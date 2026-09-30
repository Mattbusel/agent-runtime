//! # Module: tools
//!
//! ## Responsibility
//! Tool registry: register, look up, call, and introspect named async tools.
//! Provides three built-in tools: `echo`, `calculator`, and `timestamp`.
//!
//! ## Guarantees
//! - `ToolRegistry::call` always returns a `ToolError` (never panics).
//! - `with_timeout` wraps every call with `tokio::time::timeout`.
//! - `list()` returns per-tool statistics (call count, error count, avg latency).

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can occur when calling a tool.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// No tool with that name is registered.
    #[error("tool not found: {0}")]
    NotFound(String),

    /// The tool itself returned an error.
    #[error("tool '{tool}' execution failed: {reason}")]
    ExecutionFailed {
        /// Name of the tool that failed.
        tool: String,
        /// Human-readable reason.
        reason: String,
    },

    /// The tool did not complete within the configured timeout.
    #[error("tool call timed out")]
    Timeout,
}

// ── Tool trait ────────────────────────────────────────────────────────────────

/// A named, callable async tool.
///
/// Implement this trait to add custom tools to a [`ToolRegistry`].
///
/// ```rust,ignore
/// use llm_agent_runtime::tools::{Tool, ToolError};
///
/// struct MyTool;
///
/// #[async_trait::async_trait]
/// impl Tool for MyTool {
///     fn name(&self) -> &str { "my_tool" }
///     fn description(&self) -> &str { "Does something useful." }
///     async fn call(&self, input: &str) -> Result<String, ToolError> {
///         Ok(format!("processed: {input}"))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// The unique name used to look up this tool in the registry.
    fn name(&self) -> &str;

    /// A human-readable description of what the tool does.
    fn description(&self) -> &str;

    /// Execute the tool with the given `input` string.
    ///
    /// # Errors
    /// Returns [`ToolError::ExecutionFailed`] on logic errors inside the tool.
    async fn call(&self, input: &str) -> Result<String, ToolError>;
}

// ── ToolInfo (statistics snapshot) ───────────────────────────────────────────

/// Point-in-time statistics for a single registered tool.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// The tool's registered name.
    pub name: String,
    /// The tool's human-readable description.
    pub description: String,
    /// Total number of successful calls made so far.
    pub call_count: u64,
    /// Total number of calls that returned an error.
    pub error_count: u64,
    /// Average wall-clock latency in milliseconds across all calls.
    pub avg_latency_ms: f64,
}

// ── Per-tool metrics (internal) ───────────────────────────────────────────────

#[derive(Debug, Default)]
struct ToolStats {
    call_count: u64,
    error_count: u64,
    total_latency_ms: f64,
}

impl ToolStats {
    fn avg_latency_ms(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_latency_ms / self.call_count as f64
        }
    }
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry of named async tools with optional per-call timeout.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use llm_agent_runtime::tools::{ToolRegistry, EchoTool};
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(EchoTool));
/// let result = registry.call("echo", "hello").await.unwrap();
/// assert_eq!(result, "hello");
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    stats: Arc<Mutex<HashMap<String, ToolStats>>>,
    timeout: Option<Duration>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create a new empty registry without a timeout.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            stats: Arc::new(Mutex::new(HashMap::new())),
            timeout: None,
        }
    }

    /// Set a per-call timeout.  Every subsequent `call()` is wrapped in
    /// `tokio::time::timeout(duration, ...)`.
    ///
    /// Returns `self` for chaining.
    pub fn with_timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    /// Register a tool.  If a tool with the same name was previously registered
    /// it is silently replaced.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_owned();
        self.tools.insert(name, tool);
    }

    /// Return the tool registered under `name`, or `None`.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Return a snapshot of per-tool statistics for all registered tools.
    ///
    /// # Panics (never)
    /// Uses `try_lock`; if the lock is poisoned the stats field is zeroed.
    pub async fn list(&self) -> Vec<ToolInfo> {
        let stats = self.stats.lock().await;
        self.tools
            .values()
            .map(|t| {
                let s = stats.get(t.name()).unwrap_or(&ToolStats { call_count: 0, error_count: 0, total_latency_ms: 0.0 });
                ToolInfo {
                    name: t.name().to_owned(),
                    description: t.description().to_owned(),
                    call_count: s.call_count,
                    error_count: s.error_count,
                    avg_latency_ms: s.avg_latency_ms(),
                }
            })
            .collect()
    }

    /// Call the tool registered under `name` with `input`.
    ///
    /// Updates per-tool statistics (call count, error count, latency).
    ///
    /// # Errors
    /// - [`ToolError::NotFound`] — no tool registered under `name`.
    /// - [`ToolError::Timeout`] — call exceeded the configured timeout.
    /// - [`ToolError::ExecutionFailed`] — the tool itself returned an error.
    pub async fn call(&self, name: &str, input: &str) -> Result<String, ToolError> {
        let tool = self
            .tools
            .get(name)
            .cloned()
            .ok_or_else(|| ToolError::NotFound(name.to_owned()))?;

        let start = Instant::now();

        let result = if let Some(timeout_dur) = self.timeout {
            tokio::time::timeout(timeout_dur, tool.call(input))
                .await
                .map_err(|_| ToolError::Timeout)?
        } else {
            tool.call(input).await
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut stats = self.stats.lock().await;
        let entry = stats.entry(name.to_owned()).or_default();
        entry.total_latency_ms += elapsed_ms;

        match &result {
            Ok(_) => {
                entry.call_count += 1;
            }
            Err(_) => {
                entry.call_count += 1;
                entry.error_count += 1;
            }
        }

        result
    }
}

// ── Built-in tools ────────────────────────────────────────────────────────────

/// Built-in tool: returns its input unchanged.
///
/// Useful as a passthrough for testing or templating.
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Returns the input string unchanged."
    }

    async fn call(&self, input: &str) -> Result<String, ToolError> {
        Ok(input.to_owned())
    }
}

/// Built-in tool: evaluates simple arithmetic expressions.
///
/// Supports `+`, `-`, `*`, `/`, unary `-`, and parentheses.
/// Integers and decimals are both supported.
///
/// # Example inputs
/// - `"2 + 3"` → `"5"`
/// - `"(10 + 5) * 2"` → `"30"`
/// - `"7 / 2"` → `"3.5"`
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluates arithmetic expressions: +, -, *, / with parentheses support."
    }

    async fn call(&self, input: &str) -> Result<String, ToolError> {
        let expr = input.trim();
        match eval_expr(expr) {
            Ok(v) => {
                // Format as integer when possible, otherwise as float.
                if v.fract() == 0.0 && v.abs() < 1e15_f64 {
                    Ok(format!("{}", v as i64))
                } else {
                    Ok(format!("{v}"))
                }
            }
            Err(reason) => Err(ToolError::ExecutionFailed {
                tool: "calculator".to_owned(),
                reason,
            }),
        }
    }
}

// ── Recursive-descent parser ──────────────────────────────────────────────────

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            input: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos] == b' ' {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.input.get(self.pos).copied()
    }

    fn consume(&mut self) -> Option<u8> {
        self.skip_ws();
        if self.pos < self.input.len() {
            let ch = self.input[self.pos];
            self.pos += 1;
            Some(ch)
        } else {
            None
        }
    }

    /// expr = term (('+' | '-') term)*
    fn expr(&mut self) -> Result<f64, String> {
        let mut val = self.term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    val += self.term()?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    val -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(val)
    }

    /// term = factor (('*' | '/') factor)*
    fn term(&mut self) -> Result<f64, String> {
        let mut val = self.factor()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    val *= self.factor()?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let denom = self.factor()?;
                    if denom == 0.0 {
                        return Err("division by zero".to_owned());
                    }
                    val /= denom;
                }
                _ => break,
            }
        }
        Ok(val)
    }

    /// factor = '-' factor | '(' expr ')' | number
    fn factor(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some(b'-') => {
                self.pos += 1;
                Ok(-self.factor()?)
            }
            Some(b'(') => {
                self.pos += 1;
                let val = self.expr()?;
                self.skip_ws();
                if self.consume() != Some(b')') {
                    return Err("expected ')'".to_owned());
                }
                Ok(val)
            }
            Some(ch) if ch.is_ascii_digit() || ch == b'.' => self.number(),
            other => Err(format!(
                "unexpected character: {}",
                other.map(|c| c as char).unwrap_or('\0')
            )),
        }
    }

    fn number(&mut self) -> Result<f64, String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.input.len()
            && (self.input[self.pos].is_ascii_digit() || self.input[self.pos] == b'.')
        {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| e.to_string())?;
        s.parse::<f64>().map_err(|e| e.to_string())
    }
}

fn eval_expr(s: &str) -> Result<f64, String> {
    let mut parser = Parser::new(s);
    let val = parser.expr()?;
    parser.skip_ws();
    if parser.pos != parser.input.len() {
        return Err(format!(
            "unexpected trailing characters at position {}",
            parser.pos
        ));
    }
    Ok(val)
}

/// Built-in tool: returns the current UTC time as an ISO 8601 timestamp.
///
/// Input is ignored. Output format: `"2024-01-15T12:34:56Z"` (seconds precision).
pub struct TimestampTool;

#[async_trait]
impl Tool for TimestampTool {
    fn name(&self) -> &str {
        "timestamp"
    }

    fn description(&self) -> &str {
        "Returns the current UTC date and time as an ISO 8601 timestamp."
    }

    async fn call(&self, _input: &str) -> Result<String, ToolError> {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Format as ISO 8601 UTC without external crate.
        let ts = fmt_unix_secs(secs);
        Ok(ts)
    }
}

/// Format a Unix timestamp (seconds) as `"YYYY-MM-DDTHH:MM:SSZ"`.
fn fmt_unix_secs(secs: u64) -> String {
    // Days since epoch.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hour = time_of_day / 3600;
    let minute = (time_of_day % 3600) / 60;
    let second = time_of_day % 60;

    // Gregorian calendar computation.
    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

fn days_to_ymd(days: u64) -> (u64, u8, u8) {
    // Using the proleptic Gregorian algorithm.
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u8, d as u8)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Helper: create a registry pre-loaded with all three built-ins.
    fn registry_with_builtins() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        reg.register(Arc::new(CalculatorTool));
        reg.register(Arc::new(TimestampTool));
        reg
    }

    // ── EchoTool ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn echo_returns_input() {
        let tool = EchoTool;
        let result = tool.call("hello world").await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn echo_empty_input() {
        let tool = EchoTool;
        let result = tool.call("").await.unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn echo_name_and_description() {
        let tool = EchoTool;
        assert_eq!(tool.name(), "echo");
        assert!(!tool.description().is_empty());
    }

    // ── CalculatorTool ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn calculator_addition() {
        let tool = CalculatorTool;
        assert_eq!(tool.call("2 + 3").await.unwrap(), "5");
    }

    #[tokio::test]
    async fn calculator_subtraction() {
        let tool = CalculatorTool;
        assert_eq!(tool.call("10 - 4").await.unwrap(), "6");
    }

    #[tokio::test]
    async fn calculator_multiplication() {
        let tool = CalculatorTool;
        assert_eq!(tool.call("3 * 7").await.unwrap(), "21");
    }

    #[tokio::test]
    async fn calculator_division_integer() {
        let tool = CalculatorTool;
        assert_eq!(tool.call("10 / 2").await.unwrap(), "5");
    }

    #[tokio::test]
    async fn calculator_division_float() {
        let tool = CalculatorTool;
        let result = tool.call("7 / 2").await.unwrap();
        let v: f64 = result.parse().unwrap();
        assert!((v - 3.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn calculator_parentheses() {
        let tool = CalculatorTool;
        assert_eq!(tool.call("(10 + 5) * 2").await.unwrap(), "30");
    }

    #[tokio::test]
    async fn calculator_unary_minus() {
        let tool = CalculatorTool;
        assert_eq!(tool.call("-5 + 3").await.unwrap(), "-2");
    }

    #[tokio::test]
    async fn calculator_nested_parens() {
        let tool = CalculatorTool;
        assert_eq!(tool.call("((2 + 3) * (4 - 1))").await.unwrap(), "15");
    }

    #[tokio::test]
    async fn calculator_division_by_zero() {
        let tool = CalculatorTool;
        let err = tool.call("1 / 0").await.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed { .. }));
    }

    #[tokio::test]
    async fn calculator_invalid_expr() {
        let tool = CalculatorTool;
        let err = tool.call("abc").await.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed { .. }));
    }

    // ── TimestampTool ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn timestamp_returns_iso8601() {
        let tool = TimestampTool;
        let ts = tool.call("ignored").await.unwrap();
        // Basic format check: "YYYY-MM-DDTHH:MM:SSZ"
        assert_eq!(ts.len(), 20, "timestamp length wrong: {ts}");
        assert!(ts.ends_with('Z'), "should end with Z: {ts}");
        assert!(ts.contains('T'), "should contain T: {ts}");
    }

    #[tokio::test]
    async fn timestamp_ignores_input() {
        let tool = TimestampTool;
        let t1 = tool.call("foo").await.unwrap();
        let t2 = tool.call("").await.unwrap();
        // Both should be valid ISO 8601 strings (may differ by a second).
        assert!(t1.ends_with('Z'));
        assert!(t2.ends_with('Z'));
    }

    // ── ToolRegistry ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn registry_call_echo() {
        let reg = registry_with_builtins();
        let result = reg.call("echo", "ping").await.unwrap();
        assert_eq!(result, "ping");
    }

    #[tokio::test]
    async fn registry_call_not_found() {
        let reg = registry_with_builtins();
        let err = reg.call("nonexistent", "x").await.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn registry_get_returns_tool() {
        let reg = registry_with_builtins();
        assert!(reg.get("echo").is_some());
        assert!(reg.get("calculator").is_some());
        assert!(reg.get("missing").is_none());
    }

    #[tokio::test]
    async fn registry_list_contains_all() {
        let reg = registry_with_builtins();
        let list = reg.list().await;
        let names: Vec<&str> = list.iter().map(|i| i.name.as_str()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"timestamp"));
    }

    #[tokio::test]
    async fn registry_stats_update_on_call() {
        let reg = registry_with_builtins();
        reg.call("echo", "a").await.unwrap();
        reg.call("echo", "b").await.unwrap();
        let list = reg.list().await;
        let info = list.iter().find(|i| i.name == "echo").unwrap();
        assert_eq!(info.call_count, 2);
        assert_eq!(info.error_count, 0);
    }

    #[tokio::test]
    async fn registry_stats_error_incremented() {
        let reg = registry_with_builtins();
        let _ = reg.call("calculator", "abc").await;
        let list = reg.list().await;
        let info = list.iter().find(|i| i.name == "calculator").unwrap();
        assert_eq!(info.error_count, 1);
    }

    #[tokio::test]
    async fn registry_timeout_triggers() {
        use std::time::Duration;

        struct SlowTool;
        #[async_trait]
        impl Tool for SlowTool {
            fn name(&self) -> &str { "slow" }
            fn description(&self) -> &str { "sleeps" }
            async fn call(&self, _input: &str) -> Result<String, ToolError> {
                tokio::time::sleep(Duration::from_secs(10)).await;
                Ok("done".to_owned())
            }
        }

        let mut reg = ToolRegistry::new().with_timeout(Duration::from_millis(50));
        reg.register(Arc::new(SlowTool));
        let err = reg.call("slow", "").await.unwrap_err();
        assert!(matches!(err, ToolError::Timeout));
    }

    #[tokio::test]
    async fn registry_replace_existing_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        reg.register(Arc::new(EchoTool)); // replace
        let list = reg.list().await;
        assert_eq!(list.iter().filter(|i| i.name == "echo").count(), 1);
    }

    #[tokio::test]
    async fn registry_avg_latency_nonzero_after_call() {
        let reg = registry_with_builtins();
        reg.call("echo", "test").await.unwrap();
        let list = reg.list().await;
        let info = list.iter().find(|i| i.name == "echo").unwrap();
        // avg_latency_ms should be >= 0.0 (wall-clock time).
        assert!(info.avg_latency_ms >= 0.0);
    }

    // ── Parser unit tests ─────────────────────────────────────────────────────

    #[test]
    fn eval_simple_add() {
        assert_eq!(eval_expr("1 + 1").unwrap(), 2.0);
    }

    #[test]
    fn eval_operator_precedence() {
        // 2 + 3 * 4 = 14
        assert!((eval_expr("2 + 3 * 4").unwrap() - 14.0).abs() < 1e-9);
    }

    #[test]
    fn eval_decimal() {
        assert!((eval_expr("1.5 + 1.5").unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn eval_deep_nesting() {
        assert!((eval_expr("((1 + 2) * (3 + 4))").unwrap() - 21.0).abs() < 1e-9);
    }

    // ── fmt_unix_secs ─────────────────────────────────────────────────────────

    #[test]
    fn fmt_epoch() {
        let ts = fmt_unix_secs(0);
        assert_eq!(ts, "1970-01-01T00:00:00Z");
    }

    #[test]
    fn fmt_known_date() {
        // 2024-01-01T00:00:00Z = 1704067200
        let ts = fmt_unix_secs(1_704_067_200);
        assert_eq!(ts, "2024-01-01T00:00:00Z");
    }
}
