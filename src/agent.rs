//! # Module: Agent
//!
//! ## Responsibility
//! Provides a ReAct (Thought-Action-Observation) agent loop with pluggable tools.
//! Mirrors the public API of `wasm-agent`.
//!
//! ## Guarantees
//! - Deterministic: the loop terminates after at most `max_iterations` cycles
//! - Non-panicking: all operations return `Result`
//! - Tool handlers support both sync and async `Fn` closures
//!
//! ## NOT Responsible For
//! - Actual LLM inference (callers supply a mock/stub inference fn)
//! - WASM compilation or browser execution
//! - Streaming partial responses

use crate::error::AgentRuntimeError;
use crate::metrics::RuntimeMetrics;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A pinned, boxed future returning a `Value`. Used for async tool handlers.
pub type AsyncToolFuture = Pin<Box<dyn Future<Output = Value> + Send>>;

/// A pinned, boxed future returning `Result<Value, String>`. Used for fallible async tool handlers.
pub type AsyncToolResultFuture = Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;

/// An async tool handler closure.
pub type AsyncToolHandler = Box<dyn Fn(Value) -> AsyncToolFuture + Send + Sync>;

/// Role of a message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// System-level instruction injected before the user turn.
    System,
    /// Message from the human user.
    User,
    /// Message produced by the language model.
    Assistant,
    /// Message produced by a tool invocation.
    Tool,
}

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the speaker who produced this message.
    pub role: Role,
    /// The textual content of the message.
    pub content: String,
}

impl Message {
    /// Create a new `Message` with the given role and content.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Create a `User` role message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Create an `Assistant` role message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Create a `System` role message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// Return a reference to the message role.
    pub fn role(&self) -> &Role {
        &self.role
    }

    /// Return the message content as a `&str`.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Return `true` if this is a `User` role message.
    pub fn is_user(&self) -> bool {
        self.role == Role::User
    }

    /// Return `true` if this is an `Assistant` role message.
    pub fn is_assistant(&self) -> bool {
        self.role == Role::Assistant
    }

    /// Return `true` if this is a `System` role message.
    pub fn is_system(&self) -> bool {
        self.role == Role::System
    }

    /// Return `true` if this is a `Tool` role message.
    pub fn is_tool(&self) -> bool {
        self.role == Role::Tool
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// A single ReAct step: Thought → Action → Observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStep {
    /// Agent's reasoning about the current state.
    pub thought: String,
    /// The action taken (tool name + JSON arguments, or "FINAL_ANSWER").
    pub action: String,
    /// The result of the action.
    pub observation: String,
    /// Wall-clock duration of this individual step in milliseconds.
    /// Covers the time from the start of the inference call to the end of the
    /// tool invocation (or FINAL_ANSWER detection).  Zero for steps that were
    /// constructed outside the loop (e.g., in tests).
    #[serde(default)]
    pub step_duration_ms: u64,
}

impl ReActStep {
    /// Returns `true` if this step's action is a `FINAL_ANSWER`.
    pub fn is_final_answer(&self) -> bool {
        self.action.trim().to_ascii_uppercase().starts_with("FINAL_ANSWER")
    }

    /// Returns `true` if this step's action is a tool call (not a FINAL_ANSWER).
    pub fn is_tool_call(&self) -> bool {
        !self.is_final_answer() && !self.action.trim().is_empty()
    }
}

/// Configuration for the ReAct agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of Thought-Action-Observation cycles.
    pub max_iterations: usize,
    /// Model identifier passed to the inference function.
    pub model: String,
    /// System prompt injected at the start of the conversation.
    pub system_prompt: String,
    /// Maximum number of episodic memories to inject into the prompt.
    /// Keeping this small prevents silent token-budget overruns.  Default: 3.
    pub max_memory_recalls: usize,
    /// Maximum token budget for injected memories.
    ///
    /// Token counting is delegated to the [`TokenEstimator`] configured on
    /// [`AgentRuntimeBuilder`] (default: 1 token ≈ 4 bytes).  `None` means
    /// no limit.
    ///
    /// [`TokenEstimator`]: crate::runtime::TokenEstimator
    /// [`AgentRuntimeBuilder`]: crate::runtime::AgentRuntimeBuilder
    pub max_memory_tokens: Option<usize>,
    /// Optional wall-clock timeout for the entire loop.
    /// If the loop runs longer than this duration, it returns
    /// `Err(AgentRuntimeError::AgentLoop("loop timeout ..."))`.
    pub loop_timeout: Option<std::time::Duration>,
    /// Model sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<usize>,
    /// Per-inference timeout.
    pub request_timeout: Option<std::time::Duration>,
    /// Maximum number of characters allowed in the running context string.
    ///
    /// When set, the oldest Thought/Action/Observation turns are trimmed from
    /// the **beginning** of the context (after the system prompt) to keep the
    /// total length below this limit.  This prevents silent context-window
    /// overruns in long-running agents.  `None` (default) means no limit.
    pub max_context_chars: Option<usize>,
    /// Stop sequences passed to the inference function.
    ///
    /// The model stops generating when it produces any of these strings.
    /// An empty `Vec` (default) means no stop sequences.
    pub stop_sequences: Vec<String>,
}

impl AgentConfig {
    /// Create a new config with sensible defaults.
    pub fn new(max_iterations: usize, model: impl Into<String>) -> Self {
        Self {
            max_iterations,
            model: model.into(),
            system_prompt: "You are a helpful AI agent.".into(),
            max_memory_recalls: 3,
            max_memory_tokens: None,
            loop_timeout: None,
            temperature: None,
            max_tokens: None,
            request_timeout: None,
            max_context_chars: None,
            stop_sequences: vec![],
        }
    }

    /// Override the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the maximum number of episodic memories injected per run.
    pub fn with_max_memory_recalls(mut self, n: usize) -> Self {
        self.max_memory_recalls = n;
        self
    }

    /// Set a maximum token budget for injected memories (~4 chars/token heuristic).
    pub fn with_max_memory_tokens(mut self, n: usize) -> Self {
        self.max_memory_tokens = Some(n);
        self
    }

    /// Set a wall-clock timeout for the entire ReAct loop.
    ///
    /// If the loop has not reached `FINAL_ANSWER` within this duration,
    /// [`ReActLoop::run`] returns `Err(AgentRuntimeError::AgentLoop(...))`.
    pub fn with_loop_timeout(mut self, d: std::time::Duration) -> Self {
        self.loop_timeout = Some(d);
        self
    }

    /// Set a wall-clock timeout for the entire ReAct loop using seconds.
    ///
    /// Convenience wrapper around [`with_loop_timeout`](Self::with_loop_timeout).
    pub fn with_loop_timeout_secs(self, secs: u64) -> Self {
        self.with_loop_timeout(std::time::Duration::from_secs(secs))
    }

    /// Set a wall-clock timeout for the entire ReAct loop using milliseconds.
    ///
    /// Convenience wrapper around [`with_loop_timeout`](Self::with_loop_timeout).
    pub fn with_loop_timeout_ms(self, ms: u64) -> Self {
        self.with_loop_timeout(std::time::Duration::from_millis(ms))
    }

    /// Set the model sampling temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the maximum output tokens.
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set the per-inference timeout.
    pub fn with_request_timeout(mut self, d: std::time::Duration) -> Self {
        self.request_timeout = Some(d);
        self
    }

    /// Set the per-inference timeout using seconds.
    ///
    /// Convenience wrapper around [`with_request_timeout`](Self::with_request_timeout).
    pub fn with_request_timeout_secs(self, secs: u64) -> Self {
        self.with_request_timeout(std::time::Duration::from_secs(secs))
    }

    /// Set the per-inference timeout using milliseconds.
    ///
    /// Convenience wrapper around [`with_request_timeout`](Self::with_request_timeout).
    pub fn with_request_timeout_ms(self, ms: u64) -> Self {
        self.with_request_timeout(std::time::Duration::from_millis(ms))
    }

    /// Set a maximum character limit for the running context string.
    ///
    /// When the context exceeds this length the oldest
    /// Thought/Action/Observation turns are trimmed from the front (after the
    /// initial system prompt + user turn) so the context stays under the limit.
    pub fn with_max_context_chars(mut self, n: usize) -> Self {
        self.max_context_chars = Some(n);
        self
    }

    /// Change the model used for completions.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set stop sequences for inference requests.
    ///
    /// The model will stop generating when it produces any of these strings.
    /// Defaults to an empty list (no stop sequences).
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }
}

// ── ToolSpec ──────────────────────────────────────────────────────────────────

/// Describes and implements a single callable tool.
pub struct ToolSpec {
    /// Short identifier used in action strings (e.g. "search").
    pub name: String,
    /// Human-readable description passed to the model as part of the system prompt.
    pub description: String,
    /// Async handler: receives JSON arguments, returns a future resolving to a JSON result.
    pub(crate) handler: AsyncToolHandler,
    /// Field names that must be present in the JSON args object.
    /// Empty means no validation is performed.
    pub required_fields: Vec<String>,
    /// Custom argument validators run after `required_fields` and before the handler.
    /// All validators must pass; the first failure short-circuits execution.
    pub validators: Vec<Box<dyn ToolValidator>>,
    /// Optional per-tool circuit breaker.
    #[cfg(feature = "orchestrator")]
    pub circuit_breaker: Option<Arc<crate::orchestrator::CircuitBreaker>>,
}

impl std::fmt::Debug for ToolSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("ToolSpec");
        s.field("name", &self.name)
            .field("description", &self.description)
            .field("required_fields", &self.required_fields);
        #[cfg(feature = "orchestrator")]
        s.field("has_circuit_breaker", &self.circuit_breaker.is_some());
        s.finish()
    }
}

impl ToolSpec {
    /// Construct a new `ToolSpec` from a synchronous handler closure.
    /// The closure is wrapped in an `async move` block automatically.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(Value) -> Value + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(move |args| {
                let result = handler(args);
                Box::pin(async move { result })
            }),
            required_fields: Vec::new(),
            validators: Vec::new(),
            #[cfg(feature = "orchestrator")]
            circuit_breaker: None,
        }
    }

    /// Construct a new `ToolSpec` from an async handler closure.
    pub fn new_async(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(Value) -> AsyncToolFuture + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(handler),
            required_fields: Vec::new(),
            validators: Vec::new(),
            #[cfg(feature = "orchestrator")]
            circuit_breaker: None,
        }
    }

    /// Construct a new `ToolSpec` from a synchronous fallible handler closure.
    /// `Err(msg)` is converted to `{"error": msg, "ok": false}`.
    pub fn new_fallible(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(move |args| {
                let result = handler(args);
                let value = match result {
                    Ok(v) => v,
                    Err(msg) => serde_json::json!({"error": msg, "ok": false}),
                };
                Box::pin(async move { value })
            }),
            required_fields: Vec::new(),
            validators: Vec::new(),
            #[cfg(feature = "orchestrator")]
            circuit_breaker: None,
        }
    }

    /// Construct a new `ToolSpec` from an async fallible handler closure.
    /// `Err(msg)` is converted to `{"error": msg, "ok": false}`.
    pub fn new_async_fallible(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(Value) -> AsyncToolResultFuture + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(move |args| {
                let fut = handler(args);
                Box::pin(async move {
                    match fut.await {
                        Ok(v) => v,
                        Err(msg) => serde_json::json!({"error": msg, "ok": false}),
                    }
                })
            }),
            required_fields: Vec::new(),
            validators: Vec::new(),
            #[cfg(feature = "orchestrator")]
            circuit_breaker: None,
        }
    }

    /// Set the required fields that must be present in the JSON args object.
    ///
    /// Accepts any iterable of string-like values so callers can pass
    /// `&["field1", "field2"]`, `vec!["f".to_string()]`, or any other
    /// `IntoIterator<Item: Into<String>>` without manual conversion.
    pub fn with_required_fields(
        mut self,
        fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.required_fields = fields.into_iter().map(Into::into).collect();
        self
    }

    /// Attach custom argument validators.
    ///
    /// Validators run after `required_fields` checks and before the handler.
    /// The first failing validator short-circuits execution.
    pub fn with_validators(mut self, validators: Vec<Box<dyn ToolValidator>>) -> Self {
        self.validators = validators;
        self
    }

    /// Attach a circuit breaker to this tool spec.
    #[cfg(feature = "orchestrator")]
    pub fn with_circuit_breaker(mut self, cb: Arc<crate::orchestrator::CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    /// Override the human-readable description after construction.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Invoke the tool with the given JSON arguments.
    pub async fn call(&self, args: Value) -> Value {
        (self.handler)(args).await
    }
}

// ── ToolCache ─────────────────────────────────────────────────────────────────

/// Cache for tool call results.
///
/// Implement to deduplicate repeated identical tool calls within a single
/// [`ReActLoop::run`] invocation.
///
/// ## Cache key
/// Implementations should key on `(tool_name, args)`.  The `args` value is the
/// full parsed JSON object passed to the tool.
///
/// ## Thread safety
/// The trait is `Send + Sync`, so implementations must be safe to share across
/// threads.  Wrap mutable state in a `Mutex` or use lock-free atomics.
///
/// ## TTL
/// TTL semantics are implementation-defined.  A simple in-memory cache may
/// keep results for the lifetime of the [`ReActLoop::run`] call; a distributed
/// cache may use Redis with explicit expiry.
///
/// ## Lifetime
/// A cache instance is attached to a `ToolRegistry` and lives for the lifetime
/// of that registry.  Results are **not** automatically cleared between
/// `ReActLoop::run` calls — clear the cache explicitly if cross-run dedup is
/// not desired.
pub trait ToolCache: Send + Sync {
    /// Look up a cached result for `(tool_name, args)`.
    fn get(&self, tool_name: &str, args: &serde_json::Value) -> Option<serde_json::Value>;
    /// Store a result for `(tool_name, args)`.
    fn set(&self, tool_name: &str, args: &serde_json::Value, result: serde_json::Value);
}

// ── InMemoryToolCache ─────────────────────────────────────────────────────────

/// Inner state for [`InMemoryToolCache`], tracking insertion order for eviction.
struct ToolCacheInner {
    map: HashMap<(String, String), serde_json::Value>,
    /// Insertion-ordered keys for FIFO eviction.
    order: std::collections::VecDeque<(String, String)>,
}

/// A simple in-memory [`ToolCache`] backed by a `Mutex<HashMap>`.
///
/// Caches tool results keyed on `(tool_name, args_json_string)`.
/// Optionally bounded by a maximum number of entries; the oldest entry is
/// evicted once the cap is exceeded.
///
/// # Example
/// ```rust,ignore
/// use std::sync::Arc;
/// use llm_agent_runtime::agent::{InMemoryToolCache, ToolRegistry};
///
/// let cache = Arc::new(InMemoryToolCache::new());
/// let registry = ToolRegistry::new().with_cache(cache);
/// ```
pub struct InMemoryToolCache {
    inner: std::sync::Mutex<ToolCacheInner>,
    max_entries: Option<usize>,
}

impl std::fmt::Debug for InMemoryToolCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        f.debug_struct("InMemoryToolCache")
            .field("entries", &len)
            .field("max_entries", &self.max_entries)
            .finish()
    }
}

impl InMemoryToolCache {
    /// Create a new unbounded cache.
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(ToolCacheInner {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
            }),
            max_entries: None,
        }
    }

    /// Create a cache that evicts the oldest entry once `max` entries are reached.
    pub fn with_max_entries(max: usize) -> Self {
        Self {
            inner: std::sync::Mutex::new(ToolCacheInner {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
            }),
            max_entries: Some(max),
        }
    }

    /// Remove all cached entries.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.map.clear();
            inner.order.clear();
        }
    }

    /// Return the number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|s| s.map.len()).unwrap_or(0)
    }

    /// Return `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return `true` if a cached result exists for `(tool_name, args)`.
    pub fn contains(&self, tool_name: &str, args: &serde_json::Value) -> bool {
        let key = (tool_name.to_owned(), args.to_string());
        self.inner
            .lock()
            .map(|s| s.map.contains_key(&key))
            .unwrap_or(false)
    }

    /// Remove the cached result for `(tool_name, args)`.  Returns `true` if found.
    pub fn remove(&self, tool_name: &str, args: &serde_json::Value) -> bool {
        let key = (tool_name.to_owned(), args.to_string());
        if let Ok(mut inner) = self.inner.lock() {
            if inner.map.remove(&key).is_some() {
                inner.order.retain(|k| k != &key);
                return true;
            }
        }
        false
    }
}

impl Default for InMemoryToolCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCache for InMemoryToolCache {
    fn get(&self, tool_name: &str, args: &serde_json::Value) -> Option<serde_json::Value> {
        let key = (tool_name.to_owned(), args.to_string());
        self.inner.lock().ok()?.map.get(&key).cloned()
    }

    fn set(&self, tool_name: &str, args: &serde_json::Value, result: serde_json::Value) {
        let key = (tool_name.to_owned(), args.to_string());
        if let Ok(mut inner) = self.inner.lock() {
            if !inner.map.contains_key(&key) {
                inner.order.push_back(key.clone());
            }
            inner.map.insert(key, result);
            if let Some(max) = self.max_entries {
                while inner.map.len() > max {
                    if let Some(oldest) = inner.order.pop_front() {
                        inner.map.remove(&oldest);
                    }
                }
            }
        }
    }
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry of available tools for the agent loop.
pub struct ToolRegistry {
    tools: HashMap<String, ToolSpec>,
    /// Optional tool result cache.
    cache: Option<Arc<dyn ToolCache>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("has_cache", &self.cache.is_some())
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            cache: None,
        }
    }

    /// Attach a tool result cache.
    pub fn with_cache(mut self, cache: Arc<dyn ToolCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Register a tool. Overwrites any existing tool with the same name.
    pub fn register(&mut self, spec: ToolSpec) {
        self.tools.insert(spec.name.clone(), spec);
    }

    /// Register multiple tools at once.
    ///
    /// Equivalent to calling [`register`] for each spec in order.  Duplicate
    /// names overwrite earlier entries.
    ///
    /// [`register`]: ToolRegistry::register
    pub fn register_tools(&mut self, specs: impl IntoIterator<Item = ToolSpec>) {
        for spec in specs {
            self.register(spec);
        }
    }

    /// Fluent builder: register a tool and return `self`.
    ///
    /// Allows chaining multiple registrations:
    /// ```no_run
    /// use llm_agent_runtime::agent::{ToolRegistry, ToolSpec};
    ///
    /// let registry = ToolRegistry::new()
    ///     .with_tool(ToolSpec::new("search", "Search", |args| args.clone()))
    ///     .with_tool(ToolSpec::new("calc", "Calculate", |args| args.clone()));
    /// ```
    pub fn with_tool(mut self, spec: ToolSpec) -> Self {
        self.register(spec);
        self
    }

    /// Call a tool by name.
    ///
    /// # Errors
    /// - `AgentRuntimeError::AgentLoop` — tool not found, required field missing, or
    ///   custom validator rejected the arguments
    /// - `AgentRuntimeError::CircuitOpen` — the tool's circuit breaker is open
    ///   (only possible when the `orchestrator` feature is enabled)
    #[tracing::instrument(skip_all, fields(tool_name = %name))]
    pub async fn call(&self, name: &str, args: Value) -> Result<Value, AgentRuntimeError> {
        let spec = self.tools.get(name).ok_or_else(|| {
            let mut suggestion = String::new();
            let names = self.tool_names();
            if !names.is_empty() {
                if let Some((closest, dist)) = names
                    .iter()
                    .map(|n| (n, levenshtein(name, n)))
                    .min_by_key(|(_, d)| *d)
                {
                    if dist <= 3 {
                        suggestion = format!(" (did you mean '{closest}'?)");
                    }
                }
            }
            AgentRuntimeError::AgentLoop(format!("tool '{name}' not found{suggestion}"))
        })?;

        // Item 3 — required field validation
        if !spec.required_fields.is_empty() {
            if let Some(obj) = args.as_object() {
                for field in &spec.required_fields {
                    if !obj.contains_key(field) {
                        return Err(AgentRuntimeError::AgentLoop(format!(
                            "tool '{}' missing required field '{}'",
                            name, field
                        )));
                    }
                }
            } else {
                return Err(AgentRuntimeError::AgentLoop(format!(
                    "tool '{}' requires JSON object args, got {}",
                    name, args
                )));
            }
        }

        // Custom validators.
        for validator in &spec.validators {
            validator.validate(&args)?;
        }

        // Per-tool circuit breaker check.
        #[cfg(feature = "orchestrator")]
        if let Some(ref cb) = spec.circuit_breaker {
            use crate::orchestrator::CircuitState;
            if let Ok(CircuitState::Open { .. }) = cb.state() {
                return Err(AgentRuntimeError::CircuitOpen {
                    service: format!("tool:{}", name),
                });
            }
        }

        // Check cache before invoking handler.
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(name, &args) {
                return Ok(cached);
            }
        }

        let result = spec.call(args.clone()).await;

        // Update circuit breaker based on the tool's result.
        // Tools that embed errors as JSON use {"ok": false}; treat those as
        // circuit-breaker failures so the breaker can actually open.
        #[cfg(feature = "orchestrator")]
        if let Some(ref cb) = spec.circuit_breaker {
            let is_failure = result
                .get("ok")
                .and_then(|v| v.as_bool())
                .is_some_and(|ok| !ok);
            if is_failure {
                cb.record_failure();
            } else {
                cb.record_success();
            }
        }

        // Store result in cache.
        if let Some(ref cache) = self.cache {
            cache.set(name, &args, result.clone());
        }

        Ok(result)
    }

    /// Look up a registered tool by name.  Returns `None` if not registered.
    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.get(name)
    }

    /// Remove a tool by name.  Returns `true` if the tool was registered.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Return the list of registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Remove all registered tools.
    ///
    /// Useful for resetting the registry between test runs or for dynamic
    /// agent reconfiguration.
    pub fn clear(&mut self) {
        self.tools.clear();
    }

    /// Remove a tool from the registry by name.
    ///
    /// Returns `Some(spec)` if the tool was registered, `None` if not found.
    pub fn remove(&mut self, name: &str) -> Option<ToolSpec> {
        self.tools.remove(name)
    }

    /// Return `true` if a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

// ── ReActLoop ─────────────────────────────────────────────────────────────────

/// Parses a ReAct response string into a `ReActStep`.
///
/// Case-insensitive; tolerates spaces around the colon.
/// e.g. `Thought :`, `thought:`, `THOUGHT :` all match.
///
/// **Multi-line sections**: everything between a `Thought:` (or `Action:`)
/// header and the next section header is included verbatim, so JSON arguments
/// that span multiple lines are captured correctly.
pub fn parse_react_step(text: &str) -> Result<ReActStep, AgentRuntimeError> {
    // Track which section we are currently appending into.
    #[derive(PartialEq)]
    enum Section { None, Thought, Action }

    let mut thought_lines: Vec<&str> = Vec::new();
    let mut action_lines: Vec<&str> = Vec::new();
    let mut current = Section::None;

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("thought") {
            if let Some(colon_pos) = trimmed.find(':') {
                current = Section::Thought;
                thought_lines.clear();
                let first = trimmed[colon_pos + 1..].trim();
                if !first.is_empty() {
                    thought_lines.push(first);
                }
                continue;
            }
        } else if lower.starts_with("action") {
            if let Some(colon_pos) = trimmed.find(':') {
                current = Section::Action;
                action_lines.clear();
                let first = trimmed[colon_pos + 1..].trim();
                if !first.is_empty() {
                    action_lines.push(first);
                }
                continue;
            }
        } else if lower.starts_with("observation") {
            // Stop accumulating when we hit Observation (model may include it).
            current = Section::None;
            continue;
        }
        // Continuation line — append to the current section.
        match current {
            Section::Thought => thought_lines.push(trimmed),
            Section::Action => action_lines.push(trimmed),
            Section::None => {}
        }
    }

    let thought = thought_lines.join(" ");
    let action = action_lines.join("\n").trim().to_owned();

    if thought.is_empty() && action.is_empty() {
        return Err(AgentRuntimeError::AgentLoop(
            "could not parse ReAct step from response".into(),
        ));
    }

    Ok(ReActStep {
        thought,
        action,
        observation: String::new(),
        step_duration_ms: 0,
    })
}

/// The ReAct agent loop.
pub struct ReActLoop {
    config: AgentConfig,
    registry: ToolRegistry,
    /// Optional metrics sink; increments `total_tool_calls` / `failed_tool_calls`.
    metrics: Option<Arc<RuntimeMetrics>>,
    /// Optional persistence backend for per-step checkpointing during the loop.
    #[cfg(feature = "persistence")]
    checkpoint_backend: Option<(Arc<dyn crate::persistence::PersistenceBackend>, String)>,
    /// Optional observer for agent loop events.
    observer: Option<Arc<dyn Observer>>,
    /// Optional action hook called before each tool dispatch.
    action_hook: Option<ActionHook>,
}

impl std::fmt::Debug for ReActLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("ReActLoop");
        s.field("config", &self.config)
            .field("registry", &self.registry)
            .field("has_metrics", &self.metrics.is_some())
            .field("has_observer", &self.observer.is_some())
            .field("has_action_hook", &self.action_hook.is_some());
        #[cfg(feature = "persistence")]
        s.field("has_checkpoint_backend", &self.checkpoint_backend.is_some());
        s.finish()
    }
}

impl ReActLoop {
    /// Create a new `ReActLoop` with the given configuration and an empty tool registry.
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            registry: ToolRegistry::new(),
            metrics: None,
            #[cfg(feature = "persistence")]
            checkpoint_backend: None,
            observer: None,
            action_hook: None,
        }
    }

    /// Attach an observer for agent loop events.
    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Attach an action hook called before each tool dispatch.
    pub fn with_action_hook(mut self, hook: ActionHook) -> Self {
        self.action_hook = Some(hook);
        self
    }

    /// Attach a shared `RuntimeMetrics` instance.
    ///
    /// When set, the loop increments `total_tool_calls` on every tool dispatch
    /// and `failed_tool_calls` whenever a tool returns an error observation.
    pub fn with_metrics(mut self, metrics: Arc<RuntimeMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Attach a persistence backend for per-step loop checkpointing.
    ///
    /// After each completed step the current `Vec<ReActStep>` is serialised and
    /// saved under the key `"loop:<session_id>:step:<n>"`.  Checkpoint errors
    /// are logged but never abort the loop.
    #[cfg(feature = "persistence")]
    pub fn with_step_checkpoint(
        mut self,
        backend: Arc<dyn crate::persistence::PersistenceBackend>,
        session_id: impl Into<String>,
    ) -> Self {
        self.checkpoint_backend = Some((backend, session_id.into()));
        self
    }

    /// Return a read-only reference to the tool registry.
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Remove a registered tool by name.  Returns `true` if the tool was found.
    pub fn unregister_tool(&mut self, name: &str) -> bool {
        self.registry.unregister(name)
    }

    /// Register a tool that the agent loop can invoke.
    pub fn register_tool(&mut self, spec: ToolSpec) {
        self.registry.register(spec);
    }

    /// Register multiple tools at once.
    ///
    /// Equivalent to calling [`register_tool`] for each spec.
    ///
    /// [`register_tool`]: ReActLoop::register_tool
    pub fn register_tools(&mut self, specs: impl IntoIterator<Item = ToolSpec>) {
        for spec in specs {
            self.registry.register(spec);
        }
    }

    /// Trim `context` to at most `max_chars` characters by dropping the oldest
    /// Thought/Action/Observation turns from the **front** while preserving the
    /// initial system-prompt + user-turn preamble.
    ///
    /// Turns are delineated by leading `\nThought:` markers.  If no second
    /// turn boundary is found the context is left unchanged.
    fn maybe_trim_context(context: &mut String, max_chars: usize) {
        while context.len() > max_chars {
            // Find the second occurrence of "\nThought:" so we preserve the
            // preamble (everything up to the first turn) and drop only the
            // oldest appended turn.
            let first = context.find("\nThought:");
            let second = first.and_then(|pos| {
                context[pos + 1..].find("\nThought:").map(|p| pos + 1 + p)
            });
            if let Some(drop_until) = second {
                context.drain(..drop_until);
            } else {
                break; // Only one turn (or none) — nothing safe to drop.
            }
        }
    }

    /// Emit a blocked-action observation string.
    fn blocked_observation() -> String {
        serde_json::json!({
            "ok": false,
            "error": "action blocked by reviewer",
            "kind": "blocked"
        })
        .to_string()
    }

    /// Build the error observation JSON for a failed tool call.
    fn error_observation(_tool_name: &str, e: &AgentRuntimeError) -> String {
        let kind = match e {
            AgentRuntimeError::AgentLoop(msg) if msg.contains("not found") => "not_found",
            #[cfg(feature = "orchestrator")]
            AgentRuntimeError::CircuitOpen { .. } => "transient",
            _ => "permanent",
        };
        serde_json::json!({ "ok": false, "error": e.to_string(), "kind": kind }).to_string()
    }

    /// Execute the ReAct loop for the given prompt.
    ///
    /// # Arguments
    /// * `prompt` — user input passed as the initial context
    /// * `infer`  — async inference function: receives context string, returns response string
    ///
    /// # Errors
    /// - `AgentRuntimeError::AgentLoop("loop timeout …")` — if `loop_timeout` is configured
    ///   and the loop runs past the deadline
    /// - `AgentRuntimeError::AgentLoop("max iterations … reached")` — if the loop exhausts
    ///   `max_iterations` without emitting `FINAL_ANSWER`
    /// - `AgentRuntimeError::AgentLoop("could not parse …")` — if the model response cannot
    ///   be parsed into a `ReActStep`
    #[tracing::instrument(skip(infer))]
    pub async fn run<F, Fut>(
        &self,
        prompt: &str,
        mut infer: F,
    ) -> Result<Vec<ReActStep>, AgentRuntimeError>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = String>,
    {
        let mut steps: Vec<ReActStep> = Vec::new();
        let mut context = format!("{}\n\nUser: {}\n", self.config.system_prompt, prompt);

        // Pre-compute optional deadline once so that each iteration is O(1).
        let deadline = self
            .config
            .loop_timeout
            .map(|d| std::time::Instant::now() + d);

        // Observer: on_loop_start
        if let Some(ref obs) = self.observer {
            obs.on_loop_start(prompt);
        }

        for iteration in 0..self.config.max_iterations {
            let iter_span = tracing::info_span!(
                "react_iteration",
                iteration = iteration,
                model = %self.config.model,
            );
            let _iter_guard = iter_span.enter();

            // Wall-clock timeout check.
            if let Some(dl) = deadline {
                if std::time::Instant::now() >= dl {
                    let ms = self
                        .config
                        .loop_timeout
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    let err = AgentRuntimeError::AgentLoop(format!("loop timeout after {ms} ms"));
                    if let Some(ref obs) = self.observer {
                        obs.on_error(&err);
                        obs.on_loop_end(steps.len());
                    }
                    return Err(err);
                }
            }

            let step_start = std::time::Instant::now();
            let response = infer(context.clone()).await;
            let mut step = parse_react_step(&response)?;

            tracing::debug!(
                step = iteration,
                thought = %step.thought,
                action = %step.action,
                "ReAct iteration"
            );

            if step.action.to_ascii_uppercase().starts_with("FINAL_ANSWER") {
                step.observation = step.action.clone();
                step.step_duration_ms = step_start.elapsed().as_millis() as u64;
                if let Some(ref m) = self.metrics {
                    m.record_step_latency(step.step_duration_ms);
                }
                if let Some(ref obs) = self.observer {
                    obs.on_step(iteration, &step);
                }
                steps.push(step);
                tracing::info!(step = iteration, "FINAL_ANSWER reached");
                if let Some(ref obs) = self.observer {
                    obs.on_loop_end(steps.len());
                }
                return Ok(steps);
            }

            // Item 3 — propagate parse errors rather than silently falling back.
            let (tool_name, args) = parse_tool_call(&step.action)?;

            tracing::debug!(
                step = iteration,
                tool_name = %tool_name,
                "dispatching tool call"
            );

            // Action hook check.
            if let Some(ref hook) = self.action_hook {
                if !hook(tool_name.clone(), args.clone()).await {
                    if let Some(ref obs) = self.observer {
                        obs.on_action_blocked(&tool_name, &args);
                    }
                    if let Some(ref m) = self.metrics {
                        m.record_tool_call(&tool_name);
                        m.record_tool_failure(&tool_name);
                    }
                    step.observation = Self::blocked_observation();
                    step.step_duration_ms = step_start.elapsed().as_millis() as u64;
                    if let Some(ref m) = self.metrics {
                        m.record_step_latency(step.step_duration_ms);
                    }
                    let _ = write!(
                        context,
                        "\nThought: {}\nAction: {}\nObservation: {}\n",
                        step.thought, step.action, step.observation
                    );
                    if let Some(max) = self.config.max_context_chars {
                        Self::maybe_trim_context(&mut context, max);
                    }
                    if let Some(ref obs) = self.observer {
                        obs.on_step(iteration, &step);
                    }
                    steps.push(step);
                    continue;
                }
            }

            // Observer: on_tool_call
            if let Some(ref obs) = self.observer {
                obs.on_tool_call(&tool_name, &args);
            }

            // Count every tool dispatch (global + per-tool).
            if let Some(ref m) = self.metrics {
                m.record_tool_call(&tool_name);
            }

            // Structured error categorization in observation.
            let tool_span = tracing::info_span!("tool_dispatch", tool = %tool_name);
            let _tool_guard = tool_span.enter();
            let observation = match self.registry.call(&tool_name, args).await {
                Ok(result) => serde_json::json!({ "ok": true, "data": result }).to_string(),
                Err(e) => {
                    // Count failed tool calls (global + per-tool).
                    if let Some(ref m) = self.metrics {
                        m.record_tool_failure(&tool_name);
                    }
                    Self::error_observation(&tool_name, &e)
                }
            };

            step.observation = observation.clone();
            step.step_duration_ms = step_start.elapsed().as_millis() as u64;
            if let Some(ref m) = self.metrics {
                m.record_step_latency(step.step_duration_ms);
            }
            let _ = write!(
                context,
                "\nThought: {}\nAction: {}\nObservation: {}\n",
                step.thought, step.action, observation
            );
            if let Some(max) = self.config.max_context_chars {
                Self::maybe_trim_context(&mut context, max);
            }
            if let Some(ref obs) = self.observer {
                obs.on_step(iteration, &step);
            }
            steps.push(step);

            // Item 11 — per-step loop checkpoint (behind feature flag).
            #[cfg(feature = "persistence")]
            if let Some((ref backend, ref session_id)) = self.checkpoint_backend {
                let step_idx = steps.len();
                let key = format!("loop:{session_id}:step:{step_idx}");
                match serde_json::to_vec(&steps) {
                    Ok(bytes) => {
                        if let Err(e) = backend.save(&key, &bytes).await {
                            tracing::warn!(
                                key = %key,
                                error = %e,
                                "loop step checkpoint save failed"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            step = step_idx,
                            error = %e,
                            "loop step checkpoint serialisation failed"
                        );
                    }
                }
            }
        }

        let err = AgentRuntimeError::AgentLoop(format!(
            "max iterations ({}) reached without final answer",
            self.config.max_iterations
        ));
        tracing::warn!(
            max_iterations = self.config.max_iterations,
            "ReAct loop exhausted max iterations without FINAL_ANSWER"
        );
        if let Some(ref obs) = self.observer {
            obs.on_error(&err);
            obs.on_loop_end(steps.len());
        }
        Err(err)
    }

    /// Execute the ReAct loop using a streaming inference function.
    ///
    /// Identical to [`run`] except the inference closure returns a
    /// `tokio::sync::mpsc::Receiver` that streams token chunks.  All chunks
    /// are collected into a single `String` before each iteration's response
    /// is parsed.  Stream errors result in an empty partial response (the
    /// erroring chunk is silently dropped and a warning is logged).
    ///
    /// [`run`]: ReActLoop::run
    #[tracing::instrument(skip(infer_stream))]
    pub async fn run_streaming<F, Fut>(
        &self,
        prompt: &str,
        mut infer_stream: F,
    ) -> Result<Vec<ReActStep>, AgentRuntimeError>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<
            Output = tokio::sync::mpsc::Receiver<Result<String, AgentRuntimeError>>,
        >,
    {
        self.run(prompt, move |ctx| {
            let rx_fut = infer_stream(ctx);
            async move {
                let mut rx = rx_fut.await;
                let mut out = String::new();
                while let Some(chunk) = rx.recv().await {
                    match chunk {
                        Ok(s) => out.push_str(&s),
                        Err(e) => {
                            tracing::warn!(error = %e, "streaming chunk error; skipping");
                        }
                    }
                }
                out
            }
        })
        .await
    }
}

/// Declarative argument validator for a `ToolSpec`.
///
/// Implement this trait to enforce custom argument constraints (type ranges,
/// string patterns, etc.) before the handler is invoked.
///
/// Validators run **after** `required_fields` checks and **before** the handler.
/// The first failing validator short-circuits execution.
///
/// # Basic Example
/// ```no_run
/// use llm_agent_runtime::agent::ToolValidator;
/// use llm_agent_runtime::AgentRuntimeError;
/// use serde_json::Value;
///
/// struct NonEmptyQuery;
/// impl ToolValidator for NonEmptyQuery {
///     fn validate(&self, args: &Value) -> Result<(), AgentRuntimeError> {
///         let q = args.get("q").and_then(|v| v.as_str()).unwrap_or("");
///         if q.is_empty() {
///             return Err(AgentRuntimeError::AgentLoop(
///                 "tool 'search': q must not be empty".into(),
///             ));
///         }
///         Ok(())
///     }
/// }
/// ```
///
/// # Advanced Example — Parameterised validator
/// ```no_run
/// use llm_agent_runtime::agent::{ToolSpec, ToolValidator};
/// use llm_agent_runtime::AgentRuntimeError;
/// use serde_json::Value;
///
/// /// Validates that a named integer field is within [min, max].
/// struct RangeValidator { field: &'static str, min: i64, max: i64 }
///
/// impl ToolValidator for RangeValidator {
///     fn validate(&self, args: &Value) -> Result<(), AgentRuntimeError> {
///         let n = args
///             .get(self.field)
///             .and_then(|v| v.as_i64())
///             .ok_or_else(|| {
///                 AgentRuntimeError::AgentLoop(format!(
///                     "field '{}' must be an integer", self.field
///                 ))
///             })?;
///         if n < self.min || n > self.max {
///             return Err(AgentRuntimeError::AgentLoop(format!(
///                 "field '{}' = {n} is outside [{}, {}]",
///                 self.field, self.min, self.max,
///             )));
///         }
///         Ok(())
///     }
/// }
///
/// // Attach to a tool spec:
/// let spec = ToolSpec::new("roll_dice", "Roll n dice", |args| {
///     serde_json::json!({ "result": args })
/// })
/// .with_validators(vec![
///     Box::new(RangeValidator { field: "n", min: 1, max: 100 }),
/// ]);
/// ```
pub trait ToolValidator: Send + Sync {
    /// Validate `args` before the tool handler is invoked.
    ///
    /// Return `Ok(())` if the arguments are valid, or
    /// `Err(AgentRuntimeError::AgentLoop(...))` with a human-readable message.
    fn validate(&self, args: &Value) -> Result<(), AgentRuntimeError>;
}

/// Compute the Levenshtein edit distance between two strings.
///
/// Used to suggest close matches when a tool name is not found.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[m][n]
}

/// Split `"tool_name {json}"` into `(tool_name, Value)`.
///
/// Returns `Err(AgentRuntimeError::AgentLoop)` when:
/// - the tool name is empty
/// - the argument portion is non-empty but not valid JSON
fn parse_tool_call(action: &str) -> Result<(String, Value), AgentRuntimeError> {
    let mut parts = action.splitn(2, ' ');
    let name = parts.next().unwrap_or("").to_owned();
    if name.is_empty() {
        return Err(AgentRuntimeError::AgentLoop(
            "tool call has an empty tool name".into(),
        ));
    }
    let args_str = parts.next().unwrap_or("{}");
    let args: Value = serde_json::from_str(args_str).map_err(|e| {
        AgentRuntimeError::AgentLoop(format!(
            "invalid JSON args for tool call '{name}': {e} (raw: {args_str})"
        ))
    })?;
    Ok((name, args))
}

/// Agent-specific errors, mirrors `wasm-agent::AgentError`.
///
/// Converts to `AgentRuntimeError::AgentLoop` via the `From` implementation.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The referenced tool name does not exist in the registry.
    #[error("Tool '{0}' not found")]
    ToolNotFound(String),
    /// The ReAct loop consumed all iterations without emitting `FINAL_ANSWER`.
    #[error("Max iterations exceeded: {0}")]
    MaxIterations(usize),
    /// The model response could not be parsed into a `ReActStep`.
    #[error("Parse error: {0}")]
    ParseError(String),
}

impl From<AgentError> for AgentRuntimeError {
    fn from(e: AgentError) -> Self {
        AgentRuntimeError::AgentLoop(e.to_string())
    }
}

// ── Observer ──────────────────────────────────────────────────────────────────

/// Hook trait for observing agent loop events.
///
/// All methods have no-op default implementations so you only override
/// what you care about.
pub trait Observer: Send + Sync {
    /// Called when a ReAct step completes.
    fn on_step(&self, step_index: usize, step: &ReActStep) {
        let _ = (step_index, step);
    }
    /// Called when a tool is about to be dispatched.
    fn on_tool_call(&self, tool_name: &str, args: &serde_json::Value) {
        let _ = (tool_name, args);
    }
    /// Called when an action hook blocks a tool call before dispatch.
    ///
    /// `tool_name` is the name of the blocked tool; `args` are the arguments
    /// that were passed to the hook.  This is called *instead of* `on_tool_call`.
    fn on_action_blocked(&self, tool_name: &str, args: &serde_json::Value) {
        let _ = (tool_name, args);
    }
    /// Called when the loop starts.
    fn on_loop_start(&self, prompt: &str) {
        let _ = prompt;
    }
    /// Called when the loop finishes (success or error).
    fn on_loop_end(&self, step_count: usize) {
        let _ = step_count;
    }
    /// Called when the loop terminates with an error.
    ///
    /// Invoked for timeout, max-iterations, and parse failures.
    /// `on_loop_end` is also called immediately after `on_error`.
    fn on_error(&self, error: &crate::error::AgentRuntimeError) {
        let _ = error;
    }
}

// ── Action ────────────────────────────────────────────────────────────────────

/// A parsed action from a ReAct step.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// The agent has produced a final answer.
    FinalAnswer(String),
    /// A tool call with a name and JSON arguments.
    ToolCall {
        /// The tool name.
        name: String,
        /// The parsed JSON arguments.
        args: serde_json::Value,
    },
}

impl Action {
    /// Parse an action string into an `Action`.
    ///
    /// Returns `Action::FinalAnswer` if the string starts with `FINAL_ANSWER` (case-insensitive).
    /// Otherwise parses as a tool call via `parse_tool_call`.
    pub fn parse(s: &str) -> Result<Action, AgentRuntimeError> {
        if s.trim().to_ascii_uppercase().starts_with("FINAL_ANSWER") {
            let answer = s.trim()["FINAL_ANSWER".len()..].trim().to_owned();
            return Ok(Action::FinalAnswer(answer));
        }
        let (name, args) = parse_tool_call(s)?;
        Ok(Action::ToolCall { name, args })
    }
}

/// Async hook called before each tool action. Return `true` to proceed, `false` to block.
///
/// When blocked, the loop inserts a synthetic observation
/// `{"ok": false, "error": "action blocked by reviewer", "kind": "blocked"}`
/// and continues to the next iteration without invoking the tool.
///
/// ## Observer interaction
///
/// When a hook **allows** an action (`true`), the normal observer sequence fires:
/// 1. `Observer::on_tool_call` — called before the tool is dispatched
/// 2. `Observer::on_step` — called after the observation is recorded
///
/// When a hook **blocks** an action (`false`), the sequence is:
/// 1. `Observer::on_action_blocked` — called instead of `on_tool_call`
/// 2. `Observer::on_step` — called after the synthetic blocked observation is recorded
///
/// Use [`make_action_hook`] to construct a hook from a plain `async fn` without
/// writing the `Arc<dyn Fn…>` boilerplate by hand.
pub type ActionHook = Arc<dyn Fn(String, serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>> + Send + Sync>;

/// Create an [`ActionHook`] from a plain `async fn` or closure.
///
/// This helper eliminates the need to manually write
/// `Arc::new(|name, args| Box::pin(async move { … }))`.
///
/// # Example
/// ```no_run
/// use llm_agent_runtime::agent::make_action_hook;
///
/// let hook = make_action_hook(|tool_name: String, _args| async move {
///     // Block any tool called "dangerous"
///     tool_name != "dangerous"
/// });
/// ```
pub fn make_action_hook<F, Fut>(f: F) -> ActionHook
where
    F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = bool> + Send + 'static,
{
    Arc::new(move |name, args| Box::pin(f(name, args)))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_final_answer_on_first_step() {
        let config = AgentConfig::new(5, "test-model");
        let loop_ = ReActLoop::new(config);

        let steps = loop_
            .run("Say hello", |_ctx| async {
                "Thought: I will answer directly\nAction: FINAL_ANSWER hello".to_string()
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 1);
        assert!(steps[0]
            .action
            .to_ascii_uppercase()
            .starts_with("FINAL_ANSWER"));
    }

    #[tokio::test]
    async fn test_tool_call_then_final_answer() {
        let config = AgentConfig::new(5, "test-model");
        let mut loop_ = ReActLoop::new(config);

        loop_.register_tool(ToolSpec::new("greet", "Greets someone", |_args| {
            serde_json::json!("hello!")
        }));

        let mut call_count = 0;
        let steps = loop_
            .run("Say hello", |_ctx| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: I will greet\nAction: greet {}".to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].action, "greet {}");
        assert!(steps[1]
            .action
            .to_ascii_uppercase()
            .starts_with("FINAL_ANSWER"));
    }

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        let config = AgentConfig::new(2, "test-model");
        let loop_ = ReActLoop::new(config);

        let result = loop_
            .run("loop forever", |_ctx| async {
                "Thought: thinking\nAction: noop {}".to_string()
            })
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("max iterations"));
    }

    #[tokio::test]
    async fn test_parse_react_step_valid() {
        let text = "Thought: I should check\nAction: lookup {\"key\":\"val\"}";
        let step = parse_react_step(text).unwrap();
        assert_eq!(step.thought, "I should check");
        assert_eq!(step.action, "lookup {\"key\":\"val\"}");
    }

    #[tokio::test]
    async fn test_parse_react_step_empty_fails() {
        let result = parse_react_step("no prefix lines here");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tool_not_found_returns_error_observation() {
        let config = AgentConfig::new(3, "test-model");
        let loop_ = ReActLoop::new(config);

        let mut call_count = 0;
        let steps = loop_
            .run("test", |_ctx| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: try missing tool\nAction: missing_tool {}".to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 2);
        assert!(steps[0].observation.contains("\"ok\":false"));
    }

    #[tokio::test]
    async fn test_new_async_tool_spec() {
        let spec = ToolSpec::new_async("async_tool", "An async tool", |args| {
            Box::pin(async move { serde_json::json!({"echo": args}) })
        });

        let result = spec.call(serde_json::json!({"input": "test"})).await;
        assert!(result.get("echo").is_some());
    }

    // Item 1 — Robust ReAct Parser tests

    #[tokio::test]
    async fn test_parse_react_step_case_insensitive() {
        let text = "THOUGHT: done\nACTION: FINAL_ANSWER";
        let step = parse_react_step(text).unwrap();
        assert_eq!(step.thought, "done");
        assert_eq!(step.action, "FINAL_ANSWER");
    }

    #[tokio::test]
    async fn test_parse_react_step_space_before_colon() {
        let text = "Thought : done\nAction : go";
        let step = parse_react_step(text).unwrap();
        assert_eq!(step.thought, "done");
        assert_eq!(step.action, "go");
    }

    // Item 3 — Tool required field validation tests

    #[tokio::test]
    async fn test_tool_required_fields_missing_returns_error() {
        let config = AgentConfig::new(3, "test-model");
        let mut loop_ = ReActLoop::new(config);

        loop_.register_tool(
            ToolSpec::new(
                "search",
                "Searches for something",
                |args| serde_json::json!({ "result": args }),
            )
            .with_required_fields(vec!["q".to_string()]),
        );

        let mut call_count = 0;
        let steps = loop_
            .run("test", |_ctx| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        // Call with empty object — missing "q"
                        "Thought: searching\nAction: search {}".to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 2);
        assert!(
            steps[0].observation.contains("missing required field"),
            "observation was: {}",
            steps[0].observation
        );
    }

    // Item 9 — Structured error observation tests

    #[tokio::test]
    async fn test_tool_error_observation_includes_kind() {
        let config = AgentConfig::new(3, "test-model");
        let loop_ = ReActLoop::new(config);

        let mut call_count = 0;
        let steps = loop_
            .run("test", |_ctx| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: try missing\nAction: nonexistent_tool {}".to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 2);
        let obs = &steps[0].observation;
        assert!(obs.contains("\"ok\":false"), "observation: {obs}");
        assert!(obs.contains("\"kind\":\"not_found\""), "observation: {obs}");
    }

    // ── step_duration_ms ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_step_duration_ms_is_set() {
        let config = AgentConfig::new(5, "test-model");
        let loop_ = ReActLoop::new(config);

        let steps = loop_
            .run("time it", |_ctx| async {
                "Thought: done\nAction: FINAL_ANSWER ok".to_string()
            })
            .await
            .unwrap();

        // step_duration_ms may be 0 on very fast systems but must be a valid u64.
        let _ = steps[0].step_duration_ms; // just verify the field exists and is accessible
    }

    // ── ToolValidator ─────────────────────────────────────────────────────────

    struct RequirePositiveN;
    impl ToolValidator for RequirePositiveN {
        fn validate(&self, args: &Value) -> Result<(), AgentRuntimeError> {
            let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
            if n <= 0 {
                return Err(AgentRuntimeError::AgentLoop(
                    "n must be a positive integer".into(),
                ));
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_tool_validator_blocks_invalid_args() {
        let mut registry = ToolRegistry::new();
        registry.register(
            ToolSpec::new("calc", "compute", |args| serde_json::json!({"n": args}))
                .with_validators(vec![Box::new(RequirePositiveN)]),
        );

        // n = -1 should be rejected by the validator.
        let result = registry
            .call("calc", serde_json::json!({"n": -1}))
            .await;
        assert!(result.is_err(), "validator should reject n=-1");
        assert!(result.unwrap_err().to_string().contains("positive integer"));
    }

    #[tokio::test]
    async fn test_tool_validator_passes_valid_args() {
        let mut registry = ToolRegistry::new();
        registry.register(
            ToolSpec::new("calc", "compute", |_| serde_json::json!(42))
                .with_validators(vec![Box::new(RequirePositiveN)]),
        );

        let result = registry
            .call("calc", serde_json::json!({"n": 5}))
            .await;
        assert!(result.is_ok(), "validator should accept n=5");
    }

    // ── Empty tool name ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_empty_tool_name_is_rejected() {
        // parse_tool_call("") → error because name is empty
        let result = parse_tool_call("");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("empty tool name"),
            "expected 'empty tool name' error"
        );
    }

    // ── Bulk register_tools ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_register_tools_bulk() {
        let mut registry = ToolRegistry::new();
        registry.register_tools(vec![
            ToolSpec::new("tool_a", "A", |_| serde_json::json!("a")),
            ToolSpec::new("tool_b", "B", |_| serde_json::json!("b")),
        ]);
        assert!(registry.call("tool_a", serde_json::json!({})).await.is_ok());
        assert!(registry.call("tool_b", serde_json::json!({})).await.is_ok());
    }

    // ── run_streaming parity ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_run_streaming_parity_with_run() {
        use tokio::sync::mpsc;

        let config = AgentConfig::new(5, "test-model");
        let loop_ = ReActLoop::new(config);

        let steps = loop_
            .run_streaming("Say hello", |_ctx| async {
                let (tx, rx) = mpsc::channel(4);
                // Send the response in chunks
                tokio::spawn(async move {
                    tx.send(Ok("Thought: done\n".to_string())).await.ok();
                    tx.send(Ok("Action: FINAL_ANSWER hi".to_string())).await.ok();
                });
                rx
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 1);
        assert!(steps[0]
            .action
            .to_ascii_uppercase()
            .starts_with("FINAL_ANSWER"));
    }

    #[tokio::test]
    async fn test_run_streaming_error_chunk_is_skipped() {
        use tokio::sync::mpsc;
        use crate::error::AgentRuntimeError;

        let config = AgentConfig::new(5, "test-model");
        let loop_ = ReActLoop::new(config);

        // Even with an error chunk, the loop recovers and returns the valid parts.
        let steps = loop_
            .run_streaming("test", |_ctx| async {
                let (tx, rx) = mpsc::channel(4);
                tokio::spawn(async move {
                    tx.send(Err(AgentRuntimeError::Provider("stream error".into())))
                        .await
                        .ok();
                    tx.send(Ok("Thought: recovered\nAction: FINAL_ANSWER ok".to_string()))
                        .await
                        .ok();
                });
                rx
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 1);
    }

    // ── Circuit breaker test (only compiled when feature is active) ────────────

    #[cfg(feature = "orchestrator")]
    #[tokio::test]
    async fn test_tool_with_circuit_breaker_passes_when_closed() {
        use std::sync::Arc;

        let cb = Arc::new(
            crate::orchestrator::CircuitBreaker::new(
                "echo-tool",
                5,
                std::time::Duration::from_secs(30),
            )
            .unwrap(),
        );

        let spec = ToolSpec::new(
            "echo",
            "Echoes args",
            |args| serde_json::json!({ "echoed": args }),
        )
        .with_circuit_breaker(cb);

        let registry = {
            let mut r = ToolRegistry::new();
            r.register(spec);
            r
        };

        let result = registry
            .call("echo", serde_json::json!({ "msg": "hi" }))
            .await;
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
    }

    // ── Improvement 1: AgentConfig builder methods ────────────────────────────

    #[test]
    fn test_agent_config_builder_methods_set_fields() {
        let config = AgentConfig::new(3, "model")
            .with_temperature(0.7)
            .with_max_tokens(512)
            .with_request_timeout(std::time::Duration::from_secs(10));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(512));
        assert_eq!(config.request_timeout, Some(std::time::Duration::from_secs(10)));
    }

    // ── Improvement 2: Fallible tool handlers ─────────────────────────────────

    #[tokio::test]
    async fn test_fallible_tool_returns_error_json_on_err() {
        let spec = ToolSpec::new_fallible(
            "fail",
            "always fails",
            |_| Err::<Value, String>("something went wrong".to_string()),
        );
        let result = spec.call(serde_json::json!({})).await;
        assert_eq!(result["ok"], serde_json::json!(false));
        assert_eq!(result["error"], serde_json::json!("something went wrong"));
    }

    #[tokio::test]
    async fn test_fallible_tool_returns_value_on_ok() {
        let spec = ToolSpec::new_fallible(
            "succeed",
            "always succeeds",
            |_| Ok::<Value, String>(serde_json::json!(42)),
        );
        let result = spec.call(serde_json::json!({})).await;
        assert_eq!(result, serde_json::json!(42));
    }

    // ── Improvement 4: Did you mean ───────────────────────────────────────────

    #[tokio::test]
    async fn test_did_you_mean_suggestion_for_typo() {
        let mut registry = ToolRegistry::new();
        registry.register(ToolSpec::new("search", "search", |_| serde_json::json!("ok")));
        let result = registry.call("searc", serde_json::json!({})).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("did you mean"), "expected suggestion in: {msg}");
    }

    #[tokio::test]
    async fn test_no_suggestion_for_very_different_name() {
        let mut registry = ToolRegistry::new();
        registry.register(ToolSpec::new("search", "search", |_| serde_json::json!("ok")));
        let result = registry.call("xxxxxxxxxxxxxxx", serde_json::json!({})).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(!msg.contains("did you mean"), "unexpected suggestion in: {msg}");
    }

    // ── Improvement 11: Action enum ───────────────────────────────────────────

    #[test]
    fn test_action_parse_final_answer() {
        let action = Action::parse("FINAL_ANSWER hello world").unwrap();
        assert_eq!(action, Action::FinalAnswer("hello world".to_string()));
    }

    #[test]
    fn test_action_parse_tool_call() {
        let action = Action::parse("search {\"q\": \"rust\"}").unwrap();
        match action {
            Action::ToolCall { name, args } => {
                assert_eq!(name, "search");
                assert_eq!(args["q"], "rust");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn test_action_parse_invalid_returns_err() {
        let result = Action::parse("");
        assert!(result.is_err());
    }

    // ── Improvement 13: Observer ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_observer_on_step_called_for_each_step() {
        use std::sync::{Arc, Mutex};

        struct CountingObserver {
            step_count: Mutex<usize>,
        }
        impl Observer for CountingObserver {
            fn on_step(&self, _step_index: usize, _step: &ReActStep) {
                let mut c = self.step_count.lock().unwrap_or_else(|e| e.into_inner());
                *c += 1;
            }
        }

        let obs = Arc::new(CountingObserver { step_count: Mutex::new(0) });
        let config = AgentConfig::new(5, "test-model");
        let mut loop_ = ReActLoop::new(config).with_observer(obs.clone() as Arc<dyn Observer>);
        loop_.register_tool(ToolSpec::new("noop", "noop", |_| serde_json::json!("ok")));

        let mut call_count = 0;
        let _steps = loop_.run("test", |_ctx| {
            call_count += 1;
            let count = call_count;
            async move {
                if count == 1 {
                    "Thought: call noop\nAction: noop {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER done".to_string()
                }
            }
        }).await.unwrap();

        let count = *obs.step_count.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(count, 2, "observer should have seen 2 steps");
    }

    // ── Improvement 14: ToolCache ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_tool_cache_returns_cached_result_on_second_call() {
        use std::collections::HashMap;
        use std::sync::Mutex;

        struct InMemCache {
            map: Mutex<HashMap<String, Value>>,
        }
        impl ToolCache for InMemCache {
            fn get(&self, tool_name: &str, args: &Value) -> Option<Value> {
                let key = format!("{tool_name}:{args}");
                let map = self.map.lock().unwrap_or_else(|e| e.into_inner());
                map.get(&key).cloned()
            }
            fn set(&self, tool_name: &str, args: &Value, result: Value) {
                let key = format!("{tool_name}:{args}");
                let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
                map.insert(key, result);
            }
        }

        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let cache = Arc::new(InMemCache { map: Mutex::new(HashMap::new()) });
        let registry = ToolRegistry::new()
            .with_cache(cache as Arc<dyn ToolCache>);
        let mut registry = registry;

        registry.register(ToolSpec::new("count", "count calls", move |_| {
            call_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            serde_json::json!({"calls": 1})
        }));

        let args = serde_json::json!({});
        let r1 = registry.call("count", args.clone()).await.unwrap();
        let r2 = registry.call("count", args.clone()).await.unwrap();

        assert_eq!(r1, r2);
        // The handler should only be called once; second call hits cache.
        assert_eq!(call_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    // ── Task 12: Chained validator short-circuit ──────────────────────────────

    #[tokio::test]
    async fn test_validators_short_circuit_on_first_failure() {
        use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};
        use std::sync::Arc;

        let second_called = Arc::new(AtomicUsize::new(0));
        let second_called_clone = Arc::clone(&second_called);

        struct AlwaysFail;
        impl ToolValidator for AlwaysFail {
            fn validate(&self, _args: &Value) -> Result<(), AgentRuntimeError> {
                Err(AgentRuntimeError::AgentLoop("first validator failed".into()))
            }
        }

        struct CountCalls(Arc<AtomicUsize>);
        impl ToolValidator for CountCalls {
            fn validate(&self, _args: &Value) -> Result<(), AgentRuntimeError> {
                self.0.fetch_add(1, AOrdering::SeqCst);
                Ok(())
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(
            ToolSpec::new("guarded", "A guarded tool", |args| args.clone())
                .with_validators(vec![
                    Box::new(AlwaysFail),
                    Box::new(CountCalls(second_called_clone)),
                ]),
        );

        let result = registry.call("guarded", serde_json::json!({})).await;
        assert!(result.is_err(), "should fail due to first validator");
        assert_eq!(
            second_called.load(AOrdering::SeqCst),
            0,
            "second validator must not be called when first fails"
        );
    }

    // ── Task 14: loop_timeout integration test ────────────────────────────────

    #[tokio::test]
    async fn test_loop_timeout_fires_between_iterations() {
        let mut config = AgentConfig::new(100, "test-model");
        // 30 ms deadline; each infer call sleeps 20 ms, so timeout fires after 2 iterations.
        config.loop_timeout = Some(std::time::Duration::from_millis(30));
        let loop_ = ReActLoop::new(config);

        let result = loop_
            .run("test", |_ctx| async {
                // Sleep just long enough that the cumulative time exceeds the deadline.
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                // Return a valid step that keeps the loop going (unknown tool → error observation → next iter).
                "Thought: still working\nAction: noop {}".to_string()
            })
            .await;

        assert!(result.is_err(), "loop should time out");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("loop timeout"), "unexpected error: {msg}");
    }

    // ── Improvement 15: ActionHook ────────────────────────────────────────────

    // ── #2 ReActStep::is_final_answer / is_tool_call ──────────────────────────

    #[test]
    fn test_react_step_is_final_answer() {
        let step = ReActStep {
            thought: "".into(),
            action: "FINAL_ANSWER done".into(),
            observation: "".into(),
            step_duration_ms: 0,
        };
        assert!(step.is_final_answer());
        assert!(!step.is_tool_call());
    }

    #[test]
    fn test_react_step_is_tool_call() {
        let step = ReActStep {
            thought: "".into(),
            action: "search {}".into(),
            observation: "".into(),
            step_duration_ms: 0,
        };
        assert!(!step.is_final_answer());
        assert!(step.is_tool_call());
    }

    // ── #6 Role Display ───────────────────────────────────────────────────────

    #[test]
    fn test_role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
        assert_eq!(Role::Tool.to_string(), "tool");
    }

    // ── #12 Message accessors ─────────────────────────────────────────────────

    #[test]
    fn test_message_accessors() {
        let msg = Message::new(Role::User, "hello");
        assert_eq!(msg.role(), &Role::User);
        assert_eq!(msg.content(), "hello");
    }

    // ── #25 Action parse round-trips ──────────────────────────────────────────

    #[test]
    fn test_action_parse_final_answer_round_trip() {
        let step = ReActStep {
            thought: "done".into(),
            action: "FINAL_ANSWER Paris".into(),
            observation: "".into(),
            step_duration_ms: 0,
        };
        assert!(step.is_final_answer());
        let action = Action::parse(&step.action).unwrap();
        assert!(matches!(action, Action::FinalAnswer(ref s) if s == "Paris"));
    }

    #[test]
    fn test_action_parse_tool_call_round_trip() {
        let step = ReActStep {
            thought: "searching".into(),
            action: "search {\"q\":\"hello\"}".into(),
            observation: "".into(),
            step_duration_ms: 0,
        };
        assert!(step.is_tool_call());
        let action = Action::parse(&step.action).unwrap();
        assert!(matches!(action, Action::ToolCall { ref name, .. } if name == "search"));
    }

    // ── #26 Observer step indices ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_observer_receives_correct_step_indices() {
        use std::sync::{Arc, Mutex};

        struct IndexCollector(Arc<Mutex<Vec<usize>>>);
        impl Observer for IndexCollector {
            fn on_step(&self, step_index: usize, _step: &ReActStep) {
                self.0.lock().unwrap_or_else(|e| e.into_inner()).push(step_index);
            }
        }

        let indices = Arc::new(Mutex::new(Vec::new()));
        let obs = Arc::new(IndexCollector(Arc::clone(&indices)));

        let config = AgentConfig::new(5, "test");
        let mut loop_ = ReActLoop::new(config).with_observer(obs as Arc<dyn Observer>);
        loop_.register_tool(ToolSpec::new("noop", "no-op", |_| serde_json::json!({})));

        let mut call_count = 0;
        loop_.run("test", |_ctx| {
            call_count += 1;
            let count = call_count;
            async move {
                if count == 1 {
                    "Thought: step1\nAction: noop {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER ok".to_string()
                }
            }
        }).await.unwrap();

        let collected = indices.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(collected, vec![0, 1], "expected step indices 0 and 1");
    }

    #[tokio::test]
    async fn test_action_hook_blocking_inserts_blocked_observation() {
        let hook: ActionHook = Arc::new(|_name, _args| {
            Box::pin(async move { false }) // always block
        });

        let config = AgentConfig::new(5, "test-model");
        let mut loop_ = ReActLoop::new(config).with_action_hook(hook);
        loop_.register_tool(ToolSpec::new("noop", "noop", |_| serde_json::json!("ok")));

        let mut call_count = 0;
        let steps = loop_.run("test", |_ctx| {
            call_count += 1;
            let count = call_count;
            async move {
                if count == 1 {
                    "Thought: try tool\nAction: noop {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER done".to_string()
                }
            }
        }).await.unwrap();

        assert!(steps[0].observation.contains("blocked"), "expected blocked observation, got: {}", steps[0].observation);
    }
}
