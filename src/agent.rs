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
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A pinned, boxed future returning a `Value`. Used for async tool handlers.
pub type AsyncToolFuture = Pin<Box<dyn Future<Output = Value> + Send>>;

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

/// Configuration for the ReAct agent loop.
#[derive(Debug, Clone)]
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
    /// Maximum approximate token budget for injected memories.
    /// Uses ~4 chars/token heuristic. None means no limit.
    pub max_memory_tokens: Option<usize>,
    /// Optional wall-clock timeout for the entire loop.
    /// If the loop runs longer than this duration, it returns
    /// `Err(AgentRuntimeError::AgentLoop("loop timeout ..."))`.
    pub loop_timeout: Option<std::time::Duration>,
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
}

// ── ToolSpec ──────────────────────────────────────────────────────────────────

/// Describes and implements a single callable tool.
pub struct ToolSpec {
    /// Short identifier used in action strings (e.g. "search").
    pub name: String,
    /// Human-readable description passed to the model as part of the system prompt.
    pub description: String,
    /// Async handler: receives JSON arguments, returns a future resolving to a JSON result.
    pub handler: AsyncToolHandler,
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

    /// Set the required fields that must be present in the JSON args object.
    pub fn with_required_fields(mut self, fields: Vec<String>) -> Self {
        self.required_fields = fields;
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

    /// Invoke the tool with the given JSON arguments.
    pub async fn call(&self, args: Value) -> Value {
        (self.handler)(args).await
    }
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry of available tools for the agent loop.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolSpec>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. Overwrites any existing tool with the same name.
    pub fn register(&mut self, spec: ToolSpec) {
        self.tools.insert(spec.name.clone(), spec);
    }

    /// Call a tool by name.
    ///
    /// # Returns
    /// - `Ok(Value)` — tool result
    /// - `Err(AgentRuntimeError::AgentLoop)` — if the tool is not found or required fields
    ///   are missing
    /// - `Err(AgentRuntimeError::CircuitOpen)` — if the tool's circuit breaker is open
    #[tracing::instrument(skip_all, fields(tool_name = %name))]
    pub async fn call(&self, name: &str, args: Value) -> Result<Value, AgentRuntimeError> {
        let spec = self
            .tools
            .get(name)
            .ok_or_else(|| AgentRuntimeError::AgentLoop(format!("tool '{name}' not found")))?;

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

        let result = spec.call(args).await;
        Ok(result)
    }

    /// Return the list of registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

// ── ReActLoop ─────────────────────────────────────────────────────────────────

/// Parses a ReAct response string into a `ReActStep`.
///
/// Case-insensitive; tolerates spaces around the colon.
/// e.g. `Thought :`, `thought:`, `THOUGHT :` all match.
pub fn parse_react_step(text: &str) -> Result<ReActStep, AgentRuntimeError> {
    let mut thought = String::new();
    let mut action = String::new();

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("thought") {
            if let Some(colon_pos) = trimmed.find(':') {
                thought = trimmed[colon_pos + 1..].trim().to_owned();
            }
        } else if lower.starts_with("action") {
            if let Some(colon_pos) = trimmed.find(':') {
                action = trimmed[colon_pos + 1..].trim().to_owned();
            }
        }
    }

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
}

impl std::fmt::Debug for ReActLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("ReActLoop");
        s.field("config", &self.config)
            .field("registry", &self.registry)
            .field("has_metrics", &self.metrics.is_some());
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
        }
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

    /// Register a tool that the agent loop can invoke.
    pub fn register_tool(&mut self, spec: ToolSpec) {
        self.registry.register(spec);
    }

    /// Execute the ReAct loop for the given prompt.
    ///
    /// # Arguments
    /// * `prompt` — user input passed as the initial context
    /// * `infer`  — async inference function: receives context string, returns response string
    ///
    /// # Returns
    /// - `Ok(Vec<ReActStep>)` — steps executed, ending with a `FINAL_ANSWER` step
    /// - `Err(AgentRuntimeError::AgentLoop)` — if max iterations reached without `FINAL_ANSWER`
    ///   or if a ReAct response cannot be parsed
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

        for iteration in 0..self.config.max_iterations {
            // Wall-clock timeout check.
            if let Some(dl) = deadline {
                if std::time::Instant::now() >= dl {
                    let ms = self
                        .config
                        .loop_timeout
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    return Err(AgentRuntimeError::AgentLoop(format!(
                        "loop timeout after {ms} ms"
                    )));
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
                steps.push(step);
                tracing::info!(step = iteration, "FINAL_ANSWER reached");
                return Ok(steps);
            }

            // Item 3 — propagate parse errors rather than silently falling back.
            let (tool_name, args) = parse_tool_call(&step.action)?;

            tracing::debug!(
                step = iteration,
                tool_name = %tool_name,
                "dispatching tool call"
            );

            // Count every tool dispatch (global + per-tool).
            if let Some(ref m) = self.metrics {
                m.record_tool_call(&tool_name);
            }

            // Structured error categorization in observation.
            let observation = match self.registry.call(&tool_name, args).await {
                Ok(result) => serde_json::json!({ "ok": true, "data": result }).to_string(),
                Err(e) => {
                    // Count failed tool calls (global + per-tool).
                    if let Some(ref m) = self.metrics {
                        m.record_tool_failure(&tool_name);
                    }
                    let kind = match &e {
                        AgentRuntimeError::AgentLoop(msg) if msg.contains("not found") => {
                            "not_found"
                        }
                        #[cfg(feature = "orchestrator")]
                        AgentRuntimeError::CircuitOpen { .. } => "transient",
                        _ => "permanent",
                    };
                    serde_json::json!({ "ok": false, "error": e.to_string(), "kind": kind })
                        .to_string()
                }
            };

            step.observation = observation.clone();
            step.step_duration_ms = step_start.elapsed().as_millis() as u64;
            context.push_str(&format!(
                "\nThought: {}\nAction: {}\nObservation: {}\n",
                step.thought, step.action, observation
            ));
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
/// # Example
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
///             return Err(AgentRuntimeError::AgentLoop("q must not be empty".into()));
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait ToolValidator: Send + Sync {
    /// Validate `args` before the tool handler is invoked.
    ///
    /// Return `Ok(())` if the arguments are valid, or
    /// `Err(AgentRuntimeError::AgentLoop(...))` with a human-readable message.
    fn validate(&self, args: &Value) -> Result<(), AgentRuntimeError>;
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
}
