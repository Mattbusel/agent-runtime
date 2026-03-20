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

    /// Return `true` if the message content is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Return the approximate number of whitespace-separated words in the content.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
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
    /// Construct a step with zero `step_duration_ms`.
    ///
    /// Primarily useful in tests that need to build [`AgentSession`] values
    /// without running the full ReAct loop.
    pub fn new(
        thought: impl Into<String>,
        action: impl Into<String>,
        observation: impl Into<String>,
    ) -> Self {
        Self {
            thought: thought.into(),
            action: action.into(),
            observation: observation.into(),
            step_duration_ms: 0,
        }
    }

    /// Returns `true` if this step's action is a `FINAL_ANSWER`.
    pub fn is_final_answer(&self) -> bool {
        self.action.trim().to_ascii_uppercase().starts_with("FINAL_ANSWER")
    }

    /// Returns `true` if this step's action is a tool call (not a FINAL_ANSWER).
    pub fn is_tool_call(&self) -> bool {
        !self.is_final_answer() && !self.action.trim().is_empty()
    }

    /// Set the `step_duration_ms` field, returning `self` for chaining.
    ///
    /// Useful in tests and benchmarks that need to build `AgentSession` values
    /// with realistic timings without running the full ReAct loop.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.step_duration_ms = ms;
        self
    }

    /// Return `true` if all three text fields are empty strings.
    pub fn is_empty(&self) -> bool {
        self.thought.is_empty() && self.action.is_empty() && self.observation.is_empty()
    }

    /// Return `true` if the observation string is empty.
    ///
    /// Useful for identifying steps where the tool produced no output.
    pub fn observation_is_empty(&self) -> bool {
        self.observation.is_empty()
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

    /// Set the maximum number of ReAct iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Return the configured maximum number of ReAct iterations.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
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

    /// Clone this config with only the `model` field changed.
    ///
    /// Useful when the same configuration is shared across multiple agents that
    /// differ only in the model used for inference.
    pub fn clone_with_model(&self, model: impl Into<String>) -> Self {
        let mut copy = self.clone();
        copy.model = model.into();
        copy
    }

    /// Set stop sequences for inference requests.
    ///
    /// The model will stop generating when it produces any of these strings.
    /// Defaults to an empty list (no stop sequences).
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }

    /// Return `true` if this configuration is logically valid.
    ///
    /// Specifically, `max_iterations` must be at least 1 and `model` must be
    /// a non-empty string.
    pub fn is_valid(&self) -> bool {
        self.max_iterations >= 1 && !self.model.is_empty()
    }

    /// Validate the configuration, returning a structured error on failure.
    ///
    /// Checks the same invariants as [`is_valid`] but returns
    /// `Err(AgentRuntimeError::AgentLoop)` with a descriptive message instead
    /// of `false`, making it more useful in `?`-propagation chains.
    ///
    /// [`is_valid`]: AgentConfig::is_valid
    pub fn validate(&self) -> Result<(), crate::error::AgentRuntimeError> {
        if self.max_iterations == 0 {
            return Err(crate::error::AgentRuntimeError::AgentLoop(
                "AgentConfig: max_iterations must be >= 1".into(),
            ));
        }
        if self.model.is_empty() {
            return Err(crate::error::AgentRuntimeError::AgentLoop(
                "AgentConfig: model must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Return `true` if a loop timeout has been configured.
    pub fn has_loop_timeout(&self) -> bool {
        self.loop_timeout.is_some()
    }

    /// Return `true` if at least one stop sequence has been configured.
    pub fn has_stop_sequences(&self) -> bool {
        !self.stop_sequences.is_empty()
    }

    /// Return the number of configured stop sequences.
    pub fn stop_sequence_count(&self) -> usize {
        self.stop_sequences.len()
    }

    /// Return `true` if the agent runs at most one iteration.
    ///
    /// A single-shot agent executes exactly one Thought-Action-Observation
    /// cycle and then terminates regardless of whether a `FINAL_ANSWER` was
    /// produced.
    pub fn is_single_shot(&self) -> bool {
        self.max_iterations == 1
    }

    /// Return `true` if a sampling temperature has been configured.
    pub fn has_temperature(&self) -> bool {
        self.temperature.is_some()
    }

    /// Return the configured sampling temperature, if any.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Return the configured maximum output tokens, if any.
    pub fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }

    /// Return `true` if a per-inference request timeout has been configured.
    pub fn has_request_timeout(&self) -> bool {
        self.request_timeout.is_some()
    }

    /// Return the configured per-inference request timeout, if any.
    pub fn request_timeout(&self) -> Option<std::time::Duration> {
        self.request_timeout
    }

    /// Return `true` if a maximum context character limit has been configured.
    pub fn has_max_context_chars(&self) -> bool {
        self.max_context_chars.is_some()
    }

    /// Return the configured maximum context character limit, if any.
    pub fn max_context_chars(&self) -> Option<usize> {
        self.max_context_chars
    }

    /// Return the number of iterations still available after `n` have been completed.
    ///
    /// Uses saturating subtraction so values beyond `max_iterations` return `0`
    /// rather than wrapping.
    pub fn remaining_iterations_after(&self, n: usize) -> usize {
        self.max_iterations.saturating_sub(n)
    }

    /// Return the configured system prompt string.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Return the model name this config targets.
    pub fn model(&self) -> &str {
        &self.model
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

    /// Override the tool name after construction.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Return the number of required fields configured for this tool.
    pub fn required_field_count(&self) -> usize {
        self.required_fields.len()
    }

    /// Return `true` if this tool requires at least one field to be present in its args.
    pub fn has_required_fields(&self) -> bool {
        !self.required_fields.is_empty()
    }

    /// Return `true` if this tool has at least one custom argument validator attached.
    pub fn has_validators(&self) -> bool {
        !self.validators.is_empty()
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

    /// Return the configured maximum capacity, or `None` if the cache is unbounded.
    pub fn capacity(&self) -> Option<usize> {
        self.max_entries
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

    /// Return `true` if a tool with the given name is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Remove a tool by name.  Returns `true` if the tool was registered.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Return the list of registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Return all registered tool names as owned `String`s.
    ///
    /// Unlike [`tool_names`] this does not borrow `self`, making the result
    /// usable after the registry is moved or mutated.
    ///
    /// [`tool_names`]: ToolRegistry::tool_names
    pub fn tool_names_owned(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Return all registered tool names sorted alphabetically.
    ///
    /// Useful for deterministic output in help text, diagnostics, or tests.
    pub fn all_tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Return references to all registered `ToolSpec`s.
    pub fn tool_specs(&self) -> Vec<&ToolSpec> {
        self.tools.values().collect()
    }

    /// Return references to all `ToolSpec`s that satisfy the given predicate.
    ///
    /// # Example
    /// ```rust
    /// # use llm_agent_runtime::agent::ToolRegistry;
    /// let registry = ToolRegistry::new();
    /// let long_desc: Vec<_> = registry.filter_tools(|s| s.description.len() > 20);
    /// ```
    pub fn filter_tools<F: Fn(&ToolSpec) -> bool>(&self, pred: F) -> Vec<&ToolSpec> {
        self.tools.values().filter(|s| pred(s)).collect()
    }

    /// Rename a registered tool from `old_name` to `new_name`.
    ///
    /// The tool's `name` field and its registry key are both updated.
    /// Returns `true` if the tool was found and renamed, `false` if `old_name`
    /// is not registered.  If `new_name` is already registered, it is
    /// overwritten.
    pub fn rename_tool(&mut self, old_name: &str, new_name: impl Into<String>) -> bool {
        let Some(mut spec) = self.tools.remove(old_name) else {
            return false;
        };
        let new_name = new_name.into();
        spec.name = new_name.clone();
        self.tools.insert(new_name, spec);
        true
    }

    /// Return the number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Return `true` if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
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

    /// Return `(name, description)` pairs for all registered tools, sorted by name.
    ///
    /// Useful for generating help text or logging the tool set at startup.
    pub fn descriptions(&self) -> Vec<(&str, &str)> {
        let mut pairs: Vec<(&str, &str)> = self
            .tools
            .values()
            .map(|s| (s.name.as_str(), s.description.as_str()))
            .collect();
        pairs.sort_unstable_by_key(|(name, _)| *name);
        pairs
    }

    /// Return references to all tool specs whose description contains
    /// `keyword` (case-insensitive).
    pub fn find_by_description_keyword(&self, keyword: &str) -> Vec<&ToolSpec> {
        let lower = keyword.to_ascii_lowercase();
        self.tools
            .values()
            .filter(|s| s.description.to_ascii_lowercase().contains(&lower))
            .collect()
    }

    /// Return the number of tools that have at least one required field.
    pub fn tool_count_with_required_fields(&self) -> usize {
        self.tools.values().filter(|s| s.has_required_fields()).count()
    }

    /// Return the names of all registered tools, sorted alphabetically.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|k| k.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Return the names of all registered tools whose name starts with `prefix`,
    /// sorted alphabetically.
    pub fn tool_names_starting_with(&self, prefix: &str) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .tools
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(|k| k.as_str())
            .collect();
        names.sort_unstable();
        names
    }

    /// Return the description of the tool with the given `name`, or `None` if
    /// no such tool is registered.
    pub fn description_for(&self, name: &str) -> Option<&str> {
        self.tools.get(name).map(|s| s.description.as_str())
    }

    /// Return the count of tools whose description contains `keyword`
    /// (case-insensitive).
    pub fn count_with_description_containing(&self, keyword: &str) -> usize {
        let lower = keyword.to_ascii_lowercase();
        self.tools
            .values()
            .filter(|s| s.description.to_ascii_lowercase().contains(&lower))
            .count()
    }

    /// Remove all registered tools.
    pub fn unregister_all(&mut self) {
        self.tools.clear();
    }

    /// Return the names of tools whose description contains `keyword` (case-insensitive).
    pub fn tool_names_with_keyword(&self, keyword: &str) -> Vec<&str> {
        let kw = keyword.to_ascii_lowercase();
        self.tools
            .values()
            .filter(|s| s.description.to_ascii_lowercase().contains(&kw))
            .map(|s| s.name.as_str())
            .collect()
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

    /// Return the number of tools currently registered.
    ///
    /// Shorthand for `loop_.registry().tool_count()`.
    pub fn tool_count(&self) -> usize {
        self.registry.tool_count()
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

    #[test]
    fn test_react_step_new_constructor() {
        let s = ReActStep::new("think", "act", "obs");
        assert_eq!(s.thought, "think");
        assert_eq!(s.action, "act");
        assert_eq!(s.observation, "obs");
        assert_eq!(s.step_duration_ms, 0);
    }

    #[test]
    fn test_react_step_new_is_tool_call() {
        let s = ReActStep::new("think", "search {}", "result");
        assert!(s.is_tool_call());
        assert!(!s.is_final_answer());
    }

    #[test]
    fn test_react_step_new_is_final_answer() {
        let s = ReActStep::new("done", "FINAL_ANSWER 42", "");
        assert!(s.is_final_answer());
        assert!(!s.is_tool_call());
    }

    #[test]
    fn test_agent_config_is_valid_with_valid_config() {
        let cfg = AgentConfig::new(5, "my-model");
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_agent_config_is_valid_with_zero_iterations() {
        let mut cfg = AgentConfig::new(1, "my-model");
        cfg.max_iterations = 0;
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_agent_config_is_valid_with_empty_model() {
        let mut cfg = AgentConfig::new(5, "my-model");
        cfg.model = String::new();
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_react_loop_tool_count_delegates_to_registry() {
        let cfg = AgentConfig::new(5, "model");
        let mut loop_ = ReActLoop::new(cfg);
        assert_eq!(loop_.tool_count(), 0);
        loop_.register_tool(ToolSpec::new("t1", "desc", |_| serde_json::json!("ok")));
        loop_.register_tool(ToolSpec::new("t2", "desc", |_| serde_json::json!("ok")));
        assert_eq!(loop_.tool_count(), 2);
    }

    #[test]
    fn test_tool_registry_has_tool_returns_true_when_registered() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("my-tool", "desc", |_| serde_json::json!("ok")));
        assert!(reg.has_tool("my-tool"));
        assert!(!reg.has_tool("other-tool"));
    }

    // ── Round 3: AgentConfig::validate ───────────────────────────────────────

    #[test]
    fn test_agent_config_validate_ok_for_valid_config() {
        let cfg = AgentConfig::new(5, "my-model");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_agent_config_validate_err_for_zero_iterations() {
        let cfg = AgentConfig::new(0, "my-model");
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("max_iterations"));
    }

    #[test]
    fn test_agent_config_validate_err_for_empty_model() {
        let cfg = AgentConfig::new(5, "");
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("model"));
    }

    // ── Round 4: AgentConfig::clone_with_model / ToolSpec::with_name ─────────

    #[test]
    fn test_clone_with_model_produces_new_model_string() {
        let cfg = AgentConfig::new(5, "gpt-4");
        let new_cfg = cfg.clone_with_model("claude-3");
        assert_eq!(new_cfg.model, "claude-3");
        // Original unchanged
        assert_eq!(cfg.model, "gpt-4");
    }

    #[test]
    fn test_clone_with_model_preserves_other_fields() {
        let cfg = AgentConfig::new(10, "gpt-4").with_stop_sequences(vec!["STOP".to_string()]);
        let new_cfg = cfg.clone_with_model("o1");
        assert_eq!(new_cfg.max_iterations, 10);
        assert_eq!(new_cfg.stop_sequences, cfg.stop_sequences);
    }

    #[tokio::test]
    async fn test_tool_spec_with_name_changes_name() {
        let spec = ToolSpec::new("original", "desc", |_| serde_json::json!("ok"))
            .with_name("renamed");
        assert_eq!(spec.name, "renamed");
    }

    #[tokio::test]
    async fn test_tool_spec_with_name_and_description_chainable() {
        let spec = ToolSpec::new("old", "old desc", |_| serde_json::json!("ok"))
            .with_name("new")
            .with_description("new desc");
        assert_eq!(spec.name, "new");
        assert_eq!(spec.description, "new desc");
    }

    // ── Round 16: Message constructors, parse_react_step ─────────────────────

    #[test]
    fn test_message_user_sets_role_and_content() {
        let m = Message::user("hello");
        assert_eq!(m.content(), "hello");
        assert!(m.is_user());
        assert!(!m.is_assistant());
    }

    #[test]
    fn test_message_assistant_sets_role() {
        let m = Message::assistant("reply");
        assert!(m.is_assistant());
        assert!(!m.is_user());
        assert!(!m.is_system());
    }

    #[test]
    fn test_message_system_sets_role() {
        let m = Message::system("system prompt");
        assert!(m.is_system());
        assert_eq!(m.content(), "system prompt");
    }

    #[test]
    fn test_parse_react_step_valid_input() {
        let text = "Thought: I need to search\nAction: search[query]";
        let step = parse_react_step(text).unwrap();
        assert!(step.thought.contains("search"));
        assert!(step.action.contains("search"));
    }

    #[test]
    fn test_parse_react_step_missing_fields_returns_err() {
        let text = "no structured content here";
        assert!(parse_react_step(text).is_err());
    }

    // ── Round 18: ReActStep predicates, ToolRegistry ops, AgentConfig builders

    #[test]
    fn test_react_step_is_final_answer_true() {
        let step = ReActStep::new("t", "FINAL_ANSWER Paris", "");
        assert!(step.is_final_answer());
        assert!(!step.is_tool_call());
    }

    #[test]
    fn test_react_step_is_tool_call_true() {
        let step = ReActStep::new("t", "search {}", "result");
        assert!(step.is_tool_call());
        assert!(!step.is_final_answer());
    }

    #[test]
    fn test_tool_registry_unregister_returns_true_when_present() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("tool-x", "desc", |_| serde_json::json!("ok")));
        assert!(reg.unregister("tool-x"));
        assert!(!reg.has_tool("tool-x"));
    }

    #[test]
    fn test_tool_registry_unregister_returns_false_when_absent() {
        let mut reg = ToolRegistry::new();
        assert!(!reg.unregister("ghost"));
    }

    #[test]
    fn test_tool_registry_contains_matches_has_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("alpha", "desc", |_| serde_json::json!("ok")));
        assert!(reg.contains("alpha"));
        assert!(!reg.contains("beta"));
    }

    #[test]
    fn test_agent_config_with_system_prompt() {
        let cfg = AgentConfig::new(5, "model")
            .with_system_prompt("You are helpful.");
        assert_eq!(cfg.system_prompt, "You are helpful.");
    }

    #[test]
    fn test_agent_config_with_temperature_and_max_tokens() {
        let cfg = AgentConfig::new(3, "model")
            .with_temperature(0.7)
            .with_max_tokens(512);
        assert!((cfg.temperature.unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(cfg.max_tokens, Some(512));
    }

    #[test]
    fn test_agent_config_clone_with_model() {
        let orig = AgentConfig::new(5, "gpt-4");
        let cloned = orig.clone_with_model("claude-3");
        assert_eq!(cloned.model, "claude-3");
        assert_eq!(cloned.max_iterations, 5);
    }

    // ── Round 19: more AgentConfig builder methods, Message::is_tool ─────────

    #[test]
    fn test_agent_config_with_loop_timeout_secs() {
        let cfg = AgentConfig::new(5, "model").with_loop_timeout_secs(30);
        assert_eq!(cfg.loop_timeout, Some(std::time::Duration::from_secs(30)));
    }

    #[test]
    fn test_agent_config_with_max_context_chars() {
        let cfg = AgentConfig::new(5, "model").with_max_context_chars(4096);
        assert_eq!(cfg.max_context_chars, Some(4096));
    }

    #[test]
    fn test_agent_config_with_stop_sequences() {
        let cfg = AgentConfig::new(5, "model")
            .with_stop_sequences(vec!["STOP".to_string(), "END".to_string()]);
        assert_eq!(cfg.stop_sequences, vec!["STOP", "END"]);
    }

    #[test]
    fn test_message_is_tool_false_for_non_tool_roles() {
        assert!(!Message::user("hi").is_tool());
        assert!(!Message::assistant("reply").is_tool());
        assert!(!Message::system("prompt").is_tool());
    }

    // ── Round 6: AgentConfig::with_max_iterations / ToolRegistry::tool_names_owned

    #[test]
    fn test_agent_config_with_max_iterations() {
        let cfg = AgentConfig::new(5, "m").with_max_iterations(20);
        assert_eq!(cfg.max_iterations, 20);
    }

    #[test]
    fn test_tool_registry_tool_names_owned_returns_strings() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("alpha", "d", |_| serde_json::json!("ok")));
        reg.register(ToolSpec::new("beta", "d", |_| serde_json::json!("ok")));
        let mut names = reg.tool_names_owned();
        names.sort();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn test_tool_registry_tool_names_owned_empty_when_no_tools() {
        let reg = ToolRegistry::new();
        assert!(reg.tool_names_owned().is_empty());
    }

    // ── Round 7: ToolRegistry::tool_specs ────────────────────────────────────

    #[test]
    fn test_tool_registry_tool_specs_returns_all_specs() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "desc1", |_| serde_json::json!("ok")));
        reg.register(ToolSpec::new("t2", "desc2", |_| serde_json::json!("ok")));
        let specs = reg.tool_specs();
        assert_eq!(specs.len(), 2);
    }

    #[test]
    fn test_tool_registry_tool_specs_empty_when_no_tools() {
        let reg = ToolRegistry::new();
        assert!(reg.tool_specs().is_empty());
    }

    // ── Round 8: ToolRegistry::rename_tool ───────────────────────────────────

    #[test]
    fn test_rename_tool_updates_name_and_key() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("old", "desc", |_| serde_json::json!("ok")));
        assert!(reg.rename_tool("old", "new"));
        assert!(reg.has_tool("new"));
        assert!(!reg.has_tool("old"));
        let spec = reg.get("new").unwrap();
        assert_eq!(spec.name, "new");
    }

    #[test]
    fn test_rename_tool_returns_false_for_unknown_name() {
        let mut reg = ToolRegistry::new();
        assert!(!reg.rename_tool("ghost", "other"));
    }

    // ── Round 9: filter_tools ─────────────────────────────────────────────────

    #[test]
    fn test_filter_tools_returns_matching_specs() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("short_desc", "hi", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("long_desc", "a longer description here", |_| serde_json::json!({})));
        let long_ones = reg.filter_tools(|s| s.description.len() > 10);
        assert_eq!(long_ones.len(), 1);
        assert_eq!(long_ones[0].name, "long_desc");
    }

    #[test]
    fn test_filter_tools_returns_empty_when_none_match() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "desc", |_| serde_json::json!({})));
        let none: Vec<_> = reg.filter_tools(|_| false);
        assert!(none.is_empty());
    }

    #[test]
    fn test_filter_tools_returns_all_when_predicate_always_true() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("a", "d1", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("b", "d2", |_| serde_json::json!({})));
        let all = reg.filter_tools(|_| true);
        assert_eq!(all.len(), 2);
    }

    // ── Round 10: AgentConfig::max_iterations getter ──────────────────────────

    #[test]
    fn test_agent_config_max_iterations_getter_returns_configured_value() {
        let cfg = AgentConfig::new(5, "model-x");
        assert_eq!(cfg.max_iterations(), 5);
    }

    #[test]
    fn test_agent_config_with_max_iterations_updates_getter() {
        let cfg = AgentConfig::new(3, "m").with_max_iterations(10);
        assert_eq!(cfg.max_iterations(), 10);
    }

    // ── Round 11: ToolRegistry::is_empty / clear / remove ─────────────────────

    #[test]
    fn test_tool_registry_is_empty_true_when_new() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_tool_registry_is_empty_false_after_register() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t", "d", |_| serde_json::json!({})));
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_tool_registry_clear_empties_registry() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "d", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("t2", "d", |_| serde_json::json!({})));
        reg.clear();
        assert!(reg.is_empty());
        assert_eq!(reg.tool_count(), 0);
    }

    #[test]
    fn test_tool_registry_remove_returns_spec_and_decrements_count() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("myTool", "desc", |_| serde_json::json!({})));
        assert_eq!(reg.tool_count(), 1);
        let removed = reg.remove("myTool");
        assert!(removed.is_some());
        assert_eq!(reg.tool_count(), 0);
    }

    #[test]
    fn test_tool_registry_remove_returns_none_for_absent_tool() {
        let mut reg = ToolRegistry::new();
        assert!(reg.remove("ghost").is_none());
    }

    // ── Round 12: all_tool_names ──────────────────────────────────────────────

    #[test]
    fn test_all_tool_names_returns_sorted_names() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("zebra", "d", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("apple", "d", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("mango", "d", |_| serde_json::json!({})));
        let names = reg.all_tool_names();
        assert_eq!(names, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn test_all_tool_names_empty_for_empty_registry() {
        let reg = ToolRegistry::new();
        assert!(reg.all_tool_names().is_empty());
    }

    // ── Round 13: AgentConfig::remaining_iterations_after, ToolSpec predicates ──

    #[test]
    fn test_remaining_iterations_after_full_budget() {
        let cfg = AgentConfig::new(10, "m");
        assert_eq!(cfg.remaining_iterations_after(0), 10);
    }

    #[test]
    fn test_remaining_iterations_after_partial_use() {
        let cfg = AgentConfig::new(10, "m");
        assert_eq!(cfg.remaining_iterations_after(3), 7);
    }

    #[test]
    fn test_remaining_iterations_after_saturates_at_zero() {
        let cfg = AgentConfig::new(5, "m");
        assert_eq!(cfg.remaining_iterations_after(10), 0);
    }

    #[test]
    fn test_tool_spec_required_field_count_zero_by_default() {
        let spec = ToolSpec::new("t", "d", |_| serde_json::json!({}));
        assert_eq!(spec.required_field_count(), 0);
    }

    #[test]
    fn test_tool_spec_required_field_count_after_adding() {
        let spec = ToolSpec::new("t", "d", |_| serde_json::json!({}))
            .with_required_fields(["query", "limit"]);
        assert_eq!(spec.required_field_count(), 2);
    }

    #[test]
    fn test_tool_spec_has_required_fields_false_by_default() {
        let spec = ToolSpec::new("t", "d", |_| serde_json::json!({}));
        assert!(!spec.has_required_fields());
    }

    #[test]
    fn test_tool_spec_has_required_fields_true_after_adding() {
        let spec = ToolSpec::new("t", "d", |_| serde_json::json!({}))
            .with_required_fields(["key"]);
        assert!(spec.has_required_fields());
    }

    #[test]
    fn test_tool_spec_has_validators_false_by_default() {
        let spec = ToolSpec::new("t", "d", |_| serde_json::json!({}));
        assert!(!spec.has_validators());
    }

    // ── Round 14: ToolRegistry::contains, descriptions, tool_count ───────────

    #[test]
    fn test_tool_registry_contains_true_for_registered_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("search", "d", |_| serde_json::json!({})));
        assert!(reg.contains("search"));
    }

    #[test]
    fn test_tool_registry_contains_false_for_unknown_tool() {
        let reg = ToolRegistry::new();
        assert!(!reg.contains("missing"));
    }

    #[test]
    fn test_tool_registry_descriptions_sorted_by_name() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("zebra", "z-desc", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("apple", "a-desc", |_| serde_json::json!({})));
        let descs = reg.descriptions();
        assert_eq!(descs[0], ("apple", "a-desc"));
        assert_eq!(descs[1], ("zebra", "z-desc"));
    }

    #[test]
    fn test_tool_registry_descriptions_empty_when_no_tools() {
        let reg = ToolRegistry::new();
        assert!(reg.descriptions().is_empty());
    }

    #[test]
    fn test_tool_registry_tool_count_increments_on_register() {
        let mut reg = ToolRegistry::new();
        assert_eq!(reg.tool_count(), 0);
        reg.register(ToolSpec::new("t1", "d", |_| serde_json::json!({})));
        assert_eq!(reg.tool_count(), 1);
        reg.register(ToolSpec::new("t2", "d", |_| serde_json::json!({})));
        assert_eq!(reg.tool_count(), 2);
    }

    // ── Round 16: ReActStep::observation_is_empty ─────────────────────────────

    #[test]
    fn test_observation_is_empty_true_for_empty_string() {
        let step = ReActStep::new("think", "search", "");
        assert!(step.observation_is_empty());
    }

    #[test]
    fn test_observation_is_empty_false_for_non_empty() {
        let step = ReActStep::new("think", "search", "found results");
        assert!(!step.observation_is_empty());
    }

    // ── Round 17: AgentConfig::temperature / max_tokens / request_timeout ────

    #[test]
    fn test_agent_config_temperature_getter_none_by_default() {
        let cfg = AgentConfig::new(5, "gpt-4");
        assert!(cfg.temperature().is_none());
    }

    #[test]
    fn test_agent_config_temperature_getter_some_when_set() {
        let cfg = AgentConfig::new(5, "gpt-4").with_temperature(0.7);
        assert!((cfg.temperature().unwrap() - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_agent_config_max_tokens_getter_none_by_default() {
        let cfg = AgentConfig::new(5, "gpt-4");
        assert!(cfg.max_tokens().is_none());
    }

    #[test]
    fn test_agent_config_max_tokens_getter_some_when_set() {
        let cfg = AgentConfig::new(5, "gpt-4").with_max_tokens(512);
        assert_eq!(cfg.max_tokens(), Some(512));
    }

    #[test]
    fn test_agent_config_request_timeout_getter_none_by_default() {
        let cfg = AgentConfig::new(5, "gpt-4");
        assert!(cfg.request_timeout().is_none());
    }

    #[test]
    fn test_agent_config_request_timeout_getter_some_when_set() {
        let cfg = AgentConfig::new(5, "gpt-4")
            .with_request_timeout(std::time::Duration::from_secs(10));
        assert_eq!(cfg.request_timeout(), Some(std::time::Duration::from_secs(10)));
    }

    // ── Round 22: has_max_context_chars, max_context_chars, system_prompt, model

    #[test]
    fn test_agent_config_has_max_context_chars_false_by_default() {
        let cfg = AgentConfig::new(5, "gpt-4");
        assert!(!cfg.has_max_context_chars());
    }

    #[test]
    fn test_agent_config_has_max_context_chars_true_after_setting() {
        let cfg = AgentConfig::new(5, "gpt-4").with_max_context_chars(8192);
        assert!(cfg.has_max_context_chars());
    }

    #[test]
    fn test_agent_config_max_context_chars_none_by_default() {
        let cfg = AgentConfig::new(5, "gpt-4");
        assert_eq!(cfg.max_context_chars(), None);
    }

    #[test]
    fn test_agent_config_max_context_chars_some_after_setting() {
        let cfg = AgentConfig::new(5, "gpt-4").with_max_context_chars(4096);
        assert_eq!(cfg.max_context_chars(), Some(4096));
    }

    #[test]
    fn test_agent_config_system_prompt_returns_configured_prompt() {
        let cfg = AgentConfig::new(5, "gpt-4").with_system_prompt("Be concise.");
        assert_eq!(cfg.system_prompt(), "Be concise.");
    }

    #[test]
    fn test_agent_config_model_returns_configured_model() {
        let cfg = AgentConfig::new(5, "claude-3");
        assert_eq!(cfg.model(), "claude-3");
    }

    // ── Round 23: Message::is_system, word_count; AgentConfig flags ───────────

    #[test]
    fn test_message_is_system_true_for_system_role() {
        let m = Message::system("context");
        assert!(m.is_system());
    }

    #[test]
    fn test_message_is_system_false_for_user_role() {
        let m = Message::user("hello");
        assert!(!m.is_system());
    }

    #[test]
    fn test_message_word_count_counts_whitespace_words() {
        let m = Message::user("hello world foo");
        assert_eq!(m.word_count(), 3);
    }

    #[test]
    fn test_message_word_count_zero_for_empty_content() {
        let m = Message::user("");
        assert_eq!(m.word_count(), 0);
    }

    #[test]
    fn test_agent_config_has_loop_timeout_false_by_default() {
        let cfg = AgentConfig::new(5, "m");
        assert!(!cfg.has_loop_timeout());
    }

    #[test]
    fn test_agent_config_has_loop_timeout_true_after_setting() {
        let cfg = AgentConfig::new(5, "m")
            .with_loop_timeout(std::time::Duration::from_secs(30));
        assert!(cfg.has_loop_timeout());
    }

    #[test]
    fn test_agent_config_has_stop_sequences_false_by_default() {
        let cfg = AgentConfig::new(5, "m");
        assert!(!cfg.has_stop_sequences());
    }

    #[test]
    fn test_agent_config_has_stop_sequences_true_after_adding() {
        let cfg = AgentConfig::new(5, "m").with_stop_sequences(vec!["STOP".to_string()]);
        assert!(cfg.has_stop_sequences());
    }

    #[test]
    fn test_agent_config_is_single_shot_true_when_max_iterations_one() {
        let cfg = AgentConfig::new(1, "m");
        assert!(cfg.is_single_shot());
    }

    #[test]
    fn test_agent_config_is_single_shot_false_when_max_iterations_gt_one() {
        let cfg = AgentConfig::new(5, "m");
        assert!(!cfg.is_single_shot());
    }

    #[test]
    fn test_agent_config_has_temperature_false_by_default() {
        let cfg = AgentConfig::new(5, "m");
        assert!(!cfg.has_temperature());
    }

    #[test]
    fn test_agent_config_has_temperature_true_after_setting() {
        let cfg = AgentConfig::new(5, "m").with_temperature(0.7);
        assert!(cfg.has_temperature());
    }

    // ── Round 26: ToolSpec builders ───────────────────────────────────────────

    #[test]
    fn test_tool_spec_new_fallible_returns_ok_value() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tool = ToolSpec::new_fallible(
            "add",
            "adds numbers",
            |_args| Ok(serde_json::json!({"result": 42})),
        );
        let result = rt.block_on(tool.call(serde_json::json!({})));
        assert_eq!(result["result"], 42);
    }

    #[test]
    fn test_tool_spec_new_fallible_wraps_error_as_json() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tool = ToolSpec::new_fallible(
            "fail",
            "always fails",
            |_| Err("bad input".to_string()),
        );
        let result = rt.block_on(tool.call(serde_json::json!({})));
        assert_eq!(result["error"], "bad input");
        assert_eq!(result["ok"], false);
    }

    #[test]
    fn test_tool_spec_new_async_fallible_wraps_error() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tool = ToolSpec::new_async_fallible(
            "async_fail",
            "async error",
            |_| Box::pin(async { Err("async bad".to_string()) }),
        );
        let result = rt.block_on(tool.call(serde_json::json!({})));
        assert_eq!(result["error"], "async bad");
    }

    #[test]
    fn test_tool_spec_with_required_fields_sets_fields() {
        let tool = ToolSpec::new("t", "d", |_| serde_json::json!({}))
            .with_required_fields(["name", "value"]);
        assert_eq!(tool.required_field_count(), 2);
    }

    #[test]
    fn test_tool_spec_with_description_overrides_description() {
        let tool = ToolSpec::new("t", "original", |_| serde_json::json!({}))
            .with_description("updated description");
        assert_eq!(tool.description, "updated description");
    }

    // ── Round 25: stop_sequence_count / find_by_description_keyword ───────────

    #[test]
    fn test_agent_config_stop_sequence_count_zero_by_default() {
        let cfg = AgentConfig::new(5, "gpt-4");
        assert_eq!(cfg.stop_sequence_count(), 0);
    }

    #[test]
    fn test_agent_config_stop_sequence_count_reflects_configured_count() {
        let cfg = AgentConfig::new(5, "gpt-4")
            .with_stop_sequences(vec!["STOP".to_string(), "END".to_string()]);
        assert_eq!(cfg.stop_sequence_count(), 2);
    }

    #[test]
    fn test_tool_registry_find_by_description_keyword_empty_when_no_match() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("calc", "Performs arithmetic", |_| serde_json::json!({})));
        let results = reg.find_by_description_keyword("weather");
        assert!(results.is_empty());
    }

    #[test]
    fn test_tool_registry_find_by_description_keyword_case_insensitive() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("calc", "Performs ARITHMETIC operations", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("search", "Searches the web", |_| serde_json::json!({})));
        let results = reg.find_by_description_keyword("arithmetic");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "calc");
    }

    #[test]
    fn test_tool_registry_find_by_description_keyword_multiple_matches() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "query the database", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("t2", "query the cache", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("t3", "send a message", |_| serde_json::json!({})));
        let results = reg.find_by_description_keyword("query");
        assert_eq!(results.len(), 2);
    }

    // ── Round 31: Message::is_user/is_assistant,
    //             AgentConfig::stop_sequence_count/has_request_timeout ─────────

    #[test]
    fn test_message_is_user_true_for_user_role_r31() {
        let msg = Message::user("hello");
        assert!(msg.is_user());
        assert!(!msg.is_assistant());
    }

    #[test]
    fn test_message_is_assistant_true_for_assistant_role_r31() {
        let msg = Message::assistant("hi there");
        assert!(msg.is_assistant());
        assert!(!msg.is_user());
    }

    #[test]
    fn test_agent_config_stop_sequence_count_zero_for_new_config() {
        let cfg = AgentConfig::new(5, "model");
        assert_eq!(cfg.stop_sequence_count(), 0);
    }

    #[test]
    fn test_agent_config_stop_sequence_count_after_setting() {
        let cfg = AgentConfig::new(5, "model")
            .with_stop_sequences(vec!["<stop>".to_string(), "END".to_string()]);
        assert_eq!(cfg.stop_sequence_count(), 2);
    }

    #[test]
    fn test_agent_config_has_request_timeout_false_by_default() {
        let cfg = AgentConfig::new(5, "model");
        assert!(!cfg.has_request_timeout());
    }

    #[test]
    fn test_agent_config_has_request_timeout_true_after_setting() {
        let cfg = AgentConfig::new(5, "model")
            .with_request_timeout(std::time::Duration::from_secs(30));
        assert!(cfg.has_request_timeout());
    }

    // ── Round 29: ReActLoop::unregister_tool ──────────────────────────────────

    #[test]
    fn test_react_loop_unregister_tool_removes_registered_tool() {
        let mut agent = ReActLoop::new(AgentConfig::new(5, "m"));
        agent.register_tool(ToolSpec::new("t1", "desc", |_| serde_json::json!({})));
        assert!(agent.unregister_tool("t1"));
        assert_eq!(agent.tool_count(), 0);
    }

    #[test]
    fn test_react_loop_unregister_tool_returns_false_for_unknown() {
        let mut agent = ReActLoop::new(AgentConfig::new(5, "m"));
        assert!(!agent.unregister_tool("nonexistent"));
    }

    // ── Round 26: tool_count_with_required_fields ─────────────────────────────

    #[test]
    fn test_tool_count_with_required_fields_zero_when_empty() {
        let reg = ToolRegistry::new();
        assert_eq!(reg.tool_count_with_required_fields(), 0);
    }

    #[test]
    fn test_tool_count_with_required_fields_excludes_tools_without_fields() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "d", |_| serde_json::json!({})));
        assert_eq!(reg.tool_count_with_required_fields(), 0);
    }

    #[test]
    fn test_tool_count_with_required_fields_counts_only_tools_with_fields() {
        let mut reg = ToolRegistry::new();
        reg.register(
            ToolSpec::new("t1", "d", |_| serde_json::json!({}))
                .with_required_fields(["query"]),
        );
        reg.register(ToolSpec::new("t2", "d", |_| serde_json::json!({}))); // no required
        reg.register(
            ToolSpec::new("t3", "d", |_| serde_json::json!({}))
                .with_required_fields(["url", "method"]),
        );
        assert_eq!(reg.tool_count_with_required_fields(), 2);
    }

    // ── Round 27: ToolRegistry::names ─────────────────────────────────────────

    #[test]
    fn test_tool_registry_names_empty_when_no_tools() {
        let reg = ToolRegistry::new();
        assert!(reg.names().is_empty());
    }

    #[test]
    fn test_tool_registry_names_sorted_alphabetically() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("zebra", "d", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("alpha", "d", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("mango", "d", |_| serde_json::json!({})));
        assert_eq!(reg.names(), vec!["alpha", "mango", "zebra"]);
    }

    // ── Round 28: tool_names_starting_with ────────────────────────────────────

    #[test]
    fn test_tool_names_starting_with_empty_when_no_match() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("search", "d", |_| serde_json::json!({})));
        assert!(reg.tool_names_starting_with("calc").is_empty());
    }

    #[test]
    fn test_tool_names_starting_with_returns_sorted_matches() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("db_write", "d", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("db_read", "d", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("cache_get", "d", |_| serde_json::json!({})));
        let results = reg.tool_names_starting_with("db_");
        assert_eq!(results, vec!["db_read", "db_write"]);
    }

    // ── Round 29: description_for ─────────────────────────────────────────────

    #[test]
    fn test_tool_registry_description_for_none_when_missing() {
        let reg = ToolRegistry::new();
        assert!(reg.description_for("unknown").is_none());
    }

    #[test]
    fn test_tool_registry_description_for_returns_description() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("search", "Find web results", |_| serde_json::json!({})));
        assert_eq!(reg.description_for("search"), Some("Find web results"));
    }

    // ── Round 30: count_with_description_containing ───────────────────────────

    #[test]
    fn test_count_with_description_containing_zero_when_no_match() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "database query", |_| serde_json::json!({})));
        assert_eq!(reg.count_with_description_containing("weather"), 0);
    }

    #[test]
    fn test_count_with_description_containing_case_insensitive() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "Search the WEB", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("t2", "web scraper tool", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("t3", "database lookup", |_| serde_json::json!({})));
        assert_eq!(reg.count_with_description_containing("web"), 2);
    }

    #[test]
    fn test_unregister_all_clears_all_tools() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "tool one", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("t2", "tool two", |_| serde_json::json!({})));
        assert_eq!(reg.tool_count(), 2);
        reg.unregister_all();
        assert_eq!(reg.tool_count(), 0);
    }

    #[test]
    fn test_tool_names_with_keyword_returns_matching_tool_names() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("search", "search the web for info", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("db", "query database records", |_| serde_json::json!({})));
        reg.register(ToolSpec::new("web-fetch", "fetch a WEB page", |_| serde_json::json!({})));
        let mut names = reg.tool_names_with_keyword("web");
        names.sort_unstable();
        assert_eq!(names, vec!["search", "web-fetch"]);
    }

    #[test]
    fn test_tool_names_with_keyword_no_match_returns_empty() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t", "some tool", |_| serde_json::json!({})));
        assert!(reg.tool_names_with_keyword("missing").is_empty());
    }
}
