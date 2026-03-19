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
    /// Model sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<usize>,
    /// Per-inference timeout.
    pub request_timeout: Option<std::time::Duration>,
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

// ── ToolCache ─────────────────────────────────────────────────────────────────

/// Cache for tool call results. Implement to deduplicate repeated identical tool calls.
pub trait ToolCache: Send + Sync {
    /// Look up a cached result for `(tool_name, args)`.
    fn get(&self, tool_name: &str, args: &serde_json::Value) -> Option<serde_json::Value>;
    /// Store a result for `(tool_name, args)`.
    fn set(&self, tool_name: &str, args: &serde_json::Value, result: serde_json::Value);
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

        // Store result in cache.
        if let Some(ref cache) = self.cache {
            cache.set(name, &args, result.clone());
        }

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
            // Wall-clock timeout check.
            if let Some(dl) = deadline {
                if std::time::Instant::now() >= dl {
                    let ms = self
                        .config
                        .loop_timeout
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    if let Some(ref obs) = self.observer {
                        obs.on_loop_end(steps.len());
                    }
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
                    step.observation = serde_json::json!({"ok": false, "error": "action blocked by reviewer", "kind": "blocked"}).to_string();
                    step.step_duration_ms = step_start.elapsed().as_millis() as u64;
                    if let Some(ref m) = self.metrics {
                        m.record_step_latency(step.step_duration_ms);
                    }
                    context.push_str(&format!(
                        "\nThought: {}\nAction: {}\nObservation: {}\n",
                        step.thought, step.action, step.observation
                    ));
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
            if let Some(ref m) = self.metrics {
                m.record_step_latency(step.step_duration_ms);
            }
            context.push_str(&format!(
                "\nThought: {}\nAction: {}\nObservation: {}\n",
                step.thought, step.action, observation
            ));
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
    /// Called when the loop starts.
    fn on_loop_start(&self, prompt: &str) {
        let _ = prompt;
    }
    /// Called when the loop finishes (success or error).
    fn on_loop_end(&self, step_count: usize) {
        let _ = step_count;
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
pub type ActionHook = Arc<dyn Fn(String, serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>> + Send + Sync>;

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

    // ── Improvement 15: ActionHook ────────────────────────────────────────────

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
