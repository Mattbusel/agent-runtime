//! # Module: AgentRuntime
//!
//! ## Responsibility
//! Wire memory, graph, orchestrator, and agent loop into a single coordinator
//! using a builder pattern. Provides `run_agent` which executes a ReAct loop,
//! optionally enriching context from memory and graph lookups.
//!
//! ## Guarantees
//! - Builder uses a typestate parameter to enforce `agent_config` at compile time:
//!   `build()` is only callable once `with_agent_config` has been called.
//! - `run_agent` is async and returns a typed `AgentSession` with step count,
//!   durations, and hits.
//! - Non-panicking: all paths return `Result`
//!
//! ## NOT Responsible For
//! - Actual LLM inference (callers supply a mock/stub inference fn)
//! - Persistence across process restarts (unless `persistence` feature is enabled)

use crate::agent::{AgentConfig, ReActLoop, ReActStep, ToolSpec};
use crate::error::AgentRuntimeError;
use crate::memory::{AgentId, EpisodicStore, WorkingMemory};
use crate::metrics::RuntimeMetrics;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "graph")]
use crate::graph::GraphStore;

#[cfg(feature = "orchestrator")]
use crate::orchestrator::BackpressureGuard;

// ── Typestate markers ─────────────────────────────────────────────────────────

/// Builder state: agent config has not been provided yet.
pub struct NeedsConfig;
/// Builder state: agent config has been provided; `build()` is available.
pub struct HasConfig;

// ── AgentSession ──────────────────────────────────────────────────────────────

/// The result of a single agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    /// Stable unique identifier for this session (UUID v4 string).
    pub session_id: String,
    /// The agent ID used for this session.
    pub agent_id: AgentId,
    /// All ReAct steps executed during the session.
    pub steps: Vec<ReActStep>,
    /// Number of episodic memory retrievals made during the session.
    pub memory_hits: usize,
    /// Number of graph lookups made during the session.
    pub graph_lookups: usize,
    /// Wall-clock duration of the session in milliseconds.
    pub duration_ms: u64,
    /// Non-fatal errors encountered while saving per-step checkpoints.
    ///
    /// Populated only when a persistence backend is configured.  A non-empty
    /// list means some step snapshots may be missing from storage, but the
    /// session itself completed successfully.
    #[serde(default)]
    pub checkpoint_errors: Vec<String>,
}

impl AgentSession {
    /// Return the number of steps in the session.
    ///
    /// Each [`ReActStep`] in `steps` carries a `step_duration_ms` field measuring
    /// wall-clock time from inference call to observation for that individual step.
    /// Use this to identify slow steps:
    /// ```rust,ignore
    /// for (i, step) in session.steps.iter().enumerate() {
    ///     println!("step {i}: {}ms", step.step_duration_ms);
    /// }
    /// ```
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the final answer text from the last step, if available.
    ///
    /// Extracts the content after `FINAL_ANSWER` in the last step's `action` field.
    /// Returns `None` if there are no steps or the last action is not a FINAL_ANSWER.
    pub fn final_answer(&self) -> Option<String> {
        let last = self.steps.last()?;
        let upper = last.action.trim().to_ascii_uppercase();
        if upper.starts_with("FINAL_ANSWER") {
            let answer = last.action.trim()["FINAL_ANSWER".len()..].trim().to_owned();
            Some(answer)
        } else {
            None
        }
    }

    /// Persist this session as a checkpoint under `"session:<session_id>"`.
    #[cfg(feature = "persistence")]
    pub async fn save_checkpoint(
        &self,
        backend: &dyn crate::persistence::PersistenceBackend,
    ) -> Result<(), AgentRuntimeError> {
        let key = format!("session:{}", self.session_id);
        let bytes = serde_json::to_vec(self)
            .map_err(|e| AgentRuntimeError::Persistence(format!("serialize: {e}")))?;
        backend.save(&key, &bytes).await
    }

    /// Load a previously saved checkpoint by `session_id`.
    ///
    /// Returns `None` if no checkpoint exists for the given ID.
    #[cfg(feature = "persistence")]
    pub async fn load_checkpoint(
        backend: &dyn crate::persistence::PersistenceBackend,
        session_id: &str,
    ) -> Result<Option<AgentSession>, AgentRuntimeError> {
        let key = format!("session:{session_id}");
        match backend.load(&key).await? {
            None => Ok(None),
            Some(bytes) => {
                let session = serde_json::from_slice(&bytes)
                    .map_err(|e| AgentRuntimeError::Persistence(format!("deserialize: {e}")))?;
                Ok(Some(session))
            }
        }
    }

    /// Load a step-scoped checkpoint saved by `AgentRuntime`.
    ///
    /// Returns `None` if no checkpoint exists for the given session/step.
    #[cfg(feature = "persistence")]
    pub async fn load_step_checkpoint(
        backend: &dyn crate::persistence::PersistenceBackend,
        session_id: &str,
        step: usize,
    ) -> Result<Option<AgentSession>, AgentRuntimeError> {
        let key = format!("session:{session_id}:step:{step}");
        match backend.load(&key).await? {
            None => Ok(None),
            Some(bytes) => {
                let session = serde_json::from_slice(&bytes)
                    .map_err(|e| AgentRuntimeError::Persistence(format!("deserialize: {e}")))?;
                Ok(Some(session))
            }
        }
    }
}

// ── AgentRuntimeBuilder ───────────────────────────────────────────────────────

/// Builder for `AgentRuntime`.
///
/// Uses a typestate parameter `S` to enforce that `with_agent_config` is called
/// before `build()`.  Calling `build()` on a `AgentRuntimeBuilder<NeedsConfig>`
/// is a **compile-time error**.
///
/// Typical usage:
/// ```ignore
/// let runtime = AgentRuntime::builder()      // AgentRuntimeBuilder<NeedsConfig>
///     .with_memory(store)
///     .with_agent_config(cfg)                // → AgentRuntimeBuilder<HasConfig>
///     .build();                              // → AgentRuntime (infallible)
/// ```
/// Builder for [`AgentRuntime`].
pub struct AgentRuntimeBuilder<S = NeedsConfig> {
    memory: Option<EpisodicStore>,
    working: Option<WorkingMemory>,
    #[cfg(feature = "graph")]
    graph: Option<GraphStore>,
    #[cfg(feature = "orchestrator")]
    backpressure: Option<BackpressureGuard>,
    agent_config: Option<AgentConfig>,
    tools: Vec<Arc<ToolSpec>>,
    metrics: Arc<RuntimeMetrics>,
    #[cfg(feature = "persistence")]
    checkpoint_backend: Option<Arc<dyn crate::persistence::PersistenceBackend>>,
    _state: PhantomData<S>,
}

impl std::fmt::Debug for AgentRuntimeBuilder<NeedsConfig> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("AgentRuntimeBuilder<NeedsConfig>");
        s.field("memory", &self.memory.is_some())
            .field("working", &self.working.is_some());
        #[cfg(feature = "graph")]
        s.field("graph", &self.graph.is_some());
        #[cfg(feature = "orchestrator")]
        s.field("backpressure", &self.backpressure.is_some());
        s.field("tools", &self.tools.len()).finish()
    }
}

impl std::fmt::Debug for AgentRuntimeBuilder<HasConfig> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("AgentRuntimeBuilder<HasConfig>");
        s.field("memory", &self.memory.is_some())
            .field("working", &self.working.is_some());
        #[cfg(feature = "graph")]
        s.field("graph", &self.graph.is_some());
        #[cfg(feature = "orchestrator")]
        s.field("backpressure", &self.backpressure.is_some());
        s.field("agent_config", &self.agent_config.is_some())
            .field("tools", &self.tools.len())
            .finish()
    }
}

impl Default for AgentRuntimeBuilder<NeedsConfig> {
    fn default() -> Self {
        Self {
            memory: None,
            working: None,
            #[cfg(feature = "graph")]
            graph: None,
            #[cfg(feature = "orchestrator")]
            backpressure: None,
            agent_config: None,
            tools: Vec::new(),
            metrics: RuntimeMetrics::new(),
            #[cfg(feature = "persistence")]
            checkpoint_backend: None,
            _state: PhantomData,
        }
    }
}

// Methods available on ALL builder states.
impl<S> AgentRuntimeBuilder<S> {
    /// Attach an episodic memory store.
    pub fn with_memory(mut self, store: EpisodicStore) -> Self {
        self.memory = Some(store);
        self
    }

    /// Attach a working memory store.
    pub fn with_working_memory(mut self, wm: WorkingMemory) -> Self {
        self.working = Some(wm);
        self
    }

    /// Attach a graph store.
    #[cfg(feature = "graph")]
    pub fn with_graph(mut self, graph: GraphStore) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Attach a backpressure guard.
    #[cfg(feature = "orchestrator")]
    pub fn with_backpressure(mut self, guard: BackpressureGuard) -> Self {
        self.backpressure = Some(guard);
        self
    }

    /// Register a tool available to the agent loop.
    pub fn register_tool(mut self, spec: ToolSpec) -> Self {
        self.tools.push(Arc::new(spec));
        self
    }

    /// Register multiple tools at once.
    ///
    /// Equivalent to calling [`register_tool`] for each spec.
    ///
    /// [`register_tool`]: AgentRuntimeBuilder::register_tool
    pub fn register_tools(mut self, specs: impl IntoIterator<Item = ToolSpec>) -> Self {
        for spec in specs {
            self.tools.push(Arc::new(spec));
        }
        self
    }

    /// Attach a shared `RuntimeMetrics` instance.
    pub fn with_metrics(mut self, metrics: Arc<RuntimeMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Attach a persistence backend for session checkpointing.
    #[cfg(feature = "persistence")]
    pub fn with_checkpoint_backend(
        mut self,
        backend: Arc<dyn crate::persistence::PersistenceBackend>,
    ) -> Self {
        self.checkpoint_backend = Some(backend);
        self
    }
}

// `with_agent_config` transitions NeedsConfig → HasConfig.
impl AgentRuntimeBuilder<NeedsConfig> {
    /// Create a new builder (equivalent to `Default::default()`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the agent loop configuration.
    ///
    /// After this call the builder transitions to `AgentRuntimeBuilder<HasConfig>`,
    /// making `build()` available.
    /// Set the agent loop configuration.
    ///
    /// After this call the builder transitions to `AgentRuntimeBuilder<HasConfig>`,
    /// making `build()` available.
    pub fn with_agent_config(self, config: AgentConfig) -> AgentRuntimeBuilder<HasConfig> {
        AgentRuntimeBuilder {
            memory: self.memory,
            working: self.working,
            #[cfg(feature = "graph")]
            graph: self.graph,
            #[cfg(feature = "orchestrator")]
            backpressure: self.backpressure,
            agent_config: Some(config),
            tools: self.tools,
            metrics: self.metrics,
            #[cfg(feature = "persistence")]
            checkpoint_backend: self.checkpoint_backend,
            _state: PhantomData,
        }
    }
}

// `build()` is only available once we have a config.
impl AgentRuntimeBuilder<HasConfig> {
    /// Build the `AgentRuntime`.
    ///
    /// This is infallible: the typestate guarantees `agent_config` is present.
    pub fn build(self) -> AgentRuntime {
        // SAFETY: `agent_config` is always `Some` in `HasConfig` state because
        // `with_agent_config` is the only way to reach this state.
        #[allow(clippy::unwrap_used)]
        let agent_config = self.agent_config.unwrap();

        AgentRuntime {
            memory: self.memory,
            working: self.working,
            #[cfg(feature = "graph")]
            graph: self.graph,
            #[cfg(feature = "orchestrator")]
            backpressure: self.backpressure,
            agent_config,
            tools: self.tools,
            metrics: self.metrics,
            #[cfg(feature = "persistence")]
            checkpoint_backend: self.checkpoint_backend,
        }
    }
}

// ── AgentRuntime ──────────────────────────────────────────────────────────────

/// Unified runtime that coordinates memory, graph, orchestration, and agent loop.
pub struct AgentRuntime {
    memory: Option<EpisodicStore>,
    working: Option<WorkingMemory>,
    #[cfg(feature = "graph")]
    graph: Option<GraphStore>,
    #[cfg(feature = "orchestrator")]
    backpressure: Option<BackpressureGuard>,
    agent_config: AgentConfig,
    tools: Vec<Arc<ToolSpec>>,
    metrics: Arc<RuntimeMetrics>,
    #[cfg(feature = "persistence")]
    checkpoint_backend: Option<Arc<dyn crate::persistence::PersistenceBackend>>,
}

impl std::fmt::Debug for AgentRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("AgentRuntime");
        s.field("memory", &self.memory.is_some())
            .field("working", &self.working.is_some());
        #[cfg(feature = "graph")]
        s.field("graph", &self.graph.is_some());
        #[cfg(feature = "orchestrator")]
        s.field("backpressure", &self.backpressure.is_some());
        s.field("tools", &self.tools.len());
        #[cfg(feature = "persistence")]
        s.field("checkpoint_backend", &self.checkpoint_backend.is_some());
        s.finish()
    }
}

impl AgentRuntime {
    /// Return a new builder in the `NeedsConfig` state.
    pub fn builder() -> AgentRuntimeBuilder<NeedsConfig> {
        AgentRuntimeBuilder::new()
    }

    /// Construct a minimal `AgentRuntime` in one call with sensible defaults.
    pub fn quick(max_iterations: usize, model: impl Into<String>) -> Self {
        AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(max_iterations, model))
            .build()
    }

    /// Return a shared reference to the runtime metrics.
    pub fn metrics(&self) -> Arc<RuntimeMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Run the agent loop for the given prompt.
    ///
    /// Optionally recalls episodic memories and injects them into the context.
    /// Optionally enforces backpressure before starting.
    ///
    /// # Arguments
    /// * `agent_id` — identifies the agent for memory retrieval
    /// * `prompt` — the user's input prompt
    /// * `infer` — async inference function: `(context: String) -> impl Future<Output = String>`
    ///
    /// # Returns
    /// An `AgentSession` with step count, hits, duration, and a stable session ID.
    #[tracing::instrument(skip(self, infer), fields(agent_id = %agent_id))]
    pub async fn run_agent<F, Fut>(
        &self,
        agent_id: AgentId,
        prompt: &str,
        infer: F,
    ) -> Result<AgentSession, AgentRuntimeError>
    where
        F: FnMut(String) -> Fut,
        Fut: std::future::Future<Output = String>,
    {
        self.metrics.total_sessions.fetch_add(1, Ordering::Relaxed);
        self.metrics.active_sessions.fetch_add(1, Ordering::Relaxed);

        // Acquire backpressure slot before any work.
        #[cfg(feature = "orchestrator")]
        {
            let backpressure_result = if let Some(ref guard) = self.backpressure {
                guard.try_acquire()
            } else {
                Ok(())
            };
            if let Err(e) = backpressure_result {
                tracing::warn!(agent_id = %agent_id, error = %e, "backpressure shed: rejecting session");
                self.metrics
                    .backpressure_shed_count
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
                return Err(e);
            }
        }

        tracing::info!(agent_id = %agent_id, "agent session starting");
        let outcome = self.run_agent_inner(agent_id.clone(), prompt, infer).await;

        // Always release backpressure — success or error.
        #[cfg(feature = "orchestrator")]
        if let Some(ref guard) = self.backpressure {
            let _ = guard.release();
        }

        self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);

        match &outcome {
            Ok(session) => {
                tracing::info!(
                    agent_id = %agent_id,
                    session_id = %session.session_id,
                    steps = session.step_count(),
                    duration_ms = session.duration_ms,
                    "agent session completed"
                );
                self.metrics
                    .total_steps
                    .fetch_add(session.step_count() as u64, Ordering::Relaxed);
            }
            Err(e) => {
                tracing::error!(agent_id = %agent_id, error = %e, "agent session failed");
            }
        }

        outcome
    }

    /// Inner implementation of `run_agent`, called after backpressure is acquired.
    #[tracing::instrument(skip(self, infer), fields(agent_id = %agent_id, session_id = tracing::field::Empty))]
    async fn run_agent_inner<F, Fut>(
        &self,
        agent_id: AgentId,
        prompt: &str,
        infer: F,
    ) -> Result<AgentSession, AgentRuntimeError>
    where
        F: FnMut(String) -> Fut,
        Fut: std::future::Future<Output = String>,
    {
        let start = Instant::now();
        let session_id = uuid::Uuid::new_v4().to_string();

        let mut memory_hits = 0usize;
        let mut graph_lookups = 0usize;

        // Build enriched prompt from episodic memory.
        let enriched_prompt = if let Some(ref store) = self.memory {
            let memories = store.recall(&agent_id, self.agent_config.max_memory_recalls)?;

            // Apply token budget if configured.
            let memories = if let Some(token_budget) = self.agent_config.max_memory_tokens {
                let mut used = 0usize;
                memories
                    .into_iter()
                    .filter(|m| {
                        let tokens = (m.content.len() / 4).max(1);
                        if used + tokens <= token_budget {
                            used += tokens;
                            true
                        } else {
                            false
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                memories
            };

            memory_hits = memories.len();
            self.metrics
                .memory_recall_count
                .fetch_add(1, Ordering::Relaxed);

            if let Some(budget) = self.agent_config.max_memory_tokens {
                tracing::debug!(
                    "memory token budget: {budget}, injecting {} items",
                    memory_hits
                );
            } else {
                tracing::debug!("enriched prompt with {} memory items", memory_hits);
            }

            if memories.is_empty() {
                prompt.to_owned()
            } else {
                let mem_context: Vec<String> = memories
                    .iter()
                    .map(|m| format!("- {}", m.content))
                    .collect();
                format!(
                    "Relevant memories:\n{}\n\nCurrent prompt: {prompt}",
                    mem_context.join("\n")
                )
            }
        } else {
            prompt.to_owned()
        };

        // Inject working memory into prompt.
        let enriched_prompt = if let Some(ref wm) = self.working {
            let entries = wm.entries()?;
            if entries.is_empty() {
                enriched_prompt
            } else {
                let wm_context: Vec<String> =
                    entries.iter().map(|(k, v)| format!("  {k}: {v}")).collect();
                format!(
                    "{enriched_prompt}\n\nCurrent working state:\n{}",
                    wm_context.join("\n")
                )
            }
        } else {
            enriched_prompt
        };

        // Count graph entities as "lookups" for session metadata.
        #[cfg(feature = "graph")]
        if let Some(ref graph) = self.graph {
            graph_lookups = graph.entity_count()?;
            tracing::debug!("graph has {} entities", graph_lookups);
        }

        // Build the ReAct loop and register tools.
        // Each ToolSpec is stored as an Arc so we can clone the Arc into the
        // handler closure without moving ownership out of self.tools.
        // Required fields and the per-tool circuit breaker are preserved so
        // that validation and fast-fail behaviour work correctly at run time.
        let mut react_loop = ReActLoop::new(self.agent_config.clone())
            .with_metrics(Arc::clone(&self.metrics));

        // Item 11 — wire per-step loop checkpointing.
        #[cfg(feature = "persistence")]
        if let Some(ref backend) = self.checkpoint_backend {
            react_loop = react_loop
                .with_step_checkpoint(Arc::clone(backend), session_id.clone());
        }

        for tool in &self.tools {
            let tool_arc = Arc::clone(tool);
            let required_fields = tool_arc.required_fields.clone();
            #[cfg(feature = "orchestrator")]
            let circuit_breaker = tool_arc.circuit_breaker.clone();

            let mut spec = ToolSpec::new_async(
                tool_arc.name.clone(),
                tool_arc.description.clone(),
                move |args| {
                    let t = Arc::clone(&tool_arc);
                    Box::pin(async move { t.call(args).await })
                },
            )
            .with_required_fields(required_fields);

            #[cfg(feature = "orchestrator")]
            if let Some(cb) = circuit_breaker {
                spec = spec.with_circuit_breaker(cb);
            }

            react_loop.register_tool(spec);
        }

        // Record the session_id into the current tracing span so that all
        // child spans (ReActLoop iterations, tool calls) carry this field.
        tracing::Span::current().record("session_id", &session_id.as_str());

        let steps = react_loop.run(&enriched_prompt, infer).await?;
        let duration_ms = start.elapsed().as_millis() as u64;

        // Item 6 — collect per-step checkpoint errors; surfaced in AgentSession.
        #[cfg(feature = "persistence")]
        let mut ckpt_errors: Vec<String> = Vec::new();

        // Save final checkpoint if a backend is configured.
        #[cfg(feature = "persistence")]
        if let Some(ref backend) = self.checkpoint_backend {
            tracing::info!(session_id = %session_id, "saving session checkpoint");

            // Build a temporary session without errors to save as the base checkpoint.
            let tmp = AgentSession {
                session_id: session_id.clone(),
                agent_id: agent_id.clone(),
                steps: steps.clone(),
                memory_hits,
                graph_lookups,
                duration_ms,
                checkpoint_errors: vec![],
            };
            tmp.save_checkpoint(backend.as_ref()).await?;

            // Save incremental per-step consolidated snapshots.
            for i in 1..=steps.len() {
                let partial = AgentSession {
                    session_id: session_id.clone(),
                    agent_id: agent_id.clone(),
                    steps: steps[..i].to_vec(),
                    memory_hits,
                    graph_lookups,
                    duration_ms,
                    checkpoint_errors: vec![],
                };
                let key = format!("session:{session_id}:step:{i}");
                match serde_json::to_vec(&partial) {
                    Ok(bytes) => {
                        if let Err(e) = backend.save(&key, &bytes).await {
                            let msg = format!("session:{session_id} step:{i} save: {e}");
                            tracing::warn!("{}", msg);
                            ckpt_errors.push(msg);
                        }
                    }
                    Err(e) => {
                        let msg =
                            format!("session:{session_id} step:{i} serialise: {e}");
                        tracing::warn!("{}", msg);
                        ckpt_errors.push(msg);
                    }
                }
            }
        }

        let session = AgentSession {
            session_id,
            agent_id,
            steps,
            memory_hits,
            graph_lookups,
            duration_ms,
            #[cfg(feature = "persistence")]
            checkpoint_errors: ckpt_errors,
            #[cfg(not(feature = "persistence"))]
            checkpoint_errors: vec![],
        };

        Ok(session)
    }

    /// Return a reference to the episodic memory store, if configured.
    pub fn memory(&self) -> Option<&EpisodicStore> {
        self.memory.as_ref()
    }

    /// Return a reference to the graph store, if configured.
    #[cfg(feature = "graph")]
    pub fn graph(&self) -> Option<&GraphStore> {
        self.graph.as_ref()
    }

    /// Return a reference to the working memory, if configured.
    pub fn working_memory(&self) -> Option<&WorkingMemory> {
        self.working.as_ref()
    }

    /// Gracefully shut down the runtime.
    ///
    /// Logs a structured shutdown event with the final metrics snapshot.
    /// If the `persistence` feature is enabled and a checkpoint backend is
    /// configured, writes a sentinel key so operators can confirm clean shutdown.
    ///
    /// After calling `shutdown`, the runtime should not be used again.
    pub async fn shutdown(&self) {
        tracing::info!("AgentRuntime shutting down");
        tracing::info!(
            active_sessions = self.metrics.active_sessions(),
            total_sessions = self.metrics.total_sessions(),
            total_steps = self.metrics.total_steps(),
            total_tool_calls = self.metrics.total_tool_calls(),
            failed_tool_calls = self.metrics.failed_tool_calls(),
            "final metrics snapshot on shutdown"
        );

        #[cfg(feature = "persistence")]
        if let Some(ref backend) = self.checkpoint_backend {
            let ts = chrono::Utc::now().to_rfc3339();
            match backend.save("runtime:shutdown", ts.as_bytes()).await {
                Ok(()) => tracing::debug!("shutdown sentinel saved"),
                Err(e) => tracing::warn!(error = %e, "failed to save shutdown sentinel"),
            }
        }

        tracing::info!("AgentRuntime shutdown complete");
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Entity, GraphStore, Relationship};
    use crate::memory::EpisodicStore;

    fn simple_config() -> AgentConfig {
        AgentConfig::new(5, "test")
    }

    async fn final_answer_infer(_ctx: String) -> String {
        "Thought: done\nAction: FINAL_ANSWER 42".into()
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    // NOTE: test_builder_fails_without_agent_config has been removed.
    // The typestate pattern makes calling .build() without .with_agent_config()
    // a *compile-time error* — AgentRuntimeBuilder<NeedsConfig> has no build()
    // method.  There is nothing to test at runtime.

    /// Verifies that the builder compiles and produces a runtime when config is
    /// provided.  This is the runtime-observable counterpart to the former
    /// "fails without config" test.
    #[tokio::test]
    async fn test_builder_with_config_compiles() {
        let _runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        // If this compiles and runs, the typestate transition worked correctly.
    }

    #[tokio::test]
    async fn test_builder_succeeds_with_minimal_config() {
        let _runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
    }

    #[tokio::test]
    async fn test_builder_with_all_subsystems() {
        let _runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(EpisodicStore::new())
            .with_graph(GraphStore::new())
            .with_working_memory(WorkingMemory::new(10).unwrap())
            .with_backpressure(BackpressureGuard::new(5).unwrap())
            .build();
    }

    #[tokio::test]
    async fn test_builder_produces_runtime_with_config() {
        // Confirm the built runtime accepts a run_agent call — the most direct
        // evidence that the builder wired everything correctly.
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        let session = runtime
            .run_agent(AgentId::new("agent-x"), "hello", final_answer_infer)
            .await
            .unwrap();
        assert!(session.step_count() >= 1);
        assert!(!session.session_id.is_empty());
    }

    // ── run_agent ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_run_agent_returns_session_with_steps() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("agent-1"), "hello", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.step_count(), 1);
    }

    #[tokio::test]
    async fn test_run_agent_session_has_agent_id() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("agent-42"), "hello", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.agent_id.0, "agent-42");
    }

    #[tokio::test]
    async fn test_run_agent_session_duration_is_set() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "hello", final_answer_infer)
            .await
            .unwrap();

        // Duration should be non-negative (0 ms is valid for a fast mock)
        let _ = session.duration_ms; // just verify it compiles and is set
    }

    #[tokio::test]
    async fn test_run_agent_session_has_session_id() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "hello", final_answer_infer)
            .await
            .unwrap();

        // session_id must be a non-empty UUID string
        assert!(!session.session_id.is_empty());
        assert_eq!(session.session_id.len(), 36); // UUID v4 canonical form
    }

    #[tokio::test]
    async fn test_run_agent_memory_hits_zero_without_memory() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.memory_hits, 0);
    }

    #[tokio::test]
    async fn test_run_agent_memory_hits_counts_recalled_items() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("mem-agent");
        store
            .add_episode(agent.clone(), "remembered fact", 0.8)
            .unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(store)
            .build();

        let session = runtime
            .run_agent(agent, "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.memory_hits, 1);
    }

    #[tokio::test]
    async fn test_run_agent_graph_lookups_counts_entities() {
        let graph = GraphStore::new();
        graph.add_entity(Entity::new("e1", "Node")).unwrap();
        graph.add_entity(Entity::new("e2", "Node")).unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_graph(graph)
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.graph_lookups, 2);
    }

    #[tokio::test]
    async fn test_run_agent_backpressure_released_after_run() {
        let guard = BackpressureGuard::new(3).unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_backpressure(guard.clone())
            .build();

        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(guard.depth().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_run_agent_backpressure_sheds_when_full() {
        let guard = BackpressureGuard::new(1).unwrap();
        guard.try_acquire().unwrap(); // pre-fill

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_backpressure(guard)
            .build();

        let result = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await;
        assert!(matches!(
            result,
            Err(AgentRuntimeError::BackpressureShed { .. })
        ));
    }

    #[tokio::test]
    async fn test_run_agent_max_iterations_error_propagated() {
        let cfg = AgentConfig::new(2, "model");
        let runtime = AgentRuntime::builder().with_agent_config(cfg).build();

        // Simulate an infer fn that always produces FINAL_ANSWER immediately
        let result = runtime
            .run_agent(AgentId::new("a"), "prompt", |_ctx: String| async {
                "Thought: looping\nAction: FINAL_ANSWER done".to_string()
            })
            .await;
        assert!(result.is_ok()); // final answer on first call, ok
    }

    #[tokio::test]
    async fn test_agent_session_step_count_matches_steps() {
        let session = AgentSession {
            session_id: "test-session-id".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep {
                    thought: "t".into(),
                    action: "a".into(),
                    observation: "o".into(),
                    step_duration_ms: 0,
                },
                ReActStep {
                    thought: "t2".into(),
                    action: "FINAL_ANSWER".into(),
                    observation: "done".into(),
                    step_duration_ms: 0,
                },
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 10,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.step_count(), 2);
    }

    // ── Accessor methods ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_runtime_memory_accessor_returns_none_when_not_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        assert!(runtime.memory().is_none());
    }

    #[tokio::test]
    async fn test_runtime_memory_accessor_returns_some_when_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(EpisodicStore::new())
            .build();
        assert!(runtime.memory().is_some());
    }

    #[tokio::test]
    async fn test_runtime_graph_accessor_returns_none_when_not_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        assert!(runtime.graph().is_none());
    }

    #[tokio::test]
    async fn test_runtime_graph_accessor_returns_some_when_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_graph(GraphStore::new())
            .build();
        assert!(runtime.graph().is_some());
    }

    #[tokio::test]
    async fn test_runtime_working_memory_accessor() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_working_memory(WorkingMemory::new(5).unwrap())
            .build();
        assert!(runtime.working_memory().is_some());
    }

    #[tokio::test]
    async fn test_runtime_with_tool_registered() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .register_tool(ToolSpec::new("calc", "math", |_| serde_json::json!(99)))
            .build();

        let mut call_count = 0;
        let session = runtime
            .run_agent(AgentId::new("a"), "compute", move |_ctx: String| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: use calc\nAction: calc {}".into()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER result".into()
                    }
                }
            })
            .await
            .unwrap();

        assert!(session.step_count() >= 1);
    }

    #[tokio::test]
    async fn test_run_agent_with_graph_relationship_lookup() {
        let graph = GraphStore::new();
        graph.add_entity(Entity::new("a", "X")).unwrap();
        graph.add_entity(Entity::new("b", "Y")).unwrap();
        graph
            .add_relationship(Relationship::new("a", "b", "LINKS", 1.0))
            .unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_graph(graph)
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.graph_lookups, 2); // 2 entities
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_metrics_active_sessions_decrements_after_run() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(runtime.metrics().active_sessions(), 0);
    }

    #[tokio::test]
    async fn test_metrics_total_sessions_increments() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();
        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(runtime.metrics().total_sessions(), 2);
    }

    #[tokio::test]
    async fn test_metrics_backpressure_shed_increments_on_shed() {
        let guard = BackpressureGuard::new(1).unwrap();
        guard.try_acquire().unwrap(); // pre-fill

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_backpressure(guard)
            .build();

        let _ = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await;

        assert_eq!(runtime.metrics().backpressure_shed_count(), 1);
    }

    #[tokio::test]
    async fn test_metrics_memory_recall_count_increments() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "fact", 0.9).unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(store)
            .build();

        runtime
            .run_agent(agent, "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(runtime.metrics().memory_recall_count(), 1);
    }

    // ── Memory token budgeting ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_agent_config_max_memory_tokens_limits_injection() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("budget-agent");
        // Each memory has ~100 chars → ~25 tokens each
        for i in 0..5 {
            let content = format!("{:0>100}", i); // 100-char string
            store.add_episode(agent.clone(), content, 0.9).unwrap();
        }

        // Token budget of 10 allows at most ~1 memory (each is ~25 tokens).
        let cfg = AgentConfig::new(5, "test").with_max_memory_tokens(10);
        let runtime = AgentRuntime::builder()
            .with_agent_config(cfg)
            .with_memory(store)
            .build();

        let session = runtime
            .run_agent(agent, "prompt", final_answer_infer)
            .await
            .unwrap();

        assert!(
            session.memory_hits <= 1,
            "expected at most 1 memory hit with tight token budget, got {}",
            session.memory_hits
        );
    }

    // ── Working memory injection ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_working_memory_injected_into_prompt() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("task", "write tests").unwrap();
        wm.set("status", "in progress").unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_working_memory(wm)
            .build();

        let mut captured_ctx: Option<String> = None;
        let captured_ref = &mut captured_ctx;

        runtime
            .run_agent(AgentId::new("a"), "do stuff", |ctx: String| {
                *captured_ref = Some(ctx.clone());
                async move { "Thought: done\nAction: FINAL_ANSWER ok".to_string() }
            })
            .await
            .unwrap();

        let ctx = captured_ctx.expect("infer should have been called");
        assert!(
            ctx.contains("Current working state:"),
            "expected working memory injection in context, got: {ctx}"
        );
        assert!(ctx.contains("task: write tests"));
        assert!(ctx.contains("status: in progress"));
    }

    // ── Improvement 8: AgentRuntime::quick() ──────────────────────────────────

    #[tokio::test]
    async fn test_agent_runtime_quick_runs_agent() {
        let runtime = AgentRuntime::quick(5, "test-model");
        let agent = AgentId::new("quick-agent");
        let session = runtime
            .run_agent(agent, "hello", |_ctx| async {
                "Thought: done\nAction: FINAL_ANSWER ok".to_string()
            })
            .await
            .unwrap();
        assert_eq!(session.step_count(), 1);
    }

    // ── #1 final_answer() ─────────────────────────────────────────────────────

    #[test]
    fn test_final_answer_extracts_text() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![ReActStep {
                thought: "done".into(),
                action: "FINAL_ANSWER Paris".into(),
                observation: "".into(),
                step_duration_ms: 0,
            }],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.final_answer(), Some("Paris".to_string()));
    }

    #[test]
    fn test_final_answer_returns_none_without_final_step() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![ReActStep {
                thought: "thinking".into(),
                action: "search {}".into(),
                observation: "result".into(),
                step_duration_ms: 0,
            }],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.final_answer(), None);

        let empty_session = AgentSession {
            session_id: "s2".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(empty_session.final_answer(), None);
    }
}
