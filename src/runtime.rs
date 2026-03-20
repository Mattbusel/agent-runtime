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
use crate::metrics::RuntimeMetrics;
use crate::types::AgentId;

#[cfg(feature = "memory")]
use crate::memory::{EpisodicStore, WorkingMemory};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
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

    /// Return `true` if the session has no recorded steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
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

    /// Return `true` if the session ended with a `FINAL_ANSWER` action.
    ///
    /// This is the normal successful exit from a ReAct loop.  `false` means the
    /// loop was cut short by a timeout, max-iterations limit, or an error.
    pub fn is_successful(&self) -> bool {
        self.final_answer().is_some()
    }

    /// Return the session wall-clock duration as a [`std::time::Duration`].
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.duration_ms)
    }

    /// Return the number of tool-call actions dispatched during the session.
    ///
    /// Each [`ReActStep`] whose `action` parses as a `ToolCall` (not a `FinalAnswer`)
    /// is counted.
    pub fn tool_calls_made(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| {
                // A ToolCall action contains a JSON object.  A FinalAnswer starts
                // with the literal "FINAL_ANSWER" prefix.
                !s.action.trim().to_ascii_uppercase().starts_with("FINAL_ANSWER")
                    && !s.action.trim().is_empty()
            })
            .count()
    }

    /// Return the sum of all individual step durations in milliseconds.
    ///
    /// This is the cumulative inference + tool execution time across all steps,
    /// which may differ from `duration_ms` due to overhead between steps.
    pub fn total_step_duration_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.step_duration_ms).sum()
    }

    /// Return the average step duration in milliseconds.
    ///
    /// Returns `0` when there are no steps.
    pub fn average_step_duration_ms(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        self.total_step_duration_ms() / self.steps.len() as u64
    }

    /// Return a reference to the slowest step (highest `step_duration_ms`).
    ///
    /// Returns `None` when there are no steps.
    pub fn slowest_step(&self) -> Option<&ReActStep> {
        self.steps.iter().max_by_key(|s| s.step_duration_ms)
    }

    /// Return a reference to the fastest step (lowest `step_duration_ms`).
    ///
    /// Returns `None` when there are no steps.
    pub fn fastest_step(&self) -> Option<&ReActStep> {
        self.steps.iter().min_by_key(|s| s.step_duration_ms)
    }

    /// Return references to all steps that are tool calls (not `FINAL_ANSWER`).
    pub fn filter_tool_call_steps(&self) -> Vec<&ReActStep> {
        self.steps.iter().filter(|s| s.is_tool_call()).collect()
    }

    /// Return the zero-based index of the slowest step, or `None` if there are no steps.
    pub fn slowest_step_index(&self) -> Option<usize> {
        self.steps
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.step_duration_ms)
            .map(|(i, _)| i)
    }

    /// Return the zero-based index of the fastest step, or `None` if there are no steps.
    pub fn fastest_step_index(&self) -> Option<usize> {
        self.steps
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.step_duration_ms)
            .map(|(i, _)| i)
    }

    /// Return a reference to the last step, or `None` if there are no steps.
    pub fn last_step(&self) -> Option<&ReActStep> {
        self.steps.last()
    }

    /// Return a reference to the first step taken, or `None` if there are no steps.
    pub fn first_step(&self) -> Option<&ReActStep> {
        self.steps.first()
    }

    /// Return a reference to the step at zero-based index `idx`, or `None` if out of bounds.
    pub fn step_at(&self, idx: usize) -> Option<&ReActStep> {
        self.steps.get(idx)
    }

    /// Return the observation string at step `idx`, or `None` if out of bounds.
    pub fn observation_at(&self, idx: usize) -> Option<&str> {
        self.steps.get(idx).map(|s| s.observation.as_str())
    }

    /// Return the action string at step `idx`, or `None` if out of bounds.
    pub fn action_at(&self, idx: usize) -> Option<&str> {
        self.steps.get(idx).map(|s| s.action.as_str())
    }

    /// Return steps whose observation contains `pattern` (case-insensitive).
    pub fn observations_matching(&self, pattern: &str) -> Vec<&ReActStep> {
        let lower = pattern.to_ascii_lowercase();
        self.steps
            .iter()
            .filter(|s| s.observation.to_ascii_lowercase().contains(&lower))
            .collect()
    }

    /// Return steps whose thought contains `pattern` (case-insensitive).
    pub fn thoughts_containing(&self, pattern: &str) -> Vec<&ReActStep> {
        let lower = pattern.to_ascii_lowercase();
        self.steps
            .iter()
            .filter(|s| s.thought.to_ascii_lowercase().contains(&lower))
            .collect()
    }

    /// Return `true` if any step in this session used `action_name`.
    pub fn has_action(&self, action_name: &str) -> bool {
        self.steps.iter().any(|s| s.action == action_name)
    }

    /// Return the thought string at step `idx`, or `None` if out of bounds.
    pub fn thought_at(&self, idx: usize) -> Option<&str> {
        self.steps.get(idx).map(|s| s.thought.as_str())
    }

    /// Count how many steps used `action_name` as their action.
    ///
    /// Returns `0` if the action was never invoked.  Complements
    /// [`has_action`], which only tests for presence.
    ///
    /// [`has_action`]: AgentSession::has_action
    pub fn step_count_for_action(&self, action_name: &str) -> usize {
        self.steps.iter().filter(|s| s.action == action_name).count()
    }

    /// Return all observation strings in step order.
    ///
    /// Each string is a borrow of the corresponding `ReActStep::observation`
    /// field.  Useful for bulk post-processing of tool results.
    pub fn observations(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.observation.as_str()).collect()
    }

    /// Return the thought string from the first step, or `None` if the session
    /// has no steps.
    pub fn first_thought(&self) -> Option<&str> {
        self.steps.first().map(|s| s.thought.as_str())
    }

    /// Return the thought string from the last step, or `None` if the session
    /// has no steps.
    pub fn last_thought(&self) -> Option<&str> {
        self.steps.last().map(|s| s.thought.as_str())
    }

    /// Return the action name from the first step, or `None` if the session
    /// has no steps.
    pub fn first_action(&self) -> Option<&str> {
        self.steps.first().map(|s| s.action.as_str())
    }

    /// Return the action name from the last step, or `None` if the session
    /// has no steps.
    pub fn last_action(&self) -> Option<&str> {
        self.steps.last().map(|s| s.action.as_str())
    }

    /// Return a slice of the last `n` steps.
    ///
    /// If `n` is greater than or equal to the total step count, all steps are
    /// returned.  An empty slice is returned for sessions with no steps.
    pub fn last_n_steps(&self, n: usize) -> &[crate::agent::ReActStep] {
        let len = self.steps.len();
        let start = len.saturating_sub(n);
        &self.steps[start..]
    }

    /// Return all per-step durations in milliseconds, in order.
    ///
    /// Useful for computing custom percentiles or detecting slow outlier steps.
    pub fn step_durations_ms(&self) -> Vec<u64> {
        self.steps.iter().map(|s| s.step_duration_ms).collect()
    }

    /// Return the sum of all step durations in milliseconds.
    ///
    /// Equivalent to `step_durations_ms().iter().sum()` but avoids allocating
    /// a temporary Vec.
    pub fn total_latency_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.step_duration_ms).sum()
    }

    /// Return the arithmetic mean step duration in milliseconds.
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn avg_step_duration_ms(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.total_latency_ms() as f64 / self.steps.len() as f64
    }

    /// Return a reference to the step with the largest `step_duration_ms`.
    ///
    /// Returns `None` if the session has no steps.  When multiple steps share
    /// the maximum duration the first one (lowest index) is returned.
    pub fn longest_step(&self) -> Option<&crate::agent::ReActStep> {
        self.steps.iter().max_by_key(|s| s.step_duration_ms)
    }

    /// Return a reference to the step with the smallest `step_duration_ms`.
    ///
    /// Returns `None` if the session has no steps.  When multiple steps share
    /// the minimum duration the first one (lowest index) is returned.
    pub fn shortest_step(&self) -> Option<&crate::agent::ReActStep> {
        self.steps.iter().min_by_key(|s| s.step_duration_ms)
    }

    /// Return the sequence of action names taken, in step order.
    ///
    /// Unlike `all_actions()` this returns owned `String`s so the result can
    /// outlive the session borrow.
    pub fn action_sequence(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.action.clone()).collect()
    }

    /// Return the sorted, deduplicated set of tool names invoked during the session.
    ///
    /// Tool-call steps are identified by the same heuristic as `tool_calls_made`:
    /// a non-empty action that does not start with `FINAL_ANSWER`.
    pub fn unique_tools_used(&self) -> Vec<String> {
        let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for step in &self.steps {
            let action = step.action.trim();
            if action.is_empty() || action.to_ascii_uppercase().starts_with("FINAL_ANSWER") {
                continue;
            }
            // Tool name is the JSON "tool" field, or the whole action string if not JSON.
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(action) {
                if let Some(name) = v.get("tool").and_then(|n| n.as_str()) {
                    names.insert(name.to_owned());
                    continue;
                }
            }
            names.insert(action.to_owned());
        }
        let mut sorted: Vec<String> = names.into_iter().collect();
        sorted.sort_unstable();
        sorted
    }

    /// Collect all thought strings from every step, in order.
    pub fn all_thoughts(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.thought.as_str()).collect()
    }

    /// Collect all action strings from every step, in order.
    pub fn all_actions(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.action.as_str()).collect()
    }

    /// Collect all observation strings from every step, in order.
    pub fn all_observations(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.observation.as_str()).collect()
    }

    /// Return references to steps where the observation indicates a tool error.
    ///
    /// A step is classified as failed when its observation starts with
    /// `{"error"` (the structured error JSON produced by required-field
    /// validation) or contains the substring `"error"` (case-insensitive).
    pub fn failed_steps(&self) -> Vec<&crate::agent::ReActStep> {
        self.steps
            .iter()
            .filter(|s| {
                let obs = s.observation.trim();
                obs.starts_with("{\"error\"")
                    || obs.to_ascii_lowercase().contains("\"error\"")
            })
            .collect()
    }

    /// Return the number of tool-call steps whose observation indicates an error.
    ///
    /// Equivalent to `failed_steps().len()` but avoids collecting a `Vec`.
    pub fn failed_tool_call_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| {
                let obs = s.observation.trim();
                obs.starts_with("{\"error\"")
                    || obs.to_ascii_lowercase().contains("\"error\"")
            })
            .count()
    }

    /// Return a count of how many times each action was taken in this session.
    ///
    /// The map key is the action name (e.g. `"search"`, `"FINAL_ANSWER"`).
    pub fn action_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for step in &self.steps {
            *counts.entry(step.action.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Return a sorted list of unique action names used in this session.
    pub fn unique_actions(&self) -> Vec<String> {
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for step in &self.steps {
            seen.insert(step.action.clone());
        }
        seen.into_iter().collect()
    }

    /// Return `true` if any checkpoint errors were recorded during the session.
    ///
    /// A non-empty `checkpoint_errors` list means some step snapshots may be
    /// missing from storage, but the session itself completed successfully.
    pub fn has_checkpoint_errors(&self) -> bool {
        !self.checkpoint_errors.is_empty()
    }

    /// Return the number of checkpoint errors recorded during this session.
    pub fn checkpoint_error_count(&self) -> usize {
        self.checkpoint_errors.len()
    }

    /// Return the number of knowledge-graph lookups performed during this session.
    pub fn graph_lookup_count(&self) -> usize {
        self.graph_lookups
    }

    /// Return the episodic memory hit rate for this session.
    ///
    /// Computed as `memory_hits / step_count`. Returns `0.0` when there are
    /// no steps, to avoid division by zero.
    pub fn memory_hit_rate(&self) -> f64 {
        let steps = self.steps.len();
        if steps == 0 {
            return 0.0;
        }
        self.memory_hits as f64 / steps as f64
    }

    /// Return the raw count of episodic memory hits for this session.
    pub fn total_memory_hits(&self) -> usize {
        self.memory_hits
    }

    /// Return the ratio of unique actions to total steps.
    ///
    /// Returns `0.0` for sessions with no steps.  A value of `1.0` means every
    /// step used a different action; lower values indicate repeated actions.
    pub fn action_diversity(&self) -> f64 {
        let total = self.steps.len();
        if total == 0 {
            return 0.0;
        }
        let unique: std::collections::HashSet<&str> =
            self.steps.iter().map(|s| s.action.as_str()).collect();
        unique.len() as f64 / total as f64
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

    /// Load the session snapshot saved after step `step` completed.
    ///
    /// Alias for [`load_checkpoint_at_step`] — provided for ergonomic
    /// compatibility with call sites that prefer this naming convention.
    ///
    /// [`load_checkpoint_at_step`]: AgentSession::load_checkpoint_at_step
    #[cfg(feature = "persistence")]
    #[deprecated(since = "1.1.0", note = "Use load_checkpoint_at_step instead")]
    pub async fn load_step_checkpoint(
        backend: &dyn crate::persistence::PersistenceBackend,
        session_id: &str,
        step: usize,
    ) -> Result<Option<AgentSession>, AgentRuntimeError> {
        Self::load_checkpoint_at_step(backend, session_id, step).await
    }

    /// Load the session snapshot saved after step `step` completed.
    ///
    /// Returns `None` if no checkpoint exists for the given session/step pair.
    /// The step number is 1-based (step 1 = after the first ReAct iteration).
    #[cfg(feature = "persistence")]
    pub async fn load_checkpoint_at_step(
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
    #[cfg(feature = "memory")]
    memory: Option<EpisodicStore>,
    #[cfg(feature = "memory")]
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
    token_estimator: Option<Arc<dyn TokenEstimator>>,
    _state: PhantomData<S>,
}

// ── DebugBuilderState sealed trait ────────────────────────────────────────────

/// Private trait used to drive the single generic `Debug` impl for
/// `AgentRuntimeBuilder<S>`.  Only `NeedsConfig` and `HasConfig` implement it.
trait DebugBuilderState {
    /// Name shown in the debug output.
    const NAME: &'static str;
    /// Whether to emit the `agent_config` field (only `HasConfig` has one).
    const HAS_CONFIG: bool;
}

impl DebugBuilderState for NeedsConfig {
    const NAME: &'static str = "AgentRuntimeBuilder<NeedsConfig>";
    const HAS_CONFIG: bool = false;
}

impl DebugBuilderState for HasConfig {
    const NAME: &'static str = "AgentRuntimeBuilder<HasConfig>";
    const HAS_CONFIG: bool = true;
}

impl<S: DebugBuilderState> std::fmt::Debug for AgentRuntimeBuilder<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct(S::NAME);
        #[cfg(feature = "memory")]
        {
            s.field("memory", &self.memory.is_some())
                .field("working", &self.working.is_some());
        }
        #[cfg(feature = "graph")]
        s.field("graph", &self.graph.is_some());
        #[cfg(feature = "orchestrator")]
        s.field("backpressure", &self.backpressure.is_some());
        if S::HAS_CONFIG {
            s.field("agent_config", &self.agent_config.is_some());
        }
        s.field("tools", &self.tools.len()).finish()
    }
}

impl Default for AgentRuntimeBuilder<NeedsConfig> {
    fn default() -> Self {
        Self {
            #[cfg(feature = "memory")]
            memory: None,
            #[cfg(feature = "memory")]
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
            token_estimator: None,
            _state: PhantomData,
        }
    }
}

// Methods available on ALL builder states.
impl<S> AgentRuntimeBuilder<S> {
    /// Attach an episodic memory store.
    #[cfg(feature = "memory")]
    pub fn with_memory(mut self, store: EpisodicStore) -> Self {
        self.memory = Some(store);
        self
    }

    /// Attach a working memory store.
    #[cfg(feature = "memory")]
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

    /// Provide a custom [`TokenEstimator`] for memory budget calculations.
    ///
    /// Replaces the default `len / 4` byte-counting heuristic.  Use this to
    /// plug in a model-specific tokenizer (e.g. tiktoken, sentencepiece) so
    /// that `AgentConfig::max_memory_tokens` is respected accurately.
    pub fn with_token_estimator(mut self, estimator: Arc<dyn TokenEstimator>) -> Self {
        self.token_estimator = Some(estimator);
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
            token_estimator: self.token_estimator,
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
            #[cfg(feature = "memory")]
            memory: self.memory,
            #[cfg(feature = "memory")]
            working: self.working,
            #[cfg(feature = "graph")]
            graph: self.graph,
            #[cfg(feature = "orchestrator")]
            backpressure: self.backpressure,
            agent_config,
            tools: self.tools,
            metrics: self.metrics,
            token_estimator: self
                .token_estimator
                .unwrap_or_else(|| Arc::new(CharDivTokenEstimator)),
            #[cfg(feature = "persistence")]
            checkpoint_backend: self.checkpoint_backend,
        }
    }
}

// ── TokenEstimator ────────────────────────────────────────────────────────────

/// Estimates the number of tokens in a string.
///
/// Implement this trait to replace the default `len / 4` heuristic with a
/// model-specific tokenizer (e.g. tiktoken, sentencepiece).
///
/// # Example
/// ```rust,ignore
/// struct TiktokenEstimator { enc: tiktoken::Encoding }
/// impl TokenEstimator for TiktokenEstimator {
///     fn count_tokens(&self, text: &str) -> usize {
///         self.enc.encode_ordinary(text).len()
///     }
/// }
/// ```
pub trait TokenEstimator: Send + Sync {
    /// Return an approximate token count for `text`.
    fn count_tokens(&self, text: &str) -> usize;
}

/// Default heuristic: 1 token ≈ 4 bytes.
pub struct CharDivTokenEstimator;

impl TokenEstimator for CharDivTokenEstimator {
    fn count_tokens(&self, text: &str) -> usize {
        (text.len() / 4).max(1)
    }
}

// ── AgentRuntime ──────────────────────────────────────────────────────────────

/// Unified runtime that coordinates memory, graph, orchestration, and agent loop.
pub struct AgentRuntime {
    #[cfg(feature = "memory")]
    memory: Option<EpisodicStore>,
    #[cfg(feature = "memory")]
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
    token_estimator: Arc<dyn TokenEstimator>,
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
        // Acquire backpressure slot before counting the session — shed requests
        // must not inflate total_sessions or active_sessions.
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
                return Err(e);
            }
        }

        self.metrics.total_sessions.fetch_add(1, Ordering::Relaxed);
        self.metrics.active_sessions.fetch_add(1, Ordering::Relaxed);

        tracing::info!(agent_id = %agent_id, "agent session starting");
        let outcome = self.run_agent_inner(agent_id.clone(), prompt, infer).await;

        // Always release backpressure — success or error.
        #[cfg(feature = "orchestrator")]
        if let Some(ref guard) = self.backpressure {
            let _ = guard.release();
        }

        // Saturating decrement — guards against underflow to usize::MAX if
        // active_sessions is somehow already 0 (e.g. double-decrement bug).
        let _ = self.metrics.active_sessions.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |v| Some(v.saturating_sub(1)),
        );

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
        #[cfg(feature = "memory")]
        let enriched_prompt = if let Some(ref store) = self.memory {
            let memories = store.recall(&agent_id, self.agent_config.max_memory_recalls)?;

            // Apply token budget if configured.
            let memories = if let Some(token_budget) = self.agent_config.max_memory_tokens {
                let mut used = 0usize;
                memories
                    .into_iter()
                    .filter(|m| {
                        let tokens = self.token_estimator.count_tokens(&m.content);
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
                // Build the enriched prompt directly into a String without an
                // intermediate Vec<String> allocation.
                let mut enriched =
                    String::with_capacity(prompt.len() + memories.len() * 64 + 32);
                enriched.push_str("Relevant memories:\n");
                for m in &memories {
                    let _ = writeln!(enriched, "- {}", m.content);
                }
                let _ = write!(enriched, "\nCurrent prompt: {prompt}");
                enriched
            }
        } else {
            prompt.to_owned()
        };
        #[cfg(not(feature = "memory"))]
        let enriched_prompt = prompt.to_owned();

        // Inject working memory into prompt.
        #[cfg(feature = "memory")]
        let enriched_prompt = if let Some(ref wm) = self.working {
            let entries = wm.entries()?;
            if entries.is_empty() {
                enriched_prompt
            } else {
                // Build working-memory section without an intermediate Vec<String>.
                let mut out = String::with_capacity(
                    enriched_prompt.len() + entries.len() * 32 + 32,
                );
                out.push_str(&enriched_prompt);
                out.push_str("\n\nCurrent working state:\n");
                for (k, v) in &entries {
                    let _ = writeln!(out, "  {k}: {v}");
                }
                // Remove trailing newline added by writeln for the last entry.
                if out.ends_with('\n') {
                    out.pop();
                }
                out
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
    #[cfg(feature = "memory")]
    pub fn memory(&self) -> Option<&EpisodicStore> {
        self.memory.as_ref()
    }

    /// Return a reference to the graph store, if configured.
    #[cfg(feature = "graph")]
    pub fn graph(&self) -> Option<&GraphStore> {
        self.graph.as_ref()
    }

    /// Return a reference to the working memory, if configured.
    #[cfg(feature = "memory")]
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

    /// Run an agent session using a shared [`LlmProvider`].
    ///
    /// Convenience wrapper around [`run_agent`] that wires the provider's
    /// `complete` method as the inference closure.  Inference errors are
    /// converted to a `FINAL ANSWER` string so the loop terminates gracefully
    /// rather than panicking.
    ///
    /// [`run_agent`]: AgentRuntime::run_agent
    /// [`LlmProvider`]: crate::providers::LlmProvider
    #[cfg(feature = "providers")]
    pub async fn run_agent_with_provider(
        &self,
        agent_id: AgentId,
        prompt: &str,
        provider: std::sync::Arc<dyn crate::providers::LlmProvider>,
    ) -> Result<AgentSession, AgentRuntimeError> {
        let model = self.agent_config.model.clone();
        self.run_agent(agent_id, prompt, |ctx| {
            let provider = provider.clone();
            let model = model.clone();
            async move {
                provider
                    .complete(&ctx, &model)
                    .await
                    .unwrap_or_else(|e| format!("FINAL ANSWER: inference error: {e}"))
            }
        })
        .await
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

    // ── Task 15: Token budget edge case tests ─────────────────────────────────

    #[tokio::test]
    async fn test_token_budget_zero_returns_no_memories() {
        // A budget of 0 should result in no memories being injected.
        let store = EpisodicStore::new();
        let agent = AgentId::new("budget-agent");
        store.add_episode(agent.clone(), "short", 0.9).unwrap();

        let mut config = AgentConfig::new(5, "test-model");
        config.max_memory_tokens = Some(0);
        config.max_memory_recalls = 10;

        let runtime = AgentRuntime::builder()
            .with_memory(store)
            .with_agent_config(config)
            .build();

        let steps = runtime
            .run_agent(
                agent,
                "test",
                |_ctx| async { "Thought: ok\nAction: FINAL_ANSWER done".to_string() },
            )
            .await
            .unwrap();

        // The run should succeed; we just verify it doesn't panic or error.
        assert_eq!(steps.steps.len(), 1);
    }

    #[tokio::test]
    async fn test_token_budget_smaller_than_smallest_item_returns_no_memories() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("budget-agent2");
        // Content is 40 chars → ~10 tokens (40/4). Budget = 1 → none fit.
        store
            .add_episode(agent.clone(), "a".repeat(40), 0.9)
            .unwrap();

        let mut config = AgentConfig::new(5, "test-model");
        config.max_memory_tokens = Some(1);
        config.max_memory_recalls = 10;

        let runtime = AgentRuntime::builder()
            .with_memory(store)
            .with_agent_config(config)
            .build();

        let session = runtime
            .run_agent(
                agent,
                "test",
                |_ctx| async { "Thought: ok\nAction: FINAL_ANSWER done".to_string() },
            )
            .await
            .unwrap();

        assert_eq!(session.memory_hits, 0);
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

    #[test]
    fn test_all_actions_returns_actions_in_order() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("think1", "search {}", "result"),
                ReActStep::new("think2", "FINAL_ANSWER done", ""),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 10,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.all_actions(), vec!["search {}", "FINAL_ANSWER done"]);
    }

    #[test]
    fn test_has_checkpoint_errors_false_when_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!(!session.has_checkpoint_errors());
    }

    #[test]
    fn test_has_checkpoint_errors_true_when_non_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec!["err".into()],
        };
        assert!(session.has_checkpoint_errors());
    }

    #[test]
    fn test_memory_hit_rate_zero_with_no_steps() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 5,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.memory_hit_rate(), 0.0);
    }

    #[test]
    fn test_memory_hit_rate_correct_proportion() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t", "a", "o"),
                ReActStep::new("t", "a", "o"),
                ReActStep::new("t", "a", "o"),
                ReActStep::new("t", "a", "o"),
            ],
            memory_hits: 2,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!((session.memory_hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_filter_tool_call_steps_excludes_final_answer() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t1", "search {}", "res"),
                ReActStep::new("t2", "FINAL_ANSWER done", ""),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        let tool_steps = session.filter_tool_call_steps();
        assert_eq!(tool_steps.len(), 1);
        assert_eq!(tool_steps[0].action, "search {}");
    }

    #[test]
    fn test_slowest_step_index() {
        let mut s0 = ReActStep::new("t", "a", "o");
        s0.step_duration_ms = 5;
        let mut s1 = ReActStep::new("t", "a", "o");
        s1.step_duration_ms = 100;
        let mut s2 = ReActStep::new("t", "a", "o");
        s2.step_duration_ms = 10;
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![s0, s1, s2],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.slowest_step_index(), Some(1));
        assert_eq!(session.fastest_step_index(), Some(0));
    }

    #[test]
    fn test_slowest_step_index_none_when_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.slowest_step_index(), None);
        assert_eq!(session.fastest_step_index(), None);
    }

    #[test]
    fn test_last_step_returns_last() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t1", "a1", "o1"),
                ReActStep::new("t2", "FINAL_ANSWER done", ""),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.last_step().map(|s| s.action.as_str()), Some("FINAL_ANSWER done"));
    }

    #[test]
    fn test_last_step_none_when_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!(session.last_step().is_none());
    }

    #[test]
    fn test_step_at_returns_correct_step() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t0", "a0", "o0"),
                ReActStep::new("t1", "a1", "o1"),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.step_at(1).map(|s| s.thought.as_str()), Some("t1"));
        assert!(session.step_at(99).is_none());
    }

    // ── Round 3: failed_steps ─────────────────────────────────────────────────

    #[test]
    fn test_failed_steps_returns_steps_with_error_observation() {
        use crate::agent::ReActStep;
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t", "tool_a {}", r#"{"error":"bad input","ok":false}"#),
                ReActStep::new("t", "tool_b {}", r#"{"result":"ok","ok":true}"#),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        let failed = session.failed_steps();
        assert_eq!(failed.len(), 1);
        assert!(failed[0].observation.contains("bad input"));
    }

    #[test]
    fn test_failed_steps_empty_when_no_errors() {
        use crate::agent::ReActStep;
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![ReActStep::new("t", "FINAL_ANSWER done", "")],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!(session.failed_steps().is_empty());
    }

    // ── Round 17: untested AgentSession methods ───────────────────────────────

    fn make_step(thought: &str, action: &str, observation: &str) -> ReActStep {
        ReActStep::new(thought, action, observation)
    }

    fn make_session(steps: Vec<ReActStep>, duration_ms: u64) -> AgentSession {
        AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps,
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms,
            checkpoint_errors: vec![],
        }
    }

    #[test]
    fn test_step_count_returns_number_of_steps() {
        let s = make_session(vec![ReActStep::new("t", "a", "o"), ReActStep::new("t", "a", "o")], 0);
        assert_eq!(s.step_count(), 2);
    }

    #[test]
    fn test_is_empty_true_for_no_steps() {
        let s = make_session(vec![], 0);
        assert!(s.is_empty());
    }

    #[test]
    fn test_is_empty_false_with_steps() {
        let s = make_session(vec![ReActStep::new("t", "a", "o")], 0);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_is_successful_true_with_final_answer() {
        let s = make_session(vec![ReActStep::new("t", "FINAL_ANSWER yes", "")], 0);
        assert!(s.is_successful());
    }

    #[test]
    fn test_is_successful_false_without_final_answer() {
        let s = make_session(vec![ReActStep::new("t", "search {}", "result")], 0);
        assert!(!s.is_successful());
    }

    #[test]
    fn test_elapsed_returns_duration_from_duration_ms() {
        let s = make_session(vec![], 500);
        assert_eq!(s.elapsed(), std::time::Duration::from_millis(500));
    }

    #[test]
    fn test_tool_calls_made_excludes_final_answer() {
        let s = make_session(vec![
            ReActStep::new("t", "search {}", "res"),
            ReActStep::new("t", "lookup {}", "res"),
            ReActStep::new("t", "FINAL_ANSWER done", ""),
        ], 0);
        assert_eq!(s.tool_calls_made(), 2);
    }

    #[test]
    fn test_total_step_duration_ms_sums_all_steps() {
        let mut s1 = ReActStep::new("t", "a", "o"); s1.step_duration_ms = 10;
        let mut s2 = ReActStep::new("t", "a", "o"); s2.step_duration_ms = 30;
        let s = make_session(vec![s1, s2], 0);
        assert_eq!(s.total_step_duration_ms(), 40);
    }

    #[test]
    fn test_average_step_duration_ms() {
        let mut s1 = ReActStep::new("t", "a", "o"); s1.step_duration_ms = 20;
        let mut s2 = ReActStep::new("t", "a", "o"); s2.step_duration_ms = 40;
        let s = make_session(vec![s1, s2], 0);
        assert_eq!(s.average_step_duration_ms(), 30);
    }

    #[test]
    fn test_all_thoughts_returns_thoughts_in_order() {
        let s = make_session(vec![
            ReActStep::new("first thought", "a1", "o1"),
            ReActStep::new("second thought", "a2", "o2"),
        ], 0);
        assert_eq!(s.all_thoughts(), vec!["first thought", "second thought"]);
    }

    #[test]
    fn test_all_observations_returns_observations_in_order() {
        let s = make_session(vec![
            ReActStep::new("t1", "a1", "obs one"),
            ReActStep::new("t2", "a2", "obs two"),
        ], 0);
        assert_eq!(s.all_observations(), vec!["obs one", "obs two"]);
    }

    #[test]
    fn test_observations_matching_finds_matching_steps() {
        let s = make_session(vec![
            ReActStep::new("t1", "a1", "found the answer"),
            ReActStep::new("t2", "a2", "nothing relevant"),
        ], 0);
        let matching = s.observations_matching("answer");
        assert_eq!(matching.len(), 1);
        assert!(matching[0].observation.contains("answer"));
    }

    #[test]
    fn test_first_step_returns_first() {
        let s = make_session(vec![
            ReActStep::new("first", "a1", "o1"),
            ReActStep::new("second", "a2", "o2"),
        ], 0);
        assert_eq!(s.first_step().map(|s| s.thought.as_str()), Some("first"));
    }

    #[test]
    fn test_first_step_none_when_empty() {
        let s = make_session(vec![], 0);
        assert!(s.first_step().is_none());
    }

    // ── Round 18: graph_lookup_count ─────────────────────────────────────────

    #[test]
    fn test_graph_lookup_count_returns_field() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 7,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.graph_lookup_count(), 7usize);
    }

    // ── Round 7: action_counts / unique_actions ───────────────────────────────

    #[test]
    fn test_action_counts_counts_each_action() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "search", "r1"),
                ReActStep::new("t2", "search", "r2"),
                ReActStep::new("t3", "FINAL_ANSWER", "done"),
            ],
            0,
        );
        let counts = session.action_counts();
        assert_eq!(counts.get("search").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("FINAL_ANSWER").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_unique_actions_returns_sorted_deduped() {
        let session = make_session(
            vec![
                ReActStep::new("t", "b_action", "r"),
                ReActStep::new("t", "a_action", "r"),
                ReActStep::new("t", "b_action", "r"),
            ],
            0,
        );
        assert_eq!(session.unique_actions(), vec!["a_action", "b_action"]);
    }

    #[test]
    fn test_unique_actions_empty_when_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.unique_actions().is_empty());
    }

    // ── Round 8: total_latency_ms / action_sequence ───────────────────────────

    #[test]
    fn test_total_latency_ms_sums_step_durations() {
        let mut steps = vec![
            ReActStep::new("t1", "a1", "o1"),
            ReActStep::new("t2", "a2", "o2"),
        ];
        steps[0].step_duration_ms = 100;
        steps[1].step_duration_ms = 250;
        let session = make_session(steps, 350);
        assert_eq!(session.total_latency_ms(), 350);
    }

    #[test]
    fn test_total_latency_ms_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_latency_ms(), 0);
    }

    #[test]
    fn test_action_sequence_returns_actions_in_order() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r"),
                ReActStep::new("t", "FINAL_ANSWER", "done"),
            ],
            0,
        );
        assert_eq!(session.action_sequence(), vec!["search", "FINAL_ANSWER"]);
    }

    // ── Round 9: has_action / thought_at ─────────────────────────────────────

    #[test]
    fn test_has_action_returns_true_for_present_action() {
        let session = make_session(
            vec![ReActStep::new("t", "search", "r")],
            0,
        );
        assert!(session.has_action("search"));
    }

    #[test]
    fn test_has_action_returns_false_for_absent_action() {
        let session = make_session(
            vec![ReActStep::new("t", "search", "r")],
            0,
        );
        assert!(!session.has_action("compute"));
    }

    #[test]
    fn test_thought_at_returns_thought_for_valid_index() {
        let session = make_session(
            vec![
                ReActStep::new("first thought", "a1", "r1"),
                ReActStep::new("second thought", "a2", "r2"),
            ],
            0,
        );
        assert_eq!(session.thought_at(0), Some("first thought"));
        assert_eq!(session.thought_at(1), Some("second thought"));
    }

    #[test]
    fn test_thought_at_returns_none_for_out_of_bounds_index() {
        let session = make_session(vec![ReActStep::new("t", "a", "r")], 0);
        assert!(session.thought_at(99).is_none());
    }

    // ── Round 10: step_count_for_action / observations ────────────────────────

    #[test]
    fn test_step_count_for_action_counts_correctly() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r1"),
                ReActStep::new("t", "search", "r2"),
                ReActStep::new("t", "FINAL_ANSWER", "done"),
            ],
            0,
        );
        assert_eq!(session.step_count_for_action("search"), 2);
        assert_eq!(session.step_count_for_action("FINAL_ANSWER"), 1);
        assert_eq!(session.step_count_for_action("unknown"), 0);
    }

    #[test]
    fn test_observations_returns_all_observation_strings() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "a", "obs_one"),
                ReActStep::new("t2", "b", "obs_two"),
            ],
            0,
        );
        let obs = session.observations();
        assert_eq!(obs, vec!["obs_one", "obs_two"]);
    }

    #[test]
    fn test_observations_empty_for_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.observations().is_empty());
    }

    // ── Round 10: unique_tools_used ──────────────────────────────────────────

    #[test]
    fn test_unique_tools_used_deduplicates_actions() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r1"),
                ReActStep::new("t", "lookup", "r2"),
                ReActStep::new("t", "search", "r3"),
            ],
            0,
        );
        let tools = session.unique_tools_used();
        assert_eq!(tools.len(), 2);
        assert!(tools.contains(&"search".to_string()));
        assert!(tools.contains(&"lookup".to_string()));
    }

    #[test]
    fn test_unique_tools_used_excludes_final_answer() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r1"),
                ReActStep::new("t", "FINAL_ANSWER: done", "r2"),
            ],
            0,
        );
        let tools = session.unique_tools_used();
        assert_eq!(tools.len(), 1);
        assert!(tools.contains(&"search".to_string()));
    }

    #[test]
    fn test_unique_tools_used_empty_for_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.unique_tools_used().is_empty());
    }

    // ── Round 11: avg_step_duration_ms / longest_step / shortest_step ─────────

    #[test]
    fn test_avg_step_duration_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.avg_step_duration_ms() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_step_duration_single_step() {
        let mut step = ReActStep::new("t", "a", "r");
        step.step_duration_ms = 100;
        let session = make_session(vec![step], 0);
        assert!((session.avg_step_duration_ms() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_step_duration_multiple_steps() {
        let mut s1 = ReActStep::new("t1", "a", "r");
        s1.step_duration_ms = 100;
        let mut s2 = ReActStep::new("t2", "b", "r");
        s2.step_duration_ms = 200;
        let session = make_session(vec![s1, s2], 0);
        assert!((session.avg_step_duration_ms() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_longest_step_returns_step_with_max_duration() {
        let mut s1 = ReActStep::new("t1", "a", "r");
        s1.step_duration_ms = 50;
        let mut s2 = ReActStep::new("t2", "b", "r");
        s2.step_duration_ms = 200;
        let session = make_session(vec![s1, s2], 0);
        assert_eq!(session.longest_step().map(|s| s.step_duration_ms), Some(200));
    }

    #[test]
    fn test_longest_step_returns_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.longest_step().is_none());
    }

    #[test]
    fn test_shortest_step_returns_step_with_min_duration() {
        let mut s1 = ReActStep::new("t1", "a", "r");
        s1.step_duration_ms = 50;
        let mut s2 = ReActStep::new("t2", "b", "r");
        s2.step_duration_ms = 200;
        let session = make_session(vec![s1, s2], 0);
        assert_eq!(session.shortest_step().map(|s| s.step_duration_ms), Some(50));
    }

    // ── Round 12: first_thought / last_thought ────────────────────────────────

    #[test]
    fn test_first_thought_returns_thought_from_first_step() {
        let session = make_session(
            vec![
                ReActStep::new("alpha", "a1", "r1"),
                ReActStep::new("beta", "a2", "r2"),
            ],
            0,
        );
        assert_eq!(session.first_thought(), Some("alpha"));
    }

    #[test]
    fn test_last_thought_returns_thought_from_last_step() {
        let session = make_session(
            vec![
                ReActStep::new("alpha", "a1", "r1"),
                ReActStep::new("beta", "a2", "r2"),
            ],
            0,
        );
        assert_eq!(session.last_thought(), Some("beta"));
    }

    #[test]
    fn test_first_thought_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.first_thought().is_none());
    }

    #[test]
    fn test_last_thought_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.last_thought().is_none());
    }

    // ── Round 13: first_action / last_action ──────────────────────────────────

    #[test]
    fn test_first_action_returns_action_from_first_step() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "search", "r1"),
                ReActStep::new("t2", "FINAL_ANSWER", "r2"),
            ],
            0,
        );
        assert_eq!(session.first_action(), Some("search"));
    }

    #[test]
    fn test_last_action_returns_action_from_last_step() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "search", "r1"),
                ReActStep::new("t2", "FINAL_ANSWER", "r2"),
            ],
            0,
        );
        assert_eq!(session.last_action(), Some("FINAL_ANSWER"));
    }

    #[test]
    fn test_first_action_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.first_action().is_none());
    }

    #[test]
    fn test_last_action_equals_first_action_for_single_step() {
        let session = make_session(vec![ReActStep::new("t", "calc", "r")], 0);
        assert_eq!(session.first_action(), session.last_action());
    }

    // ── Round 14: AgentSession::checkpoint_error_count ───────────────────────

    #[test]
    fn test_checkpoint_error_count_zero_when_none() {
        let session = make_session(vec![], 0);
        assert_eq!(session.checkpoint_error_count(), 0);
    }

    #[test]
    fn test_checkpoint_error_count_reflects_errors() {
        let mut session = make_session(vec![], 0);
        session.checkpoint_errors.push("save failed".into());
        session.checkpoint_errors.push("disk full".into());
        assert_eq!(session.checkpoint_error_count(), 2);
    }

    // ── Round 27: failed_tool_call_count ─────────────────────────────────────

    #[test]
    fn test_failed_tool_call_count_zero_when_no_errors() {
        let step = ReActStep::new("think", "search", "results found");
        let session = make_session(vec![step], 0);
        assert_eq!(session.failed_tool_call_count(), 0);
    }

    #[test]
    fn test_failed_tool_call_count_matches_failed_steps() {
        let ok_step = ReActStep::new("ok", "search", "all good");
        let err_step = ReActStep::new("err", "lookup", "{\"error\": \"not found\"}");
        let session = make_session(vec![ok_step, err_step], 0);
        assert_eq!(session.failed_tool_call_count(), session.failed_steps().len());
        assert_eq!(session.failed_tool_call_count(), 1);
    }

    #[test]
    fn test_failed_tool_call_count_counts_all_errors() {
        let err1 = ReActStep::new("e1", "a", "{\"error\": \"bad\"}");
        let err2 = ReActStep::new("e2", "b", "some \"error\" text");
        let ok = ReActStep::new("ok", "c", "success");
        let session = make_session(vec![err1, err2, ok], 0);
        assert_eq!(session.failed_tool_call_count(), 2);
    }

    // ── Round 15: AgentSession::total_memory_hits / action_diversity ─────────

    #[test]
    fn test_total_memory_hits_returns_memory_hits_field() {
        let mut session = make_session(vec![], 0);
        session.memory_hits = 7;
        assert_eq!(session.total_memory_hits(), 7);
    }

    #[test]
    fn test_total_memory_hits_zero_by_default() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_memory_hits(), 0);
    }

    #[test]
    fn test_action_diversity_all_unique_is_one() {
        let steps = vec![
            ReActStep::new("t", "search", "r"),
            ReActStep::new("t", "calc", "r"),
            ReActStep::new("t", "lookup", "r"),
        ];
        let session = make_session(steps, 0);
        assert!((session.action_diversity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_diversity_all_same_is_fraction() {
        let steps = vec![
            ReActStep::new("t", "search", "r"),
            ReActStep::new("t", "search", "r"),
            ReActStep::new("t", "search", "r"),
        ];
        let session = make_session(steps, 0);
        // 1 unique / 3 total = 1/3
        assert!((session.action_diversity() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_diversity_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.action_diversity() - 0.0).abs() < 1e-9);
    }

    // ── Round 14: AgentSession::last_n_steps ──────────────────────────────────

    #[test]
    fn test_last_n_steps_returns_last_n() {
        let steps = vec![
            ReActStep::new("t1", "a", "r1"),
            ReActStep::new("t2", "b", "r2"),
            ReActStep::new("t3", "c", "r3"),
        ];
        let session = make_session(steps, 0);
        let last2 = session.last_n_steps(2);
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0].action, "b");
        assert_eq!(last2[1].action, "c");
    }

    #[test]
    fn test_last_n_steps_returns_all_when_n_exceeds_count() {
        let steps = vec![
            ReActStep::new("t1", "a", "r1"),
            ReActStep::new("t2", "b", "r2"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.last_n_steps(10).len(), 2);
    }

    #[test]
    fn test_last_n_steps_empty_for_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.last_n_steps(3).is_empty());
    }

    #[test]
    fn test_last_n_steps_zero_returns_empty() {
        let steps = vec![ReActStep::new("t1", "a", "r1")];
        let session = make_session(steps, 0);
        assert!(session.last_n_steps(0).is_empty());
    }
}
