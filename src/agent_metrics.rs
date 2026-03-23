//! # Module: Agent Metrics
//!
//! Structured, per-agent metrics tracking for multi-agent systems.
//!
//! ## Overview
//!
//! [`AgentMetrics`] tracks counters and timing data for a single agent session.
//! [`AgentMetricsRegistry`] manages a collection of [`AgentMetrics`] instances
//! (one per agent) behind a shared `Arc<Mutex<_>>` so they can be safely updated
//! from any async task.
//!
//! ## Quick Start
//!
//! ```rust
//! use llm_agent_runtime::agent_metrics::{AgentMetrics, AgentMetricsRegistry};
//!
//! // Registry is cheap to clone — all clones share the same data.
//! let registry = AgentMetricsRegistry::new();
//!
//! // Register a new agent and record activity.
//! registry.get_or_create("agent-1");
//! registry.record_step("agent-1", 42);
//! registry.record_tool_call("agent-1", false);
//! registry.record_tool_call("agent-1", true);  // failure
//!
//! let snap = registry.snapshot("agent-1").expect("agent exists");
//! assert_eq!(snap.total_steps, 1);
//! assert_eq!(snap.tool_calls_made, 2);
//! assert_eq!(snap.tool_call_failures, 1);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ── AgentMetrics ──────────────────────────────────────────────────────────────

/// Live, mutable metrics for a single agent session.
///
/// All fields are updated in-place by the `record_*` methods.
/// Call [`AgentMetrics::snapshot`] to obtain a serialisable point-in-time copy.
#[derive(Debug)]
pub struct AgentMetrics {
    /// Total number of ReAct (or plan-execute) steps completed.
    pub total_steps: u64,
    /// Total input tokens consumed across all LLM calls in this session.
    pub total_tokens_in: u64,
    /// Total output tokens produced across all LLM calls in this session.
    pub total_tokens_out: u64,
    /// Total number of tool calls dispatched (successful + failed).
    pub tool_calls_made: u64,
    /// Total number of tool calls that returned an error.
    pub tool_call_failures: u64,
    /// Running average of per-step wall-clock latency in milliseconds.
    ///
    /// Updated using an incremental mean: `avg = avg + (new - avg) / n`.
    pub avg_step_latency_ms: f64,
    /// Peak resident-set size in kilobytes, sampled at each `record_step` call.
    ///
    /// Because a portable `peak_rss` is not available without OS-specific APIs,
    /// callers supply this value explicitly; it defaults to `0`.
    pub peak_memory_kb: u64,
    /// Wall-clock instant at which this metrics object was created.
    pub session_start: Instant,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            total_steps: 0,
            total_tokens_in: 0,
            total_tokens_out: 0,
            tool_calls_made: 0,
            tool_call_failures: 0,
            avg_step_latency_ms: 0.0,
            peak_memory_kb: 0,
            session_start: Instant::now(),
        }
    }
}

impl AgentMetrics {
    /// Create a fresh `AgentMetrics` instance, starting the session timer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the completion of one agent step.
    ///
    /// `step_latency_ms` — wall-clock duration of the step in milliseconds.
    /// `memory_kb`       — current memory usage in KB (pass `0` if unknown).
    ///
    /// The running average latency is updated using an incremental mean so that
    /// the entire history does not need to be stored.
    pub fn record_step(&mut self, step_latency_ms: u64, memory_kb: u64) {
        self.total_steps += 1;
        // Incremental mean: avg_n = avg_{n-1} + (x_n - avg_{n-1}) / n
        let n = self.total_steps as f64;
        self.avg_step_latency_ms += (step_latency_ms as f64 - self.avg_step_latency_ms) / n;
        if memory_kb > self.peak_memory_kb {
            self.peak_memory_kb = memory_kb;
        }
    }

    /// Record a tool call.
    ///
    /// Set `failed` to `true` if the call returned an error or produced an
    /// error observation.
    pub fn record_tool_call(&mut self, failed: bool) {
        self.tool_calls_made += 1;
        if failed {
            self.tool_call_failures += 1;
        }
    }

    /// Record a token usage event for one LLM inference call.
    pub fn record_tokens(&mut self, tokens_in: u64, tokens_out: u64) {
        self.total_tokens_in += tokens_in;
        self.total_tokens_out += tokens_out;
    }

    /// Record a generic failure (increments `tool_call_failures` without
    /// incrementing `tool_calls_made`).
    ///
    /// Use this for failures that are not directly tied to a tool call, such
    /// as checkpoint errors or parsing failures.
    pub fn record_failure(&mut self) {
        self.tool_call_failures += 1;
    }

    /// Return the elapsed wall-clock time since the session started.
    pub fn session_duration(&self) -> std::time::Duration {
        self.session_start.elapsed()
    }

    /// Return the fraction of tool calls that failed, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no tool calls have been made.
    pub fn failure_rate(&self) -> f64 {
        if self.tool_calls_made == 0 {
            return 0.0;
        }
        self.tool_call_failures as f64 / self.tool_calls_made as f64
    }

    /// Capture a serialisable [`AgentMetricsSnapshot`] of the current state.
    pub fn snapshot(&self) -> AgentMetricsSnapshot {
        AgentMetricsSnapshot {
            total_steps: self.total_steps,
            total_tokens_in: self.total_tokens_in,
            total_tokens_out: self.total_tokens_out,
            tool_calls_made: self.tool_calls_made,
            tool_call_failures: self.tool_call_failures,
            avg_step_latency_ms: self.avg_step_latency_ms,
            peak_memory_kb: self.peak_memory_kb,
            session_duration_secs: self.session_duration().as_secs_f64(),
            failure_rate: self.failure_rate(),
        }
    }
}

// ── AgentMetricsSnapshot ──────────────────────────────────────────────────────

/// A serialisable, point-in-time capture of [`AgentMetrics`].
///
/// Unlike [`AgentMetrics`], this struct is `Clone + Serialize + Deserialize`
/// and can be logged, sent over the network, or stored to disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentMetricsSnapshot {
    /// Total ReAct/plan steps completed at snapshot time.
    pub total_steps: u64,
    /// Total input tokens consumed at snapshot time.
    pub total_tokens_in: u64,
    /// Total output tokens produced at snapshot time.
    pub total_tokens_out: u64,
    /// Total tool calls dispatched at snapshot time.
    pub tool_calls_made: u64,
    /// Total tool call failures at snapshot time.
    pub tool_call_failures: u64,
    /// Running average step latency in milliseconds at snapshot time.
    pub avg_step_latency_ms: f64,
    /// Peak memory usage in KB observed up to snapshot time.
    pub peak_memory_kb: u64,
    /// Elapsed session duration in seconds at snapshot time.
    pub session_duration_secs: f64,
    /// Tool-call failure rate `[0.0, 1.0]` at snapshot time.
    pub failure_rate: f64,
}

impl AgentMetricsSnapshot {
    /// Return `true` if no tool call failures have been recorded.
    pub fn is_healthy(&self) -> bool {
        self.tool_call_failures == 0
    }

    /// Return the total number of tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens_in + self.total_tokens_out
    }
}

// ── AgentMetricsRegistry ──────────────────────────────────────────────────────

/// A shared, multi-agent metrics registry.
///
/// All clones of an `AgentMetricsRegistry` refer to the same underlying data
/// (shared via `Arc<Mutex<_>>`).  It is cheap to clone and safe to share across
/// async tasks.
#[derive(Debug, Clone, Default)]
pub struct AgentMetricsRegistry {
    inner: Arc<Mutex<HashMap<String, AgentMetrics>>>,
}

impl AgentMetricsRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new agent (if not already present) and return a snapshot of
    /// the freshly created (or pre-existing) metrics.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the internal mutex has been poisoned (which cannot
    /// happen under normal operation).
    pub fn get_or_create(&self, agent_id: &str) -> AgentMetricsSnapshot {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let metrics = guard
            .entry(agent_id.to_string())
            .or_insert_with(AgentMetrics::new);
        metrics.snapshot()
    }

    /// Remove the metrics entry for `agent_id` and return its final snapshot.
    ///
    /// Returns `None` if the agent was not registered.
    pub fn remove(&self, agent_id: &str) -> Option<AgentMetricsSnapshot> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.remove(agent_id).map(|m| m.snapshot())
    }

    /// Record the completion of one step for `agent_id`.
    ///
    /// If no metrics entry exists for `agent_id` one is created automatically.
    ///
    /// `step_latency_ms` — step duration in ms.
    /// `memory_kb`       — current memory usage in KB (pass `0` if unknown).
    pub fn record_step(&self, agent_id: &str, step_latency_ms: u64) {
        self.record_step_with_memory(agent_id, step_latency_ms, 0);
    }

    /// Like [`record_step`] but also supplies a `memory_kb` sample.
    ///
    /// [`record_step`]: AgentMetricsRegistry::record_step
    pub fn record_step_with_memory(&self, agent_id: &str, step_latency_ms: u64, memory_kb: u64) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .entry(agent_id.to_string())
            .or_insert_with(AgentMetrics::new)
            .record_step(step_latency_ms, memory_kb);
    }

    /// Record a tool call for `agent_id`.
    ///
    /// Set `failed` to `true` if the call returned an error.
    pub fn record_tool_call(&self, agent_id: &str, failed: bool) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .entry(agent_id.to_string())
            .or_insert_with(AgentMetrics::new)
            .record_tool_call(failed);
    }

    /// Record token usage for `agent_id`.
    pub fn record_tokens(&self, agent_id: &str, tokens_in: u64, tokens_out: u64) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .entry(agent_id.to_string())
            .or_insert_with(AgentMetrics::new)
            .record_tokens(tokens_in, tokens_out);
    }

    /// Record a generic failure (not tied to a specific tool call) for `agent_id`.
    pub fn record_failure(&self, agent_id: &str) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .entry(agent_id.to_string())
            .or_insert_with(AgentMetrics::new)
            .record_failure();
    }

    /// Return a serialisable snapshot of the current metrics for `agent_id`.
    ///
    /// Returns `None` if no entry exists for that agent.
    pub fn snapshot(&self, agent_id: &str) -> Option<AgentMetricsSnapshot> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get(agent_id).map(|m| m.snapshot())
    }

    /// Return snapshots for all registered agents as a `HashMap`.
    pub fn snapshot_all(&self) -> HashMap<String, AgentMetricsSnapshot> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .iter()
            .map(|(id, m)| (id.clone(), m.snapshot()))
            .collect()
    }

    /// Return the number of agents currently tracked by the registry.
    pub fn agent_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Return `true` if no agents are registered.
    pub fn is_empty(&self) -> bool {
        self.agent_count() == 0
    }

    /// Return the IDs of all registered agents (order is unspecified).
    pub fn agent_ids(&self) -> Vec<String> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .cloned()
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    // ── AgentMetrics unit tests ───────────────────────────────────────────────

    #[test]
    fn new_metrics_start_at_zero() {
        let m = AgentMetrics::new();
        assert_eq!(m.total_steps, 0);
        assert_eq!(m.tool_calls_made, 0);
        assert_eq!(m.tool_call_failures, 0);
        assert_eq!(m.total_tokens_in, 0);
        assert_eq!(m.total_tokens_out, 0);
        assert_eq!(m.peak_memory_kb, 0);
        assert!((m.avg_step_latency_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn record_step_increments_counter() {
        let mut m = AgentMetrics::new();
        m.record_step(100, 512);
        assert_eq!(m.total_steps, 1);
        assert_eq!(m.peak_memory_kb, 512);
    }

    #[test]
    fn record_step_updates_average_latency() {
        let mut m = AgentMetrics::new();
        m.record_step(100, 0);
        m.record_step(200, 0);
        // avg should be 150
        assert!((m.avg_step_latency_ms - 150.0).abs() < 1e-9);
    }

    #[test]
    fn peak_memory_tracks_maximum() {
        let mut m = AgentMetrics::new();
        m.record_step(0, 100);
        m.record_step(0, 500);
        m.record_step(0, 200);
        assert_eq!(m.peak_memory_kb, 500);
    }

    #[test]
    fn record_tool_call_success() {
        let mut m = AgentMetrics::new();
        m.record_tool_call(false);
        assert_eq!(m.tool_calls_made, 1);
        assert_eq!(m.tool_call_failures, 0);
    }

    #[test]
    fn record_tool_call_failure() {
        let mut m = AgentMetrics::new();
        m.record_tool_call(true);
        assert_eq!(m.tool_calls_made, 1);
        assert_eq!(m.tool_call_failures, 1);
    }

    #[test]
    fn failure_rate_zero_when_no_calls() {
        let m = AgentMetrics::new();
        assert!((m.failure_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn failure_rate_half_when_half_fail() {
        let mut m = AgentMetrics::new();
        m.record_tool_call(false);
        m.record_tool_call(true);
        assert!((m.failure_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn record_tokens_accumulates() {
        let mut m = AgentMetrics::new();
        m.record_tokens(100, 50);
        m.record_tokens(200, 75);
        assert_eq!(m.total_tokens_in, 300);
        assert_eq!(m.total_tokens_out, 125);
    }

    #[test]
    fn record_failure_increments_failures_only() {
        let mut m = AgentMetrics::new();
        m.record_failure();
        assert_eq!(m.tool_call_failures, 1);
        assert_eq!(m.tool_calls_made, 0);
    }

    // ── AgentMetricsSnapshot tests ────────────────────────────────────────────

    #[test]
    fn snapshot_reflects_live_state() {
        let mut m = AgentMetrics::new();
        m.record_step(80, 256);
        m.record_tool_call(false);
        m.record_tokens(10, 5);
        let snap = m.snapshot();
        assert_eq!(snap.total_steps, 1);
        assert_eq!(snap.tool_calls_made, 1);
        assert_eq!(snap.total_tokens_in, 10);
        assert_eq!(snap.total_tokens_out, 5);
        assert_eq!(snap.peak_memory_kb, 256);
    }

    #[test]
    fn snapshot_is_healthy_when_no_failures() {
        let m = AgentMetrics::new();
        assert!(m.snapshot().is_healthy());
    }

    #[test]
    fn snapshot_not_healthy_after_failure() {
        let mut m = AgentMetrics::new();
        m.record_tool_call(true);
        assert!(!m.snapshot().is_healthy());
    }

    #[test]
    fn snapshot_total_tokens_sum() {
        let mut m = AgentMetrics::new();
        m.record_tokens(40, 60);
        let snap = m.snapshot();
        assert_eq!(snap.total_tokens(), 100);
    }

    // ── AgentMetricsRegistry tests ────────────────────────────────────────────

    #[test]
    fn registry_starts_empty() {
        let r = AgentMetricsRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.agent_count(), 0);
    }

    #[test]
    fn get_or_create_registers_new_agent() {
        let r = AgentMetricsRegistry::new();
        r.get_or_create("agent-1");
        assert_eq!(r.agent_count(), 1);
    }

    #[test]
    fn record_step_creates_entry_if_absent() {
        let r = AgentMetricsRegistry::new();
        r.record_step("agent-x", 50);
        let snap = r.snapshot("agent-x").expect("should exist");
        assert_eq!(snap.total_steps, 1);
    }

    #[test]
    fn record_tool_call_tracked_per_agent() {
        let r = AgentMetricsRegistry::new();
        r.record_tool_call("a1", false);
        r.record_tool_call("a1", true);
        r.record_tool_call("a2", false);

        let s1 = r.snapshot("a1").unwrap();
        let s2 = r.snapshot("a2").unwrap();

        assert_eq!(s1.tool_calls_made, 2);
        assert_eq!(s1.tool_call_failures, 1);
        assert_eq!(s2.tool_calls_made, 1);
        assert_eq!(s2.tool_call_failures, 0);
    }

    #[test]
    fn snapshot_all_covers_all_agents() {
        let r = AgentMetricsRegistry::new();
        r.record_step("a", 10);
        r.record_step("b", 20);
        let all = r.snapshot_all();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("a"));
        assert!(all.contains_key("b"));
    }

    #[test]
    fn remove_returns_final_snapshot() {
        let r = AgentMetricsRegistry::new();
        r.record_step("agent-z", 99);
        let snap = r.remove("agent-z").expect("should have been present");
        assert_eq!(snap.total_steps, 1);
        assert_eq!(r.agent_count(), 0);
    }

    #[test]
    fn clone_shares_data() {
        let r1 = AgentMetricsRegistry::new();
        let r2 = r1.clone();
        r1.record_step("shared", 10);
        // r2 should see the update because both share the same Arc
        assert!(r2.snapshot("shared").is_some());
    }

    #[test]
    fn record_tokens_via_registry() {
        let r = AgentMetricsRegistry::new();
        r.record_tokens("a", 100, 50);
        r.record_tokens("a", 200, 75);
        let snap = r.snapshot("a").unwrap();
        assert_eq!(snap.total_tokens_in, 300);
        assert_eq!(snap.total_tokens_out, 125);
    }

    #[test]
    fn record_failure_via_registry() {
        let r = AgentMetricsRegistry::new();
        r.record_failure("bot");
        let snap = r.snapshot("bot").unwrap();
        assert_eq!(snap.tool_call_failures, 1);
        assert_eq!(snap.tool_calls_made, 0);
    }

    #[test]
    fn agent_ids_lists_all_agents() {
        let r = AgentMetricsRegistry::new();
        r.get_or_create("alpha");
        r.get_or_create("beta");
        let mut ids = r.agent_ids();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn record_step_with_memory_updates_peak() {
        let r = AgentMetricsRegistry::new();
        r.record_step_with_memory("m", 10, 1024);
        r.record_step_with_memory("m", 20, 512);
        let snap = r.snapshot("m").unwrap();
        assert_eq!(snap.peak_memory_kb, 1024);
    }
}
