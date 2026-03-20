//! # Module: Metrics
//!
//! Runtime observability counters for `AgentRuntime`.
//! All global counters use atomics for lock-free, thread-safe increment/read.
//! Per-tool counters use a `Mutex<HashMap>` to avoid requiring a concurrent
//! map dependency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ── LatencyHistogram ──────────────────────────────────────────────────────────

/// A simple fixed-bucket latency histogram.
///
/// Bucket upper bounds are **inclusive** (i.e., a sample of exactly 1 ms falls into bucket 0).
/// Bucket index mapping:
/// - 0: ≤ 1 ms
/// - 1: 2 – 5 ms
/// - 2: 6 – 10 ms
/// - 3: 11 – 50 ms
/// - 4: 51 – 100 ms
/// - 5: 101 – 500 ms
/// - 6: > 500 ms
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Counts per bucket. Index 0 = ≤1ms, …, index 6 = >500ms.
    /// Bucket upper bounds are **inclusive**.
    buckets: [AtomicU64; 7],
    total_count: AtomicU64,
    total_sum_ms: AtomicU64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            buckets: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            total_count: AtomicU64::new(0),
            total_sum_ms: AtomicU64::new(0),
        }
    }
}

impl LatencyHistogram {
    /// Bucket upper bounds in milliseconds.
    ///
    /// Boundaries were chosen to cover the full range of observed latencies in
    /// LLM-backed agent systems:
    ///
    /// | Bucket | Range      | Typical source                         |
    /// |--------|------------|----------------------------------------|
    /// | 0      | ≤ 1 ms     | In-process tool calls, cache hits      |
    /// | 1      | ≤ 5 ms     | Fast local I/O, simple calculations    |
    /// | 2      | ≤ 10 ms    | Network round-trips to local services  |
    /// | 3      | ≤ 50 ms    | p50 LLM token latency (streaming)      |
    /// | 4      | ≤ 100 ms   | p95 for small LLM completions          |
    /// | 5      | ≤ 500 ms   | p99 for medium LLM completions         |
    /// | 6      | > 500 ms   | Slow completions, network retries      |
    const BOUNDS: [u64; 7] = [1, 5, 10, 50, 100, 500, u64::MAX];

    /// Record a latency sample in milliseconds.
    pub fn record(&self, ms: u64) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_sum_ms.fetch_add(ms, Ordering::Relaxed);
        for (i, &bound) in Self::BOUNDS.iter().enumerate() {
            if ms <= bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Return the mean latency in ms, or 0.0 if no samples.
    pub fn mean_ms(&self) -> f64 {
        let count = self.total_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.total_sum_ms.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Return the total sample count.
    pub fn count(&self) -> u64 {
        self.total_count.load(Ordering::Relaxed)
    }

    /// Estimate the p-th percentile latency in milliseconds from the histogram.
    ///
    /// `p` must be in `[0.0, 1.0]`.  Returns the **upper bound** of the first
    /// bucket that contains the p-th percentile.  Returns `0` if no samples
    /// have been recorded.
    ///
    /// # Accuracy
    ///
    /// This is a bucket-boundary estimate, not an exact value.  The error is
    /// bounded by the bucket width at that percentile.
    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        let target = (p.clamp(0.0, 1.0) * total as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                return Self::BOUNDS[i];
            }
        }
        // All samples accounted for — return the last bound.
        *Self::BOUNDS.last().unwrap_or(&u64::MAX)
    }

    /// Return bucket counts as `(upper_bound_ms, count)` pairs.
    pub fn buckets(&self) -> Vec<(u64, u64)> {
        Self::BOUNDS
            .iter()
            .zip(self.buckets.iter())
            .map(|(&b, a)| (b, a.load(Ordering::Relaxed)))
            .collect()
    }

    /// Return the minimum recorded latency in ms, or `None` if no samples.
    pub fn min_ms(&self) -> Option<u64> {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        // Walk buckets from the fastest; the first non-empty bucket's lower
        // bound is 0 (or the previous bound), so return the upper bound as
        // the conservative minimum estimate.
        for (i, bucket) in self.buckets.iter().enumerate() {
            if bucket.load(Ordering::Relaxed) > 0 {
                return Some(if i == 0 { 0 } else { Self::BOUNDS[i - 1] + 1 });
            }
        }
        None
    }

    /// Return the maximum recorded latency in ms, or `None` if no samples.
    pub fn max_ms(&self) -> Option<u64> {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        // Walk from the slowest bucket; return the upper bound of the last non-empty bucket.
        for (i, bucket) in self.buckets.iter().enumerate().rev() {
            if bucket.load(Ordering::Relaxed) > 0 {
                return Some(Self::BOUNDS[i]);
            }
        }
        None
    }

    /// Return the 50th-percentile (median) latency in milliseconds.
    pub fn p50(&self) -> u64 {
        self.percentile(0.50)
    }

    /// Return the 95th-percentile latency in milliseconds.
    pub fn p95(&self) -> u64 {
        self.percentile(0.95)
    }

    /// Return the 99th-percentile latency in milliseconds.
    pub fn p99(&self) -> u64 {
        self.percentile(0.99)
    }

    /// Reset all histogram counters to zero.
    pub fn reset(&self) {
        self.total_count.store(0, Ordering::Relaxed);
        self.total_sum_ms.store(0, Ordering::Relaxed);
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
    }
}

impl MetricsSnapshot {
    /// Compute the difference between `after` and `before` (i.e., `after - before`).
    ///
    /// Useful for per-request instrumentation:
    /// ```rust,ignore
    /// let before = metrics.snapshot();
    /// // ... run one agent invocation ...
    /// let after = metrics.snapshot();
    /// let delta = MetricsSnapshot::delta(&after, &before);
    /// println!("steps this run: {}", delta.total_steps);
    /// ```
    ///
    /// Saturating subtraction is used so callers don't need to guard against
    /// races where a counter is read before the full increment propagates.
    pub fn delta(after: &Self, before: &Self) -> Self {
        Self {
            active_sessions: after.active_sessions.saturating_sub(before.active_sessions),
            total_sessions: after.total_sessions.saturating_sub(before.total_sessions),
            total_steps: after.total_steps.saturating_sub(before.total_steps),
            total_tool_calls: after.total_tool_calls.saturating_sub(before.total_tool_calls),
            failed_tool_calls: after.failed_tool_calls.saturating_sub(before.failed_tool_calls),
            backpressure_shed_count: after
                .backpressure_shed_count
                .saturating_sub(before.backpressure_shed_count),
            memory_recall_count: after
                .memory_recall_count
                .saturating_sub(before.memory_recall_count),
            checkpoint_errors: after
                .checkpoint_errors
                .saturating_sub(before.checkpoint_errors),
            per_tool_calls: {
                let mut m = after.per_tool_calls.clone();
                for (k, v) in &before.per_tool_calls {
                    let entry = m.entry(k.clone()).or_default();
                    *entry = entry.saturating_sub(*v);
                }
                m
            },
            per_tool_failures: {
                let mut m = after.per_tool_failures.clone();
                for (k, v) in &before.per_tool_failures {
                    let entry = m.entry(k.clone()).or_default();
                    *entry = entry.saturating_sub(*v);
                }
                m
            },
            step_latency_buckets: after
                .step_latency_buckets
                .iter()
                .zip(before.step_latency_buckets.iter())
                .map(|((bound, a), (_, b))| (*bound, a.saturating_sub(*b)))
                .collect(),
            step_latency_mean_ms: after.step_latency_mean_ms - before.step_latency_mean_ms,
            per_agent_tool_calls: after.per_agent_tool_calls.clone(),
            per_agent_tool_failures: after.per_agent_tool_failures.clone(),
        }
    }

    /// Serialize the snapshot to a `serde_json::Value` for logging or export.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "active_sessions": self.active_sessions,
            "total_sessions": self.total_sessions,
            "total_steps": self.total_steps,
            "total_tool_calls": self.total_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "backpressure_shed_count": self.backpressure_shed_count,
            "memory_recall_count": self.memory_recall_count,
            "step_latency_mean_ms": self.step_latency_mean_ms,
            "per_tool_calls": self.per_tool_calls,
            "per_tool_failures": self.per_tool_failures,
        })
    }
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MetricsSnapshot {{ sessions: active={} total={}, steps={}, \
             tool_calls={} (failed={}), backpressure_shed={}, \
             memory_recalls={}, checkpoint_errors={}, latency_mean={:.1}ms }}",
            self.active_sessions,
            self.total_sessions,
            self.total_steps,
            self.total_tool_calls,
            self.failed_tool_calls,
            self.backpressure_shed_count,
            self.memory_recall_count,
            self.checkpoint_errors,
            self.step_latency_mean_ms,
        )
    }
}

/// A point-in-time snapshot of all runtime counters.
///
/// Obtained by calling [`RuntimeMetrics::snapshot`].  All fields are plain
/// integers so the snapshot can be logged, serialised, or diffed without
/// holding any locks.
///
/// See also [`snapshot`] for a richer snapshot including per-tool and histogram data.
///
/// # Example
/// ```rust
/// use llm_agent_runtime::metrics::RuntimeMetrics;
///
/// let m = RuntimeMetrics::new();
/// let snap = m.snapshot();
/// assert_eq!(snap.active_sessions, 0);
/// assert_eq!(snap.total_sessions, 0);
/// ```
///
/// [`snapshot`]: RuntimeMetrics::snapshot
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Number of agent sessions currently in progress.
    pub active_sessions: usize,
    /// Total number of sessions started since the runtime was created.
    pub total_sessions: u64,
    /// Total number of ReAct steps executed across all sessions.
    pub total_steps: u64,
    /// Total number of tool calls dispatched (across all tool names).
    pub total_tool_calls: u64,
    /// Total number of tool calls that returned an error observation.
    pub failed_tool_calls: u64,
    /// Total number of requests shed due to backpressure.
    pub backpressure_shed_count: u64,
    /// Total number of memory recall operations.
    pub memory_recall_count: u64,
    /// Total number of checkpoint failures encountered during `run_agent`.
    pub checkpoint_errors: u64,
    /// Per-tool call counts: `tool_name → total_calls`.
    pub per_tool_calls: HashMap<String, u64>,
    /// Per-tool failure counts: `tool_name → failed_calls`.
    pub per_tool_failures: HashMap<String, u64>,
    /// Step latency histogram bucket counts as `(upper_bound_ms_inclusive, count)`.
    pub step_latency_buckets: Vec<(u64, u64)>,
    /// Mean step latency in milliseconds.
    pub step_latency_mean_ms: f64,
    /// Per-agent, per-tool call counts: `agent_id → tool_name → count`.
    pub per_agent_tool_calls: HashMap<String, HashMap<String, u64>>,
    /// Per-agent, per-tool failure counts: `agent_id → tool_name → count`.
    pub per_agent_tool_failures: HashMap<String, HashMap<String, u64>>,
}

/// All four per-tool / per-agent counter maps, protected by a single lock.
///
/// Grouping them under one `Mutex` halves lock acquisitions on the hot path
/// (a single `record_tool_call` + `record_agent_tool_call` pair now requires
/// only one acquire/release) and makes snapshot reads cheaper too.
#[derive(Debug, Default)]
struct PerToolMaps {
    /// Per-tool call counts: `tool_name → total_calls`.
    calls: HashMap<String, u64>,
    /// Per-tool failure counts: `tool_name → failed_calls`.
    failures: HashMap<String, u64>,
    /// Per-agent, per-tool call counts: `agent_id → tool_name → count`.
    agent_calls: HashMap<String, HashMap<String, u64>>,
    /// Per-agent, per-tool failure counts: `agent_id → tool_name → count`.
    agent_failures: HashMap<String, HashMap<String, u64>>,
}

/// Shared runtime metrics. Clone the `Arc` to share across threads.
#[derive(Debug)]
pub struct RuntimeMetrics {
    /// Number of agent sessions currently in progress.
    pub active_sessions: AtomicUsize,
    /// Total number of sessions started since the runtime was created.
    pub total_sessions: AtomicU64,
    /// Total number of ReAct steps executed across all sessions.
    pub total_steps: AtomicU64,
    /// Total number of tool calls dispatched (across all tool names).
    pub total_tool_calls: AtomicU64,
    /// Total number of tool calls that returned an error observation.
    pub failed_tool_calls: AtomicU64,
    /// Total number of requests shed due to backpressure.
    pub backpressure_shed_count: AtomicU64,
    /// Total number of memory recall operations.
    pub memory_recall_count: AtomicU64,
    /// Total number of checkpoint failures encountered during `run_agent`.
    pub checkpoint_errors: AtomicU64,
    /// All four per-tool / per-agent maps under a single lock.
    per_tool: Mutex<PerToolMaps>,
    /// Per-step latency histogram.
    pub step_latency: LatencyHistogram,
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self {
            active_sessions: AtomicUsize::new(0),
            total_sessions: AtomicU64::new(0),
            total_steps: AtomicU64::new(0),
            total_tool_calls: AtomicU64::new(0),
            failed_tool_calls: AtomicU64::new(0),
            backpressure_shed_count: AtomicU64::new(0),
            memory_recall_count: AtomicU64::new(0),
            checkpoint_errors: AtomicU64::new(0),
            per_tool: Mutex::new(PerToolMaps::default()),
            step_latency: LatencyHistogram::default(),
        }
    }
}

impl RuntimeMetrics {
    /// Allocate a new `RuntimeMetrics` instance wrapped in an `Arc`.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Return the number of agent sessions currently in progress.
    pub fn active_sessions(&self) -> usize {
        self.active_sessions.load(Ordering::Relaxed)
    }

    /// Return the total number of sessions started since the runtime was created.
    pub fn total_sessions(&self) -> u64 {
        self.total_sessions.load(Ordering::Relaxed)
    }

    /// Return the total number of ReAct steps executed across all sessions.
    pub fn total_steps(&self) -> u64 {
        self.total_steps.load(Ordering::Relaxed)
    }

    /// Return the total number of tool calls dispatched.
    pub fn total_tool_calls(&self) -> u64 {
        self.total_tool_calls.load(Ordering::Relaxed)
    }

    /// Return the total number of tool calls that returned an error observation.
    pub fn failed_tool_calls(&self) -> u64 {
        self.failed_tool_calls.load(Ordering::Relaxed)
    }

    /// Return the total number of requests shed due to backpressure.
    pub fn backpressure_shed_count(&self) -> u64 {
        self.backpressure_shed_count.load(Ordering::Relaxed)
    }

    /// Return the total number of memory recall operations performed.
    pub fn memory_recall_count(&self) -> u64 {
        self.memory_recall_count.load(Ordering::Relaxed)
    }

    /// Return the total number of checkpoint failures encountered during `run_agent`.
    pub fn checkpoint_errors(&self) -> u64 {
        self.checkpoint_errors.load(Ordering::Relaxed)
    }

    /// Increment the call counter for `tool_name` by 1.
    ///
    /// Called automatically by the agent loop when `with_metrics` is configured.
    pub fn record_tool_call(&self, tool_name: &str) {
        self.total_tool_calls.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut maps) = self.per_tool.lock() {
            *maps.calls.entry(tool_name.to_owned()).or_insert(0) += 1;
        }
    }

    /// Increment the failure counter for `tool_name` by 1.
    ///
    /// Called automatically by the agent loop when a tool returns an error.
    pub fn record_tool_failure(&self, tool_name: &str) {
        self.failed_tool_calls.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut maps) = self.per_tool.lock() {
            *maps.failures.entry(tool_name.to_owned()).or_insert(0) += 1;
        }
    }

    /// Return a snapshot of per-tool call counts as a `HashMap<tool_name, count>`.
    pub fn per_tool_calls_snapshot(&self) -> HashMap<String, u64> {
        self.per_tool
            .lock()
            .map(|m| m.calls.clone())
            .unwrap_or_default()
    }

    /// Return a snapshot of per-tool failure counts as a `HashMap<tool_name, count>`.
    pub fn per_tool_failures_snapshot(&self) -> HashMap<String, u64> {
        self.per_tool
            .lock()
            .map(|m| m.failures.clone())
            .unwrap_or_default()
    }

    /// Increment call counter for (agent_id, tool_name).
    pub fn record_agent_tool_call(&self, agent_id: &str, tool_name: &str) {
        if let Ok(mut maps) = self.per_tool.lock() {
            *maps
                .agent_calls
                .entry(agent_id.to_owned())
                .or_default()
                .entry(tool_name.to_owned())
                .or_insert(0) += 1;
        }
    }

    /// Increment failure counter for (agent_id, tool_name).
    pub fn record_agent_tool_failure(&self, agent_id: &str, tool_name: &str) {
        if let Ok(mut maps) = self.per_tool.lock() {
            *maps
                .agent_failures
                .entry(agent_id.to_owned())
                .or_default()
                .entry(tool_name.to_owned())
                .or_insert(0) += 1;
        }
    }

    /// Snapshot of per-agent, per-tool call counts.
    pub fn per_agent_tool_calls_snapshot(&self) -> HashMap<String, HashMap<String, u64>> {
        self.per_tool
            .lock()
            .map(|m| m.agent_calls.clone())
            .unwrap_or_default()
    }

    /// Snapshot of per-agent, per-tool failure counts.
    pub fn per_agent_tool_failures_snapshot(&self) -> HashMap<String, HashMap<String, u64>> {
        self.per_tool
            .lock()
            .map(|m| m.agent_failures.clone())
            .unwrap_or_default()
    }

    /// Capture a complete snapshot of all counters, including per-tool breakdowns.
    ///
    /// This is the preferred alternative to [`to_snapshot`] — it returns a
    /// named [`MetricsSnapshot`] struct instead of an opaque tuple.
    ///
    /// [`to_snapshot`]: RuntimeMetrics::to_snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        // Acquire the single per-tool lock once for all four maps.
        let (per_tool_calls, per_tool_failures, per_agent_tool_calls, per_agent_tool_failures) =
            self.per_tool
                .lock()
                .map(|m| {
                    (
                        m.calls.clone(),
                        m.failures.clone(),
                        m.agent_calls.clone(),
                        m.agent_failures.clone(),
                    )
                })
                .unwrap_or_default();

        MetricsSnapshot {
            active_sessions: self.active_sessions.load(Ordering::Relaxed),
            total_sessions: self.total_sessions.load(Ordering::Relaxed),
            total_steps: self.total_steps.load(Ordering::Relaxed),
            total_tool_calls: self.total_tool_calls.load(Ordering::Relaxed),
            failed_tool_calls: self.failed_tool_calls.load(Ordering::Relaxed),
            backpressure_shed_count: self.backpressure_shed_count.load(Ordering::Relaxed),
            memory_recall_count: self.memory_recall_count.load(Ordering::Relaxed),
            checkpoint_errors: self.checkpoint_errors.load(Ordering::Relaxed),
            per_tool_calls,
            per_tool_failures,
            step_latency_buckets: self.step_latency.buckets(),
            step_latency_mean_ms: self.step_latency.mean_ms(),
            per_agent_tool_calls,
            per_agent_tool_failures,
        }
    }

    /// Record a step latency sample.
    pub fn record_step_latency(&self, ms: u64) {
        self.step_latency.record(ms);
    }

    /// Reset all counters to zero.
    ///
    /// Intended for testing. In production, counters are monotonically increasing.
    pub fn reset(&self) {
        self.active_sessions.store(0, Ordering::Relaxed);
        self.total_sessions.store(0, Ordering::Relaxed);
        self.total_steps.store(0, Ordering::Relaxed);
        self.total_tool_calls.store(0, Ordering::Relaxed);
        self.failed_tool_calls.store(0, Ordering::Relaxed);
        self.backpressure_shed_count.store(0, Ordering::Relaxed);
        self.memory_recall_count.store(0, Ordering::Relaxed);
        self.checkpoint_errors.store(0, Ordering::Relaxed);
        if let Ok(mut maps) = self.per_tool.lock() {
            maps.calls.clear();
            maps.failures.clear();
            maps.agent_calls.clear();
            maps.agent_failures.clear();
        }
        self.step_latency.reset();
    }

    /// Capture a snapshot of global counters as plain integers.
    ///
    /// Returns `(active_sessions, total_sessions, total_steps,
    ///           total_tool_calls, failed_tool_calls,
    ///           backpressure_shed_count, memory_recall_count)`.
    /// For per-tool breakdowns use [`per_tool_calls_snapshot`] and
    /// [`per_tool_failures_snapshot`].
    ///
    /// # Deprecation
    ///
    /// Prefer [`snapshot`] which returns the named [`MetricsSnapshot`] struct
    /// and includes per-tool, per-agent, and histogram data.  This method
    /// returns an anonymous tuple whose field order is easy to misread.
    ///
    /// [`snapshot`]: RuntimeMetrics::snapshot
    /// [`per_tool_calls_snapshot`]: RuntimeMetrics::per_tool_calls_snapshot
    /// [`per_tool_failures_snapshot`]: RuntimeMetrics::per_tool_failures_snapshot
    #[deprecated(since = "1.0.3", note = "use `snapshot()` which returns the named MetricsSnapshot struct")]
    pub fn to_snapshot(&self) -> (usize, u64, u64, u64, u64, u64, u64) {
        (
            self.active_sessions.load(Ordering::Relaxed),
            self.total_sessions.load(Ordering::Relaxed),
            self.total_steps.load(Ordering::Relaxed),
            self.total_tool_calls.load(Ordering::Relaxed),
            self.failed_tool_calls.load(Ordering::Relaxed),
            self.backpressure_shed_count.load(Ordering::Relaxed),
            self.memory_recall_count.load(Ordering::Relaxed),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new_returns_arc_with_zero_counters() {
        let m = RuntimeMetrics::new();
        assert_eq!(m.active_sessions(), 0);
        assert_eq!(m.total_sessions(), 0);
        assert_eq!(m.total_steps(), 0);
        assert_eq!(m.total_tool_calls(), 0);
        assert_eq!(m.failed_tool_calls(), 0);
        assert_eq!(m.backpressure_shed_count(), 0);
        assert_eq!(m.memory_recall_count(), 0);
    }

    #[test]
    fn test_active_sessions_increments_and_decrements() {
        let m = RuntimeMetrics::new();
        m.active_sessions.fetch_add(1, Ordering::Relaxed);
        assert_eq!(m.active_sessions(), 1);
        m.active_sessions.fetch_sub(1, Ordering::Relaxed);
        assert_eq!(m.active_sessions(), 0);
    }

    #[test]
    fn test_total_sessions_increments() {
        let m = RuntimeMetrics::new();
        m.total_sessions.fetch_add(1, Ordering::Relaxed);
        m.total_sessions.fetch_add(1, Ordering::Relaxed);
        assert_eq!(m.total_sessions(), 2);
    }

    #[test]
    fn test_total_steps_increments() {
        let m = RuntimeMetrics::new();
        m.total_steps.fetch_add(5, Ordering::Relaxed);
        assert_eq!(m.total_steps(), 5);
    }

    #[test]
    fn test_total_tool_calls_increments() {
        let m = RuntimeMetrics::new();
        m.total_tool_calls.fetch_add(3, Ordering::Relaxed);
        assert_eq!(m.total_tool_calls(), 3);
    }

    #[test]
    fn test_failed_tool_calls_increments() {
        let m = RuntimeMetrics::new();
        m.failed_tool_calls.fetch_add(2, Ordering::Relaxed);
        assert_eq!(m.failed_tool_calls(), 2);
    }

    #[test]
    fn test_backpressure_shed_count_increments() {
        let m = RuntimeMetrics::new();
        m.backpressure_shed_count.fetch_add(7, Ordering::Relaxed);
        assert_eq!(m.backpressure_shed_count(), 7);
    }

    #[test]
    fn test_memory_recall_count_increments() {
        let m = RuntimeMetrics::new();
        m.memory_recall_count.fetch_add(4, Ordering::Relaxed);
        assert_eq!(m.memory_recall_count(), 4);
    }

    #[test]
    fn test_reset_zeroes_all_counters() {
        let m = RuntimeMetrics::new();
        m.active_sessions.store(3, Ordering::Relaxed);
        m.total_sessions.store(10, Ordering::Relaxed);
        m.total_steps.store(50, Ordering::Relaxed);
        m.total_tool_calls.store(20, Ordering::Relaxed);
        m.failed_tool_calls.store(2, Ordering::Relaxed);
        m.backpressure_shed_count.store(1, Ordering::Relaxed);
        m.memory_recall_count.store(8, Ordering::Relaxed);

        m.reset();

        assert_eq!(m.active_sessions(), 0);
        assert_eq!(m.total_sessions(), 0);
        assert_eq!(m.total_steps(), 0);
        assert_eq!(m.total_tool_calls(), 0);
        assert_eq!(m.failed_tool_calls(), 0);
        assert_eq!(m.backpressure_shed_count(), 0);
        assert_eq!(m.memory_recall_count(), 0);
    }

    #[test]
    fn test_to_snapshot_captures_correct_values() {
        let m = RuntimeMetrics::new();
        m.active_sessions.store(1, Ordering::Relaxed);
        m.total_sessions.store(2, Ordering::Relaxed);
        m.total_steps.store(3, Ordering::Relaxed);
        m.total_tool_calls.store(4, Ordering::Relaxed);
        m.failed_tool_calls.store(5, Ordering::Relaxed);
        m.backpressure_shed_count.store(6, Ordering::Relaxed);
        m.memory_recall_count.store(7, Ordering::Relaxed);

        let snap = m.to_snapshot();
        assert_eq!(snap, (1, 2, 3, 4, 5, 6, 7));
    }

    #[test]
    fn test_metrics_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RuntimeMetrics>();
    }

    #[test]
    fn test_multiple_increments_are_cumulative() {
        let m = RuntimeMetrics::new();
        for _ in 0..100 {
            m.total_sessions.fetch_add(1, Ordering::Relaxed);
        }
        assert_eq!(m.total_sessions(), 100);
    }

    #[test]
    fn test_arc_clone_shares_state() {
        let m = RuntimeMetrics::new();
        let m2 = Arc::clone(&m);
        m.total_sessions.fetch_add(1, Ordering::Relaxed);
        assert_eq!(m2.total_sessions(), 1);
    }

    // ── Per-tool metrics ──────────────────────────────────────────────────────

    #[test]
    fn test_record_tool_call_increments_global_and_per_tool() {
        let m = RuntimeMetrics::new();
        m.record_tool_call("search");
        m.record_tool_call("search");
        m.record_tool_call("lookup");
        assert_eq!(m.total_tool_calls(), 3);
        let snap = m.per_tool_calls_snapshot();
        assert_eq!(snap.get("search").copied(), Some(2));
        assert_eq!(snap.get("lookup").copied(), Some(1));
    }

    #[test]
    fn test_record_tool_failure_increments_global_and_per_tool() {
        let m = RuntimeMetrics::new();
        m.record_tool_failure("search");
        m.record_tool_failure("lookup");
        m.record_tool_failure("search");
        assert_eq!(m.failed_tool_calls(), 3);
        let snap = m.per_tool_failures_snapshot();
        assert_eq!(snap.get("search").copied(), Some(2));
        assert_eq!(snap.get("lookup").copied(), Some(1));
    }

    #[test]
    fn test_reset_clears_per_tool_counters() {
        let m = RuntimeMetrics::new();
        m.record_tool_call("foo");
        m.record_tool_failure("foo");
        m.reset();
        assert!(m.per_tool_calls_snapshot().is_empty());
        assert!(m.per_tool_failures_snapshot().is_empty());
    }

    #[test]
    fn test_per_tool_snapshot_is_independent_for_unknown_tools() {
        let m = RuntimeMetrics::new();
        let snap = m.per_tool_calls_snapshot();
        assert!(snap.is_empty());
    }

    // ── LatencyHistogram ───────────────────────────────────────────────────────

    #[test]
    fn test_latency_histogram_records_sample() {
        let h = LatencyHistogram::default();
        h.record(10);
        assert_eq!(h.count(), 1);
    }

    #[test]
    fn test_latency_histogram_mean_ms() {
        let h = LatencyHistogram::default();
        h.record(10);
        h.record(20);
        assert!((h.mean_ms() - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_latency_histogram_buckets_correct_bucket() {
        let h = LatencyHistogram::default();
        h.record(3); // falls in ≤5ms bucket (index 1)
        let buckets = h.buckets();
        // bucket at index 1 is ≤5ms
        assert_eq!(buckets[1].1, 1, "3ms should land in ≤5ms bucket");
        // other buckets should be zero
        assert_eq!(buckets[0].1, 0);
        assert_eq!(buckets[2].1, 0);
    }

    // ── MetricsSnapshot ───────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_returns_all_fields() {
        let m = RuntimeMetrics::new();
        m.active_sessions.store(1, Ordering::Relaxed);
        m.total_sessions.store(2, Ordering::Relaxed);
        m.total_steps.store(3, Ordering::Relaxed);
        m.backpressure_shed_count.store(6, Ordering::Relaxed);
        m.memory_recall_count.store(7, Ordering::Relaxed);
        // Use record_* methods so global and per-tool counters stay consistent.
        m.record_tool_call("my_tool");
        m.record_tool_call("my_tool");
        m.record_tool_failure("my_tool");

        let snap = m.snapshot();
        assert_eq!(snap.active_sessions, 1);
        assert_eq!(snap.total_sessions, 2);
        assert_eq!(snap.total_steps, 3);
        assert_eq!(snap.total_tool_calls, 2);
        assert_eq!(snap.failed_tool_calls, 1);
        assert_eq!(snap.backpressure_shed_count, 6);
        assert_eq!(snap.memory_recall_count, 7);
        assert_eq!(snap.per_tool_calls.get("my_tool").copied(), Some(2));
        assert_eq!(snap.per_tool_failures.get("my_tool").copied(), Some(1));
    }

    #[test]
    fn test_snapshot_default_is_zeroed() {
        let snap = MetricsSnapshot::default();
        assert_eq!(snap.active_sessions, 0);
        assert_eq!(snap.total_sessions, 0);
        assert_eq!(snap.total_steps, 0);
        assert!(snap.per_tool_calls.is_empty());
        assert!(snap.per_tool_failures.is_empty());
    }

    // ── #8 MetricsSnapshot histogram fields ───────────────────────────────────

    #[test]
    fn test_metrics_snapshot_contains_all_fields() {
        let m = RuntimeMetrics::new();
        m.record_step_latency(5);
        m.record_step_latency(200);
        let snap = m.snapshot();
        // Should have 7 buckets
        assert_eq!(snap.step_latency_buckets.len(), 7);
        assert!(snap.step_latency_mean_ms > 0.0);
    }

    // ── #9 per-agent tool call tracking ──────────────────────────────────────

    #[test]
    fn test_per_agent_tool_call_tracking() {
        let m = RuntimeMetrics::new();
        m.record_agent_tool_call("agent-1", "search");
        m.record_agent_tool_call("agent-1", "search");
        m.record_agent_tool_call("agent-2", "lookup");
        m.record_agent_tool_failure("agent-1", "search");

        let calls = m.per_agent_tool_calls_snapshot();
        assert_eq!(calls.get("agent-1").and_then(|t| t.get("search")).copied(), Some(2));
        assert_eq!(calls.get("agent-2").and_then(|t| t.get("lookup")).copied(), Some(1));

        let failures = m.per_agent_tool_failures_snapshot();
        assert_eq!(failures.get("agent-1").and_then(|t| t.get("search")).copied(), Some(1));

        // Also check snapshot includes them
        let snap = m.snapshot();
        assert_eq!(snap.per_agent_tool_calls.get("agent-1").and_then(|t| t.get("search")).copied(), Some(2));

        // Reset clears them
        m.reset();
        assert!(m.per_agent_tool_calls_snapshot().is_empty());
        assert!(m.per_agent_tool_failures_snapshot().is_empty());
    }

    // ── New API tests (Rounds 4-8) ────────────────────────────────────────────

    #[test]
    fn test_latency_histogram_min_max_ms() {
        let h = LatencyHistogram::default();
        assert!(h.min_ms().is_none());
        assert!(h.max_ms().is_none());

        h.record(3);  // bucket 1 (≤5ms)
        h.record(200); // bucket 5 (≤500ms)
        assert!(h.min_ms().is_some());
        assert!(h.max_ms().is_some());
        assert!(h.min_ms().unwrap() <= h.max_ms().unwrap());
    }

    #[test]
    fn test_latency_histogram_p50_p95_p99() {
        let h = LatencyHistogram::default();
        for _ in 0..100 {
            h.record(5); // all in ≤5ms bucket
        }
        // p50, p95, p99 should all resolve to the same bucket bound
        let p50 = h.p50();
        let p95 = h.p95();
        let p99 = h.p99();
        assert_eq!(p50, p95);
        assert_eq!(p95, p99);
    }

    #[test]
    fn test_metrics_snapshot_delta_reflects_increments() {
        let m = RuntimeMetrics::new();
        let before = m.snapshot();
        m.total_steps.fetch_add(5, std::sync::atomic::Ordering::Relaxed);
        m.total_tool_calls.fetch_add(3, std::sync::atomic::Ordering::Relaxed);
        let after = m.snapshot();
        let delta = MetricsSnapshot::delta(&after, &before);
        assert_eq!(delta.total_steps, 5);
        assert_eq!(delta.total_tool_calls, 3);
    }

    #[test]
    fn test_metrics_snapshot_display_contains_key_fields() {
        let m = RuntimeMetrics::new();
        let snap = m.snapshot();
        let s = snap.to_string();
        assert!(s.contains("sessions"));
        assert!(s.contains("steps"));
        assert!(s.contains("latency_mean"));
    }
}
