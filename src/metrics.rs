//! # Module: Metrics
//!
//! Runtime observability counters for `AgentRuntime`.
//! All global counters use atomics for lock-free, thread-safe increment/read.
//! Per-tool counters use a `Mutex<HashMap>` to avoid requiring a concurrent
//! map dependency.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ── LatencyHistogram ──────────────────────────────────────────────────────────

/// A simple fixed-bucket latency histogram.
///
/// Buckets (upper bound, ms): 1, 5, 10, 50, 100, 500, ∞
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Counts per bucket. Index 0 = ≤1ms, …, index 6 = >500ms.
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

    /// Return bucket counts as `(upper_bound_ms, count)` pairs.
    pub fn buckets(&self) -> Vec<(u64, u64)> {
        Self::BOUNDS
            .iter()
            .zip(self.buckets.iter())
            .map(|(&b, a)| (b, a.load(Ordering::Relaxed)))
            .collect()
    }
}

/// A point-in-time snapshot of all runtime counters.
///
/// Obtained by calling [`RuntimeMetrics::snapshot`].  All fields are plain
/// integers so the snapshot can be logged, serialised, or diffed without
/// holding any locks.
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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
    /// Per-tool call counts: `tool_name → total_calls`.
    pub per_tool_calls: HashMap<String, u64>,
    /// Per-tool failure counts: `tool_name → failed_calls`.
    pub per_tool_failures: HashMap<String, u64>,
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
    /// Per-tool call counts: `tool_name → total_calls`.
    per_tool_calls: Mutex<HashMap<String, u64>>,
    /// Per-tool failure counts: `tool_name → failed_calls`.
    per_tool_failures: Mutex<HashMap<String, u64>>,
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
            per_tool_calls: Mutex::new(HashMap::new()),
            per_tool_failures: Mutex::new(HashMap::new()),
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

    /// Increment the call counter for `tool_name` by 1.
    ///
    /// Called automatically by the agent loop when `with_metrics` is configured.
    pub fn record_tool_call(&self, tool_name: &str) {
        self.total_tool_calls.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut map) = self.per_tool_calls.lock() {
            *map.entry(tool_name.to_owned()).or_insert(0) += 1;
        }
    }

    /// Increment the failure counter for `tool_name` by 1.
    ///
    /// Called automatically by the agent loop when a tool returns an error.
    pub fn record_tool_failure(&self, tool_name: &str) {
        self.failed_tool_calls.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut map) = self.per_tool_failures.lock() {
            *map.entry(tool_name.to_owned()).or_insert(0) += 1;
        }
    }

    /// Return a snapshot of per-tool call counts as a `HashMap<tool_name, count>`.
    pub fn per_tool_calls_snapshot(&self) -> HashMap<String, u64> {
        self.per_tool_calls
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default()
    }

    /// Return a snapshot of per-tool failure counts as a `HashMap<tool_name, count>`.
    pub fn per_tool_failures_snapshot(&self) -> HashMap<String, u64> {
        self.per_tool_failures
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default()
    }

    /// Capture a complete snapshot of all counters, including per-tool breakdowns.
    ///
    /// This is the preferred alternative to [`to_snapshot`] — it returns a
    /// named [`MetricsSnapshot`] struct instead of an opaque tuple.
    ///
    /// [`to_snapshot`]: RuntimeMetrics::to_snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            active_sessions: self.active_sessions.load(Ordering::Relaxed),
            total_sessions: self.total_sessions.load(Ordering::Relaxed),
            total_steps: self.total_steps.load(Ordering::Relaxed),
            total_tool_calls: self.total_tool_calls.load(Ordering::Relaxed),
            failed_tool_calls: self.failed_tool_calls.load(Ordering::Relaxed),
            backpressure_shed_count: self.backpressure_shed_count.load(Ordering::Relaxed),
            memory_recall_count: self.memory_recall_count.load(Ordering::Relaxed),
            per_tool_calls: self.per_tool_calls_snapshot(),
            per_tool_failures: self.per_tool_failures_snapshot(),
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
        if let Ok(mut m) = self.per_tool_calls.lock() {
            m.clear();
        }
        if let Ok(mut m) = self.per_tool_failures.lock() {
            m.clear();
        }
        self.step_latency.total_count.store(0, Ordering::Relaxed);
        self.step_latency.total_sum_ms.store(0, Ordering::Relaxed);
        for b in &self.step_latency.buckets {
            b.store(0, Ordering::Relaxed);
        }
    }

    /// Capture a snapshot of global counters as plain integers.
    ///
    /// Returns `(active_sessions, total_sessions, total_steps,
    ///           total_tool_calls, failed_tool_calls,
    ///           backpressure_shed_count, memory_recall_count)`.
    /// For per-tool breakdowns use [`per_tool_calls_snapshot`] and
    /// [`per_tool_failures_snapshot`].
    ///
    /// [`per_tool_calls_snapshot`]: RuntimeMetrics::per_tool_calls_snapshot
    /// [`per_tool_failures_snapshot`]: RuntimeMetrics::per_tool_failures_snapshot
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
}
