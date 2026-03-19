//! # Module: Metrics
//!
//! Runtime observability counters for `AgentRuntime`.
//! All global counters use atomics for lock-free, thread-safe increment/read.
//! Per-tool counters use a `Mutex<HashMap>` to avoid requiring a concurrent
//! map dependency.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
}
