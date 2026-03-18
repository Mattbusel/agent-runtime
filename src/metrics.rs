//! # Module: Metrics
//!
//! Runtime observability counters for `AgentRuntime`.
//! All counters use atomics for lock-free, thread-safe increment/read.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Shared runtime metrics. Clone the `Arc` to share across threads.
#[derive(Debug, Default)]
pub struct RuntimeMetrics {
    /// Number of agent sessions currently in progress.
    pub active_sessions: AtomicUsize,
    /// Total number of sessions started since the runtime was created.
    pub total_sessions: AtomicU64,
    /// Total number of ReAct steps executed across all sessions.
    pub total_steps: AtomicU64,
    /// Total number of tool calls dispatched.
    pub total_tool_calls: AtomicU64,
    /// Total number of tool calls that returned an error observation.
    pub failed_tool_calls: AtomicU64,
    /// Total number of requests shed due to backpressure.
    pub backpressure_shed_count: AtomicU64,
    /// Total number of memory recall operations.
    pub memory_recall_count: AtomicU64,
}

impl RuntimeMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn active_sessions(&self) -> usize {
        self.active_sessions.load(Ordering::Relaxed)
    }

    pub fn total_sessions(&self) -> u64 {
        self.total_sessions.load(Ordering::Relaxed)
    }

    pub fn total_steps(&self) -> u64 {
        self.total_steps.load(Ordering::Relaxed)
    }

    pub fn total_tool_calls(&self) -> u64 {
        self.total_tool_calls.load(Ordering::Relaxed)
    }

    pub fn failed_tool_calls(&self) -> u64 {
        self.failed_tool_calls.load(Ordering::Relaxed)
    }

    pub fn backpressure_shed_count(&self) -> u64 {
        self.backpressure_shed_count.load(Ordering::Relaxed)
    }

    pub fn memory_recall_count(&self) -> u64 {
        self.memory_recall_count.load(Ordering::Relaxed)
    }
}
