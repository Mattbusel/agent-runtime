//! # Module: resource_manager
//!
//! ## Responsibility
//! Resource budget management for agent tasks. Enforces per-task limits on
//! token consumption, tool call count, wall-clock time, and memory usage.
//! Tracks cumulative usage across multiple calls and exposes Prometheus-style
//! in-memory atomic counters.
//!
//! ## Key types
//! - [`ResourceBudget`] — configures limits for a task or session.
//! - [`ResourceUsage`] — a snapshot of resource consumption.
//! - [`ResourceManager`] — stateless budget checker.
//! - [`ResourceError`] — typed error returned when a limit is exceeded.
//! - [`BudgetEnforcer`] — stateful wrapper that accumulates usage across calls.
//! - [`ResourceMetrics`] — atomic Prometheus-style counters.
//!
//! ## Guarantees
//! - No panics in production paths.
//! - Thread-safe: `BudgetEnforcer` and `ResourceMetrics` use atomics / `Arc<Mutex<_>>`.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

// ── ResourceBudget ────────────────────────────────────────────────────────────

/// Configures resource limits for an agent task or session.
///
/// `0` means "no limit" for all numeric fields.
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    /// Maximum number of tokens that may be consumed (`0` = unlimited).
    pub max_tokens: u64,
    /// Maximum number of tool calls that may be made (`0` = unlimited).
    pub max_tool_calls: u32,
    /// Maximum allowed wall-clock time in milliseconds (`0` = unlimited).
    pub max_wall_time_ms: u64,
    /// Maximum allowed memory footprint in megabytes (`0` = unlimited).
    pub max_memory_mb: u64,
}

impl ResourceBudget {
    /// Create a budget with all limits set to zero (unlimited).
    pub fn unlimited() -> Self {
        Self {
            max_tokens: 0,
            max_tool_calls: 0,
            max_wall_time_ms: 0,
            max_memory_mb: 0,
        }
    }

    /// Set a token limit.
    pub fn with_max_tokens(mut self, limit: u64) -> Self {
        self.max_tokens = limit;
        self
    }

    /// Set a tool-call limit.
    pub fn with_max_tool_calls(mut self, limit: u32) -> Self {
        self.max_tool_calls = limit;
        self
    }

    /// Set a wall-time limit in milliseconds.
    pub fn with_max_wall_time_ms(mut self, limit: u64) -> Self {
        self.max_wall_time_ms = limit;
        self
    }

    /// Set a memory limit in megabytes.
    pub fn with_max_memory_mb(mut self, limit: u64) -> Self {
        self.max_memory_mb = limit;
        self
    }
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

// ── ResourceUsage ─────────────────────────────────────────────────────────────

/// A snapshot of resource consumption for a task or session.
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// Number of tokens consumed so far.
    pub tokens_used: u64,
    /// Number of tool calls made so far.
    pub tool_calls: u32,
    /// Wall-clock time elapsed in milliseconds.
    pub elapsed_ms: u64,
    /// Current memory footprint in megabytes.
    pub memory_mb: u64,
}

impl ResourceUsage {
    /// Create a zeroed-out usage record.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Accumulate another usage snapshot into this one (saturating arithmetic).
    pub fn accumulate(&mut self, other: &ResourceUsage) {
        self.tokens_used = self.tokens_used.saturating_add(other.tokens_used);
        self.tool_calls = self.tool_calls.saturating_add(other.tool_calls);
        self.elapsed_ms = self.elapsed_ms.saturating_add(other.elapsed_ms);
        // Memory is a peak measure — take the maximum.
        if other.memory_mb > self.memory_mb {
            self.memory_mb = other.memory_mb;
        }
    }
}

// ── ResourceError ─────────────────────────────────────────────────────────────

/// Error returned by [`ResourceManager::check_budget`] when a limit is exceeded.
#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone)]
pub enum ResourceError {
    /// Token limit exceeded.
    #[error("Token budget exceeded: used {used}, limit {limit}")]
    TokensExceeded {
        /// Tokens consumed at the time of the check.
        used: u64,
        /// Configured token limit.
        limit: u64,
    },
    /// Tool-call limit exceeded.
    #[error("Tool-call budget exceeded: used {used}, limit {limit}")]
    ToolCallsExceeded {
        /// Tool calls made at the time of the check.
        used: u32,
        /// Configured tool-call limit.
        limit: u32,
    },
    /// Wall-clock time limit exceeded.
    #[error("Wall-time budget exceeded: elapsed {elapsed_ms} ms, limit {limit_ms} ms")]
    TimeExceeded {
        /// Elapsed milliseconds at the time of the check.
        elapsed_ms: u64,
        /// Configured wall-time limit in milliseconds.
        limit_ms: u64,
    },
    /// Memory limit exceeded.
    #[error("Memory budget exceeded: used {used_mb} MB, limit {limit_mb} MB")]
    MemoryExceeded {
        /// Memory in megabytes at the time of the check.
        used_mb: u64,
        /// Configured memory limit in megabytes.
        limit_mb: u64,
    },
}

// ── ResourceManager ───────────────────────────────────────────────────────────

/// Stateless budget checker.
///
/// Call [`check_budget`] to verify that a usage snapshot does not exceed the
/// configured limits.  The first violated limit is returned; checks are applied
/// in declaration order: tokens → tool_calls → time → memory.
///
/// [`check_budget`]: ResourceManager::check_budget
#[derive(Debug, Clone, Default)]
pub struct ResourceManager;

impl ResourceManager {
    /// Create a new `ResourceManager` (stateless; no config required).
    pub fn new() -> Self {
        Self
    }

    /// Check whether `usage` is within `budget`.
    ///
    /// Returns `Ok(())` when all limits are satisfied, or the first
    /// [`ResourceError`] that is violated.
    ///
    /// A limit value of `0` means "unlimited" and is never violated.
    pub fn check_budget(
        &self,
        usage: &ResourceUsage,
        budget: &ResourceBudget,
    ) -> Result<(), ResourceError> {
        if budget.max_tokens > 0 && usage.tokens_used > budget.max_tokens {
            return Err(ResourceError::TokensExceeded {
                used: usage.tokens_used,
                limit: budget.max_tokens,
            });
        }
        if budget.max_tool_calls > 0 && usage.tool_calls > budget.max_tool_calls {
            return Err(ResourceError::ToolCallsExceeded {
                used: usage.tool_calls,
                limit: budget.max_tool_calls,
            });
        }
        if budget.max_wall_time_ms > 0 && usage.elapsed_ms > budget.max_wall_time_ms {
            return Err(ResourceError::TimeExceeded {
                elapsed_ms: usage.elapsed_ms,
                limit_ms: budget.max_wall_time_ms,
            });
        }
        if budget.max_memory_mb > 0 && usage.memory_mb > budget.max_memory_mb {
            return Err(ResourceError::MemoryExceeded {
                used_mb: usage.memory_mb,
                limit_mb: budget.max_memory_mb,
            });
        }
        Ok(())
    }
}

// ── BudgetEnforcer ────────────────────────────────────────────────────────────

/// Stateful budget enforcer that accumulates usage across multiple calls.
///
/// Wraps a [`ResourceBudget`] and a [`ResourceMetrics`] instance.  Each call
/// to [`record_and_check`] adds the new usage to the running total and then
/// checks the combined total against the budget.
///
/// [`record_and_check`]: BudgetEnforcer::record_and_check
#[derive(Debug, Clone)]
pub struct BudgetEnforcer {
    budget: ResourceBudget,
    manager: ResourceManager,
    cumulative: Arc<Mutex<ResourceUsage>>,
    /// Prometheus-style atomic metric counters.
    pub metrics: ResourceMetrics,
}

impl BudgetEnforcer {
    /// Create a new enforcer for the given `budget`.
    pub fn new(budget: ResourceBudget) -> Self {
        Self {
            budget,
            manager: ResourceManager::new(),
            cumulative: Arc::new(Mutex::new(ResourceUsage::zero())),
            metrics: ResourceMetrics::new(),
        }
    }

    /// Accumulate `delta` into the running total and check against the budget.
    ///
    /// # Errors
    /// Returns a [`ResourceError`] if the cumulative usage exceeds any limit.
    /// The `delta` is still recorded (callers can inspect cumulative usage
    /// even after a budget breach).
    pub fn record_and_check(&self, delta: &ResourceUsage) -> Result<(), ResourceError> {
        // Update atomics.
        self.metrics
            .tokens_used
            .fetch_add(delta.tokens_used, Ordering::Relaxed);
        self.metrics
            .tool_calls
            .fetch_add(u64::from(delta.tool_calls), Ordering::Relaxed);
        self.metrics
            .elapsed_ms
            .fetch_add(delta.elapsed_ms, Ordering::Relaxed);
        // memory_mb: track peak
        let prev_mem = self.metrics.memory_mb.load(Ordering::Relaxed);
        if delta.memory_mb > prev_mem {
            self.metrics
                .memory_mb
                .store(delta.memory_mb, Ordering::Relaxed);
        }

        // Update cumulative.
        let cumulative = {
            let mut guard = self
                .cumulative
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            guard.accumulate(delta);
            guard.clone()
        };

        // Check.
        let result = self.manager.check_budget(&cumulative, &self.budget);
        if result.is_err() {
            self.metrics.budget_breaches.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Return a snapshot of the current cumulative usage.
    pub fn cumulative_usage(&self) -> ResourceUsage {
        self.cumulative
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Reset cumulative usage and metric counters to zero.
    pub fn reset(&self) {
        if let Ok(mut guard) = self.cumulative.lock() {
            *guard = ResourceUsage::zero();
        }
        self.metrics.reset();
    }

    /// Return a reference to the configured budget.
    pub fn budget(&self) -> &ResourceBudget {
        &self.budget
    }
}

// ── ResourceMetrics ───────────────────────────────────────────────────────────

/// Prometheus-style in-memory atomic counters for resource usage.
///
/// All counters use `Relaxed` ordering — they are for observability only and
/// do not participate in synchronisation logic.
#[derive(Debug)]
pub struct ResourceMetrics {
    /// Total tokens consumed since last reset.
    pub tokens_used: AtomicU64,
    /// Total tool calls made since last reset.
    pub tool_calls: AtomicU64,
    /// Total wall-clock milliseconds elapsed since last reset.
    pub elapsed_ms: AtomicU64,
    /// Peak memory usage in megabytes since last reset.
    pub memory_mb: AtomicU64,
    /// Number of times `record_and_check` returned a budget error.
    pub budget_breaches: AtomicU64,
}

impl Clone for ResourceMetrics {
    fn clone(&self) -> Self {
        Self {
            tokens_used: AtomicU64::new(self.tokens_used.load(Ordering::Relaxed)),
            tool_calls: AtomicU64::new(self.tool_calls.load(Ordering::Relaxed)),
            elapsed_ms: AtomicU64::new(self.elapsed_ms.load(Ordering::Relaxed)),
            memory_mb: AtomicU64::new(self.memory_mb.load(Ordering::Relaxed)),
            budget_breaches: AtomicU64::new(self.budget_breaches.load(Ordering::Relaxed)),
        }
    }
}

impl ResourceMetrics {
    /// Create zeroed-out metrics.
    pub fn new() -> Self {
        Self {
            tokens_used: AtomicU64::new(0),
            tool_calls: AtomicU64::new(0),
            elapsed_ms: AtomicU64::new(0),
            memory_mb: AtomicU64::new(0),
            budget_breaches: AtomicU64::new(0),
        }
    }

    /// Return a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> ResourceMetricsSnapshot {
        ResourceMetricsSnapshot {
            tokens_used: self.tokens_used.load(Ordering::Relaxed),
            tool_calls: self.tool_calls.load(Ordering::Relaxed),
            elapsed_ms: self.elapsed_ms.load(Ordering::Relaxed),
            memory_mb: self.memory_mb.load(Ordering::Relaxed),
            budget_breaches: self.budget_breaches.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.tokens_used.store(0, Ordering::Relaxed);
        self.tool_calls.store(0, Ordering::Relaxed);
        self.elapsed_ms.store(0, Ordering::Relaxed);
        self.memory_mb.store(0, Ordering::Relaxed);
        self.budget_breaches.store(0, Ordering::Relaxed);
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// An immutable snapshot of [`ResourceMetrics`] values.
#[derive(Debug, Clone, Default)]
pub struct ResourceMetricsSnapshot {
    /// Total tokens consumed since last reset.
    pub tokens_used: u64,
    /// Total tool calls made since last reset.
    pub tool_calls: u64,
    /// Total wall-clock milliseconds elapsed since last reset.
    pub elapsed_ms: u64,
    /// Peak memory usage in megabytes since last reset.
    pub memory_mb: u64,
    /// Number of budget breaches recorded.
    pub budget_breaches: u64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn usage(tokens: u64, calls: u32, elapsed_ms: u64, memory_mb: u64) -> ResourceUsage {
        ResourceUsage {
            tokens_used: tokens,
            tool_calls: calls,
            elapsed_ms,
            memory_mb,
        }
    }

    // ── ResourceManager ───────────────────────────────────────────────────────

    #[test]
    fn check_budget_ok_when_unlimited() {
        let mgr = ResourceManager::new();
        let budget = ResourceBudget::unlimited();
        let u = usage(u64::MAX, u32::MAX, u64::MAX, u64::MAX);
        assert!(mgr.check_budget(&u, &budget).is_ok());
    }

    #[test]
    fn check_budget_ok_within_limits() {
        let mgr = ResourceManager::new();
        let budget = ResourceBudget::unlimited()
            .with_max_tokens(100)
            .with_max_tool_calls(5)
            .with_max_wall_time_ms(1_000)
            .with_max_memory_mb(512);
        assert!(mgr.check_budget(&usage(50, 3, 500, 256), &budget).is_ok());
    }

    #[test]
    fn check_budget_tokens_exceeded() {
        let mgr = ResourceManager::new();
        let budget = ResourceBudget::unlimited().with_max_tokens(100);
        let result = mgr.check_budget(&usage(101, 0, 0, 0), &budget);
        assert!(matches!(result, Err(ResourceError::TokensExceeded { .. })));
    }

    #[test]
    fn check_budget_tool_calls_exceeded() {
        let mgr = ResourceManager::new();
        let budget = ResourceBudget::unlimited().with_max_tool_calls(3);
        let result = mgr.check_budget(&usage(0, 4, 0, 0), &budget);
        assert!(matches!(result, Err(ResourceError::ToolCallsExceeded { .. })));
    }

    #[test]
    fn check_budget_time_exceeded() {
        let mgr = ResourceManager::new();
        let budget = ResourceBudget::unlimited().with_max_wall_time_ms(1_000);
        let result = mgr.check_budget(&usage(0, 0, 1_001, 0), &budget);
        assert!(matches!(result, Err(ResourceError::TimeExceeded { .. })));
    }

    #[test]
    fn check_budget_memory_exceeded() {
        let mgr = ResourceManager::new();
        let budget = ResourceBudget::unlimited().with_max_memory_mb(512);
        let result = mgr.check_budget(&usage(0, 0, 0, 513), &budget);
        assert!(matches!(result, Err(ResourceError::MemoryExceeded { .. })));
    }

    #[test]
    fn check_budget_exact_limit_is_ok() {
        let mgr = ResourceManager::new();
        let budget = ResourceBudget::unlimited().with_max_tokens(100);
        assert!(mgr.check_budget(&usage(100, 0, 0, 0), &budget).is_ok());
    }

    #[test]
    fn error_display_tokens() {
        let e = ResourceError::TokensExceeded { used: 200, limit: 100 };
        assert!(e.to_string().contains("200"));
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn error_display_time() {
        let e = ResourceError::TimeExceeded { elapsed_ms: 2000, limit_ms: 1000 };
        assert!(e.to_string().contains("2000"));
    }

    // ── BudgetEnforcer ────────────────────────────────────────────────────────

    #[test]
    fn enforcer_accumulates_usage() {
        let budget = ResourceBudget::unlimited().with_max_tokens(1_000);
        let enforcer = BudgetEnforcer::new(budget);
        enforcer.record_and_check(&usage(300, 1, 100, 10)).unwrap();
        enforcer.record_and_check(&usage(300, 1, 100, 10)).unwrap();
        let cumulative = enforcer.cumulative_usage();
        assert_eq!(cumulative.tokens_used, 600);
        assert_eq!(cumulative.tool_calls, 2);
    }

    #[test]
    fn enforcer_returns_error_on_breach() {
        let budget = ResourceBudget::unlimited().with_max_tokens(100);
        let enforcer = BudgetEnforcer::new(budget);
        enforcer.record_and_check(&usage(60, 0, 0, 0)).unwrap();
        let result = enforcer.record_and_check(&usage(60, 0, 0, 0));
        assert!(matches!(result, Err(ResourceError::TokensExceeded { .. })));
    }

    #[test]
    fn enforcer_metrics_incremented_on_record() {
        let enforcer = BudgetEnforcer::new(ResourceBudget::unlimited());
        enforcer.record_and_check(&usage(50, 2, 200, 64)).unwrap();
        let snap = enforcer.metrics.snapshot();
        assert_eq!(snap.tokens_used, 50);
        assert_eq!(snap.tool_calls, 2);
        assert_eq!(snap.elapsed_ms, 200);
        assert_eq!(snap.memory_mb, 64);
    }

    #[test]
    fn enforcer_breach_counter_incremented() {
        let budget = ResourceBudget::unlimited().with_max_tokens(10);
        let enforcer = BudgetEnforcer::new(budget);
        let _ = enforcer.record_and_check(&usage(20, 0, 0, 0));
        let snap = enforcer.metrics.snapshot();
        assert_eq!(snap.budget_breaches, 1);
    }

    #[test]
    fn enforcer_reset_clears_state() {
        let budget = ResourceBudget::unlimited().with_max_tokens(1_000);
        let enforcer = BudgetEnforcer::new(budget);
        enforcer.record_and_check(&usage(500, 3, 1_000, 128)).unwrap();
        enforcer.reset();
        let cumulative = enforcer.cumulative_usage();
        assert_eq!(cumulative.tokens_used, 0);
        let snap = enforcer.metrics.snapshot();
        assert_eq!(snap.tokens_used, 0);
        assert_eq!(snap.budget_breaches, 0);
    }

    #[test]
    fn enforcer_memory_tracks_peak() {
        let enforcer = BudgetEnforcer::new(ResourceBudget::unlimited());
        enforcer.record_and_check(&usage(0, 0, 0, 256)).unwrap();
        enforcer.record_and_check(&usage(0, 0, 0, 128)).unwrap();
        // cumulative memory should be peak = 256
        let cumulative = enforcer.cumulative_usage();
        assert_eq!(cumulative.memory_mb, 256);
    }

    // ── ResourceUsage::accumulate ─────────────────────────────────────────────

    #[test]
    fn usage_accumulate_adds_tokens_and_calls() {
        let mut u = ResourceUsage::zero();
        u.accumulate(&usage(100, 2, 500, 64));
        u.accumulate(&usage(200, 3, 300, 32));
        assert_eq!(u.tokens_used, 300);
        assert_eq!(u.tool_calls, 5);
        assert_eq!(u.elapsed_ms, 800);
        assert_eq!(u.memory_mb, 64); // peak
    }
}
