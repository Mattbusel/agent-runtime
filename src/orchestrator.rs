//! # Module: Orchestrator
//!
//! ## Responsibility
//! Provides a composable LLM pipeline with circuit breaking, retry, deduplication,
//! and backpressure. Mirrors the public API of `tokio-prompt-orchestrator`.
//!
//! ## Guarantees
//! - Thread-safe: all types wrap state in `Arc<Mutex<_>>` or atomics
//! - Circuit breaker opens after `threshold` failures within `window` calls
//! - RetryPolicy delays grow exponentially and are capped at `MAX_RETRY_DELAY`
//! - Deduplicator is deterministic and non-blocking
//! - BackpressureGuard never exceeds declared capacity
//! - Non-panicking: all operations return `Result`
//!
//! ## NOT Responsible For
//! - Cross-node circuit breakers (single-process only)
//! - Persistent deduplication (in-memory, bounded TTL)
//! - Distributed backpressure

use crate::error::AgentRuntimeError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Maximum delay between retries — caps exponential growth.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

// ── RetryPolicy ───────────────────────────────────────────────────────────────

/// Exponential backoff retry policy.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Base delay for the first retry.
    pub base_delay: Duration,
}

impl RetryPolicy {
    /// Create an exponential retry policy.
    ///
    /// # Arguments
    /// * `max_attempts` — total attempt budget (must be ≥ 1)
    /// * `base_ms` — base delay in milliseconds for attempt 1
    ///
    /// # Returns
    /// - `Ok(RetryPolicy)` — on success
    /// - `Err(AgentRuntimeError::Orchestration)` — if `max_attempts == 0`
    pub fn exponential(max_attempts: u32, base_ms: u64) -> Result<Self, AgentRuntimeError> {
        if max_attempts == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "max_attempts must be >= 1".into(),
            ));
        }
        Ok(Self {
            max_attempts,
            base_delay: Duration::from_millis(base_ms),
        })
    }

    /// Compute the delay before the given attempt number (1-based).
    ///
    /// Delay = `base_delay * 2^(attempt-1)`, capped at `MAX_RETRY_DELAY`.
    pub fn delay_for(&self, attempt: u32) -> Duration {
        let exp = attempt.saturating_sub(1);
        let multiplier = 1u64.checked_shl(exp.min(63) as u32).unwrap_or(u64::MAX);
        let millis = self
            .base_delay
            .as_millis()
            .saturating_mul(multiplier as u128);
        let raw = Duration::from_millis(millis.min(u64::MAX as u128) as u64);
        raw.min(MAX_RETRY_DELAY)
    }
}

// ── CircuitBreaker ────────────────────────────────────────────────────────────

/// Tracks failure rates and opens when the threshold is exceeded.
///
/// States: `Closed` (normal) → `Open` (fast-fail) → `HalfOpen` (probe).
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,
    Open { opened_at: Instant },
    HalfOpen,
}

/// Circuit breaker guarding a fallible operation.
///
/// ## Guarantees
/// - Opens after `threshold` consecutive failures
/// - Transitions to `HalfOpen` after `recovery_window` has elapsed
/// - Closes on the first successful probe in `HalfOpen`
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    inner: Arc<Mutex<CircuitBreakerInner>>,
}

#[derive(Debug)]
struct CircuitBreakerInner {
    threshold: u32,
    recovery_window: Duration,
    consecutive_failures: u32,
    state: CircuitState,
    service: String,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// # Arguments
    /// * `service` — name used in error messages
    /// * `threshold` — consecutive failures before opening
    /// * `recovery_window` — how long to stay open before probing
    pub fn new(
        service: impl Into<String>,
        threshold: u32,
        recovery_window: Duration,
    ) -> Result<Self, AgentRuntimeError> {
        if threshold == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "circuit breaker threshold must be >= 1".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(CircuitBreakerInner {
                threshold,
                recovery_window,
                consecutive_failures: 0,
                state: CircuitState::Closed,
                service: service.into(),
            })),
        })
    }

    /// Attempt to call `f`, respecting the circuit breaker state.
    ///
    /// # Returns
    /// - `Ok(T)` — if `f` succeeds (resets failure count)
    /// - `Err(AgentRuntimeError::CircuitOpen)` — if the breaker is open
    /// - `Err(...)` — if `f` fails (may open the breaker)
    pub fn call<T, E, F>(&self, f: F) -> Result<T, AgentRuntimeError>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        // Check and potentially transition state
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;

            match &inner.state {
                CircuitState::Open { opened_at } => {
                    if opened_at.elapsed() >= inner.recovery_window {
                        inner.state = CircuitState::HalfOpen;
                    } else {
                        return Err(AgentRuntimeError::CircuitOpen {
                            service: inner.service.clone(),
                        });
                    }
                }
                CircuitState::Closed | CircuitState::HalfOpen => {}
            }
        }

        // Execute the operation
        match f() {
            Ok(val) => {
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;
                inner.consecutive_failures = 0;
                inner.state = CircuitState::Closed;
                Ok(val)
            }
            Err(e) => {
                let mut inner = self.inner.lock().map_err(|e2| {
                    AgentRuntimeError::Orchestration(format!("lock poisoned: {e2}"))
                })?;
                inner.consecutive_failures += 1;
                if inner.consecutive_failures >= inner.threshold {
                    inner.state = CircuitState::Open {
                        opened_at: Instant::now(),
                    };
                }
                Err(AgentRuntimeError::Orchestration(e.to_string()))
            }
        }
    }

    /// Return the current circuit state.
    pub fn state(&self) -> Result<CircuitState, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;
        Ok(inner.state.clone())
    }

    /// Return the consecutive failure count.
    pub fn failure_count(&self) -> Result<u32, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;
        Ok(inner.consecutive_failures)
    }
}

// ── DeduplicationResult ───────────────────────────────────────────────────────

/// Result of a deduplication check.
#[derive(Debug, Clone, PartialEq)]
pub enum DeduplicationResult {
    /// This is a new, unseen request.
    New,
    /// A cached result exists for this key.
    Cached(String),
    /// A matching request is currently in-flight.
    InProgress,
}

/// Deduplicates requests by key within a TTL window.
///
/// ## Guarantees
/// - Deterministic: same key always maps to the same result
/// - Thread-safe via `Arc<Mutex<_>>`
/// - Entries expire after `ttl`
#[derive(Debug, Clone)]
pub struct Deduplicator {
    ttl: Duration,
    inner: Arc<Mutex<DeduplicatorInner>>,
}

#[derive(Debug)]
struct DeduplicatorInner {
    cache: HashMap<String, (String, Instant)>, // key → (result, inserted_at)
    in_flight: HashMap<String, Instant>,       // key → started_at
}

impl Deduplicator {
    /// Create a new deduplicator with the given TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            inner: Arc::new(Mutex::new(DeduplicatorInner {
                cache: HashMap::new(),
                in_flight: HashMap::new(),
            })),
        }
    }

    /// Check whether `key` is new, cached, or in-flight.
    ///
    /// Marks the key as in-flight if it is new.
    pub fn check_and_register(&self, key: &str) -> Result<DeduplicationResult, AgentRuntimeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;

        let now = Instant::now();

        // Expire stale cache entries
        inner
            .cache
            .retain(|_, (_, ts)| now.duration_since(*ts) < self.ttl);
        inner
            .in_flight
            .retain(|_, ts| now.duration_since(*ts) < self.ttl);

        if let Some((result, _)) = inner.cache.get(key) {
            return Ok(DeduplicationResult::Cached(result.clone()));
        }

        if inner.in_flight.contains_key(key) {
            return Ok(DeduplicationResult::InProgress);
        }

        inner.in_flight.insert(key.to_owned(), now);
        Ok(DeduplicationResult::New)
    }

    /// Complete a request: move from in-flight to cached with the given result.
    pub fn complete(&self, key: &str, result: impl Into<String>) -> Result<(), AgentRuntimeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;
        inner.in_flight.remove(key);
        inner
            .cache
            .insert(key.to_owned(), (result.into(), Instant::now()));
        Ok(())
    }
}

// ── BackpressureGuard ─────────────────────────────────────────────────────────

/// Tracks in-flight work count and enforces a capacity limit.
///
/// ## Guarantees
/// - Thread-safe via `Arc<Mutex<_>>`
/// - `try_acquire` is non-blocking
/// - `release` decrements the counter; no-op if counter is already 0
#[derive(Debug, Clone)]
pub struct BackpressureGuard {
    capacity: usize,
    inner: Arc<Mutex<usize>>,
}

impl BackpressureGuard {
    /// Create a new guard with the given capacity.
    ///
    /// # Returns
    /// - `Ok(BackpressureGuard)` — on success
    /// - `Err(AgentRuntimeError::Orchestration)` — if `capacity == 0`
    pub fn new(capacity: usize) -> Result<Self, AgentRuntimeError> {
        if capacity == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "BackpressureGuard capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            capacity,
            inner: Arc::new(Mutex::new(0)),
        })
    }

    /// Try to acquire a slot.
    ///
    /// # Returns
    /// - `Ok(())` — slot acquired
    /// - `Err(AgentRuntimeError::BackpressureShed)` — capacity exceeded
    pub fn try_acquire(&self) -> Result<(), AgentRuntimeError> {
        let mut depth = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;
        if *depth >= self.capacity {
            return Err(AgentRuntimeError::BackpressureShed {
                depth: *depth,
                capacity: self.capacity,
            });
        }
        *depth += 1;
        Ok(())
    }

    /// Release a previously acquired slot.
    pub fn release(&self) -> Result<(), AgentRuntimeError> {
        let mut depth = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;
        *depth = depth.saturating_sub(1);
        Ok(())
    }

    /// Return the current depth.
    pub fn depth(&self) -> Result<usize, AgentRuntimeError> {
        let depth = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Orchestration(format!("lock poisoned: {e}")))?;
        Ok(*depth)
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// A single named stage in the pipeline.
pub struct Stage {
    pub name: String,
    pub handler: Box<dyn Fn(String) -> Result<String, AgentRuntimeError> + Send + Sync>,
}

impl std::fmt::Debug for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage").field("name", &self.name).finish()
    }
}

/// A composable pipeline that passes a string through a sequence of named stages.
///
/// ## Guarantees
/// - Stages execute in insertion order
/// - First stage failure short-circuits remaining stages
/// - Non-panicking
#[derive(Debug)]
pub struct Pipeline {
    stages: Vec<Stage>,
}

impl Pipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage to the pipeline.
    pub fn add_stage(
        mut self,
        name: impl Into<String>,
        handler: impl Fn(String) -> Result<String, AgentRuntimeError> + Send + Sync + 'static,
    ) -> Self {
        self.stages.push(Stage {
            name: name.into(),
            handler: Box::new(handler),
        });
        self
    }

    /// Execute the pipeline, passing `input` through each stage in order.
    pub fn run(&self, input: String) -> Result<String, AgentRuntimeError> {
        let mut current = input;
        for stage in &self.stages {
            current = (stage.handler)(current)?;
        }
        Ok(current)
    }

    /// Return the number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RetryPolicy ───────────────────────────────────────────────────────────

    #[test]
    fn test_retry_policy_rejects_zero_attempts() {
        assert!(RetryPolicy::exponential(0, 100).is_err());
    }

    #[test]
    fn test_retry_policy_delay_attempt_1_equals_base() {
        let p = RetryPolicy::exponential(3, 100).unwrap();
        assert_eq!(p.delay_for(1), Duration::from_millis(100));
    }

    #[test]
    fn test_retry_policy_delay_doubles_each_attempt() {
        let p = RetryPolicy::exponential(5, 100).unwrap();
        assert_eq!(p.delay_for(2), Duration::from_millis(200));
        assert_eq!(p.delay_for(3), Duration::from_millis(400));
        assert_eq!(p.delay_for(4), Duration::from_millis(800));
    }

    #[test]
    fn test_retry_policy_delay_capped_at_max() {
        let p = RetryPolicy::exponential(10, 10_000).unwrap();
        assert_eq!(p.delay_for(10), MAX_RETRY_DELAY);
    }

    #[test]
    fn test_retry_policy_delay_never_exceeds_max_for_any_attempt() {
        let p = RetryPolicy::exponential(10, 1000).unwrap();
        for attempt in 1..=10 {
            assert!(p.delay_for(attempt) <= MAX_RETRY_DELAY);
        }
    }

    // ── CircuitBreaker ────────────────────────────────────────────────────────

    #[test]
    fn test_circuit_breaker_rejects_zero_threshold() {
        assert!(CircuitBreaker::new("svc", 0, Duration::from_secs(1)).is_err());
    }

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_success_keeps_closed() {
        let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
        let result: Result<i32, AgentRuntimeError> = cb.call(|| Ok::<i32, AgentRuntimeError>(42));
        assert!(result.is_ok());
        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold_failures() {
        let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
        for _ in 0..3 {
            let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("oops".to_string()));
        }
        assert!(matches!(cb.state().unwrap(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_circuit_breaker_open_fast_fails() {
        let cb = CircuitBreaker::new("svc", 1, Duration::from_secs(3600)).unwrap();
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
        let result: Result<(), AgentRuntimeError> = cb.call(|| Ok::<(), AgentRuntimeError>(()));
        assert!(matches!(result, Err(AgentRuntimeError::CircuitOpen { .. })));
    }

    #[test]
    fn test_circuit_breaker_success_resets_failure_count() {
        let cb = CircuitBreaker::new("svc", 5, Duration::from_secs(60)).unwrap();
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
        let _: Result<i32, AgentRuntimeError> = cb.call(|| Ok::<i32, AgentRuntimeError>(1));
        assert_eq!(cb.failure_count().unwrap(), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_on_recovery() {
        // Use a zero recovery window to immediately go half-open
        let cb = CircuitBreaker::new("svc", 1, Duration::ZERO).unwrap();
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
        // After recovery window, next call should probe (half-open → closed on success)
        let result: Result<i32, AgentRuntimeError> = cb.call(|| Ok::<i32, AgentRuntimeError>(99));
        assert_eq!(result.unwrap_or(0), 99);
        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    }

    // ── Deduplicator ──────────────────────────────────────────────────────────

    #[test]
    fn test_deduplicator_new_key_is_new() {
        let d = Deduplicator::new(Duration::from_secs(60));
        let r = d.check_and_register("key-1").unwrap();
        assert_eq!(r, DeduplicationResult::New);
    }

    #[test]
    fn test_deduplicator_second_check_is_in_progress() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("key-1").unwrap();
        let r = d.check_and_register("key-1").unwrap();
        assert_eq!(r, DeduplicationResult::InProgress);
    }

    #[test]
    fn test_deduplicator_complete_makes_cached() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("key-1").unwrap();
        d.complete("key-1", "result-value").unwrap();
        let r = d.check_and_register("key-1").unwrap();
        assert_eq!(r, DeduplicationResult::Cached("result-value".into()));
    }

    #[test]
    fn test_deduplicator_different_keys_are_independent() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("key-a").unwrap();
        let r = d.check_and_register("key-b").unwrap();
        assert_eq!(r, DeduplicationResult::New);
    }

    #[test]
    fn test_deduplicator_expired_entry_is_new() {
        let d = Deduplicator::new(Duration::ZERO); // instant TTL
        d.check_and_register("key-1").unwrap();
        d.complete("key-1", "old").unwrap();
        // Immediately expired — should be New again
        let r = d.check_and_register("key-1").unwrap();
        assert_eq!(r, DeduplicationResult::New);
    }

    // ── BackpressureGuard ─────────────────────────────────────────────────────

    #[test]
    fn test_backpressure_guard_rejects_zero_capacity() {
        assert!(BackpressureGuard::new(0).is_err());
    }

    #[test]
    fn test_backpressure_guard_acquire_within_capacity() {
        let g = BackpressureGuard::new(5).unwrap();
        assert!(g.try_acquire().is_ok());
        assert_eq!(g.depth().unwrap(), 1);
    }

    #[test]
    fn test_backpressure_guard_sheds_when_full() {
        let g = BackpressureGuard::new(2).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        let result = g.try_acquire();
        assert!(matches!(
            result,
            Err(AgentRuntimeError::BackpressureShed { .. })
        ));
    }

    #[test]
    fn test_backpressure_guard_release_decrements_depth() {
        let g = BackpressureGuard::new(3).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        g.release().unwrap();
        assert_eq!(g.depth().unwrap(), 1);
    }

    #[test]
    fn test_backpressure_guard_release_on_empty_is_noop() {
        let g = BackpressureGuard::new(3).unwrap();
        g.release().unwrap(); // Should not fail
        assert_eq!(g.depth().unwrap(), 0);
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_runs_stages_in_order() {
        let p = Pipeline::new()
            .add_stage("upper", |s| Ok(s.to_uppercase()))
            .add_stage("append", |s| Ok(format!("{s}!")));
        let result = p.run("hello".into()).unwrap();
        assert_eq!(result, "HELLO!");
    }

    #[test]
    fn test_pipeline_empty_pipeline_returns_input() {
        let p = Pipeline::new();
        assert_eq!(p.run("test".into()).unwrap(), "test");
    }

    #[test]
    fn test_pipeline_stage_failure_short_circuits() {
        let p = Pipeline::new()
            .add_stage("fail", |_| {
                Err(AgentRuntimeError::Orchestration("boom".into()))
            })
            .add_stage("never", |s| Ok(s));
        assert!(p.run("input".into()).is_err());
    }

    #[test]
    fn test_pipeline_stage_count() {
        let p = Pipeline::new()
            .add_stage("s1", |s| Ok(s))
            .add_stage("s2", |s| Ok(s));
        assert_eq!(p.stage_count(), 2);
    }
}
