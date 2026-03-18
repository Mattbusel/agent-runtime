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
//! - Cross-node circuit breakers (single-process only, unless a distributed backend is provided)
//! - Persistent deduplication (in-memory, bounded TTL)
//! - Distributed backpressure

use crate::error::AgentRuntimeError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Maximum delay between retries — caps exponential growth.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

// ── Lock recovery helpers ──────────────────────────────────────────────────────

/// Recover from a poisoned mutex by extracting the inner value.
///
/// Instead of propagating a lock-poison error, this logs a warning and
/// returns the inner guard so the caller can continue operating.
#[allow(dead_code)]
fn recover_lock<'a, T>(
    result: std::sync::LockResult<std::sync::MutexGuard<'a, T>>,
    ctx: &str,
) -> std::sync::MutexGuard<'a, T>
where
    T: ?Sized,
{
    match result {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("mutex poisoned in {ctx}, recovering inner value");
            poisoned.into_inner()
        }
    }
}

/// Acquire a lock, recovering from poisoning and logging slow acquisitions.
///
/// Times the lock acquisition; if it exceeds 5 ms a warning is emitted so
/// contention hot-spots can be identified in production logs.
fn timed_lock<'a, T>(mutex: &'a Mutex<T>, ctx: &str) -> std::sync::MutexGuard<'a, T>
where
    T: ?Sized,
{
    let start = std::time::Instant::now();
    let result = mutex.lock();
    let elapsed = start.elapsed();
    if elapsed > std::time::Duration::from_millis(5) {
        tracing::warn!(
            duration_ms = elapsed.as_millis(),
            ctx = ctx,
            "slow mutex acquisition"
        );
    }
    match result {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("mutex poisoned in {ctx}, recovering inner value");
            poisoned.into_inner()
        }
    }
}

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
        let multiplier = 1u64.checked_shl(exp.min(63)).unwrap_or(u64::MAX);
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
    /// Circuit is operating normally; requests pass through.
    Closed,
    /// Circuit has tripped; requests are fast-failed without calling the operation.
    Open {
        /// The instant at which the circuit was opened.
        opened_at: Instant,
    },
    /// Recovery probe period; the next request will be attempted to test recovery.
    HalfOpen,
}

/// Backend for circuit breaker state storage.
///
/// Implement this trait to share circuit breaker state across processes
/// (e.g., via Redis). The in-process default is `InMemoryCircuitBreakerBackend`.
///
/// Note: Methods are synchronous to avoid pulling in `async-trait`. A
/// distributed backend (e.g., Redis) can internally spawn a Tokio runtime.
pub trait CircuitBreakerBackend: Send + Sync {
    fn increment_failures(&self, service: &str) -> u32;
    fn reset_failures(&self, service: &str);
    fn get_failures(&self, service: &str) -> u32;
    fn set_open_at(&self, service: &str, at: std::time::Instant);
    fn clear_open_at(&self, service: &str);
    fn get_open_at(&self, service: &str) -> Option<std::time::Instant>;
}

// ── InMemoryCircuitBreakerBackend ─────────────────────────────────────────────

/// In-process circuit breaker backend backed by a `Mutex`.
pub struct InMemoryCircuitBreakerBackend {
    inner: Arc<Mutex<InMemoryBackendState>>,
}

struct InMemoryBackendState {
    consecutive_failures: u32,
    open_at: Option<std::time::Instant>,
}

impl InMemoryCircuitBreakerBackend {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(InMemoryBackendState {
                consecutive_failures: 0,
                open_at: None,
            })),
        }
    }
}

impl Default for InMemoryCircuitBreakerBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerBackend for InMemoryCircuitBreakerBackend {
    fn increment_failures(&self, _service: &str) -> u32 {
        let mut state = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::increment_failures");
        state.consecutive_failures += 1;
        state.consecutive_failures
    }

    fn reset_failures(&self, _service: &str) {
        let mut state = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::reset_failures");
        state.consecutive_failures = 0;
    }

    fn get_failures(&self, _service: &str) -> u32 {
        let state = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::get_failures");
        state.consecutive_failures
    }

    fn set_open_at(&self, _service: &str, at: std::time::Instant) {
        let mut state = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::set_open_at");
        state.open_at = Some(at);
    }

    fn clear_open_at(&self, _service: &str) {
        let mut state = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::clear_open_at");
        state.open_at = None;
    }

    fn get_open_at(&self, _service: &str) -> Option<std::time::Instant> {
        let state = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::get_open_at");
        state.open_at
    }
}

// ── CircuitBreaker ────────────────────────────────────────────────────────────

/// Circuit breaker guarding a fallible operation.
///
/// ## Guarantees
/// - Opens after `threshold` consecutive failures
/// - Transitions to `HalfOpen` after `recovery_window` has elapsed
/// - Closes on the first successful probe in `HalfOpen`
#[derive(Clone)]
pub struct CircuitBreaker {
    threshold: u32,
    recovery_window: Duration,
    service: String,
    backend: Arc<dyn CircuitBreakerBackend>,
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("threshold", &self.threshold)
            .field("recovery_window", &self.recovery_window)
            .field("service", &self.service)
            .finish()
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker backed by an in-memory backend.
    ///
    /// # Arguments
    /// * `service` — name used in error messages and logs
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
        let service = service.into();
        Ok(Self {
            threshold,
            recovery_window,
            service,
            backend: Arc::new(InMemoryCircuitBreakerBackend::new()),
        })
    }

    /// Replace the default in-memory backend with a custom one.
    ///
    /// Useful for sharing circuit breaker state across processes.
    pub fn with_backend(mut self, backend: Arc<dyn CircuitBreakerBackend>) -> Self {
        self.backend = backend;
        self
    }

    /// Derive the current `CircuitState` from backend storage.
    #[allow(dead_code)]
    fn current_state(&self) -> CircuitState {
        match self.backend.get_open_at(&self.service) {
            Some(opened_at) => CircuitState::Open { opened_at },
            None => {
                // We encode HalfOpen as failures == threshold but no open_at.
                // However, the transition to HalfOpen is done in `call`, so
                // outside of `call` we can only observe Closed here.
                // The `call` method manages the HalfOpen flag via a per-call
                // transient field — see below.
                CircuitState::Closed
            }
        }
    }

    /// Attempt to call `f`, respecting the circuit breaker state.
    ///
    /// # Returns
    /// - `Ok(T)` — if `f` succeeds (resets failure count)
    /// - `Err(AgentRuntimeError::CircuitOpen)` — if the breaker is open
    /// - `Err(...)` — if `f` fails (may open the breaker)
    #[tracing::instrument(skip(self, f))]
    pub fn call<T, E, F>(&self, f: F) -> Result<T, AgentRuntimeError>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        // Determine effective state, potentially transitioning Open → HalfOpen.
        let effective_state = match self.backend.get_open_at(&self.service) {
            Some(opened_at) => {
                if opened_at.elapsed() >= self.recovery_window {
                    // Clear open_at to signal HalfOpen; failures remain.
                    self.backend.clear_open_at(&self.service);
                    tracing::info!("circuit moved to half-open for {}", self.service);
                    CircuitState::HalfOpen
                } else {
                    CircuitState::Open { opened_at }
                }
            }
            None => {
                // Either Closed or HalfOpen (after a prior transition).
                // We distinguish by checking whether failures >= threshold
                // but no open_at is set — that means we are in HalfOpen.
                let failures = self.backend.get_failures(&self.service);
                if failures >= self.threshold {
                    CircuitState::HalfOpen
                } else {
                    CircuitState::Closed
                }
            }
        };

        tracing::debug!("circuit state: {:?}", effective_state);

        match effective_state {
            CircuitState::Open { .. } => {
                return Err(AgentRuntimeError::CircuitOpen {
                    service: self.service.clone(),
                });
            }
            CircuitState::Closed | CircuitState::HalfOpen => {}
        }

        // Execute the operation.
        match f() {
            Ok(val) => {
                self.backend.reset_failures(&self.service);
                self.backend.clear_open_at(&self.service);
                tracing::info!("circuit closed for {}", self.service);
                Ok(val)
            }
            Err(e) => {
                let failures = self.backend.increment_failures(&self.service);
                if failures >= self.threshold {
                    let now = Instant::now();
                    self.backend.set_open_at(&self.service, now);
                    tracing::info!("circuit opened for {}", self.service);
                }
                Err(AgentRuntimeError::Orchestration(e.to_string()))
            }
        }
    }

    /// Return the current circuit state.
    pub fn state(&self) -> Result<CircuitState, AgentRuntimeError> {
        let state = match self.backend.get_open_at(&self.service) {
            Some(opened_at) => {
                if opened_at.elapsed() >= self.recovery_window {
                    // Would transition to HalfOpen on next call; report HalfOpen.
                    let failures = self.backend.get_failures(&self.service);
                    if failures >= self.threshold {
                        CircuitState::HalfOpen
                    } else {
                        CircuitState::Closed
                    }
                } else {
                    CircuitState::Open { opened_at }
                }
            }
            None => {
                let failures = self.backend.get_failures(&self.service);
                if failures >= self.threshold {
                    CircuitState::HalfOpen
                } else {
                    CircuitState::Closed
                }
            }
        };
        Ok(state)
    }

    /// Return the consecutive failure count.
    pub fn failure_count(&self) -> Result<u32, AgentRuntimeError> {
        Ok(self.backend.get_failures(&self.service))
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
        let mut inner = timed_lock(&self.inner, "Deduplicator::check_and_register");

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
        let mut inner = timed_lock(&self.inner, "Deduplicator::complete");
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
/// - Optional soft limit emits a warning when depth reaches the threshold
#[derive(Debug, Clone)]
pub struct BackpressureGuard {
    capacity: usize,
    soft_capacity: Option<usize>,
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
            soft_capacity: None,
            inner: Arc::new(Mutex::new(0)),
        })
    }

    /// Set a soft capacity threshold. When depth reaches this level, a warning
    /// is logged but the request is still accepted (up to hard capacity).
    pub fn with_soft_limit(mut self, soft: usize) -> Result<Self, AgentRuntimeError> {
        if soft >= self.capacity {
            return Err(AgentRuntimeError::Orchestration(
                "soft_capacity must be less than hard capacity".into(),
            ));
        }
        self.soft_capacity = Some(soft);
        Ok(self)
    }

    /// Try to acquire a slot.
    ///
    /// Emits a warning when the soft limit is reached (if configured), but
    /// still accepts the request until hard capacity is exceeded.
    ///
    /// # Returns
    /// - `Ok(())` — slot acquired
    /// - `Err(AgentRuntimeError::BackpressureShed)` — hard capacity exceeded
    pub fn try_acquire(&self) -> Result<(), AgentRuntimeError> {
        let mut depth = timed_lock(&self.inner, "BackpressureGuard::try_acquire");
        if *depth >= self.capacity {
            return Err(AgentRuntimeError::BackpressureShed {
                depth: *depth,
                capacity: self.capacity,
            });
        }
        *depth += 1;
        if let Some(soft) = self.soft_capacity {
            if *depth >= soft {
                tracing::warn!(
                    depth = *depth,
                    soft_capacity = soft,
                    hard_capacity = self.capacity,
                    "backpressure approaching hard limit"
                );
            }
        }
        Ok(())
    }

    /// Release a previously acquired slot.
    pub fn release(&self) -> Result<(), AgentRuntimeError> {
        let mut depth = timed_lock(&self.inner, "BackpressureGuard::release");
        *depth = depth.saturating_sub(1);
        Ok(())
    }

    /// Return the current depth.
    pub fn depth(&self) -> Result<usize, AgentRuntimeError> {
        let depth = timed_lock(&self.inner, "BackpressureGuard::depth");
        Ok(*depth)
    }

    /// Return the ratio of current depth to soft capacity as a value in `[0.0, ∞)`.
    ///
    /// Returns `0.0` if no soft limit has been configured.
    /// Values above `1.0` mean the soft limit has been exceeded.
    pub fn soft_depth_ratio(&self) -> f32 {
        match self.soft_capacity {
            None => 0.0,
            Some(soft) => {
                let depth = timed_lock(&self.inner, "BackpressureGuard::soft_depth_ratio");
                *depth as f32 / soft as f32
            }
        }
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
    #[tracing::instrument(skip(self))]
    pub fn run(&self, input: String) -> Result<String, AgentRuntimeError> {
        let mut current = input;
        for stage in &self.stages {
            tracing::debug!("running stage: {}", stage.name);
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

    #[test]
    fn test_circuit_breaker_with_custom_backend_uses_backend_state() {
        // Build a custom backend and share it between two circuit breakers
        // to verify that state is read from and written to the backend.
        let shared_backend: Arc<dyn CircuitBreakerBackend> =
            Arc::new(InMemoryCircuitBreakerBackend::new());

        let cb1 = CircuitBreaker::new("svc", 2, Duration::from_secs(60))
            .unwrap()
            .with_backend(Arc::clone(&shared_backend));

        let cb2 = CircuitBreaker::new("svc", 2, Duration::from_secs(60))
            .unwrap()
            .with_backend(Arc::clone(&shared_backend));

        // Trigger one failure via cb1
        let _: Result<(), AgentRuntimeError> = cb1.call(|| Err::<(), _>("fail".to_string()));

        // cb2 should observe the failure recorded by cb1
        assert_eq!(cb2.failure_count().unwrap(), 1);

        // Trigger the second failure to open the circuit via cb1
        let _: Result<(), AgentRuntimeError> = cb1.call(|| Err::<(), _>("fail again".to_string()));

        // cb2 should now see the circuit as open
        assert!(matches!(cb2.state().unwrap(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_in_memory_backend_increments_and_resets() {
        let backend = InMemoryCircuitBreakerBackend::new();

        assert_eq!(backend.get_failures("svc"), 0);

        let count = backend.increment_failures("svc");
        assert_eq!(count, 1);

        let count = backend.increment_failures("svc");
        assert_eq!(count, 2);

        backend.reset_failures("svc");
        assert_eq!(backend.get_failures("svc"), 0);

        // open_at round-trip
        assert!(backend.get_open_at("svc").is_none());
        let now = Instant::now();
        backend.set_open_at("svc", now);
        assert!(backend.get_open_at("svc").is_some());
        backend.clear_open_at("svc");
        assert!(backend.get_open_at("svc").is_none());
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

    // ── Item 13: BackpressureGuard soft limit ──────────────────────────────────

    #[test]
    fn test_backpressure_soft_limit_rejects_invalid_config() {
        // soft >= capacity must be rejected
        let g = BackpressureGuard::new(5).unwrap();
        assert!(g.with_soft_limit(5).is_err());
        let g = BackpressureGuard::new(5).unwrap();
        assert!(g.with_soft_limit(6).is_err());
    }

    #[test]
    fn test_backpressure_soft_limit_accepts_requests_below_soft() {
        let g = BackpressureGuard::new(5)
            .unwrap()
            .with_soft_limit(2)
            .unwrap();
        // Both acquires below soft limit should succeed
        assert!(g.try_acquire().is_ok());
        assert!(g.try_acquire().is_ok());
        assert_eq!(g.depth().unwrap(), 2);
    }

    #[test]
    fn test_backpressure_with_soft_limit_still_sheds_at_hard_capacity() {
        let g = BackpressureGuard::new(3)
            .unwrap()
            .with_soft_limit(2)
            .unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap(); // reaches hard limit
        let result = g.try_acquire();
        assert!(matches!(
            result,
            Err(AgentRuntimeError::BackpressureShed { .. })
        ));
    }

    // ── Item 14: timed_lock concurrency correctness ───────────────────────────

    #[test]
    fn test_backpressure_concurrent_acquires_are_consistent() {
        use std::sync::Arc;
        use std::thread;

        let g = Arc::new(BackpressureGuard::new(100).unwrap());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let g_clone = Arc::clone(&g);
            handles.push(thread::spawn(move || {
                g_clone.try_acquire().ok();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // All 10 threads acquired a slot; depth must be exactly 10
        assert_eq!(g.depth().unwrap(), 10);
    }
}
