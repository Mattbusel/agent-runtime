//! # Module: Orchestrator
//!
//! ## Responsibility
//! Provides a composable LLM pipeline with circuit breaking, retry, deduplication,
//! and backpressure. Mirrors the public API of `tokio-prompt-orchestrator`.
//!
//! ## Guarantees
//! - Thread-safe: all types wrap state in `Arc<Mutex<_>>` or atomics
//! - Circuit breaker opens after `threshold` consecutive failures
//! - RetryPolicy delays grow exponentially and are capped at [`MAX_RETRY_DELAY`]
//! - Deduplicator is deterministic and non-blocking
//! - BackpressureGuard never exceeds declared hard capacity
//! - Non-panicking: all operations return `Result`
//!
//! ## NOT Responsible For
//! - Cross-node circuit breakers (single-process only, unless a distributed backend is provided)
//! - Persistent deduplication (in-memory, bounded TTL)
//! - Distributed backpressure
//!
//! ## Composing the Primitives
//!
//! The four primitives are designed to be layered. A typical production setup:
//!
//! ```text
//! request
//!   │
//!   ▼
//! BackpressureGuard  ← shed if too many in-flight requests
//!   │
//!   ▼
//! Deduplicator       ← return cached result for duplicate keys
//!   │
//!   ▼
//! CircuitBreaker     ← fast-fail if the downstream is unhealthy
//!   │
//!   ▼
//! RetryPolicy        ← retry transient failures with exponential backoff
//!   │
//!   ▼
//! Pipeline           ← transform request/response through named stages
//! ```

use crate::error::AgentRuntimeError;
use crate::util::timed_lock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Maximum delay between retries — caps exponential growth.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

// ── RetryPolicy ───────────────────────────────────────────────────────────────

/// Retry mode: exponential backoff or constant interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryKind {
    /// Delay doubles each attempt: `base_delay * 2^(attempt-1)`.
    Exponential,
    /// Delay is fixed at `base_delay` for every attempt.
    Constant,
}

/// Configurable retry policy with exponential backoff or constant interval.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Base delay for the first retry.
    pub base_delay: Duration,
    /// Whether to use exponential or constant delay.
    pub kind: RetryKind,
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
        if base_ms == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "base_ms must be >= 1 to avoid zero-delay busy-loop retries".into(),
            ));
        }
        Ok(Self {
            max_attempts,
            base_delay: Duration::from_millis(base_ms),
            kind: RetryKind::Exponential,
        })
    }

    /// Create a constant (fixed-interval) retry policy.
    ///
    /// Every retry waits exactly `delay_ms` milliseconds regardless of attempt
    /// number, unlike [`exponential`] which doubles the delay each time.
    ///
    /// [`exponential`]: RetryPolicy::exponential
    ///
    /// # Returns
    /// - `Ok(RetryPolicy)` — on success
    /// - `Err(AgentRuntimeError::Orchestration)` — if `max_attempts == 0` or `delay_ms == 0`
    pub fn constant(max_attempts: u32, delay_ms: u64) -> Result<Self, AgentRuntimeError> {
        if max_attempts == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "max_attempts must be >= 1".into(),
            ));
        }
        if delay_ms == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "delay_ms must be >= 1 to avoid busy-loop retries".into(),
            ));
        }
        Ok(Self {
            max_attempts,
            base_delay: Duration::from_millis(delay_ms),
            kind: RetryKind::Constant,
        })
    }

    /// Create a no-retry policy (single attempt, no delay).
    ///
    /// Useful for one-shot operations or when the caller manages retry logic externally.
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            base_delay: Duration::ZERO,
            kind: RetryKind::Constant,
        }
    }

    /// Return `true` if this policy makes at most one attempt with no delay.
    ///
    /// Equivalent to `max_attempts == 1 && base_delay == Duration::ZERO`.
    pub fn is_none(&self) -> bool {
        self.max_attempts == 1 && self.base_delay == Duration::ZERO
    }

    /// Return a copy of this policy with `max_attempts` changed.
    ///
    /// # Errors
    /// Returns `Err` if `n == 0`.
    pub fn with_max_attempts(mut self, n: u32) -> Result<Self, AgentRuntimeError> {
        if n == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "max_attempts must be >= 1".into(),
            ));
        }
        self.max_attempts = n;
        Ok(self)
    }

    /// Return the configured maximum number of attempts.
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Return `true` if this policy performs no retries (max_attempts ≤ 1).
    ///
    /// Useful for short-circuiting retry logic in hot paths.
    pub fn is_no_retry(&self) -> bool {
        self.max_attempts <= 1
    }

    /// Return `true` if this policy allows at least one retry (max_attempts > 1).
    ///
    /// Complement of [`is_no_retry`].
    ///
    /// [`is_no_retry`]: RetryPolicy::is_no_retry
    pub fn will_retry_at_all(&self) -> bool {
        self.max_attempts > 1
    }

    /// Return `true` if this policy uses exponential back-off between retries.
    pub fn is_exponential(&self) -> bool {
        matches!(self.kind, RetryKind::Exponential)
    }

    /// Return `true` if this policy uses a constant (fixed-interval) delay between retries.
    pub fn is_constant(&self) -> bool {
        matches!(self.kind, RetryKind::Constant)
    }

    /// Return the configured base delay in milliseconds.
    ///
    /// For constant policies this equals every per-retry delay.  For
    /// exponential policies this is the delay before the first retry.
    pub fn base_delay_ms(&self) -> u64 {
        self.base_delay.as_millis() as u64
    }

    /// Return a copy of this policy with the base delay changed to `ms` milliseconds.
    ///
    /// # Errors
    /// Returns `Err` if `ms == 0`.
    pub fn with_base_delay_ms(mut self, ms: u64) -> Result<Self, AgentRuntimeError> {
        if ms == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "base_delay_ms must be >= 1 to avoid busy-loop retries".into(),
            ));
        }
        self.base_delay = Duration::from_millis(ms);
        Ok(self)
    }

    /// Return the delay for `attempt` in whole milliseconds.
    ///
    /// Convenience wrapper around [`delay_for`] for use in logging and metrics
    /// where a `u64` is easier to handle than a `Duration`.
    ///
    /// [`delay_for`]: RetryPolicy::delay_for
    pub fn delay_ms_for(&self, attempt: u32) -> u64 {
        self.delay_for(attempt).as_millis() as u64
    }

    /// Return the total maximum delay in milliseconds across all retry attempts.
    ///
    /// Sums `delay_for(attempt)` for every attempt from 1 to `max_attempts`.
    /// Useful for estimating worst-case latency budgets.
    pub fn total_max_delay_ms(&self) -> u64 {
        (1..=self.max_attempts)
            .map(|a| self.delay_for(a).as_millis() as u64)
            .sum()
    }

    /// Return the number of attempts still available after `attempt` have been made.
    ///
    /// Returns `0` once the budget is exhausted (`attempt >= max_attempts`).
    pub fn attempts_remaining(&self, attempt: u32) -> u32 {
        self.max_attempts.saturating_sub(attempt)
    }

    /// Return `true` if another attempt is permitted after `attempt` failures.
    ///
    /// `attempt` is the number of attempts already made (0-based: `0` means
    /// no attempt has been made yet).  Returns `false` once the budget is
    /// exhausted (i.e. `attempt >= max_attempts`).
    pub fn can_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }

    /// Compute the delay before the given attempt number (1-based).
    ///
    /// - [`RetryKind::Exponential`]: `base_delay * 2^(attempt-1)`, capped at `MAX_RETRY_DELAY`.
    /// - [`RetryKind::Constant`]: always returns `base_delay`.
    pub fn delay_for(&self, attempt: u32) -> Duration {
        match self.kind {
            RetryKind::Constant => self.base_delay.min(MAX_RETRY_DELAY),
            RetryKind::Exponential => {
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
    }
}

// ── CircuitBreaker ────────────────────────────────────────────────────────────

/// Tracks failure rates and opens when the threshold is exceeded.
///
/// States: `Closed` (normal) → `Open` (fast-fail) → `HalfOpen` (probe).
///
/// Note: `PartialEq` is implemented manually because the `Open` variant
/// contains `std::time::Instant` which does not implement `Eq`. The manual
/// implementation compares only the variant discriminant, not the timestamp.
#[derive(Debug, Clone)]
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

impl PartialEq for CircuitState {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CircuitState::Closed, CircuitState::Closed) => true,
            (CircuitState::Open { .. }, CircuitState::Open { .. }) => true,
            (CircuitState::HalfOpen, CircuitState::HalfOpen) => true,
            _ => false,
        }
    }
}

impl Eq for CircuitState {}

/// Backend for circuit breaker state storage.
///
/// Implement this trait to share circuit breaker state across processes
/// (e.g., via Redis). The in-process default is `InMemoryCircuitBreakerBackend`.
///
/// Note: Methods are synchronous to avoid pulling in `async-trait`. A
/// distributed backend (e.g., Redis) can internally spawn a Tokio runtime.
pub trait CircuitBreakerBackend: Send + Sync {
    /// Increment the consecutive failure count for `service` and return the new count.
    fn increment_failures(&self, service: &str) -> u32;
    /// Reset the consecutive failure count for `service` to zero.
    fn reset_failures(&self, service: &str);
    /// Return the current consecutive failure count for `service`.
    fn get_failures(&self, service: &str) -> u32;
    /// Record the instant at which the circuit was opened for `service`.
    fn set_open_at(&self, service: &str, at: std::time::Instant);
    /// Clear the open-at timestamp, effectively moving the circuit to Closed or HalfOpen.
    fn clear_open_at(&self, service: &str);
    /// Return the instant at which the circuit was opened, or `None` if it is not open.
    fn get_open_at(&self, service: &str) -> Option<std::time::Instant>;
}

// ── InMemoryCircuitBreakerBackend ─────────────────────────────────────────────

/// In-process circuit breaker backend backed by a `Mutex<HashMap>`.
///
/// Each service name gets its own independent failure counter and open-at
/// timestamp.  Multiple `CircuitBreaker` instances that share the same
/// backend (via [`CircuitBreaker::with_backend`]) will correctly track
/// failures per service rather than sharing a single counter.
pub struct InMemoryCircuitBreakerBackend {
    inner: Arc<Mutex<HashMap<String, InMemoryServiceState>>>,
}

#[derive(Default)]
struct InMemoryServiceState {
    consecutive_failures: u32,
    open_at: Option<std::time::Instant>,
}

impl InMemoryCircuitBreakerBackend {
    /// Create a new in-memory backend with all counters at zero.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryCircuitBreakerBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerBackend for InMemoryCircuitBreakerBackend {
    fn increment_failures(&self, service: &str) -> u32 {
        let mut map = timed_lock(
            &self.inner,
            "InMemoryCircuitBreakerBackend::increment_failures",
        );
        let state = map.entry(service.to_owned()).or_default();
        state.consecutive_failures += 1;
        state.consecutive_failures
    }

    fn reset_failures(&self, service: &str) {
        let mut map = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::reset_failures");
        if let Some(state) = map.get_mut(service) {
            state.consecutive_failures = 0;
        }
    }

    fn get_failures(&self, service: &str) -> u32 {
        let map = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::get_failures");
        map.get(service).map_or(0, |s| s.consecutive_failures)
    }

    fn set_open_at(&self, service: &str, at: std::time::Instant) {
        let mut map = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::set_open_at");
        map.entry(service.to_owned()).or_default().open_at = Some(at);
    }

    fn clear_open_at(&self, service: &str) {
        let mut map = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::clear_open_at");
        if let Some(state) = map.get_mut(service) {
            state.open_at = None;
        }
    }

    fn get_open_at(&self, service: &str) -> Option<std::time::Instant> {
        let map = timed_lock(&self.inner, "InMemoryCircuitBreakerBackend::get_open_at");
        map.get(service).and_then(|s| s.open_at)
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

    /// Attempt to call `f`, respecting the circuit breaker state.
    ///
    /// # Errors
    /// - `AgentRuntimeError::CircuitOpen` — the breaker is in the `Open` state
    ///   and the recovery window has not yet elapsed
    /// - `AgentRuntimeError::Orchestration` — `f` returned an error; the error
    ///   message is the `Display` of the inner error. This call may open the
    ///   breaker if it pushes the consecutive failure count above `threshold`.
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

    /// Record a successful call, resetting the consecutive failure counter.
    ///
    /// Call this when a protected operation succeeds so the circuit can
    /// transition back to `Closed` after a `HalfOpen` probe.
    pub fn record_success(&self) {
        self.backend.reset_failures(&self.service);
        self.backend.clear_open_at(&self.service);
    }

    /// Record a failed call, incrementing the consecutive failure counter.
    ///
    /// Opens the circuit when the failure count reaches `threshold`.
    pub fn record_failure(&self) {
        let failures = self.backend.increment_failures(&self.service);
        if failures >= self.threshold {
            self.backend.set_open_at(&self.service, Instant::now());
            tracing::info!("circuit opened for {} (manual record)", self.service);
        }
    }

    /// Return the service name this circuit breaker is protecting.
    pub fn service_name(&self) -> &str {
        &self.service
    }

    /// Return `true` if the circuit is currently `Closed` (healthy).
    pub fn is_closed(&self) -> bool {
        matches!(self.state(), Ok(CircuitState::Closed))
    }

    /// Return `true` if the circuit is currently `Open` (fast-failing).
    pub fn is_open(&self) -> bool {
        matches!(self.state(), Ok(CircuitState::Open { .. }))
    }

    /// Return `true` if the circuit is currently `HalfOpen` (probing).
    pub fn is_half_open(&self) -> bool {
        matches!(self.state(), Ok(CircuitState::HalfOpen))
    }

    /// Return `true` if the circuit is in a state that allows calls to proceed.
    ///
    /// Calls are allowed in both `Closed` and `HalfOpen` states; only `Open`
    /// fast-fails.
    pub fn is_healthy(&self) -> bool {
        !self.is_open()
    }

    /// Return the configured consecutive-failure threshold.
    ///
    /// The circuit opens when `failure_count()` reaches this value.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Return the current failure count as a ratio of the threshold.
    ///
    /// Returns a value in `[0.0, 1.0]` where `1.0` (or greater) means the
    /// circuit will open (or is already open).  Returns `0.0` when the
    /// threshold is zero to avoid division by zero.
    pub fn failure_rate(&self) -> f64 {
        if self.threshold == 0 {
            return 0.0;
        }
        let failures = self.backend.get_failures(&self.service);
        failures as f64 / self.threshold as f64
    }

    /// Return `true` when `failure_count()` has reached the configured threshold.
    ///
    /// The circuit opens immediately when this returns `true` on the next
    /// `record_failure` call.
    pub fn is_at_threshold(&self) -> bool {
        let failures = self.backend.get_failures(&self.service);
        failures >= self.threshold
    }

    /// Return the number of additional failures needed to open the circuit.
    ///
    /// Returns `0` when the circuit is already at or beyond threshold.
    pub fn failures_until_open(&self) -> u32 {
        let failures = self.backend.get_failures(&self.service);
        self.threshold.saturating_sub(failures)
    }

    /// Return the configured recovery window duration.
    ///
    /// After the circuit has been `Open` for this long, it transitions to
    /// `HalfOpen` and allows the next call through as a recovery probe.
    pub fn recovery_window(&self) -> std::time::Duration {
        self.recovery_window
    }

    /// Force the circuit back to `Closed` state, resetting all failure counters.
    ///
    /// Useful for tests and manual operator recovery.  Under normal operation
    /// the circuit closes automatically after a successful `HalfOpen` probe.
    pub fn reset(&self) {
        self.backend.reset_failures(&self.service);
        self.backend.clear_open_at(&self.service);
        tracing::info!("circuit manually reset to Closed for {}", self.service);
    }

    /// Execute an async fallible operation under the circuit breaker using an
    /// [`AsyncCircuitBreakerBackend`].
    ///
    /// This is the async counterpart of [`call`] and is intended for backends
    /// that perform genuine async I/O (e.g. Redis, etcd, distributed stores).
    /// The in-process default can be used via [`InMemoryCircuitBreakerBackend`]
    /// which trivially implements `AsyncCircuitBreakerBackend`.
    ///
    /// [`call`]: CircuitBreaker::call
    #[tracing::instrument(skip(self, backend, f))]
    pub async fn async_call<T, E, F, Fut>(
        &self,
        backend: &dyn AsyncCircuitBreakerBackend,
        f: F,
    ) -> Result<T, AgentRuntimeError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        // Determine effective state via async backend.
        let effective_state = match backend.get_open_at(&self.service).await {
            Some(opened_at) => {
                if opened_at.elapsed() >= self.recovery_window {
                    backend.clear_open_at(&self.service).await;
                    tracing::info!("circuit async moved to half-open for {}", self.service);
                    CircuitState::HalfOpen
                } else {
                    CircuitState::Open { opened_at }
                }
            }
            None => {
                let failures = backend.get_failures(&self.service).await;
                if failures >= self.threshold {
                    CircuitState::HalfOpen
                } else {
                    CircuitState::Closed
                }
            }
        };

        if let CircuitState::Open { .. } = effective_state {
            return Err(AgentRuntimeError::CircuitOpen {
                service: self.service.clone(),
            });
        }

        match f().await {
            Ok(val) => {
                backend.reset_failures(&self.service).await;
                backend.clear_open_at(&self.service).await;
                Ok(val)
            }
            Err(e) => {
                let failures = backend.increment_failures(&self.service).await;
                if failures >= self.threshold {
                    backend
                        .set_open_at(&self.service, Instant::now())
                        .await;
                    tracing::info!("circuit async opened for {}", self.service);
                }
                Err(AgentRuntimeError::Orchestration(e.to_string()))
            }
        }
    }
}

// ── AsyncCircuitBreakerBackend ────────────────────────────────────────────────

/// Async counterpart of [`CircuitBreakerBackend`] for distributed backends.
///
/// Implement this trait for backends that require genuine async I/O — e.g. Redis,
/// etcd, or any network-based store — so they don't need to embed their own
/// blocking runtime.
///
/// [`InMemoryCircuitBreakerBackend`] implements this trait with trivially-async
/// wrappers for use in testing and single-process deployments.
#[async_trait::async_trait]
pub trait AsyncCircuitBreakerBackend: Send + Sync {
    /// Increment the consecutive failure count and return the new count.
    async fn increment_failures(&self, service: &str) -> u32;
    /// Reset the consecutive failure count to zero.
    async fn reset_failures(&self, service: &str);
    /// Return the current consecutive failure count.
    async fn get_failures(&self, service: &str) -> u32;
    /// Record the instant at which the circuit was opened.
    async fn set_open_at(&self, service: &str, at: Instant);
    /// Clear the open-at timestamp.
    async fn clear_open_at(&self, service: &str);
    /// Return the instant at which the circuit was opened, or `None`.
    async fn get_open_at(&self, service: &str) -> Option<Instant>;
}

#[async_trait::async_trait]
impl AsyncCircuitBreakerBackend for InMemoryCircuitBreakerBackend {
    async fn increment_failures(&self, service: &str) -> u32 {
        <Self as CircuitBreakerBackend>::increment_failures(self, service)
    }
    async fn reset_failures(&self, service: &str) {
        <Self as CircuitBreakerBackend>::reset_failures(self, service);
    }
    async fn get_failures(&self, service: &str) -> u32 {
        <Self as CircuitBreakerBackend>::get_failures(self, service)
    }
    async fn set_open_at(&self, service: &str, at: Instant) {
        <Self as CircuitBreakerBackend>::set_open_at(self, service, at);
    }
    async fn clear_open_at(&self, service: &str) {
        <Self as CircuitBreakerBackend>::clear_open_at(self, service);
    }
    async fn get_open_at(&self, service: &str) -> Option<Instant> {
        <Self as CircuitBreakerBackend>::get_open_at(self, service)
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
/// - Optional `max_entries` cap bounds memory independently of TTL
#[derive(Debug, Clone)]
pub struct Deduplicator {
    ttl: Duration,
    /// Optional hard cap on cached entries. When exceeded the oldest entry is
    /// evicted before inserting the new one, bounding memory growth even when
    /// all keys are unique and none have expired yet.
    max_entries: Option<usize>,
    inner: Arc<Mutex<DeduplicatorInner>>,
}

#[derive(Debug)]
struct DeduplicatorInner {
    cache: HashMap<String, (String, Instant)>, // key → (result, inserted_at)
    in_flight: HashMap<String, Instant>,       // key → started_at
    /// Insertion-ordered keys for O(1) FIFO eviction when `max_entries` is set.
    cache_order: std::collections::VecDeque<String>,
    /// Tracks calls since the last full expiry scan. Full scans run every
    /// `EXPIRY_INTERVAL` calls; per-key inline checks maintain correctness
    /// between scans.
    call_count: u64,
}

impl Deduplicator {
    /// Create a new deduplicator with the given TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            max_entries: None,
            inner: Arc::new(Mutex::new(DeduplicatorInner {
                cache: HashMap::new(),
                in_flight: HashMap::new(),
                cache_order: std::collections::VecDeque::new(),
                call_count: 0,
            })),
        }
    }

    /// Set a hard cap on the number of cached (completed) entries.
    ///
    /// When the cache is full the oldest entry (by insertion time) is evicted
    /// before the new entry is stored.  This bounds memory growth for workloads
    /// where all request keys are unique and the TTL has not yet expired.
    ///
    /// # Returns
    /// - `Err(AgentRuntimeError::Orchestration)` if `max == 0`
    pub fn with_max_entries(mut self, max: usize) -> Result<Self, AgentRuntimeError> {
        if max == 0 {
            return Err(AgentRuntimeError::Orchestration(
                "Deduplicator max_entries must be >= 1".into(),
            ));
        }
        self.max_entries = Some(max);
        Ok(self)
    }

    /// Check whether `key` is new, cached, or in-flight.
    ///
    /// Marks the key as in-flight if it is new.
    pub fn check_and_register(&self, key: &str) -> Result<DeduplicationResult, AgentRuntimeError> {
        let mut inner = timed_lock(&self.inner, "Deduplicator::check_and_register");

        let now = Instant::now();

        // Lazy expiry: full O(n) retain scan runs only every EXPIRY_INTERVAL
        // calls, amortising the cost. Per-key inline checks below keep
        // correctness between scans.
        const EXPIRY_INTERVAL: u64 = 64;
        inner.call_count = inner.call_count.wrapping_add(1);
        if inner.call_count % EXPIRY_INTERVAL == 0 {
            let ttl = self.ttl;
            inner.cache.retain(|_, (_, ts)| now.duration_since(*ts) < ttl);
            inner
                .in_flight
                .retain(|_, ts| now.duration_since(*ts) < ttl);
        }

        // Inline expiry check for this specific key.
        match inner.cache.get(key) {
            Some((result, ts)) if now.duration_since(*ts) < self.ttl => {
                return Ok(DeduplicationResult::Cached(result.clone()));
            }
            Some(_) => {
                inner.cache.remove(key); // entry is expired
            }
            None => {}
        }
        match inner.in_flight.get(key) {
            Some(ts) if now.duration_since(*ts) < self.ttl => {
                return Ok(DeduplicationResult::InProgress);
            }
            Some(_) => {
                inner.in_flight.remove(key); // in-flight entry is expired
            }
            None => {}
        }

        inner.in_flight.insert(key.to_owned(), now);
        Ok(DeduplicationResult::New)
    }

    /// Check deduplication state for a key with a per-call TTL override.
    ///
    /// Marks the key as in-flight if it is new. Ignores the stored TTL and uses
    /// `ttl` instead for expiry checks.
    pub fn check(&self, key: &str, ttl: std::time::Duration) -> Result<DeduplicationResult, AgentRuntimeError> {
        let mut inner = timed_lock(&self.inner, "Deduplicator::check");
        let now = Instant::now();

        // Lazy expiry: full scan every EXPIRY_INTERVAL calls.
        const EXPIRY_INTERVAL: u64 = 64;
        inner.call_count = inner.call_count.wrapping_add(1);
        if inner.call_count % EXPIRY_INTERVAL == 0 {
            inner.cache.retain(|_, (_, ts)| now.duration_since(*ts) < ttl);
            inner.in_flight.retain(|_, ts| now.duration_since(*ts) < ttl);
        }

        match inner.cache.get(key) {
            Some((result, ts)) if now.duration_since(*ts) < ttl => {
                return Ok(DeduplicationResult::Cached(result.clone()));
            }
            Some(_) => {
                inner.cache.remove(key);
            }
            None => {}
        }
        match inner.in_flight.get(key) {
            Some(ts) if now.duration_since(*ts) < ttl => {
                return Ok(DeduplicationResult::InProgress);
            }
            Some(_) => {
                inner.in_flight.remove(key);
            }
            None => {}
        }

        inner.in_flight.insert(key.to_owned(), now);
        Ok(DeduplicationResult::New)
    }

    /// Check deduplication state for multiple keys at once.
    ///
    /// Returns results in the same order as `requests`.
    /// Each entry is `(key, ttl)` — same signature as `check`.
    ///
    /// Acquires the internal mutex **once** for the entire batch, avoiding the
    /// per-key lock overhead of calling `check` in a loop.
    pub fn dedup_many(
        &self,
        requests: &[(&str, std::time::Duration)],
    ) -> Result<Vec<DeduplicationResult>, AgentRuntimeError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let mut inner = timed_lock(&self.inner, "Deduplicator::dedup_many");
        let now = std::time::Instant::now();
        let mut results = Vec::with_capacity(requests.len());

        for &(key, ttl) in requests {
            // Expire stale entries using this request's TTL.
            inner.cache.retain(|_, (_, ts)| now.duration_since(*ts) < ttl);
            inner.in_flight.retain(|_, ts| now.duration_since(*ts) < ttl);

            let result = if let Some((cached_result, _)) = inner.cache.get(key) {
                DeduplicationResult::Cached(cached_result.clone())
            } else if inner.in_flight.contains_key(key) {
                DeduplicationResult::InProgress
            } else {
                inner.in_flight.insert(key.to_owned(), now);
                DeduplicationResult::New
            };
            results.push(result);
        }

        Ok(results)
    }

    /// Complete a request: move from in-flight to cached with the given result.
    ///
    /// If `max_entries` is configured and the cache is full, the oldest cached
    /// entry (by insertion time) is evicted before the new one is stored.
    pub fn complete(&self, key: &str, result: impl Into<String>) -> Result<(), AgentRuntimeError> {
        let mut inner = timed_lock(&self.inner, "Deduplicator::complete");
        inner.in_flight.remove(key);

        // Enforce max_entries cap: evict via insertion-ordered VecDeque (O(1) amortised).
        // Ghost entries (already expired by a prior `retain`) are skipped by looping.
        if let Some(max) = self.max_entries {
            while inner.cache.len() >= max {
                match inner.cache_order.pop_front() {
                    Some(oldest_key) => {
                        inner.cache.remove(&oldest_key);
                    }
                    None => break,
                }
            }
        }

        let owned_key = key.to_owned();
        inner.cache_order.push_back(owned_key.clone());
        inner.cache.insert(owned_key, (result.into(), Instant::now()));
        Ok(())
    }

    /// Remove a key from in-flight tracking without caching a result.
    ///
    /// Call this when an in-flight operation fails so that subsequent callers
    /// are not permanently blocked by a stuck `InProgress` entry for the full TTL.
    pub fn fail(&self, key: &str) -> Result<(), AgentRuntimeError> {
        let mut inner = timed_lock(&self.inner, "Deduplicator::fail");
        inner.in_flight.remove(key);
        Ok(())
    }

    /// Return the number of keys currently in-flight (not yet completed or failed).
    pub fn in_flight_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::in_flight_count");
        Ok(inner.in_flight.len())
    }

    /// Return a snapshot of all keys currently in-flight.
    pub fn in_flight_keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::in_flight_keys");
        Ok(inner.in_flight.keys().cloned().collect())
    }

    /// Return the number of keys currently in the completed result cache.
    ///
    /// Note: expired entries are only removed lazily on the next `check*` call.
    pub fn cached_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::cached_count");
        Ok(inner.cache.len())
    }

    /// Return a snapshot of all keys that have cached results.
    ///
    /// Expired entries are included (they are removed lazily).  Use
    /// [`purge_expired`] first for a clean list of live keys.
    ///
    /// [`purge_expired`]: Deduplicator::purge_expired
    pub fn cached_keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::cached_keys");
        Ok(inner.cache.keys().cloned().collect())
    }

    /// Return the configured time-to-live for cached results.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Return the configured maximum number of cached entries, if any.
    ///
    /// Returns `None` if no cap was set via [`with_max_entries`].
    ///
    /// [`with_max_entries`]: Deduplicator::with_max_entries
    pub fn max_entries(&self) -> Option<usize> {
        self.max_entries
    }

    /// Return `true` if there are no in-flight requests.
    pub fn is_idle(&self) -> Result<bool, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::is_idle");
        Ok(inner.in_flight.is_empty())
    }

    /// Return the total number of items tracked by the deduplicator
    /// (in-flight + cached results, regardless of TTL expiry).
    pub fn total_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::total_count");
        Ok(inner.in_flight.len() + inner.cache.len())
    }

    /// Return `true` if `key` is currently in-flight or has a cached result.
    ///
    /// Unlike [`check_and_register`] this is a read-only inspection — it does
    /// not register the key or consume a deduplication slot.
    ///
    /// [`check_and_register`]: Deduplicator::check_and_register
    pub fn contains(&self, key: &str) -> Result<bool, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::contains");
        Ok(inner.in_flight.contains_key(key) || inner.cache.contains_key(key))
    }

    /// Return the cached result for `key` if one exists and has not expired.
    ///
    /// Returns `None` when the key is not in the cache (either not yet
    /// completed or already expired).  Does not modify any state.
    pub fn get_result(&self, key: &str) -> Result<Option<String>, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::get_result");
        let ttl = self.ttl;
        let now = std::time::Instant::now();
        Ok(inner.cache.get(key).and_then(|(result, inserted_at)| {
            if now.duration_since(*inserted_at) <= ttl {
                Some(result.clone())
            } else {
                None
            }
        }))
    }

    /// Remove all in-flight entries and cached results.
    ///
    /// Useful for test teardown or hard resets.
    pub fn clear(&self) -> Result<(), AgentRuntimeError> {
        let mut inner = timed_lock(&self.inner, "Deduplicator::clear");
        inner.cache.clear();
        inner.in_flight.clear();
        inner.cache_order.clear();
        Ok(())
    }

    /// Eagerly evict all cache entries whose TTL has elapsed.
    ///
    /// Under normal operation expired entries are removed lazily on the next
    /// `check*` call.  Call `purge_expired` for deterministic memory reclamation
    /// (e.g. before a `cached_count` snapshot or in a maintenance loop).
    ///
    /// Returns the number of entries that were removed.
    pub fn purge_expired(&self) -> Result<usize, AgentRuntimeError> {
        let mut inner = timed_lock(&self.inner, "Deduplicator::purge_expired");
        let ttl = self.ttl;
        let now = std::time::Instant::now();
        let before = inner.cache.len();
        inner.cache.retain(|_, (_, inserted_at)| {
            now.duration_since(*inserted_at) <= ttl
        });
        let removed = before - inner.cache.len();
        // Rebuild cache_order to drop ghost entries (keys purged from cache but
        // still referenced in the VecDeque).
        if removed > 0 {
            let live_keys: std::collections::HashSet<String> =
                inner.cache.keys().cloned().collect();
            inner.cache_order.retain(|k| live_keys.contains(k));
        }
        Ok(removed)
    }

    /// Remove the oldest cached result entry (FIFO order).
    ///
    /// Returns `true` if an entry was removed, `false` if the cache was empty.
    pub fn evict_oldest(&self) -> Result<bool, AgentRuntimeError> {
        let mut inner = timed_lock(&self.inner, "Deduplicator::evict_oldest");
        while let Some(key) = inner.cache_order.pop_front() {
            if inner.cache.remove(&key).is_some() {
                return Ok(true);
            }
        }
        Ok(false)
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

    /// Reset the current depth to zero.
    ///
    /// Useful in tests or after a controlled shutdown when all in-flight
    /// requests have been cancelled and the guard should start fresh.
    pub fn reset(&self) {
        let mut depth = timed_lock(&self.inner, "BackpressureGuard::reset");
        *depth = 0;
    }

    /// Return `true` if the guard is at or over its hard capacity.
    pub fn is_full(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.depth()? >= self.capacity)
    }

    /// Return `true` if no slots are currently in use.
    pub fn is_empty(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.depth()? == 0)
    }

    /// Return the number of additional request slots available before the hard cap.
    pub fn available_capacity(&self) -> Result<usize, AgentRuntimeError> {
        Ok(self.capacity.saturating_sub(self.depth()?))
    }

    /// Return the hard capacity (maximum concurrent slots) configured for this guard.
    pub fn hard_capacity(&self) -> usize {
        self.capacity
    }

    /// Return the soft capacity limit if one was configured, or `None`.
    pub fn soft_limit(&self) -> Option<usize> {
        self.soft_capacity
    }

    /// Return `true` if a soft capacity limit has been configured.
    ///
    /// Equivalent to `self.soft_limit().is_some()` but more readable at call
    /// sites that only need a boolean check.
    pub fn is_soft_limited(&self) -> bool {
        self.soft_capacity.is_some()
    }

    /// Return the current depth.
    pub fn depth(&self) -> Result<usize, AgentRuntimeError> {
        let depth = timed_lock(&self.inner, "BackpressureGuard::depth");
        Ok(*depth)
    }

    /// Return the current depth as a percentage of the hard capacity.
    ///
    /// Returns a value in `[0.0, 100.0]`.  When `depth > capacity` (which
    /// cannot happen in normal operation) the result is clamped to `100.0`.
    pub fn percent_full(&self) -> Result<f64, AgentRuntimeError> {
        let depth = self.depth()?;
        Ok((depth as f64 / self.capacity as f64 * 100.0).min(100.0))
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

    /// Return the fraction of the hard capacity currently in use: `depth / capacity`.
    ///
    /// Returns `0.0` when no slots are in use, `1.0` when fully saturated.
    pub fn utilization_ratio(&self) -> Result<f32, AgentRuntimeError> {
        if self.capacity == 0 {
            return Ok(0.0);
        }
        let depth = self.depth()?;
        Ok(depth as f32 / self.capacity as f32)
    }

    /// Return the number of additional slots that can be acquired before hitting
    /// the hard capacity limit.
    ///
    /// Returns `0` when the guard is full.
    pub fn remaining_capacity(&self) -> Result<usize, AgentRuntimeError> {
        let depth = self.depth()?;
        Ok(self.capacity.saturating_sub(depth))
    }

    /// Force the in-flight depth counter to zero.
    ///
    /// Useful for test teardown or hard resets where acquired slots will never
    /// be released normally (e.g., after a test panics before calling `release`).
    pub fn reset_depth(&self) -> Result<(), AgentRuntimeError> {
        let mut depth = timed_lock(&self.inner, "BackpressureGuard::reset_depth");
        *depth = 0;
        Ok(())
    }

    /// Return the fraction of capacity that is still available, in `[0.0, 1.0]`.
    ///
    /// `1.0` means completely empty; `0.0` means at full capacity.
    pub fn headroom_ratio(&self) -> Result<f64, AgentRuntimeError> {
        Ok(self.available_capacity()? as f64 / self.capacity as f64)
    }

    /// Return the number of currently held (acquired) slots.
    ///
    /// Equivalent to `capacity - available_capacity()`.
    pub fn acquired_count(&self) -> Result<usize, AgentRuntimeError> {
        Ok(self.capacity - self.available_capacity()?)
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Result of executing a pipeline, including per-stage timing.
#[derive(Debug)]
pub struct PipelineResult {
    /// Final output value after all stages.
    pub output: String,
    /// Per-stage timing: `(stage_name, duration_ms)` in execution order.
    pub stage_timings: Vec<(String, u64)>,
}

/// A single named stage in the pipeline.
pub struct Stage {
    /// Human-readable name used in log output and error messages.
    pub name: String,
    /// The transform function; receives the current string and returns the transformed string.
    pub handler: Box<dyn Fn(String) -> Result<String, AgentRuntimeError> + Send + Sync>,
}

impl std::fmt::Debug for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage").field("name", &self.name).finish()
    }
}

/// Error handler callback type for pipeline stage failures.
type StageErrorHandler = Box<dyn Fn(&str, &str) -> String + Send + Sync>;

/// A composable pipeline that passes a string through a sequence of named stages.
///
/// ## Guarantees
/// - Stages execute in insertion order
/// - First stage failure short-circuits remaining stages (unless an error handler is set)
/// - Non-panicking
pub struct Pipeline {
    stages: Vec<Stage>,
    error_handler: Option<StageErrorHandler>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("stages", &self.stages)
            .field("has_error_handler", &self.error_handler.is_some())
            .finish()
    }
}

impl Pipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new(), error_handler: None }
    }

    /// Attach a recovery callback for stage failures.
    ///
    /// When a stage fails, `handler(stage_name, error_message)` is called.
    /// The returned string becomes the input to the next stage.
    /// If no handler is set, stage failures propagate as errors.
    pub fn with_error_handler(
        mut self,
        handler: impl Fn(&str, &str) -> String + Send + Sync + 'static,
    ) -> Self {
        self.error_handler = Some(Box::new(handler));
        self
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

    /// Insert a stage at the **front** of the pipeline (index 0).
    ///
    /// All existing stages are shifted to higher indices.  The pipeline's
    /// stage names remain unique only if the caller ensures uniqueness.
    pub fn prepend_stage(
        mut self,
        name: impl Into<String>,
        handler: impl Fn(String) -> Result<String, AgentRuntimeError> + Send + Sync + 'static,
    ) -> Self {
        self.stages.insert(0, Stage {
            name: name.into(),
            handler: Box::new(handler),
        });
        self
    }

    /// Return `true` if the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Return `true` if a stage error handler has been configured via
    /// [`with_error_handler`].
    ///
    /// [`with_error_handler`]: Pipeline::with_error_handler
    pub fn has_error_handler(&self) -> bool {
        self.error_handler.is_some()
    }

    /// Return the number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Return `true` if a stage with the given name is registered.
    pub fn has_stage(&self, name: &str) -> bool {
        self.stages.iter().any(|s| s.name == name)
    }

    /// Return the names of all stages in execution order.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name.as_str()).collect()
    }

    /// Return the names of all stages as owned `String`s.
    ///
    /// Unlike [`stage_names`] this does not borrow `self`, making it easier to
    /// use the result after `self` is moved or mutated.
    ///
    /// [`stage_names`]: Pipeline::stage_names
    pub fn stage_names_owned(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.name.clone()).collect()
    }

    /// Return the name of the stage at zero-based `index`, or `None` if out of bounds.
    pub fn get_stage_name_at(&self, index: usize) -> Option<&str> {
        self.stages.get(index).map(|s| s.name.as_str())
    }

    /// Return the zero-based index of the first stage with the given name.
    ///
    /// Returns `None` if no stage with that name exists.
    pub fn stage_index(&self, name: &str) -> Option<usize> {
        self.stages.iter().position(|s| s.name == name)
    }

    /// Return the name of the first stage in the pipeline, or `None` if empty.
    pub fn first_stage_name(&self) -> Option<&str> {
        self.stages.first().map(|s| s.name.as_str())
    }

    /// Return the name of the last stage in the pipeline, or `None` if empty.
    pub fn last_stage_name(&self) -> Option<&str> {
        self.stages.last().map(|s| s.name.as_str())
    }

    /// Remove the first stage whose name equals `name`.
    ///
    /// Returns `true` if a stage was found and removed, `false` if no stage
    /// with that name was registered.
    pub fn remove_stage(&mut self, name: &str) -> bool {
        if let Some(pos) = self.stages.iter().position(|s| s.name == name) {
            self.stages.remove(pos);
            true
        } else {
            false
        }
    }

    /// Rename the first stage whose name equals `old_name` to `new_name`.
    ///
    /// Returns `true` if a stage was found and renamed, `false` if no stage
    /// with `old_name` exists.
    pub fn rename_stage(&mut self, old_name: &str, new_name: impl Into<String>) -> bool {
        if let Some(stage) = self.stages.iter_mut().find(|s| s.name == old_name) {
            stage.name = new_name.into();
            true
        } else {
            false
        }
    }

    /// Remove all stages from the pipeline.
    ///
    /// The error handler (if any) is preserved; only the stage list is cleared.
    pub fn clear(&mut self) {
        self.stages.clear();
    }

    /// Swap the positions of two stages by name.
    ///
    /// Returns `true` if both stages were found and swapped.  Returns `false`
    /// if either name is not present in the pipeline (no state change).
    pub fn swap_stages(&mut self, a: &str, b: &str) -> bool {
        let idx_a = self.stages.iter().position(|s| s.name == a);
        let idx_b = self.stages.iter().position(|s| s.name == b);
        match (idx_a, idx_b) {
            (Some(i), Some(j)) => {
                self.stages.swap(i, j);
                true
            }
            _ => false,
        }
    }

    /// Execute the pipeline, passing `input` through each stage in order.
    #[tracing::instrument(skip(self))]
    pub fn run(&self, input: String) -> Result<String, AgentRuntimeError> {
        let mut current = input;
        for stage in &self.stages {
            tracing::debug!(stage = %stage.name, "running pipeline stage");
            match (stage.handler)(current) {
                Ok(out) => current = out,
                Err(e) => {
                    tracing::error!(stage = %stage.name, error = %e, "pipeline stage failed");
                    if let Some(ref handler) = self.error_handler {
                        current = handler(&stage.name, &e.to_string());
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Ok(current)
    }

    /// Execute the pipeline with per-stage timing.
    ///
    /// Returns a [`PipelineResult`] whose `stage_timings` contains
    /// `(stage_name, duration_ms)` pairs in execution order.
    pub fn execute_timed(&self, input: String) -> Result<PipelineResult, AgentRuntimeError> {
        let mut current = input;
        let mut stage_timings = Vec::new();
        for stage in &self.stages {
            let start = std::time::Instant::now();
            tracing::debug!(stage = %stage.name, "running timed pipeline stage");
            match (stage.handler)(current) {
                Ok(out) => current = out,
                Err(e) => {
                    tracing::error!(stage = %stage.name, error = %e, "timed pipeline stage failed");
                    if let Some(ref handler) = self.error_handler {
                        current = handler(&stage.name, &e.to_string());
                    } else {
                        return Err(e);
                    }
                }
            }
            let duration_ms = start.elapsed().as_millis() as u64;
            stage_timings.push((stage.name.clone(), duration_ms));
        }
        Ok(PipelineResult {
            output: current,
            stage_timings,
        })
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
        use super::CircuitBreakerBackend as CB;
        let backend = InMemoryCircuitBreakerBackend::new();

        assert_eq!(CB::get_failures(&backend, "svc"), 0);

        let count = CB::increment_failures(&backend, "svc");
        assert_eq!(count, 1);

        let count = CB::increment_failures(&backend, "svc");
        assert_eq!(count, 2);

        CB::reset_failures(&backend, "svc");
        assert_eq!(CB::get_failures(&backend, "svc"), 0);

        // open_at round-trip
        assert!(CB::get_open_at(&backend, "svc").is_none());
        let now = Instant::now();
        CB::set_open_at(&backend, "svc", now);
        assert!(CB::get_open_at(&backend, "svc").is_some());
        CB::clear_open_at(&backend, "svc");
        assert!(CB::get_open_at(&backend, "svc").is_none());
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

    #[test]
    fn test_pipeline_execute_timed_captures_stage_durations() {
        let p = Pipeline::new()
            .add_stage("s1", |s| Ok(format!("{s}1")))
            .add_stage("s2", |s| Ok(format!("{s}2")));
        let result = p.execute_timed("x".to_string()).unwrap();
        assert_eq!(result.output, "x12");
        assert_eq!(result.stage_timings.len(), 2);
        assert_eq!(result.stage_timings[0].0, "s1");
        assert_eq!(result.stage_timings[1].0, "s2");
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

    // ── #4/#31 BackpressureGuard::hard_capacity ───────────────────────────────

    #[test]
    fn test_backpressure_hard_capacity_matches_new() {
        let g = BackpressureGuard::new(7).unwrap();
        assert_eq!(g.hard_capacity(), 7);
    }

    // ── #10 Pipeline::with_error_handler ──────────────────────────────────────

    #[test]
    fn test_pipeline_error_handler_recovers_from_stage_failure() {
        let p = Pipeline::new()
            .add_stage("fail_stage", |_| {
                Err(AgentRuntimeError::Orchestration("oops".into()))
            })
            .add_stage("append", |s| Ok(format!("{s}-recovered")))
            .with_error_handler(|stage_name, _err| format!("recovered_from_{stage_name}"));
        let result = p.run("input".to_string()).unwrap();
        assert_eq!(result, "recovered_from_fail_stage-recovered");
    }

    // ── #11/#32 CircuitState PartialEq/Eq ────────────────────────────────────

    #[test]
    fn test_circuit_state_eq() {
        assert_eq!(CircuitState::Closed, CircuitState::Closed);
        assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
        assert_eq!(
            CircuitState::Open { opened_at: std::time::Instant::now() },
            CircuitState::Open { opened_at: std::time::Instant::now() }
        );
        assert_ne!(CircuitState::Closed, CircuitState::HalfOpen);
        assert_ne!(CircuitState::Closed, CircuitState::Open { opened_at: std::time::Instant::now() });
    }

    // ── #18 Deduplicator::dedup_many ──────────────────────────────────────────

    #[test]
    fn test_dedup_many_independent_keys() {
        let d = Deduplicator::new(Duration::from_secs(60));
        let ttl = Duration::from_secs(60);
        let results = d.dedup_many(&[("key-a", ttl), ("key-b", ttl), ("key-c", ttl)]).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| matches!(r, DeduplicationResult::New)));
    }

    // ── Task 11: Concurrent CircuitBreaker state transition tests ─────────────

    #[test]
    fn test_concurrent_circuit_breaker_opens_under_concurrent_failures() {
        use std::sync::Arc;
        use std::thread;

        let cb = Arc::new(
            CircuitBreaker::new("svc", 5, Duration::from_secs(60)).unwrap(),
        );
        let n_threads = 8;
        let failures_per_thread = 2;

        let mut handles = Vec::new();
        for _ in 0..n_threads {
            let cb = Arc::clone(&cb);
            handles.push(thread::spawn(move || {
                for _ in 0..failures_per_thread {
                    let _ = cb.call(|| Err::<(), &str>("fail"));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // After n_threads * failures_per_thread = 16 failures with threshold=5,
        // the circuit must be Open.
        let state = cb.state().unwrap();
        assert!(
            matches!(state, CircuitState::Open { .. }),
            "circuit should be open after many concurrent failures; got: {state:?}"
        );
    }

    #[test]
    fn test_per_service_tracking_is_independent() {
        let backend = Arc::new(InMemoryCircuitBreakerBackend::new());

        let cb_a = CircuitBreaker::new("service-a", 3, Duration::from_secs(60))
            .unwrap()
            .with_backend(Arc::clone(&backend) as Arc<dyn CircuitBreakerBackend>);
        let cb_b = CircuitBreaker::new("service-b", 3, Duration::from_secs(60))
            .unwrap()
            .with_backend(Arc::clone(&backend) as Arc<dyn CircuitBreakerBackend>);

        // Fail service-a 3 times → opens
        for _ in 0..3 {
            let _ = cb_a.call(|| Err::<(), &str>("fail"));
        }

        // service-b should still be Closed
        let state_b = cb_b.state().unwrap();
        assert_eq!(
            state_b,
            CircuitState::Closed,
            "service-b should be unaffected by service-a failures"
        );

        // service-a should be Open
        let state_a = cb_a.state().unwrap();
        assert!(
            matches!(state_a, CircuitState::Open { .. }),
            "service-a should be open"
        );
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

    // ── New API tests (Rounds 4-8) ────────────────────────────────────────────

    #[test]
    fn test_retry_policy_constant_has_fixed_delay() {
        let p = RetryPolicy::constant(3, 100).unwrap();
        assert_eq!(p.delay_for(1), Duration::from_millis(100));
        assert_eq!(p.delay_for(2), Duration::from_millis(100));
        assert_eq!(p.delay_for(10), Duration::from_millis(100));
    }

    #[test]
    fn test_retry_policy_exponential_doubles() {
        let p = RetryPolicy::exponential(5, 10).unwrap();
        assert_eq!(p.delay_for(1), Duration::from_millis(10));
        assert_eq!(p.delay_for(2), Duration::from_millis(20));
        assert_eq!(p.delay_for(3), Duration::from_millis(40));
    }

    #[test]
    fn test_retry_policy_with_max_attempts() {
        let p = RetryPolicy::constant(3, 50).unwrap();
        let p2 = p.with_max_attempts(7).unwrap();
        assert_eq!(p2.max_attempts, 7);
        assert!(RetryPolicy::constant(1, 50).unwrap().with_max_attempts(0).is_err());
    }

    #[test]
    fn test_circuit_breaker_reset_returns_to_closed() {
        let cb = CircuitBreaker::new("svc", 2, Duration::from_secs(60)).unwrap();
        cb.record_failure();
        cb.record_failure(); // should open
        assert_ne!(cb.state().unwrap(), CircuitState::Closed);
        cb.reset();
        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
        assert_eq!(cb.failure_count().unwrap(), 0);
    }

    #[test]
    fn test_deduplicator_clear_resets_all_state() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("k1").unwrap();
        d.check_and_register("k2").unwrap();
        d.complete("k1", "r1").unwrap();
        assert_eq!(d.in_flight_count().unwrap(), 1);
        assert_eq!(d.cached_count().unwrap(), 1);
        d.clear().unwrap();
        assert_eq!(d.in_flight_count().unwrap(), 0);
        assert_eq!(d.cached_count().unwrap(), 0);
    }

    #[test]
    fn test_deduplicator_purge_expired_removes_stale() {
        let d = Deduplicator::new(Duration::from_millis(1));
        d.check_and_register("x").unwrap();
        d.complete("x", "result").unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let removed = d.purge_expired().unwrap();
        assert_eq!(removed, 1);
        assert_eq!(d.cached_count().unwrap(), 0);
    }

    #[test]
    fn test_backpressure_utilization_ratio() {
        let g = BackpressureGuard::new(4).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        let ratio = g.utilization_ratio().unwrap();
        assert!((ratio - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_pipeline_stage_count_and_names() {
        let p = Pipeline::new()
            .add_stage("first", |s| Ok(s + "1"))
            .add_stage("second", |s| Ok(s + "2"));
        assert_eq!(p.stage_count(), 2);
        assert_eq!(p.stage_names(), vec!["first", "second"]);
    }

    #[test]
    fn test_pipeline_is_empty_true_for_new() {
        let p = Pipeline::new();
        assert!(p.is_empty());
    }

    #[test]
    fn test_pipeline_is_empty_false_after_add_stage() {
        let p = Pipeline::new().add_stage("s", |s: String| Ok(s));
        assert!(!p.is_empty());
    }

    #[test]
    fn test_circuit_breaker_service_name() {
        let cb = CircuitBreaker::new("my-service", 3, Duration::from_secs(1)).unwrap();
        assert_eq!(cb.service_name(), "my-service");
    }

    #[test]
    fn test_retry_policy_none_has_max_one_attempt() {
        let p = RetryPolicy::none();
        assert_eq!(p.max_attempts, 1);
        assert_eq!(p.delay_for(0), Duration::ZERO);
    }

    #[test]
    fn test_backpressure_is_full_false_when_empty() {
        let g = BackpressureGuard::new(5).unwrap();
        assert!(!g.is_full().unwrap());
    }

    #[test]
    fn test_backpressure_is_full_true_when_at_capacity() {
        let g = BackpressureGuard::new(2).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        assert!(g.is_full().unwrap());
    }

    #[test]
    fn test_deduplicator_ttl_returns_configured_value() {
        let d = Deduplicator::new(Duration::from_secs(42));
        assert_eq!(d.ttl(), Duration::from_secs(42));
    }

    #[test]
    fn test_circuit_breaker_is_closed_initially() {
        let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(1)).unwrap();
        assert!(cb.is_closed());
        assert!(!cb.is_open());
        assert!(!cb.is_half_open());
    }

    #[test]
    fn test_circuit_breaker_is_open_after_threshold_failures() {
        let cb = CircuitBreaker::new("svc", 2, Duration::from_secs(60)).unwrap();
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());
        assert!(!cb.is_closed());
    }

    #[test]
    fn test_retry_policy_total_max_delay_constant() {
        // constant 100ms × 3 attempts = 300ms total
        let p = RetryPolicy::constant(3, 100).unwrap();
        assert_eq!(p.total_max_delay_ms(), 300);
    }

    #[test]
    fn test_retry_policy_total_max_delay_none_is_zero() {
        let p = RetryPolicy::none();
        assert_eq!(p.total_max_delay_ms(), 0);
    }

    #[test]
    fn test_retry_policy_is_none_true_for_none() {
        let p = RetryPolicy::none();
        assert!(p.is_none());
    }

    #[test]
    fn test_retry_policy_is_none_false_for_exponential() {
        let p = RetryPolicy::exponential(3, 10).unwrap();
        assert!(!p.is_none());
    }

    #[test]
    fn test_pipeline_has_error_handler_false_by_default() {
        let p = Pipeline::new().add_stage("s", |s: String| Ok(s));
        assert!(!p.has_error_handler());
    }

    #[test]
    fn test_pipeline_has_error_handler_true_after_set() {
        let p = Pipeline::new()
            .with_error_handler(|_stage, _err| "recovered".to_string());
        assert!(p.has_error_handler());
    }

    #[test]
    fn test_backpressure_reset_clears_depth() {
        let g = BackpressureGuard::new(5).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        assert_eq!(g.depth().unwrap(), 2);
        g.reset();
        assert_eq!(g.depth().unwrap(), 0);
    }

    #[test]
    fn test_deduplicator_in_flight_keys_returns_started_keys() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check("key-a", Duration::from_secs(60)).unwrap();
        d.check("key-b", Duration::from_secs(60)).unwrap();
        let mut keys = d.in_flight_keys().unwrap();
        keys.sort();
        assert_eq!(keys, vec!["key-a", "key-b"]);
    }

    // ── Round 3: new methods ──────────────────────────────────────────────────

    #[test]
    fn test_retry_policy_with_base_delay_ms_changes_delay() {
        let p = RetryPolicy::exponential(3, 100)
            .unwrap()
            .with_base_delay_ms(200)
            .unwrap();
        assert_eq!(p.delay_for(1), Duration::from_millis(200));
    }

    #[test]
    fn test_retry_policy_with_base_delay_ms_rejects_zero() {
        let p = RetryPolicy::exponential(3, 100).unwrap();
        assert!(p.with_base_delay_ms(0).is_err());
    }

    #[test]
    fn test_backpressure_reset_depth_clears_counter() {
        let guard = BackpressureGuard::new(5).unwrap();
        guard.try_acquire().unwrap();
        guard.try_acquire().unwrap();
        assert_eq!(guard.depth().unwrap(), 2);
        guard.reset_depth().unwrap();
        assert_eq!(guard.depth().unwrap(), 0);
    }

    #[test]
    fn test_pipeline_remove_stage_returns_true_if_found() {
        let mut p = Pipeline::new()
            .add_stage("a", |s| Ok(s))
            .add_stage("b", |s| Ok(s));
        assert!(p.remove_stage("a"));
        assert_eq!(p.stage_count(), 1);
        assert_eq!(p.stage_names(), vec!["b"]);
    }

    #[test]
    fn test_pipeline_remove_stage_returns_false_if_missing() {
        let mut p = Pipeline::new().add_stage("x", |s| Ok(s));
        assert!(!p.remove_stage("nope"));
        assert_eq!(p.stage_count(), 1);
    }

    #[test]
    fn test_pipeline_clear_removes_all_stages() {
        let mut p = Pipeline::new()
            .add_stage("a", |s| Ok(s))
            .add_stage("b", |s| Ok(s));
        p.clear();
        assert!(p.is_empty());
    }

    // ── Round 4: CircuitBreaker accessors / Pipeline::get_stage_name_at ──────

    #[test]
    fn test_circuit_breaker_threshold_accessor() {
        let cb = CircuitBreaker::new("svc", 5, Duration::from_secs(30)).unwrap();
        assert_eq!(cb.threshold(), 5);
    }

    #[test]
    fn test_circuit_breaker_recovery_window_accessor() {
        let window = Duration::from_secs(45);
        let cb = CircuitBreaker::new("svc", 3, window).unwrap();
        assert_eq!(cb.recovery_window(), window);
    }

    #[test]
    fn test_pipeline_get_stage_name_at_returns_correct_names() {
        let p = Pipeline::new()
            .add_stage("first", |s| Ok(s))
            .add_stage("second", |s| Ok(s));
        assert_eq!(p.get_stage_name_at(0), Some("first"));
        assert_eq!(p.get_stage_name_at(1), Some("second"));
        assert_eq!(p.get_stage_name_at(2), None);
    }

    // ── Round 16: can_retry ───────────────────────────────────────────────────

    #[test]
    fn test_retry_policy_can_retry_within_budget() {
        let p = RetryPolicy::exponential(3, 100).unwrap();
        assert!(p.can_retry(0));
        assert!(p.can_retry(1));
        assert!(p.can_retry(2));
    }

    #[test]
    fn test_retry_policy_can_retry_false_when_exhausted() {
        let p = RetryPolicy::exponential(3, 100).unwrap();
        assert!(!p.can_retry(3));
        assert!(!p.can_retry(99));
    }

    #[test]
    fn test_retry_policy_none_only_allows_first_attempt() {
        let p = RetryPolicy::none();
        assert!(p.can_retry(0));
        assert!(!p.can_retry(1));
    }

    // ── Round 5: RetryPolicy::max_attempts / Pipeline::stage_names_owned ─────

    #[test]
    fn test_retry_policy_max_attempts_accessor() {
        let p = RetryPolicy::exponential(7, 100).unwrap();
        assert_eq!(p.max_attempts(), 7);
    }

    #[test]
    fn test_pipeline_stage_names_owned_returns_strings() {
        let p = Pipeline::new()
            .add_stage("alpha", |s| Ok(s))
            .add_stage("beta", |s| Ok(s));
        let owned = p.stage_names_owned();
        assert_eq!(owned, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn test_pipeline_stage_names_owned_empty_when_no_stages() {
        let p = Pipeline::new();
        assert!(p.stage_names_owned().is_empty());
    }

    // ── Round 17: attempts_remaining ─────────────────────────────────────────

    #[test]
    fn test_attempts_remaining_full_at_zero() {
        let p = RetryPolicy::exponential(4, 100).unwrap();
        assert_eq!(p.attempts_remaining(0), 4);
    }

    #[test]
    fn test_attempts_remaining_decrements_correctly() {
        let p = RetryPolicy::exponential(4, 100).unwrap();
        assert_eq!(p.attempts_remaining(2), 2);
        assert_eq!(p.attempts_remaining(4), 0);
    }

    #[test]
    fn test_attempts_remaining_zero_when_exhausted() {
        let p = RetryPolicy::exponential(3, 100).unwrap();
        assert_eq!(p.attempts_remaining(10), 0);
    }

    // ── Round 18: untested circuit-breaker, deduplicator, backpressure methods

    #[test]
    fn test_retry_policy_max_attempts_getter() {
        let p = RetryPolicy::exponential(7, 50).unwrap();
        assert_eq!(p.max_attempts(), 7);
    }

    #[test]
    fn test_circuit_breaker_failure_count_increments() {
        let cb = CircuitBreaker::new("svc2", 3, std::time::Duration::from_secs(60)).unwrap();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count().unwrap(), 2);
    }

    #[test]
    fn test_circuit_breaker_record_success_resets_failures() {
        let cb = CircuitBreaker::new("svc3", 5, std::time::Duration::from_secs(60)).unwrap();
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.failure_count().unwrap(), 0);
        assert!(cb.is_closed());
    }

    #[test]
    fn test_circuit_breaker_threshold_and_recovery_window() {
        let cb = CircuitBreaker::new("svc4", 3, std::time::Duration::from_secs(30)).unwrap();
        assert_eq!(cb.threshold(), 3);
        assert_eq!(cb.recovery_window(), std::time::Duration::from_secs(30));
    }

    #[test]
    fn test_circuit_breaker_reset_clears_state() {
        let cb = CircuitBreaker::new("svc5", 2, std::time::Duration::from_secs(60)).unwrap();
        cb.record_failure();
        cb.record_failure(); // should open circuit
        assert!(cb.is_open());
        cb.reset();
        assert!(cb.is_closed());
        assert_eq!(cb.failure_count().unwrap(), 0);
    }

    #[test]
    fn test_deduplicator_cached_count_after_complete() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check("key1", Duration::from_secs(60)).unwrap();
        d.complete("key1", "result").unwrap();
        assert_eq!(d.cached_count().unwrap(), 1);
    }

    #[test]
    fn test_deduplicator_ttl_matches_configured() {
        let d = Deduplicator::new(Duration::from_secs(42));
        assert_eq!(d.ttl(), Duration::from_secs(42));
    }

    #[test]
    fn test_deduplicator_purge_expired_removes_stale_entries() {
        let d = Deduplicator::new(Duration::ZERO); // instant TTL
        d.check("stale", Duration::ZERO).unwrap();
        d.complete("stale", "val").unwrap();
        // Sleep briefly to ensure the entry expires
        std::thread::sleep(std::time::Duration::from_millis(1));
        let removed = d.purge_expired().unwrap();
        assert!(removed >= 1);
    }

    #[test]
    fn test_backpressure_remaining_capacity() {
        let g = BackpressureGuard::new(5).unwrap();
        g.try_acquire().unwrap();
        assert_eq!(g.remaining_capacity().unwrap(), 4);
    }

    #[test]
    fn test_backpressure_soft_depth_ratio_without_soft_limit() {
        let g = BackpressureGuard::new(5).unwrap();
        assert_eq!(g.soft_depth_ratio(), 0.0);
    }

    #[test]
    fn test_backpressure_soft_depth_ratio_with_soft_limit() {
        let g = BackpressureGuard::new(10).unwrap()
            .with_soft_limit(4).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        let ratio = g.soft_depth_ratio();
        assert!((ratio - 0.5).abs() < 1e-6);
    }

    // ── Round 7: delay_ms_for / soft_limit / has_stage ───────────────────────

    #[test]
    fn test_retry_delay_ms_for_matches_delay_for() {
        let p = RetryPolicy::exponential(5, 100).unwrap();
        assert_eq!(p.delay_ms_for(1), p.delay_for(1).as_millis() as u64);
        assert_eq!(p.delay_ms_for(3), p.delay_for(3).as_millis() as u64);
    }

    #[test]
    fn test_backpressure_soft_limit_returns_configured_value() {
        let g = BackpressureGuard::new(10).unwrap()
            .with_soft_limit(5).unwrap();
        assert_eq!(g.soft_limit(), Some(5));
    }

    #[test]
    fn test_backpressure_soft_limit_none_when_not_set() {
        let g = BackpressureGuard::new(10).unwrap();
        assert_eq!(g.soft_limit(), None);
    }

    #[test]
    fn test_pipeline_has_stage_returns_true_when_present() {
        let p = Pipeline::new().add_stage("step1", |s| Ok(s));
        assert!(p.has_stage("step1"));
        assert!(!p.has_stage("step2"));
    }

    #[test]
    fn test_pipeline_has_stage_false_for_empty_pipeline() {
        let p = Pipeline::new();
        assert!(!p.has_stage("anything"));
    }

    // ── Round 20: Deduplicator::max_entries / RetryPolicy::delay_for ─────────

    #[test]
    fn test_deduplicator_max_entries_none_by_default() {
        let d = Deduplicator::new(Duration::from_secs(60));
        assert_eq!(d.max_entries(), None);
    }

    #[test]
    fn test_deduplicator_max_entries_set_via_builder() {
        let d = Deduplicator::new(Duration::from_secs(60))
            .with_max_entries(50)
            .unwrap();
        assert_eq!(d.max_entries(), Some(50));
    }

    #[test]
    fn test_retry_policy_delay_for_exponential_grows() {
        let p = RetryPolicy::exponential(5, 100).unwrap();
        // delay_for uses saturating_sub(1): attempt 1 => multiplier 1, attempt 2 => multiplier 2
        let d1 = p.delay_for(1);
        let d2 = p.delay_for(2);
        assert!(d2 > d1, "exponential delay should grow: attempt 2 > attempt 1");
    }

    #[test]
    fn test_retry_policy_delay_for_constant_stays_same() {
        let p = RetryPolicy::constant(5, 200).unwrap();
        assert_eq!(p.delay_for(0), p.delay_for(1));
        assert_eq!(p.delay_for(1), p.delay_for(3));
    }

    // ── Round 9: is_no_retry ──────────────────────────────────────────────────

    #[test]
    fn test_is_no_retry_true_for_none_policy() {
        let p = RetryPolicy::none();
        assert!(p.is_no_retry());
    }

    #[test]
    fn test_is_no_retry_false_for_exponential_policy() {
        let p = RetryPolicy::exponential(3, 50).unwrap();
        assert!(!p.is_no_retry());
    }

    #[test]
    fn test_is_no_retry_false_for_constant_policy_with_multiple_attempts() {
        let p = RetryPolicy::constant(2, 100).unwrap();
        assert!(!p.is_no_retry());
    }

    // ── Round 10: is_exponential ──────────────────────────────────────────────

    #[test]
    fn test_is_exponential_true_for_exponential_policy() {
        let p = RetryPolicy::exponential(3, 50).unwrap();
        assert!(p.is_exponential());
    }

    #[test]
    fn test_is_exponential_false_for_constant_policy() {
        let p = RetryPolicy::constant(3, 50).unwrap();
        assert!(!p.is_exponential());
    }

    #[test]
    fn test_is_exponential_false_for_none_policy() {
        let p = RetryPolicy::none();
        assert!(!p.is_exponential());
    }

    // ── Round 11: BackpressureGuard::is_soft_limited ──────────────────────────

    #[test]
    fn test_is_soft_limited_false_without_soft_limit() {
        let g = BackpressureGuard::new(10).unwrap();
        assert!(!g.is_soft_limited());
    }

    #[test]
    fn test_is_soft_limited_true_when_soft_limit_set() {
        let g = BackpressureGuard::new(10)
            .unwrap()
            .with_soft_limit(5)
            .unwrap();
        assert!(g.is_soft_limited());
    }

    // ── Round 12: RetryPolicy::base_delay_ms, BackpressureGuard::percent_full ─

    #[test]
    fn test_retry_policy_base_delay_ms_exponential() {
        let p = RetryPolicy::exponential(3, 250).unwrap();
        assert_eq!(p.base_delay_ms(), 250);
    }

    #[test]
    fn test_retry_policy_base_delay_ms_constant() {
        let p = RetryPolicy::constant(5, 100).unwrap();
        assert_eq!(p.base_delay_ms(), 100);
    }

    #[test]
    fn test_retry_policy_base_delay_ms_none_is_zero() {
        let p = RetryPolicy::none();
        assert_eq!(p.base_delay_ms(), 0);
    }

    #[test]
    fn test_backpressure_percent_full_zero_when_empty() {
        let g = BackpressureGuard::new(100).unwrap();
        let pct = g.percent_full().unwrap();
        assert!((pct - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_backpressure_percent_full_capped_at_100() {
        let g = BackpressureGuard::new(10).unwrap();
        // Fill all slots via try_acquire
        for _ in 0..10 {
            g.try_acquire().unwrap();
        }
        let pct = g.percent_full().unwrap();
        assert!((pct - 100.0).abs() < 1e-9);
    }

    // ── Round 27: get_result, rename_stage, failure_rate ─────────────────────

    #[test]
    fn test_deduplicator_get_result_returns_cached_value() {
        let d = Deduplicator::new(std::time::Duration::from_secs(60));
        d.check_and_register("req-1").unwrap();
        d.complete("req-1", "the answer").unwrap();
        let result = d.get_result("req-1").unwrap();
        assert_eq!(result, Some("the answer".to_string()));
    }

    #[test]
    fn test_deduplicator_get_result_missing_key_returns_none() {
        let d = Deduplicator::new(std::time::Duration::from_secs(60));
        assert_eq!(d.get_result("ghost").unwrap(), None);
    }

    #[test]
    fn test_pipeline_rename_stage_succeeds() {
        let mut p = Pipeline::new().add_stage("old-name", |s: String| Ok(s));
        let renamed = p.rename_stage("old-name", "new-name");
        assert!(renamed);
        assert!(p.has_stage("new-name"));
        assert!(!p.has_stage("old-name"));
    }

    #[test]
    fn test_pipeline_rename_stage_missing_returns_false() {
        let mut p = Pipeline::new();
        assert!(!p.rename_stage("nonexistent", "anything"));
    }

    #[test]
    fn test_circuit_breaker_failure_rate_zero_initially() {
        let cb = CircuitBreaker::new("svc", 5, std::time::Duration::from_secs(10)).unwrap();
        assert!((cb.failure_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_circuit_breaker_failure_rate_increases_with_failures() {
        let cb = CircuitBreaker::new("svc-fr", 4, std::time::Duration::from_secs(10)).unwrap();
        cb.record_failure();
        cb.record_failure();
        // 2 failures / threshold 4 = 0.5
        assert!((cb.failure_rate() - 0.5).abs() < 1e-9);
    }

    // ── Round 14: Pipeline::prepend_stage ────────────────────────────────────

    #[test]
    fn test_prepend_stage_inserts_at_front() {
        let p = Pipeline::new()
            .add_stage("second", |s| Ok(s))
            .prepend_stage("first", |s| Ok(s));
        let names = p.stage_names_owned();
        assert_eq!(names[0], "first");
        assert_eq!(names[1], "second");
    }

    #[test]
    fn test_prepend_stage_executes_before_existing_stages() {
        let p = Pipeline::new()
            .add_stage("append", |s| Ok(format!("{s}_appended")))
            .prepend_stage("prefix", |s| Ok(format!("pre_{s}")));
        let result = p.run("input".to_string()).unwrap();
        assert_eq!(result, "pre_input_appended");
    }

    #[test]
    fn test_prepend_stage_on_empty_pipeline() {
        let p = Pipeline::new().prepend_stage("only", |s| Ok(s.to_uppercase()));
        let result = p.run("hello".to_string()).unwrap();
        assert_eq!(result, "HELLO");
    }

    // ── Round 21: CircuitBreaker::is_at_threshold, BackpressureGuard::headroom_ratio ──

    #[test]
    fn test_circuit_breaker_is_at_threshold_false_initially() {
        let cb = CircuitBreaker::new("svc", 3, std::time::Duration::from_secs(10)).unwrap();
        assert!(!cb.is_at_threshold());
    }

    #[test]
    fn test_circuit_breaker_is_at_threshold_true_when_failures_reach_threshold() {
        let cb = CircuitBreaker::new("svc-t", 2, std::time::Duration::from_secs(10)).unwrap();
        cb.record_failure();
        assert!(!cb.is_at_threshold());
        cb.record_failure();
        assert!(cb.is_at_threshold());
    }

    #[test]
    fn test_backpressure_headroom_ratio_one_when_empty() {
        let g = BackpressureGuard::new(10).unwrap();
        let ratio = g.headroom_ratio().unwrap();
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_backpressure_headroom_ratio_decreases_on_acquire() {
        let g = BackpressureGuard::new(4).unwrap();
        g.try_acquire().unwrap(); // 1/4 used → headroom = 3/4
        let ratio = g.headroom_ratio().unwrap();
        assert!((ratio - 0.75).abs() < 1e-9);
    }

    // ── Round 17: Pipeline first/last/stage_index, BackpressureGuard is_empty/available ──

    #[test]
    fn test_pipeline_first_stage_name_returns_first() {
        let p = Pipeline::new()
            .add_stage("alpha", |s| Ok(s))
            .add_stage("beta", |s| Ok(s));
        assert_eq!(p.first_stage_name(), Some("alpha"));
    }

    #[test]
    fn test_pipeline_first_stage_name_none_when_empty() {
        let p = Pipeline::new();
        assert!(p.first_stage_name().is_none());
    }

    #[test]
    fn test_pipeline_last_stage_name_returns_last() {
        let p = Pipeline::new()
            .add_stage("alpha", |s| Ok(s))
            .add_stage("omega", |s| Ok(s));
        assert_eq!(p.last_stage_name(), Some("omega"));
    }

    #[test]
    fn test_pipeline_stage_index_returns_correct_position() {
        let p = Pipeline::new()
            .add_stage("first", |s| Ok(s))
            .add_stage("second", |s| Ok(s))
            .add_stage("third", |s| Ok(s));
        assert_eq!(p.stage_index("first"), Some(0));
        assert_eq!(p.stage_index("second"), Some(1));
        assert_eq!(p.stage_index("third"), Some(2));
        assert_eq!(p.stage_index("missing"), None);
    }

    #[test]
    fn test_backpressure_is_empty_true_when_no_slots_acquired() {
        let g = BackpressureGuard::new(10).unwrap();
        assert!(g.is_empty().unwrap());
    }

    #[test]
    fn test_backpressure_is_empty_false_after_acquire() {
        let g = BackpressureGuard::new(10).unwrap();
        g.try_acquire().unwrap();
        assert!(!g.is_empty().unwrap());
    }

    #[test]
    fn test_backpressure_available_capacity_decrements_on_acquire() {
        let g = BackpressureGuard::new(5).unwrap();
        assert_eq!(g.available_capacity().unwrap(), 5);
        g.try_acquire().unwrap();
        assert_eq!(g.available_capacity().unwrap(), 4);
    }

    // ── Round 16: Deduplicator::evict_oldest ─────────────────────────────────

    #[test]
    fn test_evict_oldest_removes_first_cached_entry() {
        let d = Deduplicator::new(std::time::Duration::from_secs(60));
        // Register and complete two entries to put them in cache
        d.check_and_register("alpha").unwrap();
        d.check_and_register("beta").unwrap();
        d.complete("alpha", "result_a").unwrap();
        d.complete("beta", "result_b").unwrap();
        // Evict the oldest (alpha)
        let removed = d.evict_oldest().unwrap();
        assert!(removed);
        assert!(d.get_result("alpha").unwrap().is_none());
        assert!(d.get_result("beta").unwrap().is_some());
    }

    #[test]
    fn test_evict_oldest_returns_false_when_empty() {
        let d = Deduplicator::new(std::time::Duration::from_secs(60));
        assert!(!d.evict_oldest().unwrap());
    }

    // ── Round 17: CircuitBreaker::is_at_threshold three-failure variant ──────

    #[test]
    fn test_circuit_breaker_is_at_threshold_true_after_three_failures() {
        let cb = CircuitBreaker::new("svc-3", 3, std::time::Duration::from_secs(60)).unwrap();
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_at_threshold());
    }

    // ── Round 22: CircuitBreaker::failures_until_open ─────────────────────────

    #[test]
    fn test_failures_until_open_equals_threshold_initially() {
        let cb = CircuitBreaker::new("svc-fuo", 5, std::time::Duration::from_secs(60)).unwrap();
        assert_eq!(cb.failures_until_open(), 5);
    }

    #[test]
    fn test_failures_until_open_decrements_with_each_failure() {
        let cb = CircuitBreaker::new("svc-fuo2", 4, std::time::Duration::from_secs(60)).unwrap();
        cb.record_failure();
        assert_eq!(cb.failures_until_open(), 3);
        cb.record_failure();
        assert_eq!(cb.failures_until_open(), 2);
    }

    #[test]
    fn test_failures_until_open_zero_when_at_threshold() {
        let cb = CircuitBreaker::new("svc-fuo3", 2, std::time::Duration::from_secs(60)).unwrap();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failures_until_open(), 0);
    }

    // ── Round 29: Deduplicator::cached_keys ──────────────────────────────────

    #[test]
    fn test_deduplicator_cached_keys_empty_initially() {
        let d = Deduplicator::new(Duration::from_secs(60));
        assert!(d.cached_keys().unwrap().is_empty());
    }

    #[test]
    fn test_deduplicator_cached_keys_contains_completed_key() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("ck-key").unwrap();
        d.complete("ck-key", "result").unwrap();
        let keys = d.cached_keys().unwrap();
        assert!(keys.contains(&"ck-key".to_string()));
    }

    #[test]
    fn test_deduplicator_cached_keys_excludes_in_flight() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("pending-key").unwrap();
        // In-flight keys live in a separate map, not in cache
        assert!(!d.cached_keys().unwrap().contains(&"pending-key".to_string()));
    }

    #[test]
    fn test_deduplicator_cached_keys_multiple_entries() {
        let d = Deduplicator::new(Duration::from_secs(60));
        for k in ["alpha", "beta", "gamma"] {
            d.check_and_register(k).unwrap();
            d.complete(k, "v").unwrap();
        }
        let keys = d.cached_keys().unwrap();
        assert_eq!(keys.len(), 3);
    }

    // ── Round 30: RetryPolicy::is_constant, total_max_delay_ms ───────────────

    #[test]
    fn test_retry_policy_is_constant_true_for_constant() {
        let p = RetryPolicy::constant(3, 100).unwrap();
        assert!(p.is_constant());
        assert!(!p.is_exponential());
    }

    #[test]
    fn test_retry_policy_is_constant_false_for_exponential() {
        let p = RetryPolicy::exponential(3, 100).unwrap();
        assert!(!p.is_constant());
    }

    #[test]
    fn test_retry_policy_total_max_delay_ms_constant() {
        // constant(3, 100) → delays [100, 100, 100] = 300
        let p = RetryPolicy::constant(3, 100).unwrap();
        assert_eq!(p.total_max_delay_ms(), 300);
    }

    #[test]
    fn test_retry_policy_total_max_delay_ms_exponential() {
        // exponential(3, 100) → delays [100, 200, 400] (capped at MAX)
        let p = RetryPolicy::exponential(3, 100).unwrap();
        let total = p.total_max_delay_ms();
        assert!(total >= 300); // at minimum 100+100+100
    }

    // ── Round 30: CircuitBreaker::is_half_open, is_healthy ───────────────────

    #[test]
    fn test_circuit_breaker_is_healthy_true_when_closed() {
        let cb = CircuitBreaker::new("svc-ih1", 3, Duration::from_secs(60)).unwrap();
        assert!(cb.is_healthy());
    }

    #[test]
    fn test_circuit_breaker_is_healthy_false_when_open() {
        let cb = CircuitBreaker::new("svc-ih2", 1, Duration::from_secs(60)).unwrap();
        let _: Result<(), _> = cb.call(|| Err::<(), _>("fail".to_string()));
        assert!(!cb.is_healthy());
    }

    #[test]
    fn test_circuit_breaker_is_half_open_after_zero_recovery() {
        let cb = CircuitBreaker::new("svc-ho1", 1, Duration::ZERO).unwrap();
        let _: Result<(), _> = cb.call(|| Err::<(), _>("fail".to_string()));
        // With zero recovery window the circuit immediately enters HalfOpen
        assert!(cb.is_half_open() || cb.is_healthy()); // HalfOpen or recovered
    }

    // ── Round 30: Deduplicator::is_idle ──────────────────────────────────────

    #[test]
    fn test_deduplicator_is_idle_true_when_empty() {
        let d = Deduplicator::new(Duration::from_secs(60));
        assert!(d.is_idle().unwrap());
    }

    #[test]
    fn test_deduplicator_is_idle_false_when_in_flight() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("req-x").unwrap();
        assert!(!d.is_idle().unwrap());
    }

    #[test]
    fn test_deduplicator_is_idle_true_after_complete() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("req-y").unwrap();
        d.complete("req-y", "done").unwrap();
        assert!(d.is_idle().unwrap());
    }

    // ── Round 26: in_flight_count ─────────────────────────────────────────────

    #[test]
    fn test_deduplicator_in_flight_count_zero_initially() {
        let d = Deduplicator::new(Duration::from_secs(60));
        assert_eq!(d.in_flight_count().unwrap(), 0);
    }

    #[test]
    fn test_deduplicator_in_flight_count_increments_on_register() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("k1").unwrap();
        d.check_and_register("k2").unwrap();
        assert_eq!(d.in_flight_count().unwrap(), 2);
    }

    #[test]
    fn test_deduplicator_in_flight_count_decrements_after_complete() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("k1").unwrap();
        d.complete("k1", "result").unwrap();
        assert_eq!(d.in_flight_count().unwrap(), 0);
    }

    // ── Round 27: total_count, acquired_count, swap_stages, will_retry_at_all

    #[test]
    fn test_deduplicator_total_count_sums_in_flight_and_cached() {
        let d = Deduplicator::new(Duration::from_secs(60));
        d.check_and_register("k1").unwrap(); // in-flight
        d.check_and_register("k2").unwrap(); // in-flight
        d.complete("k1", "done").unwrap();   // moves to cache
        // 1 in-flight + 1 cached = 2
        assert_eq!(d.total_count().unwrap(), 2);
    }

    #[test]
    fn test_deduplicator_total_count_zero_when_empty() {
        let d = Deduplicator::new(Duration::from_secs(60));
        assert_eq!(d.total_count().unwrap(), 0);
    }

    #[test]
    fn test_backpressure_acquired_count_zero_initially() {
        let g = BackpressureGuard::new(5).unwrap();
        assert_eq!(g.acquired_count().unwrap(), 0);
    }

    #[test]
    fn test_backpressure_acquired_count_increments_on_acquire() {
        let g = BackpressureGuard::new(5).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        assert_eq!(g.acquired_count().unwrap(), 2);
    }

    #[test]
    fn test_pipeline_swap_stages_swaps_positions() {
        let mut p = Pipeline::new()
            .add_stage("a", |s| Ok(s + "A"))
            .add_stage("b", |s| Ok(s + "B"));
        let swapped = p.swap_stages("a", "b");
        assert!(swapped);
        assert_eq!(p.first_stage_name().unwrap(), "b");
        assert_eq!(p.last_stage_name().unwrap(), "a");
    }

    #[test]
    fn test_pipeline_swap_stages_returns_false_for_unknown_stage() {
        let mut p = Pipeline::new().add_stage("a", |s| Ok(s));
        assert!(!p.swap_stages("a", "missing"));
    }

    #[test]
    fn test_retry_policy_will_retry_at_all_false_for_none() {
        let p = RetryPolicy::none();
        assert!(!p.will_retry_at_all());
    }

    #[test]
    fn test_retry_policy_will_retry_at_all_true_for_exponential() {
        let p = RetryPolicy::exponential(3, 100).unwrap();
        assert!(p.will_retry_at_all());
    }

    #[test]
    fn test_agent_config_stop_sequence_count_zero_by_default() {
        // Tested via AgentConfig but placed in orchestrator block for proximity
        use crate::agent::AgentConfig;
        let cfg = AgentConfig::new(5, "m");
        assert_eq!(cfg.stop_sequence_count(), 0);
    }

    #[test]
    fn test_agent_config_stop_sequence_count_after_adding() {
        use crate::agent::AgentConfig;
        let cfg = AgentConfig::new(5, "m")
            .with_stop_sequences(vec!["STOP".to_string(), "END".to_string()]);
        assert_eq!(cfg.stop_sequence_count(), 2);
    }
}
