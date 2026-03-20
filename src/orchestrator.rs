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

    /// Return the total maximum delay in milliseconds across all retry attempts.
    ///
    /// Sums `delay_for(attempt)` for every attempt from 1 to `max_attempts`.
    /// Useful for estimating worst-case latency budgets.
    pub fn total_max_delay_ms(&self) -> u64 {
        (1..=self.max_attempts)
            .map(|a| self.delay_for(a).as_millis() as u64)
            .sum()
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

    /// Return the number of keys currently in the completed result cache.
    ///
    /// Note: expired entries are only removed lazily on the next `check*` call.
    pub fn cached_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = timed_lock(&self.inner, "Deduplicator::cached_count");
        Ok(inner.cache.len())
    }

    /// Return the configured time-to-live for cached results.
    pub fn ttl(&self) -> Duration {
        self.ttl
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

    /// Return `true` if the guard is at or over its hard capacity.
    pub fn is_full(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.depth()? >= self.capacity)
    }

    /// Return the hard capacity (maximum concurrent slots) configured for this guard.
    pub fn hard_capacity(&self) -> usize {
        self.capacity
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

    /// Return `true` if the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Return the number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Return the names of all stages in execution order.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name.as_str()).collect()
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
}
