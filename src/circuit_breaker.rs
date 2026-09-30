//! # Module: Circuit Breaker
//!
//! Circuit breaker for service call protection.
//!
//! ## Overview
//!
//! [`CircuitBreaker`] tracks rolling success/failure counts and transitions
//! through three states: Closed (normal), Open (calls rejected), and HalfOpen
//! (probe mode). [`CircuitBreakerManager`] manages multiple named breakers.
//!
//! ## States
//!
//! - **Closed**: calls pass through; failures increment the window counter.
//! - **Open**: calls are rejected immediately; after `timeout_secs` the breaker
//!   moves to HalfOpen automatically.
//! - **HalfOpen**: a single probe call is allowed; on success the breaker
//!   transitions back to Closed, on failure it returns to Open.
//!
//! ## Quick Start
//!
//! ```rust
//! use llm_agent_runtime::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
//!
//! let cfg = CircuitBreakerConfig {
//!     failure_threshold: 3,
//!     success_threshold: 2,
//!     timeout_secs: 30,
//!     window_size: 10,
//! };
//! let cb = CircuitBreaker::new(cfg);
//! let result = cb.call(|| Ok::<_, String>("ok"));
//! assert!(result.is_ok());
//! ```

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::Instant,
};


// ── Error type ──────────────────────────────────────────────────────────────

/// Errors returned by [`CircuitBreaker::call`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitError {
    /// The circuit is open; the call was rejected without execution.
    Open,
    /// The underlying call was executed but returned an error.
    CallFailed(String),
}

impl std::fmt::Display for CircuitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitError::Open => write!(f, "circuit is open"),
            CircuitError::CallFailed(msg) => write!(f, "call failed: {msg}"),
        }
    }
}

impl std::error::Error for CircuitError {}

// ── State ────────────────────────────────────────────────────────────────────

/// Current state of a [`CircuitBreaker`].
#[derive(Debug, Clone)]
pub enum CircuitState {
    /// Normal operation — calls are forwarded.
    Closed,
    /// Circuit is tripped — calls are rejected; carries the time the circuit opened.
    Open {
        /// When the circuit transitioned to Open.
        opened_at: Instant,
    },
    /// One probe call is allowed to test whether the downstream has recovered.
    HalfOpen,
}

impl CircuitState {
    /// Human-readable label for the state.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            CircuitState::Closed => "Closed",
            CircuitState::Open { .. } => "Open",
            CircuitState::HalfOpen => "HalfOpen",
        }
    }
}

// ── Config ───────────────────────────────────────────────────────────────────

/// Configuration parameters for a [`CircuitBreaker`].
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures within `window_size` calls before the circuit opens.
    pub failure_threshold: usize,
    /// Number of consecutive successes in HalfOpen before the circuit closes.
    pub success_threshold: usize,
    /// Seconds to wait in Open state before transitioning to HalfOpen.
    pub timeout_secs: u64,
    /// Rolling window size (number of recent call outcomes to track).
    pub window_size: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_secs: 60,
            window_size: 20,
        }
    }
}

// ── WindowStats ──────────────────────────────────────────────────────────────

/// A snapshot of the rolling-window statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowStats {
    /// Number of successful calls in the current window.
    pub successes: usize,
    /// Number of failed calls in the current window.
    pub failures: usize,
    /// Total calls recorded in the current window.
    pub total: usize,
}

// ── Inner state ──────────────────────────────────────────────────────────────

struct Inner {
    state: CircuitState,
    /// Rolling window: `true` = success, `false` = failure.
    window: VecDeque<bool>,
    /// Successes counted while in HalfOpen (for threshold check).
    half_open_successes: usize,
}

impl Inner {
    fn new(window_size: usize) -> Self {
        Self {
            state: CircuitState::Closed,
            window: VecDeque::with_capacity(window_size),
            half_open_successes: 0,
        }
    }

    fn push_outcome(&mut self, success: bool, window_size: usize) {
        if self.window.len() >= window_size {
            self.window.pop_front();
        }
        self.window.push_back(success);
    }

    fn window_stats(&self) -> WindowStats {
        let successes = self.window.iter().filter(|&&s| s).count();
        let total = self.window.len();
        WindowStats {
            successes,
            failures: total - successes,
            total,
        }
    }
}

// ── CircuitBreaker ────────────────────────────────────────────────────────────

/// A circuit breaker that protects a downstream service from cascading failures.
pub struct CircuitBreaker {
    inner: Arc<Mutex<Inner>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    /// Create a new [`CircuitBreaker`] with the given configuration.
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let window_size = config.window_size;
        Self {
            inner: Arc::new(Mutex::new(Inner::new(window_size))),
            config,
        }
    }

    /// Execute `f` if the circuit allows it.
    ///
    /// Returns `Err(CircuitError::Open)` when the circuit is open.
    /// Returns `Err(CircuitError::CallFailed(_))` when `f` returns `Err`.
    pub fn call<F, T, E>(&self, f: F) -> Result<T, CircuitError>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        // Check state (and handle Open → HalfOpen auto-transition).
        let current = self.state();
        match current {
            CircuitState::Open { .. } => return Err(CircuitError::Open),
            CircuitState::Closed | CircuitState::HalfOpen => {}
        }

        match f() {
            Ok(v) => {
                self.record_success();
                Ok(v)
            }
            Err(e) => {
                self.record_failure();
                Err(CircuitError::CallFailed(e.to_string()))
            }
        }
    }

    /// Record a successful call outcome.
    pub fn record_success(&self) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        inner.push_outcome(true, self.config.window_size);

        if let CircuitState::HalfOpen = inner.state {
            inner.half_open_successes += 1;
            if inner.half_open_successes >= self.config.success_threshold {
                inner.state = CircuitState::Closed;
                inner.half_open_successes = 0;
            }
        }
    }

    /// Record a failed call outcome.
    pub fn record_failure(&self) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        inner.push_outcome(false, self.config.window_size);
        inner.half_open_successes = 0;

        match inner.state {
            CircuitState::HalfOpen => {
                inner.state = CircuitState::Open {
                    opened_at: Instant::now(),
                };
            }
            CircuitState::Closed => {
                let stats = inner.window_stats();
                if stats.failures >= self.config.failure_threshold {
                    inner.state = CircuitState::Open {
                        opened_at: Instant::now(),
                    };
                }
            }
            CircuitState::Open { .. } => {}
        }
    }

    /// Return the current state, automatically transitioning Open → HalfOpen
    /// after the configured timeout has elapsed.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return CircuitState::Open {
                opened_at: Instant::now(),
            },
        };

        if let CircuitState::Open { opened_at } = inner.state {
            let elapsed = opened_at.elapsed().as_secs();
            if elapsed >= self.config.timeout_secs {
                inner.state = CircuitState::HalfOpen;
                inner.half_open_successes = 0;
            }
        }

        inner.state.clone()
    }

    /// Return a snapshot of the rolling-window statistics.
    #[must_use]
    pub fn window_stats(&self) -> WindowStats {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return WindowStats { successes: 0, failures: 0, total: 0 },
        };
        inner.window_stats()
    }

    /// Force the circuit to the Closed state and clear the rolling window.
    pub fn reset(&self) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        inner.state = CircuitState::Closed;
        inner.window.clear();
        inner.half_open_successes = 0;
    }
}

// ── CircuitBreakerManager ─────────────────────────────────────────────────────

/// Manages a collection of named [`CircuitBreaker`] instances.
pub struct CircuitBreakerManager {
    breakers: Mutex<HashMap<String, Arc<CircuitBreaker>>>,
}

impl Default for CircuitBreakerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            breakers: Mutex::new(HashMap::new()),
        }
    }

    /// Return the named breaker, creating it with `config` if it does not exist.
    #[must_use]
    pub fn get_or_create(&self, name: &str, config: CircuitBreakerConfig) -> Arc<CircuitBreaker> {
        let mut map = match self.breakers.lock() {
            Ok(g) => g,
            Err(_) => return Arc::new(CircuitBreaker::new(config)),
        };
        if let Some(cb) = map.get(name) {
            return Arc::clone(cb);
        }
        let cb = Arc::new(CircuitBreaker::new(config));
        map.insert(name.to_owned(), Arc::clone(&cb));
        cb
    }

    /// Return a snapshot mapping each breaker name to its current state label.
    #[must_use]
    pub fn all_states(&self) -> HashMap<String, String> {
        let map = match self.breakers.lock() {
            Ok(g) => g,
            Err(_) => return HashMap::new(),
        };
        map.iter()
            .map(|(name, cb)| (name.clone(), cb.state().label().to_owned()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #[allow(clippy::unwrap_used)]
    use super::*;

    fn cfg(failure_threshold: usize, window_size: usize) -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold,
            success_threshold: 2,
            timeout_secs: 60,
            window_size,
        }
    }

    #[test]
    fn closed_by_default() {
        let cb = CircuitBreaker::new(cfg(3, 10));
        assert!(matches!(cb.state(), CircuitState::Closed));
    }

    #[test]
    fn opens_after_threshold_failures() {
        let cb = CircuitBreaker::new(cfg(3, 10));
        for _ in 0..3 {
            cb.record_failure();
        }
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn call_rejected_when_open() {
        let cb = CircuitBreaker::new(cfg(1, 10));
        cb.record_failure();
        let result = cb.call(|| Ok::<_, String>("ok"));
        assert_eq!(result, Err(CircuitError::Open));
    }

    #[test]
    fn reset_closes_circuit() {
        let cb = CircuitBreaker::new(cfg(1, 10));
        cb.record_failure();
        cb.reset();
        assert!(matches!(cb.state(), CircuitState::Closed));
    }

    #[test]
    fn manager_get_or_create() {
        let mgr = CircuitBreakerManager::new();
        let cb1 = mgr.get_or_create("svc", cfg(3, 10));
        let cb2 = mgr.get_or_create("svc", cfg(3, 10));
        // Same Arc pointer.
        assert!(Arc::ptr_eq(&cb1, &cb2));
    }
}
