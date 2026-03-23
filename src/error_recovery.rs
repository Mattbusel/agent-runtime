//! # Module: error_recovery
//!
//! ## Responsibility
//! Intelligent error recovery strategies for agent failures. Maintains a
//! per-error-type policy registry, tracks error history for rate calculations,
//! manages circuit-breaker open windows, and selects outcomes (Recovered,
//! Escalated, Failed, CircuitOpen) based on the registered strategy.

use std::collections::HashMap;

// ── Error types ───────────────────────────────────────────────────────────────

/// Classification of the error that occurred.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorType {
    /// A network call timed out.
    NetworkTimeout,
    /// The upstream API rate-limited this client.
    RateLimitExceeded,
    /// The supplied input was malformed or invalid.
    InvalidInput,
    /// A tool invocation returned a failure.
    ToolFailure,
    /// The agent exhausted its allocated memory budget.
    MemoryExhausted,
    /// Authentication or authorization was rejected.
    AuthenticationFailed,
    /// An error that does not fit any specific category.
    UnknownError,
}

impl ErrorType {
    /// Returns the default [`RecoveryStrategy`] for this error type.
    pub fn default_strategy(&self) -> RecoveryStrategy {
        match self {
            ErrorType::NetworkTimeout => RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff_ms: 500,
            },
            ErrorType::RateLimitExceeded => RecoveryStrategy::Retry {
                max_attempts: 5,
                backoff_ms: 2_000,
            },
            ErrorType::InvalidInput => RecoveryStrategy::Ignore,
            ErrorType::ToolFailure => RecoveryStrategy::Fallback("default_tool".to_string()),
            ErrorType::MemoryExhausted => RecoveryStrategy::CircuitBreak(30_000),
            ErrorType::AuthenticationFailed => RecoveryStrategy::Escalate,
            ErrorType::UnknownError => RecoveryStrategy::Escalate,
        }
    }

    /// Returns a stable string key for map storage.
    fn key(&self) -> &'static str {
        match self {
            ErrorType::NetworkTimeout => "NetworkTimeout",
            ErrorType::RateLimitExceeded => "RateLimitExceeded",
            ErrorType::InvalidInput => "InvalidInput",
            ErrorType::ToolFailure => "ToolFailure",
            ErrorType::MemoryExhausted => "MemoryExhausted",
            ErrorType::AuthenticationFailed => "AuthenticationFailed",
            ErrorType::UnknownError => "UnknownError",
        }
    }
}

// ── Recovery strategy ─────────────────────────────────────────────────────────

/// The action the engine should take when an error is encountered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry the operation up to `max_attempts` times with `backoff_ms` base delay.
    Retry {
        /// Maximum number of retry attempts.
        max_attempts: u32,
        /// Base back-off delay in milliseconds.
        backoff_ms: u64,
    },
    /// Delegate to a named fallback handler.
    Fallback(String),
    /// Swallow the error and continue.
    Ignore,
    /// Escalate to a human or supervisory agent.
    Escalate,
    /// Execute a list of compensating actions to undo side effects.
    Compensate(Vec<String>),
    /// Open a circuit breaker for the given duration (milliseconds).
    CircuitBreak(u64),
}

// ── Recovery outcome ──────────────────────────────────────────────────────────

/// The result produced by the recovery engine after processing an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// The error was handled and the operation can continue.
    Recovered,
    /// The error was forwarded to the named supervisory target.
    Escalated(String),
    /// Recovery was not possible; contains a human-readable reason.
    Failed(String),
    /// A circuit breaker is open; the caller must wait before retrying.
    CircuitOpen,
}

// ── RecoveryError ─────────────────────────────────────────────────────────────

/// An error event with full context for recovery decision-making.
#[derive(Debug, Clone)]
pub struct RecoveryError {
    /// Unique identifier for this error event.
    pub error_id: String,
    /// Classification of the error.
    pub error_type: ErrorType,
    /// Human-readable description.
    pub message: String,
    /// Arbitrary key-value context (e.g. service name, request id).
    pub context: HashMap<String, String>,
    /// Unix-epoch millisecond timestamp when the error occurred.
    pub timestamp: u64,
    /// Whether this error is potentially recoverable.
    pub recoverable: bool,
}

// ── RecoveryPolicy ────────────────────────────────────────────────────────────

/// Associates a [`RecoveryStrategy`] with an error type and a rate cap.
#[derive(Debug, Clone)]
pub struct RecoveryPolicy {
    /// The error type this policy applies to.
    pub error_type: ErrorType,
    /// The strategy to apply when this error type occurs.
    pub strategy: RecoveryStrategy,
    /// Maximum number of errors of this type allowed per hour before escalating.
    pub max_occurrences_per_hour: u32,
}

// ── Internal history record ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ErrorRecord {
    timestamp: u64,
}

#[derive(Debug)]
struct CircuitWindow {
    opened_at: u64,
    duration_ms: u64,
}

// ── ErrorRecoveryEngine ───────────────────────────────────────────────────────

/// Central engine that applies recovery policies to agent errors.
///
/// Stores per-type policies, error history, and circuit-breaker windows.
/// All timestamps are Unix-epoch milliseconds supplied by the caller so that
/// the engine remains deterministic and test-friendly.
#[derive(Debug, Default)]
pub struct ErrorRecoveryEngine {
    policies: HashMap<String, RecoveryPolicy>,
    history: HashMap<String, Vec<ErrorRecord>>,
    circuits: HashMap<String, CircuitWindow>,
}

impl ErrorRecoveryEngine {
    /// Creates a new, empty engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a recovery policy for an error type, replacing any existing one.
    pub fn register_policy(&mut self, policy: RecoveryPolicy) {
        self.policies.insert(policy.error_type.key().to_string(), policy);
    }

    /// Records an error and decides a [`RecoveryOutcome`].
    ///
    /// * `error`   – the error to handle.
    /// * `attempt` – zero-based attempt number (0 = first occurrence).
    /// * `now`     – current time in Unix-epoch milliseconds.
    pub fn handle_error(
        &mut self,
        error: &RecoveryError,
        attempt: u32,
        now: u64,
    ) -> RecoveryOutcome {
        // Record the event in history.
        let key = error.error_type.key().to_string();
        self.history
            .entry(key.clone())
            .or_default()
            .push(ErrorRecord { timestamp: now });

        // If circuit is open, reject immediately.
        if self.circuit_open(&error.error_type, now) {
            return RecoveryOutcome::CircuitOpen;
        }

        if !error.recoverable {
            return RecoveryOutcome::Failed("error marked non-recoverable".to_string());
        }

        // Look up registered policy or fall back to the type's default.
        let strategy = self
            .policies
            .get(&key)
            .map(|p| p.strategy.clone())
            .unwrap_or_else(|| error.error_type.default_strategy());

        // Check hourly rate cap.
        let hourly_count = self.error_rate(&error.error_type, 3600, now) as u64;
        let cap = self
            .policies
            .get(&key)
            .map(|p| p.max_occurrences_per_hour as u64)
            .unwrap_or(u64::MAX);
        if hourly_count > cap {
            return RecoveryOutcome::Escalated(format!(
                "error rate {} exceeded cap {} for {:?}",
                hourly_count, cap, error.error_type
            ));
        }

        match &strategy {
            RecoveryStrategy::Retry { max_attempts, .. } => {
                if attempt < *max_attempts {
                    RecoveryOutcome::Recovered
                } else {
                    RecoveryOutcome::Failed(format!(
                        "exhausted {} retry attempts",
                        max_attempts
                    ))
                }
            }
            RecoveryStrategy::Fallback(_name) => RecoveryOutcome::Recovered,
            RecoveryStrategy::Ignore => RecoveryOutcome::Recovered,
            RecoveryStrategy::Escalate => {
                RecoveryOutcome::Escalated("policy: escalate".to_string())
            }
            RecoveryStrategy::Compensate(_actions) => RecoveryOutcome::Recovered,
            RecoveryStrategy::CircuitBreak(duration_ms) => {
                // Open the circuit.
                self.circuits.insert(
                    key,
                    CircuitWindow {
                        opened_at: now,
                        duration_ms: *duration_ms,
                    },
                );
                RecoveryOutcome::CircuitOpen
            }
        }
    }

    /// Returns `true` if the operation should be retried for this error at
    /// the given zero-based `attempt` number.
    pub fn should_retry(&self, error: &RecoveryError, attempt: u32) -> bool {
        let key = error.error_type.key();
        let strategy = self
            .policies
            .get(key)
            .map(|p| p.strategy.clone())
            .unwrap_or_else(|| error.error_type.default_strategy());

        match strategy {
            RecoveryStrategy::Retry { max_attempts, .. } => attempt < max_attempts,
            _ => false,
        }
    }

    /// Computes the back-off delay for a given attempt.
    ///
    /// Formula: `base_ms * 2^attempt + attempt * 137` (deterministic, no RNG).
    pub fn backoff_delay(attempt: u32, base_ms: u64) -> u64 {
        // 2^attempt, capped at u64::MAX to avoid overflow.
        let exp: u64 = if attempt < 64 { 1u64 << attempt } else { u64::MAX };
        base_ms
            .saturating_mul(exp)
            .saturating_add((attempt as u64).saturating_mul(137))
    }

    /// Returns the number of errors of the given type within the last
    /// `window_secs` seconds, expressed as a count (cast to f64 for callers
    /// that need a rate).
    pub fn error_rate(&self, error_type: &ErrorType, window_secs: u64, now: u64) -> f64 {
        let key = error_type.key();
        let window_ms = window_secs.saturating_mul(1_000);
        let cutoff = now.saturating_sub(window_ms);
        self.history
            .get(key)
            .map(|records| {
                records
                    .iter()
                    .filter(|r| r.timestamp >= cutoff)
                    .count() as f64
            })
            .unwrap_or(0.0)
    }

    /// Returns `true` when a circuit breaker for the given error type is
    /// currently open (i.e. the cool-down window has not yet elapsed).
    pub fn circuit_open(&self, error_type: &ErrorType, now: u64) -> bool {
        let key = error_type.key();
        self.circuits
            .get(key)
            .map(|w| now < w.opened_at.saturating_add(w.duration_ms))
            .unwrap_or(false)
    }

    /// Returns a count of errors by type string within the given window.
    pub fn error_summary(&self, window_secs: u64, now: u64) -> HashMap<String, u64> {
        let window_ms = window_secs.saturating_mul(1_000);
        let cutoff = now.saturating_sub(window_ms);
        let mut out = HashMap::new();
        for (key, records) in &self.history {
            let count = records.iter().filter(|r| r.timestamp >= cutoff).count() as u64;
            if count > 0 {
                out.insert(key.clone(), count);
            }
        }
        out
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn make_error(error_type: ErrorType, recoverable: bool, ts: u64) -> RecoveryError {
        RecoveryError {
            error_id: "err-1".to_string(),
            error_type,
            message: "test error".to_string(),
            context: HashMap::new(),
            timestamp: ts,
            recoverable,
        }
    }

    #[test]
    fn test_default_strategy_network_timeout() {
        let s = ErrorType::NetworkTimeout.default_strategy();
        assert!(matches!(s, RecoveryStrategy::Retry { max_attempts: 3, .. }));
    }

    #[test]
    fn test_default_strategy_invalid_input() {
        assert_eq!(ErrorType::InvalidInput.default_strategy(), RecoveryStrategy::Ignore);
    }

    #[test]
    fn test_default_strategy_auth_failed() {
        assert_eq!(ErrorType::AuthenticationFailed.default_strategy(), RecoveryStrategy::Escalate);
    }

    #[test]
    fn test_backoff_delay_deterministic() {
        // attempt 0: 500 * 1 + 0*137 = 500
        assert_eq!(ErrorRecoveryEngine::backoff_delay(0, 500), 500);
        // attempt 1: 500 * 2 + 1*137 = 1137
        assert_eq!(ErrorRecoveryEngine::backoff_delay(1, 500), 1137);
        // attempt 2: 500 * 4 + 2*137 = 2274
        assert_eq!(ErrorRecoveryEngine::backoff_delay(2, 500), 2274);
    }

    #[test]
    fn test_handle_error_retry_recovered() {
        let mut engine = ErrorRecoveryEngine::new();
        let err = make_error(ErrorType::NetworkTimeout, true, 1000);
        let outcome = engine.handle_error(&err, 0, 1000);
        assert_eq!(outcome, RecoveryOutcome::Recovered);
    }

    #[test]
    fn test_handle_error_retry_exhausted() {
        let mut engine = ErrorRecoveryEngine::new();
        let err = make_error(ErrorType::NetworkTimeout, true, 1000);
        // attempt 3 >= max_attempts(3) → Failed
        let outcome = engine.handle_error(&err, 3, 1000);
        assert!(matches!(outcome, RecoveryOutcome::Failed(_)));
    }

    #[test]
    fn test_handle_non_recoverable() {
        let mut engine = ErrorRecoveryEngine::new();
        let err = make_error(ErrorType::NetworkTimeout, false, 1000);
        assert!(matches!(engine.handle_error(&err, 0, 1000), RecoveryOutcome::Failed(_)));
    }

    #[test]
    fn test_circuit_break_strategy() {
        let mut engine = ErrorRecoveryEngine::new();
        engine.register_policy(RecoveryPolicy {
            error_type: ErrorType::MemoryExhausted,
            strategy: RecoveryStrategy::CircuitBreak(5_000),
            max_occurrences_per_hour: 100,
        });
        let err = make_error(ErrorType::MemoryExhausted, true, 1000);
        let outcome = engine.handle_error(&err, 0, 1000);
        assert_eq!(outcome, RecoveryOutcome::CircuitOpen);
        // Circuit should still be open 4 seconds later.
        assert!(engine.circuit_open(&ErrorType::MemoryExhausted, 5_999));
        // But closed after 5 seconds have elapsed.
        assert!(!engine.circuit_open(&ErrorType::MemoryExhausted, 6_001));
    }

    #[test]
    fn test_should_retry() {
        let engine = ErrorRecoveryEngine::new();
        let err = make_error(ErrorType::NetworkTimeout, true, 0);
        assert!(engine.should_retry(&err, 0));
        assert!(engine.should_retry(&err, 2));
        assert!(!engine.should_retry(&err, 3));
    }

    #[test]
    fn test_error_rate_within_window() {
        let mut engine = ErrorRecoveryEngine::new();
        let err = make_error(ErrorType::ToolFailure, true, 1000);
        engine.handle_error(&err, 0, 1000);
        engine.handle_error(&err, 0, 2000);
        engine.handle_error(&err, 0, 10_000);
        // window_secs=10, now=10_000 → cutoff=0 → all 3 events included
        assert_eq!(engine.error_rate(&ErrorType::ToolFailure, 10, 10_000), 3.0);
        // window_secs=5, now=10_000 → cutoff=5_000 → only event at 10_000
        assert_eq!(engine.error_rate(&ErrorType::ToolFailure, 5, 10_000), 1.0);
    }

    #[test]
    fn test_error_summary() {
        let mut engine = ErrorRecoveryEngine::new();
        let e1 = make_error(ErrorType::NetworkTimeout, true, 100);
        let e2 = make_error(ErrorType::RateLimitExceeded, true, 200);
        engine.handle_error(&e1, 0, 100);
        engine.handle_error(&e2, 0, 200);
        let summary = engine.error_summary(3600, 1000);
        assert_eq!(summary.get("NetworkTimeout").copied().unwrap_or(0), 1);
        assert_eq!(summary.get("RateLimitExceeded").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_rate_cap_triggers_escalation() {
        let mut engine = ErrorRecoveryEngine::new();
        engine.register_policy(RecoveryPolicy {
            error_type: ErrorType::InvalidInput,
            strategy: RecoveryStrategy::Ignore,
            max_occurrences_per_hour: 2,
        });
        let err = make_error(ErrorType::InvalidInput, true, 1000);
        engine.handle_error(&err, 0, 1000);
        engine.handle_error(&err, 0, 1001);
        // Third occurrence exceeds cap of 2 → Escalated
        let outcome = engine.handle_error(&err, 0, 1002);
        assert!(matches!(outcome, RecoveryOutcome::Escalated(_)));
    }

    #[test]
    fn test_escalate_strategy() {
        let mut engine = ErrorRecoveryEngine::new();
        let err = make_error(ErrorType::AuthenticationFailed, true, 0);
        assert!(matches!(engine.handle_error(&err, 0, 0), RecoveryOutcome::Escalated(_)));
    }

    #[test]
    fn test_compensate_strategy() {
        let mut engine = ErrorRecoveryEngine::new();
        engine.register_policy(RecoveryPolicy {
            error_type: ErrorType::ToolFailure,
            strategy: RecoveryStrategy::Compensate(vec!["rollback_db".to_string()]),
            max_occurrences_per_hour: 100,
        });
        let err = make_error(ErrorType::ToolFailure, true, 0);
        assert_eq!(engine.handle_error(&err, 0, 0), RecoveryOutcome::Recovered);
    }
}
