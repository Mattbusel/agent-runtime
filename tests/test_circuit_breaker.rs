//! Tests for CircuitBreaker state machine: closed, open, half-open transitions,
//! threshold triggering, and recovery timing.

use agent_runtime::orchestrator::{CircuitBreaker, CircuitState, InMemoryCircuitBreakerBackend};
use agent_runtime::AgentRuntimeError;
use std::sync::Arc;
use std::time::Duration;

// ── Closed state ───────────────────────────────────────────────────────────────

#[test]
fn cb_starts_in_closed_state() {
    let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
    assert_eq!(cb.state().unwrap(), CircuitState::Closed);
}

#[test]
fn cb_failure_count_starts_at_zero() {
    let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
    assert_eq!(cb.failure_count().unwrap(), 0);
}

#[test]
fn cb_success_does_not_change_closed_state() {
    let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
    let _: Result<(), AgentRuntimeError> = cb.call(|| Ok::<(), AgentRuntimeError>(()));
    assert_eq!(cb.state().unwrap(), CircuitState::Closed);
}

#[test]
fn cb_one_failure_below_threshold_stays_closed() {
    let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
    let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("oops".to_string()));
    assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    assert_eq!(cb.failure_count().unwrap(), 1);
}

#[test]
fn cb_two_failures_below_threshold_stays_closed() {
    let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
    for _ in 0..2 {
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("err".to_string()));
    }
    assert_eq!(cb.state().unwrap(), CircuitState::Closed);
}

// ── Open state ────────────────────────────────────────────────────────────────

#[test]
fn cb_opens_exactly_at_threshold() {
    let cb = CircuitBreaker::new("svc", 3, Duration::from_secs(60)).unwrap();
    for _ in 0..3 {
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("err".to_string()));
    }
    assert!(matches!(cb.state().unwrap(), CircuitState::Open { .. }));
}

#[test]
fn cb_open_state_fast_fails_without_calling_f() {
    let cb = CircuitBreaker::new("svc", 1, Duration::from_secs(3600)).unwrap();
    // Trigger open
    let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));

    let mut called = false;
    let result: Result<(), AgentRuntimeError> = cb.call(|| {
        called = true;
        Ok::<(), AgentRuntimeError>(())
    });

    assert!(matches!(result, Err(AgentRuntimeError::CircuitOpen { .. })));
    assert!(!called, "f should not be called when circuit is open");
}

#[test]
fn cb_open_error_includes_service_name() {
    let cb = CircuitBreaker::new("my-service", 1, Duration::from_secs(3600)).unwrap();
    let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
    let result: Result<(), AgentRuntimeError> = cb.call(|| Ok::<(), AgentRuntimeError>(()));
    if let Err(AgentRuntimeError::CircuitOpen { service }) = result {
        assert!(service.contains("my-service"));
    } else {
        panic!("expected CircuitOpen error");
    }
}

// ── Half-open / recovery ──────────────────────────────────────────────────────

#[test]
fn cb_transitions_to_half_open_after_recovery_window() {
    let cb = CircuitBreaker::new("svc", 1, Duration::ZERO).unwrap();
    let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
    // Recovery window is zero, so next call should probe
    let result: Result<i32, AgentRuntimeError> = cb.call(|| Ok::<i32, AgentRuntimeError>(42));
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn cb_successful_probe_in_half_open_closes_circuit() {
    let cb = CircuitBreaker::new("svc", 1, Duration::ZERO).unwrap();
    let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
    // Probe succeeds
    let _: Result<i32, AgentRuntimeError> = cb.call(|| Ok::<i32, AgentRuntimeError>(1));
    assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    assert_eq!(cb.failure_count().unwrap(), 0);
}

#[test]
fn cb_success_resets_failure_count_to_zero() {
    let cb = CircuitBreaker::new("svc", 5, Duration::from_secs(60)).unwrap();
    for _ in 0..3 {
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("e".to_string()));
    }
    assert_eq!(cb.failure_count().unwrap(), 3);
    let _: Result<i32, AgentRuntimeError> = cb.call(|| Ok::<i32, AgentRuntimeError>(0));
    assert_eq!(cb.failure_count().unwrap(), 0);
}

// ── Shared backend ────────────────────────────────────────────────────────────

#[test]
fn cb_shared_backend_propagates_failures_across_instances() {
    let backend: Arc<dyn agent_runtime::orchestrator::CircuitBreakerBackend> =
        Arc::new(InMemoryCircuitBreakerBackend::new());

    let cb1 = CircuitBreaker::new("svc", 2, Duration::from_secs(60))
        .unwrap()
        .with_backend(Arc::clone(&backend));

    let cb2 = CircuitBreaker::new("svc", 2, Duration::from_secs(60))
        .unwrap()
        .with_backend(Arc::clone(&backend));

    let _: Result<(), AgentRuntimeError> = cb1.call(|| Err::<(), _>("fail".to_string()));
    assert_eq!(cb2.failure_count().unwrap(), 1);

    let _: Result<(), AgentRuntimeError> = cb1.call(|| Err::<(), _>("fail2".to_string()));
    assert!(matches!(cb2.state().unwrap(), CircuitState::Open { .. }));
}

// ── Validation ────────────────────────────────────────────────────────────────

#[test]
fn cb_rejects_zero_threshold() {
    assert!(CircuitBreaker::new("svc", 0, Duration::from_secs(1)).is_err());
}

#[test]
fn cb_accepts_threshold_of_one() {
    assert!(CircuitBreaker::new("svc", 1, Duration::from_secs(1)).is_ok());
}

// ── Large threshold ───────────────────────────────────────────────────────────

#[test]
fn cb_with_large_threshold_only_opens_at_limit() {
    let threshold = 10u32;
    let cb = CircuitBreaker::new("svc", threshold, Duration::from_secs(60)).unwrap();

    for i in 0..threshold - 1 {
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>(format!("err {i}")));
        assert_eq!(cb.state().unwrap(), CircuitState::Closed, "should still be closed at {i}");
    }

    let _: Result<(), AgentRuntimeError> =
        cb.call(|| Err::<(), _>("final err".to_string()));
    assert!(matches!(cb.state().unwrap(), CircuitState::Open { .. }));
}

// ── state() returns Err not panic when open ──────────────────────────────────

#[test]
fn cb_state_returns_ok_not_panics_when_open() {
    let cb = CircuitBreaker::new("svc", 1, Duration::from_secs(3600)).unwrap();
    let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
    // Must return Ok(Open), not panic
    let state = cb.state();
    assert!(state.is_ok());
    assert!(matches!(state.unwrap(), CircuitState::Open { .. }));
}
