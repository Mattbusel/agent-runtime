//! Property-based tests using `proptest`.
//!
//! These tests verify algebraic / invariant properties of core algorithms
//! rather than specific input/output pairs:
//!
//! - `DecayPolicy::apply` always returns a value in `[0.0, 1.0]`.
//! - `cosine_similarity` (via the public SemanticStore API) always returns a
//!   value in `[-1.0, 1.0]`.
//! - `RetryPolicy::delay_for` is monotonically non-decreasing and never exceeds
//!   `MAX_RETRY_DELAY`.
//! - The circuit breaker reaches the `Open` state after exactly `threshold`
//!   consecutive failures.

use llm_agent_runtime::{
    memory::DecayPolicy,
    orchestrator::{CircuitBreaker, CircuitState, RetryPolicy, MAX_RETRY_DELAY},
};
use proptest::prelude::*;
use std::time::Duration;

// ── DecayPolicy ───────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn decay_result_is_always_in_unit_interval(
        importance in 0.0f32..=1.0f32,
        age_hours in 0.0f64..1_000_000.0f64,
        half_life in 0.001f64..10_000.0f64,
    ) {
        let policy = DecayPolicy::exponential(half_life).unwrap();
        let decayed = policy.apply(importance, age_hours);
        prop_assert!(
            decayed >= 0.0 && decayed <= 1.0,
            "decayed={decayed} out of [0,1]: importance={importance}, age={age_hours}, half_life={half_life}"
        );
    }
}

proptest! {
    #[test]
    fn decay_is_monotonically_decreasing_with_age(
        importance in 0.01f32..=1.0f32,
        age1 in 0.0f64..500.0f64,
        age2 in 501.0f64..1000.0f64,
        half_life in 0.1f64..100.0f64,
    ) {
        let policy = DecayPolicy::exponential(half_life).unwrap();
        let d1 = policy.apply(importance, age1);
        let d2 = policy.apply(importance, age2);
        prop_assert!(
            d1 >= d2,
            "decay should be non-increasing: age1={age1} -> {d1}, age2={age2} -> {d2}"
        );
    }
}

proptest! {
    #[test]
    fn decay_at_zero_age_equals_importance(
        importance in 0.0f32..=1.0f32,
        half_life in 0.1f64..1000.0f64,
    ) {
        let policy = DecayPolicy::exponential(half_life).unwrap();
        let decayed = policy.apply(importance, 0.0);
        prop_assert!(
            (decayed - importance).abs() < 1e-5,
            "at age=0 decayed={decayed} should equal importance={importance}"
        );
    }
}

// ── RetryPolicy ───────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn retry_delay_never_exceeds_cap(
        attempt in 1u32..=64u32,
        base_ms in 1u64..=1000u64,
    ) {
        let policy = RetryPolicy::exponential(64, base_ms).unwrap();
        let delay = policy.delay_for(attempt);
        prop_assert!(
            delay <= MAX_RETRY_DELAY,
            "delay={delay:?} exceeds MAX_RETRY_DELAY={MAX_RETRY_DELAY:?}"
        );
    }
}

proptest! {
    #[test]
    fn retry_delay_is_non_decreasing(
        attempt in 1u32..=31u32,
        base_ms in 1u64..=100u64,
    ) {
        let policy = RetryPolicy::exponential(64, base_ms).unwrap();
        let d1 = policy.delay_for(attempt);
        let d2 = policy.delay_for(attempt + 1);
        prop_assert!(
            d2 >= d1,
            "delay decreased: attempt {attempt} -> {d1:?}, attempt {} -> {d2:?}",
            attempt + 1
        );
    }
}

proptest! {
    #[test]
    fn retry_delay_attempt_1_equals_base_delay(
        base_ms in 1u64..=60_000u64,
    ) {
        let policy = RetryPolicy::exponential(5, base_ms).unwrap();
        prop_assert_eq!(policy.delay_for(1), Duration::from_millis(base_ms));
    }
}

// ── CircuitBreaker ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn circuit_opens_after_threshold_consecutive_failures(
        threshold in 1u32..=20u32,
    ) {
        let cb = CircuitBreaker::new("test-service", threshold, Duration::from_secs(60))
            .unwrap();

        // Trigger exactly threshold failures.
        for _ in 0..threshold {
            let _ = cb.call(|| -> Result<(), &str> { Err("fail") });
        }

        let state = cb.state().unwrap();
        prop_assert!(
            matches!(state, CircuitState::Open { .. }),
            "expected Open after {threshold} failures, got {state:?}"
        );
    }
}

proptest! {
    #[test]
    fn circuit_stays_closed_below_threshold(
        threshold in 2u32..=20u32,
    ) {
        let cb = CircuitBreaker::new("test-service", threshold, Duration::from_secs(60))
            .unwrap();

        // Trigger threshold - 1 failures (one below the trip point).
        for _ in 0..(threshold - 1) {
            let _ = cb.call(|| -> Result<(), &str> { Err("fail") });
        }

        let state = cb.state().unwrap();
        prop_assert!(
            !matches!(state, CircuitState::Open { .. }),
            "circuit should not open below threshold: {threshold} threshold, got {state:?}"
        );
    }
}

proptest! {
    #[test]
    fn circuit_resets_after_success(
        threshold in 1u32..=10u32,
    ) {
        let cb = CircuitBreaker::new("test-service", threshold, Duration::from_millis(1))
            .unwrap();

        // Open the circuit.
        for _ in 0..threshold {
            let _ = cb.call(|| -> Result<(), &str> { Err("fail") });
        }

        // Wait for recovery window then send a success.
        std::thread::sleep(Duration::from_millis(10));
        let _ = cb.call(|| -> Result<(), &str> { Ok(()) });

        let state = cb.state().unwrap();
        prop_assert!(
            matches!(state, CircuitState::Closed),
            "circuit should be Closed after success probe, got {state:?}"
        );
    }
}
