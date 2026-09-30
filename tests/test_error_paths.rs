//! Integration tests: error paths — invalid config, provider failure simulation,
//! circuit breaker, backpressure, missing tools, and required-field validation.

use llm_agent_runtime::prelude::*;

// ── Invalid / zero configuration ─────────────────────────────────────────────

#[test]
fn error_working_memory_zero_capacity_is_rejected() {
    let result = WorkingMemory::new(0);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("capacity"),
        "expected capacity error, got: {msg}"
    );
}

#[test]
fn error_decay_policy_zero_half_life_is_rejected() {
    let result = DecayPolicy::exponential(0.0);
    assert!(result.is_err());
}

#[test]
fn error_decay_policy_negative_half_life_is_rejected() {
    let result = DecayPolicy::exponential(-5.0);
    assert!(result.is_err());
}

#[test]
fn error_backpressure_guard_zero_capacity_is_rejected() {
    let result = BackpressureGuard::new(0);
    assert!(result.is_err());
}

#[test]
fn error_retry_policy_zero_attempts_is_rejected() {
    let result = RetryPolicy::exponential(0, 100);
    assert!(result.is_err());
}

// ── Backpressure shedding ─────────────────────────────────────────────────────

#[tokio::test]
async fn error_backpressure_shed_when_at_capacity() {
    let guard = BackpressureGuard::new(1).unwrap();
    guard.try_acquire().unwrap(); // pre-fill to capacity

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .with_backpressure(guard)
        .build();

    let result = runtime
        .run_agent(
            AgentId::new("shed-agent"),
            "will be shed",
            |_ctx: String| async { "Thought: done\nAction: FINAL_ANSWER ok".to_string() },
        )
        .await;

    assert!(
        matches!(result, Err(AgentRuntimeError::BackpressureShed { .. })),
        "expected BackpressureShed, got: {:?}",
        result
    );
}

#[tokio::test]
async fn error_backpressure_shed_increments_metric() {
    let guard = BackpressureGuard::new(1).unwrap();
    guard.try_acquire().unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .with_backpressure(guard)
        .build();

    let _ = runtime
        .run_agent(AgentId::new("a"), "prompt", |_ctx: String| async {
            "Thought: done\nAction: FINAL_ANSWER ok".to_string()
        })
        .await;

    assert_eq!(runtime.metrics().backpressure_shed_count(), 1);
}

// ── Max iterations exceeded ───────────────────────────────────────────────────

#[tokio::test]
async fn error_max_iterations_returns_agent_loop_error() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(2, "test-model"))
        .register_tool(ToolSpec::new("loop", "loops", |_| serde_json::Value::Null))
        .build();

    let result = runtime
        .run_agent(
            AgentId::new("infinite"),
            "never answer",
            |_ctx: String| async { "Thought: again\nAction: loop {}".to_string() },
        )
        .await;

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("max iterations"),
        "expected max iterations error, got: {err_str}"
    );
}

// ── Tool not found ────────────────────────────────────────────────────────────

#[tokio::test]
async fn error_missing_tool_produces_error_observation_not_panic() {
    // Calling an unregistered tool should not crash the runtime — instead it
    // should return an error observation and allow the loop to continue.
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .build();

    let mut call_n = 0u32;
    let session = runtime
        .run_agent(
            AgentId::new("tool-err-agent"),
            "try missing tool",
            move |_ctx: String| {
                call_n += 1;
                let n = call_n;
                async move {
                    if n == 1 {
                        "Thought: try ghost tool\nAction: ghost_tool {}".to_string()
                    } else {
                        "Thought: handled error\nAction: FINAL_ANSWER recovered".to_string()
                    }
                }
            },
        )
        .await
        .unwrap();

    // Step 1 should have an error observation.
    assert!(session.step_count() >= 1);
    let obs = &session.steps[0].observation;
    assert!(
        obs.contains("\"ok\":false"),
        "observation should contain error flag, got: {obs}"
    );
}

// ── Required-field validation ─────────────────────────────────────────────────

#[tokio::test]
async fn error_missing_required_field_produces_structured_observation() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .register_tool(
            ToolSpec::new(
                "search",
                "web search",
                |args| serde_json::json!({ "results": args }),
            )
            .with_required_fields(vec!["query".to_string()]),
        )
        .build();

    let mut call_n = 0u32;
    let session = runtime
        .run_agent(
            AgentId::new("req-field-agent"),
            "search without required field",
            move |_ctx: String| {
                call_n += 1;
                let n = call_n;
                async move {
                    if n == 1 {
                        // Missing the required "query" field.
                        r#"Thought: search
Action: search {}"#
                            .to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            },
        )
        .await
        .unwrap();

    let obs = &session.steps[0].observation;
    assert!(
        obs.contains("missing required field"),
        "expected missing-field error in observation, got: {obs}"
    );
}

// ── Provider failure simulation ───────────────────────────────────────────────

#[tokio::test]
async fn error_provider_failure_propagated_as_agent_loop_error() {
    // Simulate a provider that returns an unparseable response on every call.
    // The runtime should propagate an AgentLoop error rather than panic.
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(1, "test-model"))
        .build();

    let result = runtime
        .run_agent(
            AgentId::new("bad-provider-agent"),
            "use the model",
            |_ctx: String| async {
                // Returns a response with no recognisable Thought:/Action: lines.
                "Sorry, the service is unavailable.".to_string()
            },
        )
        .await;

    // Either the loop fails to parse the step, or it runs out of iterations.
    assert!(
        result.is_err(),
        "expected an error when the provider returns unparseable output"
    );
}

// ── Circuit breaker (orchestrator feature) ────────────────────────────────────

#[cfg(feature = "orchestrator")]
#[tokio::test]
async fn error_circuit_open_fast_fails_tool() {
    use std::sync::Arc;

    let cb = Arc::new(
        CircuitBreaker::new("failing-service", 1, std::time::Duration::from_secs(60)).unwrap(),
    );

    // Trip the circuit by executing a failing call (threshold=1, so one failure opens it).
    let _: Result<(), _> = cb.call(|| Err::<(), &str>("simulated failure"));
    // After 1 failure with threshold=1 the circuit should be open.

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .register_tool(
            ToolSpec::new("remote", "calls a remote service", |_| {
                serde_json::Value::Null
            })
            .with_circuit_breaker(Arc::clone(&cb)),
        )
        .build();

    let mut call_n = 0u32;
    let session = runtime
        .run_agent(
            AgentId::new("cb-agent"),
            "use remote service",
            move |_ctx: String| {
                call_n += 1;
                let n = call_n;
                async move {
                    if n == 1 {
                        "Thought: call remote\nAction: remote {}".to_string()
                    } else {
                        "Thought: handled\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            },
        )
        .await
        .unwrap();

    // The circuit-open error should appear in the observation of step 1.
    let obs = &session.steps[0].observation;
    assert!(
        obs.contains("\"ok\":false"),
        "expected circuit-open error observation, got: {obs}"
    );
}

// ── Loop timeout ──────────────────────────────────────────────────────────────

/// Verify that `with_loop_timeout_ms` causes the ReAct loop to return an error
/// when iterations collectively exceed the configured wall-clock deadline.
///
/// The deadline is checked at the top of each iteration. Each inference call
/// sleeps 15 ms without returning `FINAL_ANSWER`, so after two iterations the
/// cumulative time exceeds the 20 ms deadline.
#[tokio::test]
async fn error_loop_timeout_fires_when_inference_is_slow() {
    use llm_agent_runtime::agent::{AgentConfig, ReActLoop};

    let config = AgentConfig::new(100, "timeout-model").with_loop_timeout_ms(20);
    let loop_ = ReActLoop::new(config);

    let result = loop_
        .run("hello", |_ctx| async {
            // Sleep per iteration so cumulative time exceeds the deadline.
            tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
            // Keep the loop going (not FINAL_ANSWER); unknown tool produces an error observation.
            "Thought: still working\nAction: unknown_tool {}".to_string()
        })
        .await;

    assert!(result.is_err(), "expected timeout error, got Ok");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("timeout"),
        "error message should mention timeout; was: {msg}"
    );
}
