//! Example: Resilient Tool with Circuit Breaker and Retry
//!
//! Demonstrates how to attach a per-tool circuit breaker so that a flaky
//! external service automatically stops being called after repeated failures,
//! and how to combine that with a `RetryPolicy` in the agent loop.
//!
//! Run with:
//! ```
//! cargo run --example resilient_tool --features orchestrator
//! ```

use llm_agent_runtime::agent::{AgentConfig, ReActLoop, ToolSpec};
use llm_agent_runtime::error::AgentRuntimeError;
use llm_agent_runtime::orchestrator::{CircuitBreaker, RetryPolicy};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // ── Set up a flaky tool that fails the first 2 times ─────────────────────

    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = Arc::clone(&call_count);

    // Circuit breaker: open after 3 consecutive failures, recover after 1 second.
    let circuit_breaker = Arc::new(
        CircuitBreaker::new("weather-api", 3, Duration::from_secs(1)).unwrap(),
    );

    let tool = ToolSpec::new_async(
        "get_weather",
        "Fetch current weather for a city. Required field: city",
        move |args| {
            let count = Arc::clone(&call_count_clone);
            Box::pin(async move {
                let n = count.fetch_add(1, Ordering::Relaxed);
                if n < 2 {
                    // Simulate first two calls failing
                    serde_json::json!({ "error": "service temporarily unavailable" })
                } else {
                    let city = args
                        .get("city")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    serde_json::json!({ "city": city, "temp_c": 22, "condition": "sunny" })
                }
            })
        },
    )
    .with_required_fields(vec!["city".to_string()])
    .with_circuit_breaker(circuit_breaker);

    // ── Build the agent loop ──────────────────────────────────────────────────

    let config = AgentConfig::new(6, "my-model");
    let mut loop_ = ReActLoop::new(config);
    loop_.register_tool(tool);

    // ── Retry policy: try up to 4 times with 10 ms base delay ────────────────

    let retry = RetryPolicy::exponential(4, 10).unwrap();

    // ── Run the agent ─────────────────────────────────────────────────────────

    let mut attempt = 0u32;
    let steps = loop {
        attempt += 1;
        let result = loop_
            .run("What is the weather in London?", |_ctx: String| async {
                // Simulate the model's response changing on each attempt
                if call_count.load(Ordering::Relaxed) < 2 {
                    "Thought: fetch weather\nAction: get_weather {\"city\":\"London\"}".to_string()
                } else {
                    "Thought: got weather\nAction: FINAL_ANSWER It is sunny and 22°C in London.".to_string()
                }
            })
            .await;

        match result {
            Ok(s) => break s,
            Err(e) => {
                if attempt >= retry.max_attempts {
                    return Err(e);
                }
                let delay = retry.delay_for(attempt);
                println!("[attempt {attempt}] error: {e} — retrying after {delay:?}");
                tokio::time::sleep(delay).await;
            }
        }
    };

    println!("Completed in {} steps ({attempt} attempt(s))", steps.len());
    for (i, step) in steps.iter().enumerate() {
        println!(
            "  Step {}: thought={:?} action={:?}",
            i + 1,
            step.thought,
            step.action
        );
    }

    Ok(())
}
