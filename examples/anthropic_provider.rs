//! Example: End-to-end wiring of `AnthropicProvider` with `AgentRuntime`
//!
//! This example shows the full integration path:
//!
//! 1. Create an `AnthropicProvider` with a real API key (or a test key).
//! 2. Wire it into `AgentRuntime` so the agent loop calls the Anthropic
//!    Messages API for every inference step.
//! 3. Register a simple "calculator" tool.
//! 4. Run the agent.
//!
//! # Running
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_provider \
//!     --features anthropic,memory
//! ```
//!
//! If `ANTHROPIC_API_KEY` is not set the example exits with a clear message.
//!
//! # Note
//!
//! The `infer` closure passed to `run_agent` is where you inject your LLM
//! provider.  In production, call `provider.complete(context, model).await`
//! and propagate errors gracefully.  The example uses `unwrap_or_default` so
//! the code stays readable; real code should handle errors explicitly.

use llm_agent_runtime::{
    agent::{AgentConfig, ToolSpec},
    memory::AgentId,
    runtime::AgentRuntime,
};
use std::sync::Arc;

#[cfg(feature = "anthropic")]
use llm_agent_runtime::providers::{AnthropicProvider, LlmProvider};

#[tokio::main]
async fn main() {
    // ── API key ───────────────────────────────────────────────────────────────
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "ANTHROPIC_API_KEY is not set.  Set it and re-run:\n\
                 ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_provider \
                 --features anthropic,memory"
            );
            std::process::exit(1);
        }
    };

    // ── Provider ──────────────────────────────────────────────────────────────
    #[cfg(feature = "anthropic")]
    let provider = Arc::new(AnthropicProvider::new(api_key));

    #[cfg(not(feature = "anthropic"))]
    {
        eprintln!("Enable the `anthropic` feature to run this example.");
        std::process::exit(1);
    }

    // ── Tool: simple calculator ───────────────────────────────────────────────
    let calc_tool = ToolSpec::new("calculator", "Evaluate a simple math expression: {\"expr\": \"2 + 2\"}", |args| {
        let expr = args["expr"].as_str().unwrap_or("0");
        // Very limited: only handles `a OP b` for demo purposes.
        let parts: Vec<&str> = expr.split_whitespace().collect();
        let result = if parts.len() == 3 {
            let a: f64 = parts[0].parse().unwrap_or(0.0);
            let b: f64 = parts[2].parse().unwrap_or(0.0);
            match parts[1] {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" if b != 0.0 => a / b,
                _ => f64::NAN,
            }
        } else {
            f64::NAN
        };
        serde_json::json!({ "result": result })
    });

    // ── Runtime ───────────────────────────────────────────────────────────────
    let config = AgentConfig::new(6, "claude-sonnet-4-6")
        .with_system_prompt(
            "You are a helpful math assistant.  \
             Use the `calculator` tool for arithmetic, then report FINAL_ANSWER.",
        )
        .with_loop_timeout_secs(30);

    let runtime = AgentRuntime::builder()
        .with_agent_config(config)
        .register_tool(calc_tool)
        .build();

    // ── Inference closure (calls Anthropic) ───────────────────────────────────
    #[cfg(feature = "anthropic")]
    let session = {
        let provider_ref = Arc::clone(&provider);
        runtime
            .run_agent(
                AgentId::new("math-agent"),
                "What is 137 * 42?",
                move |context: String| {
                    let p = Arc::clone(&provider_ref);
                    async move {
                        p.complete(&context, "claude-sonnet-4-6")
                            .await
                            .unwrap_or_else(|e| format!("Error: {e}"))
                    }
                },
            )
            .await
    };

    match session {
        Ok(s) => {
            println!("Completed in {} step(s), {}ms", s.step_count(), s.duration_ms);
            if let Some(answer) = s.final_answer() {
                println!("Answer: {answer}");
            }
        }
        Err(e) => eprintln!("Agent error: {e}"),
    }
}
