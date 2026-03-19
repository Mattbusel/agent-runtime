//! Example: Streaming Inference
//!
//! Demonstrates `ReActLoop::run_streaming()` with a channel-based
//! inference function that emits token chunks incrementally.

#![allow(clippy::print_stdout)]

use llm_agent_runtime::agent::{AgentConfig, ReActLoop, ToolSpec};
use llm_agent_runtime::error::AgentRuntimeError;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    let config = AgentConfig::new(3, "streaming-model");
    let mut loop_ = ReActLoop::new(config);

    loop_.register_tool(ToolSpec::new(
        "echo",
        "Echoes the input",
        |args| serde_json::json!({ "echoed": args }),
    ));

    let steps = loop_
        .run_streaming("What is the capital of France?", |_ctx: String| async {
            let (tx, rx) = mpsc::channel::<Result<String, AgentRuntimeError>>(16);
            tokio::spawn(async move {
                // Simulate streaming token chunks
                let chunks = ["Thought", ": done\n", "Action", ": FINAL_ANSWER Paris"];
                for chunk in &chunks {
                    let _ = tx.send(Ok(chunk.to_string())).await;
                }
            });
            rx
        })
        .await?;

    println!("Steps: {}", steps.len());
    if let Some(step) = steps.last() {
        println!("Final answer action: {}", step.action);
    }

    Ok(())
}
