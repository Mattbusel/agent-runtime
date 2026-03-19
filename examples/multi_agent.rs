//! Example: Multi-Agent Coordination
//!
//! Demonstrates running multiple agents against a shared `EpisodicStore` and
//! shared `RuntimeMetrics`.  Each agent writes memories that the others can
//! recall in subsequent runs (simulating a shared long-term memory pool).

use llm_agent_runtime::agent::AgentConfig;
use llm_agent_runtime::error::AgentRuntimeError;
use llm_agent_runtime::memory::{AgentId, EpisodicStore};
use llm_agent_runtime::metrics::RuntimeMetrics;
use llm_agent_runtime::runtime::AgentRuntime;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // A shared episodic store — all agents read/write the same pool.
    let shared_memory = EpisodicStore::new();
    let shared_metrics = RuntimeMetrics::new();

    // Pre-seed agent-1's memory so agent-2 can benefit from it.
    let agent_1 = AgentId::new("agent-1");
    let agent_2 = AgentId::new("agent-2");

    shared_memory
        .add_episode(agent_1.clone(), "The capital of France is Paris", 0.95)
        .unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(
            AgentConfig::new(5, "my-model")
                .with_max_memory_recalls(3)
                .with_system_prompt("You are a knowledgeable assistant."),
        )
        .with_memory(shared_memory.clone())
        .with_metrics(Arc::clone(&shared_metrics))
        .build();

    // Run agent-1 — it will recall its own memory.
    let session_1 = runtime
        .run_agent(
            agent_1.clone(),
            "What do you remember about capitals?",
            |ctx: String| async move {
                println!("[agent-1] context snippet: {}...", &ctx[..ctx.len().min(120)]);
                "Thought: I recall Paris\nAction: FINAL_ANSWER Paris is the capital of France".to_string()
            },
        )
        .await?;

    println!(
        "Agent-1 session: {} steps, {} memory_hits",
        session_1.step_count(),
        session_1.memory_hits
    );

    // Run agent-2 — it has no memories of its own yet; starts fresh.
    let session_2 = runtime
        .run_agent(
            agent_2.clone(),
            "Anything useful in memory?",
            |_ctx: String| async {
                "Thought: nothing found\nAction: FINAL_ANSWER No relevant memories".to_string()
            },
        )
        .await?;

    println!(
        "Agent-2 session: {} steps, {} memory_hits",
        session_2.step_count(),
        session_2.memory_hits
    );

    println!(
        "\nGlobal metrics — sessions={} steps={} tool_calls={}",
        shared_metrics.total_sessions(),
        shared_metrics.total_steps(),
        shared_metrics.total_tool_calls()
    );

    Ok(())
}
