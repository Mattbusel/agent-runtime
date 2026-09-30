//! Example: Working Memory Evolution
//!
//! Demonstrates how `WorkingMemory` can track transient agent state across
//! multiple sessions on the same runtime, and how its bounded LRU eviction
//! keeps memory usage constant.
//!
//! Run with:
//! ```
//! cargo run --example working_memory_evolution --features memory
//! ```

use llm_agent_runtime::agent::AgentConfig;
use llm_agent_runtime::error::AgentRuntimeError;
use llm_agent_runtime::memory::{AgentId, WorkingMemory};
use llm_agent_runtime::runtime::AgentRuntime;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // ── Bounded working memory: holds at most 4 keys ──────────────────────────

    let wm = WorkingMemory::new(4).unwrap();

    // Pre-populate some state before the agent runs.
    wm.set("task", "Summarise quarterly report").unwrap();
    wm.set("status", "pending").unwrap();
    wm.set("priority", "high").unwrap();

    println!("Initial working memory ({} entries):", wm.len().unwrap());
    for (k, v) in wm.entries().unwrap() {
        println!("  {k} = {v}");
    }

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(3, "my-model"))
        .with_working_memory(wm.clone())
        .build();

    // ── First run: agent sees the pre-populated state ─────────────────────────

    let session = runtime
        .run_agent(
            AgentId::new("analyst"),
            "What is my current task?",
            |ctx: String| async move {
                println!("\n[run-1] context snippet:\n{}\n", &ctx[..ctx.len().min(300)]);
                "Thought: I can see the task in working state\nAction: FINAL_ANSWER Summarise quarterly report (high priority)".to_string()
            },
        )
        .await?;

    println!("Run-1 completed: {} step(s)", session.step_count());

    // ── Update working memory between runs ───────────────────────────────────

    wm.set("status", "in-progress").unwrap();
    wm.set("progress_pct", "30").unwrap();

    // Adding a 5th key evicts the oldest one (LRU: "task" was inserted first).
    println!(
        "\nAfter update ({} entries):",
        wm.len().unwrap()
    );
    for (k, v) in wm.entries().unwrap() {
        println!("  {k} = {v}");
    }

    // ── Second run: agent sees updated state ──────────────────────────────────

    let session2 = runtime
        .run_agent(
            AgentId::new("analyst"),
            "How far along am I?",
            |ctx: String| async move {
                println!("\n[run-2] context snippet:\n{}\n", &ctx[..ctx.len().min(300)]);
                "Thought: progress is visible\nAction: FINAL_ANSWER 30% complete".to_string()
            },
        )
        .await?;

    println!("Run-2 completed: {} step(s)", session2.step_count());

    Ok(())
}
