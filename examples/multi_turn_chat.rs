//! Example: Multi-Turn Chat
//!
//! Demonstrates a simple multi-turn conversation loop using `AgentRuntime::quick`
//! and `EpisodicStore` to retain context across turns.
//!
//! Each turn:
//! 1. Recalls recent context for the agent from `EpisodicStore`.
//! 2. Runs the agent with the recalled context prepended to the prompt.
//! 3. Stores both the user message and agent reply in `EpisodicStore`.
//!
//! This example does **not** call a real LLM — the inference closure echoes the
//! prompt so the example compiles and runs without any API keys.

use llm_agent_runtime::memory::{AgentId, EpisodicStore};
use llm_agent_runtime::runtime::AgentRuntime;

#[tokio::main]
async fn main() {
    // ── Setup ────────────────────────────────────────────────────────────────
    let runtime = AgentRuntime::quick(5, "echo-model");
    let memory = EpisodicStore::new();
    let agent_id = AgentId::new("chat-agent");

    // Simulated user turns.
    let turns = vec![
        "Hello, who are you?",
        "What can you do?",
        "Tell me a fun fact.",
    ];

    for user_input in turns {
        println!("User: {user_input}");

        // Recall up to 3 recent episodes to provide context.
        let context = memory
            .recall(&agent_id, 3)
            .unwrap_or_default();

        let context_prefix = if context.is_empty() {
            String::new()
        } else {
            let lines = context
                .iter()
                .map(|item| format!("[context] {}", item.content))
                .collect::<Vec<_>>()
                .join("\n");
            format!("{lines}\n\n")
        };

        let full_prompt = format!("{context_prefix}User: {user_input}");

        // Run the agent — the inference closure simply echoes the prompt.
        let session = runtime
            .run_agent(agent_id.clone(), &full_prompt, |ctx| async move {
                format!("Echo: {ctx}")
            })
            .await;

        let reply = match session {
            Ok(ref s) => s
                .steps
                .last()
                .map(|step| step.observation.clone())
                .unwrap_or_else(|| "(no response)".to_string()),
            Err(ref e) => format!("(error: {e})"),
        };

        println!("Agent: {reply}\n");

        // Store user message and agent reply for future context.
        let _ = memory.add_episode(agent_id.clone(), user_input, 0.8);
        let _ = memory.add_episode(agent_id.clone(), &reply, 0.7);
    }
}
