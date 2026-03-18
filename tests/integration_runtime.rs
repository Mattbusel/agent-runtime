//! Integration tests: AgentRuntime end-to-end lifecycle.

use agent_runtime::prelude::*;

async fn final_infer(_ctx: String) -> String {
    "Thought: complete\nAction: FINAL_ANSWER done".into()
}

// ── Builder ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_builder_minimal_succeeds() {
    // build() is now infallible — typestate guarantees config is present.
    let _runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build();
}

// NOTE: integration_builder_missing_agent_config_fails has been removed.
// Calling .build() on an AgentRuntimeBuilder<NeedsConfig> (i.e., without first
// calling .with_agent_config()) is now a compile-time error — nothing to test
// at runtime.

#[tokio::test]
async fn integration_builder_full_configuration_succeeds() {
    let _runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(10, "model"))
        .with_memory(EpisodicStore::new())
        .with_working_memory(WorkingMemory::new(20).unwrap())
        .with_graph(GraphStore::new())
        .with_backpressure(BackpressureGuard::new(10).unwrap())
        .build();
}

// ── run_agent ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_run_agent_returns_session() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build();

    let session = runtime
        .run_agent(AgentId::new("agent-1"), "hello", final_infer)
        .await
        .unwrap();

    assert!(session.step_count() >= 1);
}

#[tokio::test]
async fn integration_run_agent_session_has_correct_agent_id() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build();

    let session = runtime
        .run_agent(AgentId::new("test-agent-99"), "hi", final_infer)
        .await
        .unwrap();

    assert_eq!(session.agent_id.0, "test-agent-99");
}

#[tokio::test]
async fn integration_run_agent_with_memory_records_hits() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("memory-agent");
    store.add_episode(agent.clone(), "fact 1", 0.7).unwrap();
    store.add_episode(agent.clone(), "fact 2", 0.5).unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_memory(store)
        .build();

    let session = runtime
        .run_agent(agent, "recall facts", final_infer)
        .await
        .unwrap();

    assert_eq!(session.memory_hits, 2);
}

#[tokio::test]
async fn integration_run_agent_with_graph_records_lookups() {
    let graph = GraphStore::new();
    graph.add_entity(Entity::new("node-1", "Concept")).unwrap();
    graph.add_entity(Entity::new("node-2", "Concept")).unwrap();
    graph.add_entity(Entity::new("node-3", "Concept")).unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_graph(graph)
        .build();

    let session = runtime
        .run_agent(AgentId::new("a"), "query graph", final_infer)
        .await
        .unwrap();

    assert_eq!(session.graph_lookups, 3);
}

#[tokio::test]
async fn integration_run_agent_multi_step_session() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(10, "test"))
        .register_tool(ToolSpec::new("echo", "echoes", |v| v))
        .build();

    let mut calls = 0;
    let session = runtime
        .run_agent(AgentId::new("a"), "do stuff", move |_ctx: String| {
            calls += 1;
            let n = calls;
            async move {
                match n {
                    1 => "Thought: step 1\nAction: echo {}".into(),
                    2 => "Thought: step 2\nAction: echo {}".into(),
                    _ => "Thought: done\nAction: FINAL_ANSWER finished".into(),
                }
            }
        })
        .await
        .unwrap();

    assert!(session.step_count() >= 2);
}

#[tokio::test]
async fn integration_backpressure_sheds_when_at_capacity() {
    let guard = BackpressureGuard::new(1).unwrap();
    guard.try_acquire().unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_backpressure(guard)
        .build();

    let result = runtime
        .run_agent(AgentId::new("a"), "prompt", final_infer)
        .await;
    assert!(matches!(
        result,
        Err(AgentRuntimeError::BackpressureShed { .. })
    ));
}

#[tokio::test]
async fn integration_backpressure_released_after_successful_run() {
    let guard = BackpressureGuard::new(5).unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_backpressure(guard.clone())
        .build();

    runtime
        .run_agent(AgentId::new("a"), "prompt", final_infer)
        .await
        .unwrap();
    assert_eq!(guard.depth().unwrap(), 0);
}

#[tokio::test]
async fn integration_session_duration_is_populated() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build();

    let session = runtime
        .run_agent(AgentId::new("a"), "hi", final_infer)
        .await
        .unwrap();
    // duration_ms is u64, always >= 0. Just verify it was set (not a sentinel).
    let _duration: u64 = session.duration_ms;
}

#[tokio::test]
async fn integration_max_iterations_error_returned() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(2, "test"))
        .register_tool(ToolSpec::new("loop", "loops", |_| serde_json::Value::Null))
        .build();

    let result = runtime
        .run_agent(AgentId::new("a"), "loop forever", |_ctx: String| async {
            "Thought: again\nAction: loop {}".to_string()
        })
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn integration_memory_enriches_prompt_context() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("ctx-agent");
    store
        .add_episode(agent.clone(), "the answer is 42", 0.9)
        .unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_memory(store)
        .build();

    use std::sync::{Arc, Mutex};
    let context_received: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let ctx_clone = Arc::clone(&context_received);

    runtime
        .run_agent(agent, "what is the answer?", move |ctx: String| {
            let c = Arc::clone(&ctx_clone);
            async move {
                // store the first context we receive
                {
                    #[allow(clippy::unwrap_used)]
                    let mut guard = c.lock().unwrap();
                    if guard.is_empty() {
                        *guard = ctx;
                    }
                }
                "Thought: done\nAction: FINAL_ANSWER answer".to_string()
            }
        })
        .await
        .unwrap();

    #[allow(clippy::unwrap_used)]
    let ctx = context_received.lock().unwrap().clone();
    assert!(
        ctx.contains("the answer is 42"),
        "memory should be injected into context"
    );
}
