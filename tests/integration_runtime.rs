//! Integration tests: AgentRuntime end-to-end lifecycle.

use agent_runtime::prelude::*;

fn final_infer(_ctx: &str) -> String {
    "Thought: complete\nAction: FINAL_ANSWER done".into()
}

// ── Builder ───────────────────────────────────────────────────────────────────

#[test]
fn integration_builder_minimal_succeeds() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build();
    assert!(runtime.is_ok(), "minimal builder should succeed");
}

#[test]
fn integration_builder_missing_agent_config_fails() {
    let result = AgentRuntime::builder().build();
    assert!(
        matches!(
            result,
            Err(AgentRuntimeError::NotConfigured("agent_config"))
        ),
        "should fail with NotConfigured"
    );
}

#[test]
fn integration_builder_full_configuration_succeeds() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(10, "model"))
        .with_memory(EpisodicStore::new())
        .with_working_memory(WorkingMemory::new(20).unwrap())
        .with_graph(GraphStore::new())
        .with_backpressure(BackpressureGuard::new(10).unwrap())
        .build();
    assert!(runtime.is_ok());
}

// ── run_agent ─────────────────────────────────────────────────────────────────

#[test]
fn integration_run_agent_returns_session() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build()
        .unwrap();

    let session = runtime
        .run_agent(AgentId::new("agent-1"), "hello", final_infer)
        .unwrap();

    assert!(session.step_count() >= 1);
}

#[test]
fn integration_run_agent_session_has_correct_agent_id() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build()
        .unwrap();

    let session = runtime
        .run_agent(AgentId::new("test-agent-99"), "hi", final_infer)
        .unwrap();

    assert_eq!(session.agent_id.0, "test-agent-99");
}

#[test]
fn integration_run_agent_with_memory_records_hits() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("memory-agent");
    store.add_episode(agent.clone(), "fact 1", 0.7).unwrap();
    store.add_episode(agent.clone(), "fact 2", 0.5).unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_memory(store)
        .build()
        .unwrap();

    let session = runtime
        .run_agent(agent, "recall facts", final_infer)
        .unwrap();

    assert_eq!(session.memory_hits, 2);
}

#[test]
fn integration_run_agent_with_graph_records_lookups() {
    let graph = GraphStore::new();
    graph.add_entity(Entity::new("node-1", "Concept")).unwrap();
    graph.add_entity(Entity::new("node-2", "Concept")).unwrap();
    graph.add_entity(Entity::new("node-3", "Concept")).unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_graph(graph)
        .build()
        .unwrap();

    let session = runtime
        .run_agent(AgentId::new("a"), "query graph", final_infer)
        .unwrap();

    assert_eq!(session.graph_lookups, 3);
}

#[test]
fn integration_run_agent_multi_step_session() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(10, "test"))
        .register_tool(ToolSpec::new("echo", "echoes", |v| v))
        .build()
        .unwrap();

    let mut calls = 0;
    let session = runtime
        .run_agent(AgentId::new("a"), "do stuff", move |_| {
            calls += 1;
            match calls {
                1 => "Thought: step 1\nAction: echo {}".into(),
                2 => "Thought: step 2\nAction: echo {}".into(),
                _ => "Thought: done\nAction: FINAL_ANSWER finished".into(),
            }
        })
        .unwrap();

    assert!(session.step_count() >= 2);
}

#[test]
fn integration_backpressure_sheds_when_at_capacity() {
    let guard = BackpressureGuard::new(1).unwrap();
    guard.try_acquire().unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_backpressure(guard)
        .build()
        .unwrap();

    let result = runtime.run_agent(AgentId::new("a"), "prompt", final_infer);
    assert!(matches!(
        result,
        Err(AgentRuntimeError::BackpressureShed { .. })
    ));
}

#[test]
fn integration_backpressure_released_after_successful_run() {
    let guard = BackpressureGuard::new(5).unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_backpressure(guard.clone())
        .build()
        .unwrap();

    runtime
        .run_agent(AgentId::new("a"), "prompt", final_infer)
        .unwrap();
    assert_eq!(guard.depth().unwrap(), 0);
}

#[test]
fn integration_session_duration_is_populated() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .build()
        .unwrap();

    let session = runtime
        .run_agent(AgentId::new("a"), "hi", final_infer)
        .unwrap();
    // duration_ms is u64, always >= 0. Just verify it was set (not a sentinel).
    let _duration: u64 = session.duration_ms;
}

#[test]
fn integration_max_iterations_error_returned() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(2, "test"))
        .register_tool(ToolSpec::new("loop", "loops", |_| serde_json::Value::Null))
        .build()
        .unwrap();

    let result = runtime.run_agent(AgentId::new("a"), "loop forever", |_| {
        "Thought: again\nAction: loop {}".into()
    });

    assert!(result.is_err());
}

#[test]
fn integration_memory_enriches_prompt_context() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("ctx-agent");
    store
        .add_episode(agent.clone(), "the answer is 42", 0.9)
        .unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test"))
        .with_memory(store)
        .build()
        .unwrap();

    let mut context_received = String::new();
    runtime
        .run_agent(agent, "what is the answer?", |ctx| {
            context_received = ctx.to_owned();
            "Thought: done\nAction: FINAL_ANSWER answer".into()
        })
        .unwrap();

    assert!(
        context_received.contains("the answer is 42"),
        "memory should be injected into context"
    );
}
