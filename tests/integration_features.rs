//! Integration tests: cross-feature interactions and edge cases.

use agent_runtime::prelude::*;

// ── Memory × Graph ────────────────────────────────────────────────────────────

#[test]
fn integration_memory_and_graph_coexist_in_runtime() {
    let store = EpisodicStore::new();
    let graph = GraphStore::new();

    graph.add_entity(Entity::new("rust", "Language")).unwrap();
    graph.add_entity(Entity::new("tokio", "Runtime")).unwrap();
    graph.add_relationship(Relationship::new("tokio", "rust", "IMPLEMENTED_IN", 1.0)).unwrap();

    let agent = AgentId::new("cross-agent");
    store.add_episode(agent.clone(), "Rust is fast", 0.9).unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "model"))
        .with_memory(store)
        .with_graph(graph)
        .build()
        .unwrap();

    let session = runtime.run_agent(agent, "what do you know?", |_| {
        "Thought: done\nAction: FINAL_ANSWER ok".into()
    }).unwrap();

    assert_eq!(session.memory_hits, 1);
    assert_eq!(session.graph_lookups, 2);
}

// ── Orchestrator features ─────────────────────────────────────────────────────

#[test]
fn integration_pipeline_transforms_input() {
    let p = Pipeline::new()
        .add_stage("trim", |s| Ok(s.trim().to_owned()))
        .add_stage("lower", |s| Ok(s.to_lowercase()))
        .add_stage("exclaim", |s| Ok(format!("{s}!")));

    let result = p.run("  HELLO WORLD  ".into()).unwrap();
    assert_eq!(result, "hello world!");
}

#[test]
fn integration_circuit_breaker_and_retry_policy_combined() {
    let cb = CircuitBreaker::new("llm", 3, std::time::Duration::from_secs(60)).unwrap();
    let policy = RetryPolicy::exponential(3, 10).unwrap();

    let mut attempt = 0u32;
    let mut last_delay = std::time::Duration::ZERO;

    for _ in 0..2 {
        attempt += 1;
        last_delay = policy.delay_for(attempt);
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("transient".to_string()));
    }

    // After 2 failures, breaker should still be closed (threshold=3)
    assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    // Delay should have grown
    assert!(last_delay >= std::time::Duration::from_millis(20));
}

#[test]
fn integration_deduplicator_caches_across_complete() {
    let dedup = Deduplicator::new(std::time::Duration::from_secs(60));

    let r1 = dedup.check_and_register("prompt-hash-abc").unwrap();
    assert_eq!(r1, DeduplicationResult::New);

    dedup.complete("prompt-hash-abc", "cached-response").unwrap();

    let r2 = dedup.check_and_register("prompt-hash-abc").unwrap();
    assert_eq!(r2, DeduplicationResult::Cached("cached-response".into()));
}

#[test]
fn integration_backpressure_allows_sequential_requests() {
    let guard = BackpressureGuard::new(1).unwrap();
    guard.try_acquire().unwrap();
    assert_eq!(guard.depth().unwrap(), 1);
    guard.release().unwrap();

    // After release, can acquire again
    assert!(guard.try_acquire().is_ok());
    guard.release().unwrap();
}

// ── Agent loop ────────────────────────────────────────────────────────────────

#[test]
fn integration_react_loop_tool_chain() {
    let cfg = AgentConfig::new(10, "model");
    let mut loop_ = ReActLoop::new(cfg);

    loop_.register_tool(ToolSpec::new("step1", "first step", |_| {
        serde_json::json!("step1-done")
    }));
    loop_.register_tool(ToolSpec::new("step2", "second step", |_| {
        serde_json::json!("step2-done")
    }));

    let mut count = 0;
    let steps = loop_.run("execute pipeline", move |_| {
        count += 1;
        match count {
            1 => "Thought: run step1\nAction: step1 {}".into(),
            2 => "Thought: run step2\nAction: step2 {}".into(),
            _ => "Thought: complete\nAction: FINAL_ANSWER done".into(),
        }
    }).unwrap();

    assert_eq!(steps.len(), 3); // 2 tool calls + final answer
}

#[test]
fn integration_react_loop_with_json_tool_args() {
    let cfg = AgentConfig::new(5, "model");
    let mut loop_ = ReActLoop::new(cfg);

    loop_.register_tool(ToolSpec::new("calc", "calculator", |args| {
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        serde_json::json!(n * 2)
    }));

    let mut count = 0;
    let steps = loop_.run("double 21", move |_| {
        count += 1;
        if count == 1 {
            r#"Thought: call calc
Action: calc {"n":21}"#.into()
        } else {
            "Thought: done\nAction: FINAL_ANSWER 42".into()
        }
    }).unwrap();

    assert!(steps[0].observation.contains("42"));
}

// ── Working memory ────────────────────────────────────────────────────────────

#[test]
fn integration_working_memory_lru_eviction_under_load() {
    let wm = WorkingMemory::new(3).unwrap();

    for i in 0..10 {
        wm.set(format!("key-{i}"), format!("val-{i}")).unwrap();
    }

    // Should never have more than 3 entries
    assert_eq!(wm.len().unwrap(), 3);
    // Latest 3 keys should be present (7, 8, 9)
    assert_eq!(wm.get("key-9").unwrap(), Some("val-9".into()));
    assert_eq!(wm.get("key-0").unwrap(), None);
}

// ── Graph traversal ───────────────────────────────────────────────────────────

#[test]
fn integration_graph_bfs_finds_multi_hop_paths() {
    let graph = GraphStore::new();
    for id in ["a", "b", "c", "d", "e"] {
        graph.add_entity(Entity::new(id, "Node")).unwrap();
    }
    graph.add_relationship(Relationship::new("a", "b", "→", 1.0)).unwrap();
    graph.add_relationship(Relationship::new("b", "c", "→", 1.0)).unwrap();
    graph.add_relationship(Relationship::new("c", "d", "→", 1.0)).unwrap();
    graph.add_relationship(Relationship::new("a", "e", "→", 1.0)).unwrap();

    let visited = graph.bfs(&EntityId::new("a")).unwrap();
    assert_eq!(visited.len(), 4);
    // BFS visits b and e before c and d
    assert_eq!(visited[0], EntityId::new("b"));
}

#[test]
fn integration_graph_shortest_path_prefers_fewer_hops() {
    let graph = GraphStore::new();
    for id in ["a", "b", "c", "d"] {
        graph.add_entity(Entity::new(id, "Node")).unwrap();
    }
    // Long path: a -> b -> c -> d
    graph.add_relationship(Relationship::new("a", "b", "→", 1.0)).unwrap();
    graph.add_relationship(Relationship::new("b", "c", "→", 1.0)).unwrap();
    graph.add_relationship(Relationship::new("c", "d", "→", 1.0)).unwrap();
    // Short path: a -> d
    graph.add_relationship(Relationship::new("a", "d", "→", 1.0)).unwrap();

    let path = graph.shortest_path(&EntityId::new("a"), &EntityId::new("d")).unwrap().unwrap();
    assert_eq!(path.len(), 2); // [a, d]
}

#[test]
fn integration_graph_transitive_closure_full_chain() {
    let graph = GraphStore::new();
    for id in ["a", "b", "c", "d"] {
        graph.add_entity(Entity::new(id, "Node")).unwrap();
    }
    graph.add_relationship(Relationship::new("a", "b", "→", 1.0)).unwrap();
    graph.add_relationship(Relationship::new("b", "c", "→", 1.0)).unwrap();
    graph.add_relationship(Relationship::new("c", "d", "→", 1.0)).unwrap();

    let closure = graph.transitive_closure(&EntityId::new("a")).unwrap();
    assert_eq!(closure.len(), 4); // a, b, c, d
}

// ── Semantic memory ───────────────────────────────────────────────────────────

#[test]
fn integration_semantic_store_multi_tag_search() {
    let store = SemanticStore::new();
    store.store("doc1", "Rust async", vec!["rust".into(), "async".into(), "tokio".into()]).unwrap();
    store.store("doc2", "Python sync", vec!["python".into(), "sync".into()]).unwrap();
    store.store("doc3", "Rust sync", vec!["rust".into(), "sync".into()]).unwrap();

    let rust_only = store.retrieve(&["rust"]).unwrap();
    assert_eq!(rust_only.len(), 2);

    let rust_async = store.retrieve(&["rust", "async"]).unwrap();
    assert_eq!(rust_async.len(), 1);
    assert_eq!(rust_async[0].0, "doc1");
}

// ── Decay ────────────────────────────────────────────────────────────────────

#[test]
fn integration_decay_policy_applied_to_episodic_store() {
    let policy = DecayPolicy::exponential(1.0).unwrap(); // 1-hour half-life
    let store = EpisodicStore::with_decay(policy);
    let agent = AgentId::new("decay-agent");

    // Insert an item backdated 2 hours → should have ~25% of original importance
    store.add_episode_at(
        agent.clone(),
        "old fact",
        1.0,
        chrono::Utc::now() - chrono::Duration::hours(2),
    ).unwrap();

    let items = store.recall(&agent, 10).unwrap();
    assert_eq!(items.len(), 1);
    assert!(
        items[0].importance < 0.3,
        "importance after 2 half-lives should be ~0.25, got {}",
        items[0].importance
    );
}

// ── Error propagation ─────────────────────────────────────────────────────────

#[test]
fn integration_graph_error_wraps_in_runtime_error() {
    let graph = GraphStore::new();
    // Try to get an entity that doesn't exist
    let result = graph.get_entity(&EntityId::new("nonexistent"));
    assert!(matches!(result, Err(AgentRuntimeError::Graph(_))));
}

#[test]
fn integration_orchestration_circuit_open_error_variant() {
    let cb = CircuitBreaker::new("svc", 1, std::time::Duration::from_secs(3600)).unwrap();
    let _: Result<(), _> = cb.call(|| Err::<(), _>("fail".to_string()));
    let result: Result<(), AgentRuntimeError> = cb.call(|| Ok::<(), String>(()));
    assert!(matches!(result, Err(AgentRuntimeError::CircuitOpen { .. })));
}

#[test]
fn integration_memory_working_boundary_respected() {
    let wm = WorkingMemory::new(2).unwrap();
    wm.set("a", "1").unwrap();
    wm.set("b", "2").unwrap();
    wm.set("c", "3").unwrap(); // evicts "a"

    assert_eq!(wm.get("a").unwrap(), None);
    assert_eq!(wm.get("b").unwrap(), Some("2".into()));
    assert_eq!(wm.get("c").unwrap(), Some("3".into()));
}
