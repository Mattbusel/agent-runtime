//! Concurrency stress tests for agent-runtime.
//!
//! These tests spawn multiple async tasks simultaneously to verify that
//! shared state (EpisodicStore, GraphStore, SemanticStore, WorkingMemory)
//! is correctly handled under concurrent access.

use agent_runtime::{
    agent::AgentConfig,
    graph::{Entity, GraphStore},
    memory::{AgentId, EpisodicStore, SemanticStore, WorkingMemory},
    orchestrator::BackpressureGuard,
    runtime::AgentRuntime,
    AgentRuntimeError,
};

// Helper: simple infer function that returns FINAL_ANSWER immediately
async fn instant_answer(_ctx: String) -> String {
    "Thought: done\nAction: FINAL_ANSWER ok".to_string()
}

// ── Test 1 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_episodic_store_writes() {
    use std::sync::Arc;
    let store = Arc::new(EpisodicStore::new());
    let mut handles = Vec::new();

    for i in 0u32..20 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let agent = AgentId::new(format!("agent-{i}"));
            for j in 0..10 {
                store
                    .add_episode(agent.clone(), format!("event {j}"), 0.5)
                    .unwrap();
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(store.len().unwrap(), 200);
}

// ── Test 2 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_graph_mutations() {
    use std::sync::Arc;
    let graph = Arc::new(GraphStore::new());
    let mut handles = Vec::new();

    for i in 0u32..10 {
        let graph = Arc::clone(&graph);
        handles.push(tokio::spawn(async move {
            for j in 0u32..5 {
                let id = format!("node-{i}-{j}");
                graph.add_entity(Entity::new(id.clone(), "Node")).unwrap();
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(graph.entity_count().unwrap(), 50);
}

// ── Test 3 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_agent_sessions_shared_memory() {
    use std::sync::Arc;

    let store = EpisodicStore::new();
    // Pre-populate with some memories
    for i in 0..5 {
        store
            .add_episode(
                AgentId::new("shared"),
                format!("fact {i}"),
                0.5 + i as f32 * 0.1,
            )
            .unwrap();
    }

    let runtime = Arc::new(
        AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(3, "test"))
            .with_memory(store)
            .build(),
    );

    let mut handles = Vec::new();
    for i in 0u32..10 {
        let runtime = Arc::clone(&runtime);
        handles.push(tokio::spawn(async move {
            runtime
                .run_agent(AgentId::new(format!("agent-{i}")), "hello", instant_answer)
                .await
                .unwrap()
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }
    assert_eq!(results.len(), 10);
    for s in &results {
        assert!(s.step_count() >= 1);
    }
}

// ── Test 4 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_working_memory_writes() {
    use std::sync::Arc;
    let wm = Arc::new(WorkingMemory::new(100).unwrap());
    let mut handles = Vec::new();

    for i in 0u32..10 {
        let wm = Arc::clone(&wm);
        handles.push(tokio::spawn(async move {
            for j in 0u32..5 {
                wm.set(format!("key-{i}-{j}"), format!("val-{i}-{j}"))
                    .unwrap();
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
    // 10*5 = 50 unique keys, all should fit in capacity=100
    assert_eq!(wm.len().unwrap(), 50);
}

// ── Test 5 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_semantic_store_similarity_search() {
    use std::sync::Arc;
    let store = Arc::new(SemanticStore::new());
    let mut handles = Vec::new();

    // Writers
    for i in 0u32..5 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            store
                .store_with_embedding(
                    format!("key-{i}"),
                    format!("val-{i}"),
                    vec![],
                    vec![i as f32, 0.0, 0.0],
                )
                .unwrap();
        }));
    }

    // Readers (concurrent with writers)
    for _ in 0u32..5 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let _ = store.retrieve_similar(&[1.0, 0.0, 0.0], 3);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
    // At least some entries should be stored by the time we check
    assert!(store.len().unwrap() > 0);
}

// ── Test 6 ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_backpressure_concurrent_saturates_correctly() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    let guard = Arc::new(BackpressureGuard::new(5).unwrap());
    let success = Arc::new(AtomicUsize::new(0));
    let shed = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    for _ in 0..10 {
        let guard = Arc::clone(&guard);
        let success = Arc::clone(&success);
        let shed = Arc::clone(&shed);
        handles.push(tokio::spawn(async move {
            match guard.try_acquire() {
                Ok(()) => {
                    success.fetch_add(1, Ordering::Relaxed);
                }
                Err(AgentRuntimeError::BackpressureShed { .. }) => {
                    shed.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {}
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let s = success.load(Ordering::Relaxed);
    let e = shed.load(Ordering::Relaxed);
    assert_eq!(s + e, 10);
    assert!(s <= 5, "should not exceed capacity, got {s}");
}
