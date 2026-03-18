//! Integration tests: memory persistence — write, read back, and verify durability.

use llm_agent_runtime::prelude::*;

// ── EpisodicStore: write and read back ───────────────────────────────────────

#[test]
fn memory_episodic_write_and_read_back() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("persist-agent");

    let id = store
        .add_episode(agent.clone(), "Paris is the capital of France.", 0.9)
        .unwrap();

    assert!(!id.0.is_empty());

    let items = store.recall(&agent, 10).unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].content, "Paris is the capital of France.");
    assert!((items[0].importance - 0.9).abs() < 1e-5);
}

#[test]
fn memory_episodic_multiple_writes_read_back_in_importance_order() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("order-agent");

    store.add_episode(agent.clone(), "low", 0.1).unwrap();
    store.add_episode(agent.clone(), "high", 0.9).unwrap();
    store.add_episode(agent.clone(), "mid", 0.5).unwrap();

    let items = store.recall(&agent, 10).unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].content, "high");
    assert_eq!(items[1].content, "mid");
    assert_eq!(items[2].content, "low");
}

#[test]
fn memory_episodic_agent_isolation() {
    let store = EpisodicStore::new();
    let a = AgentId::new("agent-a");
    let b = AgentId::new("agent-b");

    store.add_episode(a.clone(), "only for A", 0.8).unwrap();
    store.add_episode(b.clone(), "only for B", 0.8).unwrap();

    let a_items = store.recall(&a, 10).unwrap();
    let b_items = store.recall(&b, 10).unwrap();

    assert_eq!(a_items.len(), 1);
    assert_eq!(a_items[0].content, "only for A");
    assert_eq!(b_items.len(), 1);
    assert_eq!(b_items[0].content, "only for B");
}

#[test]
fn memory_episodic_recall_limit_respected() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("limit-agent");

    for i in 0..10u32 {
        store
            .add_episode(agent.clone(), format!("fact {i}"), 0.5)
            .unwrap();
    }

    let items = store.recall(&agent, 3).unwrap();
    assert_eq!(items.len(), 3);
}

#[test]
fn memory_episodic_shared_store_clone_shares_data() {
    // Clone shares the Arc — the cloned handle sees data added via the original.
    let store = EpisodicStore::new();
    let agent = AgentId::new("shared-agent");
    let store2 = store.clone();

    store
        .add_episode(agent.clone(), "written via original", 0.7)
        .unwrap();
    let items = store2.recall(&agent, 10).unwrap();

    assert_eq!(
        items.len(),
        1,
        "clone should see data written by the original"
    );
    assert_eq!(items[0].content, "written via original");
}

// ── WorkingMemory: write and read back ───────────────────────────────────────

#[test]
fn memory_working_set_and_get_persists() {
    let wm = WorkingMemory::new(10).unwrap();
    wm.set("session_goal", "write tests").unwrap();

    let val = wm.get("session_goal").unwrap();
    assert_eq!(val, Some("write tests".to_string()));
}

#[test]
fn memory_working_overwrite_persists_new_value() {
    let wm = WorkingMemory::new(10).unwrap();
    wm.set("key", "first").unwrap();
    wm.set("key", "second").unwrap();

    assert_eq!(wm.get("key").unwrap(), Some("second".to_string()));
}

#[test]
fn memory_working_entries_returns_all_in_insertion_order() {
    let wm = WorkingMemory::new(10).unwrap();
    wm.set("alpha", "1").unwrap();
    wm.set("beta", "2").unwrap();
    wm.set("gamma", "3").unwrap();

    let entries = wm.entries().unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0], ("alpha".to_string(), "1".to_string()));
    assert_eq!(entries[1], ("beta".to_string(), "2".to_string()));
    assert_eq!(entries[2], ("gamma".to_string(), "3".to_string()));
}

#[test]
fn memory_working_clear_removes_all_entries() {
    let wm = WorkingMemory::new(10).unwrap();
    wm.set("a", "x").unwrap();
    wm.set("b", "y").unwrap();
    wm.clear().unwrap();

    assert!(wm.is_empty().unwrap());
    assert_eq!(wm.get("a").unwrap(), None);
}

#[test]
fn memory_working_clone_shares_state() {
    let wm = WorkingMemory::new(10).unwrap();
    let wm2 = wm.clone();
    wm.set("shared", "value").unwrap();

    assert_eq!(
        wm2.get("shared").unwrap(),
        Some("value".to_string()),
        "clone must share state via Arc"
    );
}

// ── SemanticStore: write and read back ───────────────────────────────────────

#[test]
fn memory_semantic_store_and_retrieve_all() {
    let store = SemanticStore::new();
    store
        .store("fact-1", "Rust is memory-safe", vec!["lang".into()])
        .unwrap();
    store
        .store("fact-2", "Tokio is async", vec!["async".into()])
        .unwrap();

    let all = store.retrieve(&[]).unwrap();
    assert_eq!(all.len(), 2);
}

#[test]
fn memory_semantic_tag_filter_matches_correct_entries() {
    let store = SemanticStore::new();
    store
        .store("k1", "v1", vec!["rust".into(), "async".into()])
        .unwrap();
    store.store("k2", "v2", vec!["python".into()]).unwrap();

    let results = store.retrieve(&["rust"]).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "k1");
}

#[test]
fn memory_semantic_similarity_search_returns_closest() {
    let store = SemanticStore::new();
    store
        .store_with_embedding("near", "near val", vec![], vec![1.0f32, 0.0, 0.0])
        .unwrap();
    store
        .store_with_embedding("far", "far val", vec![], vec![0.0f32, 1.0, 0.0])
        .unwrap();

    let results = store.retrieve_similar(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "near");
    assert!((results[0].2 - 1.0).abs() < 1e-5);
}

// ── Memory injected into AgentRuntime prompt ─────────────────────────────────

#[tokio::test]
async fn memory_injected_into_agent_context() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("ctx-agent");
    store
        .add_episode(agent.clone(), "the answer is always 42", 0.95)
        .unwrap();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .with_memory(store)
        .build();

    use std::sync::{Arc, Mutex};
    let captured: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let cap2 = Arc::clone(&captured);

    runtime
        .run_agent(agent, "what is the answer?", move |ctx: String| {
            let c = Arc::clone(&cap2);
            async move {
                #[allow(clippy::unwrap_used)]
                let mut guard = c.lock().unwrap();
                if guard.is_empty() {
                    *guard = ctx;
                }
                "Thought: done\nAction: FINAL_ANSWER 42".to_string()
            }
        })
        .await
        .unwrap();

    #[allow(clippy::unwrap_used)]
    let ctx = captured.lock().unwrap().clone();
    assert!(
        ctx.contains("the answer is always 42"),
        "expected episodic memory to be injected, got: {ctx}"
    );
}
