//! Tests for memory management: eviction under capacity, TTL expiry, concurrent access,
//! and the full spectrum of memory subsystem behaviors.

use llm_agent_runtime::memory::{
    AgentId, DecayPolicy, EpisodicStore, RecallPolicy, SemanticStore, WorkingMemory,
};
use std::sync::Arc;

// ── WorkingMemory: eviction under capacity ────────────────────────────────────

#[test]
fn wm_zero_capacity_is_rejected() {
    assert!(WorkingMemory::new(0).is_err());
}

#[test]
fn wm_capacity_one_evicts_first_on_second_insert() {
    let wm = WorkingMemory::new(1).unwrap();
    wm.set("a", "1").unwrap();
    wm.set("b", "2").unwrap();
    assert_eq!(wm.get("a").unwrap(), None);
    assert_eq!(wm.get("b").unwrap(), Some("2".into()));
}

#[test]
fn wm_always_holds_exactly_capacity_entries() {
    let cap = 4usize;
    let wm = WorkingMemory::new(cap).unwrap();
    for i in 0..20 {
        wm.set(format!("k{i}"), format!("v{i}")).unwrap();
        assert!(wm.len().unwrap() <= cap, "len exceeded capacity at i={i}");
    }
}

#[test]
fn wm_update_of_existing_key_does_not_evict() {
    let wm = WorkingMemory::new(2).unwrap();
    wm.set("a", "1").unwrap();
    wm.set("b", "2").unwrap();
    wm.set("a", "updated").unwrap(); // update, not new key
    assert_eq!(wm.len().unwrap(), 2);
    assert_eq!(wm.get("a").unwrap(), Some("updated".into()));
    assert_eq!(wm.get("b").unwrap(), Some("2".into()));
}

#[test]
fn wm_lru_eviction_order_is_oldest_first() {
    let wm = WorkingMemory::new(3).unwrap();
    wm.set("k1", "v1").unwrap();
    wm.set("k2", "v2").unwrap();
    wm.set("k3", "v3").unwrap();
    wm.set("k4", "v4").unwrap(); // evicts k1
    wm.set("k5", "v5").unwrap(); // evicts k2

    assert_eq!(wm.get("k1").unwrap(), None);
    assert_eq!(wm.get("k2").unwrap(), None);
    assert_eq!(wm.get("k3").unwrap(), Some("v3".into()));
    assert_eq!(wm.get("k4").unwrap(), Some("v4".into()));
    assert_eq!(wm.get("k5").unwrap(), Some("v5".into()));
}

#[test]
fn wm_entries_preserves_insertion_order() {
    let wm = WorkingMemory::new(10).unwrap();
    wm.set("z", "1").unwrap();
    wm.set("a", "2").unwrap();
    wm.set("m", "3").unwrap();

    let entries = wm.entries().unwrap();
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["z", "a", "m"]);
}

#[test]
fn wm_clear_empties_all_entries() {
    let wm = WorkingMemory::new(10).unwrap();
    for i in 0..5 {
        wm.set(format!("k{i}"), "v").unwrap();
    }
    wm.clear().unwrap();
    assert_eq!(wm.len().unwrap(), 0);
    assert!(wm.is_empty().unwrap());
}

// ── WorkingMemory: concurrent access ─────────────────────────────────────────

#[tokio::test]
async fn wm_concurrent_writes_stay_within_capacity() {
    let cap = 20usize;
    let wm = Arc::new(WorkingMemory::new(cap).unwrap());
    let mut handles = Vec::new();

    for i in 0u32..50 {
        let wm = Arc::clone(&wm);
        handles.push(tokio::spawn(async move {
            wm.set(format!("key-{i}"), format!("val-{i}")).unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert!(
        wm.len().unwrap() <= cap,
        "concurrent inserts exceeded capacity"
    );
}

// ── EpisodicStore: TTL / decay ────────────────────────────────────────────────

#[test]
fn episodic_fast_decay_reduces_importance_to_near_zero() {
    let policy = DecayPolicy::exponential(0.0001).unwrap(); // extremely fast decay
    let store = EpisodicStore::with_decay(policy);
    let agent = AgentId::new("agent");

    // Insert a memory item timestamped 1 hour in the past using the public API.
    let old_ts = chrono::Utc::now() - chrono::Duration::hours(1);
    store
        .add_episode_at(agent.clone(), "old", 1.0, old_ts)
        .unwrap();

    let items = store.recall(&agent, 10).unwrap();
    assert!(items[0].importance < 0.01, "expected near-zero importance");
}

#[test]
fn episodic_no_decay_importance_unchanged() {
    let store = EpisodicStore::new(); // no decay
    let agent = AgentId::new("agent");
    store.add_episode(agent.clone(), "fact", 0.75).unwrap();
    let items = store.recall(&agent, 10).unwrap();
    assert!((items[0].importance - 0.75).abs() < 0.01);
}

#[test]
fn episodic_per_agent_capacity_evicts_lowest_importance() {
    let store = EpisodicStore::with_per_agent_capacity(3);
    let agent = AgentId::new("agent");

    store.add_episode(agent.clone(), "mid", 0.5).unwrap();
    store.add_episode(agent.clone(), "high", 0.9).unwrap();
    store.add_episode(agent.clone(), "low", 0.1).unwrap();
    // 3 items at capacity; adding one more evicts the lowest
    store.add_episode(agent.clone(), "new", 0.6).unwrap();

    assert_eq!(store.len().unwrap(), 3);
    let items = store.recall(&agent, 10).unwrap();
    let contents: Vec<&str> = items.iter().map(|i| i.content.as_str()).collect();
    assert!(
        !contents.contains(&"low"),
        "lowest-importance item should have been evicted; got: {contents:?}"
    );
}

#[test]
fn episodic_different_agents_do_not_interfere() {
    let store = EpisodicStore::with_per_agent_capacity(2);
    let a = AgentId::new("a");
    let b = AgentId::new("b");

    store.add_episode(a.clone(), "a1", 0.8).unwrap();
    store.add_episode(a.clone(), "a2", 0.7).unwrap();
    store.add_episode(b.clone(), "b1", 0.9).unwrap();
    store.add_episode(b.clone(), "b2", 0.6).unwrap();

    // Neither agent should be affected by the other's capacity
    assert_eq!(store.recall(&a, 10).unwrap().len(), 2);
    assert_eq!(store.recall(&b, 10).unwrap().len(), 2);
}

#[test]
fn episodic_recall_increments_recall_count() {
    let store = EpisodicStore::new();
    let agent = AgentId::new("agent");
    store.add_episode(agent.clone(), "fact", 0.5).unwrap();

    let items1 = store.recall(&agent, 10).unwrap();
    assert_eq!(items1[0].recall_count, 1);

    let items2 = store.recall(&agent, 10).unwrap();
    assert_eq!(items2[0].recall_count, 2);
}

#[test]
fn episodic_hybrid_policy_ranks_frequent_over_new() {
    let store = EpisodicStore::with_recall_policy(RecallPolicy::Hybrid {
        recency_weight: 0.0,
        frequency_weight: 10.0,
    });
    let agent = AgentId::new("agent");

    store.add_episode(agent.clone(), "frequent", 0.5).unwrap();
    store.add_episode(agent.clone(), "new", 0.5).unwrap();

    // Simulate 50 prior recalls of "frequent" using the test-helper method.
    store.bump_recall_count_by_content("frequent", 50);

    // With frequency_weight = 10.0 and recency_weight = 0.0,
    // the item with the highest recall_count should sort first.
    let items = store.recall(&agent, 10).unwrap();
    assert_eq!(
        items[0].content, "frequent",
        "hybrid policy should rank frequently recalled item first"
    );
}

// ── EpisodicStore: concurrent access ─────────────────────────────────────────

#[tokio::test]
async fn episodic_concurrent_writes_all_recorded() {
    let store = Arc::new(EpisodicStore::new());
    let mut handles = Vec::new();

    for i in 0u32..30 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let agent = AgentId::new(format!("a{i}"));
            store.add_episode(agent, format!("event {i}"), 0.5).unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(store.len().unwrap(), 30);
}

#[tokio::test]
async fn episodic_concurrent_reads_never_panic() {
    let store = Arc::new(EpisodicStore::new());
    let agent = AgentId::new("shared-agent");

    for i in 0..10 {
        store
            .add_episode(agent.clone(), format!("fact {i}"), 0.5)
            .unwrap();
    }

    let mut handles = Vec::new();
    for _ in 0..20 {
        let store = Arc::clone(&store);
        let agent = agent.clone();
        handles.push(tokio::spawn(async move {
            let _ = store.recall(&agent, 5);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}

// ── SemanticStore ─────────────────────────────────────────────────────────────

#[test]
fn semantic_store_and_retrieve_by_single_tag() {
    let store = SemanticStore::new();
    store.store("doc1", "val1", vec!["rust".into()]).unwrap();
    store.store("doc2", "val2", vec!["python".into()]).unwrap();
    let result = store.retrieve(&["rust"]).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, "doc1");
}

#[test]
fn semantic_retrieve_empty_tags_returns_all() {
    let store = SemanticStore::new();
    store.store("a", "v", vec!["x".into()]).unwrap();
    store.store("b", "v", vec!["y".into()]).unwrap();
    let all = store.retrieve(&[]).unwrap();
    assert_eq!(all.len(), 2);
}

#[test]
fn semantic_similarity_search_returns_top_k_in_order() {
    let store = SemanticStore::new();
    store
        .store_with_embedding("close", "c", vec![], vec![1.0, 0.0])
        .unwrap();
    store
        .store_with_embedding("far", "f", vec![], vec![0.0, 1.0])
        .unwrap();
    store
        .store_with_embedding("mid", "m", vec![], vec![0.7071, 0.7071])
        .unwrap();

    let results = store.retrieve_similar(&[1.0, 0.0], 3).unwrap();
    assert_eq!(results[0].0, "close");
    assert!(
        results[0].2 > results[1].2,
        "results should be sorted by descending similarity"
    );
}

#[test]
fn semantic_similarity_top_k_limits_results() {
    let store = SemanticStore::new();
    for i in 0..10 {
        store
            .store_with_embedding(format!("k{i}"), "v", vec![], vec![i as f32, 0.0])
            .unwrap();
    }
    let results = store.retrieve_similar(&[5.0, 0.0], 3).unwrap();
    assert_eq!(results.len(), 3);
}

// ── DecayPolicy ───────────────────────────────────────────────────────────────

#[test]
fn decay_exactly_half_at_half_life() {
    let policy = DecayPolicy::exponential(24.0).unwrap();
    let decayed = policy.apply(1.0, 24.0);
    assert!((decayed - 0.5).abs() < 1e-5, "got {decayed}");
}

#[test]
fn decay_zero_age_preserves_importance() {
    let policy = DecayPolicy::exponential(24.0).unwrap();
    assert!((policy.apply(0.8, 0.0) - 0.8).abs() < 1e-5);
}

#[test]
fn decay_rejects_non_positive_half_life() {
    assert!(DecayPolicy::exponential(0.0).is_err());
    assert!(DecayPolicy::exponential(-5.0).is_err());
}

#[test]
fn decay_output_clamped_to_zero_one() {
    let policy = DecayPolicy::exponential(0.001).unwrap();
    let result = policy.apply(1.0, 100_000.0);
    assert!(result >= 0.0);
    assert!(result <= 1.0);
}

#[test]
fn hybrid_policy_with_decay_ranks_by_combined_score() {
    // Decay half-life = 1 hour; recency_weight=0.0, frequency_weight=10.0
    // (pure frequency ranking, ignoring recency).
    let decay = DecayPolicy::exponential(1.0).unwrap();
    let store = EpisodicStore::with_decay_and_recall_policy(
        decay,
        RecallPolicy::Hybrid {
            recency_weight: 0.0,
            frequency_weight: 10.0,
        },
    );
    let agent = AgentId::new("agent");

    // "old_frequent" was added 2 hours ago (decayed importance ~0.25) but
    // recalled many times.
    let old_ts = chrono::Utc::now() - chrono::Duration::hours(2);
    store
        .add_episode_at(agent.clone(), "old_frequent", 1.0, old_ts)
        .unwrap();
    store.bump_recall_count_by_content("old_frequent", 100);

    // "new_rare" was just added with high raw importance but never recalled.
    store.add_episode(agent.clone(), "new_rare", 1.0).unwrap();

    let items = store.recall(&agent, 10).unwrap();
    // With frequency_weight=10.0 and recency_weight=0.0, the frequency term
    // dominates: old_frequent has recall_count=100 and new_rare has 0, so
    // old_frequent's combined score is far higher.
    assert_eq!(
        items[0].content, "old_frequent",
        "old_frequent should rank first due to high recall count; got: {:?}",
        items.iter().map(|i| &i.content).collect::<Vec<_>>()
    );
}
