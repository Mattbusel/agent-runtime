//! Integration tests for feature-gated code paths.
//!
//! Each test uses `#[cfg(feature = "...")]` to ensure it only compiles
//! when the corresponding feature is enabled, mirroring the production gate.

// ── persistence feature ───────────────────────────────────────────────────────

#[cfg(feature = "persistence")]
mod persistence_tests {
    use llm_agent_runtime::agent::{AgentConfig, ToolSpec};
    use llm_agent_runtime::memory::AgentId;
    use llm_agent_runtime::persistence::{FilePersistenceBackend, PersistenceBackend};
    use llm_agent_runtime::runtime::{AgentRuntime, AgentSession};
    use std::sync::Arc;

    struct TempDir {
        path: std::path::PathBuf,
    }
    impl TempDir {
        async fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("agent_rt_feat_test_{}", uuid::Uuid::new_v4()));
            tokio::fs::create_dir_all(&path).await.unwrap();
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[tokio::test]
    async fn test_file_persistence_backend_save_load_delete_roundtrip() {
        let tmp = TempDir::new().await;
        let backend = FilePersistenceBackend::new(&tmp.path);

        backend.save("key-1", b"hello").await.unwrap();
        let loaded = backend.load("key-1").await.unwrap();
        assert_eq!(loaded, Some(b"hello".to_vec()));

        backend.delete("key-1").await.unwrap();
        let after_delete = backend.load("key-1").await.unwrap();
        assert_eq!(after_delete, None);
    }

    #[tokio::test]
    async fn test_persistence_backend_as_trait_object() {
        let tmp = TempDir::new().await;
        let backend: Arc<dyn PersistenceBackend> = Arc::new(FilePersistenceBackend::new(&tmp.path));

        backend.save("obj-safe-key", b"works").await.unwrap();
        let r = backend.load("obj-safe-key").await.unwrap();
        assert_eq!(r, Some(b"works".to_vec()));
    }

    #[tokio::test]
    async fn test_persistence_load_missing_key_returns_none() {
        let tmp = TempDir::new().await;
        let backend = FilePersistenceBackend::new(&tmp.path);
        let result = backend.load("does-not-exist").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_persistence_delete_missing_key_is_noop() {
        let tmp = TempDir::new().await;
        let backend = FilePersistenceBackend::new(&tmp.path);
        // Should not return an error
        backend.delete("no-such-key").await.unwrap();
    }

    #[tokio::test]
    async fn test_agent_session_save_and_load_checkpoint() {
        let tmp = TempDir::new().await;
        let backend: Arc<dyn PersistenceBackend> = Arc::new(FilePersistenceBackend::new(&tmp.path));

        let runtime = AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(5, "test"))
            .with_checkpoint_backend(Arc::clone(&backend))
            .build();

        let session = runtime
            .run_agent(
                AgentId::new("persist-agent"),
                "test",
                |_ctx: String| async { "Thought: done\nAction: FINAL_ANSWER ok".to_string() },
            )
            .await
            .unwrap();

        // The runtime should have auto-saved the checkpoint
        let loaded = AgentSession::load_checkpoint(backend.as_ref(), &session.session_id)
            .await
            .unwrap();
        assert!(loaded.is_some());
        let loaded_session = loaded.unwrap();
        assert_eq!(loaded_session.session_id, session.session_id);
        assert_eq!(loaded_session.step_count(), session.step_count());
    }

    #[tokio::test]
    async fn test_agent_session_load_checkpoint_missing_returns_none() {
        let tmp = TempDir::new().await;
        let backend = FilePersistenceBackend::new(&tmp.path);
        let result = AgentSession::load_checkpoint(&backend, "nonexistent-session-id")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_step_checkpoints_saved_incrementally() {
        let tmp = TempDir::new().await;
        let backend: Arc<dyn PersistenceBackend> = Arc::new(FilePersistenceBackend::new(&tmp.path));

        let mut call_count = 0;
        let runtime = AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(10, "test"))
            .with_checkpoint_backend(Arc::clone(&backend))
            .register_tool(ToolSpec::new("noop", "does nothing", |_| {
                serde_json::json!(null)
            }))
            .build();

        let session = runtime
            .run_agent(AgentId::new("step-agent"), "test", move |_ctx: String| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: step 1\nAction: noop {}".to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER result".to_string()
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(session.step_count(), 2);

        // Verify step-1 checkpoint was saved
        let step1 = AgentSession::load_step_checkpoint(backend.as_ref(), &session.session_id, 1)
            .await
            .unwrap();
        assert!(step1.is_some());
        assert_eq!(step1.unwrap().step_count(), 1);
    }
}

// ── providers feature ─────────────────────────────────────────────────────────

#[cfg(feature = "providers")]
mod providers_tests {
    use llm_agent_runtime::providers::LlmProvider;
    use llm_agent_runtime::AgentRuntimeError;
    use async_trait::async_trait;
    use std::sync::Arc;

    /// Echo provider: returns the prompt unchanged.
    struct EchoProvider;

    #[async_trait]
    impl LlmProvider for EchoProvider {
        async fn complete(&self, prompt: &str, _model: &str) -> Result<String, AgentRuntimeError> {
            Ok(prompt.to_owned())
        }
    }

    /// Error provider: always returns an error.
    struct ErrorProvider;

    #[async_trait]
    impl LlmProvider for ErrorProvider {
        async fn complete(&self, _prompt: &str, _model: &str) -> Result<String, AgentRuntimeError> {
            Err(AgentRuntimeError::Provider(
                "intentional test failure".into(),
            ))
        }
    }

    #[tokio::test]
    async fn test_llm_provider_trait_is_object_safe() {
        let p: Arc<dyn LlmProvider> = Arc::new(EchoProvider);
        let result = p.complete("hello", "model").await.unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn test_echo_provider_returns_prompt_as_completion() {
        let p = EchoProvider;
        let result = p.complete("test prompt", "any-model").await.unwrap();
        assert_eq!(result, "test prompt");
    }

    #[tokio::test]
    async fn test_error_provider_returns_provider_error() {
        let p = ErrorProvider;
        let result = p.complete("prompt", "model").await;
        assert!(matches!(result, Err(AgentRuntimeError::Provider(_))));
    }

    #[tokio::test]
    async fn test_default_stream_complete_wraps_complete() {
        let p = EchoProvider;
        let mut rx = p.stream_complete("streamed prompt", "model").await.unwrap();
        let mut collected = String::new();
        while let Some(chunk) = rx.recv().await {
            collected.push_str(&chunk.unwrap());
        }
        assert_eq!(collected, "streamed prompt");
    }

    #[tokio::test]
    async fn test_stream_complete_channel_closes_after_all_chunks() {
        let p = EchoProvider;
        let mut rx = p.stream_complete("data", "model").await.unwrap();
        while let Some(_) = rx.recv().await {}
        assert!(rx.recv().await.is_none(), "channel should be closed");
    }

    #[tokio::test]
    async fn test_multiple_providers_behind_arc() {
        let providers: Vec<Arc<dyn LlmProvider>> =
            vec![Arc::new(EchoProvider), Arc::new(EchoProvider)];
        for p in &providers {
            let r = p.complete("ping", "model").await.unwrap();
            assert_eq!(r, "ping");
        }
    }
}

// ── orchestrator feature ──────────────────────────────────────────────────────

#[cfg(feature = "orchestrator")]
mod orchestrator_tests {
    use llm_agent_runtime::orchestrator::{
        BackpressureGuard, CircuitBreaker, CircuitState, DeduplicationResult, Deduplicator,
        Pipeline, RetryPolicy, MAX_RETRY_DELAY,
    };
    use llm_agent_runtime::AgentRuntimeError;
    use std::time::Duration;

    #[test]
    fn test_retry_policy_exponential_creates_correctly() {
        let p = RetryPolicy::exponential(3, 50).unwrap();
        assert_eq!(p.max_attempts, 3);
        assert_eq!(p.base_delay, Duration::from_millis(50));
    }

    #[test]
    fn test_retry_policy_zero_attempts_is_error() {
        assert!(RetryPolicy::exponential(0, 100).is_err());
    }

    #[test]
    fn test_retry_policy_delay_is_capped_at_max() {
        let p = RetryPolicy::exponential(5, 100_000).unwrap();
        for attempt in 1..=5 {
            assert!(p.delay_for(attempt) <= MAX_RETRY_DELAY);
        }
    }

    #[test]
    fn test_circuit_breaker_starts_in_closed_state() {
        let cb = CircuitBreaker::new("test-svc", 3, Duration::from_secs(60)).unwrap();
        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new("svc", 2, Duration::from_secs(3600)).unwrap();
        for _ in 0..2 {
            let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
        }
        assert!(matches!(cb.state().unwrap(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_circuit_breaker_success_resets_to_closed() {
        let cb = CircuitBreaker::new("svc", 5, Duration::from_secs(3600)).unwrap();
        let _: Result<(), AgentRuntimeError> = cb.call(|| Err::<(), _>("fail".to_string()));
        let _: Result<i32, AgentRuntimeError> = cb.call(|| Ok::<i32, String>(1));
        assert_eq!(cb.state().unwrap(), CircuitState::Closed);
        assert_eq!(cb.failure_count().unwrap(), 0);
    }

    #[test]
    fn test_circuit_breaker_zero_threshold_is_error() {
        assert!(CircuitBreaker::new("svc", 0, Duration::from_secs(1)).is_err());
    }

    #[test]
    fn test_deduplicator_new_then_in_progress_then_cached() {
        let d = Deduplicator::new(Duration::from_secs(60));
        assert_eq!(
            d.check_and_register("req-1").unwrap(),
            DeduplicationResult::New
        );
        assert_eq!(
            d.check_and_register("req-1").unwrap(),
            DeduplicationResult::InProgress
        );
        d.complete("req-1", "result").unwrap();
        assert_eq!(
            d.check_and_register("req-1").unwrap(),
            DeduplicationResult::Cached("result".into())
        );
    }

    #[test]
    fn test_backpressure_guard_zero_capacity_is_error() {
        assert!(BackpressureGuard::new(0).is_err());
    }

    #[test]
    fn test_backpressure_guard_sheds_at_capacity() {
        let g = BackpressureGuard::new(2).unwrap();
        g.try_acquire().unwrap();
        g.try_acquire().unwrap();
        let r = g.try_acquire();
        assert!(matches!(r, Err(AgentRuntimeError::BackpressureShed { .. })));
    }

    #[test]
    fn test_pipeline_runs_all_stages_in_order() {
        let p = Pipeline::new()
            .add_stage("step1", |s| Ok(format!("[{s}]")))
            .add_stage("step2", |s| Ok(format!("{s}!")));
        assert_eq!(p.run("x".into()).unwrap(), "[x]!");
    }

    #[test]
    fn test_pipeline_empty_returns_input_unchanged() {
        let p = Pipeline::new();
        assert_eq!(p.run("unchanged".into()).unwrap(), "unchanged");
    }

    #[test]
    fn test_pipeline_error_short_circuits() {
        let p = Pipeline::new()
            .add_stage("fail", |_| {
                Err(AgentRuntimeError::Orchestration("boom".into()))
            })
            .add_stage("unreachable", |s| Ok(s));
        assert!(p.run("input".into()).is_err());
    }
}

// ── memory feature ────────────────────────────────────────────────────────────

#[cfg(feature = "memory")]
mod memory_tests {
    use llm_agent_runtime::memory::{
        AgentId, DecayPolicy, EpisodicStore, MemoryId, SemanticStore, WorkingMemory,
    };

    #[test]
    fn test_agent_id_display() {
        let id = AgentId::new("agent-abc");
        assert_eq!(id.to_string(), "agent-abc");
    }

    #[test]
    fn test_memory_id_display() {
        let id = MemoryId::new("mem-xyz");
        assert_eq!(id.to_string(), "mem-xyz");
    }

    #[test]
    fn test_agent_id_random_unique() {
        let a = AgentId::random();
        let b = AgentId::random();
        assert_ne!(a, b);
    }

    #[test]
    fn test_episodic_store_default_is_empty() {
        let s: EpisodicStore = Default::default();
        assert!(s.is_empty().unwrap());
    }

    #[test]
    fn test_episodic_store_recall_returns_sorted_by_importance() {
        let s = EpisodicStore::new();
        let a = AgentId::new("a");
        s.add_episode(a.clone(), "low", 0.1).unwrap();
        s.add_episode(a.clone(), "high", 0.9).unwrap();
        let items = s.recall(&a, 10).unwrap();
        assert_eq!(items[0].content, "high");
    }

    #[test]
    fn test_decay_policy_half_life_zero_is_error() {
        assert!(DecayPolicy::exponential(0.0).is_err());
    }

    #[test]
    fn test_decay_policy_applies_correctly_at_half_life() {
        let p = DecayPolicy::exponential(1.0).unwrap();
        let decayed = p.apply(1.0, 1.0);
        assert!((decayed - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_working_memory_zero_capacity_is_error() {
        assert!(WorkingMemory::new(0).is_err());
    }

    #[test]
    fn test_working_memory_entries_in_insertion_order() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("first", "1").unwrap();
        wm.set("second", "2").unwrap();
        wm.set("third", "3").unwrap();
        let entries = wm.entries().unwrap();
        assert_eq!(entries[0].0, "first");
        assert_eq!(entries[1].0, "second");
        assert_eq!(entries[2].0, "third");
    }

    #[test]
    fn test_semantic_store_retrieve_empty_tags_returns_all() {
        let s = SemanticStore::new();
        s.store("k1", "v1", vec!["a".into()]).unwrap();
        s.store("k2", "v2", vec!["b".into()]).unwrap();
        let all = s.retrieve(&[]).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_semantic_store_retrieve_similar_top_k_respected() {
        let s = SemanticStore::new();
        for i in 0..5 {
            s.store_with_embedding(
                format!("k{i}"),
                format!("v{i}"),
                vec![],
                vec![i as f32, 0.0],
            )
            .unwrap();
        }
        let results = s.retrieve_similar(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
    }
}

// ── graph feature ─────────────────────────────────────────────────────────────

#[cfg(feature = "graph")]
mod graph_tests {
    use llm_agent_runtime::graph::{Entity, EntityId, GraphStore, Relationship};
    use llm_agent_runtime::AgentRuntimeError;

    #[test]
    fn test_graph_store_add_and_get_entity() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("e1", "TypeA")).unwrap();
        let entity = g.get_entity(&EntityId::new("e1")).unwrap();
        assert_eq!(entity.label, "TypeA");
    }

    #[test]
    fn test_graph_store_get_nonexistent_entity_is_error() {
        let g = GraphStore::new();
        let result = g.get_entity(&EntityId::new("ghost"));
        assert!(matches!(result, Err(AgentRuntimeError::Graph(_))));
    }

    #[test]
    fn test_graph_store_entity_count() {
        let g = GraphStore::new();
        for i in 0..5 {
            g.add_entity(Entity::new(format!("n{i}"), "Node")).unwrap();
        }
        assert_eq!(g.entity_count().unwrap(), 5);
    }

    #[test]
    fn test_graph_store_bfs_finds_direct_neighbor() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("src", "Node")).unwrap();
        g.add_entity(Entity::new("dst", "Node")).unwrap();
        g.add_relationship(Relationship::new("src", "dst", "LINKS", 1.0))
            .unwrap();
        let visited = g.bfs(&EntityId::new("src")).unwrap();
        assert_eq!(visited.len(), 1);
        assert_eq!(visited[0], EntityId::new("dst"));
    }

    #[test]
    fn test_graph_store_bfs_visits_all_reachable() {
        let g = GraphStore::new();
        for id in ["a", "b", "c"] {
            g.add_entity(Entity::new(id, "N")).unwrap();
        }
        g.add_relationship(Relationship::new("a", "b", "→", 1.0))
            .unwrap();
        g.add_relationship(Relationship::new("b", "c", "→", 1.0))
            .unwrap();
        let visited = g.bfs(&EntityId::new("a")).unwrap();
        assert_eq!(visited.len(), 2); // b and c (not the start node)
    }

    #[test]
    fn test_graph_store_shortest_path_direct() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "N")).unwrap();
        g.add_entity(Entity::new("y", "N")).unwrap();
        g.add_relationship(Relationship::new("x", "y", "→", 1.0))
            .unwrap();
        let path = g
            .shortest_path(&EntityId::new("x"), &EntityId::new("y"))
            .unwrap()
            .unwrap();
        assert_eq!(path, vec![EntityId::new("x"), EntityId::new("y")]);
    }

    #[test]
    fn test_graph_store_shortest_path_unreachable_returns_none() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "N")).unwrap();
        g.add_entity(Entity::new("y", "N")).unwrap();
        // No relationship
        let path = g
            .shortest_path(&EntityId::new("x"), &EntityId::new("y"))
            .unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_graph_store_transitive_closure_includes_all_reachable() {
        let g = GraphStore::new();
        for id in ["a", "b", "c", "d"] {
            g.add_entity(Entity::new(id, "N")).unwrap();
        }
        g.add_relationship(Relationship::new("a", "b", "→", 1.0))
            .unwrap();
        g.add_relationship(Relationship::new("b", "c", "→", 1.0))
            .unwrap();
        g.add_relationship(Relationship::new("c", "d", "→", 1.0))
            .unwrap();
        // "d" is isolated from the chain going backwards
        let closure = g.transitive_closure(&EntityId::new("a")).unwrap();
        // should include a, b, c, d
        assert_eq!(closure.len(), 4);
    }

    #[test]
    fn test_graph_store_remove_entity_decrements_count() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("r", "N")).unwrap();
        assert_eq!(g.entity_count().unwrap(), 1);
        g.remove_entity(&EntityId::new("r")).unwrap();
        assert_eq!(g.entity_count().unwrap(), 0);
    }
}
