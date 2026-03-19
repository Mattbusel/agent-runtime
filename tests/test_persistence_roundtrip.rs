//! Integration test: persistence round-trip with many steps.

#[cfg(feature = "persistence")]
mod persistence_roundtrip {
    use llm_agent_runtime::persistence::{FilePersistenceBackend, PersistenceBackend};

    #[tokio::test]
    async fn test_session_checkpoint_round_trip_many_steps() {
        // Create a temporary directory.
        let dir = std::env::temp_dir()
            .join(format!("art_roundtrip_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();

        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(dir.clone());

        let backend = FilePersistenceBackend::new(&dir);

        // Simulate 50 agent step checkpoints.
        for step in 0u32..50 {
            let key = format!("session:agent-1:step:{step}");
            let value = format!(
                r#"{{"step":{step},"thought":"thinking about step {step}","action":"tool_{step}"}}"#
            );
            backend.save(&key, value.as_bytes()).await.unwrap();
        }

        // Verify all 50 checkpoints can be loaded back.
        for step in 0u32..50 {
            let key = format!("session:agent-1:step:{step}");
            let loaded = backend.load(&key).await.unwrap();
            assert!(loaded.is_some(), "step {step} should exist");
            let bytes = loaded.unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(
                text.contains(&format!("\"step\":{step}")),
                "round-trip data mismatch at step {step}"
            );
        }

        // Verify list_keys returns 50 entries.
        let keys = backend.list_keys().await.unwrap();
        assert_eq!(keys.len(), 50, "expected 50 stored checkpoints, got {}", keys.len());

        // Delete all and confirm they're gone.
        for step in 0u32..50 {
            let key = format!("session:agent-1:step:{step}");
            backend.delete(&key).await.unwrap();
            let loaded = backend.load(&key).await.unwrap();
            assert!(loaded.is_none(), "step {step} should be deleted");
        }
    }
}
