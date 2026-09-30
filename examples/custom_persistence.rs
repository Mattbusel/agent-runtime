//! Example: Custom Persistence Backend
//!
//! Shows how to implement `PersistenceBackend` to persist agent sessions
//! to an in-memory store (useful as a test double or a starting template
//! for a Redis / S3 / database backend).

use async_trait::async_trait;
use llm_agent_runtime::agent::AgentConfig;
use llm_agent_runtime::error::AgentRuntimeError;
use llm_agent_runtime::memory::AgentId;
use llm_agent_runtime::persistence::PersistenceBackend;
use llm_agent_runtime::runtime::AgentRuntime;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ── Custom in-memory persistence backend ─────────────────────────────────────

/// A trivial in-memory persistence backend.
///
/// Useful for testing; not suitable for production (state is lost on restart).
#[derive(Debug, Clone, Default)]
struct InMemoryBackend {
    store: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

#[async_trait]
impl PersistenceBackend for InMemoryBackend {
    async fn save(&self, key: &str, value: &[u8]) -> Result<(), AgentRuntimeError> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| AgentRuntimeError::Persistence("lock poisoned".into()))?;
        store.insert(key.to_owned(), value.to_vec());
        println!("[InMemoryBackend] saved key={key} ({} bytes)", value.len());
        Ok(())
    }

    async fn load(&self, key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError> {
        let store = self
            .store
            .lock()
            .map_err(|_| AgentRuntimeError::Persistence("lock poisoned".into()))?;
        Ok(store.get(key).cloned())
    }

    async fn delete(&self, key: &str) -> Result<(), AgentRuntimeError> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| AgentRuntimeError::Persistence("lock poisoned".into()))?;
        store.remove(key);
        Ok(())
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    let backend = Arc::new(InMemoryBackend::default());

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(3, "my-model"))
        .with_checkpoint_backend(Arc::clone(&backend) as Arc<dyn PersistenceBackend>)
        .build();

    let session = runtime
        .run_agent(
            AgentId::new("agent-1"),
            "What is 2 + 2?",
            |_ctx: String| async { "Thought: easy\nAction: FINAL_ANSWER 4".to_string() },
        )
        .await?;

    println!("Session ID: {}", session.session_id);
    println!("Steps: {}", session.step_count());

    // Verify the checkpoint was saved.
    let checkpoint_key = format!("session:{}", session.session_id);
    if let Some(bytes) = backend.load(&checkpoint_key).await? {
        println!(
            "Checkpoint saved ({} bytes) — load it back with AgentSession::load_checkpoint",
            bytes.len()
        );
    }

    Ok(())
}
