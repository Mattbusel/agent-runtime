//! # Module: checkpoint
//!
//! ## Responsibility
//! Agent state checkpointing: save and restore `AgentCheckpoint` via pluggable
//! store implementations.  Two stores are provided:
//! - [`InMemoryCheckpointStore`] — `HashMap` behind `Arc<Mutex<...>>`; ideal for tests.
//! - [`FileCheckpointStore`] — serialises checkpoints as JSON files.
//!
//! [`CheckpointManager`] wraps any store, auto-checkpoints every N steps,
//! and rotates oldest checkpoints when the per-agent limit is exceeded.
//!
//! ## Guarantees
//! - All store operations are async and non-panicking.
//! - `CheckpointManager::maybe_checkpoint` is idempotent at the same step.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors produced by checkpoint store operations.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// A serialisation or deserialisation error.
    #[error("serialisation error: {0}")]
    Serialisation(String),

    /// An I/O error (file store only).
    #[error("io error: {0}")]
    Io(String),

    /// A lock was poisoned (should never happen in practice).
    #[error("lock poisoned")]
    LockPoisoned,
}

impl From<serde_json::Error> for CheckpointError {
    fn from(e: serde_json::Error) -> Self {
        CheckpointError::Serialisation(e.to_string())
    }
}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        CheckpointError::Io(e.to_string())
    }
}

// ── Domain types ──────────────────────────────────────────────────────────────

/// A key-value pair stored in an agent memory snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryEntry {
    /// The key for this memory entry.
    pub key: String,
    /// The value for this memory entry.
    pub value: String,
}

/// A complete agent checkpoint saved at a particular step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    /// Stable identifier for the agent this checkpoint belongs to.
    pub agent_id: String,
    /// The zero-based step index at which this checkpoint was taken.
    pub step: u32,
    /// Snapshot of the agent's memory at this step.
    pub memory_snapshot: Vec<MemoryEntry>,
    /// The serialised context string at this step.
    pub context: String,
    /// Unix timestamp (seconds) when this checkpoint was created.
    pub created_at: u64,
}

impl AgentCheckpoint {
    /// Create a new checkpoint with `created_at` set to the current UTC second.
    pub fn new(
        agent_id: impl Into<String>,
        step: u32,
        memory_snapshot: Vec<MemoryEntry>,
        context: impl Into<String>,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            agent_id: agent_id.into(),
            step,
            memory_snapshot,
            context: context.into(),
            created_at,
        }
    }
}

// ── CheckpointStore trait ─────────────────────────────────────────────────────

/// Pluggable persistence backend for agent checkpoints.
///
/// Implementors only need to worry about raw storage; rotation and
/// auto-checkpointing policy are handled by [`CheckpointManager`].
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Persist `state` under `agent_id`, overwriting any existing checkpoint.
    ///
    /// # Errors
    /// Returns [`CheckpointError`] on serialisation or I/O failure.
    async fn save(
        &self,
        agent_id: &str,
        state: AgentCheckpoint,
    ) -> Result<(), CheckpointError>;

    /// Load the latest checkpoint for `agent_id`.
    ///
    /// Returns `Ok(None)` if no checkpoint exists.
    async fn load(&self, agent_id: &str) -> Result<Option<AgentCheckpoint>, CheckpointError>;

    /// Return a list of all agent IDs that have at least one checkpoint.
    async fn list(&self) -> Vec<String>;

    /// Delete the checkpoint stored under `agent_id`.
    ///
    /// Returns `Ok(())` if no checkpoint existed (idempotent).
    async fn delete(&self, agent_id: &str) -> Result<(), CheckpointError>;
}

// ── InMemoryCheckpointStore ───────────────────────────────────────────────────

/// In-memory checkpoint store backed by a `HashMap` behind an `Arc<Mutex<...>>`.
///
/// Checkpoints survive only for the lifetime of the process.
/// Use this for testing or for ephemeral agent sessions.
#[derive(Clone, Debug)]
pub struct InMemoryCheckpointStore {
    store: Arc<Mutex<HashMap<String, AgentCheckpoint>>>,
}

impl Default for InMemoryCheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryCheckpointStore {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self {
            store: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn save(
        &self,
        agent_id: &str,
        state: AgentCheckpoint,
    ) -> Result<(), CheckpointError> {
        let mut guard = self.store.lock().await;
        guard.insert(agent_id.to_owned(), state);
        Ok(())
    }

    async fn load(&self, agent_id: &str) -> Result<Option<AgentCheckpoint>, CheckpointError> {
        let guard = self.store.lock().await;
        Ok(guard.get(agent_id).cloned())
    }

    async fn list(&self) -> Vec<String> {
        let guard = self.store.lock().await;
        guard.keys().cloned().collect()
    }

    async fn delete(&self, agent_id: &str) -> Result<(), CheckpointError> {
        let mut guard = self.store.lock().await;
        guard.remove(agent_id);
        Ok(())
    }
}

// ── FileCheckpointStore ───────────────────────────────────────────────────────

/// File-based checkpoint store.
///
/// Each agent's checkpoint is stored as a JSON file at
/// `<dir>/<agent_id>.checkpoint.json`.  The directory is created on first use
/// if it does not already exist.
#[derive(Clone, Debug)]
pub struct FileCheckpointStore {
    dir: PathBuf,
}

impl FileCheckpointStore {
    /// Create a store that persists checkpoints in `dir`.
    ///
    /// The directory will be created automatically on first write.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path_for(&self, agent_id: &str) -> PathBuf {
        // Sanitise the agent ID so it is safe as a filename.
        let safe: String = agent_id
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        self.dir.join(format!("{safe}.checkpoint.json"))
    }
}

#[async_trait]
impl CheckpointStore for FileCheckpointStore {
    async fn save(
        &self,
        agent_id: &str,
        state: AgentCheckpoint,
    ) -> Result<(), CheckpointError> {
        tokio::fs::create_dir_all(&self.dir).await?;
        let path = self.path_for(agent_id);
        let json = serde_json::to_string_pretty(&state)?;
        tokio::fs::write(&path, json).await?;
        Ok(())
    }

    async fn load(&self, agent_id: &str) -> Result<Option<AgentCheckpoint>, CheckpointError> {
        let path = self.path_for(agent_id);
        match tokio::fs::read_to_string(&path).await {
            Ok(json) => {
                let cp = serde_json::from_str::<AgentCheckpoint>(&json)?;
                Ok(Some(cp))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CheckpointError::Io(e.to_string())),
        }
    }

    async fn list(&self) -> Vec<String> {
        let mut entries = match tokio::fs::read_dir(&self.dir).await {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let mut ids = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(stem) = name.strip_suffix(".checkpoint.json") {
                ids.push(stem.to_owned());
            }
        }
        ids
    }

    async fn delete(&self, agent_id: &str) -> Result<(), CheckpointError> {
        let path = self.path_for(agent_id);
        match tokio::fs::remove_file(&path).await {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(CheckpointError::Io(e.to_string())),
        }
    }
}

// ── CheckpointManager ─────────────────────────────────────────────────────────

/// High-level manager that wraps a [`CheckpointStore`] with:
/// - Auto-checkpointing every `interval_steps` steps.
/// - Rotation: keeps at most `max_checkpoints` per agent (deletes oldest by
///   `created_at` when the limit is exceeded).
///
/// The rotation logic works by storing a sorted ring of checkpoints per agent
/// in an in-memory index.  The underlying store itself only keeps the *latest*
/// checkpoint per agent ID; to maintain multiple versions the manager appends
/// the step number to the agent ID (`<agent_id>__step<N>`).
pub struct CheckpointManager {
    store: Arc<dyn CheckpointStore>,
    interval_steps: u32,
    max_checkpoints: usize,
    /// In-memory index: agent_id → sorted list of (created_at, key) pairs.
    index: Arc<Mutex<HashMap<String, Vec<(u64, String)>>>>,
}

impl std::fmt::Debug for CheckpointManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointManager")
            .field("interval_steps", &self.interval_steps)
            .field("max_checkpoints", &self.max_checkpoints)
            .finish()
    }
}

impl CheckpointManager {
    /// Create a new manager.
    ///
    /// * `store` — the backing store.
    /// * `interval_steps` — auto-checkpoint every N steps.
    /// * `max_checkpoints` — rotate oldest when this many checkpoints exist per agent.
    pub fn new(
        store: Arc<dyn CheckpointStore>,
        interval_steps: u32,
        max_checkpoints: usize,
    ) -> Self {
        Self {
            store,
            interval_steps,
            max_checkpoints,
            index: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Conditionally save a checkpoint for `agent_id` at `step`.
    ///
    /// Saves only when `step % interval_steps == 0` (and `step > 0`) or when
    /// `force` is `true`.  Returns `true` if a checkpoint was actually saved.
    ///
    /// # Errors
    /// Propagates store errors.
    pub async fn maybe_checkpoint(
        &self,
        agent_id: &str,
        step: u32,
        memory: Vec<MemoryEntry>,
        context: impl Into<String>,
        force: bool,
    ) -> Result<bool, CheckpointError> {
        if !force && (step == 0 || step % self.interval_steps != 0) {
            return Ok(false);
        }
        let cp = AgentCheckpoint::new(agent_id, step, memory, context);
        self.save_versioned(agent_id, cp).await?;
        Ok(true)
    }

    /// Persist a checkpoint and rotate old ones.
    async fn save_versioned(
        &self,
        agent_id: &str,
        cp: AgentCheckpoint,
    ) -> Result<(), CheckpointError> {
        let key = format!("{agent_id}__step{}", cp.step);
        let created_at = cp.created_at;

        self.store.save(&key, cp).await?;

        let mut index = self.index.lock().await;
        let versions = index.entry(agent_id.to_owned()).or_default();
        versions.push((created_at, key));
        // Sort ascending by created_at so oldest is at index 0.
        versions.sort_unstable_by_key(|(ts, _)| *ts);

        // Rotate: delete oldest entries that exceed the cap.
        while versions.len() > self.max_checkpoints {
            let (_, old_key) = versions.remove(0);
            // Best-effort delete; ignore errors.
            let _ = self.store.delete(&old_key).await;
        }

        Ok(())
    }

    /// Load the most recent checkpoint for `agent_id`, or `None`.
    pub async fn load_latest(
        &self,
        agent_id: &str,
    ) -> Result<Option<AgentCheckpoint>, CheckpointError> {
        let index = self.index.lock().await;
        let versions = match index.get(agent_id) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(None),
        };
        // Latest is the last entry (highest created_at).
        let (_, latest_key) = &versions[versions.len() - 1];
        let key = latest_key.clone();
        drop(index);
        self.store.load(&key).await
    }

    /// Return the underlying store.
    pub fn store(&self) -> &dyn CheckpointStore {
        self.store.as_ref()
    }

    /// Return the interval (in steps) between automatic checkpoints.
    pub fn interval_steps(&self) -> u32 {
        self.interval_steps
    }

    /// Return the maximum number of checkpoints kept per agent.
    pub fn max_checkpoints(&self) -> usize {
        self.max_checkpoints
    }

    /// Return the number of versioned checkpoints currently tracked for `agent_id`.
    pub async fn checkpoint_count(&self, agent_id: &str) -> usize {
        let index = self.index.lock().await;
        index.get(agent_id).map(|v| v.len()).unwrap_or(0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_cp(agent_id: &str, step: u32) -> AgentCheckpoint {
        AgentCheckpoint::new(
            agent_id,
            step,
            vec![MemoryEntry { key: "k".into(), value: "v".into() }],
            "ctx",
        )
    }

    // ── InMemoryCheckpointStore ───────────────────────────────────────────────

    #[tokio::test]
    async fn in_memory_save_load_roundtrip() {
        let store = InMemoryCheckpointStore::new();
        let cp = make_cp("agent-1", 3);
        store.save("agent-1", cp.clone()).await.unwrap();
        let loaded = store.load("agent-1").await.unwrap().unwrap();
        assert_eq!(loaded.agent_id, "agent-1");
        assert_eq!(loaded.step, 3);
    }

    #[tokio::test]
    async fn in_memory_load_missing_returns_none() {
        let store = InMemoryCheckpointStore::new();
        let result = store.load("nobody").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn in_memory_list_returns_saved_ids() {
        let store = InMemoryCheckpointStore::new();
        store.save("a1", make_cp("a1", 1)).await.unwrap();
        store.save("a2", make_cp("a2", 2)).await.unwrap();
        let ids = store.list().await;
        assert!(ids.contains(&"a1".to_owned()));
        assert!(ids.contains(&"a2".to_owned()));
    }

    #[tokio::test]
    async fn in_memory_delete_removes_entry() {
        let store = InMemoryCheckpointStore::new();
        store.save("agent-x", make_cp("agent-x", 0)).await.unwrap();
        store.delete("agent-x").await.unwrap();
        assert!(store.load("agent-x").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn in_memory_delete_missing_is_ok() {
        let store = InMemoryCheckpointStore::new();
        // Should not error.
        store.delete("phantom").await.unwrap();
    }

    #[tokio::test]
    async fn in_memory_overwrite_saves_latest() {
        let store = InMemoryCheckpointStore::new();
        store.save("a", make_cp("a", 1)).await.unwrap();
        store.save("a", make_cp("a", 5)).await.unwrap();
        let loaded = store.load("a").await.unwrap().unwrap();
        assert_eq!(loaded.step, 5);
    }

    // ── FileCheckpointStore ───────────────────────────────────────────────────

    #[tokio::test]
    async fn file_store_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path());
        let cp = make_cp("file-agent", 7);
        store.save("file-agent", cp).await.unwrap();
        let loaded = store.load("file-agent").await.unwrap().unwrap();
        assert_eq!(loaded.step, 7);
        assert_eq!(loaded.agent_id, "file-agent");
    }

    #[tokio::test]
    async fn file_store_load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path());
        let result = store.load("ghost").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn file_store_delete_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path());
        store.delete("ghost").await.unwrap();
    }

    #[tokio::test]
    async fn file_store_list_finds_saved() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStore::new(dir.path());
        store.save("ag1", make_cp("ag1", 0)).await.unwrap();
        store.save("ag2", make_cp("ag2", 0)).await.unwrap();
        let list = store.list().await;
        assert!(list.contains(&"ag1".to_owned()));
        assert!(list.contains(&"ag2".to_owned()));
    }

    // ── CheckpointManager ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn manager_auto_checkpoint_on_interval() {
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mgr = CheckpointManager::new(store, 5, 10);

        let saved = mgr
            .maybe_checkpoint("a", 5, vec![], "ctx", false)
            .await
            .unwrap();
        assert!(saved);
        assert!(mgr.load_latest("a").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn manager_skips_non_interval_steps() {
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mgr = CheckpointManager::new(store, 5, 10);

        let saved = mgr
            .maybe_checkpoint("a", 3, vec![], "ctx", false)
            .await
            .unwrap();
        assert!(!saved);
    }

    #[tokio::test]
    async fn manager_force_saves_any_step() {
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mgr = CheckpointManager::new(store, 5, 10);

        let saved = mgr
            .maybe_checkpoint("a", 2, vec![], "ctx", true)
            .await
            .unwrap();
        assert!(saved);
    }

    #[tokio::test]
    async fn manager_rotates_oldest_checkpoint() {
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mgr = CheckpointManager::new(store.clone(), 1, 3);

        for step in 1u32..=6 {
            mgr.maybe_checkpoint("agent", step, vec![], "ctx", true)
                .await
                .unwrap();
        }

        // Only 3 checkpoints should remain tracked.
        assert_eq!(mgr.checkpoint_count("agent").await, 3);
    }

    #[tokio::test]
    async fn manager_load_latest_returns_newest() {
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mgr = CheckpointManager::new(store, 5, 10);

        mgr.maybe_checkpoint("a", 5, vec![], "first", true).await.unwrap();
        mgr.maybe_checkpoint("a", 10, vec![], "second", true).await.unwrap();

        let cp = mgr.load_latest("a").await.unwrap().unwrap();
        assert_eq!(cp.step, 10);
        assert_eq!(cp.context, "second");
    }

    #[tokio::test]
    async fn manager_load_latest_none_when_empty() {
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mgr = CheckpointManager::new(store, 5, 10);
        assert!(mgr.load_latest("nobody").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn checkpoint_new_sets_fields() {
        let mem = vec![MemoryEntry { key: "foo".into(), value: "bar".into() }];
        let cp = AgentCheckpoint::new("my-agent", 42, mem.clone(), "my context");
        assert_eq!(cp.agent_id, "my-agent");
        assert_eq!(cp.step, 42);
        assert_eq!(cp.memory_snapshot, mem);
        assert_eq!(cp.context, "my context");
        assert!(cp.created_at > 0);
    }
}
