//! # Module: Persistence
//!
//! ## Responsibility
//! Provides a pluggable persistence backend trait and a `FilePersistenceBackend`
//! that stores serialized state as files on disk using `tokio::fs`.
//!
//! ## Guarantees
//! - `PersistenceBackend` is object-safe via `async-trait`
//! - `FilePersistenceBackend` stores each key as a file in a configurable directory
//! - Non-panicking: all operations return `Result`
//!
//! ## Feature Gate
//! This module is only compiled when the `persistence` feature is enabled.

use crate::error::AgentRuntimeError;
use crate::util::djb2;
use async_trait::async_trait;
use futures::future::try_join_all;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;
use tokio::io::AsyncWriteExt as _;

// ── PersistenceBackend ────────────────────────────────────────────────────────

/// Pluggable storage backend for agent state (memory, graph, session checkpoints).
///
/// Implement this trait to persist agent state to any storage system
/// (disk, Redis, S3, databases, etc.).
#[async_trait]
pub trait PersistenceBackend: Send + Sync {
    /// Persist raw bytes under the given key.
    async fn save(&self, key: &str, value: &[u8]) -> Result<(), AgentRuntimeError>;

    /// Load raw bytes for the given key.
    ///
    /// Returns `None` if no data has been stored under this key.
    async fn load(&self, key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError>;

    /// Delete the entry for the given key. No-op if the key does not exist.
    async fn delete(&self, key: &str) -> Result<(), AgentRuntimeError>;

    /// Save multiple key-value pairs concurrently.
    ///
    /// The default implementation issues all `save` calls concurrently using
    /// [`futures::future::try_join_all`] so backends that incur per-call I/O
    /// latency benefit without any additional code.  Backends with their own
    /// native batch API should override this method.
    async fn batch_save(&self, items: &[(&str, &[u8])]) -> Result<(), AgentRuntimeError> {
        try_join_all(items.iter().map(|(key, value)| self.save(key, value))).await?;
        Ok(())
    }

    /// Load multiple keys concurrently. Returns a vec of `Option<Vec<u8>>` in the same order.
    ///
    /// The default implementation issues all `load` calls concurrently using
    /// [`futures::future::try_join_all`].  Backends with a native multi-get
    /// API should override this method.
    async fn batch_load(&self, keys: &[&str]) -> Result<Vec<Option<Vec<u8>>>, AgentRuntimeError> {
        try_join_all(keys.iter().map(|key| self.load(key))).await
    }
}

// ── FilePersistenceBackend ────────────────────────────────────────────────────

/// Persists data as files in a directory on disk.
///
/// Each key maps to a file named `<key>.bin` inside the base directory.
/// The base directory must exist before calling any methods.
///
/// # Key Sanitization
///
/// Characters that are invalid in file names on common operating systems
/// (`/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`) are replaced with `_`
/// before the path is constructed. This prevents path-traversal attacks
/// and ensures portability across Linux, macOS, and Windows.
///
/// # Concurrency
///
/// `FilePersistenceBackend` is `Clone` and `Send + Sync`. Each clone
/// shares the same `base_dir` via `Arc` and can be used from multiple
/// async tasks simultaneously.
///
/// Writes are atomic at the OS level: data is written to a uniquely-named
/// temporary file in the same directory and then renamed into place, so a
/// reader never observes a half-written file even if the process crashes.
/// Concurrent writes to the **same key** from multiple tasks are safe but
/// are not serialized — the final file content is determined by whichever
/// rename completes last.
///
/// # Crash Safety
///
/// When `sync_writes` is enabled (the default), the backend calls
/// `file.sync_all()` on the temporary file before renaming it into place.
/// This flushes the data and metadata to durable storage so that a process
/// crash between the write and the rename cannot leave data in the OS page
/// cache that is never persisted.  Disable `sync_writes` only when throughput
/// matters more than crash durability (e.g. in tests or non-critical caches).
#[derive(Debug, Clone)]
pub struct FilePersistenceBackend {
    /// Absolute path to the directory where `<key>.bin` files are stored.
    base_dir: Arc<PathBuf>,
    /// When `true` (the default), `sync_all()` is called on the temporary
    /// file before it is renamed into place, guaranteeing durability on crash.
    sync_writes: bool,
}

impl FilePersistenceBackend {
    /// Create a new backend that stores files in `base_dir`.
    ///
    /// The directory is not created automatically — it must already exist.
    /// `sync_writes` defaults to `true` for crash-safe durability.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: Arc::new(base_dir.into()),
            sync_writes: true,
        }
    }

    /// Create a backend with `sync_writes` explicitly configured.
    ///
    /// Pass `sync_writes: false` to skip the `sync_all()` call for higher
    /// write throughput when crash durability is not required (e.g. tests).
    pub fn new_with_sync(base_dir: impl Into<PathBuf>, sync_writes: bool) -> Self {
        Self {
            base_dir: Arc::new(base_dir.into()),
            sync_writes,
        }
    }

    /// Return whether this backend calls `sync_all()` after each write.
    pub fn sync_writes(&self) -> bool {
        self.sync_writes
    }

    /// Return the base directory that this backend stores files in.
    pub fn base_dir(&self) -> &std::path::Path {
        self.base_dir.as_ref()
    }

    /// Compute the file path for a given key.
    ///
    /// Uses a `<readable_prefix>-<djb2_hash>.bin` scheme to guarantee uniqueness
    /// even when two keys sanitize to the same string (e.g. `"a/b"` and `"a_b"`).
    /// The readable prefix aids manual inspection of the directory; the 16-digit
    /// hex hash is the canonical disambiguator.
    fn path_for(&self, key: &str) -> PathBuf {
        let hash = djb2(key);
        let sanitized = key.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        self.base_dir
            .join(format!("{sanitized}-{hash:016x}.bin"))
    }
}

#[async_trait]
impl PersistenceBackend for FilePersistenceBackend {
    #[tracing::instrument(skip(self, value), fields(key))]
    async fn save(&self, key: &str, value: &[u8]) -> Result<(), AgentRuntimeError> {
        let path = self.path_for(key);
        // Write to a unique temp file then atomically rename to the target path.
        // This prevents a reader from observing a half-written file if the
        // process crashes mid-write.
        let tmp_path = path.with_extension(format!("tmp-{}", Uuid::new_v4().simple()));
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("open tmp {tmp_path:?}: {e}")))?;
        file.write_all(value)
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("write tmp {tmp_path:?}: {e}")))?;
        // Flush OS page-cache to durable storage before rename so a crash
        // between write and rename cannot lose data that was never synced.
        if self.sync_writes {
            file.sync_all().await.map_err(|e| {
                AgentRuntimeError::Persistence(format!("sync_all {tmp_path:?}: {e}"))
            })?;
        }
        // Release the file handle so the rename succeeds on Windows.
        drop(file);
        tokio::fs::rename(&tmp_path, &path).await.map_err(|e| {
            AgentRuntimeError::Persistence(format!(
                "rename {tmp_path:?} -> {path:?}: {e}"
            ))
        })?;
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(key))]
    async fn load(&self, key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError> {
        let path = self.path_for(key);
        match tokio::fs::read(&path).await {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(AgentRuntimeError::Persistence(format!(
                "read {path:?}: {e}"
            ))),
        }
    }

    async fn delete(&self, key: &str) -> Result<(), AgentRuntimeError> {
        let path = self.path_for(key);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(AgentRuntimeError::Persistence(format!(
                "delete {path:?}: {e}"
            ))),
        }
    }

    /// Save multiple key-value pairs concurrently.
    ///
    /// All writes are issued simultaneously using `try_join_all` instead of
    /// the default sequential loop, so throughput scales with disk/OS concurrency.
    async fn batch_save(&self, items: &[(&str, &[u8])]) -> Result<(), AgentRuntimeError> {
        let futs: Vec<_> = items.iter().map(|(key, value)| self.save(key, value)).collect();
        try_join_all(futs).await?;
        Ok(())
    }

    /// Load multiple keys concurrently.
    ///
    /// All reads are issued simultaneously using `try_join_all`.
    /// Returns results in the same order as `keys`.
    async fn batch_load(&self, keys: &[&str]) -> Result<Vec<Option<Vec<u8>>>, AgentRuntimeError> {
        let futs: Vec<_> = keys.iter().map(|key| self.load(key)).collect();
        try_join_all(futs).await
    }
}

impl FilePersistenceBackend {
    /// List all keys currently stored in the backend.
    ///
    /// Returns the readable prefix portion of stored filenames (the part before
    /// the hash suffix). Useful for backup/restore tooling and debugging.
    pub async fn list_keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let mut entries = tokio::fs::read_dir(self.base_dir.as_ref())
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("list_keys readdir: {e}")))?;

        let mut keys = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("list_keys entry: {e}")))?
        {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".bin") {
                    if let Some(stem) = name.strip_suffix(".bin") {
                        // Strip the "-<16hexdigits>" suffix to recover the readable prefix.
                        if stem.len() > 17 {
                            let prefix = &stem[..stem.len() - 17]; // 16 hex + 1 dash
                            keys.push(prefix.to_owned());
                        }
                    }
                }
            }
        }
        Ok(keys)
    }

    /// Check whether a key exists in the backend without loading its value.
    ///
    /// Returns `Ok(true)` if the file for `key` is present in the store.
    pub async fn exists(&self, key: &str) -> Result<bool, AgentRuntimeError> {
        let path = self.path_for(key);
        Ok(tokio::fs::metadata(&path).await.is_ok())
    }

    /// Return the number of keys currently stored in the backend.
    ///
    /// More efficient than `list_keys().await?.len()` for callers that only
    /// need the count, as it avoids building the string list.
    pub async fn key_count(&self) -> Result<usize, AgentRuntimeError> {
        let mut entries = tokio::fs::read_dir(self.base_dir.as_ref())
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("key_count readdir: {e}")))?;
        let mut count = 0usize;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| AgentRuntimeError::Persistence(format!("key_count entry: {e}")))?
        {
            if entry
                .file_name()
                .to_str()
                .map_or(false, |n| n.ends_with(".bin"))
            {
                count += 1;
            }
        }
        Ok(count)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn temp_backend() -> (FilePersistenceBackend, tempdir::Handle) {
        // Fall back to a simple temp path approach without the tempdir crate.
        // We use tokio's temp file support via std's temp_dir.
        let dir = std::env::temp_dir().join(format!("agent_runtime_test_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let backend = FilePersistenceBackend::new(dir.clone());
        (backend, tempdir::Handle { path: dir })
    }

    // Use a simple inline struct instead of a crate dependency.
    mod tempdir {
        pub struct Handle {
            pub path: std::path::PathBuf,
        }
        impl Drop for Handle {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.path);
            }
        }
    }

    #[tokio::test]
    async fn test_file_backend_save_and_load() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        backend.save("test-key", b"hello world").await.unwrap();
        let loaded = backend.load("test-key").await.unwrap();
        assert_eq!(loaded, Some(b"hello world".to_vec()));
    }

    #[tokio::test]
    async fn test_file_backend_load_missing_returns_none() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        let loaded = backend.load("nonexistent").await.unwrap();
        assert_eq!(loaded, None);
    }

    #[tokio::test]
    async fn test_file_backend_delete_removes_file() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        backend.save("to-delete", b"data").await.unwrap();
        backend.delete("to-delete").await.unwrap();
        let loaded = backend.load("to-delete").await.unwrap();
        assert_eq!(loaded, None);
    }

    #[tokio::test]
    async fn test_file_backend_delete_missing_is_noop() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        // Should not error
        backend.delete("never-existed").await.unwrap();
    }

    #[tokio::test]
    async fn test_file_backend_key_sanitization() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        // Key with path separators must not escape base_dir
        backend.save("agent/session:1", b"data").await.unwrap();
        let loaded = backend.load("agent/session:1").await.unwrap();
        assert_eq!(loaded, Some(b"data".to_vec()));
    }

    #[tokio::test]
    async fn test_file_backend_collision_resistant_keys() {
        // "a/b" and "a_b" both sanitize to "a_b" with naive replacement.
        // The hash suffix must differentiate them.
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        backend.save("a/b", b"slash").await.unwrap();
        backend.save("a_b", b"underscore").await.unwrap();

        let loaded_slash = backend.load("a/b").await.unwrap();
        let loaded_under = backend.load("a_b").await.unwrap();
        assert_eq!(loaded_slash, Some(b"slash".to_vec()));
        assert_eq!(loaded_under, Some(b"underscore".to_vec()));
        assert_ne!(
            loaded_slash, loaded_under,
            "keys with the same sanitized form must not collide"
        );
    }

    #[tokio::test]
    async fn test_batch_save_and_load() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        let data: Vec<(&str, Vec<u8>)> = vec![
            ("batch-key-1", b"value1".to_vec()),
            ("batch-key-2", b"value2".to_vec()),
        ];
        let refs: Vec<(&str, &[u8])> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        backend.batch_save(&refs).await.unwrap();

        let keys = vec!["batch-key-1", "batch-key-2", "batch-key-missing"];
        let results = backend.batch_load(&keys).await.unwrap();
        assert_eq!(results[0], Some(b"value1".to_vec()));
        assert_eq!(results[1], Some(b"value2".to_vec()));
        assert_eq!(results[2], None);
    }

    #[tokio::test]
    async fn test_file_backend_list_keys() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);

        backend.save("my-session", b"data1").await.unwrap();
        backend.save("another-key", b"data2").await.unwrap();

        let keys = backend.list_keys().await.unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn test_persistence_backend_object_safe() {
        // Verify FilePersistenceBackend can be used as a trait object.
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend: Arc<dyn PersistenceBackend> = Arc::new(FilePersistenceBackend::new(&dir));
        backend.save("obj-safe", b"ok").await.unwrap();
        let r = backend.load("obj-safe").await.unwrap();
        assert_eq!(r, Some(b"ok".to_vec()));
    }

    // ── Round 11: key_count ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_key_count_zero_for_empty_directory() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);
        assert_eq!(backend.key_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_key_count_matches_number_of_saves() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);
        backend.save("k1", b"v1").await.unwrap();
        backend.save("k2", b"v2").await.unwrap();
        backend.save("k3", b"v3").await.unwrap();
        assert_eq!(backend.key_count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_key_count_decrements_after_delete() {
        let dir = std::env::temp_dir().join(format!("art_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let _guard = tempdir::Handle { path: dir.clone() };
        let backend = FilePersistenceBackend::new(&dir);
        backend.save("key", b"val").await.unwrap();
        assert_eq!(backend.key_count().await.unwrap(), 1);
        backend.delete("key").await.unwrap();
        assert_eq!(backend.key_count().await.unwrap(), 0);
    }
}
