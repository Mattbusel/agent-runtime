//! # Module: snapshot
//!
//! ## Responsibility
//! Snapshotting and state restoration for agent state.
//!
//! Provides:
//! - [`SnapshotId`]: a sequential newtype wrapping `u64`.
//! - [`Snapshot`]: a stored byte blob with metadata and a checksum.
//! - [`Snapshottable`]: trait for types that can be serialised to/from bytes.
//! - [`SnapshotStore`]: bounded deque of snapshots with eviction of oldest.
//! - [`AgentStateSnapshot`]: example [`Snapshottable`] implementation.
//! - [`SnapshotError`]: errors that can arise during snapshot operations.
//!
//! ## Checksum
//! A CRC-32 using the reversed polynomial `0xEDB88320` (standard CRC-32/ISO-HDLC).
//!
//! ## Guarantees
//! - Eviction happens automatically when `max_snapshots` is reached.
//! - `verify()` always recomputes the checksum from stored data.
//! - No `.unwrap()` or `.expect()` in production paths.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── CRC-32 (polynomial 0xEDB88320) ───────────────────────────────────────────

const CRC32_POLY: u32 = 0xEDB8_8320;

/// Compute CRC-32/ISO-HDLC of a byte slice using the reflected polynomial.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ CRC32_POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

// ── SnapshotId ────────────────────────────────────────────────────────────────

/// A sequential, opaque snapshot identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SnapshotId(pub u64);

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "snap:{}", self.0)
    }
}

// ── SnapshotError ─────────────────────────────────────────────────────────────

/// Errors that can arise during snapshot operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SnapshotError {
    /// No snapshot with the requested ID exists.
    #[error("snapshot not found: {0}")]
    NotFound(SnapshotId),
    /// The stored checksum does not match the recomputed checksum.
    #[error("snapshot data is corrupted (checksum mismatch)")]
    CorruptedData,
    /// Deserialisation of the raw bytes failed.
    #[error("deserialisation failed: {0}")]
    DeserializationFailed(String),
    /// The store is at capacity and eviction is disabled.
    #[error("maximum snapshot count reached")]
    MaxSnapshotsReached,
}

// ── Snapshot ──────────────────────────────────────────────────────────────────

/// A stored snapshot: serialised bytes, metadata, and a checksum.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Unique, sequential identifier.
    pub id: SnapshotId,
    /// Human-readable label for this snapshot.
    pub label: String,
    /// Unix timestamp (seconds) when the snapshot was created.
    pub created_at: u64,
    /// Raw serialised state bytes.
    pub data: Vec<u8>,
    /// CRC-32 checksum of `data`.
    pub checksum: u32,
}

impl Snapshot {
    /// Create a new snapshot, computing the checksum automatically.
    pub fn new(id: SnapshotId, label: impl Into<String>, data: Vec<u8>) -> Self {
        let checksum = crc32(&data);
        Self {
            id,
            label: label.into(),
            created_at: now_secs(),
            data,
            checksum,
        }
    }

    /// Returns `true` if the stored checksum matches the recomputed checksum.
    pub fn verify(&self) -> bool {
        crc32(&self.data) == self.checksum
    }
}

// ── Snapshottable trait ───────────────────────────────────────────────────────

/// Types that can be serialised to a byte vector and deserialised back.
pub trait Snapshottable: Sized {
    /// Serialise `self` to a byte vector.
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserialise from a byte slice.
    ///
    /// # Errors
    /// Returns [`SnapshotError::DeserializationFailed`] if the bytes cannot be parsed.
    fn from_bytes(data: &[u8]) -> Result<Self, SnapshotError>;
}

// ── SnapshotStore ─────────────────────────────────────────────────────────────

/// A bounded, ordered store of snapshots.
///
/// When `max_snapshots` is reached, the oldest snapshot is evicted before a
/// new one is inserted.
#[derive(Debug)]
pub struct SnapshotStore {
    snapshots: VecDeque<Snapshot>,
    max_snapshots: usize,
    next_id: u64,
}

impl SnapshotStore {
    /// Create a new store with the given capacity limit.
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: VecDeque::new(),
            max_snapshots,
            next_id: 1,
        }
    }

    /// Serialise `value`, store it as a new snapshot, and return its [`SnapshotId`].
    ///
    /// If the store is at capacity, the oldest snapshot is evicted first.
    pub fn save<T: Snapshottable>(&mut self, value: &T, label: &str) -> SnapshotId {
        if self.snapshots.len() >= self.max_snapshots && self.max_snapshots > 0 {
            self.snapshots.pop_front();
        }
        let id = SnapshotId(self.next_id);
        self.next_id += 1;
        let data = value.to_bytes();
        let snap = Snapshot::new(id, label, data);
        self.snapshots.push_back(snap);
        id
    }

    /// Restore a snapshot by ID, verifying its checksum before deserialising.
    ///
    /// # Errors
    /// - [`SnapshotError::NotFound`] if no snapshot with `id` exists.
    /// - [`SnapshotError::CorruptedData`] if the checksum fails.
    /// - [`SnapshotError::DeserializationFailed`] if deserialisation fails.
    pub fn restore<T: Snapshottable>(&self, id: SnapshotId) -> Result<T, SnapshotError> {
        let snap = self
            .snapshots
            .iter()
            .find(|s| s.id == id)
            .ok_or(SnapshotError::NotFound(id))?;
        if !snap.verify() {
            return Err(SnapshotError::CorruptedData);
        }
        T::from_bytes(&snap.data)
    }

    /// Restore the most recently saved snapshot.
    ///
    /// # Errors
    /// - [`SnapshotError::NotFound`] with id `SnapshotId(0)` if the store is empty.
    /// - [`SnapshotError::CorruptedData`] if the checksum fails.
    /// - [`SnapshotError::DeserializationFailed`] if deserialisation fails.
    pub fn restore_latest<T: Snapshottable>(&self) -> Result<T, SnapshotError> {
        let snap = self
            .snapshots
            .back()
            .ok_or(SnapshotError::NotFound(SnapshotId(0)))?;
        if !snap.verify() {
            return Err(SnapshotError::CorruptedData);
        }
        T::from_bytes(&snap.data)
    }

    /// Return a list of `(id, label, created_at)` tuples for all stored snapshots,
    /// ordered from oldest to newest.
    pub fn list(&self) -> Vec<(SnapshotId, &str, u64)> {
        self.snapshots
            .iter()
            .map(|s| (s.id, s.label.as_str(), s.created_at))
            .collect()
    }

    /// Delete the snapshot with the given ID.
    ///
    /// Returns `true` if the snapshot was found and removed, `false` otherwise.
    pub fn delete(&mut self, id: SnapshotId) -> bool {
        let before = self.snapshots.len();
        self.snapshots.retain(|s| s.id != id);
        self.snapshots.len() < before
    }

    /// Return the number of snapshots currently stored.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Return `true` if the store contains no snapshots.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

// ── AgentStateSnapshot ────────────────────────────────────────────────────────

/// An example [`Snapshottable`] type representing an agent's runtime state.
///
/// Serialised format (line-delimited):
/// ```text
/// task_id=<task_id>
/// step=<step>
/// <key>=<value>
/// ...
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentStateSnapshot {
    /// Identifier of the current task.
    pub task_id: String,
    /// Current step index.
    pub step: u32,
    /// Named variables in the agent's working state.
    pub variables: HashMap<String, String>,
}

impl AgentStateSnapshot {
    /// Create a new agent state snapshot.
    pub fn new(task_id: impl Into<String>, step: u32, variables: HashMap<String, String>) -> Self {
        Self {
            task_id: task_id.into(),
            step,
            variables,
        }
    }
}

impl Snapshottable for AgentStateSnapshot {
    fn to_bytes(&self) -> Vec<u8> {
        let mut lines = Vec::new();
        lines.push(format!("task_id={}", self.task_id));
        lines.push(format!("step={}", self.step));
        // Sort variables for deterministic output.
        let mut vars: Vec<(&String, &String)> = self.variables.iter().collect();
        vars.sort_by_key(|(k, _)| k.as_str());
        for (k, v) in vars {
            lines.push(format!("{k}={v}"));
        }
        lines.join("\n").into_bytes()
    }

    fn from_bytes(data: &[u8]) -> Result<Self, SnapshotError> {
        let text = std::str::from_utf8(data).map_err(|e| {
            SnapshotError::DeserializationFailed(format!("utf8 error: {e}"))
        })?;

        let mut task_id = None::<String>;
        let mut step = None::<u32>;
        let mut variables = HashMap::new();

        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| {
                SnapshotError::DeserializationFailed(format!("malformed line: {line:?}"))
            })?;
            match key {
                "task_id" => task_id = Some(value.to_owned()),
                "step" => {
                    step = Some(value.parse::<u32>().map_err(|e| {
                        SnapshotError::DeserializationFailed(format!("bad step: {e}"))
                    })?);
                }
                other => {
                    variables.insert(other.to_owned(), value.to_owned());
                }
            }
        }

        let task_id = task_id.ok_or_else(|| {
            SnapshotError::DeserializationFailed("missing task_id".to_owned())
        })?;
        let step = step.ok_or_else(|| {
            SnapshotError::DeserializationFailed("missing step".to_owned())
        })?;

        Ok(Self {
            task_id,
            step,
            variables,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_state(task_id: &str, step: u32) -> AgentStateSnapshot {
        let mut vars = HashMap::new();
        vars.insert("foo".to_owned(), "bar".to_owned());
        vars.insert("count".to_owned(), "42".to_owned());
        AgentStateSnapshot::new(task_id, step, vars)
    }

    // ── Checksum ──────────────────────────────────────────────────────────────

    #[test]
    fn checksum_verifies_on_intact_data() {
        let state = make_state("t1", 1);
        let data = state.to_bytes();
        let snap = Snapshot::new(SnapshotId(1), "label", data);
        assert!(snap.verify());
    }

    #[test]
    fn corrupted_data_fails_verify() {
        let state = make_state("t1", 1);
        let data = state.to_bytes();
        let mut snap = Snapshot::new(SnapshotId(1), "label", data);
        // Flip a byte to corrupt the data without changing the stored checksum.
        if let Some(b) = snap.data.first_mut() {
            *b ^= 0xFF;
        }
        assert!(!snap.verify());
    }

    // ── Save + Restore roundtrip ──────────────────────────────────────────────

    #[test]
    fn save_and_restore_roundtrip() {
        let mut store = SnapshotStore::new(10);
        let state = make_state("task-42", 7);
        let id = store.save(&state, "checkpoint");
        let restored: AgentStateSnapshot = store.restore(id).unwrap();
        assert_eq!(restored, state);
    }

    #[test]
    fn restore_latest_returns_most_recent() {
        let mut store = SnapshotStore::new(10);
        let s1 = make_state("t1", 1);
        let s2 = make_state("t2", 2);
        store.save(&s1, "first");
        store.save(&s2, "second");
        let latest: AgentStateSnapshot = store.restore_latest().unwrap();
        assert_eq!(latest, s2);
    }

    #[test]
    fn restore_not_found_returns_error() {
        let store = SnapshotStore::new(10);
        let err = store.restore::<AgentStateSnapshot>(SnapshotId(99)).unwrap_err();
        assert_eq!(err, SnapshotError::NotFound(SnapshotId(99)));
    }

    #[test]
    fn restore_latest_empty_store_returns_error() {
        let store = SnapshotStore::new(10);
        let err = store.restore_latest::<AgentStateSnapshot>().unwrap_err();
        assert_eq!(err, SnapshotError::NotFound(SnapshotId(0)));
    }

    // ── Checksum corruption detected on restore ───────────────────────────────

    #[test]
    fn restore_corrupted_snapshot_returns_error() {
        let mut store = SnapshotStore::new(10);
        let state = make_state("t1", 1);
        let id = store.save(&state, "label");

        // Corrupt the stored data directly.
        if let Some(snap) = store.snapshots.iter_mut().find(|s| s.id == id) {
            if let Some(b) = snap.data.first_mut() {
                *b ^= 0xFF;
            }
        }

        let err = store.restore::<AgentStateSnapshot>(id).unwrap_err();
        assert_eq!(err, SnapshotError::CorruptedData);
    }

    // ── Max-snapshots eviction ────────────────────────────────────────────────

    #[test]
    fn max_snapshots_evicts_oldest() {
        let mut store = SnapshotStore::new(3);
        let s = make_state("t", 0);

        let id1 = store.save(&s, "first");
        let _id2 = store.save(&s, "second");
        let _id3 = store.save(&s, "third");

        assert_eq!(store.len(), 3);

        // Saving a fourth snapshot should evict id1.
        let _id4 = store.save(&s, "fourth");
        assert_eq!(store.len(), 3);

        // id1 should no longer be findable.
        let err = store.restore::<AgentStateSnapshot>(id1).unwrap_err();
        assert_eq!(err, SnapshotError::NotFound(id1));
    }

    // ── Delete ────────────────────────────────────────────────────────────────

    #[test]
    fn delete_removes_snapshot() {
        let mut store = SnapshotStore::new(10);
        let state = make_state("t1", 1);
        let id = store.save(&state, "label");
        assert!(store.delete(id));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn delete_unknown_returns_false() {
        let mut store = SnapshotStore::new(10);
        assert!(!store.delete(SnapshotId(99)));
    }

    // ── List ──────────────────────────────────────────────────────────────────

    #[test]
    fn list_returns_entries_in_insertion_order() {
        let mut store = SnapshotStore::new(10);
        let s = make_state("t", 0);
        let id1 = store.save(&s, "alpha");
        let id2 = store.save(&s, "beta");
        let list = store.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].0, id1);
        assert_eq!(list[0].1, "alpha");
        assert_eq!(list[1].0, id2);
        assert_eq!(list[1].1, "beta");
    }

    // ── AgentStateSnapshot serialisation ─────────────────────────────────────

    #[test]
    fn agent_state_snapshot_roundtrip_bytes() {
        let state = make_state("my-task", 99);
        let bytes = state.to_bytes();
        let restored = AgentStateSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(restored, state);
    }

    #[test]
    fn agent_state_snapshot_empty_variables() {
        let state = AgentStateSnapshot::new("empty-task", 0, HashMap::new());
        let bytes = state.to_bytes();
        let restored = AgentStateSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(restored, state);
    }

    #[test]
    fn agent_state_snapshot_bad_bytes_returns_error() {
        let err = AgentStateSnapshot::from_bytes(b"not valid at all").unwrap_err();
        assert!(matches!(err, SnapshotError::DeserializationFailed(_)));
    }

    // ── CRC-32 correctness ────────────────────────────────────────────────────

    #[test]
    fn crc32_deterministic() {
        let a = crc32(b"hello world");
        let b = crc32(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn crc32_different_inputs_differ() {
        let a = crc32(b"hello");
        let b = crc32(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn crc32_empty_input() {
        // CRC-32 of empty slice should be 0x00000000 (standard result).
        let c = crc32(b"");
        assert_eq!(c, 0x0000_0000);
    }
}
