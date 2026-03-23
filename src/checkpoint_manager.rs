//! # Module: checkpoint_manager
//!
//! Agent state checkpointing and resume: save/restore agent state with
//! checksum verification, bounded per-agent history, and step-level restore.

use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;
use std::time::Instant;

// ── CheckpointData ────────────────────────────────────────────────────────────

/// A single checkpoint of an agent's state at a specific step.
#[derive(Clone)]
pub struct CheckpointData {
    /// The agent this checkpoint belongs to.
    pub agent_id: String,
    /// Human-readable label for this state (e.g. "idle", "processing").
    pub state_label: String,
    /// Raw serialised payload bytes.
    pub payload: Vec<u8>,
    /// XOR checksum of all payload bytes.
    pub checksum: u32,
    /// Wall-clock time when the checkpoint was created.
    pub created_at: Instant,
    /// Logical step counter at the time of the checkpoint.
    pub step: u64,
}

/// Compute a simple XOR checksum over a byte slice.
fn xor_checksum(data: &[u8]) -> u32 {
    let mut acc: u32 = 0;
    for &b in data {
        acc ^= b as u32;
    }
    acc
}

impl CheckpointData {
    /// Create a new checkpoint, computing the checksum automatically.
    pub fn new(
        agent_id: impl Into<String>,
        state_label: impl Into<String>,
        payload: Vec<u8>,
        step: u64,
    ) -> Self {
        let checksum = xor_checksum(&payload);
        Self {
            agent_id: agent_id.into(),
            state_label: state_label.into(),
            checksum,
            created_at: Instant::now(),
            payload,
            step,
        }
    }

    /// Verify the stored checksum matches a freshly computed one.
    pub fn verify(&self) -> bool {
        xor_checksum(&self.payload) == self.checksum
    }
}

// ── CheckpointManager ─────────────────────────────────────────────────────────

/// Manages per-agent checkpoints with bounded history.
pub struct CheckpointManager {
    /// agent_id → bounded deque of checkpoints (oldest first).
    store: RwLock<HashMap<String, VecDeque<CheckpointData>>>,
    /// Maximum number of checkpoints to retain per agent.
    pub max_per_agent: usize,
}

impl CheckpointManager {
    /// Create a new manager with the given per-agent capacity.
    pub fn new(max_per_agent: usize) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            max_per_agent,
        }
    }

    /// Save a checkpoint, evicting the oldest if the agent is at capacity.
    pub fn save(&self, data: CheckpointData) {
        let mut store = self.store.write().unwrap();
        let deque = store
            .entry(data.agent_id.clone())
            .or_insert_with(VecDeque::new);
        deque.push_back(data);
        while deque.len() > self.max_per_agent {
            deque.pop_front();
        }
    }

    /// Return the most recent checkpoint for an agent, or `None`.
    pub fn latest(&self, agent_id: &str) -> Option<CheckpointData> {
        let store = self.store.read().unwrap();
        store.get(agent_id)?.back().cloned()
    }

    /// Return all checkpoints for an agent, newest first.
    pub fn history(&self, agent_id: &str) -> Vec<CheckpointData> {
        let store = self.store.read().unwrap();
        match store.get(agent_id) {
            None => vec![],
            Some(deque) => {
                let mut v: Vec<CheckpointData> = deque.iter().cloned().collect();
                v.reverse();
                v
            }
        }
    }

    /// Find the most recent checkpoint at or before a given step.
    pub fn restore_at_step(&self, agent_id: &str, step: u64) -> Option<CheckpointData> {
        let store = self.store.read().unwrap();
        store.get(agent_id)?.iter().rev().find(|c| c.step <= step).cloned()
    }

    /// Remove all checkpoints for an agent.
    pub fn purge_agent(&self, agent_id: &str) {
        let mut store = self.store.write().unwrap();
        store.remove(agent_id);
    }

    /// Return `(agent_id, checkpoint_count)` pairs for all agents.
    pub fn stats(&self) -> Vec<(String, usize)> {
        let store = self.store.read().unwrap();
        store
            .iter()
            .map(|(id, deque)| (id.clone(), deque.len()))
            .collect()
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_roundtrip() {
        let payload = vec![1u8, 2, 3, 4, 5];
        let ck = CheckpointData::new("a1", "init", payload, 0);
        assert!(ck.verify());
    }

    #[test]
    fn manager_evicts_oldest() {
        let mgr = CheckpointManager::new(3);
        for i in 0u64..5 {
            mgr.save(CheckpointData::new("a1", "s", vec![i as u8], i));
        }
        assert_eq!(mgr.history("a1").len(), 3);
        assert_eq!(mgr.latest("a1").unwrap().step, 4);
    }

    #[test]
    fn restore_at_step() {
        let mgr = CheckpointManager::new(10);
        for i in 0u64..5 {
            mgr.save(CheckpointData::new("a1", "s", vec![i as u8], i * 2));
        }
        // Steps are 0, 2, 4, 6, 8 — restore_at_step(5) should return step 4.
        let ck = mgr.restore_at_step("a1", 5).unwrap();
        assert_eq!(ck.step, 4);
    }

    #[test]
    fn purge_agent() {
        let mgr = CheckpointManager::new(10);
        mgr.save(CheckpointData::new("a1", "s", vec![1], 0));
        mgr.purge_agent("a1");
        assert!(mgr.latest("a1").is_none());
    }
}
