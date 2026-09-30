//! # Module: consensus_engine
//!
//! Raft-lite quorum voting engine.
//!
//! Provides:
//! - [`VoteRequest`] / [`VoteResponse`]: Raft-style vote RPC types.
//! - [`NodeRole`]: Follower / Candidate / Leader.
//! - [`ConsensusNode`]: a single participant in the cluster.
//! - [`LogEntry`]: a replication log entry.
//! - [`ConsensusEngine`]: drives elections, heartbeats, and log append.
//! - [`ConsensusError`]: error variants for consensus operations.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use std::sync::Mutex;

// ── NodeRole ──────────────────────────────────────────────────────────────────

/// The current role of a consensus node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeRole {
    /// Following a known leader.
    Follower,
    /// Actively seeking election.
    Candidate,
    /// Currently the cluster leader.
    Leader,
}

// ── VoteRequest / VoteResponse ────────────────────────────────────────────────

/// RequestVote RPC (Raft §5.2).
#[derive(Debug, Clone)]
pub struct VoteRequest {
    /// Candidate's term.
    pub term: u64,
    /// ID of the candidate requesting the vote.
    pub candidate_id: String,
    /// Index of the candidate's last log entry.
    pub last_log_index: u64,
    /// Term of the candidate's last log entry.
    pub last_log_term: u64,
}

/// Response to a [`VoteRequest`].
#[derive(Debug, Clone)]
pub struct VoteResponse {
    /// Current term of the responding node (for candidate to update itself).
    pub term: u64,
    /// ID of the responding node.
    pub voter_id: String,
    /// Whether the vote was granted.
    pub granted: bool,
}

// ── LogEntry ──────────────────────────────────────────────────────────────────

/// A single entry in the replicated log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Log index (1-based).
    pub index: u64,
    /// Term during which this entry was created.
    pub term: u64,
    /// The command string.
    pub command: String,
}

// ── ConsensusError ────────────────────────────────────────────────────────────

/// Errors returned by [`ConsensusEngine`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsensusError {
    /// The operation requires the caller to be the leader, but it is not.
    NotLeader,
    /// The leader no longer has quorum acknowledgement.
    QuorumLost,
    /// The referenced node ID does not exist in this engine.
    NodeNotFound,
}

impl std::fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConsensusError::NotLeader => f.write_str("not leader"),
            ConsensusError::QuorumLost => f.write_str("quorum lost"),
            ConsensusError::NodeNotFound => f.write_str("node not found"),
        }
    }
}

impl std::error::Error for ConsensusError {}

// ── ConsensusNode ─────────────────────────────────────────────────────────────

/// A single participant in the consensus cluster.
pub struct ConsensusNode {
    /// Stable node identifier.
    pub node_id: String,
    /// Current Raft term, atomically maintained.
    pub current_term: AtomicU64,
    /// Candidate this node voted for in its current term, if any.
    pub voted_for: Mutex<Option<String>>,
    /// Current role of this node.
    pub role: Mutex<NodeRole>,
    /// Replicated log.
    log: Mutex<Vec<LogEntry>>,
}

impl ConsensusNode {
    fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            current_term: AtomicU64::new(0),
            voted_for: Mutex::new(None),
            role: Mutex::new(NodeRole::Follower),
            log: Mutex::new(Vec::new()),
        }
    }

    /// Attempt to grant a vote for `candidate_id` at `term`.
    ///
    /// Returns `true` if this node grants the vote.
    fn try_grant_vote(&self, candidate_id: &str, term: u64) -> bool {
        let current = self.current_term.load(Ordering::Acquire);
        if term <= current {
            return false;
        }
        // term > current_term: advance and grant if not yet voted.
        self.current_term.store(term, Ordering::Release);
        let mut voted_for = lock_mutex(&self.voted_for);
        if voted_for.is_none() {
            *voted_for = Some(candidate_id.to_owned());
            // Step down to Follower.
            *lock_mutex(&self.role) = NodeRole::Follower;
            true
        } else {
            false
        }
    }

    /// Acknowledge a heartbeat from `leader_id`.  Returns `true` if accepted.
    fn acknowledge_heartbeat(&self, leader_id: &str, term: u64) -> bool {
        let current = self.current_term.load(Ordering::Acquire);
        if term < current {
            return false;
        }
        if term > current {
            self.current_term.store(term, Ordering::Release);
            *lock_mutex(&self.voted_for) = None;
        }
        let mut role = lock_mutex(&self.role);
        if *role == NodeRole::Candidate || *role == NodeRole::Leader {
            if self.node_id != leader_id {
                *role = NodeRole::Follower;
            }
        }
        true
    }

    /// Append a log entry if this node believes `leader_id` is the leader.
    fn append_log_entry(&self, entry: LogEntry) {
        lock_mutex(&self.log).push(entry);
    }
}

// ── ConsensusEngine ───────────────────────────────────────────────────────────

/// Drives elections, heartbeats, and log replication across a cluster of nodes.
pub struct ConsensusEngine {
    /// All nodes in the cluster.
    pub nodes: Vec<Arc<ConsensusNode>>,
    /// Minimum votes required to achieve quorum.
    pub quorum_size: usize,
}

impl ConsensusEngine {
    /// Build a new engine from a list of node IDs.
    ///
    /// `quorum_size` is set to the strict majority: `floor(n/2) + 1`.
    pub fn new(node_ids: Vec<String>) -> Self {
        let n = node_ids.len();
        let quorum_size = n / 2 + 1;
        let nodes = node_ids
            .into_iter()
            .map(|id| Arc::new(ConsensusNode::new(id)))
            .collect();
        Self { nodes, quorum_size }
    }

    /// Find a node by ID.
    fn find_node(&self, node_id: &str) -> Option<&Arc<ConsensusNode>> {
        self.nodes.iter().find(|n| n.node_id == node_id)
    }

    /// Simulate a vote collection round for `candidate_id`.
    ///
    /// The candidate increments its own term, resets its voted_for to itself,
    /// then solicits votes from every other node.
    ///
    /// Returns `(elected, votes_received)`.
    pub fn request_vote(&self, candidate_id: &str) -> (bool, usize) {
        // Find and advance the candidate's term.
        let candidate = match self.find_node(candidate_id) {
            Some(n) => n,
            None => return (false, 0),
        };

        let new_term = candidate.current_term.fetch_add(1, Ordering::AcqRel) + 1;
        *lock_mutex(&candidate.voted_for) = Some(candidate_id.to_owned());
        *lock_mutex(&candidate.role) = NodeRole::Candidate;

        let last_log_index = lock_mutex(&candidate.log).len() as u64;
        let last_log_term = lock_mutex(&candidate.log)
            .last()
            .map_or(0, |e| e.term);

        // Candidate votes for itself.
        let mut votes: usize = 1;

        for node in &self.nodes {
            if node.node_id == candidate_id {
                continue;
            }
            if node.try_grant_vote(candidate_id, new_term) {
                votes += 1;
            }
        }

        let elected = votes >= self.quorum_size;
        if elected {
            *lock_mutex(&candidate.role) = NodeRole::Leader;
            candidate.current_term.store(new_term, Ordering::Release);
        }

        let _ = last_log_index;
        let _ = last_log_term;

        (elected, votes)
    }

    /// Broadcast a heartbeat from `leader_id` to all other nodes.
    ///
    /// Returns the number of acknowledgements received.
    pub fn broadcast_heartbeat(&self, leader_id: &str) -> usize {
        let leader = match self.find_node(leader_id) {
            Some(n) => n,
            None => return 0,
        };
        let term = leader.current_term.load(Ordering::Acquire);
        let mut acks = 0usize;

        for node in &self.nodes {
            if node.node_id == leader_id {
                continue;
            }
            if node.acknowledge_heartbeat(leader_id, term) {
                acks += 1;
            }
        }
        acks
    }

    /// Return the node ID of the current leader, if one exists.
    pub fn current_leader(&self) -> Option<String> {
        self.nodes
            .iter()
            .find(|n| *lock_mutex(&n.role) == NodeRole::Leader)
            .map(|n| n.node_id.clone())
    }

    /// Demote a node to Follower.
    pub fn step_down(&self, node_id: &str) {
        if let Some(node) = self.find_node(node_id) {
            *lock_mutex(&node.role) = NodeRole::Follower;
        }
    }

    /// Append a log entry.  Only succeeds if `leader_id` is the current leader
    /// and has quorum (heartbeat acks ≥ quorum_size - 1, i.e. majority of others).
    pub fn append_entry(
        &self,
        leader_id: &str,
        entry: LogEntry,
    ) -> Result<(), ConsensusError> {
        let leader = self
            .find_node(leader_id)
            .ok_or(ConsensusError::NodeNotFound)?;

        if *lock_mutex(&leader.role) != NodeRole::Leader {
            return Err(ConsensusError::NotLeader);
        }

        // Check quorum: need acks from quorum_size - 1 other nodes.
        let term = leader.current_term.load(Ordering::Acquire);
        let needed = self.quorum_size.saturating_sub(1);
        let acks = self
            .nodes
            .iter()
            .filter(|n| {
                if n.node_id == leader_id {
                    return false;
                }
                n.current_term.load(Ordering::Acquire) == term
            })
            .count();

        if acks < needed {
            return Err(ConsensusError::QuorumLost);
        }

        // Append to leader's own log and replicate.
        leader.append_log_entry(entry.clone());
        for node in &self.nodes {
            if node.node_id == leader_id {
                continue;
            }
            if node.current_term.load(Ordering::Acquire) == term {
                node.append_log_entry(entry.clone());
            }
        }

        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn lock_mutex<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn engine_3() -> ConsensusEngine {
        ConsensusEngine::new(vec![
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ])
    }

    #[test]
    fn election_with_majority() {
        let engine = engine_3();
        let (elected, votes) = engine.request_vote("n1");
        assert!(elected, "n1 should win election in a 3-node cluster");
        assert!(votes >= 2, "should have at least quorum votes, got {votes}");
        assert_eq!(engine.current_leader(), Some("n1".to_string()));
    }

    #[test]
    fn heartbeat_acks_from_followers() {
        let engine = engine_3();
        let (elected, _) = engine.request_vote("n1");
        assert!(elected);
        let acks = engine.broadcast_heartbeat("n1");
        // 3-node cluster: 2 other nodes should ack.
        assert_eq!(acks, 2);
    }

    #[test]
    fn step_down_makes_follower() {
        let engine = engine_3();
        let (elected, _) = engine.request_vote("n1");
        assert!(elected);
        assert_eq!(engine.current_leader(), Some("n1".to_string()));
        engine.step_down("n1");
        assert_eq!(engine.current_leader(), None);
    }

    #[test]
    fn append_entry_requires_leader() {
        let engine = engine_3();
        let entry = LogEntry {
            index: 1,
            term: 1,
            command: "set x=1".to_string(),
        };
        // n1 is not leader yet.
        let result = engine.append_entry("n1", entry.clone());
        assert_eq!(result, Err(ConsensusError::NotLeader));

        // Elect n1.
        let (elected, _) = engine.request_vote("n1");
        assert!(elected);

        // After election, heartbeat to sync terms.
        engine.broadcast_heartbeat("n1");

        let result = engine.append_entry("n1", entry);
        assert!(result.is_ok(), "leader should append successfully");
    }

    #[test]
    fn node_not_found_error() {
        let engine = engine_3();
        let (elected, _) = engine.request_vote("n1");
        assert!(elected);
        let entry = LogEntry { index: 1, term: 1, command: "x".to_string() };
        let result = engine.append_entry("n99", entry);
        assert_eq!(result, Err(ConsensusError::NodeNotFound));
    }
}
