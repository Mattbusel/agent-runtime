//! # Agent Mesh — Peer-to-Peer Gossip Network
//!
//! Provides a lightweight in-process peer-to-peer mesh for multi-agent
//! coordination.  Agents join/leave, send directed or broadcast messages, and
//! propagate shared state via a gossip protocol with version-based merging.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

// ── Type alias ────────────────────────────────────────────────────────────────

/// Unique string identifier for an agent within the mesh.
pub type AgentId = String;

// ── MessageType ───────────────────────────────────────────────────────────────

/// Classifies the purpose of a [`MeshMessage`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    /// Gossip state propagation payload.
    Gossip,
    /// Addressed to a specific peer.
    DirectMessage,
    /// Addressed to all connected peers.
    Broadcast,
    /// Periodic liveness signal.
    HeartBeat,
    /// Signals that the sender is leaving the mesh.
    Leave,
}

// ── MeshMessage ───────────────────────────────────────────────────────────────

/// A message travelling through the agent mesh.
#[derive(Debug, Clone)]
pub struct MeshMessage {
    /// Sender agent id.
    pub from: AgentId,
    /// Recipient agent id, or `None` for broadcast.
    pub to: Option<AgentId>,
    /// Message classification.
    pub msg_type: MessageType,
    /// Opaque payload bytes.
    pub payload: Vec<u8>,
    /// Unix-epoch millisecond timestamp when the message was created.
    pub timestamp: u64,
    /// Remaining hop count; the message is dropped when this reaches zero.
    pub ttl: u8,
}

// ── PeerInfo ──────────────────────────────────────────────────────────────────

/// Known information about a remote agent.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Peer's unique identifier.
    pub agent_id: AgentId,
    /// Unix-epoch milliseconds of the last observed activity.
    pub last_seen: u64,
    /// Arbitrary metadata supplied at join time.
    pub metadata: HashMap<String, String>,
    /// Whether the peer is considered reachable.
    pub reachable: bool,
}

// ── GossipState ───────────────────────────────────────────────────────────────

/// A single versioned gossip key-value entry.
#[derive(Debug, Clone)]
pub struct GossipState {
    /// Gossip key.
    pub key: String,
    /// Gossip value.
    pub value: Vec<u8>,
    /// Monotonically increasing version; higher wins during merge.
    pub version: u64,
    /// Unix-epoch milliseconds when this entry was last updated locally.
    pub updated_at: u64,
}

// ── MeshStats ─────────────────────────────────────────────────────────────────

/// Snapshot of mesh statistics.
#[derive(Debug, Clone)]
pub struct MeshStats {
    /// Total peers known to this node (including unreachable ones).
    pub total_peers: usize,
    /// Number of peers currently marked reachable.
    pub reachable_peers: usize,
    /// Number of messages waiting in the local inbox.
    pub messages_in_inbox: usize,
    /// Number of gossip keys stored locally.
    pub gossip_keys: usize,
    /// This node's own agent id.
    pub local_id: AgentId,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns a coarse monotonic timestamp in milliseconds.
fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// FNV-1a 64-bit hash of a byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

// ── AgentMesh ─────────────────────────────────────────────────────────────────

/// Peer-to-peer agent mesh with gossip-based state propagation.
///
/// All operations are safe to call from multiple threads.
pub struct AgentMesh {
    local_id: AgentId,
    peers: Arc<Mutex<HashMap<AgentId, PeerInfo>>>,
    gossip: Arc<Mutex<HashMap<String, GossipState>>>,
    inbox: Arc<Mutex<VecDeque<MeshMessage>>>,
}

impl AgentMesh {
    /// Create a new mesh node with the given local identity.
    pub fn new(local_id: AgentId) -> Self {
        Self {
            local_id,
            peers: Arc::new(Mutex::new(HashMap::new())),
            gossip: Arc::new(Mutex::new(HashMap::new())),
            inbox: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Register a peer as having joined the mesh.
    pub fn join(&self, agent_id: AgentId, metadata: HashMap<String, String>) {
        let mut peers = self.peers.lock().unwrap_or_else(|e| e.into_inner());
        peers.insert(
            agent_id.clone(),
            PeerInfo {
                agent_id,
                last_seen: now_ms(),
                metadata,
                reachable: true,
            },
        );
    }

    /// Mark a peer as having left the mesh (sets reachable = false).
    pub fn leave(&self, agent_id: &AgentId) {
        let mut peers = self.peers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(peer) = peers.get_mut(agent_id) {
            peer.reachable = false;
        }
    }

    /// Route a message: if TTL > 0 and the message targets this node (or is a
    /// broadcast), place it in the inbox; otherwise drop it.
    pub fn send(&self, mut msg: MeshMessage) {
        if msg.ttl == 0 {
            return;
        }
        msg.ttl -= 1;

        let targeted_here = msg.to.as_deref() == Some(self.local_id.as_str())
            || msg.to.is_none()
            || msg.msg_type == MessageType::Broadcast;

        if targeted_here {
            let mut inbox = self.inbox.lock().unwrap_or_else(|e| e.into_inner());
            if inbox.len() < 1000 {
                inbox.push_back(msg);
            }
        }
    }

    /// Create and deliver a broadcast message with the given payload.
    pub fn broadcast(&self, payload: Vec<u8>) {
        let msg = MeshMessage {
            from: self.local_id.clone(),
            to: None,
            msg_type: MessageType::Broadcast,
            payload,
            timestamp: now_ms(),
            ttl: 8,
        };
        self.send(msg);
    }

    /// Pop the oldest message from the inbox, if any.
    pub fn receive(&self) -> Option<MeshMessage> {
        let mut inbox = self.inbox.lock().unwrap_or_else(|e| e.into_inner());
        inbox.pop_front()
    }

    /// Write a gossip key-value pair, incrementing the version counter.
    pub fn gossip_set(&self, key: &str, value: Vec<u8>) {
        let mut gossip = self.gossip.lock().unwrap_or_else(|e| e.into_inner());
        let entry = gossip.entry(key.to_string()).or_insert(GossipState {
            key: key.to_string(),
            value: Vec::new(),
            version: 0,
            updated_at: 0,
        });
        entry.version += 1;
        entry.value = value;
        entry.updated_at = now_ms();
    }

    /// Read a gossip value by key.
    pub fn gossip_get(&self, key: &str) -> Option<Vec<u8>> {
        let gossip = self.gossip.lock().unwrap_or_else(|e| e.into_inner());
        gossip.get(key).map(|g| g.value.clone())
    }

    /// Merge a remote gossip map into the local one, keeping the highest version.
    pub fn gossip_merge(&self, remote: HashMap<String, GossipState>) {
        let mut gossip = self.gossip.lock().unwrap_or_else(|e| e.into_inner());
        for (key, remote_entry) in remote {
            let local_version = gossip.get(&key).map(|e| e.version).unwrap_or(0);
            if remote_entry.version > local_version {
                gossip.insert(key, remote_entry);
            }
        }
    }

    /// Send the full gossip state to up to `fanout` randomly-selected reachable peers.
    ///
    /// Selection uses FNV-1a hashing of (local_id + peer_id + now_ms) as a
    /// deterministic-enough shuffler so that no `rand` dependency is required.
    pub fn fanout_gossip(&self, fanout: usize) {
        let reachable = self.reachable_peers();
        if reachable.is_empty() {
            return;
        }

        let gossip_snapshot: HashMap<String, GossipState> = {
            let gossip = self.gossip.lock().unwrap_or_else(|e| e.into_inner());
            gossip.clone()
        };

        // Encode the gossip snapshot as a simple byte payload.
        let payload: Vec<u8> = gossip_snapshot
            .iter()
            .flat_map(|(k, v)| {
                let mut entry = k.as_bytes().to_vec();
                entry.push(b':');
                entry.extend_from_slice(&v.value);
                entry.push(b'\n');
                entry
            })
            .collect();

        // Pseudo-random selection: sort peers by their FNV-1a hash against a seed.
        let seed = now_ms();
        let mut ranked: Vec<&AgentId> = reachable.iter().collect();
        ranked.sort_by_key(|id| {
            let mut data = id.as_bytes().to_vec();
            data.extend_from_slice(&seed.to_le_bytes());
            fnv1a(&data)
        });

        for peer_id in ranked.into_iter().take(fanout) {
            let msg = MeshMessage {
                from: self.local_id.clone(),
                to: Some(peer_id.clone()),
                msg_type: MessageType::Gossip,
                payload: payload.clone(),
                timestamp: now_ms(),
                ttl: 4,
            };
            self.send(msg);
        }
    }

    /// Snapshot all known peers.
    pub fn peer_list(&self) -> Vec<PeerInfo> {
        let peers = self.peers.lock().unwrap_or_else(|e| e.into_inner());
        peers.values().cloned().collect()
    }

    /// Return the ids of all currently reachable peers.
    pub fn reachable_peers(&self) -> Vec<AgentId> {
        let peers = self.peers.lock().unwrap_or_else(|e| e.into_inner());
        peers
            .values()
            .filter(|p| p.reachable)
            .map(|p| p.agent_id.clone())
            .collect()
    }

    /// Collect a stats snapshot without blocking the mesh.
    pub fn mesh_stats(&self) -> MeshStats {
        let peers = self.peers.lock().unwrap_or_else(|e| e.into_inner());
        let inbox = self.inbox.lock().unwrap_or_else(|e| e.into_inner());
        let gossip = self.gossip.lock().unwrap_or_else(|e| e.into_inner());

        let total = peers.len();
        let reachable = peers.values().filter(|p| p.reachable).count();

        MeshStats {
            total_peers: total,
            reachable_peers: reachable,
            messages_in_inbox: inbox.len(),
            gossip_keys: gossip.len(),
            local_id: self.local_id.clone(),
        }
    }
}
