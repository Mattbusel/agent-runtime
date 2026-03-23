//! # Inter-Agent Messaging Protocol
//!
//! Provides typed, async message passing between agents running in the same
//! process or connected over a network. The bus is in-process using Tokio
//! broadcast channels. Network extension uses JSON serialization.
//!
//! Message types cover the full lifecycle of agent collaboration:
//! task delegation, capability queries, progress updates, and result sharing.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::messaging::{AgentId, AgentMessage, MessageBus, MessageEnvelope};
//! use std::collections::HashMap;
//!
//! # tokio_test::block_on(async {
//! let bus = MessageBus::new(64);
//!
//! let sender = AgentId::new("agent-a");
//! let receiver = AgentId::new("agent-b");
//!
//! let mut mailbox = bus.subscribe_for(receiver.clone());
//!
//! let msg = AgentMessage::Heartbeat { load_pct: 10, queue_depth: 3 };
//! let envelope = MessageEnvelope::broadcast(sender.clone(), msg);
//! bus.publish(envelope).unwrap();
//!
//! let received = mailbox.recv().await.unwrap();
//! assert!(received.is_for(&receiver));
//! # });
//! ```

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{broadcast, RwLock};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── AgentId ───────────────────────────────────────────────────────────────────

/// Unique agent identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    /// Create an [`AgentId`] from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Create a randomly generated [`AgentId`] backed by a UUID v4.
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── AgentMessage ──────────────────────────────────────────────────────────────

/// All message types that agents can exchange.
///
/// Every variant is serializable so envelopes can be forwarded over the
/// network as JSON if needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentMessage {
    /// Delegate a task to another agent.
    TaskDelegate {
        /// Unique identifier for this task.
        task_id: String,
        /// Human-readable description of what must be done.
        task_description: String,
        /// Priority level: 0=low, 1=normal, 2=high, 3=critical.
        priority: u8,
        /// Optional wall-clock deadline expressed as seconds from now.
        deadline_secs: Option<u64>,
        /// Arbitrary key/value context to pass to the assignee.
        context: HashMap<String, serde_json::Value>,
    },
    /// Accept a delegated task.
    TaskAccepted {
        /// The task being accepted.
        task_id: String,
        /// Optional estimated duration in seconds.
        estimated_duration_secs: Option<u64>,
    },
    /// Reject a delegated task with a reason.
    TaskRejected {
        /// The task being rejected.
        task_id: String,
        /// Human-readable reason for rejection.
        reason: String,
    },
    /// Report progress on an in-flight task.
    TaskProgress {
        /// The in-flight task.
        task_id: String,
        /// Completion percentage, 0–100.
        progress_pct: u8,
        /// Short status message for humans or monitors.
        status_message: String,
    },
    /// Report task completion.
    TaskCompleted {
        /// The completed task.
        task_id: String,
        /// Structured result produced by the task.
        result: serde_json::Value,
        /// Wall-clock time spent executing the task, in milliseconds.
        duration_ms: u64,
    },
    /// Report task failure.
    TaskFailed {
        /// The failed task.
        task_id: String,
        /// Human-readable error description.
        error: String,
        /// Whether the caller may safely retry this task.
        retryable: bool,
    },
    /// Query another agent's capabilities.
    CapabilityQuery {
        /// Unique identifier for this query (used to correlate the response).
        query_id: String,
        /// Tags that the responding agent must possess to be considered capable.
        required_tags: Vec<String>,
    },
    /// Response to a capability query.
    CapabilityResponse {
        /// Mirrors the `query_id` from the originating [`AgentMessage::CapabilityQuery`].
        query_id: String,
        /// Capabilities advertised by the responding agent.
        capabilities: Vec<String>,
        /// Whether the agent is currently available to take on new work.
        available: bool,
    },
    /// Share a fact or piece of knowledge.
    KnowledgeShare {
        /// Namespaced key for the fact.
        key: String,
        /// Structured value.
        value: serde_json::Value,
        /// Optional time-to-live in seconds after which the fact expires.
        ttl_secs: Option<u64>,
    },
    /// Heartbeat / keepalive message.
    Heartbeat {
        /// Current CPU / work load as a percentage (0–100).
        load_pct: u8,
        /// Number of tasks waiting in the agent's local queue.
        queue_depth: usize,
    },
    /// Request to collaborate on a multi-step plan.
    CollaborationRequest {
        /// Unique identifier for this plan.
        plan_id: String,
        /// Ordered list of steps that together make up the plan.
        steps: Vec<String>,
        /// ID of the agent that is requesting collaboration.
        requesting_agent: String,
    },
}

// ── MessageEnvelope ──────────────────────────────────────────────────────────

/// An envelope wrapping a message with routing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Globally unique envelope ID.
    pub id: String,
    /// The agent that sent this message.
    pub from: AgentId,
    /// The intended recipient, or `None` for a broadcast to all agents.
    pub to: Option<AgentId>,
    /// The payload.
    pub message: AgentMessage,
    /// Unix timestamp in milliseconds when the envelope was created.
    pub timestamp: u64,
    /// Optional correlation ID for request/response pairing.
    pub correlation_id: Option<String>,
}

impl MessageEnvelope {
    /// Create an envelope addressed to a specific agent (or broadcast when
    /// `to` is `None`).
    pub fn new(from: AgentId, to: Option<AgentId>, message: AgentMessage) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from,
            to,
            message,
            timestamp: unix_ms(),
            correlation_id: None,
        }
    }

    /// Create an envelope that is broadcast to all agents (no specific `to`).
    pub fn broadcast(from: AgentId, message: AgentMessage) -> Self {
        Self::new(from, None, message)
    }

    /// Return `true` if this envelope is addressed to `agent_id` or is a
    /// broadcast (i.e. `to` is `None`).
    pub fn is_for(&self, agent_id: &AgentId) -> bool {
        match &self.to {
            None => true,
            Some(recipient) => recipient == agent_id,
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

// ── BusStats ──────────────────────────────────────────────────────────────────

/// Runtime statistics for a [`MessageBus`].
#[derive(Debug, Default, Clone)]
pub struct BusStats {
    /// Total envelopes successfully published.
    pub messages_sent: u64,
    /// Total envelopes dropped because all receiver buffers were full.
    pub messages_dropped: u64,
    /// Current number of active subscribers.
    pub active_subscribers: usize,
}

// ── MessageBus ────────────────────────────────────────────────────────────────

/// In-process message bus backed by a Tokio broadcast channel.
///
/// Clone to share across tasks. All subscribers receive all messages;
/// each subscriber is responsible for filtering envelopes addressed to it.
///
/// # Capacity
///
/// `capacity` controls the Tokio broadcast channel buffer size. Lagged
/// receivers drop messages once the buffer is full.
#[derive(Clone)]
pub struct MessageBus {
    sender: broadcast::Sender<MessageEnvelope>,
    capacity: usize,
    stats: Arc<RwLock<BusStats>>,
}

impl MessageBus {
    /// Create a new [`MessageBus`] with the given channel buffer `capacity`.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            capacity,
            stats: Arc::new(RwLock::new(BusStats::default())),
        }
    }

    /// Publish an envelope to all subscribers.
    ///
    /// Returns the number of active receivers that received the message, or
    /// an error string if publication failed entirely.
    pub fn publish(&self, envelope: MessageEnvelope) -> Result<usize, String> {
        match self.sender.send(envelope) {
            Ok(n) => {
                // Fire-and-forget stats update; don't block the caller.
                let stats = Arc::clone(&self.stats);
                tokio::spawn(async move {
                    let mut s = stats.write().await;
                    s.messages_sent += 1;
                    s.active_subscribers = n;
                });
                Ok(n)
            }
            Err(_) => {
                // No receivers — increment dropped count.
                let stats = Arc::clone(&self.stats);
                tokio::spawn(async move {
                    let mut s = stats.write().await;
                    s.messages_dropped += 1;
                });
                // Return 0: the send technically succeeded from the channel's
                // perspective (no receivers is not a hard error in broadcast).
                Ok(0)
            }
        }
    }

    /// Subscribe to receive all messages on the bus.
    ///
    /// Messages for other agents must be filtered by the caller.
    pub fn subscribe(&self) -> broadcast::Receiver<MessageEnvelope> {
        self.sender.subscribe()
    }

    /// Subscribe and return an [`AgentMailbox`] that automatically filters
    /// messages to only those addressed to `agent_id` or broadcast messages.
    pub fn subscribe_for(&self, agent_id: AgentId) -> AgentMailbox {
        AgentMailbox {
            agent_id,
            receiver: self.sender.subscribe(),
        }
    }

    /// Return a snapshot of current bus statistics.
    pub async fn stats(&self) -> BusStats {
        self.stats.read().await.clone()
    }

    /// Channel capacity this bus was created with.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── AgentMailbox ──────────────────────────────────────────────────────────────

/// A filtered view of the [`MessageBus`] for a specific agent.
///
/// Only envelopes addressed to this agent or broadcast envelopes are surfaced.
pub struct AgentMailbox {
    agent_id: AgentId,
    receiver: broadcast::Receiver<MessageEnvelope>,
}

impl AgentMailbox {
    /// Receive the next message addressed to this agent (or broadcast).
    ///
    /// Skips envelopes intended for other agents transparently.
    /// Returns `None` when the bus is permanently closed.
    pub async fn recv(&mut self) -> Option<MessageEnvelope> {
        loop {
            match self.receiver.recv().await {
                Ok(env) if env.is_for(&self.agent_id) => return Some(env),
                Ok(_) => continue, // addressed to someone else — skip
                Err(broadcast::error::RecvError::Lagged(_)) => continue, // skipped some; keep going
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    }

    /// Non-blocking attempt to receive the next message for this agent.
    ///
    /// Returns `None` immediately if no relevant message is buffered.
    pub fn try_recv(&mut self) -> Option<MessageEnvelope> {
        loop {
            match self.receiver.try_recv() {
                Ok(env) if env.is_for(&self.agent_id) => return Some(env),
                Ok(_) => continue,
                Err(_) => return None,
            }
        }
    }

    /// Return the agent ID this mailbox is filtered for.
    pub fn agent_id(&self) -> &AgentId {
        &self.agent_id
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── AgentId ──────────────────────────────────────────────────────────────

    #[test]
    fn agent_id_new_and_as_str() {
        let id = AgentId::new("worker-1");
        assert_eq!(id.as_str(), "worker-1");
    }

    #[test]
    fn agent_id_display() {
        let id = AgentId::new("planner");
        assert_eq!(id.to_string(), "planner");
    }

    #[test]
    fn agent_id_random_is_unique() {
        let a = AgentId::random();
        let b = AgentId::random();
        assert_ne!(a, b);
    }

    #[test]
    fn agent_id_equality_and_hash() {
        let a = AgentId::new("x");
        let b = AgentId::new("x");
        let c = AgentId::new("y");
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut map = HashMap::new();
        map.insert(a.clone(), 1u32);
        assert_eq!(*map.get(&b).unwrap(), 1);
    }

    // ── MessageEnvelope ───────────────────────────────────────────────────────

    #[test]
    fn envelope_is_for_specific_recipient() {
        let from = AgentId::new("a");
        let to = AgentId::new("b");
        let other = AgentId::new("c");
        let env = MessageEnvelope::new(
            from,
            Some(to.clone()),
            AgentMessage::Heartbeat { load_pct: 0, queue_depth: 0 },
        );
        assert!(env.is_for(&to));
        assert!(!env.is_for(&other));
    }

    #[test]
    fn broadcast_envelope_is_for_everyone() {
        let from = AgentId::new("a");
        let env = MessageEnvelope::broadcast(
            from,
            AgentMessage::Heartbeat { load_pct: 5, queue_depth: 1 },
        );
        assert!(env.is_for(&AgentId::new("anyone")));
        assert!(env.is_for(&AgentId::new("also-anyone")));
        assert!(env.to.is_none());
    }

    #[test]
    fn envelope_has_unique_ids() {
        let from = AgentId::new("a");
        let e1 = MessageEnvelope::broadcast(
            from.clone(),
            AgentMessage::Heartbeat { load_pct: 0, queue_depth: 0 },
        );
        let e2 = MessageEnvelope::broadcast(
            from,
            AgentMessage::Heartbeat { load_pct: 0, queue_depth: 0 },
        );
        assert_ne!(e1.id, e2.id);
    }

    // ── MessageBus / AgentMailbox ────────────────────────────────────────────

    #[tokio::test]
    async fn mailbox_receives_broadcast() {
        let bus = MessageBus::new(16);
        let agent = AgentId::new("agent-1");
        let mut mailbox = bus.subscribe_for(agent.clone());

        let env = MessageEnvelope::broadcast(
            AgentId::new("sender"),
            AgentMessage::Heartbeat { load_pct: 20, queue_depth: 0 },
        );
        bus.publish(env).unwrap();

        let received = mailbox.recv().await.unwrap();
        assert!(received.is_for(&agent));
    }

    #[tokio::test]
    async fn mailbox_receives_targeted_message() {
        let bus = MessageBus::new(16);
        let target = AgentId::new("target");
        let other = AgentId::new("other");

        let mut target_mailbox = bus.subscribe_for(target.clone());
        let mut other_mailbox = bus.subscribe_for(other.clone());

        // Send specifically to `target`.
        let env = MessageEnvelope::new(
            AgentId::new("sender"),
            Some(target.clone()),
            AgentMessage::Heartbeat { load_pct: 0, queue_depth: 0 },
        );
        bus.publish(env).unwrap();

        // `target` gets it.
        let msg = target_mailbox.recv().await.unwrap();
        assert_eq!(msg.to.as_ref().unwrap(), &target);

        // `other` should NOT receive a targeted message meant for `target`.
        assert!(other_mailbox.try_recv().is_none());
    }

    #[tokio::test]
    async fn publish_to_no_subscribers_returns_zero() {
        let bus = MessageBus::new(8);
        let env = MessageEnvelope::broadcast(
            AgentId::new("a"),
            AgentMessage::Heartbeat { load_pct: 0, queue_depth: 0 },
        );
        let count = bus.publish(env).unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn task_delegate_message_round_trips_through_bus() {
        let bus = MessageBus::new(32);
        let worker = AgentId::new("worker");
        let mut mailbox = bus.subscribe_for(worker.clone());

        let msg = AgentMessage::TaskDelegate {
            task_id: "t-001".into(),
            task_description: "Summarise document".into(),
            priority: 2,
            deadline_secs: Some(300),
            context: HashMap::new(),
        };
        let env = MessageEnvelope::new(AgentId::new("planner"), Some(worker.clone()), msg);
        bus.publish(env).unwrap();

        let received = mailbox.recv().await.unwrap();
        match received.message {
            AgentMessage::TaskDelegate { task_id, priority, .. } => {
                assert_eq!(task_id, "t-001");
                assert_eq!(priority, 2);
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[tokio::test]
    async fn capability_query_response_cycle() {
        let bus = MessageBus::new(32);
        let responder = AgentId::new("responder");
        let querier = AgentId::new("querier");
        let mut responder_mb = bus.subscribe_for(responder.clone());
        let mut querier_mb = bus.subscribe_for(querier.clone());

        // Querier sends a capability query.
        let query_env = MessageEnvelope::new(
            querier.clone(),
            Some(responder.clone()),
            AgentMessage::CapabilityQuery {
                query_id: "q-1".into(),
                required_tags: vec!["nlp".into()],
            },
        );
        bus.publish(query_env).unwrap();

        // Responder receives and replies.
        let q = responder_mb.recv().await.unwrap();
        let query_id = match &q.message {
            AgentMessage::CapabilityQuery { query_id, .. } => query_id.clone(),
            _ => panic!("expected CapabilityQuery"),
        };

        let reply_env = MessageEnvelope::new(
            responder.clone(),
            Some(querier.clone()),
            AgentMessage::CapabilityResponse {
                query_id,
                capabilities: vec!["nlp".into(), "summarise".into()],
                available: true,
            },
        );
        bus.publish(reply_env).unwrap();

        // Querier receives the response.
        let resp = querier_mb.recv().await.unwrap();
        match resp.message {
            AgentMessage::CapabilityResponse { available, capabilities, .. } => {
                assert!(available);
                assert!(capabilities.contains(&"nlp".to_string()));
            }
            _ => panic!("expected CapabilityResponse"),
        }
    }

    #[tokio::test]
    async fn knowledge_share_serializes_correctly() {
        let msg = AgentMessage::KnowledgeShare {
            key: "company.name".into(),
            value: serde_json::json!("Acme Corp"),
            ttl_secs: Some(3600),
        };
        let serialized = serde_json::to_string(&msg).unwrap();
        let de: AgentMessage = serde_json::from_str(&serialized).unwrap();
        match de {
            AgentMessage::KnowledgeShare { key, ttl_secs, .. } => {
                assert_eq!(key, "company.name");
                assert_eq!(ttl_secs, Some(3600));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn multiple_subscribers_all_receive_broadcast() {
        let bus = MessageBus::new(32);
        let mut mb_a = bus.subscribe_for(AgentId::new("a"));
        let mut mb_b = bus.subscribe_for(AgentId::new("b"));
        let mut mb_c = bus.subscribe_for(AgentId::new("c"));

        let env = MessageEnvelope::broadcast(
            AgentId::new("root"),
            AgentMessage::Heartbeat { load_pct: 50, queue_depth: 2 },
        );
        bus.publish(env).unwrap();

        assert!(mb_a.recv().await.is_some());
        assert!(mb_b.recv().await.is_some());
        assert!(mb_c.recv().await.is_some());
    }

    #[test]
    fn all_message_variants_are_serializable() {
        let variants: Vec<AgentMessage> = vec![
            AgentMessage::TaskDelegate {
                task_id: "t1".into(),
                task_description: "do something".into(),
                priority: 1,
                deadline_secs: None,
                context: HashMap::new(),
            },
            AgentMessage::TaskAccepted { task_id: "t1".into(), estimated_duration_secs: Some(60) },
            AgentMessage::TaskRejected { task_id: "t1".into(), reason: "busy".into() },
            AgentMessage::TaskProgress {
                task_id: "t1".into(),
                progress_pct: 50,
                status_message: "halfway".into(),
            },
            AgentMessage::TaskCompleted {
                task_id: "t1".into(),
                result: serde_json::json!({"ok": true}),
                duration_ms: 1500,
            },
            AgentMessage::TaskFailed {
                task_id: "t1".into(),
                error: "timeout".into(),
                retryable: true,
            },
            AgentMessage::CapabilityQuery {
                query_id: "q1".into(),
                required_tags: vec!["nlp".into()],
            },
            AgentMessage::CapabilityResponse {
                query_id: "q1".into(),
                capabilities: vec!["nlp".into()],
                available: false,
            },
            AgentMessage::KnowledgeShare {
                key: "k".into(),
                value: serde_json::json!(42),
                ttl_secs: None,
            },
            AgentMessage::Heartbeat { load_pct: 80, queue_depth: 5 },
            AgentMessage::CollaborationRequest {
                plan_id: "p1".into(),
                steps: vec!["a".into(), "b".into()],
                requesting_agent: "coord".into(),
            },
        ];

        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialization failed");
            let _: AgentMessage = serde_json::from_str(&json).expect("deserialization failed");
        }
    }
}
