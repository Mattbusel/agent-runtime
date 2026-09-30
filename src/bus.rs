//! # Module: Agent Message Bus
//!
//! ## Responsibility
//! Provides an async broadcast message bus so that agents running inside the
//! same [`AgentRuntime`] can communicate with each other without sharing
//! mutable state directly.
//!
//! ## Design
//! - [`AgentBus`] holds a `tokio::sync::broadcast` channel.  Every subscriber
//!   receives a clone of every message that passes through the bus.
//! - [`BusSubscription`] is the handle returned by [`AgentBus::subscribe`].
//!   Call [`BusSubscription::recv`] in a loop to consume messages addressed to
//!   your agent or broadcast to all agents.
//! - [`AgentMessage`] carries the sender identity, an [`AgentTarget`]
//!   (specific agent, broadcast, or role-based), the message text, and an
//!   optional reply-to [`MessageId`] for threaded conversations.
//! - [`AgentRole`] allows agents to register a named role (e.g. `"planner"`,
//!   `"executor"`, `"critic"`).  Role-targeted messages are delivered to every
//!   subscriber whose registered role matches.
//!
//! ## Guarantees
//! - Non-panicking: all public methods return `Result`.
//! - `Send + Sync`: safe to share across Tokio tasks.
//! - Lagged receivers get a [`BusError::Lagged`] error that reports how many
//!   messages were skipped, matching Tokio's broadcast semantics.
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_agent_runtime::bus::{AgentBus, AgentMessage, AgentTarget};
//! use llm_agent_runtime::types::AgentId;
//!
//! # tokio_test::block_on(async {
//! let bus = AgentBus::new(128);
//!
//! let planner = AgentId::new("planner");
//! let executor = AgentId::new("executor");
//!
//! // Subscribe both agents.
//! let mut planner_sub = bus.subscribe(planner.clone(), None);
//! let mut executor_sub = bus.subscribe(executor.clone(), Some("executor".to_string()));
//!
//! // Planner sends a task to the executor by ID.
//! bus.send(AgentMessage::new(
//!     planner.clone(),
//!     AgentTarget::Specific(executor.clone()),
//!     "Process batch-001",
//! )).unwrap();
//!
//! // Executor receives the message.
//! let msg = executor_sub.recv().await.unwrap();
//! assert_eq!(msg.content, "Process batch-001");
//! # });
//! ```

use crate::error::AgentRuntimeError;
use crate::types::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::broadcast;
use uuid::Uuid;

// ── MessageId ─────────────────────────────────────────────────────────────────

/// Stable unique identifier for a bus message.
///
/// Generated automatically by [`AgentMessage::new`].  Use the `reply_to` field
/// on a subsequent message to create a threaded conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub String);

impl MessageId {
    /// Create a new `MessageId` from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a random `MessageId` backed by a UUID v4.
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Return the inner ID string as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── AgentTarget ───────────────────────────────────────────────────────────────

/// Describes who should receive an [`AgentMessage`].
///
/// - `Specific(id)` — delivered only to the agent with the given [`AgentId`].
/// - `Broadcast` — delivered to **all** current subscribers.
/// - `Role(name)` — delivered to every subscriber registered with the named role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentTarget {
    /// Deliver the message to a single named agent.
    Specific(AgentId),
    /// Deliver the message to every subscribed agent.
    Broadcast,
    /// Deliver the message to every agent registered under the given role name.
    Role(String),
}

impl std::fmt::Display for AgentTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Specific(id) => write!(f, "agent:{id}"),
            Self::Broadcast => write!(f, "broadcast"),
            Self::Role(r) => write!(f, "role:{r}"),
        }
    }
}

// ── AgentRole ─────────────────────────────────────────────────────────────────

/// A named functional role that an agent can register on the bus.
///
/// Roles allow higher-level coordination without coupling senders to specific
/// agent IDs.  For example, a pipeline leader can send to `Role("executor")`
/// and any agent that has registered that role will receive the message.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::bus::AgentRole;
/// let role = AgentRole::new("critic");
/// assert_eq!(role.name(), "critic");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentRole {
    name: String,
}

impl AgentRole {
    /// Create a new `AgentRole` with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Return the role name as a `&str`.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ── AgentMessage ──────────────────────────────────────────────────────────────

/// A message sent through the [`AgentBus`].
///
/// # Fields
///
/// - `id` — unique identifier generated at construction time.
/// - `from` — the [`AgentId`] of the sending agent.
/// - `to` — the [`AgentTarget`] describing intended recipients.
/// - `content` — the free-form text payload.
/// - `reply_to` — if set, the [`MessageId`] of the message this is a reply to.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::bus::{AgentMessage, AgentTarget};
/// use llm_agent_runtime::types::AgentId;
///
/// let msg = AgentMessage::new(
///     AgentId::new("planner"),
///     AgentTarget::Broadcast,
///     "Starting task decomposition",
/// );
/// assert!(msg.is_broadcast());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Unique identifier for this message.
    pub id: MessageId,
    /// The agent that sent this message.
    pub from: AgentId,
    /// The intended recipient(s) of this message.
    pub to: AgentTarget,
    /// The free-form text payload.
    pub content: String,
    /// If set, the [`MessageId`] of the message this is a reply to.
    pub reply_to: Option<MessageId>,
}

impl AgentMessage {
    /// Create a new `AgentMessage` with a freshly generated [`MessageId`].
    ///
    /// # Arguments
    /// * `from` — the sending agent's [`AgentId`].
    /// * `to` — the [`AgentTarget`] describing who should receive the message.
    /// * `content` — the message text.
    pub fn new(
        from: impl Into<AgentId>,
        to: AgentTarget,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: MessageId::random(),
            from: from.into(),
            to,
            content: content.into(),
            reply_to: None,
        }
    }

    /// Attach a reply-to reference to this message, returning `self` for chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_agent_runtime::bus::{AgentMessage, AgentTarget, MessageId};
    /// use llm_agent_runtime::types::AgentId;
    ///
    /// let original_id = MessageId::new("orig-001");
    /// let reply = AgentMessage::new(
    ///     AgentId::new("executor"),
    ///     AgentTarget::Specific(AgentId::new("planner")),
    ///     "Task complete",
    /// )
    /// .with_reply_to(original_id.clone());
    ///
    /// assert_eq!(reply.reply_to.as_ref().unwrap().as_str(), "orig-001");
    /// ```
    pub fn with_reply_to(mut self, message_id: MessageId) -> Self {
        self.reply_to = Some(message_id);
        self
    }

    /// Return `true` if this message targets all agents (`AgentTarget::Broadcast`).
    pub fn is_broadcast(&self) -> bool {
        matches!(self.to, AgentTarget::Broadcast)
    }

    /// Return `true` if this message is targeted at a specific [`AgentId`].
    pub fn is_specific(&self) -> bool {
        matches!(self.to, AgentTarget::Specific(_))
    }

    /// Return `true` if this message targets a role.
    pub fn is_role_targeted(&self) -> bool {
        matches!(self.to, AgentTarget::Role(_))
    }

    /// Return `true` if this message is a reply to another message.
    pub fn is_reply(&self) -> bool {
        self.reply_to.is_some()
    }

    /// Return the content byte length.
    pub fn content_len(&self) -> usize {
        self.content.len()
    }
}

// ── BusError ──────────────────────────────────────────────────────────────────

/// Errors that can occur when receiving from a [`BusSubscription`].
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum BusError {
    /// The channel has been closed and no further messages will be delivered.
    #[error("Bus channel closed")]
    Closed,

    /// This receiver fell behind: `count` messages were skipped.
    ///
    /// The receiver is still alive; subsequent calls to `recv` will return
    /// the next available message.
    #[error("Bus receiver lagged by {count} message(s)")]
    Lagged {
        /// Number of messages that were skipped due to slow consumption.
        count: u64,
    },
}

impl From<broadcast::error::RecvError> for BusError {
    fn from(e: broadcast::error::RecvError) -> Self {
        match e {
            broadcast::error::RecvError::Closed => BusError::Closed,
            broadcast::error::RecvError::Lagged(n) => BusError::Lagged { count: n },
        }
    }
}

impl From<BusError> for AgentRuntimeError {
    fn from(e: BusError) -> Self {
        AgentRuntimeError::AgentLoop(format!("bus error: {e}"))
    }
}

// ── BusSubscription ───────────────────────────────────────────────────────────

/// A subscription handle returned by [`AgentBus::subscribe`].
///
/// Use [`BusSubscription::recv`] in a loop to consume messages.  Only messages
/// whose [`AgentTarget`] matches this agent's ID or registered role are
/// surfaced; all other traffic is silently skipped.
///
/// Drop the subscription to unsubscribe.
///
/// # Example
///
/// ```rust,no_run
/// # use llm_agent_runtime::bus::{AgentBus, AgentMessage, AgentTarget};
/// # use llm_agent_runtime::types::AgentId;
/// # tokio_test::block_on(async {
/// let bus = AgentBus::new(64);
/// let agent = AgentId::new("worker");
/// let mut sub = bus.subscribe(agent.clone(), Some("executor".to_string()));
///
/// // In a real agent this would be a tokio::select! loop.
/// bus.send(AgentMessage::new(
///     AgentId::new("planner"),
///     AgentTarget::Role("executor".to_string()),
///     "run step 1",
/// )).unwrap();
///
/// let msg = sub.recv().await.unwrap();
/// assert_eq!(msg.content, "run step 1");
/// # });
/// ```
pub struct BusSubscription {
    /// The agent identity this subscription belongs to.
    agent_id: AgentId,
    /// The optional role this agent has registered.
    role: Option<String>,
    /// The underlying broadcast receiver.
    rx: broadcast::Receiver<AgentMessage>,
}

impl BusSubscription {
    /// Return the [`AgentId`] this subscription belongs to.
    pub fn agent_id(&self) -> &AgentId {
        &self.agent_id
    }

    /// Return the registered role, if any.
    pub fn role(&self) -> Option<&str> {
        self.role.as_deref()
    }

    /// Receive the next message addressed to this agent.
    ///
    /// Messages targeting a *different* specific agent are skipped
    /// transparently.  Only messages matching one of these criteria are
    /// returned:
    ///
    /// - [`AgentTarget::Broadcast`]
    /// - [`AgentTarget::Specific`] where the target ID equals this agent's ID
    /// - [`AgentTarget::Role`] where the role name equals this agent's
    ///   registered role
    ///
    /// Returns [`BusError::Closed`] when the bus has been dropped, or
    /// [`BusError::Lagged`] if this receiver fell behind the channel capacity.
    pub async fn recv(&mut self) -> Result<AgentMessage, BusError> {
        loop {
            let msg = self.rx.recv().await?;
            if self.matches(&msg) {
                return Ok(msg);
            }
        }
    }

    /// Try to receive a message without blocking.
    ///
    /// Returns `None` if no matching message is immediately available.
    /// Returns `Err` on channel close or lag.
    pub fn try_recv(&mut self) -> Result<Option<AgentMessage>, BusError> {
        loop {
            match self.rx.try_recv() {
                Ok(msg) => {
                    if self.matches(&msg) {
                        return Ok(Some(msg));
                    }
                    // skip non-matching message and try next
                }
                Err(broadcast::error::TryRecvError::Empty) => return Ok(None),
                Err(broadcast::error::TryRecvError::Closed) => return Err(BusError::Closed),
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    return Err(BusError::Lagged { count: n })
                }
            }
        }
    }

    /// Return `true` if `msg` is addressed to this subscription.
    fn matches(&self, msg: &AgentMessage) -> bool {
        match &msg.to {
            AgentTarget::Broadcast => true,
            AgentTarget::Specific(id) => id == &self.agent_id,
            AgentTarget::Role(role) => self.role.as_deref() == Some(role.as_str()),
        }
    }
}

// ── BusStats ──────────────────────────────────────────────────────────────────

/// A snapshot of bus statistics for monitoring and observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusStats {
    /// Total number of messages sent through the bus since creation.
    pub total_sent: u64,
    /// Current number of active subscribers.
    pub subscriber_count: usize,
    /// Number of distinct roles currently registered.
    pub role_count: usize,
}

// ── AgentBus ──────────────────────────────────────────────────────────────────

/// An async broadcast message bus for inter-agent communication.
///
/// `AgentBus` is cheap to clone — every clone shares the same underlying
/// channel, metrics, and role registry.
///
/// # Channel capacity
///
/// The `capacity` argument passed to [`AgentBus::new`] is the number of
/// messages the channel will buffer before lagging slow receivers.  Choose a
/// value large enough for your expected burst traffic.  A typical starting
/// point is `128` to `1024`.
///
/// # Example
///
/// ```rust,no_run
/// use llm_agent_runtime::bus::{AgentBus, AgentMessage, AgentTarget};
/// use llm_agent_runtime::types::AgentId;
///
/// # tokio_test::block_on(async {
/// let bus = AgentBus::new(256);
///
/// let mut sub = bus.subscribe(AgentId::new("critic"), Some("critic".to_string()));
///
/// bus.send(AgentMessage::new(
///     AgentId::new("planner"),
///     AgentTarget::Role("critic".to_string()),
///     "Review this plan: ...",
/// )).unwrap();
///
/// let msg = sub.recv().await.unwrap();
/// println!("critic received: {}", msg.content);
/// # });
/// ```
#[derive(Clone)]
pub struct AgentBus {
    inner: Arc<BusInner>,
}

struct BusInner {
    tx: broadcast::Sender<AgentMessage>,
    /// Maps AgentId → role name for role-targeted delivery.
    roles: RwLock<HashMap<AgentId, String>>,
    /// Monotonic counter of total messages sent.
    total_sent: std::sync::atomic::AtomicU64,
}

impl AgentBus {
    /// Create a new `AgentBus` with the given channel capacity.
    ///
    /// `capacity` is the number of messages that will be buffered before slow
    /// receivers start lagging.  Must be at least `1`.
    ///
    /// # Panics (debug only)
    ///
    /// Triggers a `debug_assert!` if `capacity` is `0`.
    pub fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0, "AgentBus capacity must be > 0");
        let (tx, _) = broadcast::channel(capacity.max(1));
        Self {
            inner: Arc::new(BusInner {
                tx,
                roles: RwLock::new(HashMap::new()),
                total_sent: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }

    /// Subscribe an agent to the bus, optionally registering a role.
    ///
    /// Returns a [`BusSubscription`] that will receive messages matching the
    /// agent's ID or registered role.
    ///
    /// If the agent was previously registered with a different role, the role
    /// is updated to the new value.
    ///
    /// # Arguments
    /// * `agent_id` — the subscribing agent's identity.
    /// * `role` — optional role name (e.g. `"planner"`, `"executor"`).
    pub fn subscribe(
        &self,
        agent_id: AgentId,
        role: Option<String>,
    ) -> BusSubscription {
        // Register (or update) the role.
        if let Ok(mut roles) = self.inner.roles.write() {
            match &role {
                Some(r) => {
                    roles.insert(agent_id.clone(), r.clone());
                }
                None => {
                    // Do not erase an existing role registration.
                }
            }
        }

        BusSubscription {
            agent_id,
            role,
            rx: self.inner.tx.subscribe(),
        }
    }

    /// Publish a message to the bus.
    ///
    /// All current subscribers will receive the message.  Each subscriber
    /// then applies its own filter (see [`BusSubscription::matches`]) to
    /// decide whether to surface the message to its agent.
    ///
    /// Returns `Ok(n)` where `n` is the number of active receivers at the
    /// time of the send.  Returns `Err` if there are no receivers.
    ///
    /// # Errors
    ///
    /// Returns [`AgentRuntimeError::AgentLoop`] if the channel has no active
    /// receivers (i.e. all subscriptions have been dropped).
    pub fn send(&self, message: AgentMessage) -> Result<usize, AgentRuntimeError> {
        self.inner
            .total_sent
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.tx.send(message).map_err(|e| {
            AgentRuntimeError::AgentLoop(format!(
                "bus send failed — no active receivers: {e}"
            ))
        })
    }

    /// Return the number of active subscribers on the bus.
    pub fn subscriber_count(&self) -> usize {
        self.inner.tx.receiver_count()
    }

    /// Return the number of distinct roles currently registered.
    pub fn role_count(&self) -> usize {
        self.inner
            .roles
            .read()
            .map(|r| r.len())
            .unwrap_or(0)
    }

    /// Return the total number of messages sent through this bus since creation.
    pub fn total_sent(&self) -> u64 {
        self.inner
            .total_sent
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Return a snapshot of bus statistics.
    pub fn stats(&self) -> BusStats {
        BusStats {
            total_sent: self.total_sent(),
            subscriber_count: self.subscriber_count(),
            role_count: self.role_count(),
        }
    }

    /// Look up the role registered for `agent_id`, if any.
    pub fn role_for(&self, agent_id: &AgentId) -> Option<String> {
        self.inner
            .roles
            .read()
            .ok()
            .and_then(|r| r.get(agent_id).cloned())
    }

    /// Return the sorted list of all registered role names.
    pub fn registered_roles(&self) -> Vec<String> {
        let mut roles: Vec<String> = self
            .inner
            .roles
            .read()
            .map(|r| r.values().cloned().collect())
            .unwrap_or_default();
        roles.sort_unstable();
        roles.dedup();
        roles
    }

    /// Return `true` if at least one subscriber is registered with `role`.
    pub fn has_role(&self, role: &str) -> bool {
        self.inner
            .roles
            .read()
            .map(|r| r.values().any(|v| v == role))
            .unwrap_or(false)
    }
}

impl std::fmt::Debug for AgentBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBus")
            .field("subscribers", &self.subscriber_count())
            .field("total_sent", &self.total_sent())
            .finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_id_random_unique() {
        let a = MessageId::random();
        let b = MessageId::random();
        assert_ne!(a, b);
    }

    #[test]
    fn test_message_id_display() {
        let id = MessageId::new("msg-001");
        assert_eq!(id.to_string(), "msg-001");
    }

    #[test]
    fn test_agent_role_name() {
        let r = AgentRole::new("planner");
        assert_eq!(r.name(), "planner");
    }

    #[test]
    fn test_agent_message_new_is_not_reply() {
        let msg = AgentMessage::new(
            AgentId::new("a"),
            AgentTarget::Broadcast,
            "hello",
        );
        assert!(!msg.is_reply());
        assert!(msg.is_broadcast());
        assert!(!msg.is_specific());
        assert!(!msg.is_role_targeted());
    }

    #[test]
    fn test_agent_message_with_reply_to() {
        let orig = MessageId::new("orig");
        let reply = AgentMessage::new(
            AgentId::new("b"),
            AgentTarget::Specific(AgentId::new("a")),
            "done",
        )
        .with_reply_to(orig.clone());
        assert!(reply.is_reply());
        assert_eq!(reply.reply_to.as_ref().unwrap().as_str(), "orig");
    }

    #[test]
    fn test_agent_target_display() {
        assert_eq!(
            AgentTarget::Specific(AgentId::new("x")).to_string(),
            "agent:x"
        );
        assert_eq!(AgentTarget::Broadcast.to_string(), "broadcast");
        assert_eq!(AgentTarget::Role("critic".to_string()).to_string(), "role:critic");
    }

    #[test]
    fn test_bus_subscriber_count() {
        let bus = AgentBus::new(16);
        assert_eq!(bus.subscriber_count(), 0);
        let _s1 = bus.subscribe(AgentId::new("a"), None);
        let _s2 = bus.subscribe(AgentId::new("b"), None);
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[test]
    fn test_bus_send_no_receivers_returns_error() {
        let bus = AgentBus::new(16);
        let msg = AgentMessage::new(AgentId::new("x"), AgentTarget::Broadcast, "hi");
        let result = bus.send(msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_bus_role_registration() {
        let bus = AgentBus::new(16);
        let _sub = bus.subscribe(AgentId::new("critic"), Some("critic".to_string()));
        assert!(bus.has_role("critic"));
        assert!(!bus.has_role("planner"));
    }

    #[test]
    fn test_bus_registered_roles_sorted_deduped() {
        let bus = AgentBus::new(16);
        let _s1 = bus.subscribe(AgentId::new("a"), Some("executor".to_string()));
        let _s2 = bus.subscribe(AgentId::new("b"), Some("planner".to_string()));
        let _s3 = bus.subscribe(AgentId::new("c"), Some("executor".to_string()));
        let roles = bus.registered_roles();
        assert_eq!(roles, vec!["executor", "planner"]);
    }

    #[tokio::test]
    async fn test_broadcast_message_received_by_all() {
        let bus = AgentBus::new(16);
        let mut s1 = bus.subscribe(AgentId::new("a"), None);
        let mut s2 = bus.subscribe(AgentId::new("b"), None);

        bus.send(AgentMessage::new(
            AgentId::new("sender"),
            AgentTarget::Broadcast,
            "hello all",
        ))
        .unwrap();

        let m1 = s1.recv().await.unwrap();
        let m2 = s2.recv().await.unwrap();
        assert_eq!(m1.content, "hello all");
        assert_eq!(m2.content, "hello all");
    }

    #[tokio::test]
    async fn test_specific_message_only_received_by_target() {
        let bus = AgentBus::new(16);
        let mut target_sub = bus.subscribe(AgentId::new("target"), None);
        let mut other_sub = bus.subscribe(AgentId::new("other"), None);

        // Send a broadcast so other_sub has something to drain, then specific.
        bus.send(AgentMessage::new(
            AgentId::new("sender"),
            AgentTarget::Specific(AgentId::new("target")),
            "only for target",
        ))
        .unwrap();

        // target gets it
        let m = target_sub.recv().await.unwrap();
        assert_eq!(m.content, "only for target");

        // other should not get a specific message for target
        let result = other_sub.try_recv();
        // Either None (skipped) or we need to check content doesn't match
        // The subscription filter skips non-matching messages in try_recv.
        assert!(matches!(result, Ok(None)));
    }

    #[tokio::test]
    async fn test_role_targeted_message() {
        let bus = AgentBus::new(16);
        let mut executor_sub =
            bus.subscribe(AgentId::new("exec-1"), Some("executor".to_string()));
        let mut planner_sub =
            bus.subscribe(AgentId::new("plan-1"), Some("planner".to_string()));

        bus.send(AgentMessage::new(
            AgentId::new("leader"),
            AgentTarget::Role("executor".to_string()),
            "run task",
        ))
        .unwrap();

        // executor gets it
        let m = executor_sub.recv().await.unwrap();
        assert_eq!(m.content, "run task");

        // planner does not
        assert!(matches!(planner_sub.try_recv(), Ok(None)));
    }

    #[tokio::test]
    async fn test_total_sent_counter() {
        let bus = AgentBus::new(16);
        let _sub = bus.subscribe(AgentId::new("a"), None);

        for _ in 0..5 {
            bus.send(AgentMessage::new(
                AgentId::new("s"),
                AgentTarget::Broadcast,
                "ping",
            ))
            .unwrap();
        }
        assert_eq!(bus.total_sent(), 5);
    }

    #[tokio::test]
    async fn test_try_recv_returns_none_when_empty() {
        let bus = AgentBus::new(16);
        let mut sub = bus.subscribe(AgentId::new("a"), None);
        assert!(matches!(sub.try_recv(), Ok(None)));
    }

    #[test]
    fn test_bus_stats() {
        let bus = AgentBus::new(16);
        let _s = bus.subscribe(AgentId::new("a"), Some("planner".to_string()));
        let stats = bus.stats();
        assert_eq!(stats.subscriber_count, 1);
        assert_eq!(stats.role_count, 1);
        assert_eq!(stats.total_sent, 0);
    }

    #[test]
    fn test_bus_debug_format() {
        let bus = AgentBus::new(8);
        let debug = format!("{bus:?}");
        assert!(debug.contains("AgentBus"));
    }

    #[test]
    fn test_bus_error_display() {
        let e = BusError::Closed;
        assert!(e.to_string().contains("closed"));
        let e2 = BusError::Lagged { count: 7 };
        assert!(e2.to_string().contains("7"));
    }

    #[test]
    fn test_subscription_agent_id_and_role() {
        let bus = AgentBus::new(8);
        let sub = bus.subscribe(AgentId::new("worker"), Some("executor".to_string()));
        assert_eq!(sub.agent_id().as_str(), "worker");
        assert_eq!(sub.role(), Some("executor"));
    }
}
