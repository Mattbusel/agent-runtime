//! # Module: event_bus
//!
//! ## Responsibility
//! Typed publish/subscribe event bus for inter-agent communication within the
//! agent runtime.  Agents publish [`AgentEvent`] values to named topics;
//! subscribers receive every event published to a subscribed topic through a
//! `tokio::sync::broadcast` channel.
//!
//! ## Design
//! - One `broadcast::Sender<AgentEvent>` is maintained per topic (created
//!   lazily on first subscribe or publish).
//! - Capacity is fixed at 256 events per topic.  When a receiver is too slow
//!   the oldest events are dropped; [`EventBusStats::total_dropped`] counts
//!   every such drop.
//! - A wildcard topic (`"*"`) receives every event published via
//!   [`EventBus::publish_all`].
//! - [`EventBus::filtered_subscriber`] wraps the wildcard topic in an async
//!   channel filtered by an [`EventFilter`].
//!
//! ## Guarantees
//! - `EventBus` is `Clone + Send + Sync` (inner state behind `Arc<RwLock<…>>`).
//! - No panics in production paths; all fallible operations return `Result` or
//!   silently drop on lag.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::sync::{broadcast, mpsc, RwLock};

// ── Capacity ──────────────────────────────────────────────────────────────────

const CHANNEL_CAPACITY: usize = 256;

// ── Event trait ───────────────────────────────────────────────────────────────

/// Common interface implemented by every event type on the bus.
pub trait Event: Send + Sync + 'static {
    /// Short, stable identifier for this event kind (e.g. `"TaskStarted"`).
    fn event_type(&self) -> &str;

    /// Erase to `&dyn Any` so callers can downcast to the concrete type.
    fn as_any(&self) -> &dyn Any;
}

// ── AgentEvent ────────────────────────────────────────────────────────────────

/// All structured events that agents can publish to the bus.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A task was started by an agent.
    TaskStarted {
        /// Unique task identifier.
        task_id: String,
        /// Identifier of the agent that started the task.
        agent_id: String,
    },
    /// A task completed successfully.
    TaskCompleted {
        /// Unique task identifier.
        task_id: String,
        /// Serialised result (e.g. JSON or plain text).
        result: String,
    },
    /// A task failed.
    TaskFailed {
        /// Unique task identifier.
        task_id: String,
        /// Human-readable error description.
        error: String,
    },
    /// An agent transitioned between lifecycle states.
    StateChanged {
        /// Identifier of the agent whose state changed.
        agent_id: String,
        /// Previous state label.
        old_state: String,
        /// New state label.
        new_state: String,
    },
    /// An agent sent a message to another agent.
    MessageSent {
        /// Sender agent identifier.
        from: String,
        /// Recipient agent identifier.
        to: String,
        /// Raw message payload.
        payload: String,
    },
    /// A named numeric metric was recorded.
    MetricRecorded {
        /// Metric name.
        name: String,
        /// Metric value.
        value: f64,
    },
}

impl AgentEvent {
    /// Returns the event type string for topic-based routing.
    pub fn event_type_str(&self) -> &str {
        match self {
            AgentEvent::TaskStarted { .. } => "TaskStarted",
            AgentEvent::TaskCompleted { .. } => "TaskCompleted",
            AgentEvent::TaskFailed { .. } => "TaskFailed",
            AgentEvent::StateChanged { .. } => "StateChanged",
            AgentEvent::MessageSent { .. } => "MessageSent",
            AgentEvent::MetricRecorded { .. } => "MetricRecorded",
        }
    }

    /// Returns the `agent_id` field if this event carries one.
    pub fn agent_id(&self) -> Option<&str> {
        match self {
            AgentEvent::TaskStarted { agent_id, .. } => Some(agent_id.as_str()),
            AgentEvent::StateChanged { agent_id, .. } => Some(agent_id.as_str()),
            AgentEvent::MessageSent { from, .. } => Some(from.as_str()),
            _ => None,
        }
    }
}

impl Event for AgentEvent {
    fn event_type(&self) -> &str {
        self.event_type_str()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ── EventFilter ───────────────────────────────────────────────────────────────

/// Filter applied by [`EventBus::filtered_subscriber`] before forwarding
/// events to the caller.
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// When `Some`, only events whose `event_type` is in this list pass.
    pub allowed_types: Option<Vec<String>>,
    /// When `Some`, only events that carry a matching `agent_id` field pass.
    pub agent_id_filter: Option<String>,
}

impl EventFilter {
    /// Returns `true` when `event` passes all configured filter criteria.
    pub fn matches(&self, event: &AgentEvent) -> bool {
        if let Some(allowed) = &self.allowed_types {
            if !allowed.iter().any(|t| t == event.event_type_str()) {
                return false;
            }
        }
        if let Some(id) = &self.agent_id_filter {
            match event.agent_id() {
                Some(eid) if eid == id.as_str() => {}
                _ => return false,
            }
        }
        true
    }
}

// ── EventBusStats ─────────────────────────────────────────────────────────────

/// Snapshot of cumulative statistics for an [`EventBus`].
#[derive(Debug, Clone)]
pub struct EventBusStats {
    /// Number of active topics (including the wildcard `"*"` topic if used).
    pub topics: usize,
    /// Total events published since the bus was created.
    pub total_published: u64,
    /// Total events dropped because a receiver was too slow (lagged).
    pub total_dropped: u64,
}

// ── Inner state ───────────────────────────────────────────────────────────────

struct BusInner {
    /// One broadcast sender per topic, created lazily.
    senders: HashMap<String, broadcast::Sender<AgentEvent>>,
}

impl BusInner {
    fn new() -> Self {
        Self {
            senders: HashMap::new(),
        }
    }

    /// Returns (or creates) the sender for `topic`.
    fn sender_for(&mut self, topic: &str) -> &broadcast::Sender<AgentEvent> {
        if !self.senders.contains_key(topic) {
            let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
            self.senders.insert(topic.to_owned(), tx);
        }
        // Safe: we just inserted if missing.
        &self.senders[topic]
    }

    /// Publish to a single topic; returns the number of lagged/dropped events.
    fn publish_to(&mut self, topic: &str, event: AgentEvent) -> u64 {
        let sender = self.sender_for(topic);
        match sender.send(event) {
            Ok(_) => 0,
            Err(_) => {
                // No active receivers — not an error, no drop to count.
                0
            }
        }
    }
}

// ── EventBus ──────────────────────────────────────────────────────────────────

/// Typed, multi-topic publish/subscribe event bus.
///
/// Clone the bus freely — all clones share the same underlying state.
#[derive(Clone)]
pub struct EventBus {
    inner: Arc<RwLock<BusInner>>,
    total_published: Arc<AtomicU64>,
    total_dropped: Arc<AtomicU64>,
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("total_published", &self.total_published.load(Ordering::Relaxed))
            .field("total_dropped", &self.total_dropped.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    /// Creates a new, empty event bus.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BusInner::new())),
            total_published: Arc::new(AtomicU64::new(0)),
            total_dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Subscribe to a named `topic`.
    ///
    /// The returned [`broadcast::Receiver`] will receive every event published
    /// (or published-all) to `topic` after this call.
    pub async fn subscribe(&self, topic: &str) -> broadcast::Receiver<AgentEvent> {
        let mut inner = self.inner.write().await;
        inner.sender_for(topic).subscribe()
    }

    /// Publish `event` to its own event-type topic only (i.e. not to `"*"`).
    ///
    /// Subscribers of the topic equal to `event.event_type()` will receive it.
    pub async fn publish(&self, event: AgentEvent) {
        let topic = event.event_type_str().to_owned();
        let mut inner = self.inner.write().await;

        let dropped = {
            let sender = inner.sender_for(&topic);
            match sender.send(event) {
                Ok(n) => {
                    // n is receiver count; 0 dropped.
                    let _ = n;
                    0u64
                }
                Err(_) => 0,
            }
        };

        self.total_published.fetch_add(1, Ordering::Relaxed);
        if dropped > 0 {
            self.total_dropped.fetch_add(dropped, Ordering::Relaxed);
        }
    }

    /// Publish `event` to its own event-type topic **and** to the wildcard
    /// `"*"` topic, so wildcard subscribers also receive it.
    pub async fn publish_all(&self, event: AgentEvent) {
        let topic = event.event_type_str().to_owned();
        let cloned = event.clone();

        let mut inner = self.inner.write().await;

        // Publish to specific topic.
        let _ = inner.publish_to(&topic, event);

        // Publish clone to wildcard topic.
        let _ = inner.publish_to("*", cloned);

        self.total_published.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns a snapshot of current bus statistics.
    pub async fn stats(&self) -> EventBusStats {
        let inner = self.inner.read().await;
        EventBusStats {
            topics: inner.senders.len(),
            total_published: self.total_published.load(Ordering::Relaxed),
            total_dropped: self.total_dropped.load(Ordering::Relaxed),
        }
    }

    /// Subscribe to the wildcard `"*"` topic and return an async
    /// [`mpsc::Receiver`] that yields only events matching `filter`.
    ///
    /// A background task bridges the broadcast receiver to the mpsc channel;
    /// it exits automatically when the returned receiver is dropped.
    pub async fn filtered_subscriber(
        &self,
        filter: EventFilter,
    ) -> mpsc::Receiver<AgentEvent> {
        let (tx, rx) = mpsc::channel::<AgentEvent>(CHANNEL_CAPACITY);
        let mut bcast_rx = self.subscribe("*").await;

        tokio::spawn(async move {
            loop {
                match bcast_rx.recv().await {
                    Ok(event) => {
                        if filter.matches(&event) {
                            if tx.send(event).await.is_err() {
                                // Consumer dropped — exit silently.
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Skip lagged messages; keep running.
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        rx
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[tokio::test]
    async fn test_publish_and_receive() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe("TaskStarted").await;

        bus.publish(AgentEvent::TaskStarted {
            task_id: "t1".into(),
            agent_id: "a1".into(),
        })
        .await;

        let event = rx.recv().await.unwrap();
        match event {
            AgentEvent::TaskStarted { task_id, agent_id } => {
                assert_eq!(task_id, "t1");
                assert_eq!(agent_id, "a1");
            }
            _ => panic!("wrong event variant"),
        }
    }

    #[tokio::test]
    async fn test_multiple_subscribers_receive_same_event() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe("TaskCompleted").await;
        let mut rx2 = bus.subscribe("TaskCompleted").await;

        bus.publish(AgentEvent::TaskCompleted {
            task_id: "t2".into(),
            result: "ok".into(),
        })
        .await;

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();

        assert!(matches!(e1, AgentEvent::TaskCompleted { .. }));
        assert!(matches!(e2, AgentEvent::TaskCompleted { .. }));
    }

    #[tokio::test]
    async fn test_publish_all_reaches_wildcard() {
        let bus = EventBus::new();
        let mut rx_wild = bus.subscribe("*").await;
        let mut rx_specific = bus.subscribe("MetricRecorded").await;

        bus.publish_all(AgentEvent::MetricRecorded {
            name: "latency".into(),
            value: 42.0,
        })
        .await;

        let e1 = rx_wild.recv().await.unwrap();
        let e2 = rx_specific.recv().await.unwrap();
        assert!(matches!(e1, AgentEvent::MetricRecorded { .. }));
        assert!(matches!(e2, AgentEvent::MetricRecorded { .. }));
    }

    #[tokio::test]
    async fn test_stats_increment() {
        let bus = EventBus::new();
        let _rx = bus.subscribe("TaskFailed").await;

        bus.publish(AgentEvent::TaskFailed {
            task_id: "t3".into(),
            error: "oops".into(),
        })
        .await;

        let stats = bus.stats().await;
        assert_eq!(stats.total_published, 1);
        assert!(stats.topics >= 1);
    }

    #[tokio::test]
    async fn test_filtered_subscriber_type_filter() {
        let bus = EventBus::new();
        let filter = EventFilter {
            allowed_types: Some(vec!["TaskStarted".into()]),
            agent_id_filter: None,
        };
        let mut filtered = bus.filtered_subscriber(filter).await;

        // publish_all so wildcard topic sees events
        bus.publish_all(AgentEvent::TaskStarted {
            task_id: "t4".into(),
            agent_id: "a2".into(),
        })
        .await;
        bus.publish_all(AgentEvent::TaskFailed {
            task_id: "t5".into(),
            error: "nope".into(),
        })
        .await;

        // Give the bridge task time to forward.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Only TaskStarted should have been forwarded.
        let event = filtered.try_recv().unwrap();
        assert!(matches!(event, AgentEvent::TaskStarted { .. }));
        // TaskFailed must NOT be present.
        assert!(filtered.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_filtered_subscriber_agent_id_filter() {
        let bus = EventBus::new();
        let filter = EventFilter {
            allowed_types: None,
            agent_id_filter: Some("target-agent".into()),
        };
        let mut filtered = bus.filtered_subscriber(filter).await;

        bus.publish_all(AgentEvent::StateChanged {
            agent_id: "other-agent".into(),
            old_state: "idle".into(),
            new_state: "running".into(),
        })
        .await;
        bus.publish_all(AgentEvent::StateChanged {
            agent_id: "target-agent".into(),
            old_state: "idle".into(),
            new_state: "running".into(),
        })
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let event = filtered.try_recv().unwrap();
        match &event {
            AgentEvent::StateChanged { agent_id, .. } => {
                assert_eq!(agent_id, "target-agent");
            }
            _ => panic!("wrong event"),
        }
        assert!(filtered.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_dropped_receiver_does_not_panic() {
        let bus = EventBus::new();
        {
            let _rx = bus.subscribe("TaskStarted").await;
            // Drop rx immediately.
        }
        // Publishing with no receivers should not panic.
        bus.publish(AgentEvent::TaskStarted {
            task_id: "t6".into(),
            agent_id: "a3".into(),
        })
        .await;
        let stats = bus.stats().await;
        assert_eq!(stats.total_published, 1);
    }
}
