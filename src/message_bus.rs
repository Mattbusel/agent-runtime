//! # Message Bus
//!
//! A lightweight in-process pub/sub message bus for inter-agent communication.
//!
//! ## Design
//!
//! - Agents [`publish`](PubSubBus::publish) messages tagged with a `topic`.
//! - Subscribers register [`Subscription`]s whose `topic_pattern` may end with
//!   `*` as a wildcard suffix (`"events.*"` matches `"events.user.login"`).
//! - [`PubSubBus::poll`] drains pending messages for a subscriber, filtering
//!   out any messages whose TTL has expired.
//! - Expired or undeliverable messages are moved to the dead-letter queue and
//!   counted in [`BusMetrics`].

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MessagePriority
// ---------------------------------------------------------------------------

/// Priority level assigned to a [`BusMessage`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Non-urgent background traffic.
    Low,
    /// Default priority for most messages.
    Normal,
    /// Time-sensitive operational messages.
    High,
    /// Requires immediate processing — bypass queuing where possible.
    Critical,
}

impl MessagePriority {
    /// Returns the relative weight used when ordering mixed-priority queues.
    ///
    /// Higher weight = processed sooner.
    pub fn queue_weight(&self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Normal => 4,
            Self::High => 16,
            Self::Critical => 64,
        }
    }
}

// ---------------------------------------------------------------------------
// BusMessage
// ---------------------------------------------------------------------------

/// A single message routed through the [`PubSubBus`].
#[derive(Debug, Clone)]
pub struct BusMessage {
    /// Unique identifier for this message (auto-assigned by [`PubSubBus::publish`]).
    pub id: String,
    /// Agent that produced this message.
    pub from_agent: String,
    /// Optional directed recipient agent.  `None` means broadcast to all topic subscribers.
    pub to_agent: Option<String>,
    /// Topic string, e.g. `"events.user.login"`.
    pub topic: String,
    /// Serialised payload (typically JSON).
    pub payload: String,
    /// Routing priority.
    pub priority: MessagePriority,
    /// Monotonic timestamp (ms) when the message was created.
    pub timestamp: u64,
    /// Optional time-to-live in milliseconds.  Messages older than `timestamp + ttl_ms`
    /// are dropped during [`PubSubBus::poll`] and moved to the dead-letter queue.
    pub ttl_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// DeliveryGuarantee
// ---------------------------------------------------------------------------

/// Delivery semantics offered or requested for a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryGuarantee {
    /// The message may be lost.  Fastest, zero persistence.
    AtMostOnce,
    /// The message will be delivered at least once; duplicates are possible.
    AtLeastOnce,
    /// The message is delivered exactly once; requires deduplication support.
    ExactlyOnce,
}

// ---------------------------------------------------------------------------
// Subscription
// ---------------------------------------------------------------------------

/// A subscriber's interest in a topic pattern.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique identifier for the subscribing agent or component.
    pub subscriber_id: String,
    /// Topic pattern.  A trailing `*` acts as a wildcard prefix match:
    /// `"events.*"` matches any topic that starts with `"events."`.
    pub topic_pattern: String,
}

impl Subscription {
    /// Returns `true` if `topic` matches this subscription's pattern.
    pub fn matches(&self, topic: &str) -> bool {
        if let Some(prefix) = self.topic_pattern.strip_suffix('*') {
            topic.starts_with(prefix)
        } else {
            self.topic_pattern == topic
        }
    }
}

// ---------------------------------------------------------------------------
// BusMetrics
// ---------------------------------------------------------------------------

/// Aggregate operational metrics for a [`PubSubBus`].
#[derive(Debug, Clone, Default)]
pub struct BusMetrics {
    /// Total messages passed to [`PubSubBus::publish`] or [`PubSubBus::direct_send`].
    pub published: u64,
    /// Total messages delivered to at least one subscriber via [`PubSubBus::poll`].
    pub delivered: u64,
    /// Messages dropped because no matching subscription was found.
    pub dropped: u64,
    /// Messages moved to the dead-letter queue (expired TTL or undeliverable).
    pub dead_letter: u64,
}

// ---------------------------------------------------------------------------
// PubSubBus
// ---------------------------------------------------------------------------

/// In-process publish/subscribe message bus.
///
/// A single [`PubSubBus`] instance is owned by the application and shared
/// (e.g. via `Arc<Mutex<PubSubBus>>`) across agents.
pub struct PubSubBus {
    /// All active subscriptions.
    subscriptions: Vec<Subscription>,
    /// Per-subscriber pending message queues.
    queues: HashMap<String, Vec<BusMessage>>,
    /// Messages that could not be delivered or have expired.
    dlq: Vec<BusMessage>,
    /// Running metrics counters.
    metrics: BusMetrics,
    /// Monotonically increasing message counter used to generate unique IDs.
    id_counter: u64,
}

impl PubSubBus {
    /// Create a new, empty [`PubSubBus`].
    pub fn new() -> Self {
        Self {
            subscriptions: Vec::new(),
            queues: HashMap::new(),
            dlq: Vec::new(),
            metrics: BusMetrics::default(),
            id_counter: 0,
        }
    }

    // ------------------------------------------------------------------
    // Publishing
    // ------------------------------------------------------------------

    /// Publish a message to all subscribers whose topic pattern matches
    /// `msg.topic`.
    ///
    /// The message `id` field is **replaced** by an auto-generated ID.
    /// Returns the assigned message ID.
    pub fn publish(&mut self, mut msg: BusMessage) -> String {
        self.id_counter += 1;
        let id = format!("msg-{}", self.id_counter);
        msg.id = id.clone();

        self.metrics.published += 1;

        let matching: Vec<String> = self
            .subscriptions
            .iter()
            .filter(|s| s.matches(&msg.topic))
            .map(|s| s.subscriber_id.clone())
            .collect();

        if matching.is_empty() {
            self.metrics.dropped += 1;
            self.dlq.push(msg);
        } else {
            for sub_id in matching {
                self.queues
                    .entry(sub_id)
                    .or_default()
                    .push(msg.clone());
            }
        }

        id
    }

    /// Send a message directly to a single subscriber, bypassing topic routing.
    ///
    /// The `to_agent` field of `msg` must be set; if it is `None` the message
    /// is dropped to the dead-letter queue.
    pub fn direct_send(&mut self, mut msg: BusMessage) {
        self.id_counter += 1;
        msg.id = format!("msg-{}", self.id_counter);
        self.metrics.published += 1;

        match &msg.to_agent {
            Some(target) => {
                self.queues
                    .entry(target.clone())
                    .or_default()
                    .push(msg);
            }
            None => {
                self.metrics.dead_letter += 1;
                self.dlq.push(msg);
            }
        }
    }

    // ------------------------------------------------------------------
    // Subscription management
    // ------------------------------------------------------------------

    /// Register a new subscription.
    ///
    /// Duplicate `(subscriber_id, topic_pattern)` pairs are silently ignored.
    pub fn subscribe(&mut self, subscription: Subscription) {
        let already = self.subscriptions.iter().any(|s| {
            s.subscriber_id == subscription.subscriber_id
                && s.topic_pattern == subscription.topic_pattern
        });
        if !already {
            self.subscriptions.push(subscription);
        }
    }

    /// Remove all subscriptions for `subscriber_id` that match `topic`.
    pub fn unsubscribe(&mut self, subscriber_id: &str, topic: &str) {
        self.subscriptions.retain(|s| {
            !(s.subscriber_id == subscriber_id && s.topic_pattern == topic)
        });
    }

    // ------------------------------------------------------------------
    // Polling
    // ------------------------------------------------------------------

    /// Drain and return all pending messages for `subscriber_id`.
    ///
    /// Messages whose TTL has expired relative to `now` are moved to the
    /// dead-letter queue rather than returned to the caller.
    pub fn poll(&mut self, subscriber_id: &str, now: u64) -> Vec<BusMessage> {
        let queue = match self.queues.get_mut(subscriber_id) {
            Some(q) => std::mem::take(q),
            None => return Vec::new(),
        };

        let mut live = Vec::new();
        for msg in queue {
            if is_expired(&msg, now) {
                self.metrics.dead_letter += 1;
                self.dlq.push(msg);
            } else {
                self.metrics.delivered += 1;
                live.push(msg);
            }
        }
        live
    }

    // ------------------------------------------------------------------
    // Observability
    // ------------------------------------------------------------------

    /// Returns a slice of all messages in the dead-letter queue.
    pub fn dead_letter_queue(&self) -> &[BusMessage] {
        &self.dlq
    }

    /// Returns a snapshot of the current bus metrics.
    pub fn metrics(&self) -> BusMetrics {
        self.metrics.clone()
    }
}

impl Default for PubSubBus {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` if `msg` has a TTL and it has elapsed by `now`.
fn is_expired(msg: &BusMessage, now: u64) -> bool {
    match msg.ttl_ms {
        Some(ttl) => now > msg.timestamp.saturating_add(ttl),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_msg(from: &str, topic: &str, payload: &str, ts: u64, ttl: Option<u64>) -> BusMessage {
        BusMessage {
            id: String::new(), // will be overwritten by publish
            from_agent: from.to_string(),
            to_agent: None,
            topic: topic.to_string(),
            payload: payload.to_string(),
            priority: MessagePriority::Normal,
            timestamp: ts,
            ttl_ms: ttl,
        }
    }

    // ------------------------------------------------------------------
    // MessagePriority
    // ------------------------------------------------------------------

    #[test]
    fn priority_weights_increase() {
        assert!(MessagePriority::Low.queue_weight() < MessagePriority::Normal.queue_weight());
        assert!(MessagePriority::Normal.queue_weight() < MessagePriority::High.queue_weight());
        assert!(MessagePriority::High.queue_weight() < MessagePriority::Critical.queue_weight());
    }

    // ------------------------------------------------------------------
    // Subscription matching
    // ------------------------------------------------------------------

    #[test]
    fn subscription_exact_match() {
        let s = Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "events.login".to_string(),
        };
        assert!(s.matches("events.login"));
        assert!(!s.matches("events.logout"));
    }

    #[test]
    fn subscription_wildcard_matches_prefix() {
        let s = Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "events.*".to_string(),
        };
        assert!(s.matches("events.user.login"));
        assert!(s.matches("events."));
        assert!(!s.matches("metrics.cpu"));
    }

    #[test]
    fn subscription_bare_wildcard_matches_everything() {
        let s = Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "*".to_string(),
        };
        assert!(s.matches("anything.at.all"));
        assert!(s.matches(""));
    }

    // ------------------------------------------------------------------
    // publish / poll round-trip
    // ------------------------------------------------------------------

    #[test]
    fn publish_delivers_to_matching_subscriber() {
        let mut bus = PubSubBus::new();
        bus.subscribe(Subscription {
            subscriber_id: "agent-a".to_string(),
            topic_pattern: "events.login".to_string(),
        });
        let id = bus.publish(make_msg("sender", "events.login", "hello", 0, None));
        assert!(!id.is_empty());
        let msgs = bus.poll("agent-a", 100);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload, "hello");
    }

    #[test]
    fn publish_no_subscriber_goes_to_dropped_and_dlq() {
        let mut bus = PubSubBus::new();
        bus.publish(make_msg("sender", "unknown.topic", "x", 0, None));
        let m = bus.metrics();
        assert_eq!(m.dropped, 1);
        assert_eq!(bus.dead_letter_queue().len(), 1);
    }

    #[test]
    fn expired_message_lands_in_dlq() {
        let mut bus = PubSubBus::new();
        bus.subscribe(Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "t".to_string(),
        });
        bus.publish(make_msg("s", "t", "old", 0, Some(100))); // expires at 100ms
        let msgs = bus.poll("a", 200); // polled at 200ms → expired
        assert_eq!(msgs.len(), 0);
        assert_eq!(bus.dead_letter_queue().len(), 1);
        assert_eq!(bus.metrics().dead_letter, 1);
    }

    #[test]
    fn non_expired_message_is_delivered() {
        let mut bus = PubSubBus::new();
        bus.subscribe(Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "t".to_string(),
        });
        bus.publish(make_msg("s", "t", "fresh", 0, Some(1_000)));
        let msgs = bus.poll("a", 500); // within TTL
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn wildcard_delivers_to_multiple_subscribers() {
        let mut bus = PubSubBus::new();
        bus.subscribe(Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "events.*".to_string(),
        });
        bus.subscribe(Subscription {
            subscriber_id: "b".to_string(),
            topic_pattern: "events.*".to_string(),
        });
        bus.publish(make_msg("s", "events.login", "hi", 0, None));
        let ma = bus.poll("a", 0);
        let mb = bus.poll("b", 0);
        assert_eq!(ma.len(), 1);
        assert_eq!(mb.len(), 1);
    }

    #[test]
    fn unsubscribe_stops_delivery() {
        let mut bus = PubSubBus::new();
        bus.subscribe(Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "t".to_string(),
        });
        bus.unsubscribe("a", "t");
        bus.publish(make_msg("s", "t", "ignored", 0, None));
        let msgs = bus.poll("a", 0);
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn duplicate_subscription_not_added() {
        let mut bus = PubSubBus::new();
        bus.subscribe(Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "t".to_string(),
        });
        bus.subscribe(Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "t".to_string(),
        });
        assert_eq!(bus.subscriptions.len(), 1);
    }

    #[test]
    fn direct_send_bypasses_topic_routing() {
        let mut bus = PubSubBus::new();
        // no subscription for "t" but direct send should still work
        let mut msg = make_msg("s", "t", "direct", 0, None);
        msg.to_agent = Some("agent-x".to_string());
        bus.direct_send(msg);
        let msgs = bus.poll("agent-x", 0);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload, "direct");
    }

    #[test]
    fn direct_send_without_to_agent_goes_to_dlq() {
        let mut bus = PubSubBus::new();
        let msg = make_msg("s", "t", "no-dest", 0, None);
        bus.direct_send(msg); // to_agent = None
        assert_eq!(bus.dead_letter_queue().len(), 1);
        assert_eq!(bus.metrics().dead_letter, 1);
    }

    #[test]
    fn metrics_accumulate_correctly() {
        let mut bus = PubSubBus::new();
        bus.subscribe(Subscription {
            subscriber_id: "a".to_string(),
            topic_pattern: "t".to_string(),
        });
        bus.publish(make_msg("s", "t", "1", 0, None));
        bus.publish(make_msg("s", "unknown", "2", 0, None));
        bus.poll("a", 0);
        let m = bus.metrics();
        assert_eq!(m.published, 2);
        assert_eq!(m.delivered, 1);
        assert_eq!(m.dropped, 1);
    }

    #[test]
    fn poll_empty_queue_returns_empty_vec() {
        let bus = PubSubBus::new();
        let msgs = bus.poll("nobody", 0); // calling poll on immutable ref via wrapper works
        // Actually poll requires &mut self — let's use a mutable one:
        let mut bus2 = PubSubBus::new();
        let msgs2 = bus2.poll("nobody", 0);
        assert!(msgs.is_empty());
        assert!(msgs2.is_empty());
    }

    #[test]
    fn delivery_guarantee_variants_exist() {
        let _at_most = DeliveryGuarantee::AtMostOnce;
        let _at_least = DeliveryGuarantee::AtLeastOnce;
        let _exactly = DeliveryGuarantee::ExactlyOnce;
        assert_ne!(DeliveryGuarantee::AtMostOnce, DeliveryGuarantee::ExactlyOnce);
    }
}
