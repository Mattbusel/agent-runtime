//! Event store with projections and replay.
//!
//! Provides an append-only event log per aggregate, snapshot support,
//! and a `Projection` trait for building read-model views from replayed events.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ── EventMetadata ────────────────────────────────────────────────────────────

/// Metadata attached to every stored event.
#[derive(Debug, Clone)]
pub struct EventMetadata {
    /// Globally unique, monotonically increasing event id.
    pub id: u64,
    /// The aggregate this event belongs to.
    pub aggregate_id: String,
    /// Discriminator string for the event payload type.
    pub event_type: String,
    /// Unix-epoch milliseconds when the event was appended.
    pub timestamp_ms: u64,
    /// Per-aggregate monotonically increasing sequence number.
    pub sequence: u64,
    /// Optional correlation / causation id.
    pub correlation_id: Option<String>,
}

// ── StoredEvent ──────────────────────────────────────────────────────────────

/// An event as it lives inside the store.
#[derive(Debug, Clone)]
pub struct StoredEvent {
    /// Standard metadata.
    pub metadata: EventMetadata,
    /// Serialised payload (expected to be a JSON string, but not validated here).
    pub payload: String,
}

// ── EventStore ───────────────────────────────────────────────────────────────

/// Append-only, thread-safe in-memory event store.
pub struct EventStore {
    events: Mutex<Vec<StoredEvent>>,
    next_id: AtomicU64,
    sequences: Mutex<HashMap<String, u64>>,
}

impl EventStore {
    /// Create an empty event store.
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            sequences: Mutex::new(HashMap::new()),
        }
    }

    /// Append an event to the store.
    ///
    /// Returns the globally-unique event id assigned to the new event.
    pub fn append(
        &self,
        aggregate_id: &str,
        event_type: &str,
        payload: String,
        correlation_id: Option<String>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let sequence = {
            let mut seqs = self.sequences.lock().unwrap();
            let seq = seqs.entry(aggregate_id.to_string()).or_insert(0);
            *seq += 1;
            *seq
        };

        let event = StoredEvent {
            metadata: EventMetadata {
                id,
                aggregate_id: aggregate_id.to_string(),
                event_type: event_type.to_string(),
                timestamp_ms: 0, // caller sets via payload; kept 0 for simplicity
                sequence,
                correlation_id,
            },
            payload,
        };

        self.events.lock().unwrap().push(event);
        id
    }

    /// Return all events for the given aggregate in sequence order.
    pub fn events_for_aggregate(&self, aggregate_id: &str) -> Vec<StoredEvent> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.metadata.aggregate_id == aggregate_id)
            .cloned()
            .collect()
    }

    /// Return events for the given aggregate with sequence **greater than** `seq`.
    pub fn events_since_sequence(&self, aggregate_id: &str, seq: u64) -> Vec<StoredEvent> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.metadata.aggregate_id == aggregate_id && e.metadata.sequence > seq)
            .cloned()
            .collect()
    }

    /// Return all events whose `event_type` matches the given string.
    pub fn all_events_of_type(&self, event_type: &str) -> Vec<StoredEvent> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.metadata.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Replay all events for `aggregate_id` through a projection, in sequence order.
    pub fn replay(&self, aggregate_id: &str, projection: &mut dyn Projection) {
        let mut events = self.events_for_aggregate(aggregate_id);
        events.sort_by_key(|e| e.metadata.sequence);
        for event in &events {
            projection.apply(event);
        }
    }
}

impl Default for EventStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of aggregate state.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The aggregate this snapshot belongs to.
    pub aggregate_id: String,
    /// Serialised state (JSON expected but not validated).
    pub state: String,
    /// The sequence number at which this snapshot was taken.
    pub at_sequence: u64,
    /// Unix-epoch milliseconds when the snapshot was taken.
    pub at_timestamp_ms: u64,
}

// ── SnapshotStore ─────────────────────────────────────────────────────────────

/// Thread-safe in-memory snapshot store (one snapshot per aggregate).
pub struct SnapshotStore {
    snapshots: Mutex<HashMap<String, Snapshot>>,
}

impl SnapshotStore {
    /// Create an empty snapshot store.
    pub fn new() -> Self {
        Self {
            snapshots: Mutex::new(HashMap::new()),
        }
    }

    /// Persist a snapshot, replacing any previous one for the same aggregate.
    pub fn save(&self, snap: Snapshot) {
        self.snapshots
            .lock()
            .unwrap()
            .insert(snap.aggregate_id.clone(), snap);
    }

    /// Load the most recent snapshot for an aggregate, if any.
    pub fn load(&self, aggregate_id: &str) -> Option<Snapshot> {
        self.snapshots.lock().unwrap().get(aggregate_id).cloned()
    }
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Projection trait ──────────────────────────────────────────────────────────

/// A read-model projection built by consuming a stream of stored events.
pub trait Projection {
    /// Incorporate a single event into the projection's state.
    fn apply(&mut self, event: &StoredEvent);
    /// Return the current projection state as a serialised string.
    fn state(&self) -> String;
}

// ── CountProjection ───────────────────────────────────────────────────────────

/// Projection that counts how many events of each type have been seen.
#[derive(Debug, Default)]
pub struct CountProjection {
    /// Maps event-type string to the number of events seen.
    pub counts: HashMap<String, usize>,
}

impl Projection for CountProjection {
    fn apply(&mut self, event: &StoredEvent) {
        *self.counts.entry(event.metadata.event_type.clone()).or_insert(0) += 1;
    }

    fn state(&self) -> String {
        // Simple deterministic serialisation
        let mut pairs: Vec<_> = self.counts.iter().collect();
        pairs.sort_by_key(|(k, _)| k.as_str());
        let inner = pairs
            .iter()
            .map(|(k, v)| format!("\"{}\":{}", k, v))
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{}}}", inner)
    }
}

// ── TimelineProjection ────────────────────────────────────────────────────────

/// Projection that records a (timestamp_ms, event_type) timeline.
#[derive(Debug, Default)]
pub struct TimelineProjection {
    /// Ordered list of `(timestamp_ms, event_type)` tuples.
    pub timeline: Vec<(u64, String)>,
}

impl Projection for TimelineProjection {
    fn apply(&mut self, event: &StoredEvent) {
        self.timeline
            .push((event.metadata.timestamp_ms, event.metadata.event_type.clone()));
    }

    fn state(&self) -> String {
        let items = self
            .timeline
            .iter()
            .map(|(ts, et)| format!("[{},\"{}\"]", ts, et))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{}]", items)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_retrieve() {
        let store = EventStore::new();
        let id1 = store.append("agg-1", "Created", r#"{"v":1}"#.into(), None);
        let id2 = store.append("agg-1", "Updated", r#"{"v":2}"#.into(), Some("corr-123".into()));
        store.append("agg-2", "Created", r#"{"v":1}"#.into(), None);

        let events = store.events_for_aggregate("agg-1");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].metadata.id, id1);
        assert_eq!(events[1].metadata.id, id2);
        assert_eq!(events[1].metadata.correlation_id.as_deref(), Some("corr-123"));
    }

    #[test]
    fn sequence_ordering() {
        let store = EventStore::new();
        store.append("agg-1", "A", "{}".into(), None);
        store.append("agg-1", "B", "{}".into(), None);
        store.append("agg-1", "C", "{}".into(), None);

        let events = store.events_for_aggregate("agg-1");
        assert_eq!(events[0].metadata.sequence, 1);
        assert_eq!(events[1].metadata.sequence, 2);
        assert_eq!(events[2].metadata.sequence, 3);
    }

    #[test]
    fn events_since_sequence() {
        let store = EventStore::new();
        store.append("agg-1", "A", "{}".into(), None); // seq 1
        store.append("agg-1", "B", "{}".into(), None); // seq 2
        store.append("agg-1", "C", "{}".into(), None); // seq 3

        let events = store.events_since_sequence("agg-1", 1);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.metadata.sequence > 1));
    }

    #[test]
    fn projection_replay() {
        let store = EventStore::new();
        store.append("agg-1", "Created", "{}".into(), None);
        store.append("agg-1", "Updated", "{}".into(), None);
        store.append("agg-1", "Updated", "{}".into(), None);

        let mut proj = CountProjection::default();
        store.replay("agg-1", &mut proj);

        assert_eq!(*proj.counts.get("Created").unwrap(), 1);
        assert_eq!(*proj.counts.get("Updated").unwrap(), 2);
    }

    #[test]
    fn snapshot_save_and_load() {
        let store = SnapshotStore::new();
        let snap = Snapshot {
            aggregate_id: "agg-1".into(),
            state: r#"{"balance":100}"#.into(),
            at_sequence: 5,
            at_timestamp_ms: 9999,
        };
        store.save(snap.clone());

        let loaded = store.load("agg-1").expect("snapshot must exist");
        assert_eq!(loaded.at_sequence, 5);
        assert_eq!(loaded.state, r#"{"balance":100}"#);

        // Overwrite with newer snapshot
        store.save(Snapshot {
            aggregate_id: "agg-1".into(),
            state: r#"{"balance":200}"#.into(),
            at_sequence: 10,
            at_timestamp_ms: 99999,
        });
        let loaded2 = store.load("agg-1").unwrap();
        assert_eq!(loaded2.at_sequence, 10);

        // Non-existent aggregate
        assert!(store.load("agg-99").is_none());
    }

    #[test]
    fn all_events_of_type() {
        let store = EventStore::new();
        store.append("agg-1", "TypeA", "{}".into(), None);
        store.append("agg-2", "TypeA", "{}".into(), None);
        store.append("agg-3", "TypeB", "{}".into(), None);

        let type_a = store.all_events_of_type("TypeA");
        assert_eq!(type_a.len(), 2);

        let type_b = store.all_events_of_type("TypeB");
        assert_eq!(type_b.len(), 1);
    }

    #[test]
    fn timeline_projection() {
        let store = EventStore::new();
        store.append("agg-1", "Alpha", "{}".into(), None);
        store.append("agg-1", "Beta", "{}".into(), None);

        let mut proj = TimelineProjection::default();
        store.replay("agg-1", &mut proj);

        assert_eq!(proj.timeline.len(), 2);
        assert_eq!(proj.timeline[0].1, "Alpha");
        assert_eq!(proj.timeline[1].1, "Beta");
    }
}
