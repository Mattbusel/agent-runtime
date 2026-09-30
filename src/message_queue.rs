//! # Module: Message Queue
//!
//! In-memory bounded message queue with priority ordering and TTL expiry.
//!
//! ## Overview
//!
//! [`MessageQueue`] is a generic, thread-safe bounded queue that orders messages
//! by [`Priority`] (highest first) and within the same priority by insertion
//! order (FIFO). Messages may carry an optional TTL; expired messages are
//! silently discarded on dequeue.
//!
//! ## Quick Start
//!
//! ```rust
//! use llm_agent_runtime::message_queue::{MessageQueue, Priority};
//! use std::sync::Arc;
//!
//! let q: Arc<MessageQueue<String>> = Arc::new(MessageQueue::new(64));
//! let id = q.send("hello".to_string(), Priority::Normal).unwrap();
//! let msg = q.try_recv().unwrap();
//! assert_eq!(msg.payload, "hello");
//! ```

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

// ── Priority ──────────────────────────────────────────────────────────────────

/// Message priority level — higher variants are dequeued first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Priority {
    /// Lowest priority — dropped first when the queue is full.
    Low = 0,
    /// Standard priority.
    Normal = 1,
    /// Elevated priority.
    High = 2,
    /// Highest priority — never dropped; sender blocks until space is available.
    Critical = 3,
}

// ── QueueError ────────────────────────────────────────────────────────────────

/// Errors returned by [`MessageQueue::send`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    /// The queue is at capacity and the message could not be enqueued.
    Full,
    /// The queue has been closed and no longer accepts messages.
    Closed,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueError::Full => write!(f, "queue is full"),
            QueueError::Closed => write!(f, "queue is closed"),
        }
    }
}

impl std::error::Error for QueueError {}

// ── Message ───────────────────────────────────────────────────────────────────

/// A message envelope carrying a `payload` with priority and optional TTL.
pub struct Message<T> {
    /// Monotonically increasing identifier assigned at enqueue time.
    pub id: u64,
    /// The contained value.
    pub payload: T,
    /// The priority at which this message was enqueued.
    pub priority: Priority,
    /// When this message was enqueued.
    pub enqueued_at: Instant,
    /// Optional time-to-live in milliseconds; `None` means never expires.
    pub ttl_ms: Option<u64>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Message<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Message")
            .field("id", &self.id)
            .field("payload", &self.payload)
            .field("priority", &self.priority)
            .field("ttl_ms", &self.ttl_ms)
            .finish()
    }
}

impl<T> Message<T> {
    /// Returns `true` when the message has exceeded its TTL.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        match self.ttl_ms {
            None => false,
            Some(ttl) => self.enqueued_at.elapsed().as_millis() as u64 > ttl,
        }
    }
}

// ── QueueStats ────────────────────────────────────────────────────────────────

/// A statistics snapshot for a [`MessageQueue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueStats {
    /// Total messages enqueued since creation.
    pub enqueued: u64,
    /// Total messages dequeued (delivered to callers) since creation.
    pub dequeued: u64,
    /// Total messages dropped (due to full queue or TTL expiry) since creation.
    pub dropped: u64,
    /// Current number of messages in the queue.
    pub current_len: usize,
    /// Maximum queue capacity.
    pub capacity: usize,
}

// ── Ordering key ─────────────────────────────────────────────────────────────

/// Ordering key: highest priority first, then FIFO within the same priority.
///
/// We store `(u8::MAX - priority_u8, sequence_id)` so that the `BTreeMap`'s
/// natural ascending order gives us the desired dequeue order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderKey(u8, u64);

impl OrderKey {
    fn new(priority: Priority, id: u64) -> Self {
        // Invert priority so higher values sort first in a min-ordered BTreeMap.
        Self(u8::MAX - priority as u8, id)
    }
}

// ── Inner state ───────────────────────────────────────────────────────────────

struct Inner<T> {
    map: BTreeMap<OrderKey, Message<T>>,
    closed: bool,
    enqueued: u64,
    dequeued: u64,
    dropped: u64,
}

impl<T> Inner<T> {
    fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            closed: false,
            enqueued: 0,
            dequeued: 0,
            dropped: 0,
        }
    }

    /// Remove and discard all expired messages, incrementing the dropped counter.
    fn evict_expired(&mut self) {
        let expired_keys: Vec<OrderKey> = self
            .map
            .iter()
            .filter(|(_, msg)| msg.is_expired())
            .map(|(k, _)| *k)
            .collect();
        let count = expired_keys.len() as u64;
        for k in expired_keys {
            self.map.remove(&k);
        }
        self.dropped += count;
    }

    /// Pop the highest-priority non-expired message.
    fn pop_next(&mut self) -> Option<Message<T>> {
        self.evict_expired();
        let key = *self.map.keys().next()?;
        let msg = self.map.remove(&key)?;
        self.dequeued += 1;
        Some(msg)
    }
}

// ── MessageQueue ──────────────────────────────────────────────────────────────

/// Bounded, priority-ordered in-memory message queue.
pub struct MessageQueue<T> {
    inner: Mutex<Inner<T>>,
    condvar: Condvar,
    capacity: usize,
    next_id: AtomicU64,
}

impl<T: Send + 'static> MessageQueue<T> {
    /// Create a new queue with the given capacity limit.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::new()),
            condvar: Condvar::new(),
            capacity,
            next_id: AtomicU64::new(1),
        }
    }

    /// Wrap `self` in an `Arc`.
    #[must_use]
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn make_message(id: u64, payload: T, priority: Priority, ttl_ms: Option<u64>) -> Message<T> {
        Message {
            id,
            payload,
            priority,
            enqueued_at: Instant::now(),
            ttl_ms,
        }
    }

    /// Enqueue `payload` at the given `priority`.
    ///
    /// - `Critical` messages: if the queue is full, block until space is available.
    /// - `Low` messages: dropped immediately if the queue is full.
    /// - All other messages: return `Err(QueueError::Full)` if the queue is full.
    ///
    /// Returns the assigned message id on success.
    pub fn send(&self, payload: T, priority: Priority) -> Result<u64, QueueError> {
        self.send_with_ttl(payload, priority, None)
    }

    /// Like [`send`](Self::send) but with an explicit TTL in milliseconds.
    pub fn send_with_ttl(
        &self,
        payload: T,
        priority: Priority,
        ttl_ms: Option<u64>,
    ) -> Result<u64, QueueError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        if priority == Priority::Critical {
            // Block until there is space.
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return Err(QueueError::Closed),
            };
            loop {
                if inner.closed {
                    return Err(QueueError::Closed);
                }
                if inner.map.len() < self.capacity {
                    break;
                }
                inner = match self.condvar.wait(inner) {
                    Ok(g) => g,
                    Err(_) => return Err(QueueError::Closed),
                };
            }
            let msg = Self::make_message(id, payload, priority, ttl_ms);
            inner.map.insert(OrderKey::new(priority, id), msg);
            inner.enqueued += 1;
            self.condvar.notify_one();
            return Ok(id);
        }

        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return Err(QueueError::Closed),
        };
        if inner.closed {
            return Err(QueueError::Closed);
        }
        if inner.map.len() >= self.capacity {
            if priority == Priority::Low {
                inner.dropped += 1;
                return Err(QueueError::Full);
            }
            return Err(QueueError::Full);
        }
        let msg = Self::make_message(id, payload, priority, ttl_ms);
        inner.map.insert(OrderKey::new(priority, id), msg);
        inner.enqueued += 1;
        self.condvar.notify_one();
        Ok(id)
    }

    /// Dequeue the highest-priority message, blocking until one is available.
    ///
    /// Returns `None` only when the queue has been closed and is empty.
    pub fn recv(&self) -> Option<Message<T>> {
        let mut inner = self.inner.lock().ok()?;
        loop {
            if let Some(msg) = inner.pop_next() {
                self.condvar.notify_one();
                return Some(msg);
            }
            if inner.closed {
                return None;
            }
            inner = self.condvar.wait(inner).ok()?;
        }
    }

    /// Dequeue with a timeout; returns `None` if no message arrives in time.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<Message<T>> {
        let mut inner = self.inner.lock().ok()?;
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(msg) = inner.pop_next() {
                self.condvar.notify_one();
                return Some(msg);
            }
            if inner.closed {
                return None;
            }
            let remaining = deadline.checked_duration_since(Instant::now())?;
            let (g, _timeout) = self.condvar.wait_timeout(inner, remaining).ok()?;
            inner = g;
            if Instant::now() >= deadline {
                return inner.pop_next().map(|msg| {
                    self.condvar.notify_one();
                    msg
                });
            }
        }
    }

    /// Non-blocking dequeue; returns `None` immediately if the queue is empty.
    pub fn try_recv(&self) -> Option<Message<T>> {
        let mut inner = self.inner.lock().ok()?;
        let msg = inner.pop_next()?;
        self.condvar.notify_one();
        Some(msg)
    }

    /// Return a statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> QueueStats {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return QueueStats {
                enqueued: 0,
                dequeued: 0,
                dropped: 0,
                current_len: 0,
                capacity: self.capacity,
            },
        };
        QueueStats {
            enqueued: inner.enqueued,
            dequeued: inner.dequeued,
            dropped: inner.dropped,
            current_len: inner.map.len(),
            capacity: self.capacity,
        }
    }

    /// Drain all messages from the queue, returning them in priority order.
    pub fn drain(&self) -> Vec<Message<T>> {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        inner.evict_expired();
        let keys: Vec<OrderKey> = inner.map.keys().copied().collect();
        let mut msgs = Vec::with_capacity(keys.len());
        for k in keys {
            if let Some(m) = inner.map.remove(&k) {
                inner.dequeued += 1;
                msgs.push(m);
            }
        }
        self.condvar.notify_all();
        msgs
    }

    /// Close the queue; pending receivers will drain and then receive `None`.
    pub fn close(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.closed = true;
        }
        self.condvar.notify_all();
    }
}

#[cfg(test)]
mod tests {
    #[allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn basic_send_recv() {
        let q: MessageQueue<i32> = MessageQueue::new(8);
        let id = q.send(42, Priority::Normal).unwrap();
        assert!(id >= 1);
        let msg = q.try_recv().unwrap();
        assert_eq!(msg.payload, 42);
        assert_eq!(msg.priority, Priority::Normal);
    }

    #[test]
    fn priority_ordering() {
        let q: MessageQueue<&str> = MessageQueue::new(8);
        q.send("low", Priority::Low).unwrap();
        q.send("high", Priority::High).unwrap();
        q.send("normal", Priority::Normal).unwrap();
        assert_eq!(q.try_recv().unwrap().payload, "high");
        assert_eq!(q.try_recv().unwrap().payload, "normal");
        assert_eq!(q.try_recv().unwrap().payload, "low");
    }

    #[test]
    fn full_queue_returns_error() {
        let q: MessageQueue<i32> = MessageQueue::new(2);
        q.send(1, Priority::Normal).unwrap();
        q.send(2, Priority::Normal).unwrap();
        let result = q.send(3, Priority::Normal);
        assert_eq!(result, Err(QueueError::Full));
    }

    #[test]
    fn ttl_expiry() {
        let q: MessageQueue<i32> = MessageQueue::new(8);
        // TTL of 0 ms — expires immediately.
        q.send_with_ttl(99, Priority::Normal, Some(0)).unwrap();
        // Give the Instant a moment to advance past 0 ms.
        std::thread::sleep(Duration::from_millis(2));
        assert!(q.try_recv().is_none());
    }

    #[test]
    fn drain_empties_queue() {
        let q: MessageQueue<i32> = MessageQueue::new(8);
        q.send(1, Priority::Low).unwrap();
        q.send(2, Priority::High).unwrap();
        let msgs = q.drain();
        assert_eq!(msgs.len(), 2);
        assert!(q.try_recv().is_none());
    }

    #[test]
    fn stats_tracking() {
        let q: MessageQueue<i32> = MessageQueue::new(4);
        q.send(1, Priority::Normal).unwrap();
        q.send(2, Priority::Normal).unwrap();
        q.try_recv();
        let s = q.stats();
        assert_eq!(s.enqueued, 2);
        assert_eq!(s.dequeued, 1);
        assert_eq!(s.current_len, 1);
        assert_eq!(s.capacity, 4);
    }
}
