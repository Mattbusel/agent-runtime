//! # Module: task_scheduler
//!
//! Priority-based task scheduler with deadlines and FIFO tie-breaking.
//!
//! Provides:
//! - [`Priority`]: task urgency level (Critical > High > Normal > Low).
//! - [`ScheduledTask`]: a submitted task with metadata.
//! - [`TaskScheduler`]: thread-safe max-heap scheduler with deadline support.

use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
    time::{Duration, Instant},
};

use std::sync::Mutex;

// ── Priority ──────────────────────────────────────────────────────────────────

/// Task urgency level.
///
/// `Critical(0)` is the highest possible priority (lower u8 = more urgent).
/// Ordering: `Critical < High < Normal < Low` when used as max-heap keys
/// (we flip internally so the heap gives us the most-urgent task first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Priority {
    /// Critical task; inner value 0 = highest within Critical tier.
    Critical(u8),
    /// High priority.
    High,
    /// Normal priority.
    Normal,
    /// Low priority.
    Low,
}

/// Numeric rank used for comparison: higher rank = more urgent.
impl Priority {
    fn rank(&self) -> u32 {
        match self {
            // Critical tier: rank 1_000_000 down to 999_745 (255 steps).
            Priority::Critical(v) => 1_000_000u32.saturating_sub(u32::from(*v)),
            Priority::High => 3_000,
            Priority::Normal => 2_000,
            Priority::Low => 1_000,
        }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ── ScheduledTask ─────────────────────────────────────────────────────────────

/// A task submitted to the scheduler.
#[derive(Debug)]
pub struct ScheduledTask {
    /// Monotonically increasing task ID.
    pub id: u64,
    /// Task urgency.
    pub priority: Priority,
    /// Optional absolute deadline.
    pub deadline: Option<Instant>,
    /// Caller-supplied payload string.
    pub payload: String,
    /// When the task was submitted.
    pub enqueued_at: Instant,
}

// ── Heap key ──────────────────────────────────────────────────────────────────

/// Wrapper used as the BinaryHeap element.
///
/// Ordering: higher priority first; among equal priority, earlier deadline
/// first; among equal deadline, lower id (FIFO) first.
struct HeapEntry {
    priority: Priority,
    /// `None` treated as "far future" (deadline = MAX).
    deadline: Option<Instant>,
    id: u64,
    task: ScheduledTask,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Higher priority wins.
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {}
            ord => return ord,
        }
        // 2. Earlier deadline wins (None = far future, so loses to Some).
        let dl_cmp = deadline_cmp(self.deadline, other.deadline);
        match dl_cmp {
            Ordering::Equal => {}
            ord => return ord,
        }
        // 3. FIFO: lower id wins (in a max-heap, "wins" means Greater).
        other.id.cmp(&self.id) // reversed: lower id → Greater in max-heap
    }
}

/// Compare two optional deadlines: earlier deadline is "greater" (wins in max-heap).
fn deadline_cmp(a: Option<Instant>, b: Option<Instant>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Greater, // a has deadline, b doesn't → a more urgent
        (None, Some(_)) => Ordering::Less,
        (Some(da), Some(db)) => db.cmp(&da), // earlier deadline → Greater
    }
}

// ── TaskScheduler ─────────────────────────────────────────────────────────────

/// Thread-safe priority task scheduler backed by a binary max-heap.
pub struct TaskScheduler {
    heap: Arc<Mutex<BinaryHeap<HeapEntry>>>,
    next_id: Arc<AtomicU64>,
}

impl TaskScheduler {
    /// Create a new empty scheduler.
    pub fn new() -> Self {
        Self {
            heap: Arc::new(Mutex::new(BinaryHeap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Submit a task and return its assigned ID.
    pub fn submit(
        &self,
        priority: Priority,
        payload: String,
        deadline: Option<Duration>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let now = Instant::now();
        let deadline_instant = deadline.map(|d| now + d);

        let task = ScheduledTask {
            id,
            priority: priority.clone(),
            deadline: deadline_instant,
            payload,
            enqueued_at: now,
        };

        let entry = HeapEntry {
            priority,
            deadline: deadline_instant,
            id,
            task,
        };

        let mut heap = lock_heap(&self.heap);
        heap.push(entry);
        id
    }

    /// Remove and return the highest-priority task, or `None` if empty.
    pub fn poll(&self) -> Option<ScheduledTask> {
        let mut heap = lock_heap(&self.heap);
        heap.pop().map(|e| e.task)
    }

    /// Peek at the priority of the next task without removing it.
    pub fn peek_priority(&self) -> Option<Priority> {
        let heap = lock_heap(&self.heap);
        heap.peek().map(|e| e.priority.clone())
    }

    /// Number of tasks currently in the queue.
    pub fn len(&self) -> usize {
        lock_heap(&self.heap).len()
    }

    /// Returns `true` if no tasks are queued.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Count of tasks whose deadline has already passed.
    pub fn deadline_expired_count(&self) -> usize {
        let now = Instant::now();
        let heap = lock_heap(&self.heap);
        heap.iter()
            .filter(|e| e.deadline.map_or(false, |d| d <= now))
            .count()
    }

    /// Remove and return all tasks whose deadline has already passed.
    pub fn drain_expired(&self) -> Vec<ScheduledTask> {
        let now = Instant::now();
        let mut heap = lock_heap(&self.heap);

        let mut remaining: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut expired: Vec<ScheduledTask> = Vec::new();

        while let Some(entry) = heap.pop() {
            if entry.deadline.map_or(false, |d| d <= now) {
                expired.push(entry.task);
            } else {
                remaining.push(entry);
            }
        }

        *heap = remaining;
        expired
    }
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Acquire the mutex, recovering from poison.
fn lock_heap(heap: &Arc<Mutex<BinaryHeap<HeapEntry>>>) -> std::sync::MutexGuard<'_, BinaryHeap<HeapEntry>> {
    match heap.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering_critical_beats_high() {
        assert!(Priority::Critical(0) > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn critical_lower_value_beats_higher_value() {
        // Critical(0) is more urgent than Critical(1).
        assert!(Priority::Critical(0) > Priority::Critical(1));
        assert!(Priority::Critical(1) > Priority::Critical(255));
    }

    #[test]
    fn poll_returns_highest_priority() {
        let sched = TaskScheduler::new();
        sched.submit(Priority::Low, "low".to_string(), None);
        sched.submit(Priority::Critical(0), "crit".to_string(), None);
        sched.submit(Priority::Normal, "norm".to_string(), None);

        let first = sched.poll().unwrap();
        assert_eq!(first.payload, "crit");
        let second = sched.poll().unwrap();
        assert_eq!(second.payload, "norm");
        let third = sched.poll().unwrap();
        assert_eq!(third.payload, "low");
    }

    #[test]
    fn fifo_within_same_priority() {
        let sched = TaskScheduler::new();
        sched.submit(Priority::Normal, "first".to_string(), None);
        sched.submit(Priority::Normal, "second".to_string(), None);
        sched.submit(Priority::Normal, "third".to_string(), None);

        let a = sched.poll().unwrap();
        let b = sched.poll().unwrap();
        let c = sched.poll().unwrap();
        assert_eq!(a.payload, "first");
        assert_eq!(b.payload, "second");
        assert_eq!(c.payload, "third");
    }

    #[test]
    fn deadline_expired_count_and_drain() {
        let sched = TaskScheduler::new();
        // Submit a task with a deadline that already passed.
        sched.submit(Priority::Normal, "expired".to_string(), Some(Duration::from_nanos(1)));
        sched.submit(Priority::Normal, "fresh".to_string(), Some(Duration::from_secs(9999)));

        // Give the nanosecond deadline time to expire.
        std::thread::sleep(Duration::from_millis(5));

        assert_eq!(sched.deadline_expired_count(), 1);
        let expired = sched.drain_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].payload, "expired");
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn len_and_is_empty() {
        let sched = TaskScheduler::new();
        assert!(sched.is_empty());
        sched.submit(Priority::High, "x".to_string(), None);
        assert_eq!(sched.len(), 1);
        assert!(!sched.is_empty());
        let _ = sched.poll();
        assert!(sched.is_empty());
    }
}
