//! # Module: agent_coordinator
//!
//! ## Responsibility
//! Multi-agent coordination primitives: barriers, shared counters, and
//! leader election — all built on standard-library atomics and Mutex/Condvar
//! pairs so the module has no additional dependencies.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

// ── CoordinationRole ──────────────────────────────────────────────────────────

/// Role of an agent in a coordination group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationRole {
    /// Drives coordination decisions.
    Leader,
    /// Follows the leader's instructions.
    Follower,
    /// Observes without voting.
    Observer,
    /// Participating in a leader election.
    Candidate,
}

// ── BarrierState ──────────────────────────────────────────────────────────────

/// State returned after waiting on an [`AgentBarrier`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarrierState {
    /// The barrier has not yet been released; shows progress.
    Waiting {
        /// Agents that have arrived so far.
        arrived: usize,
        /// Agents needed to release the barrier.
        needed: usize,
    },
    /// The barrier was successfully released.
    Released,
    /// The wait timed out before the barrier was released.
    Timeout,
}

// ── AgentBarrier ─────────────────────────────────────────────────────────────

/// A synchronisation barrier: blocks until `needed` agents have arrived.
pub struct AgentBarrier {
    /// Human-readable name for diagnostics.
    pub name: String,
    /// Number of agents required to release the barrier.
    pub needed: usize,
    arrived: AtomicUsize,
    /// Condvar is used together with the Mutex<bool> (released flag).
    condvar: Condvar,
    released: Mutex<bool>,
}

impl std::fmt::Debug for AgentBarrier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBarrier")
            .field("name", &self.name)
            .field("needed", &self.needed)
            .field("arrived", &self.arrived.load(Ordering::SeqCst))
            .finish()
    }
}

impl AgentBarrier {
    /// Create a new barrier with the given name and arrival count.
    pub fn new(name: impl Into<String>, needed: usize) -> Self {
        Self {
            name: name.into(),
            needed,
            arrived: AtomicUsize::new(0),
            condvar: Condvar::new(),
            released: Mutex::new(false),
        }
    }

    /// Record one arrival.  Returns `true` if this arrival released the barrier.
    pub fn arrive(&self) -> bool {
        let prev = self.arrived.fetch_add(1, Ordering::SeqCst);
        let now = prev + 1;
        if now >= self.needed {
            let mut released = self.released.lock().unwrap_or_else(|e| e.into_inner());
            *released = true;
            self.condvar.notify_all();
            return true;
        }
        false
    }

    /// Block the current thread until the barrier is released or `timeout` expires.
    pub fn wait(&self, timeout: Duration) -> BarrierState {
        let guard = self.released.lock().unwrap_or_else(|e| e.into_inner());
        if *guard {
            return BarrierState::Released;
        }
        let (guard, timed_out) = self
            .condvar
            .wait_timeout_while(guard, timeout, |released| !*released)
            .unwrap_or_else(|e| e.into_inner());
        if *guard {
            drop(guard);
            BarrierState::Released
        } else if timed_out.timed_out() {
            let arrived = self.arrived.load(Ordering::SeqCst);
            BarrierState::Waiting {
                arrived,
                needed: self.needed,
            }
        } else {
            // Spurious wakeup but not yet released
            let arrived = self.arrived.load(Ordering::SeqCst);
            BarrierState::Waiting {
                arrived,
                needed: self.needed,
            }
        }
    }

    /// Reset the barrier so it can be reused.
    pub fn reset(&self) {
        self.arrived.store(0, Ordering::SeqCst);
        let mut released = self.released.lock().unwrap_or_else(|e| e.into_inner());
        *released = false;
    }
}

// ── SharedCounter ─────────────────────────────────────────────────────────────

/// A named atomic 64-bit signed counter shared across agents.
#[derive(Debug)]
pub struct SharedCounter {
    /// Human-readable name for diagnostics.
    pub name: String,
    value: AtomicI64,
}

impl SharedCounter {
    /// Create a counter starting at `initial`.
    pub fn new(name: impl Into<String>, initial: i64) -> Self {
        Self {
            name: name.into(),
            value: AtomicI64::new(initial),
        }
    }

    /// Add `by` to the counter and return the new value.
    pub fn increment(&self, by: i64) -> i64 {
        self.value.fetch_add(by, Ordering::SeqCst) + by
    }

    /// Subtract `by` from the counter and return the new value.
    pub fn decrement(&self, by: i64) -> i64 {
        self.value.fetch_sub(by, Ordering::SeqCst) - by
    }

    /// Read the current value.
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::SeqCst)
    }

    /// Atomically set the value to `new` if it equals `expected`.
    ///
    /// Returns `true` on success.
    pub fn compare_exchange(&self, expected: i64, new: i64) -> bool {
        self.value
            .compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}

// ── LeaderElection ────────────────────────────────────────────────────────────

/// A simple priority-based leader election: the candidate with the lowest
/// `priority` u64 value wins (treat 0 as highest priority).
#[derive(Debug)]
pub struct LeaderElection {
    /// ID of this node's candidate entry.
    pub candidate_id: String,
    /// All registered candidates: priority -> agent_id.
    candidates: Arc<Mutex<BTreeMap<u64, String>>>,
    /// Currently elected leader.
    elected: Arc<Mutex<Option<String>>>,
    /// Monotonically increasing term counter.
    term: AtomicU64,
}

impl LeaderElection {
    /// Create a new election context.
    pub fn new(candidate_id: impl Into<String>) -> Self {
        Self {
            candidate_id: candidate_id.into(),
            candidates: Arc::new(Mutex::new(BTreeMap::new())),
            elected: Arc::new(Mutex::new(None)),
            term: AtomicU64::new(0),
        }
    }

    /// Register a candidate with the given priority (lower wins).
    pub fn register(&self, priority: u64, agent_id: &str) {
        let mut cands = self.candidates.lock().unwrap_or_else(|e| e.into_inner());
        cands.insert(priority, agent_id.to_string());
    }

    /// Run an election and return the winner, if any candidates exist.
    pub fn elect(&self) -> Option<String> {
        let cands = self.candidates.lock().unwrap_or_else(|e| e.into_inner());
        let winner = cands.values().next().cloned();
        drop(cands);
        let mut elected = self.elected.lock().unwrap_or_else(|e| e.into_inner());
        *elected = winner.clone();
        if winner.is_some() {
            self.term.fetch_add(1, Ordering::SeqCst);
        }
        winner
    }

    /// Remove `agent_id` from the candidates and trigger a re-election.
    pub fn step_down(&self, agent_id: &str) {
        let mut cands = self.candidates.lock().unwrap_or_else(|e| e.into_inner());
        cands.retain(|_, v| v != agent_id);
        drop(cands);
        // Trigger re-election
        let _ = self.elect();
    }

    /// Return the currently elected leader without running a new election.
    pub fn current_leader(&self) -> Option<String> {
        self.elected
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Return the current election term.
    pub fn term(&self) -> u64 {
        self.term.load(Ordering::SeqCst)
    }
}

// ── CoordinatorStats ──────────────────────────────────────────────────────────

/// Snapshot of coordinator-wide statistics.
#[derive(Debug, Clone)]
pub struct CoordinatorStats {
    /// Number of registered barriers.
    pub barrier_count: usize,
    /// Number of registered counters.
    pub counter_count: usize,
    /// Number of registered elections.
    pub election_count: usize,
    /// Total barriers that have been released since creation.
    pub total_barriers_released: usize,
}

// ── AgentCoordinator ──────────────────────────────────────────────────────────

/// Central coordinator that manages barriers, counters, and leader elections.
///
/// Uses `Mutex<HashMap<…>>` for interior mutability without requiring DashMap.
#[derive(Debug, Default)]
pub struct AgentCoordinator {
    barriers: Mutex<HashMap<String, Arc<AgentBarrier>>>,
    counters: Mutex<HashMap<String, Arc<SharedCounter>>>,
    elections: Mutex<HashMap<String, Arc<LeaderElection>>>,
    total_barriers_released: AtomicUsize,
}

impl AgentCoordinator {
    /// Create a new, empty coordinator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get an existing barrier by name or create one requiring `needed` arrivals.
    pub fn get_or_create_barrier(&self, name: &str, needed: usize) -> Arc<AgentBarrier> {
        let mut map = self.barriers.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(AgentBarrier::new(name, needed)))
            .clone()
    }

    /// Get an existing counter by name or create one starting at `initial`.
    pub fn get_or_create_counter(&self, name: &str, initial: i64) -> Arc<SharedCounter> {
        let mut map = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(SharedCounter::new(name, initial)))
            .clone()
    }

    /// Get an existing election by name or create one for `candidate_id`.
    pub fn get_or_create_election(
        &self,
        name: &str,
        candidate_id: &str,
    ) -> Arc<LeaderElection> {
        let mut map = self.elections.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(LeaderElection::new(candidate_id)))
            .clone()
    }

    /// Record that a barrier has been released (called by users after `arrive` returns true).
    pub fn record_barrier_released(&self) {
        self.total_barriers_released.fetch_add(1, Ordering::Relaxed);
    }

    /// Return a snapshot of coordinator statistics.
    pub fn coordinator_stats(&self) -> CoordinatorStats {
        let barrier_count = self
            .barriers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        let counter_count = self
            .counters
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        let election_count = self
            .elections
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        CoordinatorStats {
            barrier_count,
            counter_count,
            election_count,
            total_barriers_released: self.total_barriers_released.load(Ordering::Relaxed),
        }
    }
}
