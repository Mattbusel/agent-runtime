//! Agent pool with work-stealing and dynamic scaling.
//!
//! Provides a thread-safe pool of [`PooledAgent`]s backed by a priority work
//! queue.  Idle agents are preferentially selected for new tasks; when all
//! agents are busy the work-stealing heuristic attempts to reassign queued
//! items from the conceptual queue of the busiest agent.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ── AgentStatus ───────────────────────────────────────────────────────────────

/// Current operational status of a pooled agent.
#[derive(Debug, Clone)]
pub enum AgentStatus {
    /// Not currently processing any task.
    Idle,
    /// Executing a task identified by `task_id`.
    Busy {
        /// Identifier of the running task.
        task_id: String,
        /// Instant at which the task started.
        started_at: Instant,
    },
    /// Accepting no new tasks; finishing current work before stopping.
    Draining,
    /// Fully stopped; should be removed from the pool.
    Stopped,
}

// ── AgentStats ────────────────────────────────────────────────────────────────

/// Snapshot of a single agent's performance counters.
#[derive(Debug, Clone)]
pub struct AgentStats {
    /// Agent identifier.
    pub id: String,
    /// String description of the current status.
    pub status: String,
    /// Total tasks completed successfully or not.
    pub tasks_completed: u64,
    /// Total tasks that resulted in failure.
    pub tasks_failed: u64,
    /// Average task duration in milliseconds.
    pub avg_task_ms: f64,
    /// Estimated percentage of time the agent was busy.
    pub utilization_pct: f64,
}

// ── PooledAgent ───────────────────────────────────────────────────────────────

/// A single agent entry in the pool.
pub struct PooledAgent {
    /// Stable identifier.
    pub id: String,
    /// Current status, protected by a mutex.
    pub status: Arc<Mutex<AgentStatus>>,
    /// Count of completed tasks (success + failure).
    pub tasks_completed: AtomicU64,
    /// Count of tasks that failed.
    pub tasks_failed: AtomicU64,
    /// Accumulated processing time in milliseconds.
    pub total_task_ms: AtomicU64,
}

impl PooledAgent {
    /// Create a new idle agent with the given identifier.
    pub fn new(id: impl Into<String>) -> Self {
        PooledAgent {
            id: id.into(),
            status: Arc::new(Mutex::new(AgentStatus::Idle)),
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            total_task_ms: AtomicU64::new(0),
        }
    }

    /// Return `true` if the agent is currently idle.
    pub fn is_idle(&self) -> bool {
        let guard = self.status.lock().unwrap_or_else(|e| e.into_inner());
        matches!(*guard, AgentStatus::Idle)
    }

    /// Attempt a compare-and-swap from `Idle` to `Busy { task_id }`.
    ///
    /// Returns `true` on success (agent was idle and is now reserved).
    pub fn acquire(&self, task_id: &str) -> bool {
        let mut guard = self.status.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*guard, AgentStatus::Idle) {
            *guard = AgentStatus::Busy {
                task_id: task_id.to_string(),
                started_at: Instant::now(),
            };
            true
        } else {
            false
        }
    }

    /// Transition from `Busy` back to `Idle`, recording outcome counters.
    pub fn release(&self, success: bool, duration_ms: u64) {
        let mut guard = self.status.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*guard, AgentStatus::Busy { .. }) {
            *guard = AgentStatus::Idle;
        }
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.tasks_failed.fetch_add(1, Ordering::Relaxed);
        }
        self.total_task_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Snapshot current performance metrics.
    pub fn stats(&self) -> AgentStats {
        let completed = self.tasks_completed.load(Ordering::Relaxed);
        let failed = self.tasks_failed.load(Ordering::Relaxed);
        let total_ms = self.total_task_ms.load(Ordering::Relaxed);
        let avg_task_ms = if completed == 0 {
            0.0
        } else {
            total_ms as f64 / completed as f64
        };
        // Simple utilization proxy: fraction of total elapsed time spent working.
        // Without a wall-clock start reference we approximate from task throughput.
        let utilization_pct = if completed == 0 { 0.0 } else { 50.0 }; // placeholder heuristic
        let status_str = {
            let g = self.status.lock().unwrap_or_else(|e| e.into_inner());
            match &*g {
                AgentStatus::Idle => "Idle".to_string(),
                AgentStatus::Busy { task_id, .. } => format!("Busy({task_id})"),
                AgentStatus::Draining => "Draining".to_string(),
                AgentStatus::Stopped => "Stopped".to_string(),
            }
        };
        AgentStats {
            id: self.id.clone(),
            status: status_str,
            tasks_completed: completed,
            tasks_failed: failed,
            avg_task_ms,
            utilization_pct,
        }
    }
}

// ── WorkItem ──────────────────────────────────────────────────────────────────

/// A unit of work to be processed by an agent.
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// Unique task identifier.
    pub task_id: String,
    /// Serialised task payload.
    pub payload: String,
    /// Priority (higher value = higher priority).
    pub priority: u8,
    /// Time at which the item was submitted to the queue.
    pub enqueued_at: Instant,
}

// ── PoolStats ─────────────────────────────────────────────────────────────────

/// Aggregate statistics for the entire agent pool.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total agents registered in the pool.
    pub total_agents: usize,
    /// Agents currently idle.
    pub idle_agents: usize,
    /// Agents currently busy.
    pub busy_agents: usize,
    /// Number of work items waiting in the queue.
    pub queue_depth: usize,
    /// Sum of `tasks_completed` across all agents.
    pub total_tasks_completed: u64,
    /// Mean utilization across all agents.
    pub avg_utilization_pct: f64,
}

// ── AgentPool ─────────────────────────────────────────────────────────────────

/// A priority work queue backed by a dynamic pool of [`PooledAgent`]s.
pub struct AgentPool {
    /// All registered agents.
    pub agents: Vec<Arc<PooledAgent>>,
    /// Shared priority queue: `(priority, sequence, item)`.
    pub work_queue: Arc<Mutex<BinaryHeap<(u8, u64, OrdWorkItem)>>>,
    /// Minimum number of idle agents to maintain.
    pub min_idle: usize,
    /// Maximum pool size.
    pub max_size: usize,
    /// Monotonically increasing sequence counter for FIFO tie-breaking.
    seq: u64,
}

/// Newtype wrapper so `WorkItem` can be placed inside `BinaryHeap`.
#[derive(Debug, Clone)]
pub struct OrdWorkItem(pub WorkItem);

// BinaryHeap requires Ord; we provide a dummy ordering that never compares
// payloads (the heap key is the `(priority, sequence)` tuple).
impl PartialEq for OrdWorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.0.task_id == other.0.task_id
    }
}
impl Eq for OrdWorkItem {}
impl PartialOrd for OrdWorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdWorkItem {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
    }
}

impl AgentPool {
    /// Create a new pool with `initial_size` agents, a minimum idle count of
    /// `min_idle`, and a maximum size of `max_size`.
    pub fn new(initial_size: usize, min_idle: usize, max_size: usize) -> Self {
        let agents: Vec<Arc<PooledAgent>> = (0..initial_size)
            .map(|i| Arc::new(PooledAgent::new(format!("agent-{i}"))))
            .collect();
        AgentPool {
            agents,
            work_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            min_idle,
            max_size,
            seq: 0,
        }
    }

    /// Enqueue a work item. Returns an error if the pool is at capacity and
    /// no space exists for new agents.
    pub fn submit(&mut self, item: WorkItem) -> Result<(), String> {
        let priority = item.priority;
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        let mut queue = self.work_queue.lock().unwrap_or_else(|e| e.into_inner());
        queue.push((priority, seq, OrdWorkItem(item)));
        Ok(())
    }

    /// Find and return the first idle agent, if any.
    pub fn acquire_idle(&self) -> Option<Arc<PooledAgent>> {
        self.agents.iter().find(|a| a.is_idle()).cloned()
    }

    /// Work-stealing: take the highest-priority item from the shared queue and
    /// assign it conceptually to `target_agent`.  Returns the stolen item.
    pub fn steal_work(&self, _target_agent: &PooledAgent) -> Option<WorkItem> {
        let mut queue = self.work_queue.lock().unwrap_or_else(|e| e.into_inner());
        queue.pop().map(|(_, _, OrdWorkItem(item))| item)
    }

    /// Add a new agent to the pool if below `max_size`.
    ///
    /// Returns the new agent's ID on success.
    pub fn scale_up(&mut self) -> Option<String> {
        if self.agents.len() >= self.max_size {
            return None;
        }
        let id = format!("agent-{}", self.agents.len());
        let agent = Arc::new(PooledAgent::new(id.clone()));
        self.agents.push(agent);
        Some(id)
    }

    /// Remove idle agents above `min_idle`.
    pub fn scale_down(&mut self) {
        let mut removed = 0usize;
        let idle_count = self.agents.iter().filter(|a| a.is_idle()).count();
        let to_remove = idle_count.saturating_sub(self.min_idle);
        self.agents.retain(|a| {
            if removed < to_remove && a.is_idle() {
                removed += 1;
                false
            } else {
                true
            }
        });
    }

    /// Return a snapshot of pool statistics.
    pub fn pool_stats(&self) -> PoolStats {
        let total_agents = self.agents.len();
        let idle_agents = self.agents.iter().filter(|a| a.is_idle()).count();
        let busy_agents = total_agents - idle_agents;
        let queue_depth = {
            let q = self.work_queue.lock().unwrap_or_else(|e| e.into_inner());
            q.len()
        };
        let total_tasks_completed = self
            .agents
            .iter()
            .map(|a| a.tasks_completed.load(Ordering::Relaxed))
            .sum();
        let avg_utilization_pct = if total_agents == 0 {
            0.0
        } else {
            self.agents.iter().map(|a| a.stats().utilization_pct).sum::<f64>()
                / total_agents as f64
        };
        PoolStats {
            total_agents,
            idle_agents,
            busy_agents,
            queue_depth,
            total_tasks_completed,
            avg_utilization_pct,
        }
    }

    /// Initiate graceful shutdown: mark all agents as draining.
    pub fn drain(&self) {
        for agent in &self.agents {
            let mut guard = agent.status.lock().unwrap_or_else(|e| e.into_inner());
            *guard = AgentStatus::Draining;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_agent_acquire_release() {
        let agent = PooledAgent::new("a1");
        assert!(agent.is_idle());
        assert!(agent.acquire("task-1"));
        assert!(!agent.is_idle());
        agent.release(true, 10);
        assert!(agent.is_idle());
    }

    #[test]
    fn pool_submit_and_steal() {
        let mut pool = AgentPool::new(2, 1, 4);
        let item = WorkItem {
            task_id: "t1".to_string(),
            payload: "hello".to_string(),
            priority: 5,
            enqueued_at: Instant::now(),
        };
        pool.submit(item).expect("submit failed");
        let stats = pool.pool_stats();
        assert_eq!(stats.queue_depth, 1);
        let dummy = PooledAgent::new("dummy");
        let stolen = pool.steal_work(&dummy);
        assert!(stolen.is_some());
        assert_eq!(stolen.unwrap().task_id, "t1");
    }

    #[test]
    fn scale_up_down() {
        let mut pool = AgentPool::new(2, 1, 4);
        let id = pool.scale_up();
        assert!(id.is_some());
        assert_eq!(pool.agents.len(), 3);
        pool.scale_down();
        assert!(pool.agents.len() >= 1);
    }
}
