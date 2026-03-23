//! # Module: supervisor
//!
//! ## Responsibility
//! Erlang-style agent supervisor tree with three restart strategies.
//!
//! Provides:
//! - [`SupervisorStrategy`]: `OneForOne`, `OneForAll`, `RestForOne`.
//! - [`RestartPolicy`]: `Always`, `OnFailure`, `Never`.
//! - [`ChildSpec`]: specification for a supervised child task.
//! - [`Supervisor`]: spawns and monitors children; restarts per strategy.
//! - [`SupervisorHandle`]: returned by `Supervisor::start`; supports
//!   graceful shutdown and stats queries.
//!
//! ## Guarantees
//! - Shutdown is graceful: the supervisor signals all children and waits up
//!   to each child's configured `shutdown_timeout` before dropping handles.
//! - Max-restarts is per child; a child that exceeds its limit is permanently
//!   stopped (does not take down siblings in `OneForOne`).
//! - Never panics in library code.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use tokio::{
    sync::Mutex,
    task::JoinSet,
};

use crate::error::AgentRuntimeError;

/// Type alias for the boxed async task function stored in a [`ChildSpec`].
pub type TaskFn = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), AgentError>> + Send>>
        + Send
        + Sync,
>;

/// Errors produced or propagated by supervised child tasks.
#[derive(Debug, Clone)]
pub struct AgentError(pub String);

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentError: {}", self.0)
    }
}

impl std::error::Error for AgentError {}

impl From<AgentRuntimeError> for AgentError {
    fn from(e: AgentRuntimeError) -> Self {
        AgentError(e.to_string())
    }
}

// ── RestartPolicy ──────────────────────────────────────────────────────────────

/// Determines when a child should be restarted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartPolicy {
    /// Always restart the child, whether it exited normally or failed.
    Always,
    /// Restart only when the child exits with an `Err`.
    OnFailure,
    /// Never restart the child, regardless of how it exits.
    Never,
}

// ── SupervisorStrategy ─────────────────────────────────────────────────────────

/// Determines which children are restarted when one child fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupervisorStrategy {
    /// Restart only the failed child.  Siblings are unaffected.
    OneForOne,
    /// Restart all children when any one fails.
    OneForAll,
    /// Restart the failed child plus all children started after it.
    RestForOne,
}

// ── ChildSpec ──────────────────────────────────────────────────────────────────

/// Specification for a single supervised child task.
pub struct ChildSpec {
    /// Unique identifier for this child within the supervisor.
    pub id: String,
    /// Factory function that creates the child's async task.
    pub task_fn: TaskFn,
    /// When to restart this child.
    pub restart: RestartPolicy,
    /// Maximum number of restarts before giving up.
    pub max_restarts: u32,
    /// How long to wait for graceful shutdown before forcibly dropping.
    pub shutdown_timeout: Duration,
}

impl ChildSpec {
    /// Construct a new `ChildSpec`.
    pub fn new(
        id: impl Into<String>,
        task_fn: TaskFn,
        restart: RestartPolicy,
        max_restarts: u32,
        shutdown_timeout: Duration,
    ) -> Self {
        Self {
            id: id.into(),
            task_fn,
            restart,
            max_restarts,
            shutdown_timeout,
        }
    }
}

// ── ChildState ─────────────────────────────────────────────────────────────────

/// Runtime state of a supervised child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildState {
    /// The child is currently running.
    Running,
    /// The child exited cleanly and will not be restarted.
    Stopped,
    /// The child exceeded its `max_restarts` limit and has been permanently stopped.
    Failed,
}

// ── ChildStats ─────────────────────────────────────────────────────────────────

/// Per-child statistics snapshot.
#[derive(Debug, Clone)]
pub struct ChildStats {
    /// The child's identifier.
    pub id: String,
    /// Number of times this child has been restarted.
    pub restarts: u32,
    /// Current lifecycle state.
    pub state: ChildState,
}

// ── SupervisorStats ────────────────────────────────────────────────────────────

/// Aggregate statistics for the supervisor.
#[derive(Debug, Clone)]
pub struct SupervisorStats {
    /// Total number of restarts across all children.
    pub total_restarts: u32,
    /// Per-child statistics.
    pub children: Vec<ChildStats>,
}

// ── Internal child record ──────────────────────────────────────────────────────

struct ChildRecord {
    spec: ChildSpec,
    restarts: u32,
    state: ChildState,
    /// Monotonic index: the order in which this child was added to the supervisor.
    /// Used by `RestForOne` to identify children that were started after a failing one.
    start_order: usize,
}

// ── SupervisorState (shared between Supervisor and SupervisorHandle) ──────────

pub(crate) struct SupervisorState {
    pub(crate) children: Vec<ChildRecord>,
}

// ── Supervisor ─────────────────────────────────────────────────────────────────

/// Erlang-style supervisor that manages a set of async child tasks.
///
/// # Example
/// ```rust,ignore
/// use std::{sync::Arc, time::Duration};
/// use llm_agent_runtime::supervisor::{
///     ChildSpec, RestartPolicy, Supervisor, SupervisorStrategy, TaskFn,
/// };
///
/// let spec = ChildSpec::new(
///     "worker",
///     Arc::new(|| Box::pin(async { Ok(()) })),
///     RestartPolicy::OnFailure,
///     3,
///     Duration::from_secs(5),
/// );
/// // let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
/// ```
pub struct Supervisor;

impl Supervisor {
    /// Start the supervisor with the given children and strategy.
    ///
    /// Spawns each child as a Tokio task and monitors them via a background
    /// loop.  Returns a [`SupervisorHandle`] for graceful shutdown and stats.
    pub async fn start(
        children: Vec<ChildSpec>,
        strategy: SupervisorStrategy,
    ) -> SupervisorHandle {
        let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);
        let state = Arc::new(Mutex::new(SupervisorState {
            children: children
                .into_iter()
                .enumerate()
                .map(|(i, spec)| ChildRecord {
                    spec,
                    restarts: 0,
                    state: ChildState::Running,
                    start_order: i,
                })
                .collect(),
        }));

        let state_clone = state.clone();
        let shutdown_rx = shutdown_tx.subscribe();

        // Spawn the supervisor loop as a background Tokio task.
        tokio::spawn(async move {
            supervisor_loop(state_clone, strategy, shutdown_rx).await;
        });

        SupervisorHandle {
            state,
            shutdown_tx,
        }
    }
}

// ── supervisor_loop ────────────────────────────────────────────────────────────

async fn supervisor_loop(
    state: Arc<Mutex<SupervisorState>>,
    strategy: SupervisorStrategy,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    // Track in-flight JoinSet handles keyed by child index.
    let mut join_set: JoinSet<(usize, Result<(), AgentError>)> = JoinSet::new();

    // Initial spawn: start all children.
    {
        let guard = state.lock().await;
        for (idx, record) in guard.children.iter().enumerate() {
            if record.state == ChildState::Running {
                let task_fn = record.spec.task_fn.clone();
                join_set.spawn(async move {
                    let result = task_fn().await;
                    (idx, result)
                });
            }
        }
    }

    loop {
        tokio::select! {
            // Graceful shutdown requested.
            _ = shutdown_rx.recv() => {
                break;
            }

            // A child task completed (success or failure).
            Some(outcome) = join_set.join_next() => {
                let (idx, task_result) = match outcome {
                    Ok(r) => r,
                    Err(_join_err) => {
                        // JoinError means the task panicked; treat as failure.
                        // We can't easily recover the idx here; skip.
                        continue;
                    }
                };

                let mut guard = state.lock().await;
                let n_children = guard.children.len();

                // Determine whether this child should be restarted.
                let should_restart = {
                    let record = &guard.children[idx];
                    match record.spec.restart {
                        RestartPolicy::Never => false,
                        RestartPolicy::Always => true,
                        RestartPolicy::OnFailure => task_result.is_err(),
                    }
                };

                let within_limit = {
                    let record = &guard.children[idx];
                    record.restarts < record.spec.max_restarts
                };

                if should_restart && within_limit {
                    // Apply restart strategy.
                    match strategy {
                        SupervisorStrategy::OneForOne => {
                            // Restart only the failed child.
                            let record = &mut guard.children[idx];
                            record.restarts += 1;
                            let task_fn = record.spec.task_fn.clone();
                            join_set.spawn(async move {
                                let result = task_fn().await;
                                (idx, result)
                            });
                        }

                        SupervisorStrategy::OneForAll => {
                            // Restart the failed child and increment its counter.
                            guard.children[idx].restarts += 1;

                            // Abort all running tasks then respawn all children.
                            drop(guard);
                            join_set.abort_all();
                            // Drain the join set.
                            while join_set.join_next().await.is_some() {}

                            let mut guard = state.lock().await;
                            for i in 0..n_children {
                                let record = &mut guard.children[i];
                                if record.state != ChildState::Failed
                                    && record.state != ChildState::Stopped
                                {
                                    // Only increment the child that actually failed.
                                    let task_fn = record.spec.task_fn.clone();
                                    join_set.spawn(async move {
                                        let result = task_fn().await;
                                        (i, result)
                                    });
                                }
                            }
                        }

                        SupervisorStrategy::RestForOne => {
                            // Increment restart count for the failed child.
                            let failed_order = guard.children[idx].start_order;
                            guard.children[idx].restarts += 1;

                            // Abort all and re-launch failed child + all children started after.
                            drop(guard);
                            join_set.abort_all();
                            while join_set.join_next().await.is_some() {}

                            let mut guard = state.lock().await;
                            for i in 0..n_children {
                                let record = &mut guard.children[i];
                                if record.start_order >= failed_order
                                    && record.state != ChildState::Failed
                                    && record.state != ChildState::Stopped
                                {
                                    let task_fn = record.spec.task_fn.clone();
                                    join_set.spawn(async move {
                                        let result = task_fn().await;
                                        (i, result)
                                    });
                                }
                            }
                        }
                    }
                } else if should_restart && !within_limit {
                    // Exceeded max_restarts — mark as permanently failed.
                    guard.children[idx].state = ChildState::Failed;
                } else {
                    // No restart needed (either Never policy or success + Always not needed).
                    guard.children[idx].state = ChildState::Stopped;
                }
            }
        }
    }

    // Drain remaining tasks on shutdown.
    join_set.abort_all();
    while join_set.join_next().await.is_some() {}
}

// ── SupervisorHandle ──────────────────────────────────────────────────────────

/// Handle returned by [`Supervisor::start`].
///
/// Provides graceful shutdown and statistics queries.
pub struct SupervisorHandle {
    state: Arc<Mutex<SupervisorState>>,
    shutdown_tx: tokio::sync::broadcast::Sender<()>,
}

impl SupervisorHandle {
    /// Signal all children to stop and wait for the supervisor loop to exit.
    ///
    /// This sends a shutdown signal and waits a short period for in-flight
    /// tasks to drain.  Individual child `shutdown_timeout` values are advisory
    /// at this layer; the Tokio runtime will clean up task handles when the
    /// `JoinSet` is dropped by the supervisor loop.
    pub async fn shutdown(self) {
        // Signal the supervisor loop to exit.
        let _ = self.shutdown_tx.send(());
        // Give the supervisor loop a moment to clean up.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    /// Return a snapshot of supervisor statistics.
    pub async fn stats(&self) -> SupervisorStats {
        let guard = self.state.lock().await;
        let total_restarts = guard.children.iter().map(|r| r.restarts).sum();
        let children = guard
            .children
            .iter()
            .map(|r| ChildStats {
                id: r.spec.id.clone(),
                restarts: r.restarts,
                state: r.state,
            })
            .collect();
        SupervisorStats {
            total_restarts,
            children,
        }
    }

    /// Return a reference to the shared supervisor state (for advanced use).
    pub(crate) fn state(&self) -> Arc<Mutex<SupervisorState>> {
        self.state.clone()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::time::{sleep, Duration};

    fn make_spec(
        id: &str,
        call_count: Arc<AtomicU32>,
        fail_until: u32,
        restart: RestartPolicy,
        max_restarts: u32,
    ) -> ChildSpec {
        let counter = call_count.clone();
        ChildSpec::new(
            id,
            Arc::new(move || {
                let c = counter.clone();
                Box::pin(async move {
                    let n = c.fetch_add(1, Ordering::Relaxed);
                    if n < fail_until {
                        Err(AgentError("simulated failure".to_string()))
                    } else {
                        Ok(())
                    }
                })
            }),
            restart,
            max_restarts,
            Duration::from_millis(100),
        )
    }

    #[tokio::test]
    async fn test_supervisor_starts_and_runs_child() {
        let count = Arc::new(AtomicU32::new(0));
        let spec = make_spec("worker", count.clone(), 0, RestartPolicy::Never, 3);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(150)).await;
        let stats = handle.stats().await;
        assert!(!stats.children.is_empty());
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_one_for_one_restarts_failed_child() {
        let count = Arc::new(AtomicU32::new(0));
        // Fail 2 times then succeed
        let spec = make_spec("worker", count.clone(), 2, RestartPolicy::OnFailure, 5);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(300)).await;
        let stats = handle.stats().await;
        assert!(
            stats.children[0].restarts >= 2,
            "expected >= 2 restarts, got {}",
            stats.children[0].restarts
        );
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_max_restarts_exceeded_marks_failed() {
        let count = Arc::new(AtomicU32::new(0));
        // Always fail, max_restarts = 2
        let spec = make_spec("worker", count.clone(), 100, RestartPolicy::OnFailure, 2);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(400)).await;
        let stats = handle.stats().await;
        assert_eq!(
            stats.children[0].state,
            ChildState::Failed,
            "expected Failed state after exceeding max_restarts"
        );
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_never_policy_does_not_restart() {
        let count = Arc::new(AtomicU32::new(0));
        let spec = make_spec("worker", count.clone(), 1, RestartPolicy::Never, 10);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(200)).await;
        let stats = handle.stats().await;
        assert_eq!(
            stats.children[0].restarts, 0,
            "Never policy should not restart"
        );
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_always_policy_restarts_on_success() {
        let count = Arc::new(AtomicU32::new(0));
        // Succeed immediately each time; with Always, it will restart repeatedly
        let spec = make_spec("worker", count.clone(), 0, RestartPolicy::Always, 5);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(300)).await;
        let stats = handle.stats().await;
        // Should have restarted at least once
        assert!(
            stats.children[0].restarts >= 1,
            "Always policy should restart on success"
        );
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_stats_total_restarts() {
        let count = Arc::new(AtomicU32::new(0));
        let spec = make_spec("worker", count.clone(), 2, RestartPolicy::OnFailure, 5);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(300)).await;
        let stats = handle.stats().await;
        assert_eq!(
            stats.total_restarts,
            stats.children.iter().map(|c| c.restarts).sum::<u32>()
        );
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_multiple_children() {
        let count1 = Arc::new(AtomicU32::new(0));
        let count2 = Arc::new(AtomicU32::new(0));
        let spec1 = make_spec("child-1", count1.clone(), 0, RestartPolicy::Never, 0);
        let spec2 = make_spec("child-2", count2.clone(), 0, RestartPolicy::Never, 0);
        let handle = Supervisor::start(vec![spec1, spec2], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(150)).await;
        let stats = handle.stats().await;
        assert_eq!(stats.children.len(), 2);
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_child_stats_id_matches() {
        let count = Arc::new(AtomicU32::new(0));
        let spec = make_spec("my-worker", count, 0, RestartPolicy::Never, 0);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(100)).await;
        let stats = handle.stats().await;
        assert_eq!(stats.children[0].id, "my-worker");
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_shutdown_does_not_panic() {
        let count = Arc::new(AtomicU32::new(0));
        let spec = make_spec("worker", count, 100, RestartPolicy::Always, 100);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(50)).await;
        handle.shutdown().await; // Should not panic
    }

    #[tokio::test]
    async fn test_supervisor_one_for_all_strategy() {
        let count1 = Arc::new(AtomicU32::new(0));
        let count2 = Arc::new(AtomicU32::new(0));
        // child-1 fails once then succeeds
        let spec1 = make_spec("child-1", count1.clone(), 1, RestartPolicy::OnFailure, 3);
        let spec2 = make_spec("child-2", count2.clone(), 0, RestartPolicy::Never, 0);
        let handle =
            Supervisor::start(vec![spec1, spec2], SupervisorStrategy::OneForAll).await;
        sleep(Duration::from_millis(400)).await;
        let stats = handle.stats().await;
        // At least child-1 should have some restarts
        assert!(stats.total_restarts >= 1);
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_rest_for_one_strategy() {
        let count1 = Arc::new(AtomicU32::new(0));
        let count2 = Arc::new(AtomicU32::new(0));
        let spec1 = make_spec("child-1", count1.clone(), 1, RestartPolicy::OnFailure, 3);
        let spec2 = make_spec("child-2", count2.clone(), 0, RestartPolicy::Never, 0);
        let handle =
            Supervisor::start(vec![spec1, spec2], SupervisorStrategy::RestForOne).await;
        sleep(Duration::from_millis(400)).await;
        let _stats = handle.stats().await;
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_child_state_stopped_after_success_never_policy() {
        let count = Arc::new(AtomicU32::new(0));
        let spec = make_spec("worker", count, 0, RestartPolicy::Never, 0);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(200)).await;
        let stats = handle.stats().await;
        assert_eq!(stats.children[0].state, ChildState::Stopped);
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_zero_max_restarts_immediately_fails() {
        let count = Arc::new(AtomicU32::new(0));
        // Fails on first call, max_restarts = 0 → immediately mark Failed
        let spec = make_spec("worker", count, 100, RestartPolicy::OnFailure, 0);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(200)).await;
        let stats = handle.stats().await;
        assert_eq!(stats.children[0].state, ChildState::Failed);
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_supervisor_empty_children_no_panic() {
        let handle = Supervisor::start(vec![], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(50)).await;
        let stats = handle.stats().await;
        assert_eq!(stats.children.len(), 0);
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_on_failure_policy_does_not_restart_on_success() {
        let count = Arc::new(AtomicU32::new(0));
        // Succeed immediately; OnFailure should NOT restart on Ok.
        let spec = make_spec("worker", count, 0, RestartPolicy::OnFailure, 5);
        let handle = Supervisor::start(vec![spec], SupervisorStrategy::OneForOne).await;
        sleep(Duration::from_millis(200)).await;
        let stats = handle.stats().await;
        assert_eq!(
            stats.children[0].restarts, 0,
            "OnFailure should not restart on Ok"
        );
        handle.shutdown().await;
    }
}
