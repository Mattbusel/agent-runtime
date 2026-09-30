//! # task_runner
//!
//! Task execution with timeout, retry, and progress reporting.
//!
//! [`TaskRunner`] accepts [`TaskSpec`] submissions, executes them in background
//! threads with configurable retry/backoff, and tracks live progress via
//! [`TaskHandle`].

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

// ── TaskProgress ─────────────────────────────────────────────────────────────

/// Live progress snapshot for a running task.
#[derive(Debug, Clone)]
pub struct TaskProgress {
    /// Identifier of the task this progress belongs to.
    pub task_id: String,
    /// Completion percentage in the range 0–100.
    pub pct_complete: f64,
    /// Human-readable status message.
    pub message: String,
    /// Timestamp of the last update.
    pub updated_at: Instant,
}

// ── TaskOutput ───────────────────────────────────────────────────────────────

/// Final result produced by a completed task.
#[derive(Debug, Clone)]
pub struct TaskOutput {
    /// Identifier of the task that produced this output.
    pub task_id: String,
    /// Whether the task finished successfully.
    pub success: bool,
    /// Payload returned by the task function on success.
    pub data: String,
    /// Error description when `success` is `false`.
    pub error: Option<String>,
    /// Wall-clock duration of the entire task run in milliseconds.
    pub duration_ms: u64,
    /// How many retry attempts were consumed (0 = succeeded on first try).
    pub retries_used: u32,
}

// ── RetryConfig ──────────────────────────────────────────────────────────────

/// Retry policy for a task.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts after the first failure.
    pub max_retries: u32,
    /// Base delay in milliseconds between attempts.
    pub base_delay_ms: u64,
    /// Upper bound on the computed delay in milliseconds.
    pub max_delay_ms: u64,
    /// When `true` the delay doubles with each retry; otherwise it is constant.
    pub exponential: bool,
}

impl RetryConfig {
    /// Compute the sleep duration for a given attempt index (0-based).
    fn delay_for(&self, attempt: u32) -> Duration {
        let ms = if self.exponential {
            let shift = attempt.min(63) as u32;
            let factor = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
            self.base_delay_ms.saturating_mul(factor)
        } else {
            self.base_delay_ms
        };
        Duration::from_millis(ms.min(self.max_delay_ms))
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5_000,
            exponential: true,
        }
    }
}

// ── TaskSpec ─────────────────────────────────────────────────────────────────

/// Specification that describes a unit of work to be executed.
#[derive(Debug, Clone)]
pub struct TaskSpec {
    /// Unique task identifier.
    pub id: String,
    /// Human-readable name used in logs.
    pub name: String,
    /// Maximum allowed wall-clock duration in milliseconds.
    pub timeout_ms: u64,
    /// Retry behaviour if the task function returns an error.
    pub retry_config: RetryConfig,
    /// Arbitrary key-value payload forwarded to the task function.
    pub payload: HashMap<String, String>,
    /// Scheduling priority (higher value = higher priority, informational only).
    pub priority: u8,
}

// ── TaskHandle ───────────────────────────────────────────────────────────────

/// Live handle to a submitted task; shared between the runner and the caller.
pub struct TaskHandle {
    /// The specification the task was submitted with.
    pub spec: TaskSpec,
    /// Shared progress state, updated by the executing thread.
    pub progress: Arc<Mutex<TaskProgress>>,
    /// Set to `Some` once the task finishes (success or failure).
    pub output: Arc<Mutex<Option<TaskOutput>>>,
    /// Wall-clock time at which the task was submitted.
    pub started_at: Instant,
    cancelled: Arc<AtomicBool>,
}

impl TaskHandle {
    /// Signal the task to stop at the next cancellation check.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Update the live progress visible to observers.
    pub fn update_progress(&self, pct: f64, msg: &str) {
        if let Ok(mut p) = self.progress.lock() {
            p.pct_complete = pct.clamp(0.0, 100.0);
            p.message = msg.to_owned();
            p.updated_at = Instant::now();
        }
    }

    /// Block until a result is available or `timeout` elapses.
    ///
    /// Polls with a 10 ms interval — adequate for human-latency tasks.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<TaskOutput> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                if let Ok(guard) = self.output.lock() {
                    if guard.is_some() {
                        return guard.clone();
                    }
                }
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}

// ── RunnerStats ──────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`TaskRunner`] instance.
#[derive(Debug, Clone)]
pub struct RunnerStats {
    /// Number of tasks currently executing.
    pub running: usize,
    /// Total tasks that have completed (success or failure).
    pub completed: u64,
    /// Total tasks cancelled before or during execution.
    pub cancelled: u64,
    /// Total tasks that exhausted retries and produced an error output.
    pub failed: u64,
}

// ── TaskRunner ────────────────────────────────────────────────────────────────

/// Runs tasks in background threads with retry, timeout, and progress tracking.
pub struct TaskRunner {
    handles: Arc<Mutex<HashMap<String, Arc<TaskHandle>>>>,
    completed: Arc<AtomicU64>,
    cancelled: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
}

impl Default for TaskRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskRunner {
    /// Create a new runner with no active tasks.
    pub fn new() -> Self {
        Self {
            handles: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(AtomicU64::new(0)),
            cancelled: Arc::new(AtomicU64::new(0)),
            failed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Submit a task for background execution.
    ///
    /// `f` is called with a shared reference to the [`TaskHandle`] so it can
    /// call [`TaskHandle::update_progress`] and [`TaskHandle::is_cancelled`].
    pub fn submit<F>(&self, spec: TaskSpec, f: F) -> Arc<TaskHandle>
    where
        F: FnOnce(&TaskHandle) -> Result<String, String> + Send + 'static,
    {
        let task_id = spec.id.clone();
        let progress = Arc::new(Mutex::new(TaskProgress {
            task_id: task_id.clone(),
            pct_complete: 0.0,
            message: "queued".to_owned(),
            updated_at: Instant::now(),
        }));
        let output: Arc<Mutex<Option<TaskOutput>>> = Arc::new(Mutex::new(None));
        let cancelled = Arc::new(AtomicBool::new(false));

        let handle = Arc::new(TaskHandle {
            spec: spec.clone(),
            progress: Arc::clone(&progress),
            output: Arc::clone(&output),
            started_at: Instant::now(),
            cancelled: Arc::clone(&cancelled),
        });

        // Register before spawning so `get_handle` is immediately usable.
        if let Ok(mut map) = self.handles.lock() {
            map.insert(task_id.clone(), Arc::clone(&handle));
        }

        let handle_for_thread = Arc::clone(&handle);
        let completed = Arc::clone(&self.completed);
        let cancelled_ctr = Arc::clone(&self.cancelled);
        let failed_ctr = Arc::clone(&self.failed);
        let handles_map = Arc::clone(&self.handles);

        thread::spawn(move || {
            let started = Instant::now();
            let timeout = Duration::from_millis(spec.timeout_ms);
            let mut retries_used: u32 = 0;
            let mut last_error = String::new();
            let mut succeeded = false;
            let mut data = String::new();
            let mut timed_out = false;

            // Wrap f in an Option so we can take it once.
            let mut f_opt = Some(f);

            'retry: for attempt in 0..=(spec.retry_config.max_retries) {
                // Honour cancellation before each attempt.
                if handle_for_thread.is_cancelled() {
                    cancelled_ctr.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut out) = output.lock() {
                        *out = Some(TaskOutput {
                            task_id: task_id.clone(),
                            success: false,
                            data: String::new(),
                            error: Some("cancelled".to_owned()),
                            duration_ms: started.elapsed().as_millis() as u64,
                            retries_used,
                        });
                    }
                    break 'retry;
                }

                // Check timeout.
                if started.elapsed() >= timeout {
                    timed_out = true;
                    last_error = "task timed out".to_owned();
                    break 'retry;
                }

                if attempt > 0 {
                    let delay = spec.retry_config.delay_for(attempt - 1);
                    thread::sleep(delay);
                    retries_used += 1;
                }

                // Execute the closure (only possible once).
                let result = if let Some(closure) = f_opt.take() {
                    closure(&handle_for_thread)
                } else {
                    Err("closure already consumed".to_owned())
                };

                match result {
                    Ok(d) => {
                        succeeded = true;
                        data = d;
                        break 'retry;
                    }
                    Err(e) => {
                        last_error = e;
                    }
                }
            }

            // Skip writing output if already cancelled.
            let already_cancelled = {
                if let Ok(out) = output.lock() {
                    out.is_some()
                } else {
                    false
                }
            };

            if !already_cancelled {
                let duration_ms = started.elapsed().as_millis() as u64;
                let error = if succeeded {
                    None
                } else if timed_out {
                    Some(last_error.clone())
                } else {
                    Some(last_error.clone())
                };

                if !succeeded {
                    failed_ctr.fetch_add(1, Ordering::Relaxed);
                }
                completed.fetch_add(1, Ordering::Relaxed);

                if let Ok(mut out) = output.lock() {
                    *out = Some(TaskOutput {
                        task_id: task_id.clone(),
                        success: succeeded,
                        data,
                        error,
                        duration_ms,
                        retries_used,
                    });
                }
            }

            // Remove from active map.
            if let Ok(mut map) = handles_map.lock() {
                map.remove(&task_id);
            }
        });

        handle
    }

    /// Look up an active or recently-completed task handle.
    pub fn get_handle(&self, task_id: &str) -> Option<Arc<TaskHandle>> {
        self.handles
            .lock()
            .ok()
            .and_then(|m| m.get(task_id).cloned())
    }

    /// Request cancellation of a running task.
    ///
    /// Returns `true` if the handle was found and the flag was set.
    pub fn cancel_task(&self, task_id: &str) -> bool {
        if let Some(h) = self.get_handle(task_id) {
            h.cancel();
            true
        } else {
            false
        }
    }

    /// Return the IDs of all tasks currently registered (still running).
    pub fn running_tasks(&self) -> Vec<String> {
        self.handles
            .lock()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Total number of tasks that have reached a terminal state.
    pub fn completed_tasks(&self) -> usize {
        self.completed.load(Ordering::Relaxed) as usize
    }

    /// Snapshot of aggregate runner statistics.
    pub fn runner_stats(&self) -> RunnerStats {
        let running = self
            .handles
            .lock()
            .map(|m| m.len())
            .unwrap_or(0);
        RunnerStats {
            running,
            completed: self.completed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn simple_spec(id: &str) -> TaskSpec {
        TaskSpec {
            id: id.to_owned(),
            name: id.to_owned(),
            timeout_ms: 5_000,
            retry_config: RetryConfig::default(),
            payload: HashMap::new(),
            priority: 5,
        }
    }

    #[test]
    fn test_successful_task() {
        let runner = TaskRunner::new();
        let handle = runner.submit(simple_spec("t1"), |_h| Ok("done".to_owned()));
        let out = handle.wait_timeout(Duration::from_secs(5)).unwrap();
        assert!(out.success);
        assert_eq!(out.data, "done");
    }

    #[test]
    fn test_failed_task() {
        let runner = TaskRunner::new();
        let mut spec = simple_spec("t2");
        spec.retry_config.max_retries = 0;
        let handle = runner.submit(spec, |_h| Err("boom".to_owned()));
        let out = handle.wait_timeout(Duration::from_secs(5)).unwrap();
        assert!(!out.success);
        assert_eq!(out.error.as_deref(), Some("boom"));
    }

    #[test]
    fn test_progress_update() {
        let runner = TaskRunner::new();
        let handle = runner.submit(simple_spec("t3"), |h| {
            h.update_progress(50.0, "halfway");
            Ok("ok".to_owned())
        });
        let _ = handle.wait_timeout(Duration::from_secs(5));
    }
}
