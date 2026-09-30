//! # Agent Execution Sandbox
//!
//! Provides resource-limited execution environments for agent-run code.
//! Tracks memory, CPU time, output size, file reads, and network calls against
//! configurable limits, surfacing violations as typed errors.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};

// ── ResourceLimits ────────────────────────────────────────────────────────────

/// Upper bounds on resource consumption for a single [`Sandbox`].
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum resident memory in megabytes.
    pub max_memory_mb: u64,
    /// Maximum CPU wall-clock time in milliseconds.
    pub max_cpu_time_ms: u64,
    /// Maximum total bytes written to the output buffer.
    pub max_output_bytes: usize,
    /// Maximum number of filesystem read operations.
    pub max_file_reads: u32,
    /// Maximum number of outbound network calls.
    pub max_network_calls: u32,
    /// Allowlist of module names that may make network calls.
    pub allowed_modules: Vec<String>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 256,
            max_cpu_time_ms: 30_000,
            max_output_bytes: 1_048_576, // 1 MiB
            max_file_reads: 100,
            max_network_calls: 50,
            allowed_modules: Vec::new(),
        }
    }
}

// ── SandboxViolation ──────────────────────────────────────────────────────────

/// A resource-limit violation detected by the sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxViolation {
    /// Resident memory exceeded `max_memory_mb`.
    MemoryLimitExceeded,
    /// CPU wall-clock time exceeded `max_cpu_time_ms`.
    CpuTimeLimitExceeded,
    /// Output buffer exceeded `max_output_bytes`.
    OutputTooLarge,
    /// Filesystem read count exceeded `max_file_reads`.
    TooManyFileReads,
    /// Network call count exceeded `max_network_calls`.
    TooManyNetworkCalls,
    /// A network call was attempted by a module not in `allowed_modules`.
    ModuleNotAllowed(String),
}

impl std::fmt::Display for SandboxViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxViolation::MemoryLimitExceeded => write!(f, "memory limit exceeded"),
            SandboxViolation::CpuTimeLimitExceeded => write!(f, "CPU time limit exceeded"),
            SandboxViolation::OutputTooLarge => write!(f, "output size limit exceeded"),
            SandboxViolation::TooManyFileReads => write!(f, "file read limit exceeded"),
            SandboxViolation::TooManyNetworkCalls => write!(f, "network call limit exceeded"),
            SandboxViolation::ModuleNotAllowed(m) => {
                write!(f, "module not allowed: {m}")
            }
        }
    }
}

// ── SandboxState ──────────────────────────────────────────────────────────────

/// Live resource counters for a [`Sandbox`].
pub struct SandboxState {
    /// Current estimated resident memory in megabytes.
    pub memory_used_mb: AtomicU64,
    /// Elapsed CPU wall-clock time in milliseconds.
    pub cpu_time_ms: AtomicU64,
    /// Total bytes accumulated in the output buffer.
    pub output_bytes: AtomicUsize,
    /// Number of filesystem read operations performed.
    pub file_reads: AtomicU32,
    /// Number of outbound network calls performed.
    pub network_calls: AtomicU32,
}

impl SandboxState {
    fn new() -> Self {
        Self {
            memory_used_mb: AtomicU64::new(0),
            cpu_time_ms: AtomicU64::new(0),
            output_bytes: AtomicUsize::new(0),
            file_reads: AtomicU32::new(0),
            network_calls: AtomicU32::new(0),
        }
    }

    /// Check whether the current memory usage exceeds the limit.
    pub fn check_memory(&self, limits: &ResourceLimits) -> Result<(), SandboxViolation> {
        if self.memory_used_mb.load(Ordering::Relaxed) > limits.max_memory_mb {
            Err(SandboxViolation::MemoryLimitExceeded)
        } else {
            Ok(())
        }
    }

    /// Check whether the accumulated CPU time exceeds the limit.
    pub fn check_cpu(&self, limits: &ResourceLimits) -> Result<(), SandboxViolation> {
        if self.cpu_time_ms.load(Ordering::Relaxed) > limits.max_cpu_time_ms {
            Err(SandboxViolation::CpuTimeLimitExceeded)
        } else {
            Ok(())
        }
    }

    /// Check whether the output buffer size exceeds the limit.
    pub fn check_output(&self, limits: &ResourceLimits) -> Result<(), SandboxViolation> {
        if self.output_bytes.load(Ordering::Relaxed) > limits.max_output_bytes {
            Err(SandboxViolation::OutputTooLarge)
        } else {
            Ok(())
        }
    }

    fn reset(&self) {
        self.memory_used_mb.store(0, Ordering::Relaxed);
        self.cpu_time_ms.store(0, Ordering::Relaxed);
        self.output_bytes.store(0, Ordering::Relaxed);
        self.file_reads.store(0, Ordering::Relaxed);
        self.network_calls.store(0, Ordering::Relaxed);
    }
}

// ── ExecutionResult ───────────────────────────────────────────────────────────

/// Outcome of a [`Sandbox::execute`] call.
#[derive(Debug)]
pub struct ExecutionResult {
    /// Whether the function completed without raising a violation.
    pub success: bool,
    /// Text output accumulated during execution.
    pub output: String,
    /// Any resource violations detected during execution.
    pub violations: Vec<SandboxViolation>,
    /// Total CPU wall-clock time consumed in milliseconds.
    pub cpu_time_ms: u64,
    /// Peak resident memory in megabytes.
    pub peak_memory_mb: u64,
    /// Total output bytes written.
    pub output_bytes: usize,
}

// ── SandboxUsage ──────────────────────────────────────────────────────────────

/// Snapshot of sandbox resource usage expressed as percentages and raw counts.
#[derive(Debug, Clone)]
pub struct SandboxUsage {
    /// CPU time used as a fraction of `max_cpu_time_ms` (0.0 – 100.0).
    pub cpu_pct: f64,
    /// Memory used as a fraction of `max_memory_mb` (0.0 – 100.0).
    pub memory_pct: f64,
    /// Raw file-read count.
    pub file_reads: u32,
    /// Raw network-call count.
    pub network_calls: u32,
    /// Total number of violations recorded in the current execution.
    pub violations_count: usize,
}

// ── Sandbox ───────────────────────────────────────────────────────────────────

/// An isolated execution environment with configurable resource limits.
pub struct Sandbox {
    /// Unique identifier for this sandbox instance.
    pub id: String,
    /// Resource limits applied to every execution.
    pub limits: ResourceLimits,
    /// Shared live resource counters.
    pub state: Arc<SandboxState>,
    /// Wall-clock time when this sandbox was created.
    pub created_at: Instant,
    /// Accumulated output across all `record_output` calls.
    output_buf: Mutex<String>,
    /// Violations recorded during the last `execute` call.
    violations: Mutex<Vec<SandboxViolation>>,
}

impl Sandbox {
    /// Create a new `Sandbox` with the given `id` and resource `limits`.
    pub fn new(id: &str, limits: ResourceLimits) -> Self {
        Self {
            id: id.to_owned(),
            limits,
            state: Arc::new(SandboxState::new()),
            created_at: Instant::now(),
            output_buf: Mutex::new(String::new()),
            violations: Mutex::new(Vec::new()),
        }
    }

    /// Execute a closure inside the sandbox, measuring CPU time and checking
    /// resource limits before and after.
    ///
    /// The closure's return value is discarded; results should be communicated
    /// via [`record_output`](Self::record_output).
    pub fn execute<F, R>(&self, f: F) -> Result<ExecutionResult, SandboxViolation>
    where
        F: FnOnce() -> R,
    {
        // Clear state from any previous run.
        self.reset();

        let start = Instant::now();
        let _result = f();
        let elapsed_ms = start.elapsed().as_millis() as u64;

        self.state
            .cpu_time_ms
            .fetch_add(elapsed_ms, Ordering::Relaxed);

        // Collect violations.
        let mut violations = Vec::new();
        if let Err(v) = self.state.check_cpu(&self.limits) {
            violations.push(v);
        }
        if let Err(v) = self.state.check_memory(&self.limits) {
            violations.push(v);
        }
        if let Err(v) = self.state.check_output(&self.limits) {
            violations.push(v);
        }

        if let Ok(mut stored) = self.violations.lock() {
            stored.extend(violations.iter().cloned());
        }

        let output = self
            .output_buf
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();

        let success = violations.is_empty();
        let cpu_time_ms = self.state.cpu_time_ms.load(Ordering::Relaxed);
        let peak_memory_mb = self.state.memory_used_mb.load(Ordering::Relaxed);
        let output_bytes = self.state.output_bytes.load(Ordering::Relaxed);

        Ok(ExecutionResult {
            success,
            output,
            violations,
            cpu_time_ms,
            peak_memory_mb,
            output_bytes,
        })
    }

    /// Append `output` to the sandbox output buffer, checking the size limit.
    pub fn record_output(&self, output: &str) -> Result<(), SandboxViolation> {
        let new_bytes = self
            .state
            .output_bytes
            .fetch_add(output.len(), Ordering::Relaxed)
            + output.len();

        if new_bytes > self.limits.max_output_bytes {
            return Err(SandboxViolation::OutputTooLarge);
        }

        if let Ok(mut buf) = self.output_buf.lock() {
            buf.push_str(output);
        }
        Ok(())
    }

    /// Record a filesystem read operation.
    pub fn record_file_read(&self) -> Result<(), SandboxViolation> {
        let new_count = self.state.file_reads.fetch_add(1, Ordering::Relaxed) + 1;
        if new_count > self.limits.max_file_reads {
            Err(SandboxViolation::TooManyFileReads)
        } else {
            Ok(())
        }
    }

    /// Record an outbound network call from `module`.
    ///
    /// Fails if `allowed_modules` is non-empty and `module` is not listed, or
    /// if the network-call count would exceed `max_network_calls`.
    pub fn record_network_call(&self, module: &str) -> Result<(), SandboxViolation> {
        if !self.limits.allowed_modules.is_empty()
            && !self
                .limits
                .allowed_modules
                .iter()
                .any(|m| m == module)
        {
            return Err(SandboxViolation::ModuleNotAllowed(module.to_owned()));
        }

        let new_count = self
            .state
            .network_calls
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if new_count > self.limits.max_network_calls {
            Err(SandboxViolation::TooManyNetworkCalls)
        } else {
            Ok(())
        }
    }

    /// Return a snapshot of the current resource usage relative to limits.
    pub fn usage_report(&self) -> SandboxUsage {
        let violations_count = self
            .violations
            .lock()
            .map(|g| g.len())
            .unwrap_or(0);

        let cpu_pct = if self.limits.max_cpu_time_ms == 0 {
            0.0
        } else {
            (self.state.cpu_time_ms.load(Ordering::Relaxed) as f64
                / self.limits.max_cpu_time_ms as f64)
                * 100.0
        };

        let memory_pct = if self.limits.max_memory_mb == 0 {
            0.0
        } else {
            (self.state.memory_used_mb.load(Ordering::Relaxed) as f64
                / self.limits.max_memory_mb as f64)
                * 100.0
        };

        SandboxUsage {
            cpu_pct,
            memory_pct,
            file_reads: self.state.file_reads.load(Ordering::Relaxed),
            network_calls: self.state.network_calls.load(Ordering::Relaxed),
            violations_count,
        }
    }

    /// Reset all resource counters and the output buffer.
    pub fn reset(&self) {
        self.state.reset();
        if let Ok(mut buf) = self.output_buf.lock() {
            buf.clear();
        }
        if let Ok(mut v) = self.violations.lock() {
            v.clear();
        }
    }
}

// ── SandboxPool ───────────────────────────────────────────────────────────────

/// A pool of reusable [`Sandbox`] instances keyed by id.
pub struct SandboxPool {
    sandboxes: RwLock<HashMap<String, Arc<Sandbox>>>,
}

impl Default for SandboxPool {
    fn default() -> Self {
        Self::new()
    }
}

impl SandboxPool {
    /// Create an empty `SandboxPool`.
    pub fn new() -> Self {
        Self {
            sandboxes: RwLock::new(HashMap::new()),
        }
    }

    /// Acquire a sandbox by `id`.
    ///
    /// If no sandbox with that id exists it is created with the given `limits`.
    /// If one already exists the existing instance is returned (limits are
    /// *not* updated on re-acquisition).
    pub fn acquire(&self, id: &str, limits: ResourceLimits) -> Arc<Sandbox> {
        // Fast path: sandbox already exists.
        if let Ok(g) = self.sandboxes.read() {
            if let Some(sb) = g.get(id) {
                return Arc::clone(sb);
            }
        }

        // Slow path: create and insert.
        if let Ok(mut g) = self.sandboxes.write() {
            let sb = g
                .entry(id.to_owned())
                .or_insert_with(|| Arc::new(Sandbox::new(id, limits)));
            Arc::clone(sb)
        } else {
            // Fallback: return a temporary sandbox (not pooled).
            Arc::new(Sandbox::new(id, limits))
        }
    }

    /// Release a sandbox back to the pool (resets its state).
    pub fn release(&self, id: &str) {
        if let Ok(g) = self.sandboxes.read() {
            if let Some(sb) = g.get(id) {
                sb.reset();
            }
        }
    }

    /// Return the number of sandboxes currently in the pool.
    pub fn active_sandboxes(&self) -> usize {
        self.sandboxes
            .read()
            .map(|g| g.len())
            .unwrap_or(0)
    }
}

// Suppress "unused import" for the HashMap used in doc-example position.
#[allow(unused_imports)]
use std::collections::HashMap as _HashMap;
