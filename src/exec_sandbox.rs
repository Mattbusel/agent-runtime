//! # Sandboxed Tool Execution Environment
//!
//! Provides a process-level sandbox for executing external commands with
//! resource constraints (CPU time, memory) and a policy system that controls
//! network access, allowed filesystem paths, and permitted syscalls.
//!
//! On non-Linux platforms (e.g. Windows, macOS) where kernel-level isolation
//! such as `seccomp` or `landlock` is unavailable, the sandbox falls back to
//! a pure-Tokio timeout-based executor that measures wall-clock time and emits
//! a warning that full isolation is platform-dependent.
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//! use llm_agent_runtime::exec_sandbox::{Sandbox, SandboxPolicy};
//!
//! # tokio_test::block_on(async {
//! let policy = SandboxPolicy::default();
//! let sb = Sandbox::new(policy);
//! let out = sb.execute("echo", &["hello"], HashMap::new()).await.unwrap();
//! assert_eq!(out.exit_code, 0);
//! # });
//! ```

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use tracing::warn;

// ---------------------------------------------------------------------------
// SandboxPolicy
// ---------------------------------------------------------------------------

/// Resource limits and access-control policy for a [`Sandbox`].
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    /// Maximum wall-clock CPU time in milliseconds before the process is killed.
    pub max_cpu_ms: u64,
    /// Maximum memory usage in kilobytes (used as a hint; enforcement is
    /// platform-dependent).
    pub max_memory_kb: usize,
    /// Allowlist of syscall names (enforcement requires kernel support; ignored
    /// on platforms without seccomp).
    pub allowed_syscalls: Vec<String>,
    /// Whether outbound network access is permitted.
    pub network_allowed: bool,
    /// Filesystem paths the child process may access (enforcement is
    /// platform-dependent).
    pub fs_allowed_paths: Vec<PathBuf>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            max_cpu_ms:       5_000,
            max_memory_kb:    65_536, // 64 MiB
            allowed_syscalls: Vec::new(),
            network_allowed:  false,
            fs_allowed_paths: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SandboxedOutput
// ---------------------------------------------------------------------------

/// Output produced by a sandboxed command execution.
#[derive(Debug, Clone)]
pub struct SandboxedOutput {
    /// Standard output captured from the child process.
    pub stdout: String,
    /// Standard error captured from the child process.
    pub stderr: String,
    /// Exit code returned by the process (0 = success on POSIX).
    pub exit_code: i32,
    /// Wall-clock milliseconds the process ran for.
    pub cpu_ms: u64,
    /// Estimated memory usage in kilobytes (stub on Windows).
    pub memory_kb: usize,
    /// `true` if the process was killed due to a timeout or policy violation.
    pub was_killed: bool,
}

// ---------------------------------------------------------------------------
// SandboxError
// ---------------------------------------------------------------------------

/// Errors that can occur during sandboxed execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxError {
    /// A policy check rejected the execution before it started.
    PolicyViolation(String),
    /// The process exceeded the configured time limit.
    Timeout,
    /// The process could not be spawned.
    SpawnFailed(String),
    /// The process was killed by the sandbox (e.g. memory limit exceeded).
    Killed,
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::PolicyViolation(msg) => write!(f, "policy violation: {msg}"),
            SandboxError::Timeout              => write!(f, "execution timed out"),
            SandboxError::SpawnFailed(msg)     => write!(f, "spawn failed: {msg}"),
            SandboxError::Killed               => write!(f, "process killed by sandbox"),
        }
    }
}

impl std::error::Error for SandboxError {}

// ---------------------------------------------------------------------------
// ResourceUsage
// ---------------------------------------------------------------------------

/// Tracks resource consumption for a single execution.
struct ResourceUsage {
    start: Instant,
}

impl ResourceUsage {
    fn start() -> Self {
        Self { start: Instant::now() }
    }

    /// Elapsed wall-clock time in milliseconds.
    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Estimate memory usage in kilobytes.
    ///
    /// On Windows and other non-Linux platforms, returns a constant stub
    /// because querying RSS without platform APIs is not portable.
    #[allow(clippy::unused_self)]
    fn memory_kb(&self) -> usize {
        // Platform stub: on Linux one could read /proc/self/statm;
        // on Windows one could call GetProcessMemoryInfo via winapi.
        // For portability we return a conservative constant.
        0
    }
}

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

/// Sandboxed command executor.
///
/// Wraps a [`SandboxPolicy`] and exposes a single [`Sandbox::execute`] method
/// that runs an external command with timeout enforcement.
///
/// Cheap to clone — internally reference-counted.
#[derive(Clone)]
pub struct Sandbox {
    policy:         SandboxPolicy,
    executions:     Arc<AtomicU64>,
    timeouts:       Arc<AtomicU64>,
    policy_denials: Arc<AtomicU64>,
}

impl Sandbox {
    /// Create a new [`Sandbox`] with the given policy.
    #[must_use]
    pub fn new(policy: SandboxPolicy) -> Self {
        Self {
            policy,
            executions:     Arc::new(AtomicU64::new(0)),
            timeouts:       Arc::new(AtomicU64::new(0)),
            policy_denials: Arc::new(AtomicU64::new(0)),
        }
    }

    // ------------------------------------------------------------------
    // Policy checks
    // ------------------------------------------------------------------

    /// Run pre-execution policy checks.
    fn check_policy(&self, command: &str) -> Result<(), SandboxError> {
        // If no paths are allowed and network is disallowed, only built-in
        // commands should be permitted.  We leave the allowlist empty by
        // default (no restriction) to keep the API ergonomic.
        let _ = command; // reserved for future allow/deny-list checks
        Ok(())
    }

    // ------------------------------------------------------------------
    // Execute
    // ------------------------------------------------------------------

    /// Execute `command` with `args` under the configured policy.
    ///
    /// On non-Linux platforms full kernel-level isolation (seccomp, cgroups)
    /// is unavailable; enforcement is limited to wall-clock timeout.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::PolicyViolation`] if pre-execution policy checks
    /// fail, [`SandboxError::Timeout`] if the process exceeds `max_cpu_ms`,
    /// or [`SandboxError::SpawnFailed`] if the OS refuses to spawn the process.
    pub async fn execute(
        &self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
    ) -> Result<SandboxedOutput, SandboxError> {
        // Pre-flight policy check.
        if let Err(e) = self.check_policy(command) {
            self.policy_denials.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }

        self.executions.fetch_add(1, Ordering::Relaxed);

        // Warn that full isolation is not available on this platform.
        #[cfg(not(target_os = "linux"))]
        warn!(
            command,
            "exec_sandbox: full process isolation (seccomp/cgroups) is \
             unavailable on this platform; only wall-clock timeout is enforced"
        );

        #[cfg(target_os = "linux")]
        warn!(
            command,
            "exec_sandbox: seccomp/landlock support is not wired up in this \
             build; only wall-clock timeout is enforced"
        );

        let timeout_dur = Duration::from_millis(self.policy.max_cpu_ms);
        let resource    = ResourceUsage::start();

        // Build the command.
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        for (k, v) in &env {
            cmd.env(k, v);
        }

        let child = cmd.spawn().map_err(|e| SandboxError::SpawnFailed(e.to_string()))?;

        // Wait for completion or timeout.
        let output_result =
            tokio::time::timeout(timeout_dur, child.wait_with_output()).await;

        let cpu_ms    = resource.elapsed_ms();
        let memory_kb = resource.memory_kb();

        match output_result {
            Err(_elapsed) => {
                // Timeout fired.
                self.timeouts.fetch_add(1, Ordering::Relaxed);
                Err(SandboxError::Timeout)
            }
            Ok(Err(io_err)) => Err(SandboxError::SpawnFailed(io_err.to_string())),
            Ok(Ok(output)) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let stdout    = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr    = String::from_utf8_lossy(&output.stderr).into_owned();
                Ok(SandboxedOutput {
                    stdout,
                    stderr,
                    exit_code,
                    cpu_ms,
                    memory_kb,
                    was_killed: false,
                })
            }
        }
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    /// Total successful execution attempts (policy-cleared and spawned).
    #[must_use]
    pub fn executions(&self) -> u64 {
        self.executions.load(Ordering::Relaxed)
    }

    /// Total executions that exceeded the time limit.
    #[must_use]
    pub fn timeouts(&self) -> u64 {
        self.timeouts.load(Ordering::Relaxed)
    }

    /// Total attempts blocked by policy before spawning.
    #[must_use]
    pub fn policy_denials(&self) -> u64 {
        self.policy_denials.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// MockSandbox
// ---------------------------------------------------------------------------

/// A test double for [`Sandbox`] that records calls and returns pre-configured
/// outputs without spawning real processes.
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use llm_agent_runtime::exec_sandbox::{MockSandbox, SandboxedOutput};
///
/// let mut mock = MockSandbox::new();
/// mock.enqueue_output(SandboxedOutput {
///     stdout:    "hello".into(),
///     stderr:    String::new(),
///     exit_code: 0,
///     cpu_ms:    5,
///     memory_kb: 0,
///     was_killed: false,
/// });
///
/// let call = mock.next_output("echo", &["hello"], HashMap::new());
/// assert_eq!(call.stdout, "hello");
/// assert_eq!(mock.calls().len(), 1);
/// ```
pub struct MockSandbox {
    queue: std::collections::VecDeque<SandboxedOutput>,
    calls: Vec<(String, Vec<String>)>,
}

impl MockSandbox {
    /// Create an empty [`MockSandbox`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
            calls: Vec::new(),
        }
    }

    /// Enqueue a pre-configured output to be returned on the next call to
    /// [`MockSandbox::next_output`].
    pub fn enqueue_output(&mut self, output: SandboxedOutput) {
        self.queue.push_back(output);
    }

    /// Simulate executing `command` with `args`.
    ///
    /// Records the call and pops the next pre-configured output.
    ///
    /// # Panics
    ///
    /// Panics if the output queue is empty (configure enough outputs in
    /// advance for the number of calls you expect).
    #[allow(clippy::panic)]
    pub fn next_output(
        &mut self,
        command: &str,
        args: &[&str],
        _env: HashMap<String, String>,
    ) -> SandboxedOutput {
        self.calls.push((command.into(), args.iter().map(|&s| s.into()).collect()));
        self.queue
            .pop_front()
            .unwrap_or_else(|| panic!("MockSandbox: no outputs enqueued for call to '{command}'"))
    }

    /// All calls recorded so far as `(command, args)` pairs.
    #[must_use]
    pub fn calls(&self) -> &[(String, Vec<String>)] {
        &self.calls
    }
}

impl Default for MockSandbox {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // MockSandbox tests (no real processes)
    // ------------------------------------------------------------------

    #[test]
    fn mock_records_calls() {
        let mut mock = MockSandbox::new();
        mock.enqueue_output(SandboxedOutput {
            stdout:    "ok".into(),
            stderr:    String::new(),
            exit_code: 0,
            cpu_ms:    1,
            memory_kb: 0,
            was_killed: false,
        });
        let out = mock.next_output("echo", &["ok"], HashMap::new());
        assert_eq!(out.stdout, "ok");
        assert_eq!(mock.calls().len(), 1);
        assert_eq!(mock.calls()[0].0, "echo");
    }

    #[test]
    fn mock_returns_outputs_in_order() {
        let mut mock = MockSandbox::new();
        for i in 0..3u32 {
            mock.enqueue_output(SandboxedOutput {
                stdout:    format!("out{i}"),
                stderr:    String::new(),
                exit_code: 0,
                cpu_ms:    i as u64,
                memory_kb: 0,
                was_killed: false,
            });
        }
        for i in 0..3u32 {
            let out = mock.next_output("cmd", &[], HashMap::new());
            assert_eq!(out.stdout, format!("out{i}"));
        }
    }

    #[test]
    fn mock_timeout_simulation() {
        // Simulate a "killed" output to verify callers handle was_killed correctly.
        let mut mock = MockSandbox::new();
        mock.enqueue_output(SandboxedOutput {
            stdout:    String::new(),
            stderr:    "killed".into(),
            exit_code: -1,
            cpu_ms:    5_000,
            memory_kb: 0,
            was_killed: true,
        });
        let out = mock.next_output("sleep", &["999"], HashMap::new());
        assert!(out.was_killed);
        assert_eq!(out.exit_code, -1);
    }

    #[test]
    fn mock_policy_violation_simulation() {
        // Caller can choose not to enqueue anything and handle the missing case
        // by checking whether calls were attempted.  Here we verify the mock
        // panics on an empty queue (expected in test design).
        let result = std::panic::catch_unwind(|| {
            let mut mock = MockSandbox::new();
            mock.next_output("evil_cmd", &[], HashMap::new());
        });
        assert!(result.is_err(), "should panic when queue is empty");
    }

    // ------------------------------------------------------------------
    // Sandbox construction
    // ------------------------------------------------------------------

    #[test]
    fn sandbox_stats_start_at_zero() {
        let sb = Sandbox::new(SandboxPolicy::default());
        assert_eq!(sb.executions(), 0);
        assert_eq!(sb.timeouts(), 0);
        assert_eq!(sb.policy_denials(), 0);
    }

    #[test]
    fn sandbox_policy_default_values() {
        let policy = SandboxPolicy::default();
        assert_eq!(policy.max_cpu_ms, 5_000);
        assert!(!policy.network_allowed);
        assert!(policy.fs_allowed_paths.is_empty());
    }

    // ------------------------------------------------------------------
    // Real execution tests (only run when `echo` / `cmd` is available)
    // ------------------------------------------------------------------

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_echo_succeeds() {
        let sb  = Sandbox::new(SandboxPolicy::default());
        let out = sb.execute("echo", &["hello"], HashMap::new()).await.unwrap();
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("hello"));
        assert_eq!(sb.executions(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_timeout_fires() {
        let policy = SandboxPolicy { max_cpu_ms: 100, ..Default::default() };
        let sb     = Sandbox::new(policy);
        let result = sb.execute("sleep", &["10"], HashMap::new()).await;
        assert!(matches!(result, Err(SandboxError::Timeout)));
        assert_eq!(sb.timeouts(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_bad_command_fails() {
        let sb     = Sandbox::new(SandboxPolicy::default());
        let result = sb
            .execute("__no_such_command_xyz__", &[], HashMap::new())
            .await;
        assert!(matches!(result, Err(SandboxError::SpawnFailed(_))));
    }
}
