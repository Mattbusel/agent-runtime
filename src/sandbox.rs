//! # Tool Capability Sandbox
//!
//! Provides a capability-based security model for the tool registry.
//!
//! ## Problem
//!
//! A ReAct agent can call any registered tool with any arguments.  Without
//! sandboxing, a compromised prompt (or a misbehaving agent) can invoke
//! destructive tools (delete files, make network requests, write to databases)
//! without any policy enforcement.
//!
//! ## Solution: Capability Tokens
//!
//! Each tool declares the [`Capability`]s it requires.  Before executing a
//! tool call the [`Sandbox`] checks that the caller holds all required
//! capability tokens.  If the check fails the tool is never invoked — the
//! agent receives a structured denial observation instead.
//!
//! This follows the principle of *least privilege*: grant only the capabilities
//! the agent needs for the current task, not blanket access to every tool.
//!
//! ## Capabilities
//!
//! | Capability | What it guards |
//! |------------|---------------|
//! | `FileRead` | Reading files from disk |
//! | `FileWrite` | Writing or deleting files |
//! | `Network` | Making outbound HTTP/TCP requests |
//! | `DatabaseRead` | Querying any database |
//! | `DatabaseWrite` | Mutating any database |
//! | `ShellExec` | Running shell commands |
//! | `SecretAccess` | Reading secrets / API keys from the environment |
//! | `Custom(String)` | User-defined capability for domain-specific tools |
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::sandbox::{Sandbox, SandboxConfig, ToolPermission, ToolManifest};
//!
//! // Tools declare what they need.
//! let read_tool = ToolManifest::new("read_file").require(ToolPermission::FileRead);
//! let write_tool = ToolManifest::new("write_file")
//!     .require(ToolPermission::FileRead)
//!     .require(ToolPermission::FileWrite);
//!
//! // Grant the agent only read access for this session.
//! let sandbox = Sandbox::new(SandboxConfig::default())
//!     .grant(ToolPermission::FileRead);
//!
//! sandbox.register_tool(read_tool);
//! sandbox.register_tool(write_tool);
//!
//! // Check before tool invocation.
//! assert!(sandbox.check("read_file").is_ok());
//! assert!(sandbox.check("write_file").is_err());   // missing FileWrite
//! assert!(sandbox.check("unknown_tool").is_err()); // tool not registered
//!
//! // Inspect what was denied.
//! let log = sandbox.audit_log();
//! assert_eq!(log.len(), 2);
//! ```

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Capability
// ---------------------------------------------------------------------------

/// A named permission that a tool requires to execute.
///
/// Capabilities are cheap to clone — `Custom` carries an `Arc<str>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ToolPermission {
    /// Read files from the local filesystem.
    FileRead,
    /// Write, create, or delete files on the local filesystem.
    FileWrite,
    /// Make outbound HTTP/TCP network requests.
    Network,
    /// Issue read-only database queries.
    DatabaseRead,
    /// Issue write / mutate database operations.
    DatabaseWrite,
    /// Execute shell commands or subprocesses.
    ShellExec,
    /// Read secrets or API keys from environment variables or secret stores.
    SecretAccess,
    /// A caller-defined capability for domain-specific tools.
    Custom(Arc<str>),
}

impl ToolPermission {
    /// Create a [`ToolPermission::Custom`] from any string-like value.
    pub fn custom(name: impl AsRef<str>) -> Self {
        ToolPermission::Custom(Arc::from(name.as_ref()))
    }

    /// Human-readable label used in audit log entries.
    pub fn label(&self) -> String {
        match self {
            ToolPermission::FileRead => "file_read".to_owned(),
            ToolPermission::FileWrite => "file_write".to_owned(),
            ToolPermission::Network => "network".to_owned(),
            ToolPermission::DatabaseRead => "database_read".to_owned(),
            ToolPermission::DatabaseWrite => "database_write".to_owned(),
            ToolPermission::ShellExec => "shell_exec".to_owned(),
            ToolPermission::SecretAccess => "secret_access".to_owned(),
            ToolPermission::Custom(s) => format!("custom:{s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolManifest
// ---------------------------------------------------------------------------

/// Declares the name and required capabilities of a tool.
///
/// Registered with [`Sandbox::register_tool`] before any tool calls are made.
#[derive(Debug, Clone)]
pub struct ToolManifest {
    /// The tool name, as it appears in agent `Action` text.
    pub name: String,
    /// Set of capabilities this tool requires.
    pub required: HashSet<ToolPermission>,
    /// Optional human-readable description for audit and introspection.
    pub description: String,
}

impl ToolManifest {
    /// Create a manifest with no required capabilities.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required: HashSet::new(),
            description: String::new(),
        }
    }

    /// Add a required capability.  Returns `self` for chaining.
    pub fn require(mut self, cap: ToolPermission) -> Self {
        self.required.insert(cap);
        self
    }

    /// Add a description.  Returns `self` for chaining.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

// ---------------------------------------------------------------------------
// Audit entry
// ---------------------------------------------------------------------------

/// A single entry in the sandbox audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Wall-clock time of the check.
    pub timestamp: Instant,
    /// Tool that was checked.
    pub tool_name: String,
    /// Whether the check passed.
    pub allowed: bool,
    /// The capabilities that caused the denial (empty when `allowed`).
    pub denied_capabilities: Vec<String>,
}

// ---------------------------------------------------------------------------
// SandboxConfig
// ---------------------------------------------------------------------------

/// Configuration for [`Sandbox`].
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum number of audit entries to retain.  Oldest entries are evicted
    /// when the log is full.  Defaults to `1_000`.
    pub max_audit_entries: usize,

    /// When `true`, unregistered tools are denied by default.  When `false`,
    /// unregistered tools are allowed without capability checks (permissive
    /// mode, useful for testing).  Defaults to `true` (deny unknown).
    pub deny_unknown_tools: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_audit_entries: 1_000,
            deny_unknown_tools: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

struct Inner {
    /// Granted capability tokens for this sandbox instance.
    grants: HashSet<ToolPermission>,
    /// Registered tool manifests keyed by tool name.
    tools: HashMap<String, ToolManifest>,
    /// Bounded audit log.
    audit: Vec<AuditEntry>,
}

/// Enforces capability-based access control for tool invocations.
///
/// See the [module documentation][self] for a full usage example.
#[derive(Clone)]
pub struct Sandbox {
    config: SandboxConfig,
    inner: Arc<RwLock<Inner>>,
    // Metrics
    checks_total: Arc<AtomicU64>,
    denials_total: Arc<AtomicU64>,
}

impl Sandbox {
    /// Create a new `Sandbox` with no granted capabilities and no registered tools.
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            inner: Arc::new(RwLock::new(Inner {
                grants: HashSet::new(),
                tools: HashMap::new(),
                audit: Vec::new(),
            })),
            checks_total: Arc::new(AtomicU64::new(0)),
            denials_total: Arc::new(AtomicU64::new(0)),
        }
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /// Grant a capability token.  Returns `self` for chaining.
    pub fn grant(self, cap: ToolPermission) -> Self {
        self.inner.write().grants.insert(cap);
        self
    }

    /// Grant multiple capability tokens at once.
    pub fn grant_all(&self, caps: impl IntoIterator<Item = ToolPermission>) {
        let mut inner = self.inner.write();
        for cap in caps {
            inner.grants.insert(cap);
        }
    }

    /// Revoke a previously granted capability.
    pub fn revoke(&self, cap: &ToolPermission) {
        self.inner.write().grants.remove(cap);
    }

    /// Register a tool manifest.  Overwrites any existing manifest with the
    /// same name.
    pub fn register_tool(&self, manifest: ToolManifest) {
        self.inner.write().tools.insert(manifest.name.clone(), manifest);
    }

    // ------------------------------------------------------------------
    // Checking
    // ------------------------------------------------------------------

    /// Check whether `tool_name` may be invoked under current grants.
    ///
    /// Returns `Ok(())` if the tool is allowed, or `Err(DenialReason)` if it
    /// should be blocked.  The check is always recorded in the audit log.
    pub fn check(&self, tool_name: &str) -> Result<(), DenialReason> {
        self.checks_total.fetch_add(1, Ordering::Relaxed);

        let (result, entry) = {
            let inner = self.inner.read();

            match inner.tools.get(tool_name) {
                None if self.config.deny_unknown_tools => {
                    let entry = AuditEntry {
                        timestamp: Instant::now(),
                        tool_name: tool_name.to_owned(),
                        allowed: false,
                        denied_capabilities: vec!["[tool not registered]".into()],
                    };
                    (Err(DenialReason::UnknownTool(tool_name.to_owned())), entry)
                }
                None => {
                    let entry = AuditEntry {
                        timestamp: Instant::now(),
                        tool_name: tool_name.to_owned(),
                        allowed: true,
                        denied_capabilities: vec![],
                    };
                    (Ok(()), entry)
                }
                Some(manifest) => {
                    let missing: Vec<ToolPermission> = manifest
                        .required
                        .iter()
                        .filter(|cap| !inner.grants.contains(cap))
                        .cloned()
                        .collect();

                    if missing.is_empty() {
                        let entry = AuditEntry {
                            timestamp: Instant::now(),
                            tool_name: tool_name.to_owned(),
                            allowed: true,
                            denied_capabilities: vec![],
                        };
                        (Ok(()), entry)
                    } else {
                        let denied_labels: Vec<String> =
                            missing.iter().map(|c| c.label()).collect();
                        let entry = AuditEntry {
                            timestamp: Instant::now(),
                            tool_name: tool_name.to_owned(),
                            allowed: false,
                            denied_capabilities: denied_labels.clone(),
                        };
                        (
                            Err(DenialReason::MissingCapabilities {
                                tool: tool_name.to_owned(),
                                missing: denied_labels,
                            }),
                            entry,
                        )
                    }
                }
            }
        };

        // Record in audit log (write lock, brief).
        {
            let mut inner = self.inner.write();
            if inner.audit.len() >= self.config.max_audit_entries {
                inner.audit.remove(0);
            }
            inner.audit.push(entry);
        }

        if result.is_err() {
            self.denials_total.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    /// Return a snapshot of the audit log (oldest first).
    pub fn audit_log(&self) -> Vec<AuditEntry> {
        self.inner.read().audit.clone()
    }

    /// Return the set of currently granted capabilities.
    pub fn granted_capabilities(&self) -> Vec<ToolPermission> {
        self.inner.read().grants.iter().cloned().collect()
    }

    /// Return the names of all registered tools.
    pub fn registered_tools(&self) -> Vec<String> {
        self.inner.read().tools.keys().cloned().collect()
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> SandboxStats {
        SandboxStats {
            checks_total: self.checks_total.load(Ordering::Relaxed),
            denials_total: self.denials_total.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Reason a tool call was denied.
#[derive(Debug, Clone, PartialEq)]
pub enum DenialReason {
    /// The tool name was not registered with the sandbox.
    UnknownTool(String),
    /// The tool is registered but requires capabilities not in the grant set.
    MissingCapabilities {
        /// The tool that was blocked.
        tool: String,
        /// Human-readable labels for the missing capabilities.
        missing: Vec<String>,
    },
}

impl std::fmt::Display for DenialReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenialReason::UnknownTool(t) => write!(f, "unknown tool: {t}"),
            DenialReason::MissingCapabilities { tool, missing } => {
                write!(f, "tool '{tool}' denied: missing capabilities: {}", missing.join(", "))
            }
        }
    }
}

impl std::error::Error for DenialReason {}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`Sandbox`].
#[derive(Debug, Clone)]
pub struct SandboxStats {
    /// Total capability checks performed.
    pub checks_total: u64,
    /// Total checks that resulted in denial.
    pub denials_total: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox_with_read() -> Sandbox {
        Sandbox::new(SandboxConfig::default())
            .grant(ToolPermission::FileRead)
    }

    #[test]
    fn allow_tool_with_satisfied_caps() {
        let sb = sandbox_with_read();
        sb.register_tool(ToolManifest::new("read_file").require(ToolPermission::FileRead));
        assert!(sb.check("read_file").is_ok());
    }

    #[test]
    fn deny_tool_with_missing_caps() {
        let sb = sandbox_with_read();
        sb.register_tool(
            ToolManifest::new("write_file")
                .require(ToolPermission::FileRead)
                .require(ToolPermission::FileWrite),
        );
        let err = sb.check("write_file").unwrap_err();
        match err {
            DenialReason::MissingCapabilities { missing, .. } => {
                assert!(missing.contains(&"file_write".to_owned()));
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn deny_unknown_tool_when_configured() {
        let sb = Sandbox::new(SandboxConfig { deny_unknown_tools: true, ..Default::default() });
        assert!(matches!(sb.check("ghost_tool"), Err(DenialReason::UnknownTool(_))));
    }

    #[test]
    fn allow_unknown_tool_in_permissive_mode() {
        let sb = Sandbox::new(SandboxConfig { deny_unknown_tools: false, ..Default::default() });
        assert!(sb.check("anything").is_ok());
    }

    #[test]
    fn revoke_capability_blocks_tool() {
        let sb = sandbox_with_read();
        sb.register_tool(ToolManifest::new("reader").require(ToolPermission::FileRead));
        assert!(sb.check("reader").is_ok());
        sb.revoke(&ToolPermission::FileRead);
        assert!(sb.check("reader").is_err());
    }

    #[test]
    fn audit_log_records_both_allows_and_denials() {
        let sb = sandbox_with_read();
        sb.register_tool(ToolManifest::new("r").require(ToolPermission::FileRead));
        sb.register_tool(ToolManifest::new("w").require(ToolPermission::FileWrite));
        sb.check("r").ok();
        sb.check("w").ok();
        let log = sb.audit_log();
        assert_eq!(log.len(), 2);
        assert!(log[0].allowed);
        assert!(!log[1].allowed);
    }

    #[test]
    fn audit_log_bounded_by_config() {
        let sb = Sandbox::new(SandboxConfig {
            max_audit_entries: 3,
            deny_unknown_tools: false,
        });
        for i in 0..5 {
            sb.check(&format!("tool{i}")).ok();
        }
        assert_eq!(sb.audit_log().len(), 3);
    }

    #[test]
    fn stats_track_checks_and_denials() {
        let sb = sandbox_with_read();
        sb.register_tool(ToolManifest::new("r").require(ToolPermission::FileRead));
        sb.register_tool(ToolManifest::new("w").require(ToolPermission::FileWrite));
        sb.check("r").ok();
        sb.check("w").ok();
        let s = sb.stats();
        assert_eq!(s.checks_total, 2);
        assert_eq!(s.denials_total, 1);
    }

    #[test]
    fn custom_capability_works() {
        let cap = ToolPermission::custom("billing:charge");
        let sb = Sandbox::new(SandboxConfig::default()).grant(cap.clone());
        sb.register_tool(ToolManifest::new("charge").require(cap));
        assert!(sb.check("charge").is_ok());
    }

    #[test]
    fn no_required_caps_always_allowed() {
        let sb = Sandbox::new(SandboxConfig::default()); // no grants
        sb.register_tool(ToolManifest::new("echo"));
        assert!(sb.check("echo").is_ok());
    }

    #[test]
    fn grant_all_grants_multiple() {
        let sb = Sandbox::new(SandboxConfig::default());
        sb.grant_all([ToolPermission::Network, ToolPermission::DatabaseRead]);
        sb.register_tool(
            ToolManifest::new("fetch_db")
                .require(ToolPermission::Network)
                .require(ToolPermission::DatabaseRead),
        );
        assert!(sb.check("fetch_db").is_ok());
    }
}
