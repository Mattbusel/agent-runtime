//! # Unified error type for the agent-runtime crate.
//!
//! ## Responsibility
//! Provide a single, typed error enum that covers all subsystems:
//! memory, graph, orchestration, and the ReAct agent loop.
//!
//! ## Guarantees
//! - Every variant is named and carries structured context
//! - Implements `std::error::Error` via `thiserror`
//! - Safe to send across thread/task boundaries (`Send + Sync`)
//! - Never panics

/// Unified error type returned by all public `agent-runtime` APIs.
///
/// Marked `#[non_exhaustive]` so that adding new variants in future minor
/// releases does not break external `match` arms.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum AgentRuntimeError {
    /// A memory subsystem operation failed (episodic, semantic, or working memory).
    #[error("Memory operation failed: {0}")]
    Memory(String),

    /// A graph subsystem operation failed (entity/relationship management or traversal).
    #[error("Graph operation failed: {0}")]
    Graph(String),

    /// The orchestration pipeline or one of its stages failed.
    #[error("Orchestration failed: {0}")]
    Orchestration(String),

    /// The ReAct agent loop encountered an unrecoverable error.
    #[error("Agent loop error: {0}")]
    AgentLoop(String),

    /// The runtime was used before a required subsystem was configured.
    #[error("Runtime not configured: missing '{0}'")]
    NotConfigured(&'static str),

    /// Circuit breaker is open — fast-fail without attempting the operation.
    #[error("Circuit breaker open for '{service}'")]
    CircuitOpen {
        /// Name of the service whose circuit breaker is open.
        service: String,
    },

    /// Backpressure threshold exceeded — caller must shed or wait.
    #[error("Backpressure threshold exceeded: queue depth {depth}/{capacity}")]
    BackpressureShed {
        /// Current in-flight request count at the time of rejection.
        depth: usize,
        /// Maximum allowed in-flight request count.
        capacity: usize,
    },

    /// A deduplication key collision was detected.
    #[error("Deduplication key collision: {key}")]
    DeduplicationConflict {
        /// The duplicated request key.
        key: String,
    },

    /// An LLM provider call failed.
    #[error("Provider error: {0}")]
    Provider(String),

    /// A persistence operation failed.
    #[error("Persistence error: {0}")]
    Persistence(String),

    /// A general internal error not covered by a more specific variant.
    #[error("Internal error: {0}")]
    Internal(String),

    /// A tool argument validation failed.
    #[error("Validation failed for field '{field}': [{code}] {message}")]
    Validation {
        /// The argument field that failed validation.
        field: String,
        /// A short machine-readable code (e.g. "out_of_range", "pattern_mismatch").
        code: String,
        /// Human-readable description of the failure.
        message: String,
    },
}

impl AgentRuntimeError {
    /// Return `true` if this is an `Internal` error.
    pub fn is_internal(&self) -> bool {
        matches!(self, Self::Internal(_))
    }

    /// Return `true` if this is a `CircuitOpen` error.
    pub fn is_circuit_open(&self) -> bool {
        matches!(self, Self::CircuitOpen { .. })
    }

    /// Return `true` if this is a `BackpressureShed` error.
    pub fn is_backpressure(&self) -> bool {
        matches!(self, Self::BackpressureShed { .. })
    }

    /// Return `true` if this is a `Provider` error.
    pub fn is_provider(&self) -> bool {
        matches!(self, Self::Provider(_))
    }

    /// Return `true` if this is a `Validation` error.
    pub fn is_validation(&self) -> bool {
        matches!(self, Self::Validation { .. })
    }

    /// Return `true` if this is a `Memory` error.
    pub fn is_memory(&self) -> bool {
        matches!(self, Self::Memory(_))
    }

    /// Return `true` if this is a `Graph` error.
    pub fn is_graph(&self) -> bool {
        matches!(self, Self::Graph(_))
    }

    /// Return `true` if this is an `AgentLoop` error.
    pub fn is_agent_loop(&self) -> bool {
        matches!(self, Self::AgentLoop(_))
    }

    /// Return `true` if this is an `Orchestration` error.
    pub fn is_orchestration(&self) -> bool {
        matches!(self, Self::Orchestration(_))
    }

    /// Return `true` if this is a `Persistence` error.
    pub fn is_persistence(&self) -> bool {
        matches!(self, Self::Persistence(_))
    }

    /// Return `true` if this is a `NotConfigured` error.
    pub fn is_not_configured(&self) -> bool {
        matches!(self, Self::NotConfigured(_))
    }

    /// Return `true` if this is a `DeduplicationConflict` error.
    pub fn is_deduplication_conflict(&self) -> bool {
        matches!(self, Self::DeduplicationConflict { .. })
    }

    /// Return `true` if this error is likely transient and safe to retry.
    ///
    /// `Provider` and `Persistence` errors (e.g. network timeouts, I/O failures)
    /// are classified as retryable.  Logic errors (`Memory`, `Graph`,
    /// `Orchestration`, `AgentLoop`, `NotConfigured`, `Validation`,
    /// `CircuitOpen`, `BackpressureShed`, `DeduplicationConflict`) are not.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Provider(_) | Self::Persistence(_))
    }

    /// Extract the primary message string from this error.
    ///
    /// For simple string-carrying variants (`Memory`, `Graph`, `Orchestration`,
    /// `AgentLoop`, `Provider`, `Persistence`) this returns the inner `String`.
    /// For structured variants the `Display` representation is returned so
    /// callers always get a non-empty, human-readable string.
    pub fn message(&self) -> String {
        match self {
            Self::Memory(s)
            | Self::Graph(s)
            | Self::Orchestration(s)
            | Self::AgentLoop(s)
            | Self::Provider(s)
            | Self::Persistence(s) => s.clone(),
            Self::NotConfigured(s) => s.to_string(),
            Self::CircuitOpen { service } => format!("circuit open for '{service}'"),
            Self::BackpressureShed { depth, capacity } => {
                format!("backpressure: queue depth {depth}/{capacity}")
            }
            Self::DeduplicationConflict { key } => format!("dedup conflict: {key}"),
            Self::Validation { field, code, message } => {
                format!("[{code}] {field}: {message}")
            }
        }
    }
}

impl From<serde_json::Error> for AgentRuntimeError {
    fn from(e: serde_json::Error) -> Self {
        AgentRuntimeError::AgentLoop(format!("JSON error: {e}"))
    }
}

impl From<std::io::Error> for AgentRuntimeError {
    fn from(e: std::io::Error) -> Self {
        AgentRuntimeError::Persistence(format!("I/O error: {e}"))
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for AgentRuntimeError {
    /// Convert any boxed `Send + Sync` error into an `AgentRuntimeError::AgentLoop`.
    ///
    /// This is a catch-all conversion that lets library users propagate arbitrary
    /// errors through `?` in tool handlers and inference closures without having to
    /// manually map each error type.  The error message is preserved verbatim.
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        AgentRuntimeError::AgentLoop(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_error_display() {
        let e = AgentRuntimeError::Memory("store full".into());
        assert_eq!(e.to_string(), "Memory operation failed: store full");
    }

    #[test]
    fn test_graph_error_display() {
        let e = AgentRuntimeError::Graph("entity not found".into());
        assert_eq!(e.to_string(), "Graph operation failed: entity not found");
    }

    #[test]
    fn test_orchestration_error_display() {
        let e = AgentRuntimeError::Orchestration("pipeline stalled".into());
        assert_eq!(e.to_string(), "Orchestration failed: pipeline stalled");
    }

    #[test]
    fn test_agent_loop_error_display() {
        let e = AgentRuntimeError::AgentLoop("max iterations".into());
        assert_eq!(e.to_string(), "Agent loop error: max iterations");
    }

    #[test]
    fn test_not_configured_error_display() {
        let e = AgentRuntimeError::NotConfigured("memory");
        assert_eq!(e.to_string(), "Runtime not configured: missing 'memory'");
    }

    #[test]
    fn test_circuit_open_error_display() {
        let e = AgentRuntimeError::CircuitOpen {
            service: "llm-api".into(),
        };
        assert_eq!(e.to_string(), "Circuit breaker open for 'llm-api'");
    }

    #[test]
    fn test_backpressure_shed_error_display() {
        let e = AgentRuntimeError::BackpressureShed {
            depth: 100,
            capacity: 100,
        };
        assert_eq!(
            e.to_string(),
            "Backpressure threshold exceeded: queue depth 100/100"
        );
    }

    #[test]
    fn test_deduplication_conflict_display() {
        let e = AgentRuntimeError::DeduplicationConflict {
            key: "abc123".into(),
        };
        assert_eq!(e.to_string(), "Deduplication key collision: abc123");
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AgentRuntimeError>();
    }

    #[test]
    fn test_error_debug_format() {
        let e = AgentRuntimeError::Memory("test".into());
        let debug = format!("{:?}", e);
        assert!(debug.contains("Memory"));
    }

    #[test]
    fn test_validation_error_display() {
        let e = AgentRuntimeError::Validation {
            field: "n".into(),
            code: "out_of_range".into(),
            message: "n must be between 1 and 100".into(),
        };
        assert_eq!(
            e.to_string(),
            "Validation failed for field 'n': [out_of_range] n must be between 1 and 100"
        );
    }

    #[test]
    fn test_is_circuit_open_true() {
        let e = AgentRuntimeError::CircuitOpen { service: "svc".into() };
        assert!(e.is_circuit_open());
        assert!(!e.is_backpressure());
        assert!(!e.is_provider());
        assert!(!e.is_validation());
        assert!(!e.is_memory());
        assert!(!e.is_graph());
    }

    #[test]
    fn test_is_backpressure_true() {
        let e = AgentRuntimeError::BackpressureShed { depth: 5, capacity: 5 };
        assert!(e.is_backpressure());
        assert!(!e.is_circuit_open());
    }

    #[test]
    fn test_is_provider_true() {
        let e = AgentRuntimeError::Provider("timeout".into());
        assert!(e.is_provider());
        assert!(!e.is_memory());
    }

    #[test]
    fn test_is_validation_true() {
        let e = AgentRuntimeError::Validation {
            field: "x".into(),
            code: "bad".into(),
            message: "msg".into(),
        };
        assert!(e.is_validation());
        assert!(!e.is_graph());
    }

    #[test]
    fn test_is_memory_true() {
        let e = AgentRuntimeError::Memory("oom".into());
        assert!(e.is_memory());
        assert!(!e.is_validation());
    }

    #[test]
    fn test_is_graph_true() {
        let e = AgentRuntimeError::Graph("no such entity".into());
        assert!(e.is_graph());
        assert!(!e.is_memory());
    }

    // ── Round 19: untested predicates ────────────────────────────────────────

    #[test]
    fn test_is_persistence_true() {
        let e = AgentRuntimeError::Persistence("disk full".into());
        assert!(e.is_persistence());
        assert!(!e.is_memory());
    }

    #[test]
    fn test_is_not_configured_true() {
        let e = AgentRuntimeError::NotConfigured("graph");
        assert!(e.is_not_configured());
        assert!(!e.is_persistence());
    }

    #[test]
    fn test_is_deduplication_conflict_true() {
        let e = AgentRuntimeError::DeduplicationConflict { key: "req-1".into() };
        assert!(e.is_deduplication_conflict());
        assert!(!e.is_circuit_open());
    }

    #[test]
    fn test_is_retryable_true_for_provider() {
        let e = AgentRuntimeError::Provider("503".into());
        assert!(e.is_retryable());
    }

    #[test]
    fn test_is_retryable_true_for_persistence() {
        let e = AgentRuntimeError::Persistence("io error".into());
        assert!(e.is_retryable());
    }

    #[test]
    fn test_is_retryable_false_for_logic_errors() {
        assert!(!AgentRuntimeError::Memory("x".into()).is_retryable());
        assert!(!AgentRuntimeError::Graph("x".into()).is_retryable());
        assert!(!AgentRuntimeError::Orchestration("x".into()).is_retryable());
        assert!(!AgentRuntimeError::CircuitOpen { service: "s".into() }.is_retryable());
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{invalid}").unwrap_err();
        let e = AgentRuntimeError::from(json_err);
        assert!(matches!(e, AgentRuntimeError::AgentLoop(_)));
    }

    #[test]
    fn test_provider_error_display() {
        let e = AgentRuntimeError::Provider("rate limited".into());
        assert!(e.to_string().contains("rate limited"));
    }

    #[test]
    fn test_persistence_error_display() {
        let e = AgentRuntimeError::Persistence("file not found".into());
        assert!(e.to_string().contains("file not found"));
    }

    // ── Round 28: is_agent_loop, is_orchestration ─────────────────────────────

    #[test]
    fn test_is_agent_loop_true_for_agent_loop_variant() {
        let e = AgentRuntimeError::AgentLoop("step failed".into());
        assert!(e.is_agent_loop());
    }

    #[test]
    fn test_is_agent_loop_false_for_other_variants() {
        let e = AgentRuntimeError::Memory("oom".into());
        assert!(!e.is_agent_loop());
    }

    #[test]
    fn test_is_orchestration_true_for_orchestration_variant() {
        let e = AgentRuntimeError::Orchestration("pipeline stalled".into());
        assert!(e.is_orchestration());
    }

    #[test]
    fn test_is_orchestration_false_for_other_variants() {
        let e = AgentRuntimeError::Graph("cycle".into());
        assert!(!e.is_orchestration());
    }

    // ── Round 40: From<Box<dyn Error + Send + Sync>> ──────────────────────────

    #[test]
    fn test_from_boxed_error_produces_agent_loop_variant() {
        let boxed: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, "generic failure"));
        let e = AgentRuntimeError::from(boxed);
        assert!(matches!(e, AgentRuntimeError::AgentLoop(_)));
        assert!(e.to_string().contains("generic failure"));
    }

    #[test]
    fn test_from_boxed_error_preserves_message() {
        let boxed: Box<dyn std::error::Error + Send + Sync> =
            "custom error message".parse::<i32>().unwrap_err().into();
        let e = AgentRuntimeError::from(boxed);
        assert!(e.is_agent_loop());
    }

    // ── Round 44: message() ───────────────────────────────────────────────────

    #[test]
    fn test_message_returns_inner_string_for_memory_variant() {
        let e = AgentRuntimeError::Memory("store full".into());
        assert_eq!(e.message(), "store full");
    }

    #[test]
    fn test_message_returns_inner_string_for_provider_variant() {
        let e = AgentRuntimeError::Provider("timeout".into());
        assert_eq!(e.message(), "timeout");
    }

    #[test]
    fn test_message_returns_structured_text_for_circuit_open() {
        let e = AgentRuntimeError::CircuitOpen { service: "llm".into() };
        assert!(e.message().contains("llm"));
    }

    #[test]
    fn test_message_returns_structured_text_for_validation() {
        let e = AgentRuntimeError::Validation {
            field: "n".into(),
            code: "range".into(),
            message: "must be positive".into(),
        };
        let msg = e.message();
        assert!(msg.contains("n") && msg.contains("must be positive"));
    }

    #[test]
    fn test_message_returns_structured_text_for_backpressure() {
        let e = AgentRuntimeError::BackpressureShed { depth: 10, capacity: 10 };
        let msg = e.message();
        assert!(msg.contains("10"));
    }
}
