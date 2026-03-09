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
    CircuitOpen { service: String },

    /// Backpressure threshold exceeded — caller must shed or wait.
    #[error("Backpressure threshold exceeded: queue depth {depth}/{capacity}")]
    BackpressureShed { depth: usize, capacity: usize },

    /// A deduplication key collision was detected.
    #[error("Deduplication key collision: {key}")]
    DeduplicationConflict { key: String },
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
}
