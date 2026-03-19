//! Convenience re-exports for the most commonly used types.
//!
//! ```rust
//! use llm_agent_runtime::prelude::*;
//! ```

pub use crate::error::AgentRuntimeError;

#[cfg(feature = "memory")]
pub use crate::runtime::{AgentRuntime, AgentRuntimeBuilder, AgentSession};

#[cfg(feature = "memory")]
pub use crate::memory::{
    AgentId, DecayPolicy, EpisodicStore, MemoryId, MemoryItem, SemanticStore, WorkingMemory,
};

#[cfg(feature = "graph")]
pub use crate::graph::{Entity, EntityId, GraphStore, MemGraphError, Relationship};

#[cfg(feature = "orchestrator")]
pub use crate::orchestrator::{
    BackpressureGuard, CircuitBreaker, CircuitState, DeduplicationResult, Deduplicator, Pipeline,
    RetryPolicy, MAX_RETRY_DELAY,
};

pub use crate::agent::{
    parse_react_step, Action, AgentConfig, AgentError, AsyncToolFuture, AsyncToolResultFuture,
    Message, ReActLoop, ReActStep, Role, ToolRegistry, ToolSpec, ToolValidator,
};

#[cfg(feature = "memory")]
pub use crate::memory::RecallPolicy;

#[cfg(feature = "orchestrator")]
pub use crate::orchestrator::{PipelineResult, Stage};

#[cfg(feature = "persistence")]
pub use crate::persistence::{FilePersistenceBackend, PersistenceBackend};

pub use crate::metrics::{MetricsSnapshot, RuntimeMetrics};
