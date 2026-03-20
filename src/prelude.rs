//! Convenience re-exports for the most commonly used types.
//!
//! ```rust
//! use llm_agent_runtime::prelude::*;
//! ```

pub use crate::error::AgentRuntimeError;

#[cfg(feature = "memory")]
pub use crate::runtime::{
    AgentRuntime, AgentRuntimeBuilder, AgentSession, CharDivTokenEstimator, TokenEstimator,
};

#[cfg(feature = "memory")]
pub use crate::memory::{
    AgentId, DecayPolicy, EpisodicStore, EpisodicStoreBuilder, EvictionPolicy, MemoryId,
    MemoryItem, SemanticStore, WorkingMemory,
};

#[cfg(feature = "graph")]
pub use crate::graph::{Entity, EntityId, GraphStore, MemGraphError, Relationship};

#[cfg(feature = "orchestrator")]
pub use crate::orchestrator::{
    BackpressureGuard, CircuitBreaker, CircuitState, DeduplicationResult, Deduplicator, Pipeline,
    RetryKind, RetryPolicy,
};

pub use crate::agent::{
    parse_react_step, Action, ActionHook, AgentConfig, AgentError, InMemoryToolCache, Message,
    Observer, ReActLoop, ReActStep, Role, ToolCache, ToolRegistry, ToolSpec, ToolValidator,
};

#[cfg(feature = "memory")]
pub use crate::memory::RecallPolicy;

#[cfg(feature = "orchestrator")]
pub use crate::orchestrator::{PipelineResult, Stage};

#[cfg(feature = "persistence")]
pub use crate::persistence::{FilePersistenceBackend, PersistenceBackend};

#[cfg(feature = "providers")]
pub use crate::providers::{CompletionOptions, LlmProvider};

pub use crate::metrics::{LatencyHistogram, MetricsSnapshot, RuntimeMetrics};
