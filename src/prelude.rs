//! Convenience re-exports for the most commonly used types.
//!
//! ```rust
//! use agent_runtime::prelude::*;
//! ```

pub use crate::error::AgentRuntimeError;
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

#[cfg(feature = "wasm")]
pub use crate::agent::{
    parse_react_step, AgentConfig, AgentError, Message, ReActLoop, ReActStep, Role, ToolRegistry,
    ToolSpec,
};
