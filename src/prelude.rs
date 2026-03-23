//! Convenience re-exports for the most commonly used types.
//!
//! ```rust
//! use llm_agent_runtime::prelude::*;
//! ```

pub use crate::error::AgentRuntimeError;

pub use crate::runtime::{
    AgentRuntime, AgentRuntimeBuilder, AgentSession, CharDivTokenEstimator, TokenEstimator,
};

pub use crate::types::{AgentId, MemoryId};

#[cfg(feature = "memory")]
pub use crate::memory::{
    DecayPolicy, EpisodicStore, EpisodicStoreBuilder, EvictionPolicy, MemoryItem, SemanticStore,
    WorkingMemory,
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

pub use crate::conversation::{ConversationHistory, SummaryEntry, Turn};

pub use crate::budget::{BudgetConfig, BudgetReport, BudgetTracker, ModelPrice, PriceTable, UsageRecord};

#[cfg(feature = "distributed")]
pub use crate::distributed::{CoordinatorConfig, DistributedCoordinator, WorkItem};

#[cfg(feature = "otel")]
pub use crate::telemetry::{OtelTracer, ToolCallSpan};

pub use crate::bus::{AgentBus, AgentMessage, AgentRole, AgentTarget, BusError, BusStats, MessageId};

pub use crate::team::{
    AgentTeam, CommunicationTopology, ConsensusStrategy, TeamConfig, TeamOrchestrator, TeamResult,
};

pub use crate::streaming::{
    AgentEvent, AgentEventStream, InferenceToken, StreamBroadcaster, StreamingCallbacks,
    StreamingInference, StreamingReActLoop, StreamingSession, StreamingStep, TokenStream,
};
