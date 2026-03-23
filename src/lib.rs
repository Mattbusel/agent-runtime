//! # agent-runtime
//!
//! A unified Tokio agent runtime that brings together orchestration, memory,
//! knowledge graph, and a ReAct agent loop in a single crate.
//!
//! ## Feature Flags
//!
//! | Feature | Content |
//! |---------|---------|
//! | `orchestrator` (default) | Circuit breaker, retry, dedup, backpressure, pipeline |
//! | `memory` (default) | Episodic, semantic, working memory + decay |
//! | `graph` (default) | In-memory knowledge graph (BFS/DFS/shortest-path) |
//! | `wasm` | ReAct agent loop, tool registry |
//! | `full` | All of the above |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use llm_agent_runtime::prelude::*;
//!
//! // build() is now infallible — the typestate pattern enforces that
//! // with_agent_config() is called before build() at compile time.
//! let runtime = AgentRuntime::builder()
//!     .with_agent_config(AgentConfig::new(5, "my-model"))
//!     .build();
//!
//! // run_agent is async; drive it with your async runtime.
//! // tokio::runtime::Runtime::new().unwrap().block_on(async {
//! //     let session = runtime
//! //         .run_agent(AgentId::new("agent-1"), "Hello", |_ctx: String| async {
//! //             "Thought: done\nAction: FINAL_ANSWER hi".to_string()
//! //         })
//! //         .await
//! //         .unwrap();
//! //     println!("Steps: {}", session.step_count());
//! // });
//! ```
//!
//! ## Extension Points
//!
//! The primary extension point is the [`PersistenceBackend`] trait (enabled by
//! the `persistence` feature). Implement it to persist agent sessions and
//! per-step checkpoints to any storage backend — disk, Redis, S3, or a
//! database. The bundled [`FilePersistenceBackend`] stores each checkpoint as a
//! `<key>.bin` file in a local directory and serves as a reference
//! implementation.
//!
//! Additional extension points:
//! - [`LlmProvider`] (`providers` feature) — plug in any LLM API.
//! - `CircuitBreakerBackend` (`redis-circuit-breaker` feature) — distributed
//!   circuit-breaker state via Redis.
//!
//! ## Upstream Crates
//!
//! This crate implements standalone versions of the APIs from:
//! - `tokio-prompt-orchestrator` (Mattbusel/tokio-prompt-orchestrator)
//! - `tokio-agent-memory` (Mattbusel/tokio-agent-memory)
//! - `mem-graph` (Mattbusel/mem-graph)
//! - `wasm-agent` (Mattbusel/wasm-agent)
//! - `tokio-memory` (Mattbusel/tokio-memory)

// Enforce documentation on all public items in production code.
#![deny(missing_docs)]

// ── Public modules ─────────────────────────────────────────────────────────

pub mod util;

pub mod error;
pub mod prelude;
pub mod types;

pub mod memory_compression;
pub mod discovery;

/// Multi-turn conversation history with automatic summarisation.
pub mod conversation;

/// Token budget tracking and cost estimation.
pub mod budget;

#[cfg(feature = "distributed")]
/// Distributed agent coordination via Redis (work queue and leader election).
pub mod distributed;

#[cfg(feature = "otel")]
/// OpenTelemetry tracing integration for tool call spans.
pub mod telemetry;

#[cfg(feature = "memory")]
pub mod memory;

#[cfg(feature = "graph")]
pub mod graph;

#[cfg(feature = "orchestrator")]
pub mod orchestrator;

pub mod agent;
/// Async broadcast message bus for inter-agent communication.
pub mod bus;
/// Typed inter-agent messaging protocol with task delegation, capability queries, and heartbeats.
pub mod messaging;
pub mod dialogue;
pub mod metrics;
pub mod planner;
pub mod runtime;
pub mod sandbox;
pub mod streaming;
/// Multi-agent team orchestration with configurable topology and consensus.
pub mod team;

#[cfg(feature = "persistence")]
pub mod persistence;

#[cfg(feature = "providers")]
pub mod providers;

// ── Top-level re-exports ────────────────────────────────────────────────────

pub use error::AgentRuntimeError;

pub use runtime::{AgentRuntime, AgentRuntimeBuilder, AgentSession};

// Core identifier types — always available.
pub use types::{AgentId, MemoryId};

// Memory-specific re-exports.
#[cfg(feature = "memory")]
pub use memory::MemoryItem;

#[cfg(feature = "graph")]
pub use graph::{Entity, EntityId, GraphStore, Relationship};

#[cfg(feature = "orchestrator")]
pub use orchestrator::{CircuitBreaker, Pipeline, RetryKind, RetryPolicy};

pub use agent::{Action, AgentConfig, ReActLoop, ReActStep, ToolSpec, ToolValidator};
pub use metrics::MetricsSnapshot;

#[cfg(feature = "persistence")]
pub use persistence::{FilePersistenceBackend, PersistenceBackend};

#[cfg(feature = "providers")]
pub use providers::LlmProvider;

// Memory compression re-exports.
pub use memory_compression::{
    ImportanceStrategy, MemoryCompressor, MemorySummary, MemoryTurn, Role as MemoryRole,
};

// Discovery re-exports.
pub use discovery::{
    AgentAdvertisement, AgentHealth, AgentRegistry, Capability, CapabilityMatch, CapabilityQuery,
};

// Conversation history re-exports.
pub use conversation::{ConversationHistory, SummaryEntry, Turn};

// Budget tracking re-exports.
pub use budget::{BudgetConfig, BudgetReport, BudgetTracker, ModelPrice, PriceTable, UsageRecord};

// Distributed coordination re-exports (feature-gated).
#[cfg(feature = "distributed")]
pub use distributed::{CoordinatorConfig, DistributedCoordinator, WorkItem};

// Tool sandbox re-exports.
pub use sandbox::{
    AuditEntry, DenialReason, Sandbox, SandboxConfig, SandboxStats, ToolManifest, ToolPermission,
};

// HTN planner re-exports.
pub use planner::{
    ExecutionPlan, Method, Planner, PlannerConfig, PlannerError, PlanStep, Precondition, Task,
    TaskKind,
};

// DAG goal-decomposition planner re-exports.
pub use planner::{DagExecutionPlan, DagPlanStep, DagStepStatus, PlanBuilder};

// Inter-agent messaging protocol re-exports.
pub use messaging::{
    AgentMailbox, AgentMessage, BusStats as MessagingBusStats, MessageBus, MessageEnvelope,
    AgentId as MessagingAgentId,
};

// Message bus re-exports.
pub use bus::{AgentBus, AgentMessage, AgentRole, AgentTarget, BusError, BusStats, MessageId};

// Team orchestration re-exports.
pub use team::{
    AgentTeam, CommunicationTopology, ConsensusStrategy, TeamConfig, TeamOrchestrator, TeamResult,
};
