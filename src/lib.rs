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

/// Typed publish/subscribe event bus for inter-agent communication: topics, wildcard subscriptions, filtered streams.
pub mod event_bus;

/// Agent capability registry and negotiation: versioned capabilities, registry, set intersection.
pub mod capability;

pub mod memory_compression;
pub mod discovery;

/// High-level memory compression: temporal windowing, semantic dedup, importance filter.
pub mod memory_compress;

/// Agent swarm coordination v2: subtask decomposition, convergence threshold, result synthesis.
pub mod swarm_v2;

/// Skill marketplace: global registry, filesystem loader, task-based matcher, pipeline composer.
pub mod marketplace;

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
/// Background memory consolidation using TF-IDF k-means clustering.
pub mod consolidation;
/// Plan → Execute → Verify structured agent loop.
pub mod plan_execute;
/// Typed inter-agent messaging protocol with task delegation, capability queries, and heartbeats.
pub mod messaging;
pub mod dialogue;
pub mod metrics;
pub mod planner;
pub mod runtime;
pub mod sandbox;

/// Finite state machine for agent lifecycle management: guarded transitions, history, error recovery.
pub mod fsm;

/// Sandboxed process execution environment: resource limits, timeout enforcement, MockSandbox for testing.
pub mod exec_sandbox;

/// Agent workflow engine: declarative step graphs with branching, parallelism, and loops.
pub mod workflow;

/// Tool registry: named async tools with call-count, error-count, and latency tracking.
pub mod tools;

/// Agent checkpointing: save and restore agent state via in-memory or file-backed stores.
pub mod checkpoint;
pub mod streaming;
/// Multi-agent team orchestration with configurable topology and consensus.
pub mod team;

/// Declarative tool pipeline: output of one tool feeds as input to the next.
pub mod tool_chain;

/// Concurrent tool execution using `FuturesUnordered` with dependency detection.
pub mod parallel_tools;

/// Agent self-reflection: score reasoning after each ReAct cycle and retry on low scores.
pub mod reflection;

/// Named, parameterised skill sequences definable in TOML.
pub mod skills;

/// Multi-agent swarm: decompose tasks into subtasks and run worker agents concurrently.
pub mod swarm;

/// World model: typed key-value belief store with temporal history and planning integration.
pub mod world_model;
pub mod debate;

/// In-memory knowledge graph: entities, relations, BFS shortest-path, subgraph extraction, DOT export.
pub mod knowledge;

/// Vector similarity memory search: embeddings, cosine search, TF-IDF bag-of-words, semantic memory.
pub mod vector_memory;

/// Erlang-style agent supervisor tree: OneForOne/OneForAll/RestForOne strategies with typed restart policies.
pub mod supervisor;

/// STRIPS-style backward-chaining planner for goal-directed action sequencing.
pub mod strips;

/// Multi-agent conversation protocol: message types, threading, routing, and consensus voting.
pub mod protocol;

/// Persistent goal tracking with priority, deadlines, and dependency relationships.
pub mod goals;

/// Cognitive load monitor: tracks active goals, reasoning depth, and memory pressure.
pub mod cognitive_load;

/// Long-term memory with Ebbinghaus forgetting-curve decay and consolidation.
pub mod ltm;

/// Agent persona system: tones, built-in personas, registry, and scoped application.
pub mod persona;

/// JSON-schema-like validation of tool call arguments before execution.
pub mod tool_validator;

/// Structured per-agent metrics tracking for multi-agent systems.
pub mod agent_metrics;

/// DAG-based workflow execution with topological sort and parallel execution.
pub mod dag;

/// Fixed-size slab memory pool for agent message buffers with use-after-free protection.
pub mod memory_pool;

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
pub use metrics::{Counter, Gauge, Histogram, MetricsRegistry, MetricsSnapshot};

// Workflow engine re-exports.
pub use workflow::{
    InferFn, Workflow, WorkflowContext, WorkflowEngine, WorkflowError, WorkflowResult,
    WorkflowStep,
};

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
// Note: `AgentMessage` and `BusStats` are already re-exported from `messaging` above,
// so we alias the bus equivalents to avoid ambiguity.
pub use bus::{
    AgentBus,
    AgentMessage as BusAgentMessage,
    AgentRole,
    AgentTarget,
    BusError,
    BusStats as AgentBusStats,
    MessageId,
};

// Team orchestration re-exports.
pub use team::{
    AgentTeam, CommunicationTopology, ConsensusStrategy, TeamConfig, TeamOrchestrator, TeamResult,
};

// Swarm coordination v2 re-exports.
pub use swarm_v2::{
    AgentSwarm, SubTask, SubTaskResult, SubTaskStatus, SwarmConfig as SwarmConfigV2,
    SwarmOrchestrator, SwarmResult as SwarmResultV2, TaskSplitStrategy,
    converge_results,
};

// High-level memory compression re-exports.
pub use memory_compress::{
    CompressedEpisode, CompressorConfig,
    ImportanceFilter, MemoryCompressor as HighLevelMemoryCompressor,
    SemanticDedup, TemporalCompression,
};

// Marketplace re-exports.
pub use marketplace::{
    Skill as MarketplaceSkill, SkillComposer, SkillLoader, SkillMatch, SkillMatcher,
    SkillPipeline, SkillRegistry,
};

// World model re-exports.
pub use world_model::{Belief, FactValue, TemporalRecord, WorldChange, WorldState};

// STRIPS planner re-exports.
pub use strips::{StripsFact, StripsAction, StripsPlanner, StripsPlannerError};

// Multi-agent conversation protocol re-exports.
pub use protocol::{
    AgentMessage as ProtocolAgentMessage,
    ConsensusError, ConsensusPolicy, ConsensusRound,
    ConversationThread, MessageKind,
    ProtocolMessageId, ProtocolRouter, Vote,
};

// Persistent goal tracking re-exports.
pub use goals::{Goal, GoalId, GoalPriority, GoalStatus, GoalStore};

// Cognitive load monitor re-exports.
pub use cognitive_load::{
    CognitiveLoadConfig, CognitiveLoadMonitor, CognitiveLoadSnapshot, LoadAction,
};

// Tool validator re-exports.
pub use tool_validator::{
    ParameterSpec, ParameterTypeHint, SchemaValidator, ToolCall, ToolSchema, ValidationError,
};

// Agent metrics re-exports.
pub use agent_metrics::{AgentMetrics, AgentMetricsRegistry, AgentMetricsSnapshot};

// Tool registry re-exports.
pub use tools::{
    CalculatorTool, EchoTool, TimestampTool, Tool, ToolError, ToolInfo, ToolRegistry,
};

// Checkpoint re-exports.
pub use checkpoint::{
    AgentCheckpoint, CheckpointError, CheckpointManager, CheckpointStore,
    FileCheckpointStore, InMemoryCheckpointStore, MemoryEntry,
};

/// RBAC-lite policy enforcement: roles, principals, allow/deny rules, glob matching, rate limiting.
pub mod policy;

/// Distributed tracing context propagation: W3C Trace Context headers, task-local storage, timed spans.
pub mod tracing_context;

/// Multi-agent voting and consensus: proposals, weighted rules, tally, and outcome resolution.
pub mod consensus;

/// Agent state snapshotting and restoration: CRC-32-verified snapshots, bounded deque store, Snapshottable trait.
pub mod snapshot;

/// Shadow traffic duplicator: deterministic sampling, divergence scoring, aggregated reports.
pub mod shadow_traffic;

/// Resource budget manager: per-task token/tool-call/time/memory limits with Prometheus-style metrics.
pub mod resource_manager;

/// Agent-to-agent negotiation protocol: Greedy, FairShare, PriorityBased, and Vickrey auction allocation.
pub mod negotiation;
