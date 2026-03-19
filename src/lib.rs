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

#[cfg(feature = "memory")]
pub mod memory;

#[cfg(feature = "graph")]
pub mod graph;

#[cfg(feature = "orchestrator")]
pub mod orchestrator;

pub mod agent;
pub mod metrics;

#[cfg(feature = "memory")]
pub mod runtime;

#[cfg(feature = "persistence")]
pub mod persistence;

#[cfg(feature = "providers")]
pub mod providers;

// ── Top-level re-exports ────────────────────────────────────────────────────

pub use error::AgentRuntimeError;

#[cfg(feature = "memory")]
pub use runtime::{AgentRuntime, AgentRuntimeBuilder, AgentSession};

// Re-export the most fundamental types unconditionally so users don't need
// to enable specific features just to use `AgentRuntime::builder()`.

#[cfg(feature = "memory")]
pub use memory::{AgentId, MemoryId, MemoryItem};

#[cfg(feature = "graph")]
pub use graph::{Entity, EntityId, GraphStore, Relationship};

#[cfg(feature = "orchestrator")]
pub use orchestrator::{CircuitBreaker, Pipeline, RetryPolicy};

pub use agent::{AgentConfig, ReActLoop, ReActStep, ToolSpec, ToolValidator};

#[cfg(feature = "persistence")]
pub use persistence::{FilePersistenceBackend, PersistenceBackend};

#[cfg(feature = "providers")]
pub use providers::LlmProvider;
