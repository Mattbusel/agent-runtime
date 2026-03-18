//! # agent-runtime
//!
//! A unified Tokio agent runtime that brings together orchestration, memory,
//! knowledge graph, and a ReAct agent loop in a single crate.
// Enforce documentation on all public items in production code.
#![deny(missing_docs)]
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
//! use agent_runtime::prelude::*;
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
//! ## Upstream Crates
//!
//! This crate implements standalone versions of the APIs from:
//! - `tokio-prompt-orchestrator` (Mattbusel/tokio-prompt-orchestrator)
//! - `tokio-agent-memory` (Mattbusel/tokio-agent-memory)
//! - `mem-graph` (Mattbusel/mem-graph)
//! - `wasm-agent` (Mattbusel/wasm-agent)
//! - `tokio-memory` (Mattbusel/tokio-memory)

// ── Public modules ─────────────────────────────────────────────────────────

pub mod error;
pub mod prelude;
pub mod runtime;

#[cfg(feature = "memory")]
pub mod memory;

#[cfg(feature = "graph")]
pub mod graph;

#[cfg(feature = "orchestrator")]
pub mod orchestrator;

pub mod agent;
pub mod metrics;

#[cfg(feature = "persistence")]
pub mod persistence;

#[cfg(feature = "providers")]
pub mod providers;

// ── Top-level re-exports ────────────────────────────────────────────────────

pub use error::AgentRuntimeError;
pub use runtime::{AgentRuntime, AgentRuntimeBuilder, AgentSession};

// Re-export the most fundamental types unconditionally so users don't need
// to enable specific features just to use `AgentRuntime::builder()`.

#[cfg(feature = "memory")]
pub use memory::{AgentId, MemoryId, MemoryItem};

#[cfg(feature = "graph")]
pub use graph::{Entity, EntityId, GraphStore, Relationship};

#[cfg(feature = "orchestrator")]
pub use orchestrator::{CircuitBreaker, Pipeline, RetryPolicy};

pub use agent::{AgentConfig, ReActLoop, ReActStep, ToolSpec};

#[cfg(feature = "persistence")]
pub use persistence::{FilePersistenceBackend, PersistenceBackend};

#[cfg(feature = "providers")]
pub use providers::LlmProvider;
