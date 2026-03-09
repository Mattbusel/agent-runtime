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
//! ```rust
//! use agent_runtime::prelude::*;
//!
//! let runtime = AgentRuntime::builder()
//!     .with_agent_config(AgentConfig::new(5, "my-model"))
//!     .build()
//!     .unwrap();
//!
//! let session = runtime
//!     .run_agent(AgentId::new("agent-1"), "Hello", |_ctx| {
//!         "Thought: done\nAction: FINAL_ANSWER hi".into()
//!     })
//!     .unwrap();
//!
//! println!("Steps: {}", session.step_count());
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

#[cfg(feature = "wasm")]
pub mod agent;

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

#[cfg(feature = "wasm")]
pub use agent::{AgentConfig, ReActLoop, ReActStep, ToolSpec};
