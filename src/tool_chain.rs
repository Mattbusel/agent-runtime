//! # Module: Tool Chaining
//!
//! ## Responsibility
//! Provides a declarative pipeline where the output of one tool is automatically
//! fed as input to the next.  Chains are defined in TOML and loaded at runtime.
//!
//! ## Example TOML definition
//!
//! ```toml
//! [[chain]]
//! name = "research_and_summarize"
//! tools = ["web_search", "extract_text", "summarize"]
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use llm_agent_runtime::tool_chain::{ToolChain, ToolChainConfig, ToolChainRunner};
//! use serde_json::json;
//!
//! # tokio_test::block_on(async {
//! let config = ToolChainConfig::from_toml(r#"
//! [[chain]]
//! name = "research_and_summarize"
//! tools = ["web_search", "extract_text", "summarize"]
//! "#).unwrap();
//!
//! let mut runner = ToolChainRunner::new();
//! runner.register("web_search", |input| {
//!     Box::pin(async move { serde_json::json!({"text": format!("results for {input}")}) })
//! });
//! runner.register("extract_text", |input| {
//!     Box::pin(async move { serde_json::json!({"text": input["text"].as_str().unwrap_or("").to_uppercase()}) })
//! });
//! runner.register("summarize", |input| {
//!     Box::pin(async move { serde_json::json!({"summary": format!("Summary: {}", input["text"].as_str().unwrap_or(""))}) })
//! });
//!
//! let chain = config.get("research_and_summarize").unwrap();
//! let result = runner.run(chain, json!({"query": "Rust async"})).await.unwrap();
//! println!("{result}");
//! # });
//! ```

use crate::error::AgentRuntimeError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A pinned, boxed future returning a JSON `Value`. Used for chain step handlers.
pub type ChainStepFuture = Pin<Box<dyn Future<Output = Value> + Send>>;

/// A handler for a single tool step in a chain.
pub type ChainStepHandler = Box<dyn Fn(Value) -> ChainStepFuture + Send + Sync>;

// ── Config ─────────────────────────────────────────────────────────────────────

/// A single named tool chain definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChain {
    /// Human-readable name for this chain (e.g. `"research_and_summarize"`).
    pub name: String,
    /// Ordered list of tool names; output of step N becomes input to step N+1.
    pub tools: Vec<String>,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: String,
}

/// Top-level TOML structure containing one or more chain definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawChainConfig {
    #[serde(rename = "chain")]
    chains: Vec<ToolChain>,
}

/// Parsed and indexed collection of [`ToolChain`]s.
#[derive(Debug, Clone, Default)]
pub struct ToolChainConfig {
    chains: HashMap<String, ToolChain>,
}

impl ToolChainConfig {
    /// Parse chain definitions from a TOML string.
    ///
    /// # Errors
    /// Returns [`AgentRuntimeError::Internal`] if the TOML is malformed.
    pub fn from_toml(toml_str: &str) -> Result<Self, AgentRuntimeError> {
        let raw: RawChainConfig = toml::from_str(toml_str)
            .map_err(|e| AgentRuntimeError::Internal(format!("TOML parse error: {e}")))?;
        let chains = raw
            .chains
            .into_iter()
            .map(|c| (c.name.clone(), c))
            .collect();
        Ok(Self { chains })
    }

    /// Look up a chain by name.
    pub fn get(&self, name: &str) -> Option<&ToolChain> {
        self.chains.get(name)
    }

    /// Return all chain names.
    pub fn chain_names(&self) -> Vec<&str> {
        self.chains.keys().map(|s| s.as_str()).collect()
    }
}

// ── Runner ─────────────────────────────────────────────────────────────────────

/// Executes [`ToolChain`]s by calling registered step handlers in sequence.
///
/// Each step receives the full JSON output of the previous step as its input.
/// The first step receives the caller-supplied `initial_input`.
#[derive(Default)]
pub struct ToolChainRunner {
    handlers: HashMap<String, ChainStepHandler>,
}

impl ToolChainRunner {
    /// Create a new runner with no handlers registered.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a handler for a tool name.
    ///
    /// The handler is an async closure: `Fn(Value) -> impl Future<Output = Value>`.
    pub fn register<F, Fut>(&mut self, tool_name: &str, handler: F)
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Value> + Send + 'static,
    {
        self.handlers.insert(
            tool_name.to_string(),
            Box::new(move |input| Box::pin(handler(input))),
        );
    }

    /// Execute a chain, threading outputs through each step in order.
    ///
    /// # Errors
    /// Returns [`AgentRuntimeError::Internal`] if a referenced tool has no
    /// registered handler.
    pub async fn run(
        &self,
        chain: &ToolChain,
        initial_input: Value,
    ) -> Result<Value, AgentRuntimeError> {
        let mut current = initial_input;

        for tool_name in &chain.tools {
            let handler = self.handlers.get(tool_name.as_str()).ok_or_else(|| {
                AgentRuntimeError::Internal(format!(
                    "Tool '{tool_name}' not registered in ToolChainRunner"
                ))
            })?;
            tracing::debug!(tool = %tool_name, "tool_chain: executing step");
            current = handler(current).await;
        }

        Ok(current)
    }

    /// Return the names of all registered tools.
    pub fn registered_tools(&self) -> Vec<&str> {
        self.handlers.keys().map(|s| s.as_str()).collect()
    }
}

// ── Execution trace ────────────────────────────────────────────────────────────

/// A single step's input/output captured during a chain execution trace.
#[derive(Debug, Clone, Serialize)]
pub struct ChainStepTrace {
    /// Name of the tool invoked.
    pub tool: String,
    /// JSON value supplied to the tool.
    pub input: Value,
    /// JSON value returned by the tool.
    pub output: Value,
}

/// Full execution trace of a chain run.
#[derive(Debug, Clone, Serialize)]
pub struct ChainTrace {
    /// Name of the chain that was executed.
    pub chain_name: String,
    /// Per-step traces in execution order.
    pub steps: Vec<ChainStepTrace>,
    /// Final output (== last step's output).
    pub final_output: Value,
}

/// A [`ToolChainRunner`] variant that also records an execution trace.
#[derive(Default)]
pub struct TracingChainRunner {
    inner: ToolChainRunner,
}

impl TracingChainRunner {
    /// Create a new tracing runner.
    pub fn new() -> Self {
        Self {
            inner: ToolChainRunner::new(),
        }
    }

    /// Delegate handler registration to the inner runner.
    pub fn register<F, Fut>(&mut self, tool_name: &str, handler: F)
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Value> + Send + 'static,
    {
        self.inner.register(tool_name, handler);
    }

    /// Execute a chain and return both the result and a full [`ChainTrace`].
    ///
    /// # Errors
    /// Returns [`AgentRuntimeError::Internal`] when a step's handler is missing.
    pub async fn run_traced(
        &self,
        chain: &ToolChain,
        initial_input: Value,
    ) -> Result<ChainTrace, AgentRuntimeError> {
        let mut current = initial_input.clone();
        let mut steps = Vec::with_capacity(chain.tools.len());

        for tool_name in &chain.tools {
            let handler = self.inner.handlers.get(tool_name.as_str()).ok_or_else(|| {
                AgentRuntimeError::Internal(format!(
                    "Tool '{tool_name}' not registered in TracingChainRunner"
                ))
            })?;
            let input_snapshot = current.clone();
            tracing::debug!(tool = %tool_name, "tool_chain(tracing): executing step");
            let output = handler(current).await;
            steps.push(ChainStepTrace {
                tool: tool_name.clone(),
                input: input_snapshot,
                output: output.clone(),
            });
            current = output;
        }

        Ok(ChainTrace {
            chain_name: chain.name.clone(),
            steps,
            final_output: current,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SAMPLE_TOML: &str = r#"
[[chain]]
name = "research_and_summarize"
tools = ["web_search", "extract_text", "summarize"]
description = "Search the web and produce a summary"

[[chain]]
name = "echo_chain"
tools = ["echo"]
"#;

    #[test]
    fn parse_toml() {
        let cfg = ToolChainConfig::from_toml(SAMPLE_TOML).unwrap();
        let chain = cfg.get("research_and_summarize").unwrap();
        assert_eq!(chain.tools, ["web_search", "extract_text", "summarize"]);
        assert!(cfg.get("echo_chain").is_some());
    }

    #[test]
    fn chain_names() {
        let cfg = ToolChainConfig::from_toml(SAMPLE_TOML).unwrap();
        let mut names = cfg.chain_names();
        names.sort();
        assert_eq!(names, ["echo_chain", "research_and_summarize"]);
    }

    #[tokio::test]
    async fn run_chain() {
        let cfg = ToolChainConfig::from_toml(SAMPLE_TOML).unwrap();
        let chain = cfg.get("research_and_summarize").unwrap().clone();

        let mut runner = ToolChainRunner::new();
        runner.register("web_search", |v| {
            Box::pin(async move { json!({"text": format!("results for {}", v["query"].as_str().unwrap_or(""))}) })
        });
        runner.register("extract_text", |v| {
            Box::pin(async move { json!({"text": v["text"].as_str().unwrap_or("").to_uppercase()}) })
        });
        runner.register("summarize", |v| {
            Box::pin(async move { json!({"summary": format!("Summary: {}", v["text"].as_str().unwrap_or(""))}) })
        });

        let result = runner.run(&chain, json!({"query": "rust"})).await.unwrap();
        assert!(result["summary"]
            .as_str()
            .unwrap_or("")
            .contains("RESULTS FOR RUST"));
    }

    #[tokio::test]
    async fn tracing_runner_captures_steps() {
        let cfg = ToolChainConfig::from_toml(SAMPLE_TOML).unwrap();
        let chain = cfg.get("research_and_summarize").unwrap().clone();

        let mut runner = TracingChainRunner::new();
        runner.register("web_search", |v| {
            Box::pin(async move { json!({"text": format!("result:{}", v["query"].as_str().unwrap_or(""))}) })
        });
        runner.register("extract_text", |v| {
            Box::pin(async move { json!({"text": v["text"].as_str().unwrap_or("").to_uppercase()}) })
        });
        runner.register("summarize", |v| {
            Box::pin(async move { json!({"summary": v["text"].as_str().unwrap_or("")}) })
        });

        let trace = runner
            .run_traced(&chain, json!({"query": "tokio"}))
            .await
            .unwrap();
        assert_eq!(trace.steps.len(), 3);
        assert_eq!(trace.steps[0].tool, "web_search");
        assert!(trace.final_output["summary"]
            .as_str()
            .unwrap_or("")
            .contains("RESULT:TOKIO"));
    }

    #[tokio::test]
    async fn missing_handler_returns_error() {
        let cfg = ToolChainConfig::from_toml(SAMPLE_TOML).unwrap();
        let chain = cfg.get("research_and_summarize").unwrap().clone();
        let runner = ToolChainRunner::new(); // no handlers registered
        let err = runner.run(&chain, json!({})).await.unwrap_err();
        assert!(matches!(err, AgentRuntimeError::Internal(_)));
    }
}
