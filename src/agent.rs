//! # Module: Agent
//!
//! ## Responsibility
//! Provides a ReAct (Thought-Action-Observation) agent loop with pluggable tools.
//! Mirrors the public API of `wasm-agent`.
//!
//! ## Guarantees
//! - Deterministic: the loop terminates after at most `max_iterations` cycles
//! - Non-panicking: all operations return `Result`
//! - Tool handlers support both sync and async `Fn` closures
//!
//! ## NOT Responsible For
//! - Actual LLM inference (callers supply a mock/stub inference fn)
//! - WASM compilation or browser execution
//! - Streaming partial responses

use crate::error::AgentRuntimeError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A pinned, boxed future returning a `Value`. Used for async tool handlers.
pub type AsyncToolFuture = Pin<Box<dyn Future<Output = Value> + Send>>;

/// An async tool handler closure.
pub type AsyncToolHandler = Box<dyn Fn(Value) -> AsyncToolFuture + Send + Sync>;

/// Role of a message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// A single ReAct step: Thought → Action → Observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStep {
    /// Agent's reasoning about the current state.
    pub thought: String,
    /// The action taken (tool name + JSON arguments, or "FINAL_ANSWER").
    pub action: String,
    /// The result of the action.
    pub observation: String,
}

/// Configuration for the ReAct agent loop.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of Thought-Action-Observation cycles.
    pub max_iterations: usize,
    /// Model identifier passed to the inference function.
    pub model: String,
    /// System prompt injected at the start of the conversation.
    pub system_prompt: String,
    /// Maximum number of episodic memories to inject into the prompt.
    /// Keeping this small prevents silent token-budget overruns.  Default: 3.
    pub max_memory_recalls: usize,
}

impl AgentConfig {
    /// Create a new config with sensible defaults.
    pub fn new(max_iterations: usize, model: impl Into<String>) -> Self {
        Self {
            max_iterations,
            model: model.into(),
            system_prompt: "You are a helpful AI agent.".into(),
            max_memory_recalls: 3,
        }
    }

    /// Override the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the maximum number of episodic memories injected per run.
    pub fn with_max_memory_recalls(mut self, n: usize) -> Self {
        self.max_memory_recalls = n;
        self
    }
}

// ── ToolSpec ──────────────────────────────────────────────────────────────────

/// Describes and implements a single callable tool.
pub struct ToolSpec {
    /// Short identifier used in action strings (e.g. "search").
    pub name: String,
    /// Human-readable description passed to the model as part of the system prompt.
    pub description: String,
    /// Async handler: receives JSON arguments, returns a future resolving to a JSON result.
    pub handler: AsyncToolHandler,
}

impl std::fmt::Debug for ToolSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSpec")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl ToolSpec {
    /// Construct a new `ToolSpec` from a synchronous handler closure.
    /// The closure is wrapped in an `async move` block automatically.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(Value) -> Value + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(move |args| {
                let result = handler(args);
                Box::pin(async move { result })
            }),
        }
    }

    /// Construct a new `ToolSpec` from an async handler closure.
    pub fn new_async(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(Value) -> AsyncToolFuture + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(handler),
        }
    }

    /// Invoke the tool with the given JSON arguments.
    pub async fn call(&self, args: Value) -> Value {
        (self.handler)(args).await
    }
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry of available tools for the agent loop.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolSpec>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. Overwrites any existing tool with the same name.
    pub fn register(&mut self, spec: ToolSpec) {
        self.tools.insert(spec.name.clone(), spec);
    }

    /// Call a tool by name.
    ///
    /// # Returns
    /// - `Ok(Value)` — tool result
    /// - `Err(AgentRuntimeError::AgentLoop)` — if the tool is not found
    #[tracing::instrument(skip_all, fields(tool_name = %name))]
    pub async fn call(&self, name: &str, args: Value) -> Result<Value, AgentRuntimeError> {
        let spec = self
            .tools
            .get(name)
            .ok_or_else(|| AgentRuntimeError::AgentLoop(format!("tool '{name}' not found")))?;
        Ok(spec.call(args).await)
    }

    /// Return the list of registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

// ── ReActLoop ─────────────────────────────────────────────────────────────────

/// Parses a ReAct response string into a `ReActStep`.
pub fn parse_react_step(text: &str) -> Result<ReActStep, AgentRuntimeError> {
    let mut thought = String::new();
    let mut action = String::new();

    for line in text.lines() {
        if let Some(t) = line.strip_prefix("Thought:") {
            thought = t.trim().to_owned();
        } else if let Some(a) = line.strip_prefix("Action:") {
            action = a.trim().to_owned();
        }
    }

    if thought.is_empty() && action.is_empty() {
        return Err(AgentRuntimeError::AgentLoop(
            "could not parse ReAct step from response".into(),
        ));
    }

    Ok(ReActStep {
        thought,
        action,
        observation: String::new(),
    })
}

/// The ReAct agent loop.
#[derive(Debug)]
pub struct ReActLoop {
    config: AgentConfig,
    registry: ToolRegistry,
}

impl ReActLoop {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            registry: ToolRegistry::new(),
        }
    }

    pub fn register_tool(&mut self, spec: ToolSpec) {
        self.registry.register(spec);
    }

    #[tracing::instrument(skip(infer))]
    pub async fn run<F, Fut>(
        &self,
        prompt: &str,
        mut infer: F,
    ) -> Result<Vec<ReActStep>, AgentRuntimeError>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = String>,
    {
        let mut steps: Vec<ReActStep> = Vec::new();
        let mut context = format!("{}\n\nUser: {}\n", self.config.system_prompt, prompt);

        for iteration in 0..self.config.max_iterations {
            let response = infer(context.clone()).await;
            let mut step = parse_react_step(&response)?;

            tracing::debug!(
                step = iteration,
                thought = %step.thought,
                action = %step.action,
                "ReAct iteration"
            );

            if step.action.to_ascii_uppercase().starts_with("FINAL_ANSWER") {
                step.observation = step.action.clone();
                steps.push(step);
                tracing::info!(step = iteration, "FINAL_ANSWER reached");
                return Ok(steps);
            }

            let (tool_name, args) = parse_tool_call(&step.action);
            let observation = match self.registry.call(&tool_name, args).await {
                Ok(result) => serde_json::json!({ "ok": true, "data": result }).to_string(),
                Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }).to_string(),
            };

            step.observation = observation.clone();
            context.push_str(&format!(
                "\nThought: {}\nAction: {}\nObservation: {}\n",
                step.thought, step.action, observation
            ));
            steps.push(step);
        }

        Err(AgentRuntimeError::AgentLoop(format!(
            "max iterations ({}) reached without final answer",
            self.config.max_iterations
        )))
    }
}

/// Split `"tool_name {json}"` into `(tool_name, Value)`.
fn parse_tool_call(action: &str) -> (String, Value) {
    let mut parts = action.splitn(2, ' ');
    let name = parts.next().unwrap_or("").to_owned();
    let args_str = parts.next().unwrap_or("{}");
    let args: Value = serde_json::from_str(args_str).unwrap_or(Value::String(args_str.to_owned()));
    (name, args)
}

/// Agent-specific errors, mirrors `wasm-agent::AgentError`.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Tool '{0}' not found")]
    ToolNotFound(String),
    #[error("Max iterations exceeded: {0}")]
    MaxIterations(usize),
    #[error("Parse error: {0}")]
    ParseError(String),
}

impl From<AgentError> for AgentRuntimeError {
    fn from(e: AgentError) -> Self {
        AgentRuntimeError::AgentLoop(e.to_string())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_final_answer_on_first_step() {
        let config = AgentConfig::new(5, "test-model");
        let loop_ = ReActLoop::new(config);

        let steps = loop_
            .run("Say hello", |_ctx| async {
                "Thought: I will answer directly\nAction: FINAL_ANSWER hello".to_string()
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 1);
        assert!(steps[0].action.to_ascii_uppercase().starts_with("FINAL_ANSWER"));
    }

    #[tokio::test]
    async fn test_tool_call_then_final_answer() {
        let config = AgentConfig::new(5, "test-model");
        let mut loop_ = ReActLoop::new(config);

        loop_.register_tool(ToolSpec::new("greet", "Greets someone", |_args| {
            serde_json::json!("hello!")
        }));

        let mut call_count = 0;
        let steps = loop_
            .run("Say hello", |_ctx| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: I will greet\nAction: greet {}".to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].action, "greet {}");
        assert!(steps[1].action.to_ascii_uppercase().starts_with("FINAL_ANSWER"));
    }

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        let config = AgentConfig::new(2, "test-model");
        let loop_ = ReActLoop::new(config);

        let result = loop_
            .run("loop forever", |_ctx| async {
                "Thought: thinking\nAction: noop {}".to_string()
            })
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("max iterations"));
    }

    #[tokio::test]
    async fn test_parse_react_step_valid() {
        let text = "Thought: I should check\nAction: lookup {\"key\":\"val\"}";
        let step = parse_react_step(text).unwrap();
        assert_eq!(step.thought, "I should check");
        assert_eq!(step.action, "lookup {\"key\":\"val\"}");
    }

    #[tokio::test]
    async fn test_parse_react_step_empty_fails() {
        let result = parse_react_step("no prefix lines here");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tool_not_found_returns_error_observation() {
        let config = AgentConfig::new(3, "test-model");
        let loop_ = ReActLoop::new(config);

        let mut call_count = 0;
        let steps = loop_
            .run("test", |_ctx| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: try missing tool\nAction: missing_tool {}".to_string()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER done".to_string()
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(steps.len(), 2);
        assert!(steps[0].observation.contains("\"ok\":false"));
    }

    #[tokio::test]
    async fn test_new_async_tool_spec() {
        let spec = ToolSpec::new_async("async_tool", "An async tool", |args| {
            Box::pin(async move { serde_json::json!({"echo": args}) })
        });

        let result = spec.call(serde_json::json!({"input": "test"})).await;
        assert!(result.get("echo").is_some());
    }
}
