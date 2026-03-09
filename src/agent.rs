//! # Module: Agent
//!
//! ## Responsibility
//! Provides a ReAct (Thought-Action-Observation) agent loop with pluggable tools.
//! Mirrors the public API of `wasm-agent`.
//!
//! ## Guarantees
//! - Deterministic: the loop terminates after at most `max_iterations` cycles
//! - Non-panicking: all operations return `Result`
//! - Tool handlers are synchronous `Fn` closures
//!
//! ## NOT Responsible For
//! - Actual LLM inference (callers supply a mock/stub inference fn)
//! - WASM compilation or browser execution
//! - Streaming partial responses

use crate::error::AgentRuntimeError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Types ─────────────────────────────────────────────────────────────────────

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
        Self { role, content: content.into() }
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
}

impl AgentConfig {
    /// Create a new config with sensible defaults.
    pub fn new(max_iterations: usize, model: impl Into<String>) -> Self {
        Self {
            max_iterations,
            model: model.into(),
            system_prompt: "You are a helpful AI agent.".into(),
        }
    }

    /// Override the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
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
    /// Synchronous handler: receives JSON arguments, returns JSON result.
    pub handler: Box<dyn Fn(Value) -> Value + Send + Sync>,
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
    /// Construct a new `ToolSpec`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(Value) -> Value + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(handler),
        }
    }

    /// Invoke the tool with the given JSON arguments.
    pub fn call(&self, args: Value) -> Value {
        (self.handler)(args)
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
        Self { tools: HashMap::new() }
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
    pub fn call(&self, name: &str, args: Value) -> Result<Value, AgentRuntimeError> {
        let spec = self.tools.get(name).ok_or_else(|| {
            AgentRuntimeError::AgentLoop(format!("tool '{name}' not found"))
        })?;
        Ok(spec.call(args))
    }

    /// Return the list of registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

// ── ReActLoop ─────────────────────────────────────────────────────────────────

/// Parses a ReAct response string into a `ReActStep`.
///
/// Expected format (line-based):
/// ```text
/// Thought: <reasoning>
/// Action: <tool_name> <JSON args or plain text>
/// ```
///
/// The `observation` field is filled in by the loop after tool invocation.
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
///
/// Drives a Thought → Action → Observation cycle using a pluggable inference
/// function and a `ToolRegistry`. The loop halts when the action is
/// `FINAL_ANSWER` or `max_iterations` is reached.
pub struct ReActLoop {
    config: AgentConfig,
    registry: ToolRegistry,
}

impl ReActLoop {
    /// Create a new loop with the given config.
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            registry: ToolRegistry::new(),
        }
    }

    /// Register a tool with the loop.
    pub fn register_tool(&mut self, spec: ToolSpec) {
        self.registry.register(spec);
    }

    /// Run the agent loop for the given `prompt`.
    ///
    /// # Arguments
    /// * `prompt` — the initial user prompt
    /// * `infer` — callable that simulates LLM inference:
    ///   takes the current conversation as a string, returns a ReAct-formatted response
    ///
    /// # Returns
    /// The list of `ReActStep`s produced during the session.
    pub fn run(
        &self,
        prompt: &str,
        infer: impl Fn(&str) -> String,
    ) -> Result<Vec<ReActStep>, AgentRuntimeError> {
        let mut steps: Vec<ReActStep> = Vec::new();
        let mut context = format!(
            "{}\n\nUser: {}\n",
            self.config.system_prompt, prompt
        );

        for _iteration in 0..self.config.max_iterations {
            let response = infer(&context);

            let mut step = parse_react_step(&response)?;

            // Check for terminal action
            if step.action.starts_with("FINAL_ANSWER") {
                step.observation = step.action.clone();
                steps.push(step);
                return Ok(steps);
            }

            // Parse tool name and arguments
            let (tool_name, args) = parse_tool_call(&step.action);
            let observation = match self.registry.call(&tool_name, args) {
                Ok(result) => result.to_string(),
                Err(e) => format!("Error: {e}"),
            };

            step.observation = observation.clone();
            context.push_str(&format!(
                "\nThought: {}\nAction: {}\nObservation: {}\n",
                step.thought, step.action, observation
            ));
            steps.push(step);
        }

        // max_iterations reached without FINAL_ANSWER
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

// ── AgentError (mirrors upstream) ────────────────────────────────────────────

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

    // ── Types ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_message_new_stores_role_and_content() {
        let m = Message::new(Role::User, "hello");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content, "hello");
    }

    #[test]
    fn test_agent_config_new_sets_max_iterations() {
        let cfg = AgentConfig::new(5, "gpt-4");
        assert_eq!(cfg.max_iterations, 5);
    }

    #[test]
    fn test_agent_config_with_system_prompt_overrides_default() {
        let cfg = AgentConfig::new(5, "gpt-4").with_system_prompt("custom");
        assert_eq!(cfg.system_prompt, "custom");
    }

    // ── ToolSpec / ToolRegistry ───────────────────────────────────────────────

    #[test]
    fn test_tool_spec_call_invokes_handler() {
        let spec = ToolSpec::new("echo", "echoes input", |v| v);
        let result = spec.call(Value::String("hi".into()));
        assert_eq!(result, Value::String("hi".into()));
    }

    #[test]
    fn test_tool_registry_register_and_call() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("add", "adds 1", |_| Value::Number(42.into())));
        let result = reg.call("add", Value::Null).unwrap();
        assert_eq!(result, Value::Number(42.into()));
    }

    #[test]
    fn test_tool_registry_missing_tool_returns_error() {
        let reg = ToolRegistry::new();
        assert!(reg.call("ghost", Value::Null).is_err());
    }

    #[test]
    fn test_tool_registry_tool_names_lists_registered() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("t1", "d", |v| v));
        reg.register(ToolSpec::new("t2", "d", |v| v));
        let names = reg.tool_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_tool_registry_overwrite_existing_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec::new("calc", "v1", |_| Value::Number(1.into())));
        reg.register(ToolSpec::new("calc", "v2", |_| Value::Number(2.into())));
        let result = reg.call("calc", Value::Null).unwrap();
        assert_eq!(result, Value::Number(2.into()));
    }

    // ── parse_react_step ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_react_step_extracts_thought_and_action() {
        let text = "Thought: I need to search\nAction: search {\"q\":\"rust\"}";
        let step = parse_react_step(text).unwrap();
        assert_eq!(step.thought, "I need to search");
        assert_eq!(step.action, "search {\"q\":\"rust\"}");
    }

    #[test]
    fn test_parse_react_step_empty_text_returns_error() {
        assert!(parse_react_step("").is_err());
    }

    #[test]
    fn test_parse_react_step_missing_fields_partial_parse() {
        let text = "Thought: only thought";
        let step = parse_react_step(text).unwrap();
        assert_eq!(step.thought, "only thought");
        assert!(step.action.is_empty());
    }

    // ── ReActLoop ─────────────────────────────────────────────────────────────

    #[test]
    fn test_react_loop_runs_tool_and_returns_steps() {
        let cfg = AgentConfig::new(5, "test-model");
        let mut loop_ = ReActLoop::new(cfg);

        loop_.register_tool(ToolSpec::new("greet", "greets", |_| {
            Value::String("Hello, World!".into())
        }));

        let mut call_count = 0;
        let steps = loop_.run("Say hello", |_ctx| {
            call_count += 1;
            if call_count == 1 {
                "Thought: I will greet\nAction: greet {}".into()
            } else {
                "Thought: done\nAction: FINAL_ANSWER done".into()
            }
        }).unwrap();

        assert!(!steps.is_empty());
        assert_eq!(steps[0].thought, "I will greet");
        assert!(steps[0].observation.contains("Hello, World!"));
    }

    #[test]
    fn test_react_loop_final_answer_terminates_early() {
        let cfg = AgentConfig::new(10, "model");
        let loop_ = ReActLoop::new(cfg);

        let steps = loop_.run("prompt", |_| {
            "Thought: done\nAction: FINAL_ANSWER 42".into()
        }).unwrap();

        assert_eq!(steps.len(), 1);
    }

    #[test]
    fn test_react_loop_max_iterations_respected() {
        let cfg = AgentConfig::new(3, "model");
        let mut loop_ = ReActLoop::new(cfg);
        loop_.register_tool(ToolSpec::new("noop", "does nothing", |_| Value::Null));

        let result = loop_.run("prompt", |_| {
            "Thought: keep going\nAction: noop {}".into()
        });

        assert!(result.is_err());
        if let Err(AgentRuntimeError::AgentLoop(msg)) = result {
            assert!(msg.contains("max iterations"));
        }
    }

    #[test]
    fn test_react_loop_unknown_tool_records_error_observation() {
        let cfg = AgentConfig::new(5, "model");
        let loop_ = ReActLoop::new(cfg);

        let mut count = 0;
        let steps = loop_.run("prompt", |_| {
            count += 1;
            if count == 1 {
                "Thought: try ghost\nAction: ghost {}".into()
            } else {
                "Thought: done\nAction: FINAL_ANSWER ok".into()
            }
        }).unwrap();

        assert!(steps[0].observation.contains("Error"));
    }

    // ── AgentError conversion ─────────────────────────────────────────────────

    #[test]
    fn test_agent_error_tool_not_found_display() {
        let e = AgentError::ToolNotFound("search".into());
        assert_eq!(e.to_string(), "Tool 'search' not found");
    }

    #[test]
    fn test_agent_error_max_iterations_display() {
        let e = AgentError::MaxIterations(10);
        assert_eq!(e.to_string(), "Max iterations exceeded: 10");
    }

    #[test]
    fn test_agent_error_converts_to_runtime_error() {
        let e = AgentError::ToolNotFound("x".into());
        let re: AgentRuntimeError = e.into();
        assert!(matches!(re, AgentRuntimeError::AgentLoop(_)));
    }
}
