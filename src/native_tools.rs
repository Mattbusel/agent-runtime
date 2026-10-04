//! # Module: native_tools
//!
//! ## Responsibility
//! Native (structured) tool calling: the model receives your tools as JSON
//! Schema and answers with typed tool calls instead of text the runtime has
//! to parse. Anthropic `tool_use` blocks and OpenAI `tool_calls` are both
//! supported.
//!
//! ## How it fits
//! [`ReActLoop::run_native`](crate::agent::ReActLoop::run_native) and
//! [`AgentRuntime::run_agent_native`](crate::runtime::AgentRuntime::run_agent_native)
//! drive a [`ToolCallingModel`] through the same tool registry, required
//! fields, validators, circuit breakers, action hook, observer and metrics as
//! the text ReAct loop, and record the same [`ReActStep`](crate::agent::ReActStep)s,
//! so sessions, checkpoints and `final_answer()` work unchanged.
//!
//! ## Example
//! ```no_run
//! # #[cfg(feature = "anthropic")]
//! # async fn demo() -> Result<(), llm_agent_runtime::error::AgentRuntimeError> {
//! use llm_agent_runtime::prelude::*;
//! use llm_agent_runtime::providers::AnthropicProvider;
//!
//! let runtime = AgentRuntime::builder()
//!     .with_agent_config(AgentConfig::new(6, "claude-haiku-4-5"))
//!     .register_tool(
//!         ToolSpec::new("add", "Add two integers", |args| {
//!             serde_json::json!(args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0))
//!         })
//!         .with_input_schema(serde_json::json!({
//!             "type": "object",
//!             "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
//!             "required": ["a", "b"]
//!         })),
//!     )
//!     .build();
//!
//! let provider = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY").unwrap());
//! let session = runtime
//!     .run_agent_native(AgentId::new("calc"), "What is 1234 + 4321?", &provider)
//!     .await?;
//! println!("{:?}", session.final_answer());
//! # Ok(())
//! # }
//! ```

use crate::error::AgentRuntimeError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A tool as the model sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name the model calls.
    pub name: String,
    /// What the tool does; models choose tools from this text.
    pub description: String,
    /// JSON Schema of the arguments object.
    pub input_schema: Value,
}

/// One tool call requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id; the result must quote it back.
    pub id: String,
    /// Name of the tool to run.
    pub name: String,
    /// Arguments object.
    pub input: Value,
}

/// The result of one tool call, sent back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutcome {
    /// The [`ToolCall::id`] this answers.
    pub id: String,
    /// Result text (JSON for successful calls).
    pub content: String,
    /// Whether the tool failed.
    pub is_error: bool,
}

/// One turn of a tool-calling conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ChatTurn {
    /// Text from the user.
    User(String),
    /// What the model said and which tools it called.
    Assistant {
        /// Text the model produced alongside (or instead of) tool calls.
        text: String,
        /// Tool calls, in the order the model made them.
        tool_calls: Vec<ToolCall>,
    },
    /// Results for every tool call of the previous assistant turn.
    ToolResults(Vec<ToolOutcome>),
}

/// What the model returned for one request.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelTurn {
    /// Text output (may be empty when the model only calls tools).
    pub text: String,
    /// Tool calls; empty means the model is done.
    pub tool_calls: Vec<ToolCall>,
}

/// A model that supports native tool calling.
///
/// Implemented for [`AnthropicProvider`](crate::providers::AnthropicProvider)
/// and [`OpenAiProvider`](crate::providers::OpenAiProvider) (which also covers
/// OpenAI-compatible servers such as vLLM, Ollama and LM Studio). Implement it
/// yourself for other backends, or for scripted models in tests.
#[async_trait]
pub trait ToolCallingModel: Send + Sync {
    /// Send the conversation and tools; return the model's next turn.
    async fn respond(
        &self,
        model: &str,
        system: &str,
        turns: &[ChatTurn],
        tools: &[ToolDefinition],
    ) -> Result<ModelTurn, AgentRuntimeError>;
}

// ── Anthropic Messages API ───────────────────────────────────────────────────

/// Build an Anthropic `/v1/messages` request body.
pub fn anthropic_request(
    model: &str,
    max_tokens: u32,
    system: &str,
    turns: &[ChatTurn],
    tools: &[ToolDefinition],
) -> Value {
    let messages: Vec<Value> = turns
        .iter()
        .map(|turn| match turn {
            ChatTurn::User(text) => json!({ "role": "user", "content": text }),
            ChatTurn::Assistant { text, tool_calls } => {
                let mut content = Vec::new();
                if !text.is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
                for c in tool_calls {
                    content.push(json!({
                        "type": "tool_use", "id": c.id, "name": c.name, "input": c.input
                    }));
                }
                json!({ "role": "assistant", "content": content })
            }
            ChatTurn::ToolResults(results) => json!({
                "role": "user",
                "content": results.iter().map(|r| json!({
                    "type": "tool_result",
                    "tool_use_id": r.id,
                    "content": r.content,
                    "is_error": r.is_error,
                })).collect::<Vec<_>>()
            }),
        })
        .collect();
    let mut body = json!({ "model": model, "max_tokens": max_tokens, "messages": messages });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if !tools.is_empty() {
        body["tools"] = tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.input_schema }))
            .collect();
    }
    body
}

/// Parse an Anthropic `/v1/messages` response.
pub fn parse_anthropic_response(body: &Value) -> Result<ModelTurn, AgentRuntimeError> {
    let blocks = body["content"].as_array().ok_or_else(|| {
        AgentRuntimeError::Provider(format!("Anthropic response has no content array: {body}"))
    })?;
    let mut turn = ModelTurn::default();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => turn.text.push_str(block["text"].as_str().unwrap_or("")),
            Some("tool_use") => turn.tool_calls.push(ToolCall {
                id: block["id"].as_str().unwrap_or_default().to_owned(),
                name: block["name"].as_str().unwrap_or_default().to_owned(),
                input: block["input"].clone(),
            }),
            _ => {} // thinking and other block types are not part of the answer
        }
    }
    Ok(turn)
}

// ── OpenAI Chat Completions API ──────────────────────────────────────────────

/// Build an OpenAI `/chat/completions` request body.
pub fn openai_request(
    model: &str,
    system: &str,
    turns: &[ChatTurn],
    tools: &[ToolDefinition],
) -> Value {
    let mut messages = Vec::new();
    if !system.is_empty() {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for turn in turns {
        match turn {
            ChatTurn::User(text) => messages.push(json!({ "role": "user", "content": text })),
            ChatTurn::Assistant { text, tool_calls } => {
                let mut msg = json!({
                    "role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { json!(text) },
                });
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = tool_calls
                        .iter()
                        .map(|c| json!({
                            "id": c.id,
                            "type": "function",
                            "function": { "name": c.name, "arguments": c.input.to_string() }
                        }))
                        .collect();
                }
                messages.push(msg);
            }
            ChatTurn::ToolResults(results) => {
                for r in results {
                    messages.push(json!({ "role": "tool", "tool_call_id": r.id, "content": r.content }));
                }
            }
        }
    }
    let mut body = json!({ "model": model, "messages": messages });
    if !tools.is_empty() {
        body["tools"] = tools
            .iter()
            .map(|t| json!({
                "type": "function",
                "function": { "name": t.name, "description": t.description, "parameters": t.input_schema }
            }))
            .collect();
    }
    body
}

/// Parse an OpenAI `/chat/completions` response.
pub fn parse_openai_response(body: &Value) -> Result<ModelTurn, AgentRuntimeError> {
    let message = &body["choices"][0]["message"];
    if message.is_null() {
        return Err(AgentRuntimeError::Provider(format!(
            "OpenAI response has no choices[0].message: {body}"
        )));
    }
    let mut turn = ModelTurn {
        text: message["content"].as_str().unwrap_or("").to_owned(),
        tool_calls: Vec::new(),
    };
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        let raw = call["function"]["arguments"].as_str().unwrap_or("");
        // Some OpenAI-compatible servers send "" for a call with no arguments.
        let input = if raw.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(raw).map_err(|e| {
                AgentRuntimeError::Provider(format!("tool call arguments are not JSON ({e}): {raw}"))
            })?
        };
        turn.tool_calls.push(ToolCall {
            id: call["id"].as_str().unwrap_or_default().to_owned(),
            name: call["function"]["name"].as_str().unwrap_or_default().to_owned(),
            input,
        });
    }
    Ok(turn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_tool() -> ToolDefinition {
        ToolDefinition {
            name: "add".into(),
            description: "Add two integers".into(),
            input_schema: json!({ "type": "object", "properties": { "a": {}, "b": {} } }),
        }
    }

    fn conversation() -> Vec<ChatTurn> {
        vec![
            ChatTurn::User("1+2?".into()),
            ChatTurn::Assistant {
                text: "Adding.".into(),
                tool_calls: vec![ToolCall { id: "c1".into(), name: "add".into(), input: json!({ "a": 1, "b": 2 }) }],
            },
            ChatTurn::ToolResults(vec![ToolOutcome { id: "c1".into(), content: "3".into(), is_error: false }]),
        ]
    }

    #[test]
    fn anthropic_request_uses_tool_use_and_tool_result_blocks() {
        let body = anthropic_request("m", 512, "be brief", &conversation(), &[add_tool()]);
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(body["messages"][1]["content"][1]["input"]["a"], 1);
        let result = &body["messages"][2]["content"][0];
        assert_eq!(result["type"], "tool_result");
        assert_eq!(result["tool_use_id"], "c1");
        assert_eq!(body["messages"][2]["role"], "user");
    }

    #[test]
    fn anthropic_response_collects_text_and_tool_calls() {
        let body = json!({ "content": [
            { "type": "thinking", "thinking": "..." },
            { "type": "text", "text": "Let me add." },
            { "type": "tool_use", "id": "toolu_1", "name": "add", "input": { "a": 2, "b": 3 } }
        ]});
        let turn = parse_anthropic_response(&body).unwrap();
        assert_eq!(turn.text, "Let me add.");
        assert_eq!(turn.tool_calls[0].input, json!({ "a": 2, "b": 3 }));
    }

    #[test]
    fn openai_request_uses_function_tools_and_tool_messages() {
        let body = openai_request("m", "be brief", &conversation(), &[add_tool()]);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["tools"][0]["function"]["parameters"]["type"], "object");
        let call = &body["messages"][2]["tool_calls"][0];
        assert_eq!(call["function"]["arguments"], "{\"a\":1,\"b\":2}");
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_call_id"], "c1");
    }

    #[test]
    fn openai_response_parses_argument_strings() {
        let body = json!({ "choices": [{ "message": {
            "content": null,
            "tool_calls": [{ "id": "call_1", "type": "function",
                "function": { "name": "add", "arguments": "{\"a\":4,\"b\":5}" } }]
        }}]});
        let turn = parse_openai_response(&body).unwrap();
        assert_eq!(turn.text, "");
        assert_eq!(turn.tool_calls[0].name, "add");
        assert_eq!(turn.tool_calls[0].input["b"], 5);
        let bad = json!({ "choices": [{ "message": { "tool_calls": [{ "id": "x",
            "function": { "name": "add", "arguments": "{not json" } }] } }] });
        assert!(parse_openai_response(&bad).is_err());
        let empty = json!({ "choices": [{ "message": { "tool_calls": [{ "id": "y",
            "function": { "name": "now", "arguments": "" } }] } }] });
        assert_eq!(parse_openai_response(&empty).unwrap().tool_calls[0].input, json!({}));
    }
}
