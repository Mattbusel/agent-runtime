//! # Module: mcp
//!
//! ## Responsibility
//! Use the tools of any [Model Context Protocol](https://modelcontextprotocol.io)
//! server as agent tools, through the official Rust SDK ([`rmcp`]).
//!
//! Each MCP tool becomes a [`ToolSpec`] carrying the server's own JSON Schema,
//! so it works in the text ReAct loop and in native tool calling
//! ([`AgentRuntime::run_agent_native`](crate::runtime::AgentRuntime::run_agent_native)),
//! with schema validation, circuit breakers, metrics and tracing like any
//! other tool.
//!
//! Requires the `mcp` feature (Rust 1.88 or newer, the SDK's minimum).
//!
//! ## Example
//! ```no_run
//! # async fn demo() -> Result<(), llm_agent_runtime::error::AgentRuntimeError> {
//! use llm_agent_runtime::mcp::McpClient;
//! use llm_agent_runtime::prelude::*;
//!
//! // A local server over stdio...
//! let files = McpClient::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "."]).await?;
//! // ...or a remote one over streamable HTTP.
//! // let remote = McpClient::connect_http("https://example.com/mcp").await?;
//!
//! let runtime = AgentRuntime::builder()
//!     .with_agent_config(AgentConfig::new(8, "claude-haiku-4-5"))
//!     .register_tools(files.tools().await?)
//!     .build();
//! # let _ = runtime;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::ServiceExt;
use serde_json::{json, Value};

use crate::agent::ToolSpec;
use crate::error::AgentRuntimeError;

fn mcp_err(context: &str, e: impl std::fmt::Display) -> AgentRuntimeError {
    AgentRuntimeError::Provider(format!("MCP {context}: {e}"))
}

/// A connection to one MCP server. Cheap to clone; the connection closes
/// when the last clone (including the tools made from it) is dropped, or on
/// [`shutdown`](Self::shutdown).
#[derive(Clone)]
pub struct McpClient {
    service: Arc<RunningService<RoleClient, ()>>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient").finish_non_exhaustive()
    }
}

impl McpClient {
    /// Start `program` with `args` and talk to it over stdio (the usual way
    /// to run local MCP servers).
    ///
    /// # Errors
    /// The program cannot be started or the MCP handshake fails.
    pub async fn spawn(program: &str, args: &[&str]) -> Result<Self, AgentRuntimeError> {
        let mut command = tokio::process::Command::new(program);
        command.args(args);
        let transport = rmcp::transport::TokioChildProcess::new(command)
            .map_err(|e| mcp_err(&format!("could not start '{program}'"), e))?;
        Self::from_service(().serve(transport).await.map_err(|e| mcp_err("handshake", e))?)
    }

    /// Connect to a remote server over streamable HTTP.
    ///
    /// # Errors
    /// The server is unreachable or the MCP handshake fails.
    pub async fn connect_http(url: &str) -> Result<Self, AgentRuntimeError> {
        let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(url.to_owned());
        Self::from_service(().serve(transport).await.map_err(|e| mcp_err("handshake", e))?)
    }

    /// Wrap an already running rmcp client (any transport).
    pub fn from_service(service: RunningService<RoleClient, ()>) -> Result<Self, AgentRuntimeError> {
        Ok(Self { service: Arc::new(service) })
    }

    /// The server's name and version from the handshake, if it sent them.
    pub fn server_name(&self) -> Option<String> {
        self.service
            .peer_info()
            .and_then(|info| info.server_info.clone())
            .map(|server| format!("{} {}", server.name, server.version))
    }

    /// Every tool the server offers, ready to register on an agent.
    ///
    /// Each [`ToolSpec`] keeps the server's description and input schema; its
    /// handler calls the server. A call the server reports as failed comes
    /// back as `{"ok": false, "error": ...}`, the crate's usual tool-error
    /// shape, so the model sees what went wrong.
    ///
    /// # Errors
    /// Listing the tools fails.
    pub async fn tools(&self) -> Result<Vec<ToolSpec>, AgentRuntimeError> {
        let listed = self
            .service
            .list_all_tools()
            .await
            .map_err(|e| mcp_err("tools/list", e))?;
        Ok(listed
            .into_iter()
            .map(|tool| {
                let name = tool.name.to_string();
                let client = self.clone();
                let tool_name = name.clone();
                ToolSpec::new_async_fallible(
                    name,
                    tool.description.as_deref().unwrap_or_default().to_owned(),
                    move |args| {
                        let client = client.clone();
                        let tool_name = tool_name.clone();
                        Box::pin(async move { client.call(&tool_name, args).await })
                    },
                )
                .with_input_schema(Value::Object((*tool.input_schema).clone()))
            })
            .collect())
    }

    /// Call one tool directly.
    ///
    /// Returns the tool's structured content when it has any; otherwise its
    /// text (parsed as JSON when it is JSON); otherwise the raw content
    /// blocks. A failed call is `Err` with the server's message.
    pub async fn call(&self, name: &str, args: Value) -> Result<Value, String> {
        let mut params = CallToolRequestParams::new(name.to_owned());
        match args {
            Value::Object(map) => params = params.with_arguments(map),
            Value::Null => {}
            other => return Err(format!("MCP tool arguments must be an object, got {other}")),
        }
        let result = self
            .service
            .call_tool(params)
            .await
            .map_err(|e| format!("MCP tools/call '{name}' failed: {e}"))?;
        let value = result_value(&result);
        if result.is_error == Some(true) {
            Err(match value {
                Value::String(s) => s,
                other => other.to_string(),
            })
        } else {
            Ok(value)
        }
    }

    /// Close the connection (and stop a spawned server).
    pub async fn shutdown(self) {
        if let Ok(service) = Arc::try_unwrap(self.service) {
            let _ = service.cancel().await;
        }
    }
}

fn result_value(result: &CallToolResult) -> Value {
    if let Some(structured) = &result.structured_content {
        return structured.clone();
    }
    let texts: Vec<&str> = result
        .content
        .iter()
        .filter_map(|block| block.as_text().map(|t| t.text.as_str()))
        .collect();
    if texts.len() == result.content.len() && !texts.is_empty() {
        let joined = texts.join("\n");
        return serde_json::from_str(&joined).unwrap_or(Value::String(joined));
    }
    json!(result.content)
}
