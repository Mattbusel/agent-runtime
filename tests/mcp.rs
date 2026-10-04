//! MCP tools as agent tools, against a real rmcp server running in process.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use llm_agent_runtime::error::AgentRuntimeError;
use llm_agent_runtime::mcp::McpClient;
use llm_agent_runtime::native_tools::{ChatTurn, ModelTurn, ToolCall, ToolCallingModel, ToolDefinition};
use llm_agent_runtime::prelude::*;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, schemars::JsonSchema)]
struct AddArgs {
    /// First number
    a: i64,
    /// Second number
    b: i64,
}

#[derive(Clone)]
struct MathServer {
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl MathServer {
    fn new() -> Self {
        Self { tool_router: Self::tool_router() }
    }

    /// Add two integers
    #[tool(name = "add")]
    async fn add(&self, Parameters(args): Parameters<AddArgs>) -> Result<String, String> {
        Ok((args.a + args.b).to_string())
    }

    /// Always fails
    #[tool(name = "explode")]
    async fn explode(&self) -> Result<String, String> {
        Err("the reactor is offline".into())
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MathServer {}

async fn connect() -> McpClient {
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let server = MathServer::new().serve(server_io).await.expect("server starts");
        let _ = server.waiting().await;
    });
    McpClient::from_service(().serve(client_io).await.expect("handshake")).unwrap()
}

#[tokio::test]
async fn mcp_tools_keep_their_names_descriptions_and_schemas() {
    let client = connect().await;
    let tools = client.tools().await.unwrap();
    let add = tools.iter().find(|t| t.name == "add").expect("add tool");
    assert_eq!(add.description, "Add two integers");
    let schema = add.input_schema.as_ref().expect("schema from the server");
    assert_eq!(schema["properties"]["a"]["type"], "integer");
    assert!(tools.iter().any(|t| t.name == "explode"));
}

#[tokio::test]
async fn calls_go_to_the_server_and_failures_come_back_as_tool_errors() {
    let client = connect().await;
    let mut registry = ToolRegistry::new();
    registry.register_tools(client.tools().await.unwrap());

    assert_eq!(registry.call("add", json!({ "a": 20, "b": 22 })).await.unwrap(), json!(42));

    // The server's schema is enforced before the call leaves the process.
    let err = registry.call("add", json!({ "a": "twenty", "b": 22 })).await.unwrap_err();
    assert!(err.to_string().contains("integer"), "{err}");

    let failed = registry.call("explode", json!({})).await.unwrap();
    assert_eq!(failed["ok"], false);
    assert!(failed["error"].as_str().unwrap().contains("reactor is offline"));
}

/// Calls `add` on the first turn, then answers with the tool's result.
struct AddThenAnswer {
    seen_tools: Mutex<Vec<ToolDefinition>>,
}

#[async_trait]
impl ToolCallingModel for AddThenAnswer {
    async fn respond(
        &self,
        _model: &str,
        _system: &str,
        turns: &[ChatTurn],
        tools: &[ToolDefinition],
    ) -> Result<ModelTurn, AgentRuntimeError> {
        *self.seen_tools.lock().unwrap() = tools.to_vec();
        match turns.last() {
            Some(ChatTurn::ToolResults(results)) => Ok(ModelTurn {
                text: format!("It is {}.", results[0].content),
                tool_calls: vec![],
            }),
            _ => Ok(ModelTurn {
                text: String::new(),
                tool_calls: vec![ToolCall { id: "1".into(), name: "add".into(), input: json!({ "a": 1300, "b": 37 }) }],
            }),
        }
    }
}

#[tokio::test]
async fn an_agent_uses_mcp_tools_through_native_tool_calling() {
    let client = connect().await;
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(4, "scripted"))
        .register_tools(client.tools().await.unwrap())
        .build();
    let model = Arc::new(AddThenAnswer { seen_tools: Mutex::new(vec![]) });
    let session = runtime
        .run_agent_native(AgentId::new("mcp"), "1300 + 37?", model.as_ref())
        .await
        .unwrap();
    assert_eq!(session.final_answer().as_deref(), Some("It is 1337."));
    let tools = model.seen_tools.lock().unwrap();
    assert!(tools.iter().any(|t| t.name == "add" && t.input_schema["required"] == json!(["a", "b"])));
}

/// Spawns the reference MCP server over stdio (needs Node.js and network):
/// `cargo test --features mcp --test mcp stdio_ -- --ignored`
#[tokio::test]
#[ignore]
async fn stdio_reference_server_tools_work() {
    let npx = if cfg!(windows) { "npx.cmd" } else { "npx" };
    let client = McpClient::spawn(npx, &["-y", "@modelcontextprotocol/server-everything"]).await.unwrap();
    println!("server: {:?}", client.server_name());
    let mut registry = ToolRegistry::new();
    registry.register_tools(client.tools().await.unwrap());
    println!("tools: {:?}", registry.tool_names());
    let echoed = registry.call("echo", json!({ "message": "hello from llm-agent-runtime" })).await.unwrap();
    println!("echo => {echoed}");
    assert!(echoed.to_string().contains("hello from llm-agent-runtime"));
    let sum = registry.call("get-sum", json!({ "a": 40, "b": 2 })).await
        .or(registry.call("add", json!({ "a": 40, "b": 2 })).await).unwrap();
    println!("sum => {sum}");
    assert!(sum.to_string().contains("42"));
    client.shutdown().await;
}
