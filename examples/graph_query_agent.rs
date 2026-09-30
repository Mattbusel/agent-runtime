//! Example: Knowledge Graph Query Agent
//!
//! Builds a small knowledge graph, then runs an agent that uses graph
//! traversal tools (BFS / DFS / path-exists) to answer questions.

#![allow(clippy::print_stdout)]

use llm_agent_runtime::agent::{AgentConfig, ReActLoop, ToolSpec};
use llm_agent_runtime::error::AgentRuntimeError;
use llm_agent_runtime::graph::{Entity, EntityId, GraphStore, Relationship};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), AgentRuntimeError> {
    // Build the graph.
    let graph = Arc::new(GraphStore::new());
    graph.add_entity(Entity::new("paris", "City")).unwrap();
    graph.add_entity(Entity::new("france", "Country")).unwrap();
    graph.add_entity(Entity::new("europe", "Continent")).unwrap();
    graph
        .add_relationship(Relationship::new("paris", "france", "CAPITAL_OF", 1.0))
        .unwrap();
    graph
        .add_relationship(Relationship::new("france", "europe", "PART_OF", 1.0))
        .unwrap();

    let config = AgentConfig::new(5, "my-model");
    let mut loop_ = ReActLoop::new(config);

    // Register graph tools.
    let g = Arc::clone(&graph);
    loop_.register_tool(ToolSpec::new_async(
        "bfs",
        "BFS traversal from a node (arg: {\"start\": \"node_id\"})",
        move |args| {
            let g = Arc::clone(&g);
            Box::pin(async move {
                let start = args["start"].as_str().unwrap_or("");
                match g.bfs(&EntityId::new(start)) {
                    Ok(nodes) => serde_json::json!({ "nodes": nodes.iter().map(|n| n.0.clone()).collect::<Vec<_>>() }),
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                }
            })
        },
    ));

    let g2 = Arc::clone(&graph);
    loop_.register_tool(ToolSpec::new_async(
        "path_exists",
        "Check if path exists (args: {\"from\": \"a\", \"to\": \"b\"})",
        move |args| {
            let g2 = Arc::clone(&g2);
            Box::pin(async move {
                let from = args["from"].as_str().unwrap_or("");
                let to = args["to"].as_str().unwrap_or("");
                match g2.path_exists(from, to) {
                    Ok(exists) => serde_json::json!({ "exists": exists }),
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                }
            })
        },
    ));

    let steps = loop_
        .run(
            "Is Paris connected to Europe in the graph?",
            |_ctx: String| async {
                "Thought: I should check path_exists\nAction: path_exists {\"from\":\"paris\",\"to\":\"europe\"}".to_string()
            },
        )
        .await;

    match steps {
        Ok(s) => println!("Completed in {} steps", s.len()),
        Err(e) => println!("Error: {e}"),
    }

    Ok(())
}
