//! Native tool calling: the loop logic against a scripted model, plus
//! opt-in live tests against real APIs.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use llm_agent_runtime::error::AgentRuntimeError;
use llm_agent_runtime::native_tools::{
    ChatTurn, ModelTurn, ToolCall, ToolCallingModel, ToolDefinition,
};
use llm_agent_runtime::prelude::*;
use serde_json::json;

/// Plays back fixed turns and records what it was sent.
struct Scripted {
    replies: Mutex<Vec<ModelTurn>>,
    seen: Mutex<Vec<(Vec<ChatTurn>, Vec<ToolDefinition>)>>,
}

impl Scripted {
    fn new(mut replies: Vec<ModelTurn>) -> Arc<Self> {
        replies.reverse();
        Arc::new(Self { replies: Mutex::new(replies), seen: Mutex::new(Vec::new()) })
    }
}

#[async_trait]
impl ToolCallingModel for Scripted {
    async fn respond(
        &self,
        _model: &str,
        _system: &str,
        turns: &[ChatTurn],
        tools: &[ToolDefinition],
    ) -> Result<ModelTurn, AgentRuntimeError> {
        self.seen.lock().unwrap().push((turns.to_vec(), tools.to_vec()));
        Ok(self.replies.lock().unwrap().pop().unwrap_or_default())
    }
}

fn call(id: &str, name: &str, input: serde_json::Value) -> ToolCall {
    ToolCall { id: id.into(), name: name.into(), input }
}

fn add_tool() -> ToolSpec {
    ToolSpec::new("add", "Add two integers", |args| {
        json!(args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0))
    })
    .with_input_schema(json!({
        "type": "object",
        "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
        "required": ["a", "b"]
    }))
}

fn runtime_with(tools: Vec<ToolSpec>) -> AgentRuntime {
    AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model").with_system_prompt("Use tools."))
        .register_tools(tools)
        .build()
}

#[tokio::test]
async fn tool_result_goes_back_to_the_model_and_final_text_is_the_answer() {
    let model = Scripted::new(vec![
        ModelTurn { text: "Adding.".into(), tool_calls: vec![call("c1", "add", json!({ "a": 2, "b": 3 }))] },
        ModelTurn { text: "The sum is 5.".into(), tool_calls: vec![] },
    ]);
    let rt = runtime_with(vec![add_tool()]);
    let session = rt.run_agent_native(AgentId::new("a"), "2+3?", model.as_ref()).await.unwrap();

    assert_eq!(session.final_answer().as_deref(), Some("The sum is 5."));
    assert_eq!(session.steps.len(), 2);
    assert_eq!(session.steps[0].action, "add {\"a\":2,\"b\":3}");
    assert_eq!(session.steps[0].observation, "5");

    let seen = model.seen.lock().unwrap();
    let (first_turns, tools) = &seen[0];
    assert_eq!(first_turns, &vec![ChatTurn::User("2+3?".into())]);
    assert_eq!(tools[0].input_schema["required"], json!(["a", "b"]));
    let (second_turns, _) = &seen[1];
    match &second_turns[2] {
        ChatTurn::ToolResults(results) => {
            assert_eq!(results[0].id, "c1");
            assert_eq!(results[0].content, "5");
            assert!(!results[0].is_error);
        }
        other => panic!("expected tool results, got {other:?}"),
    }
}

#[tokio::test]
async fn several_calls_in_one_turn_are_answered_together_and_errors_are_reported() {
    let model = Scripted::new(vec![
        ModelTurn {
            text: String::new(),
            tool_calls: vec![
                call("c1", "add", json!({ "a": 1, "b": 1 })),
                call("c2", "nope", json!({})),
                call("c3", "add", json!({ "a": 1 })), // missing required "b"
            ],
        },
        ModelTurn { text: "done".into(), tool_calls: vec![] },
    ]);
    let rt = runtime_with(vec![add_tool().with_required_fields(["a", "b"])]);
    let session = rt.run_agent_native(AgentId::new("a"), "go", model.as_ref()).await.unwrap();
    assert_eq!(session.steps.len(), 4);

    let seen = model.seen.lock().unwrap();
    let ChatTurn::ToolResults(results) = &seen[1].0[2] else { panic!("no tool results") };
    assert_eq!(results.len(), 3);
    assert_eq!((results[0].is_error, results[0].content.as_str()), (false, "2"));
    assert!(results[1].is_error && results[1].content.contains("not found"));
    assert!(results[2].is_error && results[2].content.contains("missing required field 'b'"));
}

#[tokio::test]
async fn a_model_that_never_stops_hits_max_iterations() {
    let looping: Vec<ModelTurn> = (0..10)
        .map(|i| ModelTurn { text: String::new(), tool_calls: vec![call(&format!("c{i}"), "add", json!({ "a": 1, "b": 1 }))] })
        .collect();
    let model = Scripted::new(looping);
    let rt = runtime_with(vec![add_tool()]);
    let err = rt.run_agent_native(AgentId::new("a"), "go", model.as_ref()).await.unwrap_err();
    assert!(err.to_string().contains("max iterations (5)"), "{err}");
}

#[tokio::test]
async fn arguments_are_checked_against_the_schema_before_the_tool_runs() {
    let registry = ToolRegistry::new().with_tool(add_tool());
    let err = registry
        .call("add", json!({ "a": "get_secret_number()", "b": 1389 }))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("/a"), "{err}");
    assert!(err.contains("integer"), "{err}");
    let missing = registry.call("add", json!({ "a": 1 })).await.unwrap_err().to_string();
    assert!(missing.contains("\"b\" is a required property"), "{missing}");
    assert_eq!(registry.call("add", json!({ "a": 1, "b": 2 })).await.unwrap(), json!(3));
}

#[tokio::test]
async fn tools_without_a_schema_get_one_from_required_fields() {
    let spec = ToolSpec::new("lookup", "Look up a word", |a| a).with_required_fields(["word"]);
    let def = spec.definition();
    assert_eq!(def.input_schema["type"], "object");
    assert_eq!(def.input_schema["required"], json!(["word"]));
    assert!(def.input_schema["properties"]["word"].is_object());
}

// ── Live tests (run with --ignored and a key in the environment) ─────────────

/// A tool whose answer the model cannot guess, so a right answer proves the
/// tool really ran.
fn secret_tool() -> ToolSpec {
    ToolSpec::new("get_secret_number", "Returns today's secret number", |_| json!(48_611))
        .with_input_schema(json!({ "type": "object", "properties": {} }))
}

async fn live(model: &dyn ToolCallingModel, name: &str) {
    let rt = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(6, name).with_system_prompt("Use the tools you are given. Answer with just the number."))
        .register_tool(secret_tool())
        .register_tool(add_tool())
        .build();
    let session = rt
        .run_agent_native(AgentId::new("live"), "Get the secret number, then add 1389 to it with the add tool.", model)
        .await
        .unwrap();
    for s in &session.steps {
        println!("{} => {}", s.action, s.observation);
    }
    let answer = session.final_answer().unwrap();
    assert!(answer.replace(',', "").contains("50000"), "answer: {answer}");
    assert!(session.steps.iter().any(|s| s.action.starts_with("add ")), "add tool used");
}

/// `HF_TOKEN=... cargo test --features openai --test native_tools live_openai_compatible -- --ignored`
#[cfg(feature = "openai")]
#[tokio::test]
#[ignore]
async fn live_openai_compatible_hugging_face_router() {
    let key = std::env::var("HF_TOKEN").expect("HF_TOKEN");
    let model = std::env::var("HF_MODEL").unwrap_or_else(|_| "Qwen/Qwen2.5-72B-Instruct".into());
    let provider = llm_agent_runtime::providers::OpenAiProvider::with_base_url(key, "https://router.huggingface.co/v1");
    live(&provider, &model).await;
}

/// `OPENAI_API_KEY=... cargo test --features openai --test native_tools live_openai -- --ignored`
#[cfg(feature = "openai")]
#[tokio::test]
#[ignore]
async fn live_openai() {
    let provider = llm_agent_runtime::providers::OpenAiProvider::new(std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY"));
    live(&provider, "gpt-4o-mini").await;
}

/// `ANTHROPIC_API_KEY=... cargo test --features anthropic --test native_tools live_anthropic -- --ignored`
#[cfg(feature = "anthropic")]
#[tokio::test]
#[ignore]
async fn live_anthropic() {
    let provider = llm_agent_runtime::providers::AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY"));
    live(&provider, "claude-haiku-4-5").await;
}
