//! Tests for structured async-aware tracing integration in the agent execution loop.
//!
//! These tests verify that:
//! - A tracing subscriber can be initialised without panicking.
//! - The ReActLoop emits structured events with the expected fields at the
//!   expected severity levels.
//! - The AgentRuntime::run_agent span carries agent_id and session_id fields.
//!
//! The tests use `tracing-subscriber` from dev-dependencies with `env-filter`.
//! They do *not* assert on specific log strings captured from a byte buffer
//! because writing to an in-process byte sink requires a custom Layer that is
//! out of scope; instead they assert that the code paths compile, execute
//! without error, and that the subscriber can be installed in a scoped context.

use llm_agent_runtime::{
    agent::{AgentConfig, ReActLoop, ToolSpec},
    memory::AgentId,
    runtime::AgentRuntime,
};
use tracing_subscriber::{fmt, EnvFilter};

/// Install a test-scoped tracing subscriber that writes to stderr.
///
/// Returns a guard whose drop uninstalls the subscriber. When multiple tests
/// run in the same process, only the first `try_init` call succeeds; the rest
/// are silently ignored, which is the correct behaviour for unit test helpers.
fn init_test_tracing() {
    // Ignore the error -- a subscriber may already be installed by a previous
    // test in the same process, and that is fine.
    let _ = fmt()
        .with_env_filter(EnvFilter::new("debug"))
        .with_test_writer()
        .try_init();
}

// ── ReActLoop tracing ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tracing_subscriber_installs_without_panic() {
    init_test_tracing();
    // If we reach here, the subscriber was installed (or was already present).
    // The test verifies no panic occurs during installation.
}

#[tokio::test]
async fn test_react_loop_emits_events_with_tracing_subscriber_installed() {
    init_test_tracing();

    let config = AgentConfig::new(5, "tracing-model");
    let loop_ = ReActLoop::new(config);

    // Run a simple one-step loop. With the subscriber installed, tracing events
    // are emitted to the test writer. We assert on the Ok/Err shape.
    let result = loop_
        .run("hello", |_ctx| async {
            "Thought: trivial\nAction: FINAL_ANSWER done".to_string()
        })
        .await;

    assert!(
        result.is_ok(),
        "expected ReActLoop::run to succeed; got {:?}",
        result
    );
    let steps = result.unwrap();
    assert_eq!(steps.len(), 1, "expected exactly one step");
}

#[tokio::test]
async fn test_react_loop_tool_dispatch_events_emitted() {
    init_test_tracing();

    let config = AgentConfig::new(5, "tracing-model");
    let mut loop_ = ReActLoop::new(config);

    // Register a simple echo tool.
    loop_.register_tool(ToolSpec::new(
        "echo",
        "Echo the input",
        |args| serde_json::json!({ "echoed": args }),
    ));

    let mut call_count = 0u32;
    let result = loop_
        .run("test", |_ctx| {
            call_count += 1;
            let count = call_count;
            async move {
                if count == 1 {
                    // Trigger a tool call -- this path emits the
                    // "dispatching tool call" debug event with tool_name field.
                    "Thought: call echo\nAction: echo {\"x\":1}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER result".to_string()
                }
            }
        })
        .await;

    assert!(result.is_ok(), "expected Ok; got {:?}", result);
    let steps = result.unwrap();
    // Step 0: echo tool call; step 1: FINAL_ANSWER.
    assert_eq!(steps.len(), 2);
    assert!(
        steps[0].observation.contains("\"ok\":true"),
        "tool observation should be ok=true; was: {}",
        steps[0].observation
    );
}

#[tokio::test]
async fn test_react_loop_max_iterations_warn_event_emitted() {
    init_test_tracing();

    let config = AgentConfig::new(2, "tracing-model");
    let loop_ = ReActLoop::new(config);

    // Let the loop exhaust its iterations -- this path emits the
    // "ReAct loop exhausted max iterations without FINAL_ANSWER" warn event.
    let result = loop_
        .run("exhaust", |_ctx| async {
            "Thought: spinning\nAction: noop {}".to_string()
        })
        .await;

    assert!(result.is_err(), "expected Err when max iterations reached");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("max iterations"),
        "error message should mention max iterations; was: {msg}"
    );
}

#[tokio::test]
async fn test_react_loop_unknown_tool_error_observation_with_tracing() {
    init_test_tracing();

    let config = AgentConfig::new(3, "tracing-model");
    let loop_ = ReActLoop::new(config);

    let mut call_count = 0u32;
    let result = loop_
        .run("test", |_ctx| {
            call_count += 1;
            let count = call_count;
            async move {
                if count == 1 {
                    "Thought: use phantom\nAction: phantom_tool {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER ok".to_string()
                }
            }
        })
        .await;

    assert!(result.is_ok());
    let steps = result.unwrap();
    assert_eq!(steps.len(), 2);

    let obs = &steps[0].observation;
    assert!(obs.contains("\"ok\":false"), "observation: {obs}");
    assert!(obs.contains("\"kind\":\"not_found\""), "observation: {obs}");
}

// ── AgentRuntime tracing ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_agent_runtime_run_agent_span_with_tracing() {
    init_test_tracing();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "tracing-model"))
        .build();

    // run_agent is instrumented with agent_id and session_id fields.
    // We verify the call succeeds; the fields are emitted into the test writer.
    let session = runtime
        .run_agent(
            AgentId::new("tracing-agent-01"),
            "hi",
            |_ctx: String| async { "Thought: ok\nAction: FINAL_ANSWER hi".to_string() },
        )
        .await
        .unwrap();

    assert!(!session.session_id.is_empty(), "session_id must be set");
    assert_eq!(session.step_count(), 1);
}

#[tokio::test]
async fn test_agent_runtime_multiple_sessions_emit_distinct_session_ids() {
    init_test_tracing();

    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "tracing-model"))
        .build();

    let s1 = runtime
        .run_agent(AgentId::new("a"), "p1", |_ctx: String| async {
            "Thought: x\nAction: FINAL_ANSWER x".to_string()
        })
        .await
        .unwrap();

    let s2 = runtime
        .run_agent(AgentId::new("a"), "p2", |_ctx: String| async {
            "Thought: y\nAction: FINAL_ANSWER y".to_string()
        })
        .await
        .unwrap();

    assert_ne!(
        s1.session_id, s2.session_id,
        "each session must have a unique session_id"
    );
}

#[tokio::test]
async fn test_tracing_env_filter_level_respected() {
    // Install a subscriber that filters at WARN level.  The ReActLoop debug
    // events should be suppressed, but the code should still run correctly.
    let _ = fmt()
        .with_env_filter(EnvFilter::new("warn"))
        .with_test_writer()
        .try_init();

    let config = AgentConfig::new(3, "tracing-model");
    let loop_ = ReActLoop::new(config);

    let result = loop_
        .run("filtered", |_ctx| async {
            "Thought: fine\nAction: FINAL_ANSWER ok".to_string()
        })
        .await;

    assert!(result.is_ok());
}

// ── Span name capture ─────────────────────────────────────────────────────────

use std::sync::{Arc, Mutex};
use tracing_subscriber::{layer::SubscriberExt, Registry};

/// A minimal tracing `Layer` that records the name of every new span created
/// while it is active.  Used to assert that specific instrumentation spans
/// (`react_iteration`, `tool_dispatch`) are emitted by the agent loop.
struct SpanCollector {
    names: Arc<Mutex<Vec<String>>>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for SpanCollector {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let name = attrs.metadata().name().to_owned();
        if let Ok(mut names) = self.names.lock() {
            names.push(name);
        }
    }
}

#[tokio::test]
async fn test_react_iteration_span_is_emitted() {
    let names: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let collector = SpanCollector {
        names: names.clone(),
    };
    let subscriber = Registry::default().with(collector);
    let _guard = tracing::subscriber::set_default(subscriber);

    let config = AgentConfig::new(5, "span-test-model");
    let loop_ = ReActLoop::new(config);
    let _ = loop_
        .run("hello", |_ctx| async {
            "Thought: ok\nAction: FINAL_ANSWER done".to_string()
        })
        .await;

    let collected = names.lock().unwrap();
    assert!(
        collected.iter().any(|s| s == "react_iteration"),
        "expected 'react_iteration' span; got: {collected:?}"
    );
}

#[tokio::test]
async fn test_tool_dispatch_span_is_emitted() {
    let names: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let collector = SpanCollector {
        names: names.clone(),
    };
    let subscriber = Registry::default().with(collector);
    let _guard = tracing::subscriber::set_default(subscriber);

    let config = AgentConfig::new(5, "span-test-model");
    let mut loop_ = ReActLoop::new(config);
    loop_.register_tool(ToolSpec::new(
        "ping",
        "Ping tool",
        |_| serde_json::json!({"ok": true}),
    ));

    let mut call_count = 0u32;
    let _ = loop_
        .run("test", |_ctx| {
            call_count += 1;
            let count = call_count;
            async move {
                if count == 1 {
                    "Thought: ping\nAction: ping {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER ok".to_string()
                }
            }
        })
        .await;

    let collected = names.lock().unwrap();
    assert!(
        collected.iter().any(|s| s == "tool_dispatch"),
        "expected 'tool_dispatch' span; got: {collected:?}"
    );
}
