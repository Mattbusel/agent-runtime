//! Integration tests: agent lifecycle — create, run, stop (completion).

use agent_runtime::prelude::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn final_infer(_ctx: String) -> impl std::future::Future<Output = String> {
    async { "Thought: complete\nAction: FINAL_ANSWER done".into() }
}

// ── Create ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn lifecycle_create_minimal_runtime() {
    // A runtime can be created with only a config — no optional subsystems.
    let _runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .build();
}

#[tokio::test]
async fn lifecycle_create_runtime_with_all_subsystems() {
    let _runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(10, "test-model"))
        .with_memory(EpisodicStore::new())
        .with_working_memory(WorkingMemory::new(20).unwrap())
        .with_graph(GraphStore::new())
        .with_backpressure(BackpressureGuard::new(50).unwrap())
        .register_tool(ToolSpec::new("noop", "does nothing", |_| {
            serde_json::Value::Null
        }))
        .build();
}

// ── Run ───────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn lifecycle_run_agent_produces_session() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .build();

    let session = runtime
        .run_agent(AgentId::new("lifecycle-agent"), "what is 1+1?", final_infer)
        .await
        .unwrap();

    assert!(session.step_count() >= 1, "at least one step expected");
    assert_eq!(session.agent_id.0, "lifecycle-agent");
    assert_eq!(session.session_id.len(), 36, "session_id should be UUID v4");
}

#[tokio::test]
async fn lifecycle_run_agent_two_steps_before_answer() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(10, "test-model"))
        .register_tool(ToolSpec::new("calc", "adds numbers", |args| {
            let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
            serde_json::json!(n + 1)
        }))
        .build();

    let mut call_n = 0u32;
    let session = runtime
        .run_agent(
            AgentId::new("two-step-agent"),
            "use calc then answer",
            move |_ctx: String| {
                call_n += 1;
                let n = call_n;
                async move {
                    if n == 1 {
                        r#"Thought: call calc first
Action: calc {"n":41}"#
                            .to_string()
                    } else {
                        "Thought: got the answer\nAction: FINAL_ANSWER 42".to_string()
                    }
                }
            },
        )
        .await
        .unwrap();

    assert_eq!(session.step_count(), 2);
}

#[tokio::test]
async fn lifecycle_session_unique_ids_per_run() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .build();

    let s1 = runtime
        .run_agent(AgentId::new("a"), "prompt", final_infer)
        .await
        .unwrap();
    let s2 = runtime
        .run_agent(AgentId::new("a"), "prompt", final_infer)
        .await
        .unwrap();

    assert_ne!(
        s1.session_id, s2.session_id,
        "each run should receive a distinct session UUID"
    );
}

// ── Stop (completion / FINAL_ANSWER) ─────────────────────────────────────────

#[tokio::test]
async fn lifecycle_final_answer_terminates_loop() {
    // The loop must terminate as soon as FINAL_ANSWER appears, even when
    // max_iterations is large.
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(100, "test-model"))
        .build();

    let session = runtime
        .run_agent(
            AgentId::new("stop-agent"),
            "stop now",
            |_ctx: String| async { "Thought: I'm done\nAction: FINAL_ANSWER stopped".to_string() },
        )
        .await
        .unwrap();

    // Should have stopped on the first step even though max=100.
    assert_eq!(session.step_count(), 1);
}

#[tokio::test]
async fn lifecycle_max_iterations_returns_error() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(3, "test-model"))
        .register_tool(ToolSpec::new("spin", "never ends", |_| {
            serde_json::Value::Null
        }))
        .build();

    let result = runtime
        .run_agent(
            AgentId::new("infinite-agent"),
            "never finish",
            |_ctx: String| async { "Thought: looping\nAction: spin {}".to_string() },
        )
        .await;

    assert!(
        result.is_err(),
        "expected error when max_iterations is exhausted"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("max iterations"),
        "error message should mention max iterations, got: {err}"
    );
}

#[tokio::test]
async fn lifecycle_metrics_track_full_session() {
    let runtime = AgentRuntime::builder()
        .with_agent_config(AgentConfig::new(5, "test-model"))
        .build();

    runtime
        .run_agent(AgentId::new("a"), "p", final_infer)
        .await
        .unwrap();

    let m = runtime.metrics();
    assert_eq!(m.total_sessions(), 1);
    assert_eq!(
        m.active_sessions(),
        0,
        "active count must return to 0 after run"
    );
    assert_eq!(m.total_steps(), 1);
}
