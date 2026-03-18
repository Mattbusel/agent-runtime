//! Tests for the ReAct loop and orchestrator: action/observation cycles,
//! max-iterations cutoff, error recovery, and pipeline/deduplication.

use agent_runtime::agent::{AgentConfig, ReActLoop, ToolSpec};
use agent_runtime::orchestrator::{
    BackpressureGuard, DeduplicationResult, Deduplicator, Pipeline, RetryPolicy,
};
use agent_runtime::AgentRuntimeError;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

// ── ReAct: action/observation cycle ───────────────────────────────────────────

#[tokio::test]
async fn react_single_tool_call_produces_observation() {
    let config = AgentConfig::new(5, "m");
    let mut loop_ = ReActLoop::new(config);

    loop_.register_tool(ToolSpec::new("ping", "pings", |_| {
        serde_json::json!("pong")
    }));

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let steps = loop_
        .run("test", move |_ctx| {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            async move {
                if n == 0 {
                    "Thought: ping\nAction: ping {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER ok".to_string()
                }
            }
        })
        .await
        .unwrap();

    assert_eq!(steps.len(), 2);
    assert!(
        steps[0].observation.contains("pong"),
        "observation: {}",
        steps[0].observation
    );
}

#[tokio::test]
async fn react_tool_observation_contains_ok_true_on_success() {
    let config = AgentConfig::new(5, "m");
    let mut loop_ = ReActLoop::new(config);

    loop_.register_tool(ToolSpec::new("greet", "greets", |_| {
        serde_json::json!("hello")
    }));

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let steps = loop_
        .run("test", move |_ctx| {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            async move {
                if n == 0 {
                    "Thought: greet\nAction: greet {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER done".to_string()
                }
            }
        })
        .await
        .unwrap();

    let obs = &steps[0].observation;
    assert!(obs.contains("\"ok\":true"), "expected ok:true, got: {obs}");
}

#[tokio::test]
async fn react_multi_step_tool_chain_accumulates_all_steps() {
    let config = AgentConfig::new(10, "m");
    let mut loop_ = ReActLoop::new(config);

    for name in ["step1", "step2", "step3"] {
        loop_.register_tool(ToolSpec::new(name, name, |_| serde_json::json!("done")));
    }

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let steps = loop_
        .run("chain", move |_ctx| {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            async move {
                match n {
                    0 => "Thought: s1\nAction: step1 {}".to_string(),
                    1 => "Thought: s2\nAction: step2 {}".to_string(),
                    2 => "Thought: s3\nAction: step3 {}".to_string(),
                    _ => "Thought: done\nAction: FINAL_ANSWER all done".to_string(),
                }
            }
        })
        .await
        .unwrap();

    assert_eq!(steps.len(), 4); // 3 tool calls + final answer
}

// ── ReAct: max iterations cutoff ─────────────────────────────────────────────

#[tokio::test]
async fn react_errors_when_max_iterations_reached() {
    let config = AgentConfig::new(3, "m");
    let loop_ = ReActLoop::new(config);

    let result = loop_
        .run("run forever", |_ctx| async {
            "Thought: keep going\nAction: noop {}".to_string()
        })
        .await;

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("max iterations"), "error was: {msg}");
}

#[tokio::test]
async fn react_one_iteration_budget_terminates_on_final_answer() {
    let config = AgentConfig::new(1, "m");
    let loop_ = ReActLoop::new(config);

    let result = loop_
        .run("quick", |_ctx| async {
            "Thought: done\nAction: FINAL_ANSWER result".to_string()
        })
        .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[tokio::test]
async fn react_exactly_at_limit_with_final_answer_succeeds() {
    let config = AgentConfig::new(2, "m");
    let mut loop_ = ReActLoop::new(config);

    loop_.register_tool(ToolSpec::new("t", "t", |_| serde_json::json!(1)));

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let result = loop_
        .run("on edge", move |_ctx| {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            async move {
                if n == 0 {
                    "Thought: t\nAction: t {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER ok".to_string()
                }
            }
        })
        .await;

    // 2 calls: 1 tool call (step 0) + 1 final answer (step 1) within limit=2
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
}

// ── ReAct: error recovery ─────────────────────────────────────────────────────

#[tokio::test]
async fn react_unknown_tool_produces_error_observation_not_panic() {
    let config = AgentConfig::new(3, "m");
    let loop_ = ReActLoop::new(config);

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let steps = loop_
        .run("recover", move |_ctx| {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            async move {
                if n == 0 {
                    "Thought: try unknown\nAction: no_such_tool {}".to_string()
                } else {
                    "Thought: continue\nAction: FINAL_ANSWER done".to_string()
                }
            }
        })
        .await
        .unwrap();

    // Should not panic; first step's observation should contain error info
    assert_eq!(steps.len(), 2);
    assert!(
        steps[0].observation.contains("\"ok\":false"),
        "observation: {}",
        steps[0].observation
    );
}

#[tokio::test]
async fn react_missing_required_field_produces_error_observation() {
    let config = AgentConfig::new(3, "m");
    let mut loop_ = ReActLoop::new(config);

    loop_.register_tool(
        ToolSpec::new("search", "searches", |args| serde_json::json!(args))
            .with_required_fields(vec!["query".to_string()]),
    );

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let steps = loop_
        .run("test", move |_ctx| {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            async move {
                if n == 0 {
                    // Missing "query" field
                    r#"Thought: search
Action: search {"wrong":"field"}"#
                        .to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER done".to_string()
                }
            }
        })
        .await
        .unwrap();

    assert!(
        steps[0].observation.contains("missing required field"),
        "observation: {}",
        steps[0].observation
    );
}

#[tokio::test]
async fn react_context_grows_with_each_step() {
    let config = AgentConfig::new(5, "m");
    let mut loop_ = ReActLoop::new(config);

    loop_.register_tool(ToolSpec::new("echo", "echoes", |v| v));

    let contexts: Arc<std::sync::Mutex<Vec<usize>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ctx_clone = Arc::clone(&contexts);

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    loop_
        .run("grow", move |ctx: String| {
            let n = cc.fetch_add(1, Ordering::Relaxed);
            {
                #[allow(clippy::unwrap_used)]
                let mut guard = ctx_clone.lock().unwrap();
                guard.push(ctx.len());
            }
            async move {
                if n == 0 {
                    "Thought: first\nAction: echo {}".to_string()
                } else {
                    "Thought: done\nAction: FINAL_ANSWER done".to_string()
                }
            }
        })
        .await
        .unwrap();

    #[allow(clippy::unwrap_used)]
    let lens = contexts.lock().unwrap();
    // Context should grow between iterations
    assert!(lens[1] > lens[0], "context should grow after first step");
}

// ── RetryPolicy ───────────────────────────────────────────────────────────────

#[test]
fn retry_policy_delay_grows_exponentially() {
    let p = RetryPolicy::exponential(5, 100).unwrap();
    let d1 = p.delay_for(1);
    let d2 = p.delay_for(2);
    let d3 = p.delay_for(3);

    assert_eq!(d1, Duration::from_millis(100));
    assert_eq!(d2, Duration::from_millis(200));
    assert_eq!(d3, Duration::from_millis(400));
}

#[test]
fn retry_policy_delay_capped_at_60_seconds() {
    let p = RetryPolicy::exponential(10, 100_000).unwrap();
    for attempt in 1..=10 {
        assert!(
            p.delay_for(attempt) <= Duration::from_secs(60),
            "delay exceeded cap at attempt {attempt}"
        );
    }
}

#[test]
fn retry_policy_zero_attempts_is_invalid() {
    assert!(RetryPolicy::exponential(0, 100).is_err());
}

// ── Deduplicator ──────────────────────────────────────────────────────────────

#[test]
fn dedup_first_check_is_new() {
    let d = Deduplicator::new(Duration::from_secs(60));
    assert_eq!(d.check_and_register("k").unwrap(), DeduplicationResult::New);
}

#[test]
fn dedup_second_check_while_in_flight_is_in_progress() {
    let d = Deduplicator::new(Duration::from_secs(60));
    d.check_and_register("k").unwrap();
    assert_eq!(
        d.check_and_register("k").unwrap(),
        DeduplicationResult::InProgress
    );
}

#[test]
fn dedup_after_complete_is_cached() {
    let d = Deduplicator::new(Duration::from_secs(60));
    d.check_and_register("k").unwrap();
    d.complete("k", "result").unwrap();
    assert_eq!(
        d.check_and_register("k").unwrap(),
        DeduplicationResult::Cached("result".into())
    );
}

#[test]
fn dedup_expired_entry_becomes_new() {
    let d = Deduplicator::new(Duration::ZERO); // instant TTL
    d.check_and_register("k").unwrap();
    d.complete("k", "old").unwrap();
    assert_eq!(d.check_and_register("k").unwrap(), DeduplicationResult::New);
}

#[test]
fn dedup_different_keys_are_independent() {
    let d = Deduplicator::new(Duration::from_secs(60));
    d.check_and_register("a").unwrap();
    assert_eq!(d.check_and_register("b").unwrap(), DeduplicationResult::New);
}

// ── BackpressureGuard ─────────────────────────────────────────────────────────

#[test]
fn backpressure_accepts_up_to_capacity() {
    let g = BackpressureGuard::new(3).unwrap();
    assert!(g.try_acquire().is_ok());
    assert!(g.try_acquire().is_ok());
    assert!(g.try_acquire().is_ok());
    assert!(matches!(
        g.try_acquire(),
        Err(AgentRuntimeError::BackpressureShed { .. })
    ));
}

#[test]
fn backpressure_release_makes_room_for_more() {
    let g = BackpressureGuard::new(1).unwrap();
    g.try_acquire().unwrap();
    g.release().unwrap();
    assert!(g.try_acquire().is_ok());
}

#[test]
fn backpressure_soft_limit_must_be_less_than_hard() {
    let g = BackpressureGuard::new(5).unwrap();
    assert!(g.clone().with_soft_limit(5).is_err());
    assert!(g.clone().with_soft_limit(6).is_err());
}

#[test]
fn backpressure_soft_limit_valid_config_accepted() {
    let g = BackpressureGuard::new(10).unwrap();
    assert!(g.with_soft_limit(7).is_ok());
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

#[test]
fn pipeline_stages_run_in_insertion_order() {
    let log: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let log1 = Arc::clone(&log);
    let log2 = Arc::clone(&log);

    let p = Pipeline::new()
        .add_stage("first", move |s| {
            #[allow(clippy::unwrap_used)]
            log1.lock().unwrap().push("first".to_string());
            Ok(s)
        })
        .add_stage("second", move |s| {
            #[allow(clippy::unwrap_used)]
            log2.lock().unwrap().push("second".to_string());
            Ok(s)
        });

    p.run("input".to_string()).unwrap();

    #[allow(clippy::unwrap_used)]
    let order = log.lock().unwrap().clone();
    assert_eq!(order, vec!["first", "second"]);
}

#[test]
fn pipeline_stage_failure_short_circuits_remaining_stages() {
    let ran = Arc::new(AtomicUsize::new(0));
    let ran2 = Arc::clone(&ran);

    let p = Pipeline::new()
        .add_stage("fail", |_| {
            Err(AgentRuntimeError::Orchestration("boom".into()))
        })
        .add_stage("never", move |s| {
            ran2.fetch_add(1, Ordering::Relaxed);
            Ok(s)
        });

    assert!(p.run("input".to_string()).is_err());
    assert_eq!(ran.load(Ordering::Relaxed), 0);
}

#[test]
fn pipeline_empty_returns_input_unchanged() {
    let p = Pipeline::new();
    assert_eq!(p.run("hello".to_string()).unwrap(), "hello");
}
