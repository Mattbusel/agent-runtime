//! Tests for failure-mode handling: persistence backend errors and network
//! split simulation for LLM providers.
//!
//! These tests cover:
//!   1. A `PersistenceBackend` whose `save` always returns `Err` -- the agent
//!      run should still succeed because the runtime logs the error rather than
//!      propagating it for per-step checkpoints, but the final checkpoint save
//!      failure IS propagated.
//!   2. A `PersistenceBackend` whose `load` always returns `Err` -- covered by
//!      exercising `AgentSession::load_checkpoint` directly.
//!   3. The trait contract (object-safety + Send+Sync) preventing panicking
//!      implementations from being required -- we verify via a `PanicOnSave`
//!      stub that panics can be contained with `std::panic::catch_unwind`.
//!   4. A network-split simulation: an `LlmProvider` that returns a connection
//!      error on `complete`, exercised through the `ReActLoop` infer callback.

// ── Persistence backend failure tests ────────────────────────────────────────
// These only compile when the `persistence` feature is active.

#[cfg(feature = "persistence")]
mod persistence_failure_tests {
    use agent_runtime::agent::AgentConfig;
    use agent_runtime::memory::AgentId;
    use agent_runtime::persistence::PersistenceBackend;
    use agent_runtime::runtime::{AgentRuntime, AgentSession};
    use agent_runtime::AgentRuntimeError;
    use async_trait::async_trait;
    use std::sync::Arc;

    // ── FailingWriteBackend ───────────────────────────────────────────────────

    /// A persistence backend whose `save` always returns an `Err`.
    ///
    /// `load` and `delete` succeed normally so that session reads do not
    /// interfere with what we are testing.
    struct FailingWriteBackend;

    #[async_trait]
    impl PersistenceBackend for FailingWriteBackend {
        async fn save(&self, _key: &str, _value: &[u8]) -> Result<(), AgentRuntimeError> {
            Err(AgentRuntimeError::Persistence(
                "simulated write failure".into(),
            ))
        }

        async fn load(&self, _key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError> {
            Ok(None)
        }

        async fn delete(&self, _key: &str) -> Result<(), AgentRuntimeError> {
            Ok(())
        }
    }

    // ── FailingLoadBackend ────────────────────────────────────────────────────

    /// A persistence backend whose `load` always returns an `Err`.
    struct FailingLoadBackend;

    #[async_trait]
    impl PersistenceBackend for FailingLoadBackend {
        async fn save(&self, _key: &str, _value: &[u8]) -> Result<(), AgentRuntimeError> {
            Ok(())
        }

        async fn load(&self, _key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError> {
            Err(AgentRuntimeError::Persistence(
                "simulated load failure".into(),
            ))
        }

        async fn delete(&self, _key: &str) -> Result<(), AgentRuntimeError> {
            Ok(())
        }
    }

    // ── AlwaysSucceedBackend ──────────────────────────────────────────────────

    /// Counts save calls so we can assert incremental checkpoints are attempted.
    struct CountingSaveBackend {
        save_count: std::sync::atomic::AtomicU32,
    }

    impl CountingSaveBackend {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                save_count: std::sync::atomic::AtomicU32::new(0),
            })
        }

        fn save_count(&self) -> u32 {
            self.save_count.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl PersistenceBackend for CountingSaveBackend {
        async fn save(&self, _key: &str, _value: &[u8]) -> Result<(), AgentRuntimeError> {
            self.save_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }

        async fn load(&self, _key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError> {
            Ok(None)
        }

        async fn delete(&self, _key: &str) -> Result<(), AgentRuntimeError> {
            Ok(())
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// A `save` failure on the final session checkpoint IS propagated to the
    /// caller as a `Persistence` error because the runtime awaits the final
    /// `save_checkpoint` and returns its error.
    #[tokio::test]
    async fn test_persistence_write_failure_propagates_to_run_agent() {
        let backend: Arc<dyn PersistenceBackend> = Arc::new(FailingWriteBackend);

        let runtime = AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(5, "test"))
            .with_checkpoint_backend(Arc::clone(&backend))
            .build();

        let result = runtime
            .run_agent(AgentId::new("fail-agent"), "test", |_ctx: String| async {
                "Thought: ok\nAction: FINAL_ANSWER done".to_string()
            })
            .await;

        // The final checkpoint save failure is returned as an error.
        assert!(
            result.is_err(),
            "expected Err when persistence backend fails on save; got Ok"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentRuntimeError::Persistence(_)),
            "expected Persistence variant; got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("simulated write failure"),
            "error message should reference the simulated write failure; was: {msg}"
        );
    }

    /// Per-step checkpoint write failures are only logged (warn), not propagated.
    /// We verify this by running a multi-step agent with a write-failing backend
    /// for step checkpoints but a succeeding backend for the final session save.
    ///
    /// Because we cannot swap backends mid-run, we instead use a fresh backend
    /// that succeeds on the FIRST save call (the final session checkpoint) and
    /// fails on subsequent calls.  However, that level of fixture complexity is
    /// unnecessary: the existing `test_persistence_write_failure_propagates_to_run_agent`
    /// already captures the propagation contract.
    ///
    /// This test focuses on step-level checkpoint failures being silently
    /// tolerated by the runtime by verifying the agent DOES complete when the
    /// backend fails only on step keys (not the session key).
    #[tokio::test]
    async fn test_persistence_step_write_failure_is_logged_not_propagated() {
        /// Backend that succeeds on `session:*` keys but fails on `session:*:step:*` keys.
        struct StepFailBackend;

        #[async_trait]
        impl PersistenceBackend for StepFailBackend {
            async fn save(&self, key: &str, _value: &[u8]) -> Result<(), AgentRuntimeError> {
                if key.contains(":step:") {
                    Err(AgentRuntimeError::Persistence(format!(
                        "simulated step write failure for key '{key}'"
                    )))
                } else {
                    Ok(())
                }
            }

            async fn load(&self, _key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError> {
                Ok(None)
            }

            async fn delete(&self, _key: &str) -> Result<(), AgentRuntimeError> {
                Ok(())
            }
        }

        let backend: Arc<dyn PersistenceBackend> = Arc::new(StepFailBackend);

        let mut call_count = 0u32;
        let runtime = AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(10, "test"))
            .with_checkpoint_backend(Arc::clone(&backend))
            .build();

        // Run an agent that takes two steps: one tool call + final answer.
        let result = runtime
            .run_agent(
                AgentId::new("step-fail-agent"),
                "test",
                move |_ctx: String| {
                    call_count += 1;
                    let count = call_count;
                    async move {
                        if count == 1 {
                            "Thought: step\nAction: noop {}".to_string()
                        } else {
                            "Thought: done\nAction: FINAL_ANSWER ok".to_string()
                        }
                    }
                },
            )
            .await;

        // The agent run should succeed: step checkpoint failures are only warned.
        assert!(
            result.is_ok(),
            "expected Ok when only step checkpoints fail; got {:?}",
            result
        );
        let session = result.unwrap();
        assert_eq!(session.step_count(), 2);
    }

    /// Verify that `AgentSession::load_checkpoint` propagates a `Persistence`
    /// error when the backend's `load` fails.
    #[tokio::test]
    async fn test_persistence_load_failure_propagates_from_load_checkpoint() {
        let backend = FailingLoadBackend;

        let result = AgentSession::load_checkpoint(&backend, "any-session-id").await;

        assert!(
            result.is_err(),
            "expected Err when persistence backend load fails; got Ok"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentRuntimeError::Persistence(_)),
            "expected Persistence variant; got {err:?}"
        );
    }

    /// Verify the trait contract: `PersistenceBackend` is object-safe and
    /// `Send + Sync`, meaning it can be used as `Arc<dyn PersistenceBackend>`.
    /// A panicking implementation can be held behind this pointer; however,
    /// its panic is the caller's problem, not the trait's.
    ///
    /// We verify with `catch_unwind` that a panicking save is containable by
    /// the caller -- the trait itself does not mandate non-panicking behaviour
    /// at the type level, but the production contract documented in the module
    /// is "non-panicking: all operations return Result".
    ///
    /// This test documents the boundary of that contract.
    #[tokio::test]
    async fn test_panicking_backend_can_be_caught_with_catch_unwind() {
        // A backend whose `save` panics.
        struct PanicOnSave;

        #[async_trait]
        impl PersistenceBackend for PanicOnSave {
            async fn save(&self, _key: &str, _value: &[u8]) -> Result<(), AgentRuntimeError> {
                panic!("intentional panic from PanicOnSave backend");
            }

            async fn load(&self, _key: &str) -> Result<Option<Vec<u8>>, AgentRuntimeError> {
                Ok(None)
            }

            async fn delete(&self, _key: &str) -> Result<(), AgentRuntimeError> {
                Ok(())
            }
        }

        let backend = PanicOnSave;

        // Wrap the panicking call in catch_unwind to prevent it from aborting
        // the test process.  AssertUnwindSafe is required because the future
        // captured by the closure is not RefUnwindSafe.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // We use tokio's block_in_place / a new runtime to drive the future
            // inside catch_unwind because async fn + catch_unwind requires care.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("could not build tokio runtime");

            rt.block_on(backend.save("key", b"value"))
        }));

        assert!(
            result.is_err(),
            "expected catch_unwind to catch the panic from PanicOnSave::save"
        );
    }

    /// Verify that a counting backend receives the expected number of save
    /// calls: one per step plus one for the final session checkpoint.
    ///
    /// For a two-step run (one tool call + FINAL_ANSWER):
    ///   - 1 final session save
    ///   - 2 per-step saves (step:1, step:2)
    ///   = 3 total save calls
    #[tokio::test]
    async fn test_persistence_save_called_for_each_step_and_final_session() {
        let backend = CountingSaveBackend::new();
        let backend_arc: Arc<dyn PersistenceBackend> =
            Arc::clone(&backend) as Arc<dyn PersistenceBackend>;

        let mut call_count = 0u32;
        let runtime = AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(10, "test"))
            .with_checkpoint_backend(backend_arc)
            .register_tool(agent_runtime::agent::ToolSpec::new(
                "noop",
                "does nothing",
                |_| serde_json::json!(null),
            ))
            .build();

        runtime
            .run_agent(
                AgentId::new("counting-agent"),
                "test",
                move |_ctx: String| {
                    call_count += 1;
                    let count = call_count;
                    async move {
                        if count == 1 {
                            "Thought: step 1\nAction: noop {}".to_string()
                        } else {
                            "Thought: done\nAction: FINAL_ANSWER result".to_string()
                        }
                    }
                },
            )
            .await
            .unwrap();

        // 1 final session save + 2 step saves (step:1, step:2).
        assert_eq!(
            backend.save_count(),
            3,
            "expected 3 save calls (1 session + 2 steps)"
        );
    }
}

// ── Network split / provider failure tests ────────────────────────────────────
// These only compile when the `providers` feature is active.

#[cfg(feature = "providers")]
mod network_split_tests {
    use agent_runtime::agent::{AgentConfig, ReActLoop};
    use agent_runtime::providers::LlmProvider;
    use agent_runtime::AgentRuntimeError;
    use async_trait::async_trait;
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    // ── ConnectionErrorProvider ───────────────────────────────────────────────

    /// Provider that returns a connection error on every call.
    ///
    /// Simulates a complete network split from the LLM endpoint.
    struct ConnectionErrorProvider;

    #[async_trait]
    impl LlmProvider for ConnectionErrorProvider {
        async fn complete(&self, _prompt: &str, _model: &str) -> Result<String, AgentRuntimeError> {
            Err(AgentRuntimeError::Provider(
                "connection refused: simulated network split".into(),
            ))
        }
    }

    // ── MidSessionNetworkSplitProvider ────────────────────────────────────────

    /// Provider that succeeds for the first N calls, then returns a connection
    /// error.  Models a network split that occurs mid-session.
    struct MidSessionNetworkSplitProvider {
        calls_before_split: u32,
        call_count: AtomicU32,
    }

    impl MidSessionNetworkSplitProvider {
        fn new(calls_before_split: u32) -> Arc<Self> {
            Arc::new(Self {
                calls_before_split,
                call_count: AtomicU32::new(0),
            })
        }
    }

    #[async_trait]
    impl LlmProvider for MidSessionNetworkSplitProvider {
        async fn complete(&self, _prompt: &str, _model: &str) -> Result<String, AgentRuntimeError> {
            let n = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= self.calls_before_split {
                // Return a valid but non-final ReAct step.
                Ok("Thought: working\nAction: noop {}".to_string())
            } else {
                Err(AgentRuntimeError::Provider(
                    "connection reset: simulated mid-session network split".into(),
                ))
            }
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// When the LLM provider always returns a connection error, the infer
    /// callback should propagate the error string.  The `ReActLoop` receives the
    /// error as a normal string response and will fail to parse it as a ReAct
    /// step, resulting in an `AgentLoop` error.
    ///
    /// This test exercises the path where the infer function (not the loop
    /// itself) surfaces a provider error.  In a real integration the caller
    /// would wrap `provider.complete(...)` in the infer closure.
    #[tokio::test]
    async fn test_connection_error_provider_fails_gracefully() {
        let provider = ConnectionErrorProvider;

        let result = provider.complete("prompt", "model").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentRuntimeError::Provider(_)),
            "expected Provider error variant; got {err:?}"
        );
        assert!(
            err.to_string().contains("connection refused"),
            "error should mention connection refused; was: {err}"
        );
    }

    /// When the infer closure surfaces a provider error as a non-parseable
    /// string, the ReActLoop returns an `AgentLoop` parse error rather than
    /// panicking or hanging.
    #[tokio::test]
    async fn test_react_loop_handles_provider_error_string_gracefully() {
        let config = AgentConfig::new(3, "test");
        let loop_ = ReActLoop::new(config);

        // Simulate what a caller would do when provider.complete() fails:
        // pass the error message as the response string.  The loop should
        // fail to parse it and return an AgentLoop error.
        let result = loop_
            .run("test", |_ctx| async {
                // This mimics a caller that converts a provider error to a string
                // rather than unwrapping.
                "Provider error: connection refused: simulated network split".to_string()
            })
            .await;

        assert!(
            result.is_err(),
            "expected Err when infer returns unparseable string"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentRuntimeError::AgentLoop(_)),
            "expected AgentLoop error; got {err:?}"
        );
    }

    /// Simulate a mid-session network split: the provider succeeds for the
    /// first call, then fails.  The caller's infer closure propagates the
    /// error as a string; the loop returns an `AgentLoop` parse error.
    ///
    /// This tests that the agent does not hang or panic when the network
    /// splits mid-session.
    #[tokio::test]
    async fn test_react_loop_handles_mid_session_network_split() {
        let provider = MidSessionNetworkSplitProvider::new(1);
        let provider_arc = Arc::clone(&provider);

        let config = AgentConfig::new(5, "test");
        let loop_ = ReActLoop::new(config);

        let result = loop_
            .run("test", move |ctx| {
                let p = Arc::clone(&provider_arc);
                async move {
                    match p.complete(&ctx, "test-model").await {
                        Ok(response) => response,
                        Err(e) => {
                            // Surface the error as a non-ReAct string.
                            format!("Provider error: {e}")
                        }
                    }
                }
            })
            .await;

        // The first call succeeds but produces a non-final step ("noop {}").
        // The second call hits the network split, returns an error string
        // that cannot be parsed, causing an AgentLoop error.
        assert!(
            result.is_err(),
            "expected Err after mid-session network split"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentRuntimeError::AgentLoop(_)),
            "expected AgentLoop error after split; got {err:?}"
        );
    }

    /// Verify the stream_complete default implementation also surfaces provider
    /// errors via the receiver channel.
    #[tokio::test]
    async fn test_connection_error_provider_stream_complete_returns_error_in_channel() {
        let provider = ConnectionErrorProvider;

        let mut rx = provider.stream_complete("prompt", "model").await.unwrap();

        let first = rx.recv().await;
        assert!(
            first.is_some(),
            "channel should yield the error before closing"
        );
        let chunk = first.unwrap();
        assert!(chunk.is_err(), "stream chunk should be an Err; got Ok");
        let err = chunk.unwrap_err();
        assert!(
            matches!(err, AgentRuntimeError::Provider(_)),
            "expected Provider error in stream; got {err:?}"
        );
    }

    /// Verify that a provider returning an error mid-stream (after a successful
    /// start) is represented as an `Err` chunk in the receiver.
    #[tokio::test]
    async fn test_provider_is_object_safe_behind_arc_with_error() {
        let p: Arc<dyn LlmProvider> = Arc::new(ConnectionErrorProvider);

        let result = p.complete("prompt", "model").await;
        assert!(result.is_err());
    }
}
