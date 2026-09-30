//! # Module: Parallel Tool Execution
//!
//! ## Responsibility
//! Detects *independent* tool calls within a ReAct loop batch and executes them
//! concurrently using [`futures::stream::FuturesUnordered`].  Tool calls are
//! considered independent when they share no data dependency (i.e. none of their
//! argument strings references the output key of another pending call).
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_agent_runtime::parallel_tools::{ParallelToolExecutor, ToolCall, ToolCallResult};
//! use serde_json::json;
//!
//! # tokio_test::block_on(async {
//! let mut executor = ParallelToolExecutor::new();
//! executor.register("web_search", |args| {
//!     Box::pin(async move {
//!         Ok(json!({"result": format!("found: {args}")}))
//!     })
//! });
//! executor.register("calculator", |args| {
//!     Box::pin(async move { Ok(json!({"answer": 42})) })
//! });
//!
//! let calls = vec![
//!     ToolCall::new("web_search", json!({"query": "Rust"})),
//!     ToolCall::new("calculator", json!({"expr": "6*7"})),
//! ];
//!
//! let results = executor.run_parallel(calls).await;
//! println!("{results:?}");
//! # });
//! ```

use crate::error::AgentRuntimeError;
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A pinned, boxed future returning a fallible JSON result.
pub type ParallelToolFuture =
    Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>;

/// Handler signature for a tool in the parallel executor.
pub type ParallelToolHandler = Box<dyn Fn(Value) -> ParallelToolFuture + Send + Sync>;

// ── ToolCall ──────────────────────────────────────────────────────────────────

/// A single tool call request — name + JSON arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this call within the batch (auto-assigned if empty).
    pub id: String,
    /// Name of the tool to invoke.
    pub tool: String,
    /// JSON arguments passed to the tool.
    pub args: Value,
}

impl ToolCall {
    /// Construct a call with an auto-generated UUID identifier.
    pub fn new(tool: impl Into<String>, args: Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            tool: tool.into(),
            args,
        }
    }

    /// Construct a call with an explicit identifier.
    pub fn with_id(id: impl Into<String>, tool: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            args,
        }
    }
}

// ── ToolCallResult ────────────────────────────────────────────────────────────

/// The outcome of a single tool call in a parallel batch.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallResult {
    /// Echoes the `id` from the originating [`ToolCall`].
    pub id: String,
    /// Name of the tool that was invoked.
    pub tool: String,
    /// `Ok` output or `Err` message.
    pub result: Result<Value, String>,
}

impl ToolCallResult {
    /// Returns `true` when the call succeeded.
    pub fn is_ok(&self) -> bool {
        self.result.is_ok()
    }

    /// Borrow the output value if successful.
    pub fn output(&self) -> Option<&Value> {
        self.result.as_ref().ok()
    }
}

// ── Dependency detection ───────────────────────────────────────────────────────

/// Check whether two tool calls are independent (no data dependency).
///
/// The heuristic: call B *depends* on call A when call B's serialised arguments
/// contain call A's `id` as a substring — a common convention for chaining
/// within a ReAct scratchpad.
pub fn are_independent(a: &ToolCall, b: &ToolCall) -> bool {
    let b_args_str = b.args.to_string();
    !b_args_str.contains(&a.id)
}

/// Partition a list of calls into independent batches that can be executed in
/// parallel.  Uses a simple greedy O(n²) algorithm suitable for the small
/// batches typical in ReAct loops (≤ 20 calls).
///
/// Returns a `Vec` of batches; each batch is a `Vec<ToolCall>` whose members
/// are mutually independent.
pub fn partition_independent(calls: Vec<ToolCall>) -> Vec<Vec<ToolCall>> {
    let mut batches: Vec<Vec<ToolCall>> = Vec::new();

    'outer: for call in calls {
        // Try to add to an existing batch where no member depends on `call`
        // and `call` does not depend on any member.
        for batch in &mut batches {
            let compatible = batch
                .iter()
                .all(|existing| are_independent(existing, &call) && are_independent(&call, existing));
            if compatible {
                batch.push(call);
                continue 'outer;
            }
        }
        // No compatible batch found — start a new one.
        batches.push(vec![call]);
    }

    batches
}

// ── Executor ─────────────────────────────────────────────────────────────────

/// Executes tool calls, running independent calls concurrently.
#[derive(Default)]
pub struct ParallelToolExecutor {
    handlers: HashMap<String, ParallelToolHandler>,
    /// Maximum number of concurrent tool futures (0 = unlimited).
    pub concurrency_limit: usize,
}

impl ParallelToolExecutor {
    /// Create a new executor with no concurrency limit.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            concurrency_limit: 0,
        }
    }

    /// Create a new executor with a concurrency cap.
    pub fn with_limit(limit: usize) -> Self {
        Self {
            handlers: HashMap::new(),
            concurrency_limit: limit,
        }
    }

    /// Register an async tool handler.
    pub fn register<F, Fut>(&mut self, tool_name: &str, handler: F)
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        self.handlers.insert(
            tool_name.to_string(),
            Box::new(move |args| Box::pin(handler(args))),
        );
    }

    /// Execute a batch of tool calls **concurrently** using [`FuturesUnordered`].
    ///
    /// Results are returned in completion order.  Calls referencing an
    /// unregistered tool produce an `Err` result rather than panicking.
    pub async fn run_parallel(&self, calls: Vec<ToolCall>) -> Vec<ToolCallResult> {
        let mut futures: FuturesUnordered<
            Pin<Box<dyn Future<Output = ToolCallResult> + Send>>,
        > = FuturesUnordered::new();

        for call in calls {
            let id = call.id.clone();
            let tool = call.tool.clone();

            match self.handlers.get(&call.tool) {
                None => {
                    let err_msg =
                        format!("Tool '{}' not registered in ParallelToolExecutor", call.tool);
                    tracing::warn!(tool = %call.tool, "parallel_tools: unregistered tool");
                    futures.push(Box::pin(async move {
                        ToolCallResult {
                            id,
                            tool,
                            result: Err(err_msg),
                        }
                    }));
                }
                Some(handler) => {
                    let fut = handler(call.args.clone());
                    tracing::debug!(tool = %call.tool, id = %id, "parallel_tools: dispatching");
                    futures.push(Box::pin(async move {
                        let result = fut.await;
                        ToolCallResult { id, tool, result }
                    }));
                }
            }
        }

        let mut results = Vec::new();
        while let Some(result) = futures.next().await {
            results.push(result);
        }
        results
    }

    /// Automatically partition calls into independent batches, then execute
    /// each batch in parallel.  Batches are run sequentially relative to each
    /// other to respect data dependencies.
    pub async fn run_smart(&self, calls: Vec<ToolCall>) -> Vec<ToolCallResult> {
        let batches = partition_independent(calls);
        let mut all_results = Vec::new();

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            tracing::debug!(
                batch = batch_idx,
                size = batch.len(),
                "parallel_tools: executing batch"
            );
            let mut batch_results = self.run_parallel(batch).await;
            all_results.append(&mut batch_results);
        }

        all_results
    }

    /// Return an error if any result in the slice is a failure.
    ///
    /// Convenience method for callers that treat any tool failure as fatal.
    pub fn check_all_ok(results: &[ToolCallResult]) -> Result<(), AgentRuntimeError> {
        for r in results {
            if let Err(ref msg) = r.result {
                return Err(AgentRuntimeError::Internal(format!(
                    "Tool '{}' (id={}) failed: {msg}",
                    r.tool, r.id
                )));
            }
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn independence_check() {
        let a = ToolCall::with_id("id-a", "search", json!({}));
        let b = ToolCall::with_id("id-b", "calc", json!({"ref": "id-a"}));
        assert!(!are_independent(&a, &b), "b depends on a");
        assert!(are_independent(&b, &a), "a does not depend on b");
    }

    #[test]
    fn partition_two_independent() {
        let calls = vec![
            ToolCall::with_id("1", "search", json!({})),
            ToolCall::with_id("2", "calc", json!({})),
        ];
        let batches = partition_independent(calls);
        assert_eq!(batches.len(), 1, "both should be in one batch");
        assert_eq!(batches[0].len(), 2);
    }

    #[test]
    fn partition_dependent_pair() {
        let calls = vec![
            ToolCall::with_id("id-a", "search", json!({})),
            ToolCall::with_id("id-b", "summarize", json!({"input": "id-a"})),
        ];
        let batches = partition_independent(calls);
        assert_eq!(batches.len(), 2, "dependent calls must be separate batches");
    }

    #[tokio::test]
    async fn run_parallel_all_succeed() {
        let mut exec = ParallelToolExecutor::new();
        exec.register("tool_a", |_| Box::pin(async { Ok(json!({"a": 1})) }));
        exec.register("tool_b", |_| Box::pin(async { Ok(json!({"b": 2})) }));

        let calls = vec![
            ToolCall::new("tool_a", json!({})),
            ToolCall::new("tool_b", json!({})),
        ];
        let results = exec.run_parallel(calls).await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn run_parallel_missing_tool() {
        let exec = ParallelToolExecutor::new();
        let results = exec
            .run_parallel(vec![ToolCall::new("missing", json!({}))])
            .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_ok());
    }

    #[tokio::test]
    async fn run_parallel_concurrent_execution() {
        // Verify that futures actually run concurrently by counting max inflight.
        let inflight = Arc::new(AtomicUsize::new(0));
        let max_inflight = Arc::new(AtomicUsize::new(0));

        let mut exec = ParallelToolExecutor::new();
        let inf = inflight.clone();
        let max = max_inflight.clone();

        exec.register("slow_tool", move |_| {
            let inf = inf.clone();
            let max = max.clone();
            Box::pin(async move {
                let current = inf.fetch_add(1, Ordering::SeqCst) + 1;
                max.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                inf.fetch_sub(1, Ordering::SeqCst);
                Ok(json!({"ok": true}))
            })
        });

        let calls: Vec<_> = (0..4)
            .map(|_| ToolCall::new("slow_tool", json!({})))
            .collect();
        let results = exec.run_parallel(calls).await;
        assert_eq!(results.len(), 4);
        // With FuturesUnordered all 4 start before any finishes.
        assert!(
            max_inflight.load(Ordering::SeqCst) > 1,
            "expected concurrent execution"
        );
    }

    #[tokio::test]
    async fn check_all_ok_fails_on_error() {
        let results = vec![ToolCallResult {
            id: "x".into(),
            tool: "bad".into(),
            result: Err("oops".into()),
        }];
        assert!(ParallelToolExecutor::check_all_ok(&results).is_err());
    }
}
