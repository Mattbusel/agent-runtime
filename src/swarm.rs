//! # Module: Agent Swarm
//!
//! ## Responsibility
//! Coordinates multiple agents working on decomposed subtasks.
//! [`Swarm::run`] decomposes a high-level task into subtasks, assigns each to
//! a worker agent, runs them concurrently, and aggregates the results.
//!
//! ## Design
//!
//! ```text
//! Swarm::run(task)
//!   │
//!   ├─► decompose(task)  ── produces Vec<Subtask>
//!   │
//!   ├─► assign each Subtask to a WorkerAgent (round-robin or custom)
//!   │
//!   ├─► run all WorkerAgents concurrently (FuturesUnordered)
//!   │
//!   └─► aggregate results ── produces SwarmResult
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_agent_runtime::swarm::{Swarm, SwarmConfig, SubtaskDecomposer};
//! use llm_agent_runtime::types::AgentId;
//!
//! # tokio_test::block_on(async {
//! let config = SwarmConfig::default();
//! let mut swarm = Swarm::new(config);
//!
//! // Register worker agents (IDs only — inference is injected as a closure).
//! swarm.add_worker(AgentId::new("worker-1"));
//! swarm.add_worker(AgentId::new("worker-2"));
//!
//! // Run with a simple word-count decomposer and a stub inference fn.
//! let result = swarm.run(
//!     "Research Rust async and Tokio internals",
//!     |_agent_id, subtask: String| async move { format!("Result for: {subtask}") },
//! ).await.unwrap();
//!
//! println!("Aggregated: {}", result.aggregated);
//! # });
//! ```

use crate::error::AgentRuntimeError;
use crate::types::AgentId;
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ── Subtask ───────────────────────────────────────────────────────────────────

/// A single unit of work produced by decomposing the parent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    /// Sequential index within the decomposed list (0-based).
    pub index: usize,
    /// Human-readable description of this subtask.
    pub description: String,
    /// Portion of the original task assigned to this subtask.
    pub prompt: String,
}

// ── Decomposer ────────────────────────────────────────────────────────────────

/// Trait for splitting a high-level task string into [`Subtask`]s.
pub trait SubtaskDecomposer: Send + Sync {
    /// Decompose `task` into an ordered list of subtasks.
    fn decompose(&self, task: &str) -> Vec<Subtask>;
}

/// A simple decomposer that splits on ` and ` and sentence boundaries.
///
/// Suitable for demos; replace with an LLM-based decomposer in production.
pub struct SimpleDecomposer {
    /// Maximum number of subtasks to produce (0 = unlimited).
    pub max_subtasks: usize,
}

impl Default for SimpleDecomposer {
    fn default() -> Self {
        Self { max_subtasks: 8 }
    }
}

impl SubtaskDecomposer for SimpleDecomposer {
    fn decompose(&self, task: &str) -> Vec<Subtask> {
        // Split on common conjunction/list markers.
        let raw_parts: Vec<&str> = task
            .split(|c| c == ',' || c == ';')
            .flat_map(|s| s.split(" and "))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        let limit = if self.max_subtasks == 0 {
            raw_parts.len()
        } else {
            raw_parts.len().min(self.max_subtasks)
        };

        raw_parts
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(index, part)| Subtask {
                index,
                description: format!("Subtask {index}: {part}"),
                prompt: part.to_string(),
            })
            .collect()
    }
}

// ── Worker result ─────────────────────────────────────────────────────────────

/// The result produced by a single worker agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    /// Which agent produced this result.
    pub agent_id: AgentId,
    /// The subtask that was assigned.
    pub subtask: Subtask,
    /// The agent's output text.
    pub output: String,
}

// ── Swarm result ──────────────────────────────────────────────────────────────

/// The aggregated result returned by [`Swarm::run`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResult {
    /// Per-worker individual results in completion order.
    pub worker_results: Vec<WorkerResult>,
    /// Simple concatenation of all worker outputs.
    pub aggregated: String,
    /// Original task string.
    pub task: String,
    /// Number of subtasks that failed (workers returned an error).
    pub failure_count: usize,
}

impl SwarmResult {
    /// Returns `true` when all workers succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.failure_count == 0
    }
}

// ── Config ─────────────────────────────────────────────────────────────────────

/// Configuration for a [`Swarm`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Maximum number of subtasks to run concurrently (0 = unlimited).
    pub max_concurrency: usize,
    /// Whether to continue when a worker fails (true) or abort (false).
    pub fault_tolerant: bool,
    /// Strategy for combining worker outputs into the aggregated field.
    pub aggregation: AggregationStrategy,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 0,
            fault_tolerant: true,
            aggregation: AggregationStrategy::Concat,
        }
    }
}

/// How worker outputs are combined into the final [`SwarmResult::aggregated`] string.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AggregationStrategy {
    /// Simple newline-separated concatenation (default).
    #[default]
    Concat,
    /// Numbered list.
    Numbered,
    /// JSON array of output strings.
    JsonArray,
}

// ── Swarm ─────────────────────────────────────────────────────────────────────

/// Coordinates multiple worker agents on decomposed subtasks.
pub struct Swarm {
    config: SwarmConfig,
    workers: Vec<AgentId>,
    decomposer: Arc<dyn SubtaskDecomposer>,
}

impl Swarm {
    /// Create a new swarm with the default [`SimpleDecomposer`].
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            config,
            workers: Vec::new(),
            decomposer: Arc::new(SimpleDecomposer::default()),
        }
    }

    /// Replace the decomposer with a custom implementation.
    pub fn with_decomposer(mut self, decomposer: impl SubtaskDecomposer + 'static) -> Self {
        self.decomposer = Arc::new(decomposer);
        self
    }

    /// Add a worker agent by ID.
    pub fn add_worker(&mut self, id: AgentId) {
        self.workers.push(id);
    }

    /// Return the list of registered worker IDs.
    pub fn workers(&self) -> &[AgentId] {
        &self.workers
    }

    /// Decompose `task`, assign subtasks round-robin to workers, run all
    /// concurrently, and return the aggregated [`SwarmResult`].
    ///
    /// The `inference_fn` is called once per subtask with `(AgentId, prompt)`.
    ///
    /// # Errors
    /// Returns [`AgentRuntimeError::Internal`] when no workers are registered,
    /// or (when `fault_tolerant = false`) when any worker fails.
    pub async fn run<F, Fut>(
        &self,
        task: &str,
        inference_fn: F,
    ) -> Result<SwarmResult, AgentRuntimeError>
    where
        F: Fn(AgentId, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        if self.workers.is_empty() {
            return Err(AgentRuntimeError::Internal(
                "Swarm has no registered workers".into(),
            ));
        }

        let subtasks = self.decomposer.decompose(task);
        if subtasks.is_empty() {
            return Ok(SwarmResult {
                worker_results: vec![],
                aggregated: String::new(),
                task: task.to_string(),
                failure_count: 0,
            });
        }

        let inference_fn = Arc::new(inference_fn);
        let mut futures: FuturesUnordered<
            Pin<Box<dyn Future<Output = Result<WorkerResult, String>> + Send>>,
        > = FuturesUnordered::new();

        for subtask in subtasks {
            // Round-robin assignment.
            let worker_idx = subtask.index % self.workers.len();
            let agent_id = self.workers[worker_idx].clone();
            let prompt = subtask.prompt.clone();
            let st = subtask.clone();
            let inf = inference_fn.clone();

            tracing::debug!(
                agent = %agent_id,
                subtask = subtask.index,
                "swarm: dispatching subtask"
            );

            futures.push(Box::pin(async move {
                let output = inf(agent_id.clone(), prompt).await;
                Ok(WorkerResult {
                    agent_id,
                    subtask: st,
                    output,
                })
            }));
        }

        let mut worker_results: Vec<WorkerResult> = Vec::new();
        let mut failure_count = 0usize;

        while let Some(res) = futures.next().await {
            match res {
                Ok(wr) => worker_results.push(wr),
                Err(e) => {
                    failure_count += 1;
                    tracing::warn!(error = %e, "swarm: worker failed");
                    if !self.config.fault_tolerant {
                        return Err(AgentRuntimeError::Internal(format!(
                            "Swarm worker failed (fault_tolerant=false): {e}"
                        )));
                    }
                }
            }
        }

        // Sort by subtask index for deterministic output.
        worker_results.sort_by_key(|r| r.subtask.index);

        let aggregated = self.aggregate(&worker_results);

        Ok(SwarmResult {
            worker_results,
            aggregated,
            task: task.to_string(),
            failure_count,
        })
    }

    /// Combine worker outputs according to the configured [`AggregationStrategy`].
    fn aggregate(&self, results: &[WorkerResult]) -> String {
        match self.config.aggregation {
            AggregationStrategy::Concat => results
                .iter()
                .map(|r| r.output.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            AggregationStrategy::Numbered => results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("{}. {}", i + 1, r.output))
                .collect::<Vec<_>>()
                .join("\n"),
            AggregationStrategy::JsonArray => {
                let arr: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| serde_json::Value::String(r.output.clone()))
                    .collect();
                serde_json::to_string(&arr).unwrap_or_default()
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_decomposer_splits_on_and() {
        let d = SimpleDecomposer::default();
        let subtasks = d.decompose("research Rust and learn Tokio and write tests");
        assert_eq!(subtasks.len(), 3);
        assert_eq!(subtasks[0].prompt, "research Rust");
        assert_eq!(subtasks[2].prompt, "write tests");
    }

    #[test]
    fn simple_decomposer_respects_max() {
        let d = SimpleDecomposer { max_subtasks: 2 };
        let subtasks = d.decompose("a and b and c and d");
        assert_eq!(subtasks.len(), 2);
    }

    #[tokio::test]
    async fn swarm_basic_run() {
        let mut swarm = Swarm::new(SwarmConfig::default());
        swarm.add_worker(AgentId::new("w1"));
        swarm.add_worker(AgentId::new("w2"));

        let result = swarm
            .run(
                "research Rust and learn Tokio",
                |agent_id, subtask| async move { format!("{agent_id}: done — {subtask}") },
            )
            .await
            .unwrap();

        assert_eq!(result.worker_results.len(), 2);
        assert!(result.all_succeeded());
        assert!(result.aggregated.contains("research Rust"));
        assert!(result.aggregated.contains("learn Tokio"));
    }

    #[tokio::test]
    async fn swarm_no_workers_errors() {
        let swarm = Swarm::new(SwarmConfig::default());
        let err = swarm
            .run("task", |_id, _t| async { "out".to_string() })
            .await
            .unwrap_err();
        assert!(matches!(err, AgentRuntimeError::Internal(_)));
    }

    #[tokio::test]
    async fn swarm_round_robin_assignment() {
        let mut swarm = Swarm::new(SwarmConfig::default());
        swarm.add_worker(AgentId::new("w1"));
        swarm.add_worker(AgentId::new("w2"));

        // 3 subtasks → w1 gets 0,2 and w2 gets 1
        let result = swarm
            .run("a and b and c", |id, t| async move { format!("{id}:{t}") })
            .await
            .unwrap();

        let agents: Vec<_> = result
            .worker_results
            .iter()
            .map(|r| r.agent_id.to_string())
            .collect();
        // subtask 0 → w1, subtask 1 → w2, subtask 2 → w1
        assert!(agents.contains(&"w1".to_string()));
        assert!(agents.contains(&"w2".to_string()));
    }

    #[tokio::test]
    async fn aggregation_numbered() {
        let mut swarm = Swarm::new(SwarmConfig {
            aggregation: AggregationStrategy::Numbered,
            ..SwarmConfig::default()
        });
        swarm.add_worker(AgentId::new("w1"));

        let result = swarm
            .run("one and two", |_, t| async move { t })
            .await
            .unwrap();
        assert!(result.aggregated.starts_with("1."));
    }
}
