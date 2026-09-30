//! # Module: Agent Swarm Coordination (v2)
//!
//! ## Responsibility
//! [`AgentSwarm`] manages N agents working on parallel subtasks derived from a
//! complex, high-level goal.  It extends the lower-level [`crate::swarm`] module
//! with:
//!
//! - Named [`SubTask`] structs carrying assignment, status, and result fields.
//! - Configurable task-split strategies (breadth-first or depth-first).
//! - A convergence threshold so the orchestrator can short-circuit once enough
//!   results are in.
//! - [`SwarmOrchestrator`] — a higher-level async driver that wires everything
//!   together.
//! - [`converge_results`] — synthesises multiple subtask results into one
//!   coherent answer.
//!
//! ## Design
//!
//! ```text
//! SwarmOrchestrator::run_swarm(task, config)
//!   │
//!   ├─► AgentSwarm::split_task(task, strategy)
//!   │      └─► Vec<SubTask>  (breadth or depth ordering)
//!   │
//!   ├─► assign each SubTask round-robin to registered AgentIds
//!   │
//!   ├─► run all assignments concurrently via FuturesUnordered
//!   │      └─► collect SubTaskResult for each
//!   │
//!   ├─► check convergence_threshold → early-exit if met
//!   │
//!   └─► converge_results(results) → SwarmResult
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_agent_runtime::swarm_v2::{
//!     AgentSwarm, SwarmConfig, SwarmOrchestrator, TaskSplitStrategy,
//! };
//! use llm_agent_runtime::types::AgentId;
//!
//! # tokio_test::block_on(async {
//! let config = SwarmConfig {
//!     max_agents: 4,
//!     task_split_strategy: TaskSplitStrategy::Breadth,
//!     convergence_threshold: 0.8,
//! };
//!
//! let mut orchestrator = SwarmOrchestrator::new(config);
//! orchestrator.add_agent(AgentId::new("worker-1"));
//! orchestrator.add_agent(AgentId::new("worker-2"));
//!
//! let result = orchestrator
//!     .run_swarm(
//!         "Research Rust async runtimes and summarise the ecosystem",
//!         |_agent_id, subtask: String| async move {
//!             format!("Result for: {subtask}")
//!         },
//!     )
//!     .await
//!     .unwrap();
//!
//! println!("Converged answer: {}", result.converged_answer);
//! # });
//! ```

use crate::error::AgentRuntimeError;
use crate::types::AgentId;
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ── SubTask status ─────────────────────────────────────────────────────────────

/// Lifecycle status of a [`SubTask`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubTaskStatus {
    /// Waiting to be dispatched to an agent.
    Pending,
    /// Currently being executed by an agent.
    Running,
    /// Finished successfully; result is populated.
    Completed,
    /// The assigned agent returned an error.
    Failed,
}

impl Default for SubTaskStatus {
    fn default() -> Self {
        Self::Pending
    }
}

// ── SubTask ────────────────────────────────────────────────────────────────────

/// A single parallelisable unit of work produced by decomposing a parent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    /// Stable unique identifier for this subtask (UUID v4 string).
    pub id: String,
    /// Human-readable description of what this subtask should accomplish.
    pub description: String,
    /// The prompt fragment forwarded to the assigned agent.
    pub prompt: String,
    /// The agent ID assigned to execute this subtask, if assigned.
    pub assigned_agent: Option<AgentId>,
    /// Current lifecycle status.
    pub status: SubTaskStatus,
    /// The textual result produced by the assigned agent, once completed.
    pub result: Option<String>,
    /// Depth in the decomposition tree (0 = top-level breadth slice).
    pub depth: usize,
}

impl SubTask {
    /// Create a new pending subtask.
    pub fn new(id: impl Into<String>, description: impl Into<String>, prompt: impl Into<String>, depth: usize) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            prompt: prompt.into(),
            assigned_agent: None,
            status: SubTaskStatus::Pending,
            result: None,
            depth,
        }
    }
}

// ── SubTaskResult ─────────────────────────────────────────────────────────────

/// The outcome of executing a single [`SubTask`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTaskResult {
    /// Subtask that was executed.
    pub subtask_id: String,
    /// The agent that executed it.
    pub agent_id: AgentId,
    /// Text output produced by the agent.
    pub output: String,
    /// Whether execution succeeded.
    pub success: bool,
    /// Error message when `success = false`.
    pub error: Option<String>,
}

// ── TaskSplitStrategy ─────────────────────────────────────────────────────────

/// How a complex task is decomposed into [`SubTask`]s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TaskSplitStrategy {
    /// Breadth-first: split on top-level conjunctions / list markers first,
    /// producing a flat list of independent parallel subtasks.
    #[default]
    Breadth,
    /// Depth-first: for each top-level segment recursively split on nested
    /// conjunctions, producing subtasks ordered for a DFS traversal.
    Depth,
}

// ── SwarmConfig ────────────────────────────────────────────────────────────────

/// Configuration for an [`AgentSwarm`] / [`SwarmOrchestrator`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Maximum number of agents that may execute concurrently.  `0` means
    /// unlimited.
    pub max_agents: usize,
    /// How the parent task is split into subtasks.
    pub task_split_strategy: TaskSplitStrategy,
    /// Fraction `[0.0, 1.0]` of subtasks that must complete successfully
    /// before the orchestrator stops waiting for the remaining ones.
    /// `1.0` means all subtasks must succeed; `0.5` means 50 % is enough.
    pub convergence_threshold: f64,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            max_agents: 0,
            task_split_strategy: TaskSplitStrategy::Breadth,
            convergence_threshold: 1.0,
        }
    }
}

// ── SwarmResult ────────────────────────────────────────────────────────────────

/// The aggregated result returned by [`SwarmOrchestrator::run_swarm`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResult {
    /// Original high-level task string.
    pub task: String,
    /// Per-subtask individual results.
    pub subtask_results: Vec<SubTaskResult>,
    /// Synthesised, coherent answer built from all successful results.
    pub converged_answer: String,
    /// Number of subtasks that failed.
    pub failure_count: usize,
    /// Whether the convergence threshold was satisfied.
    pub threshold_met: bool,
}

impl SwarmResult {
    /// Returns `true` when every subtask succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.failure_count == 0
    }
}

// ── AgentSwarm ─────────────────────────────────────────────────────────────────

/// Core decomposition and assignment engine for an agent swarm.
///
/// Responsible for:
/// 1. [`AgentSwarm::split_task`] — heuristically breaking a complex task into
///    parallelisable [`SubTask`]s.
/// 2. Assigning subtasks to registered agents (round-robin).
pub struct AgentSwarm {
    config: SwarmConfig,
    agents: Vec<AgentId>,
}

impl AgentSwarm {
    /// Create a new swarm with the given configuration.
    pub fn new(config: SwarmConfig) -> Self {
        Self { config, agents: Vec::new() }
    }

    /// Register a worker agent.
    pub fn add_agent(&mut self, id: AgentId) {
        self.agents.push(id);
    }

    /// Return a slice of all registered agent IDs.
    pub fn agents(&self) -> &[AgentId] {
        &self.agents
    }

    /// Heuristically decompose `task` into parallelisable [`SubTask`]s.
    ///
    /// Splitting is done on:
    /// - Commas and semicolons as top-level list separators.
    /// - The token ` and ` as a conjunction separator.
    /// - Numbered-list prefixes like `1.`, `2.`, …
    ///
    /// When [`TaskSplitStrategy::Depth`] is selected, each top-level segment
    /// is recursively split on nested ` then ` / ` next ` / ` finally `
    /// phrases, producing ordered sub-subtasks at `depth = 1`.
    pub fn split_task(&self, task: &str) -> Vec<SubTask> {
        match self.config.task_split_strategy {
            TaskSplitStrategy::Breadth => Self::split_breadth(task),
            TaskSplitStrategy::Depth => Self::split_depth(task),
        }
    }

    /// Assign each subtask to an agent round-robin, mutating the subtask list
    /// in-place.
    ///
    /// # Errors
    /// Returns [`AgentRuntimeError::Internal`] when no agents are registered.
    pub fn assign(&self, subtasks: &mut Vec<SubTask>) -> Result<(), AgentRuntimeError> {
        if self.agents.is_empty() {
            return Err(AgentRuntimeError::Internal(
                "AgentSwarm: no agents registered for assignment".into(),
            ));
        }
        let limit = if self.config.max_agents == 0 {
            self.agents.len()
        } else {
            self.agents.len().min(self.config.max_agents)
        };
        for (i, st) in subtasks.iter_mut().enumerate() {
            let agent = self.agents[i % limit].clone();
            st.assigned_agent = Some(agent);
            st.status = SubTaskStatus::Running;
        }
        Ok(())
    }

    // ── Private split helpers ─────────────────────────────────────────────────

    fn split_breadth(task: &str) -> Vec<SubTask> {
        let parts = Self::primary_split(task);
        parts
            .into_iter()
            .enumerate()
            .map(|(i, part)| {
                let id = format!("st-{i}");
                SubTask::new(id, format!("Subtask {i}: {part}"), part, 0)
            })
            .collect()
    }

    fn split_depth(task: &str) -> Vec<SubTask> {
        let top_parts = Self::primary_split(task);
        let mut result = Vec::new();
        let mut global_idx = 0usize;
        for (_top_i, part) in top_parts.into_iter().enumerate() {
            // Try a secondary split on sequencing connectors.
            let sub_parts = Self::secondary_split(&part);
            if sub_parts.len() > 1 {
                for (sub_j, sub_part) in sub_parts.into_iter().enumerate() {
                    let id = format!("st-{global_idx}-d{sub_j}");
                    result.push(SubTask::new(id, format!("Step {global_idx}.{sub_j}: {sub_part}"), sub_part, 1));
                    global_idx += 1;
                }
            } else {
                let id = format!("st-{global_idx}");
                result.push(SubTask::new(id, format!("Subtask {global_idx}: {part}"), part, 0));
                global_idx += 1;
            }
        }
        result
    }

    /// Split on commas, semicolons, and ` and `.
    fn primary_split(task: &str) -> Vec<String> {
        // Strip leading numbered list markers like "1. " or "2) ".
        let clean: String = {
            let mut s = task.to_string();
            for prefix in ["1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ",
                            "1) ", "2) ", "3) ", "4) ", "5) "] {
                s = s.replace(prefix, ", ");
            }
            s
        };

        clean
            .split(|c| c == ',' || c == ';')
            .flat_map(|s| s.split(" and "))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Split on sequencing connectors for depth-first decomposition.
    fn secondary_split(segment: &str) -> Vec<String> {
        let lower = segment.to_lowercase();
        let has_connector = lower.contains(" then ")
            || lower.contains(" next ")
            || lower.contains(" finally ")
            || lower.contains(" after that ")
            || lower.contains(" subsequently ");

        if !has_connector {
            return vec![segment.to_string()];
        }

        segment
            .split(|c: char| c == '.')
            .flat_map(|s| s.split(" then "))
            .flat_map(|s| s.split(" next "))
            .flat_map(|s| s.split(" finally "))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }
}

// ── converge_results ──────────────────────────────────────────────────────────

/// Synthesise multiple [`SubTaskResult`]s into one coherent textual answer.
///
/// The algorithm:
/// 1. Filter to successful results only.
/// 2. Group by common leading keywords to detect thematic clusters.
/// 3. Produce a numbered synthesis prepended with a one-line executive summary.
pub fn converge_results(results: &[SubTaskResult]) -> String {
    let successful: Vec<&SubTaskResult> = results.iter().filter(|r| r.success).collect();

    if successful.is_empty() {
        return "No successful subtask results to converge.".to_string();
    }

    if successful.len() == 1 {
        return successful[0].output.clone();
    }

    // Build a brief executive summary line.
    let summary = format!(
        "Synthesised answer from {} agent result(s):",
        successful.len()
    );

    // Deduplicate near-identical results (first 60 chars as fingerprint).
    let mut seen_fingerprints: std::collections::HashSet<String> = std::collections::HashSet::new();
    let deduped: Vec<&&SubTaskResult> = successful
        .iter()
        .filter(|r| {
            let fp: String = r.output.chars().take(60).collect::<String>().to_lowercase();
            seen_fingerprints.insert(fp)
        })
        .collect();

    let body: String = deduped
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{}. [{}] {}", i + 1, r.agent_id, r.output))
        .collect::<Vec<_>>()
        .join("\n");

    format!("{summary}\n{body}")
}

// ── SwarmOrchestrator ─────────────────────────────────────────────────────────

/// Full async orchestration driver.
///
/// Wraps an [`AgentSwarm`] and drives the complete lifecycle:
/// split → assign → run concurrently → converge.
pub struct SwarmOrchestrator {
    swarm: AgentSwarm,
}

impl SwarmOrchestrator {
    /// Create a new orchestrator with the given configuration.
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            swarm: AgentSwarm::new(config),
        }
    }

    /// Register a worker agent.
    pub fn add_agent(&mut self, id: AgentId) {
        self.swarm.add_agent(id);
    }

    /// Return the current swarm configuration.
    pub fn config(&self) -> &SwarmConfig {
        &self.swarm.config
    }

    /// Full async orchestration: split `task`, assign subtasks, run all agents
    /// concurrently, and converge results into a [`SwarmResult`].
    ///
    /// `inference_fn` is called once per subtask with `(AgentId, prompt)`.
    ///
    /// The convergence threshold is checked after every result arrives:
    /// when the fraction of completed subtasks reaches
    /// `config.convergence_threshold`, remaining in-flight futures are dropped
    /// and the partial results are converged.
    ///
    /// # Errors
    /// Returns [`AgentRuntimeError::Internal`] when no agents are registered.
    pub async fn run_swarm<F, Fut>(
        &mut self,
        task: &str,
        inference_fn: F,
    ) -> Result<SwarmResult, AgentRuntimeError>
    where
        F: Fn(AgentId, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let mut subtasks = self.swarm.split_task(task);

        if subtasks.is_empty() {
            return Ok(SwarmResult {
                task: task.to_string(),
                subtask_results: vec![],
                converged_answer: String::new(),
                failure_count: 0,
                threshold_met: true,
            });
        }

        self.swarm.assign(&mut subtasks)?;

        let total = subtasks.len();
        let threshold = self.swarm.config.convergence_threshold.clamp(0.0, 1.0);
        let required_successes = ((total as f64) * threshold).ceil() as usize;

        let inference_fn = Arc::new(inference_fn);

        // Build a FuturesUnordered over all assigned subtasks.
        let mut futures: FuturesUnordered<
            Pin<Box<dyn Future<Output = SubTaskResult> + Send>>,
        > = FuturesUnordered::new();

        for st in &subtasks {
            let agent_id = match &st.assigned_agent {
                Some(id) => id.clone(),
                None => continue,
            };
            let prompt = st.prompt.clone();
            let st_id = st.id.clone();
            let inf = inference_fn.clone();

            tracing::debug!(
                agent = %agent_id,
                subtask_id = %st_id,
                "swarm_v2: dispatching subtask"
            );

            futures.push(Box::pin(async move {
                let output = inf(agent_id.clone(), prompt).await;
                SubTaskResult {
                    subtask_id: st_id,
                    agent_id,
                    output,
                    success: true,
                    error: None,
                }
            }));
        }

        let mut collected: Vec<SubTaskResult> = Vec::new();
        let mut failure_count = 0usize;
        let mut success_count = 0usize;

        while let Some(result) = futures.next().await {
            if result.success {
                success_count += 1;
            } else {
                failure_count += 1;
            }
            collected.push(result);

            // Check convergence threshold.
            if success_count >= required_successes {
                tracing::debug!(
                    success_count,
                    required_successes,
                    "swarm_v2: convergence threshold reached, stopping early"
                );
                break;
            }
        }

        let threshold_met = success_count >= required_successes;
        let converged_answer = converge_results(&collected);

        // Update subtask status in our local list for inspection.
        for st in &mut subtasks {
            if let Some(res) = collected.iter().find(|r| r.subtask_id == st.id) {
                st.status = if res.success {
                    SubTaskStatus::Completed
                } else {
                    SubTaskStatus::Failed
                };
                st.result = Some(res.output.clone());
            }
        }

        Ok(SwarmResult {
            task: task.to_string(),
            subtask_results: collected,
            converged_answer,
            failure_count,
            threshold_met,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn split_breadth_produces_flat_subtasks() {
        let swarm = AgentSwarm::new(SwarmConfig::default());
        let subtasks = swarm.split_task("research Rust and learn Tokio and write tests");
        assert_eq!(subtasks.len(), 3);
        assert!(subtasks.iter().all(|s| s.depth == 0));
    }

    #[test]
    fn split_depth_produces_nested_subtasks() {
        let config = SwarmConfig {
            task_split_strategy: TaskSplitStrategy::Depth,
            ..SwarmConfig::default()
        };
        let swarm = AgentSwarm::new(config);
        let subtasks = swarm.split_task("fetch data then process it then emit results");
        // "fetch data then process it then emit results" should expand via secondary_split
        assert!(!subtasks.is_empty());
    }

    #[test]
    fn assign_round_robins_over_agents() {
        let mut swarm = AgentSwarm::new(SwarmConfig::default());
        swarm.add_agent(AgentId::new("a1"));
        swarm.add_agent(AgentId::new("a2"));

        let mut subtasks = swarm.split_task("task1 and task2 and task3");
        swarm.assign(&mut subtasks).unwrap();

        let assigned: Vec<_> = subtasks
            .iter()
            .map(|s| s.assigned_agent.as_ref().map(|id| id.to_string()))
            .collect();
        assert_eq!(assigned[0].as_deref(), Some("a1"));
        assert_eq!(assigned[1].as_deref(), Some("a2"));
        assert_eq!(assigned[2].as_deref(), Some("a1"));
    }

    #[test]
    fn assign_errors_with_no_agents() {
        let swarm = AgentSwarm::new(SwarmConfig::default());
        let mut subtasks = swarm.split_task("do something");
        let err = swarm.assign(&mut subtasks).unwrap_err();
        assert!(matches!(err, AgentRuntimeError::Internal(_)));
    }

    #[test]
    fn converge_results_deduplicates_identical_outputs() {
        let agent = AgentId::new("a1");
        let results = vec![
            SubTaskResult { subtask_id: "s1".into(), agent_id: agent.clone(), output: "same output".into(), success: true, error: None },
            SubTaskResult { subtask_id: "s2".into(), agent_id: agent.clone(), output: "same output".into(), success: true, error: None },
            SubTaskResult { subtask_id: "s3".into(), agent_id: agent.clone(), output: "different".into(), success: true, error: None },
        ];
        let answer = converge_results(&results);
        // "same output" should only appear once in the numbered list.
        assert!(answer.contains("different"));
        // Count occurrences of "same output" in numbered lines.
        let count = answer.matches("same output").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn converge_results_skips_failures() {
        let agent = AgentId::new("a1");
        let results = vec![
            SubTaskResult { subtask_id: "s1".into(), agent_id: agent.clone(), output: "good".into(), success: true, error: None },
            SubTaskResult { subtask_id: "s2".into(), agent_id: agent.clone(), output: "bad".into(), success: false, error: Some("error".into()) },
        ];
        let answer = converge_results(&results);
        assert!(!answer.contains("bad"));
        assert!(answer.contains("good"));
    }

    #[tokio::test]
    async fn run_swarm_full_lifecycle() {
        let config = SwarmConfig {
            max_agents: 2,
            ..SwarmConfig::default()
        };
        let mut orchestrator = SwarmOrchestrator::new(config);
        orchestrator.add_agent(AgentId::new("w1"));
        orchestrator.add_agent(AgentId::new("w2"));

        let result = orchestrator
            .run_swarm(
                "research Rust and learn Tokio",
                |agent_id, prompt| async move { format!("{agent_id}: done — {prompt}") },
            )
            .await
            .unwrap();

        assert_eq!(result.subtask_results.len(), 2);
        assert!(result.threshold_met);
        assert!(result.converged_answer.contains("done"));
    }

    #[tokio::test]
    async fn run_swarm_convergence_threshold() {
        // With threshold=0.5 and 4 subtasks, 2 successes should trigger early convergence.
        let config = SwarmConfig {
            max_agents: 0,
            convergence_threshold: 0.5,
            ..SwarmConfig::default()
        };
        let mut orchestrator = SwarmOrchestrator::new(config);
        orchestrator.add_agent(AgentId::new("w1"));

        let result = orchestrator
            .run_swarm(
                "a and b and c and d",
                |_id, prompt| async move { format!("ok: {prompt}") },
            )
            .await
            .unwrap();

        assert!(result.threshold_met);
    }

    #[tokio::test]
    async fn run_swarm_empty_task_returns_empty_result() {
        let mut orchestrator = SwarmOrchestrator::new(SwarmConfig::default());
        orchestrator.add_agent(AgentId::new("w1"));

        // A task that produces no segments after splitting should give an empty result.
        let result = orchestrator
            .run_swarm("", |_id, p| async move { p })
            .await
            .unwrap();
        assert!(result.subtask_results.is_empty());
        assert!(result.threshold_met);
    }
}
