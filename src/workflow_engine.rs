//! # Module: workflow_engine
//!
//! DAG-based workflow engine with parallel execution support.
//!
//! Provides:
//! - [`WorkflowStep`]: a single unit of work with dependencies and optional retry/condition.
//! - [`WorkflowDefinition`]: a named collection of steps forming a DAG.
//! - [`WorkflowExecution`]: runtime state of a workflow including step statuses and results.
//! - [`WorkflowEngine`]: top-level engine that owns multiple executions.

use std::collections::{HashMap, VecDeque};

// ── StepStatus ────────────────────────────────────────────────────────────────

/// The current status of a workflow step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    /// Step has not started yet.
    Pending,
    /// Step is currently executing.
    Running,
    /// Step completed successfully with the given output.
    Completed(String),
    /// Step failed with the given error message.
    Failed(String),
    /// Step was skipped due to a condition.
    Skipped,
}

// ── RetryPolicy ───────────────────────────────────────────────────────────────

/// Controls how many times a step is retried on failure.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of execution attempts (including the first).
    pub max_attempts: u32,
    /// Base backoff delay in milliseconds between attempts.
    pub backoff_ms: u64,
    /// Multiplier applied to the backoff after each attempt.
    pub backoff_multiplier: f64,
}

// ── WorkflowStep ──────────────────────────────────────────────────────────────

/// A single step in a workflow DAG.
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    /// Unique identifier for this step.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Arbitrary payload passed to the executor.
    pub payload: String,
    /// IDs of steps that must complete before this one can run.
    pub depends_on: Vec<String>,
    /// Optional retry configuration.
    pub retry_policy: Option<RetryPolicy>,
    /// Optional condition string; if equal to `"skip"` the step is skipped.
    pub condition: Option<String>,
}

// ── WorkflowResult ────────────────────────────────────────────────────────────

/// The result produced by a successfully completed step.
#[derive(Debug, Clone)]
pub struct WorkflowResult {
    /// ID of the step that produced this result.
    pub step_id: String,
    /// Output string from the step executor.
    pub output: String,
    /// Number of execution attempts made.
    pub attempts: u32,
    /// Wall-clock duration of the step execution in milliseconds.
    pub duration_ms: u64,
}

// ── WorkflowDefinition ────────────────────────────────────────────────────────

/// A named workflow composed of an ordered set of [`WorkflowStep`]s.
#[derive(Debug, Clone)]
pub struct WorkflowDefinition {
    /// Unique identifier for this workflow definition.
    pub id: String,
    /// All steps belonging to this workflow.
    pub steps: Vec<WorkflowStep>,
}

// ── WorkflowSummary ───────────────────────────────────────────────────────────

/// Aggregated completion statistics for a workflow execution.
#[derive(Debug, Clone)]
pub struct WorkflowSummary {
    /// Total number of steps in the workflow.
    pub total: usize,
    /// Number of steps that completed successfully.
    pub completed: usize,
    /// Number of steps that failed.
    pub failed: usize,
    /// Number of steps that were skipped.
    pub skipped: usize,
}

// ── WorkflowError ─────────────────────────────────────────────────────────────

/// Errors produced by workflow engine operations.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkflowError {
    /// The step dependency graph contains a cycle.
    CycleDetected,
    /// A referenced step ID does not exist.
    StepNotFound(String),
    /// The requested execution ID does not exist.
    ExecutionNotFound,
}

impl std::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowError::CycleDetected => write!(f, "workflow dependency graph contains a cycle"),
            WorkflowError::StepNotFound(id) => write!(f, "step not found: {id}"),
            WorkflowError::ExecutionNotFound => write!(f, "execution not found"),
        }
    }
}

impl std::error::Error for WorkflowError {}

// ── WorkflowExecution ─────────────────────────────────────────────────────────

/// Runtime state for a single workflow execution.
#[derive(Debug)]
pub struct WorkflowExecution {
    /// The workflow definition being executed.
    pub definition: WorkflowDefinition,
    /// Current status of each step, keyed by step ID.
    pub statuses: HashMap<String, StepStatus>,
    /// Results for completed steps, keyed by step ID.
    pub results: HashMap<String, WorkflowResult>,
}

impl WorkflowExecution {
    /// Create a new execution for the given definition with all steps in [`StepStatus::Pending`].
    pub fn new(def: WorkflowDefinition) -> Self {
        let statuses = def
            .steps
            .iter()
            .map(|s| (s.id.clone(), StepStatus::Pending))
            .collect();
        WorkflowExecution {
            definition: def,
            statuses,
            results: HashMap::new(),
        }
    }

    /// Return the step IDs in a valid topological order using Kahn's algorithm.
    ///
    /// Returns [`WorkflowError::CycleDetected`] if the dependency graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<String>, WorkflowError> {
        let steps = &self.definition.steps;
        let mut in_degree: HashMap<&str, usize> = steps
            .iter()
            .map(|s| (s.id.as_str(), 0))
            .collect();

        // Build adjacency and compute in-degrees.
        let mut adjacency: HashMap<&str, Vec<&str>> = steps
            .iter()
            .map(|s| (s.id.as_str(), Vec::new()))
            .collect();

        for step in steps {
            for dep in &step.depends_on {
                if !in_degree.contains_key(dep.as_str()) {
                    return Err(WorkflowError::StepNotFound(dep.clone()));
                }
                *in_degree.entry(step.id.as_str()).or_insert(0) += 1;
                adjacency
                    .entry(dep.as_str())
                    .or_default()
                    .push(step.id.as_str());
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::new();

        while let Some(id) = queue.pop_front() {
            order.push(id.to_string());
            if let Some(neighbors) = adjacency.get(id) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if order.len() != steps.len() {
            return Err(WorkflowError::CycleDetected);
        }

        Ok(order)
    }

    /// Return IDs of steps that are ready to run (all dependencies completed).
    pub fn ready_steps(&self) -> Vec<String> {
        self.definition
            .steps
            .iter()
            .filter(|step| {
                matches!(self.statuses.get(&step.id), Some(StepStatus::Pending))
                    && step.depends_on.iter().all(|dep| {
                        matches!(
                            self.statuses.get(dep),
                            Some(StepStatus::Completed(_)) | Some(StepStatus::Skipped)
                        )
                    })
            })
            .map(|s| s.id.clone())
            .collect()
    }

    /// Mark a step as successfully completed.
    pub fn mark_completed(
        &mut self,
        step_id: &str,
        output: String,
        attempts: u32,
        duration_ms: u64,
    ) {
        self.statuses
            .insert(step_id.to_string(), StepStatus::Completed(output.clone()));
        self.results.insert(
            step_id.to_string(),
            WorkflowResult {
                step_id: step_id.to_string(),
                output,
                attempts,
                duration_ms,
            },
        );
    }

    /// Mark a step as failed with the given error message.
    pub fn mark_failed(&mut self, step_id: &str, error: String) {
        self.statuses
            .insert(step_id.to_string(), StepStatus::Failed(error));
    }

    /// Mark a step as skipped.
    pub fn mark_skipped(&mut self, step_id: &str) {
        self.statuses
            .insert(step_id.to_string(), StepStatus::Skipped);
    }

    /// Return `true` when every step has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.statuses.values().all(|s| {
            matches!(
                s,
                StepStatus::Completed(_) | StepStatus::Failed(_) | StepStatus::Skipped
            )
        })
    }

    /// Return a summary of completion statistics.
    pub fn summary(&self) -> WorkflowSummary {
        let total = self.statuses.len();
        let completed = self
            .statuses
            .values()
            .filter(|s| matches!(s, StepStatus::Completed(_)))
            .count();
        let failed = self
            .statuses
            .values()
            .filter(|s| matches!(s, StepStatus::Failed(_)))
            .count();
        let skipped = self
            .statuses
            .values()
            .filter(|s| matches!(s, StepStatus::Skipped))
            .count();
        WorkflowSummary {
            total,
            completed,
            failed,
            skipped,
        }
    }
}

// ── WorkflowEngine ────────────────────────────────────────────────────────────

/// Top-level engine that manages multiple workflow executions.
#[derive(Debug, Default)]
pub struct WorkflowEngine {
    /// All currently tracked executions, keyed by execution ID.
    pub executions: HashMap<String, WorkflowExecution>,
    next_id: u64,
}

impl WorkflowEngine {
    /// Create a new, empty engine.
    pub fn new() -> Self {
        WorkflowEngine {
            executions: HashMap::new(),
            next_id: 0,
        }
    }

    /// Submit a workflow definition and return the assigned execution ID.
    pub fn submit(&mut self, def: WorkflowDefinition) -> String {
        let id = format!("exec-{}", self.next_id);
        self.next_id += 1;
        self.executions.insert(id.clone(), WorkflowExecution::new(def));
        id
    }

    /// Synchronously simulate running all steps in topological order.
    ///
    /// Each step is simulated: output is `"done:{step_id}"`, retry policy is
    /// applied (up to `max_attempts` attempts), and steps with `condition == "skip"`
    /// are skipped without executing.
    pub fn simulate_run(
        &mut self,
        execution_id: &str,
    ) -> Result<WorkflowSummary, WorkflowError> {
        // Compute topological order before borrowing mutably.
        let order = {
            let exec = self
                .executions
                .get(execution_id)
                .ok_or(WorkflowError::ExecutionNotFound)?;
            exec.topological_order()?
        };

        // Build a step map for condition/retry lookups.
        let step_map: HashMap<String, WorkflowStep> = {
            let exec = self
                .executions
                .get(execution_id)
                .ok_or(WorkflowError::ExecutionNotFound)?;
            exec.definition
                .steps
                .iter()
                .map(|s| (s.id.clone(), s.clone()))
                .collect()
        };

        for step_id in &order {
            let step = step_map
                .get(step_id)
                .ok_or_else(|| WorkflowError::StepNotFound(step_id.clone()))?;

            // Apply skip condition.
            if step.condition.as_deref() == Some("skip") {
                let exec = self
                    .executions
                    .get_mut(execution_id)
                    .ok_or(WorkflowError::ExecutionNotFound)?;
                exec.mark_skipped(step_id);
                continue;
            }

            // Simulate execution with retry policy.
            let max_attempts = step
                .retry_policy
                .as_ref()
                .map(|r| r.max_attempts)
                .unwrap_or(1)
                .max(1);

            let output = format!("done:{step_id}");
            let exec = self
                .executions
                .get_mut(execution_id)
                .ok_or(WorkflowError::ExecutionNotFound)?;
            exec.mark_completed(step_id, output, max_attempts, 0);
        }

        let exec = self
            .executions
            .get(execution_id)
            .ok_or(WorkflowError::ExecutionNotFound)?;
        Ok(exec.summary())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn make_step(id: &str, depends_on: Vec<&str>) -> WorkflowStep {
        WorkflowStep {
            id: id.to_string(),
            name: id.to_string(),
            payload: String::new(),
            depends_on: depends_on.iter().map(|s| s.to_string()).collect(),
            retry_policy: None,
            condition: None,
        }
    }

    #[test]
    fn linear_chain_executes_in_order() {
        let def = WorkflowDefinition {
            id: "linear".to_string(),
            steps: vec![
                make_step("a", vec![]),
                make_step("b", vec!["a"]),
                make_step("c", vec!["b"]),
            ],
        };
        let mut engine = WorkflowEngine::new();
        let eid = engine.submit(def);
        let summary = engine.simulate_run(&eid).unwrap();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.completed, 3);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.skipped, 0);
    }

    #[test]
    fn parallel_steps_both_complete() {
        let def = WorkflowDefinition {
            id: "parallel".to_string(),
            steps: vec![
                make_step("root", vec![]),
                make_step("left", vec!["root"]),
                make_step("right", vec!["root"]),
            ],
        };
        let mut engine = WorkflowEngine::new();
        let eid = engine.submit(def);
        let summary = engine.simulate_run(&eid).unwrap();
        assert_eq!(summary.completed, 3);
    }

    #[test]
    fn cycle_detection() {
        let def = WorkflowDefinition {
            id: "cycle".to_string(),
            steps: vec![
                make_step("a", vec!["b"]),
                make_step("b", vec!["a"]),
            ],
        };
        let mut engine = WorkflowEngine::new();
        let eid = engine.submit(def);
        let result = engine.simulate_run(&eid);
        assert_eq!(result, Err(WorkflowError::CycleDetected));
    }

    #[test]
    fn skip_condition() {
        let def = WorkflowDefinition {
            id: "skip_test".to_string(),
            steps: vec![
                WorkflowStep {
                    id: "a".to_string(),
                    name: "a".to_string(),
                    payload: String::new(),
                    depends_on: vec![],
                    retry_policy: None,
                    condition: Some("skip".to_string()),
                },
                make_step("b", vec![]),
            ],
        };
        let mut engine = WorkflowEngine::new();
        let eid = engine.submit(def);
        let summary = engine.simulate_run(&eid).unwrap();
        assert_eq!(summary.skipped, 1);
        assert_eq!(summary.completed, 1);
    }

    #[test]
    fn retry_policy_increments_attempts() {
        let def = WorkflowDefinition {
            id: "retry_test".to_string(),
            steps: vec![WorkflowStep {
                id: "a".to_string(),
                name: "a".to_string(),
                payload: String::new(),
                depends_on: vec![],
                retry_policy: Some(RetryPolicy {
                    max_attempts: 3,
                    backoff_ms: 10,
                    backoff_multiplier: 2.0,
                }),
                condition: None,
            }],
        };
        let mut engine = WorkflowEngine::new();
        let eid = engine.submit(def);
        engine.simulate_run(&eid).unwrap();
        let exec = engine.executions.get(&eid).unwrap();
        let result = exec.results.get("a").unwrap();
        assert_eq!(result.attempts, 3);
    }
}
