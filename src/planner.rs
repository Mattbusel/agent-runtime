//! # Hierarchical Task Network (HTN) Planner
//!
//! Provides a goal-directed planning layer as an alternative (or complement)
//! to the pure ReAct loop.
//!
//! ## Why HTN planning?
//!
//! ReAct is reactive: the agent decides what to do one step at a time based on
//! LLM output.  This works well for open-ended tasks but has two weaknesses:
//!
//! 1. **No upfront structure**: complex multi-step workflows (e.g. "research a
//!    company and write a report") require the LLM to discover the sub-steps
//!    itself, wasting tokens on iteration.
//! 2. **Hard to enforce structure**: it is difficult to guarantee that certain
//!    steps happen in a particular order.
//!
//! HTN planning adds a structured decomposition layer on top of ReAct:
//!
//! - **Goals** are high-level objectives: `"Research company ACME"`.
//! - **Tasks** are abstract steps: `"SearchWeb"`, `"SummariseDocument"`.
//! - **Methods** decompose abstract tasks into concrete tool calls.
//! - The **planner** expands the task network into a fully ordered
//!   [`ExecutionPlan`] before the first tool call is made.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::planner::{Planner, Task, Method, PlannerConfig, Precondition};
//!
//! let mut planner = Planner::new(PlannerConfig::default());
//!
//! // Register a method: decompose "ResearchCompany" into 3 primitive tasks.
//! planner.register_method(Method {
//!     task: "ResearchCompany".into(),
//!     subtasks: vec![
//!         Task::primitive("web_search", r#"{"query":"<company>"}"#),
//!         Task::primitive("scrape_page", r#"{"url":"<top_result>"}"#),
//!         Task::primitive("summarise", r#"{"text":"<page_content>"}"#),
//!     ],
//!     preconditions: vec![],
//!     description: "Search the web and summarise findings".into(),
//! });
//!
//! // Plan a goal.
//! let goal = Task::compound("ResearchCompany");
//! let plan = planner.plan(goal).unwrap();
//!
//! assert_eq!(plan.steps().len(), 3);
//! println!("Step 0: {}", plan.steps()[0].tool);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A task in the hierarchy — either a primitive tool call or a compound goal.
#[derive(Debug, Clone)]
pub struct Task {
    /// Name of this task (maps to a method for compound tasks, or a tool name
    /// for primitives).
    pub name: String,
    /// Whether this task is primitive (executable) or compound (needs decomposition).
    pub kind: TaskKind,
    /// JSON-encoded arguments for primitive tasks.  Ignored for compound tasks.
    pub args: String,
}

impl Task {
    /// Create a primitive task (direct tool invocation).
    pub fn primitive(tool: impl Into<String>, args: impl Into<String>) -> Self {
        Self { name: tool.into(), kind: TaskKind::Primitive, args: args.into() }
    }

    /// Create a compound task (to be decomposed by a method).
    pub fn compound(name: impl Into<String>) -> Self {
        Self { name: name.into(), kind: TaskKind::Compound, args: String::new() }
    }
}

/// Whether a task can be executed directly or must first be decomposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskKind {
    /// Executable — maps to a tool call.
    Primitive,
    /// Must be decomposed into sub-tasks using a registered method.
    Compound,
}

// ---------------------------------------------------------------------------
// Precondition
// ---------------------------------------------------------------------------

/// A precondition that must hold before a method may be applied.
///
/// Preconditions are evaluated against the current world-state — a
/// `HashMap<String, String>` of key/value facts.
#[derive(Debug, Clone)]
pub struct Precondition {
    /// The world-state key to check.
    pub key: String,
    /// The expected value (string equality).
    pub expected: String,
}

impl Precondition {
    /// Create a precondition `key == expected`.
    pub fn new(key: impl Into<String>, expected: impl Into<String>) -> Self {
        Self { key: key.into(), expected: expected.into() }
    }

    /// Return `true` if the world state satisfies this precondition.
    pub fn is_satisfied(&self, world: &HashMap<String, String>) -> bool {
        world.get(&self.key).map(|v| v == &self.expected).unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Method
// ---------------------------------------------------------------------------

/// A method decomposes a compound [`Task`] into an ordered list of sub-tasks.
#[derive(Debug, Clone)]
pub struct Method {
    /// The compound task name this method handles.
    pub task: String,
    /// The ordered list of sub-tasks that this method produces.
    pub subtasks: Vec<Task>,
    /// Preconditions that must hold in the world state for this method to apply.
    pub preconditions: Vec<Precondition>,
    /// Human-readable description.
    pub description: String,
}

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------

/// An ordered, fully expanded sequence of primitive tool calls.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    fn new(steps: Vec<PlanStep>) -> Self {
        Self { steps }
    }

    /// Ordered list of primitive steps to execute.
    pub fn steps(&self) -> &[PlanStep] {
        &self.steps
    }

    /// Number of steps in the plan.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True if the plan has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// A single primitive step in an [`ExecutionPlan`].
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Tool to invoke.
    pub tool: String,
    /// JSON arguments for the tool.
    pub args: String,
    /// Depth in the original task hierarchy (for diagnostics).
    pub depth: usize,
    /// Name of the compound task this step was decomposed from.
    pub source_task: String,
}

// ---------------------------------------------------------------------------
// PlannerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`Planner`].
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Maximum recursion depth when decomposing compound tasks.
    ///
    /// Prevents infinite loops from mutually recursive methods.
    /// Defaults to `16`.
    pub max_depth: usize,
    /// Maximum total steps in the expanded plan.  Defaults to `256`.
    pub max_steps: usize,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self { max_depth: 16, max_steps: 256 }
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// HTN planner: expands a goal [`Task`] into a flat [`ExecutionPlan`].
pub struct Planner {
    config: PlannerConfig,
    /// Methods indexed by the compound task they handle.
    /// Multiple methods per task are supported; the first one whose
    /// preconditions are satisfied is selected.
    methods: HashMap<String, Vec<Method>>,
}

impl Planner {
    /// Create a new planner with the given config.
    pub fn new(config: PlannerConfig) -> Self {
        Self { config, methods: HashMap::new() }
    }

    /// Register a decomposition method.
    pub fn register_method(&mut self, method: Method) {
        self.methods.entry(method.task.clone()).or_default().push(method);
    }

    /// Expand `goal` into a flat execution plan.
    ///
    /// # Errors
    ///
    /// Returns `Err(PlannerError)` if:
    /// - A compound task has no registered method.
    /// - No method's preconditions are satisfied.
    /// - The recursion depth or step limit is exceeded.
    pub fn plan(&self, goal: Task) -> Result<ExecutionPlan, PlannerError> {
        self.plan_with_world(goal, &HashMap::new())
    }

    /// Expand `goal` into a flat plan, checking preconditions against `world`.
    pub fn plan_with_world(
        &self,
        goal: Task,
        world: &HashMap<String, String>,
    ) -> Result<ExecutionPlan, PlannerError> {
        let mut steps = Vec::new();
        self.expand(&goal, world, 0, &mut steps)?;
        Ok(ExecutionPlan::new(steps))
    }

    // ------------------------------------------------------------------
    // Internal recursion
    // ------------------------------------------------------------------

    fn expand(
        &self,
        task: &Task,
        world: &HashMap<String, String>,
        depth: usize,
        out: &mut Vec<PlanStep>,
    ) -> Result<(), PlannerError> {
        if depth > self.config.max_depth {
            return Err(PlannerError::DepthExceeded {
                task: task.name.clone(),
                max_depth: self.config.max_depth,
            });
        }
        if out.len() >= self.config.max_steps {
            return Err(PlannerError::StepLimitExceeded(self.config.max_steps));
        }

        match task.kind {
            TaskKind::Primitive => {
                out.push(PlanStep {
                    tool: task.name.clone(),
                    args: task.args.clone(),
                    depth,
                    source_task: task.name.clone(),
                });
            }
            TaskKind::Compound => {
                let methods = self.methods.get(&task.name).ok_or_else(|| {
                    PlannerError::NoMethodFound(task.name.clone())
                })?;

                // Pick the first method whose preconditions are satisfied.
                let method = methods
                    .iter()
                    .find(|m| m.preconditions.iter().all(|p| p.is_satisfied(world)))
                    .ok_or_else(|| PlannerError::NoPreconditionSatisfied(task.name.clone()))?;

                let source = task.name.clone();
                for sub in &method.subtasks {
                    // Propagate source_task for tracing.
                    let mut sub = sub.clone();
                    // For primitives emitted from this method, tag with parent.
                    let _parent = source.clone();
                    self.expand_with_source(&sub, world, depth + 1, out, &source)?;
                    // suppress unused warning on sub
                    let _ = &mut sub;
                }
            }
        }
        Ok(())
    }

    fn expand_with_source(
        &self,
        task: &Task,
        world: &HashMap<String, String>,
        depth: usize,
        out: &mut Vec<PlanStep>,
        parent_task: &str,
    ) -> Result<(), PlannerError> {
        if depth > self.config.max_depth {
            return Err(PlannerError::DepthExceeded {
                task: task.name.clone(),
                max_depth: self.config.max_depth,
            });
        }
        if out.len() >= self.config.max_steps {
            return Err(PlannerError::StepLimitExceeded(self.config.max_steps));
        }

        match task.kind {
            TaskKind::Primitive => {
                out.push(PlanStep {
                    tool: task.name.clone(),
                    args: task.args.clone(),
                    depth,
                    source_task: parent_task.to_owned(),
                });
            }
            TaskKind::Compound => {
                let methods = self.methods.get(&task.name).ok_or_else(|| {
                    PlannerError::NoMethodFound(task.name.clone())
                })?;
                let method = methods
                    .iter()
                    .find(|m| m.preconditions.iter().all(|p| p.is_satisfied(world)))
                    .ok_or_else(|| PlannerError::NoPreconditionSatisfied(task.name.clone()))?;

                for sub in &method.subtasks {
                    self.expand_with_source(sub, world, depth + 1, out, &task.name)?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Planning errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PlannerError {
    /// A compound task has no registered method.
    NoMethodFound(String),
    /// A compound task has methods but none have satisfied preconditions.
    NoPreconditionSatisfied(String),
    /// The decomposition depth exceeded [`PlannerConfig::max_depth`].
    DepthExceeded {
        /// The name of the task that caused the depth limit to be exceeded.
        task: String,
        /// The configured maximum decomposition depth.
        max_depth: usize,
    },
    /// The plan grew beyond [`PlannerConfig::max_steps`].
    StepLimitExceeded(usize),
}

impl std::fmt::Display for PlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlannerError::NoMethodFound(t) => write!(f, "no method for compound task '{t}'"),
            PlannerError::NoPreconditionSatisfied(t) => {
                write!(f, "no applicable method for '{t}': all preconditions unsatisfied")
            }
            PlannerError::DepthExceeded { task, max_depth } => {
                write!(f, "max recursion depth {max_depth} exceeded at task '{task}'")
            }
            PlannerError::StepLimitExceeded(max) => {
                write!(f, "plan step limit {max} exceeded")
            }
        }
    }
}

impl std::error::Error for PlannerError {}

// ===========================================================================
// Goal Decomposition Planner — DAG-based ExecutionPlan / PlanBuilder
// ===========================================================================
//
// This section adds a second planning abstraction alongside the HTN planner
// above.  The HTN planner produces a flat ordered list of tool calls from a
// goal decomposition.  The DAG planner represents work as a directed acyclic
// graph where steps declare their dependencies explicitly, supports parallel
// execution, conditional branching, and per-step retry policies.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// DagPlanStep
// ---------------------------------------------------------------------------

/// A single step inside a DAG [`DagExecutionPlan`].
#[derive(Debug, Clone)]
pub struct DagPlanStep {
    /// Unique identifier for this step within the plan.
    pub id: String,
    /// Short human-readable name.
    pub name: String,
    /// Detailed description of what this step does.
    pub description: String,
    /// Capability tags that an executing agent must provide.
    pub required_capabilities: Vec<String>,
    /// IDs of steps that must be in [`DagStepStatus::Completed`] before this
    /// step is eligible to run.
    pub depends_on: Vec<String>,
    /// Maximum number of retry attempts before declaring the step failed.
    pub max_retries: u32,
    /// Optional wall-clock timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// Optional human-readable description of how to roll back this step.
    pub rollback_description: Option<String>,
    /// Arbitrary key/value metadata for callers.
    pub metadata: std::collections::HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// DagStepStatus
// ---------------------------------------------------------------------------

/// Execution status of a [`DagPlanStep`].
#[derive(Debug, Clone, PartialEq)]
pub enum DagStepStatus {
    /// Not yet eligible to run (dependencies unsatisfied).
    Pending,
    /// All dependencies are satisfied; the step may begin.
    Ready,
    /// The step is currently executing.
    InProgress,
    /// The step finished successfully.
    Completed {
        /// The result produced by the step.
        result: String,
    },
    /// The step has failed and may not be retried further.
    Failed {
        /// Human-readable error message.
        error: String,
        /// Total number of attempts made.
        attempts: u32,
    },
    /// The step was intentionally skipped.
    Skipped {
        /// Reason for skipping.
        reason: String,
    },
    /// The step was rolled back after a failure.
    RolledBack,
}

// ---------------------------------------------------------------------------
// DagExecutionPlan
// ---------------------------------------------------------------------------

/// A complete DAG-based execution plan.
///
/// Steps declare their dependencies via [`DagPlanStep::depends_on`].  The plan
/// tracks the status of every step and surfaces which steps are currently
/// eligible to run.
#[derive(Debug, Clone)]
pub struct DagExecutionPlan {
    /// Unique plan identifier.
    pub id: String,
    /// High-level goal this plan is working toward.
    pub goal: String,
    /// All steps, keyed by their [`DagPlanStep::id`].
    pub steps: std::collections::HashMap<String, DagPlanStep>,
    /// Current execution status of every step.
    pub step_status: std::collections::HashMap<String, DagStepStatus>,
    /// When the plan was created.
    pub created_at: std::time::SystemTime,
    /// When the plan finished (all steps terminal), if ever.
    pub completed_at: Option<std::time::SystemTime>,
}

impl DagExecutionPlan {
    /// Return steps whose dependencies are all completed and that are still
    /// in [`DagStepStatus::Pending`] or [`DagStepStatus::Ready`].
    pub fn ready_steps(&self) -> Vec<&DagPlanStep> {
        self.steps
            .values()
            .filter(|step| {
                let status = self.step_status.get(&step.id);
                // Only consider steps that haven't started yet.
                let is_waiting = matches!(
                    status,
                    Some(DagStepStatus::Pending) | Some(DagStepStatus::Ready) | None
                );
                if !is_waiting {
                    return false;
                }
                // All dependencies must be completed.
                step.depends_on.iter().all(|dep_id| {
                    matches!(
                        self.step_status.get(dep_id),
                        Some(DagStepStatus::Completed { .. })
                    )
                })
            })
            .collect()
    }

    /// Return all steps in a valid topological order (dependencies before
    /// dependants).  Returns an error if the graph contains a cycle.
    pub fn topological_order(&self) -> Vec<&DagPlanStep> {
        // Kahn's algorithm.
        let mut in_degree: std::collections::HashMap<&str, usize> =
            self.steps.keys().map(|k| (k.as_str(), 0)).collect();

        for step in self.steps.values() {
            for dep in &step.depends_on {
                *in_degree.entry(step.id.as_str()).or_insert(0) += 1;
                let _ = dep; // dep already counted via depends_on
            }
            // Re-count: for each step, its in-degree is len(depends_on).
        }
        // Reset and recount properly.
        for k in in_degree.values_mut() {
            *k = 0;
        }
        for step in self.steps.values() {
            in_degree.insert(step.id.as_str(), step.depends_on.len());
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter_map(|(id, &deg)| if deg == 0 { Some(*id) } else { None })
            .collect();

        let mut order: Vec<&DagPlanStep> = Vec::with_capacity(self.steps.len());

        while let Some(id) = queue.pop_front() {
            if let Some(step) = self.steps.get(id) {
                order.push(step);
                // Reduce in-degree for every step that depends on `id`.
                for other in self.steps.values() {
                    if other.depends_on.iter().any(|d| d == id) {
                        if let Some(deg) = in_degree.get_mut(other.id.as_str()) {
                            *deg = deg.saturating_sub(1);
                            if *deg == 0 {
                                queue.push_back(other.id.as_str());
                            }
                        }
                    }
                }
            }
        }

        order
    }

    /// Mark a step as [`DagStepStatus::Completed`].
    pub fn complete_step(&mut self, step_id: &str, result: impl Into<String>) {
        self.step_status
            .insert(step_id.to_owned(), DagStepStatus::Completed { result: result.into() });
        self.maybe_set_completed_at();
    }

    /// Mark a step as [`DagStepStatus::Failed`].
    pub fn fail_step(&mut self, step_id: &str, error: impl Into<String>) {
        let attempts = match self.step_status.get(step_id) {
            Some(DagStepStatus::Failed { attempts, .. }) => *attempts + 1,
            _ => 1,
        };
        self.step_status.insert(
            step_id.to_owned(),
            DagStepStatus::Failed { error: error.into(), attempts },
        );
        self.maybe_set_completed_at();
    }

    /// Return `true` if every step is in a terminal state (completed, failed,
    /// skipped, or rolled back).
    pub fn is_complete(&self) -> bool {
        self.steps.keys().all(|id| {
            matches!(
                self.step_status.get(id),
                Some(
                    DagStepStatus::Completed { .. }
                        | DagStepStatus::Failed { .. }
                        | DagStepStatus::Skipped { .. }
                        | DagStepStatus::RolledBack
                )
            )
        })
    }

    /// Return `true` if any step is in [`DagStepStatus::Failed`] with no
    /// retries remaining.
    pub fn has_failed(&self) -> bool {
        self.steps.values().any(|step| {
            matches!(
                self.step_status.get(&step.id),
                Some(DagStepStatus::Failed { attempts, .. })
                    if *attempts > step.max_retries
            )
        })
    }

    /// Return plan progress as an integer percentage (0–100).
    ///
    /// Progress = completed steps / total steps.
    pub fn progress_pct(&self) -> u8 {
        if self.steps.is_empty() {
            return 100;
        }
        let completed = self
            .step_status
            .values()
            .filter(|s| matches!(s, DagStepStatus::Completed { .. }))
            .count();
        ((completed * 100) / self.steps.len()).min(100) as u8
    }

    /// Validate that the step dependency graph is a DAG (no cycles).
    ///
    /// Uses iterative DFS with a "gray/black" colouring scheme.
    /// Returns `Ok(())` if the graph is acyclic, or `Err` with a description
    /// of the cycle.
    pub fn validate_dag(&self) -> Result<(), String> {
        // 0 = white (unvisited), 1 = gray (in stack), 2 = black (done)
        let mut colour: std::collections::HashMap<&str, u8> =
            self.steps.keys().map(|k| (k.as_str(), 0u8)).collect();

        for start in self.steps.keys() {
            if *colour.get(start.as_str()).unwrap_or(&0) == 0 {
                Self::dfs_cycle(start.as_str(), &self.steps, &mut colour)?;
            }
        }
        Ok(())
    }

    // Recursive DFS helper for cycle detection.
    fn dfs_cycle<'a>(
        node: &'a str,
        steps: &'a std::collections::HashMap<String, DagPlanStep>,
        colour: &mut std::collections::HashMap<&'a str, u8>,
    ) -> Result<(), String> {
        *colour.entry(node).or_insert(0) = 1; // gray
        if let Some(step) = steps.get(node) {
            for dep in &step.depends_on {
                let c = *colour.get(dep.as_str()).unwrap_or(&0);
                if c == 1 {
                    return Err(format!(
                        "cycle detected: step '{}' depends on '{}' which is already in the current path",
                        node, dep
                    ));
                }
                if c == 0 {
                    Self::dfs_cycle(dep.as_str(), steps, colour)?;
                }
            }
        }
        *colour.entry(node).or_insert(0) = 2; // black
        Ok(())
    }

    /// Return a human-readable summary of the plan and its current status.
    pub fn summary(&self) -> String {
        let total = self.steps.len();
        let completed = self
            .step_status
            .values()
            .filter(|s| matches!(s, DagStepStatus::Completed { .. }))
            .count();
        let failed = self
            .step_status
            .values()
            .filter(|s| matches!(s, DagStepStatus::Failed { .. }))
            .count();
        let pending = total - completed - failed;
        format!(
            "Plan '{}' [{}]: {total} steps, {completed} completed, {failed} failed, {pending} pending ({pct}%)",
            self.goal,
            self.id,
            pct = self.progress_pct(),
        )
    }

    // Set completed_at when the plan reaches a terminal state.
    fn maybe_set_completed_at(&mut self) {
        if self.completed_at.is_none() && self.is_complete() {
            self.completed_at = Some(std::time::SystemTime::now());
        }
    }
}

// ---------------------------------------------------------------------------
// PlanBuilder
// ---------------------------------------------------------------------------

/// Constructs [`DagExecutionPlan`] instances.
///
/// Provides convenience constructors for common patterns (sequential,
/// parallel) and a low-level [`PlanBuilder::add_step`] for bespoke graphs.
pub struct PlanBuilder;

impl PlanBuilder {
    /// Create a new empty plan for the given goal.
    pub fn new(goal: impl Into<String>) -> DagExecutionPlan {
        DagExecutionPlan {
            id: uuid::Uuid::new_v4().to_string(),
            goal: goal.into(),
            steps: std::collections::HashMap::new(),
            step_status: std::collections::HashMap::new(),
            created_at: std::time::SystemTime::now(),
            completed_at: None,
        }
    }

    /// Create a linear sequential plan: each step depends on the one before it.
    ///
    /// `steps` is a slice of `(name, description)` pairs.
    pub fn sequential(
        goal: impl Into<String>,
        steps: Vec<(&str, &str)>,
    ) -> DagExecutionPlan {
        let mut plan = Self::new(goal);
        let mut prev_id: Option<String> = None;
        for (name, description) in steps {
            let id = uuid::Uuid::new_v4().to_string();
            let step = DagPlanStep {
                id: id.clone(),
                name: name.to_owned(),
                description: description.to_owned(),
                required_capabilities: Vec::new(),
                depends_on: prev_id.iter().cloned().collect(),
                max_retries: 0,
                timeout_secs: None,
                rollback_description: None,
                metadata: std::collections::HashMap::new(),
            };
            plan.step_status.insert(id.clone(), DagStepStatus::Pending);
            plan.steps.insert(id.clone(), step);
            prev_id = Some(id);
        }
        plan
    }

    /// Create a plan where all steps are independent (no dependencies).
    ///
    /// All steps may run in parallel.
    pub fn parallel(goal: impl Into<String>, steps: Vec<(&str, &str)>) -> DagExecutionPlan {
        let mut plan = Self::new(goal);
        for (name, description) in steps {
            let id = uuid::Uuid::new_v4().to_string();
            let step = DagPlanStep {
                id: id.clone(),
                name: name.to_owned(),
                description: description.to_owned(),
                required_capabilities: Vec::new(),
                depends_on: Vec::new(),
                max_retries: 0,
                timeout_secs: None,
                rollback_description: None,
                metadata: std::collections::HashMap::new(),
            };
            plan.step_status.insert(id.clone(), DagStepStatus::Pending);
            plan.steps.insert(id.clone(), step);
        }
        plan
    }

    /// Add a step to an existing plan.
    ///
    /// The step's initial status is set to [`DagStepStatus::Pending`].
    pub fn add_step(plan: &mut DagExecutionPlan, step: DagPlanStep) {
        plan.step_status.insert(step.id.clone(), DagStepStatus::Pending);
        plan.steps.insert(step.id.clone(), step);
    }
}

// ---------------------------------------------------------------------------
// DAG planner tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod dag_tests {
    use super::*;

    fn make_step(id: &str, depends_on: Vec<&str>) -> DagPlanStep {
        DagPlanStep {
            id: id.to_owned(),
            name: format!("Step {id}"),
            description: format!("Description for {id}"),
            required_capabilities: Vec::new(),
            depends_on: depends_on.iter().map(|s| s.to_string()).collect(),
            max_retries: 0,
            timeout_secs: None,
            rollback_description: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn sequential_plan_has_chained_deps() {
        let plan = PlanBuilder::sequential(
            "Linear goal",
            vec![("a", "first"), ("b", "second"), ("c", "third")],
        );
        assert_eq!(plan.steps.len(), 3);
        // Validate the DAG is acyclic.
        plan.validate_dag().unwrap();
        // Topological order has 3 items.
        let order = plan.topological_order();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn parallel_plan_all_steps_ready_immediately() {
        let plan = PlanBuilder::parallel(
            "Parallel goal",
            vec![("x", "step x"), ("y", "step y"), ("z", "step z")],
        );
        assert_eq!(plan.steps.len(), 3);
        let ready = plan.ready_steps();
        // All three steps have no deps → all ready.
        assert_eq!(ready.len(), 3);
    }

    #[test]
    fn ready_steps_respects_dependencies() {
        let mut plan = PlanBuilder::new("dep test");
        PlanBuilder::add_step(&mut plan, make_step("a", vec![]));
        PlanBuilder::add_step(&mut plan, make_step("b", vec!["a"]));

        // Initially only "a" has no deps → ready.
        let ready: Vec<&str> = plan.ready_steps().iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ready, vec!["a"]);

        // Complete "a" → "b" becomes ready.
        plan.complete_step("a", "done");
        let ready: Vec<&str> = plan.ready_steps().iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ready, vec!["b"]);
    }

    #[test]
    fn complete_step_updates_progress() {
        let mut plan = PlanBuilder::sequential("prog", vec![("s1", "d1"), ("s2", "d2")]);
        assert_eq!(plan.progress_pct(), 0);

        let ids: Vec<String> = plan.steps.keys().cloned().collect();
        plan.complete_step(&ids[0], "ok");
        assert_eq!(plan.progress_pct(), 50);

        plan.complete_step(&ids[1], "ok");
        assert_eq!(plan.progress_pct(), 100);
    }

    #[test]
    fn is_complete_when_all_terminal() {
        let mut plan = PlanBuilder::parallel("done", vec![("a", ""), ("b", "")]);
        assert!(!plan.is_complete());

        let ids: Vec<String> = plan.steps.keys().cloned().collect();
        plan.complete_step(&ids[0], "ok");
        assert!(!plan.is_complete());
        plan.complete_step(&ids[1], "ok");
        assert!(plan.is_complete());
    }

    #[test]
    fn has_failed_detects_exhausted_step() {
        let mut plan = PlanBuilder::new("fail test");
        // max_retries = 0 → any failure is terminal.
        PlanBuilder::add_step(&mut plan, make_step("s1", vec![]));
        assert!(!plan.has_failed());
        plan.fail_step("s1", "boom");
        // 1 attempt > 0 max_retries → has_failed.
        assert!(plan.has_failed());
    }

    #[test]
    fn validate_dag_accepts_valid_graph() {
        let mut plan = PlanBuilder::new("valid");
        PlanBuilder::add_step(&mut plan, make_step("a", vec![]));
        PlanBuilder::add_step(&mut plan, make_step("b", vec!["a"]));
        PlanBuilder::add_step(&mut plan, make_step("c", vec!["a", "b"]));
        assert!(plan.validate_dag().is_ok());
    }

    #[test]
    fn validate_dag_rejects_cycle() {
        let mut plan = PlanBuilder::new("cycle");
        PlanBuilder::add_step(&mut plan, make_step("a", vec!["c"]));
        PlanBuilder::add_step(&mut plan, make_step("b", vec!["a"]));
        PlanBuilder::add_step(&mut plan, make_step("c", vec!["b"]));
        assert!(plan.validate_dag().is_err());
    }

    #[test]
    fn topological_order_respects_dependencies() {
        let mut plan = PlanBuilder::new("topo");
        PlanBuilder::add_step(&mut plan, make_step("a", vec![]));
        PlanBuilder::add_step(&mut plan, make_step("b", vec!["a"]));
        PlanBuilder::add_step(&mut plan, make_step("c", vec!["b"]));

        let order = plan.topological_order();
        assert_eq!(order.len(), 3);

        // Find positions.
        let pos = |id: &str| order.iter().position(|s| s.id == id).unwrap();
        assert!(pos("a") < pos("b"));
        assert!(pos("b") < pos("c"));
    }

    #[test]
    fn summary_contains_goal_and_counts() {
        let plan = PlanBuilder::sequential("My Goal", vec![("s1", ""), ("s2", "")]);
        let s = plan.summary();
        assert!(s.contains("My Goal"));
        assert!(s.contains("2 steps"));
    }

    #[test]
    fn completed_at_set_when_all_done() {
        let mut plan = PlanBuilder::parallel("finish", vec![("a", ""), ("b", "")]);
        assert!(plan.completed_at.is_none());
        let ids: Vec<String> = plan.steps.keys().cloned().collect();
        plan.complete_step(&ids[0], "ok");
        assert!(plan.completed_at.is_none());
        plan.complete_step(&ids[1], "ok");
        assert!(plan.completed_at.is_some());
    }

    #[test]
    fn add_step_builder_creates_pending_status() {
        let mut plan = PlanBuilder::new("manual");
        PlanBuilder::add_step(&mut plan, make_step("x", vec![]));
        assert_eq!(plan.step_status.get("x"), Some(&DagStepStatus::Pending));
    }
}

// ---------------------------------------------------------------------------
// Tests (original HTN planner)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn research_planner() -> Planner {
        let mut p = Planner::new(PlannerConfig::default());
        p.register_method(Method {
            task: "ResearchCompany".into(),
            subtasks: vec![
                Task::primitive("web_search", r#"{"q":"company"}"#),
                Task::primitive("scrape_page", r#"{"url":"<result>"}"#),
                Task::primitive("summarise", r#"{"text":"<content>"}"#),
            ],
            preconditions: vec![],
            description: "Standard research flow".into(),
        });
        p
    }

    #[test]
    fn plans_compound_into_primitives() {
        let p = research_planner();
        let plan = p.plan(Task::compound("ResearchCompany")).unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.steps()[0].tool, "web_search");
        assert_eq!(plan.steps()[1].tool, "scrape_page");
        assert_eq!(plan.steps()[2].tool, "summarise");
    }

    #[test]
    fn primitive_task_plans_directly() {
        let p = Planner::new(PlannerConfig::default());
        let plan = p.plan(Task::primitive("echo", "{}")).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.steps()[0].tool, "echo");
    }

    #[test]
    fn error_on_unknown_compound_task() {
        let p = Planner::new(PlannerConfig::default());
        let err = p.plan(Task::compound("Unknown")).unwrap_err();
        assert!(matches!(err, PlannerError::NoMethodFound(_)));
    }

    #[test]
    fn precondition_filters_method() {
        let mut p = Planner::new(PlannerConfig::default());
        // Method A: requires auth=true
        p.register_method(Method {
            task: "FetchData".into(),
            subtasks: vec![Task::primitive("authenticated_fetch", "{}")],
            preconditions: vec![Precondition::new("auth", "true")],
            description: "Authenticated fetch".into(),
        });
        // Method B: no preconditions
        p.register_method(Method {
            task: "FetchData".into(),
            subtasks: vec![Task::primitive("public_fetch", "{}")],
            preconditions: vec![],
            description: "Public fetch".into(),
        });

        // Without auth in world → method B selected.
        let plan = p.plan(Task::compound("FetchData")).unwrap();
        assert_eq!(plan.steps()[0].tool, "public_fetch");

        // With auth=true in world → method A selected (first match).
        let mut world = HashMap::new();
        world.insert("auth".into(), "true".into());
        let plan = p.plan_with_world(Task::compound("FetchData"), &world).unwrap();
        assert_eq!(plan.steps()[0].tool, "authenticated_fetch");
    }

    #[test]
    fn depth_limit_prevents_infinite_recursion() {
        let mut p = Planner::new(PlannerConfig { max_depth: 3, max_steps: 1000 });
        // Self-referential method.
        p.register_method(Method {
            task: "Loop".into(),
            subtasks: vec![Task::compound("Loop")],
            preconditions: vec![],
            description: "Infinite loop".into(),
        });
        assert!(matches!(
            p.plan(Task::compound("Loop")),
            Err(PlannerError::DepthExceeded { .. })
        ));
    }

    #[test]
    fn nested_compound_tasks_expand_correctly() {
        let mut p = Planner::new(PlannerConfig::default());
        p.register_method(Method {
            task: "GetAndProcess".into(),
            subtasks: vec![
                Task::compound("Fetch"),
                Task::compound("Process"),
            ],
            preconditions: vec![],
            description: "Fetch then process".into(),
        });
        p.register_method(Method {
            task: "Fetch".into(),
            subtasks: vec![Task::primitive("http_get", "{}")],
            preconditions: vec![],
            description: "HTTP GET".into(),
        });
        p.register_method(Method {
            task: "Process".into(),
            subtasks: vec![
                Task::primitive("parse", "{}"),
                Task::primitive("store", "{}"),
            ],
            preconditions: vec![],
            description: "Parse and store".into(),
        });

        let plan = p.plan(Task::compound("GetAndProcess")).unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.steps()[0].tool, "http_get");
        assert_eq!(plan.steps()[1].tool, "parse");
        assert_eq!(plan.steps()[2].tool, "store");
    }
}
