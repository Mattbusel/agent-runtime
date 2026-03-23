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
    DepthExceeded { task: String, max_depth: usize },
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

// ---------------------------------------------------------------------------
// Tests
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
