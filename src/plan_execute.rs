//! # Module: plan_execute
//!
//! ## Responsibility
//! Implements a full **Plan → Execute → Verify** agent loop as a structured
//! alternative to the open-ended ReAct loop.
//!
//! A [`PlanExecuteAgent`] works in three phases:
//!
//! 1. **Plan** — calls the LLM inference function with the goal string and
//!    expects a numbered list of steps.  Each step carries a description, an
//!    optional tool name, and an expected outcome.
//! 2. **Execute** — runs each step in sequence.  If a `tool` is named the
//!    agent looks it up in its tool registry and calls it; otherwise the
//!    inference function is called with a focused single-step prompt.
//! 3. **Verify** — calls the inference function one final time, passing the
//!    full plan + all collected outputs, and asks the LLM to judge whether the
//!    goal was achieved.  Returns a [`VerificationResult`].
//!
//! Wire into [`AgentRuntime`] via the [`AgentRuntime::run_plan_execute`]
//! method (see `runtime.rs`).
//!
//! ## Guarantees
//! - Non-panicking: all operations return `Result` or an infallible value.
//! - Bounded: execution terminates after at most `max_steps` steps.
//! - Deterministic: given the same inference function output the loop produces
//!   the same plan and verification.
//!
//! ## NOT Responsible For
//! - Actual LLM inference (callers supply an async inference closure).
//! - Parallel step execution (steps run sequentially by design).

use std::{future::Future, time::Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::agent::ToolSpec;
use crate::error::AgentRuntimeError;
use crate::types::AgentId;

// ── StepStatus ────────────────────────────────────────────────────────────────

/// Execution status of a single [`PlanStep`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StepStatus {
    /// The step has not been attempted yet.
    Pending,
    /// The step is currently executing.
    Running,
    /// The step completed successfully; carries the tool or inference output.
    Completed(String),
    /// The step failed; carries the error message.
    Failed(String),
    /// The step was skipped due to a preceding failure or policy decision.
    Skipped,
}

impl StepStatus {
    /// Return `true` if this step completed successfully.
    pub fn is_completed(&self) -> bool {
        matches!(self, StepStatus::Completed(_))
    }

    /// Return `true` if this step failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, StepStatus::Failed(_))
    }

    /// Return the output string if the step completed, or `None` otherwise.
    pub fn output(&self) -> Option<&str> {
        match self {
            StepStatus::Completed(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ── PlanStep ──────────────────────────────────────────────────────────────────

/// A single step within a [`Plan`].
///
/// Steps are created by the LLM planning call and executed sequentially by
/// [`PlanExecuteAgent::execute`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Human-readable description of what this step should do.
    pub description: String,
    /// Optional tool name to invoke.  When `None`, the inference function is
    /// called directly with a focused single-step prompt.
    pub tool: Option<String>,
    /// The outcome the planner expects after this step succeeds.
    pub expected_outcome: String,
    /// Current execution status of this step.
    pub status: StepStatus,
}

impl PlanStep {
    /// Construct a new `PlanStep` in the [`StepStatus::Pending`] state.
    pub fn new(
        description: impl Into<String>,
        tool: Option<String>,
        expected_outcome: impl Into<String>,
    ) -> Self {
        Self {
            description: description.into(),
            tool,
            expected_outcome: expected_outcome.into(),
            status: StepStatus::Pending,
        }
    }
}

// ── Plan ─────────────────────────────────────────────────────────────────────

/// A multi-step execution plan produced by the LLM planning call.
///
/// A `Plan` is immutable after creation: the steps are set once by the
/// planning phase and their `status` fields are mutated in place during
/// execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// The high-level goal this plan is intended to achieve.
    pub goal: String,
    /// Ordered list of steps to execute.
    pub steps: Vec<PlanStep>,
    /// Monotonic instant at which planning completed (not serialised).
    #[serde(skip)]
    pub created_at: Option<Instant>,
}

impl Plan {
    /// Create a new plan for the given `goal` with the supplied `steps`.
    pub fn new(goal: impl Into<String>, steps: Vec<PlanStep>) -> Self {
        Self {
            goal: goal.into(),
            steps,
            created_at: Some(Instant::now()),
        }
    }

    /// Return the number of steps in this plan.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the number of successfully completed steps.
    pub fn completed_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.status.is_completed())
            .count()
    }

    /// Return the number of failed steps.
    pub fn failed_count(&self) -> usize {
        self.steps.iter().filter(|s| s.status.is_failed()).count()
    }

    /// Return `true` if every step completed successfully.
    pub fn all_succeeded(&self) -> bool {
        self.steps.iter().all(|s| s.status.is_completed())
    }

    /// Collect all step outputs in order, skipping failed and skipped steps.
    pub fn outputs(&self) -> Vec<&str> {
        self.steps
            .iter()
            .filter_map(|s| s.status.output())
            .collect()
    }
}

// ── VerificationResult ────────────────────────────────────────────────────────

/// The outcome of the verification phase.
///
/// Produced by [`Verifier::verify`] after the LLM reviews the full plan and
/// all step outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the verifier judged the goal to have been achieved.
    pub achieved: bool,
    /// Confidence in the judgement, in `[0.0, 1.0]`.
    pub confidence: f64,
    /// List of issues identified by the verifier (empty when `achieved = true`
    /// and there are no caveats).
    pub issues: Vec<String>,
    /// Raw LLM response from the verification call (useful for debugging).
    pub raw_response: String,
}

impl VerificationResult {
    /// Return `true` when the goal was achieved with confidence above `0.5`.
    pub fn is_confident_success(&self) -> bool {
        self.achieved && self.confidence > 0.5
    }
}

// ── Verifier ─────────────────────────────────────────────────────────────────

/// Re-runs the LLM with the completed plan and all outputs to judge whether
/// the original goal was achieved.
///
/// The verifier expects the inference function to return a response that
/// contains one of the following tokens (case-insensitive):
///
/// - `"ACHIEVED"` or `"YES"` — goal was achieved.
/// - `"NOT_ACHIEVED"`, `"FAILED"`, or `"NO"` — goal was not achieved.
///
/// Anything after the status token on the same line is treated as an issue
/// or confidence note and appended to [`VerificationResult::issues`].
pub struct Verifier;

impl Verifier {
    /// Build the verification prompt from a completed plan.
    fn build_prompt(plan: &Plan) -> String {
        let mut prompt = format!(
            "You are verifying whether the following goal was achieved.\n\nGOAL: {}\n\nPLAN STEPS AND OUTPUTS:\n",
            plan.goal
        );
        for (i, step) in plan.steps.iter().enumerate() {
            let status_str = match &step.status {
                StepStatus::Completed(out) => format!("Completed — output: {out}"),
                StepStatus::Failed(err) => format!("Failed — error: {err}"),
                StepStatus::Skipped => "Skipped".to_string(),
                StepStatus::Pending => "Not started".to_string(),
                StepStatus::Running => "Still running".to_string(),
            };
            prompt.push_str(&format!(
                "Step {}: {} [expected: {}] — {}\n",
                i + 1,
                step.description,
                step.expected_outcome,
                status_str,
            ));
        }
        prompt.push_str(
            "\nRespond with ACHIEVED or NOT_ACHIEVED on the first line, \
             followed by a confidence score 0.0-1.0, then any issues on \
             subsequent lines.\nExample:\nACHIEVED 0.95\n",
        );
        prompt
    }

    /// Parse the raw inference response into a [`VerificationResult`].
    fn parse_response(raw: &str) -> VerificationResult {
        let upper = raw.trim().to_uppercase();
        let achieved = upper.contains("ACHIEVED") && !upper.contains("NOT_ACHIEVED");

        // Extract confidence: look for a float on the first line.
        let confidence = raw
            .lines()
            .next()
            .and_then(|line| {
                line.split_whitespace()
                    .find_map(|tok| tok.parse::<f64>().ok())
            })
            .unwrap_or(if achieved { 0.8 } else { 0.2 })
            .clamp(0.0, 1.0);

        // Lines after the first are issues.
        let issues: Vec<String> = raw
            .lines()
            .skip(1)
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();

        VerificationResult {
            achieved,
            confidence,
            issues,
            raw_response: raw.to_string(),
        }
    }

    /// Run the verification phase.
    ///
    /// Calls `infer` with a prompt that includes the full plan and all
    /// collected outputs, then parses the response into a
    /// [`VerificationResult`].
    ///
    /// # Arguments
    ///
    /// * `plan`  — the completed (or partially completed) plan.
    /// * `infer` — the same async inference closure used during execution.
    pub async fn verify<F, Fut>(plan: &Plan, infer: F) -> VerificationResult
    where
        F: FnOnce(String) -> Fut,
        Fut: Future<Output = String>,
    {
        let prompt = Self::build_prompt(plan);
        debug!("verifier: calling inference for goal verification");
        let raw = infer(prompt).await;
        let result = Self::parse_response(&raw);
        info!(
            achieved = result.achieved,
            confidence = result.confidence,
            issues = result.issues.len(),
            "verification complete"
        );
        result
    }
}

// ── PlanExecuteAgent ──────────────────────────────────────────────────────────

/// Configuration for a [`PlanExecuteAgent`].
#[derive(Debug, Clone)]
pub struct PlanExecuteConfig {
    /// Maximum number of steps the agent will execute.
    /// Steps beyond this limit are marked [`StepStatus::Skipped`].
    /// Default: `20`.
    pub max_steps: usize,
    /// If `true`, execution halts on the first failed step; remaining steps
    /// are marked `Skipped`.  If `false`, execution continues past failures.
    /// Default: `true`.
    pub stop_on_failure: bool,
    /// System prompt prefix injected into the planning call.
    pub system_prompt: String,
}

impl Default for PlanExecuteConfig {
    fn default() -> Self {
        Self {
            max_steps: 20,
            stop_on_failure: true,
            system_prompt: "You are a planning agent. Produce clear, actionable plans."
                .to_string(),
        }
    }
}

/// A Plan → Execute → Verify agent loop.
///
/// [`PlanExecuteAgent`] first asks the LLM to decompose the `goal` into an
/// ordered list of [`PlanStep`]s, executes each step in sequence using the
/// supplied tools or inference function, and finally calls a [`Verifier`] to
/// confirm whether the goal was achieved.
///
/// # Example
///
/// ```rust,no_run
/// use llm_agent_runtime::plan_execute::{PlanExecuteAgent, PlanExecuteConfig};
/// use llm_agent_runtime::types::AgentId;
///
/// #[tokio::main]
/// async fn main() {
///     let agent = PlanExecuteAgent::new(AgentId::new("planner"), PlanExecuteConfig::default());
///
///     let mut step_n = 0usize;
///     let (plan, verification) = agent.run(
///         "Summarise the quarterly results and send a report",
///         |ctx: String| {
///             step_n += 1;
///             let n = step_n;
///             async move {
///                 if n == 1 {
///                     // Planning response: one step per line, number-prefixed.
///                     "1. Retrieve quarterly data | tool:none | expected:data retrieved\n\
///                      2. Summarise the data      | tool:none | expected:summary written\n\
///                      3. Send the report         | tool:none | expected:report sent\n"
///                         .to_string()
///                 } else if ctx.contains("VERIFY") || ctx.contains("verifying") {
///                     "ACHIEVED 0.90\nAll steps completed successfully.".to_string()
///                 } else {
///                     "Step output: done.".to_string()
///                 }
///             }
///         },
///         &[],
///     )
///     .await
///     .unwrap();
///
///     println!(
///         "Plan: {} steps, verification: {}",
///         plan.step_count(),
///         verification.achieved
///     );
/// }
/// ```
pub struct PlanExecuteAgent {
    /// The agent identifier used for logging and tracing.
    pub agent_id: AgentId,
    /// Configuration for this agent instance.
    pub config: PlanExecuteConfig,
}

impl PlanExecuteAgent {
    /// Create a new `PlanExecuteAgent` with the given ID and configuration.
    pub fn new(agent_id: AgentId, config: PlanExecuteConfig) -> Self {
        Self { agent_id, config }
    }

    // ── Planning ──────────────────────────────────────────────────────────────

    /// Build the planning prompt for `goal`.
    fn planning_prompt(&self, goal: &str) -> String {
        format!(
            "{}\n\nGoal: {}\n\n\
             Produce a numbered step-by-step plan. Each line must follow this format:\n\
             <number>. <description> | tool:<tool_name_or_none> | expected:<outcome>\n\n\
             Example:\n\
             1. Search the web for recent data | tool:web_search | expected:search results returned\n\
             2. Summarise findings             | tool:none        | expected:summary paragraph written\n",
            self.config.system_prompt, goal
        )
    }

    /// Parse the LLM's planning response into a [`Plan`].
    ///
    /// Accepts lines matching the pattern:
    /// ```text
    /// <number>. <description> | tool:<name> | expected:<outcome>
    /// ```
    /// Lines that do not match are silently skipped.
    fn parse_plan(goal: &str, response: &str) -> Plan {
        let mut steps: Vec<PlanStep> = Vec::new();

        for line in response.lines() {
            let line = line.trim();
            // Must start with a digit + dot.
            let after_num = line
                .find(". ")
                .and_then(|dot| {
                    let prefix = &line[..dot];
                    if prefix.chars().all(|c| c.is_ascii_digit()) {
                        Some(&line[dot + 2..])
                    } else {
                        None
                    }
                });

            let body = match after_num {
                Some(b) => b,
                None => continue,
            };

            // Split on " | " separators.
            let parts: Vec<&str> = body.splitn(3, " | ").collect();
            if parts.is_empty() {
                continue;
            }

            let description = parts[0].trim().to_string();

            let tool = parts
                .get(1)
                .and_then(|p| p.strip_prefix("tool:"))
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty() && t != "none");

            let expected_outcome = parts
                .get(2)
                .and_then(|p| p.strip_prefix("expected:"))
                .map(|e| e.trim().to_string())
                .unwrap_or_else(|| "step completed".to_string());

            if !description.is_empty() {
                steps.push(PlanStep::new(description, tool, expected_outcome));
            }
        }

        Plan::new(goal, steps)
    }

    // ── Execution ─────────────────────────────────────────────────────────────

    /// Build the single-step inference prompt for executing `step` within
    /// `plan`.
    fn step_prompt(plan: &Plan, step_index: usize, step: &PlanStep) -> String {
        format!(
            "You are executing step {} of {} for the goal: {}\n\n\
             Step: {}\nExpected outcome: {}\n\n\
             Produce only the output for this step. Be concise.",
            step_index + 1,
            plan.step_count(),
            plan.goal,
            step.description,
            step.expected_outcome,
        )
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Run the full Plan → Execute → Verify loop.
    ///
    /// `infer` is called multiple times:
    /// - Once for planning (receives the planning prompt).
    /// - Once per step that has no registered tool (receives a step prompt).
    /// - Once for verification (receives the verification prompt).
    ///
    /// `tools` is a slice of [`ToolSpec`]s whose names can be referenced by
    /// plan steps.  When a step names a registered tool, the tool's handler is
    /// called synchronously with an empty JSON object `{}` and the result is
    /// used as the step output.
    ///
    /// # Errors
    ///
    /// Returns [`AgentRuntimeError`] if planning produces an empty plan (which
    /// indicates that the LLM returned an unrecognisable response).
    pub async fn run<F, Fut>(
        &self,
        goal: &str,
        infer: F,
        tools: &[std::sync::Arc<ToolSpec>],
    ) -> Result<(Plan, VerificationResult), AgentRuntimeError>
    where
        F: Fn(String) -> Fut,
        Fut: Future<Output = String>,
    {
        let start = Instant::now();
        info!(agent = %self.agent_id, goal, "plan-execute: starting");

        // ── Phase 1: Plan ─────────────────────────────────────────────────────
        let planning_prompt = self.planning_prompt(goal);
        debug!(agent = %self.agent_id, "plan-execute: calling inference for plan");
        let plan_response = infer(planning_prompt).await;
        let mut plan = Self::parse_plan(goal, &plan_response);

        if plan.step_count() == 0 {
            warn!(agent = %self.agent_id, "plan-execute: LLM returned empty plan");
            return Err(AgentRuntimeError::AgentError(
                "Plan-Execute planning phase produced no steps. \
                 Check that the inference function returns a numbered list."
                    .to_string(),
            ));
        }

        info!(
            agent = %self.agent_id,
            steps = plan.step_count(),
            "plan-execute: plan created"
        );

        // ── Phase 2: Execute ──────────────────────────────────────────────────
        let max = self.config.max_steps.min(plan.steps.len());
        let mut halted = false;

        for i in 0..plan.steps.len() {
            if halted || i >= max {
                plan.steps[i].status = StepStatus::Skipped;
                continue;
            }

            plan.steps[i].status = StepStatus::Running;
            let step_desc = plan.steps[i].description.clone();
            let tool_name = plan.steps[i].tool.clone();
            let expected = plan.steps[i].expected_outcome.clone();

            debug!(
                agent = %self.agent_id,
                step = i + 1,
                tool = ?tool_name,
                "plan-execute: executing step"
            );

            let output: Result<String, String> = if let Some(ref name) = tool_name {
                // Dispatch to a registered tool.
                match tools.iter().find(|t| t.name() == name) {
                    Some(tool_spec) => {
                        let result = tool_spec
                            .call(serde_json::Value::Object(serde_json::Map::new()))
                            .await;
                        Ok(result.to_string())
                    }
                    None => {
                        // Tool not found — fall back to inference.
                        debug!(
                            tool = name,
                            "plan-execute: tool not found, using inference fallback"
                        );
                        let prompt = Self::step_prompt(&plan, i, &plan.steps[i]);
                        Ok(infer(prompt).await)
                    }
                }
            } else {
                // No tool — call inference directly.
                let prompt = Self::step_prompt(&plan, i, &plan.steps[i]);
                Ok(infer(prompt).await)
            };

            match output {
                Ok(out) => {
                    debug!(
                        agent = %self.agent_id,
                        step = i + 1,
                        output = %out,
                        "plan-execute: step completed"
                    );
                    plan.steps[i].status = StepStatus::Completed(out);
                }
                Err(e) => {
                    warn!(
                        agent = %self.agent_id,
                        step = i + 1,
                        error = %e,
                        "plan-execute: step failed"
                    );
                    plan.steps[i].status = StepStatus::Failed(e);
                    if self.config.stop_on_failure {
                        halted = true;
                    }
                }
            }

            let _ = (step_desc, expected); // suppress unused-variable warnings
        }

        // ── Phase 3: Verify ───────────────────────────────────────────────────
        debug!(agent = %self.agent_id, "plan-execute: running verifier");
        let verification = Verifier::verify(&plan, |prompt| infer(prompt)).await;

        let elapsed_ms = start.elapsed().as_millis();
        info!(
            agent = %self.agent_id,
            elapsed_ms,
            completed = plan.completed_count(),
            failed = plan.failed_count(),
            achieved = verification.achieved,
            confidence = verification.confidence,
            "plan-execute: loop finished"
        );

        Ok((plan, verification))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_agent() -> PlanExecuteAgent {
        PlanExecuteAgent::new(AgentId::new("test"), PlanExecuteConfig::default())
    }

    #[test]
    fn test_parse_plan_three_steps() {
        let response = "1. Search the web | tool:web_search | expected:results found\n\
                        2. Analyse results | tool:none | expected:analysis done\n\
                        3. Write report    | tool:none | expected:report written\n";
        let plan = PlanExecuteAgent::parse_plan("test goal", response);
        assert_eq!(plan.step_count(), 3);
        assert_eq!(plan.steps[0].tool, Some("web_search".to_string()));
        assert!(plan.steps[1].tool.is_none());
        assert_eq!(plan.steps[2].description, "Write report");
        assert_eq!(plan.steps[2].expected_outcome, "report written");
    }

    #[test]
    fn test_parse_plan_empty_response() {
        let plan = PlanExecuteAgent::parse_plan("goal", "No numbered items here.");
        assert_eq!(plan.step_count(), 0);
    }

    #[test]
    fn test_parse_plan_goal_propagated() {
        let plan = PlanExecuteAgent::parse_plan("my goal", "1. Do thing | tool:none | expected:done\n");
        assert_eq!(plan.goal, "my goal");
    }

    #[test]
    fn test_plan_step_status_completed() {
        let mut step = PlanStep::new("desc", None, "done");
        assert_eq!(step.status, StepStatus::Pending);
        step.status = StepStatus::Completed("output".to_string());
        assert!(step.status.is_completed());
        assert_eq!(step.status.output(), Some("output"));
    }

    #[test]
    fn test_plan_step_status_failed() {
        let mut step = PlanStep::new("desc", None, "done");
        step.status = StepStatus::Failed("oops".to_string());
        assert!(step.status.is_failed());
        assert!(step.status.output().is_none());
    }

    #[test]
    fn test_plan_all_succeeded_and_outputs() {
        let steps = vec![
            {
                let mut s = PlanStep::new("s1", None, "done");
                s.status = StepStatus::Completed("r1".to_string());
                s
            },
            {
                let mut s = PlanStep::new("s2", None, "done");
                s.status = StepStatus::Completed("r2".to_string());
                s
            },
        ];
        let plan = Plan::new("goal", steps);
        assert!(plan.all_succeeded());
        assert_eq!(plan.outputs(), vec!["r1", "r2"]);
    }

    #[test]
    fn test_plan_failed_count() {
        let steps = vec![
            {
                let mut s = PlanStep::new("s1", None, "done");
                s.status = StepStatus::Failed("err".to_string());
                s
            },
            {
                let mut s = PlanStep::new("s2", None, "done");
                s.status = StepStatus::Completed("ok".to_string());
                s
            },
        ];
        let plan = Plan::new("goal", steps);
        assert!(!plan.all_succeeded());
        assert_eq!(plan.failed_count(), 1);
        assert_eq!(plan.completed_count(), 1);
    }

    #[test]
    fn test_verifier_parse_achieved() {
        let raw = "ACHIEVED 0.95\nAll steps completed.";
        let result = Verifier::parse_response(raw);
        assert!(result.achieved);
        assert!((result.confidence - 0.95).abs() < 1e-6);
        assert_eq!(result.issues.len(), 1);
        assert_eq!(result.issues[0], "All steps completed.");
    }

    #[test]
    fn test_verifier_parse_not_achieved() {
        let raw = "NOT_ACHIEVED 0.30\nStep 2 failed.\nData was incomplete.";
        let result = Verifier::parse_response(raw);
        assert!(!result.achieved);
        assert!((result.confidence - 0.30).abs() < 1e-6);
        assert_eq!(result.issues.len(), 2);
    }

    #[test]
    fn test_verification_result_confident_success() {
        let r = VerificationResult {
            achieved: true,
            confidence: 0.9,
            issues: vec![],
            raw_response: "ACHIEVED 0.9".to_string(),
        };
        assert!(r.is_confident_success());
    }

    #[test]
    fn test_verification_result_not_confident() {
        let r = VerificationResult {
            achieved: true,
            confidence: 0.3,
            issues: vec![],
            raw_response: "ACHIEVED 0.3".to_string(),
        };
        assert!(!r.is_confident_success());
    }

    #[tokio::test]
    async fn test_run_plan_execute_full_loop() {
        let agent = make_agent();
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();

        let (plan, verification) = agent
            .run(
                "Double the number 21",
                move |ctx: String| {
                    let n = cc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let _ctx = ctx;
                    async move {
                        if n == 0 {
                            // Planning response.
                            "1. Call the double tool | tool:none | expected:42 returned\n"
                                .to_string()
                        } else if n == 1 {
                            // Step 1 execution.
                            "42".to_string()
                        } else {
                            // Verification.
                            "ACHIEVED 0.99\nThe number was doubled correctly.".to_string()
                        }
                    }
                },
                &[],
            )
            .await
            .unwrap();

        assert_eq!(plan.step_count(), 1);
        assert!(plan.all_succeeded());
        assert_eq!(plan.outputs(), vec!["42"]);
        assert!(verification.achieved);
        assert!((verification.confidence - 0.99).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_run_plan_execute_empty_plan_error() {
        let agent = make_agent();
        let result = agent
            .run(
                "some goal",
                |_ctx: String| async move { "no plan here".to_string() },
                &[],
            )
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_execute_config_default() {
        let cfg = PlanExecuteConfig::default();
        assert_eq!(cfg.max_steps, 20);
        assert!(cfg.stop_on_failure);
    }
}
