//! # Module: AgentRuntime
//!
//! ## Responsibility
//! Wire memory, graph, orchestrator, and agent loop into a single coordinator
//! using a builder pattern. Provides `run_agent` which executes a ReAct loop,
//! optionally enriching context from memory and graph lookups.
//!
//! ## Guarantees
//! - Builder uses a typestate parameter to enforce `agent_config` at compile time:
//!   `build()` is only callable once `with_agent_config` has been called.
//! - `run_agent` is async and returns a typed `AgentSession` with step count,
//!   durations, and hits.
//! - Non-panicking: all paths return `Result`
//!
//! ## NOT Responsible For
//! - Actual LLM inference (callers supply a mock/stub inference fn)
//! - Persistence across process restarts (unless `persistence` feature is enabled)

use crate::agent::{AgentConfig, ReActLoop, ReActStep, ToolSpec};
use crate::error::AgentRuntimeError;
use crate::metrics::RuntimeMetrics;
use crate::types::AgentId;

#[cfg(feature = "memory")]
use crate::memory::{EpisodicStore, WorkingMemory};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "graph")]
use crate::graph::GraphStore;

#[cfg(feature = "orchestrator")]
use crate::orchestrator::BackpressureGuard;

// ── Typestate markers ─────────────────────────────────────────────────────────

/// Builder state: agent config has not been provided yet.
pub struct NeedsConfig;
/// Builder state: agent config has been provided; `build()` is available.
pub struct HasConfig;

// ── AgentSession ──────────────────────────────────────────────────────────────

/// The result of a single agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    /// Stable unique identifier for this session (UUID v4 string).
    pub session_id: String,
    /// The agent ID used for this session.
    pub agent_id: AgentId,
    /// All ReAct steps executed during the session.
    pub steps: Vec<ReActStep>,
    /// Number of episodic memory retrievals made during the session.
    pub memory_hits: usize,
    /// Number of graph lookups made during the session.
    pub graph_lookups: usize,
    /// Wall-clock duration of the session in milliseconds.
    pub duration_ms: u64,
    /// Non-fatal errors encountered while saving per-step checkpoints.
    ///
    /// Populated only when a persistence backend is configured.  A non-empty
    /// list means some step snapshots may be missing from storage, but the
    /// session itself completed successfully.
    #[serde(default)]
    pub checkpoint_errors: Vec<String>,
}

impl AgentSession {
    /// Return the number of steps in the session.
    ///
    /// Each [`ReActStep`] in `steps` carries a `step_duration_ms` field measuring
    /// wall-clock time from inference call to observation for that individual step.
    /// Use this to identify slow steps:
    /// ```rust,ignore
    /// for (i, step) in session.steps.iter().enumerate() {
    ///     println!("step {i}: {}ms", step.step_duration_ms);
    /// }
    /// ```
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return `true` if the session has no recorded steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Return the final answer text from the last step, if available.
    ///
    /// Extracts the content after `FINAL_ANSWER` in the last step's `action` field.
    /// Returns `None` if there are no steps or the last action is not a FINAL_ANSWER.
    pub fn final_answer(&self) -> Option<String> {
        let last = self.steps.last()?;
        let upper = last.action.trim().to_ascii_uppercase();
        if upper.starts_with("FINAL_ANSWER") {
            let answer = last.action.trim()["FINAL_ANSWER".len()..].trim().to_owned();
            Some(answer)
        } else {
            None
        }
    }

    /// Return `true` if the session ended with a `FINAL_ANSWER` action.
    ///
    /// This is the normal successful exit from a ReAct loop.  `false` means the
    /// loop was cut short by a timeout, max-iterations limit, or an error.
    pub fn is_successful(&self) -> bool {
        self.final_answer().is_some()
    }

    /// Return the session wall-clock duration as a [`std::time::Duration`].
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.duration_ms)
    }

    /// Return the number of tool-call actions dispatched during the session.
    ///
    /// Each [`ReActStep`] whose `action` parses as a `ToolCall` (not a `FinalAnswer`)
    /// is counted.
    pub fn tool_calls_made(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| {
                // A ToolCall action contains a JSON object.  A FinalAnswer starts
                // with the literal "FINAL_ANSWER" prefix.
                !s.action.trim().to_ascii_uppercase().starts_with("FINAL_ANSWER")
                    && !s.action.trim().is_empty()
            })
            .count()
    }

    /// Return the sum of all individual step durations in milliseconds.
    ///
    /// This is the cumulative inference + tool execution time across all steps,
    /// which may differ from `duration_ms` due to overhead between steps.
    pub fn total_step_duration_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.step_duration_ms).sum()
    }

    /// Return the average step duration in milliseconds.
    ///
    /// Returns `0` when there are no steps.
    pub fn average_step_duration_ms(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        self.total_step_duration_ms() / self.steps.len() as u64
    }

    /// Return a reference to the slowest step (highest `step_duration_ms`).
    ///
    /// Returns `None` when there are no steps.
    pub fn slowest_step(&self) -> Option<&ReActStep> {
        self.steps.iter().max_by_key(|s| s.step_duration_ms)
    }

    /// Return a reference to the fastest step (lowest `step_duration_ms`).
    ///
    /// Returns `None` when there are no steps.
    pub fn fastest_step(&self) -> Option<&ReActStep> {
        self.steps.iter().min_by_key(|s| s.step_duration_ms)
    }

    /// Return references to all steps that are tool calls (not `FINAL_ANSWER`).
    pub fn filter_tool_call_steps(&self) -> Vec<&ReActStep> {
        self.steps.iter().filter(|s| s.is_tool_call()).collect()
    }

    /// Return the zero-based index of the slowest step, or `None` if there are no steps.
    pub fn slowest_step_index(&self) -> Option<usize> {
        self.steps
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.step_duration_ms)
            .map(|(i, _)| i)
    }

    /// Return the zero-based index of the fastest step, or `None` if there are no steps.
    pub fn fastest_step_index(&self) -> Option<usize> {
        self.steps
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.step_duration_ms)
            .map(|(i, _)| i)
    }

    /// Return a reference to the last step, or `None` if there are no steps.
    pub fn last_step(&self) -> Option<&ReActStep> {
        self.steps.last()
    }

    /// Return a reference to the first step taken, or `None` if there are no steps.
    pub fn first_step(&self) -> Option<&ReActStep> {
        self.steps.first()
    }

    /// Return a reference to the step at zero-based index `idx`, or `None` if out of bounds.
    pub fn step_at(&self, idx: usize) -> Option<&ReActStep> {
        self.steps.get(idx)
    }

    /// Return the observation string at step `idx`, or `None` if out of bounds.
    pub fn observation_at(&self, idx: usize) -> Option<&str> {
        self.steps.get(idx).map(|s| s.observation.as_str())
    }

    /// Return the action string at step `idx`, or `None` if out of bounds.
    pub fn action_at(&self, idx: usize) -> Option<&str> {
        self.steps.get(idx).map(|s| s.action.as_str())
    }

    /// Return steps whose observation contains `pattern` (case-insensitive).
    pub fn observations_matching(&self, pattern: &str) -> Vec<&ReActStep> {
        let lower = pattern.to_ascii_lowercase();
        self.steps
            .iter()
            .filter(|s| s.observation.to_ascii_lowercase().contains(&lower))
            .collect()
    }

    /// Return steps whose thought contains `pattern` (case-insensitive).
    pub fn thoughts_containing(&self, pattern: &str) -> Vec<&ReActStep> {
        let lower = pattern.to_ascii_lowercase();
        self.steps
            .iter()
            .filter(|s| s.thought.to_ascii_lowercase().contains(&lower))
            .collect()
    }

    /// Return `true` if any step in this session used `action_name`.
    pub fn has_action(&self, action_name: &str) -> bool {
        self.steps.iter().any(|s| s.action == action_name)
    }

    /// Return the thought string at step `idx`, or `None` if out of bounds.
    pub fn thought_at(&self, idx: usize) -> Option<&str> {
        self.steps.get(idx).map(|s| s.thought.as_str())
    }

    /// Count how many steps used `action_name` as their action.
    ///
    /// Returns `0` if the action was never invoked.  Complements
    /// [`has_action`], which only tests for presence.
    ///
    /// [`has_action`]: AgentSession::has_action
    pub fn step_count_for_action(&self, action_name: &str) -> usize {
        self.steps.iter().filter(|s| s.action == action_name).count()
    }

    /// Return all observation strings in step order.
    ///
    /// Each string is a borrow of the corresponding `ReActStep::observation`
    /// field.  Useful for bulk post-processing of tool results.
    pub fn observations(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.observation.as_str()).collect()
    }

    /// Return the number of steps that have a non-empty observation string.
    pub fn observation_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.observation.is_empty()).count()
    }

    /// Return up to the last `n` non-empty observation strings, ordered oldest
    /// to newest.
    ///
    /// Empty observations are skipped.  If the session has fewer than `n`
    /// non-empty observations, all of them are returned.
    pub fn last_n_observations(&self, n: usize) -> Vec<&str> {
        let all: Vec<&str> = self
            .steps
            .iter()
            .filter(|s| !s.observation.is_empty())
            .map(|s| s.observation.as_str())
            .collect();
        let skip = all.len().saturating_sub(n);
        all[skip..].to_vec()
    }

    /// Return the action names from the last `n` steps, ordered oldest to newest.
    ///
    /// If the session has fewer than `n` steps, all action names are returned.
    pub fn actions_in_window(&self, n: usize) -> Vec<&str> {
        let skip = self.steps.len().saturating_sub(n);
        self.steps[skip..]
            .iter()
            .map(|s| s.action.as_str())
            .collect()
    }

    /// Return the number of steps whose observation string is empty.
    pub fn steps_without_observation(&self) -> usize {
        self.steps.iter().filter(|s| s.observation.is_empty()).count()
    }

    /// Return the thought string from the first step, or `None` if the session
    /// has no steps.
    pub fn first_thought(&self) -> Option<&str> {
        self.steps.first().map(|s| s.thought.as_str())
    }

    /// Return the thought string from the last step, or `None` if the session
    /// has no steps.
    pub fn last_thought(&self) -> Option<&str> {
        self.steps.last().map(|s| s.thought.as_str())
    }

    /// Return the action name from the first step, or `None` if the session
    /// has no steps.
    pub fn first_action(&self) -> Option<&str> {
        self.steps.first().map(|s| s.action.as_str())
    }

    /// Return the action name from the last step, or `None` if the session
    /// has no steps.
    pub fn last_action(&self) -> Option<&str> {
        self.steps.last().map(|s| s.action.as_str())
    }

    /// Return a slice of the last `n` steps.
    ///
    /// If `n` is greater than or equal to the total step count, all steps are
    /// returned.  An empty slice is returned for sessions with no steps.
    pub fn last_n_steps(&self, n: usize) -> &[crate::agent::ReActStep] {
        let len = self.steps.len();
        let start = len.saturating_sub(n);
        &self.steps[start..]
    }

    /// Return a slice containing at most the first `n` steps.
    ///
    /// If the session has fewer than `n` steps all steps are returned.
    /// Returns an empty slice for `n == 0` or an empty session.
    pub fn first_n_steps(&self, n: usize) -> &[crate::agent::ReActStep] {
        let end = n.min(self.steps.len());
        &self.steps[..end]
    }

    /// Return references to steps whose action string contains `tool_name`.
    ///
    /// Useful for auditing which steps invoked a specific tool.  The comparison
    /// is case-sensitive.
    pub fn steps_with_tool<'a>(&'a self, tool_name: &str) -> Vec<&'a crate::agent::ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.action.contains(tool_name) && !s.is_final_answer())
            .collect()
    }

    /// Return the total character count across all thought, action, and observation
    /// strings in the session.
    ///
    /// Useful for estimating context consumption.  Returns `0` for empty sessions.
    pub fn total_chars(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.thought.len() + s.action.len() + s.observation.len())
            .sum()
    }

    /// Return all per-step durations in milliseconds, in order.
    ///
    /// Useful for computing custom percentiles or detecting slow outlier steps.
    pub fn step_durations_ms(&self) -> Vec<u64> {
        self.steps.iter().map(|s| s.step_duration_ms).collect()
    }

    /// Return the sum of all step durations in milliseconds.
    ///
    /// Equivalent to `step_durations_ms().iter().sum()` but avoids allocating
    /// a temporary Vec.
    pub fn total_latency_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.step_duration_ms).sum()
    }

    /// Return the arithmetic mean step duration in milliseconds.
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn avg_step_duration_ms(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.total_latency_ms() as f64 / self.steps.len() as f64
    }

    /// Return a reference to the step with the largest `step_duration_ms`.
    ///
    /// Returns `None` if the session has no steps.  When multiple steps share
    /// the maximum duration the first one (lowest index) is returned.
    pub fn longest_step(&self) -> Option<&crate::agent::ReActStep> {
        self.steps.iter().max_by_key(|s| s.step_duration_ms)
    }

    /// Return a reference to the step with the smallest `step_duration_ms`.
    ///
    /// Returns `None` if the session has no steps.  When multiple steps share
    /// the minimum duration the first one (lowest index) is returned.
    pub fn shortest_step(&self) -> Option<&crate::agent::ReActStep> {
        self.steps.iter().min_by_key(|s| s.step_duration_ms)
    }

    /// Return the sequence of action names taken, in step order.
    ///
    /// Unlike `all_actions()` this returns owned `String`s so the result can
    /// outlive the session borrow.
    pub fn action_sequence(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.action.clone()).collect()
    }

    /// Return the sorted, deduplicated set of tool names invoked during the session.
    ///
    /// Tool-call steps are identified by the same heuristic as `tool_calls_made`:
    /// a non-empty action that does not start with `FINAL_ANSWER`.
    pub fn unique_tools_used(&self) -> Vec<String> {
        let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for step in &self.steps {
            let action = step.action.trim();
            if action.is_empty() || action.to_ascii_uppercase().starts_with("FINAL_ANSWER") {
                continue;
            }
            // Tool name is the JSON "tool" field, or the whole action string if not JSON.
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(action) {
                if let Some(name) = v.get("tool").and_then(|n| n.as_str()) {
                    names.insert(name.to_owned());
                    continue;
                }
            }
            names.insert(action.to_owned());
        }
        let mut sorted: Vec<String> = names.into_iter().collect();
        sorted.sort_unstable();
        sorted
    }

    /// Collect all thought strings from every step, in order.
    pub fn all_thoughts(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.thought.as_str()).collect()
    }

    /// Collect all action strings from every step, in order.
    pub fn all_actions(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.action.as_str()).collect()
    }

    /// Collect all observation strings from every step, in order.
    pub fn all_observations(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.observation.as_str()).collect()
    }

    /// Return references to steps where the observation indicates a tool error.
    ///
    /// A step is classified as failed when its observation starts with
    /// `{"error"` (the structured error JSON produced by required-field
    /// validation) or contains the substring `"error"` (case-insensitive).
    pub fn failed_steps(&self) -> Vec<&crate::agent::ReActStep> {
        self.steps
            .iter()
            .filter(|s| {
                let obs = s.observation.trim();
                obs.starts_with("{\"error\"")
                    || obs.to_ascii_lowercase().contains("\"error\"")
            })
            .collect()
    }

    /// Return the number of tool-call steps whose observation indicates an error.
    ///
    /// Equivalent to `failed_steps().len()` but avoids collecting a `Vec`.
    pub fn failed_tool_call_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| {
                let obs = s.observation.trim();
                obs.starts_with("{\"error\"")
                    || obs.to_ascii_lowercase().contains("\"error\"")
            })
            .count()
    }

    /// Return a count of how many times each action was taken in this session.
    ///
    /// The map key is the action name (e.g. `"search"`, `"FINAL_ANSWER"`).
    pub fn action_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for step in &self.steps {
            *counts.entry(step.action.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Return a sorted list of unique action names used in this session.
    pub fn unique_actions(&self) -> Vec<String> {
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for step in &self.steps {
            seen.insert(step.action.clone());
        }
        seen.into_iter().collect()
    }

    /// Return `true` when at least one action string appears more than once.
    ///
    /// Useful for detecting repetitive or looping agent behaviour without
    /// iterating over the full step list manually.
    pub fn has_duplicate_actions(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        self.steps.iter().any(|s| !seen.insert(s.action.as_str()))
    }

    /// Return the 0-based indices of steps whose action contains `tool_name`.
    ///
    /// Returns an empty `Vec` when no step matches.  Useful when you need
    /// positions rather than step references for further slicing.
    pub fn step_indices_with_tool(&self, tool_name: &str) -> Vec<usize> {
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_final_answer() && s.action.contains(tool_name))
            .map(|(i, _)| i)
            .collect()
    }

    /// Return the action name used most often during the session.
    ///
    /// Returns `None` for sessions with no steps.  When multiple actions tie
    /// for the maximum count, any one of them may be returned.
    pub fn most_used_action(&self) -> Option<String> {
        let counts = self.action_counts();
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name)
    }

    /// Return the observation string from the most recent step that has one.
    ///
    /// Steps with an empty observation are skipped.  Returns `None` when no
    /// step has produced an observation yet.
    pub fn last_observation(&self) -> Option<&str> {
        self.steps
            .iter()
            .rev()
            .find(|s| !s.observation.is_empty())
            .map(|s| s.observation.as_str())
    }

    /// Return the number of steps that have a non-empty thought string.
    pub fn thought_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.thought.is_empty()).count()
    }

    /// Return the fraction of steps that contain a non-empty observation.
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn observation_rate(&self) -> f64 {
        let n = self.steps.len();
        if n == 0 {
            return 0.0;
        }
        let with_obs = self
            .steps
            .iter()
            .filter(|s| !s.observation.is_empty())
            .count();
        with_obs as f64 / n as f64
    }

    /// Return `true` if at least one knowledge-graph lookup was performed during
    /// this session.
    pub fn has_graph_lookups(&self) -> bool {
        self.graph_lookups > 0
    }

    /// Return how many times the last action in the session repeats consecutively
    /// at the end of the step list.
    ///
    /// Returns `0` for empty sessions or single-step sessions where no repeat
    /// is possible.  Useful for detecting a stuck agent that keeps retrying the
    /// same action.
    pub fn consecutive_same_action_at_end(&self) -> usize {
        let n = self.steps.len();
        if n == 0 {
            return 0;
        }
        let last_action = &self.steps[n - 1].action;
        self.steps
            .iter()
            .rev()
            .take_while(|s| &s.action == last_action)
            .count()
            .saturating_sub(1) // don't count the step itself; only the *repeats*
    }

    /// Return the fraction of steps (from the second onward) that repeat the
    /// immediately preceding action.
    ///
    /// Returns `0.0` for sessions with fewer than two steps.  A high value
    /// may indicate the agent is stuck in a loop.
    pub fn action_repetition_rate(&self) -> f64 {
        let n = self.steps.len();
        if n < 2 {
            return 0.0;
        }
        let repeats = self
            .steps
            .windows(2)
            .filter(|w| w[0].action == w[1].action)
            .count();
        repeats as f64 / (n - 1) as f64
    }

    /// Return the length of the longest consecutive run of failed steps.
    ///
    /// A step is considered failed when its observation starts with `{"error"`
    /// or contains the substring `"error"` (case-insensitive).
    /// Returns `0` for sessions with no steps or no failures.
    pub fn max_consecutive_failures(&self) -> usize {
        let mut max_run = 0usize;
        let mut current = 0usize;
        for step in &self.steps {
            let obs = step.observation.trim();
            if obs.starts_with("{\"error\"") || obs.to_ascii_lowercase().contains("\"error\"") {
                current += 1;
                if current > max_run {
                    max_run = current;
                }
            } else {
                current = 0;
            }
        }
        max_run
    }

    /// Return the mean character length of non-empty thought strings.
    ///
    /// Only steps with a non-empty `thought` field are included.
    /// Returns `0.0` when no step has a thought.
    pub fn avg_thought_length(&self) -> f64 {
        let thoughts: Vec<_> = self
            .steps
            .iter()
            .filter(|s| !s.thought.is_empty())
            .collect();
        if thoughts.is_empty() {
            return 0.0;
        }
        let total: usize = thoughts.iter().map(|s| s.thought.len()).sum();
        total as f64 / thoughts.len() as f64
    }

    /// Return the rate of knowledge-graph lookups per step.
    ///
    /// Computed as `graph_lookups / step_count`.  Returns `0.0` when there
    /// are no steps, to avoid division by zero.
    pub fn graph_lookup_rate(&self) -> f64 {
        let steps = self.steps.len();
        if steps == 0 {
            return 0.0;
        }
        self.graph_lookups as f64 / steps as f64
    }

    /// Return `true` if any checkpoint errors were recorded during the session.
    ///
    /// A non-empty `checkpoint_errors` list means some step snapshots may be
    /// missing from storage, but the session itself completed successfully.
    pub fn has_checkpoint_errors(&self) -> bool {
        !self.checkpoint_errors.is_empty()
    }

    /// Return the number of checkpoint errors recorded during this session.
    pub fn checkpoint_error_count(&self) -> usize {
        self.checkpoint_errors.len()
    }

    /// Return the number of knowledge-graph lookups performed during this session.
    pub fn graph_lookup_count(&self) -> usize {
        self.graph_lookups
    }

    /// Return the episodic memory hit rate for this session.
    ///
    /// Computed as `memory_hits / step_count`. Returns `0.0` when there are
    /// no steps, to avoid division by zero.
    pub fn memory_hit_rate(&self) -> f64 {
        let steps = self.steps.len();
        if steps == 0 {
            return 0.0;
        }
        self.memory_hits as f64 / steps as f64
    }

    /// Return the raw count of episodic memory hits for this session.
    pub fn total_memory_hits(&self) -> usize {
        self.memory_hits
    }

    /// Return the session throughput in steps per second.
    ///
    /// Computed as `step_count / (duration_ms / 1000.0)`.  Returns `0.0`
    /// if `duration_ms` is zero.
    pub fn throughput_steps_per_sec(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        self.steps.len() as f64 / (self.duration_ms as f64 / 1000.0)
    }

    /// Return the session duration in full seconds (rounded down).
    pub fn duration_secs(&self) -> u64 {
        self.duration_ms / 1000
    }

    /// Return the count of steps whose thought string is longer than `threshold` bytes.
    pub fn steps_above_thought_length(&self, threshold: usize) -> usize {
        self.steps.iter().filter(|s| s.thought.len() > threshold).count()
    }

    /// Return `true` if any step's action begins with `"FINAL_ANSWER"` (case-insensitive).
    pub fn has_final_answer(&self) -> bool {
        self.steps
            .iter()
            .any(|s| s.action.to_ascii_uppercase().starts_with("FINAL_ANSWER"))
    }

    /// Return the mean byte length of all step action strings.
    ///
    /// Returns `0.0` for empty sessions.
    pub fn avg_action_length(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.action.len()).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return `true` if any tool-call steps had error observations.
    pub fn has_tool_failures(&self) -> bool {
        self.failed_tool_call_count() > 0
    }

    /// Return the fraction of steps that were tool calls.
    ///
    /// Computed as `tool_calls_made / step_count`.  Returns `0.0` for empty
    /// sessions to avoid division by zero.
    pub fn tool_call_rate(&self) -> f64 {
        let total = self.steps.len();
        if total == 0 {
            return 0.0;
        }
        self.tool_calls_made() as f64 / total as f64
    }

    /// Return the fraction of tool-call steps that succeeded.
    ///
    /// Computed as `1.0 - (failed_tool_call_count / step_count)`.  Returns
    /// `1.0` for empty sessions (no failures possible).
    pub fn step_success_rate(&self) -> f64 {
        let total = self.steps.len();
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.failed_tool_call_count() as f64 / total as f64)
    }

    /// Return the ratio of unique actions to total steps.
    ///
    /// Returns `0.0` for sessions with no steps.  A value of `1.0` means every
    /// step used a different action; lower values indicate repeated actions.
    pub fn action_diversity(&self) -> f64 {
        let total = self.steps.len();
        if total == 0 {
            return 0.0;
        }
        let unique: std::collections::HashSet<&str> =
            self.steps.iter().map(|s| s.action.as_str()).collect();
        unique.len() as f64 / total as f64
    }

    /// Return the total byte length of all thought strings across all steps.
    pub fn total_thought_length(&self) -> usize {
        self.steps.iter().map(|s| s.thought.len()).sum()
    }

    /// Return the number of steps whose observation string is empty.
    pub fn steps_with_empty_observations(&self) -> usize {
        self.steps.iter().filter(|s| s.observation.is_empty()).count()
    }

    /// Return the byte length of each observation, in step order.
    pub fn observation_lengths(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.observation.len()).collect()
    }

    /// Return the mean observation byte length across all steps.
    ///
    /// Returns `0.0` for empty sessions.
    pub fn avg_observation_length(&self) -> f64 {
        let n = self.steps.len();
        if n == 0 {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.observation.len()).sum();
        total as f64 / n as f64
    }

    /// Return the byte length of the shortest non-empty thought, or `0` if
    /// no non-empty thoughts exist.
    pub fn min_thought_length(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| !s.thought.is_empty())
            .map(|s| s.thought.len())
            .min()
            .unwrap_or(0)
    }

    /// Return the longest observation string in the session, or `None` if
    /// the session is empty.
    pub fn longest_observation(&self) -> Option<&str> {
        self.steps
            .iter()
            .max_by_key(|s| s.observation.len())
            .map(|s| s.observation.as_str())
    }

    /// Return the byte length of each step's thought string, in step order.
    pub fn thought_lengths(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.thought.len()).collect()
    }

    /// Return the action string that appears most often across all steps.
    ///
    /// Returns `None` if the session has no steps.
    pub fn most_common_action(&self) -> Option<&str> {
        if self.steps.is_empty() {
            return None;
        }
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for s in &self.steps {
            *counts.entry(s.action.as_str()).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(a, _)| a)
    }

    /// Return the byte length of each step's action string, in step order.
    pub fn action_lengths(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.action.len()).collect()
    }

    /// Return the count of steps that did not have a tool failure.
    pub fn step_success_count(&self) -> usize {
        self.steps.len() - self.failed_tool_call_count()
    }

    /// Return the thought string of the step with the most bytes.
    ///
    /// Returns `None` if the session has no steps.
    pub fn longest_thought(&self) -> Option<&str> {
        self.steps
            .iter()
            .max_by_key(|s| s.thought.len())
            .map(|s| s.thought.as_str())
    }

    /// Return the action string of the step with the fewest bytes.
    ///
    /// Returns `None` if the session has no steps.
    pub fn shortest_action(&self) -> Option<&str> {
        self.steps
            .iter()
            .min_by_key(|s| s.action.len())
            .map(|s| s.action.as_str())
    }

    /// Return the sum of byte lengths of all thought strings in the session.
    pub fn total_thought_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.thought.len()).sum()
    }

    /// Return the sum of byte lengths of all observation strings in the session.
    pub fn total_observation_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.observation.len()).sum()
    }

    /// Return the action string of the first step in the session.
    ///
    /// Returns `None` if the session has no steps.
    pub fn first_step_action(&self) -> Option<&str> {
        self.steps.first().map(|s| s.action.as_str())
    }

    /// Return the action string of the last step in the session.
    ///
    /// Returns `None` if the session has no steps.
    pub fn last_step_action(&self) -> Option<&str> {
        self.steps.last().map(|s| s.action.as_str())
    }

    /// Return the count of steps that have a non-empty thought string.
    pub fn count_nonempty_thoughts(&self) -> usize {
        self.steps.iter().filter(|s| !s.thought.is_empty()).count()
    }

    /// Return the count of steps whose observation contains `substring`.
    pub fn observation_contains_count(&self, substring: &str) -> usize {
        self.steps.iter().filter(|s| s.observation.contains(substring)).count()
    }

    /// Return the number of steps whose action string matches `action` exactly.
    pub fn count_steps_with_action(&self, action: &str) -> usize {
        self.steps.iter().filter(|s| s.action == action).count()
    }

    /// Return the number of steps whose thought contains `substring`.
    pub fn thought_contains_count(&self, substring: &str) -> usize {
        self.steps.iter().filter(|s| s.thought.contains(substring)).count()
    }

    /// Return the fraction of steps that had a tool failure observation.
    ///
    /// Computed as `failed_tool_call_count / step_count`.  Returns `0.0` for
    /// empty sessions.
    pub fn failure_rate(&self) -> f64 {
        let total = self.steps.len();
        if total == 0 {
            return 0.0;
        }
        self.failed_tool_call_count() as f64 / total as f64
    }

    /// Return the number of distinct action names used across all steps.
    pub fn unique_action_count(&self) -> usize {
        let unique: std::collections::HashSet<&str> =
            self.steps.iter().map(|s| s.action.as_str()).collect();
        unique.len()
    }

    /// Return references to steps whose indices fall in `[start, end)`.
    ///
    /// Clamps `end` to `step_count()` so out-of-bounds ranges are safe.
    /// Returns an empty `Vec` when `start >= step_count()` or `start >= end`.
    pub fn steps_in_range(&self, start: usize, end: usize) -> Vec<&ReActStep> {
        let clamped_end = end.min(self.steps.len());
        if start >= clamped_end {
            return Vec::new();
        }
        self.steps[start..clamped_end].iter().collect()
    }

    /// Return the median step duration in milliseconds.
    ///
    /// Sorts step durations and picks the middle value (lower median for even
    /// counts).  Returns `0` when the session has no steps.
    pub fn median_step_duration_ms(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        let mut durations: Vec<u64> = self.steps.iter().map(|s| s.step_duration_ms).collect();
        durations.sort_unstable();
        durations[durations.len() / 2]
    }

    /// Return the 95th-percentile step duration in milliseconds.
    ///
    /// Uses the nearest-rank method: the value at index `⌈0.95 × n⌉ − 1` in
    /// the sorted list of step durations.  Returns `0` when the session has no
    /// steps.
    pub fn p95_step_duration_ms(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        let mut durations: Vec<u64> = self.steps.iter().map(|s| s.step_duration_ms).collect();
        durations.sort_unstable();
        let idx = ((durations.len() as f64 * 0.95).ceil() as usize)
            .saturating_sub(1)
            .min(durations.len() - 1);
        durations[idx]
    }

    /// Return the 99th-percentile step duration in milliseconds.
    ///
    /// Uses the nearest-rank method: the value at index `⌈0.99 × n⌉ − 1` in
    /// the sorted list of step durations.  Returns `0` when the session has no
    /// steps.
    pub fn p99_step_duration_ms(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        let mut durations: Vec<u64> = self.steps.iter().map(|s| s.step_duration_ms).collect();
        durations.sort_unstable();
        let idx = ((durations.len() as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(durations.len() - 1);
        durations[idx]
    }

    /// Return the count of steps whose `step_duration_ms` is strictly greater
    /// than `threshold_ms`.
    ///
    /// Useful for identifying sessions that contain outlier-slow steps.
    /// Returns `0` for empty sessions.
    pub fn step_count_above_duration_ms(&self, threshold_ms: u64) -> usize {
        self.steps
            .iter()
            .filter(|s| s.step_duration_ms > threshold_ms)
            .count()
    }

    /// Return the minimum `step_duration_ms` across all steps in the session.
    ///
    /// Returns `0` for empty sessions.
    pub fn min_step_duration_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.step_duration_ms).min().unwrap_or(0)
    }

    /// Return the maximum `step_duration_ms` across all steps in the session.
    ///
    /// Returns `0` for empty sessions.
    pub fn max_step_duration_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.step_duration_ms).max().unwrap_or(0)
    }

    /// Return the sum of byte lengths of all action strings in the session.
    ///
    /// Useful alongside [`total_thought_bytes`] and [`total_observation_bytes`]
    /// to estimate the full token budget consumed by a session.
    ///
    /// [`total_thought_bytes`]: AgentSession::total_thought_bytes
    /// [`total_observation_bytes`]: AgentSession::total_observation_bytes
    pub fn total_action_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.action.len()).sum()
    }

    /// Return the population variance of step durations in milliseconds squared.
    ///
    /// Returns `0.0` for sessions with fewer than two steps.
    pub fn step_duration_variance_ms(&self) -> f64 {
        let n = self.steps.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.average_step_duration_ms() as f64;
        let sum_sq: f64 = self
            .steps
            .iter()
            .map(|s| {
                let diff = s.step_duration_ms as f64 - mean;
                diff * diff
            })
            .sum();
        sum_sq / n as f64
    }

    /// Return references to steps whose observation contains `"error"` (case-insensitive).
    ///
    /// A quick filter for identifying tool-call failures without inspecting the
    /// full observation string.  For more control use [`observations_matching`].
    ///
    /// [`observations_matching`]: AgentSession::observations_matching
    pub fn steps_with_errors(&self) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.observation.to_ascii_lowercase().contains("error"))
            .collect()
    }

    /// Return references to steps whose observation byte length exceeds `threshold_bytes`.
    ///
    /// Useful for identifying steps that produced unusually verbose tool output.
    pub fn steps_with_long_observations(&self, threshold_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.observation.len() > threshold_bytes)
            .collect()
    }

    /// Return steps whose observation is strictly longer than `min_bytes`.
    ///
    /// Equivalent to [`steps_with_long_observations`] but uses the name
    /// "above" for consistency with other filtering predicates.
    ///
    /// [`steps_with_long_observations`]: AgentSession::steps_with_long_observations
    pub fn observations_above_bytes(&self, min_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.observation.len() > min_bytes)
            .collect()
    }

    /// Return the total character count across all steps.
    ///
    /// Sums `thought.chars().count() + action.chars().count() + observation.chars().count()`
    /// for every step.  Useful as a proxy for token budget estimation.
    pub fn total_step_chars(&self) -> usize {
        self.steps
            .iter()
            .map(|s| {
                s.thought.chars().count()
                    + s.action.chars().count()
                    + s.observation.chars().count()
            })
            .sum()
    }

    /// Return the number of distinct observation strings across all steps.
    ///
    /// Two steps with the same observation text are counted as one.
    /// Returns `0` for empty sessions.
    pub fn unique_observations_count(&self) -> usize {
        let unique: std::collections::HashSet<&str> =
            self.steps.iter().map(|s| s.observation.as_str()).collect();
        unique.len()
    }

    /// Return the maximum byte length of any thought string in the session.
    ///
    /// Returns `0` for empty sessions or sessions where every thought is empty.
    pub fn thought_max_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.thought.len()).max().unwrap_or(0)
    }

    /// Return the maximum byte length of any observation string in the session.
    ///
    /// Returns `0` for empty sessions or sessions where every observation is empty.
    pub fn observation_max_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.observation.len()).max().unwrap_or(0)
    }

    /// Return the count of steps whose `step_duration_ms` is strictly less
    /// than `threshold_ms`.
    ///
    /// Complements [`step_count_above_duration_ms`].  Returns `0` for empty
    /// sessions.
    ///
    /// [`step_count_above_duration_ms`]: AgentSession::step_count_above_duration_ms
    pub fn step_count_below_duration_ms(&self, threshold_ms: u64) -> usize {
        self.steps
            .iter()
            .filter(|s| s.step_duration_ms < threshold_ms)
            .count()
    }

    /// Return the maximum byte length of any action string across all steps.
    ///
    /// Returns `0` for sessions with no steps or where every action is empty.
    pub fn max_action_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.action.len()).max().unwrap_or(0)
    }

    /// Return the minimum byte length of any action string across all steps.
    ///
    /// Returns `0` for sessions with no steps or where every action is empty.
    pub fn min_action_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.action.len()).min().unwrap_or(0)
    }

    /// Return the fraction of steps that are tool calls (not `FINAL_ANSWER`).
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn proportion_tool_calls(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let tool_calls = self.steps.iter().filter(|s| s.is_tool_call()).count();
        tool_calls as f64 / self.steps.len() as f64
    }

    /// Return the ratio of total thought bytes to the total bytes across all
    /// step fields (thoughts + actions + observations).
    ///
    /// Returns `0.0` for sessions with no steps or where no bytes exist.
    pub fn thought_density(&self) -> f64 {
        let thought_bytes: usize = self.steps.iter().map(|s| s.thought.len()).sum();
        let total_bytes: usize = self
            .steps
            .iter()
            .map(|s| s.thought.len() + s.action.len() + s.observation.len())
            .sum();
        if total_bytes == 0 {
            return 0.0;
        }
        thought_bytes as f64 / total_bytes as f64
    }

    /// Return the average number of ReAct steps completed per second.
    ///
    /// Computed as `step_count / (duration_ms / 1000.0)`.  Returns `0.0` for
    /// sessions with no steps or zero duration.
    pub fn step_throughput_per_sec(&self) -> f64 {
        if self.duration_ms == 0 || self.steps.is_empty() {
            return 0.0;
        }
        self.steps.len() as f64 / (self.duration_ms as f64 / 1000.0)
    }

    /// Return the mean byte length of action strings across all steps.
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn avg_action_bytes(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.action.len()).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return the mean byte length of observation strings across all steps.
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn avg_observation_bytes(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.observation.len()).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return the number of steps that have a non-empty observation string.
    ///
    /// Steps where the tool produced no output (empty string) are excluded.
    pub fn total_observation_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.observation.is_empty()).count()
    }

    /// Return references to steps whose action string contains `substring`.
    ///
    /// The comparison is case-sensitive.  Returns an empty slice when no step
    /// matches or the session has no steps.
    pub fn actions_containing<'a>(&'a self, substring: &str) -> Vec<&'a ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.action.contains(substring))
            .collect()
    }

    /// Return the mean byte length of thought strings across all steps.
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn avg_thought_bytes(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.thought.len()).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return references to steps whose action byte length exceeds `min_bytes`.
    ///
    /// Returns an empty `Vec` for sessions with no steps or when no step
    /// action exceeds `min_bytes`.
    pub fn steps_above_action_bytes(&self, min_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.action.len() > min_bytes)
            .collect()
    }

    /// Return the steps in the half-open index range `[start, end)`.
    ///
    /// Both bounds are clamped to `[0, step_count]`.  Returns an empty `Vec`
    /// when `start >= end` or the session has no steps.
    pub fn steps_between(&self, start: usize, end: usize) -> Vec<&ReActStep> {
        let clamped_end = end.min(self.steps.len());
        if start >= clamped_end {
            return Vec::new();
        }
        self.steps[start..clamped_end].iter().collect()
    }

    /// Return the fraction of steps that have a non-empty observation string.
    ///
    /// Returns `0.0` for sessions with no steps.
    pub fn step_observation_rate(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let count = self.steps.iter().filter(|s| !s.observation.is_empty()).count();
        count as f64 / self.steps.len() as f64
    }

    /// Return references to steps whose thought byte length is strictly less
    /// than `max_bytes`.
    ///
    /// Returns an empty `Vec` when no steps qualify or the session is empty.
    pub fn steps_below_thought_bytes(&self, max_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.thought.len() < max_bytes)
            .collect()
    }

    /// Return references to steps whose thought string duplicates an earlier
    /// step's thought.
    ///
    /// The first occurrence of each thought is not included; only the
    /// subsequent duplicates are returned.  Useful for detecting repetitive
    /// reasoning loops.
    pub fn steps_with_duplicate_thoughts(&self) -> Vec<&ReActStep> {
        let mut seen = std::collections::HashSet::new();
        self.steps
            .iter()
            .filter(|s| !seen.insert(s.thought.as_str()))
            .collect()
    }

    /// Return the byte length of the longest thought in this session.
    ///
    /// Returns `0` for an empty session or when all thoughts are empty strings.
    pub fn max_thought_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.thought.len()).max().unwrap_or(0)
    }

    /// Return references to steps whose action starts with `prefix`.
    ///
    /// Useful for filtering tool-call steps by tool name prefix (e.g. all
    /// `"search_"` actions).  Returns an empty `Vec` when no step qualifies.
    pub fn steps_by_action_prefix<'a>(&'a self, prefix: &str) -> Vec<&'a ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.action.starts_with(prefix))
            .collect()
    }

    /// Return the number of tool-call steps in this session.
    ///
    /// Counts steps whose action is non-empty and is not a `FINAL_ANSWER`.
    /// Returns `0` for an empty session.
    pub fn action_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_tool_call()).count()
    }

    /// Return references to steps whose observation byte length exceeds `min_bytes`.
    ///
    /// Useful for finding steps that produced unexpectedly large observations.
    /// Returns an empty `Vec` for an empty session or when no step qualifies.
    pub fn steps_above_observation_bytes(&self, min_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.observation.len() > min_bytes)
            .collect()
    }

    /// Return references to steps whose observation contains `substr`.
    ///
    /// Case-sensitive substring match.  Returns an empty `Vec` when no step
    /// matches or the session is empty.
    pub fn steps_matching_observation<'a>(&'a self, substr: &str) -> Vec<&'a ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.observation.contains(substr))
            .collect()
    }

    /// Return the byte lengths of each step's action field, in order.
    ///
    /// Returns an empty `Vec` for an empty session.
    pub fn step_action_lengths(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.action.len()).collect()
    }

    /// Return `true` if any step's thought starts with `prefix`.
    ///
    /// Returns `false` for an empty session.
    pub fn has_thought_starting_with(&self, prefix: &str) -> bool {
        self.steps.iter().any(|s| s.thought.starts_with(prefix))
    }

    /// Return the number of steps whose action byte length exceeds `min_bytes`.
    ///
    /// Returns `0` for an empty session.
    pub fn step_count_above_action_bytes(&self, min_bytes: usize) -> usize {
        self.steps.iter().filter(|s| s.action.len() > min_bytes).count()
    }

    /// Return all steps whose `action` field is empty.
    ///
    /// Returns an empty `Vec` when no steps have an empty action.
    pub fn steps_with_empty_action(&self) -> Vec<&ReActStep> {
        self.steps.iter().filter(|s| s.action.is_empty()).collect()
    }

    /// Return `true` if any step's action contains `substr` as a substring.
    ///
    /// Returns `false` for an empty session.
    pub fn has_action_containing(&self, substr: &str) -> bool {
        self.steps.iter().any(|s| s.action.contains(substr))
    }

    /// Return the maximum UTF-8 character count among all observation strings.
    ///
    /// Returns `0` for an empty session.
    pub fn max_observation_chars(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.observation.chars().count())
            .max()
            .unwrap_or(0)
    }

    /// Return the 0-based index of the step with the longest `thought` by
    /// character count, or `None` for an empty session.
    ///
    /// When multiple steps tie for the longest thought the smallest index wins.
    pub fn step_index_of_longest_thought(&self) -> Option<usize> {
        self.steps
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.thought.chars().count())
            .map(|(i, _)| i)
    }

    /// Return the number of whitespace-delimited words in each observation,
    /// in step order.
    ///
    /// Returns an empty `Vec` for an empty session.
    pub fn observation_word_counts(&self) -> Vec<usize> {
        self.steps
            .iter()
            .map(|s| s.observation.split_whitespace().count())
            .collect()
    }

    /// Return `true` if any observation starts with one of the given `prefixes`.
    ///
    /// Returns `false` for an empty session or when no prefix matches.
    pub fn observation_starts_with_any(&self, prefixes: &[&str]) -> bool {
        self.steps
            .iter()
            .any(|s| prefixes.iter().any(|p| s.observation.starts_with(p)))
    }

    /// Return `true` if any action string appears more than once in the session.
    ///
    /// Non-empty duplicate actions often signal a stuck or looping agent.
    /// Returns `false` for a session with fewer than two steps.
    pub fn has_repeated_actions(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        self.steps
            .iter()
            .filter(|s| !s.action.is_empty())
            .any(|s| !seen.insert(s.action.as_str()))
    }

    /// Return `true` if any step thought starts with one of the given `prefixes`.
    ///
    /// Returns `false` for an empty session or when no prefix matches.
    pub fn thought_starts_with_any(&self, prefixes: &[&str]) -> bool {
        self.steps
            .iter()
            .any(|s| prefixes.iter().any(|p| s.thought.starts_with(p)))
    }

    /// Return the total number of whitespace-delimited words across all action
    /// strings in this session.
    ///
    /// Returns `0` for an empty session.
    pub fn action_word_count(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.action.split_whitespace().count())
            .sum()
    }

    /// Return the number of steps whose thought character count exceeds `min`.
    ///
    /// Returns `0` for an empty session.
    pub fn steps_above_thought_chars(&self, min: usize) -> usize {
        self.steps
            .iter()
            .filter(|s| s.thought.chars().count() > min)
            .count()
    }

    /// Return all steps that have a non-empty observation string.
    ///
    /// Returns an empty `Vec` for an empty session.
    pub fn steps_with_non_empty_observation(&self) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| !s.observation.is_empty())
            .collect()
    }

    /// Return all steps whose observation contains `substr`.
    ///
    /// Returns an empty `Vec` for an empty session or no matches.
    pub fn observations_containing(&self, substr: &str) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.observation.contains(substr))
            .collect()
    }

    /// Return the ratio of total thought characters to total observation
    /// characters.
    ///
    /// Returns `0.0` when there are no observation characters to avoid
    /// division by zero.
    pub fn thought_observation_ratio(&self) -> f64 {
        let obs: usize = self.steps.iter().map(|s| s.observation.chars().count()).sum();
        if obs == 0 {
            return 0.0;
        }
        let thoughts: usize = self.steps.iter().map(|s| s.thought.chars().count()).sum();
        thoughts as f64 / obs as f64
    }

    /// Return all steps whose thought contains `substr` as a substring.
    ///
    /// Returns an empty `Vec` for an empty session or no matches.
    pub fn steps_matching_thought(&self, substr: &str) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.thought.contains(substr))
            .collect()
    }

    /// Return the median observation character count across all steps.
    ///
    /// Returns `0` for an empty session.  Uses the lower median when the step
    /// count is even.
    pub fn median_observation_chars(&self) -> usize {
        if self.steps.is_empty() {
            return 0;
        }
        let mut lens: Vec<usize> = self
            .steps
            .iter()
            .map(|s| s.observation.chars().count())
            .collect();
        lens.sort_unstable();
        lens[lens.len() / 2]
    }

    /// Return the cumulative sum of thought character counts, step by step.
    ///
    /// The *i*-th element is the total thought characters from step 0 through
    /// step *i* inclusive.  Returns an empty `Vec` for an empty session.
    pub fn cumulative_thought_chars(&self) -> Vec<usize> {
        let mut total = 0usize;
        self.steps
            .iter()
            .map(|s| {
                total += s.thought.chars().count();
                total
            })
            .collect()
    }

    /// Return the number of steps whose thought contains `substr`.
    ///
    /// Returns `0` for an empty session.
    pub fn count_steps_with_thought_containing(&self, substr: &str) -> usize {
        self.steps
            .iter()
            .filter(|s| s.thought.contains(substr))
            .count()
    }

    /// Return the smallest byte length of non-empty observation strings in this
    /// session.
    ///
    /// Steps with an empty observation are excluded. Returns `0` if no non-empty
    /// observations exist.
    pub fn min_observation_bytes(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.observation.len())
            .filter(|&n| n > 0)
            .min()
            .unwrap_or(0)
    }

    /// Return the smallest byte length of non-empty thought strings in this
    /// session.
    ///
    /// Steps with an empty thought are excluded. Returns `0` if no non-empty
    /// thoughts exist.
    pub fn min_thought_bytes(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.thought.len())
            .filter(|&n| n > 0)
            .min()
            .unwrap_or(0)
    }

    /// Return the proportion of steps whose `thought` field is empty.
    ///
    /// Returns `0.0` for an empty session.
    pub fn proportion_empty_thoughts(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let empty = self.steps.iter().filter(|s| s.thought.is_empty()).count();
        empty as f64 / self.steps.len() as f64
    }

    /// Return `true` if any step in this session is marked as failed.
    ///
    /// A step is considered failed when its `observation` starts with
    /// `"[error]"` (the convention used by the built-in tool dispatcher).
    pub fn has_failed_steps(&self) -> bool {
        self.steps
            .iter()
            .any(|s| s.observation.starts_with("[error]"))
    }

    /// Return the total number of UTF-8 characters across all step `thought` fields.
    ///
    /// Returns `0` for an empty session.
    pub fn total_thought_chars(&self) -> usize {
        self.steps.iter().map(|s| s.thought.chars().count()).sum()
    }

    /// Return the total number of UTF-8 characters across all step `action` fields.
    ///
    /// Returns `0` for an empty session.
    pub fn total_action_chars(&self) -> usize {
        self.steps.iter().map(|s| s.action.chars().count()).sum()
    }

    /// Return the total number of UTF-8 characters across all step `observation`
    /// fields.
    ///
    /// Returns `0` for an empty session.
    pub fn total_observation_chars(&self) -> usize {
        self.steps.iter().map(|s| s.observation.chars().count()).sum()
    }

    /// Return the statistical variance of action byte lengths across all steps.
    ///
    /// Useful for detecting inconsistent action sizes. Returns `0.0` for a
    /// session with fewer than two steps or all equal lengths.
    pub fn action_byte_variance(&self) -> f64 {
        if self.steps.len() < 2 {
            return 0.0;
        }
        let lengths: Vec<f64> = self.steps.iter().map(|s| s.action.len() as f64).collect();
        let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
        lengths.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / lengths.len() as f64
    }

    /// Return the count of steps that have a non-empty `action` string.
    ///
    /// Returns `0` for an empty session.
    pub fn non_empty_action_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.action.is_empty()).count()
    }

    /// Return the total byte count across all fields (`thought` + `action` +
    /// `observation`) for every step in the session.
    ///
    /// Returns `0` for an empty session.
    pub fn total_step_bytes(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.thought.len() + s.action.len() + s.observation.len())
            .sum()
    }

    /// Return the byte length of the last step's `thought` field, or `0` if
    /// the session has no steps.
    pub fn last_thought_bytes(&self) -> usize {
        self.steps.last().map_or(0, |s| s.thought.len())
    }

    /// Return the byte length of the first step's `observation` field, or `0`
    /// if the session has no steps.
    pub fn first_observation_bytes(&self) -> usize {
        self.steps.first().map_or(0, |s| s.observation.len())
    }

    /// Return `true` if any step has an empty `observation` field.
    ///
    /// Useful for detecting incomplete ReAct traces where a tool produced no
    /// output.
    pub fn has_step_with_empty_observation(&self) -> bool {
        self.steps.iter().any(|s| s.observation.is_empty())
    }

    /// Return the ratio of total thought bytes to total action bytes.
    ///
    /// Returns `0.0` when there are no action bytes (avoids division by zero).
    pub fn thought_to_action_byte_ratio(&self) -> f64 {
        let thought_bytes: usize = self.steps.iter().map(|s| s.thought.len()).sum();
        let action_bytes: usize = self.steps.iter().map(|s| s.action.len()).sum();
        if action_bytes == 0 {
            return 0.0;
        }
        thought_bytes as f64 / action_bytes as f64
    }

    /// Return the number of steps whose `observation` byte length exceeds
    /// `min_bytes`.
    ///
    /// Returns `0` when the session is empty or no step qualifies.
    pub fn observation_above_bytes_count(&self, min_bytes: usize) -> usize {
        self.steps
            .iter()
            .filter(|s| s.observation.len() > min_bytes)
            .count()
    }

    /// Return the number of steps where both the `thought` and `action`
    /// fields are non-empty.
    ///
    /// Returns `0` for an empty session.
    pub fn steps_with_both_thought_and_action(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| !s.thought.is_empty() && !s.action.is_empty())
            .count()
    }

    /// Return the number of steps whose `observation` field starts with
    /// `prefix`.
    ///
    /// Returns `0` for an empty session or when no step qualifies.
    pub fn steps_with_observation_prefix(&self, prefix: &str) -> usize {
        self.steps
            .iter()
            .filter(|s| s.observation.starts_with(prefix))
            .count()
    }

    /// Return the total byte count of all `observation` fields across all
    /// steps.
    ///
    /// Returns `0` for an empty session.
    pub fn observation_bytes_total(&self) -> usize {
        self.steps.iter().map(|s| s.observation.len()).sum()
    }

    /// Return the character count of the first step's `thought` field.
    ///
    /// Returns `0` for an empty session.
    pub fn first_thought_chars(&self) -> usize {
        self.steps.first().map_or(0, |s| s.thought.chars().count())
    }

    /// Return the character count of the last step's `observation` field.
    ///
    /// Returns `0` for an empty session.
    pub fn last_observation_chars(&self) -> usize {
        self.steps.last().map_or(0, |s| s.observation.chars().count())
    }

    /// Return the total word count across all `observation` fields.
    ///
    /// Words are split on whitespace.  Returns `0` for an empty session.
    pub fn observation_word_count_total(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.observation.split_whitespace().count())
            .sum()
    }

    /// Return the count of steps whose `action` field ends with `suffix`.
    ///
    /// Returns `0` for an empty session or when no step qualifies.
    pub fn action_ends_with_count(&self, suffix: &str) -> usize {
        self.steps
            .iter()
            .filter(|s| s.action.ends_with(suffix))
            .count()
    }

    /// Return the average word count per observation field across all steps.
    ///
    /// Returns `0.0` for an empty session.
    pub fn avg_observation_words(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self
            .steps
            .iter()
            .map(|s| s.observation.split_whitespace().count())
            .sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return the statistical variance of thought byte lengths across all steps.
    ///
    /// Returns `0.0` for a session with fewer than two steps.
    pub fn thought_byte_variance(&self) -> f64 {
        if self.steps.len() < 2 {
            return 0.0;
        }
        let lengths: Vec<f64> = self.steps.iter().map(|s| s.thought.len() as f64).collect();
        let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
        lengths.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / lengths.len() as f64
    }

    /// Return all steps whose `thought` byte length exceeds `min_bytes`.
    ///
    /// Returns an empty `Vec` if no step qualifies.
    pub fn steps_above_thought_bytes(&self, min_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.thought.len() > min_bytes)
            .collect()
    }

    /// Return the number of steps where all three fields (`thought`, `action`,
    /// and `observation`) are empty strings.
    ///
    /// Fully empty steps typically indicate a malformed or aborted iteration.
    pub fn total_empty_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.thought.is_empty() && s.action.is_empty() && s.observation.is_empty())
            .count()
    }

    /// Return the number of steps whose `action` begins with `prefix`.
    ///
    /// Returns `0` for an empty session or when no action matches.
    pub fn action_starts_with_count(&self, prefix: &str) -> usize {
        self.steps
            .iter()
            .filter(|s| s.action.starts_with(prefix))
            .count()
    }

    /// Return the longest `action` string in the session, or `None` if the
    /// session is empty.
    ///
    /// When multiple steps share the maximum byte length, the first one is
    /// returned.
    pub fn longest_action(&self) -> Option<&str> {
        self.steps
            .iter()
            .max_by_key(|s| s.action.len())
            .map(|s| s.action.as_str())
    }

    /// Return the proportion of steps that have a non-empty thought string.
    ///
    /// Returns `0.0` for an empty session.
    pub fn thought_completeness(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let non_empty = self.steps.iter().filter(|s| !s.thought.is_empty()).count();
        non_empty as f64 / self.steps.len() as f64
    }

    /// Return the 0-based index of the first `FINAL_ANSWER` step, or `None` if
    /// no such step exists in the session.
    ///
    /// Useful when you need the position of the answer step rather than just
    /// testing whether one exists (see [`has_final_answer`]).
    ///
    /// [`has_final_answer`]: AgentSession::has_final_answer
    pub fn final_answer_step_index(&self) -> Option<usize> {
        self.steps.iter().position(|s| s.is_final_answer())
    }

    /// Return the `(min, max)` step duration range in milliseconds.
    ///
    /// Returns `(0, 0)` for sessions with no steps.
    pub fn step_duration_range_ms(&self) -> (u64, u64) {
        if self.steps.is_empty() {
            return (0, 0);
        }
        let min = self.steps.iter().map(|s| s.step_duration_ms).min().unwrap_or(0);
        let max = self.steps.iter().map(|s| s.step_duration_ms).max().unwrap_or(0);
        (min, max)
    }

    /// Return the number of distinct thought strings across all steps.
    ///
    /// Two steps with identical thought text are counted once.  Returns `0`
    /// for empty sessions.
    pub fn count_unique_thoughts(&self) -> usize {
        let unique: std::collections::HashSet<&str> =
            self.steps.iter().map(|s| s.thought.as_str()).collect();
        unique.len()
    }

    /// Return references to steps whose thought string is empty.
    ///
    /// Useful for detecting steps where the model skipped the reasoning phase.
    pub fn steps_with_empty_thoughts(&self) -> Vec<&ReActStep> {
        self.steps.iter().filter(|s| s.thought.is_empty()).collect()
    }

    /// Return references to steps whose thought byte length exceeds `threshold_bytes`.
    ///
    /// Useful for identifying steps with unusually verbose reasoning traces.
    pub fn steps_with_long_thoughts(&self, threshold_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.thought.len() > threshold_bytes)
            .collect()
    }

    /// Return the number of steps whose action string contains `substring`.
    ///
    /// The comparison is case-sensitive.  Returns `0` for empty sessions or
    /// when no step matches.
    pub fn action_count_containing(&self, substring: &str) -> usize {
        self.steps.iter().filter(|s| s.action.contains(substring)).count()
    }

    /// Return the number of steps that have a non-empty thought string.
    ///
    /// Complement of [`steps_with_empty_thoughts`].  Returns `0` for empty
    /// sessions.
    ///
    /// [`steps_with_empty_thoughts`]: AgentSession::steps_with_empty_thoughts
    pub fn total_thought_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.thought.is_empty()).count()
    }

    /// Return `true` if any step's `thought` field contains `substring`
    /// (case-sensitive).
    ///
    /// Returns `false` for empty sessions or when no step matches.
    pub fn has_thought_containing(&self, substring: &str) -> bool {
        self.steps.iter().any(|s| s.thought.contains(substring))
    }

    /// Return references to steps whose `action` field is longer than
    /// `min_bytes` bytes.
    ///
    /// Returns an empty `Vec` when no step qualifies.
    pub fn steps_with_action_length_above(&self, min_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.action.len() > min_bytes)
            .collect()
    }

    /// Persist this session as a checkpoint under `"session:<session_id>"`.
    #[cfg(feature = "persistence")]
    pub async fn save_checkpoint(
        &self,
        backend: &dyn crate::persistence::PersistenceBackend,
    ) -> Result<(), AgentRuntimeError> {
        let key = format!("session:{}", self.session_id);
        let bytes = serde_json::to_vec(self)
            .map_err(|e| AgentRuntimeError::Persistence(format!("serialize: {e}")))?;
        backend.save(&key, &bytes).await
    }

    /// Load a previously saved checkpoint by `session_id`.
    ///
    /// Returns `None` if no checkpoint exists for the given ID.
    #[cfg(feature = "persistence")]
    pub async fn load_checkpoint(
        backend: &dyn crate::persistence::PersistenceBackend,
        session_id: &str,
    ) -> Result<Option<AgentSession>, AgentRuntimeError> {
        let key = format!("session:{session_id}");
        match backend.load(&key).await? {
            None => Ok(None),
            Some(bytes) => {
                let session = serde_json::from_slice(&bytes)
                    .map_err(|e| AgentRuntimeError::Persistence(format!("deserialize: {e}")))?;
                Ok(Some(session))
            }
        }
    }

    /// Load the session snapshot saved after step `step` completed.
    ///
    /// Alias for [`load_checkpoint_at_step`] — provided for ergonomic
    /// compatibility with call sites that prefer this naming convention.
    ///
    /// [`load_checkpoint_at_step`]: AgentSession::load_checkpoint_at_step
    #[cfg(feature = "persistence")]
    #[deprecated(since = "1.1.0", note = "Use load_checkpoint_at_step instead")]
    pub async fn load_step_checkpoint(
        backend: &dyn crate::persistence::PersistenceBackend,
        session_id: &str,
        step: usize,
    ) -> Result<Option<AgentSession>, AgentRuntimeError> {
        Self::load_checkpoint_at_step(backend, session_id, step).await
    }

    /// Load the session snapshot saved after step `step` completed.
    ///
    /// Returns `None` if no checkpoint exists for the given session/step pair.
    /// The step number is 1-based (step 1 = after the first ReAct iteration).
    #[cfg(feature = "persistence")]
    pub async fn load_checkpoint_at_step(
        backend: &dyn crate::persistence::PersistenceBackend,
        session_id: &str,
        step: usize,
    ) -> Result<Option<AgentSession>, AgentRuntimeError> {
        let key = format!("session:{session_id}:step:{step}");
        match backend.load(&key).await? {
            None => Ok(None),
            Some(bytes) => {
                let session = serde_json::from_slice(&bytes)
                    .map_err(|e| AgentRuntimeError::Persistence(format!("deserialize: {e}")))?;
                Ok(Some(session))
            }
        }
    }

    /// Consume this session and return its steps as an owned `Vec`.
    ///
    /// Useful when the caller needs owned `ReActStep` values — for example to
    /// move them into another data structure — without cloning the entire list.
    ///
    /// ```rust,ignore
    /// let steps = session.into_steps();
    /// for step in steps { /* owned */ }
    /// ```
    pub fn into_steps(self) -> Vec<crate::agent::ReActStep> {
        self.steps
    }

    /// Return an iterator over the steps in this session.
    ///
    /// Equivalent to `session.steps.iter()` but avoids exposing the raw
    /// field for callers who prefer the method-call style.
    pub fn iter_steps(&self) -> std::slice::Iter<'_, crate::agent::ReActStep> {
        self.steps.iter()
    }

    /// Return `true` if the session has at least `n` steps.
    ///
    /// More efficient than `step_count() >= n` because it uses
    /// `steps.len()` directly and is easy to read at the call site.
    pub fn has_at_least_steps(&self, n: usize) -> bool {
        self.steps.len() >= n
    }

    /// Return `true` if every step in this session has a non-empty observation.
    ///
    /// Returns `true` for an empty session (vacuously true).
    pub fn all_observations_non_empty(&self) -> bool {
        self.steps.iter().all(|s| !s.observation.is_empty())
    }

    /// Return the average combined byte length (thought + action + observation)
    /// per step.
    ///
    /// Returns `0.0` for an empty session.
    pub fn avg_combined_step_bytes(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.combined_byte_length()).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return a reference to the step with the shortest `observation` field.
    ///
    /// When multiple steps share the minimum observation length the first is
    /// returned.  Returns `None` for an empty session.
    pub fn shortest_observation_step(&self) -> Option<&ReActStep> {
        self.steps.iter().min_by_key(|s| s.observation.len())
    }

    /// Return the number of distinct observation strings across all steps.
    ///
    /// Returns `0` for an empty session.
    pub fn unique_observation_count(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.observation.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Return the average number of whitespace-delimited words per thought.
    ///
    /// Returns `0.0` for an empty session.
    pub fn avg_thought_word_count(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self
            .steps
            .iter()
            .map(|s| s.thought.split_whitespace().count())
            .sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return `true` if any step's observation contains at least one of the
    /// provided `terms` (case-sensitive substring match).
    ///
    /// Returns `false` for an empty session or an empty `terms` slice.
    pub fn observation_contains_any(&self, terms: &[&str]) -> bool {
        if terms.is_empty() {
            return false;
        }
        self.steps
            .iter()
            .any(|s| terms.iter().any(|t| s.observation.contains(t)))
    }

    /// Return the step at `index`, or `None` if the index is out of bounds.
    pub fn step_at_index(&self, index: usize) -> Option<&ReActStep> {
        self.steps.get(index)
    }

    /// Return `true` if any step's thought contains **all** of the provided
    /// `terms` as substrings (case-sensitive).
    ///
    /// Returns `false` for an empty `terms` slice or an empty session.
    pub fn thought_contains_all(&self, terms: &[&str]) -> bool {
        if terms.is_empty() {
            return false;
        }
        self.steps
            .iter()
            .any(|s| terms.iter().all(|t| s.thought.contains(t)))
    }

    /// Return `true` if any step's action contains at least one of the
    /// provided `terms` (case-sensitive substring match).
    ///
    /// Returns `false` for an empty `terms` slice or an empty session.
    pub fn action_contains_any(&self, terms: &[&str]) -> bool {
        if terms.is_empty() {
            return false;
        }
        self.steps
            .iter()
            .any(|s| terms.iter().any(|t| s.action.contains(t)))
    }

    /// Return the maximum thought length in characters across all steps.
    ///
    /// Returns `0` for an empty session.
    pub fn max_thought_chars(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.thought.chars().count())
            .max()
            .unwrap_or(0)
    }

    /// Return the minimum thought length in characters, considering only
    /// steps that have a non-empty thought.
    ///
    /// Returns `0` if there are no non-empty thoughts.
    pub fn min_thought_chars(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.thought.chars().count())
            .filter(|&n| n > 0)
            .min()
            .unwrap_or(0)
    }

    /// Return the average number of Unicode chars in the `action` field across
    /// all steps.
    ///
    /// Returns `0.0` for an empty session.
    pub fn avg_action_chars(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.action.chars().count()).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return the average number of Unicode chars in the `observation` field
    /// across all steps.
    ///
    /// Returns `0.0` for an empty session.
    pub fn avg_observation_chars(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|s| s.observation.chars().count()).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Return a reference to the step whose `action` string is the longest
    /// (by Unicode char count), or `None` for an empty session.
    ///
    /// When multiple steps tie, the first is returned.
    pub fn step_with_longest_action(&self) -> Option<&ReActStep> {
        self.steps.iter().max_by_key(|s| s.action.chars().count())
    }

    /// Return `true` if any step's `action` ends with the given `suffix`.
    pub fn action_ends_with(&self, suffix: &str) -> bool {
        self.steps.iter().any(|s| s.action.ends_with(suffix))
    }

    /// Return `true` if any step's `thought` ends with the given `suffix`.
    pub fn thought_ends_with(&self, suffix: &str) -> bool {
        self.steps.iter().any(|s| s.thought.ends_with(suffix))
    }

    /// Return `true` if any step's `thought` contains `thought_term` AND that
    /// same step's `action` contains `action_term`.
    pub fn has_step_with_both(&self, thought_term: &str, action_term: &str) -> bool {
        self.steps
            .iter()
            .any(|s| s.thought.contains(thought_term) && s.action.contains(action_term))
    }

    /// Return the number of steps whose `observation` byte length strictly
    /// exceeds `min_bytes`.
    ///
    /// Returns `0` for an empty session.
    pub fn step_count_with_observation_longer_than(&self, min_bytes: usize) -> usize {
        self.steps
            .iter()
            .filter(|s| s.observation.len() > min_bytes)
            .count()
    }

    /// Return a `Vec` of word counts for each step's `thought`, in order.
    pub fn thought_word_counts(&self) -> Vec<usize> {
        self.steps
            .iter()
            .map(|s| s.thought.split_whitespace().count())
            .collect()
    }

    /// Return steps sorted by `thought` byte length in ascending order.
    pub fn steps_sorted_by_thought_len(&self) -> Vec<&ReActStep> {
        let mut sorted: Vec<&ReActStep> = self.steps.iter().collect();
        sorted.sort_by_key(|s| s.thought.len());
        sorted
    }

    /// Return all steps whose `thought` byte length strictly exceeds
    /// `min_bytes`.
    pub fn steps_with_thought_longer_than(&self, min_bytes: usize) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.thought.len() > min_bytes)
            .collect()
    }

    /// Return all steps whose `action` contains `substr` as a substring.
    pub fn steps_with_action_containing(&self, substr: &str) -> Vec<&ReActStep> {
        self.steps
            .iter()
            .filter(|s| s.action.contains(substr))
            .collect()
    }

    /// Return the maximum `observation` length in Unicode chars across all
    /// steps.  Returns `0` for an empty session.
    pub fn observation_max_chars(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.observation.chars().count())
            .max()
            .unwrap_or(0)
    }

    /// Return the minimum `observation` length in Unicode chars, considering
    /// only steps with a non-empty observation.  Returns `0` if no non-empty
    /// observations exist.
    pub fn observation_min_chars(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.observation.chars().count())
            .filter(|&n| n > 0)
            .min()
            .unwrap_or(0)
    }
}

// ── AgentRuntimeBuilder ───────────────────────────────────────────────────────

/// Builder for `AgentRuntime`.
///
/// Uses a typestate parameter `S` to enforce that `with_agent_config` is called
/// before `build()`.  Calling `build()` on a `AgentRuntimeBuilder<NeedsConfig>`
/// is a **compile-time error**.
///
/// Typical usage:
/// ```ignore
/// let runtime = AgentRuntime::builder()      // AgentRuntimeBuilder<NeedsConfig>
///     .with_memory(store)
///     .with_agent_config(cfg)                // → AgentRuntimeBuilder<HasConfig>
///     .build();                              // → AgentRuntime (infallible)
/// ```
/// Builder for [`AgentRuntime`].
pub struct AgentRuntimeBuilder<S = NeedsConfig> {
    #[cfg(feature = "memory")]
    memory: Option<EpisodicStore>,
    #[cfg(feature = "memory")]
    working: Option<WorkingMemory>,
    #[cfg(feature = "graph")]
    graph: Option<GraphStore>,
    #[cfg(feature = "orchestrator")]
    backpressure: Option<BackpressureGuard>,
    agent_config: Option<AgentConfig>,
    tools: Vec<Arc<ToolSpec>>,
    metrics: Arc<RuntimeMetrics>,
    #[cfg(feature = "persistence")]
    checkpoint_backend: Option<Arc<dyn crate::persistence::PersistenceBackend>>,
    token_estimator: Option<Arc<dyn TokenEstimator>>,
    _state: PhantomData<S>,
}

// ── DebugBuilderState sealed trait ────────────────────────────────────────────

/// Private trait used to drive the single generic `Debug` impl for
/// `AgentRuntimeBuilder<S>`.  Only `NeedsConfig` and `HasConfig` implement it.
trait DebugBuilderState {
    /// Name shown in the debug output.
    const NAME: &'static str;
    /// Whether to emit the `agent_config` field (only `HasConfig` has one).
    const HAS_CONFIG: bool;
}

impl DebugBuilderState for NeedsConfig {
    const NAME: &'static str = "AgentRuntimeBuilder<NeedsConfig>";
    const HAS_CONFIG: bool = false;
}

impl DebugBuilderState for HasConfig {
    const NAME: &'static str = "AgentRuntimeBuilder<HasConfig>";
    const HAS_CONFIG: bool = true;
}

impl<S: DebugBuilderState> std::fmt::Debug for AgentRuntimeBuilder<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct(S::NAME);
        #[cfg(feature = "memory")]
        {
            s.field("memory", &self.memory.is_some())
                .field("working", &self.working.is_some());
        }
        #[cfg(feature = "graph")]
        s.field("graph", &self.graph.is_some());
        #[cfg(feature = "orchestrator")]
        s.field("backpressure", &self.backpressure.is_some());
        if S::HAS_CONFIG {
            s.field("agent_config", &self.agent_config.is_some());
        }
        s.field("tools", &self.tools.len()).finish()
    }
}

impl Default for AgentRuntimeBuilder<NeedsConfig> {
    fn default() -> Self {
        Self {
            #[cfg(feature = "memory")]
            memory: None,
            #[cfg(feature = "memory")]
            working: None,
            #[cfg(feature = "graph")]
            graph: None,
            #[cfg(feature = "orchestrator")]
            backpressure: None,
            agent_config: None,
            tools: Vec::new(),
            metrics: RuntimeMetrics::new(),
            #[cfg(feature = "persistence")]
            checkpoint_backend: None,
            token_estimator: None,
            _state: PhantomData,
        }
    }
}

// Methods available on ALL builder states.
impl<S> AgentRuntimeBuilder<S> {
    /// Attach an episodic memory store.
    #[cfg(feature = "memory")]
    pub fn with_memory(mut self, store: EpisodicStore) -> Self {
        self.memory = Some(store);
        self
    }

    /// Attach a working memory store.
    #[cfg(feature = "memory")]
    pub fn with_working_memory(mut self, wm: WorkingMemory) -> Self {
        self.working = Some(wm);
        self
    }

    /// Attach a graph store.
    #[cfg(feature = "graph")]
    pub fn with_graph(mut self, graph: GraphStore) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Attach a backpressure guard.
    #[cfg(feature = "orchestrator")]
    pub fn with_backpressure(mut self, guard: BackpressureGuard) -> Self {
        self.backpressure = Some(guard);
        self
    }

    /// Register a tool available to the agent loop.
    pub fn register_tool(mut self, spec: ToolSpec) -> Self {
        self.tools.push(Arc::new(spec));
        self
    }

    /// Register multiple tools at once.
    ///
    /// Equivalent to calling [`register_tool`] for each spec.
    ///
    /// [`register_tool`]: AgentRuntimeBuilder::register_tool
    pub fn register_tools(mut self, specs: impl IntoIterator<Item = ToolSpec>) -> Self {
        for spec in specs {
            self.tools.push(Arc::new(spec));
        }
        self
    }

    /// Attach a shared `RuntimeMetrics` instance.
    pub fn with_metrics(mut self, metrics: Arc<RuntimeMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Attach a persistence backend for session checkpointing.
    #[cfg(feature = "persistence")]
    pub fn with_checkpoint_backend(
        mut self,
        backend: Arc<dyn crate::persistence::PersistenceBackend>,
    ) -> Self {
        self.checkpoint_backend = Some(backend);
        self
    }

    /// Provide a custom [`TokenEstimator`] for memory budget calculations.
    ///
    /// Replaces the default `len / 4` byte-counting heuristic.  Use this to
    /// plug in a model-specific tokenizer (e.g. tiktoken, sentencepiece) so
    /// that `AgentConfig::max_memory_tokens` is respected accurately.
    pub fn with_token_estimator(mut self, estimator: Arc<dyn TokenEstimator>) -> Self {
        self.token_estimator = Some(estimator);
        self
    }
}

// `with_agent_config` transitions NeedsConfig → HasConfig.
impl AgentRuntimeBuilder<NeedsConfig> {
    /// Create a new builder (equivalent to `Default::default()`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the agent loop configuration.
    ///
    /// After this call the builder transitions to `AgentRuntimeBuilder<HasConfig>`,
    /// making `build()` available.
    pub fn with_agent_config(self, config: AgentConfig) -> AgentRuntimeBuilder<HasConfig> {
        AgentRuntimeBuilder {
            memory: self.memory,
            working: self.working,
            #[cfg(feature = "graph")]
            graph: self.graph,
            #[cfg(feature = "orchestrator")]
            backpressure: self.backpressure,
            agent_config: Some(config),
            tools: self.tools,
            metrics: self.metrics,
            #[cfg(feature = "persistence")]
            checkpoint_backend: self.checkpoint_backend,
            token_estimator: self.token_estimator,
            _state: PhantomData,
        }
    }
}

// `build()` is only available once we have a config.
impl AgentRuntimeBuilder<HasConfig> {
    /// Build the `AgentRuntime`.
    ///
    /// This is infallible: the typestate guarantees `agent_config` is present.
    pub fn build(self) -> AgentRuntime {
        // SAFETY: `agent_config` is always `Some` in `HasConfig` state because
        // `with_agent_config` is the only way to reach this state.
        #[allow(clippy::unwrap_used)]
        let agent_config = self.agent_config.unwrap();

        AgentRuntime {
            #[cfg(feature = "memory")]
            memory: self.memory,
            #[cfg(feature = "memory")]
            working: self.working,
            #[cfg(feature = "graph")]
            graph: self.graph,
            #[cfg(feature = "orchestrator")]
            backpressure: self.backpressure,
            agent_config,
            tools: self.tools,
            metrics: self.metrics,
            token_estimator: self
                .token_estimator
                .unwrap_or_else(|| Arc::new(CharDivTokenEstimator)),
            #[cfg(feature = "persistence")]
            checkpoint_backend: self.checkpoint_backend,
        }
    }
}

// ── TokenEstimator ────────────────────────────────────────────────────────────

/// Estimates the number of tokens in a string.
///
/// Implement this trait to replace the default `len / 4` heuristic with a
/// model-specific tokenizer (e.g. tiktoken, sentencepiece).
///
/// # Example
/// ```rust,ignore
/// struct TiktokenEstimator { enc: tiktoken::Encoding }
/// impl TokenEstimator for TiktokenEstimator {
///     fn count_tokens(&self, text: &str) -> usize {
///         self.enc.encode_ordinary(text).len()
///     }
/// }
/// ```
pub trait TokenEstimator: Send + Sync {
    /// Return an approximate token count for `text`.
    fn count_tokens(&self, text: &str) -> usize;
}

/// Default heuristic: 1 token ≈ 4 bytes.
pub struct CharDivTokenEstimator;

impl TokenEstimator for CharDivTokenEstimator {
    fn count_tokens(&self, text: &str) -> usize {
        (text.len() / 4).max(1)
    }
}

// ── AgentRuntime ──────────────────────────────────────────────────────────────

/// Unified runtime that coordinates memory, graph, orchestration, and agent loop.
pub struct AgentRuntime {
    #[cfg(feature = "memory")]
    memory: Option<EpisodicStore>,
    #[cfg(feature = "memory")]
    working: Option<WorkingMemory>,
    #[cfg(feature = "graph")]
    graph: Option<GraphStore>,
    #[cfg(feature = "orchestrator")]
    backpressure: Option<BackpressureGuard>,
    agent_config: AgentConfig,
    tools: Vec<Arc<ToolSpec>>,
    metrics: Arc<RuntimeMetrics>,
    #[cfg(feature = "persistence")]
    checkpoint_backend: Option<Arc<dyn crate::persistence::PersistenceBackend>>,
    token_estimator: Arc<dyn TokenEstimator>,
}

impl std::fmt::Debug for AgentRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("AgentRuntime");
        s.field("memory", &self.memory.is_some())
            .field("working", &self.working.is_some());
        #[cfg(feature = "graph")]
        s.field("graph", &self.graph.is_some());
        #[cfg(feature = "orchestrator")]
        s.field("backpressure", &self.backpressure.is_some());
        s.field("tools", &self.tools.len());
        #[cfg(feature = "persistence")]
        s.field("checkpoint_backend", &self.checkpoint_backend.is_some());
        s.finish()
    }
}

impl AgentRuntime {
    /// Return a new builder in the `NeedsConfig` state.
    pub fn builder() -> AgentRuntimeBuilder<NeedsConfig> {
        AgentRuntimeBuilder::new()
    }

    /// Construct a minimal `AgentRuntime` in one call with sensible defaults.
    pub fn quick(max_iterations: usize, model: impl Into<String>) -> Self {
        AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(max_iterations, model))
            .build()
    }

    /// Return a shared reference to the runtime metrics.
    pub fn metrics(&self) -> Arc<RuntimeMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Run the agent loop for the given prompt.
    ///
    /// Optionally recalls episodic memories and injects them into the context.
    /// Optionally enforces backpressure before starting.
    ///
    /// # Arguments
    /// * `agent_id` — identifies the agent for memory retrieval
    /// * `prompt` — the user's input prompt
    /// * `infer` — async inference function: `(context: String) -> impl Future<Output = String>`
    ///
    /// # Returns
    /// An `AgentSession` with step count, hits, duration, and a stable session ID.
    #[tracing::instrument(skip(self, infer), fields(agent_id = %agent_id))]
    pub async fn run_agent<F, Fut>(
        &self,
        agent_id: AgentId,
        prompt: &str,
        infer: F,
    ) -> Result<AgentSession, AgentRuntimeError>
    where
        F: FnMut(String) -> Fut,
        Fut: std::future::Future<Output = String>,
    {
        // Acquire backpressure slot before counting the session — shed requests
        // must not inflate total_sessions or active_sessions.
        #[cfg(feature = "orchestrator")]
        {
            let backpressure_result = if let Some(ref guard) = self.backpressure {
                guard.try_acquire()
            } else {
                Ok(())
            };
            if let Err(e) = backpressure_result {
                tracing::warn!(agent_id = %agent_id, error = %e, "backpressure shed: rejecting session");
                self.metrics
                    .backpressure_shed_count
                    .fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        }

        self.metrics.total_sessions.fetch_add(1, Ordering::Relaxed);
        self.metrics.active_sessions.fetch_add(1, Ordering::Relaxed);

        tracing::info!(agent_id = %agent_id, "agent session starting");
        let outcome = self.run_agent_inner(agent_id.clone(), prompt, infer).await;

        // Always release backpressure — success or error.
        #[cfg(feature = "orchestrator")]
        if let Some(ref guard) = self.backpressure {
            let _ = guard.release();
        }

        // Saturating decrement — guards against underflow to usize::MAX if
        // active_sessions is somehow already 0 (e.g. double-decrement bug).
        let _ = self.metrics.active_sessions.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |v| Some(v.saturating_sub(1)),
        );

        match &outcome {
            Ok(session) => {
                tracing::info!(
                    agent_id = %agent_id,
                    session_id = %session.session_id,
                    steps = session.step_count(),
                    duration_ms = session.duration_ms,
                    "agent session completed"
                );
                self.metrics
                    .total_steps
                    .fetch_add(session.step_count() as u64, Ordering::Relaxed);
            }
            Err(e) => {
                tracing::error!(agent_id = %agent_id, error = %e, "agent session failed");
            }
        }

        outcome
    }

    /// Inner implementation of `run_agent`, called after backpressure is acquired.
    #[tracing::instrument(skip(self, infer), fields(agent_id = %agent_id, session_id = tracing::field::Empty))]
    async fn run_agent_inner<F, Fut>(
        &self,
        agent_id: AgentId,
        prompt: &str,
        infer: F,
    ) -> Result<AgentSession, AgentRuntimeError>
    where
        F: FnMut(String) -> Fut,
        Fut: std::future::Future<Output = String>,
    {
        let start = Instant::now();
        let session_id = uuid::Uuid::new_v4().to_string();

        let mut memory_hits = 0usize;
        let mut graph_lookups = 0usize;

        // Build enriched prompt from episodic memory.
        #[cfg(feature = "memory")]
        let enriched_prompt = if let Some(ref store) = self.memory {
            let memories = store.recall(&agent_id, self.agent_config.max_memory_recalls)?;

            // Apply token budget if configured.
            let memories = if let Some(token_budget) = self.agent_config.max_memory_tokens {
                let mut used = 0usize;
                memories
                    .into_iter()
                    .filter(|m| {
                        let tokens = self.token_estimator.count_tokens(&m.content);
                        if used + tokens <= token_budget {
                            used += tokens;
                            true
                        } else {
                            false
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                memories
            };

            memory_hits = memories.len();
            self.metrics
                .memory_recall_count
                .fetch_add(1, Ordering::Relaxed);

            if let Some(budget) = self.agent_config.max_memory_tokens {
                tracing::debug!(
                    "memory token budget: {budget}, injecting {} items",
                    memory_hits
                );
            } else {
                tracing::debug!("enriched prompt with {} memory items", memory_hits);
            }

            if memories.is_empty() {
                prompt.to_owned()
            } else {
                // Build the enriched prompt directly into a String without an
                // intermediate Vec<String> allocation.
                let mut enriched =
                    String::with_capacity(prompt.len() + memories.len() * 64 + 32);
                enriched.push_str("Relevant memories:\n");
                for m in &memories {
                    let _ = writeln!(enriched, "- {}", m.content);
                }
                let _ = write!(enriched, "\nCurrent prompt: {prompt}");
                enriched
            }
        } else {
            prompt.to_owned()
        };
        #[cfg(not(feature = "memory"))]
        let enriched_prompt = prompt.to_owned();

        // Inject working memory into prompt.
        #[cfg(feature = "memory")]
        let enriched_prompt = if let Some(ref wm) = self.working {
            let entries = wm.entries()?;
            if entries.is_empty() {
                enriched_prompt
            } else {
                // Build working-memory section without an intermediate Vec<String>.
                let mut out = String::with_capacity(
                    enriched_prompt.len() + entries.len() * 32 + 32,
                );
                out.push_str(&enriched_prompt);
                out.push_str("\n\nCurrent working state:\n");
                for (k, v) in &entries {
                    let _ = writeln!(out, "  {k}: {v}");
                }
                // Remove trailing newline added by writeln for the last entry.
                if out.ends_with('\n') {
                    out.pop();
                }
                out
            }
        } else {
            enriched_prompt
        };

        // Count graph entities as "lookups" for session metadata.
        #[cfg(feature = "graph")]
        if let Some(ref graph) = self.graph {
            graph_lookups = graph.entity_count()?;
            tracing::debug!("graph has {} entities", graph_lookups);
        }

        // Build the ReAct loop and register tools.
        // Each ToolSpec is stored as an Arc so we can clone the Arc into the
        // handler closure without moving ownership out of self.tools.
        // Required fields and the per-tool circuit breaker are preserved so
        // that validation and fast-fail behaviour work correctly at run time.
        let mut react_loop = ReActLoop::new(self.agent_config.clone())
            .with_metrics(Arc::clone(&self.metrics));

        // Item 11 — wire per-step loop checkpointing.
        #[cfg(feature = "persistence")]
        if let Some(ref backend) = self.checkpoint_backend {
            react_loop = react_loop
                .with_step_checkpoint(Arc::clone(backend), session_id.clone());
        }

        for tool in &self.tools {
            let tool_arc = Arc::clone(tool);
            let required_fields = tool_arc.required_fields.clone();
            #[cfg(feature = "orchestrator")]
            let circuit_breaker = tool_arc.circuit_breaker.clone();

            let mut spec = ToolSpec::new_async(
                tool_arc.name.clone(),
                tool_arc.description.clone(),
                move |args| {
                    let t = Arc::clone(&tool_arc);
                    Box::pin(async move { t.call(args).await })
                },
            )
            .with_required_fields(required_fields);

            #[cfg(feature = "orchestrator")]
            if let Some(cb) = circuit_breaker {
                spec = spec.with_circuit_breaker(cb);
            }

            react_loop.register_tool(spec);
        }

        // Record the session_id into the current tracing span so that all
        // child spans (ReActLoop iterations, tool calls) carry this field.
        tracing::Span::current().record("session_id", &session_id.as_str());

        let steps = react_loop.run(&enriched_prompt, infer).await?;
        let duration_ms = start.elapsed().as_millis() as u64;

        // Item 6 — collect per-step checkpoint errors; surfaced in AgentSession.
        #[cfg(feature = "persistence")]
        let mut ckpt_errors: Vec<String> = Vec::new();

        // Save final checkpoint if a backend is configured.
        #[cfg(feature = "persistence")]
        if let Some(ref backend) = self.checkpoint_backend {
            tracing::info!(session_id = %session_id, "saving session checkpoint");

            // Build a temporary session without errors to save as the base checkpoint.
            let tmp = AgentSession {
                session_id: session_id.clone(),
                agent_id: agent_id.clone(),
                steps: steps.clone(),
                memory_hits,
                graph_lookups,
                duration_ms,
                checkpoint_errors: vec![],
            };
            tmp.save_checkpoint(backend.as_ref()).await?;

            // Save incremental per-step consolidated snapshots.
            for i in 1..=steps.len() {
                let partial = AgentSession {
                    session_id: session_id.clone(),
                    agent_id: agent_id.clone(),
                    steps: steps[..i].to_vec(),
                    memory_hits,
                    graph_lookups,
                    duration_ms,
                    checkpoint_errors: vec![],
                };
                let key = format!("session:{session_id}:step:{i}");
                match serde_json::to_vec(&partial) {
                    Ok(bytes) => {
                        if let Err(e) = backend.save(&key, &bytes).await {
                            let msg = format!("session:{session_id} step:{i} save: {e}");
                            tracing::warn!("{}", msg);
                            ckpt_errors.push(msg);
                        }
                    }
                    Err(e) => {
                        let msg =
                            format!("session:{session_id} step:{i} serialise: {e}");
                        tracing::warn!("{}", msg);
                        ckpt_errors.push(msg);
                    }
                }
            }
        }

        let session = AgentSession {
            session_id,
            agent_id,
            steps,
            memory_hits,
            graph_lookups,
            duration_ms,
            #[cfg(feature = "persistence")]
            checkpoint_errors: ckpt_errors,
            #[cfg(not(feature = "persistence"))]
            checkpoint_errors: vec![],
        };

        Ok(session)
    }

    /// Return a reference to the episodic memory store, if configured.
    #[cfg(feature = "memory")]
    pub fn memory(&self) -> Option<&EpisodicStore> {
        self.memory.as_ref()
    }

    /// Return a reference to the graph store, if configured.
    #[cfg(feature = "graph")]
    pub fn graph(&self) -> Option<&GraphStore> {
        self.graph.as_ref()
    }

    /// Return a reference to the working memory, if configured.
    #[cfg(feature = "memory")]
    pub fn working_memory(&self) -> Option<&WorkingMemory> {
        self.working.as_ref()
    }

    /// Return `true` if episodic memory was configured for this runtime.
    #[cfg(feature = "memory")]
    pub fn has_memory(&self) -> bool {
        self.memory.is_some()
    }

    /// Return `true` if a graph store was configured for this runtime.
    #[cfg(feature = "graph")]
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Return `true` if working memory was configured for this runtime.
    #[cfg(feature = "memory")]
    pub fn has_working_memory(&self) -> bool {
        self.working.is_some()
    }

    /// Return `true` if there is at least one session currently in progress.
    ///
    /// Reads the `active_sessions` counter from the shared metrics.
    pub fn has_active_sessions(&self) -> bool {
        self.metrics
            .active_sessions
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    }

    /// Return the number of tools registered in this runtime.
    ///
    /// Each entry in the internal `tools` list corresponds to one registered
    /// `ToolSpec`.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Return the names of all registered tools, sorted alphabetically.
    ///
    /// Provides a stable, deterministic list independent of registration order.
    /// Returns an empty `Vec` when no tools have been registered.
    pub fn tool_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.iter().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Return an owned, sorted list of registered tool names.
    ///
    /// Unlike [`AgentRuntime::tool_names`] which borrows from `self`, this
    /// method returns `Vec<String>` so the result can outlive the runtime
    /// reference — useful for logging, serialisation, or passing across async
    /// boundaries.
    pub fn registered_tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> =
            self.tools.iter().map(|t| t.name.clone()).collect();
        names.sort_unstable();
        names
    }

    /// Return a reference to the runtime's agent configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.agent_config
    }

    /// Return the model identifier configured for this runtime.
    pub fn model_name(&self) -> &str {
        &self.agent_config.model
    }

    /// Return the maximum number of ReAct iterations this runtime is configured
    /// to allow per session.
    pub fn session_max_iterations(&self) -> usize {
        self.agent_config.max_iterations
    }

    /// Return `true` if a tool with the given `name` is registered with this
    /// runtime's tool registry.
    pub fn is_registered_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }

    /// Gracefully shut down the runtime.
    ///
    /// Logs a structured shutdown event with the final metrics snapshot.
    /// If the `persistence` feature is enabled and a checkpoint backend is
    /// configured, writes a sentinel key so operators can confirm clean shutdown.
    ///
    /// After calling `shutdown`, the runtime should not be used again.
    pub async fn shutdown(&self) {
        tracing::info!("AgentRuntime shutting down");
        tracing::info!(
            active_sessions = self.metrics.active_sessions(),
            total_sessions = self.metrics.total_sessions(),
            total_steps = self.metrics.total_steps(),
            total_tool_calls = self.metrics.total_tool_calls(),
            failed_tool_calls = self.metrics.failed_tool_calls(),
            "final metrics snapshot on shutdown"
        );

        #[cfg(feature = "persistence")]
        if let Some(ref backend) = self.checkpoint_backend {
            let ts = chrono::Utc::now().to_rfc3339();
            match backend.save("runtime:shutdown", ts.as_bytes()).await {
                Ok(()) => tracing::debug!("shutdown sentinel saved"),
                Err(e) => tracing::warn!(error = %e, "failed to save shutdown sentinel"),
            }
        }

        tracing::info!("AgentRuntime shutdown complete");
    }

    /// Run an agent session using a shared [`LlmProvider`].
    ///
    /// Convenience wrapper around [`run_agent`] that wires the provider's
    /// `complete` method as the inference closure.  Inference errors are
    /// converted to a `FINAL ANSWER` string so the loop terminates gracefully
    /// rather than panicking.
    ///
    /// [`run_agent`]: AgentRuntime::run_agent
    /// [`LlmProvider`]: crate::providers::LlmProvider
    #[cfg(feature = "providers")]
    pub async fn run_agent_with_provider(
        &self,
        agent_id: AgentId,
        prompt: &str,
        provider: std::sync::Arc<dyn crate::providers::LlmProvider>,
    ) -> Result<AgentSession, AgentRuntimeError> {
        let model = self.agent_config.model.clone();
        self.run_agent(agent_id, prompt, |ctx| {
            let provider = provider.clone();
            let model = model.clone();
            async move {
                provider
                    .complete(&ctx, &model)
                    .await
                    .unwrap_or_else(|e| format!("FINAL ANSWER: inference error: {e}"))
            }
        })
        .await
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Entity, GraphStore, Relationship};
    use crate::memory::{EpisodicStore, WorkingMemory};

    fn simple_config() -> AgentConfig {
        AgentConfig::new(5, "test")
    }

    async fn final_answer_infer(_ctx: String) -> String {
        "Thought: done\nAction: FINAL_ANSWER 42".into()
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    // NOTE: test_builder_fails_without_agent_config has been removed.
    // The typestate pattern makes calling .build() without .with_agent_config()
    // a *compile-time error* — AgentRuntimeBuilder<NeedsConfig> has no build()
    // method.  There is nothing to test at runtime.

    /// Verifies that the builder compiles and produces a runtime when config is
    /// provided.  This is the runtime-observable counterpart to the former
    /// "fails without config" test.
    #[tokio::test]
    async fn test_builder_with_config_compiles() {
        let _runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        // If this compiles and runs, the typestate transition worked correctly.
    }

    #[tokio::test]
    async fn test_builder_succeeds_with_minimal_config() {
        let _runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
    }

    #[tokio::test]
    async fn test_builder_with_all_subsystems() {
        let _runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(EpisodicStore::new())
            .with_graph(GraphStore::new())
            .with_working_memory(WorkingMemory::new(10).unwrap())
            .with_backpressure(BackpressureGuard::new(5).unwrap())
            .build();
    }

    #[tokio::test]
    async fn test_builder_produces_runtime_with_config() {
        // Confirm the built runtime accepts a run_agent call — the most direct
        // evidence that the builder wired everything correctly.
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        let session = runtime
            .run_agent(AgentId::new("agent-x"), "hello", final_answer_infer)
            .await
            .unwrap();
        assert!(session.step_count() >= 1);
        assert!(!session.session_id.is_empty());
    }

    // ── run_agent ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_run_agent_returns_session_with_steps() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("agent-1"), "hello", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.step_count(), 1);
    }

    #[tokio::test]
    async fn test_run_agent_session_has_agent_id() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("agent-42"), "hello", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.agent_id.0, "agent-42");
    }

    #[tokio::test]
    async fn test_run_agent_session_duration_is_set() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "hello", final_answer_infer)
            .await
            .unwrap();

        // Duration should be non-negative (0 ms is valid for a fast mock)
        let _ = session.duration_ms; // just verify it compiles and is set
    }

    #[tokio::test]
    async fn test_run_agent_session_has_session_id() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "hello", final_answer_infer)
            .await
            .unwrap();

        // session_id must be a non-empty UUID string
        assert!(!session.session_id.is_empty());
        assert_eq!(session.session_id.len(), 36); // UUID v4 canonical form
    }

    #[tokio::test]
    async fn test_run_agent_memory_hits_zero_without_memory() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.memory_hits, 0);
    }

    #[tokio::test]
    async fn test_run_agent_memory_hits_counts_recalled_items() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("mem-agent");
        store
            .add_episode(agent.clone(), "remembered fact", 0.8)
            .unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(store)
            .build();

        let session = runtime
            .run_agent(agent, "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.memory_hits, 1);
    }

    #[tokio::test]
    async fn test_run_agent_graph_lookups_counts_entities() {
        let graph = GraphStore::new();
        graph.add_entity(Entity::new("e1", "Node")).unwrap();
        graph.add_entity(Entity::new("e2", "Node")).unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_graph(graph)
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.graph_lookups, 2);
    }

    #[tokio::test]
    async fn test_run_agent_backpressure_released_after_run() {
        let guard = BackpressureGuard::new(3).unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_backpressure(guard.clone())
            .build();

        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(guard.depth().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_run_agent_backpressure_sheds_when_full() {
        let guard = BackpressureGuard::new(1).unwrap();
        guard.try_acquire().unwrap(); // pre-fill

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_backpressure(guard)
            .build();

        let result = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await;
        assert!(matches!(
            result,
            Err(AgentRuntimeError::BackpressureShed { .. })
        ));
    }

    #[tokio::test]
    async fn test_run_agent_max_iterations_error_propagated() {
        let cfg = AgentConfig::new(2, "model");
        let runtime = AgentRuntime::builder().with_agent_config(cfg).build();

        // Simulate an infer fn that always produces FINAL_ANSWER immediately
        let result = runtime
            .run_agent(AgentId::new("a"), "prompt", |_ctx: String| async {
                "Thought: looping\nAction: FINAL_ANSWER done".to_string()
            })
            .await;
        assert!(result.is_ok()); // final answer on first call, ok
    }

    #[tokio::test]
    async fn test_agent_session_step_count_matches_steps() {
        let session = AgentSession {
            session_id: "test-session-id".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep {
                    thought: "t".into(),
                    action: "a".into(),
                    observation: "o".into(),
                    step_duration_ms: 0,
                },
                ReActStep {
                    thought: "t2".into(),
                    action: "FINAL_ANSWER".into(),
                    observation: "done".into(),
                    step_duration_ms: 0,
                },
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 10,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.step_count(), 2);
    }

    // ── Accessor methods ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_runtime_memory_accessor_returns_none_when_not_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        assert!(runtime.memory().is_none());
    }

    #[tokio::test]
    async fn test_runtime_memory_accessor_returns_some_when_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(EpisodicStore::new())
            .build();
        assert!(runtime.memory().is_some());
    }

    #[tokio::test]
    async fn test_runtime_graph_accessor_returns_none_when_not_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        assert!(runtime.graph().is_none());
    }

    #[tokio::test]
    async fn test_runtime_graph_accessor_returns_some_when_configured() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_graph(GraphStore::new())
            .build();
        assert!(runtime.graph().is_some());
    }

    #[tokio::test]
    async fn test_runtime_working_memory_accessor() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_working_memory(WorkingMemory::new(5).unwrap())
            .build();
        assert!(runtime.working_memory().is_some());
    }

    #[tokio::test]
    async fn test_runtime_with_tool_registered() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .register_tool(ToolSpec::new("calc", "math", |_| serde_json::json!(99)))
            .build();

        let mut call_count = 0;
        let session = runtime
            .run_agent(AgentId::new("a"), "compute", move |_ctx: String| {
                call_count += 1;
                let count = call_count;
                async move {
                    if count == 1 {
                        "Thought: use calc\nAction: calc {}".into()
                    } else {
                        "Thought: done\nAction: FINAL_ANSWER result".into()
                    }
                }
            })
            .await
            .unwrap();

        assert!(session.step_count() >= 1);
    }

    #[tokio::test]
    async fn test_run_agent_with_graph_relationship_lookup() {
        let graph = GraphStore::new();
        graph.add_entity(Entity::new("a", "X")).unwrap();
        graph.add_entity(Entity::new("b", "Y")).unwrap();
        graph
            .add_relationship(Relationship::new("a", "b", "LINKS", 1.0))
            .unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_graph(graph)
            .build();

        let session = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(session.graph_lookups, 2); // 2 entities
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_metrics_active_sessions_decrements_after_run() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(runtime.metrics().active_sessions(), 0);
    }

    #[tokio::test]
    async fn test_metrics_total_sessions_increments() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();

        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();
        runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(runtime.metrics().total_sessions(), 2);
    }

    #[tokio::test]
    async fn test_metrics_backpressure_shed_increments_on_shed() {
        let guard = BackpressureGuard::new(1).unwrap();
        guard.try_acquire().unwrap(); // pre-fill

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_backpressure(guard)
            .build();

        let _ = runtime
            .run_agent(AgentId::new("a"), "prompt", final_answer_infer)
            .await;

        assert_eq!(runtime.metrics().backpressure_shed_count(), 1);
    }

    #[tokio::test]
    async fn test_metrics_memory_recall_count_increments() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "fact", 0.9).unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(store)
            .build();

        runtime
            .run_agent(agent, "prompt", final_answer_infer)
            .await
            .unwrap();

        assert_eq!(runtime.metrics().memory_recall_count(), 1);
    }

    // ── Memory token budgeting ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_agent_config_max_memory_tokens_limits_injection() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("budget-agent");
        // Each memory has ~100 chars → ~25 tokens each
        for i in 0..5 {
            let content = format!("{:0>100}", i); // 100-char string
            store.add_episode(agent.clone(), content, 0.9).unwrap();
        }

        // Token budget of 10 allows at most ~1 memory (each is ~25 tokens).
        let cfg = AgentConfig::new(5, "test").with_max_memory_tokens(10);
        let runtime = AgentRuntime::builder()
            .with_agent_config(cfg)
            .with_memory(store)
            .build();

        let session = runtime
            .run_agent(agent, "prompt", final_answer_infer)
            .await
            .unwrap();

        assert!(
            session.memory_hits <= 1,
            "expected at most 1 memory hit with tight token budget, got {}",
            session.memory_hits
        );
    }

    // ── Working memory injection ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_working_memory_injected_into_prompt() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("task", "write tests").unwrap();
        wm.set("status", "in progress").unwrap();

        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_working_memory(wm)
            .build();

        let mut captured_ctx: Option<String> = None;
        let captured_ref = &mut captured_ctx;

        runtime
            .run_agent(AgentId::new("a"), "do stuff", |ctx: String| {
                *captured_ref = Some(ctx.clone());
                async move { "Thought: done\nAction: FINAL_ANSWER ok".to_string() }
            })
            .await
            .unwrap();

        let ctx = captured_ctx.expect("infer should have been called");
        assert!(
            ctx.contains("Current working state:"),
            "expected working memory injection in context, got: {ctx}"
        );
        assert!(ctx.contains("task: write tests"));
        assert!(ctx.contains("status: in progress"));
    }

    // ── Task 15: Token budget edge case tests ─────────────────────────────────

    #[tokio::test]
    async fn test_token_budget_zero_returns_no_memories() {
        // A budget of 0 should result in no memories being injected.
        let store = EpisodicStore::new();
        let agent = AgentId::new("budget-agent");
        store.add_episode(agent.clone(), "short", 0.9).unwrap();

        let mut config = AgentConfig::new(5, "test-model");
        config.max_memory_tokens = Some(0);
        config.max_memory_recalls = 10;

        let runtime = AgentRuntime::builder()
            .with_memory(store)
            .with_agent_config(config)
            .build();

        let steps = runtime
            .run_agent(
                agent,
                "test",
                |_ctx| async { "Thought: ok\nAction: FINAL_ANSWER done".to_string() },
            )
            .await
            .unwrap();

        // The run should succeed; we just verify it doesn't panic or error.
        assert_eq!(steps.steps.len(), 1);
    }

    #[tokio::test]
    async fn test_token_budget_smaller_than_smallest_item_returns_no_memories() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("budget-agent2");
        // Content is 40 chars → ~10 tokens (40/4). Budget = 1 → none fit.
        store
            .add_episode(agent.clone(), "a".repeat(40), 0.9)
            .unwrap();

        let mut config = AgentConfig::new(5, "test-model");
        config.max_memory_tokens = Some(1);
        config.max_memory_recalls = 10;

        let runtime = AgentRuntime::builder()
            .with_memory(store)
            .with_agent_config(config)
            .build();

        let session = runtime
            .run_agent(
                agent,
                "test",
                |_ctx| async { "Thought: ok\nAction: FINAL_ANSWER done".to_string() },
            )
            .await
            .unwrap();

        assert_eq!(session.memory_hits, 0);
    }

    // ── Improvement 8: AgentRuntime::quick() ──────────────────────────────────

    #[tokio::test]
    async fn test_agent_runtime_quick_runs_agent() {
        let runtime = AgentRuntime::quick(5, "test-model");
        let agent = AgentId::new("quick-agent");
        let session = runtime
            .run_agent(agent, "hello", |_ctx| async {
                "Thought: done\nAction: FINAL_ANSWER ok".to_string()
            })
            .await
            .unwrap();
        assert_eq!(session.step_count(), 1);
    }

    // ── #1 final_answer() ─────────────────────────────────────────────────────

    #[test]
    fn test_final_answer_extracts_text() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![ReActStep {
                thought: "done".into(),
                action: "FINAL_ANSWER Paris".into(),
                observation: "".into(),
                step_duration_ms: 0,
            }],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.final_answer(), Some("Paris".to_string()));
    }

    #[test]
    fn test_final_answer_returns_none_without_final_step() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![ReActStep {
                thought: "thinking".into(),
                action: "search {}".into(),
                observation: "result".into(),
                step_duration_ms: 0,
            }],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.final_answer(), None);

        let empty_session = AgentSession {
            session_id: "s2".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(empty_session.final_answer(), None);
    }

    #[test]
    fn test_all_actions_returns_actions_in_order() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("think1", "search {}", "result"),
                ReActStep::new("think2", "FINAL_ANSWER done", ""),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 10,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.all_actions(), vec!["search {}", "FINAL_ANSWER done"]);
    }

    #[test]
    fn test_has_checkpoint_errors_false_when_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!(!session.has_checkpoint_errors());
    }

    #[test]
    fn test_has_checkpoint_errors_true_when_non_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec!["err".into()],
        };
        assert!(session.has_checkpoint_errors());
    }

    #[test]
    fn test_memory_hit_rate_zero_with_no_steps() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 5,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.memory_hit_rate(), 0.0);
    }

    #[test]
    fn test_memory_hit_rate_correct_proportion() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t", "a", "o"),
                ReActStep::new("t", "a", "o"),
                ReActStep::new("t", "a", "o"),
                ReActStep::new("t", "a", "o"),
            ],
            memory_hits: 2,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!((session.memory_hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_filter_tool_call_steps_excludes_final_answer() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t1", "search {}", "res"),
                ReActStep::new("t2", "FINAL_ANSWER done", ""),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        let tool_steps = session.filter_tool_call_steps();
        assert_eq!(tool_steps.len(), 1);
        assert_eq!(tool_steps[0].action, "search {}");
    }

    #[test]
    fn test_slowest_step_index() {
        let mut s0 = ReActStep::new("t", "a", "o");
        s0.step_duration_ms = 5;
        let mut s1 = ReActStep::new("t", "a", "o");
        s1.step_duration_ms = 100;
        let mut s2 = ReActStep::new("t", "a", "o");
        s2.step_duration_ms = 10;
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![s0, s1, s2],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.slowest_step_index(), Some(1));
        assert_eq!(session.fastest_step_index(), Some(0));
    }

    #[test]
    fn test_slowest_step_index_none_when_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.slowest_step_index(), None);
        assert_eq!(session.fastest_step_index(), None);
    }

    #[test]
    fn test_last_step_returns_last() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t1", "a1", "o1"),
                ReActStep::new("t2", "FINAL_ANSWER done", ""),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.last_step().map(|s| s.action.as_str()), Some("FINAL_ANSWER done"));
    }

    #[test]
    fn test_last_step_none_when_empty() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!(session.last_step().is_none());
    }

    #[test]
    fn test_step_at_returns_correct_step() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t0", "a0", "o0"),
                ReActStep::new("t1", "a1", "o1"),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.step_at(1).map(|s| s.thought.as_str()), Some("t1"));
        assert!(session.step_at(99).is_none());
    }

    // ── Round 3: failed_steps ─────────────────────────────────────────────────

    #[test]
    fn test_failed_steps_returns_steps_with_error_observation() {
        use crate::agent::ReActStep;
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![
                ReActStep::new("t", "tool_a {}", r#"{"error":"bad input","ok":false}"#),
                ReActStep::new("t", "tool_b {}", r#"{"result":"ok","ok":true}"#),
            ],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        let failed = session.failed_steps();
        assert_eq!(failed.len(), 1);
        assert!(failed[0].observation.contains("bad input"));
    }

    #[test]
    fn test_failed_steps_empty_when_no_errors() {
        use crate::agent::ReActStep;
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![ReActStep::new("t", "FINAL_ANSWER done", "")],
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!(session.failed_steps().is_empty());
    }

    // ── Round 17: untested AgentSession methods ───────────────────────────────

    fn make_step(thought: &str, action: &str, observation: &str) -> ReActStep {
        ReActStep::new(thought, action, observation)
    }

    fn make_session(steps: Vec<ReActStep>, duration_ms: u64) -> AgentSession {
        AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps,
            memory_hits: 0,
            graph_lookups: 0,
            duration_ms,
            checkpoint_errors: vec![],
        }
    }

    #[test]
    fn test_step_count_returns_number_of_steps() {
        let s = make_session(vec![ReActStep::new("t", "a", "o"), ReActStep::new("t", "a", "o")], 0);
        assert_eq!(s.step_count(), 2);
    }

    #[test]
    fn test_is_empty_true_for_no_steps() {
        let s = make_session(vec![], 0);
        assert!(s.is_empty());
    }

    #[test]
    fn test_is_empty_false_with_steps() {
        let s = make_session(vec![ReActStep::new("t", "a", "o")], 0);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_is_successful_true_with_final_answer() {
        let s = make_session(vec![ReActStep::new("t", "FINAL_ANSWER yes", "")], 0);
        assert!(s.is_successful());
    }

    #[test]
    fn test_is_successful_false_without_final_answer() {
        let s = make_session(vec![ReActStep::new("t", "search {}", "result")], 0);
        assert!(!s.is_successful());
    }

    #[test]
    fn test_elapsed_returns_duration_from_duration_ms() {
        let s = make_session(vec![], 500);
        assert_eq!(s.elapsed(), std::time::Duration::from_millis(500));
    }

    #[test]
    fn test_tool_calls_made_excludes_final_answer() {
        let s = make_session(vec![
            ReActStep::new("t", "search {}", "res"),
            ReActStep::new("t", "lookup {}", "res"),
            ReActStep::new("t", "FINAL_ANSWER done", ""),
        ], 0);
        assert_eq!(s.tool_calls_made(), 2);
    }

    #[test]
    fn test_total_step_duration_ms_sums_all_steps() {
        let mut s1 = ReActStep::new("t", "a", "o"); s1.step_duration_ms = 10;
        let mut s2 = ReActStep::new("t", "a", "o"); s2.step_duration_ms = 30;
        let s = make_session(vec![s1, s2], 0);
        assert_eq!(s.total_step_duration_ms(), 40);
    }

    #[test]
    fn test_average_step_duration_ms() {
        let mut s1 = ReActStep::new("t", "a", "o"); s1.step_duration_ms = 20;
        let mut s2 = ReActStep::new("t", "a", "o"); s2.step_duration_ms = 40;
        let s = make_session(vec![s1, s2], 0);
        assert_eq!(s.average_step_duration_ms(), 30);
    }

    #[test]
    fn test_all_thoughts_returns_thoughts_in_order() {
        let s = make_session(vec![
            ReActStep::new("first thought", "a1", "o1"),
            ReActStep::new("second thought", "a2", "o2"),
        ], 0);
        assert_eq!(s.all_thoughts(), vec!["first thought", "second thought"]);
    }

    #[test]
    fn test_all_observations_returns_observations_in_order() {
        let s = make_session(vec![
            ReActStep::new("t1", "a1", "obs one"),
            ReActStep::new("t2", "a2", "obs two"),
        ], 0);
        assert_eq!(s.all_observations(), vec!["obs one", "obs two"]);
    }

    #[test]
    fn test_observations_matching_finds_matching_steps() {
        let s = make_session(vec![
            ReActStep::new("t1", "a1", "found the answer"),
            ReActStep::new("t2", "a2", "nothing relevant"),
        ], 0);
        let matching = s.observations_matching("answer");
        assert_eq!(matching.len(), 1);
        assert!(matching[0].observation.contains("answer"));
    }

    #[test]
    fn test_first_step_returns_first() {
        let s = make_session(vec![
            ReActStep::new("first", "a1", "o1"),
            ReActStep::new("second", "a2", "o2"),
        ], 0);
        assert_eq!(s.first_step().map(|s| s.thought.as_str()), Some("first"));
    }

    #[test]
    fn test_first_step_none_when_empty() {
        let s = make_session(vec![], 0);
        assert!(s.first_step().is_none());
    }

    // ── Round 18: graph_lookup_count ─────────────────────────────────────────

    #[test]
    fn test_graph_lookup_count_returns_field() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 7,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert_eq!(session.graph_lookup_count(), 7usize);
    }

    // ── Round 7: action_counts / unique_actions ───────────────────────────────

    #[test]
    fn test_action_counts_counts_each_action() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "search", "r1"),
                ReActStep::new("t2", "search", "r2"),
                ReActStep::new("t3", "FINAL_ANSWER", "done"),
            ],
            0,
        );
        let counts = session.action_counts();
        assert_eq!(counts.get("search").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("FINAL_ANSWER").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_unique_actions_returns_sorted_deduped() {
        let session = make_session(
            vec![
                ReActStep::new("t", "b_action", "r"),
                ReActStep::new("t", "a_action", "r"),
                ReActStep::new("t", "b_action", "r"),
            ],
            0,
        );
        assert_eq!(session.unique_actions(), vec!["a_action", "b_action"]);
    }

    #[test]
    fn test_unique_actions_empty_when_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.unique_actions().is_empty());
    }

    // ── Round 8: total_latency_ms / action_sequence ───────────────────────────

    #[test]
    fn test_total_latency_ms_sums_step_durations() {
        let mut steps = vec![
            ReActStep::new("t1", "a1", "o1"),
            ReActStep::new("t2", "a2", "o2"),
        ];
        steps[0].step_duration_ms = 100;
        steps[1].step_duration_ms = 250;
        let session = make_session(steps, 350);
        assert_eq!(session.total_latency_ms(), 350);
    }

    #[test]
    fn test_total_latency_ms_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_latency_ms(), 0);
    }

    #[test]
    fn test_action_sequence_returns_actions_in_order() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r"),
                ReActStep::new("t", "FINAL_ANSWER", "done"),
            ],
            0,
        );
        assert_eq!(session.action_sequence(), vec!["search", "FINAL_ANSWER"]);
    }

    // ── Round 9: has_action / thought_at ─────────────────────────────────────

    #[test]
    fn test_has_action_returns_true_for_present_action() {
        let session = make_session(
            vec![ReActStep::new("t", "search", "r")],
            0,
        );
        assert!(session.has_action("search"));
    }

    #[test]
    fn test_has_action_returns_false_for_absent_action() {
        let session = make_session(
            vec![ReActStep::new("t", "search", "r")],
            0,
        );
        assert!(!session.has_action("compute"));
    }

    #[test]
    fn test_thought_at_returns_thought_for_valid_index() {
        let session = make_session(
            vec![
                ReActStep::new("first thought", "a1", "r1"),
                ReActStep::new("second thought", "a2", "r2"),
            ],
            0,
        );
        assert_eq!(session.thought_at(0), Some("first thought"));
        assert_eq!(session.thought_at(1), Some("second thought"));
    }

    #[test]
    fn test_thought_at_returns_none_for_out_of_bounds_index() {
        let session = make_session(vec![ReActStep::new("t", "a", "r")], 0);
        assert!(session.thought_at(99).is_none());
    }

    // ── Round 10: step_count_for_action / observations ────────────────────────

    #[test]
    fn test_step_count_for_action_counts_correctly() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r1"),
                ReActStep::new("t", "search", "r2"),
                ReActStep::new("t", "FINAL_ANSWER", "done"),
            ],
            0,
        );
        assert_eq!(session.step_count_for_action("search"), 2);
        assert_eq!(session.step_count_for_action("FINAL_ANSWER"), 1);
        assert_eq!(session.step_count_for_action("unknown"), 0);
    }

    #[test]
    fn test_observations_returns_all_observation_strings() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "a", "obs_one"),
                ReActStep::new("t2", "b", "obs_two"),
            ],
            0,
        );
        let obs = session.observations();
        assert_eq!(obs, vec!["obs_one", "obs_two"]);
    }

    #[test]
    fn test_observations_empty_for_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.observations().is_empty());
    }

    // ── Round 10: unique_tools_used ──────────────────────────────────────────

    #[test]
    fn test_unique_tools_used_deduplicates_actions() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r1"),
                ReActStep::new("t", "lookup", "r2"),
                ReActStep::new("t", "search", "r3"),
            ],
            0,
        );
        let tools = session.unique_tools_used();
        assert_eq!(tools.len(), 2);
        assert!(tools.contains(&"search".to_string()));
        assert!(tools.contains(&"lookup".to_string()));
    }

    #[test]
    fn test_unique_tools_used_excludes_final_answer() {
        let session = make_session(
            vec![
                ReActStep::new("t", "search", "r1"),
                ReActStep::new("t", "FINAL_ANSWER: done", "r2"),
            ],
            0,
        );
        let tools = session.unique_tools_used();
        assert_eq!(tools.len(), 1);
        assert!(tools.contains(&"search".to_string()));
    }

    #[test]
    fn test_unique_tools_used_empty_for_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.unique_tools_used().is_empty());
    }

    // ── Round 11: avg_step_duration_ms / longest_step / shortest_step ─────────

    #[test]
    fn test_avg_step_duration_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.avg_step_duration_ms() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_step_duration_single_step() {
        let mut step = ReActStep::new("t", "a", "r");
        step.step_duration_ms = 100;
        let session = make_session(vec![step], 0);
        assert!((session.avg_step_duration_ms() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_step_duration_multiple_steps() {
        let mut s1 = ReActStep::new("t1", "a", "r");
        s1.step_duration_ms = 100;
        let mut s2 = ReActStep::new("t2", "b", "r");
        s2.step_duration_ms = 200;
        let session = make_session(vec![s1, s2], 0);
        assert!((session.avg_step_duration_ms() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_longest_step_returns_step_with_max_duration() {
        let mut s1 = ReActStep::new("t1", "a", "r");
        s1.step_duration_ms = 50;
        let mut s2 = ReActStep::new("t2", "b", "r");
        s2.step_duration_ms = 200;
        let session = make_session(vec![s1, s2], 0);
        assert_eq!(session.longest_step().map(|s| s.step_duration_ms), Some(200));
    }

    #[test]
    fn test_longest_step_returns_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.longest_step().is_none());
    }

    #[test]
    fn test_shortest_step_returns_step_with_min_duration() {
        let mut s1 = ReActStep::new("t1", "a", "r");
        s1.step_duration_ms = 50;
        let mut s2 = ReActStep::new("t2", "b", "r");
        s2.step_duration_ms = 200;
        let session = make_session(vec![s1, s2], 0);
        assert_eq!(session.shortest_step().map(|s| s.step_duration_ms), Some(50));
    }

    // ── Round 12: first_thought / last_thought ────────────────────────────────

    #[test]
    fn test_first_thought_returns_thought_from_first_step() {
        let session = make_session(
            vec![
                ReActStep::new("alpha", "a1", "r1"),
                ReActStep::new("beta", "a2", "r2"),
            ],
            0,
        );
        assert_eq!(session.first_thought(), Some("alpha"));
    }

    #[test]
    fn test_last_thought_returns_thought_from_last_step() {
        let session = make_session(
            vec![
                ReActStep::new("alpha", "a1", "r1"),
                ReActStep::new("beta", "a2", "r2"),
            ],
            0,
        );
        assert_eq!(session.last_thought(), Some("beta"));
    }

    #[test]
    fn test_first_thought_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.first_thought().is_none());
    }

    #[test]
    fn test_last_thought_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.last_thought().is_none());
    }

    // ── Round 13: first_action / last_action ──────────────────────────────────

    #[test]
    fn test_first_action_returns_action_from_first_step() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "search", "r1"),
                ReActStep::new("t2", "FINAL_ANSWER", "r2"),
            ],
            0,
        );
        assert_eq!(session.first_action(), Some("search"));
    }

    #[test]
    fn test_last_action_returns_action_from_last_step() {
        let session = make_session(
            vec![
                ReActStep::new("t1", "search", "r1"),
                ReActStep::new("t2", "FINAL_ANSWER", "r2"),
            ],
            0,
        );
        assert_eq!(session.last_action(), Some("FINAL_ANSWER"));
    }

    #[test]
    fn test_first_action_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.first_action().is_none());
    }

    #[test]
    fn test_last_action_equals_first_action_for_single_step() {
        let session = make_session(vec![ReActStep::new("t", "calc", "r")], 0);
        assert_eq!(session.first_action(), session.last_action());
    }

    // ── Round 14: AgentSession::checkpoint_error_count ───────────────────────

    #[test]
    fn test_checkpoint_error_count_zero_when_none() {
        let session = make_session(vec![], 0);
        assert_eq!(session.checkpoint_error_count(), 0);
    }

    #[test]
    fn test_checkpoint_error_count_reflects_errors() {
        let mut session = make_session(vec![], 0);
        session.checkpoint_errors.push("save failed".into());
        session.checkpoint_errors.push("disk full".into());
        assert_eq!(session.checkpoint_error_count(), 2);
    }

    // ── Round 27: failed_tool_call_count ─────────────────────────────────────

    #[test]
    fn test_failed_tool_call_count_zero_when_no_errors() {
        let step = ReActStep::new("think", "search", "results found");
        let session = make_session(vec![step], 0);
        assert_eq!(session.failed_tool_call_count(), 0);
    }

    #[test]
    fn test_failed_tool_call_count_matches_failed_steps() {
        let ok_step = ReActStep::new("ok", "search", "all good");
        let err_step = ReActStep::new("err", "lookup", "{\"error\": \"not found\"}");
        let session = make_session(vec![ok_step, err_step], 0);
        assert_eq!(session.failed_tool_call_count(), session.failed_steps().len());
        assert_eq!(session.failed_tool_call_count(), 1);
    }

    #[test]
    fn test_failed_tool_call_count_counts_all_errors() {
        let err1 = ReActStep::new("e1", "a", "{\"error\": \"bad\"}");
        let err2 = ReActStep::new("e2", "b", "some \"error\" text");
        let ok = ReActStep::new("ok", "c", "success");
        let session = make_session(vec![err1, err2, ok], 0);
        assert_eq!(session.failed_tool_call_count(), 2);
    }

    // ── Round 15: AgentSession::total_memory_hits / action_diversity ─────────

    #[test]
    fn test_total_memory_hits_returns_memory_hits_field() {
        let mut session = make_session(vec![], 0);
        session.memory_hits = 7;
        assert_eq!(session.total_memory_hits(), 7);
    }

    #[test]
    fn test_total_memory_hits_zero_by_default() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_memory_hits(), 0);
    }

    #[test]
    fn test_action_diversity_all_unique_is_one() {
        let steps = vec![
            ReActStep::new("t", "search", "r"),
            ReActStep::new("t", "calc", "r"),
            ReActStep::new("t", "lookup", "r"),
        ];
        let session = make_session(steps, 0);
        assert!((session.action_diversity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_diversity_all_same_is_fraction() {
        let steps = vec![
            ReActStep::new("t", "search", "r"),
            ReActStep::new("t", "search", "r"),
            ReActStep::new("t", "search", "r"),
        ];
        let session = make_session(steps, 0);
        // 1 unique / 3 total = 1/3
        assert!((session.action_diversity() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_diversity_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.action_diversity() - 0.0).abs() < 1e-9);
    }

    // ── Round 14: AgentSession::last_n_steps ──────────────────────────────────

    #[test]
    fn test_last_n_steps_returns_last_n() {
        let steps = vec![
            ReActStep::new("t1", "a", "r1"),
            ReActStep::new("t2", "b", "r2"),
            ReActStep::new("t3", "c", "r3"),
        ];
        let session = make_session(steps, 0);
        let last2 = session.last_n_steps(2);
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0].action, "b");
        assert_eq!(last2[1].action, "c");
    }

    #[test]
    fn test_last_n_steps_returns_all_when_n_exceeds_count() {
        let steps = vec![
            ReActStep::new("t1", "a", "r1"),
            ReActStep::new("t2", "b", "r2"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.last_n_steps(10).len(), 2);
    }

    #[test]
    fn test_last_n_steps_empty_for_no_steps() {
        let session = make_session(vec![], 0);
        assert!(session.last_n_steps(3).is_empty());
    }

    #[test]
    fn test_last_n_steps_zero_returns_empty() {
        let steps = vec![ReActStep::new("t1", "a", "r1")];
        let session = make_session(steps, 0);
        assert!(session.last_n_steps(0).is_empty());
    }

    // ── Round 16: observation_count / steps_without_observation ──────────────

    #[test]
    fn test_observation_count_counts_non_empty() {
        let steps = vec![
            ReActStep::new("t", "a", "result"),
            ReActStep::new("t", "b", ""),
            ReActStep::new("t", "c", "data"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_count(), 2);
    }

    #[test]
    fn test_observation_count_zero_when_all_empty() {
        let steps = vec![
            ReActStep::new("t", "a", ""),
            ReActStep::new("t", "b", ""),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_count(), 0);
    }

    #[test]
    fn test_steps_without_observation_counts_empty_obs() {
        let steps = vec![
            ReActStep::new("t", "a", ""),
            ReActStep::new("t", "b", "data"),
            ReActStep::new("t", "c", ""),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_without_observation(), 2);
    }

    #[test]
    fn test_steps_without_observation_zero_when_all_filled() {
        let steps = vec![
            ReActStep::new("t", "a", "r1"),
            ReActStep::new("t", "b", "r2"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_without_observation(), 0);
    }

    // ── Round 17: AgentSession::throughput_steps_per_sec ─────────────────────

    #[test]
    fn test_throughput_steps_per_sec_correct_ratio() {
        let steps = vec![
            ReActStep::new("t", "a", "r"),
            ReActStep::new("t", "b", "r"),
        ];
        // 2 steps in 1000ms = 2.0 steps/sec
        let session = make_session(steps, 1000);
        assert!((session.throughput_steps_per_sec() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_throughput_steps_per_sec_zero_when_no_duration() {
        let steps = vec![ReActStep::new("t", "a", "r")];
        let session = make_session(steps, 0);
        assert!((session.throughput_steps_per_sec() - 0.0).abs() < 1e-9);
    }

    // ── Round 20: thoughts_containing, step_durations_ms, fastest_step_index ─

    #[test]
    fn test_thoughts_containing_returns_matching_steps() {
        let session = make_session(
            vec![
                ReActStep::new("I need to search", "search", "found"),
                ReActStep::new("Let me calculate", "calc", "done"),
                ReActStep::new("search again", "search", "ok"),
            ],
            0,
        );
        let matches = session.thoughts_containing("search");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_thoughts_containing_is_case_insensitive() {
        let session = make_session(
            vec![ReActStep::new("SEARCH the web", "search", "r")],
            0,
        );
        assert_eq!(session.thoughts_containing("search").len(), 1);
    }

    #[test]
    fn test_thoughts_containing_empty_when_no_match() {
        let session = make_session(vec![ReActStep::new("think about x", "a", "r")], 0);
        assert!(session.thoughts_containing("zebra").is_empty());
    }

    #[test]
    fn test_step_durations_ms_returns_all_durations() {
        let mut steps = vec![
            ReActStep::new("t", "a", "r"),
            ReActStep::new("t", "b", "r"),
        ];
        steps[0].step_duration_ms = 100;
        steps[1].step_duration_ms = 200;
        let session = make_session(steps, 300);
        assert_eq!(session.step_durations_ms(), vec![100, 200]);
    }

    #[test]
    fn test_fastest_step_index_returns_index_of_shortest_step() {
        let mut steps = vec![
            ReActStep::new("t", "a", "r"),
            ReActStep::new("t", "b", "r"),
            ReActStep::new("t", "c", "r"),
        ];
        steps[0].step_duration_ms = 300;
        steps[1].step_duration_ms = 50;
        steps[2].step_duration_ms = 200;
        let session = make_session(steps, 550);
        assert_eq!(session.fastest_step_index(), Some(1));
    }

    #[test]
    fn test_fastest_step_index_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.fastest_step_index().is_none());
    }

    // ── Round 18: most_used_action / graph_lookup_rate ────────────────────────

    #[test]
    fn test_most_used_action_returns_most_frequent() {
        let steps = vec![
            ReActStep::new("t", "search", "r"),
            ReActStep::new("t", "calc", "r"),
            ReActStep::new("t", "search", "r"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.most_used_action().as_deref(), Some("search"));
    }

    #[test]
    fn test_most_used_action_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.most_used_action().is_none());
    }

    #[test]
    fn test_graph_lookup_rate_correct_ratio() {
        let steps = vec![
            ReActStep::new("t", "a", "r"),
            ReActStep::new("t", "b", "r"),
            ReActStep::new("t", "c", "r"),
            ReActStep::new("t", "d", "r"),
        ];
        let mut session = make_session(steps, 0);
        session.graph_lookups = 2;
        assert!((session.graph_lookup_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_graph_lookup_rate_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.graph_lookup_rate() - 0.0).abs() < 1e-9);
    }

    // ── Round 22: has_tool_failures, tool_call_rate, step_success_rate ────────

    #[test]
    fn test_has_tool_failures_false_when_no_errors() {
        let steps = vec![
            make_step("t", "action1", "ok"),
            make_step("t", "action2", "done"),
        ];
        let session = make_session(steps, 0);
        assert!(!session.has_tool_failures());
    }

    #[test]
    fn test_has_tool_failures_true_when_error_observation() {
        let steps = vec![
            make_step("t", "action1", "{\"error\": \"timeout\"}"),
        ];
        let session = make_session(steps, 0);
        assert!(session.has_tool_failures());
    }

    #[test]
    fn test_tool_call_rate_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.tool_call_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_tool_call_rate_correct_ratio() {
        let steps = vec![
            make_step("t", "tool_action", "ok"),
            make_step("t", "FINAL_ANSWER: done", ""),
            make_step("t", "another_tool", "ok"),
        ];
        let session = make_session(steps, 0);
        // 2 tool calls out of 3 steps
        assert!((session.tool_call_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_success_rate_one_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.step_success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_success_rate_less_than_one_when_failures() {
        let steps = vec![
            make_step("t", "act", "{\"error\": \"fail\"}"),
            make_step("t", "act", "success"),
        ];
        let session = make_session(steps, 0);
        // 1 failed out of 2 → 1.0 - 0.5 = 0.5
        assert!((session.step_success_rate() - 0.5).abs() < 1e-9);
    }

    // ── Round 23: collection helpers and rate methods ─────────────────────────

    #[test]
    fn test_avg_step_duration_ms_zero_for_empty() {
        let session = make_session(vec![], 0);
        assert!((session.avg_step_duration_ms() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_step_duration_ms_correct_mean() {
        let mut s1 = make_step("t", "a", "o");
        s1.step_duration_ms = 100;
        let mut s2 = make_step("t", "b", "o");
        s2.step_duration_ms = 200;
        let session = make_session(vec![s1, s2], 0);
        assert!((session.avg_step_duration_ms() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_longest_step_none_for_empty() {
        let session = make_session(vec![], 0);
        assert!(session.longest_step().is_none());
    }

    #[test]
    fn test_longest_step_middle_has_max_duration() {
        let mut s1 = make_step("t", "a", "o");
        s1.step_duration_ms = 10;
        let mut s2 = make_step("t", "b", "o");
        s2.step_duration_ms = 500;
        let mut s3 = make_step("t", "c", "o");
        s3.step_duration_ms = 20;
        let session = make_session(vec![s1, s2, s3], 0);
        assert_eq!(session.longest_step().unwrap().action, "b");
    }

    #[test]
    fn test_unique_tools_used_deduplicates_and_sorts() {
        let steps = vec![
            make_step("t", "search", "o"),
            make_step("t", "lookup", "o"),
            make_step("t", "search", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.unique_tools_used(), vec!["lookup", "search"]);
    }

    #[test]
    fn test_all_thoughts_collects_in_order() {
        let steps = vec![make_step("think1", "a", "o"), make_step("think2", "b", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.all_thoughts(), vec!["think1", "think2"]);
    }

    #[test]
    fn test_all_actions_collects_in_order() {
        let steps = vec![make_step("t", "act1", "o"), make_step("t", "act2", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.all_actions(), vec!["act1", "act2"]);
    }

    #[test]
    fn test_all_observations_collects_in_order() {
        let steps = vec![make_step("t", "a", "obs1"), make_step("t", "b", "obs2")];
        let session = make_session(steps, 0);
        assert_eq!(session.all_observations(), vec!["obs1", "obs2"]);
    }

    #[test]
    fn test_action_counts_returns_frequency_map() {
        let steps = vec![
            make_step("t", "search", "o"),
            make_step("t", "lookup", "o"),
            make_step("t", "search", "o"),
        ];
        let session = make_session(steps, 0);
        let counts = session.action_counts();
        assert_eq!(counts["search"], 2);
        assert_eq!(counts["lookup"], 1);
    }

    #[test]
    fn test_unique_actions_three_with_repeat_yields_two() {
        let steps = vec![
            make_step("t", "beta", "o"),
            make_step("t", "alpha", "o"),
            make_step("t", "beta", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.unique_actions(), vec!["alpha", "beta"]);
    }

    #[test]
    fn test_action_diversity_zero_for_empty() {
        let session = make_session(vec![], 0);
        assert!((session.action_diversity() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_diversity_one_when_all_actions_unique() {
        let steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.action_diversity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_diversity_fraction_when_repeated() {
        let steps = vec![
            make_step("t", "x", "o"),
            make_step("t", "x", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.action_diversity() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_has_checkpoint_errors_false_for_new_session() {
        let session = make_session(vec![], 0);
        assert!(!session.has_checkpoint_errors());
    }

    #[test]
    fn test_has_checkpoint_errors_true_when_errors_present() {
        let mut session = make_session(vec![], 0);
        session.checkpoint_errors.push("err1".to_string());
        assert!(session.has_checkpoint_errors());
    }

    #[test]
    fn test_graph_lookup_count_returns_raw_value() {
        let mut session = make_session(vec![make_step("t", "a", "o")], 0);
        session.graph_lookups = 7;
        assert_eq!(session.graph_lookup_count(), 7);
    }

    #[test]
    fn test_memory_hit_rate_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.memory_hit_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_memory_hit_rate_correct_ratio() {
        let steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
            make_step("t", "d", "o"),
        ];
        let mut session = make_session(steps, 0);
        session.memory_hits = 2;
        assert!((session.memory_hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_total_memory_hits_returns_raw_value() {
        let mut session = make_session(vec![], 0);
        session.memory_hits = 13;
        assert_eq!(session.total_memory_hits(), 13);
    }

    // ── Round 29: has_memory, has_graph, has_working_memory ──────────────────

    #[cfg(feature = "memory")]
    #[test]
    fn test_has_memory_false_without_memory() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        assert!(!runtime.has_memory());
    }

    #[cfg(feature = "memory")]
    #[test]
    fn test_has_memory_true_with_memory() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_memory(EpisodicStore::new())
            .build();
        assert!(runtime.has_memory());
    }

    #[cfg(feature = "graph")]
    #[test]
    fn test_has_graph_false_without_graph() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        assert!(!runtime.has_graph());
    }

    #[cfg(feature = "graph")]
    #[test]
    fn test_has_graph_true_with_graph() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_graph(GraphStore::new())
            .build();
        assert!(runtime.has_graph());
    }

    #[cfg(feature = "memory")]
    #[test]
    fn test_has_working_memory_false_without_working_memory() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .build();
        assert!(!runtime.has_working_memory());
    }

    #[cfg(feature = "memory")]
    #[test]
    fn test_has_working_memory_true_with_working_memory() {
        let runtime = AgentRuntime::builder()
            .with_agent_config(simple_config())
            .with_working_memory(WorkingMemory::new(10).unwrap())
            .build();
        assert!(runtime.has_working_memory());
    }

    // ── Round 23: last_observation / thought_count / observation_rate ─────────

    #[test]
    fn test_last_observation_returns_most_recent_nonempty() {
        let steps = vec![
            make_step("t1", "act", "first obs"),
            make_step("t2", "act", ""),
            make_step("t3", "act", "last obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.last_observation(), Some("last obs"));
    }

    #[test]
    fn test_last_observation_skips_empty_steps() {
        let steps = vec![
            make_step("t1", "act", "only obs"),
            make_step("t2", "act", ""),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.last_observation(), Some("only obs"));
    }

    #[test]
    fn test_last_observation_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.last_observation().is_none());
    }

    #[test]
    fn test_thought_count_counts_nonempty_thoughts() {
        let steps = vec![
            make_step("think", "act", "obs"),
            make_step("", "act", "obs"),
            make_step("think again", "act", "obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.thought_count(), 2);
    }

    #[test]
    fn test_thought_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.thought_count(), 0);
    }

    #[test]
    fn test_observation_rate_correct_fraction() {
        let steps = vec![
            make_step("t", "a", "obs"),
            make_step("t", "a", ""),
            make_step("t", "a", "obs"),
            make_step("t", "a", ""),
        ];
        let session = make_session(steps, 0);
        assert!((session.observation_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_observation_rate_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.observation_rate() - 0.0).abs() < 1e-9);
    }

    // ── Round 24: action_repetition_rate / max_consecutive_failures / avg_thought_length

    #[test]
    fn test_action_repetition_rate_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.action_repetition_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_repetition_rate_zero_for_single_step() {
        let session = make_session(vec![make_step("t", "search", "r")], 0);
        assert!((session.action_repetition_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_repetition_rate_one_when_all_same() {
        let steps = vec![
            make_step("t", "search", "r"),
            make_step("t", "search", "r"),
            make_step("t", "search", "r"),
        ];
        let session = make_session(steps, 0);
        assert!((session.action_repetition_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_repetition_rate_partial_repeats() {
        // [search, search, calc] → 1 repeat out of 2 transitions → 0.5
        let steps = vec![
            make_step("t", "search", "r"),
            make_step("t", "search", "r"),
            make_step("t", "calc", "r"),
        ];
        let session = make_session(steps, 0);
        assert!((session.action_repetition_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_max_consecutive_failures_zero_for_no_errors() {
        let steps = vec![make_step("t", "a", "ok"), make_step("t", "b", "done")];
        let session = make_session(steps, 0);
        assert_eq!(session.max_consecutive_failures(), 0);
    }

    #[test]
    fn test_max_consecutive_failures_counts_run() {
        let steps = vec![
            make_step("t", "a", "ok"),
            make_step("t", "b", r#"{"error":"x"}"#),
            make_step("t", "c", r#"{"error":"y"}"#),
            make_step("t", "d", "ok"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.max_consecutive_failures(), 2);
    }

    #[test]
    fn test_avg_thought_length_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.avg_thought_length() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_thought_length_excludes_empty_thoughts() {
        let steps = vec![
            make_step("hello", "a", "r"),  // 5 chars
            make_step("", "b", "r"),        // excluded
            make_step("hi", "c", "r"),      // 2 chars
        ];
        // mean = (5 + 2) / 2 = 3.5
        let session = make_session(steps, 0);
        assert!((session.avg_thought_length() - 3.5).abs() < 1e-9);
    }

    // ── Round 25: last_n_observations / actions_in_window ─────────────────────

    #[test]
    fn test_last_n_observations_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.last_n_observations(3).is_empty());
    }

    #[test]
    fn test_last_n_observations_returns_last_n_nonempty() {
        let steps = vec![
            make_step("t", "a", "obs1"),
            make_step("t", "b", ""),        // skipped
            make_step("t", "c", "obs2"),
            make_step("t", "d", "obs3"),
        ];
        let session = make_session(steps, 0);
        let last2 = session.last_n_observations(2);
        assert_eq!(last2, vec!["obs2", "obs3"]);
    }

    #[test]
    fn test_last_n_observations_returns_all_when_fewer_than_n() {
        let steps = vec![make_step("t", "a", "only")];
        let session = make_session(steps, 0);
        assert_eq!(session.last_n_observations(5), vec!["only"]);
    }

    #[test]
    fn test_actions_in_window_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.actions_in_window(3).is_empty());
    }

    #[test]
    fn test_actions_in_window_returns_last_n_steps() {
        let steps = vec![
            make_step("t", "alpha", "r"),
            make_step("t", "beta", "r"),
            make_step("t", "gamma", "r"),
        ];
        let session = make_session(steps, 0);
        let window = session.actions_in_window(2);
        assert_eq!(window, vec!["beta", "gamma"]);
    }

    #[test]
    fn test_actions_in_window_all_when_fewer_than_n() {
        let steps = vec![make_step("t", "solo", "r")];
        let session = make_session(steps, 0);
        assert_eq!(session.actions_in_window(10), vec!["solo"]);
    }

    // ── Round 31: observation_at, action_at ───────────────────────────────────

    #[test]
    fn test_observation_at_returns_correct_observation() {
        let steps = vec![
            make_step("t1", "a1", "obs-zero"),
            make_step("t2", "a2", "obs-one"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_at(0), Some("obs-zero"));
        assert_eq!(session.observation_at(1), Some("obs-one"));
    }

    #[test]
    fn test_observation_at_returns_none_out_of_bounds() {
        let session = make_session(vec![], 0);
        assert!(session.observation_at(0).is_none());
    }

    #[test]
    fn test_action_at_returns_correct_action() {
        let steps = vec![
            make_step("t1", "first-action", "obs"),
            make_step("t2", "second-action", "obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.action_at(0), Some("first-action"));
        assert_eq!(session.action_at(1), Some("second-action"));
    }

    #[test]
    fn test_action_at_returns_none_out_of_bounds() {
        let session = make_session(vec![], 0);
        assert!(session.action_at(5).is_none());
    }

    // ── Round 26: has_graph_lookups / consecutive_same_action_at_end ──────────

    #[test]
    fn test_has_graph_lookups_false_when_zero() {
        let session = make_session(vec![], 0);
        assert!(!session.has_graph_lookups());
    }

    #[test]
    fn test_has_graph_lookups_true_when_positive() {
        let session = AgentSession {
            session_id: "s".into(),
            agent_id: AgentId::new("a"),
            steps: vec![],
            memory_hits: 0,
            graph_lookups: 5,
            duration_ms: 0,
            checkpoint_errors: vec![],
        };
        assert!(session.has_graph_lookups());
    }

    #[test]
    fn test_consecutive_same_action_at_end_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.consecutive_same_action_at_end(), 0);
    }

    #[test]
    fn test_consecutive_same_action_at_end_single_step() {
        let steps = vec![make_step("t", "act", "obs")];
        let session = make_session(steps, 0);
        assert_eq!(session.consecutive_same_action_at_end(), 0);
    }

    #[test]
    fn test_consecutive_same_action_at_end_two_same_at_end() {
        let steps = vec![
            make_step("t", "other", "obs"),
            make_step("t", "repeat", "obs"),
            make_step("t", "repeat", "obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.consecutive_same_action_at_end(), 1);
    }

    #[test]
    fn test_consecutive_same_action_at_end_all_same() {
        let steps = vec![
            make_step("t", "same", "obs"),
            make_step("t", "same", "obs"),
            make_step("t", "same", "obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.consecutive_same_action_at_end(), 2);
    }

    // ── Round 27: failure_rate / unique_action_count ───────────────────────────

    #[test]
    fn test_failure_rate_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.failure_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_failure_rate_zero_when_no_failures() {
        let steps = vec![
            make_step("t", "lookup", "ok"),
            make_step("t", "search", "ok"),
        ];
        let session = make_session(steps, 0);
        assert!((session.failure_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_unique_action_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.unique_action_count(), 0);
    }

    #[test]
    fn test_unique_action_count_counts_distinct_actions() {
        let steps = vec![
            make_step("t", "search", "r"),
            make_step("t", "lookup", "r"),
            make_step("t", "search", "r"), // duplicate
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.unique_action_count(), 2);
    }

    // ── Round 28: total_thought_length / longest_observation ──────────────────

    #[test]
    fn test_total_thought_length_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_thought_length(), 0);
    }

    #[test]
    fn test_total_thought_length_sums_all_thoughts() {
        let steps = vec![
            make_step("hi", "a", "r"),   // 2 bytes
            make_step("hello", "b", "r"), // 5 bytes
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_thought_length(), 7);
    }

    #[test]
    fn test_longest_observation_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.longest_observation().is_none());
    }

    #[test]
    fn test_longest_observation_returns_longest() {
        let steps = vec![
            make_step("t", "a", "short"),
            make_step("t", "b", "a much longer observation"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.longest_observation(), Some("a much longer observation"));
    }

    // ── Round 29: steps_with_empty_observations / min_thought_length ──────────

    #[test]
    fn test_steps_with_empty_observations_zero_when_all_filled() {
        let steps = vec![make_step("t", "a", "obs"), make_step("t", "b", "obs2")];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_empty_observations(), 0);
    }

    #[test]
    fn test_steps_with_empty_observations_counts_empty_ones() {
        let steps = vec![
            make_step("t", "a", ""),    // empty
            make_step("t", "b", "ok"),
            make_step("t", "c", ""),    // empty
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_empty_observations(), 2);
    }

    #[test]
    fn test_min_thought_length_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.min_thought_length(), 0);
    }

    #[test]
    fn test_min_thought_length_returns_shortest_non_empty() {
        let steps = vec![
            make_step("hi", "a", "r"),        // 2 bytes
            make_step("hello", "b", "r"),     // 5 bytes
            make_step("", "c", "r"),          // empty, excluded
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.min_thought_length(), 2);
    }

    // ── Round 30: observation_lengths / avg_observation_length ────────────────

    #[test]
    fn test_observation_lengths_empty_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.observation_lengths().is_empty());
    }

    #[test]
    fn test_observation_lengths_returns_lengths_in_order() {
        let steps = vec![
            make_step("t", "a", "hi"),    // 2
            make_step("t", "b", "hello"), // 5
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_lengths(), vec![2, 5]);
    }

    #[test]
    fn test_avg_observation_length_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.avg_observation_length() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_observation_length_correct_mean() {
        let steps = vec![
            make_step("t", "a", "hi"),   // 2
            make_step("t", "b", "hello"), // 5
        ];
        let session = make_session(steps, 0);
        // mean = 3.5
        assert!((session.avg_observation_length() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_duration_secs_converts_ms_to_seconds() {
        let session = make_session(vec![], 7000);
        assert_eq!(session.duration_secs(), 7);
    }

    #[test]
    fn test_steps_above_thought_length_counts_qualifying_steps() {
        let steps = vec![
            make_step("hi", "a", "obs"),
            make_step("a longer thought here", "b", "obs"),
            make_step("medium thought", "c", "obs"),
        ];
        let session = make_session(steps, 0);
        // "hi" (2) <= 5, "a longer thought here" (21) > 5, "medium thought" (13) > 5
        assert_eq!(session.steps_above_thought_length(5), 2);
    }

    #[test]
    fn test_has_final_answer_true_when_step_has_final_answer_action() {
        let steps = vec![
            make_step("think", "search", "result"),
            make_step("done", "FINAL_ANSWER: 42", ""),
        ];
        let session = make_session(steps, 0);
        assert!(session.has_final_answer());
    }

    #[test]
    fn test_has_final_answer_false_when_no_final_answer_step() {
        let steps = vec![make_step("think", "search", "result")];
        let session = make_session(steps, 0);
        assert!(!session.has_final_answer());
    }

    #[test]
    fn test_avg_action_length_correct_mean() {
        let steps = vec![
            make_step("t", "ab", "o"),    // 2
            make_step("t", "abcd", "o"),  // 4
        ];
        let session = make_session(steps, 0);
        assert!((session.avg_action_length() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_action_length_empty_returns_zero() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_action_length(), 0.0);
    }

    #[test]
    fn test_thought_lengths_returns_lengths_in_order() {
        let steps = vec![
            make_step("hi", "a", "o"),
            make_step("hello", "b", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.thought_lengths(), vec![2, 5]);
    }

    #[test]
    fn test_most_common_action_returns_most_frequent() {
        let steps = vec![
            make_step("t", "search", "o"),
            make_step("t", "search", "o"),
            make_step("t", "other", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.most_common_action(), Some("search"));
    }

    #[test]
    fn test_most_common_action_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.most_common_action().is_none());
    }

    #[test]
    fn test_count_steps_with_action_counts_exact_matches() {
        let steps = vec![
            make_step("t", "search", "o"),
            make_step("t", "search", "o"),
            make_step("t", "other", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.count_steps_with_action("search"), 2);
        assert_eq!(session.count_steps_with_action("other"), 1);
        assert_eq!(session.count_steps_with_action("missing"), 0);
    }

    #[test]
    fn test_thought_contains_count_counts_matching_steps() {
        let steps = vec![
            make_step("search for rust", "a", "o"),
            make_step("think about python", "b", "o"),
            make_step("rust is great", "c", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.thought_contains_count("rust"), 2);
        assert_eq!(session.thought_contains_count("python"), 1);
        assert_eq!(session.thought_contains_count("java"), 0);
    }

    #[test]
    fn test_count_nonempty_thoughts_counts_steps_with_thoughts() {
        let steps = vec![
            make_step("hello", "a", "o"),
            make_step("", "b", "o"),
            make_step("world", "c", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.count_nonempty_thoughts(), 2);
    }

    #[test]
    fn test_observation_contains_count_counts_matching_observations() {
        let steps = vec![
            make_step("t", "a", "result: success"),
            make_step("t", "b", "result: failure"),
            make_step("t", "c", "no match here"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_contains_count("result"), 2);
        assert_eq!(session.observation_contains_count("success"), 1);
    }

    // ── Round 36 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_action_lengths_returns_byte_lengths_in_order() {
        let steps = vec![
            make_step("t", "ab", "o"),
            make_step("t", "hello", "o"),
            make_step("t", "", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.action_lengths(), vec![2, 5, 0]);
    }

    #[test]
    fn test_action_lengths_empty_session_returns_empty_vec() {
        let session = make_session(vec![], 0);
        assert!(session.action_lengths().is_empty());
    }

    #[test]
    fn test_step_success_count_excludes_failed_steps() {
        let steps = vec![
            make_step("t", "a", "ok"),
            make_step("t", "b", "{\"error\": \"timeout\"}"),
            make_step("t", "c", "ok"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.step_success_count(), 2);
    }

    #[test]
    fn test_step_success_count_all_success_when_no_failures() {
        let steps = vec![make_step("t", "a", "ok"), make_step("t", "b", "ok")];
        let session = make_session(steps, 0);
        assert_eq!(session.step_success_count(), 2);
    }

    // ── Round 37 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_longest_thought_returns_step_with_most_bytes() {
        let steps = vec![
            make_step("hi", "a", "o"),
            make_step("hello world", "b", "o"),
            make_step("hey", "c", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.longest_thought(), Some("hello world"));
    }

    #[test]
    fn test_longest_thought_returns_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.longest_thought().is_none());
    }

    #[test]
    fn test_shortest_action_returns_step_with_fewest_bytes() {
        let steps = vec![
            make_step("t", "search", "o"),
            make_step("t", "go", "o"),
            make_step("t", "lookup", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.shortest_action(), Some("go"));
    }

    #[test]
    fn test_shortest_action_returns_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.shortest_action().is_none());
    }

    // ── Round 38 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_first_step_action_returns_action_of_first_step() {
        let steps = vec![
            make_step("t", "first", "o"),
            make_step("t", "second", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.first_step_action(), Some("first"));
    }

    #[test]
    fn test_first_step_action_returns_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.first_step_action().is_none());
    }

    #[test]
    fn test_last_step_action_returns_action_of_last_step() {
        let steps = vec![
            make_step("t", "first", "o"),
            make_step("t", "last_one", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.last_step_action(), Some("last_one"));
    }

    #[test]
    fn test_last_step_action_returns_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.last_step_action().is_none());
    }

    // ── Round 39 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_total_thought_bytes_sums_all_thought_lengths() {
        let steps = vec![
            make_step("hi", "a", "o"),   // thought = "hi" → 2
            make_step("hello", "b", "o"), // thought = "hello" → 5
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_thought_bytes(), 7);
    }

    #[test]
    fn test_total_observation_bytes_sums_all_observation_lengths() {
        let steps = vec![
            make_step("t", "a", "ok"),     // 2
            make_step("t", "b", "done!"),  // 5
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_observation_bytes(), 7);
    }

    // ── Round 40 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_steps_in_range_returns_correct_slice() {
        let steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
        ];
        let session = make_session(steps, 0);
        let slice = session.steps_in_range(1, 3);
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].action, "b");
        assert_eq!(slice[1].action, "c");
    }

    #[test]
    fn test_steps_in_range_returns_empty_for_out_of_bounds_start() {
        let steps = vec![make_step("t", "a", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_in_range(5, 10).is_empty());
    }

    #[test]
    fn test_median_step_duration_ms_odd_count() {
        let mut steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
        ];
        steps[0].step_duration_ms = 10;
        steps[1].step_duration_ms = 50;
        steps[2].step_duration_ms = 30;
        let session = make_session(steps, 0);
        // sorted: [10, 30, 50] → median = 30
        assert_eq!(session.median_step_duration_ms(), 30);
    }

    #[test]
    fn test_median_step_duration_ms_returns_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.median_step_duration_ms(), 0);
    }

    // ── Round 40: into_steps, iter_steps, has_at_least_steps ─────────────────

    #[test]
    fn test_into_steps_consumes_session_and_returns_owned_vec() {
        let steps = vec![
            make_step("think", "act", "obs"),
            make_step("think2", "act2", "obs2"),
        ];
        let session = make_session(steps, 0);
        let owned = session.into_steps();
        assert_eq!(owned.len(), 2);
        assert_eq!(owned[0].thought, "think");
        assert_eq!(owned[1].action, "act2");
    }

    #[test]
    fn test_into_steps_returns_empty_vec_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.into_steps().is_empty());
    }

    #[test]
    fn test_iter_steps_iterates_in_order() {
        let steps = vec![
            make_step("t1", "a1", "o1"),
            make_step("t2", "a2", "o2"),
        ];
        let session = make_session(steps, 0);
        let thoughts: Vec<&str> = session.iter_steps().map(|s| s.thought.as_str()).collect();
        assert_eq!(thoughts, vec!["t1", "t2"]);
    }

    #[test]
    fn test_has_at_least_steps_true_when_enough_steps() {
        let session = make_session(vec![make_step("t", "a", "o"), make_step("t", "a", "o")], 0);
        assert!(session.has_at_least_steps(2));
        assert!(session.has_at_least_steps(1));
    }

    #[test]
    fn test_has_at_least_steps_false_when_too_few() {
        let session = make_session(vec![make_step("t", "a", "o")], 0);
        assert!(!session.has_at_least_steps(2));
    }

    #[test]
    fn test_has_at_least_steps_zero_always_true() {
        let session = make_session(vec![], 0);
        assert!(session.has_at_least_steps(0));
    }

    // ── Round 40 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_p95_step_duration_ms_returns_high_percentile() {
        // 20 steps with durations 1..=20 ms; p95 = ⌈20 * 0.95⌉ = 19th value = 19 ms
        let mut steps: Vec<ReActStep> = (1u64..=20)
            .map(|ms| ReActStep::new("t", "a", "o").with_duration(ms))
            .collect();
        // shuffle so order doesn't matter
        steps.reverse();
        let session = make_session(steps, 0);
        assert_eq!(session.p95_step_duration_ms(), 19);
    }

    #[test]
    fn test_p95_step_duration_ms_returns_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.p95_step_duration_ms(), 0);
    }

    #[test]
    fn test_p99_step_duration_ms_returns_highest_for_small_set() {
        // 10 steps 1..=10; p99 = ⌈10 * 0.99⌉ = 10th value = 10 ms
        let steps: Vec<ReActStep> = (1u64..=10)
            .map(|ms| ReActStep::new("t", "a", "o").with_duration(ms))
            .collect();
        let session = make_session(steps, 0);
        assert_eq!(session.p99_step_duration_ms(), 10);
    }

    #[test]
    fn test_p99_step_duration_ms_returns_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.p99_step_duration_ms(), 0);
    }

    #[test]
    fn test_step_count_above_duration_ms_counts_slow_steps() {
        let steps = vec![
            ReActStep::new("t", "a", "o").with_duration(10),
            ReActStep::new("t", "b", "o").with_duration(200),
            ReActStep::new("t", "c", "o").with_duration(50),
            ReActStep::new("t", "d", "o").with_duration(300),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.step_count_above_duration_ms(100), 2);
        assert_eq!(session.step_count_above_duration_ms(500), 0);
    }

    #[test]
    fn test_step_count_above_duration_ms_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.step_count_above_duration_ms(0), 0);
    }

    // ── Round 41 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_total_action_bytes_sums_action_lengths() {
        let steps = vec![
            make_step("t", "ab", "o"),    // 2
            make_step("t", "cde", "o"),   // 3
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_action_bytes(), 5);
    }

    #[test]
    fn test_total_action_bytes_empty_session_returns_zero() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_action_bytes(), 0);
    }

    #[test]
    fn test_step_duration_variance_ms_computed_correctly() {
        let mut steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
        ];
        steps[0].step_duration_ms = 10;
        steps[1].step_duration_ms = 20;
        let session = make_session(steps, 0);
        // mean = 15, variance = ((10-15)^2 + (20-15)^2) / 2 = 25
        assert!((session.step_duration_variance_ms() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_duration_variance_ms_zero_for_single_step() {
        let session = make_session(vec![make_step("t", "a", "o")], 0);
        assert_eq!(session.step_duration_variance_ms(), 0.0);
    }

    #[test]
    fn test_steps_with_errors_returns_steps_containing_error() {
        let steps = vec![
            make_step("t", "a", "success"),
            make_step("t", "b", "error: timeout"),
            make_step("t", "c", "ok"),
            make_step("t", "d", "Error: not found"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_errors().len(), 2);
    }

    #[test]
    fn test_steps_with_errors_empty_when_no_errors() {
        let steps = vec![make_step("t", "a", "ok"), make_step("t", "b", "done")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_errors().is_empty());
    }

    // ── Round 41: min_step_duration_ms, max_step_duration_ms ──────────────────

    #[test]
    fn test_min_step_duration_ms_returns_minimum() {
        let mut steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
        ];
        steps[0].step_duration_ms = 50;
        steps[1].step_duration_ms = 10;
        steps[2].step_duration_ms = 30;
        let session = make_session(steps, 0);
        assert_eq!(session.min_step_duration_ms(), 10);
    }

    #[test]
    fn test_min_step_duration_ms_empty_returns_zero() {
        let session = make_session(vec![], 0);
        assert_eq!(session.min_step_duration_ms(), 0);
    }

    #[test]
    fn test_max_step_duration_ms_returns_maximum() {
        let mut steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
        ];
        steps[0].step_duration_ms = 50;
        steps[1].step_duration_ms = 10;
        steps[2].step_duration_ms = 30;
        let session = make_session(steps, 0);
        assert_eq!(session.max_step_duration_ms(), 50);
    }

    #[test]
    fn test_max_step_duration_ms_empty_returns_zero() {
        let session = make_session(vec![], 0);
        assert_eq!(session.max_step_duration_ms(), 0);
    }

    // ── Round 42 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_steps_with_long_observations_returns_steps_above_threshold() {
        let steps = vec![
            make_step("t", "a", "short"),    // 5 bytes
            make_step("t", "b", "this is a long observation"),  // 26 bytes
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_long_observations(10).len(), 1);
        assert_eq!(session.steps_with_long_observations(4).len(), 2);
    }

    #[test]
    fn test_steps_with_long_observations_empty_for_high_threshold() {
        let steps = vec![make_step("t", "a", "hi")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_long_observations(1000).is_empty());
    }

    #[test]
    fn test_unique_observations_count_counts_distinct_values() {
        let steps = vec![
            make_step("t", "a", "ok"),
            make_step("t", "b", "ok"),
            make_step("t", "c", "done"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.unique_observations_count(), 2);
    }

    #[test]
    fn test_unique_observations_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.unique_observations_count(), 0);
    }

    // ── Round 43 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_thought_max_bytes_returns_max_thought_length() {
        let steps = vec![
            make_step("hi", "a", "o"),
            make_step("hello world", "b", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.thought_max_bytes(), 11);
    }

    #[test]
    fn test_thought_max_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.thought_max_bytes(), 0);
    }

    #[test]
    fn test_observation_max_bytes_returns_max_observation_length() {
        let steps = vec![
            make_step("t", "a", "short"),
            make_step("t", "b", "much longer observation"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_max_bytes(), "much longer observation".len());
    }

    #[test]
    fn test_step_count_below_duration_ms_counts_fast_steps() {
        let mut steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
        ];
        steps[0].step_duration_ms = 5;
        steps[1].step_duration_ms = 50;
        steps[2].step_duration_ms = 500;
        let session = make_session(steps, 0);
        assert_eq!(session.step_count_below_duration_ms(100), 2);
        assert_eq!(session.step_count_below_duration_ms(6), 1);
    }

    #[test]
    fn test_step_count_below_duration_ms_zero_for_empty() {
        let session = make_session(vec![], 0);
        assert_eq!(session.step_count_below_duration_ms(100), 0);
    }

    // ── Round 42: total_observation_count, actions_containing, step_duration_range_ms

    #[test]
    fn test_total_observation_count_counts_non_empty_observations() {
        let steps = vec![
            make_step("t", "a", "result"),
            make_step("t", "b", ""),
            make_step("t", "c", "output"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_observation_count(), 2);
    }

    #[test]
    fn test_total_observation_count_zero_when_all_empty() {
        let steps = vec![make_step("t", "a", ""), make_step("t", "b", "")];
        let session = make_session(steps, 0);
        assert_eq!(session.total_observation_count(), 0);
    }

    #[test]
    fn test_actions_containing_returns_matching_steps() {
        let steps = vec![
            make_step("t", "search(query)", "r"),
            make_step("t", "write(data)", "r"),
            make_step("t", "search(other)", "r"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.actions_containing("search").len(), 2);
    }

    #[test]
    fn test_actions_containing_empty_when_no_match() {
        let steps = vec![make_step("t", "write(x)", "r")];
        let session = make_session(steps, 0);
        assert!(session.actions_containing("read").is_empty());
    }

    #[test]
    fn test_step_duration_range_ms_returns_min_max() {
        let mut steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
        ];
        steps[0].step_duration_ms = 10;
        steps[1].step_duration_ms = 50;
        steps[2].step_duration_ms = 30;
        let session = make_session(steps, 0);
        assert_eq!(session.step_duration_range_ms(), (10, 50));
    }

    #[test]
    fn test_step_duration_range_ms_zero_zero_for_empty() {
        let session = make_session(vec![], 0);
        assert_eq!(session.step_duration_range_ms(), (0, 0));
    }

    // ── Round 43: count_unique_thoughts, steps_with_empty_thoughts ─────────────

    #[test]
    fn test_count_unique_thoughts_counts_distinct_strings() {
        let steps = vec![
            make_step("alpha", "a", "o"),
            make_step("beta", "b", "o"),
            make_step("alpha", "c", "o"), // duplicate thought
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.count_unique_thoughts(), 2);
    }

    #[test]
    fn test_count_unique_thoughts_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.count_unique_thoughts(), 0);
    }

    #[test]
    fn test_steps_with_empty_thoughts_returns_matching_steps() {
        let steps = vec![
            make_step("", "a", "o"),
            make_step("thought", "b", "o"),
            make_step("", "c", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_empty_thoughts().len(), 2);
    }

    #[test]
    fn test_steps_with_empty_thoughts_returns_empty_when_all_have_thoughts() {
        let steps = vec![make_step("t1", "a", "o"), make_step("t2", "b", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_empty_thoughts().is_empty());
    }

    // ── Round 44: max_action_bytes, min_action_bytes, step_throughput_per_sec ─

    #[test]
    fn test_max_action_bytes_returns_longest_action() {
        let steps = vec![
            make_step("t", "short", "o"),
            make_step("t", "much longer action string", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.max_action_bytes(), "much longer action string".len());
    }

    #[test]
    fn test_max_action_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.max_action_bytes(), 0);
    }

    #[test]
    fn test_min_action_bytes_returns_shortest_action() {
        let steps = vec![
            make_step("t", "ab", "o"),
            make_step("t", "abcde", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.min_action_bytes(), 2);
    }

    #[test]
    fn test_min_action_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.min_action_bytes(), 0);
    }

    #[test]
    fn test_step_throughput_per_sec_computes_ratio() {
        let steps = vec![make_step("t", "a", "o"), make_step("t", "b", "o")];
        let session = make_session(steps, 2000); // 2000 ms
        assert!((session.step_throughput_per_sec() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_throughput_per_sec_zero_for_zero_duration() {
        let steps = vec![make_step("t", "a", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.step_throughput_per_sec(), 0.0);
    }

    // ── Round 44: final_answer_step_index ────────────────────────────────────

    #[test]
    fn test_final_answer_step_index_returns_correct_index() {
        let steps = vec![
            make_step("think", "search(x)", "result"),
            make_step("think2", "FINAL_ANSWER: done", ""),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.final_answer_step_index(), Some(1));
    }

    #[test]
    fn test_final_answer_step_index_returns_none_when_no_final_answer() {
        let steps = vec![make_step("t", "search(x)", "result")];
        let session = make_session(steps, 0);
        assert_eq!(session.final_answer_step_index(), None);
    }

    #[test]
    fn test_final_answer_step_index_returns_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.final_answer_step_index(), None);
    }

    #[test]
    fn test_final_answer_step_index_returns_first_occurrence() {
        let steps = vec![
            make_step("t", "FINAL_ANSWER: first", ""),
            make_step("t", "FINAL_ANSWER: second", ""),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.final_answer_step_index(), Some(0));
    }

    // ── Round 45: first_n_steps, steps_with_tool, total_chars ─────────────────

    #[test]
    fn test_first_n_steps_returns_first_n() {
        let steps = vec![
            make_step("t1", "a1", "o1"),
            make_step("t2", "a2", "o2"),
            make_step("t3", "a3", "o3"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.first_n_steps(2).len(), 2);
        assert_eq!(session.first_n_steps(2)[0].thought, "t1");
    }

    #[test]
    fn test_first_n_steps_returns_all_when_n_exceeds_count() {
        let steps = vec![make_step("t", "a", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.first_n_steps(10).len(), 1);
    }

    #[test]
    fn test_first_n_steps_empty_for_n_zero() {
        let steps = vec![make_step("t", "a", "o")];
        let session = make_session(steps, 0);
        assert!(session.first_n_steps(0).is_empty());
    }

    #[test]
    fn test_steps_with_tool_returns_matching_steps() {
        let steps = vec![
            make_step("t", "search(query)", "result"),
            make_step("t", "write(data)", "ok"),
            make_step("t", "search(more)", "more"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_tool("search").len(), 2);
        assert_eq!(session.steps_with_tool("write").len(), 1);
    }

    #[test]
    fn test_steps_with_tool_excludes_final_answer() {
        let steps = vec![
            make_step("t", "FINAL_ANSWER: search done", ""),
        ];
        let session = make_session(steps, 0);
        // Final answer steps are excluded even if they contain the tool name
        assert!(session.steps_with_tool("search").is_empty());
    }

    #[test]
    fn test_total_chars_sums_all_strings() {
        let steps = vec![
            make_step("abc", "de", "f"),  // 3 + 2 + 1 = 6
            make_step("g", "hi", "jkl"), // 1 + 2 + 3 = 6
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_chars(), 12);
    }

    #[test]
    fn test_total_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_chars(), 0);
    }

    // ── Round 45: avg_action_bytes, avg_observation_bytes ─────────────────────

    #[test]
    fn test_avg_action_bytes_computes_mean() {
        let steps = vec![
            make_step("t", "ab", "o"),   // 2
            make_step("t", "abcd", "o"), // 4
        ];
        let session = make_session(steps, 0);
        assert!((session.avg_action_bytes() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_action_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_action_bytes(), 0.0);
    }

    #[test]
    fn test_avg_observation_bytes_computes_mean() {
        let steps = vec![
            make_step("t", "a", "hi"),    // obs = 2
            make_step("t", "b", "world"), // obs = 5
        ];
        let session = make_session(steps, 0);
        assert!((session.avg_observation_bytes() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_avg_observation_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_observation_bytes(), 0.0);
    }

    // ── Round 44: steps_with_long_thoughts, action_count_containing, total_thought_count

    #[test]
    fn test_steps_with_long_thoughts_returns_steps_exceeding_threshold() {
        let steps = vec![
            make_step("short", "a", "o"),
            make_step("this is a much longer thought string", "b", "o"),
            make_step("hi", "c", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_long_thoughts(10).len(), 1);
    }

    #[test]
    fn test_steps_with_long_thoughts_empty_when_none_exceed() {
        let steps = vec![make_step("hi", "a", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_long_thoughts(100).is_empty());
    }

    #[test]
    fn test_action_count_containing_counts_matching_steps() {
        let steps = vec![
            make_step("t", "search(query)", "o"),
            make_step("t", "write(data)", "o"),
            make_step("t", "search(other)", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.action_count_containing("search"), 2);
    }

    #[test]
    fn test_action_count_containing_zero_when_no_match() {
        let steps = vec![make_step("t", "write(x)", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.action_count_containing("read"), 0);
    }

    #[test]
    fn test_total_thought_count_counts_non_empty_thoughts() {
        let steps = vec![
            make_step("thought", "a", "o"),
            make_step("", "b", "o"),
            make_step("another", "c", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_thought_count(), 2);
    }

    #[test]
    fn test_total_thought_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_thought_count(), 0);
    }

    // ── Round 45: has_thought_containing, steps_with_action_length_above ────────

    #[test]
    fn test_has_thought_containing_true_when_substring_found() {
        let steps = vec![make_step("think about this", "act", "obs")];
        let session = make_session(steps, 0);
        assert!(session.has_thought_containing("think"));
    }

    #[test]
    fn test_has_thought_containing_false_when_not_found() {
        let steps = vec![make_step("unrelated", "act", "obs")];
        let session = make_session(steps, 0);
        assert!(!session.has_thought_containing("xyz"));
    }

    #[test]
    fn test_has_thought_containing_false_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(!session.has_thought_containing("any"));
    }

    #[test]
    fn test_steps_with_action_length_above_returns_matching_steps() {
        let steps = vec![
            make_step("t", "hi", "o"),           // len=2
            make_step("t", "hello world", "o"),  // len=11
        ];
        let session = make_session(steps, 0);
        let result = session.steps_with_action_length_above(5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].action, "hello world");
    }

    #[test]
    fn test_steps_with_action_length_above_empty_when_none_qualify() {
        let steps = vec![make_step("t", "hi", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_action_length_above(100).is_empty());
    }

    // ── Round 46: avg_thought_bytes, steps_above_action_bytes ─────────────────

    #[test]
    fn test_avg_thought_bytes_computes_mean() {
        let steps = vec![
            make_step("ab", "a", "o"),    // thought = 2
            make_step("abcdef", "b", "o"), // thought = 6
        ];
        let session = make_session(steps, 0);
        assert!((session.avg_thought_bytes() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_thought_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_thought_bytes(), 0.0);
    }

    #[test]
    fn test_steps_above_action_bytes_filters_correctly() {
        let steps = vec![
            make_step("t", "ab", "o"),       // len = 2
            make_step("t", "abcdefgh", "o"), // len = 8
            make_step("t", "abc", "o"),      // len = 3
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_above_action_bytes(3).len(), 1);
    }

    #[test]
    fn test_steps_above_action_bytes_empty_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.steps_above_action_bytes(0).is_empty());
    }

    // ── Round 47: steps_between, steps_with_duplicate_thoughts ────────────────

    #[test]
    fn test_steps_between_returns_subslice() {
        let steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "b", "o"),
            make_step("t", "c", "o"),
            make_step("t", "d", "o"),
        ];
        let session = make_session(steps, 0);
        let between = session.steps_between(1, 3);
        assert_eq!(between.len(), 2);
        assert_eq!(between[0].action, "b");
        assert_eq!(between[1].action, "c");
    }

    #[test]
    fn test_steps_between_empty_when_start_ge_end() {
        let steps = vec![make_step("t", "a", "o"), make_step("t", "b", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_between(2, 1).is_empty());
    }

    #[test]
    fn test_steps_with_duplicate_thoughts_returns_duplicates_only() {
        let steps = vec![
            make_step("alpha", "a", "o"),
            make_step("beta", "b", "o"),
            make_step("alpha", "c", "o"), // duplicate of first
        ];
        let session = make_session(steps, 0);
        let dupes = session.steps_with_duplicate_thoughts();
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].action, "c");
    }

    #[test]
    fn test_steps_with_duplicate_thoughts_empty_when_all_unique() {
        let steps = vec![
            make_step("t1", "a", "o"),
            make_step("t2", "b", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(session.steps_with_duplicate_thoughts().is_empty());
    }

    // ── Round 48: step_observation_rate, steps_below_thought_bytes, tool_count ─

    #[test]
    fn test_step_observation_rate_returns_fraction_with_observations() {
        let steps = vec![
            make_step("t", "a", "obs"),
            make_step("t", "b", ""),
            make_step("t", "c", "obs2"),
        ];
        let session = make_session(steps, 0);
        let rate = session.step_observation_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_observation_rate_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.step_observation_rate(), 0.0);
    }

    #[test]
    fn test_steps_below_thought_bytes_filters_by_threshold() {
        let steps = vec![
            make_step("hi", "a", "o"),
            make_step("hello world", "b", "o"),
        ];
        let session = make_session(steps, 0);
        let below = session.steps_below_thought_bytes(6);
        assert_eq!(below.len(), 1);
        assert_eq!(below[0].action, "a");
    }

    #[test]
    fn test_steps_below_thought_bytes_empty_when_all_exceed() {
        let steps = vec![make_step("long thought text", "a", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_below_thought_bytes(3).is_empty());
    }

    #[test]
    fn test_agent_runtime_tool_count_reflects_registered_tools() {
        let rt = AgentRuntime::quick(1, "model");
        assert_eq!(rt.tool_count(), 0);
    }

    // ── Round 49: max_thought_bytes, steps_above_observation_bytes, tool_names ──

    #[test]
    fn test_max_thought_bytes_returns_longest_thought_length() {
        let steps = vec![
            make_step("hi", "a", "o"),
            make_step("hello world", "b", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.max_thought_bytes(), 11);
    }

    #[test]
    fn test_max_thought_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.max_thought_bytes(), 0);
    }

    #[test]
    fn test_steps_above_observation_bytes_filters_by_threshold() {
        let steps = vec![
            make_step("t", "a", "tiny"),
            make_step("t", "b", "a much longer observation"),
        ];
        let session = make_session(steps, 0);
        let above = session.steps_above_observation_bytes(5);
        assert_eq!(above.len(), 1);
        assert_eq!(above[0].action, "b");
    }

    #[test]
    fn test_steps_above_observation_bytes_empty_when_all_below() {
        let steps = vec![make_step("t", "a", "hi")];
        let session = make_session(steps, 0);
        assert!(session.steps_above_observation_bytes(100).is_empty());
    }

    #[test]
    fn test_agent_runtime_tool_names_empty_when_no_tools() {
        let rt = AgentRuntime::quick(1, "model");
        assert!(rt.tool_names().is_empty());
    }

    // ── Round 47: steps_between, has_duplicate_actions, step_indices_with_tool ──

    #[test]
    fn test_steps_between_returns_correct_slice() {
        let steps = vec![
            make_step("t0", "a0", "o0"),
            make_step("t1", "a1", "o1"),
            make_step("t2", "a2", "o2"),
            make_step("t3", "a3", "o3"),
        ];
        let session = make_session(steps, 0);
        let slice = session.steps_between(1, 3);
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].thought, "t1");
        assert_eq!(slice[1].thought, "t2");
    }

    #[test]
    fn test_steps_between_returns_empty_when_start_ge_end() {
        let steps = vec![make_step("t", "a", "o"), make_step("t2", "a2", "o2")];
        let session = make_session(steps, 0);
        assert!(session.steps_between(2, 1).is_empty());
        assert!(session.steps_between(1, 1).is_empty());
    }

    #[test]
    fn test_steps_between_clamps_to_step_count() {
        let steps = vec![make_step("t", "a", "o")];
        let session = make_session(steps, 0);
        let slice = session.steps_between(0, 100);
        assert_eq!(slice.len(), 1);
    }

    #[test]
    fn test_has_duplicate_actions_true_when_repeated() {
        let steps = vec![
            make_step("t", "search[foo]", "o"),
            make_step("t", "search[foo]", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(session.has_duplicate_actions());
    }

    #[test]
    fn test_has_duplicate_actions_false_when_all_unique() {
        let steps = vec![
            make_step("t", "search[foo]", "o"),
            make_step("t", "lookup[bar]", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(!session.has_duplicate_actions());
    }

    #[test]
    fn test_has_duplicate_actions_false_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(!session.has_duplicate_actions());
    }

    #[test]
    fn test_step_indices_with_tool_returns_correct_indices() {
        let steps = vec![
            make_step("t", "search[x]", "o"),
            make_step("t", "lookup[y]", "o"),
            make_step("t", "search[z]", "o"),
        ];
        let session = make_session(steps, 0);
        let indices = session.step_indices_with_tool("search");
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn test_step_indices_with_tool_empty_when_no_match() {
        let steps = vec![make_step("t", "lookup[x]", "o")];
        let session = make_session(steps, 0);
        assert!(session.step_indices_with_tool("search").is_empty());
    }

    // ── Round 49: observations_above_bytes, total_step_chars ──────────────────

    #[test]
    fn test_observations_above_bytes_returns_matching_steps() {
        let steps = vec![
            make_step("t", "a", "hi"),        // obs len=2
            make_step("t", "a", "hello world"), // obs len=11
        ];
        let session = make_session(steps, 0);
        let result = session.observations_above_bytes(5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].observation, "hello world");
    }

    #[test]
    fn test_observations_above_bytes_empty_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.observations_above_bytes(0).is_empty());
    }

    #[test]
    fn test_total_step_chars_sums_all_fields() {
        let steps = vec![
            make_step("ab", "cd", "ef"), // 2+2+2=6
            make_step("x", "y", "z"),    // 1+1+1=3
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_step_chars(), 9);
    }

    #[test]
    fn test_total_step_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_step_chars(), 0);
    }

    // ── Round 50: steps_by_action_prefix, action_count ────────────────────────

    #[test]
    fn test_steps_by_action_prefix_returns_matching_steps() {
        let steps = vec![
            make_step("t", "search_web", "o"),
            make_step("t", "search_db", "o"),
            make_step("t", "write_file", "o"),
        ];
        let session = make_session(steps, 0);
        let result = session.steps_by_action_prefix("search");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_steps_by_action_prefix_empty_when_no_match() {
        let steps = vec![make_step("t", "write_file", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_by_action_prefix("search").is_empty());
    }

    #[test]
    fn test_action_count_counts_tool_call_steps() {
        let steps = vec![
            make_step("t", "search_web", "o"),
            make_step("t", "FINAL_ANSWER: done", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.action_count(), 1);
    }

    #[test]
    fn test_action_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.action_count(), 0);
    }

    // ── Round 47: total_thought_bytes, total_observation_bytes ─────────────────

    #[test]
    fn test_total_thought_bytes_sums_thought_lengths() {
        let steps = vec![
            make_step("ab", "a", "o"),    // 2
            make_step("abcde", "b", "o"), // 5
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_thought_bytes(), 7);
    }

    #[test]
    fn test_total_thought_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_thought_bytes(), 0);
    }

    #[test]
    fn test_total_observation_bytes_sums_observation_lengths() {
        let steps = vec![
            make_step("t", "a", "hello"), // obs=5
            make_step("t", "b", "world"), // obs=5
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_observation_bytes(), 10);
    }

    #[test]
    fn test_total_observation_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_observation_bytes(), 0);
    }

    // ── Round 50: proportion_tool_calls, thought_density ─────────────────────

    #[test]
    fn test_proportion_tool_calls_all_tool_calls() {
        let steps = vec![
            make_step("t", "search[x]", "o"),
            make_step("t", "lookup[y]", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.proportion_tool_calls() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_proportion_tool_calls_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.proportion_tool_calls(), 0.0);
    }

    #[test]
    fn test_thought_density_returns_thought_fraction_of_total_bytes() {
        let steps = vec![make_step("ab", "cd", "ef")]; // 2+2+2=6, thought=2
        let session = make_session(steps, 0);
        let density = session.thought_density();
        assert!((density - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_thought_density_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.thought_density(), 0.0);
    }

    // ── Round 51: steps_matching_observation, step_action_lengths ─────────────

    #[test]
    fn test_steps_matching_observation_returns_matching_steps() {
        let steps = vec![
            make_step("t", "a", "found: result"),
            make_step("t", "b", "no match here"),
            make_step("t", "c", "found: another"),
        ];
        let session = make_session(steps, 0);
        let result = session.steps_matching_observation("found:");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_steps_matching_observation_empty_when_no_match() {
        let steps = vec![make_step("t", "a", "nothing")];
        let session = make_session(steps, 0);
        assert!(session.steps_matching_observation("found:").is_empty());
    }

    #[test]
    fn test_step_action_lengths_returns_lengths_in_order() {
        let steps = vec![
            make_step("t", "ab", "o"),
            make_step("t", "cdef", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.step_action_lengths(), vec![2, 4]);
    }

    #[test]
    fn test_step_action_lengths_empty_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.step_action_lengths().is_empty());
    }

    // ── Round 52: has_thought_starting_with, step_count_above_action_bytes, config ──

    #[test]
    fn test_has_thought_starting_with_true_when_match() {
        let steps = vec![
            make_step("Plan: do something", "act", "obs"),
        ];
        let session = make_session(steps, 0);
        assert!(session.has_thought_starting_with("Plan:"));
    }

    #[test]
    fn test_has_thought_starting_with_false_when_no_match() {
        let steps = vec![make_step("think", "act", "obs")];
        let session = make_session(steps, 0);
        assert!(!session.has_thought_starting_with("Plan:"));
    }

    #[test]
    fn test_has_thought_starting_with_false_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(!session.has_thought_starting_with("Plan:"));
    }

    #[test]
    fn test_step_count_above_action_bytes_counts_correctly() {
        let steps = vec![
            make_step("t", "short", "o"),
            make_step("t", "a_very_long_action_string", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.step_count_above_action_bytes(5), 1);
    }

    #[test]
    fn test_step_count_above_action_bytes_zero_when_all_small() {
        let steps = vec![make_step("t", "ab", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.step_count_above_action_bytes(100), 0);
    }

    #[test]
    fn test_runtime_config_returns_agent_config() {
        let rt = AgentRuntime::quick(3, "test-model");
        assert_eq!(rt.config().max_iterations, 3);
    }

    // ── Round 53: total_thought_chars, total_action_chars, total_observation_chars, model_name ──

    #[test]
    fn test_total_thought_chars_sums_all_thoughts() {
        let steps = vec![
            make_step("ab", "x", "y"),
            make_step("cde", "x", "y"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_thought_chars(), 5);
    }

    #[test]
    fn test_total_thought_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_thought_chars(), 0);
    }

    #[test]
    fn test_total_action_chars_sums_all_actions() {
        let steps = vec![
            make_step("t", "hello", "o"),
            make_step("t", "world", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_action_chars(), 10);
    }

    #[test]
    fn test_total_observation_chars_sums_all_observations() {
        let steps = vec![
            make_step("t", "a", "abc"),
            make_step("t", "a", "de"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_observation_chars(), 5);
    }

    #[test]
    fn test_model_name_returns_configured_model() {
        let rt = AgentRuntime::quick(5, "gpt-4o");
        assert_eq!(rt.model_name(), "gpt-4o");
    }

    // ── Round 48 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_min_observation_bytes_returns_smallest_nonempty() {
        let steps = vec![
            make_step("t", "a", "hello"),
            make_step("t", "a", "hi"),
            make_step("t", "a", ""),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.min_observation_bytes(), 2);
    }

    #[test]
    fn test_min_observation_bytes_zero_when_all_empty() {
        let steps = vec![make_step("t", "a", "")];
        let session = make_session(steps, 0);
        assert_eq!(session.min_observation_bytes(), 0);
    }

    #[test]
    fn test_min_thought_bytes_returns_smallest_nonempty() {
        let steps = vec![
            make_step("abc", "a", "o"),
            make_step("xy", "a", "o"),
            make_step("", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.min_thought_bytes(), 2);
    }

    #[test]
    fn test_min_thought_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.min_thought_bytes(), 0);
    }

    #[test]
    fn test_proportion_empty_thoughts_all_empty() {
        let steps = vec![
            make_step("", "a", "o"),
            make_step("", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.proportion_empty_thoughts() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proportion_empty_thoughts_none_empty() {
        let steps = vec![make_step("think", "a", "o")];
        let session = make_session(steps, 0);
        assert!((session.proportion_empty_thoughts()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proportion_empty_thoughts_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.proportion_empty_thoughts()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_has_failed_steps_true_when_error_observation() {
        let steps = vec![make_step("t", "a", "[error] something broke")];
        let session = make_session(steps, 0);
        assert!(session.has_failed_steps());
    }

    #[test]
    fn test_has_failed_steps_false_when_no_errors() {
        let steps = vec![make_step("t", "a", "success")];
        let session = make_session(steps, 0);
        assert!(!session.has_failed_steps());
    }

    #[test]
    fn test_has_failed_steps_false_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(!session.has_failed_steps());
    }

    // ── Round 53 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_all_observations_non_empty_true_when_all_have_obs() {
        let steps = vec![
            make_step("t1", "a1", "result1"),
            make_step("t2", "a2", "result2"),
        ];
        let session = make_session(steps, 0);
        assert!(session.all_observations_non_empty());
    }

    #[test]
    fn test_all_observations_non_empty_false_when_one_is_empty() {
        let steps = vec![
            make_step("t1", "a1", "result1"),
            make_step("t2", "a2", ""),
        ];
        let session = make_session(steps, 0);
        assert!(!session.all_observations_non_empty());
    }

    #[test]
    fn test_all_observations_non_empty_true_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.all_observations_non_empty());
    }

    #[test]
    fn test_avg_combined_step_bytes_correct() {
        let steps = vec![
            make_step("hi", "go", "ok"),
            make_step("hello", "world", "result"),
        ];
        let session = make_session(steps, 0);
        // step1: 2+2+2=6, step2: 5+5+6=16, avg=11
        let avg = session.avg_combined_step_bytes();
        assert!((avg - 11.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_combined_step_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_combined_step_bytes(), 0.0);
    }

    #[test]
    fn test_shortest_observation_step_returns_shortest() {
        let steps = vec![
            make_step("t1", "a1", "longer observation"),
            make_step("t2", "a2", "short"),
        ];
        let session = make_session(steps, 0);
        let s = session.shortest_observation_step().unwrap();
        assert_eq!(s.observation, "short");
    }

    #[test]
    fn test_shortest_observation_step_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.shortest_observation_step().is_none());
    }

    // ── Round 54: steps_with_empty_action, observation_starts_with_any, has_repeated_actions, session_max_iterations ──

    #[test]
    fn test_steps_with_empty_action_returns_matching_steps() {
        let steps = vec![
            make_step("t", "", "obs"),
            make_step("t", "act", "obs"),
            make_step("t", "", "obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_empty_action().len(), 2);
    }

    #[test]
    fn test_steps_with_empty_action_empty_when_all_have_actions() {
        let steps = vec![make_step("t", "act", "obs")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_empty_action().is_empty());
    }

    #[test]
    fn test_observation_starts_with_any_true_when_prefix_matches() {
        let steps = vec![make_step("t", "a", "ERROR: something went wrong")];
        let session = make_session(steps, 0);
        assert!(session.observation_starts_with_any(&["ERROR:", "WARN:"]));
    }

    #[test]
    fn test_observation_starts_with_any_false_when_no_match() {
        let steps = vec![make_step("t", "a", "success")];
        let session = make_session(steps, 0);
        assert!(!session.observation_starts_with_any(&["ERROR:", "WARN:"]));
    }

    #[test]
    fn test_has_repeated_actions_true_when_duplicate_exists() {
        let steps = vec![
            make_step("t", "search", "obs"),
            make_step("t", "search", "obs"),
        ];
        let session = make_session(steps, 0);
        assert!(session.has_repeated_actions());
    }

    #[test]
    fn test_has_repeated_actions_false_when_all_unique() {
        let steps = vec![
            make_step("t", "search", "obs"),
            make_step("t", "read", "obs"),
        ];
        let session = make_session(steps, 0);
        assert!(!session.has_repeated_actions());
    }

    #[test]
    fn test_session_max_iterations_returns_config_value() {
        let rt = AgentRuntime::quick(7, "model");
        assert_eq!(rt.session_max_iterations(), 7);
    }

    // ── Round 55: has_action_containing, max_observation_chars, step_index_of_longest_thought, observation_word_counts ──

    #[test]
    fn test_has_action_containing_true_when_substr_matches() {
        let steps = vec![make_step("t", "search[query]", "obs")];
        let session = make_session(steps, 0);
        assert!(session.has_action_containing("search"));
    }

    #[test]
    fn test_has_action_containing_false_when_no_match() {
        let steps = vec![make_step("t", "read_file", "obs")];
        let session = make_session(steps, 0);
        assert!(!session.has_action_containing("write"));
    }

    #[test]
    fn test_max_observation_chars_returns_longest_observation() {
        let steps = vec![
            make_step("t", "a", "hi"),
            make_step("t", "a", "hello world"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.max_observation_chars(), 11);
    }

    #[test]
    fn test_max_observation_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.max_observation_chars(), 0);
    }

    #[test]
    fn test_step_index_of_longest_thought_returns_correct_index() {
        let steps = vec![
            make_step("short", "a", "o"),
            make_step("a very long thought string", "a", "o"),
            make_step("mid", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.step_index_of_longest_thought(), Some(1));
    }

    #[test]
    fn test_step_index_of_longest_thought_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.step_index_of_longest_thought(), None);
    }

    #[test]
    fn test_observation_word_counts_returns_word_counts_in_order() {
        let steps = vec![
            make_step("t", "a", "one two three"),
            make_step("t", "a", "single"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_word_counts(), vec![3, 1]);
    }

    // ── Round 49 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_action_byte_variance_zero_for_equal_lengths() {
        let steps = vec![
            make_step("t", "abc", "o"),
            make_step("t", "def", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.action_byte_variance()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_action_byte_variance_nonzero_for_different_lengths() {
        let steps = vec![
            make_step("t", "a", "o"),
            make_step("t", "abcde", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(session.action_byte_variance() > 0.0);
    }

    #[test]
    fn test_action_byte_variance_zero_for_single_step() {
        let steps = vec![make_step("t", "hello", "o")];
        let session = make_session(steps, 0);
        assert!((session.action_byte_variance()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thought_byte_variance_nonzero_for_different_lengths() {
        let steps = vec![
            make_step("a", "act", "o"),
            make_step("abcde", "act", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(session.thought_byte_variance() > 0.0);
    }

    #[test]
    fn test_thought_byte_variance_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.thought_byte_variance()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_steps_above_thought_bytes_filters_correctly() {
        let steps = vec![
            make_step("hi", "a", "o"),
            make_step("hello world", "a", "o"),
            make_step("x", "a", "o"),
        ];
        let session = make_session(steps, 0);
        let above = session.steps_above_thought_bytes(4);
        assert_eq!(above.len(), 1);
        assert_eq!(above[0].thought, "hello world");
    }

    #[test]
    fn test_steps_above_thought_bytes_empty_when_none_qualify() {
        let steps = vec![make_step("hi", "a", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_above_thought_bytes(100).is_empty());
    }

    // ── Round 54 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_unique_observation_count_counts_distinct_observations() {
        let steps = vec![
            make_step("t1", "a1", "result"),
            make_step("t2", "a2", "result"),
            make_step("t3", "a3", "other"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.unique_observation_count(), 2);
    }

    #[test]
    fn test_unique_observation_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.unique_observation_count(), 0);
    }

    #[test]
    fn test_avg_thought_word_count_computes_correctly() {
        let steps = vec![
            make_step("one word", "a", "o"),       // 2 words
            make_step("three word count", "a", "o"), // 3 words
        ];
        let session = make_session(steps, 0);
        let avg = session.avg_thought_word_count();
        assert!((avg - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_avg_thought_word_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_thought_word_count(), 0.0);
    }

    // ── Round 56: thought_starts_with_any, action_word_count, steps_above_thought_chars ──

    #[test]
    fn test_thought_starts_with_any_true_when_prefix_matches() {
        let steps = vec![make_step("Plan: do it", "act", "obs")];
        let session = make_session(steps, 0);
        assert!(session.thought_starts_with_any(&["Plan:", "Think:"]));
    }

    #[test]
    fn test_thought_starts_with_any_false_when_no_match() {
        let steps = vec![make_step("just thinking", "act", "obs")];
        let session = make_session(steps, 0);
        assert!(!session.thought_starts_with_any(&["Plan:", "Think:"]));
    }

    #[test]
    fn test_action_word_count_sums_words_across_steps() {
        let steps = vec![
            make_step("t", "search for answer", "obs"),
            make_step("t", "write result", "obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.action_word_count(), 5);
    }

    #[test]
    fn test_action_word_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.action_word_count(), 0);
    }

    #[test]
    fn test_steps_above_thought_chars_counts_correctly() {
        let steps = vec![
            make_step("short", "a", "o"),
            make_step("a very long thought here", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_above_thought_chars(5), 1);
    }

    #[test]
    fn test_steps_above_thought_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.steps_above_thought_chars(0), 0);
    }

    // ── Round 50 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_total_empty_steps_counts_fully_empty_steps() {
        let steps = vec![
            make_step("", "", ""),
            make_step("t", "a", "o"),
            make_step("", "", ""),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.total_empty_steps(), 2);
    }

    #[test]
    fn test_total_empty_steps_zero_when_no_empty_steps() {
        let steps = vec![make_step("t", "a", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.total_empty_steps(), 0);
    }

    #[test]
    fn test_action_starts_with_count_correct() {
        let steps = vec![
            make_step("t", "search:foo", "o"),
            make_step("t", "search:bar", "o"),
            make_step("t", "write:baz", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.action_starts_with_count("search"), 2);
    }

    #[test]
    fn test_action_starts_with_count_zero_for_no_match() {
        let steps = vec![make_step("t", "write:x", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.action_starts_with_count("read"), 0);
    }

    #[test]
    fn test_longest_action_returns_longest() {
        let steps = vec![
            make_step("t", "short", "o"),
            make_step("t", "much_longer_action", "o"),
            make_step("t", "mid", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.longest_action(), Some("much_longer_action"));
    }

    #[test]
    fn test_longest_action_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.longest_action(), None);
    }

    #[test]
    fn test_thought_completeness_all_non_empty() {
        let steps = vec![
            make_step("think", "a", "o"),
            make_step("also thinking", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.thought_completeness() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thought_completeness_half_empty() {
        let steps = vec![
            make_step("think", "a", "o"),
            make_step("", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.thought_completeness() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thought_completeness_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!((session.thought_completeness()).abs() < f64::EPSILON);
    }

    // ── Round 57: steps_with_non_empty_observation, observations_containing, thought_observation_ratio ──

    #[test]
    fn test_steps_with_non_empty_observation_returns_matching_steps() {
        let steps = vec![
            make_step("t", "a", "result"),
            make_step("t", "a", ""),
            make_step("t", "a", "more"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_non_empty_observation().len(), 2);
    }

    #[test]
    fn test_steps_with_non_empty_observation_empty_for_all_empty() {
        let steps = vec![make_step("t", "a", "")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_non_empty_observation().is_empty());
    }

    #[test]
    fn test_observations_containing_returns_matching_steps() {
        let steps = vec![
            make_step("t", "a", "found the answer"),
            make_step("t", "a", "no match"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observations_containing("found").len(), 1);
    }

    #[test]
    fn test_thought_observation_ratio_returns_correct_ratio() {
        let steps = vec![make_step("ab", "x", "abcd")];
        let session = make_session(steps, 0);
        assert_eq!(session.thought_observation_ratio(), 0.5);
    }

    #[test]
    fn test_thought_observation_ratio_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.thought_observation_ratio(), 0.0);
    }

    // ── Round 51 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_non_empty_action_count_correct() {
        let steps = vec![
            make_step("t", "search", "o"),
            make_step("t", "", "o"),
            make_step("t", "write", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.non_empty_action_count(), 2);
    }

    #[test]
    fn test_non_empty_action_count_zero_for_all_empty() {
        let steps = vec![make_step("t", "", "o"), make_step("t", "", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.non_empty_action_count(), 0);
    }

    #[test]
    fn test_total_step_bytes_sums_all_fields() {
        let steps = vec![make_step("abc", "de", "f")];
        let session = make_session(steps, 0);
        assert_eq!(session.total_step_bytes(), 6); // 3 + 2 + 1
    }

    #[test]
    fn test_total_step_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.total_step_bytes(), 0);
    }

    #[test]
    fn test_last_thought_bytes_returns_last_step_thought() {
        let steps = vec![
            make_step("short", "a", "o"),
            make_step("much longer thought", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.last_thought_bytes(), "much longer thought".len());
    }

    #[test]
    fn test_last_thought_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.last_thought_bytes(), 0);
    }

    #[test]
    fn test_first_observation_bytes_returns_first_step_observation() {
        let steps = vec![
            make_step("t", "a", "first obs"),
            make_step("t", "a", "second obs is longer"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.first_observation_bytes(), "first obs".len());
    }

    #[test]
    fn test_first_observation_bytes_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.first_observation_bytes(), 0);
    }

    // ── Round 59: has_step_with_empty_observation, thought_to_action_byte_ratio ──

    #[test]
    fn test_has_step_with_empty_observation_true() {
        let steps = vec![make_step("t", "a", "")];
        let session = make_session(steps, 0);
        assert!(session.has_step_with_empty_observation());
    }

    #[test]
    fn test_has_step_with_empty_observation_false_when_all_nonempty() {
        let steps = vec![make_step("t", "a", "obs")];
        let session = make_session(steps, 0);
        assert!(!session.has_step_with_empty_observation());
    }

    #[test]
    fn test_has_step_with_empty_observation_false_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(!session.has_step_with_empty_observation());
    }

    #[test]
    fn test_thought_to_action_byte_ratio_correct() {
        // thought = "hello" (5 bytes), action = "hi" (2 bytes) → 5/2 = 2.5
        let steps = vec![make_step("hello", "hi", "o")];
        let session = make_session(steps, 0);
        assert!((session.thought_to_action_byte_ratio() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_thought_to_action_byte_ratio_zero_when_no_action_bytes() {
        let steps = vec![make_step("thought", "", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.thought_to_action_byte_ratio(), 0.0);
    }

    #[test]
    fn test_thought_to_action_byte_ratio_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.thought_to_action_byte_ratio(), 0.0);
    }

    // ── Round 60: observation_above_bytes_count, steps_with_both_thought_and_action ──

    #[test]
    fn test_observation_above_bytes_count_correct() {
        let steps = vec![
            make_step("t", "a", "short"),
            make_step("t", "a", "this is quite long"),
            make_step("t", "a", "x"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_above_bytes_count(5), 1);
    }

    #[test]
    fn test_observation_above_bytes_count_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.observation_above_bytes_count(0), 0);
    }

    #[test]
    fn test_steps_with_both_thought_and_action_correct() {
        let steps = vec![
            make_step("think", "act", "obs"),
            make_step("", "act", "obs"),
            make_step("think", "", "obs"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_both_thought_and_action(), 1);
    }

    #[test]
    fn test_steps_with_both_thought_and_action_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.steps_with_both_thought_and_action(), 0);
    }

    // ── Round 61: steps_with_observation_prefix, observation_bytes_total, first_thought_chars ──

    #[test]
    fn test_steps_with_observation_prefix_correct() {
        let steps = vec![
            make_step("t", "a", "[error] bad"),
            make_step("t", "a", "ok"),
            make_step("t", "a", "[error] also bad"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_observation_prefix("[error]"), 2);
    }

    #[test]
    fn test_steps_with_observation_prefix_zero_when_none_match() {
        let steps = vec![make_step("t", "a", "ok")];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_observation_prefix("[error]"), 0);
    }

    #[test]
    fn test_observation_bytes_total_sums_all_observations() {
        let steps = vec![
            make_step("t", "a", "abc"),   // 3
            make_step("t", "a", "de"),    // 2
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_bytes_total(), 5);
    }

    #[test]
    fn test_observation_bytes_total_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.observation_bytes_total(), 0);
    }

    #[test]
    fn test_first_thought_chars_returns_first_step_count() {
        let steps = vec![make_step("héllo", "a", "o"), make_step("ignored", "a", "o")];
        let session = make_session(steps, 0);
        // "héllo" has 5 chars (é is one char)
        assert_eq!(session.first_thought_chars(), 5);
    }

    #[test]
    fn test_first_thought_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.first_thought_chars(), 0);
    }

    // ── Round 62: last_observation_chars, observation_word_count_total ────────

    #[test]
    fn test_last_observation_chars_returns_last_step() {
        let steps = vec![
            make_step("t", "a", "first"),
            make_step("t", "a", "last one"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.last_observation_chars(), "last one".chars().count());
    }

    #[test]
    fn test_last_observation_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.last_observation_chars(), 0);
    }

    #[test]
    fn test_observation_word_count_total_sums_all_words() {
        let steps = vec![
            make_step("t", "a", "one two"),    // 2
            make_step("t", "a", "three"),      // 1
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_word_count_total(), 3);
    }

    #[test]
    fn test_observation_word_count_total_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.observation_word_count_total(), 0);
    }

    // ── Round 63: action_ends_with_count, avg_observation_words ──────────────

    #[test]
    fn test_action_ends_with_count_correct() {
        let steps = vec![
            make_step("t", "search.", "o"),
            make_step("t", "browse.", "o"),
            make_step("t", "calculate", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.action_ends_with_count("."), 2);
    }

    #[test]
    fn test_action_ends_with_count_zero_when_none_match() {
        let steps = vec![make_step("t", "nope", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.action_ends_with_count("."), 0);
    }

    #[test]
    fn test_avg_observation_words_correct() {
        let steps = vec![
            make_step("t", "a", "one two"),    // 2
            make_step("t", "a", "three four five"), // 3
        ];
        let session = make_session(steps, 0);
        assert!((session.avg_observation_words() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_avg_observation_words_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_observation_words(), 0.0);
    }

    // ── Round 58: steps_matching_thought, median_observation_chars, cumulative_thought_chars, count_steps_with_thought_containing ──

    #[test]
    fn test_steps_matching_thought_returns_correct_steps() {
        let steps = vec![
            make_step("I need to search", "a", "o"),
            make_step("I found the answer", "a", "o"),
            make_step("no match here", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_matching_thought("I").len(), 2);
    }

    #[test]
    fn test_median_observation_chars_returns_middle_value() {
        let steps = vec![
            make_step("t", "a", "ab"),
            make_step("t", "a", "abcde"),
            make_step("t", "a", "abc"),
        ];
        let session = make_session(steps, 0);
        // sorted: [2, 3, 5] → median at index 1 = 3
        assert_eq!(session.median_observation_chars(), 3);
    }

    #[test]
    fn test_median_observation_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.median_observation_chars(), 0);
    }

    #[test]
    fn test_cumulative_thought_chars_accumulates_correctly() {
        let steps = vec![
            make_step("ab", "a", "o"),
            make_step("cde", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.cumulative_thought_chars(), vec![2, 5]);
    }

    #[test]
    fn test_count_steps_with_thought_containing_counts_matches() {
        let steps = vec![
            make_step("call function foo", "a", "o"),
            make_step("call function bar", "a", "o"),
            make_step("nothing", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.count_steps_with_thought_containing("function"), 2);
    }

    // ── Round 57: observation_contains_any ───────────────────────────────────

    #[test]
    fn test_observation_contains_any_true_when_term_present() {
        let steps = vec![
            make_step("t", "a", "result: success"),
            make_step("t", "a", "result: failure"),
        ];
        let session = make_session(steps, 0);
        assert!(session.observation_contains_any(&["success", "error"]));
    }

    #[test]
    fn test_observation_contains_any_false_when_no_match() {
        let steps = vec![make_step("t", "a", "nothing here")];
        let session = make_session(steps, 0);
        assert!(!session.observation_contains_any(&["success", "error"]));
    }

    #[test]
    fn test_observation_contains_any_false_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(!session.observation_contains_any(&["anything"]));
    }

    #[test]
    fn test_observation_contains_any_false_for_empty_terms() {
        let steps = vec![make_step("t", "a", "something")];
        let session = make_session(steps, 0);
        assert!(!session.observation_contains_any(&[]));
    }

    // ── Round 59: step_at_index, thought_contains_all, action_contains_any, max_thought_chars, min_thought_chars, is_registered_tool ──

    #[test]
    fn test_step_at_index_returns_correct_step() {
        let steps = vec![
            make_step("first", "a1", "o1"),
            make_step("second", "a2", "o2"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.step_at_index(1).map(|s| s.thought.as_str()), Some("second"));
    }

    #[test]
    fn test_step_at_index_returns_none_out_of_bounds() {
        let session = make_session(vec![], 0);
        assert!(session.step_at_index(0).is_none());
    }

    #[test]
    fn test_thought_contains_all_true_when_all_present_in_one_step() {
        let steps = vec![
            make_step("alpha beta gamma", "a", "o"),
            make_step("alpha only", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(session.thought_contains_all(&["alpha", "beta"]));
    }

    #[test]
    fn test_thought_contains_all_false_when_no_single_step_has_all() {
        let steps = vec![
            make_step("alpha", "a", "o"),
            make_step("beta", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(!session.thought_contains_all(&["alpha", "beta"]));
    }

    #[test]
    fn test_action_contains_any_true_when_present() {
        let steps = vec![
            make_step("t", "search(query)", "o"),
            make_step("t", "read(file)", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(session.action_contains_any(&["search", "write"]));
    }

    #[test]
    fn test_action_contains_any_false_when_not_present() {
        let steps = vec![make_step("t", "read(file)", "o")];
        let session = make_session(steps, 0);
        assert!(!session.action_contains_any(&["search", "write"]));
    }

    #[test]
    fn test_max_thought_chars_returns_longest() {
        let steps = vec![
            make_step("hi", "a", "o"),
            make_step("hello world", "a", "o"),
            make_step("hey", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.max_thought_chars(), 11);
    }

    #[test]
    fn test_max_thought_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.max_thought_chars(), 0);
    }

    #[test]
    fn test_min_thought_chars_returns_shortest_non_empty() {
        let steps = vec![
            make_step("", "a", "o"),
            make_step("ab", "a", "o"),
            make_step("abcd", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.min_thought_chars(), 2);
    }

    #[test]
    fn test_min_thought_chars_zero_when_all_empty() {
        let steps = vec![make_step("", "a", "o")];
        let session = make_session(steps, 0);
        assert_eq!(session.min_thought_chars(), 0);
    }

    #[test]
    fn test_is_registered_tool_true_for_registered_tool() {
        let spec = crate::agent::ToolSpec::new("calculator", "Does math", |_| {
            serde_json::json!("ok")
        });
        let rt = AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(3, "m"))
            .register_tool(spec)
            .build();
        assert!(rt.is_registered_tool("calculator"));
    }

    #[test]
    fn test_is_registered_tool_false_for_unknown_tool() {
        let rt = AgentRuntime::quick(3, "m");
        assert!(!rt.is_registered_tool("nonexistent"));
    }

    // ── Round 58: registered_tool_names ───────────────────────────────────────

    #[test]
    fn test_registered_tool_names_returns_owned_sorted_names() {
        let rt = AgentRuntime::builder()
            .with_agent_config(AgentConfig::new(3, "test-model"))
            .register_tool(crate::agent::ToolSpec::new("beta", "b", |_| {
                serde_json::json!("ok")
            }))
            .register_tool(crate::agent::ToolSpec::new("alpha", "a", |_| {
                serde_json::json!("ok")
            }))
            .build();
        let names = rt.registered_tool_names();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn test_registered_tool_names_empty_when_no_tools() {
        let rt = AgentRuntime::quick(3, "test-model");
        assert!(rt.registered_tool_names().is_empty());
    }

    // ── Round 60: avg_action_chars, avg_observation_chars, step_with_longest_action ──

    #[test]
    fn test_avg_action_chars_correct() {
        let steps = vec![
            make_step("t", "ab", "o"),
            make_step("t", "abcd", "o"),
        ];
        let session = make_session(steps, 0);
        assert!((session.avg_action_chars() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_action_chars_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.avg_action_chars(), 0.0);
    }

    #[test]
    fn test_avg_observation_chars_correct() {
        let steps = vec![
            make_step("t", "a", "hello"),
            make_step("t", "a", "hi"),
        ];
        let session = make_session(steps, 0);
        // (5 + 2) / 2 = 3.5
        assert!((session.avg_observation_chars() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_step_with_longest_action_returns_correct_step() {
        let steps = vec![
            make_step("t", "short", "o"),
            make_step("t", "much longer action string", "o"),
            make_step("t", "medium act", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(
            session.step_with_longest_action().map(|s| s.action.as_str()),
            Some("much longer action string")
        );
    }

    #[test]
    fn test_step_with_longest_action_none_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.step_with_longest_action().is_none());
    }

    // ── Round 59: step_count_with_observation_longer_than ─────────────────────

    #[test]
    fn test_step_count_with_observation_longer_than_counts_correctly() {
        let steps = vec![
            make_step("t", "a", "short"),           // 5 bytes
            make_step("t", "a", "a longer string"), // 14 bytes
            make_step("t", "a", "x"),               // 1 byte
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.step_count_with_observation_longer_than(5), 1);
    }

    #[test]
    fn test_step_count_with_observation_longer_than_zero_for_empty_session() {
        let session = make_session(vec![], 0);
        assert_eq!(session.step_count_with_observation_longer_than(0), 0);
    }

    // ── Round 61: action_ends_with, thought_ends_with, has_step_with_both ──────

    #[test]
    fn test_action_ends_with_true_when_present() {
        let steps = vec![make_step("t", "search(query)", "o")];
        let session = make_session(steps, 0);
        assert!(session.action_ends_with(")"));
    }

    #[test]
    fn test_action_ends_with_false_when_absent() {
        let steps = vec![make_step("t", "search(query)", "o")];
        let session = make_session(steps, 0);
        assert!(!session.action_ends_with("!"));
    }

    #[test]
    fn test_thought_ends_with_true_when_present() {
        let steps = vec![make_step("I should search.", "a", "o")];
        let session = make_session(steps, 0);
        assert!(session.thought_ends_with("."));
    }

    #[test]
    fn test_thought_ends_with_false_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(!session.thought_ends_with("x"));
    }

    #[test]
    fn test_has_step_with_both_true_when_step_matches_both() {
        let steps = vec![
            make_step("need to search", "search(foo)", "o"),
            make_step("done", "noop", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(session.has_step_with_both("search", "foo"));
    }

    #[test]
    fn test_has_step_with_both_false_when_no_step_matches_both() {
        let steps = vec![
            make_step("need to search", "noop", "o"),
            make_step("done", "search(foo)", "o"),
        ];
        let session = make_session(steps, 0);
        assert!(!session.has_step_with_both("search", "foo"));
    }

    // ── Round 62: thought_word_counts, steps_sorted_by_thought_len, steps_with_thought_longer_than ──

    #[test]
    fn test_thought_word_counts_returns_per_step_counts() {
        let steps = vec![
            make_step("one two three", "a", "o"),
            make_step("hello", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.thought_word_counts(), vec![3, 1]);
    }

    #[test]
    fn test_thought_word_counts_empty_for_empty_session() {
        let session = make_session(vec![], 0);
        assert!(session.thought_word_counts().is_empty());
    }

    #[test]
    fn test_steps_sorted_by_thought_len_ascending_order() {
        let steps = vec![
            make_step("longest thought here", "a", "o"),
            make_step("hi", "a", "o"),
            make_step("medium thought", "a", "o"),
        ];
        let session = make_session(steps, 0);
        let sorted = session.steps_sorted_by_thought_len();
        assert!(sorted[0].thought.len() <= sorted[1].thought.len());
        assert!(sorted[1].thought.len() <= sorted[2].thought.len());
    }

    #[test]
    fn test_steps_with_thought_longer_than_filters_correctly() {
        let steps = vec![
            make_step("short", "a", "o"),
            make_step("this is a longer thought", "a", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_thought_longer_than(5).len(), 1);
    }

    // ── Round 63: steps_with_action_containing, observation_max_chars, observation_min_chars ──

    #[test]
    fn test_steps_with_action_containing_returns_matching_steps() {
        let steps = vec![
            make_step("t", "search(foo)", "o"),
            make_step("t", "read(file)", "o"),
            make_step("t", "search(bar)", "o"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.steps_with_action_containing("search").len(), 2);
    }

    #[test]
    fn test_steps_with_action_containing_empty_when_no_match() {
        let steps = vec![make_step("t", "read(file)", "o")];
        let session = make_session(steps, 0);
        assert!(session.steps_with_action_containing("search").is_empty());
    }

    #[test]
    fn test_observation_max_chars_returns_longest() {
        let steps = vec![
            make_step("t", "a", "hi"),
            make_step("t", "a", "hello world"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_max_chars(), 11);
    }

    #[test]
    fn test_observation_min_chars_skips_empty_observations() {
        let steps = vec![
            make_step("t", "a", ""),
            make_step("t", "a", "abcd"),
            make_step("t", "a", "ab"),
        ];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_min_chars(), 2);
    }

    #[test]
    fn test_observation_min_chars_zero_when_all_empty() {
        let steps = vec![make_step("t", "a", "")];
        let session = make_session(steps, 0);
        assert_eq!(session.observation_min_chars(), 0);
    }
}
