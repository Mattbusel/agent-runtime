//! # Module: Agent Self-Reflection
//!
//! ## Responsibility
//! After each ReAct cycle the agent scores its own reasoning on a 0–10 scale
//! using a secondary LLM call.  If the score falls below a configurable
//! threshold, the agent retries with a modified reasoning prompt.  All
//! reflection events are accumulated in a [`ReflectionHistory`].
//!
//! ## Design
//!
//! ```text
//! ReActLoop step ──► ReflectionEngine::evaluate()
//!                          │
//!                   score < threshold?
//!                          │
//!                    yes ──┤──► produce retry hint, increment attempt counter
//!                          │
//!                    no  ──┴──► accept step, append to history
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_agent_runtime::reflection::{ReflectionEngine, ReflectionConfig};
//!
//! # tokio_test::block_on(async {
//! let mut engine = ReflectionEngine::new(ReflectionConfig::default());
//!
//! // Supply a scorer that calls the real LLM in production.
//! engine.set_scorer(|reasoning: String| {
//!     Box::pin(async move {
//!         // Parse score from LLM response; return a value 0–10.
//!         if reasoning.contains("FINAL_ANSWER") { 9.0_f64 } else { 4.0_f64 }
//!     })
//! });
//!
//! let outcome = engine.evaluate("Thought: I should search\nAction: web_search").await;
//! println!("{outcome:?}");
//! # });
//! ```

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A future that resolves to a reflection score in `[0.0, 10.0]`.
pub type ScorerFuture = Pin<Box<dyn Future<Output = f64> + Send>>;

/// A scorer closure: receives the agent's reasoning text and returns a score.
pub type ScorerFn = Box<dyn Fn(String) -> ScorerFuture + Send + Sync>;

// ── Config ─────────────────────────────────────────────────────────────────────

/// Configuration for the self-reflection engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionConfig {
    /// Minimum acceptable score; reasoning below this triggers a retry.
    pub score_threshold: f64,
    /// Maximum number of retry attempts before accepting the result anyway.
    pub max_retries: usize,
    /// Whether to prepend a reflection hint to the retry prompt.
    pub prepend_hint: bool,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 6.0,
            max_retries: 2,
            prepend_hint: true,
        }
    }
}

// ── Reflection record ─────────────────────────────────────────────────────────

/// The outcome of a single reflection evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflectionOutcome {
    /// Score met the threshold — reasoning is accepted.
    Accepted {
        /// The score assigned by the scorer.
        score: f64,
        /// Number of retry attempts before acceptance (0 = first try).
        attempt: usize,
    },
    /// Score was below the threshold on all allowed attempts.
    Exhausted {
        /// Score on the final attempt.
        final_score: f64,
        /// Maximum attempts taken.
        attempts: usize,
    },
    /// Scorer was not configured — reasoning was accepted without scoring.
    Unscored,
}

/// One entry in the reflection history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionEntry {
    /// Cycle index (0-based) within the enclosing ReAct loop.
    pub cycle: usize,
    /// The reasoning text that was evaluated.
    pub reasoning: String,
    /// Outcome of this evaluation.
    pub outcome: ReflectionOutcome,
    /// Hint injected into the retry prompt, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_hint: Option<String>,
}

/// Accumulated reflection history for an agent session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReflectionHistory {
    entries: Vec<ReflectionEntry>,
}

impl ReflectionHistory {
    /// Append a new entry.
    pub fn push(&mut self, entry: ReflectionEntry) {
        self.entries.push(entry);
    }

    /// All entries in chronological order.
    pub fn entries(&self) -> &[ReflectionEntry] {
        &self.entries
    }

    /// Mean score across all scored, accepted entries.
    pub fn mean_score(&self) -> Option<f64> {
        let scores: Vec<f64> = self
            .entries
            .iter()
            .filter_map(|e| match e.outcome {
                ReflectionOutcome::Accepted { score, .. } => Some(score),
                ReflectionOutcome::Exhausted { final_score, .. } => Some(final_score),
                ReflectionOutcome::Unscored => None,
            })
            .collect();

        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Number of cycles where the agent had to retry.
    pub fn retry_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e.outcome, ReflectionOutcome::Accepted { attempt, .. } if attempt > 0))
            .count()
    }

    /// Number of cycles where all retries were exhausted.
    pub fn exhausted_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e.outcome, ReflectionOutcome::Exhausted { .. }))
            .count()
    }
}

// ── Engine ─────────────────────────────────────────────────────────────────────

/// Drives agent self-reflection after each ReAct cycle.
pub struct ReflectionEngine {
    config: ReflectionConfig,
    scorer: Option<ScorerFn>,
    history: ReflectionHistory,
    cycle: usize,
}

impl ReflectionEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: ReflectionConfig) -> Self {
        Self {
            config,
            scorer: None,
            history: ReflectionHistory::default(),
            cycle: 0,
        }
    }

    /// Replace the scorer closure.
    ///
    /// In production, supply a closure that calls a secondary LLM and parses
    /// its response into a `f64` in `[0.0, 10.0]`.
    pub fn set_scorer<F, Fut>(&mut self, scorer: F)
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = f64> + Send + 'static,
    {
        self.scorer = Some(Box::new(move |r| Box::pin(scorer(r))));
    }

    /// Evaluate a reasoning string.
    ///
    /// If the score is below the threshold, returns a retry hint.
    /// Regardless of outcome, appends an entry to [`ReflectionHistory`].
    pub async fn evaluate(&mut self, reasoning: &str) -> ReflectionOutcome {
        let cycle = self.cycle;
        self.cycle += 1;

        let scorer = match self.scorer.as_ref() {
            None => {
                self.history.push(ReflectionEntry {
                    cycle,
                    reasoning: reasoning.to_string(),
                    outcome: ReflectionOutcome::Unscored,
                    retry_hint: None,
                });
                return ReflectionOutcome::Unscored;
            }
            Some(s) => s,
        };

        let mut attempt = 0usize;
        let mut current_reasoning = reasoning.to_string();
        let mut last_score = 0.0_f64;

        loop {
            let score = scorer(current_reasoning.clone()).await;
            last_score = score;

            tracing::debug!(
                cycle,
                attempt,
                score,
                threshold = self.config.score_threshold,
                "reflection: score"
            );

            if score >= self.config.score_threshold {
                let outcome = ReflectionOutcome::Accepted { score, attempt };
                self.history.push(ReflectionEntry {
                    cycle,
                    reasoning: reasoning.to_string(),
                    outcome: outcome.clone(),
                    retry_hint: None,
                });
                return outcome;
            }

            if attempt >= self.config.max_retries {
                let outcome = ReflectionOutcome::Exhausted {
                    final_score: score,
                    attempts: attempt + 1,
                };
                self.history.push(ReflectionEntry {
                    cycle,
                    reasoning: reasoning.to_string(),
                    outcome: outcome.clone(),
                    retry_hint: None,
                });
                return outcome;
            }

            // Generate a retry hint and augment the reasoning.
            let hint = self.build_hint(score, attempt);
            tracing::debug!(cycle, attempt, hint = %hint, "reflection: injecting retry hint");

            if self.config.prepend_hint {
                current_reasoning = format!("[Reflection hint: {hint}]\n{reasoning}");
            }

            attempt += 1;
        }
    }

    /// Build a human-readable hint based on the current score and attempt.
    fn build_hint(&self, score: f64, attempt: usize) -> String {
        let urgency = if attempt == 0 { "Consider" } else { "You must" };
        format!(
            "{urgency} revising your reasoning. Current score: {score:.1}/10. \
             Aim to be more specific, verify each step, and confirm your final answer."
        )
    }

    /// Borrow the accumulated reflection history.
    pub fn history(&self) -> &ReflectionHistory {
        &self.history
    }

    /// Take ownership of the history, resetting it to empty.
    pub fn take_history(&mut self) -> ReflectionHistory {
        std::mem::take(&mut self.history)
    }

    /// Current cycle counter (number of evaluations performed so far).
    pub fn cycle(&self) -> usize {
        self.cycle
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with_fixed_score(score: f64) -> ReflectionEngine {
        let mut engine = ReflectionEngine::new(ReflectionConfig::default());
        engine.set_scorer(move |_| Box::pin(async move { score }));
        engine
    }

    #[tokio::test]
    async fn accepts_high_score() {
        let mut engine = engine_with_fixed_score(8.0);
        let outcome = engine.evaluate("Thought: done\nAction: FINAL_ANSWER").await;
        assert!(matches!(
            outcome,
            ReflectionOutcome::Accepted { score, attempt: 0 } if score == 8.0
        ));
        assert_eq!(engine.history().entries().len(), 1);
    }

    #[tokio::test]
    async fn exhausts_retries_on_low_score() {
        let cfg = ReflectionConfig {
            score_threshold: 7.0,
            max_retries: 2,
            prepend_hint: false,
        };
        let mut engine = ReflectionEngine::new(cfg);
        engine.set_scorer(|_| Box::pin(async { 3.0_f64 }));

        let outcome = engine.evaluate("Bad reasoning").await;
        assert!(matches!(
            outcome,
            ReflectionOutcome::Exhausted { attempts: 3, .. }
        ));
    }

    #[tokio::test]
    async fn unscored_when_no_scorer() {
        let mut engine = ReflectionEngine::new(ReflectionConfig::default());
        let outcome = engine.evaluate("Any reasoning").await;
        assert!(matches!(outcome, ReflectionOutcome::Unscored));
    }

    #[tokio::test]
    async fn history_mean_score() {
        let mut engine = engine_with_fixed_score(9.0);
        engine.evaluate("step 1").await;
        engine.evaluate("step 2").await;
        let mean = engine.history().mean_score().unwrap();
        assert!((mean - 9.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn retry_count_tracked() {
        // Score below threshold on attempt 0, above on attempt 1.
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let mut engine = ReflectionEngine::new(ReflectionConfig {
            score_threshold: 7.0,
            max_retries: 3,
            prepend_hint: false,
        });
        engine.set_scorer(move |_| {
            let cc = cc.clone();
            Box::pin(async move {
                let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 { 4.0 } else { 8.0 }
            })
        });

        let outcome = engine.evaluate("reasoning").await;
        assert!(matches!(
            outcome,
            ReflectionOutcome::Accepted { attempt: 1, .. }
        ));
        assert_eq!(engine.history().retry_count(), 1);
    }
}
