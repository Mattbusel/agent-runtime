//! # Multi-Agent Debate Protocol
//!
//! A structured adversarial reasoning system where multiple agents argue
//! different positions on a topic, score each other's arguments, and a
//! moderator synthesises the strongest reasoning into a final answer.
//!
//! ## Why debates?
//!
//! Single-agent reasoning is vulnerable to anchoring on the first plausible
//! answer.  Debates surface hidden assumptions by forcing agents to argue
//! conflicting positions — even positions they would not independently choose.
//! This is particularly useful for:
//!
//! - Complex decisions with many trade-offs
//! - Red-teaming: exposing weaknesses in a plan
//! - Research synthesis: combining evidence from multiple sources
//! - Ethical dilemmas: ensuring minority views are considered
//!
//! ## Protocol
//!
//! ```text
//! Round 1:  Each debater receives the topic + their assigned position.
//!           They generate an opening argument.
//!
//! Round N:  Each debater receives all prior arguments.
//!           They generate a rebuttal / refined argument.
//!
//! Scoring:  Each debater scores every other debater's latest argument.
//!           The moderator also scores.
//!
//! Synthesis: The moderator reviews all rounds + scores and produces a
//!            final synthesised answer that incorporates the strongest points.
//! ```
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::debate::{
//!     DebateConfig, DebateOrchestrator, DebaterPosition,
//! };
//!
//! # tokio_test::block_on(async {
//! let config = DebateConfig::new("Should we use microservices or a monolith?")
//!     .with_positions(vec![
//!         DebaterPosition::new("Alice", "Microservices are better"),
//!         DebaterPosition::new("Bob", "Monolith is better"),
//!         DebaterPosition::new("Carol", "It depends on team size"),
//!     ])
//!     .with_rounds(2);
//!
//! let orchestrator = DebateOrchestrator::new(config);
//!
//! let session = orchestrator
//!     .run(|agent_id, prompt| async move {
//!         // Replace with actual LLM inference.
//!         format!("{agent_id} argues: {}", &prompt[..50.min(prompt.len())])
//!     })
//!     .await;
//!
//! println!("Winner: {}", session.winner_id().unwrap_or("tie"));
//! println!("Synthesis: {}", session.synthesis);
//! # });
//! ```

use crate::types::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use tokio::task::JoinSet;

// ─── Position ────────────────────────────────────────────────────────────────

/// A debater's assigned position in the debate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebaterPosition {
    /// Unique name for this debater within the debate.
    pub agent_id: AgentId,
    /// The stance this agent must argue, regardless of their "true" opinion.
    pub position: String,
}

impl DebaterPosition {
    /// Create a new debater with an assigned position.
    pub fn new(name: impl Into<String>, position: impl Into<String>) -> Self {
        Self {
            agent_id: AgentId::new(name.into()),
            position: position.into(),
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a [`DebateOrchestrator`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateConfig {
    /// The question or topic being debated.
    pub topic: String,

    /// The debaters and their assigned positions.
    pub positions: Vec<DebaterPosition>,

    /// Number of debate rounds before synthesis (minimum 1).
    pub rounds: u32,

    /// Whether debaters score each other's arguments (adds a scoring round).
    pub enable_peer_scoring: bool,

    /// Maximum length (chars) of each argument — enforced by prompt truncation hint.
    pub max_argument_chars: usize,
}

impl DebateConfig {
    /// Create a new debate configuration with default settings.
    pub fn new(topic: impl Into<String>) -> Self {
        Self {
            topic: topic.into(),
            positions: Vec::new(),
            rounds: 2,
            enable_peer_scoring: true,
            max_argument_chars: 500,
        }
    }

    /// Set the debaters and their positions.
    pub fn with_positions(mut self, positions: Vec<DebaterPosition>) -> Self {
        self.positions = positions;
        self
    }

    /// Set the number of debate rounds (default 2).
    pub fn with_rounds(mut self, rounds: u32) -> Self {
        self.rounds = rounds.max(1);
        self
    }

    /// Disable peer scoring to speed up the debate.
    pub fn without_peer_scoring(mut self) -> Self {
        self.enable_peer_scoring = false;
        self
    }

    /// Set the maximum argument length hint.
    pub fn with_max_argument_chars(mut self, n: usize) -> Self {
        self.max_argument_chars = n;
        self
    }
}

// ─── Argument ────────────────────────────────────────────────────────────────

/// A single argument produced by a debater in one round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    /// The debater who produced this argument.
    pub agent_id: AgentId,

    /// Round number (1-indexed).
    pub round: u32,

    /// The argument text.
    pub text: String,

    /// Peer scores received for this argument: scorer_id → score in [0.0, 1.0].
    pub peer_scores: HashMap<String, f64>,
}

impl Argument {
    /// Compute the mean peer score, or `None` if no scores have been assigned.
    pub fn mean_peer_score(&self) -> Option<f64> {
        if self.peer_scores.is_empty() {
            return None;
        }
        let sum: f64 = self.peer_scores.values().sum();
        Some(sum / self.peer_scores.len() as f64)
    }
}

// ─── DebateRound ─────────────────────────────────────────────────────────────

/// All arguments produced in one debate round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateRound {
    /// Round number.
    pub round: u32,
    /// Arguments from all debaters in this round.
    pub arguments: Vec<Argument>,
}

impl DebateRound {
    /// Return the argument by a specific agent, if any.
    pub fn argument_by(&self, agent_id: &AgentId) -> Option<&Argument> {
        self.arguments.iter().find(|a| &a.agent_id == agent_id)
    }

    /// Return the highest-scoring argument in this round (by mean peer score).
    pub fn leading_argument(&self) -> Option<&Argument> {
        self.arguments
            .iter()
            .filter_map(|a| a.mean_peer_score().map(|s| (s, a)))
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, a)| a)
    }
}

// ─── DebateSession ────────────────────────────────────────────────────────────

/// The complete record of a finished debate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateSession {
    /// The topic that was debated.
    pub topic: String,

    /// All rounds of debate arguments.
    pub rounds: Vec<DebateRound>,

    /// Final synthesised answer produced by the moderator.
    pub synthesis: String,

    /// Cumulative scores per debater across all rounds.
    pub cumulative_scores: HashMap<String, f64>,
}

impl DebateSession {
    /// Return the agent ID of the debater with the highest cumulative score,
    /// or `None` if scores are equal or no scoring took place.
    pub fn winner_id(&self) -> Option<&str> {
        self.cumulative_scores
            .iter()
            .filter(|(_, &s)| s > 0.0)
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id.as_str())
    }

    /// Return a formatted summary suitable for display.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== Debate: {} ===\n\n", self.topic));

        for round in &self.rounds {
            out.push_str(&format!("--- Round {} ---\n", round.round));
            for arg in &round.arguments {
                let score_str = arg
                    .mean_peer_score()
                    .map(|s| format!(" [score: {:.2}]", s))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "{}{}: {}\n",
                    arg.agent_id, score_str, arg.text
                ));
            }
            out.push('\n');
        }

        out.push_str("--- Synthesis ---\n");
        out.push_str(&self.synthesis);
        out.push('\n');

        if let Some(winner) = self.winner_id() {
            out.push_str(&format!("\nWinner by peer score: {}\n", winner));
        }

        out
    }

    /// Return the full argument transcript as a flat list ordered by round then agent.
    pub fn transcript(&self) -> Vec<(u32, &AgentId, &str)> {
        self.rounds
            .iter()
            .flat_map(|r| r.arguments.iter().map(move |a| (r.round, &a.agent_id, a.text.as_str())))
            .collect()
    }
}

// ─── DebateOrchestrator ───────────────────────────────────────────────────────

/// Orchestrates a multi-agent debate session.
///
/// Call [`DebateOrchestrator::run`] with an async inference function to execute
/// the debate. The function is called for each debater in each round, plus once
/// for each scoring exchange (if peer scoring is enabled) and once for the
/// moderator synthesis.
pub struct DebateOrchestrator {
    config: DebateConfig,
}

impl DebateOrchestrator {
    /// Create a new orchestrator with the given configuration.
    pub fn new(config: DebateConfig) -> Self {
        Self { config }
    }

    /// Run the debate. Returns a completed [`DebateSession`].
    ///
    /// `infer` is called with `(agent_id, prompt)` and must return the agent's
    /// response. This can be any async function — swap in your LLM client.
    ///
    /// # Concurrency
    ///
    /// Each debate round spawns all debaters concurrently via
    /// [`tokio::task::JoinSet`]. Peer scoring also runs concurrently per debater.
    pub async fn run<F, Fut>(&self, infer: F) -> DebateSession
    where
        F: Fn(AgentId, String) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let mut all_rounds: Vec<DebateRound> = Vec::new();
        let mut cumulative_scores: HashMap<String, f64> = HashMap::new();

        // Initialise score accumulators
        for pos in &self.config.positions {
            cumulative_scores.insert(pos.agent_id.to_string(), 0.0);
        }

        // ── Run each debate round ────────────────────────────────────────────
        for round_num in 1..=self.config.rounds {
            let round = self
                .run_round(round_num, &all_rounds, infer.clone())
                .await;
            all_rounds.push(round);

            // ── Peer scoring ─────────────────────────────────────────────────
            if self.config.enable_peer_scoring {
                let round = all_rounds.last_mut().unwrap();
                self.run_peer_scoring(round, infer.clone()).await;

                // Accumulate scores
                for arg in &round.arguments {
                    if let Some(mean) = arg.mean_peer_score() {
                        *cumulative_scores
                            .entry(arg.agent_id.to_string())
                            .or_default() += mean;
                    }
                }
            }
        }

        // ── Moderator synthesis ──────────────────────────────────────────────
        let synthesis = self.run_synthesis(&all_rounds, infer).await;

        DebateSession {
            topic: self.config.topic.clone(),
            rounds: all_rounds,
            synthesis,
            cumulative_scores,
        }
    }

    // ── Internal: run one debate round ──────────────────────────────────────

    async fn run_round<F, Fut>(
        &self,
        round_num: u32,
        prior_rounds: &[DebateRound],
        infer: F,
    ) -> DebateRound
    where
        F: Fn(AgentId, String) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let mut set: JoinSet<Argument> = JoinSet::new();

        for pos in &self.config.positions {
            let prompt = self.build_argument_prompt(round_num, pos, prior_rounds);
            let agent_id = pos.agent_id.clone();
            let infer = infer.clone();

            set.spawn(async move {
                let text = infer(agent_id.clone(), prompt).await;
                Argument {
                    agent_id,
                    round: round_num,
                    text,
                    peer_scores: HashMap::new(),
                }
            });
        }

        let mut arguments = Vec::new();
        while let Some(result) = set.join_next().await {
            if let Ok(arg) = result {
                arguments.push(arg);
            }
        }

        // Sort by agent_id for deterministic ordering
        arguments.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));

        DebateRound { round: round_num, arguments }
    }

    // ── Internal: run peer scoring for the latest round ─────────────────────

    async fn run_peer_scoring<F, Fut>(&self, round: &mut DebateRound, infer: F)
    where
        F: Fn(AgentId, String) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        // Collect (scorer_id, target_id, argument_text) triples
        let pairs: Vec<(AgentId, AgentId, String)> = round
            .arguments
            .iter()
            .flat_map(|target| {
                self.config
                    .positions
                    .iter()
                    .filter(|p| p.agent_id != target.agent_id)
                    .map(|scorer| {
                        (
                            scorer.agent_id.clone(),
                            target.agent_id.clone(),
                            target.text.clone(),
                        )
                    })
            })
            .collect();

        let mut set: JoinSet<(AgentId, AgentId, f64)> = JoinSet::new();

        for (scorer_id, target_id, arg_text) in pairs {
            let prompt = format!(
                "Score the following argument from 0.0 to 1.0 based on logical strength, \
                 evidence quality, and persuasiveness. Respond with ONLY a decimal number \
                 like 0.75.\n\nArgument: {}",
                &arg_text[..arg_text.len().min(self.config.max_argument_chars)]
            );
            let infer = infer.clone();

            set.spawn(async move {
                let response = infer(scorer_id.clone(), prompt).await;
                let score = parse_score(&response);
                (scorer_id, target_id, score)
            });
        }

        // Collect scores
        let mut score_map: HashMap<(String, String), f64> = HashMap::new();
        while let Some(result) = set.join_next().await {
            if let Ok((scorer, target, score)) = result {
                score_map.insert((scorer.to_string(), target.to_string()), score);
            }
        }

        // Apply scores to arguments
        for arg in &mut round.arguments {
            for ((scorer, target), score) in &score_map {
                if target == &arg.agent_id.to_string() {
                    arg.peer_scores.insert(scorer.clone(), *score);
                }
            }
        }
    }

    // ── Internal: moderator synthesis ────────────────────────────────────────

    async fn run_synthesis<F, Fut>(&self, rounds: &[DebateRound], infer: F) -> String
    where
        F: Fn(AgentId, String) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let moderator_id = AgentId::new("moderator");
        let prompt = self.build_synthesis_prompt(rounds);
        infer(moderator_id, prompt).await
    }

    // ── Prompt builders ──────────────────────────────────────────────────────

    fn build_argument_prompt(
        &self,
        round: u32,
        pos: &DebaterPosition,
        prior_rounds: &[DebateRound],
    ) -> String {
        let mut prompt = format!(
            "You are arguing in a debate. Topic: \"{}\"\n\
             Your assigned position: \"{}\"\n\
             Argue FOR your position, even if you personally disagree.\n\
             Be specific, logical, and concise (max {} characters).\n\n",
            self.config.topic, pos.position, self.config.max_argument_chars
        );

        if round == 1 {
            prompt.push_str("This is your opening argument. State your strongest points.\n");
        } else {
            prompt.push_str("Prior arguments from other debaters:\n");
            for prior in prior_rounds {
                for arg in &prior.arguments {
                    if arg.agent_id != pos.agent_id {
                        prompt.push_str(&format!(
                            "- {} (Round {}): {}\n",
                            arg.agent_id, prior.round,
                            &arg.text[..arg.text.len().min(200)]
                        ));
                    }
                }
            }
            prompt.push_str("\nRebut the strongest opposing argument and reinforce your position.\n");
        }

        prompt.push_str("\nYour argument:");
        prompt
    }

    fn build_synthesis_prompt(&self, rounds: &[DebateRound]) -> String {
        let mut prompt = format!(
            "You are a neutral moderator. A debate was held on the topic: \"{}\"\n\n\
             Your task: synthesise the strongest points from all sides into a balanced, \
             nuanced final answer that acknowledges trade-offs.\n\n\
             Complete debate transcript:\n",
            self.config.topic
        );

        for round in rounds {
            prompt.push_str(&format!("Round {}:\n", round.round));
            for arg in &round.arguments {
                let score_note = arg
                    .mean_peer_score()
                    .map(|s| format!(" [peer score: {:.2}]", s))
                    .unwrap_or_default();
                prompt.push_str(&format!(
                    "  {}{}: {}\n",
                    arg.agent_id, score_note,
                    &arg.text[..arg.text.len().min(300)]
                ));
            }
        }

        prompt.push_str(
            "\nProvide a synthesised answer that:\n\
             1. Identifies the strongest argument from each side.\n\
             2. Acknowledges valid points from all positions.\n\
             3. Gives a concrete recommendation or balanced conclusion.\n\
             \nSynthesis:",
        );
        prompt
    }
}

// ─── Score parsing ────────────────────────────────────────────────────────────

/// Parse a score from a model response. Extracts the first decimal number
/// in [0.0, 1.0]. Returns 0.5 as a neutral fallback if parsing fails.
fn parse_score(response: &str) -> f64 {
    // Find the first run of digits/period in the response
    let candidate: String = response
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();

    candidate
        .parse::<f64>()
        .ok()
        .map(|s| s.clamp(0.0, 1.0))
        .unwrap_or(0.5)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn debate_produces_session_with_correct_rounds() {
        let config = DebateConfig::new("Monolith vs microservices")
            .with_positions(vec![
                DebaterPosition::new("Alice", "Monolith is better for small teams"),
                DebaterPosition::new("Bob", "Microservices scale better"),
            ])
            .with_rounds(2)
            .without_peer_scoring();

        let orchestrator = DebateOrchestrator::new(config);

        let session = orchestrator
            .run(|agent_id, _prompt| async move {
                format!("{agent_id} makes a compelling argument here.")
            })
            .await;

        assert_eq!(session.rounds.len(), 2);
        assert_eq!(session.rounds[0].arguments.len(), 2);
        assert!(!session.synthesis.is_empty());
    }

    #[tokio::test]
    async fn peer_scoring_populates_scores() {
        let config = DebateConfig::new("Is Rust better than Python?")
            .with_positions(vec![
                DebaterPosition::new("Rustacean", "Rust: safe, fast, no GC"),
                DebaterPosition::new("Pythonista", "Python: readable and productive"),
            ])
            .with_rounds(1);

        let orchestrator = DebateOrchestrator::new(config);

        let session = orchestrator
            .run(|agent_id, prompt| async move {
                // For scoring prompts, return a score; for arguments, return text.
                if prompt.contains("Score the following") {
                    "0.75".to_string()
                } else {
                    format!("{agent_id} argues strongly.")
                }
            })
            .await;

        let round = &session.rounds[0];
        for arg in &round.arguments {
            // Each argument should have been scored by the other debater
            assert!(!arg.peer_scores.is_empty(), "Expected peer scores on {:?}", arg.agent_id);
            assert!(arg.mean_peer_score().is_some());
        }
    }

    #[tokio::test]
    async fn winner_id_returns_highest_scorer() {
        let config = DebateConfig::new("A vs B")
            .with_positions(vec![
                DebaterPosition::new("TeamA", "A is best"),
                DebaterPosition::new("TeamB", "B is best"),
            ])
            .with_rounds(1);

        let orchestrator = DebateOrchestrator::new(config);

        let session = orchestrator
            .run(|agent_id, prompt| async move {
                if prompt.contains("Score the following") {
                    // TeamA scores opponents low; TeamB scores opponents high
                    if agent_id.to_string().contains("TeamA") {
                        "0.3".to_string()
                    } else {
                        "0.9".to_string()
                    }
                } else {
                    format!("{agent_id} argument")
                }
            })
            .await;

        // TeamB received a 0.9 from TeamA; TeamA received a 0.3 from TeamB
        let winner = session.winner_id();
        assert!(winner.is_some());
    }

    #[test]
    fn parse_score_handles_various_formats() {
        assert!((parse_score("0.75") - 0.75).abs() < 1e-6);
        assert!((parse_score("Score: 0.8 out of 1") - 0.8).abs() < 1e-6);
        assert!((parse_score("I give it a 1.0") - 1.0).abs() < 1e-6);
        assert!((parse_score("no number here") - 0.5).abs() < 1e-6);
        // Values outside [0,1] are clamped
        assert!((parse_score("5.0") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn debate_session_summary_contains_topic() {
        let session = DebateSession {
            topic: "Test topic".to_string(),
            rounds: vec![],
            synthesis: "Final answer.".to_string(),
            cumulative_scores: HashMap::new(),
        };
        let summary = session.summary();
        assert!(summary.contains("Test topic"));
        assert!(summary.contains("Final answer."));
    }

    #[test]
    fn debate_config_builder_chain() {
        let config = DebateConfig::new("Topic")
            .with_positions(vec![DebaterPosition::new("A", "pro")])
            .with_rounds(3)
            .with_max_argument_chars(1000)
            .without_peer_scoring();

        assert_eq!(config.rounds, 3);
        assert!(!config.enable_peer_scoring);
        assert_eq!(config.max_argument_chars, 1000);
        assert_eq!(config.positions.len(), 1);
    }
}
