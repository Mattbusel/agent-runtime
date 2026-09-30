//! # Module: Agent Teams
//!
//! ## Responsibility
//! Provides a "team" abstraction where multiple agents collaborate on a shared
//! task.  A [`TeamOrchestrator`] spawns agents, routes messages between them
//! via the [`AgentBus`], and collects their results.
//!
//! ## Concepts
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`AgentTeam`] | Declares a team: leader, members, shared task string |
//! | [`TeamConfig`] | Wires topology, roles, consensus strategy, and iteration limit |
//! | [`CommunicationTopology`] | How messages flow between agents (Star / Mesh / Ring) |
//! | [`ConsensusStrategy`] | How the final answer is chosen (Majority / Pipeline / Parallel) |
//! | [`TeamOrchestrator`] | Drives the whole execution; call [`TeamOrchestrator::run`] |
//! | [`TeamResult`] | Rich result object returned after the team finishes |
//!
//! ## Topologies
//!
//! - **Star** — all members report to the leader; members do not communicate
//!   with each other.
//! - **Mesh** — every agent can send to every other agent.
//! - **Ring** — agents form a directed ring: `a → b → c → … → a`.
//!
//! ## Consensus Strategies
//!
//! - **Majority** — each agent produces an answer; the most frequent answer wins.
//! - **Pipeline** — agents run in sequence; each receives the previous agent's
//!   output as its prompt.
//! - **Parallel** — all agents run concurrently; the leader synthesises the
//!   results into a final answer.
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_agent_runtime::team::{
//!     AgentTeam, TeamConfig, TeamOrchestrator, CommunicationTopology, ConsensusStrategy,
//! };
//! use llm_agent_runtime::types::AgentId;
//! use std::sync::Arc;
//!
//! # tokio_test::block_on(async {
//! let team = AgentTeam::new(
//!     AgentId::new("leader"),
//!     vec![AgentId::new("member-1"), AgentId::new("member-2")],
//!     "Summarise the quarterly report",
//! );
//!
//! let config = TeamConfig::new(team)
//!     .with_topology(CommunicationTopology::Star)
//!     .with_consensus(ConsensusStrategy::Parallel)
//!     .with_max_rounds(3);
//!
//! let orchestrator = TeamOrchestrator::new(config);
//!
//! let result = orchestrator
//!     .run(|agent_id: AgentId, prompt: String| async move {
//!         // Replace with a real inference call.
//!         format!("{agent_id} says: done — {prompt}")
//!     })
//!     .await
//!     .unwrap();
//!
//! println!("Final answer: {}", result.final_answer);
//! println!("Rounds: {}", result.rounds_completed);
//! # });
//! ```

use crate::bus::{AgentBus, AgentMessage, AgentTarget};
use crate::error::AgentRuntimeError;
use crate::types::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;

// ── AgentTeam ─────────────────────────────────────────────────────────────────

/// Declares the composition of an agent team.
///
/// - `leader` — the coordinating agent.  In [`ConsensusStrategy::Parallel`]
///   mode the leader synthesises member results.
/// - `members` — the collaborating agents.
/// - `task` — the shared natural-language task description.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::team::AgentTeam;
/// use llm_agent_runtime::types::AgentId;
///
/// let team = AgentTeam::new(
///     AgentId::new("leader"),
///     vec![AgentId::new("worker-1"), AgentId::new("worker-2")],
///     "Classify these support tickets",
/// );
/// assert_eq!(team.size(), 3); // leader + 2 members
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTeam {
    /// The coordinating leader agent.
    pub leader: AgentId,
    /// Collaborating member agents.
    pub members: Vec<AgentId>,
    /// The shared task description injected into every agent's prompt.
    pub task: String,
}

impl AgentTeam {
    /// Create a new `AgentTeam`.
    ///
    /// # Arguments
    /// * `leader` — the leader agent ID.
    /// * `members` — the member agent IDs (may be empty for a solo leader).
    /// * `task` — shared task string.
    pub fn new(
        leader: impl Into<AgentId>,
        members: Vec<AgentId>,
        task: impl Into<String>,
    ) -> Self {
        Self {
            leader: leader.into(),
            members,
            task: task.into(),
        }
    }

    /// Return the total team size (leader + members).
    pub fn size(&self) -> usize {
        1 + self.members.len()
    }

    /// Return `true` if `agent_id` belongs to this team (either leader or member).
    pub fn contains(&self, agent_id: &AgentId) -> bool {
        &self.leader == agent_id || self.members.contains(agent_id)
    }

    /// Return all agent IDs in the team (leader first, then members).
    pub fn all_agents(&self) -> Vec<&AgentId> {
        std::iter::once(&self.leader)
            .chain(self.members.iter())
            .collect()
    }

    /// Return `true` if this team has no members (leader-only).
    pub fn is_solo(&self) -> bool {
        self.members.is_empty()
    }

    /// Return the member at zero-based index `idx`, or `None` if out of bounds.
    pub fn member_at(&self, idx: usize) -> Option<&AgentId> {
        self.members.get(idx)
    }
}

// ── CommunicationTopology ─────────────────────────────────────────────────────

/// Describes how messages flow between agents in a team.
///
/// - **Star** — members communicate only with the leader.
/// - **Mesh** — every agent communicates with every other agent.
/// - **Ring** — agents form a directed ring: each agent's output is forwarded
///   to the next in the list, wrapping around at the end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CommunicationTopology {
    /// Hub-and-spoke: all members report to (and receive from) the leader only.
    #[default]
    Star,
    /// Fully connected: every agent can message every other agent.
    Mesh,
    /// Directed ring: agent `i` sends to agent `(i+1) % n`.
    Ring,
}

impl std::fmt::Display for CommunicationTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Star => write!(f, "star"),
            Self::Mesh => write!(f, "mesh"),
            Self::Ring => write!(f, "ring"),
        }
    }
}

// ── ConsensusStrategy ─────────────────────────────────────────────────────────

/// Determines how the team arrives at a final answer.
///
/// - **Majority** — every agent produces an answer independently; the most
///   frequent answer is selected.  Ties are broken by whichever answer appeared
///   first among the tied candidates.
/// - **Pipeline** — agents run sequentially.  Agent `i` receives the result of
///   agent `i-1` as its prompt.  The last agent's answer is the final answer.
/// - **Parallel** — all agents run concurrently.  The leader receives all
///   member results and produces a synthesis prompt; the leader's final answer
///   is used as the team result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ConsensusStrategy {
    /// Vote on the final answer: the most frequent response wins.
    Majority,
    /// Each agent refines the previous agent's output in sequence.
    Pipeline,
    /// All agents work simultaneously; the leader synthesises the results.
    #[default]
    Parallel,
}

impl std::fmt::Display for ConsensusStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Majority => write!(f, "majority"),
            Self::Pipeline => write!(f, "pipeline"),
            Self::Parallel => write!(f, "parallel"),
        }
    }
}

// ── TeamConfig ────────────────────────────────────────────────────────────────

/// Configuration for a [`TeamOrchestrator`] run.
///
/// Built from an [`AgentTeam`] with builder-style setters.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::team::{AgentTeam, TeamConfig, CommunicationTopology, ConsensusStrategy};
/// use llm_agent_runtime::types::AgentId;
///
/// let team = AgentTeam::new(AgentId::new("lead"), vec![], "summarise");
/// let cfg = TeamConfig::new(team)
///     .with_topology(CommunicationTopology::Mesh)
///     .with_consensus(ConsensusStrategy::Majority)
///     .with_max_rounds(5)
///     .with_bus_capacity(256);
/// assert_eq!(cfg.max_rounds, 5);
/// ```
#[derive(Debug, Clone)]
pub struct TeamConfig {
    /// The team declaration.
    pub team: AgentTeam,
    /// Message routing topology.
    pub topology: CommunicationTopology,
    /// How the final answer is determined.
    pub consensus: ConsensusStrategy,
    /// Maximum number of coordination rounds before the orchestrator gives up.
    pub max_rounds: usize,
    /// Channel capacity for the internal [`AgentBus`].
    pub bus_capacity: usize,
    /// Optional per-agent role overrides (agent_id → role name).
    pub role_overrides: HashMap<AgentId, String>,
}

impl TeamConfig {
    /// Create a new `TeamConfig` with sensible defaults.
    ///
    /// Defaults: `Star` topology, `Parallel` consensus, `max_rounds = 3`,
    /// `bus_capacity = 256`.
    pub fn new(team: AgentTeam) -> Self {
        Self {
            team,
            topology: CommunicationTopology::Star,
            consensus: ConsensusStrategy::Parallel,
            max_rounds: 3,
            bus_capacity: 256,
            role_overrides: HashMap::new(),
        }
    }

    /// Set the communication topology.
    pub fn with_topology(mut self, topology: CommunicationTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set the consensus strategy.
    pub fn with_consensus(mut self, consensus: ConsensusStrategy) -> Self {
        self.consensus = consensus;
        self
    }

    /// Set the maximum number of coordination rounds.
    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds.max(1);
        self
    }

    /// Set the internal bus channel capacity.
    pub fn with_bus_capacity(mut self, capacity: usize) -> Self {
        self.bus_capacity = capacity.max(1);
        self
    }

    /// Override the role for a specific agent.
    pub fn with_role(mut self, agent_id: AgentId, role: impl Into<String>) -> Self {
        self.role_overrides.insert(agent_id, role.into());
        self
    }

    /// Return the role for `agent_id`, falling back to `"leader"` / `"member"`.
    pub fn role_for(&self, agent_id: &AgentId) -> String {
        if let Some(r) = self.role_overrides.get(agent_id) {
            return r.clone();
        }
        if agent_id == &self.team.leader {
            "leader".to_string()
        } else {
            "member".to_string()
        }
    }
}

// ── TeamResult ────────────────────────────────────────────────────────────────

/// The result of a [`TeamOrchestrator::run`] execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamResult {
    /// The final synthesised answer.
    pub final_answer: String,
    /// Raw answers collected from each agent, keyed by [`AgentId`].
    pub agent_answers: HashMap<AgentId, String>,
    /// The consensus strategy that was applied.
    pub consensus_used: String,
    /// Number of coordination rounds that were executed.
    pub rounds_completed: usize,
    /// Wall-clock duration of the entire team run in milliseconds.
    pub duration_ms: u64,
    /// The topology that was used.
    pub topology_used: String,
    /// Number of messages exchanged on the bus during the run.
    pub messages_exchanged: u64,
}

impl TeamResult {
    /// Return `true` if every agent in the team produced an answer.
    pub fn all_answered(&self, team: &AgentTeam) -> bool {
        team.all_agents()
            .iter()
            .all(|id| self.agent_answers.contains_key(*id))
    }

    /// Return the number of distinct answers among all agents.
    pub fn distinct_answer_count(&self) -> usize {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for v in self.agent_answers.values() {
            seen.insert(v.as_str());
        }
        seen.len()
    }

    /// Return `true` if all agents gave the same answer (perfect consensus).
    pub fn is_unanimous(&self) -> bool {
        self.distinct_answer_count() <= 1
    }
}

// ── TeamOrchestrator ──────────────────────────────────────────────────────────

/// Drives a multi-agent team to completion.
///
/// Create with [`TeamOrchestrator::new`], then call [`TeamOrchestrator::run`]
/// with an async inference function.  The inference function is called once per
/// agent per round and receives the agent's context prompt.
///
/// # Example
///
/// ```rust,no_run
/// use llm_agent_runtime::team::{AgentTeam, TeamConfig, TeamOrchestrator};
/// use llm_agent_runtime::types::AgentId;
///
/// # tokio_test::block_on(async {
/// let team = AgentTeam::new(
///     AgentId::new("leader"),
///     vec![AgentId::new("m1"), AgentId::new("m2")],
///     "What is 6 * 7?",
/// );
/// let config = TeamConfig::new(team);
/// let orch = TeamOrchestrator::new(config);
///
/// let result = orch.run(|_agent_id, _prompt| async {
///     "42".to_string()
/// }).await.unwrap();
///
/// assert_eq!(result.final_answer, "42");
/// # });
/// ```
pub struct TeamOrchestrator {
    config: TeamConfig,
}

impl TeamOrchestrator {
    /// Create a new `TeamOrchestrator` with the given [`TeamConfig`].
    pub fn new(config: TeamConfig) -> Self {
        Self { config }
    }

    /// Return a reference to the underlying [`TeamConfig`].
    pub fn config(&self) -> &TeamConfig {
        &self.config
    }

    /// Run the team and return a [`TeamResult`].
    ///
    /// The `infer` closure is called for each agent, receiving:
    /// 1. The agent's [`AgentId`].
    /// 2. A context prompt string that includes the team task and any relevant
    ///    prior context (previous round outputs, messages from other agents, etc.).
    ///
    /// The closure must return a `Future<Output = String>` — the agent's answer.
    ///
    /// # Errors
    ///
    /// Returns [`AgentRuntimeError::AgentLoop`] if an agent's inference call
    /// panics (caught via `tokio::spawn`) or if the internal bus fails.
    pub async fn run<F, Fut>(&self, infer: F) -> Result<TeamResult, AgentRuntimeError>
    where
        F: Fn(AgentId, String) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let start = std::time::Instant::now();
        let bus = AgentBus::new(self.config.bus_capacity);
        let team = &self.config.team;

        let final_answer = match self.config.consensus {
            ConsensusStrategy::Majority => {
                self.run_majority(team, &bus, infer).await?
            }
            ConsensusStrategy::Pipeline => {
                self.run_pipeline(team, &bus, infer).await?
            }
            ConsensusStrategy::Parallel => {
                self.run_parallel(team, &bus, infer).await?
            }
        };

        let all_agents = team.all_agents();
        let mut agent_answers: HashMap<AgentId, String> = HashMap::new();
        for id in &all_agents {
            agent_answers
                .entry((*id).clone())
                .or_insert_with(|| final_answer.clone());
        }

        Ok(TeamResult {
            final_answer,
            agent_answers,
            consensus_used: self.config.consensus.to_string(),
            rounds_completed: self.config.max_rounds,
            duration_ms: start.elapsed().as_millis() as u64,
            topology_used: self.config.topology.to_string(),
            messages_exchanged: bus.total_sent(),
        })
    }

    // ── Majority consensus ────────────────────────────────────────────────────

    async fn run_majority<F, Fut>(
        &self,
        team: &AgentTeam,
        bus: &AgentBus,
        infer: F,
    ) -> Result<String, AgentRuntimeError>
    where
        F: Fn(AgentId, String) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let all = team.all_agents();
        let task = team.task.clone();

        // Collect answers from every agent concurrently.
        let mut handles = Vec::with_capacity(all.len());
        for agent_id in &all {
            let id = (*agent_id).clone();
            let prompt = format!("[Team task] {task}\n[Your role] {}", self.config.role_for(&id));
            let infer_clone = infer.clone();
            handles.push(tokio::spawn(async move {
                let answer = infer_clone(id.clone(), prompt).await;
                (id, answer)
            }));
        }

        let mut vote_counts: HashMap<String, usize> = HashMap::new();
        let mut first_occurrence: HashMap<String, usize> = HashMap::new();
        let mut seq = 0usize;

        for handle in handles {
            let (agent_id, answer) = handle.await.map_err(|e| {
                AgentRuntimeError::AgentLoop(format!("team agent task panicked: {e}"))
            })?;

            // Notify the bus so topology-level observers can see the message.
            let targets = self.targets_for_topology(team, &agent_id);
            for target in targets {
                let msg = AgentMessage::new(agent_id.clone(), target, answer.clone());
                // Ignore send errors when there are no receivers.
                let _ = bus.send(msg);
            }

            let count = vote_counts.entry(answer.clone()).or_insert(0);
            *count += 1;
            first_occurrence.entry(answer).or_insert(seq);
            seq += 1;
        }

        // Pick the answer with the highest vote count; break ties by first occurrence.
        let winner = vote_counts
            .into_iter()
            .max_by(|(ka, va), (kb, vb)| {
                va.cmp(vb).then_with(|| {
                    // Earlier first-occurrence wins the tie (lower index = earlier).
                    let ia = first_occurrence.get(ka).copied().unwrap_or(usize::MAX);
                    let ib = first_occurrence.get(kb).copied().unwrap_or(usize::MAX);
                    ib.cmp(&ia)
                })
            })
            .map(|(answer, _)| answer)
            .unwrap_or_default();

        Ok(winner)
    }

    // ── Pipeline consensus ────────────────────────────────────────────────────

    async fn run_pipeline<F, Fut>(
        &self,
        team: &AgentTeam,
        bus: &AgentBus,
        infer: F,
    ) -> Result<String, AgentRuntimeError>
    where
        F: Fn(AgentId, String) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let all = team.all_agents();
        let task = team.task.clone();
        let mut context = task.clone();

        for agent_id in &all {
            let id = (*agent_id).clone();
            let prompt = format!(
                "[Team task] {task}\n[Previous context] {context}\n[Your role] {}",
                self.config.role_for(&id)
            );
            let infer_clone = infer.clone();
            let answer = infer_clone(id.clone(), prompt).await;

            // Route to next agents via the bus.
            let targets = self.targets_for_topology(team, &id);
            for target in targets {
                let msg = AgentMessage::new(id.clone(), target, answer.clone());
                let _ = bus.send(msg);
            }

            context = answer;
        }

        Ok(context)
    }

    // ── Parallel consensus ────────────────────────────────────────────────────

    async fn run_parallel<F, Fut>(
        &self,
        team: &AgentTeam,
        bus: &AgentBus,
        infer: F,
    ) -> Result<String, AgentRuntimeError>
    where
        F: Fn(AgentId, String) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let task = team.task.clone();

        // Run all members concurrently.
        let mut handles = Vec::with_capacity(team.members.len());
        for member_id in &team.members {
            let id = member_id.clone();
            let role = self.config.role_for(&id);
            let prompt =
                format!("[Team task] {task}\n[Your role] {role}");
            let infer_clone = infer.clone();
            handles.push(tokio::spawn(async move {
                let answer = infer_clone(id.clone(), prompt).await;
                (id, answer)
            }));
        }

        let mut member_results: Vec<String> = Vec::with_capacity(team.members.len());
        for handle in handles {
            let (member_id, answer) = handle.await.map_err(|e| {
                AgentRuntimeError::AgentLoop(format!("team member panicked: {e}"))
            })?;
            // Send member answer to leader.
            let msg = AgentMessage::new(
                member_id.clone(),
                AgentTarget::Specific(team.leader.clone()),
                answer.clone(),
            );
            let _ = bus.send(msg);
            member_results.push(answer);
        }

        // Leader synthesises all member results.
        let synthesis_prompt = format!(
            "[Team task] {task}\n[Member answers]\n{}\n[Your role] leader — synthesise the above into a final answer.",
            member_results.join("\n")
        );
        let leader_answer = infer(team.leader.clone(), synthesis_prompt).await;

        // Broadcast leader's final answer.
        let _ = bus.send(AgentMessage::new(
            team.leader.clone(),
            AgentTarget::Broadcast,
            leader_answer.clone(),
        ));

        Ok(leader_answer)
    }

    // ── Topology helpers ──────────────────────────────────────────────────────

    /// Return the [`AgentTarget`]s that `sender` should route to given the
    /// configured topology.
    fn targets_for_topology(&self, team: &AgentTeam, sender: &AgentId) -> Vec<AgentTarget> {
        match self.config.topology {
            CommunicationTopology::Star => {
                // Members → leader; leader → all members.
                if sender == &team.leader {
                    team.members
                        .iter()
                        .map(|id| AgentTarget::Specific(id.clone()))
                        .collect()
                } else {
                    vec![AgentTarget::Specific(team.leader.clone())]
                }
            }
            CommunicationTopology::Mesh => {
                // Each agent broadcasts to all others.
                vec![AgentTarget::Broadcast]
            }
            CommunicationTopology::Ring => {
                // Find sender index in the all-agents list, send to next.
                let all = team.all_agents();
                let pos = all.iter().position(|id| *id == sender);
                match pos {
                    Some(i) => {
                        let next = all[(i + 1) % all.len()];
                        vec![AgentTarget::Specific(next.clone())]
                    }
                    None => vec![AgentTarget::Broadcast],
                }
            }
        }
    }
}

impl std::fmt::Debug for TeamOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamOrchestrator")
            .field("topology", &self.config.topology)
            .field("consensus", &self.config.consensus)
            .field("max_rounds", &self.config.max_rounds)
            .field("team_size", &self.config.team.size())
            .finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_team() -> AgentTeam {
        AgentTeam::new(
            AgentId::new("leader"),
            vec![AgentId::new("m1"), AgentId::new("m2")],
            "test task",
        )
    }

    #[test]
    fn test_agent_team_size() {
        let t = simple_team();
        assert_eq!(t.size(), 3);
    }

    #[test]
    fn test_agent_team_contains() {
        let t = simple_team();
        assert!(t.contains(&AgentId::new("leader")));
        assert!(t.contains(&AgentId::new("m1")));
        assert!(!t.contains(&AgentId::new("stranger")));
    }

    #[test]
    fn test_agent_team_all_agents_order() {
        let t = simple_team();
        let ids: Vec<&str> = t.all_agents().iter().map(|id| id.as_str()).collect();
        assert_eq!(ids, vec!["leader", "m1", "m2"]);
    }

    #[test]
    fn test_agent_team_is_solo() {
        let solo = AgentTeam::new(AgentId::new("alone"), vec![], "task");
        assert!(solo.is_solo());
        assert!(!simple_team().is_solo());
    }

    #[test]
    fn test_agent_team_member_at() {
        let t = simple_team();
        assert_eq!(t.member_at(0).map(|id| id.as_str()), Some("m1"));
        assert_eq!(t.member_at(1).map(|id| id.as_str()), Some("m2"));
        assert!(t.member_at(2).is_none());
    }

    #[test]
    fn test_team_config_defaults() {
        let cfg = TeamConfig::new(simple_team());
        assert_eq!(cfg.topology, CommunicationTopology::Star);
        assert_eq!(cfg.consensus, ConsensusStrategy::Parallel);
        assert_eq!(cfg.max_rounds, 3);
        assert_eq!(cfg.bus_capacity, 256);
    }

    #[test]
    fn test_team_config_builder() {
        let cfg = TeamConfig::new(simple_team())
            .with_topology(CommunicationTopology::Mesh)
            .with_consensus(ConsensusStrategy::Majority)
            .with_max_rounds(7)
            .with_bus_capacity(512);
        assert_eq!(cfg.topology, CommunicationTopology::Mesh);
        assert_eq!(cfg.consensus, ConsensusStrategy::Majority);
        assert_eq!(cfg.max_rounds, 7);
        assert_eq!(cfg.bus_capacity, 512);
    }

    #[test]
    fn test_team_config_role_for() {
        let cfg = TeamConfig::new(simple_team());
        assert_eq!(cfg.role_for(&AgentId::new("leader")), "leader");
        assert_eq!(cfg.role_for(&AgentId::new("m1")), "member");
    }

    #[test]
    fn test_team_config_role_override() {
        let cfg = TeamConfig::new(simple_team())
            .with_role(AgentId::new("m1"), "executor");
        assert_eq!(cfg.role_for(&AgentId::new("m1")), "executor");
    }

    #[test]
    fn test_team_config_max_rounds_minimum_one() {
        let cfg = TeamConfig::new(simple_team()).with_max_rounds(0);
        assert_eq!(cfg.max_rounds, 1);
    }

    #[test]
    fn test_consensus_strategy_display() {
        assert_eq!(ConsensusStrategy::Majority.to_string(), "majority");
        assert_eq!(ConsensusStrategy::Pipeline.to_string(), "pipeline");
        assert_eq!(ConsensusStrategy::Parallel.to_string(), "parallel");
    }

    #[test]
    fn test_topology_display() {
        assert_eq!(CommunicationTopology::Star.to_string(), "star");
        assert_eq!(CommunicationTopology::Mesh.to_string(), "mesh");
        assert_eq!(CommunicationTopology::Ring.to_string(), "ring");
    }

    #[test]
    fn test_orchestrator_debug() {
        let orch = TeamOrchestrator::new(TeamConfig::new(simple_team()));
        let s = format!("{orch:?}");
        assert!(s.contains("TeamOrchestrator"));
    }

    #[tokio::test]
    async fn test_parallel_consensus_run() {
        let team = AgentTeam::new(
            AgentId::new("leader"),
            vec![AgentId::new("m1"), AgentId::new("m2")],
            "What is 2+2?",
        );
        let config = TeamConfig::new(team).with_consensus(ConsensusStrategy::Parallel);
        let orch = TeamOrchestrator::new(config);

        let result = orch
            .run(|_id, _prompt| async { "4".to_string() })
            .await
            .unwrap();

        // Leader synthesises, so final answer comes from leader's infer call.
        assert_eq!(result.final_answer, "4");
        assert_eq!(result.consensus_used, "parallel");
    }

    #[tokio::test]
    async fn test_pipeline_consensus_run() {
        let team = AgentTeam::new(
            AgentId::new("leader"),
            vec![AgentId::new("m1")],
            "task",
        );
        let config = TeamConfig::new(team).with_consensus(ConsensusStrategy::Pipeline);
        let orch = TeamOrchestrator::new(config);

        let result = orch
            .run(|id, _prompt| {
                let answer = format!("{id}-done");
                async move { answer }
            })
            .await
            .unwrap();

        // Pipeline: leader runs last, so its answer wins.
        assert!(result.final_answer.contains("leader-done"));
        assert_eq!(result.consensus_used, "pipeline");
    }

    #[tokio::test]
    async fn test_majority_consensus_pick_most_frequent() {
        let team = AgentTeam::new(
            AgentId::new("leader"),
            vec![AgentId::new("m1"), AgentId::new("m2")],
            "task",
        );
        let config = TeamConfig::new(team).with_consensus(ConsensusStrategy::Majority);
        let orch = TeamOrchestrator::new(config);

        // leader → "yes", m1 → "no", m2 → "yes"  →  "yes" wins
        let result = orch
            .run(|id, _prompt| {
                let answer = if id.as_str() == "m1" { "no" } else { "yes" };
                async move { answer.to_string() }
            })
            .await
            .unwrap();

        assert_eq!(result.final_answer, "yes");
        assert_eq!(result.consensus_used, "majority");
    }

    #[tokio::test]
    async fn test_solo_team_parallel() {
        let team = AgentTeam::new(AgentId::new("solo"), vec![], "solo task");
        let config = TeamConfig::new(team).with_consensus(ConsensusStrategy::Parallel);
        let orch = TeamOrchestrator::new(config);

        let result = orch
            .run(|_id, _prompt| async { "solo answer".to_string() })
            .await
            .unwrap();

        assert_eq!(result.final_answer, "solo answer");
    }

    #[test]
    fn test_team_result_distinct_answer_count() {
        let mut agent_answers = HashMap::new();
        agent_answers.insert(AgentId::new("a"), "yes".to_string());
        agent_answers.insert(AgentId::new("b"), "no".to_string());
        agent_answers.insert(AgentId::new("c"), "yes".to_string());
        let result = TeamResult {
            final_answer: "yes".to_string(),
            agent_answers,
            consensus_used: "majority".to_string(),
            rounds_completed: 1,
            duration_ms: 0,
            topology_used: "star".to_string(),
            messages_exchanged: 0,
        };
        assert_eq!(result.distinct_answer_count(), 2);
        assert!(!result.is_unanimous());
    }

    #[test]
    fn test_team_result_is_unanimous() {
        let mut agent_answers = HashMap::new();
        agent_answers.insert(AgentId::new("a"), "42".to_string());
        agent_answers.insert(AgentId::new("b"), "42".to_string());
        let result = TeamResult {
            final_answer: "42".to_string(),
            agent_answers,
            consensus_used: "majority".to_string(),
            rounds_completed: 1,
            duration_ms: 0,
            topology_used: "star".to_string(),
            messages_exchanged: 0,
        };
        assert!(result.is_unanimous());
    }
}
