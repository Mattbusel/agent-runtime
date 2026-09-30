//! # Agent Supervisor
//!
//! Fault-tolerant agent supervision with restart strategies.
//!
//! Modelled after Erlang/OTP supervisor semantics:
//!
//! - **OneForOne** — only the failed agent is restarted.
//! - **OneForAll** — all supervised agents are restarted.
//! - **RestForOne** — the failed agent plus all agents registered *after* it
//!   are restarted.
//!
//! Restart rate limiting prevents crash loops: if an agent exceeds
//! `max_restarts` within the configured time window the supervisor escalates
//! instead of restarting.

use std::collections::{HashMap, VecDeque};

// ── RestartStrategy ──────────────────────────────────────────────────────────

/// Determines which agents are restarted when one fails.
#[derive(Debug, Clone, PartialEq)]
pub enum RestartStrategy {
    /// Restart only the failed agent.
    OneForOne,
    /// Restart all supervised agents.
    OneForAll,
    /// Restart the failed agent and all agents registered after it.
    RestForOne,
}

// ── RestartLimit ─────────────────────────────────────────────────────────────

/// Caps the allowed restart rate to prevent crash loops.
#[derive(Debug, Clone)]
pub struct RestartLimit {
    /// Maximum number of restarts permitted within `within_ms`.
    pub max_restarts: u32,
    /// Sliding time window in milliseconds over which `max_restarts` is counted.
    pub within_ms: u64,
}

impl RestartLimit {
    /// Create a new `RestartLimit`.
    pub fn new(max_restarts: u32, within_ms: u64) -> Self {
        Self { max_restarts, within_ms }
    }
}

// ── AgentState ───────────────────────────────────────────────────────────────

/// Lifecycle state of a supervised agent.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    /// The agent is initialising.
    Starting,
    /// The agent is operating normally.
    Running,
    /// The agent is shutting down gracefully.
    Stopping,
    /// The agent has stopped cleanly.
    Stopped,
    /// The agent has failed with the given reason string.
    Failed(String),
}

// ── RestartDecision ──────────────────────────────────────────────────────────

/// The supervisor's verdict after an agent reports failure.
#[derive(Debug, Clone, PartialEq)]
pub enum RestartDecision {
    /// Restart the listed agents after `after_ms` milliseconds.
    Restart {
        /// Delay before restarting, in milliseconds.
        after_ms: u64,
        /// IDs of agents that should be restarted.
        agents_to_restart: Vec<String>,
    },
    /// Restart limit exceeded — escalate to the parent supervisor or operator.
    Escalate {
        /// Human-readable reason for escalation.
        reason: String,
    },
    /// No action required (e.g., agent was already stopped intentionally).
    Ignore,
}

// ── SupervisedAgent ──────────────────────────────────────────────────────────

/// Internal record for a single supervised agent.
#[derive(Debug, Clone)]
pub struct SupervisedAgent {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Current lifecycle state.
    pub state: AgentState,
    /// Total number of times this agent has been restarted.
    pub restart_count: u32,
    /// Wall-clock milliseconds of the most recent restart, if any.
    pub last_restart_ms: Option<u64>,
    /// Ring-buffer of restart timestamps (milliseconds) used for rate-limit checks.
    pub restart_times: VecDeque<u64>,
}

impl SupervisedAgent {
    fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            state: AgentState::Starting,
            restart_count: 0,
            last_restart_ms: None,
            restart_times: VecDeque::new(),
        }
    }
}

// ── SupervisorConfig ─────────────────────────────────────────────────────────

/// Configuration for an [`AgentSupervisor`].
#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    /// Restart strategy applied when an agent fails.
    pub strategy: RestartStrategy,
    /// Rate limit that gates whether a restart is allowed.
    pub restart_limit: RestartLimit,
    /// Maximum milliseconds allowed for an agent to transition from
    /// `Starting` to `Running` before it is considered stuck.
    pub init_timeout_ms: u64,
}

impl SupervisorConfig {
    /// Create a new `SupervisorConfig`.
    pub fn new(strategy: RestartStrategy, restart_limit: RestartLimit, init_timeout_ms: u64) -> Self {
        Self { strategy, restart_limit, init_timeout_ms }
    }
}

// ── AgentSupervisor ──────────────────────────────────────────────────────────

/// Fault-tolerant supervisor for a group of agents.
pub struct AgentSupervisor {
    /// Supervisor configuration.
    pub config: SupervisorConfig,
    /// All supervised agents, keyed by ID.
    pub agents: HashMap<String, SupervisedAgent>,
    /// Unique identifier for this supervisor instance.
    pub supervisor_id: String,
    /// Ordered registration list (used for RestForOne strategy).
    registration_order: Vec<String>,
}

impl AgentSupervisor {
    /// Create a new `AgentSupervisor` with the given ID and configuration.
    pub fn new(id: &str, config: SupervisorConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            supervisor_id: id.to_string(),
            registration_order: Vec::new(),
        }
    }

    /// Register a new agent.  The agent starts in [`AgentState::Starting`].
    pub fn add_agent(&mut self, agent_id: &str, name: &str) {
        self.agents
            .insert(agent_id.to_string(), SupervisedAgent::new(agent_id, name));
        self.registration_order.push(agent_id.to_string());
    }

    /// Transition an agent to [`AgentState::Running`].
    ///
    /// Returns `true` if the transition succeeded, `false` if the agent was
    /// not found or was not in `Starting` state.
    pub fn mark_running(&mut self, agent_id: &str, _now_ms: u64) -> bool {
        match self.agents.get_mut(agent_id) {
            Some(agent) if agent.state == AgentState::Starting => {
                agent.state = AgentState::Running;
                true
            }
            _ => false,
        }
    }

    /// Report that `agent_id` has failed with `reason`.
    ///
    /// Returns the supervisor's [`RestartDecision`].
    pub fn report_failure(
        &mut self,
        agent_id: &str,
        reason: String,
        now_ms: u64,
    ) -> RestartDecision {
        let agent = match self.agents.get_mut(agent_id) {
            Some(a) => a,
            None => return RestartDecision::Ignore,
        };
        agent.state = AgentState::Failed(reason.clone());

        // Clone what we need before releasing the mutable borrow on `agents`.
        let should = self.should_restart(self.agents.get(agent_id).unwrap_or_else(|| {
            // Safety: we just set the state above, so it must exist.
            unreachable!()
        }), now_ms);

        if !should {
            return RestartDecision::Escalate {
                reason: format!(
                    "agent '{}' exceeded restart limit: {}",
                    agent_id, reason
                ),
            };
        }

        let targets = self.agents_to_restart_for(agent_id);
        self.apply_restart(&targets.clone(), now_ms);
        RestartDecision::Restart {
            after_ms: 0,
            agents_to_restart: targets,
        }
    }

    /// Return `true` if `agent` is allowed to restart given the current rate window.
    pub fn should_restart(&self, agent: &SupervisedAgent, now_ms: u64) -> bool {
        let window_start = now_ms.saturating_sub(self.config.restart_limit.within_ms);
        let recent = agent
            .restart_times
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u32;
        recent < self.config.restart_limit.max_restarts
    }

    /// Return the list of agent IDs that should be restarted for a given
    /// failed agent, according to the configured strategy.
    pub fn agents_to_restart_for(&self, failed_id: &str) -> Vec<String> {
        match self.config.strategy {
            RestartStrategy::OneForOne => vec![failed_id.to_string()],
            RestartStrategy::OneForAll => self.registration_order.clone(),
            RestartStrategy::RestForOne => {
                let pos = self
                    .registration_order
                    .iter()
                    .position(|id| id == failed_id)
                    .unwrap_or(0);
                self.registration_order[pos..].to_vec()
            }
        }
    }

    /// Transition the given agents back to `Starting` and record the restart timestamp.
    pub fn apply_restart(&mut self, agents: &[String], now_ms: u64) {
        for id in agents {
            if let Some(agent) = self.agents.get_mut(id) {
                agent.state = AgentState::Starting;
                agent.restart_count += 1;
                agent.last_restart_ms = Some(now_ms);
                agent.restart_times.push_back(now_ms);
            }
        }
    }

    /// Return the IDs of all agents currently in `Running` state.
    pub fn running_agents(&self) -> Vec<&str> {
        self.agents
            .values()
            .filter(|a| a.state == AgentState::Running)
            .map(|a| a.id.as_str())
            .collect()
    }

    /// Return `(id, reason)` pairs for all agents currently in `Failed` state.
    pub fn failed_agents(&self) -> Vec<(&str, &str)> {
        self.agents
            .values()
            .filter_map(|a| {
                if let AgentState::Failed(ref r) = a.state {
                    Some((a.id.as_str(), r.as_str()))
                } else {
                    None
                }
            })
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn default_supervisor(strategy: RestartStrategy) -> AgentSupervisor {
        AgentSupervisor::new(
            "sup-1",
            SupervisorConfig::new(
                strategy,
                RestartLimit::new(3, 10_000),
                5_000,
            ),
        )
    }

    #[test]
    fn mark_running_transitions_state() {
        let mut sup = default_supervisor(RestartStrategy::OneForOne);
        sup.add_agent("a1", "Alpha");
        assert!(sup.mark_running("a1", 0));
        assert_eq!(sup.agents["a1"].state, AgentState::Running);
        assert_eq!(sup.running_agents(), vec!["a1"]);
    }

    #[test]
    fn failure_triggers_restart() {
        let mut sup = default_supervisor(RestartStrategy::OneForOne);
        sup.add_agent("a1", "Alpha");
        sup.mark_running("a1", 0);
        let decision = sup.report_failure("a1", "crash".into(), 100);
        assert!(matches!(decision, RestartDecision::Restart { .. }));
        if let RestartDecision::Restart { agents_to_restart, .. } = decision {
            assert_eq!(agents_to_restart, vec!["a1"]);
        }
        assert_eq!(sup.agents["a1"].state, AgentState::Starting);
    }

    #[test]
    fn restart_limit_triggers_escalate() {
        let mut sup = AgentSupervisor::new(
            "sup-2",
            SupervisorConfig::new(
                RestartStrategy::OneForOne,
                // Allow only 1 restart within the window.
                RestartLimit::new(1, 10_000),
                5_000,
            ),
        );
        sup.add_agent("a1", "Alpha");
        sup.mark_running("a1", 0);

        // First failure — should restart.
        let d1 = sup.report_failure("a1", "crash1".into(), 100);
        assert!(matches!(d1, RestartDecision::Restart { .. }));

        // Second failure within window — should escalate.
        sup.mark_running("a1", 101);
        let d2 = sup.report_failure("a1", "crash2".into(), 200);
        assert!(matches!(d2, RestartDecision::Escalate { .. }));
    }

    #[test]
    fn one_for_all_restarts_all() {
        let mut sup = default_supervisor(RestartStrategy::OneForAll);
        sup.add_agent("a1", "Alpha");
        sup.add_agent("a2", "Beta");
        sup.add_agent("a3", "Gamma");
        sup.mark_running("a1", 0);
        sup.mark_running("a2", 0);
        sup.mark_running("a3", 0);

        let decision = sup.report_failure("a2", "boom".into(), 100);
        if let RestartDecision::Restart { agents_to_restart, .. } = decision {
            let mut got = agents_to_restart.clone();
            got.sort();
            assert_eq!(got, vec!["a1", "a2", "a3"]);
        } else {
            panic!("expected Restart, got {decision:?}");
        }
    }

    #[test]
    fn rest_for_one_restarts_failed_and_later() {
        let mut sup = default_supervisor(RestartStrategy::RestForOne);
        sup.add_agent("a1", "Alpha");
        sup.add_agent("a2", "Beta");
        sup.add_agent("a3", "Gamma");
        sup.mark_running("a1", 0);
        sup.mark_running("a2", 0);
        sup.mark_running("a3", 0);

        // Fail a2 — should restart a2 and a3 (registered after a2).
        let decision = sup.report_failure("a2", "boom".into(), 100);
        if let RestartDecision::Restart { agents_to_restart, .. } = decision {
            assert_eq!(agents_to_restart, vec!["a2", "a3"]);
        } else {
            panic!("expected Restart, got {decision:?}");
        }
        // a1 should still be running.
        assert_eq!(sup.agents["a1"].state, AgentState::Running);
    }
}
