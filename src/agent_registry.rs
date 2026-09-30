//! # Agent Registry
//!
//! A capability-based registry of available agents.
//!
//! Agents register their [`AgentMetadata`] with an [`AgentRegistry`], report
//! periodic heartbeats to update their status, and can be discovered by
//! callers that require specific [`AgentCapability`] combinations.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AgentCapability
// ---------------------------------------------------------------------------

/// A discrete capability that an agent can advertise.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentCapability {
    /// Generate natural-language text.
    TextGeneration,
    /// Execute arbitrary code in a sandboxed environment.
    CodeExecution,
    /// Issue live queries to the web.
    WebSearch,
    /// Read and write files on the local file-system.
    FileSystem,
    /// Query structured databases.
    DatabaseQuery,
    /// Classify or describe image content.
    ImageAnalysis,
    /// Transcribe, synthesise, or analyse audio.
    AudioProcessing,
    /// Call external tools / functions.
    ToolUse,
    /// Process combinations of text, images, audio, etc.
    MultiModal,
}

impl AgentCapability {
    /// Returns a short human-readable description of this capability.
    pub fn description(&self) -> &str {
        match self {
            Self::TextGeneration => "Generate natural-language text",
            Self::CodeExecution => "Execute code in a sandboxed environment",
            Self::WebSearch => "Issue live queries to the web",
            Self::FileSystem => "Read and write local files",
            Self::DatabaseQuery => "Query structured databases",
            Self::ImageAnalysis => "Classify or describe image content",
            Self::AudioProcessing => "Transcribe, synthesise, or analyse audio",
            Self::ToolUse => "Call external tools or functions",
            Self::MultiModal => "Process combinations of modalities",
        }
    }
}

// ---------------------------------------------------------------------------
// AgentMetadata
// ---------------------------------------------------------------------------

/// Static description of an agent as supplied at registration time.
#[derive(Debug, Clone)]
pub struct AgentMetadata {
    /// Unique identifier for this agent (caller-supplied or auto-generated).
    pub agent_id: String,
    /// Human-readable display name.
    pub name: String,
    /// Semantic version string (e.g. `"1.2.3"`).
    pub version: String,
    /// List of capabilities this agent supports.
    pub capabilities: Vec<AgentCapability>,
    /// Maximum number of tasks the agent can handle concurrently.
    pub max_concurrent_tasks: u32,
    /// Moving-average response time in milliseconds.
    pub avg_response_ms: u64,
    /// Historical reliability score in `[0.0, 1.0]`.  Higher = more reliable.
    pub reliability_score: f64,
}

// ---------------------------------------------------------------------------
// AgentStatus
// ---------------------------------------------------------------------------

/// Current operational status of a registered agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    /// Ready to accept new tasks.
    Available,
    /// Currently handling `tasks` tasks.
    Busy(u32),
    /// Unreachable or shut down.
    Offline,
    /// Temporarily unavailable for scheduled maintenance.
    Maintenance,
}

impl AgentStatus {
    /// Returns `true` if the agent can accept additional work.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available | Self::Busy(_))
    }
}

// ---------------------------------------------------------------------------
// AgentRegistration
// ---------------------------------------------------------------------------

/// The registry entry for a single agent.
#[derive(Debug, Clone)]
pub struct AgentRegistration {
    /// Static agent description.
    pub metadata: AgentMetadata,
    /// Live operational status.
    pub status: AgentStatus,
    /// Monotonic timestamp (ms) when the agent registered.
    pub registered_at: u64,
    /// Monotonic timestamp (ms) of the most recent heartbeat.
    pub last_heartbeat: u64,
}

// ---------------------------------------------------------------------------
// AgentRegistry
// ---------------------------------------------------------------------------

/// In-process registry of agents, indexed by `agent_id`.
#[derive(Default)]
pub struct AgentRegistry {
    agents: HashMap<String, AgentRegistration>,
}

impl AgentRegistry {
    /// Create an empty [`AgentRegistry`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent and return its `agent_id`.
    ///
    /// If an agent with the same `agent_id` already exists it is replaced.
    pub fn register(&mut self, metadata: AgentMetadata) -> String {
        let id = metadata.agent_id.clone();
        let now = 0u64; // callers should pass `now` via heartbeat; use 0 as sentinel
        self.agents.insert(
            id.clone(),
            AgentRegistration {
                metadata,
                status: AgentStatus::Available,
                registered_at: now,
                last_heartbeat: now,
            },
        );
        id
    }

    /// Register an agent with an explicit registration timestamp.
    pub fn register_at(&mut self, metadata: AgentMetadata, now_ms: u64) -> String {
        let id = metadata.agent_id.clone();
        self.agents.insert(
            id.clone(),
            AgentRegistration {
                metadata,
                status: AgentStatus::Available,
                registered_at: now_ms,
                last_heartbeat: now_ms,
            },
        );
        id
    }

    /// Remove an agent from the registry.
    ///
    /// Returns `true` if the agent was present and has been removed.
    pub fn deregister(&mut self, agent_id: &str) -> bool {
        self.agents.remove(agent_id).is_some()
    }

    /// Update an agent's status and heartbeat timestamp.
    ///
    /// Returns `true` on success, `false` if the `agent_id` is unknown.
    pub fn heartbeat(&mut self, agent_id: &str, status: AgentStatus, now: u64) -> bool {
        if let Some(reg) = self.agents.get_mut(agent_id) {
            reg.status = status;
            reg.last_heartbeat = now;
            true
        } else {
            false
        }
    }

    // ------------------------------------------------------------------
    // Discovery
    // ------------------------------------------------------------------

    /// Return all registered agents that advertise `cap`.
    pub fn find_by_capability(&self, cap: &AgentCapability) -> Vec<&AgentRegistration> {
        self.agents
            .values()
            .filter(|r| r.metadata.capabilities.contains(cap))
            .collect()
    }

    /// Find the single best agent that supports **all** `required_caps`.
    ///
    /// Selection criterion: `reliability_score × availability_factor` where
    /// `availability_factor` is `1.0` for `Available`, `0.5` for `Busy`, and
    /// `0.0` for `Offline`/`Maintenance`.
    pub fn find_best_for_task(
        &self,
        required_caps: &[AgentCapability],
    ) -> Option<&AgentRegistration> {
        self.agents
            .values()
            .filter(|r| {
                required_caps
                    .iter()
                    .all(|c| r.metadata.capabilities.contains(c))
            })
            .max_by(|a, b| {
                let score_a = composite_score(a);
                let score_b = composite_score(b);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    // ------------------------------------------------------------------
    // Housekeeping
    // ------------------------------------------------------------------

    /// Return the `agent_id`s of agents whose `last_heartbeat` is more than
    /// `timeout_ms` milliseconds before `now`.
    pub fn stale_agents(&self, timeout_ms: u64, now: u64) -> Vec<String> {
        self.agents
            .values()
            .filter(|r| now.saturating_sub(r.last_heartbeat) > timeout_ms)
            .map(|r| r.metadata.agent_id.clone())
            .collect()
    }

    /// Returns a map of `agent_id → (active_tasks, max_tasks)`.
    pub fn capacity_report(&self) -> HashMap<String, (u32, u32)> {
        self.agents
            .values()
            .map(|r| {
                let active = match &r.status {
                    AgentStatus::Busy(n) => *n,
                    _ => 0,
                };
                (
                    r.metadata.agent_id.clone(),
                    (active, r.metadata.max_concurrent_tasks),
                )
            })
            .collect()
    }
}

/// Composite score used for `find_best_for_task`.
fn composite_score(reg: &AgentRegistration) -> f64 {
    let availability = match &reg.status {
        AgentStatus::Available => 1.0,
        AgentStatus::Busy(_) => 0.5,
        AgentStatus::Offline | AgentStatus::Maintenance => 0.0,
    };
    reg.metadata.reliability_score * availability
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_meta(id: &str, caps: Vec<AgentCapability>, reliability: f64) -> AgentMetadata {
        AgentMetadata {
            agent_id: id.to_string(),
            name: id.to_string(),
            version: "1.0.0".to_string(),
            capabilities: caps,
            max_concurrent_tasks: 10,
            avg_response_ms: 50,
            reliability_score: reliability,
        }
    }

    #[test]
    fn register_returns_agent_id() {
        let mut r = AgentRegistry::new();
        let id = r.register(make_meta("a1", vec![AgentCapability::TextGeneration], 0.9));
        assert_eq!(id, "a1");
    }

    #[test]
    fn deregister_removes_agent() {
        let mut r = AgentRegistry::new();
        r.register(make_meta("a1", vec![], 0.9));
        assert!(r.deregister("a1"));
        assert!(!r.deregister("a1")); // second call → false
    }

    #[test]
    fn heartbeat_updates_status() {
        let mut r = AgentRegistry::new();
        r.register_at(make_meta("a1", vec![], 1.0), 0);
        r.heartbeat("a1", AgentStatus::Busy(3), 1_000);
        let reg = r.agents.get("a1").unwrap();
        assert_eq!(reg.status, AgentStatus::Busy(3));
        assert_eq!(reg.last_heartbeat, 1_000);
    }

    #[test]
    fn heartbeat_unknown_agent_returns_false() {
        let mut r = AgentRegistry::new();
        assert!(!r.heartbeat("ghost", AgentStatus::Available, 0));
    }

    #[test]
    fn find_by_capability_returns_matching() {
        let mut r = AgentRegistry::new();
        r.register(make_meta(
            "a1",
            vec![AgentCapability::TextGeneration],
            0.9,
        ));
        r.register(make_meta("a2", vec![AgentCapability::CodeExecution], 0.8));
        let found = r.find_by_capability(&AgentCapability::TextGeneration);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].metadata.agent_id, "a1");
    }

    #[test]
    fn find_best_for_task_selects_highest_reliability() {
        let mut r = AgentRegistry::new();
        r.register(make_meta(
            "low",
            vec![AgentCapability::TextGeneration],
            0.5,
        ));
        r.register(make_meta(
            "high",
            vec![AgentCapability::TextGeneration],
            0.95,
        ));
        let best = r
            .find_best_for_task(&[AgentCapability::TextGeneration])
            .unwrap();
        assert_eq!(best.metadata.agent_id, "high");
    }

    #[test]
    fn find_best_requires_all_caps() {
        let mut r = AgentRegistry::new();
        r.register(make_meta(
            "partial",
            vec![AgentCapability::TextGeneration],
            0.99,
        ));
        r.register(make_meta(
            "full",
            vec![
                AgentCapability::TextGeneration,
                AgentCapability::CodeExecution,
            ],
            0.80,
        ));
        let best = r
            .find_best_for_task(&[
                AgentCapability::TextGeneration,
                AgentCapability::CodeExecution,
            ])
            .unwrap();
        assert_eq!(best.metadata.agent_id, "full");
    }

    #[test]
    fn stale_agents_detects_old_heartbeats() {
        let mut r = AgentRegistry::new();
        r.register_at(make_meta("old", vec![], 1.0), 0);
        r.register_at(make_meta("fresh", vec![], 1.0), 9_000);
        let stale = r.stale_agents(5_000, 10_000);
        assert_eq!(stale, vec!["old"]);
    }

    #[test]
    fn capacity_report_reflects_busy_count() {
        let mut r = AgentRegistry::new();
        r.register(make_meta("a1", vec![], 1.0));
        r.heartbeat("a1", AgentStatus::Busy(4), 0);
        let report = r.capacity_report();
        let (active, max) = report["a1"];
        assert_eq!(active, 4);
        assert_eq!(max, 10);
    }

    #[test]
    fn agent_status_is_available_variants() {
        assert!(AgentStatus::Available.is_available());
        assert!(AgentStatus::Busy(1).is_available());
        assert!(!AgentStatus::Offline.is_available());
        assert!(!AgentStatus::Maintenance.is_available());
    }

    #[test]
    fn capability_descriptions_are_non_empty() {
        let caps = [
            AgentCapability::TextGeneration,
            AgentCapability::CodeExecution,
            AgentCapability::WebSearch,
            AgentCapability::FileSystem,
            AgentCapability::DatabaseQuery,
            AgentCapability::ImageAnalysis,
            AgentCapability::AudioProcessing,
            AgentCapability::ToolUse,
            AgentCapability::MultiModal,
        ];
        for cap in &caps {
            assert!(!cap.description().is_empty(), "{cap:?} has empty description");
        }
    }
}
