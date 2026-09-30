//! Agent capability advertising, discovery, and matching.
//!
//! Provides a registry for agents to advertise their capabilities,
//! and for other agents to discover and query for matching agents.

use std::collections::HashMap;

// ── Capability kind ──────────────────────────────────────────────────────────

/// The broad functional category of an agent capability.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CapabilityKind {
    /// Data ingestion, transformation, or analysis capability.
    DataProcessing,
    /// Web search and retrieval capability.
    WebSearch,
    /// Code execution or compilation capability.
    CodeExecution,
    /// File system read/write capability.
    FileSystem,
    /// Email, chat, or other communication capability.
    Communication,
    /// Machine learning inference or training capability.
    MachineLearning,
    /// A user-defined capability kind.
    Custom(String),
}

impl CapabilityKind {
    /// Returns a stable string key for use in histograms.
    fn as_key(&self) -> String {
        match self {
            CapabilityKind::DataProcessing => "DataProcessing".into(),
            CapabilityKind::WebSearch => "WebSearch".into(),
            CapabilityKind::CodeExecution => "CodeExecution".into(),
            CapabilityKind::FileSystem => "FileSystem".into(),
            CapabilityKind::Communication => "Communication".into(),
            CapabilityKind::MachineLearning => "MachineLearning".into(),
            CapabilityKind::Custom(s) => format!("Custom({})", s),
        }
    }
}

// ── Capability ───────────────────────────────────────────────────────────────

/// A single, versioned capability advertised by an agent.
#[derive(Debug, Clone)]
pub struct Capability {
    /// Unique identifier for this capability instance.
    pub id: String,
    /// The broad functional category.
    pub kind: CapabilityKind,
    /// Human-readable name.
    pub name: String,
    /// Semantic version as `(major, minor, patch)`.
    pub version: (u32, u32, u32),
    /// Arbitrary tags for filtering (e.g. `"gpu"`, `"fast"`).
    pub tags: Vec<String>,
    /// Key-value constraints (e.g. `"max_tokens" -> "4096"`).
    pub constraints: HashMap<String, String>,
}

// ── AgentCapabilities ────────────────────────────────────────────────────────

/// All capabilities advertised by a single agent.
#[derive(Debug, Clone)]
pub struct AgentCapabilities {
    /// The agent identifier.
    pub agent_id: String,
    /// The capabilities this agent advertises.
    pub capabilities: Vec<Capability>,
    /// Unix-epoch milliseconds when this record was first registered.
    pub registered_at_ms: u64,
    /// Unix-epoch milliseconds of the most recent heartbeat.
    pub last_heartbeat_ms: u64,
}

// ── CapabilityQuery ──────────────────────────────────────────────────────────

/// Criteria used to search the registry for matching agents.
#[derive(Debug, Clone, Default)]
pub struct CapabilityQuery {
    /// If set, only match agents that have at least one capability of this kind.
    pub required_kind: Option<CapabilityKind>,
    /// All of these tags must be present on at least one matching capability.
    pub required_tags: Vec<String>,
    /// The matched capability version must be `>=` this value (lex order on `(major,minor,patch)`).
    pub min_version: Option<(u32, u32, u32)>,
    /// All of these key-value pairs must appear in the matched capability's constraints.
    pub constraints_match: HashMap<String, String>,
}

// ── CapabilityRegistry ───────────────────────────────────────────────────────

/// Central registry of all agent capabilities.
#[derive(Debug, Default)]
pub struct CapabilityRegistry {
    /// Maps agent-id to its capability record.
    pub agents: HashMap<String, AgentCapabilities>,
}

impl CapabilityRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Register or replace an agent's capability set.
    ///
    /// Both `registered_at_ms` and `last_heartbeat_ms` are set to `timestamp_ms`.
    pub fn register(
        &mut self,
        agent_id: String,
        capabilities: Vec<Capability>,
        timestamp_ms: u64,
    ) {
        self.agents.insert(
            agent_id.clone(),
            AgentCapabilities {
                agent_id,
                capabilities,
                registered_at_ms: timestamp_ms,
                last_heartbeat_ms: timestamp_ms,
            },
        );
    }

    /// Update the heartbeat timestamp for an existing agent.
    ///
    /// Returns `true` if the agent was found and updated, `false` otherwise.
    pub fn heartbeat(&mut self, agent_id: &str, timestamp_ms: u64) -> bool {
        if let Some(entry) = self.agents.get_mut(agent_id) {
            entry.last_heartbeat_ms = timestamp_ms;
            true
        } else {
            false
        }
    }

    /// Remove an agent from the registry.
    ///
    /// Returns `true` if the agent existed and was removed.
    pub fn deregister(&mut self, agent_id: &str) -> bool {
        self.agents.remove(agent_id).is_some()
    }

    /// Return the IDs of all agents that have at least one capability satisfying *all* query criteria.
    pub fn find_agents(&self, query: &CapabilityQuery) -> Vec<&str> {
        self.agents
            .values()
            .filter(|ac| {
                ac.capabilities.iter().any(|cap| {
                    capability_matches(cap, query)
                })
            })
            .map(|ac| ac.agent_id.as_str())
            .collect()
    }

    /// Return the capability record for a specific agent, if registered.
    pub fn capabilities_of(&self, agent_id: &str) -> Option<&AgentCapabilities> {
        self.agents.get(agent_id)
    }

    /// Return the IDs of all agents that have at least one capability of the given kind.
    pub fn agents_with_kind(&self, kind: &CapabilityKind) -> Vec<&str> {
        self.agents
            .values()
            .filter(|ac| ac.capabilities.iter().any(|c| &c.kind == kind))
            .map(|ac| ac.agent_id.as_str())
            .collect()
    }

    /// Return the IDs of agents whose `last_heartbeat_ms` is older than `now_ms - max_age_ms`.
    pub fn stale_agents(&self, now_ms: u64, max_age_ms: u64) -> Vec<&str> {
        let threshold = now_ms.saturating_sub(max_age_ms);
        self.agents
            .values()
            .filter(|ac| ac.last_heartbeat_ms < threshold)
            .map(|ac| ac.agent_id.as_str())
            .collect()
    }

    /// Remove all stale agents and return the number removed.
    pub fn evict_stale(&mut self, now_ms: u64, max_age_ms: u64) -> usize {
        let threshold = now_ms.saturating_sub(max_age_ms);
        let before = self.agents.len();
        self.agents
            .retain(|_, ac| ac.last_heartbeat_ms >= threshold);
        before - self.agents.len()
    }

    /// Produce an aggregate summary of the registry.
    pub fn summary(&self) -> RegistrySummary {
        let total_agents = self.agents.len();
        let mut total_capabilities = 0usize;
        let mut kinds_histogram: HashMap<String, usize> = HashMap::new();

        for ac in self.agents.values() {
            total_capabilities += ac.capabilities.len();
            for cap in &ac.capabilities {
                *kinds_histogram.entry(cap.kind.as_key()).or_insert(0) += 1;
            }
        }

        RegistrySummary {
            total_agents,
            total_capabilities,
            kinds_histogram,
        }
    }
}

/// Returns `true` when `cap` satisfies all criteria in `query`.
fn capability_matches(cap: &Capability, query: &CapabilityQuery) -> bool {
    // Kind filter
    if let Some(ref required_kind) = query.required_kind {
        if &cap.kind != required_kind {
            return false;
        }
    }
    // Version filter (lex order on (major, minor, patch))
    if let Some(min_ver) = query.min_version {
        if cap.version < min_ver {
            return false;
        }
    }
    // Tag filter — all required tags must be present
    for tag in &query.required_tags {
        if !cap.tags.contains(tag) {
            return false;
        }
    }
    // Constraint filter — all required key-value pairs must match
    for (k, v) in &query.constraints_match {
        if cap.constraints.get(k).map(String::as_str) != Some(v.as_str()) {
            return false;
        }
    }
    true
}

// ── RegistrySummary ──────────────────────────────────────────────────────────

/// Aggregate statistics for the registry.
#[derive(Debug, Clone)]
pub struct RegistrySummary {
    /// Total number of registered agents.
    pub total_agents: usize,
    /// Sum of all capabilities across all agents.
    pub total_capabilities: usize,
    /// How many capabilities of each kind exist.
    pub kinds_histogram: HashMap<String, usize>,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cap(
        id: &str,
        kind: CapabilityKind,
        version: (u32, u32, u32),
        tags: &[&str],
        constraints: &[(&str, &str)],
    ) -> Capability {
        Capability {
            id: id.to_string(),
            kind,
            name: id.to_string(),
            version,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            constraints: constraints
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn register_and_find_by_kind() {
        let mut reg = CapabilityRegistry::new();
        reg.register(
            "agent-a".into(),
            vec![make_cap("c1", CapabilityKind::WebSearch, (1, 0, 0), &[], &[])],
            1000,
        );
        reg.register(
            "agent-b".into(),
            vec![make_cap("c2", CapabilityKind::CodeExecution, (1, 0, 0), &[], &[])],
            1000,
        );

        let query = CapabilityQuery {
            required_kind: Some(CapabilityKind::WebSearch),
            ..Default::default()
        };
        let found = reg.find_agents(&query);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "agent-a");
    }

    #[test]
    fn version_filter() {
        let mut reg = CapabilityRegistry::new();
        reg.register(
            "old".into(),
            vec![make_cap("c1", CapabilityKind::DataProcessing, (1, 0, 0), &[], &[])],
            1000,
        );
        reg.register(
            "new".into(),
            vec![make_cap("c2", CapabilityKind::DataProcessing, (2, 0, 0), &[], &[])],
            1000,
        );

        let query = CapabilityQuery {
            required_kind: Some(CapabilityKind::DataProcessing),
            min_version: Some((2, 0, 0)),
            ..Default::default()
        };
        let found = reg.find_agents(&query);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "new");
    }

    #[test]
    fn tag_filter() {
        let mut reg = CapabilityRegistry::new();
        reg.register(
            "gpu-agent".into(),
            vec![make_cap(
                "c1",
                CapabilityKind::MachineLearning,
                (1, 0, 0),
                &["gpu", "fast"],
                &[],
            )],
            1000,
        );
        reg.register(
            "cpu-agent".into(),
            vec![make_cap(
                "c2",
                CapabilityKind::MachineLearning,
                (1, 0, 0),
                &["cpu"],
                &[],
            )],
            1000,
        );

        let query = CapabilityQuery {
            required_kind: Some(CapabilityKind::MachineLearning),
            required_tags: vec!["gpu".to_string()],
            ..Default::default()
        };
        let found = reg.find_agents(&query);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "gpu-agent");
    }

    #[test]
    fn stale_eviction() {
        let mut reg = CapabilityRegistry::new();
        reg.register("a1".into(), vec![], 100);
        reg.register("a2".into(), vec![], 200);
        // a1 is stale if now=500, max_age=300 => threshold=200, a1.heartbeat(100) < 200
        let stale = reg.stale_agents(500, 300);
        assert!(stale.contains(&"a1"));
        assert!(!stale.contains(&"a2"));

        let evicted = reg.evict_stale(500, 300);
        assert_eq!(evicted, 1);
        assert!(reg.agents.contains_key("a2"));
        assert!(!reg.agents.contains_key("a1"));
    }

    #[test]
    fn deregister() {
        let mut reg = CapabilityRegistry::new();
        reg.register("agent-x".into(), vec![], 1000);
        assert!(reg.deregister("agent-x"));
        assert!(!reg.deregister("agent-x")); // already gone
        assert!(reg.capabilities_of("agent-x").is_none());
    }

    #[test]
    fn summary_counts() {
        let mut reg = CapabilityRegistry::new();
        reg.register(
            "a".into(),
            vec![
                make_cap("c1", CapabilityKind::WebSearch, (1, 0, 0), &[], &[]),
                make_cap("c2", CapabilityKind::WebSearch, (1, 0, 0), &[], &[]),
            ],
            1000,
        );
        reg.register(
            "b".into(),
            vec![make_cap("c3", CapabilityKind::CodeExecution, (1, 0, 0), &[], &[])],
            1000,
        );

        let s = reg.summary();
        assert_eq!(s.total_agents, 2);
        assert_eq!(s.total_capabilities, 3);
        assert_eq!(*s.kinds_histogram.get("WebSearch").unwrap(), 2);
        assert_eq!(*s.kinds_histogram.get("CodeExecution").unwrap(), 1);
    }

    #[test]
    fn agents_with_kind() {
        let mut reg = CapabilityRegistry::new();
        reg.register(
            "fs-agent".into(),
            vec![make_cap("c1", CapabilityKind::FileSystem, (1, 0, 0), &[], &[])],
            1000,
        );
        reg.register(
            "ml-agent".into(),
            vec![make_cap("c2", CapabilityKind::MachineLearning, (1, 0, 0), &[], &[])],
            1000,
        );

        let found = reg.agents_with_kind(&CapabilityKind::FileSystem);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "fs-agent");
    }
}
