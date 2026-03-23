//! # Agent Discovery
//!
//! Peer-to-peer capability advertisement for multi-agent systems.
//! Agents publish their capabilities to a shared [`AgentRegistry`].
//! Other agents query the registry to find the best peer for a task.
//!
//! ## Matching algorithm
//!
//! Capability matching uses a two-pass tag-overlap score:
//!
//! 1. **Required tags** — any capability that does *not* cover every required
//!    tag is discarded entirely.
//! 2. **Score** — for capabilities that pass the filter, the score is:
//!    ```text
//!    score = jaccard(capability.tags, required_tags ∪ preferred_tags)
//!          × health_penalty
//!    ```
//!    where `health_penalty` is `1.0` for `Healthy`, `0.6` for `Degraded`,
//!    and `0.0` for `Unhealthy`/`Unknown`.
//! 3. Results are sorted by score descending and returned as a
//!    `Vec<CapabilityMatch>`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

// ── Capability ────────────────────────────────────────────────────────────────

/// A single capability that an agent can advertise.
#[derive(Debug, Clone)]
pub struct Capability {
    /// Short machine-readable name, e.g. `"summarise-text"`.
    pub name: String,
    /// Human-readable description shown in discovery results.
    pub description: String,
    /// Searchable tags used for capability matching.
    pub tags: Vec<String>,
    /// JSON Schema describing the expected input to this capability, if any.
    pub input_schema: Option<serde_json::Value>,
    /// JSON Schema describing the output produced by this capability, if any.
    pub output_schema: Option<serde_json::Value>,
    /// Observed average latency in milliseconds, if known.
    pub avg_latency_ms: Option<u64>,
    /// Indicative cost per invocation in USD, if known.
    pub cost_per_call_usd: Option<f64>,
}

// ── AgentHealth ───────────────────────────────────────────────────────────────

/// Health status reported by or inferred for an agent.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentHealth {
    /// Agent is operating normally.
    Healthy,
    /// Agent is operational but experiencing elevated error rates or latency.
    Degraded,
    /// Agent is not accepting work.
    Unhealthy,
    /// Health is not yet known (e.g. no heartbeat received since registration).
    Unknown,
}

// ── AgentAdvertisement ────────────────────────────────────────────────────────

/// An agent's self-description as stored in the registry.
#[derive(Debug, Clone)]
pub struct AgentAdvertisement {
    /// Unique, stable identifier for the advertising agent.
    pub agent_id: String,
    /// Human-readable display name.
    pub agent_name: String,
    /// Capabilities this agent exposes.
    pub capabilities: Vec<Capability>,
    /// Optional network endpoint, e.g. `"http://agent-2:8080"`.
    pub endpoint: Option<String>,
    /// Wall-clock time of initial registration.
    pub registered_at: SystemTime,
    /// Wall-clock time of the most recent heartbeat.
    pub last_heartbeat: SystemTime,
    /// Current health status.
    pub health: AgentHealth,
}

// ── CapabilityQuery ───────────────────────────────────────────────────────────

/// A query submitted to [`AgentRegistry::query`].
#[derive(Debug, Clone)]
pub struct CapabilityQuery {
    /// Tags that a capability *must* have.  Capabilities missing any of these
    /// are excluded from results.
    pub required_tags: Vec<String>,
    /// Tags that improve the score of a capability but are not mandatory.
    pub preferred_tags: Vec<String>,
    /// If set, exclude capabilities whose `avg_latency_ms` exceeds this value.
    pub max_latency_ms: Option<u64>,
    /// If set, exclude capabilities whose `cost_per_call_usd` exceeds this value.
    pub max_cost_usd: Option<f64>,
}

// ── CapabilityMatch ───────────────────────────────────────────────────────────

/// One match returned from [`AgentRegistry::query`].
#[derive(Debug, Clone)]
pub struct CapabilityMatch {
    /// ID of the agent that owns this capability.
    pub agent_id: String,
    /// The matched capability.
    pub capability: Capability,
    /// Match quality score in `[0.0, 1.0]`.  Higher is better.
    pub score: f32,
}

// ── AgentRegistry ─────────────────────────────────────────────────────────────

/// In-memory capability registry with TTL-based stale-agent eviction.
///
/// All methods are `async` and safe to call from multiple tasks simultaneously.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::discovery::{
///     AgentRegistry, AgentAdvertisement, AgentHealth,
///     Capability, CapabilityQuery,
/// };
/// use std::time::SystemTime;
///
/// # #[tokio::main]
/// # async fn main() {
/// let registry = AgentRegistry::new(60); // 60-second TTL
///
/// let ad = AgentAdvertisement {
///     agent_id: "agent-1".to_string(),
///     agent_name: "Summariser".to_string(),
///     capabilities: vec![Capability {
///         name: "summarise".to_string(),
///         description: "Summarises long documents".to_string(),
///         tags: vec!["nlp".to_string(), "summarise".to_string()],
///         input_schema: None,
///         output_schema: None,
///         avg_latency_ms: Some(200),
///         cost_per_call_usd: None,
///     }],
///     endpoint: Some("http://agent-1:8080".to_string()),
///     registered_at: SystemTime::now(),
///     last_heartbeat: SystemTime::now(),
///     health: AgentHealth::Healthy,
/// };
///
/// registry.register(ad).await;
///
/// let matches = registry.query(&CapabilityQuery {
///     required_tags: vec!["nlp".to_string()],
///     preferred_tags: vec!["summarise".to_string()],
///     max_latency_ms: Some(500),
///     max_cost_usd: None,
/// }).await;
///
/// assert_eq!(matches.len(), 1);
/// assert_eq!(matches[0].agent_id, "agent-1");
/// # }
/// ```
pub struct AgentRegistry {
    agents: Arc<RwLock<HashMap<String, AgentAdvertisement>>>,
    /// Agents whose last heartbeat is older than `ttl_secs` seconds are
    /// considered stale and removed by [`Self::evict_stale`].
    ttl_secs: u64,
}

impl AgentRegistry {
    /// Create a new registry.  Agents that go `ttl_secs` seconds without a
    /// heartbeat are treated as stale and removed by [`Self::evict_stale`].
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            ttl_secs,
        }
    }

    /// Register or fully replace an agent's advertisement.
    ///
    /// If an advertisement for `ad.agent_id` already exists it is overwritten.
    pub async fn register(&self, ad: AgentAdvertisement) {
        let mut guard = self.agents.write().await;
        guard.insert(ad.agent_id.clone(), ad);
    }

    /// Remove an agent from the registry.
    ///
    /// Does nothing if the agent is not registered.
    pub async fn deregister(&self, agent_id: &str) {
        let mut guard = self.agents.write().await;
        guard.remove(agent_id);
    }

    /// Record a heartbeat for `agent_id`, refreshing its TTL and updating its
    /// health status.
    ///
    /// Does nothing if `agent_id` is not registered.
    pub async fn heartbeat(&self, agent_id: &str, health: AgentHealth) {
        let mut guard = self.agents.write().await;
        if let Some(ad) = guard.get_mut(agent_id) {
            ad.last_heartbeat = SystemTime::now();
            ad.health = health;
        }
    }

    /// Return all capability matches for query `q`, sorted by score descending.
    ///
    /// Unhealthy and unknown-health agents are included in results but penalised
    /// heavily (score multiplied by `0.0`), so they appear last if at all.
    pub async fn query(&self, q: &CapabilityQuery) -> Vec<CapabilityMatch> {
        let guard = self.agents.read().await;
        let mut results: Vec<CapabilityMatch> = guard
            .values()
            .flat_map(|ad| self.score_agent(ad, q))
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Remove all agents whose `last_heartbeat` is older than `ttl_secs`.
    pub async fn evict_stale(&self) {
        let now = SystemTime::now();
        let ttl = std::time::Duration::from_secs(self.ttl_secs);
        let mut guard = self.agents.write().await;
        guard.retain(|_id, ad| {
            now.duration_since(ad.last_heartbeat)
                .map(|elapsed| elapsed < ttl)
                .unwrap_or(false)
        });
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Score every capability in `ad` against `q`.
    ///
    /// Capabilities that fail required-tag or constraint filters are excluded.
    fn score_agent(&self, ad: &AgentAdvertisement, q: &CapabilityQuery) -> Vec<CapabilityMatch> {
        let health_factor = match ad.health {
            AgentHealth::Healthy => 1.0_f32,
            AgentHealth::Degraded => 0.6,
            AgentHealth::Unhealthy | AgentHealth::Unknown => 0.0,
        };

        let required_set: HashSet<&str> = q.required_tags.iter().map(String::as_str).collect();

        // Build the combined query tag set for Jaccard scoring.
        let query_tags: Vec<String> = q
            .required_tags
            .iter()
            .chain(q.preferred_tags.iter())
            .cloned()
            .collect();

        ad.capabilities
            .iter()
            .filter_map(|cap| {
                // ── Required-tag filter ──────────────────────────────────────
                let cap_tag_set: HashSet<&str> = cap.tags.iter().map(String::as_str).collect();
                if !required_set.is_subset(&cap_tag_set) {
                    return None;
                }

                // ── Latency filter ───────────────────────────────────────────
                if let (Some(max_lat), Some(lat)) = (q.max_latency_ms, cap.avg_latency_ms) {
                    if lat > max_lat {
                        return None;
                    }
                }

                // ── Cost filter ──────────────────────────────────────────────
                if let (Some(max_cost), Some(cost)) = (q.max_cost_usd, cap.cost_per_call_usd) {
                    if cost > max_cost {
                        return None;
                    }
                }

                // ── Tag-overlap score ────────────────────────────────────────
                let jaccard = Self::tag_similarity(&cap.tags, &query_tags);
                let score = (jaccard * health_factor).clamp(0.0, 1.0);

                Some(CapabilityMatch {
                    agent_id: ad.agent_id.clone(),
                    capability: cap.clone(),
                    score,
                })
            })
            .collect()
    }

    /// Jaccard similarity between two tag lists treated as sets.
    ///
    /// `J(A, B) = |A ∩ B| / |A ∪ B|`
    ///
    /// Returns `0.0` when both sets are empty.
    fn tag_similarity(a: &[String], b: &[String]) -> f32 {
        let set_a: HashSet<&str> = a.iter().map(String::as_str).collect();
        let set_b: HashSet<&str> = b.iter().map(String::as_str).collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            return 0.0;
        }
        intersection as f32 / union as f32
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn make_registry() -> AgentRegistry {
        AgentRegistry::new(60)
    }

    fn make_ad(id: &str, tags: Vec<&str>, health: AgentHealth) -> AgentAdvertisement {
        let now = SystemTime::now();
        AgentAdvertisement {
            agent_id: id.to_string(),
            agent_name: format!("Agent {id}"),
            capabilities: vec![Capability {
                name: "cap".to_string(),
                description: "test capability".to_string(),
                tags: tags.into_iter().map(str::to_string).collect(),
                input_schema: None,
                output_schema: None,
                avg_latency_ms: Some(100),
                cost_per_call_usd: Some(0.01),
            }],
            endpoint: None,
            registered_at: now,
            last_heartbeat: now,
            health,
        }
    }

    #[tokio::test]
    async fn register_and_query_basic() {
        let registry = make_registry();
        registry.register(make_ad("a1", vec!["nlp", "summarise"], AgentHealth::Healthy)).await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["nlp".to_string()],
            preferred_tags: vec![],
            max_latency_ms: None,
            max_cost_usd: None,
        }).await;

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].agent_id, "a1");
    }

    #[tokio::test]
    async fn required_tag_filter_excludes_non_matching() {
        let registry = make_registry();
        registry.register(make_ad("a1", vec!["nlp"], AgentHealth::Healthy)).await;
        registry.register(make_ad("a2", vec!["vision"], AgentHealth::Healthy)).await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["vision".to_string()],
            preferred_tags: vec![],
            max_latency_ms: None,
            max_cost_usd: None,
        }).await;

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].agent_id, "a2");
    }

    #[tokio::test]
    async fn unhealthy_agent_scores_zero() {
        let registry = make_registry();
        registry.register(make_ad("a1", vec!["nlp"], AgentHealth::Unhealthy)).await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["nlp".to_string()],
            preferred_tags: vec![],
            max_latency_ms: None,
            max_cost_usd: None,
        }).await;

        // Match is present but score is 0.
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].score, 0.0);
    }

    #[tokio::test]
    async fn deregister_removes_agent() {
        let registry = make_registry();
        registry.register(make_ad("a1", vec!["nlp"], AgentHealth::Healthy)).await;
        registry.deregister("a1").await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["nlp".to_string()],
            preferred_tags: vec![],
            max_latency_ms: None,
            max_cost_usd: None,
        }).await;

        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn heartbeat_updates_health() {
        let registry = make_registry();
        registry.register(make_ad("a1", vec!["nlp"], AgentHealth::Unhealthy)).await;
        registry.heartbeat("a1", AgentHealth::Healthy).await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["nlp".to_string()],
            preferred_tags: vec![],
            max_latency_ms: None,
            max_cost_usd: None,
        }).await;

        assert!(matches[0].score > 0.0);
    }

    #[tokio::test]
    async fn evict_stale_removes_old_agents() {
        // Use 0-second TTL so every agent is immediately stale.
        let registry = AgentRegistry::new(0);
        registry.register(make_ad("a1", vec!["nlp"], AgentHealth::Healthy)).await;

        // Sleep briefly to ensure the heartbeat timestamp is definitely in the past.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        registry.evict_stale().await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["nlp".to_string()],
            preferred_tags: vec![],
            max_latency_ms: None,
            max_cost_usd: None,
        }).await;

        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn latency_filter_excludes_slow_agents() {
        let registry = make_registry();
        registry.register(make_ad("fast", vec!["nlp"], AgentHealth::Healthy)).await;

        // Manually insert a slow agent.
        let mut slow = make_ad("slow", vec!["nlp"], AgentHealth::Healthy);
        slow.capabilities[0].avg_latency_ms = Some(5000);
        registry.register(slow).await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["nlp".to_string()],
            preferred_tags: vec![],
            max_latency_ms: Some(200),
            max_cost_usd: None,
        }).await;

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].agent_id, "fast");
    }

    #[tokio::test]
    async fn results_sorted_by_score_descending() {
        let registry = make_registry();
        // "a1" has both required + preferred; "a2" has only required.
        registry.register(make_ad("a1", vec!["nlp", "summarise"], AgentHealth::Healthy)).await;
        registry.register(make_ad("a2", vec!["nlp"], AgentHealth::Healthy)).await;

        let matches = registry.query(&CapabilityQuery {
            required_tags: vec!["nlp".to_string()],
            preferred_tags: vec!["summarise".to_string()],
            max_latency_ms: None,
            max_cost_usd: None,
        }).await;

        assert_eq!(matches.len(), 2);
        assert!(matches[0].score >= matches[1].score);
        assert_eq!(matches[0].agent_id, "a1");
    }

    #[test]
    fn tag_similarity_empty() {
        assert_eq!(AgentRegistry::tag_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn tag_similarity_identical() {
        let tags = vec!["a".to_string(), "b".to_string()];
        assert!((AgentRegistry::tag_similarity(&tags, &tags) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn tag_similarity_disjoint() {
        let a = vec!["x".to_string()];
        let b = vec!["y".to_string()];
        assert_eq!(AgentRegistry::tag_similarity(&a, &b), 0.0);
    }
}
