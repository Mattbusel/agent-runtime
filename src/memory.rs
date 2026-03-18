//! # Module: Memory
//!
//! ## Responsibility
//! Provides episodic, semantic, and working memory stores for agents.
//! Mirrors the public API of `tokio-agent-memory` and `tokio-memory`.
//!
//! ## Guarantees
//! - Thread-safe: all stores wrap their state in `Arc<Mutex<_>>`
//! - Bounded: WorkingMemory evicts the oldest entry when capacity is exceeded
//! - Decaying: DecayPolicy reduces importance scores over time
//! - Non-panicking: all operations return `Result`
//! - Lock-poisoning resilient: a panicking thread does not permanently break a store
//!
//! ## NOT Responsible For
//! - Cross-agent shared memory (see runtime.rs coordinator)
//! - Persistence to disk or external store

use crate::error::AgentRuntimeError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// ── Lock-poisoning recovery ───────────────────────────────────────────────────

/// Acquire a mutex guard, recovering from a poisoned mutex rather than
/// propagating an error.  A panicking thread does not permanently break the
/// store; we simply take ownership of the inner value and emit a warning.
fn recover_lock<'a, T>(
    result: std::sync::LockResult<std::sync::MutexGuard<'a, T>>,
    ctx: &str,
) -> std::sync::MutexGuard<'a, T>
where
    T: ?Sized,
{
    match result {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("mutex poisoned in {ctx}, recovering inner value");
            poisoned.into_inner()
        }
    }
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ── Newtype IDs ───────────────────────────────────────────────────────────────

/// Stable identifier for an agent instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    /// Create a new `AgentId` from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a random `AgentId` backed by a UUID v4.
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stable identifier for a memory item.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub String);

impl MemoryId {
    /// Create a new `MemoryId` from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a random `MemoryId` backed by a UUID v4.
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── MemoryItem ────────────────────────────────────────────────────────────────

/// A single memory record stored for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Unique identifier for this memory.
    pub id: MemoryId,
    /// The agent this memory belongs to.
    pub agent_id: AgentId,
    /// Textual content of the memory.
    pub content: String,
    /// Importance score in `[0.0, 1.0]`. Higher = more important.
    pub importance: f32,
    /// UTC timestamp when this memory was recorded.
    pub timestamp: DateTime<Utc>,
    /// Searchable tags attached to this memory.
    pub tags: Vec<String>,
    /// Number of times this memory has been recalled. Updated in-place by `recall`.
    #[serde(default)]
    pub recall_count: u64,
}

impl MemoryItem {
    /// Construct a new `MemoryItem` with the current timestamp and a random ID.
    pub fn new(
        agent_id: AgentId,
        content: impl Into<String>,
        importance: f32,
        tags: Vec<String>,
    ) -> Self {
        Self {
            id: MemoryId::random(),
            agent_id,
            content: content.into(),
            importance: importance.clamp(0.0, 1.0),
            timestamp: Utc::now(),
            tags,
            recall_count: 0,
        }
    }
}

// ── DecayPolicy ───────────────────────────────────────────────────────────────

/// Governs how memory importance decays over time.
#[derive(Debug, Clone)]
pub struct DecayPolicy {
    /// The half-life duration in hours. After this many hours, importance is halved.
    half_life_hours: f64,
}

impl DecayPolicy {
    /// Create an exponential decay policy with the given half-life in hours.
    ///
    /// # Arguments
    /// * `half_life_hours` — time after which importance is halved; must be > 0
    ///
    /// # Returns
    /// - `Ok(DecayPolicy)` — on success
    /// - `Err(AgentRuntimeError::Memory)` — if `half_life_hours <= 0`
    pub fn exponential(half_life_hours: f64) -> Result<Self, AgentRuntimeError> {
        if half_life_hours <= 0.0 {
            return Err(AgentRuntimeError::Memory(
                "half_life_hours must be positive".into(),
            ));
        }
        Ok(Self { half_life_hours })
    }

    /// Apply decay to an importance score based on elapsed time.
    ///
    /// # Arguments
    /// * `importance` — original importance in `[0.0, 1.0]`
    /// * `age_hours` — how many hours have passed since the memory was recorded
    ///
    /// # Returns
    /// Decayed importance clamped to `[0.0, 1.0]`.
    pub fn apply(&self, importance: f32, age_hours: f64) -> f32 {
        let decay = (-age_hours * std::f64::consts::LN_2 / self.half_life_hours).exp();
        (importance as f64 * decay).clamp(0.0, 1.0) as f32
    }

    /// Apply decay in-place to a mutable `MemoryItem`.
    pub fn decay_item(&self, item: &mut MemoryItem) {
        let age_hours = (Utc::now() - item.timestamp).num_seconds().max(0) as f64 / 3600.0;
        item.importance = self.apply(item.importance, age_hours);
    }
}

// ── RecallPolicy ──────────────────────────────────────────────────────────────

/// Controls how memories are scored and ranked during recall.
#[derive(Debug, Clone)]
pub enum RecallPolicy {
    /// Rank purely by importance score (default).
    Importance,
    /// Hybrid score: blends importance, recency, and recall frequency.
    ///
    /// `score = importance + recency_score * recency_weight + frequency_score * frequency_weight`
    /// where `recency_score = exp(-age_hours / 24.0)` and
    /// `frequency_score = recall_count / (max_recall_count + 1)` (normalized).
    Hybrid {
        /// Weight applied to the recency component of the hybrid score.
        recency_weight: f32,
        /// Weight applied to the recall-frequency component of the hybrid score.
        frequency_weight: f32,
    },
}

// ── Hybrid scoring helper ─────────────────────────────────────────────────────

fn compute_hybrid_score(
    item: &MemoryItem,
    recency_weight: f32,
    frequency_weight: f32,
    max_recall: u64,
    now: chrono::DateTime<Utc>,
) -> f32 {
    let age_hours = (now - item.timestamp).num_seconds().max(0) as f64 / 3600.0;
    let recency_score = (-age_hours / 24.0).exp() as f32;
    let frequency_score = item.recall_count as f32 / (max_recall as f32 + 1.0);
    item.importance + recency_score * recency_weight + frequency_score * frequency_weight
}

// ── EpisodicStore ─────────────────────────────────────────────────────────────

/// Stores episodic (event-based) memories for agents, ordered by insertion time.
///
/// ## Guarantees
/// - Thread-safe via `Arc<Mutex<_>>`
/// - Ordered: recall returns items in descending importance order
/// - Bounded by optional capacity
#[derive(Debug, Clone)]
pub struct EpisodicStore {
    inner: Arc<Mutex<EpisodicInner>>,
}

#[derive(Debug)]
struct EpisodicInner {
    items: Vec<MemoryItem>,
    decay: Option<DecayPolicy>,
    recall_policy: RecallPolicy,
    /// Maximum items stored per agent. Oldest (lowest-importance) items evicted when exceeded.
    per_agent_capacity: Option<usize>,
}

impl EpisodicStore {
    /// Create a new unbounded episodic store without decay.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: Vec::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: None,
            })),
        }
    }

    /// Create a new episodic store with the given decay policy.
    pub fn with_decay(policy: DecayPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: Vec::new(),
                decay: Some(policy),
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: None,
            })),
        }
    }

    /// Create a new episodic store with the given recall policy.
    pub fn with_recall_policy(policy: RecallPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: Vec::new(),
                decay: None,
                recall_policy: policy,
                per_agent_capacity: None,
            })),
        }
    }

    /// Create a new episodic store with the given per-agent capacity limit.
    ///
    /// When an agent exceeds this capacity, the lowest-importance item for that
    /// agent is evicted.
    pub fn with_per_agent_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: Vec::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: Some(capacity),
            })),
        }
    }

    /// Record a new episode for the given agent.
    ///
    /// # Returns
    /// The `MemoryId` of the newly created memory item.
    #[tracing::instrument(skip(self))]
    pub fn add_episode(
        &self,
        agent_id: AgentId,
        content: impl Into<String> + std::fmt::Debug,
        importance: f32,
    ) -> Result<MemoryId, AgentRuntimeError> {
        let item = MemoryItem::new(agent_id.clone(), content, importance, Vec::new());
        let id = item.id.clone();
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::add_episode");
        inner.items.push(item);
        if let Some(cap) = inner.per_agent_capacity {
            let agent_count = inner.items.iter().filter(|i| i.agent_id == agent_id).count();
            if agent_count > cap {
                if let Some(pos) = inner
                    .items
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| i.agent_id == agent_id)
                    .min_by(|(_, a), (_, b)| {
                        a.importance
                            .partial_cmp(&b.importance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(pos, _)| pos)
                {
                    inner.items.remove(pos);
                }
            }
        }
        Ok(id)
    }

    /// Add an episode with an explicit timestamp.
    #[tracing::instrument(skip(self))]
    pub fn add_episode_at(
        &self,
        agent_id: AgentId,
        content: impl Into<String> + std::fmt::Debug,
        importance: f32,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<MemoryId, AgentRuntimeError> {
        let mut item = MemoryItem::new(agent_id.clone(), content, importance, Vec::new());
        item.timestamp = timestamp;
        let id = item.id.clone();
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::add_episode_at");
        inner.items.push(item);
        if let Some(cap) = inner.per_agent_capacity {
            let agent_count = inner.items.iter().filter(|i| i.agent_id == agent_id).count();
            if agent_count > cap {
                if let Some(pos) = inner
                    .items
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| i.agent_id == agent_id)
                    .min_by(|(_, a), (_, b)| {
                        a.importance
                            .partial_cmp(&b.importance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(pos, _)| pos)
                {
                    inner.items.remove(pos);
                }
            }
        }
        Ok(id)
    }

    /// Recall up to `limit` memories for the given agent.
    ///
    /// Applies decay if configured, increments `recall_count` for each recalled
    /// item in-place, then returns items sorted according to the configured
    /// `RecallPolicy` (default: descending importance).
    #[tracing::instrument(skip(self))]
    pub fn recall(
        &self,
        agent_id: &AgentId,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::recall");

        // Apply decay in-place
        let decay_clone: Option<DecayPolicy> = inner.decay.clone();
        if let Some(policy) = decay_clone {
            for item in inner.items.iter_mut() {
                policy.decay_item(item);
            }
        }

        // Collect IDs of items belonging to this agent
        let agent_ids_to_update: Vec<MemoryId> = inner
            .items
            .iter()
            .filter(|i| &i.agent_id == agent_id)
            .map(|i| i.id.clone())
            .collect();

        // Increment recall_count in-place for all matching items
        for item in inner.items.iter_mut() {
            if agent_ids_to_update.contains(&item.id) {
                item.recall_count += 1;
            }
        }

        let mut items: Vec<MemoryItem> = inner
            .items
            .iter()
            .filter(|i| &i.agent_id == agent_id)
            .cloned()
            .collect();

        match inner.recall_policy {
            RecallPolicy::Importance => {
                items.sort_by(|a, b| {
                    b.importance
                        .partial_cmp(&a.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            RecallPolicy::Hybrid {
                recency_weight,
                frequency_weight,
            } => {
                let max_recall = items.iter().map(|i| i.recall_count).max().unwrap_or(1).max(1);
                let now = Utc::now();
                items.sort_by(|a, b| {
                    let score_a = compute_hybrid_score(
                        a,
                        recency_weight,
                        frequency_weight,
                        max_recall,
                        now,
                    );
                    let score_b = compute_hybrid_score(
                        b,
                        recency_weight,
                        frequency_weight,
                        max_recall,
                        now,
                    );
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        items.truncate(limit);
        tracing::debug!("recalled {} items", items.len());
        Ok(items)
    }

    /// Return the total number of stored episodes across all agents.
    pub fn len(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::len");
        Ok(inner.items.len())
    }

    /// Return `true` if no episodes have been stored.
    pub fn is_empty(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.len()? == 0)
    }

    /// Bump the `recall_count` of every item whose content equals `content` by `amount`.
    ///
    /// This method exists to support integration tests that need to simulate prior recall
    /// history without accessing private fields. It is not intended for production use.
    #[doc(hidden)]
    pub fn bump_recall_count_by_content(&self, content: &str, amount: u64) {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::bump_recall_count_by_content");
        for item in inner.items.iter_mut() {
            if item.content == content {
                item.recall_count = item.recall_count.saturating_add(amount);
            }
        }
    }
}

impl Default for EpisodicStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── SemanticStore ─────────────────────────────────────────────────────────────

/// Stores semantic (fact-based) knowledge as tagged key-value pairs.
///
/// ## Guarantees
/// - Thread-safe via `Arc<Mutex<_>>`
/// - Retrieval by tag intersection
/// - Optional vector-based similarity search via stored embeddings
#[derive(Debug, Clone)]
pub struct SemanticStore {
    inner: Arc<Mutex<Vec<SemanticEntry>>>,
}

#[derive(Debug, Clone)]
struct SemanticEntry {
    key: String,
    value: String,
    tags: Vec<String>,
    embedding: Option<Vec<f32>>,
}

impl SemanticStore {
    /// Create a new empty semantic store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Store a key-value pair with associated tags.
    #[tracing::instrument(skip(self))]
    pub fn store(
        &self,
        key: impl Into<String> + std::fmt::Debug,
        value: impl Into<String> + std::fmt::Debug,
        tags: Vec<String>,
    ) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::store");
        inner.push(SemanticEntry {
            key: key.into(),
            value: value.into(),
            tags,
            embedding: None,
        });
        Ok(())
    }

    /// Store a key-value pair with an embedding vector for similarity search.
    #[tracing::instrument(skip(self))]
    pub fn store_with_embedding(
        &self,
        key: impl Into<String> + std::fmt::Debug,
        value: impl Into<String> + std::fmt::Debug,
        tags: Vec<String>,
        embedding: Vec<f32>,
    ) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::store_with_embedding");
        inner.push(SemanticEntry {
            key: key.into(),
            value: value.into(),
            tags,
            embedding: Some(embedding),
        });
        Ok(())
    }

    /// Retrieve all entries that contain **all** of the given tags.
    ///
    /// If `tags` is empty, returns all entries.
    #[tracing::instrument(skip(self))]
    pub fn retrieve(&self, tags: &[&str]) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::retrieve");

        let results = inner
            .iter()
            .filter(|entry| {
                tags.iter()
                    .all(|t| entry.tags.iter().any(|et| et.as_str() == *t))
            })
            .map(|e| (e.key.clone(), e.value.clone()))
            .collect();

        Ok(results)
    }

    /// Retrieve top-k entries by cosine similarity to `query_embedding`.
    ///
    /// Only entries that were stored with an embedding (via [`store_with_embedding`])
    /// are considered.  Returns `(key, value, similarity)` sorted by descending
    /// similarity.
    ///
    /// [`store_with_embedding`]: SemanticStore::store_with_embedding
    #[tracing::instrument(skip(self, query_embedding))]
    pub fn retrieve_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, String, f32)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::retrieve_similar");

        let mut scored: Vec<(String, String, f32)> = inner
            .iter()
            .filter_map(|entry| {
                entry.embedding.as_ref().map(|emb| {
                    let sim = cosine_similarity(query_embedding, emb);
                    (entry.key.clone(), entry.value.clone(), sim)
                })
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Return the total number of stored entries.
    pub fn len(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::len");
        Ok(inner.len())
    }

    /// Return `true` if no entries have been stored.
    pub fn is_empty(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.len()? == 0)
    }
}

impl Default for SemanticStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── WorkingMemory ─────────────────────────────────────────────────────────────

/// A bounded, key-value working memory for transient agent state.
///
/// When capacity is exceeded, the oldest entry (by insertion order) is evicted.
///
/// ## Guarantees
/// - Thread-safe via `Arc<Mutex<_>>`
/// - Bounded: never exceeds `capacity` entries
/// - Deterministic eviction: LRU (oldest insertion first)
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    capacity: usize,
    inner: Arc<Mutex<WorkingInner>>,
}

#[derive(Debug)]
struct WorkingInner {
    map: HashMap<String, String>,
    order: VecDeque<String>,
}

impl WorkingMemory {
    /// Create a new `WorkingMemory` with the given capacity.
    ///
    /// # Returns
    /// - `Ok(WorkingMemory)` — on success
    /// - `Err(AgentRuntimeError::Memory)` — if `capacity == 0`
    pub fn new(capacity: usize) -> Result<Self, AgentRuntimeError> {
        if capacity == 0 {
            return Err(AgentRuntimeError::Memory(
                "WorkingMemory capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            capacity,
            inner: Arc::new(Mutex::new(WorkingInner {
                map: HashMap::new(),
                order: VecDeque::new(),
            })),
        })
    }

    /// Insert or update a key-value pair, evicting the oldest entry if over capacity.
    #[tracing::instrument(skip(self))]
    pub fn set(
        &self,
        key: impl Into<String> + std::fmt::Debug,
        value: impl Into<String> + std::fmt::Debug,
    ) -> Result<(), AgentRuntimeError> {
        let key = key.into();
        let value = value.into();
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::set");

        // Remove existing key from order tracking if present
        if inner.map.contains_key(&key) {
            inner.order.retain(|k| k != &key);
        } else if inner.map.len() >= self.capacity {
            // Evict oldest
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
            }
        }

        inner.order.push_back(key.clone());
        inner.map.insert(key, value);
        Ok(())
    }

    /// Retrieve a value by key.
    ///
    /// # Returns
    /// - `Some(value)` — if the key exists
    /// - `None` — if not found
    #[tracing::instrument(skip(self))]
    pub fn get(&self, key: &str) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::get");
        Ok(inner.map.get(key).cloned())
    }

    /// Remove all entries from working memory.
    pub fn clear(&self) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::clear");
        inner.map.clear();
        inner.order.clear();
        Ok(())
    }

    /// Return the current number of entries.
    pub fn len(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::len");
        Ok(inner.map.len())
    }

    /// Return `true` if no entries are stored.
    pub fn is_empty(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.len()? == 0)
    }

    /// Return all key-value pairs in insertion order.
    pub fn entries(&self) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::entries");
        let entries = inner
            .order
            .iter()
            .filter_map(|k| inner.map.get(k).map(|v| (k.clone(), v.clone())))
            .collect();
        Ok(entries)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AgentId / MemoryId ────────────────────────────────────────────────────

    #[test]
    fn test_agent_id_new_stores_string() {
        let id = AgentId::new("agent-1");
        assert_eq!(id.0, "agent-1");
    }

    #[test]
    fn test_agent_id_random_is_unique() {
        let a = AgentId::random();
        let b = AgentId::random();
        assert_ne!(a, b);
    }

    #[test]
    fn test_memory_id_new_stores_string() {
        let id = MemoryId::new("mem-1");
        assert_eq!(id.0, "mem-1");
    }

    #[test]
    fn test_memory_id_random_is_unique() {
        let a = MemoryId::random();
        let b = MemoryId::random();
        assert_ne!(a, b);
    }

    // ── MemoryItem ────────────────────────────────────────────────────────────

    #[test]
    fn test_memory_item_new_clamps_importance_above_one() {
        let item = MemoryItem::new(AgentId::new("a"), "test", 1.5, vec![]);
        assert_eq!(item.importance, 1.0);
    }

    #[test]
    fn test_memory_item_new_clamps_importance_below_zero() {
        let item = MemoryItem::new(AgentId::new("a"), "test", -0.5, vec![]);
        assert_eq!(item.importance, 0.0);
    }

    #[test]
    fn test_memory_item_new_preserves_valid_importance() {
        let item = MemoryItem::new(AgentId::new("a"), "test", 0.7, vec![]);
        assert!((item.importance - 0.7).abs() < 1e-6);
    }

    // ── DecayPolicy ───────────────────────────────────────────────────────────

    #[test]
    fn test_decay_policy_rejects_zero_half_life() {
        assert!(DecayPolicy::exponential(0.0).is_err());
    }

    #[test]
    fn test_decay_policy_rejects_negative_half_life() {
        assert!(DecayPolicy::exponential(-1.0).is_err());
    }

    #[test]
    fn test_decay_policy_no_decay_at_age_zero() {
        let p = DecayPolicy::exponential(24.0).unwrap();
        let decayed = p.apply(1.0, 0.0);
        assert!((decayed - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_decay_policy_half_importance_at_half_life() {
        let p = DecayPolicy::exponential(24.0).unwrap();
        let decayed = p.apply(1.0, 24.0);
        assert!((decayed - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_decay_policy_quarter_importance_at_two_half_lives() {
        let p = DecayPolicy::exponential(24.0).unwrap();
        let decayed = p.apply(1.0, 48.0);
        assert!((decayed - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_decay_policy_result_is_clamped_to_zero_one() {
        let p = DecayPolicy::exponential(1.0).unwrap();
        let decayed = p.apply(0.0, 1000.0);
        assert!(decayed >= 0.0 && decayed <= 1.0);
    }

    // ── EpisodicStore ─────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_add_episode_returns_id() {
        let store = EpisodicStore::new();
        let id = store.add_episode(AgentId::new("a"), "event", 0.8).unwrap();
        assert!(!id.0.is_empty());
    }

    #[test]
    fn test_episodic_store_recall_returns_stored_item() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-1");
        store
            .add_episode(agent.clone(), "hello world", 0.9)
            .unwrap();
        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "hello world");
    }

    #[test]
    fn test_episodic_store_recall_filters_by_agent() {
        let store = EpisodicStore::new();
        let a = AgentId::new("agent-a");
        let b = AgentId::new("agent-b");
        store.add_episode(a.clone(), "for a", 0.5).unwrap();
        store.add_episode(b.clone(), "for b", 0.5).unwrap();
        let items = store.recall(&a, 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "for a");
    }

    #[test]
    fn test_episodic_store_recall_sorted_by_descending_importance() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-1");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items[0].content, "high");
        assert_eq!(items[1].content, "mid");
        assert_eq!(items[2].content, "low");
    }

    #[test]
    fn test_episodic_store_recall_respects_limit() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-1");
        for i in 0..5 {
            store
                .add_episode(agent.clone(), format!("item {i}"), 0.5)
                .unwrap();
        }
        let items = store.recall(&agent, 3).unwrap();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_episodic_store_len_tracks_insertions() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "a", 0.5).unwrap();
        store.add_episode(agent.clone(), "b", 0.5).unwrap();
        assert_eq!(store.len().unwrap(), 2);
    }

    #[test]
    fn test_episodic_store_is_empty_initially() {
        let store = EpisodicStore::new();
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn test_episodic_store_with_decay_reduces_importance() {
        let policy = DecayPolicy::exponential(0.001).unwrap(); // very fast decay
        let store = EpisodicStore::with_decay(policy);
        let agent = AgentId::new("a");

        // Manually insert an old item by directly manipulating timestamps
        {
            let mut inner = store.inner.lock().unwrap();
            let mut item = MemoryItem::new(agent.clone(), "old event", 1.0, vec![]);
            // Set the timestamp to 1 hour ago
            item.timestamp = Utc::now() - chrono::Duration::hours(1);
            inner.items.push(item);
        }

        let items = store.recall(&agent, 10).unwrap();
        // With half_life=0.001h and age=1h, importance should be near 0
        assert_eq!(items.len(), 1);
        assert!(
            items[0].importance < 0.01,
            "expected near-zero importance, got {}",
            items[0].importance
        );
    }

    // ── Item 10: RecallPolicy / per-agent capacity tests ──────────────────────

    #[test]
    fn test_recall_increments_recall_count() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-rc");
        store.add_episode(agent.clone(), "memory", 0.5).unwrap();

        // First recall — count becomes 1
        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items[0].recall_count, 1);

        // Second recall — count becomes 2
        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items[0].recall_count, 2);
    }

    #[test]
    fn test_hybrid_recall_policy_prefers_recently_used() {
        // "old_frequent": added 48 h ago with importance 0.5, recall_count bumped
        // manually to simulate frequent use.
        // "new_never": added just now with importance 0.5, never recalled.
        // With a large frequency_weight the frequently-recalled item should rank higher.
        let store = EpisodicStore::with_recall_policy(RecallPolicy::Hybrid {
            recency_weight: 0.1,
            frequency_weight: 2.0,
        });
        let agent = AgentId::new("agent-hybrid");

        let old_ts = Utc::now() - chrono::Duration::hours(48);
        store
            .add_episode_at(agent.clone(), "old_frequent", 0.5, old_ts)
            .unwrap();
        store
            .add_episode(agent.clone(), "new_never", 0.5)
            .unwrap();

        // Simulate many prior recalls of "old_frequent" by manually bumping its count
        {
            let mut inner = store.inner.lock().unwrap();
            for item in inner.items.iter_mut() {
                if item.content == "old_frequent" {
                    item.recall_count = 100;
                }
            }
        }

        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].content, "old_frequent",
            "hybrid policy should rank the frequently-recalled item first"
        );
    }

    #[test]
    fn test_per_agent_capacity_evicts_lowest_importance() {
        let store = EpisodicStore::with_per_agent_capacity(2);
        let agent = AgentId::new("agent-cap");

        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        // Adding "low" (0.1) should trigger eviction of the lowest-importance item
        store.add_episode(agent.clone(), "low", 0.1).unwrap();

        assert_eq!(
            store.len().unwrap(),
            2,
            "store should hold exactly 2 items after eviction"
        );

        let items = store.recall(&agent, 10).unwrap();
        let contents: Vec<&str> = items.iter().map(|i| i.content.as_str()).collect();
        assert!(
            !contents.contains(&"low"),
            "the lowest-importance item should have been evicted; remaining: {:?}",
            contents
        );
    }

    // ── SemanticStore ─────────────────────────────────────────────────────────

    #[test]
    fn test_semantic_store_store_and_retrieve_all() {
        let store = SemanticStore::new();
        store.store("key1", "value1", vec!["tag-a".into()]).unwrap();
        store.store("key2", "value2", vec!["tag-b".into()]).unwrap();
        let results = store.retrieve(&[]).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_semantic_store_retrieve_filters_by_tag() {
        let store = SemanticStore::new();
        store
            .store("k1", "v1", vec!["rust".into(), "async".into()])
            .unwrap();
        store.store("k2", "v2", vec!["rust".into()]).unwrap();
        let results = store.retrieve(&["async"]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "k1");
    }

    #[test]
    fn test_semantic_store_retrieve_requires_all_tags() {
        let store = SemanticStore::new();
        store
            .store("k1", "v1", vec!["a".into(), "b".into()])
            .unwrap();
        store.store("k2", "v2", vec!["a".into()]).unwrap();
        let results = store.retrieve(&["a", "b"]).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_semantic_store_is_empty_initially() {
        let store = SemanticStore::new();
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn test_semantic_store_len_tracks_insertions() {
        let store = SemanticStore::new();
        store.store("k", "v", vec![]).unwrap();
        assert_eq!(store.len().unwrap(), 1);
    }

    #[test]
    fn test_semantic_store_retrieve_similar_returns_closest() {
        let store = SemanticStore::new();
        // "close" is in the same direction as the query
        store
            .store_with_embedding("close", "close value", vec![], vec![1.0, 0.0, 0.0])
            .unwrap();
        // "far" is orthogonal to the query
        store
            .store_with_embedding("far", "far value", vec![], vec![0.0, 1.0, 0.0])
            .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = store.retrieve_similar(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
        // The closest result should be "close"
        assert_eq!(results[0].0, "close");
        assert!((results[0].2 - 1.0).abs() < 1e-5, "expected similarity ~1.0, got {}", results[0].2);
        // The far result should have similarity 0.0
        assert!((results[1].2).abs() < 1e-5, "expected similarity ~0.0, got {}", results[1].2);
    }

    #[test]
    fn test_semantic_store_retrieve_similar_ignores_unembedded_entries() {
        let store = SemanticStore::new();
        // This entry has no embedding — must not appear in similarity results
        store.store("no-emb", "no embedding value", vec![]).unwrap();
        // This entry has an embedding
        store
            .store_with_embedding("with-emb", "with embedding value", vec![], vec![1.0, 0.0])
            .unwrap();

        let query = vec![1.0, 0.0];
        let results = store.retrieve_similar(&query, 10).unwrap();
        assert_eq!(results.len(), 1, "only the embedded entry should appear");
        assert_eq!(results[0].0, "with-emb");
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors_return_zero() {
        // Exercise cosine_similarity through retrieve_similar with orthogonal vectors
        let store = SemanticStore::new();
        store
            .store_with_embedding("a", "va", vec![], vec![1.0, 0.0])
            .unwrap();
        store
            .store_with_embedding("b", "vb", vec![], vec![0.0, 1.0])
            .unwrap();

        // query is along [1, 0]; "b" is orthogonal → similarity should be 0
        let query = vec![1.0, 0.0];
        let results = store.retrieve_similar(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
        let b_result = results.iter().find(|(k, _, _)| k == "b").unwrap();
        assert!(
            b_result.2.abs() < 1e-5,
            "expected cosine similarity 0.0 for orthogonal vectors, got {}",
            b_result.2
        );
    }

    // ── WorkingMemory ─────────────────────────────────────────────────────────

    #[test]
    fn test_working_memory_new_rejects_zero_capacity() {
        assert!(WorkingMemory::new(0).is_err());
    }

    #[test]
    fn test_working_memory_set_and_get() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("foo", "bar").unwrap();
        let val = wm.get("foo").unwrap();
        assert_eq!(val, Some("bar".into()));
    }

    #[test]
    fn test_working_memory_get_missing_key_returns_none() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.get("missing").unwrap(), None);
    }

    #[test]
    fn test_working_memory_bounded_evicts_oldest() {
        let wm = WorkingMemory::new(3).unwrap();
        wm.set("k1", "v1").unwrap();
        wm.set("k2", "v2").unwrap();
        wm.set("k3", "v3").unwrap();
        wm.set("k4", "v4").unwrap(); // k1 should be evicted
        assert_eq!(wm.get("k1").unwrap(), None);
        assert_eq!(wm.get("k4").unwrap(), Some("v4".into()));
    }

    #[test]
    fn test_working_memory_update_existing_key_no_eviction() {
        let wm = WorkingMemory::new(2).unwrap();
        wm.set("k1", "v1").unwrap();
        wm.set("k2", "v2").unwrap();
        wm.set("k1", "v1-updated").unwrap(); // update, not eviction
        assert_eq!(wm.len().unwrap(), 2);
        assert_eq!(wm.get("k1").unwrap(), Some("v1-updated".into()));
        assert_eq!(wm.get("k2").unwrap(), Some("v2".into()));
    }

    #[test]
    fn test_working_memory_clear_removes_all() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        wm.clear().unwrap();
        assert!(wm.is_empty().unwrap());
    }

    #[test]
    fn test_working_memory_is_empty_initially() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(wm.is_empty().unwrap());
    }

    #[test]
    fn test_working_memory_len_tracks_entries() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        assert_eq!(wm.len().unwrap(), 2);
    }

    #[test]
    fn test_working_memory_capacity_never_exceeded() {
        let cap = 5usize;
        let wm = WorkingMemory::new(cap).unwrap();
        for i in 0..20 {
            wm.set(format!("key-{i}"), format!("val-{i}")).unwrap();
            assert!(wm.len().unwrap() <= cap);
        }
    }
}
