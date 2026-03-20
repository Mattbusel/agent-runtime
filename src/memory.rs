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
//! - O(1) agent lookup: EpisodicStore indexes items per-agent for efficient recall
//!
//! ## NOT Responsible For
//! - Cross-agent shared memory (see runtime.rs coordinator)
//! - Persistence to disk or external store

use crate::error::AgentRuntimeError;
use crate::util::recover_lock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

// Re-export the core ID types so callers can import from either module.
pub use crate::types::{AgentId, MemoryId};

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Normalize `v` to unit length in-place.  Does nothing if the norm is below
/// `f32::EPSILON` (zero or near-zero vector).
fn normalize_in_place(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
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

    /// Return the age of this memory in fractional hours since it was recorded.
    ///
    /// Returns `0.0` if the current time is somehow before `timestamp`.
    pub fn age_hours(&self) -> f64 {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.timestamp);
        elapsed.num_milliseconds().max(0) as f64 / 3_600_000.0
    }
}

impl std::fmt::Display for MemoryItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] importance={:.2} recalls={} content=\"{}\"",
            self.id,
            self.importance,
            self.recall_count,
            self.content
        )
    }
}

// ── DecayPolicy ───────────────────────────────────────────────────────────────

/// Governs how memory importance decays over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Return the configured half-life in hours.
    pub fn half_life_hours(&self) -> f64 {
        self.half_life_hours
    }

    /// Apply decay in-place to a mutable `MemoryItem`.
    pub fn decay_item(&self, item: &mut MemoryItem) {
        let age_hours = (Utc::now() - item.timestamp).num_seconds().max(0) as f64 / 3600.0;
        item.importance = self.apply(item.importance, age_hours);
    }
}

// ── RecallPolicy ──────────────────────────────────────────────────────────────

/// Controls how memories are scored and ranked during recall.
///
/// # Interaction with `DecayPolicy`
///
/// When both a `DecayPolicy` and a `RecallPolicy` are configured, decay is
/// applied **before** scoring.  This means that for `RecallPolicy::Importance`,
/// an old high-importance memory may rank lower than a fresh low-importance
/// memory after decay has reduced its score.
///
/// For `RecallPolicy::Hybrid`, the `recency_weight` term already captures
/// temporal distance; combining it with a `DecayPolicy` therefore applies a
/// *double* time penalty — set one or the other, not both, unless the double
/// penalty is intentional.
///
/// ## Score Calculation Example
///
/// Given two memories, each with `importance = 0.5`:
/// - Memory A: `recall_count = 0`, inserted 1 hour ago
/// - Memory B: `recall_count = 10`, inserted 10 hours ago
///
/// With `recency_weight = 1.0` and `frequency_weight = 0.1`:
/// - Score A = `0.5 + 1.0 × 1.0 + 0.1 × 0` = `1.5` (recency wins)
/// - Score B = `0.5 + 1.0 × (−10.0) + 0.1 × 10` = `−8.5` (old → ranked lower)
///
/// Note: the recency term uses negative hours-since-creation so older items score lower.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for RecallPolicy {
    fn default() -> Self {
        RecallPolicy::Importance
    }
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

// ── EvictionPolicy ────────────────────────────────────────────────────────────

/// Policy controlling which item is evicted when the per-agent capacity is exceeded.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Evict the item with the lowest importance score (default).
    #[default]
    LowestImportance,
    /// Evict the oldest item (by insertion order / timestamp).
    Oldest,
}

// ── EpisodicStoreBuilder ──────────────────────────────────────────────────────

/// Fluent builder for [`EpisodicStore`].
///
/// Allows combining any set of options — decay, recall policy, per-agent
/// capacity, max age, and eviction policy — before creating the store.
///
/// # Example
/// ```rust
/// use llm_agent_runtime::memory::{EpisodicStore, EvictionPolicy, RecallPolicy, DecayPolicy};
///
/// let store = EpisodicStore::builder()
///     .per_agent_capacity(50)
///     .eviction_policy(EvictionPolicy::Oldest)
///     .build();
/// ```
#[derive(Default)]
pub struct EpisodicStoreBuilder {
    decay: Option<DecayPolicy>,
    recall_policy: Option<RecallPolicy>,
    per_agent_capacity: Option<usize>,
    max_age_hours: Option<f64>,
    eviction_policy: Option<EvictionPolicy>,
}

impl EpisodicStoreBuilder {
    /// Set the decay policy.
    pub fn decay(mut self, policy: DecayPolicy) -> Self {
        self.decay = Some(policy);
        self
    }

    /// Set the recall policy.
    pub fn recall_policy(mut self, policy: RecallPolicy) -> Self {
        self.recall_policy = Some(policy);
        self
    }

    /// Set the per-agent capacity. Panics if `capacity == 0`.
    pub fn per_agent_capacity(mut self, capacity: usize) -> Self {
        assert!(capacity > 0, "per_agent_capacity must be > 0");
        self.per_agent_capacity = Some(capacity);
        self
    }

    /// Set the per-agent capacity without panicking.
    ///
    /// Returns `Err` if `capacity == 0`. Prefer this over [`per_agent_capacity`]
    /// in library/user-facing code where a misconfigured capacity should be
    /// handled gracefully rather than aborting the process.
    ///
    /// [`per_agent_capacity`]: EpisodicStoreBuilder::per_agent_capacity
    pub fn try_per_agent_capacity(
        mut self,
        capacity: usize,
    ) -> Result<Self, crate::error::AgentRuntimeError> {
        if capacity == 0 {
            return Err(crate::error::AgentRuntimeError::Memory(
                "per_agent_capacity must be > 0".into(),
            ));
        }
        self.per_agent_capacity = Some(capacity);
        Ok(self)
    }

    /// Set the maximum memory age in hours. Returns `Err` if `max_age_hours <= 0`.
    pub fn max_age_hours(mut self, hours: f64) -> Result<Self, crate::error::AgentRuntimeError> {
        if hours <= 0.0 {
            return Err(crate::error::AgentRuntimeError::Memory(
                "max_age_hours must be positive".into(),
            ));
        }
        self.max_age_hours = Some(hours);
        Ok(self)
    }

    /// Set the eviction policy.
    pub fn eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = Some(policy);
        self
    }

    /// Consume the builder and create an [`EpisodicStore`].
    pub fn build(self) -> EpisodicStore {
        // Warn when both a DecayPolicy and RecallPolicy::Hybrid are active:
        // decay is applied before scoring, so Hybrid's recency term produces a
        // double time penalty.  Set one or the other unless the double penalty
        // is intentional.
        if self.decay.is_some() {
            if let Some(RecallPolicy::Hybrid { .. }) = &self.recall_policy {
                tracing::warn!(
                    "EpisodicStore configured with both DecayPolicy and RecallPolicy::Hybrid \
                     — time-based decay is applied before hybrid scoring, resulting in a \
                     double time penalty.  Set one or the other unless this is intentional."
                );
            }
        }
        EpisodicStore {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: self.decay,
                recall_policy: self.recall_policy.unwrap_or(RecallPolicy::Importance),
                per_agent_capacity: self.per_agent_capacity,
                max_age_hours: self.max_age_hours,
                eviction_policy: self.eviction_policy.unwrap_or_default(),
            })),
        }
    }
}

// ── EpisodicStore ─────────────────────────────────────────────────────────────

/// Stores episodic (event-based) memories for agents, ordered by insertion time.
///
/// ## Guarantees
/// - Thread-safe via `Arc<Mutex<_>>`
/// - Ordered: recall returns items in descending importance order
/// - Bounded by optional per-agent capacity
/// - O(1) agent lookup via per-agent `HashMap` index
/// - Automatic expiry via optional `max_age_hours`
#[derive(Debug, Clone)]
pub struct EpisodicStore {
    inner: Arc<Mutex<EpisodicInner>>,
}

#[derive(Debug)]
struct EpisodicInner {
    /// Items stored per-agent for O(1) lookup. The key is the agent ID.
    items: HashMap<AgentId, Vec<MemoryItem>>,
    decay: Option<DecayPolicy>,
    recall_policy: RecallPolicy,
    /// Maximum items stored per agent. Oldest (lowest-importance) items evicted when exceeded.
    per_agent_capacity: Option<usize>,
    /// Maximum age in hours. Items older than this are purged on the next recall or add.
    max_age_hours: Option<f64>,
    /// Eviction policy when per_agent_capacity is exceeded.
    eviction_policy: EvictionPolicy,
}

impl EpisodicInner {
    /// Purge items for `agent_id` that exceed `max_age_hours`, if configured.
    fn purge_stale(&mut self, agent_id: &AgentId) {
        if let Some(max_age_h) = self.max_age_hours {
            let cutoff = Utc::now()
                - chrono::Duration::seconds((max_age_h * 3600.0) as i64);
            if let Some(agent_items) = self.items.get_mut(agent_id) {
                agent_items.retain(|i| i.timestamp >= cutoff);
            }
        }
    }
}

/// Evict one item from `agent_items` if `len > cap`, according to `policy`.
///
/// The last element (the just-inserted item) is excluded from the
/// `LowestImportance` scan so that newly added items are never evicted.
fn evict_if_over_capacity(
    agent_items: &mut Vec<MemoryItem>,
    cap: usize,
    policy: &EvictionPolicy,
) {
    if agent_items.len() <= cap {
        return;
    }
    let pos = match policy {
        EvictionPolicy::LowestImportance => {
            let len = agent_items.len();
            agent_items[..len - 1]
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.importance
                        .partial_cmp(&b.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(pos, _)| pos)
        }
        EvictionPolicy::Oldest => {
            let len = agent_items.len();
            agent_items[..len - 1]
                .iter()
                .enumerate()
                .min_by_key(|(_, item)| item.timestamp)
                .map(|(pos, _)| pos)
        }
    };
    if let Some(pos) = pos {
        agent_items.remove(pos);
    }
}

impl EpisodicStore {
    /// Create a new unbounded episodic store without decay.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: None,
                max_age_hours: None,
                eviction_policy: EvictionPolicy::LowestImportance,
            })),
        }
    }

    /// Return a fluent builder to construct an `EpisodicStore` with any combination of options.
    pub fn builder() -> EpisodicStoreBuilder {
        EpisodicStoreBuilder::default()
    }

    /// Create a new episodic store with the given decay policy.
    pub fn with_decay(policy: DecayPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: Some(policy),
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: None,
                max_age_hours: None,
                eviction_policy: EvictionPolicy::LowestImportance,
            })),
        }
    }

    /// Create a new episodic store with both a decay policy and a recall policy.
    ///
    /// # Warning
    ///
    /// When `recall` is [`RecallPolicy::Hybrid`], decay is applied **before**
    /// scoring, producing a double time penalty.  See [`RecallPolicy`] docs for
    /// details.  Consider using the [`builder`](EpisodicStore::builder) to
    /// configure only one of these time-based mechanisms.
    pub fn with_decay_and_recall_policy(decay: DecayPolicy, recall: RecallPolicy) -> Self {
        if let RecallPolicy::Hybrid { .. } = &recall {
            tracing::warn!(
                "EpisodicStore::with_decay_and_recall_policy called with RecallPolicy::Hybrid \
                 — this applies a double time penalty.  Set DecayPolicy OR Hybrid recency \
                 weighting, not both, unless the double penalty is intentional."
            );
        }
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: Some(decay),
                recall_policy: recall,
                per_agent_capacity: None,
                max_age_hours: None,
                eviction_policy: EvictionPolicy::LowestImportance,
            })),
        }
    }

    /// Create a new episodic store with the given recall policy.
    pub fn with_recall_policy(policy: RecallPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: None,
                recall_policy: policy,
                per_agent_capacity: None,
                max_age_hours: None,
                eviction_policy: EvictionPolicy::LowestImportance,
            })),
        }
    }

    /// Create a new episodic store with the given per-agent capacity limit.
    ///
    /// When an agent exceeds this capacity, the lowest-importance item for that
    /// agent is evicted.
    ///
    /// # Soft-limit semantics
    ///
    /// The capacity is a *soft* limit.  During each [`add_episode`] call the
    /// new item is inserted first, and only then is the lowest-importance item
    /// evicted if the count exceeds `capacity`.  This means the store
    /// momentarily holds `capacity + 1` items per agent while eviction is in
    /// progress.  The newly added item is **never** the one evicted regardless
    /// of its importance score.
    ///
    /// Concurrent calls to `add_episode` may briefly exceed the cap by more
    /// than one item before each call performs its own eviction sweep.
    ///
    /// [`add_episode`]: EpisodicStore::add_episode
    pub fn with_per_agent_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "per_agent_capacity must be > 0");
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: Some(capacity),
                max_age_hours: None,
                eviction_policy: EvictionPolicy::LowestImportance,
            })),
        }
    }

    /// Create a new episodic store with a per-agent capacity limit, without panicking.
    ///
    /// Returns `Err` if `capacity == 0`. Prefer this over [`with_per_agent_capacity`]
    /// in user-facing code where a zero capacity should be a recoverable error
    /// rather than a panic.
    ///
    /// [`with_per_agent_capacity`]: EpisodicStore::with_per_agent_capacity
    pub fn try_with_per_agent_capacity(
        capacity: usize,
    ) -> Result<Self, AgentRuntimeError> {
        if capacity == 0 {
            return Err(AgentRuntimeError::Memory(
                "per_agent_capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: Some(capacity),
                max_age_hours: None,
                eviction_policy: EvictionPolicy::LowestImportance,
            })),
        })
    }

    /// Create a new episodic store with an absolute age limit.
    ///
    /// Items older than `max_age_hours` are automatically purged on the next
    /// `recall` or `add_episode` call for the owning agent.
    ///
    /// # Arguments
    /// * `max_age_hours` — maximum memory age in hours; must be > 0
    pub fn with_max_age(max_age_hours: f64) -> Result<Self, AgentRuntimeError> {
        if max_age_hours <= 0.0 {
            return Err(AgentRuntimeError::Memory(
                "max_age_hours must be positive".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: None,
                max_age_hours: Some(max_age_hours),
                eviction_policy: EvictionPolicy::LowestImportance,
            })),
        })
    }

    /// Create a new episodic store with the given eviction policy.
    pub fn with_eviction_policy(policy: EvictionPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EpisodicInner {
                items: HashMap::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: None,
                max_age_hours: None,
                eviction_policy: policy,
            })),
        }
    }

    /// Record a new episode for the given agent.
    ///
    /// # Returns
    /// The `MemoryId` of the newly created memory item.
    ///
    /// # Errors
    /// Returns `Err(AgentRuntimeError::Memory)` only if the internal mutex is
    /// poisoned (extremely unlikely in normal operation; see [`recover_lock`]).
    ///
    /// # Capacity enforcement
    ///
    /// If the store was created with [`with_per_agent_capacity`], the item is
    /// always inserted first.  If the agent's item count then exceeds the cap,
    /// the single lowest-importance item for that agent is evicted.  See
    /// [`with_per_agent_capacity`] for the full soft-limit semantics.
    ///
    /// [`with_per_agent_capacity`]: EpisodicStore::with_per_agent_capacity
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

        inner.purge_stale(&agent_id);
        let cap = inner.per_agent_capacity; // read before mutable borrow
        let eviction_policy = inner.eviction_policy.clone();
        let agent_items = inner.items.entry(agent_id).or_default();
        agent_items.push(item);

        if let Some(cap) = cap {
            evict_if_over_capacity(agent_items, cap, &eviction_policy);
        }
        Ok(id)
    }

    /// Add an episode with associated tags.
    ///
    /// Convenience wrapper around [`add_episode`] that accepts a tag list in the
    /// same call.  Episode capacity eviction follows the same rules as `add_episode`.
    ///
    /// [`add_episode`]: EpisodicStore::add_episode
    #[tracing::instrument(skip(self))]
    pub fn add_episode_with_tags(
        &self,
        agent_id: AgentId,
        content: impl Into<String> + std::fmt::Debug,
        importance: f32,
        tags: Vec<String>,
    ) -> Result<MemoryId, AgentRuntimeError> {
        let item = MemoryItem::new(agent_id.clone(), content, importance, tags);
        let id = item.id.clone();
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::add_episode_with_tags");
        inner.purge_stale(&agent_id);
        let cap = inner.per_agent_capacity;
        let eviction_policy = inner.eviction_policy.clone();
        let agent_items = inner.items.entry(agent_id).or_default();
        agent_items.push(item);
        if let Some(cap) = cap {
            evict_if_over_capacity(agent_items, cap, &eviction_policy);
        }
        Ok(id)
    }

    /// Remove a specific episode by its `MemoryId`.
    ///
    /// Returns `Ok(true)` if the episode was found and removed, `Ok(false)` if
    /// no episode with that `id` exists for `agent_id`.
    pub fn remove_by_id(
        &self,
        agent_id: &AgentId,
        id: &MemoryId,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::remove_by_id");
        if let Some(items) = inner.items.get_mut(agent_id) {
            if let Some(pos) = items.iter().position(|i| &i.id == id) {
                items.remove(pos);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Update the `tags` of an episode identified by its `MemoryId`.
    ///
    /// Returns `Ok(true)` if found and updated, `Ok(false)` otherwise.
    pub fn update_tags_by_id(
        &self,
        agent_id: &AgentId,
        id: &MemoryId,
        new_tags: Vec<String>,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::update_tags_by_id");
        if let Some(items) = inner.items.get_mut(agent_id) {
            if let Some(item) = items.iter_mut().find(|i| &i.id == id) {
                item.tags = new_tags;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Return the highest importance score across all episodes for `agent_id`.
    ///
    /// Returns `None` if the agent has no stored episodes.
    pub fn max_importance_for(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::max_importance_for");
        let max = inner
            .items
            .get(agent_id)
            .and_then(|items| {
                items
                    .iter()
                    .map(|i| i.importance)
                    .reduce(f32::max)
            });
        Ok(max)
    }

    /// Return the number of episodes stored for `agent_id`.
    ///
    /// Cheaper than `recall(agent, usize::MAX)?.len()` because it does not
    /// sort, clone, or increment recall counts.
    pub fn count_for(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::count_for");
        Ok(inner.items.get(agent_id).map_or(0, |v| v.len()))
    }

    /// Recall up to `limit` episodes for `agent_id` that carry `tag`,
    /// sorted by descending importance.  `limit = 0` returns all matches.
    pub fn recall_by_tag(
        &self,
        agent_id: &AgentId,
        tag: &str,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recall_by_tag");
        let mut matches: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map(|items| {
                items
                    .iter()
                    .filter(|i| i.tags.iter().any(|t| t == tag))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        matches.sort_unstable_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if limit > 0 {
            matches.truncate(limit);
        }
        Ok(matches)
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

        inner.purge_stale(&agent_id);
        let cap = inner.per_agent_capacity; // read before mutable borrow
        let eviction_policy = inner.eviction_policy.clone();
        let agent_items = inner.items.entry(agent_id).or_default();
        agent_items.push(item);

        if let Some(cap) = cap {
            evict_if_over_capacity(agent_items, cap, &eviction_policy);
        }
        Ok(id)
    }

    /// Add multiple episodes for the same agent in a single lock acquisition.
    ///
    /// More efficient than calling [`add_episode`] in a loop when inserting
    /// many items at once.  Returns the generated [`MemoryId`]s in the same
    /// order as `episodes`.
    ///
    /// [`add_episode`]: EpisodicStore::add_episode
    pub fn add_episodes_batch(
        &self,
        agent_id: AgentId,
        episodes: impl IntoIterator<Item = (impl Into<String>, f32)>,
    ) -> Result<Vec<MemoryId>, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::add_episodes_batch");
        inner.purge_stale(&agent_id);
        let cap = inner.per_agent_capacity;
        let eviction_policy = inner.eviction_policy.clone();
        let agent_items = inner.items.entry(agent_id.clone()).or_default();

        let mut ids = Vec::new();
        for (content, importance) in episodes {
            let item = MemoryItem::new(agent_id.clone(), content, importance, Vec::new());
            ids.push(item.id.clone());
            agent_items.push(item);
        }
        if let Some(cap) = cap {
            evict_if_over_capacity(agent_items, cap, &eviction_policy);
        }
        Ok(ids)
    }

    /// Recall up to `limit` memories for the given agent.
    ///
    /// Applies decay if configured, purges stale items if `max_age` is set,
    /// increments `recall_count` for each recalled item, then returns items
    /// sorted according to the configured `RecallPolicy`.
    ///
    /// # Errors
    /// Returns `Err(AgentRuntimeError::Memory)` only if the internal mutex is
    /// poisoned (extremely unlikely in normal operation).
    #[tracing::instrument(skip(self))]
    pub fn recall(
        &self,
        agent_id: &AgentId,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::recall");

        // Clone policy values to avoid borrow conflicts.
        let decay = inner.decay.clone();
        let max_age = inner.max_age_hours;
        let recall_policy = inner.recall_policy.clone();

        // Use get_mut to avoid creating ghost entries for unknown agents.
        if !inner.items.contains_key(agent_id) {
            return Ok(Vec::new());
        }
        let agent_items = inner.items.get_mut(agent_id).unwrap();

        // Apply decay in-place.
        if let Some(ref policy) = decay {
            for item in agent_items.iter_mut() {
                policy.decay_item(item);
            }
        }

        // Purge stale items.
        if let Some(max_age_h) = max_age {
            let cutoff =
                Utc::now() - chrono::Duration::seconds((max_age_h * 3600.0) as i64);
            agent_items.retain(|i| i.timestamp >= cutoff);
        }

        // Build a sorted index list (descending by score) without cloning all items first.
        //
        // When `limit < indices.len()` we use a two-phase partial sort:
        //   1. `select_nth_unstable_by(limit - 1, cmp)` — O(n) — partitions the
        //      slice so that indices[0..limit] are the top-limit elements (unordered)
        //      and indices[limit..] are the remaining (discarded) elements.
        //   2. Sort only indices[0..limit] — O(limit log limit).
        // Total: O(n + limit log limit) vs O(n log n) for a full sort.
        let mut indices: Vec<usize> = (0..agent_items.len()).collect();

        match recall_policy {
            RecallPolicy::Importance => {
                let cmp = |&a: &usize, &b: &usize| {
                    agent_items[b]
                        .importance
                        .partial_cmp(&agent_items[a].importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                };
                if limit > 0 && limit < indices.len() {
                    indices.select_nth_unstable_by(limit - 1, cmp);
                    indices[..limit].sort_unstable_by(cmp);
                } else {
                    indices.sort_unstable_by(cmp);
                }
            }
            RecallPolicy::Hybrid {
                recency_weight,
                frequency_weight,
            } => {
                let max_recall = agent_items
                    .iter()
                    .map(|i| i.recall_count)
                    .max()
                    .unwrap_or(1)
                    .max(1);
                let now = Utc::now();
                let cmp = |&a: &usize, &b: &usize| {
                    let score_a = compute_hybrid_score(
                        &agent_items[a],
                        recency_weight,
                        frequency_weight,
                        max_recall,
                        now,
                    );
                    let score_b = compute_hybrid_score(
                        &agent_items[b],
                        recency_weight,
                        frequency_weight,
                        max_recall,
                        now,
                    );
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                };
                if limit > 0 && limit < indices.len() {
                    indices.select_nth_unstable_by(limit - 1, cmp);
                    indices[..limit].sort_unstable_by(cmp);
                } else {
                    indices.sort_unstable_by(cmp);
                }
            }
        }

        indices.truncate(limit);

        // Increment recall_count only for the surviving items.
        for &idx in &indices {
            agent_items[idx].recall_count += 1;
        }

        // Clone only the surviving items, with already-incremented counts.
        let items: Vec<MemoryItem> = indices.iter().map(|&idx| agent_items[idx].clone()).collect();

        tracing::debug!("recalled {} items", items.len());
        Ok(items)
    }

    /// Recall episodes for `agent_id` that contain **all** of the specified `tags`.
    ///
    /// Returns at most `limit` items ordered by descending importance, consistent
    /// with [`recall`].  Pass an empty `tags` slice to match all episodes.
    ///
    /// [`recall`]: EpisodicStore::recall
    pub fn recall_tagged(
        &self,
        agent_id: &AgentId,
        tags: &[&str],
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recall_tagged");
        let items = inner.items.get(agent_id).cloned().unwrap_or_default();
        drop(inner);
        let mut matched: Vec<MemoryItem> = items
            .into_iter()
            .filter(|item| {
                tags.iter()
                    .all(|t| item.tags.iter().any(|it| it.as_str() == *t))
            })
            .collect();
        matched.sort_unstable_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 {
            matched.truncate(limit);
        }
        Ok(matched)
    }

    /// Retrieve a single episode by its `MemoryId`.
    ///
    /// Returns `Ok(Some(item))` if found, `Ok(None)` if no episode with that ID
    /// exists for `agent_id`.
    pub fn recall_by_id(
        &self,
        agent_id: &AgentId,
        id: &MemoryId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recall_by_id");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|items| items.iter().find(|i| &i.id == id).cloned()))
    }

    /// Import all episodes from `other` for `agent_id` into this store.
    ///
    /// Episodes are appended without deduplication.  Capacity eviction is applied
    /// per-episode exactly as in [`add_episode`].
    ///
    /// [`add_episode`]: EpisodicStore::add_episode
    pub fn merge_from(
        &self,
        other: &EpisodicStore,
        agent_id: &AgentId,
    ) -> Result<usize, AgentRuntimeError> {
        let other_items = {
            let inner = recover_lock(other.inner.lock(), "EpisodicStore::merge_from:read");
            inner.items.get(agent_id).cloned().unwrap_or_default()
        };
        let count = other_items.len();
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::merge_from:write");
        inner.purge_stale(agent_id);
        let cap = inner.per_agent_capacity;
        let bucket = inner.items.entry(agent_id.clone()).or_default();
        for item in other_items {
            if let Some(cap) = cap {
                while bucket.len() >= cap {
                    bucket.remove(0);
                }
            }
            bucket.push(item);
        }
        Ok(count)
    }

    /// Update the importance score of a specific episode in-place.
    ///
    /// Returns `Ok(true)` if the episode was found and updated, `Ok(false)` if no
    /// episode with that `id` exists for `agent_id`.
    ///
    /// `new_importance` is clamped to `[0.0, 1.0]`.
    pub fn update_importance(
        &self,
        agent_id: &AgentId,
        id: &MemoryId,
        new_importance: f32,
    ) -> Result<bool, AgentRuntimeError> {
        let importance = new_importance.clamp(0.0, 1.0);
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::update_importance");
        if let Some(items) = inner.items.get_mut(agent_id) {
            if let Some(item) = items.iter_mut().find(|i| &i.id == id) {
                item.importance = importance;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Recall episodes for `agent_id` inserted at or after `cutoff`.
    ///
    /// Returns items ordered by descending importance (same as [`recall`]).
    /// Pass `limit = 0` to return all matching items.
    ///
    /// [`recall`]: EpisodicStore::recall
    pub fn recall_since(
        &self,
        agent_id: &AgentId,
        cutoff: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recall_since");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|i| i.timestamp >= cutoff)
            .collect();
        drop(inner);
        items.sort_unstable_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if limit > 0 {
            items.truncate(limit);
        }
        Ok(items)
    }

    /// Update the `content` of an episode identified by its `MemoryId`.
    ///
    /// Returns `Ok(true)` if found and updated, `Ok(false)` if no episode with that
    /// `id` exists for `agent_id`.
    pub fn update_content(
        &self,
        agent_id: &AgentId,
        id: &MemoryId,
        new_content: impl Into<String>,
    ) -> Result<bool, AgentRuntimeError> {
        let new_content = new_content.into();
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::update_content");
        if let Some(items) = inner.items.get_mut(agent_id) {
            if let Some(item) = items.iter_mut().find(|i| &i.id == id) {
                item.content = new_content;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Recall the most recently added episodes for `agent_id` in reverse insertion order.
    ///
    /// Unlike [`recall`] (which ranks by importance), this returns the `limit` most
    /// recently inserted items with no re-ordering.  Useful when recency matters more
    /// than importance, e.g. retrieving the latest context window entries.
    ///
    /// [`recall`]: EpisodicStore::recall
    pub fn recall_recent(
        &self,
        agent_id: &AgentId,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recall_recent");
        let items = inner.items.get(agent_id).cloned().unwrap_or_default();
        drop(inner);
        let start = if limit > 0 && limit < items.len() {
            items.len() - limit
        } else {
            0
        };
        Ok(items[start..].iter().rev().cloned().collect())
    }

    /// Retrieve all stored episodes for `agent_id` without any limit.
    ///
    /// Returns items in descending importance order, consistent with [`recall`].
    /// For large stores, prefer [`recall`] with an explicit limit.
    ///
    /// [`recall`]: EpisodicStore::recall
    pub fn recall_all(&self, agent_id: &AgentId) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        self.recall(agent_id, usize::MAX)
    }

    /// Return the top `n` episodes for `agent_id` ordered by descending importance.
    ///
    /// When `n == 0` all episodes are returned. Does not increment `recall_count`.
    pub fn top_n(&self, agent_id: &AgentId, n: usize) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::top_n");
        let mut items = inner.items.get(agent_id).cloned().unwrap_or_default();
        items.sort_unstable_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        if n > 0 {
            items.truncate(n);
        }
        Ok(items)
    }

    /// Return episodes for `agent_id` whose importance is in `[min, max]`, most
    /// important first. Passing `limit == 0` returns all matching episodes.
    pub fn search_by_importance_range(
        &self,
        agent_id: &AgentId,
        min: f32,
        max: f32,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::search_by_importance_range");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |v| {
                v.iter().filter(|i| i.importance >= min && i.importance <= max).cloned().collect()
            });
        items.sort_unstable_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 {
            items.truncate(limit);
        }
        Ok(items)
    }

    /// Return the sum of `recall_count` across all episodes for `agent_id`.
    ///
    /// Useful for tracking aggregate access frequency per agent.
    pub fn total_recall_count(&self, agent_id: &AgentId) -> Result<u64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::total_recall_count");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().map(|i| i.recall_count).sum()))
    }

    /// Return summary statistics (count, min, max, mean importance) for `agent_id`.
    ///
    /// Returns `(0, 0.0, 0.0, 0.0)` if the agent has no stored episodes.
    pub fn importance_stats(&self, agent_id: &AgentId) -> Result<(usize, f32, f32, f32), AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::importance_stats");
        let items = inner.items.get(agent_id).map(|v| v.as_slice()).unwrap_or(&[]);
        if items.is_empty() {
            return Ok((0, 0.0, 0.0, 0.0));
        }
        let count = items.len();
        let min = items.iter().map(|i| i.importance).fold(f32::MAX, f32::min);
        let max = items.iter().map(|i| i.importance).fold(f32::MIN, f32::max);
        let mean = items.iter().map(|i| i.importance).sum::<f32>() / count as f32;
        Ok((count, min, max, mean))
    }

    /// Return the oldest (first-inserted) episode for `agent_id`, or `None`
    /// if the agent has no stored episodes.
    pub fn oldest(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::oldest");
        let item = inner.items.get(agent_id).and_then(|v| v.first()).cloned();
        Ok(item)
    }

    /// Remove all stored episodes for `agent_id`.
    ///
    /// Returns the number of episodes that were removed.
    pub fn clear_agent(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::clear_agent");
        let count = inner.items.remove(agent_id).map_or(0, |v| v.len());
        Ok(count)
    }

    /// Return the most recently inserted episode for `agent_id`, or `None`
    /// if the agent has no stored episodes.
    pub fn newest(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::newest");
        let item = inner.items.get(agent_id).and_then(|v| v.last()).cloned();
        Ok(item)
    }

    /// Return the total number of stored episodes across all agents.
    pub fn len(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::len");
        Ok(inner.items.values().map(|v| v.len()).sum())
    }

    /// Return `true` if no episodes have been stored.
    pub fn is_empty(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.len()? == 0)
    }

    /// Return the number of stored episodes for a specific agent.
    ///
    /// Returns `0` if the agent has no episodes or has not been seen before.
    pub fn agent_memory_count(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agent_memory_count");
        Ok(inner.items.get(agent_id).map_or(0, |v| v.len()))
    }

    /// Return all agent IDs that have at least one stored episode.
    ///
    /// The order of agents in the returned vector is not guaranteed.
    pub fn list_agents(&self) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::list_agents");
        Ok(inner.items.keys().cloned().collect())
    }

    /// Remove all stored episodes for `agent_id` and return the number removed.
    ///
    /// Returns `0` if the agent had no episodes.  Does not affect other agents.
    pub fn purge_agent_memories(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::purge_agent_memories");
        let removed = inner.items.remove(agent_id).map_or(0, |v| v.len());
        Ok(removed)
    }

    /// Remove all memories for the given agent.
    ///
    /// After this call, `recall` for this agent returns an empty list.
    pub fn clear_agent_memory(&self, agent_id: &AgentId) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::clear_agent_memory");
        inner.items.remove(agent_id);
        Ok(())
    }

    /// Remove **all** episodes for **all** agents.
    ///
    /// After this call `len()` returns `0` and `list_agents()` returns an empty slice.
    pub fn clear_all(&self) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::clear_all");
        inner.items.clear();
        Ok(())
    }

    /// Export all memories for the given agent as a serializable Vec.
    ///
    /// Useful for migrating agent state across runtime instances.
    pub fn export_agent_memory(&self, agent_id: &AgentId) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::export_agent_memory");
        Ok(inner.items.get(agent_id).cloned().unwrap_or_default())
    }

    /// Import a Vec of MemoryItems for the given agent, replacing any existing memories.
    ///
    /// The agent's existing memories are completely replaced by the imported items.
    pub fn import_agent_memory(&self, agent_id: &AgentId, items: Vec<MemoryItem>) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::import_agent_memory");
        inner.items.insert(agent_id.clone(), items);
        Ok(())
    }

    /// Bump the `recall_count` of every item whose content equals `content` by `amount`.
    ///
    /// This method exists to support integration tests that need to simulate prior recall
    /// history without accessing private fields. It is not intended for production use.
    #[doc(hidden)]
    pub fn bump_recall_count_by_content(&self, content: &str, amount: u64) {
        let mut inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::bump_recall_count_by_content",
        );
        for agent_items in inner.items.values_mut() {
            for item in agent_items.iter_mut() {
                if item.content == content {
                    item.recall_count = item.recall_count.saturating_add(amount);
                }
            }
        }
    }

    /// Search episodes for a given `agent_id` whose content contains `query` as a substring.
    ///
    /// The comparison is case-sensitive.  Returns at most `limit` matching items,
    /// ordered by descending importance (same as [`recall`]).
    ///
    /// [`recall`]: EpisodicStore::recall
    pub fn search_by_content(
        &self,
        agent_id: &AgentId,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::search_by_content");
        let items = inner.items.get(agent_id).cloned().unwrap_or_default();
        drop(inner);
        let mut matched: Vec<MemoryItem> = items
            .into_iter()
            .filter(|item| item.content.contains(query))
            .collect();
        matched.sort_unstable_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 {
            matched.truncate(limit);
        }
        Ok(matched)
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
    inner: Arc<Mutex<SemanticInner>>,
}

#[derive(Debug)]
struct SemanticInner {
    entries: Vec<SemanticEntry>,
    expected_dim: Option<usize>,
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
            inner: Arc::new(Mutex::new(SemanticInner {
                entries: Vec::new(),
                expected_dim: None,
            })),
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
        inner.entries.push(SemanticEntry {
            key: key.into(),
            value: value.into(),
            tags,
            embedding: None,
        });
        Ok(())
    }

    /// Store a key-value pair with an embedding vector for similarity search.
    ///
    /// # Errors
    /// Returns `Err(AgentRuntimeError::Memory)` if `embedding` is empty or dimension mismatches.
    #[tracing::instrument(skip(self))]
    pub fn store_with_embedding(
        &self,
        key: impl Into<String> + std::fmt::Debug,
        value: impl Into<String> + std::fmt::Debug,
        tags: Vec<String>,
        embedding: Vec<f32>,
    ) -> Result<(), AgentRuntimeError> {
        if embedding.is_empty() {
            return Err(AgentRuntimeError::Memory(
                "embedding vector must not be empty".into(),
            ));
        }
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::store_with_embedding");
        // Validate dimension consistency using expected_dim.
        if let Some(expected) = inner.expected_dim {
            if expected != embedding.len() {
                return Err(AgentRuntimeError::Memory(format!(
                    "embedding dimension mismatch: expected {expected}, got {}",
                    embedding.len()
                )));
            }
        } else {
            inner.expected_dim = Some(embedding.len());
        }
        // Pre-normalize so retrieve_similar can use dot product instead of
        // full cosine similarity for a ~3× speedup on large stores.
        let mut embedding = embedding;
        normalize_in_place(&mut embedding);
        inner.entries.push(SemanticEntry {
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
            .entries
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
    /// Returns `Err(AgentRuntimeError::Memory)` if `query_embedding` dimension mismatches.
    ///
    /// [`store_with_embedding`]: SemanticStore::store_with_embedding
    #[tracing::instrument(skip(self, query_embedding))]
    pub fn retrieve_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, String, f32)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::retrieve_similar");

        // Check dimension against expected_dim.
        if let Some(expected) = inner.expected_dim {
            if expected != query_embedding.len() {
                return Err(AgentRuntimeError::Memory(format!(
                    "query embedding dimension mismatch: expected {expected}, got {}",
                    query_embedding.len()
                )));
            }
        }

        // Normalize the query so we can use dot product against pre-normalized
        // stored embeddings (equivalent to cosine similarity, but faster).
        let query_norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm < f32::EPSILON {
            return Ok(vec![]);
        }
        let query_unit: Vec<f32> = query_embedding.iter().map(|x| x / query_norm).collect();

        let mut scored: Vec<(String, String, f32)> = inner
            .entries
            .iter()
            .filter_map(|entry| {
                entry.embedding.as_ref().map(|emb| {
                    let sim = emb
                        .iter()
                        .zip(query_unit.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        .clamp(-1.0, 1.0);
                    (entry.key.clone(), entry.value.clone(), sim)
                })
            })
            .collect();

        // Partial sort: when top_k < total candidates, select_nth_unstable_by
        // partitions in O(n), then sort only the top top_k in O(top_k log top_k).
        let cmp = |a: &(String, String, f32), b: &(String, String, f32)| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        };
        if top_k > 0 && top_k < scored.len() {
            scored.select_nth_unstable_by(top_k - 1, cmp);
            scored[..top_k].sort_unstable_by(cmp);
        } else {
            scored.sort_unstable_by(cmp);
        }
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Update the `value` of the first entry whose key matches `key`.
    ///
    /// Returns `Ok(true)` if found and updated, `Ok(false)` if not found.
    pub fn update(&self, key: &str, new_value: impl Into<String>) -> Result<bool, AgentRuntimeError> {
        let new_value = new_value.into();
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::update");
        if let Some(entry) = inner.entries.iter_mut().find(|e| e.key == key) {
            entry.value = new_value;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Look up a single entry by its exact key.
    ///
    /// Returns `Ok(Some((value, tags)))` if found, `Ok(None)` if not.
    pub fn retrieve_by_key(&self, key: &str) -> Result<Option<(String, Vec<String>)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::retrieve_by_key");
        Ok(inner.entries.iter().find(|e| e.key == key).map(|e| (e.value.clone(), e.tags.clone())))
    }

    /// Return `true` if an entry with the given key is stored.
    pub fn contains(&self, key: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::contains");
        Ok(inner.entries.iter().any(|e| e.key == key))
    }

    /// Remove all entries.
    pub fn clear(&self) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::clear");
        inner.entries.clear();
        inner.expected_dim = None;
        Ok(())
    }

    /// Count entries that contain `tag` (case-sensitive).
    pub fn count_by_tag(&self, tag: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::count_by_tag");
        Ok(inner
            .entries
            .iter()
            .filter(|e| e.tags.iter().any(|t| t.as_str() == tag))
            .count())
    }

    /// Return all unique tags present across all stored entries, in sorted order.
    pub fn list_tags(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::list_tags");
        let mut tags: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for entry in &inner.entries {
            for tag in &entry.tags {
                tags.insert(tag.clone());
            }
        }
        Ok(tags.into_iter().collect())
    }

    /// Return the number of distinct tags across all stored entries.
    ///
    /// Equivalent to `list_tags()?.len()` but avoids allocating the full tag list.
    pub fn tag_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::tag_count");
        let distinct: std::collections::HashSet<&str> = inner
            .entries
            .iter()
            .flat_map(|e| e.tags.iter().map(|t| t.as_str()))
            .collect();
        Ok(distinct.len())
    }

    /// Return all entry keys that contain `tag` (case-sensitive).
    pub fn keys_with_tag(&self, tag: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::keys_with_tag");
        Ok(inner
            .entries
            .iter()
            .filter(|e| e.tags.iter().any(|t| t.as_str() == tag))
            .map(|e| e.key.clone())
            .collect())
    }

    /// Return all stored entry keys in insertion order.
    ///
    /// Duplicate keys (if any) will appear multiple times.
    pub fn list_keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::list_keys");
        Ok(inner.entries.iter().map(|e| e.key.clone()).collect())
    }

    /// Return all stored entry keys in insertion order.
    ///
    /// Alias for [`list_keys`] using more idiomatic naming.
    ///
    /// [`list_keys`]: SemanticStore::list_keys
    pub fn keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        self.list_keys()
    }

    /// Update the tags of the first entry whose key matches `key`.
    ///
    /// Returns `Ok(true)` if found and updated, `Ok(false)` if not found.
    pub fn update_tags(
        &self,
        key: &str,
        new_tags: Vec<String>,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::update_tags");
        if let Some(entry) = inner.entries.iter_mut().find(|e| e.key == key) {
            entry.tags = new_tags;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Return the total number of stored entries.
    pub fn len(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::len");
        Ok(inner.entries.len())
    }

    /// Return `true` if no entries have been stored.
    pub fn is_empty(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.len()? == 0)
    }

    /// Return the number of stored entries.
    ///
    /// Alias for [`len`] using conventional naming.
    ///
    /// [`len`]: SemanticStore::len
    pub fn count(&self) -> Result<usize, AgentRuntimeError> {
        self.len()
    }

    /// Remove the first entry with key `key`.
    ///
    /// Returns `Ok(true)` if an entry was found and removed, `Ok(false)` if
    /// no entry with that key exists.
    pub fn remove(&self, key: &str) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::remove");
        let before = inner.entries.len();
        inner.entries.retain(|e| e.key != key);
        Ok(inner.entries.len() < before)
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

    /// Insert multiple key-value pairs with a single lock acquisition.
    ///
    /// Each entry follows the same eviction semantics as [`set`]: if the key
    /// already exists it is updated in-place; if inserting a new key would
    /// exceed capacity, the oldest key is evicted first.
    ///
    /// [`set`]: WorkingMemory::set
    pub fn set_many(
        &self,
        pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Result<(), AgentRuntimeError> {
        let capacity = self.capacity;
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::set_many");
        for (key, value) in pairs {
            let key: String = key.into();
            let value: String = value.into();
            if inner.map.contains_key(&key) {
                inner.order.retain(|k| k != &key);
            } else if inner.map.len() >= capacity {
                if let Some(oldest) = inner.order.pop_front() {
                    inner.map.remove(&oldest);
                }
            }
            inner.order.push_back(key.clone());
            inner.map.insert(key, value);
        }
        Ok(())
    }

    /// Update the value of an existing key without inserting if absent.
    ///
    /// Returns `Ok(true)` if the key existed and was updated, `Ok(false)` if not present.
    pub fn update_if_exists(
        &self,
        key: &str,
        value: impl Into<String>,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::update_if_exists");
        if let Some(v) = inner.map.get_mut(key) {
            *v = value.into();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Rename a key without changing its value or insertion order.
    ///
    /// Returns `true` if `old_key` existed and was renamed.  Returns `false` if
    /// `old_key` is not present.  If `new_key` already exists it is overwritten
    /// and its old value is lost.
    pub fn rename(
        &self,
        old_key: &str,
        new_key: impl Into<String>,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::rename");
        let value = match inner.map.remove(old_key) {
            None => return Ok(false),
            Some(v) => v,
        };
        let new_key = new_key.into();
        // Update the insertion-order list.
        if let Some(pos) = inner.order.iter().position(|k| k == old_key) {
            inner.order[pos] = new_key.clone();
        }
        inner.map.insert(new_key, value);
        Ok(true)
    }

    /// Retrieve multiple values in a single lock acquisition.
    ///
    /// Returns a `Vec` of the same length as `keys`.  Each entry is `Some(value)`
    /// if the key is present, `None` if not.
    pub fn get_many(&self, keys: &[&str]) -> Result<Vec<Option<String>>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::get_many");
        Ok(keys.iter().map(|k| inner.map.get(*k).cloned()).collect())
    }

    /// Return `true` if a value is stored under `key`.
    pub fn contains(&self, key: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::contains");
        Ok(inner.map.contains_key(key))
    }

    /// Return the value associated with `key`, or `default` if not set.
    pub fn get_or_default(
        &self,
        key: &str,
        default: impl Into<String>,
    ) -> Result<String, AgentRuntimeError> {
        Ok(self.get(key)?.unwrap_or_else(|| default.into()))
    }

    /// Remove a single entry by key.  Returns `true` if the key existed.
    pub fn remove(&self, key: &str) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::remove");
        if inner.map.remove(key).is_some() {
            inner.order.retain(|k| k != key);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Return all keys in insertion order.
    pub fn keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys");
        Ok(inner.order.iter().cloned().collect())
    }

    /// Return all values in insertion order (parallel to [`keys`]).
    ///
    /// [`keys`]: WorkingMemory::keys
    pub fn values(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::values");
        Ok(inner
            .order
            .iter()
            .filter_map(|k| inner.map.get(k).cloned())
            .collect())
    }

    /// Remove all entries from working memory.
    pub fn clear(&self) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::clear");
        inner.map.clear();
        inner.order.clear();
        Ok(())
    }

    /// Remove all entries and return them as a `Vec<(key, value)>` in insertion order.
    ///
    /// After this call the memory is empty. Useful for atomically moving the
    /// contents to another data structure.
    pub fn drain(&self) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::drain");
        let pairs: Vec<(String, String)> = inner
            .order
            .iter()
            .filter_map(|k| inner.map.get(k).map(|v| (k.clone(), v.clone())))
            .collect();
        inner.map.clear();
        inner.order.clear();
        Ok(pairs)
    }

    /// Clone all current entries into a `HashMap<String, String>`.
    ///
    /// Unlike [`drain`], this leaves the working memory unchanged.
    ///
    /// [`drain`]: WorkingMemory::drain
    pub fn snapshot(&self) -> Result<std::collections::HashMap<String, String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::snapshot");
        Ok(inner.map.clone())
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

    /// Iterate over all key-value pairs in insertion order.
    ///
    /// Equivalent to [`entries`]; provided as a more idiomatic name
    /// for `for`-loop patterns.
    ///
    /// [`entries`]: WorkingMemory::entries
    pub fn iter(&self) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        self.entries()
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

    /// Remove and return the oldest entry (first inserted that is still present).
    ///
    /// Returns `None` if the memory is empty.
    pub fn pop_oldest(&self) -> Result<Option<(String, String)>, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::pop_oldest");
        if let Some(key) = inner.order.pop_front() {
            let value = inner.map.remove(&key).unwrap_or_default();
            Ok(Some((key, value)))
        } else {
            Ok(None)
        }
    }

    /// Return the maximum number of entries this store can hold.
    ///
    /// When [`len`] reaches this value, the oldest entry is evicted on the
    /// next [`set`] call for a new key.
    ///
    /// [`len`]: WorkingMemory::len
    /// [`set`]: WorkingMemory::set
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remove all entries for which `predicate(key, value)` returns `false`.
    ///
    /// Preserves insertion order of the surviving entries.
    /// Returns the number of entries removed.
    pub fn retain<F>(&self, mut predicate: F) -> Result<usize, AgentRuntimeError>
    where
        F: FnMut(&str, &str) -> bool,
    {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::retain");
        let before = inner.map.len();
        inner.map.retain(|k, v| predicate(k.as_str(), v.as_str()));
        let surviving: std::collections::HashSet<String> =
            inner.map.keys().cloned().collect();
        inner.order.retain(|k| surviving.contains(k));
        Ok(before - inner.map.len())
    }

    /// Return all keys in insertion order.
    pub fn get_all_keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::get_all_keys");
        Ok(inner.order.iter().cloned().collect())
    }

    /// Atomically replace **all** entries with `map`.
    ///
    /// The new entries are stored in iteration order of `map`.  Capacity
    /// limits apply: if `map.len() > capacity`, only the last `capacity`
    /// entries (in iteration order) are retained.
    pub fn replace_all(
        &self,
        map: std::collections::HashMap<String, String>,
    ) -> Result<(), AgentRuntimeError> {
        let capacity = self.capacity;
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::replace_all");
        inner.map.clear();
        inner.order.clear();
        for (k, v) in map {
            if inner.map.len() >= capacity {
                if let Some(oldest) = inner.order.pop_front() {
                    inner.map.remove(&oldest);
                }
            }
            inner.order.push_back(k.clone());
            inner.map.insert(k, v);
        }
        Ok(())
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

        // Insert an old item by manipulating via add_episode_at.
        let old_ts = Utc::now() - chrono::Duration::hours(1);
        store
            .add_episode_at(agent.clone(), "old event", 1.0, old_ts)
            .unwrap();

        let items = store.recall(&agent, 10).unwrap();
        // With half_life=0.001h and age=1h, importance should be near 0
        assert_eq!(items.len(), 1);
        assert!(
            items[0].importance < 0.01,
            "expected near-zero importance, got {}",
            items[0].importance
        );
    }

    // ── max_age eviction ──────────────────────────────────────────────────────

    #[test]
    fn test_max_age_rejects_zero() {
        assert!(EpisodicStore::with_max_age(0.0).is_err());
    }

    #[test]
    fn test_max_age_rejects_negative() {
        assert!(EpisodicStore::with_max_age(-1.0).is_err());
    }

    #[test]
    fn test_max_age_evicts_old_items_on_recall() {
        // max_age = 0.001 hours (~3.6 seconds) — items 1 hour old should be evicted
        let store = EpisodicStore::with_max_age(0.001).unwrap();
        let agent = AgentId::new("a");

        let old_ts = Utc::now() - chrono::Duration::hours(1);
        store
            .add_episode_at(agent.clone(), "old", 0.9, old_ts)
            .unwrap();
        store.add_episode(agent.clone(), "new", 0.5).unwrap();

        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items.len(), 1, "old item should be evicted by max_age");
        assert_eq!(items[0].content, "new");
    }

    #[test]
    fn test_max_age_evicts_old_items_on_add() {
        let store = EpisodicStore::with_max_age(0.001).unwrap();
        let agent = AgentId::new("a");

        let old_ts = Utc::now() - chrono::Duration::hours(1);
        store
            .add_episode_at(agent.clone(), "old", 0.9, old_ts)
            .unwrap();
        // Adding a new item triggers stale purge.
        store.add_episode(agent.clone(), "new", 0.5).unwrap();

        assert_eq!(store.len().unwrap(), 1);
    }

    // ── RecallPolicy / per-agent capacity tests ───────────────────────────────

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
        let store = EpisodicStore::with_recall_policy(RecallPolicy::Hybrid {
            recency_weight: 0.1,
            frequency_weight: 2.0,
        });
        let agent = AgentId::new("agent-hybrid");

        let old_ts = Utc::now() - chrono::Duration::hours(48);
        store
            .add_episode_at(agent.clone(), "old_frequent", 0.5, old_ts)
            .unwrap();
        store.add_episode(agent.clone(), "new_never", 0.5).unwrap();

        // Simulate many prior recalls of "old_frequent".
        store.bump_recall_count_by_content("old_frequent", 100);

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

        // Add two items; capacity is full.
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        // Adding "new" triggers eviction of the EXISTING lowest-importance item
        // ("low", 0.1) — the newly added item is never the one evicted.
        store.add_episode(agent.clone(), "new", 0.5).unwrap();

        assert_eq!(
            store.len().unwrap(),
            2,
            "store should hold exactly 2 items after eviction"
        );

        let items = store.recall(&agent, 10).unwrap();
        let contents: Vec<&str> = items.iter().map(|i| i.content.as_str()).collect();
        assert!(
            !contents.contains(&"low"),
            "the pre-existing lowest-importance item should have been evicted; remaining: {:?}",
            contents
        );
        assert!(
            contents.contains(&"new"),
            "the newly added item must never be evicted; remaining: {:?}",
            contents
        );
    }

    // ── O(1) agent isolation ──────────────────────────────────────────────────

    #[test]
    fn test_many_agents_do_not_see_each_others_memories() {
        let store = EpisodicStore::new();
        let n_agents = 20usize;
        for i in 0..n_agents {
            let agent = AgentId::new(format!("agent-{i}"));
            for j in 0..5 {
                store
                    .add_episode(agent.clone(), format!("item-{i}-{j}"), 0.5)
                    .unwrap();
            }
        }
        // Each agent should only see its own 5 items.
        for i in 0..n_agents {
            let agent = AgentId::new(format!("agent-{i}"));
            let items = store.recall(&agent, 100).unwrap();
            assert_eq!(
                items.len(),
                5,
                "agent {i} should see exactly 5 items, got {}",
                items.len()
            );
            for item in &items {
                assert!(
                    item.content.starts_with(&format!("item-{i}-")),
                    "agent {i} saw foreign item: {}",
                    item.content
                );
            }
        }
    }

    // ── agent_memory_count / list_agents / purge_agent_memories ──────────────

    #[test]
    fn test_agent_memory_count_returns_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let count = store.agent_memory_count(&AgentId::new("ghost")).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_agent_memory_count_tracks_insertions() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.5).unwrap();
        store.add_episode(agent.clone(), "e2", 0.5).unwrap();
        assert_eq!(store.agent_memory_count(&agent).unwrap(), 2);
    }

    #[test]
    fn test_list_agents_returns_all_known_agents() {
        let store = EpisodicStore::new();
        let a = AgentId::new("agent-a");
        let b = AgentId::new("agent-b");
        store.add_episode(a.clone(), "x", 0.5).unwrap();
        store.add_episode(b.clone(), "y", 0.5).unwrap();
        let agents = store.list_agents().unwrap();
        assert_eq!(agents.len(), 2);
        assert!(agents.contains(&a));
        assert!(agents.contains(&b));
    }

    #[test]
    fn test_list_agents_empty_when_no_episodes() {
        let store = EpisodicStore::new();
        let agents = store.list_agents().unwrap();
        assert!(agents.is_empty());
    }

    #[test]
    fn test_purge_agent_memories_removes_all_for_agent() {
        let store = EpisodicStore::new();
        let a = AgentId::new("a");
        let b = AgentId::new("b");
        store.add_episode(a.clone(), "ep1", 0.5).unwrap();
        store.add_episode(a.clone(), "ep2", 0.5).unwrap();
        store.add_episode(b.clone(), "ep-b", 0.5).unwrap();

        let removed = store.purge_agent_memories(&a).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.agent_memory_count(&a).unwrap(), 0);
        assert_eq!(store.agent_memory_count(&b).unwrap(), 1);
        assert_eq!(store.len().unwrap(), 1);
    }

    #[test]
    fn test_purge_agent_memories_returns_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let removed = store.purge_agent_memories(&AgentId::new("ghost")).unwrap();
        assert_eq!(removed, 0);
    }

    // ── All-stale recall ──────────────────────────────────────────────────────

    #[test]
    fn test_recall_returns_empty_when_all_items_are_stale() {
        // max_age = 0.001 hours — all items inserted 1 hour ago will be evicted.
        let store = EpisodicStore::with_max_age(0.001).unwrap();
        let agent = AgentId::new("stale-agent");

        let old_ts = Utc::now() - chrono::Duration::hours(1);
        store
            .add_episode_at(agent.clone(), "stale-1", 0.9, old_ts)
            .unwrap();
        store
            .add_episode_at(agent.clone(), "stale-2", 0.7, old_ts)
            .unwrap();

        let items = store.recall(&agent, 100).unwrap();
        assert!(
            items.is_empty(),
            "all stale items should be evicted on recall, got {}",
            items.len()
        );
    }

    // ── Concurrency stress tests ──────────────────────────────────────────────

    #[test]
    fn test_concurrent_add_and_recall_are_consistent() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(EpisodicStore::new());
        let agent = AgentId::new("concurrent-agent");
        let n_threads = 8;
        let items_per_thread = 25;

        // Spawn writers.
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let s = Arc::clone(&store);
            let a = agent.clone();
            handles.push(thread::spawn(move || {
                for i in 0..items_per_thread {
                    s.add_episode(a.clone(), format!("t{t}-i{i}"), 0.5).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Spawn readers.
        let mut read_handles = Vec::new();
        for _ in 0..n_threads {
            let s = Arc::clone(&store);
            let a = agent.clone();
            read_handles.push(thread::spawn(move || {
                let items = s.recall(&a, 1000).unwrap();
                assert!(items.len() <= n_threads * items_per_thread);
            }));
        }
        for h in read_handles {
            h.join().unwrap();
        }
    }

    // ── Concurrent capacity eviction ──────────────────────────────────────────

    #[test]
    fn test_concurrent_capacity_eviction_never_exceeds_cap() {
        use std::sync::Arc;
        use std::thread;

        let cap = 5usize;
        let store = Arc::new(EpisodicStore::with_per_agent_capacity(cap));
        let agent = AgentId::new("cap-agent");
        let n_threads = 8;
        let items_per_thread = 10;

        let mut handles = Vec::new();
        for t in 0..n_threads {
            let s = Arc::clone(&store);
            let a = agent.clone();
            handles.push(thread::spawn(move || {
                for i in 0..items_per_thread {
                    let importance = (t * items_per_thread + i) as f32 / 100.0;
                    s.add_episode(a.clone(), format!("t{t}-i{i}"), importance)
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // The store may momentarily exceed cap+1 during concurrent eviction,
        // but after all threads complete the final count must be <= cap + n_threads.
        // (Each thread's last insert may not have been evicted yet.)
        // The strong invariant: agent_memory_count <= cap + n_threads - 1.
        let count = store.agent_memory_count(&agent).unwrap();
        assert!(
            count <= cap + n_threads,
            "expected at most {} items, got {}",
            cap + n_threads,
            count
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
    fn test_semantic_store_empty_embedding_is_rejected() {
        let store = SemanticStore::new();
        let result = store.store_with_embedding("k", "v", vec![], vec![]);
        assert!(result.is_err(), "empty embedding should be rejected");
    }

    #[test]
    fn test_semantic_store_dimension_mismatch_is_rejected() {
        let store = SemanticStore::new();
        store
            .store_with_embedding("k1", "v1", vec![], vec![1.0, 0.0])
            .unwrap();
        // Different dimension
        let result = store.store_with_embedding("k2", "v2", vec![], vec![1.0, 0.0, 0.0]);
        assert!(
            result.is_err(),
            "embedding dimension mismatch should be rejected"
        );
    }

    #[test]
    fn test_semantic_store_retrieve_similar_returns_closest() {
        let store = SemanticStore::new();
        store
            .store_with_embedding("close", "close value", vec![], vec![1.0, 0.0, 0.0])
            .unwrap();
        store
            .store_with_embedding("far", "far value", vec![], vec![0.0, 1.0, 0.0])
            .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = store.retrieve_similar(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "close");
        assert!(
            (results[0].2 - 1.0).abs() < 1e-5,
            "expected similarity ~1.0, got {}",
            results[0].2
        );
        assert!(
            (results[1].2).abs() < 1e-5,
            "expected similarity ~0.0, got {}",
            results[1].2
        );
    }

    #[test]
    fn test_semantic_store_retrieve_similar_ignores_unembedded_entries() {
        let store = SemanticStore::new();
        store.store("no-emb", "no embedding value", vec![]).unwrap();
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
        let store = SemanticStore::new();
        store
            .store_with_embedding("a", "va", vec![], vec![1.0, 0.0])
            .unwrap();
        store
            .store_with_embedding("b", "vb", vec![], vec![0.0, 1.0])
            .unwrap();

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

    // ── Improvement 6: SemanticStore dimension validation on retrieve ──────────

    #[test]
    fn test_semantic_dimension_mismatch_on_retrieve_returns_error() {
        let store = SemanticStore::new();
        store
            .store_with_embedding("k1", "v1", vec![], vec![1.0, 0.0, 0.0])
            .unwrap();
        // Query with wrong dimension
        let result = store.retrieve_similar(&[1.0, 0.0], 10);
        assert!(result.is_err(), "dimension mismatch on retrieve should error");
    }

    // ── Improvement 12: EvictionPolicy::Oldest ────────────────────────────────

    // ── #3 clear_agent_memory ────────────────────────────────────────────────

    #[test]
    fn test_clear_agent_memory_removes_all_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "ep1", 0.5).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.9).unwrap();
        store.clear_agent_memory(&agent).unwrap();
        let items = store.recall(&agent, 10).unwrap();
        assert!(items.is_empty(), "all memories should be cleared");
    }

    // ── #13 AgentId::as_str / MemoryId::as_str ───────────────────────────────

    #[test]
    fn test_agent_id_as_str() {
        let id = AgentId::new("hello");
        assert_eq!(id.as_str(), "hello");
    }

    // ── #15 export/import round trip ─────────────────────────────────────────

    #[test]
    fn test_export_import_agent_memory_round_trip() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("export-agent");
        store.add_episode(agent.clone(), "fact1", 0.8).unwrap();
        store.add_episode(agent.clone(), "fact2", 0.6).unwrap();

        let exported = store.export_agent_memory(&agent).unwrap();
        assert_eq!(exported.len(), 2);

        let new_store = EpisodicStore::new();
        new_store.import_agent_memory(&agent, exported).unwrap();
        let recalled = new_store.recall(&agent, 10).unwrap();
        assert_eq!(recalled.len(), 2);
    }

    // ── #19 WorkingMemory::iter ───────────────────────────────────────────────

    #[test]
    fn test_working_memory_iter_matches_entries() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        let via_iter = wm.iter().unwrap();
        let via_entries = wm.entries().unwrap();
        assert_eq!(via_iter, via_entries);
    }

    // ── #37 AsRef<str> for AgentId and MemoryId ──────────────────────────────

    #[test]
    fn test_agent_id_as_ref_str() {
        let id = AgentId::new("ref-test");
        let s: &str = id.as_ref();
        assert_eq!(s, "ref-test");
    }

    #[test]
    fn test_eviction_policy_oldest_evicts_first_inserted() {
        let store = EpisodicStore::with_eviction_policy(EvictionPolicy::Oldest);
        // Override capacity by building a combined store.
        // We need a store with capacity=2 AND oldest eviction.
        // Use `with_per_agent_capacity` approach on a new store then set policy.
        // Since we can't combine constructors directly yet, we test via a different path:
        // Insert items and check that the oldest is evicted.
        let store = {
            // Build internal store with per_agent_capacity=2 and Oldest policy
            let inner = EpisodicInner {
                items: std::collections::HashMap::new(),
                decay: None,
                recall_policy: RecallPolicy::Importance,
                per_agent_capacity: Some(2),
                max_age_hours: None,
                eviction_policy: EvictionPolicy::Oldest,
            };
            EpisodicStore {
                inner: std::sync::Arc::new(std::sync::Mutex::new(inner)),
            }
        };

        let agent = AgentId::new("agent");
        // Add items with distinct timestamps by using add_episode_at
        let t1 = chrono::Utc::now() - chrono::Duration::seconds(100);
        let t2 = chrono::Utc::now() - chrono::Duration::seconds(50);
        store.add_episode_at(agent.clone(), "oldest", 0.9, t1).unwrap();
        store.add_episode_at(agent.clone(), "newer", 0.8, t2).unwrap();
        // Adding a third item should evict "oldest" (earliest timestamp)
        store.add_episode(agent.clone(), "newest", 0.5).unwrap();

        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items.len(), 2);
        let contents: Vec<&str> = items.iter().map(|i| i.content.as_str()).collect();
        assert!(!contents.contains(&"oldest"), "oldest item should have been evicted; got: {contents:?}");
    }

    // ── New API tests (Rounds 4-8) ────────────────────────────────────────────

    #[test]
    fn test_search_by_content_finds_matching_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "the quick brown fox", 0.9).unwrap();
        store.add_episode(agent.clone(), "jumps over the lazy dog", 0.5).unwrap();
        store.add_episode(agent.clone(), "hello world", 0.7).unwrap();

        let results = store.search_by_content(&agent, "the", 10).unwrap();
        assert_eq!(results.len(), 2);
        // results are sorted by importance descending
        assert_eq!(results[0].content, "the quick brown fox");
    }

    #[test]
    fn test_search_by_content_returns_empty_on_no_match() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hello", 0.5).unwrap();
        let results = store.search_by_content(&agent, "xyz", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_recall_tagged_filters_by_all_tags() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let mut inner = store.inner.lock().unwrap();
        let item1 = MemoryItem { id: MemoryId::random(), agent_id: agent.clone(), content: "rust".into(), importance: 0.8, timestamp: Utc::now(), tags: vec!["lang".into(), "sys".into()], recall_count: 0 };
        let item2 = MemoryItem { id: MemoryId::random(), agent_id: agent.clone(), content: "python".into(), importance: 0.6, timestamp: Utc::now(), tags: vec!["lang".into()], recall_count: 0 };
        inner.items.entry(agent.clone()).or_default().push(item1);
        inner.items.entry(agent.clone()).or_default().push(item2);
        drop(inner);

        let results = store.recall_tagged(&agent, &["lang", "sys"], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "rust");

        let all = store.recall_tagged(&agent, &["lang"], 10).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_recall_recent_returns_newest_first() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "first", 0.3).unwrap();
        store.add_episode(agent.clone(), "second", 0.5).unwrap();
        store.add_episode(agent.clone(), "third", 0.9).unwrap();

        let recent = store.recall_recent(&agent, 2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "third");
        assert_eq!(recent[1].content, "second");
    }

    #[test]
    fn test_recall_by_id_finds_specific_episode() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store.add_episode(agent.clone(), "specific", 0.7).unwrap();
        store.add_episode(agent.clone(), "other", 0.5).unwrap();

        let found = store.recall_by_id(&agent, &id).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().content, "specific");
    }

    #[test]
    fn test_recall_by_id_returns_none_for_unknown_id() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let result = store.recall_by_id(&agent, &MemoryId::random()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_update_importance_changes_score() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store.add_episode(agent.clone(), "item", 0.5).unwrap();
        let updated = store.update_importance(&agent, &id, 0.9).unwrap();
        assert!(updated);
        let item = store.recall_by_id(&agent, &id).unwrap().unwrap();
        assert!((item.importance - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_merge_from_imports_episodes() {
        let src = EpisodicStore::new();
        let agent = AgentId::new("a");
        src.add_episode(agent.clone(), "ep1", 0.8).unwrap();
        src.add_episode(agent.clone(), "ep2", 0.6).unwrap();

        let dst = EpisodicStore::new();
        let count = dst.merge_from(&src, &agent).unwrap();
        assert_eq!(count, 2);
        assert_eq!(dst.agent_memory_count(&agent).unwrap(), 2);
    }

    #[test]
    fn test_memory_item_age_hours_is_non_negative() {
        let item = MemoryItem::new(AgentId::new("a"), "test", 0.5, vec![]);
        let age = item.age_hours();
        assert!(age >= 0.0);
    }

    #[test]
    fn test_working_memory_remove_and_contains() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "v").unwrap();
        assert!(wm.contains("k").unwrap());
        let removed = wm.remove("k").unwrap();
        assert!(removed);
        assert!(!wm.contains("k").unwrap());
        assert!(!wm.remove("k").unwrap()); // second remove returns false
    }

    #[test]
    fn test_working_memory_keys_in_insertion_order() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("b", "2").unwrap();
        wm.set("a", "1").unwrap();
        wm.set("c", "3").unwrap();
        assert_eq!(wm.keys().unwrap(), vec!["b", "a", "c"]);
    }

    #[test]
    fn test_working_memory_set_many_batch_insert() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set_many([("x", "1"), ("y", "2"), ("z", "3")]).unwrap();
        assert_eq!(wm.len().unwrap(), 3);
        assert_eq!(wm.get("y").unwrap(), Some("2".into()));
    }

    #[test]
    fn test_working_memory_get_or_default() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("a", "val").unwrap();
        assert_eq!(wm.get_or_default("a", "fallback").unwrap(), "val");
        assert_eq!(wm.get_or_default("missing", "fallback").unwrap(), "fallback");
    }

    #[test]
    fn test_semantic_store_remove_deletes_entry() {
        let store = SemanticStore::new();
        store.store("k", "v", vec![]).unwrap();
        assert_eq!(store.len().unwrap(), 1);
        let removed = store.remove("k").unwrap();
        assert!(removed);
        assert_eq!(store.len().unwrap(), 0);
    }

    #[test]
    fn test_semantic_store_clear_empties_store() {
        let store = SemanticStore::new();
        store.store("a", "1", vec!["t".into()]).unwrap();
        store.store("b", "2", vec!["t".into()]).unwrap();
        store.clear().unwrap();
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn test_semantic_store_update_changes_value() {
        let store = SemanticStore::new();
        store.store("k", "old", vec![]).unwrap();
        let updated = store.update("k", "new").unwrap();
        assert!(updated);
        let (val, _) = store.retrieve_by_key("k").unwrap().unwrap();
        assert_eq!(val, "new");
    }

    #[test]
    fn test_semantic_store_retrieve_by_key() {
        let store = SemanticStore::new();
        store.store("key", "value", vec!["tag1".into()]).unwrap();
        let result = store.retrieve_by_key("key").unwrap().unwrap();
        assert_eq!(result.0, "value");
        assert_eq!(result.1, vec!["tag1".to_string()]);
        assert!(store.retrieve_by_key("missing").unwrap().is_none());
    }

    #[test]
    fn test_semantic_store_list_tags() {
        let store = SemanticStore::new();
        store.store("a", "1", vec!["rust".into(), "sys".into()]).unwrap();
        store.store("b", "2", vec!["rust".into(), "ml".into()]).unwrap();
        let tags = store.list_tags().unwrap();
        assert_eq!(tags, vec!["ml", "rust", "sys"]); // sorted
    }

    #[test]
    fn test_semantic_store_count_by_tag() {
        let store = SemanticStore::new();
        store.store("a", "1", vec!["rust".into(), "sys".into()]).unwrap();
        store.store("b", "2", vec!["rust".into(), "ml".into()]).unwrap();
        store.store("c", "3", vec!["ml".into()]).unwrap();
        assert_eq!(store.count_by_tag("rust").unwrap(), 2);
        assert_eq!(store.count_by_tag("ml").unwrap(), 2);
        assert_eq!(store.count_by_tag("sys").unwrap(), 1);
        assert_eq!(store.count_by_tag("absent").unwrap(), 0);
    }

    #[test]
    fn test_working_memory_get_many_returns_present_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "alpha".to_string()).unwrap();
        wm.set("b", "beta".to_string()).unwrap();
        // get_many returns a Vec<Option<String>> in the same order as the input keys
        let results = wm.get_many(&["a", "b", "c"]).unwrap();
        assert_eq!(results[0].as_deref(), Some("alpha"));
        assert_eq!(results[1].as_deref(), Some("beta"));
        assert_eq!(results[2], None);
    }

    #[test]
    fn test_working_memory_update_if_exists_changes_value() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("x", "old".to_string()).unwrap();
        assert!(wm.update_if_exists("x", "new".to_string()).unwrap());
        assert_eq!(wm.get("x").unwrap().as_deref(), Some("new"));
    }

    #[test]
    fn test_working_memory_update_if_exists_returns_false_for_missing() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(!wm.update_if_exists("nope", "val".to_string()).unwrap());
    }

    #[test]
    fn test_episodic_total_recall_count_sums_all_accesses() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent1");
        store.add_episode(agent.clone(), "first", 0.8).unwrap();
        store.add_episode(agent.clone(), "second", 0.5).unwrap();
        // recall() increments recall_count on each returned item
        store.recall(&agent, 10).unwrap();
        store.recall(&agent, 10).unwrap();
        let total = store.total_recall_count(&agent).unwrap();
        // 2 items × 2 recalls = 4
        assert_eq!(total, 4);
    }

    #[test]
    fn test_episodic_total_recall_count_zero_before_recall() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("fresh");
        store.add_episode(agent.clone(), "ep", 0.5).unwrap();
        assert_eq!(store.total_recall_count(&agent).unwrap(), 0);
    }

    #[test]
    fn test_episodic_recall_all_returns_all_items() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-all");
        store.add_episode(agent.clone(), "one", 0.9).unwrap();
        store.add_episode(agent.clone(), "two", 0.5).unwrap();
        store.add_episode(agent.clone(), "three", 0.1).unwrap();
        let all = store.recall_all(&agent).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_episodic_recall_all_empty_agent_returns_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("nobody");
        let all = store.recall_all(&agent).unwrap();
        assert!(all.is_empty());
    }

    #[test]
    fn test_episodic_clear_all_removes_all_agents() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("a1");
        let a2 = AgentId::new("a2");
        store.add_episode(a1.clone(), "ep", 0.5).unwrap();
        store.add_episode(a2.clone(), "ep", 0.5).unwrap();
        store.clear_all().unwrap();
        assert_eq!(store.len().unwrap(), 0);
        assert!(store.list_agents().unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_drain_returns_all_and_empties() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("x", "1".to_string()).unwrap();
        wm.set("y", "2".to_string()).unwrap();
        let drained = wm.drain().unwrap();
        assert_eq!(drained.len(), 2);
        assert!(wm.is_empty().unwrap());
    }

    #[test]
    fn test_working_memory_drain_preserves_insertion_order() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1".to_string()).unwrap();
        wm.set("b", "2".to_string()).unwrap();
        wm.set("c", "3".to_string()).unwrap();
        let drained = wm.drain().unwrap();
        let keys: Vec<&str> = drained.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_semantic_store_tag_count() {
        let store = SemanticStore::new();
        store.store("a", "1", vec!["rust".into(), "sys".into()]).unwrap();
        store.store("b", "2", vec!["rust".into(), "ml".into()]).unwrap();
        assert_eq!(store.tag_count().unwrap(), 3); // rust, sys, ml
    }

    #[test]
    fn test_semantic_store_tag_count_empty() {
        let store = SemanticStore::new();
        assert_eq!(store.tag_count().unwrap(), 0);
    }

    #[test]
    fn test_episodic_top_n_returns_highest_importance() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ag");
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "med", 0.5).unwrap();
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        let top = store.top_n(&agent, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].content, "high");
        assert_eq!(top[1].content, "med");
    }

    #[test]
    fn test_episodic_top_n_zero_returns_all() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ag");
        store.add_episode(agent.clone(), "a", 0.9).unwrap();
        store.add_episode(agent.clone(), "b", 0.5).unwrap();
        assert_eq!(store.top_n(&agent, 0).unwrap().len(), 2);
    }

    #[test]
    fn test_episodic_search_by_importance_range() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ag");
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        let results = store.search_by_importance_range(&agent, 0.4, 1.0, 0).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "high");
    }

    #[test]
    fn test_semantic_store_contains_true_for_stored_key() {
        let store = SemanticStore::new();
        store.store("k", "v", vec![]).unwrap();
        assert!(store.contains("k").unwrap());
    }

    #[test]
    fn test_semantic_store_contains_false_for_missing_key() {
        let store = SemanticStore::new();
        assert!(!store.contains("missing").unwrap());
    }

    #[test]
    fn test_working_memory_rename_changes_key() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("old", "value".to_string()).unwrap();
        assert!(wm.rename("old", "new").unwrap());
        assert_eq!(wm.get("new").unwrap().as_deref(), Some("value"));
        assert!(wm.get("old").unwrap().is_none());
    }

    #[test]
    fn test_working_memory_rename_preserves_order() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1".to_string()).unwrap();
        wm.set("b", "2".to_string()).unwrap();
        wm.rename("a", "x").unwrap();
        assert_eq!(wm.keys().unwrap(), vec!["x", "b"]);
    }

    #[test]
    fn test_working_memory_rename_missing_returns_false() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(!wm.rename("nope", "other").unwrap());
    }

    #[test]
    fn test_episodic_importance_stats_basic() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ag");
        store.add_episode(agent.clone(), "a", 0.2).unwrap();
        store.add_episode(agent.clone(), "b", 0.6).unwrap();
        store.add_episode(agent.clone(), "c", 1.0).unwrap();
        let (count, min, max, mean) = store.importance_stats(&agent).unwrap();
        assert_eq!(count, 3);
        assert!((min - 0.2).abs() < 1e-5);
        assert!((max - 1.0).abs() < 1e-5);
        assert!((mean - 0.6).abs() < 1e-4);
    }

    #[test]
    fn test_episodic_importance_stats_empty_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("nobody");
        let (count, min, max, mean) = store.importance_stats(&agent).unwrap();
        assert_eq!(count, 0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_semantic_store_keys_returns_stored_keys() {
        let store = SemanticStore::new();
        store.store("alpha", "v1", vec![]).unwrap();
        store.store("beta", "v2", vec![]).unwrap();
        let keys = store.keys().unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"alpha".to_string()));
        assert!(keys.contains(&"beta".to_string()));
    }

    #[test]
    fn test_working_memory_snapshot_clones_contents() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("x", "1".to_string()).unwrap();
        wm.set("y", "2".to_string()).unwrap();
        let snap = wm.snapshot().unwrap();
        assert_eq!(snap.get("x").map(|s| s.as_str()), Some("1"));
        assert_eq!(snap.get("y").map(|s| s.as_str()), Some("2"));
        // Memory still has items
        assert_eq!(wm.len().unwrap(), 2);
    }

    // ── Round 3: new methods ──────────────────────────────────────────────────

    #[test]
    fn test_memory_item_display_includes_content() {
        let agent = AgentId::new("a");
        let item = MemoryItem::new(agent, "hello world", 0.75, vec![]);
        let s = format!("{item}");
        assert!(s.contains("hello world"), "Display should include content");
        assert!(s.contains("0.75"), "Display should include importance");
    }

    #[test]
    fn test_decay_policy_half_life_hours_accessor() {
        let policy = DecayPolicy::exponential(12.5).unwrap();
        assert!((policy.half_life_hours() - 12.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_semantic_store_list_keys_returns_all_keys() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec![]).unwrap();
        store.store("k2", "v2", vec![]).unwrap();
        let keys = store.list_keys().unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"k1".to_string()));
        assert!(keys.contains(&"k2".to_string()));
    }

    #[test]
    fn test_semantic_store_update_tags_returns_true_on_found() {
        let store = SemanticStore::new();
        store.store("k", "v", vec!["old".into()]).unwrap();
        let updated = store.update_tags("k", vec!["new".into()]).unwrap();
        assert!(updated);
        let (_, tags) = store.retrieve_by_key("k").unwrap().unwrap();
        assert_eq!(tags, vec!["new".to_string()]);
    }

    #[test]
    fn test_semantic_store_update_tags_returns_false_on_missing() {
        let store = SemanticStore::new();
        assert!(!store.update_tags("missing", vec![]).unwrap());
    }

    #[test]
    fn test_episodic_store_recall_since_filters_old() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let old_ts = Utc::now() - chrono::Duration::hours(2);
        store.add_episode_at(agent.clone(), "old", 0.9, old_ts).unwrap();
        store.add_episode(agent.clone(), "new", 0.5).unwrap();

        let cutoff = Utc::now() - chrono::Duration::minutes(30);
        let items = store.recall_since(&agent, cutoff, 0).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "new");
    }

    #[test]
    fn test_episodic_store_recall_since_limit_respected() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        for i in 0..5 {
            store.add_episode(agent.clone(), format!("item {i}"), 0.5).unwrap();
        }
        let cutoff = Utc::now() - chrono::Duration::hours(1);
        let items = store.recall_since(&agent, cutoff, 2).unwrap();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_episodic_store_update_content_returns_true_on_found() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store.add_episode(agent.clone(), "original", 0.5).unwrap();
        let updated = store.update_content(&agent, &id, "updated").unwrap();
        assert!(updated);
        let item = store.recall_by_id(&agent, &id).unwrap().unwrap();
        assert_eq!(item.content, "updated");
    }

    #[test]
    fn test_episodic_store_update_content_returns_false_on_missing() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let fake_id = MemoryId::new("does-not-exist");
        assert!(!store.update_content(&agent, &fake_id, "x").unwrap());
    }

    #[test]
    fn test_working_memory_capacity_matches_constructor() {
        let wm = WorkingMemory::new(7).unwrap();
        assert_eq!(wm.capacity(), 7);
    }

    // ── Round 4: EpisodicStore new helpers ───────────────────────────────────

    #[test]
    fn test_add_episode_with_tags_stores_and_recalls() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store
            .add_episode_with_tags(agent.clone(), "tagged", 0.8, vec!["t1".to_string(), "t2".to_string()])
            .unwrap();
        let item = store.recall_by_id(&agent, &id).unwrap().unwrap();
        assert_eq!(item.content, "tagged");
        assert_eq!(item.tags, vec!["t1", "t2"]);
    }

    #[test]
    fn test_remove_by_id_deletes_episode() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store.add_episode(agent.clone(), "to-delete", 0.5).unwrap();
        assert!(store.remove_by_id(&agent, &id).unwrap());
        assert!(store.recall_by_id(&agent, &id).unwrap().is_none());
    }

    #[test]
    fn test_remove_by_id_returns_false_for_missing() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let fake = MemoryId::new("does-not-exist");
        assert!(!store.remove_by_id(&agent, &fake).unwrap());
    }

    #[test]
    fn test_update_tags_by_id_replaces_tags() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store
            .add_episode_with_tags(agent.clone(), "x", 0.5, vec!["old".to_string()])
            .unwrap();
        let updated = store
            .update_tags_by_id(&agent, &id, vec!["new1".to_string(), "new2".to_string()])
            .unwrap();
        assert!(updated);
        let item = store.recall_by_id(&agent, &id).unwrap().unwrap();
        assert_eq!(item.tags, vec!["new1", "new2"]);
    }

    #[test]
    fn test_update_tags_by_id_returns_false_for_missing() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let fake = MemoryId::new("nope");
        assert!(!store.update_tags_by_id(&agent, &fake, vec![]).unwrap());
    }

    #[test]
    fn test_max_importance_for_returns_highest() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        let max = store.max_importance_for(&agent).unwrap().unwrap();
        assert!((max - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_max_importance_for_returns_none_when_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("empty");
        assert!(store.max_importance_for(&agent).unwrap().is_none());
    }

    // ── Round 4: WorkingMemory::snapshot / pop_oldest ────────────────────────

    #[test]
    fn test_working_memory_snapshot_returns_all_entries() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "v1").unwrap();
        wm.set("k2", "v2").unwrap();
        let snap = wm.snapshot().unwrap();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap.get("k1").unwrap(), "v1");
        assert_eq!(snap.get("k2").unwrap(), "v2");
    }

    #[test]
    fn test_working_memory_pop_oldest_removes_first_inserted() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("first", "1").unwrap();
        wm.set("second", "2").unwrap();
        let popped = wm.pop_oldest().unwrap().unwrap();
        assert_eq!(popped.0, "first");
        assert_eq!(popped.1, "1");
        assert_eq!(wm.len().unwrap(), 1);
    }

    #[test]
    fn test_working_memory_pop_oldest_returns_none_when_empty() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(wm.pop_oldest().unwrap().is_none());
    }

    // ── Round 4: SemanticStore::keys_with_tag ────────────────────────────────

    #[test]
    fn test_semantic_store_keys_with_tag_returns_matching() {
        let store = SemanticStore::new();
        store
            .store("doc1", "content one", vec!["alpha".to_string(), "beta".to_string()])
            .unwrap();
        store
            .store("doc2", "content two", vec!["alpha".to_string()])
            .unwrap();
        store
            .store("doc3", "content three", vec!["gamma".to_string()])
            .unwrap();
        let mut keys = store.keys_with_tag("alpha").unwrap();
        keys.sort();
        assert_eq!(keys, vec!["doc1", "doc2"]);
    }

    #[test]
    fn test_semantic_store_keys_with_tag_empty_when_no_match() {
        let store = SemanticStore::new();
        store
            .store("doc1", "content", vec!["x".to_string()])
            .unwrap();
        let keys = store.keys_with_tag("z").unwrap();
        assert!(keys.is_empty());
    }

    // ── Round 16: oldest, get_or_default ─────────────────────────────────────

    #[test]
    fn test_episodic_oldest_returns_first_inserted() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-oldest");
        store.add_episode(agent.clone(), "first episode", 0.5).unwrap();
        store.add_episode(agent.clone(), "second episode", 0.9).unwrap();
        let oldest = store.oldest(&agent).unwrap().unwrap();
        assert_eq!(oldest.content, "first episode");
    }

    #[test]
    fn test_episodic_oldest_none_when_no_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("no-episodes");
        assert!(store.oldest(&agent).unwrap().is_none());
    }

    #[test]
    fn test_working_memory_get_or_default_returns_value_when_present() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("key", "value").unwrap();
        assert_eq!(wm.get_or_default("key", "fallback").unwrap(), "value");
    }

    #[test]
    fn test_working_memory_get_or_default_returns_default_when_absent() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.get_or_default("missing", "fallback").unwrap(), "fallback");
    }

    // ── Round 5: EpisodicStore new helpers ───────────────────────────────────

    #[test]
    fn test_episodic_count_for_returns_correct_count() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.5).unwrap();
        store.add_episode(agent.clone(), "e2", 0.5).unwrap();
        assert_eq!(store.count_for(&agent).unwrap(), 2);
    }

    #[test]
    fn test_episodic_count_for_returns_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("unknown");
        assert_eq!(store.count_for(&agent).unwrap(), 0);
    }

    #[test]
    fn test_recall_by_tag_returns_matching_sorted_by_importance() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store
            .add_episode_with_tags(agent.clone(), "low", 0.1, vec!["news".to_string()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "high", 0.9, vec!["news".to_string()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "other", 0.8, vec!["other".to_string()])
            .unwrap();
        let items = store.recall_by_tag(&agent, "news", 0).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].content, "high");
    }

    #[test]
    fn test_recall_by_tag_respects_limit() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        for i in 0..5 {
            store
                .add_episode_with_tags(agent.clone(), format!("ep{i}"), 0.5, vec!["t".to_string()])
                .unwrap();
        }
        let items = store.recall_by_tag(&agent, "t", 2).unwrap();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_merge_from_copies_episodes() {
        let src = EpisodicStore::new();
        let dst = EpisodicStore::new();
        let agent = AgentId::new("a");
        src.add_episode(agent.clone(), "ep1", 0.5).unwrap();
        src.add_episode(agent.clone(), "ep2", 0.9).unwrap();
        let count = dst.merge_from(&src, &agent).unwrap();
        assert_eq!(count, 2);
        assert_eq!(dst.count_for(&agent).unwrap(), 2);
    }

    // ── Round 5: WorkingMemory get_all_keys / replace_all ────────────────────

    #[test]
    fn test_working_memory_get_all_keys_preserves_insertion_order() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("first", "1").unwrap();
        wm.set("second", "2").unwrap();
        wm.set("third", "3").unwrap();
        assert_eq!(wm.get_all_keys().unwrap(), vec!["first", "second", "third"]);
    }

    #[test]
    fn test_working_memory_replace_all_replaces_contents() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("old", "x").unwrap();
        let mut new_map = std::collections::HashMap::new();
        new_map.insert("a".to_string(), "1".to_string());
        new_map.insert("b".to_string(), "2".to_string());
        wm.replace_all(new_map).unwrap();
        assert_eq!(wm.len().unwrap(), 2);
        assert!(wm.get("old").unwrap().is_none());
        assert_eq!(wm.get("a").unwrap().unwrap(), "1");
    }

    // ── Round 5: SemanticStore::remove / count ───────────────────────────────

    #[test]
    fn test_semantic_store_remove_returns_true_and_removes() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec![]).unwrap();
        store.store("k2", "v2", vec![]).unwrap();
        assert!(store.remove("k1").unwrap());
        assert_eq!(store.count().unwrap(), 1);
        let keys = store.list_keys().unwrap();
        assert_eq!(keys, vec!["k2"]);
    }

    #[test]
    fn test_semantic_store_remove_returns_false_when_missing() {
        let store = SemanticStore::new();
        assert!(!store.remove("ghost").unwrap());
    }

    #[test]
    fn test_semantic_store_count_matches_len() {
        let store = SemanticStore::new();
        store.store("a", "1", vec![]).unwrap();
        store.store("b", "2", vec![]).unwrap();
        assert_eq!(store.count().unwrap(), store.len().unwrap());
    }

    // ── Round 17: newest, values ─────────────────────────────────────────────

    #[test]
    fn test_episodic_newest_returns_last_inserted() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-newest");
        store.add_episode(agent.clone(), "first", 0.3).unwrap();
        store.add_episode(agent.clone(), "last", 0.8).unwrap();
        let newest = store.newest(&agent).unwrap().unwrap();
        assert_eq!(newest.content, "last");
    }

    #[test]
    fn test_episodic_newest_none_when_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("no-ep");
        assert!(store.newest(&agent).unwrap().is_none());
    }

    #[test]
    fn test_working_memory_values_returns_values_in_insertion_order() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "apple").unwrap();
        wm.set("b", "banana").unwrap();
        let vals = wm.values().unwrap();
        assert_eq!(vals, vec!["apple", "banana"]);
    }

    #[test]
    fn test_working_memory_values_empty_when_no_entries() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.values().unwrap().is_empty());
    }

    // ── Round 18: clear_agent ────────────────────────────────────────────────

    #[test]
    fn test_episodic_clear_agent_removes_all_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-clear");
        store.add_episode(agent.clone(), "ep1", 0.5).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.7).unwrap();
        let removed = store.clear_agent(&agent).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.agent_memory_count(&agent).unwrap(), 0);
    }

    #[test]
    fn test_episodic_clear_agent_returns_zero_when_none() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("no-agent");
        assert_eq!(store.clear_agent(&agent).unwrap(), 0);
    }
}
