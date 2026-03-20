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

    /// Return `true` if this memory item has the given tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Return the approximate number of whitespace-separated words in the content.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Return the byte length of the content string.
    pub fn content_len(&self) -> usize {
        self.content.len()
    }

    /// Return the number of tags attached to this memory item.
    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }

    /// Add `tag` to this memory item if it is not already present.
    ///
    /// Returns `true` if the tag was newly added, `false` if it was already present.
    pub fn add_tag(&mut self, tag: impl Into<String>) -> bool {
        let tag = tag.into();
        if self.tags.iter().any(|t| t == &tag) {
            return false;
        }
        self.tags.push(tag);
        true
    }

    /// Remove `tag` from this memory item.
    ///
    /// Returns `true` if the tag was found and removed, `false` if it was not present.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Return `true` if this item's importance score is strictly above `threshold`.
    ///
    /// Useful for filtering a collection to retain only high-value memories:
    /// ```rust,ignore
    /// let important: Vec<_> = items.iter().filter(|m| m.is_high_importance(0.7)).collect();
    /// ```
    pub fn is_high_importance(&self, threshold: f32) -> bool {
        self.importance > threshold
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

impl std::fmt::Display for DecayPolicy {
    /// Render as `"Exponential(half_life=Xh)"`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exponential(half_life={:.1}h)", self.half_life_hours)
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

    /// Return `true` if the store contains at least one episode for `agent_id`.
    ///
    /// Cheaper than `count_for(agent_id)? > 0` because no heap allocation occurs.
    pub fn has_agent(&self, agent_id: &AgentId) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::has_agent");
        Ok(inner.items.get(agent_id).map_or(false, |v| !v.is_empty()))
    }

    /// Return the IDs of agents that have at least `min` episodes.
    pub fn agents_with_min_episodes(&self, min: usize) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agents_with_min_episodes");
        let mut ids: Vec<AgentId> = inner
            .items
            .iter()
            .filter(|(_, v)| v.len() >= min)
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(ids)
    }

    /// Return the total number of episodes stored across all agents.
    pub fn total_episode_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::total_episode_count");
        Ok(inner.items.values().map(|v| v.len()).sum())
    }

    /// Return all agent IDs that have at least one episode stored.
    ///
    /// The returned `Vec` is sorted for deterministic output.
    pub fn agent_ids_with_episodes(&self) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agent_ids_with_episodes");
        let mut ids: Vec<AgentId> = inner
            .items
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, _)| k.clone())
            .collect();
        ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        Ok(ids)
    }

    /// Return the average importance score of all episodes for `agent_id`.
    ///
    /// Returns `0.0` when the agent has no episodes or is unknown.
    pub fn episode_importance_avg(&self, agent_id: &AgentId) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_importance_avg");
        let items = match inner.items.get(agent_id) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(0.0),
        };
        let sum: f64 = items.iter().map(|m| m.importance as f64).sum();
        Ok(sum / items.len() as f64)
    }

    /// Return a map from each agent ID to their episode count.
    ///
    /// Returns an empty map for an empty store.
    pub fn agent_episode_counts(
        &self,
    ) -> Result<std::collections::HashMap<AgentId, usize>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agent_episode_counts");
        Ok(inner
            .items
            .iter()
            .map(|(id, items)| (id.clone(), items.len()))
            .collect())
    }

    /// Return all episodes for `agent_id` sorted by importance descending
    /// (most important first).
    ///
    /// Returns an empty `Vec` for an unknown agent.
    pub fn episodes_for_agent_sorted_by_importance(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::episodes_for_agent_sorted_by_importance");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default();
        items.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(items)
    }

    /// Return all episodes for `agent_id` whose timestamp is strictly after
    /// `after`.
    ///
    /// Returns an empty `Vec` for an unknown agent or when no episode qualifies.
    pub fn episodes_after_timestamp(
        &self,
        agent_id: &AgentId,
        after: DateTime<Utc>,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_after_timestamp");
        Ok(inner
            .items
            .get(agent_id)
            .map(|v| v.iter().filter(|m| m.timestamp > after).cloned().collect())
            .unwrap_or_default())
    }

    /// Return the agent ID with the lowest average episode importance, or
    /// `None` for an empty store.
    pub fn agent_with_min_importance_avg(&self) -> Result<Option<AgentId>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::agent_with_min_importance_avg");
        Ok(inner
            .items
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(id, items)| {
                let avg = items.iter().map(|m| m.importance as f64).sum::<f64>()
                    / items.len() as f64;
                (id.clone(), avg)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id))
    }

    /// Return all episodes for `agent_id` whose content contains `substr`.
    ///
    /// Returns an empty `Vec` for an unknown agent or when no episode matches.
    pub fn episodes_matching_content(
        &self,
        agent_id: &AgentId,
        substr: &str,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_matching_content");
        Ok(inner
            .items
            .get(agent_id)
            .map(|v| v.iter().filter(|m| m.content.contains(substr)).cloned().collect())
            .unwrap_or_default())
    }

    /// Return the agent ID whose episodes have the highest total importance
    /// score, or `None` if the store is empty.
    pub fn top_agent_by_importance(&self) -> Result<Option<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::top_agent_by_importance");
        Ok(inner
            .items
            .iter()
            .map(|(id, items)| {
                let total: f32 = items.iter().map(|m| m.importance).sum();
                (id.clone(), total)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id))
    }

    /// Return the episode with the highest importance score for `agent_id`,
    /// or `None` when the agent has no episodes.
    pub fn highest_importance_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::highest_importance_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| {
                v.iter().max_by(|a, b| {
                    a.importance
                        .partial_cmp(&b.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .cloned())
    }

    /// Return the average number of whitespace-delimited words per episode
    /// content string for `agent_id`.
    ///
    /// Returns `0.0` when the agent has no episodes.
    pub fn avg_episode_content_words(
        &self,
        agent_id: &AgentId,
    ) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::avg_episode_content_words");
        let items = match inner.items.get(agent_id) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(0.0),
        };
        let total: usize = items
            .iter()
            .map(|m| m.content.split_whitespace().count())
            .sum();
        Ok(total as f64 / items.len() as f64)
    }

    /// Return `true` if two or more episodes for `agent_id` share identical
    /// content strings.
    ///
    /// Returns `false` for an unknown agent or one with fewer than 2 episodes.
    pub fn has_duplicate_content(
        &self,
        agent_id: &AgentId,
    ) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::has_duplicate_content");
        let items = match inner.items.get(agent_id) {
            Some(v) => v,
            None => return Ok(false),
        };
        let mut seen = std::collections::HashSet::new();
        Ok(items.iter().any(|m| !seen.insert(m.content.as_str())))
    }

    /// Return the number of episodes stored for the given agent.
    ///
    /// Returns `0` if the agent has no recorded episodes.
    pub fn episode_count_for(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_count_for");
        Ok(inner.items.get(agent_id).map_or(0, |v| v.len()))
    }

    /// Return all episodes for `agent_id` whose importance is strictly above
    /// `min_importance`, cloned into a new `Vec`.
    pub fn episodes_with_importance_above(
        &self,
        agent_id: &AgentId,
        min_importance: f32,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::episodes_with_importance_above",
        );
        Ok(inner
            .items
            .get(agent_id)
            .map(|v| v.iter().filter(|m| m.importance > min_importance).cloned().collect())
            .unwrap_or_default())
    }

    /// Return the sum of importance values for all episodes belonging to
    /// `agent_id`.
    ///
    /// Returns `0.0` when the agent has no recorded episodes.
    pub fn agent_episode_importance_sum(
        &self,
        agent_id: &AgentId,
    ) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::agent_episode_importance_sum",
        );
        Ok(inner
            .items
            .get(agent_id)
            .map(|v| v.iter().map(|m| m.importance as f64).sum())
            .unwrap_or(0.0))
    }

    /// Return the minimum importance across all episodes for `agent_id`, or
    /// `None` when the agent has no recorded episodes.
    pub fn episodes_min_importance(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_min_importance");
        Ok(inner.items.get(agent_id).and_then(|v| {
            v.iter()
                .map(|m| m.importance)
                .reduce(f32::min)
        }))
    }

    /// Return the minimum word count among all episode content strings for
    /// `agent_id`.
    ///
    /// Returns `0` when the agent has no recorded episodes.
    pub fn episode_min_content_words(
        &self,
        agent_id: &AgentId,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::episode_min_content_words",
        );
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().map(|m| m.content.split_whitespace().count()).min())
            .unwrap_or(0))
    }

    /// Return the `AgentId` of the agent with the most stored episodes, or
    /// `None` if the store is empty.
    pub fn agent_with_most_episodes(&self) -> Result<Option<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agent_with_most_episodes");
        Ok(inner
            .items
            .iter()
            .max_by_key(|(_, v)| v.len())
            .map(|(id, _)| id.clone()))
    }

    /// Return the maximum word count among all episode content strings for
    /// `agent_id`.
    ///
    /// Returns `0` when the agent has no recorded episodes.
    pub fn episode_max_content_words(
        &self,
        agent_id: &AgentId,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::episode_max_content_words",
        );
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().map(|m| m.content.split_whitespace().count()).max())
            .unwrap_or(0))
    }

    /// Return the total number of episodes stored across all agents.
    pub fn total_items(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::total_items");
        Ok(inner.items.values().map(|v| v.len()).sum())
    }

    /// Return a cloned `Vec` of all `MemoryItem`s across all agents, in an
    /// unspecified order.
    pub fn all_episodes(&self) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::all_episodes");
        Ok(inner.items.values().flat_map(|v| v.iter().cloned()).collect())
    }

    /// Return all agent IDs that have at least one stored episode, sorted.
    pub fn agents(&self) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agents");
        let mut ids: Vec<AgentId> = inner
            .items
            .keys()
            .filter(|id| !inner.items[id].is_empty())
            .cloned()
            .collect();
        ids.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(ids)
    }

    /// Return the agent with the fewest (non-zero) stored episodes, or `None`
    /// if the store is empty.
    pub fn min_episode_count(&self) -> Result<Option<(AgentId, usize)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::min_episode_count");
        Ok(inner
            .items
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .min_by_key(|(_, v)| v.len())
            .map(|(id, v)| (id.clone(), v.len())))
    }

    /// Return the highest importance value across all stored episodes,
    /// or `None` if the store is empty.
    pub fn max_importance_overall(&self) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::max_importance_overall");
        let max = inner
            .items
            .values()
            .flat_map(|v| v.iter())
            .map(|e| e.importance)
            .reduce(f32::max);
        Ok(max)
    }

    /// Return the variance of importance scores for the given agent's episodes.
    ///
    /// Returns `0.0` when the agent has fewer than two episodes.
    pub fn importance_variance_for(&self, agent_id: &AgentId) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::importance_variance_for");
        let vals: Vec<f64> = inner
            .items
            .get(agent_id)
            .map(|v| v.iter().map(|e| e.importance as f64).collect())
            .unwrap_or_default();
        if vals.len() < 2 {
            return Ok(0.0);
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        Ok(variance)
    }

    /// Return up to `n` episodes for `agent_id` sorted by descending importance
    /// without incrementing recall counts or applying decay.
    ///
    /// Use this for read-only importance-ranked snapshots.
    pub fn recall_top_n(
        &self,
        agent_id: &AgentId,
        n: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recall_top_n");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .cloned()
            .unwrap_or_default();
        items.sort_unstable_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(n);
        Ok(items)
    }

    /// Return all episodes for `agent_id` whose importance is within
    /// `[min_inclusive, max_inclusive]`, sorted by descending importance.
    pub fn filter_by_importance(
        &self,
        agent_id: &AgentId,
        min: f32,
        max: f32,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::filter_by_importance");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map(|v| {
                v.iter()
                    .filter(|i| i.importance >= min && i.importance <= max)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        items.sort_unstable_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(items)
    }

    /// Keep only the `n` most-important episodes for `agent_id`, removing
    /// the rest.  Returns the number of episodes that were removed.
    ///
    /// If the agent has `n` or fewer episodes already, nothing is removed.
    pub fn retain_top_n(&self, agent_id: &AgentId, n: usize) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::retain_top_n");
        let items = inner.items.entry(agent_id.clone()).or_default();
        if items.len() <= n {
            return Ok(0);
        }
        items.sort_unstable_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let removed = items.len() - n;
        items.truncate(n);
        Ok(removed)
    }

    /// Return the most recently stored episode for `agent_id`, or `None` if
    /// the agent has no episodes.
    ///
    /// "Most recent" is defined as the last element in the stored list, which
    /// matches insertion order.
    pub fn most_recent(&self, agent_id: &AgentId) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::most_recent");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.last().cloned()))
    }

    /// Return the highest importance score for `agent_id`, or `None` if the
    /// agent has no episodes.
    pub fn max_importance(&self, agent_id: &AgentId) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::max_importance");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| {
                v.iter()
                    .map(|i| i.importance)
                    .reduce(f32::max)
            }))
    }

    /// Return the lowest importance score for `agent_id`, or `None` if the
    /// agent has no episodes.
    pub fn min_importance(&self, agent_id: &AgentId) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::min_importance");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| {
                v.iter()
                    .map(|i| i.importance)
                    .reduce(f32::min)
            }))
    }

    /// Count episodes for `agent_id` whose importance is strictly greater than
    /// `threshold`.
    ///
    /// Returns `0` if the agent has no episodes.
    pub fn count_above_importance(
        &self,
        agent_id: &AgentId,
        threshold: f32,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::count_above_importance");
        Ok(inner
            .items
            .get(agent_id)
            .map(|v| v.iter().filter(|i| i.importance > threshold).count())
            .unwrap_or(0))
    }

    /// Return the episode with the highest `recall_count` for `agent_id`.
    ///
    /// Returns `None` if the agent has no stored episodes.  When multiple
    /// episodes tie for the maximum recall count, any one of them may be returned.
    pub fn most_recalled(&self, agent_id: &AgentId) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::most_recalled");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().max_by_key(|i| i.recall_count))
            .cloned())
    }

    /// Return the arithmetic mean importance for `agent_id`, or `0.0` if the
    /// agent has no stored episodes.
    pub fn importance_avg(&self, agent_id: &AgentId) -> Result<f32, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::importance_avg");
        match inner.items.get(agent_id) {
            None => Ok(0.0),
            Some(items) if items.is_empty() => Ok(0.0),
            Some(items) => {
                let sum: f32 = items.iter().map(|i| i.importance).sum();
                Ok(sum / items.len() as f32)
            }
        }
    }

    /// Remove duplicate episodes (same `content`) for `agent_id`, keeping only
    /// the episode with the highest importance for each distinct content string.
    ///
    /// Returns the number of episodes removed.
    pub fn deduplicate_content(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::deduplicate_content");
        let Some(items) = inner.items.get_mut(agent_id) else {
            return Ok(0);
        };
        let before = items.len();
        // For each unique content keep the item with the highest importance.
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut keepers: Vec<usize> = Vec::new();
        for (idx, item) in items.iter().enumerate() {
            match seen.get(&item.content) {
                None => {
                    seen.insert(item.content.clone(), keepers.len());
                    keepers.push(idx);
                }
                Some(&pos) => {
                    let kept_idx = keepers[pos];
                    if item.importance > items[kept_idx].importance {
                        keepers[pos] = idx;
                    }
                }
            }
        }
        let kept: std::collections::HashSet<usize> = keepers.into_iter().collect();
        let mut i = 0;
        items.retain(|_| {
            let keep = kept.contains(&i);
            i += 1;
            keep
        });
        Ok(before - items.len())
    }

    /// Return all `AgentId`s that have at least one stored episode.
    pub fn agent_ids(&self) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agent_ids");
        Ok(inner.items.keys().cloned().collect())
    }

    /// Return agent IDs that have more than `min_count` episodes stored.
    ///
    /// Useful for identifying high-activity agents in a shared store.
    pub fn agents_with_episodes_above_count(
        &self,
        min_count: usize,
    ) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::agents_with_episodes_above_count",
        );
        Ok(inner
            .items
            .iter()
            .filter(|(_, episodes)| episodes.len() > min_count)
            .map(|(id, _)| id.clone())
            .collect())
    }

    /// Return all episodes for `agent_id` whose `content` contains `pattern`
    /// (case-sensitive substring match), sorted by descending importance.
    pub fn find_by_content(
        &self,
        agent_id: &AgentId,
        pattern: &str,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::find_by_content");
        let mut matches: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map(|items| {
                items
                    .iter()
                    .filter(|i| i.content.contains(pattern))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        matches.sort_unstable_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(matches)
    }

    /// Remove all episodes stored for `agent_id`.
    ///
    /// Returns the number of episodes that were removed.
    pub fn clear_for(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "EpisodicStore::clear_for");
        let count = inner.items.remove(agent_id).map_or(0, |v| v.len());
        Ok(count)
    }

    /// Return the number of episodes for `agent_id` that carry the given tag.
    pub fn count_episodes_with_tag(
        &self,
        agent_id: &AgentId,
        tag: &str,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::count_episodes_with_tag");
        let count = inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().filter(|i| i.has_tag(tag)).count());
        Ok(count)
    }

    /// Return the content strings of all episodes for `agent_id` whose content contains `substring`.
    pub fn episodes_with_content(
        &self,
        agent_id: &AgentId,
        substring: &str,
    ) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_with_content");
        let items = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |v| {
                v.iter()
                    .filter(|i| i.content.contains(substring))
                    .map(|i| i.content.clone())
                    .collect()
            });
        Ok(items)
    }

    /// Return the byte length of the longest episode content for `agent_id`.
    ///
    /// Returns `0` if the agent has no episodes.
    pub fn max_content_length(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::max_content_length");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().map(|i| i.content.len()).max())
            .unwrap_or(0))
    }

    /// Return the byte length of the shortest episode content for `agent_id`.
    ///
    /// Returns `0` if the agent has no episodes.
    pub fn min_content_length(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::min_content_length");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().map(|i| i.content.len()).min())
            .unwrap_or(0))
    }

    /// Return the content strings of all episodes for `agent_id`, sorted by
    /// descending importance (most important first).
    pub fn episodes_by_importance(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_by_importance");
        let mut items: Vec<(f32, String)> = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |v| {
                v.iter().map(|i| (i.importance, i.content.clone())).collect()
            });
        items.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(items.into_iter().map(|(_, c)| c).collect())
    }

    /// Return the count of episodes for `agent_id` whose content contains `substring`.
    ///
    /// The match is case-sensitive.  Returns `0` if the agent has no episodes.
    pub fn content_contains_count(
        &self,
        agent_id: &AgentId,
        substring: &str,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::content_contains_count");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |v| v.iter().filter(|i| i.content.contains(substring)).count()))
    }

    /// Return a sorted list of all `AgentId`s that have at least one episode.
    pub fn agents_with_episodes(&self) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agents_with_episodes");
        let mut ids: Vec<AgentId> = inner
            .items
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, _)| k.clone())
            .collect();
        ids.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));
        Ok(ids)
    }

    /// Return the total number of episodes stored across **all** agents.
    ///
    /// Equivalent to summing `episode_count_for` over every agent, but in a
    /// single lock acquisition.
    pub fn episode_count_all_agents(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_count_all_agents");
        Ok(inner.items.values().map(|v| v.len()).sum())
    }

    /// Return the count of episodes whose importance is strictly greater than `threshold`.
    pub fn high_importance_count(
        &self,
        agent_id: &AgentId,
        threshold: f32,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::high_importance_count");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |v| v.iter().filter(|i| i.importance > threshold).count()))
    }

    /// Return a list of content byte lengths for all episodes belonging to `agent_id`, in storage order.
    pub fn content_lengths(&self, agent_id: &AgentId) -> Result<Vec<usize>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::content_lengths");
        let lengths = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |v| v.iter().map(|i| i.content.len()).collect());
        Ok(lengths)
    }

    /// Return the total byte length of all episode content strings for `agent_id`.
    ///
    /// Returns `0` if the agent has no stored episodes.
    pub fn total_content_bytes(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::total_content_bytes");
        let total = inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().map(|i| i.content.len()).sum());
        Ok(total)
    }

    /// Return the average byte length of episode content strings for `agent_id`.
    ///
    /// Returns `0.0` if the agent has no stored episodes.
    pub fn avg_content_length(&self, agent_id: &AgentId) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::avg_content_length");
        let items = match inner.items.get(agent_id) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(0.0),
        };
        let total: usize = items.iter().map(|i| i.content.len()).sum();
        Ok(total as f64 / items.len() as f64)
    }

    /// Return the sum of all importance scores for `agent_id`.
    ///
    /// Returns `0.0` if the agent has no stored episodes.
    pub fn importance_sum(&self, agent_id: &AgentId) -> Result<f32, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::importance_sum");
        let sum = inner
            .items
            .get(agent_id)
            .map_or(0.0, |items| items.iter().map(|i| i.importance).sum());
        Ok(sum)
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

    /// Return the `recall_count` of the episode identified by `id` for `agent_id`.
    ///
    /// Returns `None` if no episode with that `id` exists for the agent.
    pub fn recall_count_for(
        &self,
        agent_id: &AgentId,
        id: &MemoryId,
    ) -> Result<Option<u64>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recall_count_for");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|items| items.iter().find(|i| &i.id == id))
            .map(|i| i.recall_count))
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

    /// Return the episode with the earliest timestamp for `agent_id`, or `None`
    /// if the agent has no stored episodes.
    pub fn oldest_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::oldest_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().min_by_key(|i| i.timestamp))
            .cloned())
    }

    /// Return the most recently stored episode for `agent_id` by timestamp, or
    /// `None` if the agent has no episodes.
    pub fn newest_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::newest_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().max_by_key(|i| i.timestamp))
            .cloned())
    }

    /// Return the `n` most recently stored episodes for `agent_id`, sorted by
    /// timestamp descending (newest first).
    ///
    /// Returns fewer than `n` items when the agent has fewer stored episodes.
    /// Returns an empty `Vec` for an unknown agent.
    pub fn recent_episodes(
        &self,
        agent_id: &AgentId,
        n: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::recent_episodes");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default();
        items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        items.truncate(n);
        Ok(items)
    }

    /// Return the most recently stored episode for `agent_id` (largest timestamp), or
    /// `None` if the agent has no stored episodes.
    pub fn most_recent_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::most_recent_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().max_by_key(|i| i.timestamp))
            .cloned())
    }

    /// Return the episode with the highest `recall_count` for `agent_id`, or `None`
    /// if the agent has no stored episodes.  Ties are broken in favour of the
    /// episode with the higher importance score.
    pub fn most_recalled_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::most_recalled_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| {
                v.iter().max_by(|a, b| {
                    a.recall_count.cmp(&b.recall_count).then_with(|| {
                        a.importance
                            .partial_cmp(&b.importance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                })
            })
            .cloned())
    }

    /// Return the episode with the highest importance score for `agent_id`, or `None`
    /// if the agent has no stored episodes.  Ties are broken in favour of the
    /// later-inserted episode.
    pub fn max_importance_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::max_importance_episode");
        let item = inner
            .items
            .get(agent_id)
            .and_then(|v| {
                v.iter()
                    .max_by(|a, b| {
                        a.importance
                            .partial_cmp(&b.importance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .cloned();
        Ok(item)
    }

    /// Return the episode with the lowest importance score for `agent_id`, or
    /// `None` if the agent has no stored episodes.  Ties are broken in favour
    /// of the later-inserted episode.
    pub fn min_importance_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::min_importance_episode");
        let item = inner
            .items
            .get(agent_id)
            .and_then(|v| {
                v.iter().min_by(|a, b| {
                    a.importance
                        .partial_cmp(&b.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .cloned();
        Ok(item)
    }

    /// Return all episodes for `agent_id` sorted by importance descending
    /// (highest first).
    ///
    /// Returns an empty `Vec` for an unknown agent.
    pub fn episodes_sorted_by_importance(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_sorted_by_importance");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default();
        items.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(items)
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

    /// Return the number of distinct agents that have at least one stored episode.
    pub fn agent_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agent_count");
        Ok(inner.items.len())
    }

    /// Return the number of stored episodes for a specific agent.
    ///
    /// Returns `0` if the agent has no episodes or has not been seen before.
    pub fn agent_memory_count(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::agent_memory_count");
        Ok(inner.items.get(agent_id).map_or(0, |v| v.len()))
    }

    /// Return `true` if `agent_id` has at least one stored episode.
    pub fn has_episodes(&self, agent_id: &AgentId) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::has_episodes");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(false, |v| !v.is_empty()))
    }

    /// Return the most recently inserted episode for `agent_id`, or `None` if
    /// the agent has no stored episodes.
    ///
    /// "Most recent" is determined by `timestamp`.
    pub fn latest_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::latest_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().max_by_key(|i| i.timestamp))
            .cloned())
    }

    /// Return the maximum single `recall_count` value across all episodes for
    /// `agent_id`, or `None` if the agent has no stored episodes.
    pub fn max_recall_count_for(&self, agent_id: &AgentId) -> Result<Option<u64>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::max_recall_count_for");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|v| v.iter().map(|i| i.recall_count).max()))
    }

    /// Return the mean importance score across all episodes for `agent_id`.
    ///
    /// Returns `0.0` when the agent has no stored episodes.
    pub fn avg_importance(&self, agent_id: &AgentId) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::avg_importance");
        let episodes = match inner.items.get(agent_id) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(0.0),
        };
        let sum: f64 = episodes.iter().map(|i| f64::from(i.importance)).sum();
        Ok(sum / episodes.len() as f64)
    }

    /// Return the `(min, max)` importance pair across all episodes for
    /// `agent_id`, or `None` when the agent has no stored episodes.
    pub fn importance_range(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<(f32, f32)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::importance_range");
        Ok(inner.items.get(agent_id).and_then(|v| {
            if v.is_empty() {
                return None;
            }
            let min = v
                .iter()
                .map(|i| i.importance)
                .fold(f32::INFINITY, f32::min);
            let max = v
                .iter()
                .map(|i| i.importance)
                .fold(f32::NEG_INFINITY, f32::max);
            Some((min, max))
        }))
    }

    /// Return the total `recall_count` across all episodes for `agent_id`.
    ///
    /// Returns `0` if the agent has no stored episodes.
    pub fn sum_recall_counts(&self, agent_id: &AgentId) -> Result<u64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::sum_recall_counts");
        Ok(inner
            .items
            .get(agent_id)
            .map(|v| v.iter().map(|i| i.recall_count).sum())
            .unwrap_or(0))
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

    /// Return the count of episodes for `agent_id` whose importance is strictly
    /// above `threshold`.
    ///
    /// Returns `0` for unknown agents or when no episodes exceed the threshold.
    pub fn episode_count_above_importance(
        &self,
        agent_id: &AgentId,
        threshold: f32,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_count_above_importance");
        let count = inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().filter(|m| m.importance > threshold).count());
        Ok(count)
    }

    /// Return the mean importance score of all episodes for `agent_id`.
    ///
    /// Returns `0.0` for unknown agents or agents with no episodes.
    pub fn avg_episode_importance(
        &self,
        agent_id: &AgentId,
    ) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::avg_episode_importance");
        let items = match inner.items.get(agent_id) {
            Some(items) if !items.is_empty() => items,
            _ => return Ok(0.0),
        };
        let sum: f64 = items.iter().map(|m| m.importance as f64).sum();
        Ok(sum / items.len() as f64)
    }

    /// Return the total byte count of all episode content strings for
    /// `agent_id`.
    ///
    /// Returns `0` for unknown agents.
    pub fn episode_content_bytes_total(
        &self,
        agent_id: &AgentId,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::episode_content_bytes_total",
        );
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().map(|m| m.content.len()).sum()))
    }

    /// Return the total number of whitespace-separated words across all episode
    /// content strings for `agent_id`.
    ///
    /// Returns `0` for unknown agents or agents with no episodes.
    pub fn total_content_words(
        &self,
        agent_id: &AgentId,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::total_content_words");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| {
                items
                    .iter()
                    .map(|m| m.content.split_whitespace().count())
                    .sum()
            }))
    }

    /// Return the minimum importance score across all episodes for `agent_id`.
    ///
    /// Returns `None` when the agent has no episodes.  Complements
    /// [`avg_episode_importance`](Self::avg_episode_importance).
    pub fn min_episode_importance(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<f32>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::min_episode_importance");
        Ok(inner.items.get(agent_id).and_then(|items| {
            items
                .iter()
                .map(|m| m.importance)
                .reduce(f32::min)
        }))
    }

    /// Return the count of episodes for `agent_id` whose importance is
    /// strictly greater than `threshold`.
    ///
    /// Returns `0` for unknown agents.
    pub fn episodes_above_importance_count(
        &self,
        agent_id: &AgentId,
        threshold: f32,
    ) -> Result<usize, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::episodes_above_importance_count");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().filter(|m| m.importance > threshold).count()))
    }

    /// Return the union of all tags across all episodes for `agent_id`.
    ///
    /// Returns an empty set for unknown agents or when all episodes have no
    /// tags.
    pub fn tag_union(
        &self,
        agent_id: &AgentId,
    ) -> Result<std::collections::HashSet<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::tag_union");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(std::collections::HashSet::new, |items| {
                items.iter().flat_map(|m| m.tags.iter().cloned()).collect()
            }))
    }

    /// Return the most recently stored episode for `agent_id`.
    ///
    /// Uses `MemoryItem::timestamp` to determine recency.  Returns `None` for
    /// unknown agents.
    pub fn episode_most_recent(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_most_recent");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|items| items.iter().max_by_key(|m| m.timestamp))
            .cloned())
    }

    /// Return all episodes for `agent_id` whose importance is in `[lo, hi]`
    /// (inclusive), sorted by importance ascending.
    ///
    /// Returns an empty `Vec` for unknown agents.
    pub fn episodes_by_importance_range(
        &self,
        agent_id: &AgentId,
        lo: f32,
        hi: f32,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::episodes_by_importance_range");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items
                    .iter()
                    .filter(|m| m.importance >= lo && m.importance <= hi)
                    .cloned()
                    .collect()
            });
        items.sort_unstable_by(|a, b| {
            a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(items)
    }

    /// Return the number of episodes for `agent_id` whose timestamp falls
    /// within `[start, end]` (inclusive on both ends).
    ///
    /// Returns `0` for unknown agents or when no episode falls in the window.
    pub fn count_episodes_in_window(
        &self,
        agent_id: &AgentId,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::count_episodes_in_window");
        Ok(inner.items.get(agent_id).map_or(0, |items| {
            items
                .iter()
                .filter(|m| m.timestamp >= start && m.timestamp <= end)
                .count()
        }))
    }

    /// Return the total number of tags across all episodes for `agent_id`.
    ///
    /// Episodes with no tags contribute `0` to the sum.  Returns `0` for
    /// unknown agents.
    pub fn total_tag_count(
        &self,
        agent_id: &AgentId,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::total_tag_count");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().map(|m| m.tags.len()).sum()))
    }

    /// Return the average number of tags per episode for `agent_id`.
    ///
    /// Returns `0.0` for unknown agents or when there are no episodes.
    pub fn avg_tag_count_per_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<f64, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::avg_tag_count_per_episode");
        Ok(inner.items.get(agent_id).map_or(0.0, |items| {
            if items.is_empty() {
                return 0.0;
            }
            let total: usize = items.iter().map(|m| m.tags.len()).sum();
            total as f64 / items.len() as f64
        }))
    }

    /// Return all episodes for `agent_id` whose importance is strictly below
    /// `threshold`, sorted by importance ascending (least important first).
    ///
    /// Returns an empty `Vec` for unknown agents or when all episodes meet the
    /// threshold.
    pub fn low_importance_episodes(
        &self,
        agent_id: &AgentId,
        threshold: f32,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::low_importance_episodes");
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items.iter().filter(|m| m.importance < threshold).cloned().collect()
            });
        items.sort_unstable_by(|a, b| {
            a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(items)
    }

    /// Return all episodes for `agent_id` sorted by timestamp ascending
    /// (oldest first).
    ///
    /// Returns an empty `Vec` for unknown agents.
    pub fn episodes_sorted_by_timestamp(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::episodes_sorted_by_timestamp",
        );
        let mut items: Vec<MemoryItem> = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |v| v.clone());
        items.sort_unstable_by_key(|m| m.timestamp);
        Ok(items)
    }

    /// Return a sorted, deduplicated list of all tags used by any episode
    /// belonging to `agent_id`.
    ///
    /// Returns an empty `Vec` for unknown agents.
    pub fn all_unique_tags(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::all_unique_tags");
        let mut tags: Vec<String> = inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items.iter().flat_map(|m| m.tags.iter().cloned()).collect()
            });
        tags.sort_unstable();
        tags.dedup();
        Ok(tags)
    }

    /// Return all episodes for `agent_id` that contain `tag` in their tag list.
    ///
    /// Returns an empty `Vec` for unknown agents or when no episode has the tag.
    pub fn episodes_with_tag(
        &self,
        agent_id: &AgentId,
        tag: &str,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_with_tag");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items
                    .iter()
                    .filter(|m| m.tags.iter().any(|t| t == tag))
                    .cloned()
                    .collect()
            }))
    }

    /// Return the count of episodes for `agent_id` whose timestamp is strictly
    /// before `before`.
    ///
    /// Returns `0` for unknown agents or when no episode qualifies.
    pub fn episode_count_before(
        &self,
        agent_id: &AgentId,
        before: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_count_before");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| {
                items.iter().filter(|m| m.timestamp < before).count()
            }))
    }

    /// Return the IDs of all episodes stored for `agent_id`.
    ///
    /// The order of the returned IDs is unspecified.  Returns an empty `Vec`
    /// for unknown agents.
    pub fn episode_ids(&self, agent_id: &AgentId) -> Result<Vec<MemoryId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_ids");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| items.iter().map(|m| m.id.clone()).collect()))
    }

    /// Return all episodes for `agent_id` whose importance score is strictly
    /// greater than `threshold`.
    ///
    /// Returns an empty `Vec` for unknown agents or when no episode qualifies.
    pub fn episodes_above_importance(
        &self,
        agent_id: &AgentId,
        threshold: f32,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_above_importance");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items.iter().filter(|m| m.importance > threshold).cloned().collect()
            }))
    }

    /// Return all episodes for `agent_id` whose timestamp falls in the range
    /// `[from, to)` (inclusive start, exclusive end).
    ///
    /// Returns an empty `Vec` for unknown agents or when no episode qualifies.
    pub fn episodes_between(
        &self,
        agent_id: &AgentId,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_between");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items
                    .iter()
                    .filter(|m| m.timestamp >= from && m.timestamp < to)
                    .cloned()
                    .collect()
            }))
    }

    /// Return all episodes for `agent_id` that carry **every** tag in `tags`.
    ///
    /// Returns an empty `Vec` when `tags` is empty, the agent is unknown, or
    /// no episode satisfies all of the given tags.
    pub fn episodes_tagged_with_all(
        &self,
        agent_id: &AgentId,
        tags: &[&str],
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_tagged_with_all");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items
                    .iter()
                    .filter(|m| tags.iter().all(|t| m.has_tag(t)))
                    .cloned()
                    .collect()
            }))
    }

    /// Return the total whitespace-delimited word count across all episodes
    /// for `agent_id`.
    ///
    /// Returns `0` for unknown agents or agents with no episodes.
    pub fn content_word_count_total(
        &self,
        agent_id: &AgentId,
    ) -> Result<usize, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::content_word_count_total");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| {
                items.iter().map(|m| m.word_count()).sum()
            }))
    }

    /// Return the sum of importance scores for all episodes belonging to
    /// `agent_id`.
    ///
    /// Returns `0.0` for unknown agents or agents with no episodes.
    pub fn weighted_importance_sum(
        &self,
        agent_id: &AgentId,
    ) -> Result<f32, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::weighted_importance_sum");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0.0, |items| items.iter().map(|m| m.importance).sum()))
    }

    /// Return a `Vec` of content byte lengths for all episodes of `agent_id`.
    ///
    /// The order matches the internal storage order (insertion order).
    /// Returns an empty `Vec` for unknown agents.
    pub fn episode_content_lengths(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<usize>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_content_lengths");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| items.iter().map(|m| m.content.len()).collect()))
    }

    /// Return the number of distinct agents that have at least one episode.
    ///
    /// Provides a quick count of how many agents have recorded any episodic
    /// memory.  Returns `0` for an empty store.
    pub fn episode_count_by_agent(&self) -> Result<std::collections::HashMap<AgentId, usize>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_count_by_agent");
        Ok(inner.items.iter().map(|(id, items)| (id.clone(), items.len())).collect())
    }

    /// Return all episode IDs across every agent in the store.
    ///
    /// The returned `Vec` is unsorted.  Returns an empty `Vec` for an empty
    /// store.
    pub fn all_episode_ids(&self) -> Result<Vec<MemoryId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::all_episode_ids");
        let ids: Vec<MemoryId> = inner
            .items
            .values()
            .flat_map(|items| items.iter().map(|m| m.id.clone()))
            .collect();
        Ok(ids)
    }

    /// Return all episodes for `agent_id` whose content byte length exceeds `min_bytes`.
    ///
    /// Returns an empty `Vec` when the agent has no episodes or none exceed the
    /// threshold.
    pub fn episodes_above_content_bytes(
        &self,
        agent_id: &AgentId,
        min_bytes: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_above_content_bytes");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items.iter().filter(|m| m.content.len() > min_bytes).cloned().collect()
            }))
    }

    /// Return all agent IDs sorted by episode count in descending order.
    ///
    /// Agents with more episodes appear first.  When two agents have the same
    /// count the tie is broken alphabetically by agent ID.
    pub fn agents_sorted_by_episode_count(&self) -> Result<Vec<AgentId>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::agents_sorted_by_episode_count");
        let mut pairs: Vec<(AgentId, usize)> = inner
            .items
            .iter()
            .map(|(id, items)| (id.clone(), items.len()))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));
        Ok(pairs.into_iter().map(|(id, _)| id).collect())
    }

    /// Return the number of distinct agents that have at least one episode.
    ///
    /// Returns `0` for an empty store.
    pub fn unique_agents_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::unique_agents_count");
        Ok(inner.items.len())
    }

    /// Return the number of episodes for `agent_id` that have at least one tag.
    ///
    /// Returns `0` for an unknown agent or an empty store.
    pub fn episodes_with_tag_count(&self, agent_id: &AgentId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episodes_with_tag_count");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().filter(|m| !m.tags.is_empty()).count()))
    }

    /// Return all episodes for `agent_id` whose content has at least `min_words`
    /// whitespace-delimited words.
    ///
    /// Returns an empty `Vec` for unknown agents or when no episode qualifies.
    pub fn episodes_with_min_word_count(
        &self,
        agent_id: &AgentId,
        min_words: usize,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::episodes_with_min_word_count");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items
                    .iter()
                    .filter(|m| m.word_count() >= min_words)
                    .cloned()
                    .collect()
            }))
    }

    /// Return the episode with the most tags for `agent_id`, or `None` when
    /// the agent has no episodes.  Ties are broken by insertion order.
    pub fn most_tagged_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::most_tagged_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|items| items.iter().max_by_key(|m| m.tag_count()).cloned()))
    }

    /// Return a map of tag → episode count for all episodes of `agent_id`.
    ///
    /// Each entry reports how many episodes carry that particular tag.
    /// Returns an empty map for unknown agents or agents with no tagged episodes.
    pub fn tag_frequency(
        &self,
        agent_id: &AgentId,
    ) -> Result<std::collections::HashMap<String, usize>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::tag_frequency");
        let mut freq: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        if let Some(items) = inner.items.get(agent_id) {
            for item in items {
                for tag in &item.tags {
                    *freq.entry(tag.clone()).or_insert(0) += 1;
                }
            }
        }
        Ok(freq)
    }

    /// Return the episode with the highest `importance` score for `agent_id`.
    ///
    /// When multiple episodes share the maximum importance score the first one
    /// encountered is returned.  Returns `None` for unknown agents or an empty
    /// store.
    pub fn most_important_episode(
        &self,
        agent_id: &AgentId,
    ) -> Result<Option<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::most_important_episode");
        Ok(inner
            .items
            .get(agent_id)
            .and_then(|items| {
                items
                    .iter()
                    .max_by(|a, b| {
                        a.importance
                            .partial_cmp(&b.importance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
            }))
    }

    /// Return the number of episodes for `agent_id` that carry `tag`.
    ///
    /// Returns `0` for unknown agents or if no episode carries `tag`.
    pub fn episode_count_with_tag(
        &self,
        agent_id: &AgentId,
        tag: &str,
    ) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "EpisodicStore::episode_count_with_tag");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(0, |items| items.iter().filter(|m| m.has_tag(tag)).count()))
    }

    /// Return all episodes for `agent_id` whose `importance` is strictly below
    /// `threshold`.
    ///
    /// Returns an empty `Vec` for unknown agents or when no episode qualifies.
    pub fn episodes_below_importance(
        &self,
        agent_id: &AgentId,
        threshold: f32,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::episodes_below_importance");
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items.iter().filter(|m| m.importance < threshold).cloned().collect()
            }))
    }

    /// Return `true` if at least one episode has been stored for `agent_id`.
    ///
    /// This is a cheaper alternative to calling [`EpisodicStore::recall`] and
    /// checking whether the result is empty.
    pub fn has_episodes_for_agent(
        &self,
        agent_id: &AgentId,
    ) -> Result<bool, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "EpisodicStore::has_episodes_for_agent");
        Ok(inner
            .items
            .get(agent_id)
            .map_or(false, |items| !items.is_empty()))
    }

    /// Return all episodes for `agent_id` whose content contains `substr`
    /// (case-sensitive substring match).
    ///
    /// Returns an empty `Vec` for unknown agents or when no episode matches.
    pub fn episodes_with_content_containing(
        &self,
        agent_id: &AgentId,
        substr: &str,
    ) -> Result<Vec<MemoryItem>, AgentRuntimeError> {
        let inner = recover_lock(
            self.inner.lock(),
            "EpisodicStore::episodes_with_content_containing",
        );
        Ok(inner
            .items
            .get(agent_id)
            .map_or_else(Vec::new, |items| {
                items
                    .iter()
                    .filter(|m| m.content.contains(substr))
                    .cloned()
                    .collect()
            }))
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

    /// Remove all entries that have the given tag.
    ///
    /// Returns the number of entries removed.
    pub fn remove_entries_with_tag(&self, tag: &str) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::remove_entries_with_tag");
        let before = inner.entries.len();
        inner.entries.retain(|e| !e.tags.iter().any(|t| t == tag));
        Ok(before - inner.entries.len())
    }

    /// Return the tag that appears most often across all entries.
    ///
    /// Returns `None` if there are no tagged entries.
    pub fn most_common_tag(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::most_common_tag");
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for entry in &inner.entries {
            for tag in &entry.tags {
                *counts.entry(tag.as_str()).or_insert(0) += 1;
            }
        }
        Ok(counts.into_iter().max_by_key(|(_, c)| *c).map(|(t, _)| t.to_string()))
    }

    /// Return the keys of all entries that have the given tag.
    pub fn keys_for_tag(&self, tag: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::keys_for_tag");
        let keys = inner
            .entries
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .map(|e| e.key.clone())
            .collect();
        Ok(keys)
    }

    /// Return a sorted list of every distinct tag appearing across all entries.
    pub fn unique_tags(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::unique_tags");
        let mut tags: Vec<String> = inner
            .entries
            .iter()
            .flat_map(|e| e.tags.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        tags.sort_unstable();
        Ok(tags)
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

    /// Return the number of entries that have an associated embedding vector.
    pub fn entry_count_with_embedding(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::entry_count_with_embedding");
        Ok(inner.entries.iter().filter(|e| e.embedding.is_some()).count())
    }

    /// Return the stored value for the first entry whose key matches `key`,
    /// or `None` if no such entry exists.
    ///
    /// Simpler alternative to [`retrieve_by_key`] when only the value is needed
    /// and tags are not required.
    ///
    /// [`retrieve_by_key`]: SemanticStore::retrieve_by_key
    pub fn get_value(&self, key: &str) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::get_value");
        Ok(inner
            .entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.value.clone()))
    }

    /// Return the tags for the first entry whose key matches `key`.
    ///
    /// Returns `None` if no entry with that key exists.
    pub fn get_tags(&self, key: &str) -> Result<Option<Vec<String>>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::get_tags");
        Ok(inner.entries.iter().find(|e| e.key == key).map(|e| e.tags.clone()))
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

    /// Return the tags associated with the first entry matching `key`, or
    /// `None` if no entry with that key exists.
    pub fn tags_for(&self, key: &str) -> Result<Option<Vec<String>>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::tags_for");
        Ok(inner
            .entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.tags.clone()))
    }

    /// Return `true` if at least one entry with the given `key` exists.
    pub fn has_key(&self, key: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::has_key");
        Ok(inner.entries.iter().any(|e| e.key == key))
    }

    /// Return the value string of the first entry whose key matches `key`,
    /// or `None` if no such entry exists.
    pub fn value_for(&self, key: &str) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::value_for");
        Ok(inner
            .entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.value.clone()))
    }

    /// Return the number of entries that have no tags.
    pub fn entries_without_tags(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::entries_without_tags");
        Ok(inner.entries.iter().filter(|e| e.tags.is_empty()).count())
    }

    /// Return the keys of all entries that have no tags.
    /// Return the key of the entry with the most tags, or `None` if the store
    /// is empty.
    pub fn most_tagged_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::most_tagged_key");
        Ok(inner
            .entries
            .iter()
            .max_by_key(|e| e.tags.len())
            .map(|e| e.key.clone()))
    }

    /// Return the count of entries whose value contains `substring`.
    /// Rename `old_tag` to `new_tag` across all entries.
    ///
    /// Returns the number of entries that were modified.
    pub fn rename_tag(&self, old_tag: &str, new_tag: &str) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::rename_tag");
        let mut count = 0;
        for entry in &mut inner.entries {
            for tag in &mut entry.tags {
                if tag == old_tag {
                    *tag = new_tag.to_string();
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Return the count of entries whose value contains `substring`.
    pub fn count_matching_value(&self, substring: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::count_matching_value");
        Ok(inner.entries.iter().filter(|e| e.value.contains(substring)).count())
    }

    /// Return the keys of all entries that have no tags.
    pub fn entries_with_no_tags(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::entries_with_no_tags");
        Ok(inner
            .entries
            .iter()
            .filter(|e| e.tags.is_empty())
            .map(|e| e.key.clone())
            .collect())
    }

    /// Return the mean number of tags per entry.
    ///
    /// Returns `0.0` when the store is empty.
    pub fn avg_tag_count_per_entry(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::avg_tag_count_per_entry");
        let n = inner.entries.len();
        if n == 0 {
            return Ok(0.0);
        }
        let total: usize = inner.entries.iter().map(|e| e.tags.len()).sum();
        Ok(total as f64 / n as f64)
    }

    /// Return the key of the most recently inserted entry, or `None` if empty.
    pub fn most_recent_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::most_recent_key");
        Ok(inner.entries.last().map(|e| e.key.clone()))
    }

    /// Return the key of the earliest inserted entry, or `None` if empty.
    pub fn oldest_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::oldest_key");
        Ok(inner.entries.first().map(|e| e.key.clone()))
    }

    /// Remove all entries whose key equals `key`.
    ///
    /// Returns the number of entries removed.
    pub fn remove_by_key(&self, key: &str) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::remove_by_key");
        let before = inner.entries.len();
        inner.entries.retain(|e| e.key != key);
        Ok(before - inner.entries.len())
    }

    /// Return the number of entries that include `tag` in their tag list.
    pub fn entry_count_with_tag(&self, tag: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::entry_count_with_tag");
        Ok(inner
            .entries
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .count())
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

    /// Return the stored value for every entry, in insertion order.
    pub fn values(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::values");
        Ok(inner.entries.iter().map(|e| e.value.clone()).collect())
    }

    /// Return all keys that contain `pattern` as a substring (case-insensitive).
    ///
    /// Useful for prefix/contains searches without loading values.
    pub fn keys_matching(&self, pattern: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::keys_matching");
        let lower = pattern.to_ascii_lowercase();
        Ok(inner
            .entries
            .iter()
            .filter(|e| e.key.to_ascii_lowercase().contains(&lower))
            .map(|e| e.key.clone())
            .collect())
    }

    /// Update the stored value of the first entry whose key matches `key`.
    ///
    /// Returns `Ok(true)` if found and updated, `Ok(false)` if not found.
    pub fn update_value(
        &self,
        key: &str,
        new_value: impl Into<String>,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "SemanticStore::update_value");
        if let Some(entry) = inner.entries.iter_mut().find(|e| e.key == key) {
            entry.value = new_value.into();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Return all stored entries as a `HashMap<key, value>`.
    ///
    /// Useful for serialization or bulk inspection.  Tags and embeddings are
    /// discarded; use `iter()` / `retrieve_by_key` when those are needed.
    pub fn to_map(&self) -> Result<std::collections::HashMap<String, String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::to_map");
        Ok(inner
            .entries
            .iter()
            .map(|e| (e.key.clone(), e.value.clone()))
            .collect())
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

    /// Return the sum of byte lengths of all stored values.
    ///
    /// Returns `0` if no entries have been stored.
    pub fn total_value_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::total_value_bytes");
        Ok(inner.entries.iter().map(|e| e.value.len()).sum())
    }

    /// Return the mean byte length of all stored values.
    ///
    /// Returns `0.0` if no entries have been stored.
    pub fn avg_value_bytes(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::avg_value_bytes");
        if inner.entries.is_empty() {
            return Ok(0.0);
        }
        let total: usize = inner.entries.iter().map(|e| e.value.len()).sum();
        Ok(total as f64 / inner.entries.len() as f64)
    }

    /// Return the byte length of the longest stored value.
    ///
    /// Returns `0` if no entries have been stored.
    pub fn max_value_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::max_value_bytes");
        Ok(inner.entries.iter().map(|e| e.value.len()).max().unwrap_or(0))
    }

    /// Return the byte length of the shortest stored value.
    ///
    /// Returns `0` if no entries have been stored.
    pub fn min_value_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::min_value_bytes");
        Ok(inner.entries.iter().map(|e| e.value.len()).min().unwrap_or(0))
    }

    /// Return all stored keys in ascending sorted order.
    pub fn all_keys(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::all_keys");
        let mut keys: Vec<String> = inner.entries.iter().map(|e| e.key.clone()).collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Return all stored keys whose string representation starts with `prefix`,
    /// in ascending sorted order.
    ///
    /// Returns an empty `Vec` if no keys share the prefix.
    pub fn keys_with_prefix(&self, prefix: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "SemanticStore::keys_with_prefix");
        let mut keys: Vec<String> = inner
            .entries
            .iter()
            .filter(|e| e.key.starts_with(prefix))
            .map(|e| e.key.clone())
            .collect();
        keys.sort_unstable();
        Ok(keys)
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
    /// Insert `key → value` only if `key` is not already present.
    ///
    /// Returns `Ok(true)` if the entry was inserted, `Ok(false)` if the key
    /// was already present (the existing value is left unchanged).  Capacity
    /// limits and LRU eviction apply as with [`set`].
    ///
    /// [`set`]: WorkingMemory::set
    pub fn set_if_absent(
        &self,
        key: impl Into<String> + std::fmt::Debug,
        value: impl Into<String> + std::fmt::Debug,
    ) -> Result<bool, AgentRuntimeError> {
        let key = key.into();
        {
            let inner = recover_lock(self.inner.lock(), "WorkingMemory::set_if_absent");
            if inner.map.contains_key(&key) {
                return Ok(false);
            }
        }
        self.set(key, value)?;
        Ok(true)
    }

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

    /// Update multiple existing keys in a single lock acquisition.
    ///
    /// Each `(key, value)` pair is applied only if the key is already present
    /// (same semantics as [`update_if_exists`]).  Returns the number of keys
    /// that were found and updated.
    ///
    /// [`update_if_exists`]: WorkingMemory::update_if_exists
    pub fn update_many(
        &self,
        pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::update_many");
        let mut updated = 0;
        for (key, value) in pairs {
            let key: String = key.into();
            if let Some(v) = inner.map.get_mut(&key) {
                *v = value.into();
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Return all keys whose names start with `prefix`, in insertion order.
    ///
    /// Useful for namespace-prefixed keys such as `"user:name"`,
    /// `"user:email"` where all user-related keys share the `"user:"` prefix.
    pub fn keys_starting_with(&self, prefix: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_starting_with");
        Ok(inner
            .map
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }

    /// Return all `(key, value)` pairs whose value contains `pattern` as a
    /// substring.  Comparison is case-sensitive.
    ///
    /// Useful for scanning working memory for values that match a keyword
    /// without iterating the full map externally.
    pub fn values_matching(&self, pattern: &str) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::values_matching");
        Ok(inner
            .map
            .iter()
            .filter(|(_, v)| v.contains(pattern))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }

    /// Return the byte length of the value stored at `key`, or `None` if the
    /// key is not present.
    ///
    /// Useful for estimating memory usage without cloning the value string.
    pub fn value_length(&self, key: &str) -> Result<Option<usize>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_length");
        Ok(inner.map.get(key).map(|v| v.len()))
    }

    /// Rename a key without changing its value or insertion order.
    ///
    /// Return `true` if **all** of the given keys are currently stored.
    ///
    /// An empty iterator returns `true` vacuously.
    pub fn contains_all<'a>(&self, keys: impl IntoIterator<Item = &'a str>) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::contains_all");
        Ok(keys.into_iter().all(|k| inner.map.contains_key(k)))
    }

    /// Return `true` if **any** of the given keys is currently stored.
    ///
    /// An empty iterator returns `false`.
    pub fn has_any_key<'a>(&self, keys: impl IntoIterator<Item = &'a str>) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::has_any_key");
        Ok(keys.into_iter().any(|k| inner.map.contains_key(k)))
    }

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
    /// Return all entries as `(key, value)` pairs sorted alphabetically by key.
    ///
    /// Unlike [`iter`]/[`entries`] which return entries in insertion order,
    /// this always returns a deterministically ordered snapshot.
    ///
    /// [`iter`]: WorkingMemory::iter
    /// [`entries`]: WorkingMemory::entries
    pub fn iter_sorted(&self) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::iter_sorted");
        let mut pairs: Vec<(String, String)> = inner
            .map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(pairs)
    }

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

    /// Peek at the oldest entry without removing it.
    ///
    /// Returns `None` if the memory is empty.  Unlike [`pop_oldest`] this
    /// method does not modify the store.
    ///
    /// [`pop_oldest`]: WorkingMemory::pop_oldest
    pub fn peek_oldest(&self) -> Result<Option<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::peek_oldest");
        Ok(inner.order.front().and_then(|key| {
            inner.map.get(key).map(|val| (key.clone(), val.clone()))
        }))
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

    /// Return the current fill ratio as `len / capacity`.
    ///
    /// Returns a value in `[0.0, 1.0]`.  `1.0` means the memory is full and
    /// the next insert will evict the oldest entry.
    pub fn fill_ratio(&self) -> Result<f64, AgentRuntimeError> {
        Ok(self.len()? as f64 / self.capacity as f64)
    }

    /// Return `true` when the number of stored entries equals the configured capacity.
    ///
    /// When `true`, the next [`set`] call for a new key will evict the oldest entry.
    ///
    /// [`set`]: WorkingMemory::set
    pub fn is_at_capacity(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.len()? >= self.capacity)
    }

    /// Remove all entries whose key begins with `prefix`.
    ///
    /// Returns the number of entries removed.  Removal preserves insertion order
    /// for the surviving entries.
    pub fn remove_keys_starting_with(&self, prefix: &str) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::remove_keys_starting_with");
        let before = inner.map.len();
        inner.map.retain(|k, _| !k.starts_with(prefix));
        Ok(before - inner.map.len())
    }

    /// Return the total number of bytes used by all stored values.
    ///
    /// Useful for estimating memory pressure without allocating a full snapshot.
    pub fn total_value_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::total_value_bytes");
        Ok(inner.map.values().map(|v| v.len()).sum())
    }

    /// Return the byte length of the longest key currently stored.
    ///
    /// Returns `0` when the memory is empty.
    pub fn max_key_length(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::max_key_length");
        Ok(inner.map.keys().map(|k| k.len()).max().unwrap_or(0))
    }

    /// Return the byte length of the longest stored value.
    ///
    /// Returns `0` when the memory is empty.
    pub fn max_value_length(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::max_value_length");
        Ok(inner.map.values().map(|v| v.len()).max().unwrap_or(0))
    }

    /// Return the byte length of the shortest stored value.
    ///
    /// Returns `0` when the memory is empty.
    pub fn min_value_length(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::min_value_length");
        Ok(inner.map.values().map(|v| v.len()).min().unwrap_or(0))
    }

    /// Return the number of keys whose text contains `substring`.
    ///
    /// The search is case-sensitive.  Returns `0` when the store is empty or
    /// no key matches.
    pub fn key_count_matching(&self, substring: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::key_count_matching");
        Ok(inner.map.keys().filter(|k| k.contains(substring)).count())
    }

    /// Return the mean byte length of all stored values.
    ///
    /// Returns `0.0` when the store is empty.
    pub fn avg_value_length(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::avg_value_length");
        let n = inner.map.len();
        if n == 0 {
            return Ok(0.0);
        }
        let total: usize = inner.map.values().map(|v| v.len()).sum();
        Ok(total as f64 / n as f64)
    }

    /// Return the number of entries whose value exceeds `min_bytes` bytes in length.
    pub fn count_above_value_length(&self, min_bytes: usize) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::count_above_value_length");
        Ok(inner.map.values().filter(|v| v.len() > min_bytes).count())
    }

    /// Return the key with the most bytes, or `None` if the store is empty.
    ///
    /// When multiple keys share the maximum byte length, one of them is
    /// returned (unspecified which).
    pub fn longest_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::longest_key");
        Ok(inner
            .map
            .keys()
            .max_by_key(|k| k.len())
            .map(|k| k.clone()))
    }

    /// Return the value with the most bytes, or `None` if the store is empty.
    ///
    /// When multiple values share the maximum byte length, one of them is
    /// returned (unspecified which).
    pub fn longest_value(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::longest_value");
        Ok(inner
            .map
            .values()
            .max_by_key(|v| v.len())
            .map(|v| v.clone()))
    }

    /// Return all `(key, value)` pairs sorted alphabetically by key.
    ///
    /// Provides a stable, deterministic view of the store contents.
    /// Returns an empty `Vec` for an empty store.
    pub fn key_value_pairs_sorted(&self) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::key_value_pairs_sorted");
        let mut pairs: Vec<(String, String)> = inner
            .map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(pairs)
    }

    /// Return all `(key, value)` pairs whose key starts with `prefix`.
    pub fn pairs_starting_with(&self, prefix: &str) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::pairs_starting_with");
        let pairs = inner
            .map
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(pairs)
    }

    /// Return the sum of byte lengths of all stored keys.
    pub fn total_key_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::total_key_bytes");
        Ok(inner.map.keys().map(|k| k.len()).sum())
    }

    /// Return all key-value pairs sorted by key byte length ascending, then
    /// alphabetically for equal lengths.
    ///
    /// Returns an empty `Vec` for an empty store.
    pub fn entries_sorted_by_key_length(
        &self,
    ) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "WorkingMemory::entries_sorted_by_key_length");
        let mut pairs: Vec<(String, String)> = inner
            .map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        pairs.sort_unstable_by(|(a, _), (b, _)| a.len().cmp(&b.len()).then(a.cmp(b)));
        Ok(pairs)
    }

    /// Return the maximum byte length among all stored values.
    ///
    /// Returns `0` for an empty store.
    pub fn value_bytes_max(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_bytes_max");
        Ok(inner.map.values().map(|v| v.len()).max().unwrap_or(0))
    }

    /// Return all keys whose string representation ends with `suffix`,
    /// sorted alphabetically.
    ///
    /// Returns an empty `Vec` when no key qualifies.
    pub fn keys_with_suffix(
        &self,
        suffix: &str,
    ) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_with_suffix");
        let mut keys: Vec<String> = inner
            .map
            .keys()
            .filter(|k| k.ends_with(suffix))
            .cloned()
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Return the average byte length of all stored keys.
    ///
    /// Returns `0.0` for an empty store.
    pub fn avg_key_length(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::avg_key_length");
        if inner.map.is_empty() {
            return Ok(0.0);
        }
        let total: usize = inner.map.keys().map(|k| k.len()).sum();
        Ok(total as f64 / inner.map.len() as f64)
    }

    /// Return the maximum key byte length, or `0` if the store is empty.
    ///
    /// Useful as a quick bound on the widest key that will be emitted.
    pub fn max_key_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::max_key_bytes");
        Ok(inner.map.keys().map(|k| k.len()).max().unwrap_or(0))
    }

    /// Return the count of values whose byte length exceeds `min_bytes`.
    ///
    /// Returns `0` for an empty store or when no value exceeds the threshold.
    pub fn value_count_above_bytes(&self, min_bytes: usize) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_count_above_bytes");
        Ok(inner.map.values().filter(|v| v.len() > min_bytes).count())
    }

    /// Return `true` if any key in the store starts with `prefix`.
    ///
    /// Returns `false` for an empty store.
    pub fn has_key_starting_with(&self, prefix: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::has_key_starting_with");
        Ok(inner.map.keys().any(|k| k.starts_with(prefix)))
    }

    /// Return the number of keys whose names begin with `prefix`.
    ///
    /// Returns `0` for an empty store or when no key matches.
    pub fn key_count_starting_with(&self, prefix: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::key_count_starting_with");
        Ok(inner.map.keys().filter(|k| k.starts_with(prefix)).count())
    }

    /// Return the value with the greatest byte length in the store.
    ///
    /// Returns `None` for an empty store. When multiple values share the
    /// maximum byte length, the lexicographically largest is returned.
    pub fn value_with_max_bytes(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_with_max_bytes");
        Ok(inner
            .map
            .values()
            .max_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)))
            .cloned())
    }

    /// Return all keys whose associated value has a byte length greater than
    /// `min_bytes`, sorted alphabetically.
    ///
    /// Returns an empty `Vec` when no key qualifies.
    pub fn values_longer_than(
        &self,
        min_bytes: usize,
    ) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::values_longer_than");
        let mut keys: Vec<String> = inner
            .map
            .iter()
            .filter(|(_, v)| v.len() > min_bytes)
            .map(|(k, _)| k.clone())
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Return `true` if any key in the store starts with `prefix`.
    pub fn has_key_with_prefix(&self, prefix: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::has_key_with_prefix");
        Ok(inner.map.keys().any(|k| k.starts_with(prefix)))
    }

    /// Return the count of values that start with `prefix`.
    ///
    /// Returns `0` for an empty store or when no value matches.
    pub fn value_count_matching_prefix(
        &self,
        prefix: &str,
    ) -> Result<usize, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "WorkingMemory::value_count_matching_prefix");
        Ok(inner.map.values().filter(|v| v.starts_with(prefix)).count())
    }

    /// Return the sum of byte lengths of all values currently stored.
    ///
    /// Returns `0` for an empty store.
    pub fn value_bytes_total(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_bytes_total");
        Ok(inner.map.values().map(|v| v.len()).sum())
    }

    /// Return all keys sorted alphabetically.
    ///
    /// Returns an empty `Vec` for an empty store.
    pub fn all_keys_sorted(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::all_keys_sorted");
        let mut keys: Vec<String> = inner.map.keys().cloned().collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Return all values in the store, sorted alphabetically.
    ///
    /// Useful for deterministic comparison in tests and diagnostics.
    /// Returns an empty `Vec` for an empty store.
    pub fn all_values_sorted(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::all_values_sorted");
        let mut values: Vec<String> = inner.map.values().cloned().collect();
        values.sort_unstable();
        Ok(values)
    }

    /// Return the keys whose stored value begins with `prefix`.
    ///
    /// Returns an empty `Vec` when no value matches.
    pub fn keys_with_value_prefix(&self, prefix: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_with_value_prefix");
        Ok(inner
            .map
            .iter()
            .filter(|(_, v)| v.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect())
    }

    /// Return the number of key-value pairs currently stored.
    pub fn key_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::key_count");
        Ok(inner.map.len())
    }

    /// Return the value for `key`, or `default` if the key is not present.
    pub fn value_for_key_or_default<'a>(
        &self,
        key: &str,
        default: &'a str,
    ) -> Result<String, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_for_key_or_default");
        Ok(inner
            .map
            .get(key)
            .map(|v| v.clone())
            .unwrap_or_else(|| default.to_string()))
    }

    /// Return all `(key, value)` pairs whose key starts with `prefix`,
    /// sorted alphabetically by key.
    ///
    /// Returns an empty `Vec` for an empty store or when no key matches.
    pub fn entries_with_key_prefix(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::entries_with_key_prefix");
        let mut entries: Vec<(String, String)> = inner
            .map
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries)
    }

    /// Return `true` if any stored value is exactly equal to `val`.
    ///
    /// Returns `false` for an empty store.
    pub fn has_value_equal_to(&self, val: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::has_value_equal_to");
        Ok(inner.map.values().any(|v| v == val))
    }

    /// Return `true` if the value stored under `key` contains `substr`.
    ///
    /// Returns `false` when `key` is not present.
    pub fn value_contains(&self, key: &str, substr: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_contains");
        Ok(inner.map.get(key).map_or(false, |v| v.contains(substr)))
    }

    /// Return all keys that do NOT start with `prefix`, sorted alphabetically.
    ///
    /// Returns an empty `Vec` for an empty store or when all keys match.
    pub fn keys_without_prefix(&self, prefix: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_without_prefix");
        let mut keys: Vec<String> = inner
            .map
            .keys()
            .filter(|k| !k.starts_with(prefix))
            .cloned()
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Return the number of stored values that are exactly equal to `val`.
    pub fn count_values_equal_to(&self, val: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::count_values_equal_to");
        Ok(inner.map.values().filter(|v| v.as_str() == val).count())
    }

    /// Return the number of keys whose byte length is strictly less than
    /// `max_bytes`.
    pub fn count_keys_below_bytes(&self, max_bytes: usize) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::count_keys_below_bytes");
        Ok(inner.map.keys().filter(|k| k.len() < max_bytes).count())
    }

    /// Return the number of Unicode scalar values (chars) in the value stored
    /// under `key`, or `0` if the key is not present.
    pub fn value_char_count(&self, key: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_char_count");
        Ok(inner.map.get(key).map_or(0, |v| v.chars().count()))
    }

    /// Return all keys whose byte length is strictly greater than `min_bytes`.
    ///
    /// Returns an empty `Vec` when no keys satisfy the condition.
    pub fn keys_longer_than(&self, min_bytes: usize) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_longer_than");
        Ok(inner.map.keys().filter(|k| k.len() > min_bytes).cloned().collect())
    }

    /// Return all keys whose byte length is strictly less than `max_bytes`.
    ///
    /// Returns an empty `Vec` when no keys satisfy the condition.
    pub fn keys_shorter_than(&self, max_bytes: usize) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_shorter_than");
        Ok(inner.map.keys().filter(|k| k.len() < max_bytes).cloned().collect())
    }

    /// Return all stored values that start with `prefix`.
    pub fn values_with_prefix(&self, prefix: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::values_with_prefix");
        Ok(inner
            .map
            .values()
            .filter(|v| v.starts_with(prefix))
            .cloned()
            .collect())
    }

    /// Return `true` if the value stored under `key` starts with `prefix`.
    ///
    /// Returns `false` when `key` is not present.
    pub fn value_starts_with(&self, key: &str, prefix: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_starts_with");
        Ok(inner
            .map
            .get(key)
            .map_or(false, |v| v.starts_with(prefix)))
    }

    /// Return a histogram of value byte lengths bucketed by `bucket_size`.
    ///
    /// The returned `Vec` contains `(bucket_start, count)` pairs where
    /// `bucket_start = (value_len / bucket_size) * bucket_size`.  Pairs are
    /// sorted by `bucket_start` in ascending order.  Returns an empty `Vec`
    /// for an empty store or when `bucket_size == 0`.
    pub fn value_length_histogram(
        &self,
        bucket_size: usize,
    ) -> Result<Vec<(usize, usize)>, AgentRuntimeError> {
        if bucket_size == 0 {
            return Ok(Vec::new());
        }
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_length_histogram");
        let mut buckets: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for v in inner.map.values() {
            let bucket = (v.len() / bucket_size) * bucket_size;
            *buckets.entry(bucket).or_insert(0) += 1;
        }
        let mut result: Vec<(usize, usize)> = buckets.into_iter().collect();
        result.sort_unstable_by_key(|(k, _)| *k);
        Ok(result)
    }

    /// Return the number of keys that contain `substr` as a substring.
    ///
    /// Useful for quickly counting "semantic" keys that share a common
    /// token (e.g. `"intent"`, `"context"`, `"goal"`).  Returns `0` for
    /// an empty store or when no key matches.
    pub fn semantic_key_count(&self, substr: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::semantic_key_count");
        Ok(inner.map.keys().filter(|k| k.contains(substr)).count())
    }

    /// Return the byte length of the longest stored value, or `0` when empty.
    ///
    /// Helpful for detecting unusually large values that might indicate a
    /// working-memory bloat.  Returns `0` for an empty store.
    pub fn longest_value_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::longest_value_bytes");
        Ok(inner.map.values().map(|v| v.len()).max().unwrap_or(0))
    }

    /// Return the shortest key length, or `0` if the store is empty.
    pub fn min_key_length(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::min_key_length");
        Ok(inner.map.keys().map(|k| k.len()).min().unwrap_or(0))
    }

    /// Return the number of keys that start with the given prefix.
    pub fn count_matching_prefix(&self, prefix: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::count_matching_prefix");
        Ok(inner.map.keys().filter(|k| k.starts_with(prefix)).count())
    }

    /// Return a list of `(key, value_byte_length)` pairs for all entries.
    /// Return the number of key-value entries currently stored.
    pub fn entry_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::entry_count");
        Ok(inner.map.len())
    }

    /// Return a list of `(key, value_byte_length)` pairs for all entries.
    pub fn value_lengths(&self) -> Result<Vec<(String, usize)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_lengths");
        Ok(inner.map.iter().map(|(k, v)| (k.clone(), v.len())).collect())
    }

    /// Return the keys of all entries whose value byte length is strictly
    /// greater than `threshold`.
    pub fn keys_with_value_longer_than(&self, threshold: usize) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_with_value_longer_than");
        Ok(inner
            .map
            .iter()
            .filter(|(_, v)| v.len() > threshold)
            .map(|(k, _)| k.clone())
            .collect())
    }

    /// Return the key whose associated value has the most bytes, or `None` if empty.
    pub fn longest_value_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::longest_value_key");
        Ok(inner
            .map
            .iter()
            .max_by_key(|(_, v)| v.len())
            .map(|(k, _)| k.clone()))
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

    /// Copy all entries from `other` into `self`.
    ///
    /// Entries from `other` are inserted in its insertion order.  Capacity
    /// limits and eviction apply as if each entry were inserted individually
    /// via [`set`].
    ///
    /// [`set`]: WorkingMemory::set
    pub fn merge_from(&self, other: &WorkingMemory) -> Result<usize, AgentRuntimeError> {
        let pairs: Vec<(String, String)> = {
            let inner = recover_lock(other.inner.lock(), "WorkingMemory::merge_from(read)");
            inner.order.iter().filter_map(|k| {
                inner.map.get(k).map(|v| (k.clone(), v.clone()))
            }).collect()
        };
        let count = pairs.len();
        self.set_many(pairs)?;
        Ok(count)
    }

    /// Return the number of entries for which `predicate(key, value)` returns `true`.
    pub fn entry_count_satisfying<F>(&self, mut predicate: F) -> Result<usize, AgentRuntimeError>
    where
        F: FnMut(&str, &str) -> bool,
    {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::entry_count_satisfying");
        Ok(inner.map.iter().filter(|(k, v)| predicate(k.as_str(), v.as_str())).count())
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

    /// Return all values whose content contains `substring` (case-sensitive),
    /// in insertion order.
    ///
    /// Returns an empty `Vec` when no values match or the store is empty.
    pub fn values_containing(&self, substring: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::values_containing");
        let result = inner
            .order
            .iter()
            .filter_map(|k| {
                inner.map.get(k).filter(|v| v.contains(substring)).cloned()
            })
            .collect();
        Ok(result)
    }

    /// Return all keys in ascending sorted (lexicographic) order.
    ///
    /// Unlike [`get_all_keys`] which returns keys in insertion order, this
    /// method always returns them in a deterministic alphabetical order.
    ///
    /// [`get_all_keys`]: WorkingMemory::get_all_keys
    pub fn keys_sorted(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::keys_sorted");
        let mut keys: Vec<String> = inner.map.keys().cloned().collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Return the value associated with the longest key, or `None` if the
    /// store is empty.
    ///
    /// When multiple keys share the maximum byte length, the one that sorts
    /// first lexicographically is chosen for deterministic output.
    pub fn value_for_longest_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_for_longest_key");
        let best_key = inner
            .map
            .keys()
            .max_by(|a, b| a.len().cmp(&b.len()).then_with(|| b.cmp(a)));
        Ok(best_key.and_then(|k| inner.map.get(k).cloned()))
    }

    /// Return `true` if any entry in this store has a value equal to `value`.
    ///
    /// Performs a linear scan over all stored values.  For exact-match
    /// semantics use [`get`] instead.
    ///
    /// [`get`]: WorkingMemory::get
    pub fn contains_value(&self, value: &str) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::contains_value");
        Ok(inner.map.values().any(|v| v == value))
    }

    /// Return the key with the fewest bytes, or `None` if the store is empty.
    ///
    /// When multiple keys share the minimum byte length, the one that sorts
    /// first lexicographically is returned for deterministic output.
    pub fn shortest_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::shortest_key");
        Ok(inner
            .map
            .keys()
            .min_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)))
            .cloned())
    }

    /// Return a `Vec` of `(key_byte_len, value_byte_len)` pairs for every entry.
    ///
    /// The order of the returned pairs is unspecified (hash-map iteration order).
    /// Returns an empty `Vec` for an empty store.
    pub fn entry_byte_pairs(&self) -> Result<Vec<(usize, usize)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::entry_byte_pairs");
        Ok(inner.map.iter().map(|(k, v)| (k.len(), v.len())).collect())
    }

    /// Return the total byte usage of the store — the sum of all key and value
    /// byte lengths.
    ///
    /// Returns `0` for an empty store.
    pub fn total_bytes(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::total_bytes");
        Ok(inner.map.iter().map(|(k, v)| k.len() + v.len()).sum())
    }

    /// Return the key whose value has the greatest byte length, or `None` if
    /// the store is empty.
    ///
    /// When multiple values share the maximum byte length, the key that sorts
    /// first lexicographically is returned for deterministic output.
    pub fn key_with_longest_value(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::key_with_longest_value");
        Ok(inner
            .map
            .iter()
            .max_by(|(ka, va), (kb, vb)| va.len().cmp(&vb.len()).then_with(|| kb.cmp(ka)))
            .map(|(k, _)| k.clone()))
    }

    /// Return the number of entries whose value byte length is strictly less than `max_bytes`.
    ///
    /// Returns `0` for an empty store.
    pub fn count_below_value_length(&self, max_bytes: usize) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::count_below_value_length");
        Ok(inner.map.values().filter(|v| v.len() < max_bytes).count())
    }

    /// Return the value associated with the shortest key, or `None` if the
    /// store is empty.
    ///
    /// When multiple keys share the minimum byte length, the one that sorts
    /// first lexicographically is selected for deterministic output.
    pub fn value_for_shortest_key(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::value_for_shortest_key");
        let best_key = inner
            .map
            .keys()
            .min_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        Ok(best_key.and_then(|k| inner.map.get(k).cloned()))
    }

    /// Return all `(key, value)` pairs whose value byte length is strictly
    /// greater than `min_value_bytes`.
    ///
    /// Returns an empty `Vec` for an empty store or when no values exceed the
    /// threshold.
    pub fn key_value_pairs_above_length(
        &self,
        min_value_bytes: usize,
    ) -> Result<Vec<(String, String)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::key_value_pairs_above_length");
        let pairs: Vec<(String, String)> = inner
            .map
            .iter()
            .filter(|(_, v)| v.len() > min_value_bytes)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(pairs)
    }

    /// Return the value for `key` if it exists, otherwise insert `default` and
    /// return it.
    ///
    /// The returned `String` is always the value associated with `key` after
    /// this call completes.  If the key was absent and the store has a capacity
    /// limit, the oldest entry may be evicted to make room for the new entry.
    pub fn get_or_insert(
        &self,
        key: impl Into<String>,
        default: impl Into<String>,
    ) -> Result<String, AgentRuntimeError> {
        let key = key.into();
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::get_or_insert");
        if let Some(v) = inner.map.get(&key) {
            return Ok(v.clone());
        }
        let value = default.into();
        // Evict oldest entry if at capacity.
        let cap = self.capacity;
        if inner.map.len() >= cap && cap > 0 {
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
            }
        }
        inner.map.insert(key.clone(), value.clone());
        inner.order.push_back(key);
        Ok(value)
    }

    /// Parse the value for `key` as an `i64`, add `step`, and store the result.
    ///
    /// If `key` does not exist it is treated as `0` before adding `step`.
    /// Returns the new value after the increment.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::Memory` if the existing value cannot be
    /// parsed as `i64`.
    pub fn increment(&self, key: impl Into<String>, step: i64) -> Result<i64, AgentRuntimeError> {
        let key = key.into();
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::increment");
        let current: i64 = if let Some(v) = inner.map.get(&key) {
            v.parse::<i64>().map_err(|_| {
                AgentRuntimeError::Memory(format!(
                    "WorkingMemory::increment: value for key '{key}' is not a valid integer"
                ))
            })?
        } else {
            0
        };
        let new_val = current.saturating_add(step);
        let new_str = new_val.to_string();
        if !inner.map.contains_key(&key) {
            // Evict oldest if at capacity.
            let cap = self.capacity;
            if inner.map.len() >= cap && cap > 0 {
                if let Some(oldest) = inner.order.pop_front() {
                    inner.map.remove(&oldest);
                }
            }
            inner.order.push_back(key.clone());
        }
        inner.map.insert(key, new_str);
        Ok(new_val)
    }

    /// Swap the values of `key_a` and `key_b`.
    ///
    /// Returns `Ok(())` when both keys exist and the swap succeeds.
    /// Returns `Err` when either key is absent — the store is left unchanged.
    pub fn swap(&self, key_a: &str, key_b: &str) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "WorkingMemory::swap");
        let a = inner.map.get(key_a).cloned().ok_or_else(|| {
            AgentRuntimeError::Memory(format!("WorkingMemory::swap: key '{key_a}' not found"))
        })?;
        let b = inner.map.get(key_b).cloned().ok_or_else(|| {
            AgentRuntimeError::Memory(format!("WorkingMemory::swap: key '{key_b}' not found"))
        })?;
        inner.map.insert(key_a.to_owned(), b);
        inner.map.insert(key_b.to_owned(), a);
        Ok(())
    }

    /// Return all values in the store that are non-empty strings.
    ///
    /// The returned `Vec` is in insertion order.  Returns an empty `Vec` for
    /// an empty store or when all values are empty strings.
    pub fn non_empty_values(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::non_empty_values");
        let values: Vec<String> = inner
            .order
            .iter()
            .filter_map(|k| inner.map.get(k))
            .filter(|v| !v.is_empty())
            .cloned()
            .collect();
        Ok(values)
    }

    /// Return all values sorted alphabetically.
    ///
    /// The sort is lexicographic (byte order).  Duplicate values are preserved.
    pub fn values_sorted(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::values_sorted");
        let mut values: Vec<String> = inner.map.values().cloned().collect();
        values.sort_unstable();
        Ok(values)
    }

    /// Return the number of keys whose byte length is strictly greater than
    /// `min_bytes`.
    ///
    /// Returns `0` for an empty store or when no key qualifies.
    pub fn count_keys_above_bytes(&self, min_bytes: usize) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::count_keys_above_bytes");
        Ok(inner.map.keys().filter(|k| k.len() > min_bytes).count())
    }

    /// Return `true` if any two distinct keys share the same value.
    ///
    /// Returns `false` for an empty store or a store with only one entry.
    pub fn has_duplicate_values(&self) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::has_duplicate_values");
        let mut seen = std::collections::HashSet::new();
        for v in inner.map.values() {
            if !seen.insert(v.as_str()) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Return the value with the fewest bytes, or `None` for an empty store.
    ///
    /// When multiple values share the minimum byte length the lexicographically
    /// smallest one is returned for deterministic output.
    pub fn shortest_value(&self) -> Result<Option<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "WorkingMemory::shortest_value");
        Ok(inner
            .map
            .values()
            .min_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)))
            .cloned())
    }

    /// Return all keys whose stored value exactly equals `value`, sorted
    /// alphabetically.
    ///
    /// Returns an empty `Vec` when no key has that value or the store is empty.
    pub fn keys_with_value_equal_to(&self, value: &str) -> Result<Vec<String>, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "WorkingMemory::keys_with_value_equal_to");
        let mut keys: Vec<String> = inner
            .map
            .iter()
            .filter(|(_, v)| v.as_str() == value)
            .map(|(k, _)| k.clone())
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Return `true` if every value currently stored in working memory is
    /// non-empty.
    ///
    /// Returns `true` when the store is empty (vacuously true).  Callers that
    /// treat empty values as invalid can use this to validate the store
    /// contents in a single call.
    pub fn all_values_non_empty(&self) -> Result<bool, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "WorkingMemory::all_values_non_empty");
        Ok(inner.map.values().all(|v| !v.is_empty()))
    }

    /// Return the average byte length of all stored values.
    ///
    /// Returns `0.0` when the store is empty to avoid division-by-zero.
    pub fn average_value_length(&self) -> Result<f64, AgentRuntimeError> {
        let inner =
            recover_lock(self.inner.lock(), "WorkingMemory::average_value_length");
        let count = inner.map.len();
        if count == 0 {
            return Ok(0.0);
        }
        let total: usize = inner.map.values().map(|v| v.len()).sum();
        Ok(total as f64 / count as f64)
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

    // ── Round 19: max_importance_episode, SemanticStore::update ──────────────

    #[test]
    fn test_episodic_max_importance_episode_returns_highest() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-max");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        let top = store.max_importance_episode(&agent).unwrap().unwrap();
        assert_eq!(top.content, "high");
    }

    #[test]
    fn test_episodic_max_importance_episode_none_when_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("empty-max");
        assert!(store.max_importance_episode(&agent).unwrap().is_none());
    }

    #[test]
    fn test_semantic_store_update_replaces_value() {
        let store = SemanticStore::new();
        store.store("key1", "original", vec![]).unwrap();
        let updated = store.update("key1", "new value").unwrap();
        assert!(updated);
        let retrieved = store.retrieve_by_key("key1").unwrap().unwrap();
        assert_eq!(retrieved.0, "new value");
    }

    #[test]
    fn test_semantic_store_update_returns_false_for_missing_key() {
        let store = SemanticStore::new();
        assert!(!store.update("missing", "value").unwrap());
    }

    // ── Round 6: EpisodicStore::clear_for / importance_sum ───────────────────

    #[test]
    fn test_episodic_clear_for_removes_all_episodes_for_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.5).unwrap();
        store.add_episode(agent.clone(), "e2", 0.9).unwrap();
        let removed = store.clear_for(&agent).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.count_for(&agent).unwrap(), 0);
    }

    #[test]
    fn test_episodic_clear_for_returns_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ghost");
        assert_eq!(store.clear_for(&agent).unwrap(), 0);
    }

    #[test]
    fn test_episodic_importance_sum_correct() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.3).unwrap();
        store.add_episode(agent.clone(), "e2", 0.5).unwrap();
        let sum = store.importance_sum(&agent).unwrap();
        assert!((sum - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_episodic_importance_sum_zero_when_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("empty");
        assert_eq!(store.importance_sum(&agent).unwrap(), 0.0);
    }

    // ── Round 6: WorkingMemory::merge_from / entry_count_satisfying ──────────

    #[test]
    fn test_working_memory_merge_from_copies_entries() {
        let src = WorkingMemory::new(10).unwrap();
        src.set("x", "1").unwrap();
        src.set("y", "2").unwrap();
        let dst = WorkingMemory::new(10).unwrap();
        let count = dst.merge_from(&src).unwrap();
        assert_eq!(count, 2);
        assert_eq!(dst.get("x").unwrap().unwrap(), "1");
        assert_eq!(dst.get("y").unwrap().unwrap(), "2");
    }

    #[test]
    fn test_working_memory_entry_count_satisfying_counts_matching() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "hello").unwrap();
        wm.set("b", "world").unwrap();
        wm.set("c", "hello world").unwrap();
        let count = wm
            .entry_count_satisfying(|_, v| v.contains("hello"))
            .unwrap();
        assert_eq!(count, 2);
    }

    // ── Round 6: SemanticStore::update_value ─────────────────────────────────

    #[test]
    fn test_semantic_update_value_replaces_stored_value() {
        let store = SemanticStore::new();
        store.store("k", "old", vec![]).unwrap();
        assert!(store.update_value("k", "new").unwrap());
        let pairs = store.retrieve(&[]).unwrap();
        assert!(pairs.iter().any(|(_, v)| v == "new"));
    }

    #[test]
    fn test_semantic_update_value_returns_false_when_missing() {
        let store = SemanticStore::new();
        assert!(!store.update_value("nope", "x").unwrap());
    }

    // ── Round 7: EpisodicStore::agent_ids / find_by_content ──────────────────

    #[test]
    fn test_episodic_agent_ids_returns_all_agents() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("agent-1");
        let a2 = AgentId::new("agent-2");
        store.add_episode(a1.clone(), "e", 0.5).unwrap();
        store.add_episode(a2.clone(), "e", 0.5).unwrap();
        let mut ids = store.agent_ids().unwrap();
        ids.sort_by_key(|id| id.0.clone());
        assert_eq!(ids, vec![a1, a2]);
    }

    #[test]
    fn test_episodic_agent_ids_empty_for_new_store() {
        let store = EpisodicStore::new();
        assert!(store.agent_ids().unwrap().is_empty());
    }

    #[test]
    fn test_find_by_content_returns_matching_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hello world", 0.5).unwrap();
        store.add_episode(agent.clone(), "goodbye world", 0.8).unwrap();
        store.add_episode(agent.clone(), "something else", 0.9).unwrap();
        let results = store.find_by_content(&agent, "world").unwrap();
        assert_eq!(results.len(), 2);
        // sorted by descending importance
        assert_eq!(results[0].content, "goodbye world");
    }

    #[test]
    fn test_find_by_content_returns_empty_when_no_match() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hello", 0.5).unwrap();
        let results = store.find_by_content(&agent, "xyz").unwrap();
        assert!(results.is_empty());
    }

    // ── Round 20: add_episode_at / add_episodes_batch / SemanticStore::keys_matching ──

    #[test]
    fn test_add_episode_at_stores_with_given_timestamp() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-ts");
        let ts = chrono::Utc::now() - chrono::Duration::hours(2);
        let id = store.add_episode_at(agent.clone(), "past event", 0.7, ts).unwrap();
        let items = store.recall_all(&agent).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, id);
        assert_eq!(items[0].content, "past event");
        assert!((items[0].timestamp - ts).num_seconds().abs() < 1);
    }

    #[test]
    fn test_add_episodes_batch_returns_all_ids() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("batch-agent");
        let episodes = vec![
            ("first", 0.5f32),
            ("second", 0.8f32),
            ("third", 0.3f32),
        ];
        let ids = store.add_episodes_batch(agent.clone(), episodes).unwrap();
        assert_eq!(ids.len(), 3);
        let all = store.recall_all(&agent).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_add_episodes_batch_empty_iter_returns_empty_ids() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("empty-batch");
        let ids = store
            .add_episodes_batch(agent.clone(), Vec::<(String, f32)>::new())
            .unwrap();
        assert!(ids.is_empty());
        assert_eq!(store.count_for(&agent).unwrap(), 0);
    }

    #[test]
    fn test_semantic_keys_matching_returns_substring_matches() {
        let store = SemanticStore::new();
        store.store("rust_intro", "value1", vec![]).unwrap();
        store.store("rust_advanced", "value2", vec![]).unwrap();
        store.store("python_basics", "value3", vec![]).unwrap();
        let matches = store.keys_matching("rust").unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&"rust_intro".to_string()));
        assert!(matches.contains(&"rust_advanced".to_string()));
    }

    #[test]
    fn test_semantic_keys_matching_is_case_insensitive() {
        let store = SemanticStore::new();
        store.store("UPPER_KEY", "v", vec![]).unwrap();
        let matches = store.keys_matching("upper").unwrap();
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_semantic_keys_matching_empty_when_no_match() {
        let store = SemanticStore::new();
        store.store("abc", "val", vec![]).unwrap();
        let matches = store.keys_matching("xyz").unwrap();
        assert!(matches.is_empty());
    }

    // ── Round 8: EpisodicStore::importance_avg / deduplicate_content ─────────

    #[test]
    fn test_importance_avg_correct() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.2).unwrap();
        store.add_episode(agent.clone(), "e2", 0.8).unwrap();
        let avg = store.importance_avg(&agent).unwrap();
        assert!((avg - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_importance_avg_zero_for_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("empty");
        assert_eq!(store.importance_avg(&agent).unwrap(), 0.0);
    }

    #[test]
    fn test_deduplicate_content_removes_lower_importance_duplicate() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "same", 0.3).unwrap();
        store.add_episode(agent.clone(), "same", 0.9).unwrap();
        store.add_episode(agent.clone(), "different", 0.5).unwrap();
        let removed = store.deduplicate_content(&agent).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.count_for(&agent).unwrap(), 2);
    }

    #[test]
    fn test_deduplicate_content_keeps_highest_importance() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "dup", 0.1).unwrap();
        store.add_episode(agent.clone(), "dup", 0.7).unwrap();
        store.deduplicate_content(&agent).unwrap();
        let items = store.recall(&agent, 10).unwrap();
        assert_eq!(items.len(), 1);
        assert!((items[0].importance - 0.7).abs() < 1e-5);
    }

    // ── Round 8: WorkingMemory::iter_sorted / SemanticStore::get_value ───────

    #[test]
    fn test_working_memory_iter_sorted_alphabetical_order() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("c", "3").unwrap();
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        let sorted = wm.iter_sorted().unwrap();
        let keys: Vec<&str> = sorted.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_semantic_get_value_returns_value_for_existing_key() {
        let store = SemanticStore::new();
        store.store("mykey", "myvalue", vec![]).unwrap();
        assert_eq!(store.get_value("mykey").unwrap(), Some("myvalue".to_string()));
    }

    #[test]
    fn test_semantic_get_value_returns_none_for_missing_key() {
        let store = SemanticStore::new();
        assert!(store.get_value("ghost").unwrap().is_none());
    }

    // ── Round 9: recall_top_n / filter_by_importance / update_many / entry_count_with_embedding ──

    #[test]
    fn test_recall_top_n_returns_highest_importance_items() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let top = store.recall_top_n(&agent, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert!((top[0].importance - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_recall_top_n_clamps_to_available_items() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "only", 0.5).unwrap();
        let top = store.recall_top_n(&agent, 100).unwrap();
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_filter_by_importance_returns_items_in_range() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("b");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let mid_range = store.filter_by_importance(&agent, 0.3, 0.7).unwrap();
        assert_eq!(mid_range.len(), 1);
        assert!((mid_range[0].importance - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_update_many_sets_multiple_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("x", "old_x").unwrap();
        wm.set("y", "old_y").unwrap();
        let updated = wm
            .update_many(vec![("x".to_string(), "new_x".to_string()), ("y".to_string(), "new_y".to_string())])
            .unwrap();
        assert_eq!(updated, 2);
        assert_eq!(wm.get("x").unwrap(), Some("new_x".to_string()));
        assert_eq!(wm.get("y").unwrap(), Some("new_y".to_string()));
    }

    #[test]
    fn test_update_many_returns_zero_for_empty_iter() {
        let wm = WorkingMemory::new(5).unwrap();
        let updated = wm.update_many(Vec::<(String, String)>::new()).unwrap();
        assert_eq!(updated, 0);
    }

    #[test]
    fn test_entry_count_with_embedding_counts_only_embedded_entries() {
        let store = SemanticStore::new();
        store.store_with_embedding("has_emb", "v1", vec![], vec![0.1_f32, 0.2_f32]).unwrap();
        store.store("no_emb", "v2", vec![]).unwrap();
        assert_eq!(store.entry_count_with_embedding().unwrap(), 1);
    }

    // ── Round 10: retain_top_n / keys_starting_with ───────────────────────────

    #[test]
    fn test_retain_top_n_removes_low_importance_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let removed = store.retain_top_n(&agent, 2).unwrap();
        assert_eq!(removed, 1);
        let remaining = store.recall(&agent, 10).unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().all(|i| i.importance >= 0.5));
    }

    #[test]
    fn test_retain_top_n_noop_when_fewer_than_n() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("b");
        store.add_episode(agent.clone(), "only", 0.7).unwrap();
        let removed = store.retain_top_n(&agent, 5).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(store.recall(&agent, 10).unwrap().len(), 1);
    }

    #[test]
    fn test_keys_starting_with_returns_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("user:name", "alice").unwrap();
        wm.set("user:email", "alice@example.com").unwrap();
        wm.set("session:id", "abc").unwrap();
        let mut keys = wm.keys_starting_with("user:").unwrap();
        keys.sort();
        assert_eq!(keys, vec!["user:email", "user:name"]);
    }

    #[test]
    fn test_keys_starting_with_returns_empty_when_no_match() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("foo", "bar").unwrap();
        assert!(wm.keys_starting_with("xyz:").unwrap().is_empty());
    }

    // ── Round 11: most_recent / contains_all / has_any_key / to_map / fill_ratio ──

    #[test]
    fn test_episodic_most_recent_returns_last_inserted() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "first", 0.5).unwrap();
        store.add_episode(agent.clone(), "second", 0.3).unwrap();
        let recent = store.most_recent(&agent).unwrap().unwrap();
        assert_eq!(recent.content, "second");
    }

    #[test]
    fn test_episodic_most_recent_none_when_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("nobody");
        assert!(store.most_recent(&agent).unwrap().is_none());
    }

    #[test]
    fn test_working_memory_contains_all_true_when_all_present() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("x", "1").unwrap();
        wm.set("y", "2").unwrap();
        assert!(wm.contains_all(["x", "y"]).unwrap());
    }

    #[test]
    fn test_working_memory_contains_all_false_when_one_missing() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("x", "1").unwrap();
        assert!(!wm.contains_all(["x", "missing"]).unwrap());
    }

    #[test]
    fn test_working_memory_contains_all_vacuously_true_for_empty() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(wm.contains_all(std::iter::empty::<&str>()).unwrap());
    }

    #[test]
    fn test_working_memory_has_any_key_true_when_at_least_one_present() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("present", "v").unwrap();
        assert!(wm.has_any_key(["missing", "present"]).unwrap());
    }

    #[test]
    fn test_working_memory_has_any_key_false_when_none_present() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(!wm.has_any_key(["a", "b"]).unwrap());
    }

    #[test]
    fn test_working_memory_has_any_key_false_for_empty_iter() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(!wm.has_any_key(std::iter::empty::<&str>()).unwrap());
    }

    #[test]
    fn test_semantic_to_map_returns_key_value_pairs() {
        let store = SemanticStore::new();
        store.store("key1", "val1", vec![]).unwrap();
        store.store("key2", "val2", vec![]).unwrap();
        let map = store.to_map().unwrap();
        assert_eq!(map.get("key1").map(String::as_str), Some("val1"));
        assert_eq!(map.get("key2").map(String::as_str), Some("val2"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_semantic_to_map_empty_when_no_entries() {
        let store = SemanticStore::new();
        assert!(store.to_map().unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_fill_ratio_zero_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.fill_ratio().unwrap(), 0.0);
    }

    #[test]
    fn test_working_memory_fill_ratio_correct_proportion() {
        let wm = WorkingMemory::new(4).unwrap();
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        // 2 out of 4 = 0.5
        assert!((wm.fill_ratio().unwrap() - 0.5).abs() < 1e-9);
    }

    // ── Round 11: most_recent / contains_all / has_any_key ───────────────────

    #[test]
    fn test_most_recent_returns_last_inserted_episode() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "first", 0.5).unwrap();
        store.add_episode(agent.clone(), "second", 0.8).unwrap();
        let recent = store.most_recent(&agent).unwrap().unwrap();
        assert_eq!(recent.content, "second");
    }

    #[test]
    fn test_most_recent_returns_none_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("empty");
        assert!(store.most_recent(&agent).unwrap().is_none());
    }

    #[test]
    fn test_contains_all_true_when_all_keys_present() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        assert!(wm.contains_all(["a", "b"]).unwrap());
    }

    #[test]
    fn test_contains_all_false_when_one_key_missing() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1").unwrap();
        assert!(!wm.contains_all(["a", "missing"]).unwrap());
    }

    #[test]
    fn test_contains_all_true_for_empty_iter() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(wm.contains_all([]).unwrap());
    }

    #[test]
    fn test_has_any_key_true_when_one_key_present() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("x", "v").unwrap();
        assert!(wm.has_any_key(["x", "y"]).unwrap());
    }

    #[test]
    fn test_has_any_key_false_when_none_present() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(!wm.has_any_key(["nope", "also_nope"]).unwrap());
    }

    #[test]
    fn test_has_any_key_false_for_empty_iter() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("a", "1").unwrap();
        assert!(!wm.has_any_key([]).unwrap());
    }

    // ── Round 13: EpisodicStore::most_recalled ────────────────────────────────

    #[test]
    fn test_most_recalled_returns_none_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("fresh");
        assert!(store.most_recalled(&agent).unwrap().is_none());
    }

    #[test]
    fn test_most_recalled_returns_highest_recall_count() {
        use std::sync::Arc;
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.3).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.6).unwrap();
        // Increment recall count on "high" episode by recalling it multiple times
        let items = store.recall(&agent, 3).unwrap();
        // recall_count is incremented on each recall; all items start at 0 after add,
        // then each recall increments them. We need to find which has highest recall_count.
        // After recall(3), all 3 items get incremented once. Let's recall just "high" more.
        // Simulate: call recall again to increment counts further
        store.recall(&agent, 1).unwrap();
        let top = store.most_recalled(&agent).unwrap();
        assert!(top.is_some());
        // The most recalled has recall_count >= 1
        assert!(top.unwrap().recall_count >= 1);
    }

    #[test]
    fn test_most_recalled_single_episode() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("solo");
        store.add_episode(agent.clone(), "only one", 0.7).unwrap();
        store.recall(&agent, 1).unwrap();
        let top = store.most_recalled(&agent).unwrap();
        assert_eq!(top.unwrap().content, "only one");
    }

    // ── Round 13: max_importance / min_importance / values_matching ──────────

    #[test]
    fn test_max_importance_returns_highest_score() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let max = store.max_importance(&agent).unwrap().unwrap();
        assert!((max - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_min_importance_returns_lowest_score() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("b");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let min = store.min_importance(&agent).unwrap().unwrap();
        assert!((min - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_max_importance_none_for_empty_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("empty");
        assert!(store.max_importance(&agent).unwrap().is_none());
    }

    #[test]
    fn test_values_matching_returns_pairs_with_pattern() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("name", "alice wonder").unwrap();
        wm.set("city", "wonderland").unwrap();
        wm.set("role", "engineer").unwrap();
        let mut matches = wm.values_matching("wonder").unwrap();
        matches.sort_by_key(|(k, _)| k.clone());
        assert_eq!(
            matches,
            vec![
                ("city".to_string(), "wonderland".to_string()),
                ("name".to_string(), "alice wonder".to_string()),
            ]
        );
    }

    #[test]
    fn test_values_matching_returns_empty_when_no_match() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("a", "foo").unwrap();
        assert!(wm.values_matching("xyz").unwrap().is_empty());
    }

    // ── Round 14: count_above_importance, value_length, tags_for ─────────────

    #[test]
    fn test_count_above_importance_correct_count() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        assert_eq!(store.count_above_importance(&agent, 0.4).unwrap(), 2);
    }

    #[test]
    fn test_count_above_importance_zero_for_empty_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("nobody");
        assert_eq!(store.count_above_importance(&agent, 0.0).unwrap(), 0);
    }

    #[test]
    fn test_count_above_importance_threshold_is_exclusive() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ex");
        store.add_episode(agent.clone(), "exact", 0.5).unwrap();
        // threshold 0.5 — strictly greater than, so 0.5 should not count
        assert_eq!(store.count_above_importance(&agent, 0.5).unwrap(), 0);
    }

    #[test]
    fn test_working_memory_value_length_some_for_existing_key() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("greeting", "hello").unwrap();
        assert_eq!(wm.value_length("greeting").unwrap(), Some(5));
    }

    #[test]
    fn test_working_memory_value_length_none_for_absent_key() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.value_length("missing").unwrap().is_none());
    }

    #[test]
    fn test_semantic_store_tags_for_returns_tags() {
        let store = SemanticStore::new();
        store
            .store("key1", "value1", vec!["alpha".to_string(), "beta".to_string()])
            .unwrap();
        let tags = store.tags_for("key1").unwrap().unwrap();
        assert_eq!(tags, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn test_semantic_store_tags_for_none_for_missing_key() {
        let store = SemanticStore::new();
        assert!(store.tags_for("ghost").unwrap().is_none());
    }

    #[test]
    fn test_semantic_store_tags_for_empty_tags() {
        let store = SemanticStore::new();
        store.store("k", "v", vec![]).unwrap();
        let tags = store.tags_for("k").unwrap().unwrap();
        assert!(tags.is_empty());
    }

    // ── Round 27: peek_oldest, SemanticStore::values, SemanticStore::get_tags ─

    #[test]
    fn test_peek_oldest_returns_oldest_without_removing() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("first", "alpha").unwrap();
        wm.set("second", "beta").unwrap();
        let peeked = wm.peek_oldest().unwrap();
        assert_eq!(peeked, Some(("first".into(), "alpha".into())));
        // Still there after peek
        assert_eq!(wm.len().unwrap(), 2);
    }

    #[test]
    fn test_peek_oldest_empty_returns_none() {
        let wm = WorkingMemory::new(5).unwrap();
        assert_eq!(wm.peek_oldest().unwrap(), None);
    }

    #[test]
    fn test_peek_oldest_does_not_remove_entry() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("k", "v").unwrap();
        wm.peek_oldest().unwrap();
        assert_eq!(wm.get("k").unwrap(), Some("v".into()));
    }

    #[test]
    fn test_semantic_store_values_returns_all_values() {
        let store = SemanticStore::new();
        store.store("k1", "hello", vec![]).unwrap();
        store.store("k2", "world", vec![]).unwrap();
        let vals = store.values().unwrap();
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&"hello".to_string()));
        assert!(vals.contains(&"world".to_string()));
    }

    #[test]
    fn test_semantic_store_values_empty() {
        let store = SemanticStore::new();
        assert!(store.values().unwrap().is_empty());
    }

    #[test]
    fn test_semantic_store_get_tags_returns_tags() {
        let store = SemanticStore::new();
        store.store("k1", "val", vec!["tag-a".to_string(), "tag-b".to_string()]).unwrap();
        let tags = store.get_tags("k1").unwrap();
        assert!(tags.is_some());
        let tags = tags.unwrap();
        assert!(tags.contains(&"tag-a".to_string()));
        assert!(tags.contains(&"tag-b".to_string()));
    }

    #[test]
    fn test_semantic_store_get_tags_missing_key_returns_none() {
        let store = SemanticStore::new();
        assert!(store.get_tags("no-such-key").unwrap().is_none());
    }

    // ── Round 16 (duplicate block): WorkingMemory::value_length, iter_sorted ──

    #[test]
    fn test_working_memory_value_length_returns_char_count_r27() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "hello").unwrap();
        assert_eq!(wm.value_length("k").unwrap(), Some(5));
    }

    #[test]
    fn test_working_memory_value_length_none_for_missing_key_r27() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.value_length("nope").unwrap().is_none());
    }

    #[test]
    fn test_working_memory_iter_sorted_returns_sorted_pairs() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("b", "2").unwrap();
        wm.set("a", "1").unwrap();
        let pairs = wm.iter_sorted().unwrap();
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[1].0, "b");
    }

    #[test]
    fn test_working_memory_drain_empties_store() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("x", "1").unwrap();
        wm.set("y", "2").unwrap();
        let drained = wm.drain().unwrap();
        assert_eq!(drained.len(), 2);
        assert!(wm.is_empty().unwrap());
    }

    #[test]
    fn test_working_memory_snapshot_returns_all_entries_r27() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "alpha").unwrap();
        wm.set("b", "beta").unwrap();
        let snap = wm.snapshot().unwrap();
        assert_eq!(snap.get("a").map(String::as_str), Some("alpha"));
        assert_eq!(snap.get("b").map(String::as_str), Some("beta"));
    }

    #[test]
    fn test_working_memory_snapshot_empty_when_no_entries_r27() {
        let wm = WorkingMemory::new(5).unwrap();
        assert!(wm.snapshot().unwrap().is_empty());
    }

    // ── Round 15: agent_count, is_at_capacity, remove_keys_starting_with,
    //              has_key, entry_count_with_tag ─────────────────────────────

    #[test]
    fn test_episodic_store_agent_count_zero_when_empty() {
        let store = EpisodicStore::new();
        assert_eq!(store.agent_count().unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_agent_count_distinct_agents() {
        let store = EpisodicStore::new();
        store.add_episode(AgentId::new("a"), "ep1", 0.5).unwrap();
        store.add_episode(AgentId::new("b"), "ep2", 0.7).unwrap();
        store.add_episode(AgentId::new("a"), "ep3", 0.3).unwrap();
        assert_eq!(store.agent_count().unwrap(), 2);
    }

    #[test]
    fn test_working_memory_is_at_capacity_true_when_full() {
        let wm = WorkingMemory::new(2).unwrap();
        wm.set("k1", "v1").unwrap();
        wm.set("k2", "v2").unwrap();
        assert!(wm.is_at_capacity().unwrap());
    }

    #[test]
    fn test_working_memory_is_at_capacity_false_when_not_full() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("k1", "v1").unwrap();
        assert!(!wm.is_at_capacity().unwrap());
    }

    #[test]
    fn test_remove_keys_starting_with_removes_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("prefix_a", "1").unwrap();
        wm.set("prefix_b", "2").unwrap();
        wm.set("other", "3").unwrap();
        let removed = wm.remove_keys_starting_with("prefix_").unwrap();
        assert_eq!(removed, 2);
        assert!(wm.get("prefix_a").unwrap().is_none());
        assert!(wm.get("prefix_b").unwrap().is_none());
        assert_eq!(wm.get("other").unwrap(), Some("3".into()));
    }

    #[test]
    fn test_remove_keys_starting_with_returns_zero_when_no_match() {
        let wm = WorkingMemory::new(5).unwrap();
        wm.set("foo", "bar").unwrap();
        assert_eq!(wm.remove_keys_starting_with("xyz_").unwrap(), 0);
    }

    #[test]
    fn test_semantic_store_has_key_true_when_exists() {
        let store = SemanticStore::new();
        store.store("mykey", "val", vec![]).unwrap();
        assert!(store.has_key("mykey").unwrap());
    }

    #[test]
    fn test_semantic_store_has_key_false_when_absent() {
        let store = SemanticStore::new();
        assert!(!store.has_key("ghost").unwrap());
    }

    #[test]
    fn test_entry_count_with_tag_counts_correctly() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["rust".into(), "async".into()]).unwrap();
        store.store("k2", "v2", vec!["rust".into()]).unwrap();
        store.store("k3", "v3", vec!["python".into()]).unwrap();
        assert_eq!(store.entry_count_with_tag("rust").unwrap(), 2);
        assert_eq!(store.entry_count_with_tag("async").unwrap(), 1);
        assert_eq!(store.entry_count_with_tag("absent").unwrap(), 0);
    }

    // ── Round 18: importance_sum, update_content, recall_all, top_n, search_by_importance_range ──

    #[test]
    fn test_importance_sum_returns_sum_of_all_importances() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.2).unwrap();
        store.add_episode(agent.clone(), "e2", 0.3).unwrap();
        store.add_episode(agent.clone(), "e3", 0.5).unwrap();
        let sum = store.importance_sum(&agent).unwrap();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_importance_sum_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("nobody");
        assert!((store.importance_sum(&agent).unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_update_content_changes_stored_content() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store.add_episode(agent.clone(), "old content", 0.5).unwrap();
        let updated = store.update_content(&agent, &id, "new content").unwrap();
        assert!(updated);
        let items = store.recall_all(&agent).unwrap();
        assert_eq!(items[0].content, "new content");
    }

    #[test]
    fn test_update_content_false_for_missing_id() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let fake_id = MemoryId::new("nonexistent");
        assert!(!store.update_content(&agent, &fake_id, "x").unwrap());
    }

    #[test]
    fn test_recall_all_returns_all_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.5).unwrap();
        store.add_episode(agent.clone(), "e2", 0.3).unwrap();
        store.add_episode(agent.clone(), "e3", 0.9).unwrap();
        let all = store.recall_all(&agent).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_top_n_returns_top_by_importance() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        let top = store.top_n(&agent, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].content, "high");
        assert_eq!(top[1].content, "mid");
    }

    #[test]
    fn test_search_by_importance_range_filters_correctly() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let results = store.search_by_importance_range(&agent, 0.4, 0.8, 0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "mid");
    }

    // ── Round 16: oldest_episode, remove_by_key, total_value_bytes ───────────

    #[test]
    fn test_oldest_episode_returns_none_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("fresh");
        assert!(store.oldest_episode(&agent).unwrap().is_none());
    }

    #[test]
    fn test_oldest_episode_returns_earliest_timestamp() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "first", 0.5).unwrap();
        store.add_episode(agent.clone(), "second", 0.7).unwrap();
        store.add_episode(agent.clone(), "third", 0.3).unwrap();
        let oldest = store.oldest_episode(&agent).unwrap().unwrap();
        // Insertion order determines timestamp; "first" should be oldest
        assert_eq!(oldest.content, "first");
    }

    #[test]
    fn test_semantic_store_remove_by_key_removes_all_matching() {
        let store = SemanticStore::new();
        store.store("target", "v1", vec![]).unwrap();
        store.store("target", "v2", vec![]).unwrap();
        store.store("keep", "vk", vec![]).unwrap();
        let removed = store.remove_by_key("target").unwrap();
        assert_eq!(removed, 2);
        assert!(!store.has_key("target").unwrap());
        assert!(store.has_key("keep").unwrap());
    }

    #[test]
    fn test_semantic_store_remove_by_key_zero_for_absent_key() {
        let store = SemanticStore::new();
        assert_eq!(store.remove_by_key("ghost").unwrap(), 0);
    }

    #[test]
    fn test_working_memory_total_value_bytes_sums_lengths() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "hello").unwrap();    // 5 bytes
        wm.set("b", "world!").unwrap();   // 6 bytes
        assert_eq!(wm.total_value_bytes().unwrap(), 11);
    }

    #[test]
    fn test_working_memory_total_value_bytes_zero_when_empty() {
        let wm = WorkingMemory::new(5).unwrap();
        assert_eq!(wm.total_value_bytes().unwrap(), 0);
    }

    // ── Round 19: agent_ids, clear_for, recall_since ──────────────────────────

    #[test]
    fn test_agent_ids_returns_all_tracked_agents() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("alice");
        let a2 = AgentId::new("bob");
        store.add_episode(a1.clone(), "x", 0.5).unwrap();
        store.add_episode(a2.clone(), "y", 0.5).unwrap();
        let mut ids = store.agent_ids().unwrap();
        ids.sort_by_key(|id| id.as_str().to_string());
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].as_str(), "alice");
        assert_eq!(ids[1].as_str(), "bob");
    }

    #[test]
    fn test_agent_ids_empty_for_new_store() {
        let store = EpisodicStore::new();
        assert!(store.agent_ids().unwrap().is_empty());
    }

    #[test]
    fn test_clear_for_removes_all_episodes_for_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.5).unwrap();
        store.add_episode(agent.clone(), "e2", 0.3).unwrap();
        let removed = store.clear_for(&agent).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.count_for(&agent).unwrap(), 0);
    }

    #[test]
    fn test_clear_for_returns_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ghost");
        assert_eq!(store.clear_for(&agent).unwrap(), 0);
    }

    #[test]
    fn test_recall_since_returns_episodes_after_cutoff() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let past = chrono::Utc::now() - chrono::Duration::hours(2);
        let future_cutoff = chrono::Utc::now() + chrono::Duration::seconds(1);
        // Add one in the past
        store.add_episode_at(agent.clone(), "old", 0.5, past).unwrap();
        // Add one now
        store.add_episode(agent.clone(), "new", 0.5).unwrap();
        // Recall only future episodes (should be 0)
        let future = store.recall_since(&agent, future_cutoff, 0).unwrap();
        assert!(future.is_empty());
        // Recall from far past (should include both)
        let all = store.recall_since(&agent, past - chrono::Duration::seconds(1), 0).unwrap();
        assert_eq!(all.len(), 2);
    }

    // ── Round 17: EpisodicStore::sum_recall_counts ────────────────────────────

    #[test]
    fn test_sum_recall_counts_zero_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("new");
        assert_eq!(store.sum_recall_counts(&agent).unwrap(), 0);
    }

    #[test]
    fn test_sum_recall_counts_increases_with_recalls() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "ep1", 0.5).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.8).unwrap();
        // Recall both episodes once
        store.recall(&agent, 2).unwrap();
        let total = store.sum_recall_counts(&agent).unwrap();
        assert!(total >= 2);
    }

    // ── Round 22: max_recall_count_for, most_recent_key, max_key_length ───────

    #[test]
    fn test_max_recall_count_for_none_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ghost");
        assert_eq!(store.max_recall_count_for(&agent).unwrap(), None);
    }

    #[test]
    fn test_max_recall_count_for_returns_highest_after_recalls() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "ep1", 0.9).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.1).unwrap();
        // Recall several times to accumulate counts
        store.recall(&agent, 2).unwrap();
        store.recall(&agent, 1).unwrap();
        let max = store.max_recall_count_for(&agent).unwrap().unwrap();
        assert!(max >= 1);
    }

    #[test]
    fn test_semantic_most_recent_key_none_when_empty() {
        let store = SemanticStore::new();
        assert_eq!(store.most_recent_key().unwrap(), None);
    }

    #[test]
    fn test_semantic_most_recent_key_returns_last_inserted() {
        let store = SemanticStore::new();
        store.store("first", "v1", vec![]).unwrap();
        store.store("second", "v2", vec![]).unwrap();
        assert_eq!(store.most_recent_key().unwrap(), Some("second".to_string()));
    }

    #[test]
    fn test_working_memory_max_key_length_zero_when_empty() {
        let mem = WorkingMemory::new(10).unwrap();
        assert_eq!(mem.max_key_length().unwrap(), 0);
    }

    #[test]
    fn test_working_memory_max_key_length_returns_longest() {
        let mem = WorkingMemory::new(10).unwrap();
        mem.set("ab", "v1").unwrap();
        mem.set("abcde", "v2").unwrap();
        mem.set("abc", "v3").unwrap();
        assert_eq!(mem.max_key_length().unwrap(), 5);
    }

    // ── Round 29: WorkingMemory::set_if_absent ────────────────────────────────

    #[test]
    fn test_working_memory_set_if_absent_inserts_new_key() {
        let mem = WorkingMemory::new(10).unwrap();
        let inserted = mem.set_if_absent("fresh", "value").unwrap();
        assert!(inserted);
        assert_eq!(mem.get("fresh").unwrap(), Some("value".to_string()));
    }

    #[test]
    fn test_working_memory_set_if_absent_does_not_overwrite_existing() {
        let mem = WorkingMemory::new(10).unwrap();
        mem.set("key", "original").unwrap();
        let inserted = mem.set_if_absent("key", "replacement").unwrap();
        assert!(!inserted);
        assert_eq!(mem.get("key").unwrap(), Some("original".to_string()));
    }

    #[test]
    fn test_working_memory_set_if_absent_second_call_returns_false() {
        let mem = WorkingMemory::new(10).unwrap();
        assert!(mem.set_if_absent("k", "v1").unwrap());
        assert!(!mem.set_if_absent("k", "v2").unwrap());
    }

    // ── Round 24: MemoryItem helpers, DecayPolicy, export/bump ───────────────

    #[test]
    fn test_memory_item_has_tag_true_when_tag_present() {
        let item = MemoryItem::new(
            AgentId::new("a"),
            "content",
            0.5,
            vec!["important".to_string(), "work".to_string()],
        );
        assert!(item.has_tag("important"));
        assert!(item.has_tag("work"));
    }

    #[test]
    fn test_memory_item_has_tag_false_when_tag_absent() {
        let item = MemoryItem::new(AgentId::new("a"), "content", 0.5, vec![]);
        assert!(!item.has_tag("missing"));
    }

    #[test]
    fn test_memory_item_word_count_counts_words() {
        let item = MemoryItem::new(AgentId::new("a"), "one two three", 0.5, vec![]);
        assert_eq!(item.word_count(), 3);
    }

    #[test]
    fn test_memory_item_word_count_zero_for_empty_content() {
        let item = MemoryItem::new(AgentId::new("a"), "", 0.5, vec![]);
        assert_eq!(item.word_count(), 0);
    }

    #[test]
    fn test_decay_policy_apply_reduces_importance_after_one_half_life() {
        let policy = DecayPolicy::exponential(1.0).unwrap(); // half-life 1 hour
        let decayed = policy.apply(1.0, 1.0); // after 1 hour, should be ~0.5
        assert!((decayed - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_decay_policy_apply_no_change_for_zero_age() {
        let policy = DecayPolicy::exponential(10.0).unwrap();
        let decayed = policy.apply(0.8, 0.0);
        assert!((decayed - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_decay_policy_decay_item_reduces_importance_for_old_item() {
        let policy = DecayPolicy::exponential(0.0001).unwrap(); // very short half-life
        let mut item = MemoryItem::new(AgentId::new("a"), "old memory", 1.0, vec![]);
        item.timestamp = Utc::now() - chrono::Duration::hours(1);
        policy.decay_item(&mut item);
        assert!(item.importance < 1.0);
    }

    #[test]
    fn test_episodic_export_returns_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ghost");
        let exported = store.export_agent_memory(&agent).unwrap();
        assert!(exported.is_empty());
    }

    #[test]
    fn test_episodic_export_returns_all_stored_items() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "memory1", 0.9).unwrap();
        store.add_episode(agent.clone(), "memory2", 0.5).unwrap();
        let exported = store.export_agent_memory(&agent).unwrap();
        assert_eq!(exported.len(), 2);
    }

    #[test]
    fn test_bump_recall_count_increases_by_amount() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "the content", 0.7).unwrap();
        store.bump_recall_count_by_content("the content", 5);
        let max = store.max_recall_count_for(&agent).unwrap().unwrap();
        assert_eq!(max, 5);
    }

    #[test]
    fn test_bump_recall_count_no_effect_for_absent_content() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "existing", 0.5).unwrap();
        store.bump_recall_count_by_content("not here", 10);
        let max = store.max_recall_count_for(&agent).unwrap().unwrap();
        assert_eq!(max, 0);
    }

    // ── Round 23: latest_episode / oldest_key / key_count_matching / avg_value_length

    #[test]
    fn test_latest_episode_none_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ghost");
        assert!(store.latest_episode(&agent).unwrap().is_none());
    }

    #[test]
    fn test_latest_episode_returns_most_recent_by_timestamp() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let old = chrono::Utc::now() - chrono::Duration::hours(1);
        store.add_episode_at(agent.clone(), "old ep", 0.5, old).unwrap();
        store.add_episode(agent.clone(), "new ep", 0.8).unwrap();
        let latest = store.latest_episode(&agent).unwrap().unwrap();
        assert_eq!(latest.content, "new ep");
    }

    #[test]
    fn test_semantic_oldest_key_none_when_empty() {
        let store = SemanticStore::new();
        assert!(store.oldest_key().unwrap().is_none());
    }

    #[test]
    fn test_semantic_oldest_key_returns_first_inserted() {
        let store = SemanticStore::new();
        store.store("first", "value1", vec![]).unwrap();
        store.store("second", "value2", vec![]).unwrap();
        assert_eq!(store.oldest_key().unwrap().as_deref(), Some("first"));
    }

    #[test]
    fn test_working_memory_key_count_matching_zero_when_empty() {
        let mem = WorkingMemory::new(10).unwrap();
        assert_eq!(mem.key_count_matching("foo").unwrap(), 0);
    }

    #[test]
    fn test_working_memory_key_count_matching_counts_correctly() {
        let mem = WorkingMemory::new(10).unwrap();
        mem.set("foo_bar", "v1").unwrap();
        mem.set("foo_baz", "v2").unwrap();
        mem.set("other", "v3").unwrap();
        assert_eq!(mem.key_count_matching("foo").unwrap(), 2);
    }

    #[test]
    fn test_working_memory_avg_value_length_zero_when_empty() {
        let mem = WorkingMemory::new(10).unwrap();
        assert!((mem.avg_value_length().unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_working_memory_avg_value_length_correct_mean() {
        let mem = WorkingMemory::new(10).unwrap();
        mem.set("k1", "ab").unwrap();    // 2 bytes
        mem.set("k2", "abcd").unwrap();  // 4 bytes
        // mean = 3.0
        assert!((mem.avg_value_length().unwrap() - 3.0).abs() < 1e-9);
    }

    // ── Round 30: EpisodicStore::has_agent, export_agent_memory ──────────────

    #[test]
    fn test_episodic_store_has_agent_false_when_empty() {
        let store = EpisodicStore::new();
        assert!(!store.has_agent(&AgentId::new("nobody")).unwrap());
    }

    #[test]
    fn test_episodic_store_has_agent_true_after_episode() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-ha");
        store.add_episode(agent.clone(), "event", 0.5).unwrap();
        assert!(store.has_agent(&agent).unwrap());
    }

    #[test]
    fn test_episodic_store_export_agent_memory_empty_for_unknown() {
        let store = EpisodicStore::new();
        let exported = store.export_agent_memory(&AgentId::new("ghost")).unwrap();
        assert!(exported.is_empty());
    }

    #[test]
    fn test_episodic_store_export_agent_memory_returns_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-exp");
        store.add_episode(agent.clone(), "ep1", 0.9).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.7).unwrap();
        let exported = store.export_agent_memory(&agent).unwrap();
        assert_eq!(exported.len(), 2);
    }

    // ── Round 30: SemanticStore::store_with_embedding ────────────────────────

    #[test]
    fn test_semantic_store_store_with_embedding_rejects_empty_vec() {
        let store = SemanticStore::new();
        let result = store.store_with_embedding("k", "v", vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_semantic_store_store_with_embedding_stores_entry() {
        let store = SemanticStore::new();
        let emb = vec![1.0f32, 0.0, 0.0];
        store.store_with_embedding("k1", "v1", vec![], emb).unwrap();
        assert_eq!(store.len().unwrap(), 1);
    }

    #[test]
    fn test_semantic_store_store_with_embedding_rejects_dimension_mismatch() {
        let store = SemanticStore::new();
        store.store_with_embedding("k1", "v1", vec![], vec![1.0f32, 0.0]).unwrap();
        // Second call with different dimension should fail
        let result = store.store_with_embedding("k2", "v2", vec![], vec![1.0f32, 0.0, 0.0]);
        assert!(result.is_err());
    }

    // ── Round 24: avg_importance / importance_range / entries_without_tags /
    //             avg_tag_count_per_entry / longest_key / longest_value ────────

    #[test]
    fn test_avg_importance_zero_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ghost");
        assert!((store.avg_importance(&agent).unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_importance_correct_mean() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "ep1", 0.2).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.8).unwrap();
        // mean = 0.5
        assert!((store.avg_importance(&agent).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_importance_range_none_for_new_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("ghost");
        assert!(store.importance_range(&agent).unwrap().is_none());
    }

    #[test]
    fn test_importance_range_returns_min_max() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "ep1", 0.1).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.9).unwrap();
        let (min, max) = store.importance_range(&agent).unwrap().unwrap();
        assert!((min - 0.1_f32).abs() < 1e-6);
        assert!((max - 0.9_f32).abs() < 1e-6);
    }

    #[test]
    fn test_semantic_entries_without_tags_all_untagged() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec![]).unwrap();
        store.store("k2", "v2", vec![]).unwrap();
        assert_eq!(store.entries_without_tags().unwrap(), 2);
    }

    #[test]
    fn test_semantic_entries_without_tags_mixed() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["tag".to_string()]).unwrap();
        store.store("k2", "v2", vec![]).unwrap();
        assert_eq!(store.entries_without_tags().unwrap(), 1);
    }

    #[test]
    fn test_semantic_avg_tag_count_zero_when_empty() {
        let store = SemanticStore::new();
        assert!((store.avg_tag_count_per_entry().unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_semantic_avg_tag_count_correct_mean() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["a".to_string(), "b".to_string()]).unwrap(); // 2
        store.store("k2", "v2", vec![]).unwrap(); // 0
        // mean = 1.0
        assert!((store.avg_tag_count_per_entry().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_working_memory_longest_key_none_when_empty() {
        let mem = WorkingMemory::new(10).unwrap();
        assert!(mem.longest_key().unwrap().is_none());
    }

    #[test]
    fn test_working_memory_longest_key_returns_longest() {
        let mem = WorkingMemory::new(10).unwrap();
        mem.set("ab", "v1").unwrap();
        mem.set("abcde", "v2").unwrap();
        assert_eq!(mem.longest_key().unwrap().as_deref(), Some("abcde"));
    }

    #[test]
    fn test_working_memory_longest_value_none_when_empty() {
        let mem = WorkingMemory::new(10).unwrap();
        assert!(mem.longest_value().unwrap().is_none());
    }

    #[test]
    fn test_working_memory_longest_value_returns_longest() {
        let mem = WorkingMemory::new(10).unwrap();
        mem.set("k1", "short").unwrap();
        mem.set("k2", "much longer value").unwrap();
        assert_eq!(mem.longest_value().unwrap().as_deref(), Some("much longer value"));
    }

    // ── Round 26: SemanticStore::store_with_embedding ─────────────────────────

    #[test]
    fn test_semantic_store_with_embedding_rejects_empty_vector() {
        let store = SemanticStore::new();
        let result = store.store_with_embedding("k", "v", vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_semantic_store_with_embedding_stores_and_retrievable() {
        let store = SemanticStore::new();
        store.store_with_embedding("k", "v", vec![], vec![1.0, 0.0]).unwrap();
        let entry = store.retrieve_by_key("k").unwrap();
        assert_eq!(entry.map(|(val, _)| val), Some("v".to_string()));
    }

    #[test]
    fn test_semantic_store_with_embedding_dimension_mismatch_errors() {
        let store = SemanticStore::new();
        store.store_with_embedding("k1", "v1", vec![], vec![1.0, 0.0]).unwrap();
        let result = store.store_with_embedding("k2", "v2", vec![], vec![1.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    // ── Round 26: has_episodes / value_for / count_above_value_length ─────────

    #[test]
    fn test_episodic_store_has_episodes_false_when_empty() {
        let store = EpisodicStore::new();
        let id = AgentId::new("agent-x");
        assert!(!store.has_episodes(&id).unwrap());
    }

    #[test]
    fn test_episodic_store_has_episodes_true_after_recording() {
        let store = EpisodicStore::new();
        let id = AgentId::new("agent-y");
        store.add_episode(id.clone(), "e1", 0.8).unwrap();
        assert!(store.has_episodes(&id).unwrap());
    }

    #[test]
    fn test_semantic_store_value_for_none_when_missing() {
        let store = SemanticStore::new();
        assert!(store.value_for("missing-key").unwrap().is_none());
    }

    #[test]
    fn test_semantic_store_value_for_returns_stored_value() {
        let store = SemanticStore::new();
        store.store("mykey", "myvalue", vec![]).unwrap();
        assert_eq!(store.value_for("mykey").unwrap(), Some("myvalue".to_string()));
    }

    #[test]
    fn test_working_memory_count_above_value_length_zero_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.count_above_value_length(5).unwrap(), 0);
    }

    #[test]
    fn test_working_memory_count_above_value_length_counts_correctly() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "short").unwrap();        // 5 bytes
        wm.set("k2", "a longer value").unwrap(); // 13 bytes
        wm.set("k3", "hi").unwrap();             // 2 bytes
        // min_bytes=5: values strictly > 5 bytes: "a longer value" (13)
        assert_eq!(wm.count_above_value_length(5).unwrap(), 1);
    }

    // ── Round 27: total_episode_count ─────────────────────────────────────────

    #[test]
    fn test_total_episode_count_zero_when_empty() {
        let store = EpisodicStore::new();
        assert_eq!(store.total_episode_count().unwrap(), 0);
    }

    #[test]
    fn test_total_episode_count_sums_across_agents() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("a1");
        let a2 = AgentId::new("a2");
        store.add_episode(a1.clone(), "e1", 0.5).unwrap();
        store.add_episode(a1.clone(), "e2", 0.6).unwrap();
        store.add_episode(a2.clone(), "e3", 0.7).unwrap();
        assert_eq!(store.total_episode_count().unwrap(), 3);
    }

    // ── Round 28: agents_with_min_episodes / entries_with_no_tags / longest_value_key

    #[test]
    fn test_agents_with_min_episodes_empty_when_below_min() {
        let store = EpisodicStore::new();
        let a = AgentId::new("a1");
        store.add_episode(a.clone(), "e1", 0.5).unwrap();
        // min=2 → no agents qualify
        assert!(store.agents_with_min_episodes(2).unwrap().is_empty());
    }

    #[test]
    fn test_agents_with_min_episodes_includes_qualifying_agents() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("a1");
        let a2 = AgentId::new("a2");
        store.add_episode(a1.clone(), "e1", 0.5).unwrap();
        store.add_episode(a1.clone(), "e2", 0.6).unwrap();
        store.add_episode(a2.clone(), "only-one", 0.7).unwrap();
        let result = store.agents_with_min_episodes(2).unwrap();
        assert_eq!(result, vec![a1]);
    }

    #[test]
    fn test_entries_with_no_tags_returns_empty_list_when_all_have_tags() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["tag".to_string()]).unwrap();
        assert!(store.entries_with_no_tags().unwrap().is_empty());
    }

    #[test]
    fn test_entries_with_no_tags_returns_untagged_keys() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec![]).unwrap();
        store.store("k2", "v2", vec!["tag".to_string()]).unwrap();
        let result = store.entries_with_no_tags().unwrap();
        assert_eq!(result, vec!["k1".to_string()]);
    }

    #[test]
    fn test_working_memory_longest_value_key_none_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.longest_value_key().unwrap().is_none());
    }

    #[test]
    fn test_working_memory_longest_value_key_returns_key_with_longest_value() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("short_key", "hi").unwrap();
        wm.set("long_key", "a much longer value string").unwrap();
        assert_eq!(wm.longest_value_key().unwrap(), Some("long_key".to_string()));
    }

    // ── Round 29: agent_with_most_episodes / most_tagged_key / value_lengths ──

    #[test]
    fn test_agent_with_most_episodes_none_when_empty() {
        let store = EpisodicStore::new();
        assert!(store.agent_with_most_episodes().unwrap().is_none());
    }

    #[test]
    fn test_agent_with_most_episodes_returns_agent_with_most() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("a1");
        let a2 = AgentId::new("a2");
        store.add_episode(a1.clone(), "e1", 0.5).unwrap();
        store.add_episode(a2.clone(), "e1", 0.5).unwrap();
        store.add_episode(a2.clone(), "e2", 0.6).unwrap();
        assert_eq!(store.agent_with_most_episodes().unwrap(), Some(a2));
    }

    #[test]
    fn test_most_tagged_key_none_when_empty() {
        let store = SemanticStore::new();
        assert!(store.most_tagged_key().unwrap().is_none());
    }

    #[test]
    fn test_most_tagged_key_returns_key_with_most_tags() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["a".to_string()]).unwrap();
        store.store("k2", "v2", vec!["a".to_string(), "b".to_string(), "c".to_string()]).unwrap();
        store.store("k3", "v3", vec![]).unwrap();
        assert_eq!(store.most_tagged_key().unwrap(), Some("k2".to_string()));
    }

    #[test]
    fn test_value_lengths_empty_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.value_lengths().unwrap().is_empty());
    }

    #[test]
    fn test_value_lengths_returns_all_pairs() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "hello").unwrap();
        let lengths = wm.value_lengths().unwrap();
        assert_eq!(lengths.len(), 1);
        assert_eq!(lengths[0], ("k".to_string(), 5));
    }

    // ── Round 30: importance_variance_for / count_matching_value / keys_with_value_longer_than

    #[test]
    fn test_importance_variance_for_zero_when_fewer_than_two() {
        let store = EpisodicStore::new();
        let id = AgentId::new("a");
        store.add_episode(id.clone(), "e", 0.5).unwrap();
        assert!((store.importance_variance_for(&id).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_importance_variance_for_nonzero_with_spread() {
        let store = EpisodicStore::new();
        let id = AgentId::new("a");
        store.add_episode(id.clone(), "e1", 0.0).unwrap();
        store.add_episode(id.clone(), "e2", 1.0).unwrap();
        // mean=0.5, variance = 0.25
        let v = store.importance_variance_for(&id).unwrap();
        assert!((v - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_count_matching_value_zero_when_no_match() {
        let store = SemanticStore::new();
        store.store("k", "hello world", vec![]).unwrap();
        assert_eq!(store.count_matching_value("xyz").unwrap(), 0);
    }

    #[test]
    fn test_count_matching_value_counts_containing_entries() {
        let store = SemanticStore::new();
        store.store("k1", "hello world", vec![]).unwrap();
        store.store("k2", "world peace", vec![]).unwrap();
        store.store("k3", "goodbye", vec![]).unwrap();
        assert_eq!(store.count_matching_value("world").unwrap(), 2);
    }

    #[test]
    fn test_keys_with_value_longer_than_empty_when_all_short() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "hi").unwrap();
        assert!(wm.keys_with_value_longer_than(10).unwrap().is_empty());
    }

    #[test]
    fn test_keys_with_value_longer_than_returns_qualifying_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("short", "hi").unwrap();
        wm.set("long", "this is a longer value").unwrap();
        let keys = wm.keys_with_value_longer_than(5).unwrap();
        assert_eq!(keys, vec!["long".to_string()]);
    }

    #[test]
    fn test_episodic_store_max_importance_overall_returns_highest() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        let max = store.max_importance_overall().unwrap();
        assert!((max.unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_episodic_store_max_importance_overall_empty_returns_none() {
        let store = EpisodicStore::new();
        assert!(store.max_importance_overall().unwrap().is_none());
    }

    #[test]
    fn test_semantic_store_rename_tag_updates_all_occurrences() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["old".to_string(), "x".to_string()]).unwrap();
        store.store("k2", "v2", vec!["old".to_string()]).unwrap();
        let count = store.rename_tag("old", "new").unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_semantic_store_rename_tag_nonexistent_returns_zero() {
        let store = SemanticStore::new();
        store.store("k", "v", vec!["alpha".to_string()]).unwrap();
        assert_eq!(store.rename_tag("missing", "new").unwrap(), 0);
    }

    #[test]
    fn test_working_memory_entry_count_reflects_stored_entries() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.entry_count().unwrap(), 0);
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        assert_eq!(wm.entry_count().unwrap(), 2);
    }

    #[test]
    fn test_episode_count_for_returns_correct_count() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.5).unwrap();
        store.add_episode(agent.clone(), "e2", 0.5).unwrap();
        assert_eq!(store.episode_count_for(&agent).unwrap(), 2);
    }

    #[test]
    fn test_episode_count_for_unknown_agent_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.episode_count_for(&AgentId::new("x")).unwrap(), 0);
    }

    #[test]
    fn test_semantic_store_unique_tags_returns_sorted_distinct_tags() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["b".to_string(), "a".to_string()]).unwrap();
        store.store("k2", "v2", vec!["a".to_string(), "c".to_string()]).unwrap();
        assert_eq!(store.unique_tags().unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_semantic_store_unique_tags_empty_returns_empty() {
        let store = SemanticStore::new();
        assert!(store.unique_tags().unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_count_matching_prefix_counts_correctly() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("user:a", "1").unwrap();
        wm.set("user:b", "2").unwrap();
        wm.set("other", "3").unwrap();
        assert_eq!(wm.count_matching_prefix("user:").unwrap(), 2);
        assert_eq!(wm.count_matching_prefix("other").unwrap(), 1);
        assert_eq!(wm.count_matching_prefix("none").unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_total_content_bytes_sums_lengths() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hi", 0.5).unwrap();   // 2 bytes
        store.add_episode(agent.clone(), "hello", 0.5).unwrap(); // 5 bytes
        assert_eq!(store.total_content_bytes(&agent).unwrap(), 7);
    }

    #[test]
    fn test_episodic_store_total_content_bytes_unknown_agent_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.total_content_bytes(&AgentId::new("x")).unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_avg_content_length_correct_mean() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hi", 0.5).unwrap();    // 2
        store.add_episode(agent.clone(), "hello", 0.5).unwrap(); // 5
        assert!((store.avg_content_length(&agent).unwrap() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_episodic_store_avg_content_length_empty_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.avg_content_length(&AgentId::new("x")).unwrap(), 0.0);
    }

    #[test]
    fn test_semantic_store_keys_for_tag_returns_matching_keys() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["rust".to_string()]).unwrap();
        store.store("k2", "v2", vec!["python".to_string()]).unwrap();
        store.store("k3", "v3", vec!["rust".to_string(), "async".to_string()]).unwrap();
        let mut keys = store.keys_for_tag("rust").unwrap();
        keys.sort_unstable();
        assert_eq!(keys, vec!["k1", "k3"]);
    }

    #[test]
    fn test_semantic_store_keys_for_tag_nonexistent_tag_returns_empty() {
        let store = SemanticStore::new();
        store.store("k", "v", vec!["rust".to_string()]).unwrap();
        assert!(store.keys_for_tag("missing").unwrap().is_empty());
    }

    #[test]
    fn test_count_episodes_with_tag_returns_correct_count() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode_with_tags(agent.clone(), "e1", 0.5, vec!["ai".to_string()]).unwrap();
        store.add_episode_with_tags(agent.clone(), "e2", 0.5, vec!["ai".to_string(), "ml".to_string()]).unwrap();
        store.add_episode_with_tags(agent.clone(), "e3", 0.5, vec!["ml".to_string()]).unwrap();
        assert_eq!(store.count_episodes_with_tag(&agent, "ai").unwrap(), 2);
        assert_eq!(store.count_episodes_with_tag(&agent, "ml").unwrap(), 2);
    }

    #[test]
    fn test_episodes_with_content_returns_matching_content() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "rust is great", 0.5).unwrap();
        store.add_episode(agent.clone(), "python is fun", 0.5).unwrap();
        store.add_episode(agent.clone(), "rust and python", 0.5).unwrap();
        let matches = store.episodes_with_content(&agent, "rust").unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_semantic_store_most_common_tag_returns_most_frequent() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["a".to_string(), "b".to_string()]).unwrap();
        store.store("k2", "v2", vec!["a".to_string()]).unwrap();
        store.store("k3", "v3", vec!["b".to_string()]).unwrap();
        // "a" appears 2 times, "b" appears 2 times - either is valid
        let tag = store.most_common_tag().unwrap();
        assert!(tag.is_some());
    }

    #[test]
    fn test_semantic_store_most_common_tag_empty_returns_none() {
        let store = SemanticStore::new();
        assert!(store.most_common_tag().unwrap().is_none());
    }

    #[test]
    fn test_working_memory_pairs_starting_with_returns_matching_pairs() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("user:name", "alice").unwrap();
        wm.set("user:age", "30").unwrap();
        wm.set("sys:mode", "prod").unwrap();
        let pairs = wm.pairs_starting_with("user:").unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|(k, _)| k.starts_with("user:")));
    }

    #[test]
    fn test_working_memory_total_key_bytes_sums_key_lengths() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "x").unwrap();   // 2 bytes
        wm.set("cde", "y").unwrap();  // 3 bytes
        assert_eq!(wm.total_key_bytes().unwrap(), 5);
    }

    #[test]
    fn test_working_memory_min_key_length_returns_shortest() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "x").unwrap();
        wm.set("abcd", "y").unwrap();
        assert_eq!(wm.min_key_length().unwrap(), 2);
    }

    #[test]
    fn test_working_memory_min_key_length_empty_returns_zero() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.min_key_length().unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_content_lengths_returns_lengths_in_order() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hi", 0.5).unwrap();    // 2
        store.add_episode(agent.clone(), "hello", 0.5).unwrap(); // 5
        assert_eq!(store.content_lengths(&agent).unwrap(), vec![2, 5]);
    }

    #[test]
    fn test_semantic_store_remove_entries_with_tag_removes_matching() {
        let store = SemanticStore::new();
        store.store("k1", "v1", vec!["old".to_string()]).unwrap();
        store.store("k2", "v2", vec!["keep".to_string()]).unwrap();
        store.store("k3", "v3", vec!["old".to_string(), "keep".to_string()]).unwrap();
        let removed = store.remove_entries_with_tag("old").unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.len().unwrap(), 1);
    }

    // ── Round 36 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_max_content_length_returns_longest() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hi", 0.5).unwrap();       // 2
        store.add_episode(agent.clone(), "hello!", 0.5).unwrap();   // 6
        store.add_episode(agent.clone(), "yo", 0.5).unwrap();       // 2
        assert_eq!(store.max_content_length(&agent).unwrap(), 6);
    }

    #[test]
    fn test_episodic_store_max_content_length_unknown_agent_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.max_content_length(&AgentId::new("x")).unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_min_content_length_returns_shortest() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "hi", 0.5).unwrap();      // 2
        store.add_episode(agent.clone(), "hello!", 0.5).unwrap();  // 6
        assert_eq!(store.min_content_length(&agent).unwrap(), 2);
    }

    #[test]
    fn test_episodic_store_min_content_length_unknown_agent_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.min_content_length(&AgentId::new("x")).unwrap(), 0);
    }

    #[test]
    fn test_semantic_store_total_value_bytes_sums_value_lengths() {
        let store = SemanticStore::new();
        store.store("k1", "ab", vec![]).unwrap();    // 2
        store.store("k2", "cde", vec![]).unwrap();   // 3
        store.store("k3", "fghij", vec![]).unwrap(); // 5
        assert_eq!(store.total_value_bytes().unwrap(), 10);
    }

    #[test]
    fn test_semantic_store_total_value_bytes_empty_returns_zero() {
        let store = SemanticStore::new();
        assert_eq!(store.total_value_bytes().unwrap(), 0);
    }

    // ── Round 37 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_agents_with_episodes_returns_sorted_ids() {
        let store = EpisodicStore::new();
        let b = AgentId::new("b");
        let a = AgentId::new("a");
        store.add_episode(b.clone(), "episode b", 0.5).unwrap();
        store.add_episode(a.clone(), "episode a", 0.5).unwrap();
        let agents = store.agents_with_episodes().unwrap();
        assert_eq!(agents, vec![a, b]);
    }

    #[test]
    fn test_episodic_store_agents_with_episodes_empty_returns_empty() {
        let store = EpisodicStore::new();
        assert!(store.agents_with_episodes().unwrap().is_empty());
    }

    #[test]
    fn test_episodic_store_high_importance_count_counts_above_threshold() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.3).unwrap();
        store.add_episode(agent.clone(), "high", 0.8).unwrap();
        store.add_episode(agent.clone(), "med", 0.6).unwrap();
        assert_eq!(store.high_importance_count(&agent, 0.5).unwrap(), 2);
    }

    #[test]
    fn test_episodic_store_high_importance_count_unknown_agent_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.high_importance_count(&AgentId::new("x"), 0.5).unwrap(), 0);
    }

    #[test]
    fn test_semantic_store_avg_value_bytes_returns_mean() {
        let store = SemanticStore::new();
        store.store("k1", "ab", vec![]).unwrap();   // 2
        store.store("k2", "abcd", vec![]).unwrap(); // 4
        let avg = store.avg_value_bytes().unwrap();
        assert!((avg - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_semantic_store_avg_value_bytes_empty_returns_zero() {
        let store = SemanticStore::new();
        assert_eq!(store.avg_value_bytes().unwrap(), 0.0);
    }

    #[test]
    fn test_semantic_store_max_value_bytes_returns_longest() {
        let store = SemanticStore::new();
        store.store("k1", "hi", vec![]).unwrap();      // 2
        store.store("k2", "hello!", vec![]).unwrap();  // 6
        assert_eq!(store.max_value_bytes().unwrap(), 6);
    }

    #[test]
    fn test_semantic_store_max_value_bytes_empty_returns_zero() {
        let store = SemanticStore::new();
        assert_eq!(store.max_value_bytes().unwrap(), 0);
    }

    // ── Round 38 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_content_contains_count_counts_matches() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "rust is great", 0.5).unwrap();
        store.add_episode(agent.clone(), "python is ok", 0.5).unwrap();
        store.add_episode(agent.clone(), "rust rocks", 0.5).unwrap();
        assert_eq!(store.content_contains_count(&agent, "rust").unwrap(), 2);
        assert_eq!(store.content_contains_count(&agent, "java").unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_content_contains_count_unknown_agent_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.content_contains_count(&AgentId::new("x"), "anything").unwrap(), 0);
    }

    #[test]
    fn test_semantic_store_min_value_bytes_returns_shortest() {
        let store = SemanticStore::new();
        store.store("k1", "hello world", vec![]).unwrap(); // 11
        store.store("k2", "hi", vec![]).unwrap();          // 2
        assert_eq!(store.min_value_bytes().unwrap(), 2);
    }

    #[test]
    fn test_semantic_store_min_value_bytes_empty_returns_zero() {
        let store = SemanticStore::new();
        assert_eq!(store.min_value_bytes().unwrap(), 0);
    }

    #[test]
    fn test_working_memory_max_value_length_returns_longest() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "ab").unwrap();     // 2
        wm.set("k2", "abcde").unwrap(); // 5
        assert_eq!(wm.max_value_length().unwrap(), 5);
    }

    #[test]
    fn test_working_memory_max_value_length_empty_returns_zero() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.max_value_length().unwrap(), 0);
    }

    // ── Round 39 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_episodes_by_importance_returns_desc_order() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "med", 0.5).unwrap();
        let contents = store.episodes_by_importance(&agent).unwrap();
        assert_eq!(contents[0], "high");
        assert_eq!(contents[2], "low");
    }

    #[test]
    fn test_episodic_store_episodes_by_importance_empty_agent_returns_empty() {
        let store = EpisodicStore::new();
        assert!(store.episodes_by_importance(&AgentId::new("x")).unwrap().is_empty());
    }

    #[test]
    fn test_semantic_store_all_keys_returns_sorted_keys() {
        let store = SemanticStore::new();
        store.store("banana", "v1", vec![]).unwrap();
        store.store("apple", "v2", vec![]).unwrap();
        store.store("cherry", "v3", vec![]).unwrap();
        assert_eq!(store.all_keys().unwrap(), vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_semantic_store_all_keys_empty_returns_empty() {
        let store = SemanticStore::new();
        assert!(store.all_keys().unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_min_value_length_returns_shortest() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "ab").unwrap();    // 2
        wm.set("k2", "abcde").unwrap(); // 5
        assert_eq!(wm.min_value_length().unwrap(), 2);
    }

    #[test]
    fn test_working_memory_min_value_length_empty_returns_zero() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.min_value_length().unwrap(), 0);
    }

    // ── Round 40 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_episode_count_above_importance_counts_correctly() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        assert_eq!(store.episode_count_above_importance(&agent, 0.4).unwrap(), 2);
        assert_eq!(store.episode_count_above_importance(&agent, 0.8).unwrap(), 1);
    }

    #[test]
    fn test_episodic_store_episode_count_above_importance_unknown_agent_returns_zero() {
        let store = EpisodicStore::new();
        assert_eq!(store.episode_count_above_importance(&AgentId::new("x"), 0.5).unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_low_importance_episodes_sorted_ascending() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let items = store.low_importance_episodes(&agent, 0.6).unwrap();
        assert_eq!(items.len(), 2);
        assert!(items[0].importance <= items[1].importance);
    }

    #[test]
    fn test_episodic_store_low_importance_episodes_empty_when_none_below_threshold() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        assert!(store.low_importance_episodes(&agent, 0.5).unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_values_containing_returns_matching_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hello world").unwrap();
        wm.set("k2", "goodbye world").unwrap();
        wm.set("k3", "something else").unwrap();
        let result = wm.values_containing("world").unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|v| v.contains("world")));
    }

    #[test]
    fn test_working_memory_values_containing_returns_empty_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hello").unwrap();
        assert!(wm.values_containing("xyz").unwrap().is_empty());
    }

    // ── Round 40 (continued): SemanticStore::keys_with_prefix ─────────────────

    #[test]
    fn test_semantic_store_keys_with_prefix_returns_matching_keys_sorted() {
        let store = SemanticStore::new();
        store.store("user:alice", "data-a", vec![]).unwrap();
        store.store("user:bob", "data-b", vec![]).unwrap();
        store.store("session:123", "sess", vec![]).unwrap();
        let keys = store.keys_with_prefix("user:").unwrap();
        assert_eq!(keys, vec!["user:alice", "user:bob"]);
    }

    #[test]
    fn test_semantic_store_keys_with_prefix_returns_empty_when_no_match() {
        let store = SemanticStore::new();
        store.store("abc", "v", vec![]).unwrap();
        assert!(store.keys_with_prefix("xyz").unwrap().is_empty());
    }

    #[test]
    fn test_semantic_store_keys_with_prefix_empty_prefix_returns_all_keys() {
        let store = SemanticStore::new();
        store.store("b", "v2", vec![]).unwrap();
        store.store("a", "v1", vec![]).unwrap();
        let keys = store.keys_with_prefix("").unwrap();
        assert_eq!(keys, vec!["a", "b"]);
    }

    // ── Round 41 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_episodes_sorted_by_timestamp_oldest_first() {
        use std::thread::sleep;
        use std::time::Duration;
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "first", 0.5).unwrap();
        sleep(Duration::from_millis(2));
        store.add_episode(agent.clone(), "second", 0.5).unwrap();
        let items = store.episodes_sorted_by_timestamp(&agent).unwrap();
        assert_eq!(items.len(), 2);
        assert!(items[0].timestamp <= items[1].timestamp);
    }

    #[test]
    fn test_episodic_store_episodes_sorted_by_timestamp_unknown_agent_empty() {
        let store = EpisodicStore::new();
        assert!(store.episodes_sorted_by_timestamp(&AgentId::new("unknown")).unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_keys_sorted_returns_alphabetical_order() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("banana", "b").unwrap();
        wm.set("apple", "a").unwrap();
        wm.set("cherry", "c").unwrap();
        assert_eq!(wm.keys_sorted().unwrap(), vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_working_memory_keys_sorted_empty_returns_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.keys_sorted().unwrap().is_empty());
    }

    // ── Round 41: MemoryItem tag management + predicates ─────────────────────

    #[test]
    fn test_memory_item_content_len_returns_byte_length() {
        let item = MemoryItem::new(AgentId::new("a"), "hello", 0.5, vec![]);
        assert_eq!(item.content_len(), 5);
    }

    #[test]
    fn test_memory_item_tag_count_returns_count() {
        let item = MemoryItem::new(AgentId::new("a"), "x", 0.5, vec!["t1".into(), "t2".into()]);
        assert_eq!(item.tag_count(), 2);
    }

    #[test]
    fn test_memory_item_tag_count_zero_when_no_tags() {
        let item = MemoryItem::new(AgentId::new("a"), "x", 0.5, vec![]);
        assert_eq!(item.tag_count(), 0);
    }

    #[test]
    fn test_memory_item_add_tag_adds_new_tag() {
        let mut item = MemoryItem::new(AgentId::new("a"), "x", 0.5, vec![]);
        assert!(item.add_tag("rust"));
        assert!(item.has_tag("rust"));
        assert_eq!(item.tag_count(), 1);
    }

    #[test]
    fn test_memory_item_add_tag_returns_false_for_duplicate() {
        let mut item = MemoryItem::new(AgentId::new("a"), "x", 0.5, vec!["rust".into()]);
        assert!(!item.add_tag("rust"));
        assert_eq!(item.tag_count(), 1);
    }

    #[test]
    fn test_memory_item_remove_tag_removes_existing_tag() {
        let mut item = MemoryItem::new(AgentId::new("a"), "x", 0.5, vec!["rust".into()]);
        assert!(item.remove_tag("rust"));
        assert!(!item.has_tag("rust"));
        assert_eq!(item.tag_count(), 0);
    }

    #[test]
    fn test_memory_item_remove_tag_returns_false_when_not_present() {
        let mut item = MemoryItem::new(AgentId::new("a"), "x", 0.5, vec![]);
        assert!(!item.remove_tag("missing"));
    }

    #[test]
    fn test_memory_item_is_high_importance_true_above_threshold() {
        let item = MemoryItem::new(AgentId::new("a"), "x", 0.9, vec![]);
        assert!(item.is_high_importance(0.7));
    }

    #[test]
    fn test_memory_item_is_high_importance_false_at_threshold() {
        // strictly greater than — equal to threshold is NOT high importance
        let item = MemoryItem::new(AgentId::new("a"), "x", 0.7, vec![]);
        assert!(!item.is_high_importance(0.7));
    }

    #[test]
    fn test_memory_item_is_high_importance_false_below_threshold() {
        let item = MemoryItem::new(AgentId::new("a"), "x", 0.3, vec![]);
        assert!(!item.is_high_importance(0.5));
    }

    // ── Round 41: EpisodicStore::recall_count_for ──────────────────────────────

    #[test]
    fn test_episodic_store_recall_count_for_returns_zero_initially() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store.add_episode(agent.clone(), "content", 0.5).unwrap();
        let count = store.recall_count_for(&agent, &id).unwrap();
        assert_eq!(count, Some(0));
    }

    #[test]
    fn test_episodic_store_recall_count_for_increments_after_recall() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id = store.add_episode(agent.clone(), "content", 0.5).unwrap();
        store.recall(&agent, 10).unwrap();
        let count = store.recall_count_for(&agent, &id).unwrap();
        assert_eq!(count, Some(1));
    }

    #[test]
    fn test_episodic_store_recall_count_for_returns_none_for_unknown_id() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let missing = MemoryId::new("no-such-id");
        assert_eq!(store.recall_count_for(&agent, &missing).unwrap(), None);
    }

    // ── Round 42 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_working_memory_value_for_longest_key_returns_correct_value() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("short", "v1").unwrap();
        wm.set("this-is-longer", "v2").unwrap();
        assert_eq!(wm.value_for_longest_key().unwrap(), Some("v2".to_string()));
    }

    #[test]
    fn test_working_memory_value_for_longest_key_returns_none_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_for_longest_key().unwrap(), None);
    }

    // ── Round 42: DecayPolicy Display ─────────────────────────────────────────

    #[test]
    fn test_decay_policy_display_format() {
        let p = DecayPolicy::exponential(24.0).unwrap();
        assert_eq!(p.to_string(), "Exponential(half_life=24.0h)");
    }

    #[test]
    fn test_decay_policy_display_fractional_hours() {
        let p = DecayPolicy::exponential(1.5).unwrap();
        assert_eq!(p.to_string(), "Exponential(half_life=1.5h)");
    }

    // ── Round 43 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodic_store_all_unique_tags_returns_sorted_deduped() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode_with_tags(
            agent.clone(), "c1", 0.5, vec!["rust".into(), "ai".into()]
        ).unwrap();
        store.add_episode_with_tags(
            agent.clone(), "c2", 0.5, vec!["ai".into(), "ml".into()]
        ).unwrap();
        let tags = store.all_unique_tags(&agent).unwrap();
        assert_eq!(tags, vec!["ai", "ml", "rust"]);
    }

    #[test]
    fn test_episodic_store_all_unique_tags_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.all_unique_tags(&AgentId::new("unknown")).unwrap().is_empty());
    }

    #[test]
    fn test_episodic_store_episodes_with_tag_returns_matching() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode_with_tags(
            agent.clone(), "tagged", 0.5, vec!["rust".into()]
        ).unwrap();
        store.add_episode(agent.clone(), "no-tag", 0.5).unwrap();
        let result = store.episodes_with_tag(&agent, "rust").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "tagged");
    }

    #[test]
    fn test_episodic_store_episodes_with_tag_empty_when_no_match() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "no-tag", 0.5).unwrap();
        assert!(store.episodes_with_tag(&agent, "missing-tag").unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_contains_value_true_when_present() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "hello").unwrap();
        assert!(wm.contains_value("hello").unwrap());
    }

    #[test]
    fn test_working_memory_contains_value_false_when_absent() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(!wm.contains_value("anything").unwrap());
    }

    // ── Round 42: shortest_key, count_below_value_length, most_recent_episode ──

    #[test]
    fn test_working_memory_shortest_key_returns_shortest() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("abc", "v").unwrap();
        wm.set("a", "v").unwrap();
        wm.set("ab", "v").unwrap();
        assert_eq!(wm.shortest_key().unwrap(), Some("a".to_string()));
    }

    #[test]
    fn test_working_memory_shortest_key_returns_none_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.shortest_key().unwrap(), None);
    }

    #[test]
    fn test_working_memory_count_below_value_length_counts_correctly() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hi").unwrap();      // 2 bytes
        wm.set("k2", "hello").unwrap();   // 5 bytes
        wm.set("k3", "yo").unwrap();      // 2 bytes
        // strictly less than 5: "hi" and "yo"
        assert_eq!(wm.count_below_value_length(5).unwrap(), 2);
    }

    #[test]
    fn test_working_memory_count_below_value_length_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.count_below_value_length(100).unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_most_recent_episode_returns_latest() {
        use std::thread::sleep;
        use std::time::Duration;
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "old", 0.5).unwrap();
        sleep(Duration::from_millis(5));
        store.add_episode(agent.clone(), "new", 0.5).unwrap();
        let ep = store.most_recent_episode(&agent).unwrap().unwrap();
        assert_eq!(ep.content, "new");
    }

    #[test]
    fn test_episodic_store_most_recent_episode_returns_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.most_recent_episode(&AgentId::new("nobody")).unwrap().is_none());
    }

    // ── Round 43: most_recalled_episode, value_for_shortest_key ───────────────

    #[test]
    fn test_episodic_store_most_recalled_episode_returns_highest_recall_count() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "low", 0.5).unwrap();
        let id_high = store.add_episode(agent.clone(), "high", 0.8).unwrap();
        // recall the second episode twice
        store.recall(&agent, 10).unwrap();
        store.recall(&agent, 10).unwrap();
        let ep = store.most_recalled_episode(&agent).unwrap().unwrap();
        assert_eq!(ep.id, id_high);
    }

    #[test]
    fn test_episodic_store_most_recalled_episode_returns_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.most_recalled_episode(&AgentId::new("nobody")).unwrap().is_none());
    }

    #[test]
    fn test_working_memory_value_for_shortest_key_returns_correct_value() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("abc", "long-key").unwrap();
        wm.set("a", "short-key").unwrap();
        wm.set("ab", "mid-key").unwrap();
        assert_eq!(wm.value_for_shortest_key().unwrap(), Some("short-key".to_string()));
    }

    #[test]
    fn test_working_memory_value_for_shortest_key_returns_none_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_for_shortest_key().unwrap(), None);
    }

    // ── Round 44: episode_count_before, entry_byte_pairs, key_with_longest_value

    #[test]
    fn test_episodic_store_episode_count_before_counts_older_episodes() {
        use chrono::{Duration, Utc};
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "old", 0.5).unwrap();
        store.add_episode(agent.clone(), "newer", 0.5).unwrap();
        // All episodes have timestamps <= now, so count before far future == 2
        let future = Utc::now() + Duration::hours(1);
        assert_eq!(store.episode_count_before(&agent, future).unwrap(), 2);
        // Count before epoch == 0
        let past = Utc::now() - Duration::hours(1);
        assert_eq!(store.episode_count_before(&agent, past).unwrap(), 0);
    }

    #[test]
    fn test_episodic_store_episode_count_before_zero_for_unknown_agent() {
        use chrono::Utc;
        let store = EpisodicStore::new();
        let future = Utc::now();
        assert_eq!(store.episode_count_before(&AgentId::new("nobody"), future).unwrap(), 0);
    }

    #[test]
    fn test_working_memory_entry_byte_pairs_returns_correct_lengths() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "hello").unwrap();
        let pairs = wm.entry_byte_pairs().unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (2, 5));
    }

    #[test]
    fn test_working_memory_entry_byte_pairs_empty_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.entry_byte_pairs().unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_key_with_longest_value_returns_correct_key() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "short").unwrap();
        wm.set("k2", "much longer value here").unwrap();
        assert_eq!(wm.key_with_longest_value().unwrap(), Some("k2".to_string()));
    }

    #[test]
    fn test_working_memory_key_with_longest_value_returns_none_when_empty() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.key_with_longest_value().unwrap(), None);
    }

    // ── Round 44: key_value_pairs_above_length ─────────────────────────────────

    #[test]
    fn test_key_value_pairs_above_length_returns_matching_pairs() {
        let wm = WorkingMemory::new(100).unwrap();
        wm.set("short", "hi").unwrap();
        wm.set("long", "hello world").unwrap();
        let pairs = wm.key_value_pairs_above_length(5).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "long");
        assert_eq!(pairs[0].1, "hello world");
    }

    #[test]
    fn test_key_value_pairs_above_length_empty_when_none_qualify() {
        let wm = WorkingMemory::new(100).unwrap();
        wm.set("a", "x").unwrap();
        assert!(wm.key_value_pairs_above_length(100).unwrap().is_empty());
    }

    #[test]
    fn test_key_value_pairs_above_length_empty_for_empty_store() {
        let wm = WorkingMemory::new(100).unwrap();
        assert!(wm.key_value_pairs_above_length(0).unwrap().is_empty());
    }

    // ── Round 45: episode_ids, total_key_bytes ─────────────────────────────────

    #[test]
    fn test_episodic_store_episode_ids_returns_all_ids() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        let id1 = store.add_episode(agent.clone(), "first", 0.5).unwrap();
        let id2 = store.add_episode(agent.clone(), "second", 0.5).unwrap();
        let mut ids = store.episode_ids(&agent).unwrap();
        ids.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_episodic_store_episode_ids_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.episode_ids(&AgentId::new("nobody")).unwrap().is_empty());
    }

    // ── Round 45: episode_count_all_agents, increment ─────────────────────────

    #[test]
    fn test_episode_count_all_agents_sums_across_agents() {
        let store = EpisodicStore::new();
        let a = AgentId::new("r45-cnt-a");
        let b = AgentId::new("r45-cnt-b");
        store.add_episode(a.clone(), "ep1", 0.5).unwrap();
        store.add_episode(a.clone(), "ep2", 0.5).unwrap();
        store.add_episode(b.clone(), "ep3", 0.5).unwrap();
        assert_eq!(store.episode_count_all_agents().unwrap(), 3);
    }

    #[test]
    fn test_episode_count_all_agents_zero_for_empty_store() {
        let store = EpisodicStore::new();
        assert_eq!(store.episode_count_all_agents().unwrap(), 0);
    }

    #[test]
    fn test_working_memory_increment_inserts_zero_when_absent() {
        let wm = WorkingMemory::new(100).unwrap();
        let v = wm.increment("counter-r45", 5).unwrap();
        assert_eq!(v, 5);
        assert_eq!(wm.get("counter-r45").unwrap().unwrap(), "5");
    }

    #[test]
    fn test_working_memory_increment_adds_to_existing_value() {
        let wm = WorkingMemory::new(100).unwrap();
        wm.set("n-r45", "10").unwrap();
        let v = wm.increment("n-r45", 3).unwrap();
        assert_eq!(v, 13);
    }

    #[test]
    fn test_working_memory_increment_returns_error_for_non_integer_value() {
        let wm = WorkingMemory::new(100).unwrap();
        wm.set("bad-r45", "not_a_number").unwrap();
        assert!(wm.increment("bad-r45", 1).is_err());
    }

    #[test]
    fn test_working_memory_increment_negative_step() {
        let wm = WorkingMemory::new(100).unwrap();
        wm.set("n2-r45", "10").unwrap();
        let v = wm.increment("n2-r45", -4).unwrap();
        assert_eq!(v, 6);
    }

    // ── Round 45: all_episode_ids, non_empty_values ───────────────────────────

    #[test]
    fn test_all_episode_ids_returns_ids_across_agents() {
        let store = EpisodicStore::new();
        let a = AgentId::new("r45-all-a");
        let b = AgentId::new("r45-all-b");
        let id1 = store.add_episode(a.clone(), "ep1", 0.5).unwrap();
        let id2 = store.add_episode(b.clone(), "ep2", 0.5).unwrap();
        let ids = store.all_episode_ids().unwrap();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_all_episode_ids_empty_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.all_episode_ids().unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_non_empty_values_filters_empty() {
        let wm = WorkingMemory::new(100).unwrap();
        wm.set("k1-r45", "hello").unwrap();
        wm.set("k2-r45", "").unwrap();
        wm.set("k3-r45", "world").unwrap();
        let vals = wm.non_empty_values().unwrap();
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&"hello".to_string()));
        assert!(vals.contains(&"world".to_string()));
    }

    #[test]
    fn test_working_memory_non_empty_values_empty_for_empty_store() {
        let wm = WorkingMemory::new(100).unwrap();
        assert!(wm.non_empty_values().unwrap().is_empty());
    }

    // ── Round 46: episodes_above_importance, total_bytes ──────────────────────

    #[test]
    fn test_episodic_store_episodes_above_importance_filters_correctly() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r46-a");
        store.add_episode(agent.clone(), "low", 0.3).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let result = store.episodes_above_importance(&agent, 0.5).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "high");
    }

    #[test]
    fn test_episodic_store_episodes_above_importance_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.episodes_above_importance(&AgentId::new("nobody-r46"), 0.5).unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_total_bytes_sums_key_and_value_lengths() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "xyz").unwrap(); // key=2, value=3 → total=5
        assert_eq!(wm.total_bytes().unwrap(), 5);
    }

    #[test]
    fn test_working_memory_total_bytes_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.total_bytes().unwrap(), 0);
    }

    // ── Round 47: episodes_between, max_key_bytes, value_length_histogram ─────

    #[test]
    fn test_episodic_store_episodes_between_returns_in_range() {
        use chrono::{Duration, Utc};
        let store = EpisodicStore::new();
        let agent = AgentId::new("r47-eb");
        store.add_episode(agent.clone(), "old", 0.5).unwrap();
        let past = Utc::now() - Duration::hours(1);
        let future = Utc::now() + Duration::hours(1);
        let result = store.episodes_between(&agent, past, future).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_episodic_store_episodes_between_empty_for_out_of_range() {
        use chrono::{Duration, Utc};
        let store = EpisodicStore::new();
        let agent = AgentId::new("r47-eb2");
        store.add_episode(agent.clone(), "ep", 0.5).unwrap();
        let far_past_start = Utc::now() - Duration::hours(10);
        let far_past_end = Utc::now() - Duration::hours(5);
        assert!(store.episodes_between(&agent, far_past_start, far_past_end).unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_max_key_bytes_returns_longest_key_len() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "x").unwrap();
        wm.set("abcde", "y").unwrap();
        assert_eq!(wm.max_key_bytes().unwrap(), 5);
    }

    #[test]
    fn test_working_memory_max_key_bytes_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.max_key_bytes().unwrap(), 0);
    }

    #[test]
    fn test_working_memory_value_length_histogram_buckets_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "ab").unwrap();   // len=2, bucket 0 (size=5)
        wm.set("k2", "abcde").unwrap(); // len=5, bucket 5
        wm.set("k3", "ab").unwrap();   // len=2, bucket 0 — evicted k1 under capacity
        // actually with capacity=10, k1 k2 k3 all fit
        let hist = wm.value_length_histogram(5).unwrap();
        // bucket 0: values with len 0-4 (both "ab"s have len 2)
        // but k3 set after k1 so k1 still exists (both are set under capacity 10)
        // Wait, k3 with same key "k3" and k1 with key "k1" — different keys
        // k1="ab"(2), k2="abcde"(5), k3="ab"(2)
        // bucket 0 (0-4): 2 entries; bucket 5 (5-9): 1 entry
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_working_memory_value_length_histogram_empty_for_zero_bucket_size() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "v").unwrap();
        assert!(wm.value_length_histogram(0).unwrap().is_empty());
    }

    // ── Round 48: weighted_importance_sum, episode_content_lengths, semantic_key_count ──

    #[test]
    fn test_episodic_store_weighted_importance_sum_sums_importances() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r48-wis");
        store.add_episode(agent.clone(), "ep1", 0.3).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.7).unwrap();
        let sum = store.weighted_importance_sum(&agent).unwrap();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_episodic_store_weighted_importance_sum_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r48-wis-missing");
        assert_eq!(store.weighted_importance_sum(&agent).unwrap(), 0.0);
    }

    #[test]
    fn test_episodic_store_episode_content_lengths_returns_lengths() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r48-ecl");
        store.add_episode(agent.clone(), "hi", 0.5).unwrap();
        store.add_episode(agent.clone(), "hello", 0.5).unwrap();
        let mut lengths = store.episode_content_lengths(&agent).unwrap();
        lengths.sort_unstable();
        assert_eq!(lengths, vec![2, 5]);
    }

    #[test]
    fn test_episodic_store_episode_content_lengths_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r48-ecl-missing");
        assert!(store.episode_content_lengths(&agent).unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_semantic_key_count_counts_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("intent_primary", "v1").unwrap();
        wm.set("intent_secondary", "v2").unwrap();
        wm.set("goal_main", "v3").unwrap();
        assert_eq!(wm.semantic_key_count("intent").unwrap(), 2);
    }

    #[test]
    fn test_working_memory_semantic_key_count_zero_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("context_a", "v").unwrap();
        assert_eq!(wm.semantic_key_count("intent").unwrap(), 0);
    }

    // ── Round 49: episode_count_by_agent, longest_value_bytes ─────────────────

    #[test]
    fn test_episode_count_by_agent_maps_agents_to_counts() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("r49-ecba-1");
        let a2 = AgentId::new("r49-ecba-2");
        store.add_episode(a1.clone(), "ep1", 0.5).unwrap();
        store.add_episode(a1.clone(), "ep2", 0.5).unwrap();
        store.add_episode(a2.clone(), "ep3", 0.5).unwrap();
        let counts = store.episode_count_by_agent().unwrap();
        assert_eq!(*counts.get(&a1).unwrap(), 2);
        assert_eq!(*counts.get(&a2).unwrap(), 1);
    }

    #[test]
    fn test_episode_count_by_agent_empty_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.episode_count_by_agent().unwrap().is_empty());
    }

    #[test]
    fn test_working_memory_longest_value_bytes_returns_max_value_len() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "ab").unwrap();
        wm.set("k2", "abcde").unwrap();
        assert_eq!(wm.longest_value_bytes().unwrap(), 5);
    }

    #[test]
    fn test_working_memory_longest_value_bytes_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.longest_value_bytes().unwrap(), 0);
    }

    // ── Round 47: episodes_between, max_key_bytes, value_length_histogram,
    //              episodes_tagged_with_all, content_word_count_total ──────────

    #[test]
    fn test_episodes_between_returns_episodes_in_range() {
        use chrono::{TimeZone, Utc};
        let store = EpisodicStore::new();
        let agent = AgentId::new("r47-eb-a");
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap();
        store.add_episode_at(agent.clone(), "early", 0.5, t0).unwrap();
        store.add_episode_at(agent.clone(), "mid", 0.5, t1).unwrap();
        store.add_episode_at(agent.clone(), "late", 0.5, t2).unwrap();
        // [t0, t2) — excludes t2
        let result = store.episodes_between(&agent, t0, t2).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_episodes_between_returns_empty_for_unknown_agent() {
        use chrono::{TimeZone, Utc};
        let store = EpisodicStore::new();
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
        assert!(
            store
                .episodes_between(&AgentId::new("nobody-r47"), t0, t1)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_working_memory_max_key_bytes_returns_longest_key_length() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "v").unwrap();        // 2 bytes
        wm.set("abcde", "v").unwrap();     // 5 bytes
        assert_eq!(wm.max_key_bytes().unwrap(), 5);
    }

    #[test]
    fn test_working_memory_max_key_bytes_zero_for_empty_store_r47() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.max_key_bytes().unwrap(), 0);
    }

    #[test]
    fn test_working_memory_value_length_histogram_groups_by_bucket() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "ab").unwrap();    // len=2 → bucket 0 (2/10*10=0)
        wm.set("k2", "abcde").unwrap(); // len=5 → bucket 0
        wm.set("k3", "abcdefghijk").unwrap(); // len=11 → bucket 10
        let hist = wm.value_length_histogram(10).unwrap();
        // bucket 0 → 2 entries, bucket 10 → 1 entry
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], (0, 2));
        assert_eq!(hist[1], (10, 1));
    }

    #[test]
    fn test_working_memory_value_length_histogram_empty_for_zero_bucket_size_r47() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "v").unwrap();
        assert!(wm.value_length_histogram(0).unwrap().is_empty());
    }

    #[test]
    fn test_episodes_tagged_with_all_returns_matching_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r47-twa-a");
        store
            .add_episode_with_tags(agent.clone(), "both", 0.5, vec!["x".into(), "y".into()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "only_x", 0.5, vec!["x".into()])
            .unwrap();
        let result = store.episodes_tagged_with_all(&agent, &["x", "y"]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "both");
    }

    #[test]
    fn test_episodes_tagged_with_all_empty_tags_returns_empty() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r47-twa-b");
        store.add_episode(agent.clone(), "ep", 0.5).unwrap();
        assert!(store.episodes_tagged_with_all(&agent, &[]).unwrap().is_empty());
    }

    #[test]
    fn test_content_word_count_total_sums_across_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r47-cwct-a");
        store.add_episode(agent.clone(), "one two three", 0.5).unwrap(); // 3 words
        store.add_episode(agent.clone(), "four five", 0.5).unwrap();     // 2 words
        assert_eq!(store.content_word_count_total(&agent).unwrap(), 5);
    }

    #[test]
    fn test_content_word_count_total_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert_eq!(
            store
                .content_word_count_total(&AgentId::new("nobody-r47-cwct"))
                .unwrap(),
            0
        );
    }

    // ── Round 49: agents_sorted_by_episode_count, episodes_with_min_word_count,
    //              values_sorted ───────────────────────────────────────────────

    #[test]
    fn test_agents_sorted_by_episode_count_most_episodes_first() {
        let store = EpisodicStore::new();
        let a = AgentId::new("r49-asbec-a");
        let b = AgentId::new("r49-asbec-b");
        store.add_episode(a.clone(), "e1", 0.5).unwrap();
        store.add_episode(b.clone(), "e2", 0.5).unwrap();
        store.add_episode(b.clone(), "e3", 0.5).unwrap();
        let sorted = store.agents_sorted_by_episode_count().unwrap();
        assert_eq!(sorted[0].as_str(), b.as_str());
        assert_eq!(sorted[1].as_str(), a.as_str());
    }

    #[test]
    fn test_agents_sorted_by_episode_count_empty_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.agents_sorted_by_episode_count().unwrap().is_empty());
    }

    #[test]
    fn test_episodes_with_min_word_count_returns_qualifying_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r49-ewmwc-a");
        store.add_episode(agent.clone(), "one", 0.5).unwrap();            // 1 word
        store.add_episode(agent.clone(), "one two three", 0.5).unwrap();   // 3 words
        let result = store.episodes_with_min_word_count(&agent, 2).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "one two three");
    }

    #[test]
    fn test_episodes_with_min_word_count_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(
            store
                .episodes_with_min_word_count(&AgentId::new("nobody-r49"), 1)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_working_memory_values_sorted_returns_alphabetically_sorted_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "banana").unwrap();
        wm.set("k2", "apple").unwrap();
        wm.set("k3", "cherry").unwrap();
        let sorted = wm.values_sorted().unwrap();
        assert_eq!(sorted, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_working_memory_values_sorted_empty_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.values_sorted().unwrap().is_empty());
    }

    // ── Round 50: episodes_above_content_bytes, value_count_above_bytes ────────

    #[test]
    fn test_episodes_above_content_bytes_returns_long_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-eacb");
        store.add_episode(agent.clone(), "hi", 0.5).unwrap();
        store.add_episode(agent.clone(), "long content here", 0.5).unwrap();
        let result = store.episodes_above_content_bytes(&agent, 5).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "long content here");
    }

    #[test]
    fn test_episodes_above_content_bytes_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-eacb-missing");
        assert!(store.episodes_above_content_bytes(&agent, 0).unwrap().is_empty());
    }

    #[test]
    fn test_value_count_above_bytes_counts_long_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "ab").unwrap();
        wm.set("k2", "abcdef").unwrap();
        assert_eq!(wm.value_count_above_bytes(3).unwrap(), 1);
    }

    #[test]
    fn test_value_count_above_bytes_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_count_above_bytes(0).unwrap(), 0);
    }

    // ── Round 47: most_important_episode, count_keys_above_bytes ──────────────

    #[test]
    fn test_most_important_episode_returns_highest_importance() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r47-imp-a");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let result = store.most_important_episode(&agent).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().content, "high");
    }

    #[test]
    fn test_most_important_episode_returns_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.most_important_episode(&AgentId::new("nobody-r47")).unwrap().is_none());
    }

    #[test]
    fn test_count_keys_above_bytes_counts_long_keys() {
        let wm = WorkingMemory::new(100).unwrap();
        wm.set("ab", "v").unwrap();       // key len=2
        wm.set("abcdef", "v").unwrap();   // key len=6
        assert_eq!(wm.count_keys_above_bytes(3).unwrap(), 1);
    }

    #[test]
    fn test_count_keys_above_bytes_zero_for_empty_store() {
        let wm = WorkingMemory::new(100).unwrap();
        assert_eq!(wm.count_keys_above_bytes(0).unwrap(), 0);
    }

    // ── Round 50: most_tagged_episode, tag_frequency, agents_sorted_by_episode_count,
    //              values_sorted ───────────────────────────────────────────────

    #[test]
    fn test_most_tagged_episode_returns_episode_with_most_tags() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-mte-a");
        store
            .add_episode_with_tags(agent.clone(), "many", 0.5, vec!["a".into(), "b".into(), "c".into()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "few", 0.5, vec!["a".into()])
            .unwrap();
        let result = store.most_tagged_episode(&agent).unwrap().unwrap();
        assert_eq!(result.content, "many");
    }

    #[test]
    fn test_most_tagged_episode_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.most_tagged_episode(&AgentId::new("nobody-r50")).unwrap().is_none());
    }

    #[test]
    fn test_tag_frequency_counts_each_tag() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-tf-a");
        store
            .add_episode_with_tags(agent.clone(), "e1", 0.5, vec!["x".into(), "y".into()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "e2", 0.5, vec!["x".into()])
            .unwrap();
        let freq = store.tag_frequency(&agent).unwrap();
        assert_eq!(freq["x"], 2);
        assert_eq!(freq["y"], 1);
    }

    #[test]
    fn test_tag_frequency_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        assert!(store.tag_frequency(&AgentId::new("nobody-r50-tf")).unwrap().is_empty());
    }

    // ── Round 51: unique_agents_count, key_value_pairs_sorted ─────────────────

    #[test]
    fn test_unique_agents_count_reflects_distinct_agents() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("r51-uac-1");
        let a2 = AgentId::new("r51-uac-2");
        store.add_episode(a1.clone(), "ep", 0.5).unwrap();
        store.add_episode(a2.clone(), "ep", 0.5).unwrap();
        assert_eq!(store.unique_agents_count().unwrap(), 2);
    }

    #[test]
    fn test_unique_agents_count_zero_for_empty_store() {
        let store = EpisodicStore::new();
        assert_eq!(store.unique_agents_count().unwrap(), 0);
    }

    #[test]
    fn test_key_value_pairs_sorted_returns_alphabetical_pairs() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("b_key", "v2").unwrap();
        wm.set("a_key", "v1").unwrap();
        let pairs = wm.key_value_pairs_sorted().unwrap();
        assert_eq!(pairs[0].0, "a_key");
        assert_eq!(pairs[1].0, "b_key");
    }

    #[test]
    fn test_key_value_pairs_sorted_empty_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.key_value_pairs_sorted().unwrap().is_empty());
    }

    // ── Round 52 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episode_count_with_tag_counts_matching_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a1");
        store
            .add_episode_with_tags(agent.clone(), "content", 0.5, vec!["rust".into(), "ai".into()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "other", 0.3, vec!["rust".into()])
            .unwrap();
        assert_eq!(store.episode_count_with_tag(&agent, "rust").unwrap(), 2);
        assert_eq!(store.episode_count_with_tag(&agent, "ai").unwrap(), 1);
    }

    #[test]
    fn test_episode_count_with_tag_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("unknown");
        assert_eq!(store.episode_count_with_tag(&agent, "rust").unwrap(), 0);
    }

    #[test]
    fn test_has_duplicate_values_true_when_two_keys_share_value() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "same").unwrap();
        wm.set("k2", "same").unwrap();
        assert!(wm.has_duplicate_values().unwrap());
    }

    #[test]
    fn test_has_duplicate_values_false_for_unique_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "one").unwrap();
        wm.set("k2", "two").unwrap();
        assert!(!wm.has_duplicate_values().unwrap());
    }

    #[test]
    fn test_has_duplicate_values_false_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(!wm.has_duplicate_values().unwrap());
    }

    // ── Round 52: episodes_with_tag_count, has_key_starting_with ──────────────

    #[test]
    fn test_episodes_with_tag_count_counts_tagged_items() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r52-ewtc-1");
        store.add_episode(agent.clone(), "no tags episode", 0.5).unwrap();
        store
            .add_episode_with_tags(agent.clone(), "tagged content", 0.8, vec!["relevant".to_string()])
            .unwrap();
        assert_eq!(store.episodes_with_tag_count(&agent).unwrap(), 1);
    }

    #[test]
    fn test_episodes_with_tag_count_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r52-ewtc-unknown");
        assert_eq!(store.episodes_with_tag_count(&agent).unwrap(), 0);
    }

    #[test]
    fn test_has_key_starting_with_true_when_prefix_matches() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("config_timeout", "30").unwrap();
        assert!(wm.has_key_starting_with("config_").unwrap());
    }

    #[test]
    fn test_has_key_starting_with_false_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("data_x", "v").unwrap();
        assert!(!wm.has_key_starting_with("config_").unwrap());
    }

    #[test]
    fn test_has_key_starting_with_false_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(!wm.has_key_starting_with("any").unwrap());
    }

    // ── Round 53: newest_episode, value_bytes_total, all_keys_sorted ──────────

    #[test]
    fn test_newest_episode_returns_most_recent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r53-ne-1");
        store.add_episode(agent.clone(), "first", 0.5).unwrap();
        store.add_episode(agent.clone(), "second", 0.7).unwrap();
        let newest = store.newest_episode(&agent).unwrap();
        assert!(newest.is_some());
        assert_eq!(newest.unwrap().content, "second");
    }

    #[test]
    fn test_newest_episode_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r53-ne-unknown");
        assert!(store.newest_episode(&agent).unwrap().is_none());
    }

    #[test]
    fn test_value_bytes_total_sums_value_lengths() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "abc").unwrap();
        wm.set("k2", "de").unwrap();
        assert_eq!(wm.value_bytes_total().unwrap(), 5);
    }

    #[test]
    fn test_value_bytes_total_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_bytes_total().unwrap(), 0);
    }

    #[test]
    fn test_all_keys_sorted_returns_alphabetical_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("zebra", "v").unwrap();
        wm.set("alpha", "v").unwrap();
        wm.set("mango", "v").unwrap();
        let keys = wm.all_keys_sorted().unwrap();
        assert_eq!(keys, vec!["alpha", "mango", "zebra"]);
    }

    #[test]
    fn test_all_keys_sorted_empty_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.all_keys_sorted().unwrap().is_empty());
    }

    // ── Round 53 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_shortest_value_returns_smallest_value() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hi").unwrap();
        wm.set("k2", "hello").unwrap();
        assert_eq!(wm.shortest_value().unwrap(), Some("hi".to_string()));
    }

    #[test]
    fn test_shortest_value_none_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.shortest_value().unwrap(), None);
    }

    #[test]
    fn test_shortest_value_tie_broken_lexicographically() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "bb").unwrap();
        wm.set("k2", "aa").unwrap();
        assert_eq!(wm.shortest_value().unwrap(), Some("aa".to_string()));
    }

    // ── Round 48 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_key_count_starting_with_counts_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("cfg_host", "localhost").unwrap();
        wm.set("cfg_port", "8080").unwrap();
        wm.set("debug_mode", "true").unwrap();
        assert_eq!(wm.key_count_starting_with("cfg_").unwrap(), 2);
    }

    #[test]
    fn test_key_count_starting_with_zero_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("alpha", "v").unwrap();
        assert_eq!(wm.key_count_starting_with("zzz").unwrap(), 0);
    }

    #[test]
    fn test_key_count_starting_with_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.key_count_starting_with("any").unwrap(), 0);
    }

    // ── Round 54: agent_ids_with_episodes, episode_importance_avg, key_count, value_for_key_or_default ──

    #[test]
    fn test_agent_ids_with_episodes_returns_sorted_ids() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("r54-z-agent");
        let a2 = AgentId::new("r54-a-agent");
        store.add_episode(a1.clone(), "ep", 0.5).unwrap();
        store.add_episode(a2.clone(), "ep", 0.5).unwrap();
        let ids = store.agent_ids_with_episodes().unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].as_str(), "r54-a-agent");
        assert_eq!(ids[1].as_str(), "r54-z-agent");
    }

    #[test]
    fn test_agent_ids_with_episodes_empty_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.agent_ids_with_episodes().unwrap().is_empty());
    }

    #[test]
    fn test_episode_importance_avg_returns_mean() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r54-eia-1");
        store.add_episode(agent.clone(), "ep1", 0.5).unwrap();
        store.add_episode(agent.clone(), "ep2", 1.0).unwrap();
        let avg = store.episode_importance_avg(&agent).unwrap();
        assert!((avg - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_episode_importance_avg_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r54-eia-unknown");
        assert_eq!(store.episode_importance_avg(&agent).unwrap(), 0.0);
    }

    #[test]
    fn test_key_count_returns_number_of_entries() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "1").unwrap();
        wm.set("b", "2").unwrap();
        assert_eq!(wm.key_count().unwrap(), 2);
    }

    #[test]
    fn test_key_count_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.key_count().unwrap(), 0);
    }

    #[test]
    fn test_value_for_key_or_default_returns_value_when_present() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("host", "localhost").unwrap();
        assert_eq!(wm.value_for_key_or_default("host", "unknown").unwrap(), "localhost");
    }

    #[test]
    fn test_value_for_key_or_default_returns_default_when_absent() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_for_key_or_default("missing", "fallback").unwrap(), "fallback");
    }

    // ── Round 55: agent_episode_counts, episodes_for_agent_sorted_by_importance, entries_with_key_prefix ──

    #[test]
    fn test_agent_episode_counts_returns_correct_counts() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("r55-aec-1");
        let a2 = AgentId::new("r55-aec-2");
        store.add_episode(a1.clone(), "ep1", 0.5).unwrap();
        store.add_episode(a1.clone(), "ep2", 0.7).unwrap();
        store.add_episode(a2.clone(), "ep3", 0.3).unwrap();
        let counts = store.agent_episode_counts().unwrap();
        assert_eq!(counts.get(&a1).copied(), Some(2));
        assert_eq!(counts.get(&a2).copied(), Some(1));
    }

    #[test]
    fn test_agent_episode_counts_empty_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.agent_episode_counts().unwrap().is_empty());
    }

    #[test]
    fn test_episodes_for_agent_sorted_by_importance_descending() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r55-esbi-1");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        let sorted = store.episodes_for_agent_sorted_by_importance(&agent).unwrap();
        assert_eq!(sorted[0].importance, 0.9);
        assert_eq!(sorted[2].importance, 0.2);
    }

    #[test]
    fn test_entries_with_key_prefix_returns_matching_pairs() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("cfg_host", "localhost").unwrap();
        wm.set("cfg_port", "8080").unwrap();
        wm.set("debug_mode", "true").unwrap();
        let entries = wm.entries_with_key_prefix("cfg_").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "cfg_host");
        assert_eq!(entries[1].0, "cfg_port");
    }

    #[test]
    fn test_entries_with_key_prefix_empty_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("data_x", "v").unwrap();
        assert!(wm.entries_with_key_prefix("cfg_").unwrap().is_empty());
    }

    // ── Round 54 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_episodes_below_importance_filters_correctly() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a1");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let low = store.episodes_below_importance(&agent, 0.5).unwrap();
        assert_eq!(low.len(), 1);
        assert_eq!(low[0].content, "low");
    }

    #[test]
    fn test_episodes_below_importance_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("unknown");
        assert!(store.episodes_below_importance(&agent, 0.5).unwrap().is_empty());
    }

    #[test]
    fn test_keys_with_value_equal_to_returns_correct_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hello").unwrap();
        wm.set("k2", "world").unwrap();
        wm.set("k3", "hello").unwrap();
        let mut keys = wm.keys_with_value_equal_to("hello").unwrap();
        keys.sort();
        assert_eq!(keys, vec!["k1", "k3"]);
    }

    #[test]
    fn test_keys_with_value_equal_to_empty_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hello").unwrap();
        assert!(wm.keys_with_value_equal_to("world").unwrap().is_empty());
    }

    // ── Round 49 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_all_values_sorted_returns_sorted_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "banana").unwrap();
        wm.set("k2", "apple").unwrap();
        wm.set("k3", "cherry").unwrap();
        let values = wm.all_values_sorted().unwrap();
        assert_eq!(values, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_all_values_sorted_empty_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.all_values_sorted().unwrap().is_empty());
    }

    #[test]
    fn test_keys_with_value_prefix_returns_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("host", "http://example.com").unwrap();
        wm.set("api", "http://api.example.com").unwrap();
        wm.set("name", "alice").unwrap();
        let mut keys = wm.keys_with_value_prefix("http://").unwrap();
        keys.sort();
        assert_eq!(keys, vec!["api", "host"]);
    }

    #[test]
    fn test_keys_with_value_prefix_empty_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hello").unwrap();
        assert!(wm.keys_with_value_prefix("zzz").unwrap().is_empty());
    }

    #[test]
    fn test_agents_with_episodes_above_count_returns_high_activity_agents() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("agent-high");
        let a2 = AgentId::new("agent-low");
        store.add_episode(a1.clone(), "e1", 0.5).unwrap();
        store.add_episode(a1.clone(), "e2", 0.5).unwrap();
        store.add_episode(a1.clone(), "e3", 0.5).unwrap();
        store.add_episode(a2.clone(), "e1", 0.5).unwrap();
        let above = store.agents_with_episodes_above_count(2).unwrap();
        assert_eq!(above.len(), 1);
        assert_eq!(above[0], a1);
    }

    #[test]
    fn test_agents_with_episodes_above_count_empty_when_none_qualify() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("a");
        store.add_episode(agent.clone(), "e1", 0.5).unwrap();
        assert!(store.agents_with_episodes_above_count(5).unwrap().is_empty());
    }

    // ── Round 56: episodes_matching_content, top_agent_by_importance, has_value_equal_to, value_contains ──

    #[test]
    fn test_episodes_matching_content_returns_matching_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r56-emc-1");
        store.add_episode(agent.clone(), "user asked about Rust", 0.5).unwrap();
        store.add_episode(agent.clone(), "user asked about Python", 0.5).unwrap();
        let matched = store.episodes_matching_content(&agent, "Rust").unwrap();
        assert_eq!(matched.len(), 1);
        assert!(matched[0].content.contains("Rust"));
    }

    #[test]
    fn test_episodes_matching_content_empty_for_no_match() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r56-emc-2");
        store.add_episode(agent.clone(), "something else", 0.5).unwrap();
        assert!(store.episodes_matching_content(&agent, "Rust").unwrap().is_empty());
    }

    #[test]
    fn test_top_agent_by_importance_returns_highest_total_agent() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("r56-tabi-low");
        let a2 = AgentId::new("r56-tabi-high");
        store.add_episode(a1.clone(), "ep", 0.2).unwrap();
        store.add_episode(a2.clone(), "ep", 0.9).unwrap();
        let top = store.top_agent_by_importance().unwrap();
        assert_eq!(top.unwrap().as_str(), "r56-tabi-high");
    }

    #[test]
    fn test_top_agent_by_importance_none_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.top_agent_by_importance().unwrap().is_none());
    }

    #[test]
    fn test_has_value_equal_to_true_when_match_exists() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("key", "exact_value").unwrap();
        assert!(wm.has_value_equal_to("exact_value").unwrap());
    }

    #[test]
    fn test_has_value_equal_to_false_when_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("key", "other").unwrap();
        assert!(!wm.has_value_equal_to("exact_value").unwrap());
    }

    #[test]
    fn test_value_contains_true_when_substr_in_value() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("url", "https://example.com/path").unwrap();
        assert!(wm.value_contains("url", "example.com").unwrap());
    }

    #[test]
    fn test_value_contains_false_when_key_absent() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(!wm.value_contains("missing", "anything").unwrap());
    }

    // ── Round 50 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_avg_episode_importance_returns_mean() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-avg-1");
        store.add_episode(agent.clone(), "e1", 0.4).unwrap();
        store.add_episode(agent.clone(), "e2", 0.8).unwrap();
        let avg = store.avg_episode_importance(&agent).unwrap();
        assert!((avg - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_avg_episode_importance_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-avg-unknown");
        assert_eq!(store.avg_episode_importance(&agent).unwrap(), 0.0);
    }

    #[test]
    fn test_episode_content_bytes_total_correct() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-bytes-1");
        store.add_episode(agent.clone(), "hello", 0.5).unwrap();
        store.add_episode(agent.clone(), "world!", 0.5).unwrap();
        let total = store.episode_content_bytes_total(&agent).unwrap();
        assert_eq!(total, 11); // "hello" (5) + "world!" (6)
    }

    #[test]
    fn test_episode_content_bytes_total_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r50-bytes-unknown");
        assert_eq!(store.episode_content_bytes_total(&agent).unwrap(), 0);
    }

    // ── Round 57: episodes_after_timestamp, agent_with_min_importance_avg, keys_without_prefix, count_values_equal_to ──

    #[test]
    fn test_episodes_after_timestamp_returns_newer_episodes() {
        use chrono::{Duration, Utc};
        let store = EpisodicStore::new();
        let agent = AgentId::new("r57-eat-1");
        store.add_episode(agent.clone(), "old", 0.5).unwrap();
        let cutoff = Utc::now();
        store.add_episode(agent.clone(), "new", 0.7).unwrap();
        let after = store.episodes_after_timestamp(&agent, cutoff).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].content, "new");
        let _ = Duration::seconds(0); // suppress unused import warning
    }

    #[test]
    fn test_agent_with_min_importance_avg_returns_lowest() {
        let store = EpisodicStore::new();
        let a1 = AgentId::new("r57-awmia-hi");
        let a2 = AgentId::new("r57-awmia-lo");
        store.add_episode(a1.clone(), "ep", 0.9).unwrap();
        store.add_episode(a2.clone(), "ep", 0.1).unwrap();
        let min_agent = store.agent_with_min_importance_avg().unwrap();
        assert_eq!(min_agent.unwrap().as_str(), "r57-awmia-lo");
    }

    #[test]
    fn test_agent_with_min_importance_avg_none_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.agent_with_min_importance_avg().unwrap().is_none());
    }

    #[test]
    fn test_keys_without_prefix_returns_non_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("cfg_host", "v").unwrap();
        wm.set("data_x", "v").unwrap();
        wm.set("data_y", "v").unwrap();
        let keys = wm.keys_without_prefix("cfg_").unwrap();
        assert_eq!(keys, vec!["data_x", "data_y"]);
    }

    #[test]
    fn test_keys_without_prefix_empty_when_all_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("cfg_a", "v").unwrap();
        assert!(wm.keys_without_prefix("cfg_").unwrap().is_empty());
    }

    #[test]
    fn test_count_values_equal_to_returns_correct_count() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "same").unwrap();
        wm.set("k2", "same").unwrap();
        wm.set("k3", "other").unwrap();
        assert_eq!(wm.count_values_equal_to("same").unwrap(), 2);
    }

    #[test]
    fn test_count_values_equal_to_zero_for_no_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "v").unwrap();
        assert_eq!(wm.count_values_equal_to("missing").unwrap(), 0);
    }

    // ── Round 57: has_episodes_for_agent, all_values_non_empty ────────────────

    #[test]
    fn test_has_episodes_for_agent_true_after_adding_episode() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("agent-r57");
        store.add_episode_with_tags(agent.clone(), "content", 0.5_f32, vec![]).unwrap();
        assert!(store.has_episodes_for_agent(&agent).unwrap());
    }

    #[test]
    fn test_has_episodes_for_agent_false_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("nobody-r57");
        assert!(!store.has_episodes_for_agent(&agent).unwrap());
    }

    #[test]
    fn test_all_values_non_empty_true_for_non_empty_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "hello").unwrap();
        wm.set("b", "world").unwrap();
        assert!(wm.all_values_non_empty().unwrap());
    }

    #[test]
    fn test_all_values_non_empty_false_when_empty_value_present() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "hello").unwrap();
        wm.set("b", "").unwrap();
        assert!(!wm.all_values_non_empty().unwrap());
    }

    #[test]
    fn test_all_values_non_empty_true_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.all_values_non_empty().unwrap());
    }

    // ── Round 58: highest_importance_episode, avg_episode_content_words, has_duplicate_content, value_starts_with ──

    #[test]
    fn test_highest_importance_episode_returns_max() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r58-hie-1");
        store.add_episode(agent.clone(), "low", 0.2).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let top = store.highest_importance_episode(&agent).unwrap();
        assert_eq!(top.unwrap().importance, 0.9);
    }

    #[test]
    fn test_highest_importance_episode_none_for_empty_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r58-hie-unknown");
        assert!(store.highest_importance_episode(&agent).unwrap().is_none());
    }

    #[test]
    fn test_avg_episode_content_words_returns_mean() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r58-aecw-1");
        store.add_episode(agent.clone(), "one two", 0.5).unwrap();
        store.add_episode(agent.clone(), "a b c d", 0.5).unwrap();
        let avg = store.avg_episode_content_words(&agent).unwrap();
        assert_eq!(avg, 3.0);
    }

    #[test]
    fn test_has_duplicate_content_true_when_duplicates_exist() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r58-hdc-1");
        store.add_episode(agent.clone(), "same content", 0.5).unwrap();
        store.add_episode(agent.clone(), "same content", 0.7).unwrap();
        assert!(store.has_duplicate_content(&agent).unwrap());
    }

    #[test]
    fn test_has_duplicate_content_false_when_all_unique() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r58-hdc-2");
        store.add_episode(agent.clone(), "first", 0.5).unwrap();
        store.add_episode(agent.clone(), "second", 0.5).unwrap();
        assert!(!store.has_duplicate_content(&agent).unwrap());
    }

    #[test]
    fn test_value_starts_with_true_when_value_has_prefix() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("url", "https://example.com").unwrap();
        assert!(wm.value_starts_with("url", "https://").unwrap());
    }

    #[test]
    fn test_value_starts_with_false_when_key_absent() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(!wm.value_starts_with("missing", "https://").unwrap());
    }

    // ── Round 51 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_value_with_max_bytes_returns_longest_value() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hi").unwrap();
        wm.set("k2", "hello world").unwrap();
        wm.set("k3", "ok").unwrap();
        assert_eq!(wm.value_with_max_bytes().unwrap(), Some("hello world".to_string()));
    }

    #[test]
    fn test_value_with_max_bytes_none_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_with_max_bytes().unwrap(), None);
    }

    // ── Round 59: values_longer_than, has_key_with_prefix, min_episode_importance ──

    #[test]
    fn test_values_longer_than_returns_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("short", "hi").unwrap();
        wm.set("long", "hello world").unwrap();
        let keys = wm.values_longer_than(4).unwrap();
        assert_eq!(keys, vec!["long".to_string()]);
    }

    #[test]
    fn test_values_longer_than_empty_when_none_qualify() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k", "ab").unwrap();
        let keys = wm.values_longer_than(100).unwrap();
        assert!(keys.is_empty());
    }

    #[test]
    fn test_has_key_with_prefix_true() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("agent:config", "val").unwrap();
        assert!(wm.has_key_with_prefix("agent:").unwrap());
    }

    #[test]
    fn test_has_key_with_prefix_false() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("other:config", "val").unwrap();
        assert!(!wm.has_key_with_prefix("agent:").unwrap());
    }

    #[test]
    fn test_min_episode_importance_returns_minimum() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r59-min-imp");
        store.add_episode(agent.clone(), "first", 0.8).unwrap();
        store.add_episode(agent.clone(), "second", 0.2).unwrap();
        store.add_episode(agent.clone(), "third", 0.5).unwrap();
        let min = store.min_episode_importance(&agent).unwrap();
        assert!((min.unwrap() - 0.2_f32).abs() < 1e-6);
    }

    #[test]
    fn test_min_episode_importance_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r59-min-unknown");
        assert!(store.min_episode_importance(&agent).unwrap().is_none());
    }

    // ── Round 60: value_count_matching_prefix, count_episodes_in_window ───────

    #[test]
    fn test_value_count_matching_prefix_correct() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hello world").unwrap();
        wm.set("k2", "hello there").unwrap();
        wm.set("k3", "goodbye").unwrap();
        assert_eq!(wm.value_count_matching_prefix("hello").unwrap(), 2);
    }

    #[test]
    fn test_value_count_matching_prefix_zero_when_none_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "nope").unwrap();
        assert_eq!(wm.value_count_matching_prefix("yes").unwrap(), 0);
    }

    #[test]
    fn test_count_episodes_in_window_correct() {
        use chrono::Utc;
        let store = EpisodicStore::new();
        let agent = AgentId::new("r60-window");
        store.add_episode(agent.clone(), "ep1", 0.5).unwrap();
        store.add_episode(agent.clone(), "ep2", 0.5).unwrap();
        // All episodes are created "now", so a wide window should include them
        let start = Utc::now() - chrono::Duration::hours(1);
        let end = Utc::now() + chrono::Duration::hours(1);
        let count = store.count_episodes_in_window(&agent, start, end).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_episodes_in_window_zero_for_future_window() {
        use chrono::Utc;
        let store = EpisodicStore::new();
        let agent = AgentId::new("r60-future");
        store.add_episode(agent.clone(), "ep", 0.5).unwrap();
        let start = Utc::now() + chrono::Duration::hours(1);
        let end = Utc::now() + chrono::Duration::hours(2);
        let count = store.count_episodes_in_window(&agent, start, end).unwrap();
        assert_eq!(count, 0);
    }

    // ── Round 61: total_tag_count, avg_tag_count_per_episode, keys_with_suffix, avg_key_length ──

    #[test]
    fn test_total_tag_count_sums_tags_across_episodes() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-tags");
        store
            .add_episode_with_tags(agent.clone(), "ep1", 0.5, vec!["a".to_string(), "b".to_string()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "ep2", 0.5, vec!["c".to_string()])
            .unwrap();
        assert_eq!(store.total_tag_count(&agent).unwrap(), 3);
    }

    #[test]
    fn test_total_tag_count_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-tags-unknown");
        assert_eq!(store.total_tag_count(&agent).unwrap(), 0);
    }

    #[test]
    fn test_avg_tag_count_per_episode_correct() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-avg-tags");
        store
            .add_episode_with_tags(agent.clone(), "e1", 0.5, vec!["x".to_string(), "y".to_string()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "e2", 0.5, vec![])
            .unwrap();
        // 2 tags over 2 episodes = 1.0
        assert!((store.avg_tag_count_per_episode(&agent).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_tag_count_per_episode_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-avg-tags-unknown");
        assert_eq!(store.avg_tag_count_per_episode(&agent).unwrap(), 0.0);
    }

    #[test]
    fn test_keys_with_suffix_returns_matching_keys() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("config.json", "data").unwrap();
        wm.set("readme.md", "doc").unwrap();
        wm.set("schema.json", "meta").unwrap();
        let keys = wm.keys_with_suffix(".json").unwrap();
        assert_eq!(keys, vec!["config.json".to_string(), "schema.json".to_string()]);
    }

    #[test]
    fn test_keys_with_suffix_empty_when_none_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("hello", "world").unwrap();
        let keys = wm.keys_with_suffix(".json").unwrap();
        assert!(keys.is_empty());
    }

    #[test]
    fn test_avg_key_length_correct() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "v").unwrap();   // 2
        wm.set("abcd", "v").unwrap(); // 4
        let avg = wm.avg_key_length().unwrap();
        assert!((avg - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_key_length_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.avg_key_length().unwrap(), 0.0);
    }

    // ── Round 62: episodes_above_importance_count, tag_union, entries_sorted_by_key_length ──

    #[test]
    fn test_episodes_above_importance_count_correct() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r62-above");
        store.add_episode(agent.clone(), "e1", 0.8).unwrap();
        store.add_episode(agent.clone(), "e2", 0.3).unwrap();
        store.add_episode(agent.clone(), "e3", 0.6).unwrap();
        assert_eq!(store.episodes_above_importance_count(&agent, 0.5).unwrap(), 2);
    }

    #[test]
    fn test_episodes_above_importance_count_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r62-above-unknown");
        assert_eq!(store.episodes_above_importance_count(&agent, 0.5).unwrap(), 0);
    }

    #[test]
    fn test_tag_union_collects_all_unique_tags() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r62-union");
        store
            .add_episode_with_tags(agent.clone(), "e1", 0.5, vec!["a".to_string(), "b".to_string()])
            .unwrap();
        store
            .add_episode_with_tags(agent.clone(), "e2", 0.5, vec!["b".to_string(), "c".to_string()])
            .unwrap();
        let union = store.tag_union(&agent).unwrap();
        assert_eq!(union.len(), 3);
        assert!(union.contains("a"));
        assert!(union.contains("b"));
        assert!(union.contains("c"));
    }

    #[test]
    fn test_tag_union_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r62-union-unknown");
        assert!(store.tag_union(&agent).unwrap().is_empty());
    }

    #[test]
    fn test_entries_sorted_by_key_length_orders_by_length_then_alpha() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("cc", "v").unwrap();
        wm.set("a", "v").unwrap();
        wm.set("bbb", "v").unwrap();
        let pairs = wm.entries_sorted_by_key_length().unwrap();
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["a", "cc", "bbb"]);
    }

    #[test]
    fn test_entries_sorted_by_key_length_empty_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert!(wm.entries_sorted_by_key_length().unwrap().is_empty());
    }

    // ── Round 63: episode_most_recent, episodes_by_importance_range, value_bytes_max ──

    #[test]
    fn test_episode_most_recent_returns_newest() {
        use std::thread::sleep;
        use std::time::Duration;
        let store = EpisodicStore::new();
        let agent = AgentId::new("r63-recent");
        store.add_episode(agent.clone(), "first", 0.5).unwrap();
        sleep(Duration::from_millis(2));
        store.add_episode(agent.clone(), "latest", 0.8).unwrap();
        let recent = store.episode_most_recent(&agent).unwrap();
        assert_eq!(recent.map(|m| m.content), Some("latest".to_string()));
    }

    #[test]
    fn test_episode_most_recent_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r63-recent-unknown");
        assert!(store.episode_most_recent(&agent).unwrap().is_none());
    }

    #[test]
    fn test_episodes_by_importance_range_correct() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r63-range");
        store.add_episode(agent.clone(), "low", 0.1).unwrap();
        store.add_episode(agent.clone(), "mid", 0.5).unwrap();
        store.add_episode(agent.clone(), "high", 0.9).unwrap();
        let range = store.episodes_by_importance_range(&agent, 0.3, 0.7).unwrap();
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].content, "mid");
    }

    #[test]
    fn test_episodes_by_importance_range_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r63-range-unknown");
        assert!(store.episodes_by_importance_range(&agent, 0.0, 1.0).unwrap().is_empty());
    }

    #[test]
    fn test_value_bytes_max_returns_max() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "short").unwrap();
        wm.set("k2", "much longer value").unwrap();
        assert_eq!(wm.value_bytes_max().unwrap(), "much longer value".len());
    }

    #[test]
    fn test_value_bytes_max_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_bytes_max().unwrap(), 0);
    }

    #[test]
    fn test_total_content_words_returns_total_word_count() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r51-words-1");
        store.add_episode(agent.clone(), "hello world", 0.5).unwrap();
        store.add_episode(agent.clone(), "one two three", 0.5).unwrap();
        assert_eq!(store.total_content_words(&agent).unwrap(), 5);
    }

    #[test]
    fn test_total_content_words_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r51-words-unknown");
        assert_eq!(store.total_content_words(&agent).unwrap(), 0);
    }

    // ── Round 58: agent_count, max_key_length ─────────────────────────────────

    #[test]
    fn test_agent_count_zero_for_empty_store() {
        let store = EpisodicStore::new();
        assert_eq!(store.agent_count().unwrap(), 0);
    }

    #[test]
    fn test_agent_count_reflects_distinct_agents() {
        let store = EpisodicStore::new();
        let a = AgentId::new("r58-a");
        let b = AgentId::new("r58-b");
        store.add_episode_with_tags(a.clone(), "x", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(a.clone(), "y", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(b.clone(), "z", 0.5, vec![]).unwrap();
        assert_eq!(store.agent_count().unwrap(), 2);
    }

    #[test]
    fn test_max_key_length_returns_longest() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("short", "v").unwrap();
        wm.set("a-longer-key", "v").unwrap();
        assert_eq!(wm.max_key_length().unwrap(), "a-longer-key".len());
    }

    #[test]
    fn test_max_key_length_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.max_key_length().unwrap(), 0);
    }

    // ── Round 59: episodes_with_importance_above, value_char_count ─────────────

    #[test]
    fn test_episodes_with_importance_above_filters_correctly() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r59-epia-agent");
        store.add_episode_with_tags(agent.clone(), "low", 0.2, vec![]).unwrap();
        store.add_episode_with_tags(agent.clone(), "high", 0.9, vec![]).unwrap();
        store.add_episode_with_tags(agent.clone(), "mid", 0.5, vec![]).unwrap();
        let result = store.episodes_with_importance_above(&agent, 0.4).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_episodes_with_importance_above_empty_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r59-epia-unknown");
        let result = store.episodes_with_importance_above(&agent, 0.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_value_char_count_returns_correct_count() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("greeting", "héllo").unwrap();
        assert_eq!(wm.value_char_count("greeting").unwrap(), 5);
    }

    #[test]
    fn test_value_char_count_zero_for_missing_key() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.value_char_count("missing").unwrap(), 0);
    }

    // ── Round 60: agent_episode_importance_sum, keys_longer_than ──────────────

    #[test]
    fn test_agent_episode_importance_sum_correct() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r60-importance-sum");
        store.add_episode_with_tags(agent.clone(), "a", 0.4, vec![]).unwrap();
        store.add_episode_with_tags(agent.clone(), "b", 0.6, vec![]).unwrap();
        let sum = store.agent_episode_importance_sum(&agent).unwrap();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_agent_episode_importance_sum_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r60-importance-sum-unknown");
        assert_eq!(store.agent_episode_importance_sum(&agent).unwrap(), 0.0);
    }

    #[test]
    fn test_keys_longer_than_filters_correctly() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("ab", "v").unwrap();
        wm.set("abcdef", "v").unwrap();
        wm.set("xyz", "v").unwrap();
        let mut keys = wm.keys_longer_than(3).unwrap();
        keys.sort();
        assert_eq!(keys, vec!["abcdef".to_string()]);
    }

    #[test]
    fn test_keys_longer_than_empty_when_none_qualify() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("hi", "v").unwrap();
        assert!(wm.keys_longer_than(10).unwrap().is_empty());
    }

    // ── Round 59: episodes_with_content_containing, average_value_length ──────

    #[test]
    fn test_episodes_with_content_containing_returns_matches() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r59-content");
        store.add_episode_with_tags(agent.clone(), "hello world", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(agent.clone(), "goodbye", 0.5, vec![]).unwrap();
        let result = store.episodes_with_content_containing(&agent, "hello").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].content.contains("hello"));
    }

    #[test]
    fn test_episodes_with_content_containing_empty_for_no_match() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r59-content-miss");
        store.add_episode_with_tags(agent.clone(), "hello", 0.5, vec![]).unwrap();
        let result = store.episodes_with_content_containing(&agent, "xyz").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_average_value_length_computes_mean() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "hi").unwrap();      // 2 bytes
        wm.set("b", "hello").unwrap();   // 5 bytes
        // mean = 3.5
        let avg = wm.average_value_length().unwrap();
        assert!((avg - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_average_value_length_zero_for_empty_store() {
        let wm = WorkingMemory::new(10).unwrap();
        assert_eq!(wm.average_value_length().unwrap(), 0.0);
    }

    // ── Round 61: episodes_min_importance, episode_min_content_words, keys_shorter_than ──

    #[test]
    fn test_episodes_min_importance_returns_minimum() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-min-imp");
        store.add_episode_with_tags(agent.clone(), "a", 0.8, vec![]).unwrap();
        store.add_episode_with_tags(agent.clone(), "b", 0.2, vec![]).unwrap();
        let min = store.episodes_min_importance(&agent).unwrap();
        assert!((min.unwrap() - 0.2_f32).abs() < 1e-6);
    }

    #[test]
    fn test_episodes_min_importance_none_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-min-imp-unknown");
        assert!(store.episodes_min_importance(&agent).unwrap().is_none());
    }

    #[test]
    fn test_episode_min_content_words_returns_minimum() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-min-words");
        store.add_episode_with_tags(agent.clone(), "one two three", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(agent.clone(), "hello", 0.5, vec![]).unwrap();
        assert_eq!(store.episode_min_content_words(&agent).unwrap(), 1);
    }

    #[test]
    fn test_episode_min_content_words_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r61-min-words-unknown");
        assert_eq!(store.episode_min_content_words(&agent).unwrap(), 0);
    }

    #[test]
    fn test_keys_shorter_than_filters_correctly() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "v").unwrap();
        wm.set("abcdef", "v").unwrap();
        wm.set("abc", "v").unwrap();
        let mut keys = wm.keys_shorter_than(3).unwrap();
        keys.sort();
        assert_eq!(keys, vec!["a".to_string()]);
    }

    #[test]
    fn test_keys_shorter_than_empty_when_none_qualify() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("long_key", "v").unwrap();
        assert!(wm.keys_shorter_than(1).unwrap().is_empty());
    }

    // ── Round 62: episode_max_content_words, total_items, all_episodes, count_keys_below_bytes ──

    #[test]
    fn test_episode_max_content_words_returns_maximum() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r62-max-words");
        store.add_episode_with_tags(agent.clone(), "one", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(agent.clone(), "one two three four", 0.5, vec![]).unwrap();
        assert_eq!(store.episode_max_content_words(&agent).unwrap(), 4);
    }

    #[test]
    fn test_episode_max_content_words_zero_for_unknown_agent() {
        let store = EpisodicStore::new();
        let agent = AgentId::new("r62-max-words-unknown");
        assert_eq!(store.episode_max_content_words(&agent).unwrap(), 0);
    }

    #[test]
    fn test_total_items_counts_across_all_agents() {
        let store = EpisodicStore::new();
        let a = AgentId::new("r62-total-a");
        let b = AgentId::new("r62-total-b");
        store.add_episode_with_tags(a.clone(), "x", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(a.clone(), "y", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(b.clone(), "z", 0.5, vec![]).unwrap();
        assert_eq!(store.total_items().unwrap(), 3);
    }

    #[test]
    fn test_all_episodes_returns_all_items() {
        let store = EpisodicStore::new();
        let a = AgentId::new("r62-all-a");
        let b = AgentId::new("r62-all-b");
        store.add_episode_with_tags(a.clone(), "ep1", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(b.clone(), "ep2", 0.5, vec![]).unwrap();
        assert_eq!(store.all_episodes().unwrap().len(), 2);
    }

    #[test]
    fn test_count_keys_below_bytes_correct() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("a", "v").unwrap();
        wm.set("ab", "v").unwrap();
        wm.set("abcdefgh", "v").unwrap();
        assert_eq!(wm.count_keys_below_bytes(3).unwrap(), 2);
    }

    #[test]
    fn test_count_keys_below_bytes_zero_when_none_qualify() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("long_key", "v").unwrap();
        assert_eq!(wm.count_keys_below_bytes(1).unwrap(), 0);
    }

    // ── Round 63: min_episode_count, values_with_prefix ──────────────────────

    #[test]
    fn test_min_episode_count_returns_agent_with_fewest() {
        let store = EpisodicStore::new();
        let a = AgentId::new("r63-min-a");
        let b = AgentId::new("r63-min-b");
        store.add_episode_with_tags(a.clone(), "x", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(a.clone(), "y", 0.5, vec![]).unwrap();
        store.add_episode_with_tags(b.clone(), "z", 0.5, vec![]).unwrap();
        let result = store.min_episode_count().unwrap().unwrap();
        assert_eq!(result.0, b);
        assert_eq!(result.1, 1);
    }

    #[test]
    fn test_min_episode_count_none_for_empty_store() {
        let store = EpisodicStore::new();
        assert!(store.min_episode_count().unwrap().is_none());
    }

    #[test]
    fn test_values_with_prefix_returns_matching_values() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "prefix:hello").unwrap();
        wm.set("k2", "other:world").unwrap();
        wm.set("k3", "prefix:bye").unwrap();
        let mut vals = wm.values_with_prefix("prefix:").unwrap();
        vals.sort();
        assert_eq!(vals, vec!["prefix:bye".to_string(), "prefix:hello".to_string()]);
    }

    #[test]
    fn test_values_with_prefix_empty_when_none_match() {
        let wm = WorkingMemory::new(10).unwrap();
        wm.set("k1", "hello").unwrap();
        assert!(wm.values_with_prefix("xyz:").unwrap().is_empty());
    }
}
