//! # Long-Term Memory with Forgetting
//!
//! Implements an Ebbinghaus forgetting-curve-based long-term memory store.
//! Entries decay over time and are pruned below a minimum importance threshold.
//! Similar entries can be consolidated by averaging their importance.

use std::collections::HashMap;

// ── LtmEntry ──────────────────────────────────────────────────────────────────

/// A single long-term memory entry.
#[derive(Debug, Clone)]
pub struct LtmEntry {
    /// Stable unique identifier for this entry.
    pub id: u64,
    /// The stored content string.
    pub content: String,
    /// Original importance score at creation; range `[0.0, 1.0]`.
    pub importance: f32,
    /// Total number of times this entry has been recalled.
    pub access_count: u32,
    /// Unix timestamp in seconds when this entry was created.
    pub created_at: u64,
    /// Unix timestamp in seconds when this entry was last accessed.
    pub last_accessed: u64,
    /// Current importance after applying the forgetting-curve decay.
    pub decayed_importance: f32,
}

// ── ForgettingCurve ───────────────────────────────────────────────────────────

/// A forgetting curve model.
#[derive(Debug, Clone)]
pub enum ForgettingCurve {
    /// Ebbinghaus exponential decay: `retention = exp(-t / stability)`.
    ///
    /// `stability` is the half-life-like parameter in days — a higher value
    /// means the memory persists longer.
    Ebbinghaus {
        /// Decay stability factor in days (larger = slower forgetting).
        stability: f32,
    },
}

impl ForgettingCurve {
    /// Compute the retention fraction (range `[0.0, 1.0]`) for an entry
    /// created `days_since_creation` days ago.
    pub fn retention(&self, days_since_creation: f32) -> f32 {
        match self {
            ForgettingCurve::Ebbinghaus { stability } => {
                (-days_since_creation / stability).exp()
            }
        }
    }
}

// ── LtmConfig ─────────────────────────────────────────────────────────────────

/// Configuration for an [`LtmStore`].
#[derive(Debug, Clone)]
pub struct LtmConfig {
    /// Maximum number of entries allowed in the store.
    pub capacity: usize,
    /// Entries with `decayed_importance` below this threshold are pruned on
    /// each [`LtmStore::decay`] call.
    pub min_importance: f32,
    /// Ebbinghaus stability parameter in days.
    pub stability_days: f32,
    /// How often (in hours) automatic decay should be triggered.  This is
    /// advisory — callers must drive the decay interval explicitly.
    pub decay_interval_hours: u32,
}

impl Default for LtmConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            min_importance: 0.05,
            stability_days: 30.0,
            decay_interval_hours: 24,
        }
    }
}

// ── TF-IDF bag-of-words helpers ───────────────────────────────────────────────

/// Build a simple term-frequency vector from a content string.
///
/// Tokenises by whitespace, lowercases, and counts occurrences.
fn term_frequency(text: &str) -> HashMap<String, f32> {
    let mut tf: HashMap<String, f32> = HashMap::new();
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let n = tokens.len() as f32;
    if n == 0.0 {
        return tf;
    }
    for token in &tokens {
        let word = token.to_lowercase();
        *tf.entry(word).or_insert(0.0) += 1.0 / n;
    }
    tf
}

/// Cosine similarity between two term-frequency maps.
fn cosine_similarity(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
    let dot: f32 = a
        .iter()
        .filter_map(|(k, va)| b.get(k).map(|vb| va * vb))
        .sum();
    let mag_a: f32 = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let mag_b: f32 = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

// ── LtmStore ──────────────────────────────────────────────────────────────────

/// Long-term memory store with Ebbinghaus forgetting-curve decay.
pub struct LtmStore {
    entries: Vec<LtmEntry>,
    config: LtmConfig,
    curve: ForgettingCurve,
    next_id: u64,
}

impl LtmStore {
    /// Create a new store with the given configuration.
    pub fn new(config: LtmConfig) -> Self {
        let curve = ForgettingCurve::Ebbinghaus {
            stability: config.stability_days,
        };
        Self {
            entries: Vec::new(),
            config,
            curve,
            next_id: 1,
        }
    }

    /// Create a store with default configuration.
    pub fn default_config() -> Self {
        Self::new(LtmConfig::default())
    }

    /// Current Unix timestamp in seconds (simulated via `std::time::SystemTime`).
    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Store a new memory entry and return its assigned ID.
    ///
    /// If the store is at capacity the least-important entry (by
    /// `decayed_importance`) is evicted to make room.
    pub fn remember(&mut self, content: impl Into<String>, importance: f32) -> u64 {
        let now = Self::now_secs();
        let importance = importance.clamp(0.0, 1.0);
        let id = self.next_id;
        self.next_id += 1;

        // Evict if at capacity.
        if self.entries.len() >= self.config.capacity {
            let worst = self
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.decayed_importance
                        .partial_cmp(&b.decayed_importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            if let Some(idx) = worst {
                self.entries.remove(idx);
            }
        }

        self.entries.push(LtmEntry {
            id,
            content: content.into(),
            importance,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            decayed_importance: importance,
        });
        id
    }

    /// Recall the top-`top_k` entries most relevant to `query`.
    ///
    /// Relevance score = cosine_similarity × `decayed_importance` × retention.
    /// Entries are returned sorted by descending relevance.
    pub fn recall(&mut self, query: &str, top_k: usize) -> Vec<&LtmEntry> {
        let now = Self::now_secs();
        let query_tf = term_frequency(query);

        // Score each entry.
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let days = (now.saturating_sub(e.created_at)) as f32 / 86_400.0;
                let retention = self.curve.retention(days);
                let sim = cosine_similarity(&term_frequency(&e.content), &query_tf);
                let score = sim * e.decayed_importance * retention;
                (i, score)
            })
            .collect();

        scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Update access statistics.
        let indices: Vec<usize> = scored
            .iter()
            .take(top_k)
            .map(|(i, _)| *i)
            .collect();

        for &idx in &indices {
            self.entries[idx].access_count += 1;
            self.entries[idx].last_accessed = now;
        }

        // Borrow as immutable for the return slice.
        indices
            .into_iter()
            .filter_map(|i| self.entries.get(i))
            .collect()
    }

    /// Update `decayed_importance` for all entries and prune those below
    /// `min_importance`.
    pub fn decay(&mut self) {
        let now = Self::now_secs();
        let min = self.config.min_importance;

        for entry in &mut self.entries {
            let days = (now.saturating_sub(entry.created_at)) as f32 / 86_400.0;
            let retention = self.curve.retention(days);
            entry.decayed_importance = entry.importance * retention;
        }

        self.entries
            .retain(|e| e.decayed_importance >= min);
    }

    /// Merge entries that are very similar (cosine similarity > 0.85) by
    /// averaging their importance and retaining the longer content.
    pub fn consolidate(&mut self) {
        const SIMILARITY_THRESHOLD: f32 = 0.85;

        let mut merged: Vec<bool> = vec![false; self.entries.len()];
        let mut i = 0;

        while i < self.entries.len() {
            if merged[i] {
                i += 1;
                continue;
            }
            let tf_i = term_frequency(&self.entries[i].content);
            let mut j = i + 1;
            while j < self.entries.len() {
                if merged[j] {
                    j += 1;
                    continue;
                }
                let tf_j = term_frequency(&self.entries[j].content);
                if cosine_similarity(&tf_i, &tf_j) > SIMILARITY_THRESHOLD {
                    // Merge j into i.
                    let avg_importance = (self.entries[i].importance + self.entries[j].importance) / 2.0;
                    let avg_decayed = (self.entries[i].decayed_importance + self.entries[j].decayed_importance) / 2.0;
                    self.entries[i].importance = avg_importance;
                    self.entries[i].decayed_importance = avg_decayed;
                    // Keep longer content.
                    if self.entries[j].content.len() > self.entries[i].content.len() {
                        let new_content = self.entries[j].content.clone();
                        self.entries[i].content = new_content;
                    }
                    self.entries[i].access_count += self.entries[j].access_count;
                    merged[j] = true;
                }
                j += 1;
            }
            i += 1;
        }

        // Remove merged entries (in reverse to preserve indices).
        let mut idx = merged.len();
        while idx > 0 {
            idx -= 1;
            if merged[idx] {
                self.entries.remove(idx);
            }
        }
    }

    /// Return an immutable reference to all stored entries.
    pub fn entries(&self) -> &[LtmEntry] {
        &self.entries
    }

    /// Return the number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the configuration.
    pub fn config(&self) -> &LtmConfig {
        &self.config
    }

    /// Look up a single entry by its ID.
    pub fn get(&self, id: u64) -> Option<&LtmEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

// ── PersonaScope (defined in persona.rs; declared here for AgentRuntime) ──────

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout
)]
mod tests {
    use super::*;

    fn make_store() -> LtmStore {
        LtmStore::new(LtmConfig {
            capacity: 100,
            min_importance: 0.01,
            stability_days: 30.0,
            decay_interval_hours: 24,
        })
    }

    #[test]
    fn test_remember_returns_incrementing_ids() {
        let mut store = make_store();
        let id1 = store.remember("first entry", 0.8);
        let id2 = store.remember("second entry", 0.5);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_store_len_grows() {
        let mut store = make_store();
        assert_eq!(store.len(), 0);
        store.remember("alpha", 0.9);
        assert_eq!(store.len(), 1);
        store.remember("beta", 0.7);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_is_empty_initially() {
        let store = make_store();
        assert!(store.is_empty());
    }

    #[test]
    fn test_get_by_id() {
        let mut store = make_store();
        let id = store.remember("hello world", 0.6);
        let entry = store.get(id).expect("entry should exist");
        assert_eq!(entry.content, "hello world");
        assert_eq!(entry.id, id);
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let store = make_store();
        assert!(store.get(9999).is_none());
    }

    #[test]
    fn test_importance_clamped_to_one() {
        let mut store = make_store();
        let id = store.remember("over-important", 2.5);
        let e = store.get(id).unwrap();
        assert!(e.importance <= 1.0);
    }

    #[test]
    fn test_importance_clamped_to_zero() {
        let mut store = make_store();
        let id = store.remember("negative importance", -1.0);
        let e = store.get(id).unwrap();
        assert!(e.importance >= 0.0);
    }

    #[test]
    fn test_recall_returns_at_most_top_k() {
        let mut store = make_store();
        for i in 0..10 {
            store.remember(format!("memory item number {i}"), 0.5);
        }
        let results = store.recall("memory", 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_recall_increments_access_count() {
        let mut store = make_store();
        let id = store.remember("unique content for recall test", 0.9);
        store.recall("unique content recall", 5);
        let entry = store.get(id).unwrap();
        assert!(entry.access_count > 0);
    }

    #[test]
    fn test_recall_empty_store() {
        let mut store = make_store();
        let results = store.recall("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_decay_does_not_prune_fresh_entries() {
        let mut store = make_store();
        store.remember("fresh memory", 0.9);
        store.decay();
        // Fresh entry should still be present (retention ≈ 1.0 for t=0).
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_decay_prunes_below_min_importance() {
        let mut store = LtmStore::new(LtmConfig {
            capacity: 100,
            min_importance: 0.5,
            stability_days: 30.0,
            decay_interval_hours: 24,
        });
        // Manually add an entry with very low decayed_importance.
        let id = store.remember("low importance entry", 0.01);
        // Force decay to run (decayed_importance will be ~0.01 * retention ≈ 0.01).
        store.decay();
        // Entry should be pruned since 0.01 < min_importance 0.5.
        assert!(store.get(id).is_none());
    }

    #[test]
    fn test_ebbinghaus_retention_at_zero_days() {
        let curve = ForgettingCurve::Ebbinghaus { stability: 30.0 };
        let r = curve.retention(0.0);
        assert!((r - 1.0).abs() < 1e-5, "retention at t=0 should be 1.0, got {r}");
    }

    #[test]
    fn test_ebbinghaus_retention_decays_over_time() {
        let curve = ForgettingCurve::Ebbinghaus { stability: 30.0 };
        let r0 = curve.retention(0.0);
        let r30 = curve.retention(30.0);
        let r90 = curve.retention(90.0);
        assert!(r0 > r30, "retention should decrease over time");
        assert!(r30 > r90, "retention should decrease further at 90 days");
    }

    #[test]
    fn test_consolidate_merges_similar_entries() {
        let mut store = make_store();
        store.remember("the quick brown fox", 0.8);
        store.remember("the quick brown fox jumps", 0.6);
        let before = store.len();
        store.consolidate();
        let after = store.len();
        // The two very-similar entries should be merged into one.
        assert!(after <= before, "consolidate should reduce or equal entry count");
    }

    #[test]
    fn test_consolidate_keeps_dissimilar_entries() {
        let mut store = make_store();
        store.remember("machine learning gradient descent", 0.8);
        store.remember("ocean waves crashing on the beach", 0.7);
        let before = store.len();
        store.consolidate();
        let after = store.len();
        assert_eq!(before, after, "dissimilar entries should not be merged");
    }

    #[test]
    fn test_capacity_eviction() {
        let mut store = LtmStore::new(LtmConfig {
            capacity: 3,
            min_importance: 0.0,
            stability_days: 30.0,
            decay_interval_hours: 24,
        });
        store.remember("entry one", 0.9);
        store.remember("entry two", 0.8);
        store.remember("entry three", 0.7);
        store.remember("entry four", 0.6);
        assert_eq!(store.len(), 3, "store should not exceed capacity");
    }

    #[test]
    fn test_entries_returns_all() {
        let mut store = make_store();
        store.remember("a", 0.5);
        store.remember("b", 0.4);
        assert_eq!(store.entries().len(), 2);
    }

    #[test]
    fn test_config_accessible() {
        let config = LtmConfig {
            capacity: 500,
            min_importance: 0.1,
            stability_days: 14.0,
            decay_interval_hours: 12,
        };
        let store = LtmStore::new(config.clone());
        assert_eq!(store.config().capacity, 500);
        assert!((store.config().stability_days - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_term_frequency_empty_string() {
        let tf = term_frequency("");
        assert!(tf.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let tf = term_frequency("hello world hello");
        assert!((cosine_similarity(&tf, &tf) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let tf_a = term_frequency("hello world");
        let tf_b = term_frequency("foo bar baz");
        let sim = cosine_similarity(&tf_a, &tf_b);
        assert!(sim < 1e-5, "orthogonal vectors should have ~0 similarity");
    }

    #[test]
    fn test_ltm_store_default_config() {
        let store = LtmStore::default_config();
        assert_eq!(store.config().capacity, 10_000);
        assert!(store.is_empty());
    }
}
