//! # Module: Memory Compression (high-level)
//!
//! ## Responsibility
//! Compresses episodic memory when it grows too large, using three composable
//! strategies that can be applied independently or in sequence:
//!
//! | Strategy | What it does |
//! |---|---|
//! | [`TemporalCompression`] | Keeps recent events verbatim; merges similar old events |
//! | [`SemanticDedup`] | Removes near-duplicate memories via Jaccard similarity |
//! | [`ImportanceFilter`] | Retains memories tagged as "important" regardless of age |
//!
//! A background-ready async interface is provided via
//! [`MemoryCompressor::run_if_needed`], which triggers compression only when
//! the memory buffer exceeds a configurable threshold.
//!
//! ## Relationship to `memory_compression`
//!
//! The existing [`crate::memory_compression`] module is a low-level,
//! token-budget-aware compressor.  This module provides a higher-level,
//! strategy-composable API on top of the same [`MemoryTurn`] type, adding
//! richer metadata in [`CompressedEpisode`] and a clean async trigger
//! mechanism.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::memory_compress::{
//!     CompressorConfig, ImportanceFilter, MemoryCompressor,
//!     SemanticDedup, TemporalCompression,
//! };
//! use llm_agent_runtime::memory_compression::{MemoryTurn, Role};
//! use std::time::SystemTime;
//!
//! let config = CompressorConfig {
//!     max_turns: 20,
//!     recency_keep: 5,
//!     jaccard_dedup_threshold: 0.8,
//!     importance_tag: "important".to_string(),
//! };
//!
//! let compressor = MemoryCompressor::new(config);
//!
//! // Build a small memory buffer.
//! let turns: Vec<MemoryTurn> = (0..25)
//!     .map(|i| MemoryTurn {
//!         id: format!("t{i}"),
//!         role: Role::User,
//!         content: format!("user message {i}"),
//!         timestamp: SystemTime::now(),
//!         importance_score: 0.0,
//!         token_count: 10,
//!         tags: if i == 2 { vec!["important".into()] } else { vec![] },
//!     })
//!     .collect();
//!
//! // Compress if over the threshold.
//! let (kept, episodes) = compressor.compress(turns);
//! println!("Kept {} turns, {} compressed episodes", kept.len(), episodes.len());
//! ```

use crate::memory_compression::MemoryTurn;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

// ── CompressedEpisode ─────────────────────────────────────────────────────────

/// A structured record summarising a batch of compressed [`MemoryTurn`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedEpisode {
    /// Unique identifier for this compressed episode block.
    pub id: String,
    /// Number of original turns that were merged into this episode.
    pub original_count: usize,
    /// Number of turns retained after deduplication and filtering.
    pub compressed_to: usize,
    /// Human-readable prose summary of what was in the compressed segment.
    pub summary: String,
    /// Inclusive wall-clock time span of the compressed turns
    /// `(earliest, latest)` as seconds since the Unix epoch.
    pub time_range: (u64, u64),
    /// Common tags found across the compressed turns.
    pub common_tags: Vec<String>,
}

impl CompressedEpisode {
    /// Build a [`CompressedEpisode`] from a batch of turns that have been
    /// reduced to `compressed_to` representative turns.
    pub fn from_batch(
        id: impl Into<String>,
        original_count: usize,
        compressed_to: usize,
        turns: &[MemoryTurn],
        summary: impl Into<String>,
    ) -> Self {
        let (earliest_secs, latest_secs) = if turns.is_empty() {
            (0u64, 0u64)
        } else {
            let times: Vec<u64> = turns
                .iter()
                .map(|t| {
                    t.timestamp
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                })
                .collect();
            (
                *times.iter().min().unwrap_or(&0),
                *times.iter().max().unwrap_or(&0),
            )
        };

        // Find tags that appear in at least half the turns.
        let tag_counts: HashMap<String, usize> = {
            let mut map: HashMap<String, usize> = HashMap::new();
            for t in turns {
                for tag in &t.tags {
                    *map.entry(tag.clone()).or_insert(0) += 1;
                }
            }
            map
        };
        let threshold = (turns.len() / 2).max(1);
        let mut common_tags: Vec<String> = tag_counts
            .into_iter()
            .filter(|(_, count)| *count >= threshold)
            .map(|(tag, _)| tag)
            .collect();
        common_tags.sort();

        Self {
            id: id.into(),
            original_count,
            compressed_to,
            summary: summary.into(),
            time_range: (earliest_secs, latest_secs),
            common_tags,
        }
    }
}

// ── TemporalCompression ───────────────────────────────────────────────────────

/// Keeps the N most-recent turns verbatim; merges older ones into
/// [`CompressedEpisode`]s grouped by a configurable time-window.
///
/// ## Algorithm
/// 1. Partition turns into `recent` (last `recency_keep`) and `old`.
/// 2. Group `old` turns into windows of `window_size` turns each.
/// 3. Within each window, keep the most-important turn as a representative.
/// 4. Emit one [`CompressedEpisode`] per window.
pub struct TemporalCompression {
    /// Always preserve this many of the most-recent turns unchanged.
    pub recency_keep: usize,
    /// Turns per compression window for old events.
    pub window_size: usize,
}

impl Default for TemporalCompression {
    fn default() -> Self {
        Self { recency_keep: 10, window_size: 5 }
    }
}

impl TemporalCompression {
    /// Create a new [`TemporalCompression`] with explicit parameters.
    pub fn new(recency_keep: usize, window_size: usize) -> Self {
        Self { recency_keep, window_size: window_size.max(1) }
    }

    /// Apply temporal compression to `turns` (assumed to be in chronological
    /// order).
    ///
    /// Returns `(kept_turns, compressed_episodes)`.
    pub fn compress(&self, mut turns: Vec<MemoryTurn>) -> (Vec<MemoryTurn>, Vec<CompressedEpisode>) {
        if turns.len() <= self.recency_keep {
            return (turns, vec![]);
        }

        // Sort chronologically for safety.
        turns.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let split_at = turns.len().saturating_sub(self.recency_keep);
        let old = turns[..split_at].to_vec();
        let recent = turns[split_at..].to_vec();

        let mut episodes: Vec<CompressedEpisode> = Vec::new();
        let mut episode_idx = 0usize;

        for window in old.chunks(self.window_size) {
            let original_count = window.len();
            // Pick the representative: highest importance_score, else first.
            let representative = window
                .iter()
                .max_by(|a, b| {
                    a.importance_score
                        .partial_cmp(&b.importance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned();

            let compressed_to = if representative.is_some() { 1 } else { 0 };
            let summary = format!(
                "[Temporal window {episode_idx}: merged {original_count} turn(s). \
                 Topics: {}]",
                window
                    .iter()
                    .flat_map(|t| t.tags.iter().cloned())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            episodes.push(CompressedEpisode::from_batch(
                format!("temporal-ep-{episode_idx}"),
                original_count,
                compressed_to,
                window,
                summary,
            ));
            episode_idx += 1;
        }

        (recent, episodes)
    }
}

// ── SemanticDedup ─────────────────────────────────────────────────────────────

/// Removes near-duplicate memories using Jaccard similarity on word sets.
///
/// Two turns are considered duplicates when
/// `jaccard(words(a), words(b)) >= threshold`.  When duplicates are found, the
/// one with the higher `importance_score` is kept (ties → later timestamp).
pub struct SemanticDedup {
    /// Similarity threshold in `[0.0, 1.0]`.  A value of `1.0` means only
    /// exact word-set matches are removed; `0.8` removes near-duplicates with
    /// ≥ 80 % word overlap.
    pub threshold: f32,
}

impl Default for SemanticDedup {
    fn default() -> Self {
        Self { threshold: 0.8 }
    }
}

impl SemanticDedup {
    /// Create a new [`SemanticDedup`] with the given threshold.
    pub fn new(threshold: f32) -> Self {
        Self { threshold: threshold.clamp(0.0, 1.0) }
    }

    /// Remove near-duplicate turns from `turns`.
    ///
    /// Returns the deduplicated list; order of surviving turns is preserved.
    pub fn dedup(&self, turns: Vec<MemoryTurn>) -> Vec<MemoryTurn> {
        if turns.len() <= 1 {
            return turns;
        }

        // Pre-compute word sets.
        let word_sets: Vec<HashSet<String>> = turns
            .iter()
            .map(|t| Self::word_set(&t.content))
            .collect();

        let n = turns.len();
        // `keep[i]` is true unless turn i is dominated by an earlier turn.
        let mut keep = vec![true; n];

        for i in 0..n {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !keep[j] {
                    continue;
                }
                let sim = Self::jaccard(&word_sets[i], &word_sets[j]);
                if sim >= self.threshold {
                    // Keep the one with higher importance; break ties by recency.
                    let i_score = turns[i].importance_score;
                    let j_score = turns[j].importance_score;
                    if j_score > i_score {
                        keep[i] = false;
                        break; // i is dropped; no point comparing i further
                    } else {
                        keep[j] = false;
                    }
                }
            }
        }

        turns.into_iter().enumerate().filter(|(i, _)| keep[*i]).map(|(_, t)| t).collect()
    }

    /// Compute Jaccard similarity between two word sets.
    ///
    /// `J(A, B) = |A ∩ B| / |A ∪ B|`
    pub fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let intersection = a.intersection(b).count();
        let union = a.union(b).count();
        if union == 0 {
            return 1.0;
        }
        intersection as f32 / union as f32
    }

    /// Tokenise `text` into a lowercase word set, stripping punctuation.
    pub fn word_set(text: &str) -> HashSet<String> {
        text.split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }
}

// ── ImportanceFilter ──────────────────────────────────────────────────────────

/// Retains memories tagged as "important" regardless of age or score.
///
/// After applying [`TemporalCompression`] and [`SemanticDedup`], call
/// [`ImportanceFilter::reintegrate`] to ensure that any turn that was removed
/// but carries the importance tag is added back.
pub struct ImportanceFilter {
    /// The tag value that marks a turn as unconditionally important.
    pub importance_tag: String,
    /// Minimum `importance_score` in `[0.0, 1.0]` that also qualifies a turn
    /// for unconditional retention (independent of tags).
    pub min_score: f32,
}

impl Default for ImportanceFilter {
    fn default() -> Self {
        Self {
            importance_tag: "important".to_string(),
            min_score: 0.8,
        }
    }
}

impl ImportanceFilter {
    /// Create a new filter with explicit parameters.
    pub fn new(importance_tag: impl Into<String>, min_score: f32) -> Self {
        Self {
            importance_tag: importance_tag.into(),
            min_score: min_score.clamp(0.0, 1.0),
        }
    }

    /// Returns `true` when `turn` must be kept regardless of other strategies.
    pub fn is_important(&self, turn: &MemoryTurn) -> bool {
        turn.tags.iter().any(|t| t == &self.importance_tag)
            || turn.importance_score >= self.min_score
    }

    /// Given the full original `all_turns` and a `kept` subset produced by
    /// prior compression steps, ensure that every important turn from
    /// `all_turns` that is missing from `kept` is re-inserted at the correct
    /// chronological position.
    pub fn reintegrate(&self, all_turns: &[MemoryTurn], mut kept: Vec<MemoryTurn>) -> Vec<MemoryTurn> {
        let kept_ids: HashSet<&str> = kept.iter().map(|t| t.id.as_str()).collect();

        let rescued: Vec<MemoryTurn> = all_turns
            .iter()
            .filter(|t| !kept_ids.contains(t.id.as_str()) && self.is_important(t))
            .cloned()
            .collect();

        if rescued.is_empty() {
            return kept;
        }

        // Merge rescued turns back in chronological order.
        kept.extend(rescued);
        kept.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        kept
    }
}

// ── CompressorConfig ──────────────────────────────────────────────────────────

/// Configuration for the high-level [`MemoryCompressor`].
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Compression is only triggered when the turn count exceeds this value.
    pub max_turns: usize,
    /// How many of the most-recent turns to always keep verbatim.
    pub recency_keep: usize,
    /// Jaccard similarity threshold for semantic deduplication.
    /// Set to `1.0` to disable deduplication.
    pub jaccard_dedup_threshold: f32,
    /// Tag string used to mark unconditionally important turns.
    pub importance_tag: String,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            max_turns: 50,
            recency_keep: 10,
            jaccard_dedup_threshold: 0.85,
            importance_tag: "important".to_string(),
        }
    }
}

// ── MemoryCompressor ──────────────────────────────────────────────────────────

/// High-level memory compressor that chains [`TemporalCompression`],
/// [`SemanticDedup`], and [`ImportanceFilter`] in sequence.
///
/// Call [`MemoryCompressor::run_if_needed`] from an async background task
/// (e.g. via `tokio::spawn`) to compress only when the buffer exceeds
/// [`CompressorConfig::max_turns`].
pub struct MemoryCompressor {
    config: CompressorConfig,
    temporal: TemporalCompression,
    dedup: SemanticDedup,
    importance: ImportanceFilter,
}

impl MemoryCompressor {
    /// Create a new [`MemoryCompressor`] from the given configuration.
    pub fn new(config: CompressorConfig) -> Self {
        let temporal = TemporalCompression::new(config.recency_keep, 5);
        let dedup = SemanticDedup::new(config.jaccard_dedup_threshold);
        let importance = ImportanceFilter::new(config.importance_tag.clone(), 0.8);
        Self { config, temporal, dedup, importance }
    }

    /// Returns `true` when `turns.len() > config.max_turns`.
    pub fn needs_compression(&self, turns: &[MemoryTurn]) -> bool {
        turns.len() > self.config.max_turns
    }

    /// Compress `turns` unconditionally, applying all three strategies in
    /// order: temporal → semantic dedup → importance reintegration.
    ///
    /// Returns `(kept_turns, compressed_episodes)`.
    pub fn compress(&self, turns: Vec<MemoryTurn>) -> (Vec<MemoryTurn>, Vec<CompressedEpisode>) {
        if turns.is_empty() {
            return (vec![], vec![]);
        }

        let original = turns.clone();

        // Step 1: temporal compression.
        let (after_temporal, episodes) = self.temporal.compress(turns);

        // Step 2: semantic deduplication on remaining turns.
        let after_dedup = self.dedup.dedup(after_temporal);

        // Step 3: reintegrate any removed-but-important turns.
        let final_kept = self.importance.reintegrate(&original, after_dedup);

        (final_kept, episodes)
    }

    /// Run compression only when the buffer exceeds the threshold.
    ///
    /// This method is `async` so it can be called from a `tokio::spawn` task.
    /// It does no I/O; the `async` annotation is purely for ergonomic use in
    /// async contexts.
    ///
    /// Returns `(kept_turns, compressed_episodes, was_triggered)`.
    pub async fn run_if_needed(
        &self,
        turns: Vec<MemoryTurn>,
    ) -> (Vec<MemoryTurn>, Vec<CompressedEpisode>, bool) {
        if self.needs_compression(&turns) {
            let (kept, episodes) = self.compress(turns);
            (kept, episodes, true)
        } else {
            (turns, vec![], false)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::memory_compression::Role;

    fn make_turn(id: &str, content: &str, importance_score: f32, tags: Vec<&str>) -> MemoryTurn {
        MemoryTurn {
            id: id.to_string(),
            role: Role::User,
            content: content.to_string(),
            timestamp: SystemTime::now(),
            importance_score,
            token_count: content.len() / 4 + 1,
            tags: tags.into_iter().map(str::to_string).collect(),
        }
    }

    // ── SemanticDedup ─────────────────────────────────────────────────────────

    #[test]
    fn jaccard_identical_sets() {
        let a = SemanticDedup::word_set("hello world");
        let b = SemanticDedup::word_set("hello world");
        assert!((SemanticDedup::jaccard(&a, &b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = SemanticDedup::word_set("hello world");
        let b = SemanticDedup::word_set("foo bar baz");
        assert!((SemanticDedup::jaccard(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn semantic_dedup_removes_near_duplicates() {
        let dedup = SemanticDedup::new(0.8);
        let turns = vec![
            make_turn("t1", "the quick brown fox", 0.5, vec![]),
            make_turn("t2", "the quick brown fox jumps", 0.6, vec![]), // ~0.8 overlap
            make_turn("t3", "completely different content here", 0.5, vec![]),
        ];
        let result = dedup.dedup(turns);
        // t2 has higher score so t1 should be dropped; t3 is distinct and kept.
        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|t| t.id == "t2"));
        assert!(result.iter().any(|t| t.id == "t3"));
    }

    #[test]
    fn semantic_dedup_keeps_all_when_threshold_is_one() {
        let dedup = SemanticDedup::new(1.0);
        let turns = vec![
            make_turn("t1", "hello world", 0.5, vec![]),
            make_turn("t2", "hello world foo", 0.5, vec![]),
        ];
        let result = dedup.dedup(turns);
        assert_eq!(result.len(), 2);
    }

    // ── TemporalCompression ───────────────────────────────────────────────────

    #[test]
    fn temporal_keeps_recent_verbatim() {
        let tc = TemporalCompression::new(5, 3);
        let turns: Vec<MemoryTurn> = (0..15)
            .map(|i| make_turn(&format!("t{i}"), &format!("content {i}"), 0.0, vec![]))
            .collect();

        let (kept, episodes) = tc.compress(turns);
        assert_eq!(kept.len(), 5);
        assert!(!episodes.is_empty());
    }

    #[test]
    fn temporal_no_compression_when_within_recency_keep() {
        let tc = TemporalCompression::new(20, 5);
        let turns: Vec<MemoryTurn> = (0..10)
            .map(|i| make_turn(&format!("t{i}"), "hi", 0.0, vec![]))
            .collect();
        let (kept, episodes) = tc.compress(turns);
        assert_eq!(kept.len(), 10);
        assert!(episodes.is_empty());
    }

    #[test]
    fn compressed_episode_metadata_is_correct() {
        let turns: Vec<MemoryTurn> = (0..4)
            .map(|i| make_turn(&format!("t{i}"), "text", 0.0, vec!["topic"]))
            .collect();
        let ep = CompressedEpisode::from_batch("ep-0", 4, 1, &turns, "summary");
        assert_eq!(ep.original_count, 4);
        assert_eq!(ep.compressed_to, 1);
        assert!(ep.common_tags.contains(&"topic".to_string()));
    }

    // ── ImportanceFilter ──────────────────────────────────────────────────────

    #[test]
    fn importance_filter_identifies_tagged_turns() {
        let filter = ImportanceFilter::default();
        let important = make_turn("t1", "critical error occurred", 0.0, vec!["important"]);
        let normal = make_turn("t2", "regular update", 0.0, vec![]);
        assert!(filter.is_important(&important));
        assert!(!filter.is_important(&normal));
    }

    #[test]
    fn importance_filter_identifies_high_score_turns() {
        let filter = ImportanceFilter::new("important", 0.9);
        let high = make_turn("t1", "anything", 0.95, vec![]);
        let low = make_turn("t2", "anything", 0.5, vec![]);
        assert!(filter.is_important(&high));
        assert!(!filter.is_important(&low));
    }

    #[test]
    fn importance_filter_reintegrates_removed_turns() {
        let filter = ImportanceFilter::default();
        let all = vec![
            make_turn("t1", "normal", 0.0, vec![]),
            make_turn("t2", "critical", 0.0, vec!["important"]),
            make_turn("t3", "normal2", 0.0, vec![]),
        ];
        // Simulate compression that dropped t2.
        let kept = vec![
            make_turn("t1", "normal", 0.0, vec![]),
            make_turn("t3", "normal2", 0.0, vec![]),
        ];
        let result = filter.reintegrate(&all, kept);
        assert_eq!(result.len(), 3);
        assert!(result.iter().any(|t| t.id == "t2"));
    }

    // ── MemoryCompressor (high-level) ─────────────────────────────────────────

    #[test]
    fn memory_compressor_does_not_trigger_below_threshold() {
        let config = CompressorConfig { max_turns: 50, ..CompressorConfig::default() };
        let compressor = MemoryCompressor::new(config);
        let turns: Vec<MemoryTurn> = (0..10)
            .map(|i| make_turn(&format!("t{i}"), "hello", 0.0, vec![]))
            .collect();
        assert!(!compressor.needs_compression(&turns));
    }

    #[test]
    fn memory_compressor_triggers_above_threshold() {
        let config = CompressorConfig { max_turns: 5, ..CompressorConfig::default() };
        let compressor = MemoryCompressor::new(config);
        let turns: Vec<MemoryTurn> = (0..10)
            .map(|i| make_turn(&format!("t{i}"), "hello", 0.0, vec![]))
            .collect();
        assert!(compressor.needs_compression(&turns));
    }

    #[test]
    fn memory_compressor_full_pipeline() {
        let config = CompressorConfig {
            max_turns: 5,
            recency_keep: 3,
            jaccard_dedup_threshold: 0.9,
            importance_tag: "important".into(),
        };
        let compressor = MemoryCompressor::new(config);
        let mut turns: Vec<MemoryTurn> = (0..10)
            .map(|i| make_turn(&format!("t{i}"), &format!("msg {i}"), 0.0, vec![]))
            .collect();
        // Mark t0 as important.
        turns[0].tags.push("important".into());

        let (kept, episodes) = compressor.compress(turns);
        // t0 should survive via reintegration.
        assert!(kept.iter().any(|t| t.id == "t0"));
        // Episodes should capture the temporal windows.
        assert!(!episodes.is_empty());
    }

    #[tokio::test]
    async fn run_if_needed_returns_untouched_when_below_threshold() {
        let config = CompressorConfig { max_turns: 100, ..CompressorConfig::default() };
        let compressor = MemoryCompressor::new(config);
        let turns: Vec<MemoryTurn> = (0..5)
            .map(|i| make_turn(&format!("t{i}"), "hi", 0.0, vec![]))
            .collect();
        let (kept, episodes, triggered) = compressor.run_if_needed(turns).await;
        assert_eq!(kept.len(), 5);
        assert!(episodes.is_empty());
        assert!(!triggered);
    }

    #[tokio::test]
    async fn run_if_needed_triggers_when_above_threshold() {
        let config = CompressorConfig { max_turns: 3, recency_keep: 2, ..CompressorConfig::default() };
        let compressor = MemoryCompressor::new(config);
        let turns: Vec<MemoryTurn> = (0..10)
            .map(|i| make_turn(&format!("t{i}"), &format!("content {i}"), 0.0, vec![]))
            .collect();
        let (_, _, triggered) = compressor.run_if_needed(turns).await;
        assert!(triggered);
    }
}
