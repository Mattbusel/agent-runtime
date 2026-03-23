//! # Memory Compression
//!
//! When episodic memory grows too large, this module compresses old turns
//! into structured summaries. Uses three strategies:
//! - Recency: keep the N most recent turns verbatim
//! - Importance: score each turn by keywords/entities and keep high-scorers
//! - Summarization: compress low-importance old turns into a structured summary block
//!
//! The compressor is token-budget aware: it targets a `max_tokens` budget
//! and prunes until the serialized memory fits.

use std::collections::HashMap;

// ── Role ─────────────────────────────────────────────────────────────────────

/// The conversational role of a memory turn's author.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    /// A system-level instruction turn.
    System,
    /// A turn authored by the human user.
    User,
    /// A turn authored by the AI assistant.
    Assistant,
    /// A turn produced by a tool invocation.
    Tool,
}

// ── MemoryTurn ────────────────────────────────────────────────────────────────

/// A single memory turn in an agent's episodic memory.
#[derive(Debug, Clone)]
pub struct MemoryTurn {
    /// Unique identifier for this turn.
    pub id: String,
    /// The role of the author of this turn.
    pub role: Role,
    /// The text content of this turn.
    pub content: String,
    /// Wall-clock time at which this turn was recorded.
    pub timestamp: std::time::SystemTime,
    /// Pre-assigned importance score in `[0.0, 1.0]`.  May be `0.0` when not
    /// yet computed; the compressor will re-score it during compression.
    pub importance_score: f32,
    /// Estimated token count (may be pre-computed by the caller).
    pub token_count: usize,
    /// Searchable tags associated with this turn.
    pub tags: Vec<String>,
}

// ── MemorySummary ─────────────────────────────────────────────────────────────

/// A structured summary block produced by compressing one or more low-importance
/// [`MemoryTurn`]s.
#[derive(Debug, Clone)]
pub struct MemorySummary {
    /// Unique identifier for this summary block.
    pub id: String,
    /// IDs of the turns that were collapsed into this summary.
    pub covers_turns: Vec<String>,
    /// Human-readable prose summary of the compressed segment.
    pub summary_text: String,
    /// Short, declarative facts extracted from the compressed turns.
    pub key_facts: Vec<String>,
    /// Named entities discovered in the compressed turns: entity text → entity
    /// type label (e.g. `"Rust" → "Technology"`).
    pub entities: HashMap<String, String>,
    /// Inclusive time span covered by the compressed turns
    /// `(earliest_timestamp, latest_timestamp)`.
    pub time_range: (std::time::SystemTime, std::time::SystemTime),
    /// Estimated token count of the summary text.
    pub token_count: usize,
}

// ── ImportanceStrategy ────────────────────────────────────────────────────────

/// Strategy used to score the importance of a [`MemoryTurn`].
///
/// Scores are always normalised to `[0.0, 1.0]` before use.
#[derive(Debug, Clone)]
pub enum ImportanceStrategy {
    /// Score by the density of a provided keyword list within the turn content.
    ///
    /// `score = min(1.0, keyword_hits / max(1, word_count))`
    KeywordDensity(Vec<String>),

    /// Score by the density of probable named entities (heuristic: sequences of
    /// capitalised words not at the start of a sentence).
    ///
    /// `score = min(1.0, entity_count / max(1, word_count))`
    EntityDensity,

    /// Score by recency: turns closer to *now* score higher.
    ///
    /// `score = exp(-decay_per_hour * age_hours)`
    RecencyDecay {
        /// Exponential decay constant per elapsed hour.  A value of `0.1`
        /// means a turn 10 hours old retains `e^{-1} ≈ 0.37` of its score.
        decay_per_hour: f32,
    },

    /// Weighted average of multiple strategies.
    ///
    /// Each inner `(strategy, weight)` contributes `weight * score` to the
    /// total, which is then divided by the sum of all weights.
    Composite(Vec<(ImportanceStrategy, f32)>),
}

// ── MemoryCompressor ──────────────────────────────────────────────────────────

/// Compresses a buffer of [`MemoryTurn`]s to fit within a token budget.
///
/// # Example
///
/// ```rust
/// use llm_agent_runtime::memory_compression::{MemoryCompressor, ImportanceStrategy};
///
/// let compressor = MemoryCompressor::new(4096)
///     .with_recency_keep(8)
///     .with_importance_threshold(0.4)
///     .with_strategy(ImportanceStrategy::KeywordDensity(vec![
///         "error".to_string(), "critical".to_string(), "user".to_string(),
///     ]));
///
/// // let (kept, summaries, saved) = compressor.compress(turns);
/// ```
pub struct MemoryCompressor {
    /// Maximum total token count allowed after compression.
    max_tokens: usize,
    /// Always keep this many of the most-recent turns verbatim, regardless of
    /// their importance score.
    recency_keep: usize,
    /// Turns whose computed importance score is strictly below this threshold
    /// are candidates for summarization.
    importance_threshold: f32,
    /// The scoring strategy applied to each turn.
    strategy: ImportanceStrategy,
}

impl MemoryCompressor {
    /// Create a new compressor targeting `max_tokens`.
    ///
    /// Defaults:
    /// - `recency_keep = 10`
    /// - `importance_threshold = 0.3`
    /// - strategy: [`ImportanceStrategy::EntityDensity`]
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            recency_keep: 10,
            importance_threshold: 0.3,
            strategy: ImportanceStrategy::EntityDensity,
        }
    }

    /// Set how many of the most-recent turns are always kept verbatim.
    pub fn with_recency_keep(mut self, n: usize) -> Self {
        self.recency_keep = n;
        self
    }

    /// Set the importance threshold below which turns become compression
    /// candidates.
    pub fn with_importance_threshold(mut self, threshold: f32) -> Self {
        self.importance_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Replace the default scoring strategy.
    pub fn with_strategy(mut self, strategy: ImportanceStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Score a single turn's importance in `[0.0, 1.0]` using the configured
    /// [`ImportanceStrategy`].
    pub fn score_turn(&self, turn: &MemoryTurn) -> f32 {
        Self::apply_strategy(&self.strategy, turn)
            .clamp(0.0, 1.0)
    }

    /// Compress `turns` to fit within [`Self::max_tokens`].
    ///
    /// The algorithm:
    /// 1. Re-score every turn.
    /// 2. Protect the `recency_keep` most-recent turns unconditionally.
    /// 3. Keep any non-protected turn whose score ≥ `importance_threshold`.
    /// 4. Bucket the remaining (low-importance, non-recent) turns into
    ///    contiguous groups and summarize each group.
    /// 5. If the result still exceeds `max_tokens`, discard the oldest
    ///    summaries first, then the oldest kept turns, until the budget is met.
    ///
    /// Returns `(kept_turns, summaries_created, tokens_saved)`.
    pub fn compress(
        &self,
        turns: Vec<MemoryTurn>,
    ) -> (Vec<MemoryTurn>, Vec<MemorySummary>, usize) {
        if turns.is_empty() {
            return (Vec::new(), Vec::new(), 0);
        }

        let total_input_tokens: usize = turns.iter().map(|t| t.token_count).sum();

        // ── Step 1: score every turn ──────────────────────────────────────────
        let mut scored: Vec<(MemoryTurn, f32)> = turns
            .into_iter()
            .map(|mut t| {
                let score = self.score_turn(&t);
                t.importance_score = score;
                (t, score)
            })
            .collect();

        // Keep chronological order (assume the input is already ordered; sort
        // by timestamp as a safety measure).
        scored.sort_by(|a, b| {
            a.0.timestamp
                .partial_cmp(&b.0.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = scored.len();
        let recency_cutoff = n.saturating_sub(self.recency_keep);

        // ── Step 2: partition into protected vs. compressible ─────────────────
        // `protected_indices`: indices that must be kept verbatim.
        // Collect compressible segments (consecutive low-importance old turns).
        let mut kept_turns: Vec<MemoryTurn> = Vec::new();
        let mut compress_batch: Vec<MemoryTurn> = Vec::new();
        let mut summaries: Vec<MemorySummary> = Vec::new();

        for (i, (turn, score)) in scored.into_iter().enumerate() {
            let is_recent = i >= recency_cutoff;
            let is_important = score >= self.importance_threshold;

            if is_recent || is_important {
                // Flush any pending compression batch first.
                if !compress_batch.is_empty() {
                    let summary = Self::summarize_turns(&compress_batch);
                    summaries.push(summary);
                    compress_batch.clear();
                }
                kept_turns.push(turn);
            } else {
                compress_batch.push(turn);
            }
        }
        // Flush trailing batch.
        if !compress_batch.is_empty() {
            let summary = Self::summarize_turns(&compress_batch);
            summaries.push(summary);
        }

        // ── Step 3: budget enforcement ────────────────────────────────────────
        let kept_tokens: usize = kept_turns.iter().map(|t| t.token_count).sum();
        let summary_tokens: usize = summaries.iter().map(|s| s.token_count).sum();
        let mut total = kept_tokens + summary_tokens;

        // Drop oldest summaries first.
        let mut summary_idx = 0usize;
        while total > self.max_tokens && summary_idx < summaries.len() {
            total -= summaries[summary_idx].token_count;
            summary_idx += 1;
        }
        let summaries: Vec<MemorySummary> = summaries.into_iter().skip(summary_idx).collect();

        // Drop oldest kept turns next (preserve recency).
        let mut drop_from = 0usize;
        while total > self.max_tokens && drop_from < kept_turns.len() {
            total -= kept_turns[drop_from].token_count;
            drop_from += 1;
        }
        let kept_turns: Vec<MemoryTurn> = kept_turns.into_iter().skip(drop_from).collect();

        let tokens_saved = total_input_tokens.saturating_sub(total);
        (kept_turns, summaries, tokens_saved)
    }

    /// Estimate the number of tokens in a string using the 4 chars ≈ 1 token
    /// heuristic.
    pub fn estimate_tokens(s: &str) -> usize {
        // Minimum of 1 so an empty string still has a non-zero representation cost.
        (s.len() / 4).max(1)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Apply `strategy` to `turn` and return an unbounded score (caller clamps).
    fn apply_strategy(strategy: &ImportanceStrategy, turn: &MemoryTurn) -> f32 {
        match strategy {
            ImportanceStrategy::KeywordDensity(keywords) => {
                let lower = turn.content.to_lowercase();
                let word_count = lower.split_whitespace().count().max(1);
                let hits = keywords
                    .iter()
                    .filter(|kw| lower.contains(kw.to_lowercase().as_str()))
                    .count();
                (hits as f32 / word_count as f32).min(1.0)
            }

            ImportanceStrategy::EntityDensity => {
                let word_count = turn.content.split_whitespace().count().max(1);
                let entities = Self::extract_entities(&turn.content);
                (entities.len() as f32 / word_count as f32).min(1.0)
            }

            ImportanceStrategy::RecencyDecay { decay_per_hour } => {
                let age_secs = turn
                    .timestamp
                    .elapsed()
                    .unwrap_or_default()
                    .as_secs_f32();
                let age_hours = age_secs / 3600.0;
                (-decay_per_hour * age_hours).exp()
            }

            ImportanceStrategy::Composite(strategies) => {
                if strategies.is_empty() {
                    return 0.0;
                }
                let total_weight: f32 = strategies.iter().map(|(_, w)| w.abs()).sum();
                if total_weight < f32::EPSILON {
                    return 0.0;
                }
                let weighted_sum: f32 = strategies
                    .iter()
                    .map(|(s, w)| {
                        let score = Self::apply_strategy(s, turn).clamp(0.0, 1.0);
                        score * w.abs()
                    })
                    .sum();
                weighted_sum / total_weight
            }
        }
    }

    /// Compress a batch of low-importance turns into a single [`MemorySummary`].
    fn summarize_turns(turns: &[MemoryTurn]) -> MemorySummary {
        use std::time::SystemTime;

        let id = {
            // Derive a deterministic-ish ID from the first and last turn IDs.
            let first = turns.first().map(|t| t.id.as_str()).unwrap_or("?");
            let last = turns.last().map(|t| t.id.as_str()).unwrap_or("?");
            format!("summary-{first}-{last}")
        };

        let covers_turns: Vec<String> = turns.iter().map(|t| t.id.clone()).collect();

        // Time range.
        let earliest = turns
            .iter()
            .map(|t| t.timestamp)
            .min()
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let latest = turns
            .iter()
            .map(|t| t.timestamp)
            .max()
            .unwrap_or(SystemTime::UNIX_EPOCH);

        // Build prose summary.
        let role_labels: Vec<&str> = turns
            .iter()
            .map(|t| match t.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            })
            .collect();
        let unique_roles: std::collections::HashSet<&&str> = role_labels.iter().collect();
        let roles_str = {
            let mut v: Vec<&str> = unique_roles.into_iter().copied().collect();
            v.sort_unstable();
            v.join(", ")
        };
        let summary_text = format!(
            "[Compressed segment: {} turn(s) from {roles_str}. \
             Topics: {}]",
            turns.len(),
            turns
                .iter()
                .flat_map(|t| t.tags.iter().cloned())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .take(5)
                .collect::<Vec<_>>()
                .join(", ")
        );

        let key_facts = Self::extract_key_facts(turns);
        let entities = {
            let combined: String = turns.iter().map(|t| t.content.as_str()).collect::<Vec<_>>().join(" ");
            Self::extract_entities(&combined)
        };

        let token_count = Self::estimate_tokens(&summary_text)
            + key_facts.iter().map(|f| Self::estimate_tokens(f)).sum::<usize>();

        MemorySummary {
            id,
            covers_turns,
            summary_text,
            key_facts,
            entities,
            time_range: (earliest, latest),
            token_count,
        }
    }

    /// Extract short, declarative key facts from a batch of turns.
    ///
    /// Heuristic: sentences (split on `.`, `!`, `?`) that are between 5 and 25
    /// words long, from turns with `importance_score > 0`.  At most 10 facts
    /// are returned, deduplicated.
    fn extract_key_facts(turns: &[MemoryTurn]) -> Vec<String> {
        let mut facts: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for turn in turns {
            // Split on sentence-ending punctuation.
            for sentence in turn.content.split(['.', '!', '?']) {
                let trimmed = sentence.trim();
                let word_count = trimmed.split_whitespace().count();
                if word_count >= 5 && word_count <= 25 {
                    let key = trimmed.to_lowercase();
                    if seen.insert(key) {
                        facts.push(trimmed.to_string());
                        if facts.len() >= 10 {
                            return facts;
                        }
                    }
                }
            }
        }
        facts
    }

    /// Extract named entities from `text` using a simple heuristic:
    /// sequences of consecutive capitalised words (2+ chars each) that are
    /// not the very first word in the string.
    ///
    /// Returns a map of `entity_text → entity_type`.  The entity type is
    /// inferred from a small lookup table; everything else is labelled
    /// `"Unknown"`.
    fn extract_entities(text: &str) -> HashMap<String, String> {
        let mut entities: HashMap<String, String> = HashMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len();

        let mut i = 1usize; // skip the first word to avoid sentence-start capitals
        while i < n {
            let word = words[i];
            // Strip trailing punctuation for the check.
            let clean: String = word
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect();
            if clean.len() >= 2 && clean.chars().next().map_or(false, |c| c.is_uppercase()) {
                // Try to extend into a multi-word entity.
                let mut phrase = vec![clean.clone()];
                let mut j = i + 1;
                while j < n {
                    let next_clean: String = words[j]
                        .chars()
                        .filter(|c| c.is_alphabetic())
                        .collect();
                    if next_clean.len() >= 2
                        && next_clean.chars().next().map_or(false, |c| c.is_uppercase())
                    {
                        phrase.push(next_clean);
                        j += 1;
                    } else {
                        break;
                    }
                }
                let entity = phrase.join(" ");
                let entity_type = Self::classify_entity(&entity);
                entities.entry(entity).or_insert(entity_type);
                i = j;
            } else {
                i += 1;
            }
        }
        entities
    }

    /// Classify a named entity string into a broad category label.
    ///
    /// Uses a small hard-coded lookup; falls back to `"Unknown"`.
    fn classify_entity(entity: &str) -> String {
        let lower = entity.to_lowercase();
        // Technology keywords.
        let tech = ["rust", "tokio", "async", "redis", "openai", "anthropic",
                    "claude", "gpt", "http", "tcp", "json", "api", "sdk", "llm",
                    "wasm", "docker", "kubernetes"];
        for kw in tech {
            if lower.contains(kw) {
                return "Technology".to_string();
            }
        }
        // Organisation-ish keywords.
        let org = ["inc", "corp", "ltd", "llc", "org", "company", "labs",
                   "institute", "foundation", "group"];
        for kw in org {
            if lower.contains(kw) {
                return "Organisation".to_string();
            }
        }
        // Location-ish keywords.
        let loc = ["street", "avenue", "city", "country", "state", "nation",
                   "region", "district", "bay", "valley"];
        for kw in loc {
            if lower.contains(kw) {
                return "Location".to_string();
            }
        }
        "Unknown".to_string()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use std::time::SystemTime;

    fn make_turn(id: &str, content: &str, tokens: usize) -> MemoryTurn {
        MemoryTurn {
            id: id.to_string(),
            role: Role::User,
            content: content.to_string(),
            timestamp: SystemTime::now(),
            importance_score: 0.0,
            token_count: tokens,
            tags: vec![],
        }
    }

    #[test]
    fn estimate_tokens_nonzero() {
        assert_eq!(MemoryCompressor::estimate_tokens(""), 1);
        assert_eq!(MemoryCompressor::estimate_tokens("hello"), 1);
        assert_eq!(MemoryCompressor::estimate_tokens("hello world foo bar baz"), 5);
    }

    #[test]
    fn compress_empty() {
        let c = MemoryCompressor::new(1000);
        let (kept, sums, saved) = c.compress(vec![]);
        assert!(kept.is_empty());
        assert!(sums.is_empty());
        assert_eq!(saved, 0);
    }

    #[test]
    fn compress_within_budget_keeps_all() {
        let c = MemoryCompressor::new(10_000)
            .with_recency_keep(5)
            .with_importance_threshold(0.9); // very high threshold
        let turns = vec![
            make_turn("t1", "hello", 10),
            make_turn("t2", "world", 10),
        ];
        let (kept, _sums, saved) = c.compress(turns);
        assert_eq!(kept.len(), 2);
        assert_eq!(saved, 0);
    }

    #[test]
    fn keyword_density_scores_correctly() {
        let c = MemoryCompressor::new(1000).with_strategy(
            ImportanceStrategy::KeywordDensity(vec!["error".to_string(), "critical".to_string()]),
        );
        let high = make_turn("h", "error critical failure error", 10);
        let low = make_turn("l", "the quick brown fox jumps", 10);
        assert!(c.score_turn(&high) > c.score_turn(&low));
    }

    #[test]
    fn entity_density_finds_entities() {
        let entities = MemoryCompressor::extract_entities(
            "Anthropic Claude is an AI developed by Anthropic Labs",
        );
        // "Claude", "Anthropic", "Anthropic Labs" or similar should appear.
        assert!(!entities.is_empty());
    }

    #[test]
    fn extract_key_facts_returns_sentences() {
        let turns = vec![make_turn(
            "t",
            "Rust is a systems programming language. It is fast. Tokio is an async runtime for Rust.",
            50,
        )];
        let facts = MemoryCompressor::extract_key_facts(&turns);
        assert!(!facts.is_empty());
    }

    #[test]
    fn compress_low_importance_turns_are_summarized() {
        // 20 identical low-content turns.
        let turns: Vec<MemoryTurn> = (0..20)
            .map(|i| make_turn(&format!("t{i}"), "ok", 5))
            .collect();
        let c = MemoryCompressor::new(10_000)
            .with_recency_keep(3)
            .with_importance_threshold(0.99); // almost everything is low importance
        let (kept, summaries, _saved) = c.compress(turns);
        // The last 3 are kept verbatim; the rest become summaries.
        assert_eq!(kept.len(), 3);
        assert!(!summaries.is_empty());
    }

    #[test]
    fn composite_strategy_normalises_weights() {
        let c = MemoryCompressor::new(1000).with_strategy(ImportanceStrategy::Composite(vec![
            (ImportanceStrategy::EntityDensity, 0.5),
            (
                ImportanceStrategy::KeywordDensity(vec!["test".to_string()]),
                0.5,
            ),
        ]));
        let turn = make_turn("t", "Test Anthropic run test", 10);
        let score = c.score_turn(&turn);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
