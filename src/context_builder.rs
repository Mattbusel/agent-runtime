//! # Context Builder
//!
//! Build rich prompt context from multiple memory and knowledge sources.
//!
//! [`ContextBuilder`] collects [`ContextChunk`] items from various
//! [`ContextSource`] types, deduplicates overlapping content, filters by
//! relevance, applies source-weight adjustments, and greedily selects chunks
//! to fit within a token budget.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::context_builder::{
//!     ContextBuilder, ContextBuildConfig, ContextChunk, ContextSource,
//! };
//! use std::collections::HashMap;
//!
//! let mut builder = ContextBuilder::new();
//! builder.add_chunk(ContextChunk {
//!     source: ContextSource::ConversationHistory,
//!     content: "User asked about Rust async.".into(),
//!     tokens: 10,
//!     relevance_score: 0.9,
//!     priority: 5,
//! });
//!
//! let config = ContextBuildConfig {
//!     max_tokens: 200,
//!     min_relevance: 0.5,
//!     source_weights: HashMap::new(),
//! };
//!
//! let chunks = builder.build(&config);
//! assert!(!chunks.is_empty());
//! let prompt = ContextBuilder::to_prompt_string(&chunks);
//! assert!(prompt.contains("User asked about Rust async"));
//! ```

use std::collections::{HashMap, HashSet};

// ── ContextSource ─────────────────────────────────────────────────────────────

/// Origin type of a context chunk.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContextSource {
    /// Prior turns in the current conversation.
    ConversationHistory,
    /// Entries retrieved from the agent's persistent memory.
    AgentMemory,
    /// Outputs from previously executed tool calls.
    ToolResults,
    /// The top-level system prompt or persona definition.
    SystemPrompt,
    /// Structured user profile data (preferences, history).
    UserProfile,
    /// Facts retrieved from an external knowledge base.
    ExternalKnowledge,
}

impl ContextSource {
    /// Human-readable label used when formatting the prompt string.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ConversationHistory => "Conversation History",
            Self::AgentMemory => "Agent Memory",
            Self::ToolResults => "Tool Results",
            Self::SystemPrompt => "System Prompt",
            Self::UserProfile => "User Profile",
            Self::ExternalKnowledge => "External Knowledge",
        }
    }

    /// Key used in [`ContextBuildConfig::source_weights`] lookups.
    pub fn weight_key(&self) -> &'static str {
        match self {
            Self::ConversationHistory => "conversation_history",
            Self::AgentMemory => "agent_memory",
            Self::ToolResults => "tool_results",
            Self::SystemPrompt => "system_prompt",
            Self::UserProfile => "user_profile",
            Self::ExternalKnowledge => "external_knowledge",
        }
    }
}

// ── ContextChunk ──────────────────────────────────────────────────────────────

/// A single unit of context to be considered for inclusion in the prompt.
#[derive(Debug, Clone)]
pub struct ContextChunk {
    /// Where this chunk came from.
    pub source: ContextSource,
    /// The text content of the chunk.
    pub content: String,
    /// Estimated number of LLM tokens occupied by `content`.
    pub tokens: usize,
    /// Relevance to the current query in `[0.0, 1.0]`.
    pub relevance_score: f64,
    /// Scheduling priority within the same relevance tier (higher = preferred).
    pub priority: u8,
}

// ── ContextBuildConfig ────────────────────────────────────────────────────────

/// Configuration controlling how [`ContextBuilder::build`] selects chunks.
#[derive(Debug, Clone)]
pub struct ContextBuildConfig {
    /// Maximum number of tokens the returned context may contain.
    pub max_tokens: usize,
    /// Chunks with `relevance_score < min_relevance` are excluded.
    pub min_relevance: f64,
    /// Per-source weight multipliers keyed by [`ContextSource::weight_key()`].
    /// Missing keys default to `1.0`.
    pub source_weights: HashMap<String, f64>,
}

impl Default for ContextBuildConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4_096,
            min_relevance: 0.0,
            source_weights: HashMap::new(),
        }
    }
}

// ── ContextBuilder ────────────────────────────────────────────────────────────

/// Aggregates context chunks and produces a filtered, weighted, token-budgeted
/// selection suitable for inclusion in an LLM prompt.
#[derive(Debug, Default)]
pub struct ContextBuilder {
    chunks: Vec<ContextChunk>,
}

impl ContextBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a context chunk to the builder's internal pool.
    pub fn add_chunk(&mut self, chunk: ContextChunk) {
        self.chunks.push(chunk);
    }

    /// Build the final context selection from the accumulated chunks.
    ///
    /// Steps applied in order:
    /// 1. Filter by `config.min_relevance`.
    /// 2. Deduplicate content (Jaccard overlap > 50%).
    /// 3. Score each chunk: `priority * relevance * source_weight`.
    /// 4. Sort descending by score.
    /// 5. Greedily select until `max_tokens` is reached.
    pub fn build(&self, config: &ContextBuildConfig) -> Vec<ContextChunk> {
        let filtered: Vec<ContextChunk> = self
            .chunks
            .iter()
            .filter(|c| c.relevance_score >= config.min_relevance)
            .cloned()
            .collect();

        let deduped = Self::deduplicate(&filtered);

        let default_weight = 1.0_f64;
        let mut scored: Vec<(f64, ContextChunk)> = deduped
            .into_iter()
            .map(|c| {
                let w = config
                    .source_weights
                    .get(c.source.weight_key())
                    .copied()
                    .unwrap_or(default_weight)
                    .max(0.0);
                let score = c.priority as f64 * c.relevance_score * w;
                (score, c)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected = Vec::new();
        let mut used = 0usize;
        for (_, chunk) in scored {
            if used.saturating_add(chunk.tokens) > config.max_tokens {
                continue;
            }
            used = used.saturating_add(chunk.tokens);
            selected.push(chunk);
        }
        selected
    }

    /// Remove chunks with word-level Jaccard similarity > 50% to any earlier chunk.
    ///
    /// Operates greedily: the first chunk in the slice is always kept; later
    /// chunks are dropped if they overlap too much with any already-kept chunk.
    pub fn deduplicate(chunks: &[ContextChunk]) -> Vec<ContextChunk> {
        let mut kept: Vec<ContextChunk> = Vec::new();
        for chunk in chunks {
            let words_new = word_set(&chunk.content);
            let is_dup = kept.iter().any(|k| {
                let words_kept = word_set(&k.content);
                jaccard(&words_new, &words_kept) > 0.5
            });
            if !is_dup {
                kept.push(chunk.clone());
            }
        }
        kept
    }

    /// Format a slice of chunks as a prompt string.
    ///
    /// Each chunk is rendered as:
    /// ```text
    /// ## <Source Label>
    /// <content>
    ///
    /// ```
    pub fn to_prompt_string(chunks: &[ContextChunk]) -> String {
        let mut out = String::new();
        for chunk in chunks {
            out.push_str("## ");
            out.push_str(chunk.source.label());
            out.push('\n');
            out.push_str(&chunk.content);
            out.push_str("\n\n");
        }
        out
    }

    /// Total token count of the given chunk slice.
    pub fn token_usage(chunks: &[ContextChunk]) -> usize {
        chunks.iter().map(|c| c.tokens).sum()
    }

    /// Token count broken down by source type label.
    pub fn coverage_report(chunks: &[ContextChunk]) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for chunk in chunks {
            let entry = map.entry(chunk.source.label().to_owned()).or_insert(0);
            *entry = entry.saturating_add(chunk.tokens);
        }
        map
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a lowercase word-set from a string (splits on non-alphanumeric chars).
fn word_set(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect()
}

/// Jaccard similarity between two word sets.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn chunk(
        source: ContextSource,
        content: &str,
        tokens: usize,
        relevance: f64,
        priority: u8,
    ) -> ContextChunk {
        ContextChunk {
            source,
            content: content.to_owned(),
            tokens,
            relevance_score: relevance,
            priority,
        }
    }

    #[test]
    fn test_add_and_build_basic() {
        let mut builder = ContextBuilder::new();
        builder.add_chunk(chunk(ContextSource::SystemPrompt, "You are a helpful assistant.", 10, 1.0, 10));
        builder.add_chunk(chunk(ContextSource::ConversationHistory, "User: Hello world!", 5, 0.8, 5));
        let config = ContextBuildConfig {
            max_tokens: 100,
            min_relevance: 0.0,
            source_weights: HashMap::new(),
        };
        let result = builder.build(&config);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_min_relevance_filter() {
        let mut builder = ContextBuilder::new();
        builder.add_chunk(chunk(ContextSource::AgentMemory, "Highly relevant.", 10, 0.9, 5));
        builder.add_chunk(chunk(ContextSource::ToolResults, "Low relevance.", 10, 0.2, 5));
        let config = ContextBuildConfig {
            max_tokens: 100,
            min_relevance: 0.5,
            source_weights: HashMap::new(),
        };
        let result = builder.build(&config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Highly relevant.");
    }

    #[test]
    fn test_token_budget() {
        let mut builder = ContextBuilder::new();
        for i in 0..5u8 {
            builder.add_chunk(chunk(
                ContextSource::ConversationHistory,
                &format!("Message {i} with unique alpha beta gamma delta epsilon zeta theta iota"),
                50,
                0.9,
                5,
            ));
        }
        let config = ContextBuildConfig {
            max_tokens: 120,
            min_relevance: 0.0,
            source_weights: HashMap::new(),
        };
        let result = builder.build(&config);
        let total = ContextBuilder::token_usage(&result);
        assert!(total <= 120);
    }

    #[test]
    fn test_deduplicate_identical() {
        let chunks = vec![
            chunk(ContextSource::AgentMemory, "the quick brown fox", 10, 1.0, 5),
            chunk(ContextSource::AgentMemory, "the quick brown fox", 10, 1.0, 5),
            chunk(ContextSource::AgentMemory, "completely different sentence here now", 10, 1.0, 5),
        ];
        let deduped = ContextBuilder::deduplicate(&chunks);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_deduplicate_partial_overlap_kept() {
        let chunks = vec![
            chunk(ContextSource::AgentMemory, "alpha beta gamma delta epsilon", 10, 1.0, 5),
            chunk(ContextSource::AgentMemory, "alpha zeta theta iota kappa lambda mu nu", 10, 1.0, 5),
        ];
        let deduped = ContextBuilder::deduplicate(&chunks);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_to_prompt_string() {
        let chunks = vec![chunk(ContextSource::SystemPrompt, "Be helpful.", 5, 1.0, 10)];
        let prompt = ContextBuilder::to_prompt_string(&chunks);
        assert!(prompt.contains("## System Prompt"));
        assert!(prompt.contains("Be helpful."));
    }

    #[test]
    fn test_token_usage() {
        let chunks = vec![
            chunk(ContextSource::AgentMemory, "a", 20, 1.0, 5),
            chunk(ContextSource::AgentMemory, "b", 30, 1.0, 5),
        ];
        assert_eq!(ContextBuilder::token_usage(&chunks), 50);
    }

    #[test]
    fn test_coverage_report() {
        let chunks = vec![
            chunk(ContextSource::AgentMemory, "a", 20, 1.0, 5),
            chunk(ContextSource::AgentMemory, "b", 30, 1.0, 5),
            chunk(ContextSource::SystemPrompt, "c", 10, 1.0, 10),
        ];
        let report = ContextBuilder::coverage_report(&chunks);
        assert_eq!(report.get("Agent Memory").copied().unwrap_or(0), 50);
        assert_eq!(report.get("System Prompt").copied().unwrap_or(0), 10);
    }

    #[test]
    fn test_source_weights_priority() {
        let mut builder = ContextBuilder::new();
        builder.add_chunk(chunk(ContextSource::AgentMemory, "memory unique content words set alpha", 10, 1.0, 5));
        builder.add_chunk(chunk(ContextSource::ExternalKnowledge, "knowledge different text here beta gamma", 10, 1.0, 5));
        let mut weights = HashMap::new();
        weights.insert("external_knowledge".to_owned(), 10.0);
        weights.insert("agent_memory".to_owned(), 1.0);
        let config = ContextBuildConfig {
            max_tokens: 10,
            min_relevance: 0.0,
            source_weights: weights,
        };
        let result = builder.build(&config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source, ContextSource::ExternalKnowledge);
    }

    #[test]
    fn test_empty_builder() {
        let builder = ContextBuilder::new();
        let config = ContextBuildConfig::default();
        let result = builder.build(&config);
        assert!(result.is_empty());
    }

    #[test]
    fn test_jaccard_identical_words() {
        let a: HashSet<String> = ["hello", "world"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard(&a, &a) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint_words() {
        let a: HashSet<String> = ["hello"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["world"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard(&a, &b) - 0.0).abs() < f64::EPSILON);
    }
}
