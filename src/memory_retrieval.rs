//! # Module: memory_retrieval
//!
//! ## Responsibility
//! BM25-based full-text retrieval over agent memory entries.
//!
//! ## Formula
//! score(D, Q) = Σ IDF(qᵢ) × tf(qᵢ, D) × (k1 + 1) / (tf(qᵢ, D) + k1 × (1 − b + b × |D| / avgdl))
//!
//! IDF(qᵢ) = ln((N − df(qᵢ) + 0.5) / (df(qᵢ) + 0.5) + 1)
//!
//! ## Guarantees
//! - All public methods are non-panicking.
//! - Tokenisation lowercases and splits on non-alphanumeric characters.

use std::collections::HashMap;

// ── Data structures ───────────────────────────────────────────────────────────

/// A single entry stored in the memory index.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: String,
    /// The text content of this memory.
    pub content: String,
    /// Categorical tags for filtering.
    pub tags: Vec<String>,
    /// Creation timestamp (milliseconds since epoch).
    pub created_at: u64,
    /// Number of times this entry has been retrieved.
    pub access_count: u32,
}

/// BM25 hyper-parameters.
#[derive(Debug, Clone)]
pub struct BM25Params {
    /// Term-frequency saturation parameter (default 1.5).
    pub k1: f64,
    /// Length normalisation factor (default 0.75).
    pub b: f64,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// A scored retrieval result wrapping a [`MemoryEntry`].
#[derive(Debug)]
pub struct RetrievalResult {
    /// The matched memory entry.
    pub entry: MemoryEntry,
    /// BM25 score.
    pub score: f64,
    /// 1-based rank in the result set.
    pub rank: usize,
}

// ── BM25Index ─────────────────────────────────────────────────────────────────

/// Inverted BM25 index over [`MemoryEntry`] documents.
pub struct BM25Index {
    params: BM25Params,
    /// Stored entries indexed by their id.
    entries: HashMap<String, MemoryEntry>,
    /// term -> set of entry ids that contain it.
    inverted: HashMap<String, Vec<String>>,
    /// entry id -> token -> count within that entry.
    tf: HashMap<String, HashMap<String, u32>>,
    /// Accumulated total document length (sum of token counts).
    total_doc_len: u64,
}

impl BM25Index {
    /// Create a new index with the given BM25 parameters.
    pub fn new(params: BM25Params) -> Self {
        Self {
            params,
            entries: HashMap::new(),
            inverted: HashMap::new(),
            tf: HashMap::new(),
            total_doc_len: 0,
        }
    }

    /// Create a new index with default BM25 parameters (k1=1.5, b=0.75).
    pub fn with_defaults() -> Self {
        Self::new(BM25Params::default())
    }

    /// Add a [`MemoryEntry`] to the index.
    ///
    /// If an entry with the same id already exists it is replaced.
    pub fn add(&mut self, entry: MemoryEntry) {
        // Remove old entry if present.
        let _ = self.remove(&entry.id);

        let tokens = tokenize(&entry.content);
        let doc_len = tokens.len() as u64;
        self.total_doc_len += doc_len;

        let mut term_freq: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }

        for token in term_freq.keys() {
            self.inverted
                .entry(token.clone())
                .or_default()
                .push(entry.id.clone());
        }

        self.tf.insert(entry.id.clone(), term_freq);
        self.entries.insert(entry.id.clone(), entry);
    }

    /// Remove an entry by id.  Returns `true` if the entry existed.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(entry) = self.entries.remove(id) {
            // Recompute doc_len from stored tf.
            if let Some(term_freq) = self.tf.remove(id) {
                let doc_len: u64 = term_freq.values().map(|&c| c as u64).sum();
                self.total_doc_len = self.total_doc_len.saturating_sub(doc_len);

                // Remove from inverted index.
                for token in term_freq.keys() {
                    if let Some(list) = self.inverted.get_mut(token) {
                        list.retain(|eid| eid != id);
                        if list.is_empty() {
                            self.inverted.remove(token);
                        }
                    }
                }
            }
            let _ = entry; // satisfy clippy
            true
        } else {
            false
        }
    }

    /// Search for entries matching `query`, returning up to `top_k` results
    /// sorted by BM25 score descending.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(f64, &MemoryEntry)> {
        if self.entries.is_empty() || top_k == 0 {
            return vec![];
        }

        let n = self.entries.len() as f64;
        let avgdl = if n > 0.0 {
            self.total_doc_len as f64 / n
        } else {
            1.0
        };

        let query_tokens = tokenize(query);
        let mut scores: HashMap<&str, f64> = HashMap::new();

        for token in &query_tokens {
            let df = self
                .inverted
                .get(token.as_str())
                .map(|v| v.len())
                .unwrap_or(0) as f64;

            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            if let Some(doc_ids) = self.inverted.get(token.as_str()) {
                for doc_id in doc_ids {
                    if let (Some(entry), Some(term_freq)) =
                        (self.entries.get(doc_id), self.tf.get(doc_id))
                    {
                        let tf_val = term_freq
                            .get(token.as_str())
                            .copied()
                            .unwrap_or(0) as f64;
                        let doc_len: f64 =
                            term_freq.values().map(|&c| c as f64).sum();

                        let k1 = self.params.k1;
                        let b = self.params.b;
                        let tf_norm = tf_val * (k1 + 1.0)
                            / (tf_val + k1 * (1.0 - b + b * doc_len / avgdl));
                        let score = idf * tf_norm;

                        *scores.entry(entry.id.as_str()).or_insert(0.0) += score;
                    }
                }
            }
        }

        let mut results: Vec<(f64, &MemoryEntry)> = scores
            .into_iter()
            .filter_map(|(id, score)| self.entries.get(id).map(|e| (score, e)))
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Return up to `top_k` entries that contain **all** of the given tags.
    pub fn search_by_tags<'a>(&'a self, tags: &[&str], top_k: usize) -> Vec<&'a MemoryEntry> {
        self.entries
            .values()
            .filter(|e| {
                tags.iter()
                    .all(|&tag| e.tags.iter().any(|t| t == tag))
            })
            .take(top_k)
            .collect()
    }

    /// Number of entries currently in the index.
    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens, splitting on
/// any non-alphanumeric character.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn entry(id: &str, content: &str, tags: Vec<&str>) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            created_at: 0,
            access_count: 0,
        }
    }

    #[test]
    fn add_and_size() {
        let mut idx = BM25Index::with_defaults();
        assert_eq!(idx.size(), 0);
        idx.add(entry("e1", "hello world", vec![]));
        assert_eq!(idx.size(), 1);
        idx.add(entry("e2", "foo bar", vec![]));
        assert_eq!(idx.size(), 2);
    }

    #[test]
    fn remove_existing_entry() {
        let mut idx = BM25Index::with_defaults();
        idx.add(entry("e1", "hello world", vec![]));
        assert!(idx.remove("e1"));
        assert_eq!(idx.size(), 0);
    }

    #[test]
    fn remove_nonexistent_returns_false() {
        let mut idx = BM25Index::with_defaults();
        assert!(!idx.remove("nope"));
    }

    #[test]
    fn search_returns_relevant_entry() {
        let mut idx = BM25Index::with_defaults();
        idx.add(entry("e1", "the quick brown fox", vec![]));
        idx.add(entry("e2", "lazy dog sleeping", vec![]));
        let results = idx.search("fox", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].1.id, "e1");
    }

    #[test]
    fn search_top_k_limits_results() {
        let mut idx = BM25Index::with_defaults();
        for i in 0..10u32 {
            idx.add(entry(
                &format!("e{i}"),
                &format!("document number {i} about rust"),
                vec![],
            ));
        }
        let results = idx.search("rust", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_scores_descending() {
        let mut idx = BM25Index::with_defaults();
        idx.add(entry("e1", "rust rust rust", vec![]));
        idx.add(entry("e2", "rust programming", vec![]));
        idx.add(entry("e3", "python is great", vec![]));
        let results = idx.search("rust", 5);
        // e1 or e2 should score higher than e3 (which has no match).
        assert!(results.iter().all(|(_, e)| e.id != "e3"));
        // Descending order check.
        for w in results.windows(2) {
            assert!(w[0].0 >= w[1].0);
        }
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let idx = BM25Index::with_defaults();
        assert!(idx.search("anything", 5).is_empty());
    }

    #[test]
    fn search_by_tags_filters_correctly() {
        let mut idx = BM25Index::with_defaults();
        idx.add(entry("e1", "data", vec!["ml", "nlp"]));
        idx.add(entry("e2", "code", vec!["rust"]));
        idx.add(entry("e3", "model", vec!["ml"]));

        let results = idx.search_by_tags(&["ml"], 10);
        let ids: Vec<&str> = results.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"e1"));
        assert!(ids.contains(&"e3"));
        assert!(!ids.contains(&"e2"));
    }

    #[test]
    fn search_by_tags_all_required() {
        let mut idx = BM25Index::with_defaults();
        idx.add(entry("e1", "data", vec!["ml", "nlp"]));
        idx.add(entry("e2", "code", vec!["ml"]));

        let results = idx.search_by_tags(&["ml", "nlp"], 10);
        let ids: Vec<&str> = results.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"e1"));
        assert!(!ids.contains(&"e2"));
    }

    #[test]
    fn replace_entry_updates_index() {
        let mut idx = BM25Index::with_defaults();
        idx.add(entry("e1", "old content about cats", vec![]));
        idx.add(entry("e1", "new content about dogs", vec![]));
        assert_eq!(idx.size(), 1);
        let results = idx.search("dogs", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].1.id, "e1");
    }

    #[test]
    fn tokenize_splits_on_punctuation() {
        let tokens = super::tokenize("hello, world! foo-bar");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
    }
}
