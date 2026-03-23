//! # Module: vector_memory
//!
//! ## Responsibility
//! Vector similarity memory search using cosine similarity.
//! Provides:
//! - `Embedding`: a `Vec<f32>` wrapper with L2 normalisation.
//! - `VectorMemory<T>`: stores `(Embedding, T)` pairs and retrieves
//!   the `top_k` most similar entries given a query embedding.
//! - `BagOfWordsEmbedder`: converts text to TF-IDF bag-of-words embeddings.
//! - `SemanticMemory<T>`: combines `VectorMemory<T>` + `BagOfWordsEmbedder`.
//!
//! ## Guarantees
//! - All similarity computations use pre-normalised embeddings (O(d) dot products).
//! - Never panics on empty inputs.
//! - `search` returns results ordered from most similar to least similar.

use std::collections::HashMap;

// ── Stopword list ──────────────────────────────────────────────────────────────

/// Common English stopwords excluded from bag-of-words embeddings.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
];

fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

// ── Embedding ──────────────────────────────────────────────────────────────────

/// A dense floating-point vector used for similarity search.
///
/// After calling [`Embedding::normalize`] the vector has unit L2 norm, which
/// makes cosine similarity equivalent to a dot product and avoids the
/// magnitude division on every search query.
#[derive(Debug, Clone)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Create an embedding from raw components.
    pub fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    /// Return the raw values slice.
    pub fn values(&self) -> &[f32] {
        &self.0
    }

    /// Return the dimensionality of this embedding.
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Normalise this embedding to unit L2 norm in place.
    ///
    /// If the norm is zero (all-zero vector) the embedding is left unchanged.
    pub fn normalize(&mut self) {
        let norm: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut self.0 {
                *v /= norm;
            }
        }
    }

    /// Return `true` if this embedding has already been normalised to unit norm.
    pub fn is_normalized(&self) -> bool {
        let norm_sq: f32 = self.0.iter().map(|x| x * x).sum();
        (norm_sq - 1.0).abs() < 1e-5
    }

    /// Compute the dot product with another embedding.
    ///
    /// When both embeddings are normalised this equals cosine similarity.
    /// Returns 0.0 if the dimensions differ.
    pub fn dot(&self, other: &Embedding) -> f32 {
        if self.0.len() != other.0.len() {
            return 0.0;
        }
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute cosine similarity with another embedding.
    ///
    /// Returns a value in `[-1.0, 1.0]`, or `0.0` when either vector is zero
    /// or they have different dimensions.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.0.len() != other.0.len() || self.0.is_empty() {
            return 0.0;
        }
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let mag_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }
        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(v: Vec<f32>) -> Self {
        Self(v)
    }
}

impl From<Embedding> for Vec<f32> {
    fn from(e: Embedding) -> Self {
        e.0
    }
}

// ── VectorMemory ───────────────────────────────────────────────────────────────

/// An in-memory vector store that retrieves items by cosine similarity.
///
/// Stores `(Embedding, T)` pairs and supports `top_k` approximate nearest
/// neighbour search using pre-normalised embeddings (O(n·d) brute force).
///
/// # Example
/// ```rust
/// use llm_agent_runtime::vector_memory::{Embedding, VectorMemory};
///
/// let mut store: VectorMemory<&str> = VectorMemory::new();
/// let mut emb = Embedding::new(vec![1.0, 0.0]);
/// emb.normalize();
/// store.insert(emb.clone(), "hello");
/// let results = store.search(&emb, 1);
/// assert_eq!(results.len(), 1);
/// ```
pub struct VectorMemory<T> {
    entries: Vec<(Embedding, T)>,
}

impl<T> VectorMemory<T> {
    /// Create an empty `VectorMemory`.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert an embedding-value pair into the store.
    ///
    /// The embedding is L2-normalised automatically before storage so that
    /// `search` can use simple dot products instead of full cosine similarity.
    pub fn insert(&mut self, mut embedding: Embedding, value: T) {
        embedding.normalize();
        self.entries.push((embedding, value));
    }

    /// Return the number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search for the `top_k` most similar entries to `query`.
    ///
    /// The query embedding is normalised internally before comparison; the
    /// stored embeddings are already normalised at insert time.  Similarity
    /// is cosine similarity (via dot product on normalised vectors).
    ///
    /// Returns up to `top_k` `(similarity, &T)` pairs ordered from highest
    /// to lowest similarity.  Returns an empty `Vec` when the store is empty
    /// or `top_k == 0`.
    pub fn search(&self, query: &Embedding, top_k: usize) -> Vec<(f32, &T)> {
        if self.entries.is_empty() || top_k == 0 {
            return Vec::new();
        }

        // Normalise the query before computing dot products.
        let mut q = query.clone();
        q.normalize();

        // Score every entry.
        let mut scored: Vec<(f32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, (emb, _))| (emb.dot(&q), i))
            .collect();

        // Sort descending by similarity; stable so insertion order breaks ties.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(score, idx)| (score, &self.entries[idx].1))
            .collect()
    }

    /// Clear all entries from the store.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<T> Default for VectorMemory<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── BagOfWordsEmbedder ─────────────────────────────────────────────────────────

/// Converts text into a TF-IDF bag-of-words [`Embedding`].
///
/// Maintains a vocabulary mapping tokens → dimension indices and accumulates
/// document frequencies for IDF weighting.  The output is L2-normalised before
/// return.
///
/// # IDF formula
///
/// `idf(t) = log(1 + N / (1 + df(t)))` (smoothed to avoid division by zero).
///
/// # Example
/// ```rust
/// use llm_agent_runtime::vector_memory::BagOfWordsEmbedder;
///
/// let mut embedder = BagOfWordsEmbedder::new();
/// let emb = embedder.embed("the quick brown fox");
/// assert!(!emb.values().is_empty());
/// ```
pub struct BagOfWordsEmbedder {
    /// Maps token → column index in the vocabulary.
    vocab: HashMap<String, usize>,
    /// Document frequency for each token (how many texts contained it).
    doc_freq: HashMap<String, usize>,
    /// Total number of documents embedded so far (used for IDF).
    doc_count: usize,
}

impl BagOfWordsEmbedder {
    /// Create a new embedder with an empty vocabulary.
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            doc_freq: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Return the current vocabulary size (number of unique tokens seen).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Tokenise `text` into lowercased, stopword-filtered tokens.
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase())
            .filter(|t| !is_stopword(t.as_str()) && t.len() >= 2)
            .collect()
    }

    /// Embed `text` as a TF-IDF bag-of-words vector.
    ///
    /// Updates the internal vocabulary and document-frequency table with
    /// the new text.  The returned `Embedding` is L2-normalised.
    pub fn embed(&mut self, text: &str) -> Embedding {
        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return Embedding::new(vec![]);
        }

        // Count term frequencies in this document.
        let mut tf: HashMap<String, f32> = HashMap::new();
        for tok in &tokens {
            *tf.entry(tok.clone()).or_insert(0.0) += 1.0;
        }

        // Normalise TF by document length.
        let doc_len = tokens.len() as f32;
        for v in tf.values_mut() {
            *v /= doc_len;
        }

        // Update vocabulary and document frequencies.
        for tok in tf.keys() {
            let next_id = self.vocab.len();
            self.vocab.entry(tok.clone()).or_insert(next_id);
            *self.doc_freq.entry(tok.clone()).or_insert(0) += 1;
        }
        self.doc_count += 1;

        // Build the TF-IDF vector across the full vocabulary.
        let dim = self.vocab.len();
        let mut values = vec![0.0_f32; dim];
        let n = self.doc_count as f32;

        for (tok, &tf_val) in &tf {
            if let Some(&idx) = self.vocab.get(tok) {
                let df = *self.doc_freq.get(tok).unwrap_or(&1) as f32;
                let idf = (1.0 + n / (1.0 + df)).ln();
                if idx < values.len() {
                    values[idx] = tf_val * idf;
                }
            }
        }

        let mut emb = Embedding::new(values);
        emb.normalize();
        emb
    }

    /// Embed `text` without updating internal state (read-only query).
    ///
    /// Uses the existing vocabulary; tokens not in the vocabulary are ignored.
    /// The returned `Embedding` is L2-normalised.
    pub fn embed_query(&self, text: &str) -> Embedding {
        if self.vocab.is_empty() {
            return Embedding::new(vec![]);
        }

        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return Embedding::new(vec![0.0_f32; self.vocab.len()]);
        }

        let mut tf: HashMap<&str, f32> = HashMap::new();
        for tok in &tokens {
            *tf.entry(tok.as_str()).or_insert(0.0) += 1.0;
        }
        let doc_len = tokens.len() as f32;
        for v in tf.values_mut() {
            *v /= doc_len;
        }

        let dim = self.vocab.len();
        let mut values = vec![0.0_f32; dim];
        let n = self.doc_count.max(1) as f32;

        for (tok, &tf_val) in &tf {
            if let Some(&idx) = self.vocab.get(*tok) {
                let df = *self.doc_freq.get(*tok).unwrap_or(&1) as f32;
                let idf = (1.0 + n / (1.0 + df)).ln();
                if idx < values.len() {
                    values[idx] = tf_val * idf;
                }
            }
        }

        let mut emb = Embedding::new(values);
        emb.normalize();
        emb
    }
}

impl Default for BagOfWordsEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

// ── SemanticMemory ─────────────────────────────────────────────────────────────

/// High-level semantic memory: text-keyed storage with similarity recall.
///
/// Combines a [`BagOfWordsEmbedder`] (text → embedding) with a
/// [`VectorMemory<T>`] (embedding → stored value) for a convenient interface
/// where you `remember(text, value)` and `recall_similar(query, top_k)`.
///
/// # Example
/// ```rust
/// use llm_agent_runtime::vector_memory::SemanticMemory;
///
/// let mut mem: SemanticMemory<String> = SemanticMemory::new();
/// mem.remember("the cat sat on the mat", "cats are comfortable".to_string());
/// mem.remember("dogs love to run outside", "dogs are energetic".to_string());
/// let results = mem.recall_similar("feline resting", 1);
/// assert_eq!(results.len(), 1);
/// ```
pub struct SemanticMemory<T> {
    embedder: BagOfWordsEmbedder,
    store: VectorMemory<T>,
}

impl<T> SemanticMemory<T> {
    /// Create an empty `SemanticMemory`.
    pub fn new() -> Self {
        Self {
            embedder: BagOfWordsEmbedder::new(),
            store: VectorMemory::new(),
        }
    }

    /// Store `value` associated with the semantic content of `text`.
    ///
    /// Embeds `text` via the internal [`BagOfWordsEmbedder`] and inserts the
    /// resulting (normalised) embedding into the vector store.
    pub fn remember(&mut self, text: &str, value: T) {
        let embedding = self.embedder.embed(text);
        self.store.insert(embedding, value);
    }

    /// Return the `top_k` most semantically similar stored values to `query`.
    ///
    /// The query is embedded using the current (read-only) vocabulary; tokens
    /// not in the vocabulary contribute nothing to the score.  Results are
    /// ordered from highest to lowest cosine similarity.
    pub fn recall_similar(&self, query: &str, top_k: usize) -> Vec<(f32, &T)> {
        let query_emb = self.embedder.embed_query(query);
        if query_emb.dim() == 0 {
            return Vec::new();
        }
        self.store.search(&query_emb, top_k)
    }

    /// Return the number of stored memories.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` if no memories have been stored.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Return a reference to the underlying embedder.
    pub fn embedder(&self) -> &BagOfWordsEmbedder {
        &self.embedder
    }

    /// Return a reference to the underlying vector store.
    pub fn vector_store(&self) -> &VectorMemory<T> {
        &self.store
    }
}

impl<T> Default for SemanticMemory<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // ── Embedding tests ──────────────────────────────────────────────────────

    #[test]
    fn test_embedding_normalize_produces_unit_vector() {
        let mut emb = Embedding::new(vec![3.0, 4.0]);
        emb.normalize();
        let norm: f32 = emb.values().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn test_embedding_normalize_zero_vector_unchanged() {
        let mut emb = Embedding::new(vec![0.0, 0.0, 0.0]);
        emb.normalize();
        // Should remain zero
        assert!(emb.values().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embedding_dot_product_orthogonal() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0]);
        assert!((a.dot(&b)).abs() < 1e-10);
    }

    #[test]
    fn test_embedding_dot_product_parallel() {
        let mut a = Embedding::new(vec![1.0, 0.0]);
        a.normalize();
        let mut b = Embedding::new(vec![2.0, 0.0]);
        b.normalize();
        assert!((a.dot(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_dot_different_dimensions_returns_zero() {
        let a = Embedding::new(vec![1.0, 2.0]);
        let b = Embedding::new(vec![1.0]);
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = Embedding::new(vec![1.0, 2.0, 3.0]);
        let b = Embedding::new(vec![1.0, 2.0, 3.0]);
        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![-1.0, 0.0]);
        let sim = a.cosine_similarity(&b);
        assert!((sim + 1.0).abs() < 1e-5, "expected ~-1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&b);
        assert!(sim.abs() < 1e-5, "expected ~0.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_zero_vector_returns_zero() {
        let a = Embedding::new(vec![0.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0]);
        let sim = a.cosine_similarity(&b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_embedding_dim() {
        let emb = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dim(), 3);
    }

    // ── VectorMemory tests ───────────────────────────────────────────────────

    #[test]
    fn test_vector_memory_insert_and_len() {
        let mut mem: VectorMemory<String> = VectorMemory::new();
        assert!(mem.is_empty());
        mem.insert(Embedding::new(vec![1.0, 0.0]), "hello".to_string());
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn test_vector_memory_search_empty_returns_empty() {
        let mem: VectorMemory<String> = VectorMemory::new();
        let q = Embedding::new(vec![1.0, 0.0]);
        assert!(mem.search(&q, 5).is_empty());
    }

    #[test]
    fn test_vector_memory_search_top_k_zero_returns_empty() {
        let mut mem: VectorMemory<String> = VectorMemory::new();
        mem.insert(Embedding::new(vec![1.0, 0.0]), "x".to_string());
        let q = Embedding::new(vec![1.0, 0.0]);
        assert!(mem.search(&q, 0).is_empty());
    }

    #[test]
    fn test_vector_memory_search_finds_most_similar() {
        let mut mem: VectorMemory<&str> = VectorMemory::new();
        // First entry: points right (+x)
        mem.insert(Embedding::new(vec![1.0, 0.0]), "right");
        // Second entry: points up (+y)
        mem.insert(Embedding::new(vec![0.0, 1.0]), "up");

        // Query pointing right
        let q = Embedding::new(vec![1.0, 0.0]);
        let results = mem.search(&q, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(*results[0].1, "right");
        assert_eq!(*results[1].1, "up");
    }

    #[test]
    fn test_vector_memory_search_top_k_limits_results() {
        let mut mem: VectorMemory<i32> = VectorMemory::new();
        for i in 0..10 {
            mem.insert(Embedding::new(vec![i as f32, 0.0]), i);
        }
        let q = Embedding::new(vec![1.0, 0.0]);
        let results = mem.search(&q, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_vector_memory_search_results_ordered_by_similarity() {
        let mut mem: VectorMemory<&str> = VectorMemory::new();
        mem.insert(Embedding::new(vec![1.0, 0.0, 0.0]), "x");
        mem.insert(Embedding::new(vec![0.0, 1.0, 0.0]), "y");
        mem.insert(Embedding::new(vec![0.0, 0.0, 1.0]), "z");

        // Query close to "x"
        let q = Embedding::new(vec![0.9, 0.1, 0.0]);
        let results = mem.search(&q, 3);
        assert_eq!(*results[0].1, "x");
        // Similarity scores should be descending
        assert!(results[0].0 >= results[1].0);
        assert!(results[1].0 >= results[2].0);
    }

    #[test]
    fn test_vector_memory_clear() {
        let mut mem: VectorMemory<i32> = VectorMemory::new();
        mem.insert(Embedding::new(vec![1.0]), 1);
        mem.clear();
        assert!(mem.is_empty());
    }

    // ── BagOfWordsEmbedder tests ─────────────────────────────────────────────

    #[test]
    fn test_bow_embedder_returns_normalised_embedding() {
        let mut embedder = BagOfWordsEmbedder::new();
        let emb = embedder.embed("hello world test");
        if emb.dim() > 0 {
            let norm: f32 = emb.values().iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "not normalised: norm={norm}");
        }
    }

    #[test]
    fn test_bow_embedder_empty_text_returns_empty_embedding() {
        let mut embedder = BagOfWordsEmbedder::new();
        let emb = embedder.embed("a an the"); // all stopwords
        // May have 0-dim or all-zero vector
        assert!(emb.dim() == 0 || emb.values().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bow_embedder_vocab_grows_with_new_tokens() {
        let mut embedder = BagOfWordsEmbedder::new();
        embedder.embed("apple banana cherry");
        let v1 = embedder.vocab_size();
        embedder.embed("dragon elderberry fig");
        let v2 = embedder.vocab_size();
        assert!(v2 > v1);
    }

    #[test]
    fn test_bow_embedder_similar_texts_have_high_similarity() {
        let mut embedder = BagOfWordsEmbedder::new();
        let e1 = embedder.embed("machine learning neural networks deep");
        let e2 = embedder.embed("machine learning neural networks training");
        let sim = e1.cosine_similarity(&e2);
        // Texts share many tokens so similarity should be reasonably high.
        assert!(sim > 0.5, "expected sim > 0.5, got {sim}");
    }

    #[test]
    fn test_bow_embedder_different_texts_have_low_similarity() {
        let mut embedder = BagOfWordsEmbedder::new();
        let e1 = embedder.embed("apple banana mango fruit salad");
        let e2 = embedder.embed("quantum physics spacetime relativity electrons");
        let sim = e1.cosine_similarity(&e2);
        // Texts share no tokens so similarity should be very low.
        assert!(sim < 0.3, "expected sim < 0.3, got {sim}");
    }

    // ── SemanticMemory tests ─────────────────────────────────────────────────

    #[test]
    fn test_semantic_memory_remember_increases_len() {
        let mut mem: SemanticMemory<String> = SemanticMemory::new();
        assert_eq!(mem.len(), 0);
        mem.remember("hello world", "greeting".to_string());
        assert_eq!(mem.len(), 1);
        mem.remember("goodbye world", "farewell".to_string());
        assert_eq!(mem.len(), 2);
    }

    #[test]
    fn test_semantic_memory_recall_empty_returns_empty() {
        let mem: SemanticMemory<String> = SemanticMemory::new();
        let results = mem.recall_similar("hello", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_semantic_memory_recall_returns_similar() {
        let mut mem: SemanticMemory<String> = SemanticMemory::new();
        mem.remember("cats are fluffy animals", "cat_fact".to_string());
        mem.remember("quantum mechanics uncertainty principle", "physics".to_string());
        let results = mem.recall_similar("fluffy cat pets", 2);
        assert!(!results.is_empty());
        // The cat-related entry should be more similar.
        assert_eq!(*results[0].1, "cat_fact");
    }

    #[test]
    fn test_semantic_memory_recall_top_k_limits_results() {
        let mut mem: SemanticMemory<i32> = SemanticMemory::new();
        for i in 0..10 {
            mem.remember(&format!("topic number {i} content words"), i);
        }
        let results = mem.recall_similar("topic content", 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_semantic_memory_is_empty_initially() {
        let mem: SemanticMemory<String> = SemanticMemory::new();
        assert!(mem.is_empty());
    }

    #[test]
    fn test_semantic_memory_not_empty_after_remember() {
        let mut mem: SemanticMemory<String> = SemanticMemory::new();
        mem.remember("hello", "hi".to_string());
        assert!(!mem.is_empty());
    }

    #[test]
    fn test_semantic_memory_recall_results_ordered_descending() {
        let mut mem: SemanticMemory<String> = SemanticMemory::new();
        mem.remember("rust programming language fast safe", "rust".to_string());
        mem.remember("python scripting dynamic slow", "python".to_string());
        mem.remember("cooking baking food recipes chef", "cooking".to_string());
        let results = mem.recall_similar("rust programming language", 3);
        // Scores should be descending.
        if results.len() >= 2 {
            assert!(results[0].0 >= results[1].0, "results not ordered");
        }
    }
}
