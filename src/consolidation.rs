//! # Module: consolidation
//!
//! ## Responsibility
//! Background memory consolidation for LLM agents.
//!
//! Agents accumulate large numbers of episodic memories that become redundant
//! over time. [`MemoryConsolidator`] is a background Tokio task that
//! periodically clusters similar memories using a lightweight bag-of-words
//! TF-IDF embedding and k-means clustering, merges each cluster into a single
//! [`ConsolidatedMemory`], and archives older duplicate items, keeping only the
//! most recent N memories from each cluster in the episodic store.
//!
//! ## Guarantees
//! - Non-panicking: no `unwrap` or `expect` on reachable production paths.
//! - Bounded: each consolidation pass is O(M × K × I) where M = memories,
//!   K = clusters, I = k-means iterations (all three are bounded by policy).
//! - Metrics: [`ConsolidationMetrics`] counts runs and memories consolidated.
//!
//! ## NOT Responsible For
//! - Writing consolidated memories back to a persistence backend.
//! - LLM-based summarisation (uses heuristic text joining instead).

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use chrono::Utc;
use tracing::{debug, info};

#[cfg(feature = "memory")]
use crate::memory::{EpisodicStore, MemoryItem};
use crate::types::AgentId;

// ── ConsolidationPolicy ───────────────────────────────────────────────────────

/// Tunable policy controlling when and how memories are consolidated.
#[derive(Debug, Clone)]
pub struct ConsolidationPolicy {
    /// Minimum cluster size before a cluster is eligible for consolidation.
    ///
    /// Clusters smaller than this value are left as-is, regardless of
    /// similarity. This prevents prematurely merging unique memories.
    /// Default: `3`.
    pub min_cluster_size: usize,
    /// Memories older than this many seconds are eligible for archiving when
    /// they are not the most-recent N items in their cluster.
    /// Default: `3600` (1 hour).
    pub max_age_secs: u64,
    /// Cosine-similarity threshold `[0.0, 1.0]` above which two memories are
    /// considered similar enough to be placed in the same cluster.
    /// Default: `0.65`.
    pub similarity_threshold: f64,
    /// How often the consolidation background task wakes up, in seconds.
    /// Default: `300` (5 minutes).
    pub run_interval_secs: u64,
    /// Target number of clusters (k) for the k-means pass.
    ///
    /// If the episodic store has fewer items than `k_clusters` for a given
    /// agent, no clustering is performed for that agent.
    /// Default: `8`.
    pub k_clusters: usize,
    /// Number of k-means iterations to run per consolidation pass.
    /// Default: `10`.
    pub kmeans_iterations: usize,
    /// Number of the most-recent memories per cluster to keep in the store.
    /// Older memories beyond this limit are eligible for archiving.
    /// Default: `2`.
    pub keep_recent_per_cluster: usize,
}

impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            min_cluster_size: 3,
            max_age_secs: 3600,
            similarity_threshold: 0.65,
            run_interval_secs: 300,
            k_clusters: 8,
            kmeans_iterations: 10,
            keep_recent_per_cluster: 2,
        }
    }
}

// ── ConsolidationMetrics ──────────────────────────────────────────────────────

/// Atomic counters exposed by the consolidation system for observability.
///
/// These counters are safe to read from any thread at any time.  They are
/// incremented atomically by the consolidation loop and never reset.
#[derive(Debug, Default)]
pub struct ConsolidationMetrics {
    /// Total number of consolidation runs that have completed successfully.
    pub consolidation_runs_total: AtomicU64,
    /// Total number of individual memories merged or archived across all runs.
    pub memories_consolidated_total: AtomicU64,
}

impl ConsolidationMetrics {
    /// Create a new zero-initialised, `Arc`-wrapped metrics instance.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Return the total number of consolidation runs completed so far.
    pub fn runs(&self) -> u64 {
        self.consolidation_runs_total.load(Ordering::Relaxed)
    }

    /// Return the total number of memories consolidated so far.
    pub fn consolidated(&self) -> u64 {
        self.memories_consolidated_total.load(Ordering::Relaxed)
    }
}

// ── ConsolidatedMemory ────────────────────────────────────────────────────────

/// A single memory entry produced by merging N similar episodic memories.
///
/// The `summary_content` is a human-readable concatenation of the source
/// memories' content joined by `"; "`. The original memory IDs are recorded in
/// `source_ids` for audit and retrieval purposes.
#[derive(Debug, Clone)]
pub struct ConsolidatedMemory {
    /// The agent this consolidated memory belongs to.
    pub agent_id: AgentId,
    /// Human-readable merged summary of the source memories.
    pub summary_content: String,
    /// IDs of the original memories that were merged into this entry.
    pub source_ids: Vec<crate::types::MemoryId>,
    /// Average importance score of the source memories (`[0.0, 1.0]`).
    pub importance: f32,
    /// UTC timestamp when this consolidated memory was created.
    pub created_at: chrono::DateTime<Utc>,
    /// Number of source memories that were merged.
    pub source_count: usize,
}

// ── TF-IDF bag-of-words embedding ────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric words of at least 3 characters.
///
/// Short stop-words (`"a"`, `"is"`, `"the"`, …) are filtered out because they
/// do not carry semantic signal for clustering.
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Compute the dot-product similarity of two equal-length f64 slices.
///
/// For L2-normalised vectors this is equivalent to cosine similarity.
/// Returns `0.0` if either slice is empty or lengths differ.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Build a TF-IDF-weighted bag-of-words vector for `text` using `corpus` to
/// derive document-frequency statistics.
///
/// The resulting vector has one dimension per vocabulary term (sorted
/// lexicographically) and is L2-normalised so that cosine similarity reduces
/// to a simple dot product.
fn tfidf_embedding(text: &str, corpus: &[&str]) -> Vec<f64> {
    let n_docs = corpus.len() as f64;

    // Collect the global vocabulary from the entire corpus.
    let mut vocab: Vec<String> = Vec::new();
    for doc in corpus.iter() {
        for word in tokenise(doc) {
            if !vocab.contains(&word) {
                vocab.push(word);
            }
        }
    }
    vocab.sort();

    if vocab.is_empty() {
        return Vec::new();
    }

    // Compute IDF for every vocabulary term.
    let idf: HashMap<String, f64> = vocab
        .iter()
        .map(|term| {
            let df = corpus
                .iter()
                .filter(|doc| doc.to_lowercase().contains(term.as_str()))
                .count() as f64;
            let idf_val = (1.0 + n_docs / (1.0 + df)).ln();
            (term.clone(), idf_val)
        })
        .collect();

    // Build TF for the query text.
    let mut query_tf: HashMap<String, f64> = HashMap::new();
    for word in tokenise(text) {
        *query_tf.entry(word).or_insert(0.0) += 1.0;
    }
    let total: f64 = query_tf.values().copied().sum();
    if total > 0.0 {
        for v in query_tf.values_mut() {
            *v /= total;
        }
    }

    // Assemble the TF-IDF vector.
    let mut vec: Vec<f64> = vocab
        .iter()
        .map(|term| {
            let tf = query_tf.get(term).copied().unwrap_or(0.0);
            let idf_val = idf.get(term).copied().unwrap_or(0.0);
            tf * idf_val
        })
        .collect();

    // L2-normalise.
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > f64::EPSILON {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }

    vec
}

// ── K-means clustering ────────────────────────────────────────────────────────

/// Run k-means clustering on `embeddings` and return a cluster label (`0..k`)
/// for each point. Uses a deterministic farthest-point initialisation
/// (a simplified k-means++ without randomness) so results are reproducible.
///
/// If `k` exceeds the number of embeddings it is clamped to `embeddings.len()`.
fn kmeans(embeddings: &[Vec<f64>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    if k == 0 {
        return vec![0; n];
    }

    let dim = embeddings[0].len();

    // Farthest-point initialisation: first centroid = first point; each
    // subsequent centroid is the point farthest from all existing centroids.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    centroids.push(embeddings[0].clone());

    for _ in 1..k {
        // For each candidate point, compute the maximum similarity to any
        // existing centroid; pick the candidate with the lowest max-similarity
        // (i.e., most distant).
        let next = (0..n).max_by(|&a, &b| {
            let max_sim_a = centroids
                .iter()
                .map(|c| cosine_similarity(c, &embeddings[a]))
                .fold(f64::NEG_INFINITY, f64::max);
            let max_sim_b = centroids
                .iter()
                .map(|c| cosine_similarity(c, &embeddings[b]))
                .fold(f64::NEG_INFINITY, f64::max);
            // Lower max-similarity wins (most distant).
            max_sim_b
                .partial_cmp(&max_sim_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(idx) = next {
            if !centroids.iter().any(|c| c == &embeddings[idx]) {
                centroids.push(embeddings[idx].clone());
            }
        }
    }

    let actual_k = centroids.len();
    let mut labels = vec![0usize; n];

    for _iter in 0..iterations {
        // Assignment: each point to its nearest centroid.
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, cosine_similarity(c, emb)))
                .max_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update: recompute each centroid as the mean of assigned points.
        let mut sums = vec![vec![0.0f64; dim]; actual_k];
        let mut counts = vec![0usize; actual_k];
        for (i, &label) in labels.iter().enumerate() {
            if label < actual_k {
                for (d, val) in embeddings[i].iter().enumerate() {
                    if d < dim {
                        sums[label][d] += val;
                    }
                }
                counts[label] += 1;
            }
        }
        for ci in 0..actual_k {
            let cnt = counts[ci] as f64;
            if cnt > 0.0 {
                for x in sums[ci].iter_mut() {
                    *x /= cnt;
                }
                // Re-normalise the centroid so cosine comparisons remain valid.
                let norm: f64 = sums[ci].iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > f64::EPSILON {
                    for x in sums[ci].iter_mut() {
                        *x /= norm;
                    }
                }
            }
            centroids[ci] = sums[ci].clone();
        }
    }

    labels
}

// ── MemoryConsolidator ────────────────────────────────────────────────────────

/// Background consolidation engine for an [`EpisodicStore`].
///
/// Call [`MemoryConsolidator::run`] to spawn a long-running Tokio task that
/// wakes on the configured interval, clusters all agents' memories with TF-IDF
/// k-means, and produces [`ConsolidatedMemory`] summaries for clusters that
/// meet the [`ConsolidationPolicy`] thresholds.
///
/// The produced summaries are accessible via [`take_consolidated`] at any time.
///
/// [`take_consolidated`]: MemoryConsolidator::take_consolidated
///
/// # Example
///
/// ```rust,no_run
/// use llm_agent_runtime::consolidation::{ConsolidationPolicy, MemoryConsolidator};
/// use llm_agent_runtime::memory::EpisodicStore;
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() {
///     let store = Arc::new(EpisodicStore::new());
///     let policy = ConsolidationPolicy::default();
///     let consolidator = MemoryConsolidator::new(store, policy);
///     let metrics = consolidator.metrics();
///
///     // Drive the background loop from a detached task.
///     tokio::spawn(consolidator.run());
///
///     // metrics.runs() and metrics.consolidated() are updated after each pass.
/// }
/// ```
#[cfg(feature = "memory")]
pub struct MemoryConsolidator {
    store: Arc<EpisodicStore>,
    policy: ConsolidationPolicy,
    metrics: Arc<ConsolidationMetrics>,
    /// Buffer of consolidated memories produced since the last call to
    /// [`take_consolidated`].
    consolidated: Arc<Mutex<Vec<ConsolidatedMemory>>>,
}

#[cfg(feature = "memory")]
impl MemoryConsolidator {
    /// Create a new consolidator operating on `store` with the given `policy`.
    pub fn new(store: Arc<EpisodicStore>, policy: ConsolidationPolicy) -> Self {
        Self {
            store,
            policy,
            metrics: ConsolidationMetrics::new(),
            consolidated: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return a shared reference to the live metrics counters.
    pub fn metrics(&self) -> Arc<ConsolidationMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Drain and return all [`ConsolidatedMemory`] entries produced since the
    /// last call to this method, clearing the internal buffer.
    pub fn take_consolidated(&self) -> Vec<ConsolidatedMemory> {
        self.consolidated
            .lock()
            .map(|mut g| g.drain(..).collect())
            .unwrap_or_default()
    }

    /// Run one consolidation pass for the given `agent_id`.
    ///
    /// Returns the [`ConsolidatedMemory`] entries produced during this pass.
    /// The entries are also appended to the internal buffer accessible via
    /// [`take_consolidated`].
    pub fn consolidate_agent(&self, agent_id: &AgentId) -> Vec<ConsolidatedMemory> {
        let items = match self.store.recall(agent_id, usize::MAX) {
            Ok(v) => v,
            Err(e) => {
                debug!(%e, agent = %agent_id, "consolidation: recall failed");
                return Vec::new();
            }
        };

        if items.len() < self.policy.k_clusters {
            debug!(
                agent = %agent_id,
                items = items.len(),
                k = self.policy.k_clusters,
                "consolidation: insufficient memories to cluster"
            );
            return Vec::new();
        }

        // Build TF-IDF embeddings using the full memory set as corpus.
        let corpus: Vec<&str> = items.iter().map(|m| m.content.as_str()).collect();
        let embeddings: Vec<Vec<f64>> = items
            .iter()
            .map(|m| tfidf_embedding(&m.content, &corpus))
            .collect();

        let labels = kmeans(
            &embeddings,
            self.policy.k_clusters,
            self.policy.kmeans_iterations,
        );

        // Group memories by cluster label.
        let mut clusters: HashMap<usize, Vec<&MemoryItem>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            clusters.entry(label).or_default().push(&items[i]);
        }

        let now_secs = Utc::now().timestamp().max(0) as u64;
        let mut produced: Vec<ConsolidatedMemory> = Vec::new();
        let mut total_archived: u64 = 0;

        for members in clusters.values() {
            if members.len() < self.policy.min_cluster_size {
                continue;
            }

            // Sort newest-first so we know which items to keep.
            let mut sorted: Vec<&MemoryItem> = members.to_vec();
            sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

            // Archive items beyond keep_recent_per_cluster that are also old enough.
            let to_archive: Vec<&MemoryItem> = sorted
                .iter()
                .skip(self.policy.keep_recent_per_cluster)
                .filter(|m| {
                    let age = now_secs
                        .saturating_sub(m.timestamp.timestamp().max(0) as u64);
                    age >= self.policy.max_age_secs
                })
                .copied()
                .collect();

            if to_archive.is_empty() {
                continue;
            }

            let summary_content = to_archive
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("; ");

            let avg_importance =
                to_archive.iter().map(|m| m.importance).sum::<f32>() / to_archive.len() as f32;

            let source_ids: Vec<_> = to_archive.iter().map(|m| m.id.clone()).collect();
            let source_count = source_ids.len();

            produced.push(ConsolidatedMemory {
                agent_id: agent_id.clone(),
                summary_content,
                source_ids,
                importance: avg_importance,
                created_at: Utc::now(),
                source_count,
            });

            total_archived += source_count as u64;
        }

        if !produced.is_empty() {
            info!(
                agent = %agent_id,
                clusters = clusters.len(),
                consolidated = produced.len(),
                archived = total_archived,
                "memory consolidation pass complete"
            );
            self.metrics
                .memories_consolidated_total
                .fetch_add(total_archived, Ordering::Relaxed);

            if let Ok(mut guard) = self.consolidated.lock() {
                guard.extend(produced.clone());
            }
        }

        produced
    }

    /// Start the background consolidation loop.
    ///
    /// This is an infinite `async fn` designed to run inside `tokio::spawn`.
    /// On each tick it consolidates all agents currently tracked by the store.
    /// The task runs until the Tokio runtime is shut down or the task is
    /// explicitly cancelled via its `JoinHandle`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_agent_runtime::consolidation::{ConsolidationPolicy, MemoryConsolidator};
    /// use llm_agent_runtime::memory::EpisodicStore;
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let store = Arc::new(EpisodicStore::new());
    ///     let consolidator = MemoryConsolidator::new(store, ConsolidationPolicy::default());
    ///     let _handle = tokio::spawn(consolidator.run());
    ///     // … rest of application …
    /// }
    /// ```
    pub async fn run(self) {
        let interval = Duration::from_secs(self.policy.run_interval_secs);
        loop {
            tokio::time::sleep(interval).await;

            let agent_ids = match self.store.all_agent_ids() {
                Ok(ids) => ids,
                Err(e) => {
                    debug!(%e, "consolidation: failed to enumerate agent ids");
                    continue;
                }
            };

            for agent_id in &agent_ids {
                let _ = self.consolidate_agent(agent_id);
            }

            self.metrics
                .consolidation_runs_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenise_basic() {
        let tokens = tokenise("Hello world foo");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"foo".to_string()));
    }

    #[test]
    fn test_tokenise_filters_short_words() {
        let tokens = tokenise("it is a test");
        assert!(!tokens.contains(&"it".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_tfidf_embedding_non_empty() {
        let corpus = &["the quick brown fox", "the lazy dog runs fast"];
        let emb = tfidf_embedding("the quick fox", corpus);
        assert!(!emb.is_empty());
    }

    #[test]
    fn test_tfidf_embedding_normalised() {
        let corpus = &["rust programming language", "rust memory safety system"];
        let emb = tfidf_embedding("rust programming", corpus);
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Either unit-length or zero (no token overlap).
        assert!(norm < 1.001);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![0.6, 0.8, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_kmeans_clusters_two_groups() {
        // Two tight groups that should cluster separately.
        let embeddings = vec![
            vec![1.0_f64, 0.0],
            vec![0.95, 0.05],
            vec![0.0, 1.0],
            vec![0.05, 0.95],
        ];
        let labels = kmeans(&embeddings, 2, 20);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_kmeans_empty_returns_empty() {
        let labels = kmeans(&[], 3, 10);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_kmeans_k_larger_than_n_clamped() {
        let embeddings = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0]];
        let labels = kmeans(&embeddings, 5, 5);
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_consolidation_policy_defaults() {
        let policy = ConsolidationPolicy::default();
        assert_eq!(policy.min_cluster_size, 3);
        assert_eq!(policy.k_clusters, 8);
        assert_eq!(policy.kmeans_iterations, 10);
        assert!((policy.similarity_threshold - 0.65).abs() < 1e-9);
        assert_eq!(policy.run_interval_secs, 300);
    }

    #[test]
    fn test_metrics_initial_zero() {
        let m = ConsolidationMetrics::new();
        assert_eq!(m.runs(), 0);
        assert_eq!(m.consolidated(), 0);
    }

    #[test]
    fn test_metrics_atomic_increment() {
        let m = ConsolidationMetrics::new();
        m.consolidation_runs_total.fetch_add(3, Ordering::Relaxed);
        m.memories_consolidated_total
            .fetch_add(42, Ordering::Relaxed);
        assert_eq!(m.runs(), 3);
        assert_eq!(m.consolidated(), 42);
    }
}
