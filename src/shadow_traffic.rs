//! # Module: shadow_traffic
//!
//! ## Responsibility
//! Shadow traffic duplicator — silently replay requests to a shadow agent for
//! testing without affecting production. Enables comparison of live vs. shadow
//! responses for divergence analysis and confidence scoring.
//!
//! ## Key types
//! - [`ShadowConfig`] — sampling rate, target agent, response capture flag.
//! - [`ShadowTrafficManager`] — deterministic sampling via FNV-1a hashing.
//! - [`ShadowRecord`] — a paired (original, shadow) response with divergence score.
//! - [`ShadowReport`] — aggregated divergence statistics over a collection window.
//!
//! ## Guarantees
//! - No panics in production paths.
//! - Thread-safe: all shared state behind `Arc<Mutex<_>>`.
//! - Deterministic sampling: same `request_id` always produces the same shadow decision.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ── ShadowConfig ──────────────────────────────────────────────────────────────

/// Configuration for the shadow traffic duplicator.
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Fraction of requests to shadow (0.0 = none, 1.0 = all).
    pub sampling_rate: f64,
    /// Identifier of the shadow agent that receives replayed requests.
    pub shadow_agent_id: String,
    /// Whether to capture and store shadow responses for analysis.
    pub capture_responses: bool,
    /// Maximum number of records to retain in the in-memory window.
    pub window_size: usize,
}

impl ShadowConfig {
    /// Create a new `ShadowConfig` with explicit sampling rate and shadow agent.
    pub fn new(
        sampling_rate: f64,
        shadow_agent_id: impl Into<String>,
        capture_responses: bool,
    ) -> Self {
        let rate = sampling_rate.clamp(0.0, 1.0);
        Self {
            sampling_rate: rate,
            shadow_agent_id: shadow_agent_id.into(),
            capture_responses,
            window_size: 1_000,
        }
    }

    /// Override the rolling window size (default: 1 000).
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }
}

// ── ShadowTrafficManager ──────────────────────────────────────────────────────

/// Manages shadow traffic decisions and accumulated records.
///
/// Uses deterministic FNV-1a hashing of the `request_id` so that the same
/// request always makes the same shadow/no-shadow decision, independent of
/// wall-clock time or process state.
#[derive(Debug, Clone)]
pub struct ShadowTrafficManager {
    config: ShadowConfig,
    inner: Arc<Mutex<ManagerInner>>,
}

#[derive(Debug)]
struct ManagerInner {
    records: VecDeque<ShadowRecord>,
    total_shadowed: u64,
    total_skipped: u64,
}

impl ManagerInner {
    fn new() -> Self {
        Self {
            records: VecDeque::new(),
            total_shadowed: 0,
            total_skipped: 0,
        }
    }
}

impl ShadowTrafficManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: ShadowConfig) -> Self {
        Self {
            config,
            inner: Arc::new(Mutex::new(ManagerInner::new())),
        }
    }

    /// Decide whether the given `request_id` should be shadowed.
    ///
    /// The decision is deterministic: identical `request_id` values always
    /// produce the same result for a given `sampling_rate`.
    pub fn should_shadow(&self, request_id: &str) -> bool {
        let hash = fnv1a_hash(request_id.as_bytes());
        // Map hash to [0.0, 1.0) and compare against the sampling rate.
        let ratio = (hash as f64) / (u64::MAX as f64);
        ratio < self.config.sampling_rate
    }

    /// Record a shadow comparison result.
    ///
    /// Stores the record in the rolling window if `capture_responses` is
    /// enabled, and updates counters either way.
    ///
    /// # Errors
    /// Returns `Err` only if the internal mutex is poisoned.
    pub fn record(&self, record: ShadowRecord) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "ShadowTrafficManager lock poisoned".to_string())?;

        inner.total_shadowed += 1;

        if self.config.capture_responses {
            if inner.records.len() >= self.config.window_size {
                inner.records.pop_front();
            }
            inner.records.push_back(record);
        }

        Ok(())
    }

    /// Increment the skipped (not shadowed) counter.
    ///
    /// # Errors
    /// Returns `Err` only if the internal mutex is poisoned.
    pub fn record_skip(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "ShadowTrafficManager lock poisoned".to_string())?;
        inner.total_skipped += 1;
        Ok(())
    }

    /// Generate a [`ShadowReport`] aggregating all retained records.
    ///
    /// # Errors
    /// Returns `Err` only if the internal mutex is poisoned.
    pub fn report(&self) -> Result<ShadowReport, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| "ShadowTrafficManager lock poisoned".to_string())?;

        let records: Vec<ShadowRecord> = inner.records.iter().cloned().collect();
        let total_shadowed = inner.total_shadowed;
        let total_skipped = inner.total_skipped;
        drop(inner);

        Ok(ShadowReport::from_records(
            records,
            total_shadowed,
            total_skipped,
        ))
    }

    /// Return the shadow agent identifier from the config.
    pub fn shadow_agent_id(&self) -> &str {
        &self.config.shadow_agent_id
    }

    /// Return the configured sampling rate.
    pub fn sampling_rate(&self) -> f64 {
        self.config.sampling_rate
    }
}

// ── ShadowRecord ─────────────────────────────────────────────────────────────

/// A paired comparison of an original and a shadow response.
#[derive(Debug, Clone)]
pub struct ShadowRecord {
    /// The request identifier that triggered this shadow comparison.
    pub request_id: String,
    /// The response produced by the production agent.
    pub original_response: String,
    /// The response produced by the shadow agent.
    pub shadow_response: String,
    /// Wall-clock latency difference in milliseconds (shadow − original).
    /// Negative means the shadow was faster.
    pub latency_delta_ms: i64,
    /// Normalised Levenshtein divergence in [0.0, 1.0].
    /// 0.0 = identical, 1.0 = completely different.
    pub divergence_score: f64,
}

impl ShadowRecord {
    /// Create a new `ShadowRecord`, computing the divergence score automatically.
    pub fn new(
        request_id: impl Into<String>,
        original_response: impl Into<String>,
        shadow_response: impl Into<String>,
        latency_delta_ms: i64,
    ) -> Self {
        let orig = original_response.into();
        let shadow = shadow_response.into();
        let divergence_score = divergence_score(&orig, &shadow);
        Self {
            request_id: request_id.into(),
            original_response: orig,
            shadow_response: shadow,
            latency_delta_ms,
            divergence_score,
        }
    }

    /// Return `true` when both responses are identical (divergence == 0.0).
    pub fn is_exact_match(&self) -> bool {
        self.divergence_score == 0.0
    }
}

// ── ShadowReport ─────────────────────────────────────────────────────────────

/// Aggregated divergence statistics over a collection of [`ShadowRecord`]s.
#[derive(Debug, Clone)]
pub struct ShadowReport {
    /// Number of requests that were shadowed (including those not stored due
    /// to `capture_responses = false`).
    pub total_shadowed: u64,
    /// Number of requests that were NOT shadowed (skipped by sampler).
    pub total_skipped: u64,
    /// Number of records in the retained window.
    pub window_size: usize,
    /// Mean divergence score across all retained records (0.0 if none).
    pub mean_divergence: f64,
    /// Maximum divergence score seen in the window (0.0 if none).
    pub max_divergence: f64,
    /// Fraction of retained records with `divergence_score == 0.0`.
    pub exact_match_rate: f64,
    /// Mean absolute latency delta in milliseconds (0.0 if none).
    pub mean_latency_delta_ms: f64,
}

impl ShadowReport {
    /// Build a report from a slice of records plus total counters.
    fn from_records(records: Vec<ShadowRecord>, total_shadowed: u64, total_skipped: u64) -> Self {
        let n = records.len();
        if n == 0 {
            return Self {
                total_shadowed,
                total_skipped,
                window_size: 0,
                mean_divergence: 0.0,
                max_divergence: 0.0,
                exact_match_rate: 0.0,
                mean_latency_delta_ms: 0.0,
            };
        }

        let mut sum_div = 0.0_f64;
        let mut max_div = 0.0_f64;
        let mut exact_matches = 0usize;
        let mut sum_lat = 0.0_f64;

        for r in &records {
            sum_div += r.divergence_score;
            if r.divergence_score > max_div {
                max_div = r.divergence_score;
            }
            if r.is_exact_match() {
                exact_matches += 1;
            }
            sum_lat += r.latency_delta_ms.unsigned_abs() as f64;
        }

        let nf = n as f64;
        Self {
            total_shadowed,
            total_skipped,
            window_size: n,
            mean_divergence: sum_div / nf,
            max_divergence: max_div,
            exact_match_rate: exact_matches as f64 / nf,
            mean_latency_delta_ms: sum_lat / nf,
        }
    }
}

// ── Divergence calculation ─────────────────────────────────────────────────────

/// Compute a normalised Levenshtein distance between two strings.
///
/// Returns a score in [0.0, 1.0]:
/// - `0.0` means the strings are identical.
/// - `1.0` means maximum divergence (edit distance equals the length of the
///   longer string).
///
/// The comparison is performed on Unicode characters (not bytes) to handle
/// multi-byte UTF-8 correctly.
pub fn divergence_score(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let distance = levenshtein(&a_chars, &b_chars);
    let max_len = a_chars.len().max(b_chars.len());
    if max_len == 0 {
        return 0.0;
    }
    (distance as f64) / (max_len as f64)
}

/// Standard Wagner-Fischer Levenshtein distance on character slices.
fn levenshtein(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Two-row DP to keep O(min(m,n)) space.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1)
                .min(prev[j] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// FNV-1a 64-bit hash — deterministic, no external deps.
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn default_config() -> ShadowConfig {
        ShadowConfig::new(0.5, "shadow-agent-1", true)
    }

    // ── divergence_score ─────────────────────────────────────────────────────

    #[test]
    fn divergence_identical_strings_is_zero() {
        assert_eq!(divergence_score("hello world", "hello world"), 0.0);
    }

    #[test]
    fn divergence_empty_strings_is_zero() {
        assert_eq!(divergence_score("", ""), 0.0);
    }

    #[test]
    fn divergence_one_empty_is_one() {
        assert_eq!(divergence_score("abc", ""), 1.0);
        assert_eq!(divergence_score("", "abc"), 1.0);
    }

    #[test]
    fn divergence_single_char_diff() {
        // "abc" vs "axc" — 1 substitution, max_len = 3 → 1/3
        let score = divergence_score("abc", "axc");
        assert!((score - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn divergence_completely_different() {
        // "abc" vs "xyz" — 3 substitutions, max_len = 3 → 1.0
        let score = divergence_score("abc", "xyz");
        assert_eq!(score, 1.0);
    }

    #[test]
    fn divergence_is_between_zero_and_one() {
        let score = divergence_score("The quick brown fox", "The slow green fox");
        assert!((0.0..=1.0).contains(&score));
    }

    // ── ShadowRecord ─────────────────────────────────────────────────────────

    #[test]
    fn shadow_record_exact_match_detected() {
        let rec = ShadowRecord::new("req-1", "response A", "response A", 0);
        assert!(rec.is_exact_match());
        assert_eq!(rec.divergence_score, 0.0);
    }

    #[test]
    fn shadow_record_divergence_computed() {
        let rec = ShadowRecord::new("req-2", "hello", "world", 5);
        assert!(!rec.is_exact_match());
        assert!(rec.divergence_score > 0.0);
    }

    // ── ShadowTrafficManager::should_shadow ───────────────────────────────────

    #[test]
    fn should_shadow_always_true_at_rate_one() {
        let mgr = ShadowTrafficManager::new(ShadowConfig::new(1.0, "shadow", true));
        for i in 0..100 {
            assert!(mgr.should_shadow(&format!("req-{i}")));
        }
    }

    #[test]
    fn should_shadow_always_false_at_rate_zero() {
        let mgr = ShadowTrafficManager::new(ShadowConfig::new(0.0, "shadow", true));
        for i in 0..100 {
            assert!(!mgr.should_shadow(&format!("req-{i}")));
        }
    }

    #[test]
    fn should_shadow_deterministic() {
        let mgr = ShadowTrafficManager::new(default_config());
        let first = mgr.should_shadow("stable-request-id");
        for _ in 0..10 {
            assert_eq!(mgr.should_shadow("stable-request-id"), first);
        }
    }

    #[test]
    fn should_shadow_rate_clamped() {
        // Values outside [0,1] are silently clamped.
        let mgr = ShadowTrafficManager::new(ShadowConfig::new(2.5, "s", true));
        assert_eq!(mgr.sampling_rate(), 1.0);
        let mgr2 = ShadowTrafficManager::new(ShadowConfig::new(-0.5, "s", true));
        assert_eq!(mgr2.sampling_rate(), 0.0);
    }

    // ── record / report ───────────────────────────────────────────────────────

    #[test]
    fn report_empty_window_has_zero_stats() {
        let mgr = ShadowTrafficManager::new(default_config());
        let report = mgr.report().unwrap();
        assert_eq!(report.window_size, 0);
        assert_eq!(report.mean_divergence, 0.0);
        assert_eq!(report.exact_match_rate, 0.0);
    }

    #[test]
    fn report_counts_shadowed_records() {
        let mgr = ShadowTrafficManager::new(default_config());
        let rec = ShadowRecord::new("r1", "orig", "orig", 0);
        mgr.record(rec).unwrap();
        let report = mgr.report().unwrap();
        assert_eq!(report.total_shadowed, 1);
        assert_eq!(report.window_size, 1);
    }

    #[test]
    fn report_exact_match_rate_all_identical() {
        let mgr = ShadowTrafficManager::new(default_config());
        for i in 0..5 {
            mgr.record(ShadowRecord::new(
                format!("r{i}"),
                "same response",
                "same response",
                0,
            ))
            .unwrap();
        }
        let report = mgr.report().unwrap();
        assert_eq!(report.exact_match_rate, 1.0);
        assert_eq!(report.mean_divergence, 0.0);
    }

    #[test]
    fn report_window_evicts_oldest() {
        let config = ShadowConfig::new(1.0, "s", true).with_window_size(3);
        let mgr = ShadowTrafficManager::new(config);
        for i in 0..5 {
            mgr.record(ShadowRecord::new(format!("r{i}"), "a", "a", 0))
                .unwrap();
        }
        let report = mgr.report().unwrap();
        assert_eq!(report.window_size, 3);
        assert_eq!(report.total_shadowed, 5);
    }

    #[test]
    fn record_skip_increments_skipped_counter() {
        let mgr = ShadowTrafficManager::new(default_config());
        mgr.record_skip().unwrap();
        mgr.record_skip().unwrap();
        let report = mgr.report().unwrap();
        assert_eq!(report.total_skipped, 2);
    }

    #[test]
    fn shadow_agent_id_returned_from_config() {
        let mgr = ShadowTrafficManager::new(ShadowConfig::new(0.5, "my-shadow", true));
        assert_eq!(mgr.shadow_agent_id(), "my-shadow");
    }

    #[test]
    fn capture_responses_false_does_not_store_records() {
        let config = ShadowConfig::new(1.0, "s", false);
        let mgr = ShadowTrafficManager::new(config);
        mgr.record(ShadowRecord::new("r1", "a", "b", 0)).unwrap();
        let report = mgr.report().unwrap();
        // counter incremented but nothing stored
        assert_eq!(report.total_shadowed, 1);
        assert_eq!(report.window_size, 0);
    }
}
