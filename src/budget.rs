//! # Module: Budget
//!
//! ## Responsibility
//! Tracks token usage and estimated cost per agent/session and enforces
//! configurable budget limits.
//!
//! ## Key types
//! - [`PriceTable`] — per-million-token prices for input and output tokens.
//! - [`BudgetTracker`] — accumulates usage, computes cost, enforces limits,
//!   and exports a JSON report.
//!
//! ## Guarantees
//! - Thread-safe: internal counters use `std::sync::Mutex`.
//! - Non-panicking: all public methods return `Result` or plain values.
//! - The tracker computes cost using `u128` micro-dollar arithmetic to avoid
//!   floating-point accumulation errors.

use crate::error::AgentRuntimeError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ── PriceTable ────────────────────────────────────────────────────────────────

/// Per-model token pricing used by [`BudgetTracker`].
///
/// Prices are expressed in **USD per million tokens** for both input (prompt)
/// and output (completion) tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrice {
    /// Price in USD per 1M input (prompt) tokens.
    pub input_per_million: f64,
    /// Price in USD per 1M output (completion) tokens.
    pub output_per_million: f64,
}

impl ModelPrice {
    /// Construct a `ModelPrice` with the given per-million rates.
    pub fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self { input_per_million, output_per_million }
    }
}

/// A table mapping model names to their token prices.
///
/// # Example
/// ```rust,no_run
/// use llm_agent_runtime::budget::{PriceTable, ModelPrice};
///
/// let prices = PriceTable::new()
///     .with_model("claude-3-5-sonnet", ModelPrice::new(3.0, 15.0))
///     .with_model("gpt-4o", ModelPrice::new(2.5, 10.0));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PriceTable {
    prices: HashMap<String, ModelPrice>,
}

impl PriceTable {
    /// Create an empty price table.
    pub fn new() -> Self {
        Self { prices: HashMap::new() }
    }

    /// Create a price table pre-loaded with common model prices (as of 2025).
    pub fn default_prices() -> Self {
        Self::new()
            .with_model("claude-3-5-sonnet-20241022", ModelPrice::new(3.0, 15.0))
            .with_model("claude-3-5-haiku-20241022", ModelPrice::new(0.8, 4.0))
            .with_model("claude-3-opus-20240229", ModelPrice::new(15.0, 75.0))
            .with_model("gpt-4o", ModelPrice::new(2.5, 10.0))
            .with_model("gpt-4o-mini", ModelPrice::new(0.15, 0.60))
            .with_model("gpt-4-turbo", ModelPrice::new(10.0, 30.0))
    }

    /// Add or overwrite pricing for `model_name`.
    pub fn with_model(mut self, model_name: impl Into<String>, price: ModelPrice) -> Self {
        self.prices.insert(model_name.into(), price);
        self
    }

    /// Look up pricing for `model_name`.
    pub fn get(&self, model_name: &str) -> Option<&ModelPrice> {
        self.prices.get(model_name)
    }

    /// Return the number of models in the table.
    pub fn model_count(&self) -> usize {
        self.prices.len()
    }
}

// ── UsageRecord ───────────────────────────────────────────────────────────────

/// Accumulated token usage for a single agent or session.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageRecord {
    /// Total input tokens consumed.
    pub input_tokens: u64,
    /// Total output tokens produced.
    pub output_tokens: u64,
    /// Total estimated cost in micro-dollars (divide by 1_000_000 for USD).
    pub cost_microdollars: u128,
    /// Number of inference calls made.
    pub call_count: u64,
}

impl UsageRecord {
    /// Compute the estimated cost in USD.
    pub fn cost_usd(&self) -> f64 {
        self.cost_microdollars as f64 / 1_000_000.0
    }

    /// Total tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ── BudgetConfig ──────────────────────────────────────────────────────────────

/// Configuration for [`BudgetTracker`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum allowed spend in USD for a tracked entity (`None` = unlimited).
    pub max_cost_usd: Option<f64>,
    /// Maximum allowed total tokens (`None` = unlimited).
    pub max_total_tokens: Option<u64>,
    /// The price table used to estimate costs.
    pub prices: PriceTable,
}

impl BudgetConfig {
    /// Create a new config with no limits and the default price table.
    pub fn new() -> Self {
        Self {
            max_cost_usd: None,
            max_total_tokens: None,
            prices: PriceTable::default_prices(),
        }
    }

    /// Set a maximum spend limit in USD.
    pub fn with_max_cost_usd(mut self, limit: f64) -> Self {
        self.max_cost_usd = Some(limit);
        self
    }

    /// Set a maximum total token limit.
    pub fn with_max_total_tokens(mut self, limit: u64) -> Self {
        self.max_total_tokens = Some(limit);
        self
    }

    /// Replace the price table.
    pub fn with_prices(mut self, prices: PriceTable) -> Self {
        self.prices = prices;
        self
    }
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── BudgetReport ─────────────────────────────────────────────────────────────

/// A point-in-time snapshot of usage for all tracked agents/sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetReport {
    /// Per-entity usage breakdown.  Key is the agent/session identifier.
    pub entities: HashMap<String, UsageRecord>,
    /// Aggregate totals across all entities.
    pub totals: UsageRecord,
    /// Whether the budget limit was exceeded at the time of this report.
    pub limit_exceeded: bool,
    /// UTC timestamp when the report was generated (ISO 8601).
    pub generated_at: String,
}

impl BudgetReport {
    /// Serialize this report as a pretty-printed JSON string.
    ///
    /// # Errors
    /// Returns `AgentRuntimeError::AgentLoop` if JSON serialization fails
    /// (should not happen for well-formed `BudgetReport` values).
    pub fn to_json(&self) -> Result<String, AgentRuntimeError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| AgentRuntimeError::AgentLoop(format!("BudgetReport JSON: {e}")))
    }

    /// Return the total estimated cost across all entities in USD.
    pub fn total_cost_usd(&self) -> f64 {
        self.totals.cost_usd()
    }
}

// ── BudgetTracker ─────────────────────────────────────────────────────────────

/// Tracks token usage and estimated cost per agent/session, enforcing limits.
///
/// # Halting on Budget Exceeded
///
/// [`record_usage`] returns `Err(AgentRuntimeError::AgentLoop(...))` when a
/// configured limit is exceeded.  Callers that propagate this error with `?`
/// automatically halt the agent loop.
///
/// # Thread Safety
///
/// `BudgetTracker` is `Clone + Send + Sync`.  Clones share the same underlying
/// state via `Arc<Mutex<_>>`.
///
/// [`record_usage`]: BudgetTracker::record_usage
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    config: BudgetConfig,
    inner: Arc<Mutex<BudgetTrackerInner>>,
}

#[derive(Debug, Default)]
struct BudgetTrackerInner {
    entities: HashMap<String, UsageRecord>,
    totals: UsageRecord,
    limit_exceeded: bool,
}

impl BudgetTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            inner: Arc::new(Mutex::new(BudgetTrackerInner::default())),
        }
    }

    /// Record token usage for `entity_id` (an agent or session identifier).
    ///
    /// Computes the cost using the configured price table for `model_name`.
    /// If no price is found for the model, cost is recorded as `0`.
    ///
    /// # Errors
    /// Returns `Err(AgentRuntimeError::AgentLoop)` if a budget limit is
    /// exceeded after recording the new usage.  The usage is still recorded
    /// so the caller can inspect it in the final report.
    pub fn record_usage(
        &self,
        entity_id: &str,
        model_name: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<(), AgentRuntimeError> {
        let cost_microdollars = self.compute_cost_microdollars(model_name, input_tokens, output_tokens);

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| AgentRuntimeError::AgentLoop("BudgetTracker lock poisoned".into()))?;

        let record = inner.entities.entry(entity_id.to_owned()).or_default();
        record.input_tokens = record.input_tokens.saturating_add(input_tokens);
        record.output_tokens = record.output_tokens.saturating_add(output_tokens);
        record.cost_microdollars = record.cost_microdollars.saturating_add(cost_microdollars);
        record.call_count = record.call_count.saturating_add(1);

        inner.totals.input_tokens = inner.totals.input_tokens.saturating_add(input_tokens);
        inner.totals.output_tokens = inner.totals.output_tokens.saturating_add(output_tokens);
        inner.totals.cost_microdollars =
            inner.totals.cost_microdollars.saturating_add(cost_microdollars);
        inner.totals.call_count = inner.totals.call_count.saturating_add(1);

        // Check limits.
        let total_tokens = inner.totals.total_tokens();
        let total_cost_usd = inner.totals.cost_usd();

        if let Some(max_tokens) = self.config.max_total_tokens {
            if total_tokens > max_tokens {
                inner.limit_exceeded = true;
                return Err(AgentRuntimeError::AgentLoop(format!(
                    "budget exceeded: token limit {max_tokens} surpassed ({total_tokens} used)"
                )));
            }
        }

        if let Some(max_cost) = self.config.max_cost_usd {
            if total_cost_usd > max_cost {
                inner.limit_exceeded = true;
                return Err(AgentRuntimeError::AgentLoop(format!(
                    "budget exceeded: cost limit ${max_cost:.4} surpassed (${total_cost_usd:.4} used)"
                )));
            }
        }

        Ok(())
    }

    /// Return a snapshot of usage for `entity_id`, or a zeroed record if not tracked.
    pub fn usage_for(&self, entity_id: &str) -> UsageRecord {
        self.inner
            .lock()
            .map(|g| g.entities.get(entity_id).cloned().unwrap_or_default())
            .unwrap_or_default()
    }

    /// Return the aggregate totals across all entities.
    pub fn totals(&self) -> UsageRecord {
        self.inner
            .lock()
            .map(|g| g.totals.clone())
            .unwrap_or_default()
    }

    /// Return `true` if any budget limit has been exceeded.
    pub fn is_limit_exceeded(&self) -> bool {
        self.inner.lock().map(|g| g.limit_exceeded).unwrap_or(false)
    }

    /// Generate a [`BudgetReport`] snapshot.
    pub fn report(&self) -> BudgetReport {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        BudgetReport {
            entities: inner.entities.clone(),
            totals: inner.totals.clone(),
            limit_exceeded: inner.limit_exceeded,
            generated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Export the current budget report as a JSON string.
    ///
    /// Convenience wrapper around [`report`] → [`BudgetReport::to_json`].
    ///
    /// # Errors
    /// Propagates serialization errors as `AgentRuntimeError::AgentLoop`.
    ///
    /// [`report`]: BudgetTracker::report
    pub fn export_json(&self) -> Result<String, AgentRuntimeError> {
        self.report().to_json()
    }

    /// Reset all accumulated usage, clearing all entity records and totals.
    pub fn reset(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            *inner = BudgetTrackerInner::default();
        }
    }

    /// Return a reference to the tracker's configuration.
    pub fn config(&self) -> &BudgetConfig {
        &self.config
    }

    // ── Private ────────────────────────────────────────────────────────────

    fn compute_cost_microdollars(
        &self,
        model_name: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> u128 {
        let Some(price) = self.config.prices.get(model_name) else {
            return 0;
        };
        // cost_microdollars = tokens / 1_000_000 * price_per_million * 1_000_000
        //                   = tokens * price_per_million
        // We scale by 1_000_000 for micro-dollar precision then round.
        let input_cost = (input_tokens as f64 * price.input_per_million) as u128;
        let output_cost = (output_tokens as f64 * price.output_per_million) as u128;
        input_cost.saturating_add(output_cost)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker_with_limit(max_tokens: u64) -> BudgetTracker {
        BudgetTracker::new(
            BudgetConfig::new()
                .with_max_total_tokens(max_tokens)
                .with_prices(PriceTable::new().with_model("test-model", ModelPrice::new(1.0, 2.0))),
        )
    }

    #[test]
    fn test_record_usage_increments_counts() {
        let tracker = BudgetTracker::new(BudgetConfig::new());
        tracker.record_usage("agent-1", "gpt-4o", 100, 50).unwrap();
        let usage = tracker.usage_for("agent-1");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.call_count, 1);
    }

    #[test]
    fn test_budget_limit_exceeded_returns_error() {
        let tracker = tracker_with_limit(10);
        let result = tracker.record_usage("agent-1", "test-model", 5, 6);
        assert!(result.is_err());
        assert!(tracker.is_limit_exceeded());
    }

    #[test]
    fn test_budget_within_limit_returns_ok() {
        let tracker = tracker_with_limit(100);
        tracker.record_usage("agent-1", "test-model", 10, 10).unwrap();
        assert!(!tracker.is_limit_exceeded());
    }

    #[test]
    fn test_export_json_is_valid() {
        let tracker = BudgetTracker::new(BudgetConfig::new());
        tracker.record_usage("agent-1", "gpt-4o", 100, 50).unwrap();
        let json = tracker.export_json().unwrap();
        assert!(json.contains("agent-1"));
    }

    #[test]
    fn test_totals_aggregate_across_entities() {
        let tracker = BudgetTracker::new(BudgetConfig::new());
        tracker.record_usage("a", "gpt-4o", 100, 50).unwrap();
        tracker.record_usage("b", "gpt-4o", 200, 100).unwrap();
        let totals = tracker.totals();
        assert_eq!(totals.input_tokens, 300);
        assert_eq!(totals.output_tokens, 150);
    }

    #[test]
    fn test_reset_clears_all_state() {
        let tracker = BudgetTracker::new(BudgetConfig::new());
        tracker.record_usage("agent-1", "gpt-4o", 100, 50).unwrap();
        tracker.reset();
        let usage = tracker.usage_for("agent-1");
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.call_count, 0);
    }

    #[test]
    fn test_cost_computed_for_known_model() {
        let tracker = BudgetTracker::new(
            BudgetConfig::new().with_prices(
                PriceTable::new().with_model("my-model", ModelPrice::new(10.0, 20.0)),
            ),
        );
        tracker.record_usage("a", "my-model", 1_000_000, 0).unwrap();
        let usage = tracker.usage_for("a");
        // 1M input tokens at $10/M = $10 = 10_000_000 micro-dollars
        assert_eq!(usage.cost_microdollars, 10_000_000);
    }
}
