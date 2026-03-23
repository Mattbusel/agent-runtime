//! # Module: negotiation
//!
//! ## Responsibility
//! Agent-to-agent negotiation protocol for resource allocation. Implements
//! multiple allocation strategies including Greedy, Fair-Share (max-min
//! fairness), Priority-Based, and Vickrey second-price auction.
//!
//! ## Key types
//! - [`NegotiationProposal`] — a single agent's resource request with priority.
//! - [`ResourceRequest`] — the actual resources requested.
//! - [`NegotiationStrategy`] — selects the allocation algorithm.
//! - [`NegotiationRound`] — groups proposals and runs the selected strategy.
//! - [`NegotiationResult`] — winner and allocated resources with fairness score.
//!
//! ## Guarantees
//! - No panics in production paths.
//! - Pure in-memory: no I/O, no async.
//! - Deterministic: tie-breaking is alphabetical by proposer_id.

// ── ResourceRequest ───────────────────────────────────────────────────────────

/// The specific resources requested in a [`NegotiationProposal`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRequest {
    /// Number of language-model tokens requested.
    pub tokens: u64,
    /// Compute units (e.g. CPU-milliseconds or GPU-ms) requested.
    pub compute_units: u32,
    /// Memory in megabytes requested.
    pub memory_mb: u64,
}

impl ResourceRequest {
    /// Create a new resource request.
    pub fn new(tokens: u64, compute_units: u32, memory_mb: u64) -> Self {
        Self { tokens, compute_units, memory_mb }
    }

    /// Compute a single scalar "weight" for comparison purposes.
    ///
    /// Weighting: tokens + compute_units×100 + memory_mb×10.
    /// These weights are arbitrary and can be tuned per deployment.
    pub fn weight(&self) -> u64 {
        self.tokens
            .saturating_add(u64::from(self.compute_units).saturating_mul(100))
            .saturating_add(self.memory_mb.saturating_mul(10))
    }
}

// ── NegotiationProposal ───────────────────────────────────────────────────────

/// A single agent's proposal in a negotiation round.
#[derive(Debug, Clone)]
pub struct NegotiationProposal {
    /// Identifier of the proposing agent.
    pub proposer_id: String,
    /// The resources this agent is requesting.
    pub resource_request: ResourceRequest,
    /// Priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// Absolute deadline by which the resources must be granted (Unix ms).
    /// `0` means no deadline constraint.
    pub deadline_ms: u64,
    /// Sealed bid value for auction-based allocation (Vickrey).
    /// Callers set this to indicate willingness-to-pay in arbitrary units.
    pub bid_value: u64,
}

impl NegotiationProposal {
    /// Create a proposal with a zero bid value (suitable for non-auction strategies).
    pub fn new(
        proposer_id: impl Into<String>,
        resource_request: ResourceRequest,
        priority: u8,
        deadline_ms: u64,
    ) -> Self {
        Self {
            proposer_id: proposer_id.into(),
            resource_request,
            priority,
            deadline_ms,
            bid_value: 0,
        }
    }

    /// Set the sealed bid value (for auction-based allocation).
    pub fn with_bid(mut self, bid_value: u64) -> Self {
        self.bid_value = bid_value;
        self
    }
}

// ── NegotiationStrategy ───────────────────────────────────────────────────────

/// Selects the algorithm used to resolve competing proposals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationStrategy {
    /// Award all resources to the proposer with the highest resource weight.
    Greedy,
    /// Max-min fair share: iteratively allocate the minimum requested share to
    /// the most-constrained proposer until resources are exhausted.
    FairShare,
    /// Award resources to the proposer with the highest priority; ties broken
    /// by highest resource weight, then alphabetically by proposer_id.
    PriorityBased,
    /// Vickrey second-price auction: the highest bidder wins but pays
    /// (is allocated) only the second-highest bid's resource weight.
    AuctionBased,
}

// ── NegotiationRound ──────────────────────────────────────────────────────────

/// A single negotiation round containing competing proposals.
///
/// Call [`run`] to execute the selected strategy and obtain a
/// [`NegotiationResult`].
///
/// [`run`]: NegotiationRound::run
#[derive(Debug, Clone)]
pub struct NegotiationRound {
    /// The proposals submitted by competing agents.
    pub proposals: Vec<NegotiationProposal>,
    /// The allocation strategy to apply.
    pub strategy: NegotiationStrategy,
}

impl NegotiationRound {
    /// Create a new round with the given proposals and strategy.
    pub fn new(proposals: Vec<NegotiationProposal>, strategy: NegotiationStrategy) -> Self {
        Self { proposals, strategy }
    }

    /// Run the negotiation and return a result.
    ///
    /// Returns `None` when there are no proposals.
    pub fn run(&self) -> Option<NegotiationResult> {
        if self.proposals.is_empty() {
            return None;
        }
        match &self.strategy {
            NegotiationStrategy::Greedy => Some(self.run_greedy()),
            NegotiationStrategy::FairShare => Some(self.run_fair_share()),
            NegotiationStrategy::PriorityBased => Some(self.run_priority_based()),
            NegotiationStrategy::AuctionBased => Some(self.run_auction()),
        }
    }

    // ── Greedy ────────────────────────────────────────────────────────────────

    fn run_greedy(&self) -> NegotiationResult {
        // Winner = highest resource weight; tie-break = alphabetical proposer_id.
        let winner = self
            .proposals
            .iter()
            .max_by(|a, b| {
                a.resource_request
                    .weight()
                    .cmp(&b.resource_request.weight())
                    .then_with(|| a.proposer_id.cmp(&b.proposer_id))
            })
            .expect("proposals is non-empty");

        let fairness = compute_fairness_score(&self.proposals, &winner.proposer_id);
        NegotiationResult {
            winner_id: winner.proposer_id.clone(),
            allocated_resources: winner.resource_request.clone(),
            fairness_score: fairness,
            strategy_used: NegotiationStrategy::Greedy,
        }
    }

    // ── Priority-Based ────────────────────────────────────────────────────────

    fn run_priority_based(&self) -> NegotiationResult {
        let winner = self
            .proposals
            .iter()
            .max_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| {
                        a.resource_request
                            .weight()
                            .cmp(&b.resource_request.weight())
                    })
                    .then_with(|| a.proposer_id.cmp(&b.proposer_id))
            })
            .expect("proposals is non-empty");

        let fairness = compute_fairness_score(&self.proposals, &winner.proposer_id);
        NegotiationResult {
            winner_id: winner.proposer_id.clone(),
            allocated_resources: winner.resource_request.clone(),
            fairness_score: fairness,
            strategy_used: NegotiationStrategy::PriorityBased,
        }
    }

    // ── Fair Share (max-min fairness) ─────────────────────────────────────────

    fn run_fair_share(&self) -> NegotiationResult {
        // Simplified max-min fairness: give each agent an equal share of a
        // virtual resource budget, then award the full round to the proposer
        // whose *request* is closest to their fair share (most constrained).
        //
        // "Most constrained" = smallest ratio of (request_weight / fair_share).
        // Tie-break: alphabetical proposer_id.
        let n = self.proposals.len() as u64;
        let total_weight: u64 = self
            .proposals
            .iter()
            .map(|p| p.resource_request.weight())
            .fold(0u64, |acc, w| acc.saturating_add(w));
        let fair_share = if n == 0 { 0 } else { total_weight / n };

        // Winner = proposal whose weight is ≤ fair_share (most constrained);
        // if none are under, pick the smallest weight (closest to fair share).
        let winner = self
            .proposals
            .iter()
            .min_by(|a, b| {
                let wa = a.resource_request.weight();
                let wb = b.resource_request.weight();
                // Prefer proposals at or below fair share.
                let a_over = wa > fair_share;
                let b_over = wb > fair_share;
                match (a_over, b_over) {
                    (false, true) => std::cmp::Ordering::Less,
                    (true, false) => std::cmp::Ordering::Greater,
                    _ => wa.cmp(&wb).then_with(|| a.proposer_id.cmp(&b.proposer_id)),
                }
            })
            .expect("proposals is non-empty");

        let fairness = compute_fairness_score(&self.proposals, &winner.proposer_id);
        NegotiationResult {
            winner_id: winner.proposer_id.clone(),
            allocated_resources: winner.resource_request.clone(),
            fairness_score: fairness,
            strategy_used: NegotiationStrategy::FairShare,
        }
    }

    // ── Vickrey Second-Price Auction ──────────────────────────────────────────

    fn run_auction(&self) -> NegotiationResult {
        // Sort proposals by bid_value descending, tie-break alphabetically.
        let mut sorted: Vec<&NegotiationProposal> = self.proposals.iter().collect();
        sorted.sort_by(|a, b| {
            b.bid_value
                .cmp(&a.bid_value)
                .then_with(|| a.proposer_id.cmp(&b.proposer_id))
        });

        let winner_proposal = sorted[0];
        // Second-price: if only one bidder, they pay their own bid (no discount).
        let second_price_bid = if sorted.len() > 1 { sorted[1].bid_value } else { winner_proposal.bid_value };

        // Allocated resources are scaled down proportionally to the second price.
        // If second_price_bid == winner_bid, full allocation is given (no scaling).
        let allocated = scale_request_by_bid(
            &winner_proposal.resource_request,
            second_price_bid,
            winner_proposal.bid_value,
        );

        let fairness = compute_fairness_score(&self.proposals, &winner_proposal.proposer_id);
        NegotiationResult {
            winner_id: winner_proposal.proposer_id.clone(),
            allocated_resources: allocated,
            fairness_score: fairness,
            strategy_used: NegotiationStrategy::AuctionBased,
        }
    }
}

/// Scale a `ResourceRequest` by the ratio `second_bid / winner_bid`.
///
/// When `winner_bid` is zero (degenerate case), returns the original request.
fn scale_request_by_bid(
    req: &ResourceRequest,
    second_bid: u64,
    winner_bid: u64,
) -> ResourceRequest {
    if winner_bid == 0 || second_bid >= winner_bid {
        return req.clone();
    }
    let ratio = second_bid as f64 / winner_bid as f64;
    ResourceRequest {
        tokens: (req.tokens as f64 * ratio).ceil() as u64,
        compute_units: (req.compute_units as f64 * ratio).ceil() as u32,
        memory_mb: (req.memory_mb as f64 * ratio).ceil() as u64,
    }
}

// ── NegotiationResult ─────────────────────────────────────────────────────────

/// The outcome of a [`NegotiationRound`].
#[derive(Debug, Clone)]
pub struct NegotiationResult {
    /// Identifier of the winning agent.
    pub winner_id: String,
    /// The resources allocated to the winner.
    pub allocated_resources: ResourceRequest,
    /// Fairness score in [0.0, 1.0].
    ///
    /// Computed as the ratio of the winner's request weight to the sum of all
    /// request weights.  A score close to `1/n` (where `n` is the number of
    /// proposers) indicates the fairest possible outcome.  `1.0` means the
    /// winner requested everything.
    pub fairness_score: f64,
    /// The strategy that produced this result.
    pub strategy_used: NegotiationStrategy,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute a fairness score: winner_weight / total_weight.
///
/// Returns `1.0` when there is only one proposer (perfectly "fair" in that
/// degenerate case) or when total weight is zero.
fn compute_fairness_score(proposals: &[NegotiationProposal], winner_id: &str) -> f64 {
    let total: u64 = proposals
        .iter()
        .map(|p| p.resource_request.weight())
        .fold(0u64, |acc, w| acc.saturating_add(w));
    if total == 0 {
        return 1.0;
    }
    let winner_weight = proposals
        .iter()
        .find(|p| p.proposer_id == winner_id)
        .map(|p| p.resource_request.weight())
        .unwrap_or(0);
    winner_weight as f64 / total as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn req(tokens: u64, compute_units: u32, memory_mb: u64) -> ResourceRequest {
        ResourceRequest::new(tokens, compute_units, memory_mb)
    }

    fn proposal(id: &str, r: ResourceRequest, priority: u8) -> NegotiationProposal {
        NegotiationProposal::new(id, r, priority, 0)
    }

    // ── ResourceRequest::weight ───────────────────────────────────────────────

    #[test]
    fn weight_zero_request_is_zero() {
        assert_eq!(req(0, 0, 0).weight(), 0);
    }

    #[test]
    fn weight_tokens_only() {
        assert_eq!(req(1_000, 0, 0).weight(), 1_000);
    }

    #[test]
    fn weight_compute_units_scaled() {
        assert_eq!(req(0, 5, 0).weight(), 500);
    }

    #[test]
    fn weight_memory_scaled() {
        assert_eq!(req(0, 0, 10).weight(), 100);
    }

    // ── Empty round ───────────────────────────────────────────────────────────

    #[test]
    fn empty_round_returns_none() {
        let round = NegotiationRound::new(vec![], NegotiationStrategy::Greedy);
        assert!(round.run().is_none());
    }

    // ── Greedy ────────────────────────────────────────────────────────────────

    #[test]
    fn greedy_picks_highest_weight() {
        let proposals = vec![
            proposal("agent-a", req(100, 0, 0), 0),
            proposal("agent-b", req(500, 0, 0), 0),
            proposal("agent-c", req(300, 0, 0), 0),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::Greedy)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "agent-b");
    }

    #[test]
    fn greedy_single_proposer_wins() {
        let proposals = vec![proposal("only-one", req(100, 1, 10), 5)];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::Greedy)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "only-one");
    }

    #[test]
    fn greedy_tie_broken_alphabetically() {
        let proposals = vec![
            proposal("bravo", req(100, 0, 0), 0),
            proposal("alpha", req(100, 0, 0), 0),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::Greedy)
            .run()
            .unwrap();
        // "bravo" > "alpha" alphabetically → "bravo" wins tie
        assert_eq!(result.winner_id, "bravo");
    }

    // ── Priority-Based ────────────────────────────────────────────────────────

    #[test]
    fn priority_highest_priority_wins() {
        let proposals = vec![
            proposal("low", req(1_000, 0, 0), 1),
            proposal("high", req(100, 0, 0), 200),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::PriorityBased)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "high");
    }

    #[test]
    fn priority_tie_uses_weight_then_alphabetical() {
        let proposals = vec![
            proposal("z-agent", req(500, 0, 0), 10),
            proposal("a-agent", req(500, 0, 0), 10),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::PriorityBased)
            .run()
            .unwrap();
        // Same priority, same weight → alphabetical: "z-agent" > "a-agent"
        assert_eq!(result.winner_id, "z-agent");
    }

    // ── Fair Share ────────────────────────────────────────────────────────────

    #[test]
    fn fair_share_picks_most_constrained() {
        // fair_share = (100 + 1000) / 2 = 550; agent-a (100) is under fair_share
        let proposals = vec![
            proposal("agent-a", req(100, 0, 0), 0),
            proposal("agent-b", req(1_000, 0, 0), 0),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::FairShare)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "agent-a");
    }

    #[test]
    fn fair_share_single_proposer_wins() {
        let proposals = vec![proposal("solo", req(200, 2, 20), 0)];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::FairShare)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "solo");
    }

    // ── Auction (Vickrey) ─────────────────────────────────────────────────────

    #[test]
    fn auction_highest_bidder_wins() {
        let proposals = vec![
            proposal("agent-a", req(100, 0, 0), 0).with_bid(500),
            proposal("agent-b", req(100, 0, 0), 0).with_bid(300),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::AuctionBased)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "agent-a");
    }

    #[test]
    fn auction_second_price_scales_allocation() {
        // Winner bid = 1000, second bid = 500 → ratio = 0.5
        let proposals = vec![
            proposal("winner", req(1_000, 0, 0), 0).with_bid(1_000),
            proposal("second", req(500, 0, 0), 0).with_bid(500),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::AuctionBased)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "winner");
        // tokens: ceil(1000 * 0.5) = 500
        assert_eq!(result.allocated_resources.tokens, 500);
    }

    #[test]
    fn auction_single_bidder_pays_own_bid() {
        let proposals = vec![proposal("solo", req(800, 4, 64), 0).with_bid(999)];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::AuctionBased)
            .run()
            .unwrap();
        assert_eq!(result.winner_id, "solo");
        // Only one bidder → full allocation
        assert_eq!(result.allocated_resources.tokens, 800);
    }

    // ── Fairness score ────────────────────────────────────────────────────────

    #[test]
    fn fairness_score_in_unit_interval() {
        let proposals = vec![
            proposal("a", req(100, 0, 0), 5),
            proposal("b", req(200, 0, 0), 3),
        ];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::Greedy)
            .run()
            .unwrap();
        assert!((0.0..=1.0).contains(&result.fairness_score));
    }

    #[test]
    fn fairness_score_single_proposer_is_one() {
        let proposals = vec![proposal("only", req(500, 0, 0), 0)];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::Greedy)
            .run()
            .unwrap();
        assert_eq!(result.fairness_score, 1.0);
    }

    // ── NegotiationResult fields ──────────────────────────────────────────────

    #[test]
    fn result_allocated_resources_match_winner_for_greedy() {
        let r = req(100, 2, 20);
        let proposals = vec![proposal("winner", r.clone(), 0)];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::Greedy)
            .run()
            .unwrap();
        assert_eq!(result.allocated_resources, r);
    }

    #[test]
    fn result_strategy_recorded_correctly() {
        let proposals = vec![proposal("a", req(1, 0, 0), 0)];
        let result = NegotiationRound::new(proposals, NegotiationStrategy::FairShare)
            .run()
            .unwrap();
        assert_eq!(result.strategy_used, NegotiationStrategy::FairShare);
    }
}
