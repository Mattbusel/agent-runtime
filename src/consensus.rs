//! # Module: consensus
//!
//! ## Responsibility
//! Multi-agent voting and consensus for distributed decisions.
//!
//! Provides:
//! - [`Vote`]: a single vote cast by an agent on a proposal.
//! - [`VoteChoice`]: Approve / Reject / Abstain.
//! - [`VotingRule`]: the rule used to determine the outcome of a vote.
//! - [`Proposal`]: a proposal that agents can vote on.
//! - [`VoteTally`]: the aggregate result of all votes on a proposal.
//! - [`ConsensusOutcome`]: Approved / Rejected / Inconclusive.
//! - [`ConsensusMgr`]: manages multiple proposals and their lifecycle.
//! - [`ConsensusError`]: errors that can arise during voting.
//!
//! ## Guarantees
//! - One vote per voter per proposal; duplicates are rejected.
//! - Votes are rejected after the proposal deadline.
//! - No `.unwrap()` or `.expect()` in production paths.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── VoteChoice ────────────────────────────────────────────────────────────────

/// The choice a voter makes on a proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoteChoice {
    /// The voter approves the proposal.
    Approve,
    /// The voter rejects the proposal.
    Reject,
    /// The voter abstains (counted toward quorum but not toward approve/reject weight).
    Abstain,
}

// ── VotingRule ────────────────────────────────────────────────────────────────

/// The rule used to determine the outcome of a vote tally.
#[derive(Debug, Clone)]
pub enum VotingRule {
    /// Approve if approve_weight > total non-abstain weight / 2.
    SimpleMajority,
    /// Approve if approve_weight / total non-abstain weight >= threshold (e.g. 0.67).
    SuperMajority(f64),
    /// Approve only if every non-abstaining voter approves.
    Unanimous,
    /// Approve if approve_weight / total_weight >= threshold (abstains count as weight).
    WeightedMajority(f64),
    /// Approve if at least `min_voters` voted (excluding abstain counted for quorum
    /// purposes) and approve fraction of non-abstain weight >= `majority`.
    Quorum {
        /// Minimum number of non-abstaining votes required.
        min_voters: usize,
        /// Required approval fraction of non-abstain weight.
        majority: f64,
    },
}

// ── Vote ──────────────────────────────────────────────────────────────────────

/// A single vote cast by an agent on a proposal.
#[derive(Debug, Clone)]
pub struct Vote {
    /// The identifier of the voter agent.
    pub voter_id: String,
    /// The proposal this vote belongs to.
    pub proposal_id: String,
    /// The voter's choice.
    pub choice: VoteChoice,
    /// Relative weight of this vote (must be > 0.0).
    pub weight: f64,
    /// Unix timestamp (seconds) when the vote was cast.
    pub timestamp: u64,
}

impl Vote {
    /// Create a new vote with `timestamp` set to now.
    pub fn new(
        voter_id: impl Into<String>,
        proposal_id: impl Into<String>,
        choice: VoteChoice,
        weight: f64,
    ) -> Self {
        Self {
            voter_id: voter_id.into(),
            proposal_id: proposal_id.into(),
            choice,
            weight,
            timestamp: now_secs(),
        }
    }
}

// ── ConsensusOutcome ──────────────────────────────────────────────────────────

/// The resolved outcome of a proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsensusOutcome {
    /// The proposal was approved.
    Approved,
    /// The proposal was rejected.
    Rejected,
    /// The outcome cannot yet be determined (e.g. quorum not met, deadline not passed).
    Inconclusive,
}

// ── VoteTally ─────────────────────────────────────────────────────────────────

/// Aggregate result of all votes on a proposal at a point in time.
#[derive(Debug, Clone)]
pub struct VoteTally {
    /// Number of Approve votes.
    pub approve_count: usize,
    /// Number of Reject votes.
    pub reject_count: usize,
    /// Number of Abstain votes.
    pub abstain_count: usize,
    /// Sum of weights of Approve votes.
    pub approve_weight: f64,
    /// Sum of weights of Reject votes.
    pub reject_weight: f64,
    /// Sum of all vote weights (including abstain).
    pub total_weight: f64,
    /// Whether the outcome is currently decided.
    pub is_decided: bool,
    /// The outcome, if decided.
    pub outcome: Option<ConsensusOutcome>,
}

// ── ConsensusError ────────────────────────────────────────────────────────────

/// Errors that can arise during consensus operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConsensusError {
    /// No proposal with that ID exists.
    #[error("proposal not found: {0}")]
    ProposalNotFound(String),
    /// The voter has already voted on this proposal.
    #[error("voter '{0}' has already voted")]
    AlreadyVoted(String),
    /// The proposal's deadline has passed; no more votes are accepted.
    #[error("proposal deadline has passed")]
    DeadlinePassed,
    /// The vote weight is not a positive finite number.
    #[error("invalid vote weight: {0}")]
    InvalidWeight(String),
}

// ── Proposal ──────────────────────────────────────────────────────────────────

/// A proposal that agents can vote on.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// Unique proposal ID.
    pub id: String,
    /// Human-readable description of the proposal.
    pub description: String,
    /// Agent ID of the agent who created the proposal.
    pub proposer_id: String,
    /// Unix timestamp (seconds) when the proposal was created.
    pub created_at: u64,
    /// Number of seconds after `created_at` that the proposal accepts votes.
    pub deadline_secs: u64,
    /// The rule used to evaluate the tally.
    pub rule: VotingRule,
    /// Map from voter_id to their Vote.
    pub votes: HashMap<String, Vote>,
}

impl Proposal {
    /// Create a new proposal.
    pub fn new(
        description: impl Into<String>,
        proposer_id: impl Into<String>,
        rule: VotingRule,
        deadline_secs: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.into(),
            proposer_id: proposer_id.into(),
            created_at: now_secs(),
            deadline_secs,
            rule,
            votes: HashMap::new(),
        }
    }

    /// Returns `true` if the proposal is still within its voting window.
    pub fn is_open(&self) -> bool {
        now_secs() < self.created_at.saturating_add(self.deadline_secs)
    }

    /// Cast a vote on this proposal.
    ///
    /// # Errors
    /// - [`ConsensusError::DeadlinePassed`] if the proposal is no longer open.
    /// - [`ConsensusError::AlreadyVoted`] if the voter has already voted.
    /// - [`ConsensusError::InvalidWeight`] if the weight is not a positive finite number.
    pub fn cast_vote(&mut self, vote: Vote) -> Result<(), ConsensusError> {
        if !self.is_open() {
            return Err(ConsensusError::DeadlinePassed);
        }
        if !vote.weight.is_finite() || vote.weight <= 0.0 {
            return Err(ConsensusError::InvalidWeight(format!("{}", vote.weight)));
        }
        if self.votes.contains_key(&vote.voter_id) {
            return Err(ConsensusError::AlreadyVoted(vote.voter_id.clone()));
        }
        self.votes.insert(vote.voter_id.clone(), vote);
        Ok(())
    }

    /// Compute the current tally and evaluate the voting rule.
    pub fn tally(&self) -> VoteTally {
        let mut approve_count = 0usize;
        let mut reject_count = 0usize;
        let mut abstain_count = 0usize;
        let mut approve_weight = 0.0f64;
        let mut reject_weight = 0.0f64;
        let mut total_weight = 0.0f64;

        for vote in self.votes.values() {
            total_weight += vote.weight;
            match vote.choice {
                VoteChoice::Approve => {
                    approve_count += 1;
                    approve_weight += vote.weight;
                }
                VoteChoice::Reject => {
                    reject_count += 1;
                    reject_weight += vote.weight;
                }
                VoteChoice::Abstain => {
                    abstain_count += 1;
                }
            }
        }

        let non_abstain_weight = approve_weight + reject_weight;
        let non_abstain_count = approve_count + reject_count;

        let outcome = self.evaluate_rule(
            approve_count,
            reject_count,
            non_abstain_count,
            approve_weight,
            reject_weight,
            non_abstain_weight,
            total_weight,
        );

        let is_decided = matches!(
            outcome,
            ConsensusOutcome::Approved | ConsensusOutcome::Rejected
        );

        VoteTally {
            approve_count,
            reject_count,
            abstain_count,
            approve_weight,
            reject_weight,
            total_weight,
            is_decided,
            outcome: Some(outcome),
        }
    }

    fn evaluate_rule(
        &self,
        _approve_count: usize,
        _reject_count: usize,
        non_abstain_count: usize,
        approve_weight: f64,
        _reject_weight: f64,
        non_abstain_weight: f64,
        total_weight: f64,
    ) -> ConsensusOutcome {
        match &self.rule {
            VotingRule::SimpleMajority => {
                if non_abstain_weight == 0.0 {
                    return ConsensusOutcome::Inconclusive;
                }
                if approve_weight > non_abstain_weight / 2.0 {
                    ConsensusOutcome::Approved
                } else {
                    ConsensusOutcome::Rejected
                }
            }
            VotingRule::SuperMajority(threshold) => {
                if non_abstain_weight == 0.0 {
                    return ConsensusOutcome::Inconclusive;
                }
                let ratio = approve_weight / non_abstain_weight;
                if ratio >= *threshold {
                    ConsensusOutcome::Approved
                } else {
                    ConsensusOutcome::Rejected
                }
            }
            VotingRule::Unanimous => {
                if non_abstain_count == 0 {
                    return ConsensusOutcome::Inconclusive;
                }
                // All non-abstaining voters must approve.
                let all_approve = self
                    .votes
                    .values()
                    .filter(|v| v.choice != VoteChoice::Abstain)
                    .all(|v| v.choice == VoteChoice::Approve);
                if all_approve {
                    ConsensusOutcome::Approved
                } else {
                    ConsensusOutcome::Rejected
                }
            }
            VotingRule::WeightedMajority(threshold) => {
                if total_weight == 0.0 {
                    return ConsensusOutcome::Inconclusive;
                }
                let ratio = approve_weight / total_weight;
                if ratio >= *threshold {
                    ConsensusOutcome::Approved
                } else {
                    ConsensusOutcome::Rejected
                }
            }
            VotingRule::Quorum {
                min_voters,
                majority,
            } => {
                if non_abstain_count < *min_voters {
                    return ConsensusOutcome::Inconclusive;
                }
                if non_abstain_weight == 0.0 {
                    return ConsensusOutcome::Inconclusive;
                }
                let ratio = approve_weight / non_abstain_weight;
                if ratio >= *majority {
                    ConsensusOutcome::Approved
                } else {
                    ConsensusOutcome::Rejected
                }
            }
        }
    }
}

// ── ConsensusMgr ──────────────────────────────────────────────────────────────

/// Manages multiple proposals and their lifecycle.
#[derive(Debug, Default)]
pub struct ConsensusMgr {
    proposals: HashMap<String, Proposal>,
}

impl ConsensusMgr {
    /// Create a new, empty consensus manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new proposal and register it.
    ///
    /// Returns the new proposal's ID.
    pub fn create_proposal(
        &mut self,
        description: &str,
        proposer: &str,
        rule: VotingRule,
        deadline_secs: u64,
    ) -> String {
        let proposal = Proposal::new(description, proposer, rule, deadline_secs);
        let id = proposal.id.clone();
        self.proposals.insert(id.clone(), proposal);
        id
    }

    /// Submit a vote on a proposal.
    ///
    /// Returns the current [`VoteTally`] after the vote is recorded.
    ///
    /// # Errors
    /// - [`ConsensusError::ProposalNotFound`] if `proposal_id` is unknown.
    /// - Propagates any error from [`Proposal::cast_vote`].
    pub fn submit_vote(
        &mut self,
        proposal_id: &str,
        vote: Vote,
    ) -> Result<VoteTally, ConsensusError> {
        let proposal = self
            .proposals
            .get_mut(proposal_id)
            .ok_or_else(|| ConsensusError::ProposalNotFound(proposal_id.to_owned()))?;
        proposal.cast_vote(vote)?;
        Ok(proposal.tally())
    }

    /// Force-resolve a proposal regardless of deadline, returning the outcome.
    ///
    /// Returns `None` if no proposal with that ID exists.
    pub fn finalize(&mut self, proposal_id: &str) -> Option<ConsensusOutcome> {
        let proposal = self.proposals.get(proposal_id)?;
        let tally = proposal.tally();
        tally.outcome
    }

    /// Return references to all proposals whose voting window has not yet closed.
    pub fn active_proposals(&self) -> Vec<&Proposal> {
        self.proposals.values().filter(|p| p.is_open()).collect()
    }

    /// Return a reference to a specific proposal, if it exists.
    pub fn get_proposal(&self, proposal_id: &str) -> Option<&Proposal> {
        self.proposals.get(proposal_id)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_vote(voter: &str, proposal_id: &str, choice: VoteChoice, weight: f64) -> Vote {
        Vote::new(voter, proposal_id, choice, weight)
    }

    // ── SimpleMajority ────────────────────────────────────────────────────────

    #[test]
    fn simple_majority_passes_with_majority_weight() {
        let mut mgr = ConsensusMgr::new();
        let pid = mgr.create_proposal("Deploy v2", "agent-1", VotingRule::SimpleMajority, 9999);

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        mgr.submit_vote(&pid, make_vote("b", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("c", &pid, VoteChoice::Reject, 1.0))
            .unwrap();

        assert_eq!(tally.outcome, Some(ConsensusOutcome::Approved));
        assert!(tally.is_decided);
    }

    #[test]
    fn simple_majority_fails_with_tied_votes() {
        let mut mgr = ConsensusMgr::new();
        let pid = mgr.create_proposal("Tied", "agent-1", VotingRule::SimpleMajority, 9999);

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("b", &pid, VoteChoice::Reject, 1.0))
            .unwrap();

        // approve_weight (1.0) is NOT > non_abstain/2 (1.0) → Rejected
        assert_eq!(tally.outcome, Some(ConsensusOutcome::Rejected));
    }

    // ── SuperMajority ─────────────────────────────────────────────────────────

    #[test]
    fn supermajority_passes_at_67_percent() {
        let mut mgr = ConsensusMgr::new();
        let pid =
            mgr.create_proposal("Super", "agent-1", VotingRule::SuperMajority(0.67), 9999);

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 2.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("b", &pid, VoteChoice::Reject, 1.0))
            .unwrap();

        // 2/3 ≈ 0.666… which is < 0.67 → Rejected
        assert_eq!(tally.outcome, Some(ConsensusOutcome::Rejected));
    }

    #[test]
    fn supermajority_fails_with_60_percent() {
        let mut mgr = ConsensusMgr::new();
        let pid =
            mgr.create_proposal("Super60", "agent-1", VotingRule::SuperMajority(0.67), 9999);

        // 3 approve, 2 reject → 60% approve
        for i in 0..3 {
            mgr.submit_vote(
                &pid,
                make_vote(&format!("approve-{i}"), &pid, VoteChoice::Approve, 1.0),
            )
            .unwrap();
        }
        for i in 0..2 {
            mgr.submit_vote(
                &pid,
                make_vote(&format!("reject-{i}"), &pid, VoteChoice::Reject, 1.0),
            )
            .unwrap();
        }

        let tally = mgr.finalize(&pid).unwrap();
        assert_eq!(tally, ConsensusOutcome::Rejected);
    }

    // ── Unanimous ─────────────────────────────────────────────────────────────

    #[test]
    fn unanimous_requires_all_to_approve() {
        let mut mgr = ConsensusMgr::new();
        let pid = mgr.create_proposal("All in", "agent-1", VotingRule::Unanimous, 9999);

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        mgr.submit_vote(&pid, make_vote("b", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("c", &pid, VoteChoice::Approve, 1.0))
            .unwrap();

        assert_eq!(tally.outcome, Some(ConsensusOutcome::Approved));
    }

    #[test]
    fn unanimous_fails_if_one_rejects() {
        let mut mgr = ConsensusMgr::new();
        let pid = mgr.create_proposal("All in", "agent-1", VotingRule::Unanimous, 9999);

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("b", &pid, VoteChoice::Reject, 1.0))
            .unwrap();

        assert_eq!(tally.outcome, Some(ConsensusOutcome::Rejected));
    }

    // ── Duplicate vote ────────────────────────────────────────────────────────

    #[test]
    fn duplicate_vote_returns_error() {
        let mut mgr = ConsensusMgr::new();
        let pid =
            mgr.create_proposal("Dup test", "agent-1", VotingRule::SimpleMajority, 9999);

        mgr.submit_vote(&pid, make_vote("voter-1", &pid, VoteChoice::Approve, 1.0))
            .unwrap();

        let err = mgr
            .submit_vote(&pid, make_vote("voter-1", &pid, VoteChoice::Reject, 1.0))
            .unwrap_err();

        assert_eq!(err, ConsensusError::AlreadyVoted("voter-1".to_owned()));
    }

    // ── Quorum ────────────────────────────────────────────────────────────────

    #[test]
    fn quorum_not_met_returns_inconclusive() {
        let mut mgr = ConsensusMgr::new();
        let pid = mgr.create_proposal(
            "Quorum test",
            "agent-1",
            VotingRule::Quorum {
                min_voters: 3,
                majority: 0.5,
            },
            9999,
        );

        // Only 2 non-abstaining votes → quorum not met
        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("b", &pid, VoteChoice::Approve, 1.0))
            .unwrap();

        assert_eq!(tally.outcome, Some(ConsensusOutcome::Inconclusive));
        assert!(!tally.is_decided);
    }

    #[test]
    fn quorum_met_and_approved() {
        let mut mgr = ConsensusMgr::new();
        let pid = mgr.create_proposal(
            "Quorum ok",
            "agent-1",
            VotingRule::Quorum {
                min_voters: 2,
                majority: 0.5,
            },
            9999,
        );

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 2.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("b", &pid, VoteChoice::Reject, 1.0))
            .unwrap();

        assert_eq!(tally.outcome, Some(ConsensusOutcome::Approved));
    }

    // ── InvalidWeight ─────────────────────────────────────────────────────────

    #[test]
    fn invalid_weight_zero_rejected() {
        let mut mgr = ConsensusMgr::new();
        let pid =
            mgr.create_proposal("Weight test", "agent-1", VotingRule::SimpleMajority, 9999);

        let err = mgr
            .submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 0.0))
            .unwrap_err();
        assert!(matches!(err, ConsensusError::InvalidWeight(_)));
    }

    #[test]
    fn invalid_weight_nan_rejected() {
        let mut mgr = ConsensusMgr::new();
        let pid =
            mgr.create_proposal("Weight nan", "agent-1", VotingRule::SimpleMajority, 9999);

        let err = mgr
            .submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, f64::NAN))
            .unwrap_err();
        assert!(matches!(err, ConsensusError::InvalidWeight(_)));
    }

    // ── ProposalNotFound ──────────────────────────────────────────────────────

    #[test]
    fn submit_to_unknown_proposal_errors() {
        let mut mgr = ConsensusMgr::new();
        let err = mgr
            .submit_vote(
                "nonexistent-id",
                make_vote("a", "nonexistent-id", VoteChoice::Approve, 1.0),
            )
            .unwrap_err();
        assert!(matches!(err, ConsensusError::ProposalNotFound(_)));
    }

    // ── Finalize ──────────────────────────────────────────────────────────────

    #[test]
    fn finalize_returns_outcome() {
        let mut mgr = ConsensusMgr::new();
        let pid =
            mgr.create_proposal("Finalize me", "agent-1", VotingRule::SimpleMajority, 9999);

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 1.0))
            .unwrap();

        let outcome = mgr.finalize(&pid).unwrap();
        assert_eq!(outcome, ConsensusOutcome::Approved);
    }

    #[test]
    fn finalize_unknown_returns_none() {
        let mut mgr = ConsensusMgr::new();
        assert!(mgr.finalize("ghost").is_none());
    }

    // ── Abstain handling ──────────────────────────────────────────────────────

    #[test]
    fn abstain_does_not_count_toward_non_abstain_weight() {
        let mut mgr = ConsensusMgr::new();
        let pid =
            mgr.create_proposal("Abstain test", "agent-1", VotingRule::SimpleMajority, 9999);

        mgr.submit_vote(&pid, make_vote("a", &pid, VoteChoice::Approve, 1.0))
            .unwrap();
        let tally = mgr
            .submit_vote(&pid, make_vote("b", &pid, VoteChoice::Abstain, 1.0))
            .unwrap();

        assert_eq!(tally.abstain_count, 1);
        // Approve weight (1.0) > non-abstain/2 (0.5) → Approved
        assert_eq!(tally.outcome, Some(ConsensusOutcome::Approved));
    }
}
