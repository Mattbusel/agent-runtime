//! # Multi-Agent Conversation Protocol
//!
//! Defines a structured protocol for agents to communicate with each other,
//! including message types, conversation threading, capability-based routing,
//! and a consensus voting mechanism.
//!
//! ## Message Types
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`Query`] | An agent asks another a question or requests work. |
//! | [`Answer`] | A direct response to a prior [`Query`]. |
//! | [`Delegate`] | Hands off a task to another agent (by capability). |
//! | [`Report`] | An unsolicited status update or observation broadcast. |
//! | [`Disagree`] | Signals disagreement with a prior message's content. |
//!
//! ## Threading
//!
//! Every message carries an optional `reply_to` field that links it to the
//! message it responds to, enabling conversation thread reconstruction.
//!
//! ## Consensus
//!
//! [`ConsensusRound`] collects votes from multiple agents on a proposal and
//! resolves them with a configurable [`ConsensusPolicy`].
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::protocol::{AgentMessage, MessageKind, ConversationThread, ProtocolRouter};
//! use llm_agent_runtime::protocol::{ConsensusRound, ConsensusPolicy, Vote};
//! use llm_agent_runtime::types::AgentId;
//!
//! // Build a message.
//! let msg = AgentMessage::query(
//!     AgentId::new("planner"),
//!     AgentId::new("researcher"),
//!     "What is the capital of France?".into(),
//! );
//!
//! // Thread it.
//! let mut thread = ConversationThread::new(msg.clone());
//! let reply = AgentMessage::answer(
//!     AgentId::new("researcher"),
//!     AgentId::new("planner"),
//!     msg.id.clone(),
//!     "Paris".into(),
//! );
//! thread.add_reply(reply);
//! assert_eq!(thread.replies().len(), 1);
//!
//! // Consensus.
//! let mut round = ConsensusRound::new("Should we proceed?".into(), ConsensusPolicy::Majority);
//! round.cast_vote(AgentId::new("a1"), Vote::Yes);
//! round.cast_vote(AgentId::new("a2"), Vote::Yes);
//! round.cast_vote(AgentId::new("a3"), Vote::No);
//! assert!(round.resolve().unwrap());
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::AgentId;

// ---------------------------------------------------------------------------
// MessageId
// ---------------------------------------------------------------------------

/// Unique identifier for a protocol message.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProtocolMessageId(pub String);

impl ProtocolMessageId {
    /// Generate a random message ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl Default for ProtocolMessageId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ProtocolMessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// MessageKind
// ---------------------------------------------------------------------------

/// The semantic type of an [`AgentMessage`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageKind {
    /// Request information or work from another agent.
    Query,
    /// Direct answer to a prior [`Query`].
    Answer,
    /// Hand off a task to another agent.
    Delegate,
    /// Unsolicited status update or broadcast.
    Report,
    /// Express disagreement with a prior message.
    Disagree,
}

impl std::fmt::Display for MessageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MessageKind::Query => "Query",
            MessageKind::Answer => "Answer",
            MessageKind::Delegate => "Delegate",
            MessageKind::Report => "Report",
            MessageKind::Disagree => "Disagree",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// AgentMessage
// ---------------------------------------------------------------------------

/// A structured message exchanged between agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Unique identifier for this message.
    pub id: ProtocolMessageId,
    /// The agent that sent this message.
    pub from: AgentId,
    /// The intended recipient (or `None` for a broadcast).
    pub to: Option<AgentId>,
    /// Semantic type of this message.
    pub kind: MessageKind,
    /// The text payload.
    pub body: String,
    /// If this is a reply, the ID of the message being replied to.
    pub reply_to: Option<ProtocolMessageId>,
    /// Optional capability tag used for routing [`Delegate`](MessageKind::Delegate) messages.
    pub required_capability: Option<String>,
    /// Timestamp when this message was created.
    pub sent_at: DateTime<Utc>,
    /// Arbitrary metadata (tool name, confidence score, etc.).
    pub metadata: HashMap<String, String>,
}

impl AgentMessage {
    /// Construct a raw message.
    pub fn new(
        from: AgentId,
        to: Option<AgentId>,
        kind: MessageKind,
        body: String,
        reply_to: Option<ProtocolMessageId>,
    ) -> Self {
        Self {
            id: ProtocolMessageId::new(),
            from,
            to,
            kind,
            body,
            reply_to,
            required_capability: None,
            sent_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a [`MessageKind::Query`] message.
    pub fn query(from: AgentId, to: AgentId, body: String) -> Self {
        Self::new(from, Some(to), MessageKind::Query, body, None)
    }

    /// Create a [`MessageKind::Answer`] in reply to a prior message.
    pub fn answer(
        from: AgentId,
        to: AgentId,
        reply_to: ProtocolMessageId,
        body: String,
    ) -> Self {
        Self::new(from, Some(to), MessageKind::Answer, body, Some(reply_to))
    }

    /// Create a [`MessageKind::Delegate`] message, optionally targeting a capability.
    pub fn delegate(
        from: AgentId,
        body: String,
        required_capability: Option<String>,
    ) -> Self {
        let mut msg = Self::new(from, None, MessageKind::Delegate, body, None);
        msg.required_capability = required_capability;
        msg
    }

    /// Create a [`MessageKind::Report`] broadcast.
    pub fn report(from: AgentId, body: String) -> Self {
        Self::new(from, None, MessageKind::Report, body, None)
    }

    /// Create a [`MessageKind::Disagree`] in reply to a prior message.
    pub fn disagree(
        from: AgentId,
        to: AgentId,
        reply_to: ProtocolMessageId,
        reason: String,
    ) -> Self {
        Self::new(from, Some(to), MessageKind::Disagree, reason, Some(reply_to))
    }

    /// Attach a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// ConversationThread
// ---------------------------------------------------------------------------

/// A rooted conversation thread: one root message and all its replies.
#[derive(Debug, Clone)]
pub struct ConversationThread {
    /// The message that started the thread.
    pub root: AgentMessage,
    /// Replies in the order they were added.
    replies: Vec<AgentMessage>,
}

impl ConversationThread {
    /// Start a thread from a root message.
    pub fn new(root: AgentMessage) -> Self {
        Self { root, replies: Vec::new() }
    }

    /// Append a reply to this thread.
    pub fn add_reply(&mut self, msg: AgentMessage) {
        self.replies.push(msg);
    }

    /// All replies in this thread (in insertion order).
    pub fn replies(&self) -> &[AgentMessage] {
        &self.replies
    }

    /// Total messages in the thread (root + replies).
    pub fn len(&self) -> usize {
        1 + self.replies.len()
    }

    /// True if the thread has no replies.
    pub fn is_empty(&self) -> bool {
        self.replies.is_empty()
    }

    /// Find all messages from a specific agent.
    pub fn messages_from(&self, agent: &AgentId) -> Vec<&AgentMessage> {
        std::iter::once(&self.root)
            .chain(self.replies.iter())
            .filter(|m| &m.from == agent)
            .collect()
    }

    /// Collect all disagreements in the thread.
    pub fn disagreements(&self) -> Vec<&AgentMessage> {
        self.replies.iter().filter(|m| m.kind == MessageKind::Disagree).collect()
    }
}

// ---------------------------------------------------------------------------
// ProtocolRouter
// ---------------------------------------------------------------------------

/// Routes incoming messages to agents by capability or direct address.
///
/// Agents register their capabilities.  When a [`MessageKind::Delegate`]
/// message arrives without a specific `to` field, the router finds all agents
/// that advertise the required capability.
#[derive(Debug, Default)]
pub struct ProtocolRouter {
    /// Maps agent ID → set of capability strings.
    capabilities: HashMap<AgentId, Vec<String>>,
}

impl ProtocolRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register capabilities for an agent.
    pub fn register(&mut self, agent: AgentId, capabilities: Vec<String>) {
        self.capabilities.insert(agent, capabilities);
    }

    /// Remove an agent from the router.
    pub fn deregister(&mut self, agent: &AgentId) {
        self.capabilities.remove(agent);
    }

    /// Find all agents that provide the given capability.
    pub fn agents_for_capability(&self, capability: &str) -> Vec<&AgentId> {
        self.capabilities
            .iter()
            .filter_map(|(agent, caps)| {
                if caps.iter().any(|c| c == capability) { Some(agent) } else { None }
            })
            .collect()
    }

    /// Route a message: return the list of target agent IDs.
    ///
    /// - For direct messages (`to` is `Some`): returns `[to]`.
    /// - For delegate messages with `required_capability`: returns all agents
    ///   that have that capability.
    /// - For broadcasts (no `to`, no capability): returns all registered agents.
    pub fn route<'a>(&'a self, msg: &'a AgentMessage) -> Vec<&'a AgentId> {
        if let Some(to) = &msg.to {
            return vec![to];
        }
        if let Some(cap) = &msg.required_capability {
            return self.agents_for_capability(cap);
        }
        // Broadcast.
        self.capabilities.keys().collect()
    }
}

// ---------------------------------------------------------------------------
// Consensus
// ---------------------------------------------------------------------------

/// A vote cast by an agent in a [`ConsensusRound`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vote {
    /// Affirmative vote.
    Yes,
    /// Negative vote.
    No,
    /// The agent abstains.
    Abstain,
}

/// How a [`ConsensusRound`] resolves votes into a binary decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusPolicy {
    /// More yes-votes than no-votes → accepted.
    Majority,
    /// All non-abstaining voters must vote yes → accepted.
    Unanimous,
    /// At least one yes-vote → accepted.
    AnyYes,
}

/// Error type for consensus resolution.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusError {
    /// No votes were cast (or all agents abstained).
    NoVotes,
}

impl std::fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "consensus failed: no non-abstaining votes cast")
    }
}

impl std::error::Error for ConsensusError {}

/// A single consensus round: a proposal that agents vote on.
#[derive(Debug, Clone)]
pub struct ConsensusRound {
    /// The proposal text.
    pub proposal: String,
    /// The policy used to resolve votes.
    pub policy: ConsensusPolicy,
    /// Votes by agent.
    votes: HashMap<AgentId, Vote>,
}

impl ConsensusRound {
    /// Start a new consensus round.
    pub fn new(proposal: String, policy: ConsensusPolicy) -> Self {
        Self { proposal, policy, votes: HashMap::new() }
    }

    /// Cast (or overwrite) a vote from an agent.
    pub fn cast_vote(&mut self, agent: AgentId, vote: Vote) {
        self.votes.insert(agent, vote);
    }

    /// Total number of votes cast (including abstentions).
    pub fn vote_count(&self) -> usize {
        self.votes.len()
    }

    /// Resolve the votes according to the policy.
    ///
    /// Returns `Ok(true)` if accepted, `Ok(false)` if rejected, or an error
    /// if no non-abstaining votes were cast.
    pub fn resolve(&self) -> Result<bool, ConsensusError> {
        let yes: usize = self.votes.values().filter(|v| **v == Vote::Yes).count();
        let no: usize = self.votes.values().filter(|v| **v == Vote::No).count();
        let total_non_abstain = yes + no;

        if total_non_abstain == 0 {
            return Err(ConsensusError::NoVotes);
        }

        let accepted = match self.policy {
            ConsensusPolicy::Majority => yes > no,
            ConsensusPolicy::Unanimous => no == 0,
            ConsensusPolicy::AnyYes => yes > 0,
        };
        Ok(accepted)
    }

    /// Return a summary of the vote counts.
    pub fn tally(&self) -> (usize, usize, usize) {
        let yes = self.votes.values().filter(|v| **v == Vote::Yes).count();
        let no = self.votes.values().filter(|v| **v == Vote::No).count();
        let abstain = self.votes.values().filter(|v| **v == Vote::Abstain).count();
        (yes, no, abstain)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_message_has_correct_kind() {
        let msg = AgentMessage::query(
            AgentId::new("a"),
            AgentId::new("b"),
            "hello".into(),
        );
        assert_eq!(msg.kind, MessageKind::Query);
        assert_eq!(msg.to, Some(AgentId::new("b")));
    }

    #[test]
    fn answer_message_links_to_original() {
        let q = AgentMessage::query(AgentId::new("a"), AgentId::new("b"), "q".into());
        let ans = AgentMessage::answer(
            AgentId::new("b"),
            AgentId::new("a"),
            q.id.clone(),
            "answer".into(),
        );
        assert_eq!(ans.reply_to, Some(q.id));
        assert_eq!(ans.kind, MessageKind::Answer);
    }

    #[test]
    fn disagree_message_has_correct_fields() {
        let q = AgentMessage::query(AgentId::new("a"), AgentId::new("b"), "q".into());
        let d = AgentMessage::disagree(
            AgentId::new("b"),
            AgentId::new("a"),
            q.id.clone(),
            "I disagree".into(),
        );
        assert_eq!(d.kind, MessageKind::Disagree);
        assert_eq!(d.body, "I disagree");
    }

    #[test]
    fn thread_tracks_replies() {
        let root = AgentMessage::query(AgentId::new("a"), AgentId::new("b"), "q".into());
        let reply =
            AgentMessage::answer(AgentId::new("b"), AgentId::new("a"), root.id.clone(), "a".into());
        let mut thread = ConversationThread::new(root);
        thread.add_reply(reply);
        assert_eq!(thread.len(), 2);
        assert_eq!(thread.replies().len(), 1);
    }

    #[test]
    fn thread_disagreements_filter() {
        let root = AgentMessage::query(AgentId::new("a"), AgentId::new("b"), "q".into());
        let d = AgentMessage::disagree(
            AgentId::new("b"),
            AgentId::new("a"),
            root.id.clone(),
            "nope".into(),
        );
        let mut thread = ConversationThread::new(root);
        thread.add_reply(d);
        assert_eq!(thread.disagreements().len(), 1);
    }

    #[test]
    fn router_routes_by_direct_address() {
        let mut router = ProtocolRouter::new();
        router.register(AgentId::new("b"), vec!["search".into()]);
        let msg = AgentMessage::query(AgentId::new("a"), AgentId::new("b"), "?".into());
        let targets = router.route(&msg);
        assert_eq!(targets, vec![&AgentId::new("b")]);
    }

    #[test]
    fn router_routes_by_capability() {
        let mut router = ProtocolRouter::new();
        router.register(AgentId::new("x"), vec!["search".into()]);
        router.register(AgentId::new("y"), vec!["write".into()]);
        let msg = AgentMessage::delegate(AgentId::new("a"), "find info".into(), Some("search".into()));
        let targets = router.route(&msg);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0], &AgentId::new("x"));
    }

    #[test]
    fn consensus_majority_policy() {
        let mut r = ConsensusRound::new("go?".into(), ConsensusPolicy::Majority);
        r.cast_vote(AgentId::new("a"), Vote::Yes);
        r.cast_vote(AgentId::new("b"), Vote::Yes);
        r.cast_vote(AgentId::new("c"), Vote::No);
        assert!(r.resolve().unwrap());
    }

    #[test]
    fn consensus_unanimous_fails_with_any_no() {
        let mut r = ConsensusRound::new("go?".into(), ConsensusPolicy::Unanimous);
        r.cast_vote(AgentId::new("a"), Vote::Yes);
        r.cast_vote(AgentId::new("b"), Vote::No);
        assert!(!r.resolve().unwrap());
    }

    #[test]
    fn consensus_any_yes_passes_with_single_yes() {
        let mut r = ConsensusRound::new("go?".into(), ConsensusPolicy::AnyYes);
        r.cast_vote(AgentId::new("a"), Vote::No);
        r.cast_vote(AgentId::new("b"), Vote::Yes);
        assert!(r.resolve().unwrap());
    }

    #[test]
    fn consensus_no_votes_returns_error() {
        let r = ConsensusRound::new("go?".into(), ConsensusPolicy::Majority);
        assert!(r.resolve().is_err());
    }

    #[test]
    fn consensus_tally_counts_correctly() {
        let mut r = ConsensusRound::new("q".into(), ConsensusPolicy::Majority);
        r.cast_vote(AgentId::new("a"), Vote::Yes);
        r.cast_vote(AgentId::new("b"), Vote::No);
        r.cast_vote(AgentId::new("c"), Vote::Abstain);
        let (yes, no, abstain) = r.tally();
        assert_eq!((yes, no, abstain), (1, 1, 1));
    }
}
