//! # Module: Conversation
//!
//! ## Responsibility
//! Provides [`ConversationHistory`] — a turn-by-turn record of a single agent
//! session, separate from episodic memory.
//!
//! Unlike `EpisodicStore` (importance-weighted, agent-global), `ConversationHistory`
//! is strictly ordered and session-scoped.  When the configured `max_turns` is
//! exceeded, the oldest turns are summarised into a single compacted entry so
//! the history never grows without bound.
//!
//! ## Guarantees
//! - All operations are synchronous and `Send + Sync`.
//! - The history is bounded: it never stores more than `max_turns` raw turns.
//! - Summarisation is pluggable: callers supply a `Fn` that produces the
//!   summary string for a batch of old turns.
//! - Non-panicking: no `unwrap`/`expect` in public APIs.

use crate::agent::{Message, Role};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ── Turn ──────────────────────────────────────────────────────────────────────

/// A single conversational turn stored in [`ConversationHistory`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Sequential index of this turn within the session (0-based).
    pub index: usize,
    /// The message that makes up this turn.
    pub message: Message,
    /// UTC timestamp when this turn was recorded (ISO 8601).
    pub timestamp: String,
}

impl Turn {
    /// Construct a new `Turn` with the current timestamp.
    pub fn new(index: usize, message: Message) -> Self {
        Self {
            index,
            message,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Return `true` if this is a user turn.
    pub fn is_user(&self) -> bool {
        self.message.role == Role::User
    }

    /// Return `true` if this is an assistant turn.
    pub fn is_assistant(&self) -> bool {
        self.message.role == Role::Assistant
    }

    /// Return the content of this turn.
    pub fn content(&self) -> &str {
        &self.message.content
    }
}

impl std::fmt::Display for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.index, self.message)
    }
}

// ── SummaryEntry ──────────────────────────────────────────────────────────────

/// A compacted summary of multiple older turns.
///
/// Produced automatically when the history exceeds `max_turns`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryEntry {
    /// The range of turn indices that were summarised (inclusive).
    pub turn_range: (usize, usize),
    /// The generated summary text.
    pub summary: String,
    /// UTC timestamp when the summary was created (ISO 8601).
    pub created_at: String,
}

impl SummaryEntry {
    /// Construct a new `SummaryEntry`.
    pub fn new(turn_range: (usize, usize), summary: impl Into<String>) -> Self {
        Self {
            turn_range,
            summary: summary.into(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Return the number of turns that were compressed into this summary.
    pub fn compressed_turn_count(&self) -> usize {
        self.turn_range.1.saturating_sub(self.turn_range.0) + 1
    }
}

// ── ConversationHistory ───────────────────────────────────────────────────────

/// Bounded turn-by-turn conversation history for a single session.
///
/// # Overflow Handling
///
/// When the number of raw turns reaches `max_turns`, the oldest half of the
/// turns are passed to the configured summarizer function.  The resulting
/// [`SummaryEntry`] is stored in `summaries` and the compressed turns are
/// removed from the live `turns` ring.  This keeps memory usage bounded
/// while preserving an auditable record.
///
/// # Usage
/// ```rust,no_run
/// use llm_agent_runtime::conversation::ConversationHistory;
/// use llm_agent_runtime::agent::Message;
///
/// let mut history = ConversationHistory::new(20);
/// history.push(Message::user("Hello!"));
/// history.push(Message::assistant("Hi there!"));
///
/// println!("Turns: {}", history.len());
/// ```
pub struct ConversationHistory {
    /// The live (unsummarised) turns, in chronological order.
    turns: VecDeque<Turn>,
    /// Summarised batches of older turns.
    summaries: Vec<SummaryEntry>,
    /// Maximum number of raw turns to keep before summarising.
    max_turns: usize,
    /// Running count of all turns ever added (never decremented).
    total_turns_added: usize,
    /// Fraction of `max_turns` to summarise at a time (default 0.5).
    summarise_fraction: f64,
    /// Summariser function: receives the turns being compressed, returns a summary string.
    summarizer: Box<dyn Fn(&[Turn]) -> String + Send + Sync>,
}

impl std::fmt::Debug for ConversationHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationHistory")
            .field("turns", &self.turns.len())
            .field("summaries", &self.summaries.len())
            .field("max_turns", &self.max_turns)
            .field("total_turns_added", &self.total_turns_added)
            .finish()
    }
}

impl ConversationHistory {
    /// Create a new history with `max_turns` capacity.
    ///
    /// The default summarizer produces a bullet-list of `"{role}: {preview}"`
    /// entries, truncated to 80 characters per turn.
    pub fn new(max_turns: usize) -> Self {
        Self {
            turns: VecDeque::new(),
            summaries: Vec::new(),
            max_turns: max_turns.max(1),
            total_turns_added: 0,
            summarise_fraction: 0.5,
            summarizer: Box::new(default_summarizer),
        }
    }

    /// Attach a custom summarizer function.
    ///
    /// The function receives the slice of turns being compressed and must
    /// return a single summary string.
    pub fn with_summarizer(mut self, f: impl Fn(&[Turn]) -> String + Send + Sync + 'static) -> Self {
        self.summarizer = Box::new(f);
        self
    }

    /// Set the fraction of `max_turns` to summarise when the history overflows.
    ///
    /// Must be in `(0.0, 1.0)`.  Defaults to `0.5` (summarise half the turns).
    pub fn with_summarise_fraction(mut self, fraction: f64) -> Self {
        self.summarise_fraction = fraction.clamp(0.01, 0.99);
        self
    }

    /// Append a `Message` as a new turn.
    ///
    /// If the live turn count reaches `max_turns`, the oldest
    /// `summarise_fraction * max_turns` turns are summarised automatically
    /// before the new turn is added.
    pub fn push(&mut self, message: Message) {
        if self.turns.len() >= self.max_turns {
            self.summarise_oldest();
        }
        let turn = Turn::new(self.total_turns_added, message);
        self.total_turns_added += 1;
        self.turns.push_back(turn);
    }

    /// Return the number of live (unsummarised) turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Return `true` if there are no live turns.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Return the total number of turns ever added (including summarised ones).
    pub fn total_turns(&self) -> usize {
        self.total_turns_added
    }

    /// Return a slice-like iterator over the live turns in chronological order.
    pub fn turns(&self) -> impl Iterator<Item = &Turn> {
        self.turns.iter()
    }

    /// Return all stored summaries.
    pub fn summaries(&self) -> &[SummaryEntry] {
        &self.summaries
    }

    /// Return the last N live turns, or all turns if fewer than N exist.
    pub fn last_n(&self, n: usize) -> Vec<&Turn> {
        let skip = self.turns.len().saturating_sub(n);
        self.turns.iter().skip(skip).collect()
    }

    /// Return the most recent turn, or `None` if the history is empty.
    pub fn last_turn(&self) -> Option<&Turn> {
        self.turns.back()
    }

    /// Clear all live turns and summaries, resetting the history.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.summaries.clear();
        self.total_turns_added = 0;
    }

    /// Build a flat `Vec<Message>` from all summaries (as `System` messages)
    /// followed by all live turns, suitable for passing to an LLM.
    pub fn to_messages(&self) -> Vec<Message> {
        let mut msgs = Vec::new();
        for s in &self.summaries {
            msgs.push(Message::system(format!(
                "[Summary of turns {}-{}]: {}",
                s.turn_range.0, s.turn_range.1, s.summary
            )));
        }
        for t in &self.turns {
            msgs.push(t.message.clone());
        }
        msgs
    }

    /// Produce a single context string by joining all messages with newlines.
    ///
    /// Format: `"{role}: {content}\n"` for each message (summaries first).
    pub fn to_context_string(&self) -> String {
        self.to_messages()
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return the number of summaries stored.
    pub fn summary_count(&self) -> usize {
        self.summaries.len()
    }

    /// Return the configured maximum number of live turns.
    pub fn max_turns(&self) -> usize {
        self.max_turns
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn summarise_oldest(&mut self) {
        let n = ((self.max_turns as f64 * self.summarise_fraction).ceil() as usize).max(1);
        let n = n.min(self.turns.len());
        let batch: Vec<Turn> = self.turns.drain(..n).collect();
        if batch.is_empty() {
            return;
        }
        let first_idx = batch[0].index;
        let last_idx = batch[batch.len() - 1].index;
        let summary_text = (self.summarizer)(&batch);
        self.summaries.push(SummaryEntry::new((first_idx, last_idx), summary_text));
    }
}

// ── Default summarizer ────────────────────────────────────────────────────────

fn default_summarizer(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(|t| {
            let preview = if t.content().len() > 80 {
                format!("{}...", &t.content()[..80])
            } else {
                t.content().to_owned()
            };
            format!("- {}: {}", t.message.role, preview)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut h = ConversationHistory::new(10);
        h.push(Message::user("hello"));
        h.push(Message::assistant("hi"));
        assert_eq!(h.len(), 2);
        assert_eq!(h.total_turns(), 2);
    }

    #[test]
    fn test_overflow_triggers_summarisation() {
        let mut h = ConversationHistory::new(4);
        for i in 0..6usize {
            h.push(Message::user(format!("msg {i}")));
        }
        // After 4 turns the oldest 2 were summarised; then 2 more were added.
        assert!(h.summary_count() >= 1, "expected at least one summary");
        assert!(h.len() <= 4, "live turns must not exceed max_turns");
    }

    #[test]
    fn test_to_messages_includes_summaries_first() {
        let mut h = ConversationHistory::new(2);
        h.push(Message::user("a"));
        h.push(Message::user("b"));
        h.push(Message::user("c")); // triggers summarisation of a & b
        let msgs = h.to_messages();
        // First message should be the summary (System role).
        assert_eq!(msgs[0].role, Role::System);
    }

    #[test]
    fn test_last_n_returns_correct_slice() {
        let mut h = ConversationHistory::new(10);
        for i in 0..5usize {
            h.push(Message::user(format!("m{i}")));
        }
        let last2 = h.last_n(2);
        assert_eq!(last2.len(), 2);
        assert!(last2[1].content().contains("m4"));
    }

    #[test]
    fn test_clear_resets_state() {
        let mut h = ConversationHistory::new(10);
        h.push(Message::user("x"));
        h.clear();
        assert!(h.is_empty());
        assert_eq!(h.total_turns(), 0);
    }

    #[test]
    fn test_custom_summarizer() {
        let mut h = ConversationHistory::new(2)
            .with_summarizer(|turns| format!("CUSTOM SUMMARY of {} turns", turns.len()));
        h.push(Message::user("a"));
        h.push(Message::user("b"));
        h.push(Message::user("c"));
        let summaries = h.summaries();
        assert!(!summaries.is_empty());
        assert!(summaries[0].summary.starts_with("CUSTOM SUMMARY"));
    }
}
