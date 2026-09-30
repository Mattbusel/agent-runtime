//! # Module: Dialogue
//!
//! ## Responsibility
//! Provides multi-turn conversation memory for agents.  Each session tracks a
//! chronological sequence of (role, content) turns and can be assembled into a
//! prompt context that blends episodic memory with live dialogue history.
//!
//! ## Guarantees
//! - Thread-safe: `DialogueStore` wraps its map in `Arc<Mutex<_>>`
//! - Non-panicking: all operations return `Result`
//! - Serialisable: all public types implement `serde::Serialize` /
//!   `serde::Deserialize`
//! - Lock-poisoning resilient: uses `crate::util::recover_lock`
//!
//! ## NOT Responsible For
//! - Persistent storage of dialogue history (see `persistence` module)
//! - Semantic search across past conversations (see `memory` module)

use crate::error::AgentRuntimeError;
use crate::util::recover_lock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ── Role ─────────────────────────────────────────────────────────────────────

/// The speaker role for a single dialogue turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A message produced by a human user.
    User,
    /// A message produced by the AI assistant.
    Assistant,
    /// A system-level instruction injected at the start of a conversation.
    System,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => f.write_str("user"),
            Role::Assistant => f.write_str("assistant"),
            Role::System => f.write_str("system"),
        }
    }
}

// ── DialogueTurn ─────────────────────────────────────────────────────────────

/// A single turn in a multi-turn dialogue — a role/content pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    /// Who produced this message.
    pub role: Role,
    /// The text content of this message.
    pub content: String,
}

impl DialogueTurn {
    /// Construct a new turn with the given role and content.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

// ── DialogueSession ───────────────────────────────────────────────────────────

/// A single conversation session holding an ordered history of turns.
///
/// The session is keyed by a `session_id` string supplied by the caller (e.g.
/// a UUID or a stable user-facing conversation ID).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueSession {
    /// Caller-assigned stable identifier for this session.
    pub session_id: String,
    /// Chronological list of dialogue turns.
    turns: Vec<DialogueTurn>,
    /// Optional single-sentence summary of turns that were truncated.
    summary: Option<String>,
}

impl DialogueSession {
    /// Create a new, empty `DialogueSession` with the given ID.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turns: Vec::new(),
            summary: None,
        }
    }

    /// Append a new turn to the session.
    ///
    /// Returns `Err` if either `role` or `content` would be empty after
    /// trimming, to guard against accidental blank entries.
    pub fn append_turn(
        &mut self,
        role: Role,
        content: impl Into<String>,
    ) -> Result<(), AgentRuntimeError> {
        let content = content.into();
        if content.trim().is_empty() {
            return Err(AgentRuntimeError::Memory(
                "dialogue turn content must not be blank".into(),
            ));
        }
        self.turns.push(DialogueTurn::new(role, content));
        Ok(())
    }

    /// Return a slice of all turns in chronological order.
    pub fn get_history(&self) -> &[DialogueTurn] {
        &self.turns
    }

    /// Return the number of turns currently stored.
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Return the optional summary string, if one has been produced by
    /// [`summarize_if_long`].
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Truncate the history to the most recent `keep_last` turns, setting
    /// `summary` to a brief sentence describing how many turns were dropped.
    ///
    /// If the total number of turns is already ≤ `keep_last`, this is a no-op
    /// and `Ok(false)` is returned.  When truncation occurs `Ok(true)` is
    /// returned.
    ///
    /// # Errors
    /// Returns `Err` if `keep_last` is zero, since a zero-length history
    /// would discard all context.
    pub fn summarize_if_long(&mut self, keep_last: usize) -> Result<bool, AgentRuntimeError> {
        if keep_last == 0 {
            return Err(AgentRuntimeError::Memory(
                "keep_last must be greater than zero".into(),
            ));
        }
        if self.turns.len() <= keep_last {
            return Ok(false);
        }
        let dropped = self.turns.len() - keep_last;
        self.summary = Some(format!(
            "[{dropped} earlier turn(s) omitted for brevity]"
        ));
        self.turns = self.turns.split_off(dropped);
        Ok(true)
    }

    /// Assemble the full session into a single prompt-ready context string.
    ///
    /// The optional `episodic_context` prefix (recalled memories) is prepended
    /// before the dialogue history.  If a summary is present it is inserted
    /// between the episodic context and the turn history.
    pub fn build_context(&self, episodic_context: Option<&str>) -> String {
        let mut parts: Vec<String> = Vec::new();

        if let Some(ctx) = episodic_context {
            if !ctx.trim().is_empty() {
                parts.push(format!("[Memory context]\n{ctx}"));
            }
        }

        if let Some(summary) = &self.summary {
            parts.push(summary.clone());
        }

        for turn in &self.turns {
            parts.push(format!("{}: {}", turn.role, turn.content));
        }

        parts.join("\n")
    }
}

// ── DialogueStore ─────────────────────────────────────────────────────────────

/// Concurrent map of session ID → [`DialogueSession`].
///
/// `DialogueStore` is cheap to clone — the inner map is wrapped in an
/// `Arc<Mutex<_>>` so all clones share the same underlying storage.
#[derive(Debug, Clone, Default)]
pub struct DialogueStore {
    sessions: Arc<Mutex<HashMap<String, DialogueSession>>>,
}

impl DialogueStore {
    /// Create an empty `DialogueStore`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or completely replace the session for `session_id`.
    pub fn upsert(&self, session: DialogueSession) -> Result<(), AgentRuntimeError> {
        let mut map = recover_lock(self.sessions.lock(), "DialogueStore::upsert");
        map.insert(session.session_id.clone(), session);
        Ok(())
    }

    /// Retrieve a clone of the session for `session_id`, or `None` if absent.
    pub fn get(&self, session_id: &str) -> Result<Option<DialogueSession>, AgentRuntimeError> {
        let map = recover_lock(self.sessions.lock(), "DialogueStore::get");
        Ok(map.get(session_id).cloned())
    }

    /// Append a turn to an existing session, creating the session if it does
    /// not yet exist.
    pub fn append_turn(
        &self,
        session_id: impl Into<String>,
        role: Role,
        content: impl Into<String>,
    ) -> Result<(), AgentRuntimeError> {
        let session_id = session_id.into();
        let mut map = recover_lock(self.sessions.lock(), "DialogueStore::append_turn");
        let session = map
            .entry(session_id.clone())
            .or_insert_with(|| DialogueSession::new(session_id));
        session.append_turn(role, content)
    }

    /// Remove and return the session for `session_id`.
    pub fn remove(&self, session_id: &str) -> Result<Option<DialogueSession>, AgentRuntimeError> {
        let mut map = recover_lock(self.sessions.lock(), "DialogueStore::remove");
        Ok(map.remove(session_id))
    }

    /// Return the number of sessions currently stored.
    pub fn session_count(&self) -> usize {
        let map = recover_lock(self.sessions.lock(), "DialogueStore::session_count");
        map.len()
    }

    /// Return a sorted list of all session IDs.
    pub fn session_ids(&self) -> Vec<String> {
        let map = recover_lock(self.sessions.lock(), "DialogueStore::session_ids");
        let mut ids: Vec<String> = map.keys().cloned().collect();
        ids.sort();
        ids
    }
}

// ── DialogueContext ───────────────────────────────────────────────────────────

/// A fully assembled prompt context combining episodic memory with the live
/// dialogue history for a session.
///
/// Construct via [`DialogueContext::build`] and pass the resulting
/// [`DialogueContext::prompt`] string to an inference closure or LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueContext {
    /// The session ID this context was assembled for.
    pub session_id: String,
    /// Final assembled prompt string ready to be sent to an LLM.
    pub prompt: String,
    /// Number of dialogue turns included (after any summarization).
    pub turn_count: usize,
    /// Whether a summary of older turns is present in the prompt.
    pub has_summary: bool,
}

impl DialogueContext {
    /// Assemble a `DialogueContext` from a `DialogueSession` and an optional
    /// episodic memory snippet.
    ///
    /// The `episodic_context` string is typically produced by recalling items
    /// from an [`crate::memory::EpisodicStore`] and formatting them as text.
    pub fn build(session: &DialogueSession, episodic_context: Option<&str>) -> Self {
        Self {
            session_id: session.session_id.clone(),
            prompt: session.build_context(episodic_context),
            turn_count: session.turn_count(),
            has_summary: session.summary().is_some(),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Role ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_role_display_user() {
        assert_eq!(Role::User.to_string(), "user");
    }

    #[test]
    fn test_role_display_assistant() {
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn test_role_display_system() {
        assert_eq!(Role::System.to_string(), "system");
    }

    #[test]
    fn test_role_serialize_roundtrip() {
        let json = serde_json::to_string(&Role::Assistant).unwrap();
        let back: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Role::Assistant);
    }

    // ── DialogueTurn ─────────────────────────────────────────────────────────

    #[test]
    fn test_dialogue_turn_new() {
        let t = DialogueTurn::new(Role::User, "hello");
        assert_eq!(t.role, Role::User);
        assert_eq!(t.content, "hello");
    }

    #[test]
    fn test_dialogue_turn_serialize_roundtrip() {
        let t = DialogueTurn::new(Role::Assistant, "hi there");
        let json = serde_json::to_string(&t).unwrap();
        let back: DialogueTurn = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "hi there");
    }

    // ── DialogueSession ───────────────────────────────────────────────────────

    #[test]
    fn test_session_new_is_empty() {
        let s = DialogueSession::new("sess-1");
        assert_eq!(s.session_id, "sess-1");
        assert_eq!(s.turn_count(), 0);
        assert!(s.summary().is_none());
    }

    #[test]
    fn test_append_turn_increments_count() {
        let mut s = DialogueSession::new("s");
        s.append_turn(Role::User, "hello").unwrap();
        assert_eq!(s.turn_count(), 1);
    }

    #[test]
    fn test_append_turn_rejects_blank_content() {
        let mut s = DialogueSession::new("s");
        let err = s.append_turn(Role::User, "   ").unwrap_err();
        assert!(matches!(err, AgentRuntimeError::Memory(_)));
    }

    #[test]
    fn test_get_history_returns_turns_in_order() {
        let mut s = DialogueSession::new("s");
        s.append_turn(Role::User, "first").unwrap();
        s.append_turn(Role::Assistant, "second").unwrap();
        let h = s.get_history();
        assert_eq!(h[0].content, "first");
        assert_eq!(h[1].content, "second");
    }

    #[test]
    fn test_summarize_if_long_truncates_and_sets_summary() {
        let mut s = DialogueSession::new("s");
        for i in 0..5 {
            s.append_turn(Role::User, format!("msg {i}")).unwrap();
        }
        let truncated = s.summarize_if_long(2).unwrap();
        assert!(truncated);
        assert_eq!(s.turn_count(), 2);
        assert!(s.summary().is_some());
        assert!(s.summary().unwrap().contains("3"));
    }

    #[test]
    fn test_summarize_if_long_noop_when_under_limit() {
        let mut s = DialogueSession::new("s");
        s.append_turn(Role::User, "only turn").unwrap();
        let truncated = s.summarize_if_long(10).unwrap();
        assert!(!truncated);
        assert!(s.summary().is_none());
    }

    #[test]
    fn test_summarize_if_long_rejects_zero_keep() {
        let mut s = DialogueSession::new("s");
        s.append_turn(Role::User, "msg").unwrap();
        assert!(s.summarize_if_long(0).is_err());
    }

    #[test]
    fn test_summarize_retains_last_n_turns() {
        let mut s = DialogueSession::new("s");
        for i in 0..6 {
            s.append_turn(Role::User, format!("turn {i}")).unwrap();
        }
        s.summarize_if_long(3).unwrap();
        let h = s.get_history();
        assert_eq!(h[0].content, "turn 3");
        assert_eq!(h[2].content, "turn 5");
    }

    #[test]
    fn test_build_context_includes_turns() {
        let mut s = DialogueSession::new("s");
        s.append_turn(Role::User, "hi").unwrap();
        let ctx = s.build_context(None);
        assert!(ctx.contains("user: hi"));
    }

    #[test]
    fn test_build_context_includes_episodic_prefix() {
        let mut s = DialogueSession::new("s");
        s.append_turn(Role::User, "hi").unwrap();
        let ctx = s.build_context(Some("Rust is fast."));
        assert!(ctx.contains("Memory context"));
        assert!(ctx.contains("Rust is fast."));
    }

    #[test]
    fn test_build_context_includes_summary() {
        let mut s = DialogueSession::new("s");
        for i in 0..5 {
            s.append_turn(Role::User, format!("m{i}")).unwrap();
        }
        s.summarize_if_long(2).unwrap();
        let ctx = s.build_context(None);
        assert!(ctx.contains("omitted"));
    }

    #[test]
    fn test_session_serialize_roundtrip() {
        let mut s = DialogueSession::new("rt");
        s.append_turn(Role::User, "hello").unwrap();
        let json = serde_json::to_string(&s).unwrap();
        let back: DialogueSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "rt");
        assert_eq!(back.turn_count(), 1);
    }

    // ── DialogueStore ─────────────────────────────────────────────────────────

    #[test]
    fn test_store_starts_empty() {
        let store = DialogueStore::new();
        assert_eq!(store.session_count(), 0);
    }

    #[test]
    fn test_store_upsert_and_get() {
        let store = DialogueStore::new();
        let mut s = DialogueSession::new("abc");
        s.append_turn(Role::User, "hello").unwrap();
        store.upsert(s).unwrap();
        let fetched = store.get("abc").unwrap().unwrap();
        assert_eq!(fetched.turn_count(), 1);
    }

    #[test]
    fn test_store_get_missing_returns_none() {
        let store = DialogueStore::new();
        assert!(store.get("missing").unwrap().is_none());
    }

    #[test]
    fn test_store_append_turn_creates_session() {
        let store = DialogueStore::new();
        store
            .append_turn("new-sess", Role::User, "first message")
            .unwrap();
        let s = store.get("new-sess").unwrap().unwrap();
        assert_eq!(s.turn_count(), 1);
    }

    #[test]
    fn test_store_append_turn_to_existing_session() {
        let store = DialogueStore::new();
        store.append_turn("s", Role::User, "a").unwrap();
        store.append_turn("s", Role::Assistant, "b").unwrap();
        let s = store.get("s").unwrap().unwrap();
        assert_eq!(s.turn_count(), 2);
    }

    #[test]
    fn test_store_remove_existing() {
        let store = DialogueStore::new();
        store.append_turn("x", Role::User, "hi").unwrap();
        let removed = store.remove("x").unwrap();
        assert!(removed.is_some());
        assert_eq!(store.session_count(), 0);
    }

    #[test]
    fn test_store_remove_missing_returns_none() {
        let store = DialogueStore::new();
        assert!(store.remove("ghost").unwrap().is_none());
    }

    #[test]
    fn test_store_session_ids_sorted() {
        let store = DialogueStore::new();
        store.append_turn("z-sess", Role::User, "z").unwrap();
        store.append_turn("a-sess", Role::User, "a").unwrap();
        store.append_turn("m-sess", Role::User, "m").unwrap();
        let ids = store.session_ids();
        assert_eq!(ids, vec!["a-sess", "m-sess", "z-sess"]);
    }

    #[test]
    fn test_store_clone_shares_state() {
        let store = DialogueStore::new();
        let clone = store.clone();
        store.append_turn("shared", Role::User, "hi").unwrap();
        assert_eq!(clone.session_count(), 1);
    }

    // ── DialogueContext ───────────────────────────────────────────────────────

    #[test]
    fn test_dialogue_context_build_no_episodic() {
        let mut s = DialogueSession::new("ctx-test");
        s.append_turn(Role::User, "what is 2+2?").unwrap();
        let ctx = DialogueContext::build(&s, None);
        assert_eq!(ctx.session_id, "ctx-test");
        assert_eq!(ctx.turn_count, 1);
        assert!(!ctx.has_summary);
        assert!(ctx.prompt.contains("user: what is 2+2?"));
    }

    #[test]
    fn test_dialogue_context_build_with_episodic() {
        let mut s = DialogueSession::new("ctx-ep");
        s.append_turn(Role::User, "hello").unwrap();
        let ctx = DialogueContext::build(&s, Some("fact: Rust is fast"));
        assert!(ctx.prompt.contains("Memory context"));
        assert!(ctx.prompt.contains("Rust is fast"));
    }

    #[test]
    fn test_dialogue_context_has_summary_flag() {
        let mut s = DialogueSession::new("summ");
        for i in 0..4 {
            s.append_turn(Role::User, format!("m{i}")).unwrap();
        }
        s.summarize_if_long(2).unwrap();
        let ctx = DialogueContext::build(&s, None);
        assert!(ctx.has_summary);
    }

    #[test]
    fn test_dialogue_context_serialize_roundtrip() {
        let s = DialogueSession::new("ser");
        let ctx = DialogueContext::build(&s, None);
        let json = serde_json::to_string(&ctx).unwrap();
        let back: DialogueContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "ser");
    }
}
