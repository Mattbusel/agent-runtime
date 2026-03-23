//! # World Model
//!
//! A mutable, typed representation of the agent's beliefs about its environment.
//!
//! ## Overview
//!
//! The [`WorldState`] is a key-value store of *facts* — typed values that the
//! agent believes to be true at a given moment.  Facts can be updated by tool
//! results or observations and are tagged with the timestamp at which they were
//! last asserted.
//!
//! ### Temporal beliefs
//!
//! Each fact carries a [`Belief`] that records both the current value and a
//! history of prior values (with timestamps).  This lets the agent ask "what
//! did I believe about X at time T?" and reason about change over time.
//!
//! ### Planning integration
//!
//! [`WorldState::changes_needed`] compares a goal state to the current world
//! state and returns the list of facts that differ — i.e., the changes the
//! agent must achieve to reach the goal.  This drives the STRIPS planner in
//! [`crate::planner`].
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::world_model::{WorldState, FactValue};
//!
//! let mut world = WorldState::new();
//! world.assert_fact("door_locked", FactValue::Bool(true));
//! world.assert_fact("room", FactValue::Text("kitchen".into()));
//!
//! assert_eq!(world.get("door_locked"), Some(&FactValue::Bool(true)));
//!
//! let mut goal = WorldState::new();
//! goal.assert_fact("door_locked", FactValue::Bool(false));
//!
//! let changes = world.changes_needed(&goal);
//! assert_eq!(changes.len(), 1);
//! assert_eq!(changes[0].0, "door_locked");
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FactValue
// ---------------------------------------------------------------------------

/// A typed value that can be stored as a fact in the world model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FactValue {
    /// Boolean fact (e.g. "door_locked = true").
    Bool(bool),
    /// Integer fact (e.g. "items_collected = 5").
    Int(i64),
    /// Floating-point fact (e.g. "confidence = 0.9").
    Float(f64),
    /// Free-form text fact (e.g. "location = kitchen").
    Text(String),
    /// JSON value for structured facts.
    Json(serde_json::Value),
}

impl std::fmt::Display for FactValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactValue::Bool(b) => write!(f, "{b}"),
            FactValue::Int(i) => write!(f, "{i}"),
            FactValue::Float(v) => write!(f, "{v}"),
            FactValue::Text(s) => write!(f, "{s}"),
            FactValue::Json(j) => write!(f, "{j}"),
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalRecord — a single historical snapshot of a fact
// ---------------------------------------------------------------------------

/// A snapshot of a fact's value at a specific point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRecord {
    /// The value that was believed at this time.
    pub value: FactValue,
    /// When this belief was asserted.
    pub asserted_at: DateTime<Utc>,
    /// Optional source (tool name, observation ID, etc.) that produced this belief.
    pub source: Option<String>,
}

// ---------------------------------------------------------------------------
// Belief — current value + history
// ---------------------------------------------------------------------------

/// The agent's belief about a single named fact, including its full history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    /// The most recent (current) value.
    pub current: FactValue,
    /// Timestamp when the current value was asserted.
    pub updated_at: DateTime<Utc>,
    /// History of prior values in chronological order (oldest first).
    pub history: Vec<TemporalRecord>,
}

impl Belief {
    fn new(value: FactValue, source: Option<String>) -> Self {
        Self {
            current: value,
            updated_at: Utc::now(),
            history: Vec::new(),
        }
    }

    fn update(&mut self, value: FactValue, source: Option<String>) {
        // Push the old value into history before replacing it.
        let old = TemporalRecord {
            value: self.current.clone(),
            asserted_at: self.updated_at,
            source,
        };
        self.history.push(old);
        self.current = value;
        self.updated_at = Utc::now();
    }

    /// Return the value that was believed at or just before `at`.
    ///
    /// Returns `None` if the belief did not exist at that time (i.e., `at`
    /// precedes the first assertion).
    pub fn value_at(&self, at: DateTime<Utc>) -> Option<&FactValue> {
        // Walk history forward; return the last value whose timestamp <= at.
        // If no history entry qualifies, check whether the current value was
        // set before `at` (meaning the fact only has one value ever).
        let hist = self.history.iter().rev().find(|r| r.asserted_at <= at);
        match hist {
            Some(record) => Some(&record.value),
            None => {
                if self.updated_at <= at {
                    Some(&self.current)
                } else {
                    // Find the earliest historical record
                    self.history.first().and_then(|r| {
                        if r.asserted_at <= at { Some(&r.value) } else { None }
                    })
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WorldChange — a single required change to reach a goal
// ---------------------------------------------------------------------------

/// Describes a single fact that must change to satisfy a goal state.
#[derive(Debug, Clone)]
pub struct WorldChange {
    /// The fact key that must change.
    pub key: String,
    /// The current believed value (or `None` if the fact is unknown).
    pub current: Option<FactValue>,
    /// The value required by the goal.
    pub required: FactValue,
}

// ---------------------------------------------------------------------------
// WorldState
// ---------------------------------------------------------------------------

/// Mutable representation of the agent's beliefs about its environment.
///
/// Acts as a typed key-value store of [`Belief`]s.  Every mutation is
/// timestamped, and the full history of each fact is retained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    facts: HashMap<String, Belief>,
}

impl WorldState {
    /// Create an empty world state.
    pub fn new() -> Self {
        Self { facts: HashMap::new() }
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Assert (insert or update) a fact, recording the source if provided.
    pub fn assert_fact_with_source(
        &mut self,
        key: impl Into<String>,
        value: FactValue,
        source: Option<String>,
    ) {
        let key = key.into();
        match self.facts.get_mut(&key) {
            Some(belief) => belief.update(value, source),
            None => {
                self.facts.insert(key, Belief::new(value, source));
            }
        }
    }

    /// Assert a fact without an explicit source.
    pub fn assert_fact(&mut self, key: impl Into<String>, value: FactValue) {
        self.assert_fact_with_source(key, value, None);
    }

    /// Assert multiple facts at once from a tool result or observation.
    pub fn apply_observation(&mut self, source: &str, facts: Vec<(String, FactValue)>) {
        for (key, value) in facts {
            self.assert_fact_with_source(key, value, Some(source.to_owned()));
        }
    }

    /// Retract (remove) a fact entirely from the world model.
    pub fn retract(&mut self, key: &str) -> Option<Belief> {
        self.facts.remove(key)
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Get the current value of a fact, or `None` if unknown.
    pub fn get(&self, key: &str) -> Option<&FactValue> {
        self.facts.get(key).map(|b| &b.current)
    }

    /// Get the full [`Belief`] for a fact (current value + history).
    pub fn belief(&self, key: &str) -> Option<&Belief> {
        self.facts.get(key)
    }

    /// Return the believed value of a fact at a specific past time.
    pub fn value_at(&self, key: &str, at: DateTime<Utc>) -> Option<&FactValue> {
        self.facts.get(key)?.value_at(at)
    }

    /// Return all currently known fact keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.facts.keys().map(String::as_str)
    }

    /// Number of known facts.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// True if no facts are known.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Return `true` if the fact exists and its current value equals `value`.
    pub fn holds(&self, key: &str, value: &FactValue) -> bool {
        self.get(key).map(|v| v == value).unwrap_or(false)
    }

    // ── Planning integration ──────────────────────────────────────────────────

    /// Compare this world state to a `goal` state and return the list of
    /// changes needed to satisfy it.
    ///
    /// Each entry in the returned `Vec` describes a fact whose current value
    /// differs from the goal's required value (or which is entirely absent
    /// from the current state).
    pub fn changes_needed(&self, goal: &WorldState) -> Vec<WorldChange> {
        let mut changes = Vec::new();
        for (key, goal_belief) in &goal.facts {
            let required = goal_belief.current.clone();
            let current = self.get(key).cloned();
            if current.as_ref() != Some(&required) {
                changes.push(WorldChange {
                    key: key.clone(),
                    current,
                    required,
                });
            }
        }
        changes
    }

    /// Return `true` if the current state satisfies every fact in `goal`.
    pub fn satisfies(&self, goal: &WorldState) -> bool {
        self.changes_needed(goal).is_empty()
    }

    /// Build a simple `HashMap<String, String>` snapshot compatible with the
    /// HTN planner's [`crate::planner::Precondition`] evaluation.
    pub fn to_string_map(&self) -> HashMap<String, String> {
        self.facts
            .iter()
            .map(|(k, b)| (k.clone(), b.current.to_string()))
            .collect()
    }
}

impl Default for WorldState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_and_get_fact() {
        let mut w = WorldState::new();
        w.assert_fact("x", FactValue::Int(42));
        assert_eq!(w.get("x"), Some(&FactValue::Int(42)));
    }

    #[test]
    fn update_fact_preserves_history() {
        let mut w = WorldState::new();
        w.assert_fact("x", FactValue::Int(1));
        w.assert_fact("x", FactValue::Int(2));
        let belief = w.belief("x").unwrap();
        assert_eq!(belief.current, FactValue::Int(2));
        assert_eq!(belief.history.len(), 1);
        assert_eq!(belief.history[0].value, FactValue::Int(1));
    }

    #[test]
    fn changes_needed_detects_difference() {
        let mut current = WorldState::new();
        current.assert_fact("door", FactValue::Bool(true));

        let mut goal = WorldState::new();
        goal.assert_fact("door", FactValue::Bool(false));

        let changes = current.changes_needed(&goal);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].key, "door");
    }

    #[test]
    fn changes_needed_empty_when_satisfied() {
        let mut current = WorldState::new();
        current.assert_fact("x", FactValue::Int(5));

        let mut goal = WorldState::new();
        goal.assert_fact("x", FactValue::Int(5));

        assert!(current.changes_needed(&goal).is_empty());
        assert!(current.satisfies(&goal));
    }

    #[test]
    fn retract_removes_fact() {
        let mut w = WorldState::new();
        w.assert_fact("y", FactValue::Text("hello".into()));
        w.retract("y");
        assert!(w.get("y").is_none());
    }

    #[test]
    fn holds_returns_true_for_matching_value() {
        let mut w = WorldState::new();
        w.assert_fact("flag", FactValue::Bool(true));
        assert!(w.holds("flag", &FactValue::Bool(true)));
        assert!(!w.holds("flag", &FactValue::Bool(false)));
    }

    #[test]
    fn to_string_map_converts_correctly() {
        let mut w = WorldState::new();
        w.assert_fact("auth", FactValue::Bool(true));
        let map = w.to_string_map();
        assert_eq!(map.get("auth").map(String::as_str), Some("true"));
    }

    #[test]
    fn apply_observation_sets_multiple_facts() {
        let mut w = WorldState::new();
        w.apply_observation(
            "tool_result",
            vec![
                ("a".into(), FactValue::Int(1)),
                ("b".into(), FactValue::Int(2)),
            ],
        );
        assert_eq!(w.get("a"), Some(&FactValue::Int(1)));
        assert_eq!(w.get("b"), Some(&FactValue::Int(2)));
    }

    #[test]
    fn default_creates_empty_state() {
        let w = WorldState::default();
        assert!(w.is_empty());
    }
}
