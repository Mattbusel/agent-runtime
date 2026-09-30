//! # Persistent Goal Tracking
//!
//! Goals are high-level objectives that the agent works toward across
//! multiple sessions.  This module provides:
//!
//! - [`Goal`] — a single objective with description, priority, deadline, and
//!   status.
//! - [`GoalTree`] — a forest of goals with parent/child dependency
//!   relationships.  A goal cannot be marked [`GoalStatus::Complete`] until
//!   all of its sub-goals are complete.
//! - [`GoalStore`] — a persistent goal registry that saves goals to disk as
//!   JSON and automatically resumes incomplete goals on startup.
//!
//! ## Goal lifecycle
//!
//! ```text
//! Pending → InProgress → Complete
//!                     ↘ Failed
//!                     ↘ Abandoned
//! ```
//!
//! ## Persistence
//!
//! [`GoalStore::save`] serialises the entire goal tree to a JSON file.
//! [`GoalStore::load`] reads it back.  Both operations are *synchronous* so
//! they can be called during startup/shutdown without requiring an async
//! executor.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::goals::{Goal, GoalStatus, GoalStore, GoalPriority};
//!
//! let mut store = GoalStore::new();
//!
//! let root = store.add_goal(Goal::new(
//!     "Publish blog post",
//!     GoalPriority::High,
//!     None,
//! ));
//!
//! let sub = store.add_goal(Goal::new(
//!     "Write draft",
//!     GoalPriority::Medium,
//!     None,
//! ));
//!
//! store.add_dependency(root.clone(), sub.clone());
//!
//! assert_eq!(store.incomplete_goals().len(), 2);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// GoalId
// ---------------------------------------------------------------------------

/// Unique identifier for a [`Goal`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GoalId(pub String);

impl GoalId {
    /// Generate a new random [`GoalId`].
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl Default for GoalId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for GoalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// GoalPriority
// ---------------------------------------------------------------------------

/// Priority level for a [`Goal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GoalPriority {
    /// Drop first under load.
    Low = 0,
    /// Normal priority.
    Medium = 1,
    /// Deprioritise other goals in favour of this one.
    High = 2,
    /// Mission-critical; never dropped automatically.
    Critical = 3,
}

// ---------------------------------------------------------------------------
// GoalStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a [`Goal`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    /// Not yet started.
    Pending,
    /// Actively being pursued.
    InProgress,
    /// Success criteria met.
    Complete,
    /// Could not be achieved.
    Failed,
    /// Deliberately dropped.
    Abandoned,
}

impl GoalStatus {
    /// Return `true` if the goal is no longer being pursued (terminal state).
    pub fn is_terminal(&self) -> bool {
        matches!(self, GoalStatus::Complete | GoalStatus::Failed | GoalStatus::Abandoned)
    }

    /// Return `true` if the goal is still open (non-terminal).
    pub fn is_open(&self) -> bool {
        !self.is_terminal()
    }
}

// ---------------------------------------------------------------------------
// Goal
// ---------------------------------------------------------------------------

/// A single agent goal with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique identifier.
    pub id: GoalId,
    /// Human-readable description of what must be achieved.
    pub description: String,
    /// Criteria that, when met, mean the goal is complete.
    pub success_criteria: Option<String>,
    /// Priority relative to other goals.
    pub priority: GoalPriority,
    /// Optional deadline by which the goal must be achieved.
    pub deadline: Option<DateTime<Utc>>,
    /// Current lifecycle state.
    pub status: GoalStatus,
    /// When this goal was created.
    pub created_at: DateTime<Utc>,
    /// When the status was last changed.
    pub updated_at: DateTime<Utc>,
    /// Free-form progress notes accumulated by the agent.
    pub progress_notes: Vec<String>,
}

impl Goal {
    /// Create a new [`Pending`](GoalStatus::Pending) goal.
    pub fn new(
        description: impl Into<String>,
        priority: GoalPriority,
        deadline: Option<DateTime<Utc>>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: GoalId::new(),
            description: description.into(),
            success_criteria: None,
            priority,
            deadline,
            status: GoalStatus::Pending,
            created_at: now,
            updated_at: now,
            progress_notes: Vec::new(),
        }
    }

    /// Attach explicit success criteria.
    pub fn with_criteria(mut self, criteria: impl Into<String>) -> Self {
        self.success_criteria = Some(criteria.into());
        self
    }

    /// Transition the goal to [`InProgress`](GoalStatus::InProgress).
    pub fn start(&mut self) {
        self.status = GoalStatus::InProgress;
        self.updated_at = Utc::now();
    }

    /// Mark the goal as [`Complete`](GoalStatus::Complete).
    pub fn complete(&mut self) {
        self.status = GoalStatus::Complete;
        self.updated_at = Utc::now();
    }

    /// Mark the goal as [`Failed`](GoalStatus::Failed).
    pub fn fail(&mut self) {
        self.status = GoalStatus::Failed;
        self.updated_at = Utc::now();
    }

    /// Mark the goal as [`Abandoned`](GoalStatus::Abandoned).
    pub fn abandon(&mut self) {
        self.status = GoalStatus::Abandoned;
        self.updated_at = Utc::now();
    }

    /// Append a progress note.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.progress_notes.push(note.into());
        self.updated_at = Utc::now();
    }

    /// Return `true` if the goal has passed its deadline (if one was set).
    pub fn is_overdue(&self) -> bool {
        self.deadline.map(|d| Utc::now() > d).unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// GoalTree — goal store with dependency tracking
// ---------------------------------------------------------------------------

/// A flat registry of [`Goal`]s with parent/child dependency relationships.
///
/// A goal with sub-goals (children) cannot transition to
/// [`GoalStatus::Complete`] until all children are complete.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoalStore {
    goals: HashMap<GoalId, Goal>,
    /// children[parent] = list of child goal IDs.
    children: HashMap<GoalId, Vec<GoalId>>,
    /// parents[child] = parent goal ID.
    parents: HashMap<GoalId, GoalId>,
}

impl GoalStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Insert a goal and return its ID.
    pub fn add_goal(&mut self, goal: Goal) -> GoalId {
        let id = goal.id.clone();
        self.goals.insert(id.clone(), goal);
        id
    }

    /// Declare that `child` is a sub-goal of `parent`.
    ///
    /// The parent will not complete until all children are complete.
    pub fn add_dependency(&mut self, parent: GoalId, child: GoalId) {
        self.children.entry(parent.clone()).or_default().push(child.clone());
        self.parents.insert(child, parent);
    }

    /// Attempt to start a goal (transition `Pending → InProgress`).
    ///
    /// Returns `true` on success, `false` if the goal is not in `Pending` state.
    pub fn start_goal(&mut self, id: &GoalId) -> bool {
        if let Some(g) = self.goals.get_mut(id) {
            if g.status == GoalStatus::Pending {
                g.start();
                return true;
            }
        }
        false
    }

    /// Attempt to complete a goal.
    ///
    /// Returns `Err` if any child goals are still incomplete.  Returns `Ok(false)`
    /// if the goal does not exist.
    pub fn complete_goal(&mut self, id: &GoalId) -> Result<bool, String> {
        // Check children first.
        if let Some(children) = self.children.get(id).cloned() {
            for child_id in &children {
                if let Some(child) = self.goals.get(child_id) {
                    if !child.status.is_terminal() {
                        return Err(format!(
                            "cannot complete '{}': sub-goal '{}' is still {}",
                            id, child_id, child.description
                        ));
                    }
                }
            }
        }
        if let Some(g) = self.goals.get_mut(id) {
            g.complete();
            return Ok(true);
        }
        Ok(false)
    }

    /// Mark a goal as failed.
    pub fn fail_goal(&mut self, id: &GoalId) -> bool {
        if let Some(g) = self.goals.get_mut(id) {
            g.fail();
            return true;
        }
        false
    }

    /// Mark a goal as abandoned.
    pub fn abandon_goal(&mut self, id: &GoalId) -> bool {
        if let Some(g) = self.goals.get_mut(id) {
            g.abandon();
            return true;
        }
        false
    }

    /// Append a progress note to a goal.
    pub fn add_note(&mut self, id: &GoalId, note: impl Into<String>) -> bool {
        if let Some(g) = self.goals.get_mut(id) {
            g.add_note(note);
            return true;
        }
        false
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Get a goal by ID.
    pub fn get(&self, id: &GoalId) -> Option<&Goal> {
        self.goals.get(id)
    }

    /// All goals, regardless of status.
    pub fn all_goals(&self) -> impl Iterator<Item = &Goal> {
        self.goals.values()
    }

    /// Goals that are still open (non-terminal), sorted by priority descending.
    pub fn incomplete_goals(&self) -> Vec<&Goal> {
        let mut open: Vec<&Goal> =
            self.goals.values().filter(|g| g.status.is_open()).collect();
        open.sort_by(|a, b| b.priority.cmp(&a.priority));
        open
    }

    /// Goals that are currently in progress.
    pub fn in_progress(&self) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| g.status == GoalStatus::InProgress)
            .collect()
    }

    /// Root goals (goals with no parent).
    pub fn roots(&self) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| !self.parents.contains_key(&g.id))
            .collect()
    }

    /// Children of a goal.
    pub fn children_of(&self, id: &GoalId) -> Vec<&Goal> {
        self.children
            .get(id)
            .map(|ids| ids.iter().filter_map(|i| self.goals.get(i)).collect())
            .unwrap_or_default()
    }

    /// Parent of a goal, if it has one.
    pub fn parent_of(&self, id: &GoalId) -> Option<&Goal> {
        self.parents.get(id).and_then(|pid| self.goals.get(pid))
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Serialize the entire goal store to a JSON file at `path`.
    pub fn save(&self, path: &std::path::Path) -> Result<(), crate::error::AgentRuntimeError> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| crate::error::AgentRuntimeError::Persistence(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| crate::error::AgentRuntimeError::Persistence(e.to_string()))
    }

    /// Load a goal store from a JSON file at `path`.
    ///
    /// Returns an empty store if the file does not exist.
    pub fn load(path: &std::path::Path) -> Result<Self, crate::error::AgentRuntimeError> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let bytes = std::fs::read(path)
            .map_err(|e| crate::error::AgentRuntimeError::Persistence(e.to_string()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| crate::error::AgentRuntimeError::Persistence(e.to_string()))
    }

    /// Return all incomplete goals — used on startup to resume work.
    pub fn resumable_goals(&self) -> Vec<&Goal> {
        self.incomplete_goals()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_retrieve_goal() {
        let mut store = GoalStore::new();
        let id = store.add_goal(Goal::new("do X", GoalPriority::Medium, None));
        assert!(store.get(&id).is_some());
    }

    #[test]
    fn start_goal_transitions_to_in_progress() {
        let mut store = GoalStore::new();
        let id = store.add_goal(Goal::new("task", GoalPriority::Low, None));
        assert!(store.start_goal(&id));
        assert_eq!(store.get(&id).unwrap().status, GoalStatus::InProgress);
    }

    #[test]
    fn complete_goal_fails_when_child_is_open() {
        let mut store = GoalStore::new();
        let parent = store.add_goal(Goal::new("parent", GoalPriority::High, None));
        let child = store.add_goal(Goal::new("child", GoalPriority::Medium, None));
        store.add_dependency(parent.clone(), child);
        store.start_goal(&parent);
        let result = store.complete_goal(&parent);
        assert!(result.is_err());
    }

    #[test]
    fn complete_goal_succeeds_when_all_children_done() {
        let mut store = GoalStore::new();
        let parent = store.add_goal(Goal::new("parent", GoalPriority::High, None));
        let child = store.add_goal(Goal::new("child", GoalPriority::Medium, None));
        store.add_dependency(parent.clone(), child.clone());
        store.start_goal(&child);
        store.complete_goal(&child).unwrap();
        store.start_goal(&parent);
        assert!(store.complete_goal(&parent).is_ok());
        assert_eq!(store.get(&parent).unwrap().status, GoalStatus::Complete);
    }

    #[test]
    fn incomplete_goals_sorted_by_priority() {
        let mut store = GoalStore::new();
        store.add_goal(Goal::new("low", GoalPriority::Low, None));
        store.add_goal(Goal::new("high", GoalPriority::High, None));
        let incomplete = store.incomplete_goals();
        assert_eq!(incomplete[0].priority, GoalPriority::High);
    }

    #[test]
    fn children_of_returns_sub_goals() {
        let mut store = GoalStore::new();
        let parent = store.add_goal(Goal::new("parent", GoalPriority::High, None));
        let child = store.add_goal(Goal::new("child", GoalPriority::Low, None));
        store.add_dependency(parent.clone(), child);
        assert_eq!(store.children_of(&parent).len(), 1);
    }

    #[test]
    fn goal_is_overdue_when_past_deadline() {
        let past = Utc::now() - chrono::Duration::hours(1);
        let g = Goal::new("late", GoalPriority::Medium, Some(past));
        assert!(g.is_overdue());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut store = GoalStore::new();
        store.add_goal(Goal::new("persistent goal", GoalPriority::Critical, None));

        let dir = std::env::temp_dir();
        let path = dir.join(format!("goals_test_{}.json", uuid::Uuid::new_v4()));
        store.save(&path).unwrap();
        let loaded = GoalStore::load(&path).unwrap();
        assert_eq!(loaded.incomplete_goals().len(), 1);

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_returns_empty_store_when_file_missing() {
        let path = std::path::Path::new("/tmp/nonexistent_goals_xyz.json");
        let store = GoalStore::load(path).unwrap();
        assert!(store.incomplete_goals().is_empty());
    }
}
