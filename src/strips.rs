//! # STRIPS-style Planner
//!
//! A goal-directed planner that uses *backward chaining* to find a sequence
//! of actions that transforms the current world state into a goal state.
//!
//! ## What is STRIPS?
//!
//! STRIPS (Stanford Research Institute Problem Solver) represents planning
//! problems as:
//!
//! - **State**: a set of ground facts that are currently true.
//! - **Goal**: a set of facts that must be true in the final state.
//! - **Actions**: each action has:
//!   - **Preconditions** — facts that must be true *before* the action runs.
//!   - **Add-effects** — facts that become true *after* the action runs.
//!   - **Delete-effects** — facts that become false *after* the action runs.
//!
//! ## Backward chaining
//!
//! The planner works *backward* from the goal:
//!
//! 1. Start with the set of unsatisfied goal facts.
//! 2. Find an action whose add-effects satisfy at least one open goal.
//! 3. Add that action's preconditions as new sub-goals.
//! 4. Recurse until all sub-goals are satisfied by the initial state.
//!
//! Circular plan detection prevents infinite loops by tracking which goal sets
//! have already been visited.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::strips::{StripsPlanner, StripsAction, StripsFact};
//! use llm_agent_runtime::world_model::{WorldState, FactValue};
//!
//! let mut planner = StripsPlanner::new(16);
//!
//! // Action: unlock_door — requires key_held; makes door_locked=false.
//! planner.register_action(StripsAction {
//!     name: "unlock_door".into(),
//!     preconditions: vec![StripsFact::new("key_held", FactValue::Bool(true))],
//!     add_effects:   vec![StripsFact::new("door_locked", FactValue::Bool(false))],
//!     delete_effects: vec![StripsFact::new("door_locked", FactValue::Bool(true))],
//!     cost: 1,
//! });
//!
//! let mut initial = WorldState::new();
//! initial.assert_fact("key_held",   FactValue::Bool(true));
//! initial.assert_fact("door_locked", FactValue::Bool(true));
//!
//! let mut goal = WorldState::new();
//! goal.assert_fact("door_locked", FactValue::Bool(false));
//!
//! let plan = planner.plan(&initial, &goal).unwrap();
//! assert_eq!(plan.len(), 1);
//! assert_eq!(plan[0].name, "unlock_door");
//! ```

use crate::world_model::{FactValue, WorldState};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// StripsFact
// ---------------------------------------------------------------------------

/// A single ground fact used in action preconditions or effects.
#[derive(Debug, Clone, PartialEq)]
pub struct StripsFact {
    /// The fact key (must match keys in [`WorldState`]).
    pub key: String,
    /// The required / asserted value.
    pub value: FactValue,
}

impl StripsFact {
    /// Create a new fact assertion.
    pub fn new(key: impl Into<String>, value: FactValue) -> Self {
        Self { key: key.into(), value }
    }

    /// Return `true` if this fact is satisfied by `state`.
    pub fn is_satisfied_by(&self, state: &WorldState) -> bool {
        state.holds(&self.key, &self.value)
    }
}

// ---------------------------------------------------------------------------
// StripsAction
// ---------------------------------------------------------------------------

/// A STRIPS action with preconditions and add/delete effects.
#[derive(Debug, Clone)]
pub struct StripsAction {
    /// Unique name for this action (used in the plan output).
    pub name: String,
    /// Facts that must hold in the current state before this action can run.
    pub preconditions: Vec<StripsFact>,
    /// Facts that become true after this action executes.
    pub add_effects: Vec<StripsFact>,
    /// Facts that become false after this action executes (removed from state).
    pub delete_effects: Vec<StripsFact>,
    /// Relative cost of this action (used for tie-breaking; lower is preferred).
    pub cost: u32,
}

impl StripsAction {
    /// Return `true` if all preconditions of this action are satisfied in `state`.
    pub fn applicable(&self, state: &WorldState) -> bool {
        self.preconditions.iter().all(|p| p.is_satisfied_by(state))
    }

    /// Apply this action to `state`, returning a new `WorldState` that reflects
    /// the add and delete effects.
    pub fn apply(&self, state: &WorldState) -> WorldState {
        let mut next = state.clone();
        for del in &self.delete_effects {
            next.retract(&del.key);
        }
        for add in &self.add_effects {
            next.assert_fact(add.key.clone(), add.value.clone());
        }
        next
    }
}

// ---------------------------------------------------------------------------
// StripsPlannerError
// ---------------------------------------------------------------------------

/// Errors returned by the STRIPS planner.
#[derive(Debug, Clone, PartialEq)]
pub enum StripsPlannerError {
    /// No plan could be found within the depth / step limits.
    NoSolution,
    /// A circular plan was detected (the same goal set was reached twice).
    CircularPlan,
    /// The search depth exceeded the configured maximum.
    DepthExceeded(usize),
}

impl std::fmt::Display for StripsPlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StripsPlannerError::NoSolution => write!(f, "no plan found"),
            StripsPlannerError::CircularPlan => write!(f, "circular plan detected"),
            StripsPlannerError::DepthExceeded(d) => {
                write!(f, "search depth {d} exceeded")
            }
        }
    }
}

impl std::error::Error for StripsPlannerError {}

// ---------------------------------------------------------------------------
// StripsPlanner
// ---------------------------------------------------------------------------

/// STRIPS-style backward-chaining planner.
pub struct StripsPlanner {
    actions: Vec<StripsAction>,
    max_depth: usize,
}

impl StripsPlanner {
    /// Create a new planner with the given maximum search depth.
    pub fn new(max_depth: usize) -> Self {
        Self { actions: Vec::new(), max_depth }
    }

    /// Register an action with the planner.
    pub fn register_action(&mut self, action: StripsAction) {
        self.actions.push(action);
    }

    /// Find a plan (ordered sequence of actions) that transforms `initial` into
    /// a state satisfying `goal`.
    ///
    /// Iterative-deepening depth-first search over applicable actions, so the
    /// plan returned has the fewest actions (ties go to actions that achieve
    /// an open goal fact, then to lower cost). States already reached by a
    /// shorter path are not expanded again.
    ///
    /// # Errors
    ///
    /// - [`StripsPlannerError::NoSolution`] — no sequence of at most
    ///   `max_depth` registered actions achieves the goal.
    ///
    /// (`CircularPlan` and `DepthExceeded` are no longer returned; repeated
    /// states are pruned instead.)
    pub fn plan(
        &self,
        initial: &WorldState,
        goal: &WorldState,
    ) -> Result<Vec<StripsAction>, StripsPlannerError> {
        if initial.satisfies(goal) {
            return Ok(Vec::new());
        }
        for depth_limit in 1..=self.max_depth {
            // Best depth at which each state was reached during this pass.
            let mut seen: HashMap<String, usize> = HashMap::new();
            let mut path = Vec::new();
            if self.search(initial, goal, &mut path, depth_limit, &mut seen) {
                return Ok(path);
            }
        }
        Err(StripsPlannerError::NoSolution)
    }

    fn search(
        &self,
        state: &WorldState,
        goal: &WorldState,
        path: &mut Vec<StripsAction>,
        remaining: usize,
        seen: &mut HashMap<String, usize>,
    ) -> bool {
        if state.satisfies(goal) {
            return true;
        }
        if remaining == 0 {
            return false;
        }
        let key = Self::state_key(state);
        if seen.get(&key).is_some_and(|&depth| depth <= path.len()) {
            return false;
        }
        seen.insert(key, path.len());

        let open_goals = state.changes_needed(goal);
        let mut candidates: Vec<&StripsAction> =
            self.actions.iter().filter(|a| a.applicable(state)).collect();
        candidates.sort_by_key(|a| {
            let achieves_goal = a.add_effects.iter().any(|eff| {
                open_goals.iter().any(|og| og.key == eff.key && og.required == eff.value)
            });
            (!achieves_goal, a.cost)
        });

        for action in candidates {
            let next = action.apply(state);
            if Self::state_key(&next) == Self::state_key(state) {
                continue; // no-op in this state
            }
            path.push(action.clone());
            if self.search(&next, goal, path, remaining - 1, seen) {
                return true;
            }
            path.pop();
        }
        false
    }

    fn state_key(state: &WorldState) -> String {
        let mut facts: Vec<String> = state
            .keys()
            .filter_map(|k| state.get(k).map(|v| format!("{k}={v}")))
            .collect();
        facts.sort();
        facts.join("|")
    }

    // Simulate a sequence of actions on an initial state, returning the
    // resulting `WorldState`.  Actions whose preconditions are not met are
    // skipped (forward-simulation is lenient; validity is checked at goal time).
    fn simulate(&self, initial: &WorldState, actions: &[StripsAction]) -> WorldState {
        let mut state = initial.clone();
        for action in actions {
            if action.applicable(&state) {
                state = action.apply(&state);
            }
        }
        state
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_model::FactValue;

    fn locked_door_planner() -> StripsPlanner {
        let mut p = StripsPlanner::new(8);
        p.register_action(StripsAction {
            name: "pick_up_key".into(),
            preconditions: vec![StripsFact::new("key_available", FactValue::Bool(true))],
            add_effects: vec![StripsFact::new("key_held", FactValue::Bool(true))],
            delete_effects: vec![StripsFact::new("key_available", FactValue::Bool(true))],
            cost: 1,
        });
        p.register_action(StripsAction {
            name: "unlock_door".into(),
            preconditions: vec![StripsFact::new("key_held", FactValue::Bool(true))],
            add_effects: vec![StripsFact::new("door_locked", FactValue::Bool(false))],
            delete_effects: vec![StripsFact::new("door_locked", FactValue::Bool(true))],
            cost: 1,
        });
        p
    }

    #[test]
    fn finds_single_action_plan() {
        let p = locked_door_planner();

        let mut initial = WorldState::new();
        initial.assert_fact("key_held", FactValue::Bool(true));
        initial.assert_fact("door_locked", FactValue::Bool(true));

        let mut goal = WorldState::new();
        goal.assert_fact("door_locked", FactValue::Bool(false));

        let plan = p.plan(&initial, &goal).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].name, "unlock_door");
    }

    #[test]
    fn finds_two_action_plan() {
        let p = locked_door_planner();

        let mut initial = WorldState::new();
        initial.assert_fact("key_available", FactValue::Bool(true));
        initial.assert_fact("door_locked", FactValue::Bool(true));

        let mut goal = WorldState::new();
        goal.assert_fact("door_locked", FactValue::Bool(false));

        let plan = p.plan(&initial, &goal).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].name, "pick_up_key");
        assert_eq!(plan[1].name, "unlock_door");
    }

    #[test]
    fn returns_no_solution_when_impossible() {
        let p = locked_door_planner();

        let initial = WorldState::new(); // no key available

        let mut goal = WorldState::new();
        goal.assert_fact("door_locked", FactValue::Bool(false));

        let err = p.plan(&initial, &goal).unwrap_err();
        assert_eq!(err, StripsPlannerError::NoSolution);
    }

    #[test]
    fn action_applicable_checks_preconditions() {
        let action = StripsAction {
            name: "test".into(),
            preconditions: vec![StripsFact::new("x", FactValue::Bool(true))],
            add_effects: vec![],
            delete_effects: vec![],
            cost: 0,
        };
        let mut state = WorldState::new();
        assert!(!action.applicable(&state));
        state.assert_fact("x", FactValue::Bool(true));
        assert!(action.applicable(&state));
    }

    #[test]
    fn action_apply_modifies_state() {
        let action = StripsAction {
            name: "flip".into(),
            preconditions: vec![],
            add_effects: vec![StripsFact::new("y", FactValue::Bool(true))],
            delete_effects: vec![StripsFact::new("x", FactValue::Bool(false))],
            cost: 0,
        };
        let mut state = WorldState::new();
        state.assert_fact("x", FactValue::Bool(false));
        let next = action.apply(&state);
        assert_eq!(next.get("y"), Some(&FactValue::Bool(true)));
        // x was deleted by delete_effects
        assert!(next.get("x").is_none());
    }

    #[test]
    fn empty_goal_is_immediately_satisfied() {
        let p = locked_door_planner();
        let initial = WorldState::new();
        let goal = WorldState::new();
        let plan = p.plan(&initial, &goal).unwrap();
        assert!(plan.is_empty());
    }
}
