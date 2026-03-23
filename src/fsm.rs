//! # Finite State Machine for Agent Lifecycle Management
//!
//! Provides a type-safe, history-tracking finite state machine that governs
//! the lifecycle of an agent from [`AgentState::Idle`] through to
//! [`AgentState::Terminating`], with guarded transitions, forced overrides for
//! error recovery, and a rolling history of the last 50 state changes.
//!
//! ## State diagram
//!
//! ```text
//!  Idle ──► Running ──► Waiting ──► Running
//!    │          │           │
//!    │          ▼           ▼
//!    │        Paused ──► Running
//!    │          │
//!    ▼          ▼
//!  Error ◄── Terminating
//! ```
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::fsm::{AgentContext, AgentState, StateMachine};
//!
//! let mut sm = StateMachine::default();
//! let ctx = AgentContext {
//!     task_id: "t-1".into(),
//!     elapsed_ms: 0,
//!     error_count: 0,
//!     last_error: None,
//! };
//!
//! // Register a transition: Idle → Running when elapsed == 0.
//! sm.add_transition(AgentState::Idle, AgentState::Running, |_ctx| true);
//!
//! let next = sm.transition(&ctx);
//! assert_eq!(next, Some(AgentState::Running));
//! assert_eq!(*sm.current(), AgentState::Running);
//! ```

use std::collections::VecDeque;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// State trait
// ---------------------------------------------------------------------------

/// Behaviour hooks called when the FSM enters or exits a state.
///
/// Implementors should be cheap to clone and have no side-effectful fields.
/// The default impls are no-ops so you only override what you need.
pub trait State {
    /// Human-readable name for this state, used in logging and history dumps.
    fn name(&self) -> &str;

    /// Called immediately after the FSM transitions **into** this state.
    fn on_enter(&self) {}

    /// Called immediately before the FSM transitions **out of** this state.
    fn on_exit(&self) {}
}

// ---------------------------------------------------------------------------
// AgentState
// ---------------------------------------------------------------------------

/// Built-in states for the agent lifecycle FSM.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentState {
    /// The agent is initialised but has not yet received a task.
    Idle,
    /// The agent is actively processing a task.
    Running,
    /// The agent is waiting for an external event (e.g. tool response).
    Waiting,
    /// The agent has been administratively paused.
    Paused,
    /// The agent is shutting down gracefully.
    Terminating,
    /// The agent has encountered an unrecoverable error.
    Error(String),
}

impl State for AgentState {
    fn name(&self) -> &str {
        match self {
            AgentState::Idle        => "Idle",
            AgentState::Running     => "Running",
            AgentState::Waiting     => "Waiting",
            AgentState::Paused      => "Paused",
            AgentState::Terminating => "Terminating",
            AgentState::Error(_)    => "Error",
        }
    }

    fn on_enter(&self) {
        tracing::debug!(state = self.name(), "FSM: entered state");
    }

    fn on_exit(&self) {
        tracing::debug!(state = self.name(), "FSM: exiting state");
    }
}

impl std::fmt::Display for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentState::Error(msg) => write!(f, "Error({msg})"),
            other                  => write!(f, "{}", other.name()),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentContext
// ---------------------------------------------------------------------------

/// Runtime context passed to transition condition functions.
///
/// The FSM itself does not mutate this; callers populate it before calling
/// [`StateMachine::transition`].
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// Identifier of the task currently being processed.
    pub task_id: String,
    /// Wall-clock milliseconds elapsed since the agent started this task.
    pub elapsed_ms: u64,
    /// Number of errors encountered during the current task.
    pub error_count: u32,
    /// The most recent error message, if any.
    pub last_error: Option<String>,
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A guarded edge in the FSM graph.
///
/// The FSM evaluates transitions in registration order; the first matching
/// transition wins.
pub struct Transition {
    /// The state this transition departs from.
    pub from: AgentState,
    /// The state this transition arrives at.
    pub to: AgentState,
    /// Predicate evaluated against the current [`AgentContext`].
    /// Returns `true` when the transition should fire.
    pub condition: Box<dyn Fn(&AgentContext) -> bool + Send + Sync>,
}

impl std::fmt::Debug for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transition")
            .field("from", &self.from)
            .field("to", &self.to)
            .field("condition", &"<fn>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Guard: allowed transition table
// ---------------------------------------------------------------------------

/// Returns `true` if the `from → to` edge is structurally valid.
///
/// Invalid edges (e.g. `Idle → Terminating`) are blocked even when a
/// matching [`Transition`] is registered and its condition passes.
fn is_allowed(from: &AgentState, to: &AgentState) -> bool {
    use AgentState::*;
    matches!(
        (from, to),
        // Normal forward progression
        (Idle,        Running)     |
        (Idle,        Error(_))    |
        (Running,     Waiting)     |
        (Running,     Paused)      |
        (Running,     Terminating) |
        (Running,     Error(_))    |
        (Waiting,     Running)     |
        (Waiting,     Terminating) |
        (Waiting,     Error(_))    |
        (Paused,      Running)     |
        (Paused,      Terminating) |
        (Paused,      Error(_))    |
        // Error recovery: allow re-entering Running from Error
        (Error(_),    Running)     |
        (Error(_),    Terminating) |
        // Terminating is a sink — only Error is reachable from it via force
        (Terminating, Error(_))
    )
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------

/// Maximum number of history entries retained.
const MAX_HISTORY: usize = 50;

/// Finite state machine governing the agent lifecycle.
///
/// Constructed via [`StateMachine::new`] or [`StateMachine::default`]
/// (both start in [`AgentState::Idle`]).
pub struct StateMachine {
    current:     AgentState,
    transitions: Vec<Transition>,
    /// Rolling history: `(state entered, time of entry)`.
    history:     VecDeque<(AgentState, SystemTime)>,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl StateMachine {
    /// Create a new [`StateMachine`] starting in [`AgentState::Idle`].
    #[must_use]
    pub fn new() -> Self {
        let initial = AgentState::Idle;
        initial.on_enter();
        let mut history = VecDeque::with_capacity(MAX_HISTORY + 1);
        history.push_back((initial.clone(), SystemTime::now()));
        Self {
            current: initial,
            transitions: Vec::new(),
            history,
        }
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /// Register a guarded transition.
    ///
    /// The condition closure receives an [`AgentContext`] reference and
    /// returns `true` when the transition should fire.  Transitions are
    /// evaluated in registration order; the first matching one is taken.
    pub fn add_transition<F>(&mut self, from: AgentState, to: AgentState, condition: F)
    where
        F: Fn(&AgentContext) -> bool + Send + Sync + 'static,
    {
        self.transitions.push(Transition {
            from,
            to,
            condition: Box::new(condition),
        });
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// The state the FSM is currently in.
    #[must_use]
    pub fn current(&self) -> &AgentState {
        &self.current
    }

    /// Ordered history of `(state, entry_time)` pairs.
    ///
    /// The oldest entry is at index 0; the most recent is at the back.
    /// At most [`MAX_HISTORY`] (50) entries are retained.
    #[must_use]
    pub fn history(&self) -> &VecDeque<(AgentState, SystemTime)> {
        &self.history
    }

    // ------------------------------------------------------------------
    // Transition logic
    // ------------------------------------------------------------------

    /// Evaluate all registered transitions from the current state.
    ///
    /// The first transition whose `from` matches the current state **and**
    /// whose condition returns `true` **and** which passes the structural
    /// guard fires.  Returns the new state on success, or `None` if no
    /// transition matched.
    ///
    /// On a successful transition:
    /// 1. [`State::on_exit`] is called on the old state.
    /// 2. The current state is updated.
    /// 3. [`State::on_enter`] is called on the new state.
    /// 4. The new state is appended to the history ring buffer.
    pub fn transition(&mut self, ctx: &AgentContext) -> Option<AgentState> {
        // Find the first applicable transition.
        let target = self.transitions.iter().find_map(|t| {
            // Variant-level equality for the `from` guard (Error(_) matches any Error).
            if !states_match_from(&t.from, &self.current) {
                return None;
            }
            if !is_allowed(&self.current, &t.to) {
                tracing::warn!(
                    from = %self.current,
                    to   = %t.to,
                    "FSM: guard blocked structurally invalid transition"
                );
                return None;
            }
            if (t.condition)(ctx) {
                Some(t.to.clone())
            } else {
                None
            }
        });

        if let Some(new_state) = target {
            self.apply(new_state.clone());
            Some(new_state)
        } else {
            None
        }
    }

    /// Force a transition to `state`, bypassing condition guards.
    ///
    /// The structural guard (valid edge table) is **also** bypassed, making
    /// this suitable for emergency error recovery where the FSM may be in an
    /// unexpected state.
    pub fn force_transition(&mut self, state: AgentState) {
        tracing::warn!(
            from  = %self.current,
            to    = %state,
            "FSM: force_transition bypassing guards"
        );
        self.apply(state);
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn apply(&mut self, new_state: AgentState) {
        self.current.on_exit();
        self.current = new_state;
        self.current.on_enter();
        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back((self.current.clone(), SystemTime::now()));
    }
}

/// Variant-level equality for the `from` field: `Error(_)` matches any `Error`.
fn states_match_from(pattern: &AgentState, current: &AgentState) -> bool {
    use AgentState::*;
    match (pattern, current) {
        (Error(_), Error(_)) => true,
        (a, b)               => a == b,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn ctx(error_count: u32) -> AgentContext {
        AgentContext {
            task_id:     "test-task".into(),
            elapsed_ms:  100,
            error_count,
            last_error:  None,
        }
    }

    // ------------------------------------------------------------------
    // Basic transition
    // ------------------------------------------------------------------

    #[test]
    fn idle_to_running_when_condition_true() {
        let mut sm = StateMachine::new();
        sm.add_transition(AgentState::Idle, AgentState::Running, |_| true);
        let result = sm.transition(&ctx(0));
        assert_eq!(result, Some(AgentState::Running));
        assert_eq!(*sm.current(), AgentState::Running);
    }

    #[test]
    fn no_transition_when_condition_false() {
        let mut sm = StateMachine::new();
        sm.add_transition(AgentState::Idle, AgentState::Running, |_| false);
        let result = sm.transition(&ctx(0));
        assert_eq!(result, None);
        assert_eq!(*sm.current(), AgentState::Idle);
    }

    #[test]
    fn first_matching_transition_wins() {
        let mut sm = StateMachine::new();
        // Both transitions are valid from Idle; first one (→ Running) fires.
        sm.add_transition(AgentState::Idle, AgentState::Running, |_| true);
        sm.add_transition(AgentState::Idle, AgentState::Error("x".into()), |_| true);
        let result = sm.transition(&ctx(0));
        assert_eq!(result, Some(AgentState::Running));
    }

    // ------------------------------------------------------------------
    // Multi-step
    // ------------------------------------------------------------------

    #[test]
    fn multi_step_lifecycle() {
        let mut sm = StateMachine::new();
        sm.add_transition(AgentState::Idle,    AgentState::Running,     |_| true);
        sm.add_transition(AgentState::Running, AgentState::Waiting,     |_| true);
        sm.add_transition(AgentState::Waiting, AgentState::Terminating, |_| true);

        sm.transition(&ctx(0)); // → Running
        sm.transition(&ctx(0)); // → Waiting
        sm.transition(&ctx(0)); // → Terminating

        assert_eq!(*sm.current(), AgentState::Terminating);
    }

    // ------------------------------------------------------------------
    // Guard enforcement
    // ------------------------------------------------------------------

    #[test]
    fn guard_blocks_idle_to_terminating() {
        let mut sm = StateMachine::new();
        // Register the structurally invalid edge.
        sm.add_transition(AgentState::Idle, AgentState::Terminating, |_| true);
        // Should be blocked by the guard even though condition is true.
        let result = sm.transition(&ctx(0));
        assert_eq!(result, None);
        assert_eq!(*sm.current(), AgentState::Idle);
    }

    #[test]
    fn guard_blocks_idle_to_paused() {
        let mut sm = StateMachine::new();
        sm.add_transition(AgentState::Idle, AgentState::Paused, |_| true);
        let result = sm.transition(&ctx(0));
        assert_eq!(result, None);
    }

    // ------------------------------------------------------------------
    // Force transition
    // ------------------------------------------------------------------

    #[test]
    fn force_transition_bypasses_guard() {
        let mut sm = StateMachine::new();
        // Force directly from Idle → Terminating (normally guarded).
        sm.force_transition(AgentState::Terminating);
        assert_eq!(*sm.current(), AgentState::Terminating);
    }

    #[test]
    fn force_transition_for_error_recovery() {
        let mut sm = StateMachine::new();
        sm.force_transition(AgentState::Error("oom".into()));
        sm.force_transition(AgentState::Running); // recovery
        assert_eq!(*sm.current(), AgentState::Running);
    }

    // ------------------------------------------------------------------
    // History tracking
    // ------------------------------------------------------------------

    #[test]
    fn history_starts_with_idle() {
        let sm = StateMachine::new();
        assert_eq!(sm.history().len(), 1);
        assert_eq!(sm.history()[0].0, AgentState::Idle);
    }

    #[test]
    fn history_grows_with_transitions() {
        let mut sm = StateMachine::new();
        sm.add_transition(AgentState::Idle, AgentState::Running, |_| true);
        sm.transition(&ctx(0));
        assert_eq!(sm.history().len(), 2);
        assert_eq!(sm.history()[1].0, AgentState::Running);
    }

    #[test]
    fn history_capped_at_50() {
        let mut sm = StateMachine::new();
        // Oscillate Running ↔ Waiting to generate many transitions.
        sm.add_transition(AgentState::Idle,    AgentState::Running, |_| true);
        sm.add_transition(AgentState::Running, AgentState::Waiting, |_| true);
        sm.add_transition(AgentState::Waiting, AgentState::Running, |_| true);

        sm.transition(&ctx(0)); // → Running
        for _ in 0..60 {
            sm.transition(&ctx(0)); // Running→Waiting or Waiting→Running
        }

        assert!(sm.history().len() <= 50);
    }

    // ------------------------------------------------------------------
    // Context-driven conditions
    // ------------------------------------------------------------------

    #[test]
    fn transition_uses_context_error_count() {
        let mut sm = StateMachine::new();
        sm.add_transition(AgentState::Idle, AgentState::Running, |_| true);
        sm.transition(&ctx(0)); // → Running

        sm.add_transition(
            AgentState::Running,
            AgentState::Error("too many errors".into()),
            |ctx| ctx.error_count >= 3,
        );

        // Not yet enough errors.
        let result = sm.transition(&ctx(2));
        assert_eq!(result, None);

        // Now enough errors.
        let result = sm.transition(&ctx(3));
        assert!(matches!(result, Some(AgentState::Error(_))));
    }

    // ------------------------------------------------------------------
    // Error variant matching
    // ------------------------------------------------------------------

    #[test]
    fn error_to_running_recovery() {
        let mut sm = StateMachine::new();
        sm.force_transition(AgentState::Error("crash".into()));
        sm.add_transition(
            AgentState::Error("any".into()), // pattern — Error(_) matches any Error
            AgentState::Running,
            |_| true,
        );
        let result = sm.transition(&ctx(0));
        assert_eq!(result, Some(AgentState::Running));
    }
}
