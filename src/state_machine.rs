//! # Hierarchical State Machine (HSM)
//!
//! Provides a hierarchical state machine with guarded transitions, entry/exit
//! actions, parent-state ancestry checks, and a fluent builder API.

use std::collections::HashMap;
use std::sync::Arc;

// ── Type aliases ─────────────────────────────────────────────────────────────

/// Opaque numeric identifier for a state.
pub type StateId = usize;

/// Opaque numeric identifier for an event.
pub type EventId = usize;

/// A callable action executed during a state transition.
pub type ActionFn = Arc<dyn Fn(&mut StateContext) + Send + Sync>;

/// A callable guard evaluated before a transition is taken.
pub type GuardFn = Arc<dyn Fn(&StateContext) -> bool + Send + Sync>;

// ── StateContext ──────────────────────────────────────────────────────────────

/// Mutable context carried through the state machine and passed to actions/guards.
pub struct StateContext {
    /// Arbitrary key-value data attached to this context.
    pub data: HashMap<String, String>,
    /// Ordered list of states visited so far (most recent last).
    pub history: Vec<StateId>,
    /// Total number of events processed.
    pub event_count: usize,
}

impl StateContext {
    /// Create a new, empty context.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            history: Vec::new(),
            event_count: 0,
        }
    }

    /// Store a key-value pair in the context data map.
    pub fn set(&mut self, key: &str, val: &str) {
        self.data.insert(key.to_string(), val.to_string());
    }

    /// Retrieve a value from the context data map.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(String::as_str)
    }
}

impl Default for StateContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── Transition ────────────────────────────────────────────────────────────────

/// A directed edge in the state machine graph.
pub struct Transition {
    /// Source state identifier.
    pub from: StateId,
    /// Event that triggers this transition.
    pub event: EventId,
    /// Target state identifier.
    pub to: StateId,
    /// Optional guard; if `None` the transition is always eligible.
    pub guard: Option<GuardFn>,
    /// Optional action executed between exit and entry callbacks.
    pub action: Option<ActionFn>,
}

// ── State ─────────────────────────────────────────────────────────────────────

/// A node in the state machine hierarchy.
pub struct State {
    /// Unique numeric identifier.
    pub id: StateId,
    /// Human-readable name.
    pub name: String,
    /// Optional parent state for hierarchical nesting.
    pub parent: Option<StateId>,
    /// Called when the machine enters this state.
    pub entry: Option<ActionFn>,
    /// Called when the machine exits this state.
    pub exit: Option<ActionFn>,
    /// Whether this is a terminal (accepting) state.
    pub is_final: bool,
}

// ── StateMachineError ─────────────────────────────────────────────────────────

/// Errors returned by [`StateMachine::process_event`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateMachineError {
    /// No transition matches the given event from the current state.
    NoTransition,
    /// A matching transition was found but its guard returned `false`.
    GuardFailed,
    /// The referenced state id does not exist in this machine.
    InvalidState(StateId),
}

impl core::fmt::Display for StateMachineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoTransition => write!(f, "no transition found for this event"),
            Self::GuardFailed => write!(f, "transition guard rejected the event"),
            Self::InvalidState(id) => write!(f, "invalid state id: {id}"),
        }
    }
}

impl std::error::Error for StateMachineError {}

// ── StateMachine ──────────────────────────────────────────────────────────────

/// A hierarchical state machine.
///
/// States are stored in a flat `Vec`; hierarchy is expressed via the
/// [`State::parent`] field.  `process_event` walks the transition table,
/// evaluates guards, fires exit/action/entry hooks, and updates history.
pub struct StateMachine {
    states: Vec<State>,
    transitions: Vec<Transition>,
    current: StateId,
    context: StateContext,
}

impl StateMachine {
    /// Create a new machine with `initial` as the starting state.
    pub fn new(initial: StateId) -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            current: initial,
            context: StateContext::new(),
        }
    }

    /// Register a state.  Duplicate ids are allowed but not recommended.
    pub fn add_state(&mut self, state: State) {
        self.states.push(state);
    }

    /// Register a transition edge.
    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Process an event: find an applicable transition, execute lifecycle hooks,
    /// and return the new active [`StateId`].
    ///
    /// Returns [`StateMachineError::NoTransition`] when no transition matches,
    /// [`StateMachineError::GuardFailed`] when a matching transition's guard
    /// rejects the event, and [`StateMachineError::InvalidState`] when the
    /// target state does not exist.
    pub fn process_event(&mut self, event: EventId) -> Result<StateId, StateMachineError> {
        // Find first transition matching current state + event.
        let transition_idx = self
            .transitions
            .iter()
            .position(|t| t.from == self.current && t.event == event)
            .ok_or(StateMachineError::NoTransition)?;

        // Evaluate guard (borrow the guard Arc before mutating context).
        let guard_ok = match &self.transitions[transition_idx].guard {
            Some(g) => {
                let g = Arc::clone(g);
                g(&self.context)
            }
            None => true,
        };
        if !guard_ok {
            return Err(StateMachineError::GuardFailed);
        }

        let to = self.transitions[transition_idx].to;

        // Verify target state exists.
        if !self.states.iter().any(|s| s.id == to) {
            return Err(StateMachineError::InvalidState(to));
        }

        // Execute exit action on the current state.
        if let Some(exit_fn) = self
            .states
            .iter()
            .find(|s| s.id == self.current)
            .and_then(|s| s.exit.as_ref())
            .map(Arc::clone)
        {
            exit_fn(&mut self.context);
        }

        // Execute the transition action.
        if let Some(action_fn) = self.transitions[transition_idx].action.as_ref().map(Arc::clone) {
            action_fn(&mut self.context);
        }

        // Update state and context bookkeeping.
        self.context.history.push(self.current);
        self.context.event_count += 1;
        self.current = to;

        // Execute entry action on the new state.
        if let Some(entry_fn) = self
            .states
            .iter()
            .find(|s| s.id == to)
            .and_then(|s| s.entry.as_ref())
            .map(Arc::clone)
        {
            entry_fn(&mut self.context);
        }

        Ok(self.current)
    }

    /// Return the active state id.
    pub fn current_state(&self) -> StateId {
        self.current
    }

    /// Return the name of a state by id, if known.
    pub fn state_name(&self, id: StateId) -> Option<&str> {
        self.states.iter().find(|s| s.id == id).map(|s| s.name.as_str())
    }

    /// Return `true` if the machine is in `id` or any descendant of `id`.
    pub fn is_in_state(&self, id: StateId) -> bool {
        if self.current == id {
            return true;
        }
        self.ancestors(self.current).contains(&id)
    }

    /// Collect all ancestor [`StateId`]s for `id` from immediate parent to root.
    pub fn ancestors(&self, id: StateId) -> Vec<StateId> {
        let mut result = Vec::new();
        let mut cursor = id;
        loop {
            match self.states.iter().find(|s| s.id == cursor).and_then(|s| s.parent) {
                Some(parent) => {
                    result.push(parent);
                    cursor = parent;
                }
                None => break,
            }
        }
        result
    }

    /// Return a slice of the state-visit history (oldest first).
    pub fn history(&self) -> &[StateId] {
        &self.context.history
    }

    /// Immutable access to the shared context.
    pub fn context(&self) -> &StateContext {
        &self.context
    }

    /// Mutable access to the shared context.
    pub fn context_mut(&mut self) -> &mut StateContext {
        &mut self.context
    }
}

// ── StateMachineBuilder ───────────────────────────────────────────────────────

/// Fluent builder for [`StateMachine`].
///
/// ```rust
/// use llm_agent_runtime::state_machine::{StateMachineBuilder, State, Transition};
///
/// let machine = StateMachineBuilder::new(0)
///     .state(State { id: 0, name: "Idle".into(), parent: None, entry: None, exit: None, is_final: false })
///     .state(State { id: 1, name: "Running".into(), parent: None, entry: None, exit: None, is_final: false })
///     .transition(Transition { from: 0, event: 1, to: 1, guard: None, action: None })
///     .build();
///
/// assert_eq!(machine.current_state(), 0);
/// ```
pub struct StateMachineBuilder {
    initial: StateId,
    states: Vec<State>,
    transitions: Vec<Transition>,
}

impl StateMachineBuilder {
    /// Create a builder with the given initial state id.
    pub fn new(initial: StateId) -> Self {
        Self {
            initial,
            states: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Add a state to the machine being built.
    pub fn state(mut self, state: State) -> Self {
        self.states.push(state);
        self
    }

    /// Add a transition to the machine being built.
    pub fn transition(mut self, transition: Transition) -> Self {
        self.transitions.push(transition);
        self
    }

    /// Consume the builder and produce a [`StateMachine`].
    pub fn build(self) -> StateMachine {
        let mut sm = StateMachine::new(self.initial);
        for s in self.states {
            sm.add_state(s);
        }
        for t in self.transitions {
            sm.add_transition(t);
        }
        sm
    }
}
