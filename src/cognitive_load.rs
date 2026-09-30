//! # Cognitive Load Monitor
//!
//! Tracks and limits the agent's "cognitive load" — a composite measure of
//! how much concurrent work the agent is holding in mind.
//!
//! ## What is cognitive load?
//!
//! The monitor aggregates three independent pressure signals:
//!
//! | Signal | Meaning |
//! |--------|---------|
//! | Active goals | How many goals are currently `InProgress`. |
//! | Reasoning depth | How deeply nested the current reasoning stack is. |
//! | Memory pressure | How many episodic memory entries are held in working memory. |
//!
//! Each signal contributes a normalised score in `[0.0, 1.0]`.  The combined
//! load is a weighted average (weights configurable via [`CognitiveLoadConfig`]).
//!
//! When load exceeds a threshold the monitor recommends a
//! [`LoadAction`]: prioritise, drop low-priority work, or pause.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::cognitive_load::{
//!     CognitiveLoadMonitor, CognitiveLoadConfig, LoadAction,
//! };
//!
//! let config = CognitiveLoadConfig::default();
//! let mut monitor = CognitiveLoadMonitor::new(config);
//!
//! // Simulate loading up the agent.
//! monitor.set_active_goals(8);
//! monitor.set_reasoning_depth(5);
//! monitor.set_memory_entries(1200);
//!
//! let load = monitor.load();
//! let action = monitor.recommended_action();
//!
//! // Under high load the monitor recommends shedding work.
//! println!("load={load:.2}, action={action:?}");
//! ```

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CognitiveLoadConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`CognitiveLoadMonitor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadConfig {
    /// Maximum number of active (in-progress) goals before this signal saturates.
    pub max_active_goals: usize,
    /// Maximum reasoning stack depth before this signal saturates.
    pub max_reasoning_depth: usize,
    /// Maximum number of episodic memory entries before this signal saturates.
    pub max_memory_entries: usize,

    /// Weight applied to the goal-count signal (default: 0.4).
    pub goal_weight: f64,
    /// Weight applied to the reasoning-depth signal (default: 0.35).
    pub depth_weight: f64,
    /// Weight applied to the memory-pressure signal (default: 0.25).
    pub memory_weight: f64,

    /// Load fraction `[0, 1]` above which the agent should prioritise and shed
    /// low-priority work.
    pub shed_threshold: f64,
    /// Load fraction above which the agent should pause all new work.
    pub pause_threshold: f64,
}

impl Default for CognitiveLoadConfig {
    fn default() -> Self {
        Self {
            max_active_goals: 10,
            max_reasoning_depth: 12,
            max_memory_entries: 2000,
            goal_weight: 0.40,
            depth_weight: 0.35,
            memory_weight: 0.25,
            shed_threshold: 0.70,
            pause_threshold: 0.90,
        }
    }
}

// ---------------------------------------------------------------------------
// LoadAction
// ---------------------------------------------------------------------------

/// Recommended action based on the current cognitive load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadAction {
    /// Load is acceptable — continue normally.
    Continue,
    /// Load is high — prioritise and drop low-priority goals.
    Prioritise,
    /// Load is critical — pause all new work until load drops.
    Pause,
}

impl std::fmt::Display for LoadAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadAction::Continue => write!(f, "Continue"),
            LoadAction::Prioritise => write!(f, "Prioritise"),
            LoadAction::Pause => write!(f, "Pause"),
        }
    }
}

// ---------------------------------------------------------------------------
// CognitiveLoadSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of all cognitive load signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadSnapshot {
    /// Number of currently active (in-progress) goals.
    pub active_goals: usize,
    /// Current depth of the reasoning stack.
    pub reasoning_depth: usize,
    /// Number of episodic memory entries held in working memory.
    pub memory_entries: usize,
    /// Normalised goal-count signal `[0, 1]`.
    pub goal_signal: f64,
    /// Normalised reasoning-depth signal `[0, 1]`.
    pub depth_signal: f64,
    /// Normalised memory-pressure signal `[0, 1]`.
    pub memory_signal: f64,
    /// Combined weighted load score `[0, 1]`.
    pub total_load: f64,
    /// Recommended action at the time of this snapshot.
    pub recommended_action: LoadAction,
}

// ---------------------------------------------------------------------------
// CognitiveLoadMonitor
// ---------------------------------------------------------------------------

/// Tracks the agent's cognitive load and recommends load-management actions.
#[derive(Debug, Clone)]
pub struct CognitiveLoadMonitor {
    config: CognitiveLoadConfig,
    active_goals: usize,
    reasoning_depth: usize,
    memory_entries: usize,
}

impl CognitiveLoadMonitor {
    /// Create a monitor with the given configuration.
    pub fn new(config: CognitiveLoadConfig) -> Self {
        Self {
            config,
            active_goals: 0,
            reasoning_depth: 0,
            memory_entries: 0,
        }
    }

    // ── Setters ───────────────────────────────────────────────────────────────

    /// Update the count of active (in-progress) goals.
    pub fn set_active_goals(&mut self, n: usize) {
        self.active_goals = n;
    }

    /// Push a reasoning frame (increment depth by 1).
    pub fn push_reasoning_frame(&mut self) {
        self.reasoning_depth = self.reasoning_depth.saturating_add(1);
    }

    /// Pop a reasoning frame (decrement depth by 1, floor at 0).
    pub fn pop_reasoning_frame(&mut self) {
        self.reasoning_depth = self.reasoning_depth.saturating_sub(1);
    }

    /// Set the reasoning depth directly.
    pub fn set_reasoning_depth(&mut self, depth: usize) {
        self.reasoning_depth = depth;
    }

    /// Update the count of episodic memory entries.
    pub fn set_memory_entries(&mut self, n: usize) {
        self.memory_entries = n;
    }

    // ── Computation ───────────────────────────────────────────────────────────

    /// Normalised goal-count signal `[0.0, 1.0]`.
    pub fn goal_signal(&self) -> f64 {
        if self.config.max_active_goals == 0 {
            return 0.0;
        }
        (self.active_goals as f64 / self.config.max_active_goals as f64).min(1.0)
    }

    /// Normalised reasoning-depth signal `[0.0, 1.0]`.
    pub fn depth_signal(&self) -> f64 {
        if self.config.max_reasoning_depth == 0 {
            return 0.0;
        }
        (self.reasoning_depth as f64 / self.config.max_reasoning_depth as f64).min(1.0)
    }

    /// Normalised memory-pressure signal `[0.0, 1.0]`.
    pub fn memory_signal(&self) -> f64 {
        if self.config.max_memory_entries == 0 {
            return 0.0;
        }
        (self.memory_entries as f64 / self.config.max_memory_entries as f64).min(1.0)
    }

    /// Combined weighted load score `[0.0, 1.0]`.
    pub fn load(&self) -> f64 {
        let g = self.goal_signal() * self.config.goal_weight;
        let d = self.depth_signal() * self.config.depth_weight;
        let m = self.memory_signal() * self.config.memory_weight;
        // Normalise by sum of weights in case they don't sum to 1.0.
        let weight_sum =
            self.config.goal_weight + self.config.depth_weight + self.config.memory_weight;
        if weight_sum < f64::EPSILON {
            return 0.0;
        }
        ((g + d + m) / weight_sum).min(1.0)
    }

    /// Recommended [`LoadAction`] for the current load.
    pub fn recommended_action(&self) -> LoadAction {
        let load = self.load();
        if load >= self.config.pause_threshold {
            LoadAction::Pause
        } else if load >= self.config.shed_threshold {
            LoadAction::Prioritise
        } else {
            LoadAction::Continue
        }
    }

    /// Build a full snapshot of all signals and the recommended action.
    pub fn snapshot(&self) -> CognitiveLoadSnapshot {
        CognitiveLoadSnapshot {
            active_goals: self.active_goals,
            reasoning_depth: self.reasoning_depth,
            memory_entries: self.memory_entries,
            goal_signal: self.goal_signal(),
            depth_signal: self.depth_signal(),
            memory_signal: self.memory_signal(),
            total_load: self.load(),
            recommended_action: self.recommended_action(),
        }
    }

    /// Return `true` if the agent should accept new work.
    pub fn can_accept_work(&self) -> bool {
        self.recommended_action() == LoadAction::Continue
    }

    /// Return `true` if the agent is overloaded and should pause.
    pub fn is_overloaded(&self) -> bool {
        self.recommended_action() == LoadAction::Pause
    }
}

impl Default for CognitiveLoadMonitor {
    fn default() -> Self {
        Self::new(CognitiveLoadConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn light_monitor() -> CognitiveLoadMonitor {
        CognitiveLoadMonitor::new(CognitiveLoadConfig::default())
    }

    #[test]
    fn zero_load_on_fresh_monitor() {
        let m = light_monitor();
        assert_eq!(m.load(), 0.0);
        assert_eq!(m.recommended_action(), LoadAction::Continue);
    }

    #[test]
    fn goal_signal_saturates_at_one() {
        let mut m = light_monitor();
        m.set_active_goals(100); // way over max
        assert_eq!(m.goal_signal(), 1.0);
    }

    #[test]
    fn depth_signal_saturates_at_one() {
        let mut m = light_monitor();
        m.set_reasoning_depth(100);
        assert_eq!(m.depth_signal(), 1.0);
    }

    #[test]
    fn memory_signal_saturates_at_one() {
        let mut m = light_monitor();
        m.set_memory_entries(100_000);
        assert_eq!(m.memory_signal(), 1.0);
    }

    #[test]
    fn full_load_recommends_pause() {
        let mut m = light_monitor();
        m.set_active_goals(10);
        m.set_reasoning_depth(12);
        m.set_memory_entries(2000);
        assert_eq!(m.recommended_action(), LoadAction::Pause);
        assert!(m.is_overloaded());
    }

    #[test]
    fn moderate_load_recommends_prioritise() {
        let mut m = light_monitor();
        // ~75 % active goals, rest at 0
        m.set_active_goals(8);
        let load = m.load();
        // 8/10 * 0.40 / 1.0 = 0.32 — still below shed threshold with other signals at 0
        // Need to make goal signal high enough to cross 0.70 threshold.
        // With max=10, goals=10, goal_signal=1.0 → weighted=0.40 (still below 0.70).
        // Use custom config to lower threshold.
        let config = CognitiveLoadConfig {
            shed_threshold: 0.30,
            pause_threshold: 0.50,
            ..Default::default()
        };
        let mut m2 = CognitiveLoadMonitor::new(config);
        m2.set_active_goals(8);
        assert_eq!(m2.recommended_action(), LoadAction::Prioritise);
        let _ = load; // silence unused warning
    }

    #[test]
    fn push_and_pop_reasoning_frame() {
        let mut m = light_monitor();
        m.push_reasoning_frame();
        m.push_reasoning_frame();
        assert_eq!(m.reasoning_depth, 2);
        m.pop_reasoning_frame();
        assert_eq!(m.reasoning_depth, 1);
    }

    #[test]
    fn pop_below_zero_stays_at_zero() {
        let mut m = light_monitor();
        m.pop_reasoning_frame();
        assert_eq!(m.reasoning_depth, 0);
    }

    #[test]
    fn snapshot_captures_all_signals() {
        let mut m = light_monitor();
        m.set_active_goals(5);
        let snap = m.snapshot();
        assert_eq!(snap.active_goals, 5);
        assert!(snap.goal_signal > 0.0);
        assert_eq!(snap.reasoning_depth, 0);
    }

    #[test]
    fn can_accept_work_false_when_overloaded() {
        let mut m = light_monitor();
        m.set_active_goals(10);
        m.set_reasoning_depth(12);
        m.set_memory_entries(2000);
        assert!(!m.can_accept_work());
    }

    #[test]
    fn default_monitor_has_zero_load() {
        let m = CognitiveLoadMonitor::default();
        assert_eq!(m.load(), 0.0);
    }
}
