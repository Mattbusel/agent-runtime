//! # Module: plan_validator
//!
//! Plan feasibility checking and constraint satisfaction.
//!
//! ## Overview
//!
//! [`PlanValidator`] checks a [`Plan`] against a set of [`Constraint`]s and
//! the validator's declared available capabilities and resources.  It returns a
//! [`ValidationResult`] that describes every violation found together with
//! optional optimisation suggestions.
//!
//! ## Usage
//!
//! ```rust
//! use llm_agent_runtime::plan_validator::{
//!     Constraint, Plan, PlanStep, PlanValidator,
//! };
//! use std::collections::HashMap;
//!
//! let mut resources = HashMap::new();
//! resources.insert("cpu".to_string(), 4.0_f64);
//!
//! let validator = PlanValidator::new(
//!     vec!["search".to_string()],
//!     resources,
//! );
//!
//! let step = PlanStep {
//!     id: "s1".to_string(),
//!     name: "Search".to_string(),
//!     estimated_duration_ms: 500,
//!     requires: vec!["search".to_string()],
//!     resource_usage: HashMap::new(),
//!     parallel_group: None,
//! };
//!
//! let plan = Plan {
//!     id: "p1".to_string(),
//!     steps: vec![step],
//!     constraints: vec![],
//! };
//!
//! let result = validator.validate(&plan);
//! assert!(result.valid);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ── Constraint ────────────────────────────────────────────────────────────────

/// A constraint that a [`Plan`] must satisfy.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// The total plan duration must not exceed this many milliseconds.
    MaxDuration(u64),
    /// All steps must have access to this capability.
    RequiresCapability(String),
    /// None of the named steps may appear together in the same plan.
    MutuallyExclusive(Vec<String>),
    /// A step with this ID must exist as a dependency of any step that
    /// references it.
    DependsOn(String),
    /// No more than this many steps may execute in parallel.
    MaxParallelism(usize),
    /// Total consumption of `resource` across all steps must not exceed `max`.
    ResourceLimit {
        /// The resource identifier (e.g. `"cpu"`, `"memory_gb"`).
        resource: String,
        /// Maximum allowed aggregate value.
        max: f64,
    },
}

// ── PlanStep ──────────────────────────────────────────────────────────────────

/// A single step within a [`Plan`].
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Unique identifier for this step within the plan.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Estimated execution time in milliseconds.
    pub estimated_duration_ms: u64,
    /// Capability names that must be present in the validator's
    /// `available_capabilities` list.
    pub requires: Vec<String>,
    /// Resource consumption this step contributes (key → quantity).
    pub resource_usage: HashMap<String, f64>,
    /// Optional label grouping steps that can run in parallel.
    pub parallel_group: Option<String>,
}

// ── Plan ──────────────────────────────────────────────────────────────────────

/// A collection of steps and constraints to be validated.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Unique plan identifier.
    pub id: String,
    /// Ordered list of steps.
    pub steps: Vec<PlanStep>,
    /// Constraints the plan must satisfy.
    pub constraints: Vec<Constraint>,
}

// ── ValidationError ───────────────────────────────────────────────────────────

/// A single validation failure.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// The step involved, if the error is step-specific.
    pub step_id: Option<String>,
    /// Human-readable description of the violated constraint.
    pub constraint: String,
    /// Detailed message.
    pub message: String,
}

// ── ValidationResult ──────────────────────────────────────────────────────────

/// The outcome of validating a [`Plan`].
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// `true` when no errors were found.
    pub valid: bool,
    /// All constraint violations detected.
    pub errors: Vec<ValidationError>,
    /// Non-fatal observations and optimisation hints.
    pub warnings: Vec<String>,
    /// Critical-path estimate of the total plan duration in milliseconds.
    pub estimated_total_ms: u64,
}

// ── PlanValidator ─────────────────────────────────────────────────────────────

/// Validates plans against capability and resource constraints.
#[derive(Debug, Clone)]
pub struct PlanValidator {
    /// Capabilities available to steps at runtime.
    pub available_capabilities: Vec<String>,
    /// Resources available at runtime (key → maximum quantity).
    pub available_resources: HashMap<String, f64>,
}

impl PlanValidator {
    /// Create a new validator with the supplied capabilities and resources.
    pub fn new(capabilities: Vec<String>, resources: HashMap<String, f64>) -> Self {
        Self {
            available_capabilities: capabilities,
            available_resources: resources,
        }
    }

    /// Validate `plan` and return a [`ValidationResult`].
    pub fn validate(&self, plan: &Plan) -> ValidationResult {
        let mut errors: Vec<ValidationError> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // Per-step capability checks.
        for step in &plan.steps {
            errors.extend(self.check_capability_requirements(step));
        }

        // Resource-limit checks (aggregate).
        errors.extend(self.check_resource_limits(plan));

        // Constraint-level checks.
        for constraint in &plan.constraints {
            match constraint {
                Constraint::MutuallyExclusive(_) => {
                    errors.extend(Self::check_mutual_exclusion(plan, constraint));
                }
                Constraint::DependsOn(_) => {
                    // Handled below via global dependency check.
                }
                Constraint::MaxDuration(limit_ms) => {
                    let est = Self::estimate_duration(plan);
                    if est > *limit_ms {
                        errors.push(ValidationError {
                            step_id: None,
                            constraint: format!("MaxDuration({})", limit_ms),
                            message: format!(
                                "Estimated duration {}ms exceeds limit {}ms",
                                est, limit_ms
                            ),
                        });
                    }
                }
                Constraint::MaxParallelism(max) => {
                    let max_parallel = Self::max_parallel_steps(plan);
                    if max_parallel > *max {
                        errors.push(ValidationError {
                            step_id: None,
                            constraint: format!("MaxParallelism({})", max),
                            message: format!(
                                "Up to {} steps may run in parallel, exceeding limit {}",
                                max_parallel, max
                            ),
                        });
                    }
                }
                Constraint::RequiresCapability(cap) => {
                    if !self.available_capabilities.contains(cap) {
                        errors.push(ValidationError {
                            step_id: None,
                            constraint: format!("RequiresCapability({})", cap),
                            message: format!("Capability '{}' is not available", cap),
                        });
                    }
                }
                Constraint::ResourceLimit { resource, max } => {
                    let total: f64 = plan
                        .steps
                        .iter()
                        .map(|s| s.resource_usage.get(resource).copied().unwrap_or(0.0))
                        .sum();
                    if total > *max {
                        errors.push(ValidationError {
                            step_id: None,
                            constraint: format!("ResourceLimit({}, {})", resource, max),
                            message: format!(
                                "Aggregate '{}' usage {:.2} exceeds limit {:.2}",
                                resource, total, max
                            ),
                        });
                    }
                }
            }
        }

        // Global dependency validity.
        errors.extend(Self::check_dependencies(plan));

        // Optimisation warnings.
        warnings.extend(Self::suggest_optimizations(plan));

        let estimated_total_ms = Self::estimate_duration(plan);

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
            estimated_total_ms,
        }
    }

    /// Check that each capability required by `step` is available.
    pub fn check_capability_requirements(&self, step: &PlanStep) -> Vec<ValidationError> {
        step.requires
            .iter()
            .filter(|cap| !self.available_capabilities.contains(cap))
            .map(|cap| ValidationError {
                step_id: Some(step.id.clone()),
                constraint: format!("RequiresCapability({})", cap),
                message: format!(
                    "Step '{}' requires capability '{}' which is not available",
                    step.id, cap
                ),
            })
            .collect()
    }

    /// Aggregate resource usage across all steps and check against any
    /// [`Constraint::ResourceLimit`] constraints in the plan.
    pub fn check_resource_limits(&self, plan: &Plan) -> Vec<ValidationError> {
        let mut totals: HashMap<String, f64> = HashMap::new();
        for step in &plan.steps {
            for (resource, qty) in &step.resource_usage {
                *totals.entry(resource.clone()).or_insert(0.0) += qty;
            }
        }

        let mut errors = Vec::new();
        for (resource, total) in &totals {
            if let Some(&available) = self.available_resources.get(resource) {
                if *total > available {
                    errors.push(ValidationError {
                        step_id: None,
                        constraint: format!("ResourceLimit({}, {})", resource, available),
                        message: format!(
                            "Aggregate '{}' usage {:.2} exceeds available {:.2}",
                            resource, total, available
                        ),
                    });
                }
            }
        }
        errors
    }

    /// Verify that a [`Constraint::MutuallyExclusive`] is not violated — i.e.
    /// no two (or more) of the named steps appear in the plan simultaneously.
    pub fn check_mutual_exclusion(plan: &Plan, constraint: &Constraint) -> Vec<ValidationError> {
        let names = match constraint {
            Constraint::MutuallyExclusive(names) => names,
            _ => return Vec::new(),
        };

        let present: Vec<&str> = plan
            .steps
            .iter()
            .filter_map(|s| {
                if names.contains(&s.id) {
                    Some(s.id.as_str())
                } else {
                    None
                }
            })
            .collect();

        if present.len() > 1 {
            vec![ValidationError {
                step_id: None,
                constraint: format!("MutuallyExclusive({:?})", names),
                message: format!(
                    "Steps {:?} are mutually exclusive but all appear in the plan",
                    present
                ),
            }]
        } else {
            Vec::new()
        }
    }

    /// Ensure that every [`Constraint::DependsOn`] reference in the plan
    /// resolves to a valid step id, and detect trivial self-cycles.
    pub fn check_dependencies(plan: &Plan) -> Vec<ValidationError> {
        let valid_ids: HashSet<&str> = plan.steps.iter().map(|s| s.id.as_str()).collect();
        let mut errors = Vec::new();

        for constraint in &plan.constraints {
            if let Constraint::DependsOn(dep_id) = constraint {
                if !valid_ids.contains(dep_id.as_str()) {
                    errors.push(ValidationError {
                        step_id: None,
                        constraint: format!("DependsOn({})", dep_id),
                        message: format!(
                            "Dependency '{}' does not reference any known step id",
                            dep_id
                        ),
                    });
                }
            }
        }

        errors
    }

    /// Estimate the total plan duration using a simple critical-path model:
    ///
    /// 1. Steps in the same `parallel_group` contribute only their maximum
    ///    duration to the total.
    /// 2. Steps without a group are sequential and their durations are summed.
    /// 3. Dependencies from [`Constraint::DependsOn`] are respected via
    ///    topological ordering when step ids can be resolved.
    pub fn estimate_duration(plan: &Plan) -> u64 {
        // Collect DependsOn constraints as a set of required step ids.
        let dep_ids: HashSet<&str> = plan
            .constraints
            .iter()
            .filter_map(|c| {
                if let Constraint::DependsOn(id) = c {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect();

        // Build a simple topological order via BFS (Kahn's algorithm).
        // For each step, count how many DependsOn constraints name it (in-degree).
        let step_ids: Vec<&str> = plan.steps.iter().map(|s| s.id.as_str()).collect();
        let mut in_degree: HashMap<&str, usize> = step_ids.iter().map(|id| (*id, 0)).collect();

        // Every step that is named in a DependsOn constraint is treated as a
        // dependency that must complete before anything that implicitly depends on it.
        for id in &dep_ids {
            if let Some(count) = in_degree.get_mut(*id) {
                *count += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();

        let mut ordered: Vec<&str> = Vec::with_capacity(plan.steps.len());
        while let Some(id) = queue.pop_front() {
            ordered.push(id);
        }

        // For any steps not yet ordered, append them.
        for step in &plan.steps {
            if !ordered.contains(&step.id.as_str()) {
                ordered.push(step.id.as_str());
            }
        }

        // Build a lookup for duration and parallel_group.
        let step_map: HashMap<&str, &PlanStep> =
            plan.steps.iter().map(|s| (s.id.as_str(), s)).collect();

        // Accumulate duration: group parallel steps, sum sequential ones.
        let mut total_ms: u64 = 0;
        let mut seen_groups: HashMap<&str, u64> = HashMap::new();

        for id in &ordered {
            if let Some(step) = step_map.get(id) {
                if let Some(ref group) = step.parallel_group {
                    let entry = seen_groups.entry(group.as_str()).or_insert(0);
                    *entry = (*entry).max(step.estimated_duration_ms);
                } else {
                    total_ms += step.estimated_duration_ms;
                }
            }
        }

        for (_, group_ms) in seen_groups {
            total_ms += group_ms;
        }

        total_ms
    }

    /// Return the number of steps that could run concurrently (largest
    /// parallel group size, or 1 for purely sequential plans).
    fn max_parallel_steps(plan: &Plan) -> usize {
        let mut group_counts: HashMap<&str, usize> = HashMap::new();
        for step in &plan.steps {
            if let Some(ref g) = step.parallel_group {
                *group_counts.entry(g.as_str()).or_insert(0) += 1;
            }
        }
        group_counts.values().copied().max().unwrap_or(1)
    }

    /// Generate human-readable optimisation suggestions for the plan.
    pub fn suggest_optimizations(plan: &Plan) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Flag steps with very long durations.
        for step in &plan.steps {
            if step.estimated_duration_ms > 30_000 {
                suggestions.push(format!(
                    "Step '{}' has an estimated duration of {}ms (>30s); consider breaking it up",
                    step.id, step.estimated_duration_ms
                ));
            }
        }

        // Identify ungrouped steps that share no dependencies and could be parallelised.
        let ungrouped: Vec<&PlanStep> = plan
            .steps
            .iter()
            .filter(|s| s.parallel_group.is_none())
            .collect();

        if ungrouped.len() > 1 {
            let ids: Vec<&str> = ungrouped.iter().map(|s| s.id.as_str()).collect();
            suggestions.push(format!(
                "Steps {:?} are sequential but have no declared dependencies — consider assigning them to a parallel group",
                ids
            ));
        }

        suggestions
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_validator() -> PlanValidator {
        let mut resources = HashMap::new();
        resources.insert("cpu".to_string(), 4.0);
        resources.insert("memory_gb".to_string(), 8.0);
        PlanValidator::new(
            vec!["search".to_string(), "compute".to_string()],
            resources,
        )
    }

    fn simple_step(id: &str, cap: &str, cpu: f64) -> PlanStep {
        let mut usage = HashMap::new();
        usage.insert("cpu".to_string(), cpu);
        PlanStep {
            id: id.to_string(),
            name: id.to_string(),
            estimated_duration_ms: 1_000,
            requires: vec![cap.to_string()],
            resource_usage: usage,
            parallel_group: None,
        }
    }

    #[test]
    fn valid_plan_passes() {
        let validator = make_validator();
        let plan = Plan {
            id: "p1".to_string(),
            steps: vec![simple_step("s1", "search", 1.0)],
            constraints: vec![],
        };
        let result = validator.validate(&plan);
        assert!(result.valid, "{:?}", result.errors);
    }

    #[test]
    fn missing_capability_fails() {
        let validator = make_validator();
        let plan = Plan {
            id: "p2".to_string(),
            steps: vec![simple_step("s1", "fly", 1.0)],
            constraints: vec![],
        };
        let result = validator.validate(&plan);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.message.contains("fly")));
    }

    #[test]
    fn exceeds_resource_fails() {
        let validator = make_validator();
        let plan = Plan {
            id: "p3".to_string(),
            steps: vec![
                simple_step("s1", "search", 3.0),
                simple_step("s2", "compute", 3.0),
            ],
            constraints: vec![],
        };
        let result = validator.validate(&plan);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.message.contains("cpu")));
    }

    #[test]
    fn mutual_exclusion_detected() {
        let validator = make_validator();
        let plan = Plan {
            id: "p4".to_string(),
            steps: vec![
                simple_step("s1", "search", 0.5),
                simple_step("s2", "compute", 0.5),
            ],
            constraints: vec![Constraint::MutuallyExclusive(vec![
                "s1".to_string(),
                "s2".to_string(),
            ])],
        };
        let result = validator.validate(&plan);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.constraint.contains("MutuallyExclusive")));
    }

    #[test]
    fn dependency_cycle_detected() {
        // A DependsOn reference to a non-existent step id is flagged.
        let validator = make_validator();
        let plan = Plan {
            id: "p5".to_string(),
            steps: vec![simple_step("s1", "search", 0.5)],
            constraints: vec![Constraint::DependsOn("ghost".to_string())],
        };
        let result = validator.validate(&plan);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.message.contains("ghost")));
    }

    #[test]
    fn duration_estimate_respects_parallel_groups() {
        let mut usage = HashMap::new();
        usage.insert("cpu".to_string(), 0.5);

        let make = |id: &str, ms: u64, group: Option<&str>| PlanStep {
            id: id.to_string(),
            name: id.to_string(),
            estimated_duration_ms: ms,
            requires: vec!["search".to_string()],
            resource_usage: usage.clone(),
            parallel_group: group.map(str::to_string),
        };

        let plan = Plan {
            id: "p6".to_string(),
            steps: vec![
                make("s1", 2_000, Some("g1")),
                make("s2", 3_000, Some("g1")),
                make("s3", 1_000, None),
            ],
            constraints: vec![],
        };

        // g1 contributes max(2000, 3000) = 3000; s3 contributes 1000.
        let est = PlanValidator::estimate_duration(&plan);
        assert_eq!(est, 4_000);
    }
}
