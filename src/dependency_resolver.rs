//! # Module: dependency_resolver
//!
//! ## Responsibility
//! Resolve and topologically order task dependencies with cycle detection.
//! Uses Kahn's algorithm to produce execution *waves* — sets of tasks that
//! can run concurrently — and computes the critical path by longest weighted
//! chain, parallelism factor, resource conflicts within a wave, and total
//! estimated duration.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Data structures ───────────────────────────────────────────────────────────

/// A task node in the dependency graph.
#[derive(Debug, Clone)]
pub struct TaskNode {
    /// Unique identifier for this task.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// IDs of tasks that must complete before this one starts.
    pub dependencies: Vec<String>,
    /// Estimated execution time in milliseconds.
    pub estimated_duration_ms: u64,
    /// Named resources required during execution (e.g. "gpu", "db-conn").
    pub resources: Vec<String>,
}

/// Errors that can occur during dependency resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionError {
    /// The graph contains a cycle; the vec holds the cycle member IDs.
    CyclicDependency(Vec<String>),
    /// A task references a dependency that has not been added.
    MissingDependency(String),
    /// Two or more tasks in the same wave share a resource.
    ResourceConflict(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolutionError::CyclicDependency(ids) => {
                write!(f, "cyclic dependency among: {}", ids.join(", "))
            }
            ResolutionError::MissingDependency(id) => {
                write!(f, "missing dependency: {id}")
            }
            ResolutionError::ResourceConflict(res) => {
                write!(f, "resource conflict: {res}")
            }
        }
    }
}

// ── DependencyResolver ────────────────────────────────────────────────────────

/// Resolves task execution order using Kahn's topological sort.
#[derive(Debug, Default)]
pub struct DependencyResolver {
    tasks: HashMap<String, TaskNode>,
}

impl DependencyResolver {
    /// Creates a new, empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a task to the resolver.
    ///
    /// Returns [`ResolutionError::MissingDependency`] if any declared
    /// dependency is not yet registered.
    pub fn add_task(&mut self, task: TaskNode) -> Result<(), ResolutionError> {
        for dep in &task.dependencies {
            if !self.tasks.contains_key(dep) {
                return Err(ResolutionError::MissingDependency(dep.clone()));
            }
        }
        self.tasks.insert(task.id.clone(), task);
        Ok(())
    }

    /// Resolves tasks into execution waves using Kahn's algorithm.
    ///
    /// Each wave is a `Vec<String>` of task IDs that can run concurrently.
    /// Returns [`ResolutionError::CyclicDependency`] if a cycle is detected.
    pub fn resolve(&self) -> Result<Vec<Vec<String>>, ResolutionError> {
        // Build in-degree map and adjacency list.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in self.tasks.keys() {
            in_degree.entry(id.as_str()).or_insert(0);
            dependents.entry(id.as_str()).or_default();
        }
        for task in self.tasks.values() {
            for dep in &task.dependencies {
                *in_degree.entry(task.id.as_str()).or_insert(0) += 1;
                dependents
                    .entry(dep.as_str())
                    .or_default()
                    .push(task.id.as_str());
            }
        }

        // Seed queue with zero-in-degree nodes.
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        // Sort for deterministic output.
        let mut queue_vec: Vec<&str> = queue.drain(..).collect();
        queue_vec.sort_unstable();
        queue.extend(queue_vec);

        let mut waves: Vec<Vec<String>> = Vec::new();
        let mut processed = 0usize;
        let total = self.tasks.len();

        // Process wave by wave.
        while !queue.is_empty() {
            let mut wave: Vec<&str> = queue.drain(..).collect();
            wave.sort_unstable();
            processed += wave.len();

            let mut next_queue: Vec<&str> = Vec::new();
            for &task_id in &wave {
                if let Some(deps_on_me) = dependents.get(task_id) {
                    for &dep_task in deps_on_me {
                        let deg = in_degree.entry(dep_task).or_insert(0);
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            next_queue.push(dep_task);
                        }
                    }
                }
            }

            waves.push(wave.iter().map(|s| s.to_string()).collect());
            next_queue.sort_unstable();
            queue.extend(next_queue);
        }

        if processed < total {
            // Cycle exists — collect the unresolved nodes.
            let resolved: HashSet<String> = waves.iter().flatten().cloned().collect();
            let cycle_members: Vec<String> = self
                .tasks
                .keys()
                .filter(|id| !resolved.contains(*id))
                .cloned()
                .collect();
            return Err(ResolutionError::CyclicDependency(cycle_members));
        }

        Ok(waves)
    }

    /// Returns the critical path as an ordered list of task IDs.
    ///
    /// The critical path is the sequence of tasks with the greatest total
    /// `estimated_duration_ms` from start to finish.
    pub fn critical_path(&self) -> Vec<String> {
        if self.tasks.is_empty() {
            return Vec::new();
        }

        // dp[id] = (longest_duration_to_complete_id, predecessor_id)
        let mut dp: HashMap<&str, (u64, Option<&str>)> = HashMap::new();

        // Topological order via resolve(); ignore errors here (return empty on cycle).
        let waves = match self.resolve() {
            Ok(w) => w,
            Err(_) => return Vec::new(),
        };

        for wave in &waves {
            for id in wave {
                let task = match self.tasks.get(id) {
                    Some(t) => t,
                    None => continue,
                };
                let (best_pred_cost, best_pred) = task
                    .dependencies
                    .iter()
                    .filter_map(|dep| dp.get(dep.as_str()).map(|&(c, _)| (c, dep.as_str())))
                    .max_by_key(|&(c, _)| c)
                    .unwrap_or((0, ""));

                let cost = best_pred_cost + task.estimated_duration_ms;
                let pred = if best_pred.is_empty() { None } else { Some(best_pred) };
                dp.insert(id.as_str(), (cost, pred));
            }
        }

        // Find end node with maximum cost.
        let end = match dp.iter().max_by_key(|(_, &(c, _))| c) {
            Some((&id, _)) => id,
            None => return Vec::new(),
        };

        // Reconstruct path by following predecessors.
        let mut path: Vec<String> = Vec::new();
        let mut current: Option<&str> = Some(end);
        while let Some(id) = current {
            path.push(id.to_string());
            current = dp.get(id).and_then(|(_, p)| *p);
        }
        path.reverse();
        path
    }

    /// Returns the parallelism factor: average tasks per wave divided by the
    /// maximum tasks in any single wave.  Returns `0.0` if there are no waves.
    pub fn parallelism_factor(&self) -> f64 {
        let waves = match self.resolve() {
            Ok(w) => w,
            Err(_) => return 0.0,
        };
        if waves.is_empty() {
            return 0.0;
        }
        let total: usize = waves.iter().map(|w| w.len()).sum();
        let max: usize = waves.iter().map(|w| w.len()).max().unwrap_or(1);
        let avg = total as f64 / waves.len() as f64;
        avg / max as f64
    }

    /// Returns resource conflicts: for each resource used by more than one task
    /// in the same wave, returns `(resource_name, [task_ids])`.
    pub fn resource_conflicts(&self) -> Vec<(String, Vec<String>)> {
        let waves = match self.resolve() {
            Ok(w) => w,
            Err(_) => return Vec::new(),
        };

        let mut result: Vec<(String, Vec<String>)> = Vec::new();

        for wave in &waves {
            // resource → tasks in this wave that need it
            let mut res_map: HashMap<&str, Vec<&str>> = HashMap::new();
            for task_id in wave {
                if let Some(task) = self.tasks.get(task_id) {
                    for res in &task.resources {
                        res_map.entry(res.as_str()).or_default().push(task_id.as_str());
                    }
                }
            }
            for (res, tasks) in res_map {
                if tasks.len() > 1 {
                    result.push((
                        res.to_string(),
                        tasks.iter().map(|s| s.to_string()).collect(),
                    ));
                }
            }
        }

        result
    }

    /// Returns the estimated total execution time given a set of waves.
    ///
    /// For each wave the cost is the maximum `estimated_duration_ms` of any
    /// task in that wave (tasks in a wave run concurrently).  The total is the
    /// sum of per-wave maxima.
    pub fn estimated_total_ms(&self, waves: &[Vec<String>]) -> u64 {
        waves
            .iter()
            .map(|wave| {
                wave.iter()
                    .filter_map(|id| self.tasks.get(id))
                    .map(|t| t.estimated_duration_ms)
                    .max()
                    .unwrap_or(0)
            })
            .sum()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn node(id: &str, deps: &[&str], duration: u64, resources: &[&str]) -> TaskNode {
        TaskNode {
            id: id.to_string(),
            name: id.to_string(),
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            estimated_duration_ms: duration,
            resources: resources.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_simple_chain() {
        let mut r = DependencyResolver::new();
        r.add_task(node("a", &[], 10, &[])).unwrap();
        r.add_task(node("b", &["a"], 20, &[])).unwrap();
        r.add_task(node("c", &["b"], 30, &[])).unwrap();
        let waves = r.resolve().unwrap();
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec!["a"]);
        assert_eq!(waves[1], vec!["b"]);
        assert_eq!(waves[2], vec!["c"]);
    }

    #[test]
    fn test_parallel_wave() {
        let mut r = DependencyResolver::new();
        r.add_task(node("a", &[], 10, &[])).unwrap();
        r.add_task(node("b", &[], 10, &[])).unwrap();
        r.add_task(node("c", &["a", "b"], 5, &[])).unwrap();
        let waves = r.resolve().unwrap();
        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0].len(), 2); // a and b in parallel
        assert_eq!(waves[1], vec!["c"]);
    }

    #[test]
    fn test_missing_dependency_error() {
        let mut r = DependencyResolver::new();
        let result = r.add_task(node("b", &["a"], 10, &[]));
        assert_eq!(result, Err(ResolutionError::MissingDependency("a".to_string())));
    }

    #[test]
    fn test_cycle_detection() {
        // Build a graph where b→c and c→b (cycle) by bypassing add_task validation.
        // We insert directly to simulate what would happen if validation were skipped.
        let mut r = DependencyResolver::new();
        // Insert a without deps.
        r.tasks.insert("a".to_string(), node("a", &[], 10, &[]));
        // Insert b depending on c (not yet added).
        r.tasks.insert("b".to_string(), node("b", &["c"], 10, &[]));
        // Insert c depending on b — cycle.
        r.tasks.insert("c".to_string(), node("c", &["b"], 10, &[]));
        let result = r.resolve();
        assert!(matches!(result, Err(ResolutionError::CyclicDependency(_))));
        if let Err(ResolutionError::CyclicDependency(members)) = result {
            assert!(members.contains(&"b".to_string()) || members.contains(&"c".to_string()));
        }
    }

    #[test]
    fn test_critical_path() {
        let mut r = DependencyResolver::new();
        // a(10) → c(30) is length 40
        // b(5)  → c(30) is length 35
        r.add_task(node("a", &[], 10, &[])).unwrap();
        r.add_task(node("b", &[], 5, &[])).unwrap();
        r.add_task(node("c", &["a", "b"], 30, &[])).unwrap();
        let path = r.critical_path();
        // Critical path should be a → c (total 40)
        assert_eq!(path, vec!["a", "c"]);
    }

    #[test]
    fn test_parallelism_factor_all_parallel() {
        let mut r = DependencyResolver::new();
        r.add_task(node("a", &[], 10, &[])).unwrap();
        r.add_task(node("b", &[], 10, &[])).unwrap();
        r.add_task(node("c", &[], 10, &[])).unwrap();
        // All in one wave: avg = 3, max = 3, factor = 1.0
        assert!((r.parallelism_factor() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parallelism_factor_chain() {
        let mut r = DependencyResolver::new();
        r.add_task(node("a", &[], 10, &[])).unwrap();
        r.add_task(node("b", &["a"], 10, &[])).unwrap();
        r.add_task(node("c", &["b"], 10, &[])).unwrap();
        // 3 waves of 1 each: avg=1, max=1, factor=1.0
        assert!((r.parallelism_factor() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_resource_conflicts() {
        let mut r = DependencyResolver::new();
        r.add_task(node("a", &[], 10, &["gpu"])).unwrap();
        r.add_task(node("b", &[], 10, &["gpu"])).unwrap();
        let conflicts = r.resource_conflicts();
        assert!(!conflicts.is_empty());
        let (res, tasks) = &conflicts[0];
        assert_eq!(res, "gpu");
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn test_estimated_total_ms() {
        let mut r = DependencyResolver::new();
        r.add_task(node("a", &[], 100, &[])).unwrap();
        r.add_task(node("b", &[], 200, &[])).unwrap();
        r.add_task(node("c", &["a", "b"], 50, &[])).unwrap();
        let waves = r.resolve().unwrap();
        // wave 0: max(100,200)=200, wave 1: 50 → total 250
        assert_eq!(r.estimated_total_ms(&waves), 250);
    }

    #[test]
    fn test_empty_resolver() {
        let r = DependencyResolver::new();
        let waves = r.resolve().unwrap();
        assert!(waves.is_empty());
        assert!(r.critical_path().is_empty());
        assert_eq!(r.parallelism_factor(), 0.0);
    }
}
