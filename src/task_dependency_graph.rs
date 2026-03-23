//! # Module: task_dependency_graph
//!
//! ## Responsibility
//! DAG-based task dependency system with cycle detection, topological sort,
//! critical path analysis, and parallel execution grouping.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Data structures ───────────────────────────────────────────────────────────

/// A task node in the dependency graph.
#[derive(Debug, Clone)]
pub struct TaskNode {
    /// Unique identifier for this task.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Estimated execution time in milliseconds.
    pub duration_estimate_ms: u64,
    /// IDs of tasks that must complete before this one starts.
    pub dependencies: Vec<String>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// A group of tasks that can all run in parallel.
#[derive(Debug, Clone)]
pub struct ExecutionGroup {
    /// Task IDs that belong to this group.
    pub tasks: Vec<String>,
    /// Zero-based level index (group 0 has no dependencies).
    pub group_id: usize,
}

/// The critical (longest) path through the DAG.
#[derive(Debug, Clone)]
pub struct CriticalPath {
    /// Ordered list of task IDs on the critical path.
    pub task_ids: Vec<String>,
    /// Sum of `duration_estimate_ms` along the critical path.
    pub total_duration_ms: u64,
}

/// Errors that can occur when working with the dependency graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyError {
    /// The graph contains a cycle; the string names the first node detected.
    CycleDetected(String),
    /// A task references a dependency ID that was never added.
    UnknownDependency(String),
    /// A task with the same ID was added twice.
    DuplicateTask(String),
}

impl std::fmt::Display for DependencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DependencyError::CycleDetected(id) => write!(f, "cycle detected at task: {id}"),
            DependencyError::UnknownDependency(id) => {
                write!(f, "unknown dependency: {id}")
            }
            DependencyError::DuplicateTask(id) => write!(f, "duplicate task id: {id}"),
        }
    }
}

impl std::error::Error for DependencyError {}

/// Full execution plan produced by [`TaskDependencyGraph::execution_plan`].
#[derive(Debug, Clone)]
pub struct DagExecutionPlan {
    /// BFS-level groups: tasks within each group are independent.
    pub groups: Vec<ExecutionGroup>,
    /// The longest weighted path through the DAG.
    pub critical_path: CriticalPath,
    /// Total number of tasks in the plan.
    pub total_tasks: usize,
    /// Estimated wall-clock duration (= critical path duration).
    pub estimated_duration_ms: u64,
    /// Average tasks per group (rough parallelism measure).
    pub parallelism_factor: f64,
}

// ── DFS colouring helpers ─────────────────────────────────────────────────────

#[derive(PartialEq, Eq, Clone, Copy)]
enum Color {
    White, // unvisited
    Gray,  // in current DFS stack
    Black, // fully processed
}

// ── TaskDependencyGraph ───────────────────────────────────────────────────────

/// A directed acyclic graph of tasks with dependency tracking.
#[derive(Debug, Default)]
pub struct TaskDependencyGraph {
    nodes: HashMap<String, TaskNode>,
}

impl TaskDependencyGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a task node to the graph.
    ///
    /// Returns [`DependencyError::DuplicateTask`] if a task with the same id
    /// already exists.
    pub fn add_task(&mut self, node: TaskNode) -> Result<(), DependencyError> {
        if self.nodes.contains_key(&node.id) {
            return Err(DependencyError::DuplicateTask(node.id.clone()));
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Remove a task from the graph by id.  No-op if the id is unknown.
    pub fn remove_task(&mut self, id: &str) {
        self.nodes.remove(id);
        // Remove this id from other tasks' dependency lists.
        for node in self.nodes.values_mut() {
            node.dependencies.retain(|dep| dep != id);
        }
    }

    /// Add a dependency edge: `task_id` depends on `depends_on`.
    ///
    /// Returns an error if either id is unknown.
    pub fn add_dependency(
        &mut self,
        task_id: &str,
        depends_on: &str,
    ) -> Result<(), DependencyError> {
        if !self.nodes.contains_key(depends_on) {
            return Err(DependencyError::UnknownDependency(depends_on.to_string()));
        }
        let node = self
            .nodes
            .get_mut(task_id)
            .ok_or_else(|| DependencyError::UnknownDependency(task_id.to_string()))?;
        if !node.dependencies.contains(&depends_on.to_string()) {
            node.dependencies.push(depends_on.to_string());
        }
        Ok(())
    }

    /// Validate that the graph is acyclic and all dependency references exist.
    ///
    /// Uses DFS colouring (white / gray / black).
    pub fn validate(&self) -> Result<(), DependencyError> {
        // First check all referenced deps exist.
        for node in self.nodes.values() {
            for dep in &node.dependencies {
                if !self.nodes.contains_key(dep) {
                    return Err(DependencyError::UnknownDependency(dep.clone()));
                }
            }
        }

        let mut colors: HashMap<&str, Color> = HashMap::new();
        for id in self.nodes.keys() {
            colors.insert(id.as_str(), Color::White);
        }

        for id in self.nodes.keys() {
            if colors[id.as_str()] == Color::White {
                Self::dfs_visit(id, &self.nodes, &mut colors)?;
            }
        }
        Ok(())
    }

    fn dfs_visit<'a>(
        id: &'a str,
        nodes: &'a HashMap<String, TaskNode>,
        colors: &mut HashMap<&'a str, Color>,
    ) -> Result<(), DependencyError> {
        colors.insert(id, Color::Gray);
        if let Some(node) = nodes.get(id) {
            for dep in &node.dependencies {
                let dep_str = dep.as_str();
                match colors.get(dep_str).copied().unwrap_or(Color::White) {
                    Color::Gray => {
                        return Err(DependencyError::CycleDetected(dep.clone()));
                    }
                    Color::White => {
                        Self::dfs_visit(dep_str, nodes, colors)?;
                    }
                    Color::Black => {}
                }
            }
        }
        colors.insert(id, Color::Black);
        Ok(())
    }

    /// Topological sort using Kahn's algorithm.
    ///
    /// Returns tasks ordered so that every dependency appears before the
    /// tasks that depend on it.
    pub fn topological_sort(&self) -> Result<Vec<String>, DependencyError> {
        self.validate()?;

        // in-degree for each task
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for id in self.nodes.keys() {
            in_degree.entry(id.as_str()).or_insert(0);
        }
        for node in self.nodes.values() {
            for dep in &node.dependencies {
                // dep must complete before node, so node is downstream of dep.
                // We track in_degree of node.
                *in_degree.entry(node.id.as_str()).or_insert(0) += 0; // ensure present
                let _ = dep; // dep is upstream; nothing to do here
            }
        }

        // Recompute properly: in-degree = number of dependencies.
        let mut in_deg: HashMap<&str, usize> = self
            .nodes
            .keys()
            .map(|k| (k.as_str(), 0usize))
            .collect();
        // Build adjacency: dep -> [tasks that depend on dep]
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in self.nodes.values() {
            for dep in &node.dependencies {
                adj.entry(dep.as_str()).or_default().push(node.id.as_str());
                *in_deg.entry(node.id.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_deg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        // Stable sort for determinism
        let mut q_vec: Vec<&str> = queue.drain(..).collect();
        q_vec.sort_unstable();
        queue.extend(q_vec);

        let mut result = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            result.push(id.to_string());
            if let Some(children) = adj.get(id) {
                let mut next: Vec<&str> = Vec::new();
                for &child in children {
                    let deg = in_deg.entry(child).or_insert(0);
                    *deg -= 1;
                    if *deg == 0 {
                        next.push(child);
                    }
                }
                next.sort_unstable();
                queue.extend(next);
            }
        }

        if result.len() != self.nodes.len() {
            return Err(DependencyError::CycleDetected(
                "cycle detected during Kahn's sort".to_string(),
            ));
        }
        Ok(result)
    }

    /// Build a full execution plan: BFS level groups + critical path.
    pub fn execution_plan(&self) -> Result<DagExecutionPlan, DependencyError> {
        self.validate()?;

        // BFS level assignment
        let mut level: HashMap<&str, usize> = HashMap::new();
        // in-degree
        let mut in_deg: HashMap<&str, usize> =
            self.nodes.keys().map(|k| (k.as_str(), 0usize)).collect();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in self.nodes.values() {
            for dep in &node.dependencies {
                adj.entry(dep.as_str()).or_default().push(node.id.as_str());
                *in_deg.entry(node.id.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_deg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        for &id in &queue {
            level.insert(id, 0);
        }

        let mut remaining_in_deg = in_deg.clone();
        let mut bfs_queue: VecDeque<&str> = queue.drain(..).collect();
        let mut visit_order: Vec<&str> = Vec::new();

        while let Some(id) = bfs_queue.pop_front() {
            visit_order.push(id);
            if let Some(children) = adj.get(id) {
                for &child in children {
                    let new_level = level[id] + 1;
                    let entry = level.entry(child).or_insert(0);
                    if new_level > *entry {
                        *entry = new_level;
                    }
                    let deg = remaining_in_deg.entry(child).or_insert(0);
                    *deg -= 1;
                    if *deg == 0 {
                        bfs_queue.push_back(child);
                    }
                }
            }
        }

        // Group by level
        let max_level = level.values().copied().max().unwrap_or(0);
        let mut groups: Vec<ExecutionGroup> = (0..=max_level)
            .map(|g| ExecutionGroup {
                tasks: Vec::new(),
                group_id: g,
            })
            .collect();
        for (id, &lv) in &level {
            groups[lv].tasks.push((*id).to_string());
        }
        for g in &mut groups {
            g.tasks.sort();
        }

        let total_tasks = self.nodes.len();
        let cp = self.critical_path()?;
        let estimated_duration_ms = cp.total_duration_ms;
        let parallelism_factor = if groups.is_empty() {
            1.0
        } else {
            total_tasks as f64 / groups.len() as f64
        };

        Ok(DagExecutionPlan {
            groups,
            critical_path: cp,
            total_tasks,
            estimated_duration_ms,
            parallelism_factor,
        })
    }

    /// Find the critical (longest weighted) path via dynamic programming.
    pub fn critical_path(&self) -> Result<CriticalPath, DependencyError> {
        let topo = self.topological_sort()?;

        // dp[id] = (longest duration ending at id, predecessor id)
        let mut dp: HashMap<&str, (u64, Option<&str>)> = HashMap::new();
        for id in self.nodes.keys() {
            dp.insert(id.as_str(), (0, None));
        }

        for id_str in &topo {
            let id = id_str.as_str();
            let node = &self.nodes[id];
            let best = node
                .dependencies
                .iter()
                .map(|dep| {
                    let dep_s = dep.as_str();
                    let (dep_dist, _) = dp[dep_s];
                    (dep_dist, dep_s)
                })
                .max_by_key(|&(d, _)| d);

            let (dist, pred) = match best {
                Some((d, p)) => (d + node.duration_estimate_ms, Some(p)),
                None => (node.duration_estimate_ms, None),
            };
            dp.insert(id, (dist, pred));
        }

        // Find end node of critical path
        let (end_id, &(total_duration_ms, _)) = dp
            .iter()
            .max_by_key(|(_, &(d, _))| d)
            .map(|(&id, v)| (id, v))
            .unwrap_or(("", &(0, None)));

        // Trace back
        let mut path: Vec<String> = Vec::new();
        let mut cur: Option<&str> = if end_id.is_empty() { None } else { Some(end_id) };
        while let Some(id) = cur {
            path.push(id.to_string());
            cur = dp[id].1;
        }
        path.reverse();

        Ok(CriticalPath {
            task_ids: path,
            total_duration_ms,
        })
    }

    /// Return the set of all transitive dependencies of `task_id`.
    pub fn transitive_dependencies(&self, task_id: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        if let Some(node) = self.nodes.get(task_id) {
            for dep in &node.dependencies {
                stack.push(dep.clone());
            }
        }
        while let Some(id) = stack.pop() {
            if visited.insert(id.clone()) {
                if let Some(node) = self.nodes.get(&id) {
                    for dep in &node.dependencies {
                        stack.push(dep.clone());
                    }
                }
            }
        }
        visited
    }

    /// Return all tasks that directly or indirectly depend on `task_id`.
    pub fn dependents(&self, task_id: &str) -> Vec<String> {
        self.nodes
            .values()
            .filter(|n| n.dependencies.contains(&task_id.to_string()))
            .map(|n| n.id.clone())
            .collect()
    }

    /// Return tasks whose dependencies are all in `completed`.
    ///
    /// Tasks that are already in `completed` are excluded from the result.
    pub fn ready_tasks(&self, completed: &HashSet<String>) -> Vec<String> {
        self.nodes
            .values()
            .filter(|n| {
                !completed.contains(&n.id)
                    && n.dependencies.iter().all(|dep| completed.contains(dep))
            })
            .map(|n| n.id.clone())
            .collect()
    }
}
