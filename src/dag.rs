//! # Module: dag
//!
//! ## Responsibility
//! DAG-based workflow execution with topological sort and parallel execution.
//!
//! Provides:
//! - [`NodeId`]: newtype wrapping `String` for type-safe node identification.
//! - [`DagNode`]: a node with a name, dependencies, and an arbitrary payload.
//! - [`DagEdge`]: a directed edge between two nodes with an optional label.
//! - [`Dag`]: the graph structure with cycle detection and topological ordering.
//! - [`DagExecutor`]: executes nodes in dependency order, with parallel variants.
//!
//! ## Guarantees
//! - [`Dag::add_edge`] rejects edges that would create a cycle.
//! - [`Dag::topological_sort`] uses Kahn's algorithm; returns [`DagError::Cycle`]
//!   if a cycle is detected.
//! - No `.unwrap()` or `.expect()` in production paths.

use std::{
    any::Any,
    collections::{HashMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
};

use tokio::task::JoinSet;

// ── NodeId ────────────────────────────────────────────────────────────────────

/// Newtype wrapping a [`String`] that uniquely identifies a node in a [`Dag`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

impl NodeId {
    /// Create a new [`NodeId`] from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        NodeId(id.into())
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── DagError ──────────────────────────────────────────────────────────────────

/// Errors produced by [`Dag`] and [`DagExecutor`] operations.
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    /// The graph contains a cycle; topological ordering is impossible.
    #[error("DAG contains a cycle")]
    Cycle,
    /// A referenced node does not exist in the graph.
    #[error("node not found: {0}")]
    NodeNotFound(String),
    /// Attempt to add a node whose id is already present.
    #[error("duplicate node: {0}")]
    DuplicateNode(String),
}

// ── DagNode ───────────────────────────────────────────────────────────────────

/// A single node in a [`Dag`].
pub struct DagNode {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// Human-readable name.
    pub name: String,
    /// Ids of nodes that must complete before this one can run.
    pub dependencies: Vec<NodeId>,
    /// Arbitrary payload attached to this node.
    pub payload: Box<dyn Any + Send + Sync>,
}

impl std::fmt::Debug for DagNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DagNode")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

// ── DagEdge ───────────────────────────────────────────────────────────────────

/// A directed edge in a [`Dag`] from `from` to `to`.
#[derive(Debug, Clone)]
pub struct DagEdge {
    /// Source node.
    pub from: NodeId,
    /// Target (dependent) node.
    pub to: NodeId,
    /// Optional human-readable label for this edge.
    pub label: Option<String>,
}

// ── Dag ───────────────────────────────────────────────────────────────────────

/// Directed Acyclic Graph used for workflow execution.
///
/// Nodes are added via [`Dag::add_node`] and directed edges (dependencies) are
/// added via [`Dag::add_edge`].  Cycle detection happens eagerly on
/// [`add_edge`](Dag::add_edge).
#[derive(Debug, Default)]
pub struct Dag {
    /// All nodes indexed by their [`NodeId`].
    pub nodes: HashMap<NodeId, DagNode>,
    /// All directed edges in the graph.
    pub edges: Vec<DagEdge>,
}

impl Dag {
    /// Create a new empty [`Dag`].
    pub fn new() -> Self {
        Dag::default()
    }

    /// Add a node to the graph.
    ///
    /// Returns [`DagError::DuplicateNode`] if a node with the same `id` already
    /// exists.
    pub fn add_node(&mut self, id: &str, name: &str) -> Result<NodeId, DagError> {
        let node_id = NodeId::new(id);
        if self.nodes.contains_key(&node_id) {
            return Err(DagError::DuplicateNode(id.to_owned()));
        }
        let node = DagNode {
            id: node_id.clone(),
            name: name.to_owned(),
            dependencies: Vec::new(),
            payload: Box::new(()),
        };
        self.nodes.insert(node_id.clone(), node);
        Ok(node_id)
    }

    /// Add a directed dependency edge `from` → `to`.
    ///
    /// Returns [`DagError::NodeNotFound`] if either node is unknown.
    /// Returns [`DagError::Cycle`] if adding the edge would create a cycle.
    pub fn add_edge(&mut self, from: &str, to: &str) -> Result<(), DagError> {
        let from_id = NodeId::new(from);
        let to_id = NodeId::new(to);

        if !self.nodes.contains_key(&from_id) {
            return Err(DagError::NodeNotFound(from.to_owned()));
        }
        if !self.nodes.contains_key(&to_id) {
            return Err(DagError::NodeNotFound(to.to_owned()));
        }

        // Temporarily add the edge and check for cycles.
        self.edges.push(DagEdge {
            from: from_id.clone(),
            to: to_id.clone(),
            label: None,
        });

        if !self.is_dag() {
            self.edges.pop();
            return Err(DagError::Cycle);
        }

        // Also record in the node's dependency list.
        if let Some(node) = self.nodes.get_mut(&to_id) {
            if !node.dependencies.contains(&from_id) {
                node.dependencies.push(from_id);
            }
        }

        Ok(())
    }

    /// Returns `true` if the graph contains no cycles (i.e., it is a valid DAG).
    pub fn is_dag(&self) -> bool {
        self.topological_sort().is_ok()
    }

    /// Return all nodes in topological order using Kahn's algorithm.
    ///
    /// Returns [`DagError::Cycle`] if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, DagError> {
        // Build adjacency list and in-degree map.
        let mut in_degree: HashMap<&NodeId, usize> =
            self.nodes.keys().map(|id| (id, 0)).collect();
        let mut adj: HashMap<&NodeId, Vec<&NodeId>> = HashMap::new();

        for edge in &self.edges {
            *in_degree.entry(&edge.to).or_insert(0) += 1;
            adj.entry(&edge.from).or_default().push(&edge.to);
        }

        let mut queue: VecDeque<&NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut result: Vec<NodeId> = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    let deg = in_degree.entry(neighbor).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if result.len() == self.nodes.len() {
            Ok(result)
        } else {
            Err(DagError::Cycle)
        }
    }

    /// Return all ancestors of `node` — every node that must complete before it.
    ///
    /// Returns an empty set if `node` is not found.
    pub fn ancestors(&self, node: &str) -> HashSet<NodeId> {
        let target = NodeId::new(node);
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from nodes that have a direct edge to `target`.
        for edge in &self.edges {
            if edge.to == target {
                queue.push_back(edge.from.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            for edge in &self.edges {
                if edge.to == current && !visited.contains(&edge.from) {
                    queue.push_back(edge.from.clone());
                }
            }
        }

        visited
    }

    /// Return all descendants of `node` — every node that depends on it (directly or transitively).
    ///
    /// Returns an empty set if `node` is not found.
    pub fn descendants(&self, node: &str) -> HashSet<NodeId> {
        let source = NodeId::new(node);
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        for edge in &self.edges {
            if edge.from == source {
                queue.push_back(edge.to.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            for edge in &self.edges {
                if edge.from == current && !visited.contains(&edge.to) {
                    queue.push_back(edge.to.clone());
                }
            }
        }

        visited
    }

    /// Render the graph as a Graphviz DOT string.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n");
        for node in self.nodes.values() {
            out.push_str(&format!(
                "    \"{}\" [label=\"{}\"];\n",
                node.id.0, node.name
            ));
        }
        for edge in &self.edges {
            if let Some(ref label) = edge.label {
                out.push_str(&format!(
                    "    \"{}\" -> \"{}\" [label=\"{}\"];\n",
                    edge.from.0, edge.to.0, label
                ));
            } else {
                out.push_str(&format!(
                    "    \"{}\" -> \"{}\";\n",
                    edge.from.0, edge.to.0
                ));
            }
        }
        out.push('}');
        out
    }
}

// ── BoxFuture helper ──────────────────────────────────────────────────────────

/// A heap-allocated, pinned, `Send` future that resolves to `T`.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ── DagExecutor ───────────────────────────────────────────────────────────────

/// Executes nodes of a [`Dag`] in dependency order.
///
/// Two execution modes are provided:
/// - [`DagExecutor::execute`]: synchronous runner wrapped in `spawn_blocking`.
/// - [`DagExecutor::execute_parallel`]: fully async runner; concurrently
///   dispatches nodes whose dependencies are all satisfied.
#[derive(Debug, Default)]
pub struct DagExecutor;

impl DagExecutor {
    /// Create a new [`DagExecutor`].
    pub fn new() -> Self {
        DagExecutor
    }

    /// Execute all nodes in topological order using a synchronous `runner`.
    ///
    /// Each node is dispatched via `tokio::task::spawn_blocking`.
    /// Returns a map of `NodeId → Ok(output) | Err(message)`.
    pub async fn execute<F>(
        &self,
        dag: &Dag,
        runner: F,
    ) -> HashMap<NodeId, Result<String, String>>
    where
        F: Fn(&NodeId) -> String + Clone + Send + 'static,
    {
        let order = match dag.topological_sort() {
            Ok(o) => o,
            Err(e) => {
                return dag
                    .nodes
                    .keys()
                    .map(|id| (id.clone(), Err(e.to_string())))
                    .collect();
            }
        };

        let mut results: HashMap<NodeId, Result<String, String>> = HashMap::new();

        for node_id in order {
            let id_clone = node_id.clone();
            let runner_clone = runner.clone();
            let outcome = tokio::task::spawn_blocking(move || runner_clone(&id_clone)).await;
            let result = match outcome {
                Ok(output) => Ok(output),
                Err(e) => Err(e.to_string()),
            };
            results.insert(node_id, result);
        }

        results
    }

    /// Execute nodes concurrently whenever their dependencies are all satisfied.
    ///
    /// `runner` is an `Arc`-wrapped async closure that accepts a [`NodeId`]
    /// reference and returns a [`BoxFuture`] resolving to a `String`.
    pub async fn execute_parallel(
        &self,
        dag: &Dag,
        runner: Arc<dyn Fn(&NodeId) -> BoxFuture<'static, String> + Send + Sync>,
    ) -> HashMap<NodeId, String> {
        let order = match dag.topological_sort() {
            Ok(o) => o,
            Err(_) => return HashMap::new(),
        };

        // Build a set of all node ids in topological order for tracking completion.
        let mut completed: HashSet<NodeId> = HashSet::new();
        let mut results: HashMap<NodeId, String> = HashMap::new();
        // remaining nodes not yet dispatched
        let mut pending: Vec<NodeId> = order;

        while !pending.is_empty() {
            // Find all nodes whose dependencies are satisfied.
            let (ready, not_ready): (Vec<NodeId>, Vec<NodeId>) =
                pending.into_iter().partition(|id| {
                    dag.nodes
                        .get(id)
                        .map(|n| n.dependencies.iter().all(|dep| completed.contains(dep)))
                        .unwrap_or(false)
                });

            if ready.is_empty() {
                // Should not happen in a valid DAG.
                break;
            }

            let mut set: JoinSet<(NodeId, String)> = JoinSet::new();
            for node_id in ready {
                let r = Arc::clone(&runner);
                let id = node_id.clone();
                // Safety: the future is 'static because NodeId is owned.
                let fut = r(&id);
                set.spawn(async move { (id, fut.await) });
            }

            while let Some(outcome) = set.join_next().await {
                if let Ok((id, output)) = outcome {
                    completed.insert(id.clone());
                    results.insert(id, output);
                }
            }

            pending = not_ready;
        }

        results
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn build_chain() -> Dag {
        let mut dag = Dag::new();
        dag.add_node("a", "Node A").unwrap();
        dag.add_node("b", "Node B").unwrap();
        dag.add_node("c", "Node C").unwrap();
        dag.add_edge("a", "b").unwrap();
        dag.add_edge("b", "c").unwrap();
        dag
    }

    #[test]
    fn simple_chain_topological_order() {
        let dag = build_chain();
        let order = dag.topological_sort().unwrap();
        // a must come before b, b before c
        let pos: HashMap<&NodeId, usize> =
            order.iter().enumerate().map(|(i, id)| (id, i)).collect();
        let a = NodeId::new("a");
        let b = NodeId::new("b");
        let c = NodeId::new("c");
        assert!(pos[&a] < pos[&b]);
        assert!(pos[&b] < pos[&c]);
    }

    #[test]
    fn diamond_dag_topological_order() {
        let mut dag = Dag::new();
        dag.add_node("a", "A").unwrap();
        dag.add_node("b", "B").unwrap();
        dag.add_node("c", "C").unwrap();
        dag.add_node("d", "D").unwrap();
        dag.add_edge("a", "b").unwrap();
        dag.add_edge("a", "c").unwrap();
        dag.add_edge("b", "d").unwrap();
        dag.add_edge("c", "d").unwrap();

        assert!(dag.is_dag());
        let order = dag.topological_sort().unwrap();
        let pos: HashMap<&NodeId, usize> =
            order.iter().enumerate().map(|(i, id)| (id, i)).collect();
        let a = NodeId::new("a");
        let b = NodeId::new("b");
        let c = NodeId::new("c");
        let d = NodeId::new("d");
        assert!(pos[&a] < pos[&b]);
        assert!(pos[&a] < pos[&c]);
        assert!(pos[&b] < pos[&d]);
        assert!(pos[&c] < pos[&d]);
    }

    #[test]
    fn cycle_detection() {
        let mut dag = Dag::new();
        dag.add_node("x", "X").unwrap();
        dag.add_node("y", "Y").unwrap();
        dag.add_edge("x", "y").unwrap();
        // Adding y→x would create a cycle
        let err = dag.add_edge("y", "x");
        assert!(matches!(err, Err(DagError::Cycle)));
        // Graph is still a valid DAG after rejection
        assert!(dag.is_dag());
    }

    #[test]
    fn duplicate_node_error() {
        let mut dag = Dag::new();
        dag.add_node("a", "A").unwrap();
        let err = dag.add_node("a", "A2");
        assert!(matches!(err, Err(DagError::DuplicateNode(_))));
    }

    #[test]
    fn ancestors_and_descendants() {
        let dag = build_chain();
        let ancs = dag.ancestors("c");
        assert!(ancs.contains(&NodeId::new("a")));
        assert!(ancs.contains(&NodeId::new("b")));
        assert!(!ancs.contains(&NodeId::new("c")));

        let descs = dag.descendants("a");
        assert!(descs.contains(&NodeId::new("b")));
        assert!(descs.contains(&NodeId::new("c")));
        assert!(!descs.contains(&NodeId::new("a")));
    }

    #[test]
    fn to_dot_contains_nodes_and_edges() {
        let dag = build_chain();
        let dot = dag.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("\"a\""));
        assert!(dot.contains("\"b\""));
        assert!(dot.contains("->"));
    }

    #[tokio::test]
    async fn execute_synchronous_runner() {
        let dag = build_chain();
        let executor = DagExecutor::new();
        let results = executor
            .execute(&dag, |id| format!("ran:{}", id))
            .await;
        assert_eq!(results.len(), 3);
        for v in results.values() {
            assert!(v.is_ok());
        }
    }

    #[tokio::test]
    async fn execute_parallel_runner() {
        let dag = build_chain();
        let executor = DagExecutor::new();
        let runner: Arc<dyn Fn(&NodeId) -> BoxFuture<'static, String> + Send + Sync> =
            Arc::new(|id: &NodeId| {
                let s = format!("async:{}", id);
                Box::pin(async move { s })
            });
        let results = executor.execute_parallel(&dag, runner).await;
        assert_eq!(results.len(), 3);
    }
}
