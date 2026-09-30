//! # Knowledge Graph
//!
//! In-memory directed graph for storing and querying entities and their
//! relationships. Supports BFS shortest-path, subgraph extraction, substring
//! search, and Graphviz DOT export.
//!
//! ## Example
//!
//! ```rust
//! use llm_agent_runtime::knowledge::{KnowledgeGraph, Entity, Relation};
//!
//! let mut g = KnowledgeGraph::new();
//! g.add_entity(Entity::new("alice", "Alice"));
//! g.add_entity(Entity::new("bob", "Bob"));
//! g.add_relation(Relation::new("alice", "bob", "knows", 1.0));
//!
//! let path = g.shortest_path("alice", "bob");
//! assert_eq!(path, Some(vec!["alice".to_string(), "bob".to_string()]));
//!
//! let dot = g.to_dot();
//! assert!(dot.contains("Alice"));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ─── Entity ──────────────────────────────────────────────────────────────────

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entity {
    /// Unique identifier for this entity.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Arbitrary key-value properties attached to this entity.
    pub properties: HashMap<String, String>,
}

impl Entity {
    /// Create a new entity with no properties.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            properties: HashMap::new(),
        }
    }

    /// Attach a property to this entity (builder pattern).
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

// ─── Relation ────────────────────────────────────────────────────────────────

/// A directed edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Relation {
    /// ID of the source entity.
    pub from_id: String,
    /// ID of the target entity.
    pub to_id: String,
    /// Relationship kind (e.g. `"knows"`, `"owns"`, `"part_of"`).
    pub kind: String,
    /// Edge weight — higher is stronger/more confident.
    pub weight: f32,
}

impl Relation {
    /// Create a new relation.
    pub fn new(
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        kind: impl Into<String>,
        weight: f32,
    ) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            kind: kind.into(),
            weight,
        }
    }
}

// ─── KnowledgeGraph ──────────────────────────────────────────────────────────

/// In-memory directed knowledge graph.
///
/// Entities are stored in a `HashMap` keyed by their ID.
/// Relations are stored as a separate list and also as an adjacency map for
/// O(1) neighbour lookup.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    entities: HashMap<String, Entity>,
    relations: Vec<Relation>,
    /// Adjacency list: `from_id → Vec<to_id>` for fast BFS.
    adjacency: HashMap<String, Vec<String>>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Mutations ────────────────────────────────────────────────────────────

    /// Add or replace an entity.
    ///
    /// If an entity with the same ID already exists it is overwritten.
    pub fn add_entity(&mut self, entity: Entity) {
        // Ensure adjacency entry exists even if no edges are added.
        self.adjacency.entry(entity.id.clone()).or_default();
        self.entities.insert(entity.id.clone(), entity);
    }

    /// Add a directed relation.
    ///
    /// Both `from_id` and `to_id` should name existing entities, but this is
    /// not enforced — dangling edges are allowed for flexibility.
    pub fn add_relation(&mut self, relation: Relation) {
        self.adjacency
            .entry(relation.from_id.clone())
            .or_default()
            .push(relation.to_id.clone());
        self.relations.push(relation);
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Return a reference to the entity with the given ID, or `None`.
    pub fn entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Return all entities as a slice.
    pub fn all_entities(&self) -> Vec<&Entity> {
        self.entities.values().collect()
    }

    /// Return all relations.
    pub fn all_relations(&self) -> &[Relation] {
        &self.relations
    }

    /// Return the direct outgoing neighbours of `id`.
    pub fn neighbors(&self, id: &str) -> Vec<&Entity> {
        self.adjacency
            .get(id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|nid| self.entities.get(nid.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// BFS shortest path from `from` to `to`.
    ///
    /// Returns `Some(Vec<id>)` with the node IDs along the shortest path
    /// (inclusive), or `None` if no path exists.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<Vec<String>> = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back(vec![from.to_string()]);

        while let Some(path) = queue.pop_front() {
            let current = path.last().expect("path is never empty");
            let empty = Vec::new();
            let neighbors = self.adjacency.get(current.as_str()).unwrap_or(&empty);

            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());

                    if neighbor == to {
                        return Some(new_path);
                    }

                    visited.insert(neighbor.clone());
                    queue.push_back(new_path);
                }
            }
        }

        None
    }

    /// Substring search on entity labels and property values.
    ///
    /// Returns all entities where `query` appears (case-insensitive) in the
    /// `label` or in any property value.
    pub fn search(&self, query: &str) -> Vec<&Entity> {
        let q = query.to_ascii_lowercase();
        self.entities
            .values()
            .filter(|e| {
                e.label.to_ascii_lowercase().contains(&q)
                    || e.properties
                        .values()
                        .any(|v| v.to_ascii_lowercase().contains(&q))
            })
            .collect()
    }

    /// Extract a subgraph containing `root` and all nodes reachable within
    /// `depth` hops via BFS.
    ///
    /// Returns a new `KnowledgeGraph` with only those entities and the
    /// relations between them.
    pub fn subgraph(&self, root: &str, depth: usize) -> KnowledgeGraph {
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier: VecDeque<(String, usize)> = VecDeque::new();

        if self.entities.contains_key(root) {
            visited.insert(root.to_string());
            frontier.push_back((root.to_string(), 0));
        }

        while let Some((node, d)) = frontier.pop_front() {
            if d >= depth {
                continue;
            }
            let empty = Vec::new();
            let neighbors = self.adjacency.get(node.as_str()).unwrap_or(&empty);
            for n in neighbors {
                if !visited.contains(n) {
                    visited.insert(n.clone());
                    frontier.push_back((n.clone(), d + 1));
                }
            }
        }

        let mut sub = KnowledgeGraph::new();
        for id in &visited {
            if let Some(entity) = self.entities.get(id.as_str()) {
                sub.add_entity(entity.clone());
            }
        }
        for rel in &self.relations {
            if visited.contains(&rel.from_id) && visited.contains(&rel.to_id) {
                sub.add_relation(rel.clone());
            }
        }

        sub
    }

    /// Export the graph in Graphviz DOT format.
    ///
    /// Useful for visualising the graph with `dot -Tsvg`.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph KnowledgeGraph {\n");
        out.push_str("  rankdir=LR;\n");
        out.push_str("  node [shape=box, style=filled, fillcolor=lightblue];\n\n");

        // Entity nodes
        for entity in self.entities.values() {
            let escaped_label = entity.label.replace('"', "\\\"");
            let escaped_id = entity.id.replace('"', "\\\"");
            let props: String = entity
                .properties
                .iter()
                .map(|(k, v)| format!("\\n{}={}", k, v))
                .collect::<Vec<_>>()
                .join("");
            out.push_str(&format!(
                "  \"{}\" [label=\"{}{}\"];\n",
                escaped_id, escaped_label, props
            ));
        }

        out.push('\n');

        // Relation edges
        for rel in &self.relations {
            let from = rel.from_id.replace('"', "\\\"");
            let to = rel.to_id.replace('"', "\\\"");
            let kind = rel.kind.replace('"', "\\\"");
            out.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{} ({:.2})\"];\n",
                from, to, kind, rel.weight
            ));
        }

        out.push_str("}\n");
        out
    }

    /// Return the number of entities in the graph.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Return the number of relations in the graph.
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample_graph() -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();
        g.add_entity(Entity::new("a", "Alice").with_property("role", "engineer"));
        g.add_entity(Entity::new("b", "Bob").with_property("role", "manager"));
        g.add_entity(Entity::new("c", "Carol").with_property("dept", "research"));
        g.add_entity(Entity::new("d", "Dave"));
        g.add_relation(Relation::new("a", "b", "reports_to", 1.0));
        g.add_relation(Relation::new("b", "c", "collaborates", 0.8));
        g.add_relation(Relation::new("a", "c", "knows", 0.5));
        g.add_relation(Relation::new("c", "d", "manages", 1.0));
        g
    }

    #[test]
    fn add_entity_and_lookup() {
        let mut g = KnowledgeGraph::new();
        g.add_entity(Entity::new("x", "X-Node"));
        assert_eq!(g.entity("x").unwrap().label, "X-Node");
        assert!(g.entity("missing").is_none());
    }

    #[test]
    fn add_relation_creates_adjacency() {
        let g = sample_graph();
        let neighbors = g.neighbors("a");
        let ids: Vec<&str> = neighbors.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn neighbors_empty_for_leaf_node() {
        let g = sample_graph();
        let ns = g.neighbors("d");
        assert!(ns.is_empty());
    }

    #[test]
    fn neighbors_unknown_node_empty() {
        let g = KnowledgeGraph::new();
        assert!(g.neighbors("no_such_node").is_empty());
    }

    #[test]
    fn shortest_path_direct() {
        let g = sample_graph();
        let path = g.shortest_path("a", "b").unwrap();
        assert_eq!(path, vec!["a", "b"]);
    }

    #[test]
    fn shortest_path_indirect() {
        let g = sample_graph();
        let path = g.shortest_path("a", "d").unwrap();
        // a→c→d or a→b→c→d; BFS guarantees shortest (2 hops: a→c→d)
        assert_eq!(path.first().unwrap(), "a");
        assert_eq!(path.last().unwrap(), "d");
        assert!(path.len() <= 4);
    }

    #[test]
    fn shortest_path_self() {
        let g = sample_graph();
        let path = g.shortest_path("a", "a").unwrap();
        assert_eq!(path, vec!["a"]);
    }

    #[test]
    fn shortest_path_no_path() {
        let mut g = KnowledgeGraph::new();
        g.add_entity(Entity::new("x", "X"));
        g.add_entity(Entity::new("y", "Y"));
        // No edges — no path.
        assert!(g.shortest_path("x", "y").is_none());
    }

    #[test]
    fn shortest_path_unknown_nodes() {
        let g = sample_graph();
        assert!(g.shortest_path("a", "zzz").is_none());
        assert!(g.shortest_path("zzz", "a").is_none());
    }

    #[test]
    fn search_by_label() {
        let g = sample_graph();
        let results = g.search("alice");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn search_by_property_value() {
        let g = sample_graph();
        let results = g.search("research");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "c");
    }

    #[test]
    fn search_case_insensitive() {
        let g = sample_graph();
        let results = g.search("ALICE");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_no_matches() {
        let g = sample_graph();
        let results = g.search("xyzzy");
        assert!(results.is_empty());
    }

    #[test]
    fn search_multiple_matches() {
        let g = sample_graph();
        // "engineer" and "manager" both contain "r" — but "research" too.
        // Search for "er" matches Alice (engineer), Bob (manager), Carol (research).
        let results = g.search("er");
        assert!(results.len() >= 2);
    }

    #[test]
    fn subgraph_depth_zero() {
        let g = sample_graph();
        let sub = g.subgraph("a", 0);
        assert_eq!(sub.entity_count(), 1);
        assert_eq!(sub.entity("a").unwrap().label, "Alice");
        assert_eq!(sub.relation_count(), 0);
    }

    #[test]
    fn subgraph_depth_one() {
        let g = sample_graph();
        let sub = g.subgraph("a", 1);
        // a's direct neighbors are b and c
        assert!(sub.entity("a").is_some());
        assert!(sub.entity("b").is_some());
        assert!(sub.entity("c").is_some());
        assert!(sub.entity("d").is_none());
    }

    #[test]
    fn subgraph_depth_two() {
        let g = sample_graph();
        let sub = g.subgraph("a", 2);
        // Should reach d via a→c→d
        assert!(sub.entity("d").is_some());
    }

    #[test]
    fn subgraph_includes_relevant_relations() {
        let g = sample_graph();
        let sub = g.subgraph("a", 1);
        let rels = sub.all_relations();
        // a→b and a→c should be included; b→c and c→d should not (d not in sub)
        let from_a: Vec<&Relation> = rels.iter().filter(|r| r.from_id == "a").collect();
        assert!(!from_a.is_empty());
    }

    #[test]
    fn subgraph_unknown_root_is_empty() {
        let g = sample_graph();
        let sub = g.subgraph("no_such", 5);
        assert_eq!(sub.entity_count(), 0);
    }

    #[test]
    fn to_dot_contains_entities() {
        let g = sample_graph();
        let dot = g.to_dot();
        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("Alice"));
        assert!(dot.contains("Bob"));
    }

    #[test]
    fn to_dot_contains_edges() {
        let g = sample_graph();
        let dot = g.to_dot();
        assert!(dot.contains("->"));
        assert!(dot.contains("reports_to"));
    }

    #[test]
    fn to_dot_empty_graph() {
        let g = KnowledgeGraph::new();
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains('}'));
    }

    #[test]
    fn entity_count_and_relation_count() {
        let g = sample_graph();
        assert_eq!(g.entity_count(), 4);
        assert_eq!(g.relation_count(), 4);
    }

    #[test]
    fn all_entities_returns_all() {
        let g = sample_graph();
        assert_eq!(g.all_entities().len(), 4);
    }

    #[test]
    fn add_entity_overwrites_existing() {
        let mut g = KnowledgeGraph::new();
        g.add_entity(Entity::new("x", "Original"));
        g.add_entity(Entity::new("x", "Replaced"));
        assert_eq!(g.entity("x").unwrap().label, "Replaced");
        assert_eq!(g.entity_count(), 1);
    }

    #[test]
    fn relation_weight_stored_correctly() {
        let mut g = KnowledgeGraph::new();
        g.add_entity(Entity::new("a", "A"));
        g.add_entity(Entity::new("b", "B"));
        g.add_relation(Relation::new("a", "b", "link", 0.42));
        let rel = &g.all_relations()[0];
        assert!((rel.weight - 0.42).abs() < 1e-5);
    }

    #[test]
    fn shortest_path_chain() {
        let mut g = KnowledgeGraph::new();
        for i in 0..5 {
            g.add_entity(Entity::new(format!("n{i}"), format!("Node {i}")));
        }
        for i in 0..4 {
            g.add_relation(Relation::new(
                format!("n{i}"),
                format!("n{}", i + 1),
                "next",
                1.0,
            ));
        }
        let path = g.shortest_path("n0", "n4").unwrap();
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn with_property_builder() {
        let e = Entity::new("id", "Label")
            .with_property("key1", "val1")
            .with_property("key2", "val2");
        assert_eq!(e.properties.get("key1").unwrap(), "val1");
        assert_eq!(e.properties.get("key2").unwrap(), "val2");
    }
}
