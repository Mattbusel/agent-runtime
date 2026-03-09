//! # Module: Graph
//!
//! ## Responsibility
//! Provides an in-memory knowledge graph with typed entities and relationships.
//! Mirrors the public API of `mem-graph`.
//!
//! ## Guarantees
//! - Thread-safe: `GraphStore` wraps state in `Arc<Mutex<_>>`
//! - BFS/DFS traversal and shortest-path are correct for directed graphs
//! - Non-panicking: all operations return `Result`
//!
//! ## NOT Responsible For
//! - Persistence to disk or external store
//! - Graph sharding / distributed graphs
//! - Weighted shortest-path (Dijkstra); only hop-count shortest-path provided

use crate::error::AgentRuntimeError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

// ── EntityId ──────────────────────────────────────────────────────────────────

/// Stable identifier for a graph entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

impl EntityId {
    /// Create a new `EntityId` from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Entity ────────────────────────────────────────────────────────────────────

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier.
    pub id: EntityId,
    /// Human-readable label (e.g. "Person", "Concept").
    pub label: String,
    /// Arbitrary key-value properties.
    pub properties: HashMap<String, Value>,
}

impl Entity {
    /// Construct a new entity with no properties.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: EntityId::new(id),
            label: label.into(),
            properties: HashMap::new(),
        }
    }

    /// Construct a new entity with the given properties.
    pub fn with_properties(
        id: impl Into<String>,
        label: impl Into<String>,
        properties: HashMap<String, Value>,
    ) -> Self {
        Self {
            id: EntityId::new(id),
            label: label.into(),
            properties,
        }
    }
}

// ── Relationship ──────────────────────────────────────────────────────────────

/// A directed, typed edge between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source entity.
    pub from: EntityId,
    /// Target entity.
    pub to: EntityId,
    /// Relationship type label (e.g. "KNOWS", "PART_OF").
    pub kind: String,
    /// Optional weight for weighted-graph use cases.
    pub weight: f32,
}

impl Relationship {
    /// Construct a new relationship with the given kind and weight.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        kind: impl Into<String>,
        weight: f32,
    ) -> Self {
        Self {
            from: EntityId::new(from),
            to: EntityId::new(to),
            kind: kind.into(),
            weight,
        }
    }
}

// ── MemGraphError (mirrors upstream) ─────────────────────────────────────────

/// Graph-specific errors, mirrors `mem-graph::MemGraphError`.
#[derive(Debug, thiserror::Error)]
pub enum MemGraphError {
    /// The requested entity was not found.
    #[error("Entity '{0}' not found")]
    EntityNotFound(String),

    /// A relationship between the two entities already exists with the same kind.
    #[error("Relationship '{kind}' from '{from}' to '{to}' already exists")]
    DuplicateRelationship {
        from: String,
        to: String,
        kind: String,
    },

    /// Generic internal error.
    #[error("Graph internal error: {0}")]
    Internal(String),
}

impl From<MemGraphError> for AgentRuntimeError {
    fn from(e: MemGraphError) -> Self {
        AgentRuntimeError::Graph(e.to_string())
    }
}

// ── GraphStore ────────────────────────────────────────────────────────────────

/// In-memory knowledge graph supporting entities, relationships, BFS/DFS,
/// shortest-path, and transitive closure.
///
/// ## Guarantees
/// - Thread-safe via `Arc<Mutex<_>>`
/// - BFS/DFS are non-recursive (stack-safe)
/// - Shortest-path is hop-count based (BFS)
#[derive(Debug, Clone)]
pub struct GraphStore {
    inner: Arc<Mutex<GraphInner>>,
}

#[derive(Debug)]
struct GraphInner {
    entities: HashMap<EntityId, Entity>,
    relationships: Vec<Relationship>,
}

impl GraphStore {
    /// Create a new, empty graph store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(GraphInner {
                entities: HashMap::new(),
                relationships: Vec::new(),
            })),
        }
    }

    /// Add an entity to the graph.
    ///
    /// If an entity with the same ID already exists, it is replaced.
    pub fn add_entity(&self, entity: Entity) -> Result<(), AgentRuntimeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;
        inner.entities.insert(entity.id.clone(), entity);
        Ok(())
    }

    /// Retrieve an entity by ID.
    pub fn get_entity(&self, id: &EntityId) -> Result<Entity, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;
        inner
            .entities
            .get(id)
            .cloned()
            .ok_or_else(|| AgentRuntimeError::Graph(format!("entity '{}' not found", id.0)))
    }

    /// Add a directed relationship between two existing entities.
    ///
    /// Both source and target entities must already exist in the graph.
    pub fn add_relationship(&self, rel: Relationship) -> Result<(), AgentRuntimeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;

        if !inner.entities.contains_key(&rel.from) {
            return Err(AgentRuntimeError::Graph(format!(
                "source entity '{}' not found",
                rel.from.0
            )));
        }
        if !inner.entities.contains_key(&rel.to) {
            return Err(AgentRuntimeError::Graph(format!(
                "target entity '{}' not found",
                rel.to.0
            )));
        }

        inner.relationships.push(rel);
        Ok(())
    }

    /// Remove an entity and all relationships involving it.
    pub fn remove_entity(&self, id: &EntityId) -> Result<(), AgentRuntimeError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;

        if inner.entities.remove(id).is_none() {
            return Err(AgentRuntimeError::Graph(format!(
                "entity '{}' not found",
                id.0
            )));
        }
        inner.relationships.retain(|r| &r.from != id && &r.to != id);
        Ok(())
    }

    /// Return all direct neighbours of the given entity (BFS layer 1).
    fn neighbours<'a>(
        relationships: &'a [Relationship],
        id: &EntityId,
    ) -> Vec<EntityId> {
        relationships
            .iter()
            .filter(|r| &r.from == id)
            .map(|r| r.to.clone())
            .collect()
    }

    /// Breadth-first search starting from `start`.
    ///
    /// Returns entity IDs in BFS discovery order (not including the start node).
    pub fn bfs(&self, start: &EntityId) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;

        if !inner.entities.contains_key(start) {
            return Err(AgentRuntimeError::Graph(format!(
                "start entity '{}' not found",
                start.0
            )));
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<EntityId> = VecDeque::new();
        let mut result: Vec<EntityId> = Vec::new();

        visited.insert(start.clone());
        queue.push_back(start.clone());

        while let Some(current) = queue.pop_front() {
            let neighbours: Vec<EntityId> = Self::neighbours(&inner.relationships, &current);
            for neighbour in neighbours {
                if visited.insert(neighbour.clone()) {
                    result.push(neighbour.clone());
                    queue.push_back(neighbour);
                }
            }
        }

        Ok(result)
    }

    /// Depth-first search starting from `start`.
    ///
    /// Returns entity IDs in DFS discovery order (not including the start node).
    pub fn dfs(&self, start: &EntityId) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;

        if !inner.entities.contains_key(start) {
            return Err(AgentRuntimeError::Graph(format!(
                "start entity '{}' not found",
                start.0
            )));
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut stack: Vec<EntityId> = Vec::new();
        let mut result: Vec<EntityId> = Vec::new();

        visited.insert(start.clone());
        stack.push(start.clone());

        while let Some(current) = stack.pop() {
            let neighbours: Vec<EntityId> = Self::neighbours(&inner.relationships, &current);
            for neighbour in neighbours {
                if visited.insert(neighbour.clone()) {
                    result.push(neighbour.clone());
                    stack.push(neighbour);
                }
            }
        }

        Ok(result)
    }

    /// Find the shortest path (by hop count) between `from` and `to`.
    ///
    /// # Returns
    /// - `Some(path)` — ordered list of `EntityId`s from `from` to `to` (inclusive)
    /// - `None` — no path exists
    pub fn shortest_path(
        &self,
        from: &EntityId,
        to: &EntityId,
    ) -> Result<Option<Vec<EntityId>>, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;

        if !inner.entities.contains_key(from) {
            return Err(AgentRuntimeError::Graph(format!(
                "source entity '{}' not found",
                from.0
            )));
        }
        if !inner.entities.contains_key(to) {
            return Err(AgentRuntimeError::Graph(format!(
                "target entity '{}' not found",
                to.0
            )));
        }

        if from == to {
            return Ok(Some(vec![from.clone()]));
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<Vec<EntityId>> = VecDeque::new();

        visited.insert(from.clone());
        queue.push_back(vec![from.clone()]);

        while let Some(path) = queue.pop_front() {
            let current = match path.last() {
                Some(c) => c.clone(),
                None => continue,
            };

            let neighbours: Vec<EntityId> = Self::neighbours(&inner.relationships, &current);

            for neighbour in neighbours {
                if &neighbour == to {
                    let mut full_path = path.clone();
                    full_path.push(neighbour);
                    return Ok(Some(full_path));
                }
                if visited.insert(neighbour.clone()) {
                    let mut new_path = path.clone();
                    new_path.push(neighbour.clone());
                    queue.push_back(new_path);
                }
            }
        }

        Ok(None)
    }

    /// Compute the transitive closure: all entities reachable from `start`.
    pub fn transitive_closure(
        &self,
        start: &EntityId,
    ) -> Result<HashSet<EntityId>, AgentRuntimeError> {
        let reachable = self.bfs(start)?;
        let mut set: HashSet<EntityId> = reachable.into_iter().collect();
        set.insert(start.clone());
        Ok(set)
    }

    /// Return the number of entities in the graph.
    pub fn entity_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;
        Ok(inner.entities.len())
    }

    /// Return the number of relationships in the graph.
    pub fn relationship_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AgentRuntimeError::Graph(format!("lock poisoned: {e}")))?;
        Ok(inner.relationships.len())
    }
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> GraphStore {
        GraphStore::new()
    }

    fn add(g: &GraphStore, id: &str) {
        g.add_entity(Entity::new(id, "Node")).unwrap();
    }

    fn link(g: &GraphStore, from: &str, to: &str) {
        g.add_relationship(Relationship::new(from, to, "CONNECTS", 1.0)).unwrap();
    }

    // ── EntityId ──────────────────────────────────────────────────────────────

    #[test]
    fn test_entity_id_equality() {
        assert_eq!(EntityId::new("a"), EntityId::new("a"));
        assert_ne!(EntityId::new("a"), EntityId::new("b"));
    }

    #[test]
    fn test_entity_id_display() {
        let id = EntityId::new("hello");
        assert_eq!(id.to_string(), "hello");
    }

    // ── Entity ────────────────────────────────────────────────────────────────

    #[test]
    fn test_entity_new_has_empty_properties() {
        let e = Entity::new("e1", "Person");
        assert!(e.properties.is_empty());
    }

    #[test]
    fn test_entity_with_properties_stores_props() {
        let mut props = HashMap::new();
        props.insert("age".into(), Value::Number(42.into()));
        let e = Entity::with_properties("e1", "Person", props);
        assert!(e.properties.contains_key("age"));
    }

    // ── GraphStore basic ops ──────────────────────────────────────────────────

    #[test]
    fn test_graph_add_entity_increments_count() {
        let g = make_graph();
        add(&g, "a");
        assert_eq!(g.entity_count().unwrap(), 1);
    }

    #[test]
    fn test_graph_get_entity_returns_entity() {
        let g = make_graph();
        g.add_entity(Entity::new("e1", "Person")).unwrap();
        let e = g.get_entity(&EntityId::new("e1")).unwrap();
        assert_eq!(e.label, "Person");
    }

    #[test]
    fn test_graph_get_entity_missing_returns_error() {
        let g = make_graph();
        assert!(g.get_entity(&EntityId::new("ghost")).is_err());
    }

    #[test]
    fn test_graph_add_relationship_increments_count() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        link(&g, "a", "b");
        assert_eq!(g.relationship_count().unwrap(), 1);
    }

    #[test]
    fn test_graph_add_relationship_missing_source_fails() {
        let g = make_graph();
        add(&g, "b");
        let result = g.add_relationship(Relationship::new("ghost", "b", "X", 1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_add_relationship_missing_target_fails() {
        let g = make_graph();
        add(&g, "a");
        let result = g.add_relationship(Relationship::new("a", "ghost", "X", 1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_remove_entity_removes_relationships() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        link(&g, "a", "b");
        g.remove_entity(&EntityId::new("a")).unwrap();
        assert_eq!(g.entity_count().unwrap(), 1);
        assert_eq!(g.relationship_count().unwrap(), 0);
    }

    #[test]
    fn test_graph_remove_entity_missing_returns_error() {
        let g = make_graph();
        assert!(g.remove_entity(&EntityId::new("ghost")).is_err());
    }

    // ── BFS ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_bfs_finds_direct_neighbours() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "b");
        link(&g, "a", "c");
        let visited = g.bfs(&EntityId::new("a")).unwrap();
        assert_eq!(visited.len(), 2);
    }

    #[test]
    fn test_bfs_traverses_chain() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c"); add(&g, "d");
        link(&g, "a", "b"); link(&g, "b", "c"); link(&g, "c", "d");
        let visited = g.bfs(&EntityId::new("a")).unwrap();
        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], EntityId::new("b"));
    }

    #[test]
    fn test_bfs_handles_isolated_node() {
        let g = make_graph();
        add(&g, "a");
        let visited = g.bfs(&EntityId::new("a")).unwrap();
        assert!(visited.is_empty());
    }

    #[test]
    fn test_bfs_missing_start_returns_error() {
        let g = make_graph();
        assert!(g.bfs(&EntityId::new("ghost")).is_err());
    }

    // ── DFS ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_dfs_visits_all_reachable_nodes() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c"); add(&g, "d");
        link(&g, "a", "b"); link(&g, "a", "c"); link(&g, "b", "d");
        let visited = g.dfs(&EntityId::new("a")).unwrap();
        assert_eq!(visited.len(), 3);
    }

    #[test]
    fn test_dfs_handles_isolated_node() {
        let g = make_graph();
        add(&g, "a");
        let visited = g.dfs(&EntityId::new("a")).unwrap();
        assert!(visited.is_empty());
    }

    #[test]
    fn test_dfs_missing_start_returns_error() {
        let g = make_graph();
        assert!(g.dfs(&EntityId::new("ghost")).is_err());
    }

    // ── Shortest path ─────────────────────────────────────────────────────────

    #[test]
    fn test_shortest_path_direct_connection() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        link(&g, "a", "b");
        let path = g.shortest_path(&EntityId::new("a"), &EntityId::new("b")).unwrap();
        assert_eq!(path, Some(vec![EntityId::new("a"), EntityId::new("b")]));
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "b"); link(&g, "b", "c");
        let path = g.shortest_path(&EntityId::new("a"), &EntityId::new("c")).unwrap();
        assert_eq!(path.as_ref().map(|p| p.len()), Some(3));
    }

    #[test]
    fn test_shortest_path_returns_none_for_disconnected() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        let path = g.shortest_path(&EntityId::new("a"), &EntityId::new("b")).unwrap();
        assert_eq!(path, None);
    }

    #[test]
    fn test_shortest_path_same_node_returns_single_element() {
        let g = make_graph();
        add(&g, "a");
        let path = g.shortest_path(&EntityId::new("a"), &EntityId::new("a")).unwrap();
        assert_eq!(path, Some(vec![EntityId::new("a")]));
    }

    #[test]
    fn test_shortest_path_missing_source_returns_error() {
        let g = make_graph();
        add(&g, "b");
        assert!(g.shortest_path(&EntityId::new("ghost"), &EntityId::new("b")).is_err());
    }

    #[test]
    fn test_shortest_path_missing_target_returns_error() {
        let g = make_graph();
        add(&g, "a");
        assert!(g.shortest_path(&EntityId::new("a"), &EntityId::new("ghost")).is_err());
    }

    // ── Transitive closure ────────────────────────────────────────────────────

    #[test]
    fn test_transitive_closure_includes_start() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        link(&g, "a", "b");
        let closure = g.transitive_closure(&EntityId::new("a")).unwrap();
        assert!(closure.contains(&EntityId::new("a")));
        assert!(closure.contains(&EntityId::new("b")));
    }

    #[test]
    fn test_transitive_closure_isolated_node_contains_only_self() {
        let g = make_graph();
        add(&g, "a");
        let closure = g.transitive_closure(&EntityId::new("a")).unwrap();
        assert_eq!(closure.len(), 1);
    }

    // ── MemGraphError conversion ──────────────────────────────────────────────

    #[test]
    fn test_mem_graph_error_converts_to_runtime_error() {
        let e = MemGraphError::EntityNotFound("x".into());
        let re: AgentRuntimeError = e.into();
        assert!(matches!(re, AgentRuntimeError::Graph(_)));
    }
}
