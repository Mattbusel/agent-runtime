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

use crate::error::AgentRuntimeError;
use crate::util::recover_lock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

// ── OrdF32 newtype ─────────────────────────────────────────────────────────────

/// Newtype wrapper for `f32` that implements `Ord`.
/// NaN is treated as `Greater` for safe use in a `BinaryHeap`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Greater)
    }
}

// ── EntityId ──────────────────────────────────────────────────────────────────

/// Stable identifier for a graph entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityId(pub String);

impl EntityId {
    /// Create a new `EntityId` from any string-like value.
    ///
    /// # Panics (debug only)
    ///
    /// Triggers a `debug_assert!` if `id` is empty.  In release builds a
    /// `tracing::warn!` is emitted instead so that the misconfiguration is
    /// surfaced in production logs without aborting the process.
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        if id.is_empty() {
            debug_assert!(false, "EntityId must not be empty");
            tracing::warn!("EntityId::new called with an empty string — entity IDs should be non-empty to avoid lookup ambiguity");
        }
        Self(id)
    }

    /// Create a validated `EntityId`, returning an error if `id` is empty.
    pub fn try_new(id: impl Into<String>) -> Result<Self, AgentRuntimeError> {
        let id = id.into();
        if id.is_empty() {
            return Err(AgentRuntimeError::Graph(
                "EntityId must not be empty".into(),
            ));
        }
        Ok(Self(id))
    }

    /// Return the inner ID string as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return `true` if the inner ID string is empty.
    ///
    /// Note: `EntityId::new` warns (debug: asserts) against empty IDs.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return `true` if the inner ID string starts with `prefix`.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.0.starts_with(prefix)
    }
}

impl AsRef<str> for EntityId {
    fn as_ref(&self) -> &str {
        &self.0
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

    /// Add a single key-value property, consuming and returning `self`.
    ///
    /// Allows fluent builder-style construction:
    /// ```rust,ignore
    /// let e = Entity::new("alice", "Person")
    ///     .with_property("age", 30.into())
    ///     .with_property("role", "engineer".into());
    /// ```
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Return `true` if the entity has a property with the given key.
    pub fn has_property(&self, key: &str) -> bool {
        self.properties.contains_key(key)
    }

    /// Return a reference to the property value for `key`, or `None` if absent.
    pub fn property_value(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
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

    /// Return `true` if this relationship is a self-loop (`from == to`).
    pub fn is_self_loop(&self) -> bool {
        self.from == self.to
    }

    /// Return a new `Relationship` with `from` and `to` swapped.
    ///
    /// The `kind` and `weight` are preserved unchanged.
    pub fn reversed(&self) -> Self {
        Self {
            from: self.to.clone(),
            to: self.from.clone(),
            kind: self.kind.clone(),
            weight: self.weight,
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
        /// Source entity ID.
        from: String,
        /// Target entity ID.
        to: String,
        /// Relationship kind label.
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
/// shortest-path, weighted shortest-path, and graph analytics.
///
/// ## Guarantees
/// - Thread-safe via `Arc<Mutex<_>>`
/// - BFS/DFS are non-recursive (stack-safe)
/// - Shortest-path is hop-count based (BFS)
/// - Weighted shortest-path uses Dijkstra's algorithm
#[derive(Debug, Clone)]
pub struct GraphStore {
    inner: Arc<Mutex<GraphInner>>,
}

#[derive(Debug)]
struct GraphInner {
    entities: HashMap<EntityId, Entity>,
    /// Flat list kept for full-edge iteration (degree/betweenness/label-propagation).
    relationships: Vec<Relationship>,
    /// Adjacency index: entity → outgoing relationships.
    /// Kept in sync with `relationships` to give O(degree) neighbour lookup
    /// in BFS/DFS/shortest-path without a full linear scan.
    adjacency: HashMap<EntityId, Vec<Relationship>>,
    /// Reverse adjacency index: entity → set of entities with edges pointing TO it.
    /// Enables O(in-degree) `in_degree`, `source_nodes`, and `isolates` without a
    /// full O(|E|) scan.
    reverse_adjacency: HashMap<EntityId, Vec<EntityId>>,
    /// Cached result of cycle detection. Invalidated on any mutation.
    cycle_cache: Option<bool>,
}

impl GraphStore {
    /// Create a new, empty graph store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(GraphInner {
                entities: HashMap::new(),
                relationships: Vec::new(),
                adjacency: HashMap::new(),
                reverse_adjacency: HashMap::new(),
                cycle_cache: None,
            })),
        }
    }

    /// Add an entity to the graph.
    ///
    /// If an entity with the same ID already exists, it is replaced.
    pub fn add_entity(&self, entity: Entity) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "add_entity");
        inner.cycle_cache = None;
        // Ensure an adjacency entry exists even if the entity has no outgoing edges.
        inner.adjacency.entry(entity.id.clone()).or_default();
        inner.entities.insert(entity.id.clone(), entity);
        Ok(())
    }

    /// Retrieve an entity by ID.
    pub fn get_entity(&self, id: &EntityId) -> Result<Entity, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "get_entity");
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
        let mut inner = recover_lock(self.inner.lock(), "add_relationship");

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

        // Reject duplicate (from, to, kind) triples — the DuplicateRelationship
        // error variant existed but was never raised, silently allowing duplicate
        // edges that corrupt relationship_count() and BFS/DFS result counts.
        let duplicate = inner
            .relationships
            .iter()
            .any(|r| r.from == rel.from && r.to == rel.to && r.kind == rel.kind);
        if duplicate {
            return Err(AgentRuntimeError::Graph(
                MemGraphError::DuplicateRelationship {
                    from: rel.from.0.clone(),
                    to: rel.to.0.clone(),
                    kind: rel.kind.clone(),
                }
                .to_string(),
            ));
        }

        inner.cycle_cache = None;
        // Keep adjacency in sync: add to outgoing list for rel.from.
        inner
            .adjacency
            .entry(rel.from.clone())
            .or_default()
            .push(rel.clone());
        // Keep reverse adjacency in sync: add rel.from to incoming list for rel.to.
        inner
            .reverse_adjacency
            .entry(rel.to.clone())
            .or_default()
            .push(rel.from.clone());
        inner.relationships.push(rel);
        Ok(())
    }

    /// Remove a specific relationship by (from, to, kind).
    ///
    /// Returns `Err` if no matching relationship exists.
    pub fn remove_relationship(
        &self,
        from: &EntityId,
        to: &EntityId,
        kind: &str,
    ) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "remove_relationship");

        let before = inner.relationships.len();
        inner
            .relationships
            .retain(|r| !(&r.from == from && &r.to == to && r.kind == kind));
        if inner.relationships.len() == before {
            return Err(AgentRuntimeError::Graph(format!(
                "relationship '{kind}' from '{}' to '{}' not found",
                from.0, to.0
            )));
        }

        // Keep adjacency in sync.
        if let Some(adj) = inner.adjacency.get_mut(from) {
            adj.retain(|r| !(&r.to == to && r.kind == kind));
        }
        // Keep reverse adjacency in sync.
        if let Some(rev) = inner.reverse_adjacency.get_mut(to) {
            rev.retain(|src| src != from);
        }

        inner.cycle_cache = None;
        Ok(())
    }

    /// Remove an entity and all relationships involving it.
    pub fn remove_entity(&self, id: &EntityId) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "remove_entity");

        if inner.entities.remove(id).is_none() {
            return Err(AgentRuntimeError::Graph(format!(
                "entity '{}' not found",
                id.0
            )));
        }
        inner.cycle_cache = None;
        inner.relationships.retain(|r| &r.from != id && &r.to != id);
        // Remove outgoing edges for this entity and scrub it from others' lists.
        inner.adjacency.remove(id);
        for adj in inner.adjacency.values_mut() {
            adj.retain(|r| &r.to != id);
        }
        // Remove incoming edges for this entity from the reverse adjacency index.
        inner.reverse_adjacency.remove(id);
        for rev in inner.reverse_adjacency.values_mut() {
            rev.retain(|src| src != id);
        }
        Ok(())
    }

    /// Return all direct outgoing neighbours of the given entity (BFS layer 1).
    ///
    /// Uses the adjacency index for O(degree) lookup instead of O(|edges|).
    fn neighbours(adjacency: &HashMap<EntityId, Vec<Relationship>>, id: &EntityId) -> Vec<EntityId> {
        adjacency
            .get(id)
            .map(|rels| rels.iter().map(|r| r.to.clone()).collect())
            .unwrap_or_default()
    }

    /// Breadth-first search starting from `start`.
    ///
    /// Returns entity IDs in BFS discovery order (not including the start node).
    #[tracing::instrument(skip(self))]
    pub fn bfs(&self, start: &EntityId) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "bfs");

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
            let neighbours: Vec<EntityId> = Self::neighbours(&inner.adjacency, &current);
            for neighbour in neighbours {
                if visited.insert(neighbour.clone()) {
                    result.push(neighbour.clone());
                    queue.push_back(neighbour);
                }
            }
        }

        tracing::debug!("BFS visited {} nodes", result.len());
        Ok(result)
    }

    /// Depth-first search starting from `start`.
    ///
    /// Returns entity IDs in DFS discovery order (not including the start node).
    #[tracing::instrument(skip(self))]
    pub fn dfs(&self, start: &EntityId) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "dfs");

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
            let neighbours: Vec<EntityId> = Self::neighbours(&inner.adjacency, &current);
            for neighbour in neighbours {
                if visited.insert(neighbour.clone()) {
                    result.push(neighbour.clone());
                    stack.push(neighbour);
                }
            }
        }

        tracing::debug!("DFS visited {} nodes", result.len());
        Ok(result)
    }

    /// Find the shortest path (by hop count) between `from` and `to`.
    ///
    /// # Returns
    /// - `Some(path)` — ordered list of `EntityId`s from `from` to `to` (inclusive)
    /// - `None` — no path exists
    #[tracing::instrument(skip(self))]
    pub fn shortest_path(
        &self,
        from: &EntityId,
        to: &EntityId,
    ) -> Result<Option<Vec<EntityId>>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "shortest_path");

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

        // Item 5 — predecessor-map BFS; O(1) per enqueue instead of O(path_len).
        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut prev: HashMap<EntityId, EntityId> = HashMap::new();
        let mut queue: VecDeque<EntityId> = VecDeque::new();

        visited.insert(from.clone());
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            for neighbour in Self::neighbours(&inner.adjacency, &current) {
                if &neighbour == to {
                    // Reconstruct path by following prev back from current.
                    let mut path = vec![neighbour, current.clone()];
                    let mut node = current;
                    while let Some(p) = prev.get(&node) {
                        path.push(p.clone());
                        node = p.clone();
                    }
                    path.reverse();
                    return Ok(Some(path));
                }
                if visited.insert(neighbour.clone()) {
                    prev.insert(neighbour.clone(), current.clone());
                    queue.push_back(neighbour);
                }
            }
        }

        Ok(None)
    }

    /// Find the shortest weighted path between `from` and `to` using Dijkstra's algorithm.
    ///
    /// Uses `Relationship::weight` as edge cost. Negative weights are not supported
    /// and will cause this method to return an error.
    ///
    /// # Returns
    /// - `Ok(Some((path, total_weight)))` — the shortest path and its total weight
    /// - `Ok(None)` — no path exists between `from` and `to`
    /// - `Err(...)` — either entity not found, or a negative edge weight was encountered
    pub fn shortest_path_weighted(
        &self,
        from: &EntityId,
        to: &EntityId,
    ) -> Result<Option<(Vec<EntityId>, f32)>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "shortest_path_weighted");

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

        // Validate: no negative weights
        for rel in &inner.relationships {
            if rel.weight < 0.0 {
                return Err(AgentRuntimeError::Graph(format!(
                    "negative weight {:.4} on edge '{}' -> '{}'",
                    rel.weight, rel.from.0, rel.to.0
                )));
            }
        }

        if from == to {
            return Ok(Some((vec![from.clone()], 0.0)));
        }

        // Dijkstra using a max-heap with negated weights to simulate a min-heap.
        // Heap entries: (negated_cost, node_id)
        let mut dist: HashMap<EntityId, f32> = HashMap::new();
        let mut prev: HashMap<EntityId, EntityId> = HashMap::new();
        // BinaryHeap is a max-heap; negate weights to get min-heap behaviour.
        let mut heap: BinaryHeap<(OrdF32, EntityId)> = BinaryHeap::new();

        dist.insert(from.clone(), 0.0);
        heap.push((OrdF32(-0.0), from.clone()));

        while let Some((OrdF32(neg_cost), current)) = heap.pop() {
            let cost = -neg_cost;

            // Skip stale entries.
            if let Some(&best) = dist.get(&current) {
                if cost > best {
                    continue;
                }
            }

            if &current == to {
                // Reconstruct path in reverse.
                let mut path = vec![to.clone()];
                let mut node = to.clone();
                while let Some(p) = prev.get(&node) {
                    path.push(p.clone());
                    node = p.clone();
                }
                path.reverse();
                return Ok(Some((path, cost)));
            }

            // Use the adjacency index for O(degree) neighbour lookup instead of
            // scanning all relationships.
            if let Some(rels) = inner.adjacency.get(&current) {
                for rel in rels {
                    let next_cost = cost + rel.weight;
                    let entry = dist.entry(rel.to.clone()).or_insert(f32::INFINITY);
                    if next_cost < *entry {
                        *entry = next_cost;
                        prev.insert(rel.to.clone(), current.clone());
                        heap.push((OrdF32(-next_cost), rel.to.clone()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// BFS that builds a `HashSet` directly (including `start`).
    ///
    /// Operates on a pre-locked `GraphInner` to avoid acquiring the mutex twice
    /// and to skip the intermediate `Vec` allocation that `bfs()` produces.
    fn bfs_into_set(inner: &GraphInner, start: &EntityId) -> HashSet<EntityId> {
        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<EntityId> = VecDeque::new();
        visited.insert(start.clone());
        queue.push_back(start.clone());
        while let Some(current) = queue.pop_front() {
            for neighbour in Self::neighbours(&inner.adjacency, &current) {
                if visited.insert(neighbour.clone()) {
                    queue.push_back(neighbour);
                }
            }
        }
        visited
    }

    /// Compute the transitive closure: all entities reachable from `start`
    /// (including `start` itself).
    ///
    /// Uses a single lock acquisition and builds the result as a `HashSet`
    /// directly, avoiding the intermediate `Vec` that would otherwise be
    /// produced by delegating to [`bfs`].
    ///
    /// [`bfs`]: GraphStore::bfs
    pub fn transitive_closure(
        &self,
        start: &EntityId,
    ) -> Result<HashSet<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "transitive_closure");
        if !inner.entities.contains_key(start) {
            return Err(AgentRuntimeError::Graph(format!(
                "start entity '{}' not found",
                start.0
            )));
        }
        Ok(Self::bfs_into_set(&inner, start))
    }

    /// Return the number of entities in the graph.
    pub fn entity_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "entity_count");
        Ok(inner.entities.len())
    }

    /// Return the number of entities in the graph.
    ///
    /// Alias for [`entity_count`] using graph-theory terminology.
    ///
    /// [`entity_count`]: GraphStore::entity_count
    pub fn node_count(&self) -> Result<usize, AgentRuntimeError> {
        self.entity_count()
    }

    /// Return the number of relationships in the graph.
    pub fn relationship_count(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "relationship_count");
        Ok(inner.relationships.len())
    }

    /// Return the average out-degree across all entities.
    ///
    /// Returns `0.0` for an empty graph.  The result is the total number of
    /// directed edges divided by the number of nodes.
    pub fn average_out_degree(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "average_out_degree");
        let n = inner.entities.len();
        if n == 0 {
            return Ok(0.0);
        }
        Ok(inner.relationships.len() as f64 / n as f64)
    }

    /// Return the in-degree (number of incoming edges) for the given entity.
    ///
    /// Returns `0` if the entity does not exist or has no incoming edges.
    pub fn in_degree_for(&self, entity_id: &EntityId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "in_degree_for");
        Ok(inner
            .reverse_adjacency
            .get(entity_id)
            .map(|v| v.len())
            .unwrap_or(0))
    }

    /// Return the out-degree (number of outgoing edges) for the given entity.
    ///
    /// Returns `0` if the entity does not exist or has no outgoing edges.
    pub fn out_degree_for(&self, entity_id: &EntityId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "out_degree_for");
        Ok(inner
            .adjacency
            .get(entity_id)
            .map(|v| v.len())
            .unwrap_or(0))
    }

    /// Return all entities that have a directed edge pointing **to** `entity_id`
    /// (its predecessors in the directed graph).
    ///
    /// Returns an empty `Vec` if the entity has no incoming edges or does not
    /// exist.
    pub fn predecessors(&self, entity_id: &EntityId) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "predecessors");
        let ids = inner
            .reverse_adjacency
            .get(entity_id)
            .cloned()
            .unwrap_or_default();
        Ok(ids
            .iter()
            .filter_map(|id| inner.entities.get(id).cloned())
            .collect())
    }

    /// Return `true` if the entity has no incoming edges (in-degree == 0).
    ///
    /// In a DAG, source nodes (roots) have no predecessors.  Returns `false`
    /// if the entity does not exist.
    pub fn is_source(&self, entity_id: &EntityId) -> Result<bool, AgentRuntimeError> {
        Ok(self.in_degree_for(entity_id)? == 0)
    }

    /// Return all entities directly reachable from `entity_id` via outgoing edges
    /// (its successors / immediate out-neighbors).
    ///
    /// Returns an empty `Vec` if the entity has no outgoing edges or does not exist.
    pub fn successors(&self, entity_id: &EntityId) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "successors");
        let rels = inner.adjacency.get(entity_id).cloned().unwrap_or_default();
        Ok(rels
            .iter()
            .filter_map(|r| inner.entities.get(&r.to).cloned())
            .collect())
    }

    /// Return `true` if the entity has no outgoing edges (out-degree == 0).
    ///
    /// In a DAG, sink nodes (leaves) have no successors.  Returns `true` if
    /// the entity does not exist (unknown nodes cannot have outgoing edges).
    pub fn is_sink(&self, entity_id: &EntityId) -> Result<bool, AgentRuntimeError> {
        Ok(self.out_degree_for(entity_id)? == 0)
    }

    /// Return the set of all entity IDs reachable from `start` via directed edges
    /// (BFS traversal).  The starting node itself is **not** included.
    ///
    /// Returns an empty `HashSet` if `start` has no outgoing edges or does not exist.
    pub fn reachable_from(
        &self,
        start: &EntityId,
    ) -> Result<std::collections::HashSet<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "reachable_from");
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        if let Some(rels) = inner.adjacency.get(start) {
            for r in rels {
                if visited.insert(r.to.clone()) {
                    queue.push_back(r.to.clone());
                }
            }
        }
        while let Some(current) = queue.pop_front() {
            if let Some(rels) = inner.adjacency.get(&current) {
                for r in rels {
                    if visited.insert(r.to.clone()) {
                        queue.push_back(r.to.clone());
                    }
                }
            }
        }
        Ok(visited)
    }

    /// Return `true` if the directed graph contains at least one cycle.
    ///
    /// Uses iterative DFS with a three-colour scheme (unvisited / in-stack / done).
    /// An empty graph is acyclic.
    pub fn contains_cycle(&self) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "contains_cycle");
        let mut color: HashMap<&EntityId, u8> = HashMap::new();

        for start in inner.entities.keys() {
            if color.get(start).copied().unwrap_or(0) != 0 {
                continue;
            }
            let mut stack: Vec<(&EntityId, usize)> = vec![(start, 0)];
            color.insert(start, 1);
            while let Some((node, idx)) = stack.last_mut() {
                let neighbors = inner.adjacency.get(node).map(|v| v.as_slice()).unwrap_or(&[]);
                if *idx < neighbors.len() {
                    let neighbor = &neighbors[*idx].to;
                    *idx += 1;
                    match color.get(neighbor).copied().unwrap_or(0) {
                        1 => return Ok(true),
                        0 => {
                            color.insert(neighbor, 1);
                            stack.push((neighbor, 0));
                        }
                        _ => {}
                    }
                } else {
                    color.insert(*node, 2);
                    stack.pop();
                }
            }
        }
        Ok(false)
    }

    /// Return the number of relationships in the graph.
    ///
    /// Alias for [`relationship_count`] using graph-theory terminology.
    ///
    /// [`relationship_count`]: GraphStore::relationship_count
    pub fn edge_count(&self) -> Result<usize, AgentRuntimeError> {
        self.relationship_count()
    }

    /// Return `true` if the directed graph contains **no** cycles.
    ///
    /// Shorthand for `!self.contains_cycle()`.
    pub fn is_acyclic(&self) -> Result<bool, AgentRuntimeError> {
        Ok(!self.contains_cycle()?)
    }

    /// Return the maximum out-degree across all entities, or `0` for empty graphs.
    pub fn max_out_degree(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "max_out_degree");
        Ok(inner
            .adjacency
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0))
    }

    /// Return the maximum in-degree across all entities, or `0` for empty graphs.
    pub fn max_in_degree(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "max_in_degree");
        Ok(inner
            .reverse_adjacency
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0))
    }

    /// Return all entity IDs in the graph without allocating full `Entity` objects.
    ///
    /// Cheaper than [`all_entities`] when only IDs are needed.
    ///
    /// [`all_entities`]: GraphStore::all_entities
    pub fn entity_ids(&self) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "entity_ids");
        Ok(inner.entities.keys().cloned().collect())
    }

    /// Return `true` if the graph contains no entities.
    pub fn is_empty(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.entity_count()? == 0)
    }

    /// Remove all entities and relationships from the graph.
    ///
    /// After this call `entity_count()` and `relationship_count()` both return `0`.
    pub fn clear(&self) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "GraphStore::clear");
        inner.entities.clear();
        inner.relationships.clear();
        inner.adjacency.clear();
        inner.reverse_adjacency.clear();
        inner.cycle_cache = None;
        Ok(())
    }

    /// Count entities whose label equals `label` (case-sensitive).
    pub fn entity_count_by_label(&self, label: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "entity_count_by_label");
        Ok(inner.entities.values().filter(|e| e.label == label).count())
    }

    /// Compute the directed graph density: |E| / (|V| × (|V| − 1)).
    ///
    /// Returns `0.0` for graphs with fewer than two entities (no edges possible).
    /// A density of `1.0` means every possible directed edge is present.
    pub fn graph_density(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "graph_density");
        let v = inner.entities.len();
        if v < 2 {
            return Ok(0.0);
        }
        let e = inner.relationships.len() as f64;
        Ok(e / (v as f64 * (v - 1) as f64))
    }

    /// Return all distinct entity label strings present in the graph, sorted.
    pub fn entity_labels(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "entity_labels");
        let mut labels: Vec<String> = inner
            .entities
            .values()
            .map(|e| e.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        labels.sort_unstable();
        Ok(labels)
    }

    /// Return all distinct relationship kind strings present in the graph, sorted.
    pub fn relationship_kinds(&self) -> Result<Vec<String>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "relationship_kinds");
        let mut kinds: Vec<String> = inner
            .relationships
            .iter()
            .map(|r| r.kind.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        kinds.sort_unstable();
        Ok(kinds)
    }

    /// Update the label of an existing entity in-place.
    ///
    /// Returns `Ok(true)` if the entity was found and updated, `Ok(false)` if not found.
    pub fn update_entity_label(
        &self,
        id: &EntityId,
        new_label: impl Into<String>,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "update_entity_label");
        if let Some(entity) = inner.entities.get_mut(id) {
            entity.label = new_label.into();
            inner.cycle_cache = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Compute normalized degree centrality for each entity.
    /// Degree = (in_degree + out_degree) / (n - 1), or 0.0 if n <= 1.
    pub fn degree_centrality(&self) -> Result<HashMap<EntityId, f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "degree_centrality");
        let n = inner.entities.len();

        // Both out-degree and in-degree are now available from the index:
        // adjacency[id].len()         == out-degree
        // reverse_adjacency[id].len() == in-degree
        let denom = if n <= 1 { 1.0 } else { (n - 1) as f32 };
        let mut result = HashMap::new();
        for id in inner.entities.keys() {
            let od = inner.adjacency.get(id).map_or(0, |v| v.len());
            let id_ = inner.reverse_adjacency.get(id).map_or(0, |v| v.len());
            let centrality = if n <= 1 {
                0.0
            } else {
                (od + id_) as f32 / denom
            };
            result.insert(id.clone(), centrality);
        }

        Ok(result)
    }

    /// Compute normalized betweenness centrality for each entity.
    /// Uses Brandes' algorithm with hop-count BFS.
    ///
    /// # Complexity
    /// O(V * E) time. Not suitable for very large graphs (>1000 nodes).
    pub fn betweenness_centrality(&self) -> Result<HashMap<EntityId, f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "betweenness_centrality");
        let n = inner.entities.len();
        let nodes: Vec<EntityId> = inner.entities.keys().cloned().collect();

        let mut centrality: HashMap<EntityId, f32> =
            nodes.iter().map(|id| (id.clone(), 0.0f32)).collect();

        // Pre-allocate per-source work buffers and reuse them each iteration
        // to avoid O(V) HashMap allocations * V source nodes = O(V²) allocations.
        let mut stack: Vec<EntityId> = Vec::with_capacity(n);
        let mut predecessors: HashMap<EntityId, Vec<EntityId>> =
            nodes.iter().map(|id| (id.clone(), vec![])).collect();
        let mut sigma: HashMap<EntityId, f32> =
            nodes.iter().map(|id| (id.clone(), 0.0f32)).collect();
        let mut dist: HashMap<EntityId, i64> =
            nodes.iter().map(|id| (id.clone(), -1i64)).collect();
        let mut delta: HashMap<EntityId, f32> =
            nodes.iter().map(|id| (id.clone(), 0.0f32)).collect();
        let mut queue: VecDeque<EntityId> = VecDeque::with_capacity(n);

        for source in &nodes {
            // BFS to find shortest path counts and predecessors.
            // Reset per-source state by clearing rather than re-allocating.
            stack.clear();
            for v in predecessors.values_mut() {
                v.clear();
            }
            for v in sigma.values_mut() {
                *v = 0.0;
            }
            for v in dist.values_mut() {
                *v = -1;
            }
            for v in delta.values_mut() {
                *v = 0.0;
            }
            queue.clear();

            *sigma.entry(source.clone()).or_insert(0.0) = 1.0;
            *dist.entry(source.clone()).or_insert(-1) = 0;
            queue.push_back(source.clone());

            while let Some(v) = queue.pop_front() {
                stack.push(v.clone());
                let d_v = *dist.get(&v).unwrap_or(&0);
                let sigma_v = *sigma.get(&v).unwrap_or(&0.0);
                // Use adjacency index for O(degree) neighbour lookup.
                if let Some(rels) = inner.adjacency.get(&v) {
                    for rel in rels {
                        let w = &rel.to;
                        let d_w = dist.get(w).copied().unwrap_or(-1);
                        if d_w < 0 {
                            queue.push_back(w.clone());
                            *dist.entry(w.clone()).or_insert(-1) = d_v + 1;
                        }
                        if dist.get(w).copied().unwrap_or(-1) == d_v + 1 {
                            *sigma.entry(w.clone()).or_insert(0.0) += sigma_v;
                            predecessors.entry(w.clone()).or_default().push(v.clone());
                        }
                    }
                }
            }

            // (delta map reset above at start of loop)

            while let Some(w) = stack.pop() {
                let delta_w = *delta.get(&w).unwrap_or(&0.0);
                let sigma_w = *sigma.get(&w).unwrap_or(&1.0);
                // Item 5 — iterate predecessors by reference to avoid cloning the Vec.
                for v in predecessors
                    .get(&w)
                    .map(|ps| ps.as_slice())
                    .unwrap_or_default()
                {
                    let sigma_v = *sigma.get(v).unwrap_or(&1.0);
                    let contribution = (sigma_v / sigma_w) * (1.0 + delta_w);
                    *delta.entry(v.clone()).or_insert(0.0) += contribution;
                }
                if &w != source {
                    *centrality.entry(w.clone()).or_insert(0.0) += delta_w;
                }
            }
        }

        // Normalize by 2 / ((n-1) * (n-2)) for directed graphs.
        if n > 2 {
            let norm = 2.0 / (((n - 1) * (n - 2)) as f32);
            for v in centrality.values_mut() {
                *v *= norm;
            }
        } else {
            for v in centrality.values_mut() {
                *v = 0.0;
            }
        }

        Ok(centrality)
    }

    /// Detect communities using label propagation.
    /// Each entity starts as its own community. In each iteration, each entity
    /// adopts the most frequent label among its neighbours.
    /// Returns a map of entity ID → community ID (usize).
    pub fn label_propagation_communities(
        &self,
        max_iterations: usize,
    ) -> Result<HashMap<EntityId, usize>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "label_propagation_communities");
        let nodes: Vec<EntityId> = inner.entities.keys().cloned().collect();

        // Build a temporary reverse-adjacency (incoming edges) so that each
        // node can cheaply query both its out-neighbours (via adjacency) and
        // its in-neighbours (via reverse_adj) without an O(|E|) scan per node.
        let mut reverse_adj: HashMap<EntityId, Vec<EntityId>> =
            nodes.iter().map(|id| (id.clone(), vec![])).collect();
        for rel in &inner.relationships {
            reverse_adj
                .entry(rel.to.clone())
                .or_default()
                .push(rel.from.clone());
        }

        // Assign each node a unique initial label (index in nodes vec).
        let mut labels: HashMap<EntityId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();

        // Hoist the frequency counter out of the inner loop to avoid
        // allocating a fresh HashMap for every node on every iteration.
        let mut freq: HashMap<usize, usize> = HashMap::new();

        for _ in 0..max_iterations {
            let mut changed = false;
            // Iterate in a stable order.
            for node in &nodes {
                // Collect neighbour labels using adjacency (out) + reverse_adj (in).
                let out_labels = inner
                    .adjacency
                    .get(node)
                    .map(|rels| {
                        rels.iter()
                            .map(|r| labels.get(&r.to).copied().unwrap_or(0))
                    })
                    .into_iter()
                    .flatten();
                let in_labels = reverse_adj
                    .get(node)
                    .map(|froms| froms.iter().map(|f| labels.get(f).copied().unwrap_or(0)))
                    .into_iter()
                    .flatten();

                freq.clear();
                for lbl in out_labels.chain(in_labels) {
                    *freq.entry(lbl).or_insert(0) += 1;
                }

                if freq.is_empty() {
                    continue;
                }

                // Find the most frequent label.
                let best = freq
                    .iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(&lbl, _)| lbl);

                if let Some(new_label) = best {
                    let current = labels.entry(node.clone()).or_insert(0);
                    if *current != new_label {
                        *current = new_label;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(labels)
    }

    /// Detect whether the directed graph contains any cycles.
    ///
    /// Uses iterative DFS with a three-color marking scheme.  The result is
    /// cached until the next mutation (`add_entity`, `add_relationship`, or
    /// `remove_entity`).
    ///
    /// # Returns
    /// - `Ok(true)` — at least one cycle exists
    /// - `Ok(false)` — the graph is acyclic (a DAG)
    pub fn detect_cycles(&self) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "detect_cycles");

        if let Some(cached) = inner.cycle_cache {
            return Ok(cached);
        }

        // Iterative DFS with three-color marking: 0=white, 1=gray, 2=black.
        let mut color: HashMap<&EntityId, u8> =
            inner.entities.keys().map(|id| (id, 0u8)).collect();

        let has_cycle = 'outer: {
            for start in inner.entities.keys() {
                if *color.get(start).unwrap_or(&0) != 0 {
                    continue;
                }

                // Stack holds (node_id, iterator position for adjacency).
                let mut stack: Vec<(&EntityId, usize)> = vec![(start, 0)];
                *color.entry(start).or_insert(0) = 1;

                while let Some((node, idx)) = stack.last_mut() {
                    // Use the adjacency index (O(1) lookup) instead of an
                    // O(|E|) relationship scan on every while-loop step.
                    let rels = inner
                        .adjacency
                        .get(*node)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);

                    if *idx < rels.len() {
                        let next = &rels[*idx].to;
                        *idx += 1;
                        match color.get(next).copied().unwrap_or(0) {
                            1 => break 'outer true, // back edge → cycle
                            0 => {
                                *color.entry(next).or_insert(0) = 1;
                                stack.push((next, 0));
                            }
                            _ => {} // already fully processed
                        }
                    } else {
                        // All neighbors processed; color black.
                        *color.entry(*node).or_insert(0) = 2;
                        stack.pop();
                    }
                }
            }
            false
        };

        inner.cycle_cache = Some(has_cycle);
        Ok(has_cycle)
    }

    /// Compute a topological ordering of all entities.
    ///
    /// Returns the entities sorted so that for every directed edge `(A → B)`,
    /// `A` appears before `B` in the result.  Uses iterative DFS (post-order).
    ///
    /// # Errors
    /// Returns `Err(AgentRuntimeError::Graph(...))` if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "topological_sort");

        // Three-color DFS: 0=white, 1=gray (in-stack), 2=black (done).
        let mut color: HashMap<&EntityId, u8> =
            inner.entities.keys().map(|id| (id, 0u8)).collect();
        let mut result: Vec<EntityId> = Vec::with_capacity(inner.entities.len());

        for start in inner.entities.keys() {
            if *color.get(start).unwrap_or(&0) != 0 {
                continue;
            }
            // Stack: (node, adjacency_index, already_pushed_post_order)
            let mut stack: Vec<(&EntityId, usize, bool)> = vec![(start, 0, false)];
            *color.entry(start).or_insert(0) = 1;

            while let Some((node, idx, pushed)) = stack.last_mut() {
                let rels = inner.adjacency.get(*node).map(|v| v.as_slice()).unwrap_or(&[]);
                if *idx < rels.len() {
                    let next = &rels[*idx].to;
                    *idx += 1;
                    match color.get(next).copied().unwrap_or(0) {
                        1 => return Err(AgentRuntimeError::Graph(
                            "topological_sort: graph contains a cycle".into(),
                        )),
                        0 => {
                            *color.entry(next).or_insert(0) = 1;
                            stack.push((next, 0, false));
                        }
                        _ => {} // already finished
                    }
                } else {
                    if !*pushed {
                        result.push((*node).clone());
                        *pushed = true;
                    }
                    *color.entry(*node).or_insert(0) = 2;
                    stack.pop();
                }
            }
        }

        result.reverse();
        Ok(result)
    }

    /// Update a single property on an existing entity.
    ///
    /// Returns `Ok(true)` if the entity was found and updated, `Ok(false)` otherwise.
    pub fn update_entity_property(
        &self,
        id: &EntityId,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Result<bool, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "update_entity_property");
        if let Some(entity) = inner.entities.get_mut(id) {
            entity.properties.insert(key.into(), value);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Return `true` if the graph is a DAG (directed acyclic graph).
    ///
    /// Equivalent to `!detect_cycles()` but reads more naturally in
    /// condition expressions.
    pub fn is_dag(&self) -> Result<bool, AgentRuntimeError> {
        Ok(!self.detect_cycles()?)
    }

    /// Count the number of weakly connected components in the graph.
    ///
    /// Treats all edges as undirected for component assignment.  Isolated
    /// nodes (no edges) each count as their own component.  Uses Union-Find.
    pub fn connected_components(&self) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "connected_components");
        let ids: Vec<&EntityId> = inner.entities.keys().collect();
        if ids.is_empty() {
            return Ok(0);
        }

        // Map EntityId → index for Union-Find.
        let idx: HashMap<&EntityId, usize> =
            ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let mut parent: Vec<usize> = (0..ids.len()).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        for rel in &inner.relationships {
            if let (Some(&a), Some(&b)) = (idx.get(&rel.from), idx.get(&rel.to)) {
                union(&mut parent, a, b);
            }
        }

        let components = ids
            .iter()
            .enumerate()
            .filter(|(i, _)| find(&mut parent, *i) == *i)
            .count();
        Ok(components)
    }

    /// Return `true` if the graph is weakly connected (i.e. has at most one
    /// connected component when all edges are treated as undirected).
    ///
    /// An empty graph is considered weakly connected by convention.
    ///
    /// # Errors
    /// Propagates any lock-poisoning error from [`connected_components`].
    pub fn weakly_connected(&self) -> Result<bool, AgentRuntimeError> {
        Ok(self.connected_components()? <= 1)
    }

    /// Return all entities that have no outgoing edges (out-degree == 0).
    ///
    /// In a DAG these are the leaf nodes. Useful for identifying terminal
    /// states in a workflow or dependency graph.
    pub fn sink_nodes(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "sink_nodes");
        // Use adjacency index: entities with no entry (or an empty entry) have no outgoing edges.
        Ok(inner
            .entities
            .values()
            .filter(|e| {
                inner
                    .adjacency
                    .get(&e.id)
                    .map_or(true, |v| v.is_empty())
            })
            .cloned()
            .collect())
    }

    /// Return all entities that have no incoming edges (in-degree == 0).
    ///
    /// In a DAG these are the root nodes. Useful for finding entry points in
    /// a workflow or dependency graph.
    pub fn source_nodes(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "source_nodes");
        // Use reverse_adjacency: entities with no entry (or an empty entry) have no incoming edges.
        Ok(inner
            .entities
            .values()
            .filter(|e| {
                inner
                    .reverse_adjacency
                    .get(&e.id)
                    .map_or(true, |v| v.is_empty())
            })
            .cloned()
            .collect())
    }

    /// Return all entities that have no edges (neither incoming nor outgoing).
    pub fn isolates(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "isolates");
        // An entity is isolated if it has no outgoing edges (adjacency) and no
        // incoming edges (reverse_adjacency).
        Ok(inner
            .entities
            .values()
            .filter(|e| {
                inner
                    .adjacency
                    .get(&e.id)
                    .map_or(true, |v| v.is_empty())
                    && inner
                        .reverse_adjacency
                        .get(&e.id)
                        .map_or(true, |v| v.is_empty())
            })
            .cloned()
            .collect())
    }

    /// Return the number of outgoing edges (out-degree) for `id`.
    ///
    /// Returns `0` if the entity has no outgoing edges or does not exist.
    pub fn out_degree(&self, id: &EntityId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "out_degree");
        Ok(inner.adjacency.get(id).map_or(0, |rels| rels.len()))
    }

    /// Return the number of incoming edges (in-degree) for `id`.
    ///
    /// Uses the reverse adjacency index for O(in-degree) lookup.
    pub fn in_degree(&self, id: &EntityId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "in_degree");
        Ok(inner
            .reverse_adjacency
            .get(id)
            .map_or(0, |srcs| srcs.len()))
    }

    /// Return `true` if there is any path from `from` to `to`.
    ///
    /// Both nodes must exist or returns `Err`. Uses BFS internally.
    /// Returns `Ok(false)` if nodes exist but are not connected.
    pub fn path_exists(&self, from: &str, to: &str) -> Result<bool, AgentRuntimeError> {
        let from_id = EntityId::new(from);
        let to_id = EntityId::new(to);
        match self.shortest_path(&from_id, &to_id) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Return the hop-count of the shortest path from `from` to `to`.
    ///
    /// Returns `Some(hops)` if a path exists, `None` if the nodes are
    /// disconnected.  Both node IDs must exist or returns `Err`.
    pub fn shortest_path_length(
        &self,
        from: &EntityId,
        to: &EntityId,
    ) -> Result<Option<usize>, AgentRuntimeError> {
        Ok(self.shortest_path(from, to)?.map(|path| path.len().saturating_sub(1)))
    }

    /// BFS limited by maximum depth and maximum node count.
    ///
    /// Returns the subset of nodes visited within those limits (including start).
    pub fn bfs_bounded(
        &self,
        start: &str,
        max_depth: usize,
        max_nodes: usize,
    ) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "bfs_bounded");
        let start_id = EntityId::new(start);
        if !inner.entities.contains_key(&start_id) {
            return Err(AgentRuntimeError::Graph(format!(
                "start entity '{start}' not found"
            )));
        }

        let mut visited: std::collections::HashMap<EntityId, usize> = std::collections::HashMap::new();
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();
        let mut result: Vec<EntityId> = Vec::new();

        visited.insert(start_id.clone(), 0);
        queue.push_back((start_id.clone(), 0));
        result.push(start_id);

        while let Some((current, depth)) = queue.pop_front() {
            if result.len() >= max_nodes {
                break;
            }
            if depth >= max_depth {
                continue;
            }
            for neighbour in Self::neighbours(&inner.adjacency, &current) {
                if !visited.contains_key(&neighbour) {
                    let new_depth = depth + 1;
                    visited.insert(neighbour.clone(), new_depth);
                    result.push(neighbour.clone());
                    if result.len() >= max_nodes {
                        break;
                    }
                    queue.push_back((neighbour, new_depth));
                }
            }
        }

        Ok(result)
    }

    /// DFS limited by maximum depth and maximum node count.
    ///
    /// Returns the subset of nodes visited within those limits (including start).
    pub fn dfs_bounded(
        &self,
        start: &str,
        max_depth: usize,
        max_nodes: usize,
    ) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "dfs_bounded");
        let start_id = EntityId::new(start);
        if !inner.entities.contains_key(&start_id) {
            return Err(AgentRuntimeError::Graph(format!(
                "start entity '{start}' not found"
            )));
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut stack: Vec<(EntityId, usize)> = Vec::new();
        let mut result: Vec<EntityId> = Vec::new();

        visited.insert(start_id.clone());
        stack.push((start_id.clone(), 0));
        result.push(start_id);

        while let Some((current, depth)) = stack.pop() {
            if result.len() >= max_nodes {
                break;
            }
            if depth >= max_depth {
                continue;
            }
            for neighbour in Self::neighbours(&inner.adjacency, &current) {
                if visited.insert(neighbour.clone()) {
                    result.push(neighbour.clone());
                    if result.len() >= max_nodes {
                        break;
                    }
                    stack.push((neighbour, depth + 1));
                }
            }
        }

        Ok(result)
    }

    /// Return all entities in the graph.
    pub fn all_entities(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "all_entities");
        Ok(inner.entities.values().cloned().collect())
    }

    /// Return all relationships in the graph.
    pub fn all_relationships(&self) -> Result<Vec<Relationship>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "all_relationships");
        Ok(inner.relationships.clone())
    }

    /// Return all relationships whose `kind` matches `kind` (case-sensitive).
    pub fn find_relationships_by_kind(&self, kind: &str) -> Result<Vec<Relationship>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "find_relationships_by_kind");
        Ok(inner.relationships.iter().filter(|r| r.kind == kind).cloned().collect())
    }

    /// Return the number of relationships whose `kind` matches `kind` (case-sensitive).
    pub fn count_relationships_by_kind(&self, kind: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "count_relationships_by_kind");
        Ok(inner.relationships.iter().filter(|r| r.kind == kind).count())
    }

    /// Return all entities whose `label` matches `label` (case-sensitive).
    pub fn find_entities_by_label(&self, label: &str) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "find_entities_by_label");
        Ok(inner
            .entities
            .values()
            .filter(|e| e.label == label)
            .cloned()
            .collect())
    }

    /// Return all entities whose label is contained in `labels`.
    ///
    /// Order of results is unspecified.  An empty `labels` slice returns an
    /// empty result (never all entities).
    pub fn find_entities_by_labels(&self, labels: &[&str]) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "find_entities_by_labels");
        let label_set: std::collections::HashSet<&str> = labels.iter().copied().collect();
        Ok(inner
            .entities
            .values()
            .filter(|e| label_set.contains(e.label.as_str()))
            .cloned()
            .collect())
    }

    /// Remove all isolated entities (those with no incoming or outgoing edges).
    ///
    /// Returns the number of entities removed.
    pub fn remove_isolated(&self) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "remove_isolated");
        let isolated: Vec<EntityId> = inner
            .entities
            .keys()
            .filter(|id| {
                inner.adjacency.get(*id).map_or(true, |v| v.is_empty())
                    && inner.reverse_adjacency.get(*id).map_or(true, |v| v.is_empty())
            })
            .cloned()
            .collect();
        let count = isolated.len();
        for id in &isolated {
            inner.entities.remove(id);
            inner.adjacency.remove(id);
            inner.reverse_adjacency.remove(id);
        }
        if count > 0 {
            inner.cycle_cache = None;
        }
        Ok(count)
    }

    /// Return all outgoing relationships from `id` (those where `id` is the source).
    pub fn get_relationships_for(
        &self,
        id: &EntityId,
    ) -> Result<Vec<Relationship>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "get_relationships_for");
        Ok(inner
            .adjacency
            .get(id)
            .cloned()
            .unwrap_or_default())
    }

    /// Return all relationships where both endpoints are `from` and `to` (any direction or kind).
    pub fn relationships_between(
        &self,
        from: &EntityId,
        to: &EntityId,
    ) -> Result<Vec<Relationship>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "relationships_between");
        Ok(inner
            .relationships
            .iter()
            .filter(|r| {
                (r.from == *from && r.to == *to) || (r.from == *to && r.to == *from)
            })
            .cloned()
            .collect())
    }

    /// Find entities that have a property `key` whose value equals `expected`.
    ///
    /// Uses `serde_json::Value` equality; for string properties pass
    /// `serde_json::Value::String(...)`.
    pub fn find_entities_by_property(
        &self,
        key: &str,
        expected: &serde_json::Value,
    ) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "find_entities_by_property");
        Ok(inner
            .entities
            .values()
            .filter(|e| e.properties.get(key) == Some(expected))
            .cloned()
            .collect())
    }

    /// Merge another `GraphStore` into this one.
    ///
    /// Entities are inserted (or replaced if the same ID exists); relationships
    /// are inserted only if the `(from, to, kind)` triple does not already exist.
    pub fn merge(&self, other: &GraphStore) -> Result<(), AgentRuntimeError> {
        let other_inner = recover_lock(other.inner.lock(), "merge:read");
        let other_entities: Vec<Entity> = other_inner.entities.values().cloned().collect();
        let other_rels: Vec<Relationship> = other_inner.relationships.clone();
        drop(other_inner);

        let mut inner = recover_lock(self.inner.lock(), "merge:write");
        inner.cycle_cache = None;
        for entity in other_entities {
            inner.adjacency.entry(entity.id.clone()).or_default();
            inner.entities.insert(entity.id.clone(), entity);
        }
        for rel in other_rels {
            let already_exists = inner
                .relationships
                .iter()
                .any(|r| r.from == rel.from && r.to == rel.to && r.kind == rel.kind);
            if !already_exists && inner.entities.contains_key(&rel.from) && inner.entities.contains_key(&rel.to) {
                inner
                    .adjacency
                    .entry(rel.from.clone())
                    .or_default()
                    .push(rel.clone());
                inner.relationships.push(rel);
            }
        }
        Ok(())
    }

    /// Return the `Entity` objects for all direct out-neighbors of `id`.
    ///
    /// Neighbors are the targets of all outgoing relationships from `id`.
    /// Entities that no longer exist (orphan edge targets) are silently skipped.
    pub fn neighbor_entities(&self, id: &EntityId) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "neighbor_entities");
        let neighbors: Vec<Entity> = inner
            .adjacency
            .get(id)
            .iter()
            .flat_map(|rels| rels.iter())
            .filter_map(|r| inner.entities.get(&r.to).cloned())
            .collect();
        Ok(neighbors)
    }

    /// Return the IDs of all directly reachable neighbours from `id`
    /// (i.e. the `to` end of every outgoing relationship).
    ///
    /// Cheaper than [`neighbor_entities`] when only IDs are needed.
    ///
    /// [`neighbor_entities`]: GraphStore::neighbor_entities
    pub fn neighbor_ids(&self, id: &EntityId) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "neighbor_ids");
        let ids: Vec<EntityId> = inner
            .adjacency
            .get(id)
            .iter()
            .flat_map(|rels| rels.iter())
            .map(|r| r.to.clone())
            .collect();
        Ok(ids)
    }

    /// Remove **all** relationships where `id` is the source (outgoing edges).
    ///
    /// Also updates the adjacency index.  Does **not** remove incoming edges
    /// from other nodes that point to `id` — use this before [`remove_entity`]
    /// to ensure consistent graph state.
    ///
    /// Returns the number of relationships removed.
    ///
    /// [`remove_entity`]: GraphStore::remove_entity
    pub fn remove_all_relationships_for(&self, id: &EntityId) -> Result<usize, AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "remove_all_relationships_for");
        inner.adjacency.remove(id);
        let before = inner.relationships.len();
        inner.relationships.retain(|r| &r.from != id);
        inner.cycle_cache = None;
        Ok(before - inner.relationships.len())
    }

    /// Check whether an entity with the given ID exists.
    pub fn entity_exists(&self, id: &EntityId) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "entity_exists");
        Ok(inner.entities.contains_key(id))
    }

    /// Check whether a relationship `(from, to, kind)` exists.
    ///
    /// Uses the adjacency index (O(out-degree of `from`)) rather than a full
    /// O(|E|) scan over all relationships.
    pub fn relationship_exists(
        &self,
        from: &EntityId,
        to: &EntityId,
        kind: &str,
    ) -> Result<bool, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "relationship_exists");
        Ok(inner
            .adjacency
            .get(from)
            .map_or(false, |rels| rels.iter().any(|r| r.to == *to && r.kind == kind)))
    }

    /// Extract a subgraph containing only the specified entities and the
    /// relationships between them.
    pub fn subgraph(&self, node_ids: &[EntityId]) -> Result<GraphStore, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "subgraph");
        let id_set: HashSet<&EntityId> = node_ids.iter().collect();

        let new_store = GraphStore::new();

        // Validate all entities exist before mutating the new store, so we
        // don't end up with a partially-populated subgraph on error.
        let entities_to_copy: Vec<Entity> = node_ids
            .iter()
            .map(|id| {
                inner
                    .entities
                    .get(id)
                    .cloned()
                    .ok_or_else(|| {
                        AgentRuntimeError::Graph(format!("entity '{}' not found", id.0))
                    })
            })
            .collect::<Result<_, _>>()?;

        // Acquire the new store's lock once for all entity insertions.
        {
            let mut new_inner = recover_lock(new_store.inner.lock(), "subgraph:add_entities");
            for entity in entities_to_copy {
                // Ensure an adjacency entry exists for this entity.
                new_inner.adjacency.entry(entity.id.clone()).or_default();
                new_inner.entities.insert(entity.id.clone(), entity);
            }
        }

        // Acquire the new store's lock once for the entire relationship batch
        // rather than once per relationship.
        {
            let mut new_inner =
                recover_lock(new_store.inner.lock(), "subgraph:add_relationships");
            for rel in inner.relationships.iter() {
                if id_set.contains(&rel.from) && id_set.contains(&rel.to) {
                    // Keep adjacency index in sync with relationships.
                    new_inner
                        .adjacency
                        .entry(rel.from.clone())
                        .or_default()
                        .push(rel.clone());
                    new_inner.relationships.push(rel.clone());
                }
            }
        }

        Ok(new_store)
    }

    /// Return a new graph with all edge directions reversed.
    ///
    /// Every relationship `A → B` becomes `B → A` in the returned graph.
    /// All entities are copied; relationship weights and kinds are preserved.
    pub fn reverse(&self) -> Result<GraphStore, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "reverse");
        let reversed = GraphStore::new();

        // Copy all entities.
        {
            let mut r_inner = recover_lock(reversed.inner.lock(), "reverse:entities");
            for entity in inner.entities.values() {
                r_inner.adjacency.entry(entity.id.clone()).or_default();
                r_inner.entities.insert(entity.id.clone(), entity.clone());
            }
        }

        // Add reversed relationships.
        for rel in &inner.relationships {
            let flipped = Relationship {
                from: rel.to.clone(),
                to: rel.from.clone(),
                kind: rel.kind.clone(),
                weight: rel.weight,
            };
            let mut r_inner = recover_lock(reversed.inner.lock(), "reverse:rels");
            r_inner
                .adjacency
                .entry(flipped.from.clone())
                .or_default()
                .push(flipped.clone());
            r_inner.relationships.push(flipped);
        }

        Ok(reversed)
    }

    /// Return entities that are out-neighbors of **both** `a` and `b`.
    ///
    /// Useful for link-prediction and community detection heuristics.
    /// Returns an empty `Vec` if either node has no outgoing edges.
    pub fn common_neighbors(
        &self,
        a: &EntityId,
        b: &EntityId,
    ) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "common_neighbors");
        let a_set: HashSet<&EntityId> = inner
            .adjacency
            .get(a)
            .map_or(HashSet::new(), |rels| rels.iter().map(|r| &r.to).collect());
        let b_set: HashSet<&EntityId> = inner
            .adjacency
            .get(b)
            .map_or(HashSet::new(), |rels| rels.iter().map(|r| &r.to).collect());
        let common: Vec<Entity> = a_set
            .intersection(&b_set)
            .filter_map(|id| inner.entities.get(*id).cloned())
            .collect();
        Ok(common)
    }

    /// Return the weight of the first relationship from `from` to `to`.
    ///
    /// Returns `None` if no such edge exists.
    pub fn weight_of(
        &self,
        from: &EntityId,
        to: &EntityId,
    ) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "weight_of");
        let weight = inner
            .adjacency
            .get(from)
            .and_then(|rels| rels.iter().find(|r| &r.to == to))
            .map(|r| r.weight);
        Ok(weight)
    }

    /// Return the IDs of all entities that have an outgoing edge pointing **to** `id`.
    ///
    /// This is the inverse of `neighbor_entities`, which returns out-neighbors.
    /// Returns an empty `Vec` if no incoming edges exist for `id`.
    pub fn neighbors_in(&self, id: &EntityId) -> Result<Vec<EntityId>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "neighbors_in");
        // Use reverse_adjacency for O(in-degree) lookup instead of O(|E|) scan.
        Ok(inner
            .reverse_adjacency
            .get(id)
            .cloned()
            .unwrap_or_default())
    }

    /// Return the graph density: `edges / (nodes * (nodes − 1))` for a directed graph.
    ///
    /// Returns `0.0` when the graph has fewer than 2 nodes (no directed edges are
    /// possible). Values range from `0.0` (sparse) to `1.0` (complete).
    pub fn density(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "density");
        let n = inner.entities.len();
        if n < 2 {
            return Ok(0.0);
        }
        let max_edges = n * (n - 1);
        Ok(inner.relationships.len() as f64 / max_edges as f64)
    }

    /// Return the average out-degree across all nodes.
    ///
    /// Returns `0.0` for an empty graph.
    pub fn avg_degree(&self) -> Result<f64, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "avg_degree");
        let n = inner.entities.len();
        if n == 0 {
            return Ok(0.0);
        }
        Ok(inner.relationships.len() as f64 / n as f64)
    }

    /// Return the sum of all edge weights in the graph.
    ///
    /// Returns `0.0` for an empty graph or a graph with no relationships.
    pub fn total_weight(&self) -> Result<f32, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "total_weight");
        Ok(inner.relationships.iter().map(|r| r.weight).sum())
    }

    /// Return the maximum edge weight, or `None` if the graph has no edges.
    pub fn max_edge_weight(&self) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "max_edge_weight");
        Ok(inner
            .relationships
            .iter()
            .map(|r| r.weight)
            .reduce(f32::max))
    }

    /// Return the minimum edge weight, or `None` if the graph has no edges.
    pub fn min_edge_weight(&self) -> Result<Option<f32>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "min_edge_weight");
        Ok(inner
            .relationships
            .iter()
            .map(|r| r.weight)
            .reduce(f32::min))
    }

    /// Return the top-N entities by out-degree (most outgoing edges first).
    ///
    /// If `n == 0` or the graph is empty, returns an empty Vec.  Ties are
    /// broken by arbitrary hash-map iteration order.
    pub fn top_n_by_out_degree(&self, n: usize) -> Result<Vec<Entity>, AgentRuntimeError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let inner = recover_lock(self.inner.lock(), "top_n_by_out_degree");
        let mut pairs: Vec<(&EntityId, usize)> = inner
            .adjacency
            .iter()
            .map(|(id, rels)| (id, rels.len()))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        Ok(pairs
            .into_iter()
            .take(n)
            .filter_map(|(id, _)| inner.entities.get(id).cloned())
            .collect())
    }

    /// Remove `id` and all relationships where it is the source or target.
    ///
    /// Equivalent to calling `remove_entity` and `remove_all_relationships_for`
    /// in a single lock acquisition, avoiding two separate look-ups.
    ///
    /// Returns `Ok(())` if the entity was found and removed.  Returns
    /// `Err(AgentRuntimeError::Graph)` if no entity with that ID exists.
    pub fn remove_entity_and_edges(&self, id: &EntityId) -> Result<(), AgentRuntimeError> {
        let mut inner = recover_lock(self.inner.lock(), "remove_entity_and_edges");
        if !inner.entities.contains_key(id) {
            return Err(AgentRuntimeError::Graph(format!(
                "entity '{}' not found",
                id.0
            )));
        }
        inner.entities.remove(id);
        inner.relationships.retain(|r| &r.from != id && &r.to != id);
        inner.adjacency.remove(id);
        for adj in inner.adjacency.values_mut() {
            adj.retain(|r| &r.to != id);
        }
        inner.reverse_adjacency.remove(id);
        for rev in inner.reverse_adjacency.values_mut() {
            rev.retain(|src| src != id);
        }
        inner.cycle_cache = None;
        Ok(())
    }

    /// Return all entities whose out-degree is at or above `threshold`.
    ///
    /// Useful for finding hub or gateway nodes in a knowledge graph.
    pub fn hub_nodes(&self, threshold: usize) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "hub_nodes");
        Ok(inner
            .entities
            .values()
            .filter(|e| {
                inner
                    .adjacency
                    .get(&e.id)
                    .map_or(0, |rels| rels.len())
                    >= threshold
            })
            .cloned()
            .collect())
    }

    /// Return all relationships incident to `entity_id`
    /// (where the entity is either the source or the target).
    pub fn incident_relationships(
        &self,
        entity_id: &EntityId,
    ) -> Result<Vec<Relationship>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "incident_relationships");
        Ok(inner
            .relationships
            .iter()
            .filter(|r| &r.from == entity_id || &r.to == entity_id)
            .cloned()
            .collect())
    }

    /// Return the entity with the highest out-degree (most outgoing edges),
    /// or `None` if the graph is empty.
    ///
    /// If multiple entities share the maximum out-degree, the first one
    /// encountered (in arbitrary hash-map iteration order) is returned.
    pub fn max_out_degree_entity(&self) -> Result<Option<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "max_out_degree_entity");
        let best = inner
            .adjacency
            .iter()
            .max_by_key(|(_, rels)| rels.len())
            .and_then(|(id, _)| inner.entities.get(id).cloned());
        Ok(best)
    }

    /// Return the entity with the most incoming edges (highest in-degree).
    ///
    /// Uses the reverse adjacency index for O(V) computation.
    /// Returns `None` for an empty graph or a graph with no edges.
    pub fn max_in_degree_entity(&self) -> Result<Option<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "max_in_degree_entity");
        let best = inner
            .reverse_adjacency
            .iter()
            .max_by_key(|(_, srcs)| srcs.len())
            .and_then(|(id, _)| inner.entities.get(id).cloned());
        Ok(best)
    }

    /// Return all entities with out-degree = 0 (no outgoing edges).
    ///
    /// Leaf nodes (also called sink nodes in directed graphs) are entities
    /// that have no outgoing relationships.  See also [`sink_nodes`] which
    /// also counts nodes with no outgoing edges but filters differently.
    ///
    /// [`sink_nodes`]: GraphStore::sink_nodes
    pub fn leaf_nodes(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "leaf_nodes");
        Ok(inner
            .entities
            .values()
            .filter(|e| {
                inner
                    .adjacency
                    .get(&e.id)
                    .map_or(true, |rels| rels.is_empty())
            })
            .cloned()
            .collect())
    }

    /// Return the top `n` entities sorted by out-degree (descending).
    ///
    /// Uses the adjacency index for O(V log V) computation.  Useful for finding
    /// hub nodes — entities with the most outgoing connections.
    /// Returns fewer than `n` entities if the graph has fewer nodes.
    pub fn top_nodes_by_out_degree(&self, n: usize) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "top_nodes_by_out_degree");
        let mut pairs: Vec<(&EntityId, usize)> = inner
            .entities
            .keys()
            .map(|id| (id, inner.adjacency.get(id).map_or(0, |v| v.len())))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(n);
        Ok(pairs
            .into_iter()
            .filter_map(|(id, _)| inner.entities.get(id).cloned())
            .collect())
    }

    /// Return the top `n` entities sorted by in-degree (descending).
    ///
    /// Uses the reverse adjacency index for O(V log V) computation.  Useful for
    /// finding sink hubs — entities with the most incoming connections.
    pub fn top_nodes_by_in_degree(&self, n: usize) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "top_nodes_by_in_degree");
        let mut pairs: Vec<(&EntityId, usize)> = inner
            .entities
            .keys()
            .map(|id| (id, inner.reverse_adjacency.get(id).map_or(0, |v| v.len())))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(n);
        Ok(pairs
            .into_iter()
            .filter_map(|(id, _)| inner.entities.get(id).cloned())
            .collect())
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
        g.add_relationship(Relationship::new(from, to, "CONNECTS", 1.0))
            .unwrap();
    }

    fn link_w(g: &GraphStore, from: &str, to: &str, weight: f32) {
        g.add_relationship(Relationship::new(from, to, "CONNECTS", weight))
            .unwrap();
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
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        link(&g, "a", "b");
        link(&g, "a", "c");
        let visited = g.bfs(&EntityId::new("a")).unwrap();
        assert_eq!(visited.len(), 2);
    }

    #[test]
    fn test_bfs_traverses_chain() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        add(&g, "d");
        link(&g, "a", "b");
        link(&g, "b", "c");
        link(&g, "c", "d");
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
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        add(&g, "d");
        link(&g, "a", "b");
        link(&g, "a", "c");
        link(&g, "b", "d");
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
        add(&g, "a");
        add(&g, "b");
        link(&g, "a", "b");
        let path = g
            .shortest_path(&EntityId::new("a"), &EntityId::new("b"))
            .unwrap();
        assert_eq!(path, Some(vec![EntityId::new("a"), EntityId::new("b")]));
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        link(&g, "a", "b");
        link(&g, "b", "c");
        let path = g
            .shortest_path(&EntityId::new("a"), &EntityId::new("c"))
            .unwrap();
        assert_eq!(path.as_ref().map(|p| p.len()), Some(3));
    }

    #[test]
    fn test_shortest_path_returns_none_for_disconnected() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        let path = g
            .shortest_path(&EntityId::new("a"), &EntityId::new("b"))
            .unwrap();
        assert_eq!(path, None);
    }

    #[test]
    fn test_shortest_path_same_node_returns_single_element() {
        let g = make_graph();
        add(&g, "a");
        let path = g
            .shortest_path(&EntityId::new("a"), &EntityId::new("a"))
            .unwrap();
        assert_eq!(path, Some(vec![EntityId::new("a")]));
    }

    #[test]
    fn test_shortest_path_missing_source_returns_error() {
        let g = make_graph();
        add(&g, "b");
        assert!(g
            .shortest_path(&EntityId::new("ghost"), &EntityId::new("b"))
            .is_err());
    }

    #[test]
    fn test_shortest_path_missing_target_returns_error() {
        let g = make_graph();
        add(&g, "a");
        assert!(g
            .shortest_path(&EntityId::new("a"), &EntityId::new("ghost"))
            .is_err());
    }

    // ── Transitive closure ────────────────────────────────────────────────────

    #[test]
    fn test_transitive_closure_includes_start() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
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

    // ── Weighted shortest path ────────────────────────────────────────────────

    #[test]
    fn test_shortest_path_weighted_simple() {
        // a --(1.0)--> b --(2.0)--> c
        // a --(10.0)--> c  (direct but heavier)
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        link_w(&g, "a", "b", 1.0);
        link_w(&g, "b", "c", 2.0);
        g.add_relationship(Relationship::new("a", "c", "DIRECT", 10.0))
            .unwrap();

        let result = g
            .shortest_path_weighted(&EntityId::new("a"), &EntityId::new("c"))
            .unwrap();
        assert!(result.is_some());
        let (path, weight) = result.unwrap();
        // The cheapest path is a -> b -> c with total weight 3.0
        assert_eq!(
            path,
            vec![EntityId::new("a"), EntityId::new("b"), EntityId::new("c")]
        );
        assert!((weight - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_shortest_path_weighted_returns_none_for_disconnected() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        let result = g
            .shortest_path_weighted(&EntityId::new("a"), &EntityId::new("b"))
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_shortest_path_weighted_same_node() {
        let g = make_graph();
        add(&g, "a");
        let result = g
            .shortest_path_weighted(&EntityId::new("a"), &EntityId::new("a"))
            .unwrap();
        assert_eq!(result, Some((vec![EntityId::new("a")], 0.0)));
    }

    #[test]
    fn test_shortest_path_weighted_negative_weight_errors() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        g.add_relationship(Relationship::new("a", "b", "NEG", -1.0))
            .unwrap();
        let result = g.shortest_path_weighted(&EntityId::new("a"), &EntityId::new("b"));
        assert!(result.is_err());
    }

    // ── Degree centrality ─────────────────────────────────────────────────────

    #[test]
    fn test_degree_centrality_basic() {
        // Star graph: a -> b, a -> c, a -> d
        // a: out=3, in=0 => (3+0)/(4-1) = 1.0
        // b: out=0, in=1 => (0+1)/3 = 0.333...
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        add(&g, "d");
        link(&g, "a", "b");
        link(&g, "a", "c");
        link(&g, "a", "d");

        let centrality = g.degree_centrality().unwrap();
        let a_cent = *centrality.get(&EntityId::new("a")).unwrap();
        let b_cent = *centrality.get(&EntityId::new("b")).unwrap();

        assert!((a_cent - 1.0).abs() < 1e-5, "a centrality was {a_cent}");
        assert!(
            (b_cent - 1.0 / 3.0).abs() < 1e-5,
            "b centrality was {b_cent}"
        );
    }

    // ── Betweenness centrality ────────────────────────────────────────────────

    #[test]
    fn test_betweenness_centrality_chain() {
        // Chain: a -> b -> c -> d
        // b and c are on all paths from a to c, a to d, b to d
        // Node b and c should have higher centrality than a and d.
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        add(&g, "d");
        link(&g, "a", "b");
        link(&g, "b", "c");
        link(&g, "c", "d");

        let centrality = g.betweenness_centrality().unwrap();
        let a_cent = *centrality.get(&EntityId::new("a")).unwrap();
        let b_cent = *centrality.get(&EntityId::new("b")).unwrap();
        let c_cent = *centrality.get(&EntityId::new("c")).unwrap();
        let d_cent = *centrality.get(&EntityId::new("d")).unwrap();

        // Endpoints should have 0 centrality; intermediate nodes should be > 0.
        assert!((a_cent).abs() < 1e-5, "expected a_cent ~ 0, got {a_cent}");
        assert!(b_cent > 0.0, "expected b_cent > 0, got {b_cent}");
        assert!(c_cent > 0.0, "expected c_cent > 0, got {c_cent}");
        assert!((d_cent).abs() < 1e-5, "expected d_cent ~ 0, got {d_cent}");
    }

    // ── Label propagation communities ─────────────────────────────────────────

    #[test]
    fn test_label_propagation_communities_two_clusters() {
        // Cluster 1: a <-> b <-> c (fully connected via bidirectional edges)
        // Cluster 2: x <-> y <-> z
        // No edges between clusters.
        let g = make_graph();
        for id in &["a", "b", "c", "x", "y", "z"] {
            add(&g, id);
        }
        // Cluster 1 (bidirectional via two directed edges each)
        link(&g, "a", "b");
        link(&g, "b", "a");
        link(&g, "b", "c");
        link(&g, "c", "b");
        link(&g, "a", "c");
        link(&g, "c", "a");
        // Cluster 2
        link(&g, "x", "y");
        link(&g, "y", "x");
        link(&g, "y", "z");
        link(&g, "z", "y");
        link(&g, "x", "z");
        link(&g, "z", "x");

        let communities = g.label_propagation_communities(100).unwrap();

        let label_a = communities[&EntityId::new("a")];
        let label_b = communities[&EntityId::new("b")];
        let label_c = communities[&EntityId::new("c")];
        let label_x = communities[&EntityId::new("x")];
        let label_y = communities[&EntityId::new("y")];
        let label_z = communities[&EntityId::new("z")];

        // All nodes in cluster 1 share a label, all in cluster 2 share a label,
        // and the two clusters have different labels.
        assert_eq!(label_a, label_b, "a and b should be in same community");
        assert_eq!(label_b, label_c, "b and c should be in same community");
        assert_eq!(label_x, label_y, "x and y should be in same community");
        assert_eq!(label_y, label_z, "y and z should be in same community");
        assert_ne!(
            label_a, label_x,
            "cluster 1 and cluster 2 should be different communities"
        );
    }

    // ── Subgraph extraction ───────────────────────────────────────────────────

    #[test]
    fn test_subgraph_extracts_correct_nodes_and_edges() {
        // Full graph: a -> b -> c -> d
        // Subgraph of {a, b, c} should contain edges a->b and b->c but not c->d.
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        add(&g, "d");
        link(&g, "a", "b");
        link(&g, "b", "c");
        link(&g, "c", "d");

        let sub = g
            .subgraph(&[EntityId::new("a"), EntityId::new("b"), EntityId::new("c")])
            .unwrap();

        assert_eq!(sub.entity_count().unwrap(), 3);
        assert_eq!(sub.relationship_count().unwrap(), 2);

        // d should not be present in the subgraph.
        assert!(sub.get_entity(&EntityId::new("d")).is_err());

        // a -> b and b -> c should be present; c -> d should not.
        let path = sub
            .shortest_path(&EntityId::new("a"), &EntityId::new("c"))
            .unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 3);
    }

    // ── detect_cycles ──────────────────────────────────────────────────────────

    #[test]
    fn test_detect_cycles_dag_returns_false() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        link(&g, "a", "b");
        link(&g, "b", "c");
        assert_eq!(g.detect_cycles().unwrap(), false);
    }

    #[test]
    fn test_detect_cycles_self_loop_returns_true() {
        let g = make_graph();
        add(&g, "a");
        // Use a different kind to avoid duplicate-relationship rejection.
        g.add_relationship(Relationship::new("a", "a", "SELF", 1.0))
            .unwrap();
        assert_eq!(g.detect_cycles().unwrap(), true);
    }

    #[test]
    fn test_detect_cycles_simple_cycle_returns_true() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        link(&g, "a", "b");
        g.add_relationship(Relationship::new("b", "a", "BACK", 1.0))
            .unwrap();
        assert_eq!(g.detect_cycles().unwrap(), true);
    }

    #[test]
    fn test_detect_cycles_empty_graph_returns_false() {
        let g = make_graph();
        assert_eq!(g.detect_cycles().unwrap(), false);
    }

    #[test]
    fn test_detect_cycles_result_is_cached() {
        let g = make_graph();
        add(&g, "x");
        add(&g, "y");
        link(&g, "x", "y");
        // First call.
        let r1 = g.detect_cycles().unwrap();
        // Second call should return the cached value.
        let r2 = g.detect_cycles().unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_detect_cycles_cache_invalidated_on_mutation() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        link(&g, "a", "b");
        assert_eq!(g.detect_cycles().unwrap(), false);

        // Add a back edge to create a cycle — cache must be invalidated.
        g.add_relationship(Relationship::new("b", "a", "BACK", 1.0))
            .unwrap();
        assert_eq!(
            g.detect_cycles().unwrap(),
            true,
            "cache should be invalidated after adding a back edge"
        );
    }

    // ── bfs_bounded / dfs_bounded ─────────────────────────────────────────────

    #[test]
    fn test_bfs_bounded_respects_max_depth() {
        // Chain: a -> b -> c -> d
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        add(&g, "d");
        link(&g, "a", "b");
        link(&g, "b", "c");
        link(&g, "c", "d");

        // max_depth=1 should only visit a and b (depth 0 and 1)
        let visited = g.bfs_bounded("a", 1, 100).unwrap();
        assert!(visited.contains(&EntityId::new("a")));
        assert!(visited.contains(&EntityId::new("b")));
        assert!(!visited.contains(&EntityId::new("c")), "c is at depth 2, should not be visited");
    }

    // ── #5/#35 path_exists ────────────────────────────────────────────────────

    #[test]
    fn test_path_exists_returns_true() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        link(&g, "a", "b");
        link(&g, "b", "c");
        assert_eq!(g.path_exists("a", "c").unwrap(), true);
    }

    #[test]
    fn test_path_exists_returns_false() {
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        assert_eq!(g.path_exists("a", "b").unwrap(), false);
    }

    // ── #13 EntityId::as_str ──────────────────────────────────────────────────

    #[test]
    fn test_entity_id_as_str() {
        let id = EntityId::new("my-entity");
        assert_eq!(id.as_str(), "my-entity");
    }

    // ── #38 EntityId AsRef<str> ───────────────────────────────────────────────

    #[test]
    fn test_entity_id_as_ref_str() {
        let id = EntityId::new("asref-test");
        let s: &str = id.as_ref();
        assert_eq!(s, "asref-test");
    }

    #[test]
    fn test_dfs_bounded_respects_max_nodes() {
        // Chain: a -> b -> c -> d
        let g = make_graph();
        add(&g, "a");
        add(&g, "b");
        add(&g, "c");
        add(&g, "d");
        link(&g, "a", "b");
        link(&g, "b", "c");
        link(&g, "c", "d");

        // max_nodes=2 means only 2 nodes total
        let visited = g.dfs_bounded("a", 100, 2).unwrap();
        assert_eq!(visited.len(), 2, "should stop at 2 nodes");
    }

    // ── New API tests (Rounds 4-8) ────────────────────────────────────────────

    #[test]
    fn test_entity_exists_and_relationship_exists() {
        let g = GraphStore::new();
        let a = EntityId::new("a");
        let b = EntityId::new("b");
        assert!(!g.entity_exists(&a).unwrap());
        g.add_entity(Entity::new("a", "Node")).unwrap();
        assert!(g.entity_exists(&a).unwrap());
        assert!(!g.relationship_exists(&a, &b, "knows").unwrap());
        g.add_entity(Entity::new("b", "Node")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "knows", 1.0)).unwrap();
        assert!(g.relationship_exists(&a, &b, "knows").unwrap());
        assert!(!g.relationship_exists(&a, &b, "likes").unwrap());
    }

    #[test]
    fn test_get_relationships_for_returns_outgoing() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        let a = EntityId::new("a");
        g.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "r", 1.0)).unwrap();
        let rels = g.get_relationships_for(&a).unwrap();
        assert_eq!(rels.len(), 2);
    }

    #[test]
    fn test_relationships_between_finds_both_directions() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "N")).unwrap();
        g.add_entity(Entity::new("y", "N")).unwrap();
        let x = EntityId::new("x");
        let y = EntityId::new("y");
        g.add_relationship(Relationship::new("x", "y", "follows", 1.0)).unwrap();
        g.add_relationship(Relationship::new("y", "x", "blocks", 1.0)).unwrap();
        let rels = g.relationships_between(&x, &y).unwrap();
        assert_eq!(rels.len(), 2);
    }

    #[test]
    fn test_find_entities_by_property() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "Person").with_property("age", serde_json::json!(30))).unwrap();
        g.add_entity(Entity::new("b", "Person").with_property("age", serde_json::json!(25))).unwrap();
        g.add_entity(Entity::new("c", "Person").with_property("age", serde_json::json!(30))).unwrap();
        let found = g.find_entities_by_property("age", &serde_json::json!(30)).unwrap();
        assert_eq!(found.len(), 2);
        let ids: Vec<_> = found.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"a") && ids.contains(&"c"));
    }

    #[test]
    fn test_neighbor_entities_returns_entity_objects() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("root", "R")).unwrap();
        g.add_entity(Entity::new("child1", "C")).unwrap();
        g.add_entity(Entity::new("child2", "C")).unwrap();
        let root = EntityId::new("root");
        g.add_relationship(Relationship::new("root", "child1", "has", 1.0)).unwrap();
        g.add_relationship(Relationship::new("root", "child2", "has", 1.0)).unwrap();
        let neighbors = g.neighbor_entities(&root).unwrap();
        assert_eq!(neighbors.len(), 2);
        let labels: Vec<_> = neighbors.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.iter().all(|l| *l == "C"));
    }

    #[test]
    fn test_remove_all_relationships_for() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        let a = EntityId::new("a");
        g.add_relationship(Relationship::new("a", "b", "r1", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "b", "r2", 1.0)).unwrap();
        let removed = g.remove_all_relationships_for(&a).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(g.relationship_count().unwrap(), 0);
    }

    #[test]
    fn test_topological_sort_returns_valid_order() {
        let g = GraphStore::new();
        for id in &["a", "b", "c", "d"] {
            g.add_entity(Entity::new(*id, "N")).unwrap();
        }
        g.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("c", "d", "r", 1.0)).unwrap();
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), 4);
        let pos: std::collections::HashMap<_, _> = order.iter().enumerate().map(|(i, id)| (id.as_str().to_owned(), i)).collect();
        assert!(pos["a"] < pos["b"]);
        assert!(pos["b"] < pos["c"]);
        assert!(pos["c"] < pos["d"]);
    }

    #[test]
    fn test_topological_sort_rejects_cycle() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "N")).unwrap();
        g.add_entity(Entity::new("y", "N")).unwrap();
        g.add_relationship(Relationship::new("x", "y", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("y", "x", "r", 1.0)).unwrap();
        assert!(g.topological_sort().is_err());
    }

    #[test]
    fn test_entity_count_by_label() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "Person")).unwrap();
        g.add_entity(Entity::new("b", "Person")).unwrap();
        g.add_entity(Entity::new("c", "Organization")).unwrap();
        assert_eq!(g.entity_count_by_label("Person").unwrap(), 2);
        assert_eq!(g.entity_count_by_label("Organization").unwrap(), 1);
        assert_eq!(g.entity_count_by_label("Unknown").unwrap(), 0);
    }

    #[test]
    fn test_update_entity_label_and_property() {
        let g = GraphStore::new();
        let id = EntityId::new("e1");
        g.add_entity(Entity::new("e1", "Old")).unwrap();
        assert!(g.update_entity_label(&id, "New").unwrap());
        assert_eq!(g.get_entity(&id).unwrap().label, "New");
        assert!(g.update_entity_property(&id, "key", serde_json::json!("val")).unwrap());
        assert_eq!(g.get_entity(&id).unwrap().properties["key"], serde_json::json!("val"));
    }

    #[test]
    fn test_connected_components_single_node() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        assert_eq!(g.connected_components().unwrap(), 1);
    }

    #[test]
    fn test_connected_components_two_isolated_nodes() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        assert_eq!(g.connected_components().unwrap(), 2);
    }

    #[test]
    fn test_connected_components_connected_pair() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "link", 1.0)).unwrap();
        assert_eq!(g.connected_components().unwrap(), 1);
    }

    #[test]
    fn test_connected_components_two_separate_pairs() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("c", "C")).unwrap();
        g.add_entity(Entity::new("d", "D")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "link", 1.0)).unwrap();
        g.add_relationship(Relationship::new("c", "d", "link", 1.0)).unwrap();
        assert_eq!(g.connected_components().unwrap(), 2);
    }

    #[test]
    fn test_connected_components_empty_graph() {
        let g = GraphStore::new();
        assert_eq!(g.connected_components().unwrap(), 0);
    }

    // ── Round 9: weakly_connected ─────────────────────────────────────────────

    #[test]
    fn test_weakly_connected_true_for_empty_graph() {
        let g = GraphStore::new();
        assert!(g.weakly_connected().unwrap());
    }

    #[test]
    fn test_weakly_connected_true_for_single_node() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        assert!(g.weakly_connected().unwrap());
    }

    #[test]
    fn test_weakly_connected_true_when_all_nodes_connected() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "link", 1.0)).unwrap();
        assert!(g.weakly_connected().unwrap());
    }

    #[test]
    fn test_weakly_connected_false_when_nodes_isolated() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        assert!(!g.weakly_connected().unwrap());
    }

    #[test]
    fn test_isolates_returns_nodes_with_no_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("c", "C")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "link", 1.0)).unwrap();
        let iso = g.isolates().unwrap();
        let mut ids: Vec<String> = iso.iter().map(|e| e.id.as_str().to_string()).collect();
        ids.sort();
        assert_eq!(ids, vec!["c".to_string()]);
    }

    #[test]
    fn test_isolates_returns_empty_when_all_connected() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "link", 1.0)).unwrap();
        assert!(g.isolates().unwrap().is_empty());
    }

    #[test]
    fn test_isolates_all_isolated() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "X")).unwrap();
        g.add_entity(Entity::new("y", "Y")).unwrap();
        let iso = g.isolates().unwrap();
        let mut ids: Vec<String> = iso.iter().map(|e| e.id.as_str().to_string()).collect();
        ids.sort();
        assert_eq!(ids, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn test_is_dag_on_dag() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "edge", 1.0)).unwrap();
        assert!(g.is_dag().unwrap());
    }

    #[test]
    fn test_is_dag_on_cyclic_graph() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "edge", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "a", "back", 1.0)).unwrap();
        assert!(!g.is_dag().unwrap());
    }

    #[test]
    fn test_in_degree_and_out_degree() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("c", "C")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "e1", 1.0)).unwrap();
        g.add_relationship(Relationship::new("c", "b", "e2", 1.0)).unwrap();
        let a = EntityId::new("a");
        let b = EntityId::new("b");
        let c = EntityId::new("c");
        assert_eq!(g.out_degree(&a).unwrap(), 1);
        assert_eq!(g.in_degree(&a).unwrap(), 0);
        assert_eq!(g.in_degree(&b).unwrap(), 2);
        assert_eq!(g.out_degree(&b).unwrap(), 0);
        assert_eq!(g.out_degree(&c).unwrap(), 1);
        assert_eq!(g.in_degree(&c).unwrap(), 0);
    }

    #[test]
    fn test_in_degree_missing_entity_returns_zero() {
        let g = GraphStore::new();
        let id = EntityId::new("ghost");
        assert_eq!(g.in_degree(&id).unwrap(), 0);
    }

    #[test]
    fn test_out_degree_missing_entity_returns_zero() {
        let g = GraphStore::new();
        let id = EntityId::new("ghost");
        assert_eq!(g.out_degree(&id).unwrap(), 0);
    }

    #[test]
    fn test_node_count_is_alias_for_entity_count() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        assert_eq!(g.node_count().unwrap(), g.entity_count().unwrap());
        assert_eq!(g.node_count().unwrap(), 2);
    }

    #[test]
    fn test_edge_count_is_alias_for_relationship_count() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "link", 1.0)).unwrap();
        assert_eq!(g.edge_count().unwrap(), g.relationship_count().unwrap());
        assert_eq!(g.edge_count().unwrap(), 1);
    }

    #[test]
    fn test_source_nodes_returns_nodes_with_no_incoming_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("root", "Root")).unwrap();
        g.add_entity(Entity::new("child", "Child")).unwrap();
        g.add_entity(Entity::new("leaf", "Leaf")).unwrap();
        g.add_relationship(Relationship::new("root", "child", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("child", "leaf", "e", 1.0)).unwrap();
        let sources = g.source_nodes().unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id.as_str(), "root");
    }

    #[test]
    fn test_sink_nodes_returns_nodes_with_no_outgoing_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("root", "Root")).unwrap();
        g.add_entity(Entity::new("child", "Child")).unwrap();
        g.add_entity(Entity::new("leaf", "Leaf")).unwrap();
        g.add_relationship(Relationship::new("root", "child", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("child", "leaf", "e", 1.0)).unwrap();
        let sinks = g.sink_nodes().unwrap();
        assert_eq!(sinks.len(), 1);
        assert_eq!(sinks[0].id.as_str(), "leaf");
    }

    #[test]
    fn test_source_and_sink_empty_on_isolated_node() {
        // An isolated node has no in or out edges, so it's both source and sink
        let g = GraphStore::new();
        g.add_entity(Entity::new("solo", "Solo")).unwrap();
        assert_eq!(g.source_nodes().unwrap().len(), 1);
        assert_eq!(g.sink_nodes().unwrap().len(), 1);
    }

    #[test]
    fn test_reverse_flips_all_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "edge", 1.0)).unwrap();
        let rev = g.reverse().unwrap();
        // Original: a→b. Reversed: b→a.
        assert_eq!(rev.entity_count().unwrap(), 2);
        assert_eq!(rev.relationship_count().unwrap(), 1);
        let b_id = EntityId::new("b");
        let a_id = EntityId::new("a");
        assert!(rev.relationship_exists(&b_id, &a_id, "edge").unwrap());
    }

    #[test]
    fn test_reverse_empty_graph_stays_empty() {
        let g = GraphStore::new();
        let rev = g.reverse().unwrap();
        assert_eq!(rev.entity_count().unwrap(), 0);
    }

    #[test]
    fn test_common_neighbors_finds_shared_targets() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("shared", "S")).unwrap();
        g.add_entity(Entity::new("only_a", "OA")).unwrap();
        g.add_relationship(Relationship::new("a", "shared", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "shared", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "only_a", "e", 1.0)).unwrap();
        let a_id = EntityId::new("a");
        let b_id = EntityId::new("b");
        let common = g.common_neighbors(&a_id, &b_id).unwrap();
        assert_eq!(common.len(), 1);
        assert_eq!(common[0].id.as_str(), "shared");
    }

    #[test]
    fn test_common_neighbors_empty_when_none_shared() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("x", "X")).unwrap();
        g.add_entity(Entity::new("y", "Y")).unwrap();
        g.add_relationship(Relationship::new("a", "x", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "y", "e", 1.0)).unwrap();
        let a_id = EntityId::new("a");
        let b_id = EntityId::new("b");
        assert!(g.common_neighbors(&a_id, &b_id).unwrap().is_empty());
    }

    // ── Round 3: entity_ids, is_empty, clear ─────────────────────────────────

    #[test]
    fn test_graph_is_empty_initially() {
        let g = GraphStore::new();
        assert!(g.is_empty().unwrap());
    }

    #[test]
    fn test_graph_is_empty_false_after_add() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        assert!(!g.is_empty().unwrap());
    }

    #[test]
    fn test_graph_entity_ids_returns_all_ids() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "X")).unwrap();
        g.add_entity(Entity::new("y", "Y")).unwrap();
        let ids = g.entity_ids().unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.iter().any(|id| id.0 == "x"));
        assert!(ids.iter().any(|id| id.0 == "y"));
    }

    #[test]
    fn test_graph_clear_removes_entities_and_relationships() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "links", 1.0))
        .unwrap();
        g.clear().unwrap();
        assert_eq!(g.entity_count().unwrap(), 0);
        assert_eq!(g.relationship_count().unwrap(), 0);
        assert!(g.is_empty().unwrap());
    }

    // ── Round 16: weight_of, neighbors_in, path_exists ───────────────────────

    #[test]
    fn test_weight_of_returns_edge_weight() {
        let g = make_graph();
        add(&g, "x"); add(&g, "y");
        link_w(&g, "x", "y", 3.5);
        let w = g.weight_of(&EntityId::new("x"), &EntityId::new("y")).unwrap();
        assert!(w.is_some());
        assert!((w.unwrap() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_weight_of_absent_edge_returns_none() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        let w = g.weight_of(&EntityId::new("a"), &EntityId::new("b")).unwrap();
        assert!(w.is_none());
    }

    #[test]
    fn test_neighbors_in_returns_predecessors() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "c"); link(&g, "b", "c");
        let mut preds: Vec<String> = g
            .neighbors_in(&EntityId::new("c"))
            .unwrap()
            .into_iter()
            .map(|id| id.as_str().to_string())
            .collect();
        preds.sort();
        assert_eq!(preds, vec!["a", "b"]);
    }

    #[test]
    fn test_neighbors_in_empty_for_node_with_no_incoming() {
        let g = make_graph();
        add(&g, "isolated");
        let preds = g.neighbors_in(&EntityId::new("isolated")).unwrap();
        assert!(preds.is_empty());
    }

    #[test]
    fn test_path_exists_reachable() {
        let g = make_graph();
        add(&g, "s"); add(&g, "m"); add(&g, "t");
        link(&g, "s", "m"); link(&g, "m", "t");
        assert!(g.path_exists("s", "t").unwrap());
    }

    #[test]
    fn test_path_exists_unreachable() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        assert!(!g.path_exists("a", "b").unwrap());
    }

    // ── Round 5: GraphStore::neighbor_ids ─────────────────────────────────────

    #[test]
    fn test_neighbor_ids_returns_direct_successors() {
        let g = make_graph();
        add(&g, "src"); add(&g, "dst1"); add(&g, "dst2");
        link(&g, "src", "dst1"); link(&g, "src", "dst2");
        let mut ids = g.neighbor_ids(&EntityId::new("src")).unwrap();
        ids.sort_by_key(|id| id.0.clone());
        assert_eq!(ids, vec![EntityId::new("dst1"), EntityId::new("dst2")]);
    }

    #[test]
    fn test_neighbor_ids_empty_for_isolated_node() {
        let g = make_graph();
        add(&g, "isolated");
        let ids = g.neighbor_ids(&EntityId::new("isolated")).unwrap();
        assert!(ids.is_empty());
    }

    // ── Round 17: density, all_entities, all_relationships, find_entities_by_label, bfs_bounded

    #[test]
    fn test_density_zero_for_empty_graph() {
        let g = make_graph();
        assert_eq!(g.density().unwrap(), 0.0);
    }

    #[test]
    fn test_density_zero_for_single_node() {
        let g = make_graph();
        add(&g, "solo");
        assert_eq!(g.density().unwrap(), 0.0);
    }

    #[test]
    fn test_density_one_for_complete_directed_graph() {
        // 2 nodes with both directed edges → density = 2 / (2*1) = 1.0
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        link(&g, "a", "b"); link(&g, "b", "a");
        assert!((g.density().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_density_partial() {
        // 3 nodes, 1 edge → max edges = 6, density = 1/6
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "b");
        let d = g.density().unwrap();
        assert!((d - 1.0/6.0).abs() < 1e-9);
    }

    #[test]
    fn test_all_entities_returns_all_nodes() {
        let g = make_graph();
        add(&g, "x"); add(&g, "y"); add(&g, "z");
        assert_eq!(g.all_entities().unwrap().len(), 3);
    }

    #[test]
    fn test_all_relationships_returns_all_edges() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "b"); link(&g, "b", "c");
        assert_eq!(g.all_relationships().unwrap().len(), 2);
    }

    #[test]
    fn test_find_entities_by_label_returns_matches() {
        let g = make_graph();
        g.add_entity(Entity::new("n1", "Person")).unwrap();
        g.add_entity(Entity::new("n2", "Person")).unwrap();
        g.add_entity(Entity::new("n3", "Car")).unwrap();
        let people = g.find_entities_by_label("Person").unwrap();
        assert_eq!(people.len(), 2);
    }

    #[test]
    fn test_bfs_bounded_limits_depth() {
        // a → b → c → d, bounded at depth 2 should only reach a,b,c
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c"); add(&g, "d");
        link(&g, "a", "b"); link(&g, "b", "c"); link(&g, "c", "d");
        let visited = g.bfs_bounded("a", 2, 100).unwrap();
        assert!(visited.contains(&EntityId::new("a")));
        assert!(visited.contains(&EntityId::new("b")));
        assert!(visited.contains(&EntityId::new("c")));
        assert!(!visited.contains(&EntityId::new("d")));
    }

    // ── Round 18: avg_degree ─────────────────────────────────────────────────

    #[test]
    fn test_avg_degree_zero_for_empty_graph() {
        let g = make_graph();
        assert_eq!(g.avg_degree().unwrap(), 0.0);
    }

    #[test]
    fn test_avg_degree_correct_value() {
        // 3 nodes, 2 edges → 2/3
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "b"); link(&g, "a", "c");
        let d = g.avg_degree().unwrap();
        assert!((d - 2.0/3.0).abs() < 1e-9);
    }

    // ── Round 19: total_weight, max/min_edge_weight ───────────────────────────

    #[test]
    fn test_total_weight_zero_for_empty_graph() {
        let g = make_graph();
        assert_eq!(g.total_weight().unwrap(), 0.0);
    }

    #[test]
    fn test_total_weight_sums_all_edges() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link_w(&g, "a", "b", 2.0); link_w(&g, "b", "c", 3.5);
        assert!((g.total_weight().unwrap() - 5.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_edge_weight_none_for_empty() {
        let g = make_graph();
        assert!(g.max_edge_weight().unwrap().is_none());
    }

    #[test]
    fn test_max_edge_weight_returns_largest() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link_w(&g, "a", "b", 1.0); link_w(&g, "a", "c", 9.5);
        assert!((g.max_edge_weight().unwrap().unwrap() - 9.5).abs() < 1e-6);
    }

    #[test]
    fn test_min_edge_weight_returns_smallest() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link_w(&g, "a", "b", 1.0); link_w(&g, "a", "c", 9.5);
        assert!((g.min_edge_weight().unwrap().unwrap() - 1.0).abs() < 1e-6);
    }

    // ── Round 6: max_out_degree_entity / leaf_nodes ───────────────────────────

    #[test]
    fn test_max_out_degree_entity_returns_node_with_most_edges() {
        let g = make_graph();
        add(&g, "hub"); add(&g, "a"); add(&g, "b"); add(&g, "leaf");
        link(&g, "hub", "a"); link(&g, "hub", "b"); link(&g, "a", "leaf");
        let best = g.max_out_degree_entity().unwrap().unwrap();
        assert_eq!(best.id, EntityId::new("hub"));
    }

    #[test]
    fn test_max_out_degree_entity_none_for_empty_graph() {
        let g = make_graph();
        assert!(g.max_out_degree_entity().unwrap().is_none());
    }

    #[test]
    fn test_leaf_nodes_returns_nodes_with_no_outgoing_edges() {
        let g = make_graph();
        add(&g, "root"); add(&g, "mid"); add(&g, "leaf1"); add(&g, "leaf2");
        link(&g, "root", "mid"); link(&g, "mid", "leaf1"); link(&g, "mid", "leaf2");
        let mut leaf_ids: Vec<String> = g
            .leaf_nodes()
            .unwrap()
            .into_iter()
            .map(|e| e.id.0.clone())
            .collect();
        leaf_ids.sort();
        assert_eq!(leaf_ids, vec!["leaf1", "leaf2"]);
    }

    #[test]
    fn test_leaf_nodes_all_are_leaves_with_no_edges() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        assert_eq!(g.leaf_nodes().unwrap().len(), 2);
    }

    // ── Round 7: top_n_by_out_degree / remove_entity_and_edges ───────────────

    #[test]
    fn test_top_n_by_out_degree_returns_descending() {
        let g = make_graph();
        add(&g, "hub"); add(&g, "mid"); add(&g, "tip"); add(&g, "leaf");
        link(&g, "hub", "mid"); link(&g, "hub", "tip"); link(&g, "hub", "leaf");
        link(&g, "mid", "leaf");
        let top2 = g.top_n_by_out_degree(2).unwrap();
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].id, EntityId::new("hub"));
    }

    #[test]
    fn test_top_n_by_out_degree_zero_returns_empty() {
        let g = make_graph();
        add(&g, "x");
        assert!(g.top_n_by_out_degree(0).unwrap().is_empty());
    }

    #[test]
    fn test_remove_entity_and_edges_removes_node_and_incident_edges() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "b"); link(&g, "b", "c");
        g.remove_entity_and_edges(&EntityId::new("b")).unwrap();
        assert_eq!(g.entity_count().unwrap(), 2);
        assert_eq!(g.relationship_count().unwrap(), 0);
    }

    #[test]
    fn test_remove_entity_and_edges_errors_for_unknown_id() {
        let g = make_graph();
        let result = g.remove_entity_and_edges(&EntityId::new("ghost"));
        assert!(result.is_err());
    }

    // ── Round 20: relationship_kinds / graph_density / EntityId::try_new ──────

    #[test]
    fn test_relationship_kinds_returns_sorted_distinct_kinds() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        g.add_relationship(Relationship::new("a", "b", "KNOWS", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "LIKES", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "KNOWS", 1.0)).unwrap();
        let kinds = g.relationship_kinds().unwrap();
        assert_eq!(kinds, vec!["KNOWS", "LIKES"]);
    }

    #[test]
    fn test_relationship_kinds_empty_graph_returns_empty() {
        let g = make_graph();
        assert!(g.relationship_kinds().unwrap().is_empty());
    }

    #[test]
    fn test_graph_density_zero_for_empty() {
        let g = make_graph();
        assert_eq!(g.graph_density().unwrap(), 0.0);
    }

    #[test]
    fn test_graph_density_correct_for_partial_graph() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        // 1 edge out of max 6 directed edges (3*2)
        link(&g, "a", "b");
        let d = g.graph_density().unwrap();
        assert!((d - 1.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_entity_id_try_new_rejects_empty() {
        let result = EntityId::try_new("");
        assert!(result.is_err());
    }

    #[test]
    fn test_entity_id_try_new_accepts_nonempty() {
        let id = EntityId::try_new("valid").unwrap();
        assert_eq!(id.as_str(), "valid");
    }

    // ── Round 8: hub_nodes / incident_relationships ───────────────────────────

    #[test]
    fn test_hub_nodes_returns_nodes_meeting_threshold() {
        let g = make_graph();
        add(&g, "hub"); add(&g, "mid"); add(&g, "leaf");
        link(&g, "hub", "mid"); link(&g, "hub", "leaf"); link(&g, "mid", "leaf");
        let hubs = g.hub_nodes(2).unwrap();
        assert_eq!(hubs.len(), 1);
        assert_eq!(hubs[0].id, EntityId::new("hub"));
    }

    #[test]
    fn test_hub_nodes_threshold_zero_returns_all() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b");
        assert_eq!(g.hub_nodes(0).unwrap().len(), 2);
    }

    #[test]
    fn test_incident_relationships_includes_outgoing_and_incoming() {
        let g = make_graph();
        add(&g, "a"); add(&g, "b"); add(&g, "c");
        link(&g, "a", "b"); link(&g, "c", "b");
        let rels = g.incident_relationships(&EntityId::new("b")).unwrap();
        assert_eq!(rels.len(), 2);
    }

    #[test]
    fn test_incident_relationships_empty_for_isolated_node() {
        let g = make_graph();
        add(&g, "iso");
        assert!(g.incident_relationships(&EntityId::new("iso")).unwrap().is_empty());
    }

    // ── Round 10: average_out_degree / in_degree_for ──────────────────────────

    #[test]
    fn test_average_out_degree_empty_graph_is_zero() {
        let g = GraphStore::new();
        assert!((g.average_out_degree().unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_average_out_degree_two_nodes_one_edge() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "link", 1.0)).unwrap();
        // 1 edge / 2 nodes = 0.5
        assert!((g.average_out_degree().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_in_degree_for_counts_incoming_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("c", "C")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("c", "b", "r", 1.0)).unwrap();
        assert_eq!(g.in_degree_for(&EntityId::new("b")).unwrap(), 2);
        assert_eq!(g.in_degree_for(&EntityId::new("a")).unwrap(), 0);
    }

    #[test]
    fn test_in_degree_for_returns_zero_for_unknown_entity() {
        let g = GraphStore::new();
        assert_eq!(g.in_degree_for(&EntityId::new("ghost")).unwrap(), 0);
    }

    // ── Round 12: EntityId::is_empty/starts_with, Entity::has_property, entity_labels, step_latency_p50/p99 ──

    #[test]
    fn test_entity_id_is_empty_false_for_nonempty() {
        let id = EntityId::new("node-1");
        assert!(!id.is_empty());
    }

    #[test]
    fn test_entity_id_starts_with_matches_prefix() {
        let id = EntityId::new("concept-42");
        assert!(id.starts_with("concept-"));
        assert!(!id.starts_with("entity-"));
    }

    #[test]
    fn test_entity_id_starts_with_empty_always_true() {
        let id = EntityId::new("anything");
        assert!(id.starts_with(""));
    }

    #[test]
    fn test_entity_has_property_returns_true_when_present() {
        let e = Entity::new("e", "Node")
            .with_property("color", serde_json::json!("blue"));
        assert!(e.has_property("color"));
        assert!(!e.has_property("size"));
    }

    #[test]
    fn test_entity_has_property_false_when_no_properties() {
        let e = Entity::new("e", "Node");
        assert!(!e.has_property("any"));
    }

    #[test]
    fn test_entity_labels_returns_distinct_sorted_labels() {
        let g = make_graph();
        g.add_entity(Entity::new("a", "Person")).unwrap();
        g.add_entity(Entity::new("b", "Concept")).unwrap();
        g.add_entity(Entity::new("c", "Person")).unwrap();
        let labels = g.entity_labels().unwrap();
        assert_eq!(labels, vec!["Concept", "Person"]);
    }

    #[test]
    fn test_entity_labels_empty_for_empty_graph() {
        let g = make_graph();
        assert!(g.entity_labels().unwrap().is_empty());
    }

    // ── Round 12: out_degree_for / predecessors / is_source ───────────────────

    #[test]
    fn test_out_degree_for_returns_outgoing_edge_count() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("c", "C")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "r", 1.0)).unwrap();
        assert_eq!(g.out_degree_for(&EntityId::new("a")).unwrap(), 2);
        assert_eq!(g.out_degree_for(&EntityId::new("b")).unwrap(), 0);
    }

    #[test]
    fn test_out_degree_for_returns_zero_for_unknown_entity() {
        let g = GraphStore::new();
        assert_eq!(g.out_degree_for(&EntityId::new("ghost")).unwrap(), 0);
    }

    #[test]
    fn test_predecessors_returns_nodes_with_incoming_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("c", "C")).unwrap();
        g.add_relationship(Relationship::new("a", "c", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "r", 1.0)).unwrap();
        let mut preds: Vec<String> = g
            .predecessors(&EntityId::new("c"))
            .unwrap()
            .iter()
            .map(|e| e.id.as_str().to_string())
            .collect();
        preds.sort();
        assert_eq!(preds, vec!["a", "b"]);
    }

    #[test]
    fn test_predecessors_empty_for_source_node() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("root", "Root")).unwrap();
        g.add_entity(Entity::new("child", "Child")).unwrap();
        g.add_relationship(Relationship::new("root", "child", "r", 1.0)).unwrap();
        assert!(g.predecessors(&EntityId::new("root")).unwrap().is_empty());
    }

    #[test]
    fn test_is_source_true_for_node_with_no_incoming_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("src", "Src")).unwrap();
        g.add_entity(Entity::new("dst", "Dst")).unwrap();
        g.add_relationship(Relationship::new("src", "dst", "r", 1.0)).unwrap();
        assert!(g.is_source(&EntityId::new("src")).unwrap());
        assert!(!g.is_source(&EntityId::new("dst")).unwrap());
    }

    // ── Round 13: Relationship::is_self_loop/reversed, find_entities_by_labels, remove_isolated ──

    #[test]
    fn test_relationship_is_self_loop_true_when_from_equals_to() {
        let r = Relationship::new("a", "a", "self", 1.0);
        assert!(r.is_self_loop());
    }

    #[test]
    fn test_relationship_is_self_loop_false_for_normal_edge() {
        let r = Relationship::new("a", "b", "edge", 1.0);
        assert!(!r.is_self_loop());
    }

    #[test]
    fn test_relationship_reversed_swaps_endpoints() {
        let r = Relationship::new("from", "to", "knows", 0.5);
        let rev = r.reversed();
        assert_eq!(rev.from.as_str(), "to");
        assert_eq!(rev.to.as_str(), "from");
        assert_eq!(rev.kind, "knows");
        assert!((rev.weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_find_entities_by_labels_returns_matching() {
        let g = make_graph();
        g.add_entity(Entity::new("p1", "Person")).unwrap();
        g.add_entity(Entity::new("p2", "Person")).unwrap();
        g.add_entity(Entity::new("c1", "Concept")).unwrap();
        let results = g.find_entities_by_labels(&["Person"]).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.label == "Person"));
    }

    #[test]
    fn test_find_entities_by_labels_empty_when_no_match() {
        let g = make_graph();
        g.add_entity(Entity::new("n1", "Node")).unwrap();
        let results = g.find_entities_by_labels(&["Missing"]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_isolated_removes_nodes_without_edges() {
        let g = make_graph();
        g.add_entity(Entity::new("connected", "N")).unwrap();
        g.add_entity(Entity::new("isolated", "N")).unwrap();
        g.add_entity(Entity::new("other", "N")).unwrap();
        g.add_relationship(Relationship::new("connected", "other", "r", 1.0)).unwrap();
        let removed = g.remove_isolated().unwrap();
        assert_eq!(removed, 1);
        assert!(g.get_entity(&EntityId::new("isolated")).is_err());
        assert!(g.get_entity(&EntityId::new("connected")).is_ok());
    }

    #[test]
    fn test_remove_isolated_zero_when_all_connected() {
        let g = make_graph();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();
        assert_eq!(g.remove_isolated().unwrap(), 0);
    }

    // ── Round 13: successors / is_sink ────────────────────────────────────────

    #[test]
    fn test_successors_returns_direct_out_neighbors() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_entity(Entity::new("c", "C")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "r", 1.0)).unwrap();
        let mut ids: Vec<String> = g
            .successors(&EntityId::new("a"))
            .unwrap()
            .iter()
            .map(|e| e.id.as_str().to_string())
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["b", "c"]);
    }

    #[test]
    fn test_successors_empty_for_sink_node() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("leaf", "L")).unwrap();
        assert!(g.successors(&EntityId::new("leaf")).unwrap().is_empty());
    }

    #[test]
    fn test_is_sink_true_for_node_with_no_outgoing_edges() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "A")).unwrap();
        g.add_entity(Entity::new("b", "B")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();
        assert!(!g.is_sink(&EntityId::new("a")).unwrap());
        assert!(g.is_sink(&EntityId::new("b")).unwrap());
    }

    #[test]
    fn test_is_sink_true_for_unknown_entity() {
        let g = GraphStore::new();
        assert!(g.is_sink(&EntityId::new("ghost")).unwrap());
    }

    // ── Round 14: reachable_from / contains_cycle ─────────────────────────────

    #[test]
    fn test_reachable_from_returns_all_downstream_nodes() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "edge", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "edge", 1.0)).unwrap();
        let reachable = g.reachable_from(&EntityId::new("a")).unwrap();
        assert!(reachable.contains(&EntityId::new("b")));
        assert!(reachable.contains(&EntityId::new("c")));
        assert!(!reachable.contains(&EntityId::new("a")));
    }

    #[test]
    fn test_reachable_from_empty_for_sink_node() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("sink", "N")).unwrap();
        let reachable = g.reachable_from(&EntityId::new("sink")).unwrap();
        assert!(reachable.is_empty());
    }

    #[test]
    fn test_reachable_from_empty_for_unknown_node() {
        let g = GraphStore::new();
        let reachable = g.reachable_from(&EntityId::new("ghost")).unwrap();
        assert!(reachable.is_empty());
    }

    #[test]
    fn test_contains_cycle_false_for_dag() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "e", 1.0)).unwrap();
        assert!(!g.contains_cycle().unwrap());
    }

    #[test]
    fn test_contains_cycle_true_for_cyclic_graph() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "N")).unwrap();
        g.add_entity(Entity::new("y", "N")).unwrap();
        g.add_relationship(Relationship::new("x", "y", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("y", "x", "e", 1.0)).unwrap();
        assert!(g.contains_cycle().unwrap());
    }

    #[test]
    fn test_contains_cycle_false_for_empty_graph() {
        let g = GraphStore::new();
        assert!(!g.contains_cycle().unwrap());
    }

    // ── Round 15: GraphStore::is_acyclic ─────────────────────────────────────

    #[test]
    fn test_is_acyclic_true_for_dag() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "e", 1.0)).unwrap();
        assert!(g.is_acyclic().unwrap());
    }

    #[test]
    fn test_is_acyclic_false_for_cyclic_graph() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("x", "N")).unwrap();
        g.add_entity(Entity::new("y", "N")).unwrap();
        g.add_relationship(Relationship::new("x", "y", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("y", "x", "e", 1.0)).unwrap();
        assert!(!g.is_acyclic().unwrap());
    }

    #[test]
    fn test_is_acyclic_true_for_empty_graph() {
        let g = GraphStore::new();
        assert!(g.is_acyclic().unwrap());
    }

    // ── Round 15: count_relationships_by_kind, merge, top_nodes_by_in/out_degree ──

    #[test]
    fn test_count_relationships_by_kind_returns_correct_count() {
        let g = make_graph();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "knows", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "knows", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "likes", 0.5)).unwrap();
        assert_eq!(g.count_relationships_by_kind("knows").unwrap(), 2);
        assert_eq!(g.count_relationships_by_kind("likes").unwrap(), 1);
        assert_eq!(g.count_relationships_by_kind("absent").unwrap(), 0);
    }

    #[test]
    fn test_merge_imports_entities_and_relationships() {
        let g1 = make_graph();
        g1.add_entity(Entity::new("a", "N")).unwrap();
        g1.add_entity(Entity::new("b", "N")).unwrap();
        g1.add_relationship(Relationship::new("a", "b", "r", 1.0)).unwrap();

        let g2 = make_graph();
        g2.add_entity(Entity::new("c", "N")).unwrap();
        g2.add_entity(Entity::new("a", "N")).unwrap(); // duplicate — should not double-add
        g2.add_relationship(Relationship::new("c", "a", "s", 0.5)).unwrap();

        g1.merge(&g2).unwrap();
        assert_eq!(g1.entity_count().unwrap(), 3); // a, b, c
        assert_eq!(g1.relationship_count().unwrap(), 2); // r + s
    }

    #[test]
    fn test_top_nodes_by_in_degree_returns_sinks() {
        let g = make_graph();
        g.add_entity(Entity::new("hub", "N")).unwrap();
        g.add_entity(Entity::new("src1", "N")).unwrap();
        g.add_entity(Entity::new("src2", "N")).unwrap();
        g.add_relationship(Relationship::new("src1", "hub", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("src2", "hub", "r", 1.0)).unwrap();
        let top = g.top_nodes_by_in_degree(1).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id.as_str(), "hub");
    }

    #[test]
    fn test_top_nodes_by_out_degree_returns_sources() {
        let g = make_graph();
        g.add_entity(Entity::new("src", "N")).unwrap();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_relationship(Relationship::new("src", "a", "r", 1.0)).unwrap();
        g.add_relationship(Relationship::new("src", "b", "r", 1.0)).unwrap();
        let top = g.top_nodes_by_out_degree(1).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id.as_str(), "src");
    }

    // ── Round 27: property_value, find_relationships_by_kind, count_relationships_by_kind ──

    #[test]
    fn test_entity_property_value_returns_value() {
        let e = Entity::new("n1", "Node")
            .with_property("age", serde_json::Value::Number(42.into()));
        let val = e.property_value("age");
        assert!(val.is_some());
        assert_eq!(val.unwrap(), &serde_json::Value::Number(42.into()));
    }

    #[test]
    fn test_entity_property_value_missing_returns_none() {
        let e = Entity::new("n1", "Node");
        assert!(e.property_value("missing").is_none());
    }

    #[test]
    fn test_find_relationships_by_kind_returns_matching() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "KNOWS", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "LIKES", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "KNOWS", 1.0)).unwrap();
        let rels = g.find_relationships_by_kind("KNOWS").unwrap();
        assert_eq!(rels.len(), 2);
        assert!(rels.iter().all(|r| r.kind == "KNOWS"));
    }

    #[test]
    fn test_find_relationships_by_kind_no_match_returns_empty() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "KNOWS", 1.0)).unwrap();
        let rels = g.find_relationships_by_kind("HATES").unwrap();
        assert!(rels.is_empty());
    }

    #[test]
    fn test_count_relationships_by_kind_correct() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        g.add_relationship(Relationship::new("a", "b", "KNOWS", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "KNOWS", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "PART_OF", 1.0)).unwrap();
        assert_eq!(g.count_relationships_by_kind("KNOWS").unwrap(), 2);
        assert_eq!(g.count_relationships_by_kind("PART_OF").unwrap(), 1);
        assert_eq!(g.count_relationships_by_kind("MISSING").unwrap(), 0);
    }

    // ── Round 16: max_out_degree / max_in_degree ──────────────────────────────

    #[test]
    fn test_max_out_degree_returns_highest_out_degree() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        // a → b, a → c  (out-degree 2)
        // b → c          (out-degree 1)
        g.add_relationship(Relationship::new("a", "b", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("a", "c", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "e", 1.0)).unwrap();
        assert_eq!(g.max_out_degree().unwrap(), 2);
    }

    #[test]
    fn test_max_out_degree_zero_for_empty_graph() {
        let g = GraphStore::new();
        assert_eq!(g.max_out_degree().unwrap(), 0);
    }

    #[test]
    fn test_max_in_degree_returns_highest_in_degree() {
        let g = GraphStore::new();
        g.add_entity(Entity::new("a", "N")).unwrap();
        g.add_entity(Entity::new("b", "N")).unwrap();
        g.add_entity(Entity::new("c", "N")).unwrap();
        // a → c, b → c  (c has in-degree 2)
        g.add_relationship(Relationship::new("a", "c", "e", 1.0)).unwrap();
        g.add_relationship(Relationship::new("b", "c", "e", 1.0)).unwrap();
        assert_eq!(g.max_in_degree().unwrap(), 2);
    }

    #[test]
    fn test_max_in_degree_zero_for_empty_graph() {
        let g = GraphStore::new();
        assert_eq!(g.max_in_degree().unwrap(), 0);
    }
}
