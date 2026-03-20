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

    /// Return the number of relationships in the graph.
    ///
    /// Alias for [`relationship_count`] using graph-theory terminology.
    ///
    /// [`relationship_count`]: GraphStore::relationship_count
    pub fn edge_count(&self) -> Result<usize, AgentRuntimeError> {
        self.relationship_count()
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
        inner.cycle_cache = None;
        Ok(())
    }

    /// Count entities whose label equals `label` (case-sensitive).
    pub fn entity_count_by_label(&self, label: &str) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "entity_count_by_label");
        Ok(inner.entities.values().filter(|e| e.label == label).count())
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

        // Out-degree is already available directly from the adjacency index:
        // adjacency[id].len() == number of outgoing edges from id.
        // Only in-degree requires a pass over relationships.
        let mut in_degree: HashMap<EntityId, usize> = inner
            .entities
            .keys()
            .map(|id| (id.clone(), 0usize))
            .collect();

        for rel in &inner.relationships {
            *in_degree.entry(rel.to.clone()).or_insert(0) += 1;
        }

        let denom = if n <= 1 { 1.0 } else { (n - 1) as f32 };
        let mut result = HashMap::new();
        for id in inner.entities.keys() {
            let od = inner.adjacency.get(id).map_or(0, |v| v.len());
            let id_ = *in_degree.get(id).unwrap_or(&0);
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

    /// Return all entities that have no outgoing edges (out-degree == 0).
    ///
    /// In a DAG these are the leaf nodes. Useful for identifying terminal
    /// states in a workflow or dependency graph.
    pub fn sink_nodes(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "sink_nodes");
        let has_outgoing: std::collections::HashSet<&EntityId> =
            inner.relationships.iter().map(|r| &r.from).collect();
        Ok(inner
            .entities
            .values()
            .filter(|e| !has_outgoing.contains(&e.id))
            .cloned()
            .collect())
    }

    /// Return all entities that have no incoming edges (in-degree == 0).
    ///
    /// In a DAG these are the root nodes. Useful for finding entry points in
    /// a workflow or dependency graph.
    pub fn source_nodes(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "source_nodes");
        let has_incoming: std::collections::HashSet<&EntityId> =
            inner.relationships.iter().map(|r| &r.to).collect();
        Ok(inner
            .entities
            .values()
            .filter(|e| !has_incoming.contains(&e.id))
            .cloned()
            .collect())
    }

    /// Return all entities that have no edges (neither incoming nor outgoing).
    pub fn isolates(&self) -> Result<Vec<Entity>, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "isolates");
        let mut has_edge: std::collections::HashSet<&EntityId> = std::collections::HashSet::new();
        for rel in &inner.relationships {
            has_edge.insert(&rel.from);
            has_edge.insert(&rel.to);
        }
        Ok(inner
            .entities
            .values()
            .filter(|e| !has_edge.contains(&e.id))
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
    /// Scans all relationships in O(|E|).
    pub fn in_degree(&self, id: &EntityId) -> Result<usize, AgentRuntimeError> {
        let inner = recover_lock(self.inner.lock(), "in_degree");
        Ok(inner.relationships.iter().filter(|r| &r.to == id).count())
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
}
