//! # Module: knowledge_graph
//!
//! Entity-relationship knowledge graph with traversal and simple reasoning.
//!
//! ## Overview
//!
//! [`KnowledgeGraph`] is an in-memory directed graph of typed [`Entity`] nodes
//! connected by [`Relation`] edges.  It supports:
//!
//! - BFS shortest-path search ([`KnowledgeGraph::bfs_path`]).
//! - Transitive closure over a single relation type
//!   ([`KnowledgeGraph::transitive_closure`]).
//! - Ancestor inference via the [`RelationType::IsA`] chain
//!   ([`KnowledgeGraph::infer_is_a`]).
//! - Common-ancestor lookup ([`KnowledgeGraph::common_ancestors`]).
//! - Connected-component counting ([`KnowledgeGraph::stats`]).
//! - Subgraph extraction ([`KnowledgeGraph::subgraph`]).
//!
//! ## Usage
//!
//! ```rust
//! use llm_agent_runtime::knowledge_graph::{KnowledgeGraph, RelationType};
//!
//! let mut kg = KnowledgeGraph::new();
//! let dog = kg.add_entity("Animal", "Dog");
//! let mammal = kg.add_entity("Class", "Mammal");
//! kg.add_relation(dog, mammal, RelationType::IsA, 1.0);
//!
//! let ancestors = kg.infer_is_a(dog);
//! assert!(ancestors.contains(&mammal));
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ── EntityId ──────────────────────────────────────────────────────────────────

/// Opaque numeric identifier for an entity in the graph.
pub type EntityId = u64;

// ── Entity ────────────────────────────────────────────────────────────────────

/// A node in the knowledge graph.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Unique numeric identifier.
    pub id: EntityId,
    /// Ontological type label (e.g. `"Animal"`, `"Concept"`).
    pub entity_type: String,
    /// Human-readable name.
    pub name: String,
    /// Arbitrary key/value properties.
    pub properties: HashMap<String, String>,
}

// ── RelationType ──────────────────────────────────────────────────────────────

/// The semantic type of a directed edge between two entities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    /// `from` is-a `to` (subtype / instance relationship).
    IsA,
    /// `from` has-a `to` (composition).
    HasA,
    /// `from` is part-of `to`.
    PartOf,
    /// `from` is related to `to` (generic association).
    RelatedTo,
    /// `from` depends-on `to`.
    DependsOn,
    /// `from` is the opposite of `to`.
    OppositeOf,
    /// User-defined relation with an arbitrary label.
    Custom(String),
}

// ── Relation ──────────────────────────────────────────────────────────────────

/// A directed, weighted edge in the knowledge graph.
#[derive(Debug, Clone)]
pub struct Relation {
    /// Unique numeric identifier.
    pub id: u64,
    /// Source entity.
    pub from_entity: EntityId,
    /// Target entity.
    pub to_entity: EntityId,
    /// Semantic type of this edge.
    pub relation_type: RelationType,
    /// Edge weight (e.g. confidence or strength).
    pub weight: f64,
    /// Arbitrary key/value properties.
    pub properties: HashMap<String, String>,
}

// ── KnowledgeGraph ────────────────────────────────────────────────────────────

/// In-memory entity-relationship graph.
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// All entities, keyed by their id.
    pub entities: HashMap<EntityId, Entity>,
    /// All relations (directed edges).
    pub relations: Vec<Relation>,
    /// Next entity id to assign.
    pub next_entity_id: u64,
    /// Next relation id to assign.
    pub next_relation_id: u64,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            relations: Vec::new(),
            next_entity_id: 1,
            next_relation_id: 1,
        }
    }

    /// Add a new entity and return its assigned [`EntityId`].
    pub fn add_entity(&mut self, entity_type: &str, name: &str) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entities.insert(
            id,
            Entity {
                id,
                entity_type: entity_type.to_string(),
                name: name.to_string(),
                properties: HashMap::new(),
            },
        );
        id
    }

    /// Add a directed relation from `from` to `to` and return the relation id.
    pub fn add_relation(
        &mut self,
        from: EntityId,
        to: EntityId,
        rel_type: RelationType,
        weight: f64,
    ) -> u64 {
        let id = self.next_relation_id;
        self.next_relation_id += 1;
        self.relations.push(Relation {
            id,
            from_entity: from,
            to_entity: to,
            relation_type: rel_type,
            weight,
            properties: HashMap::new(),
        });
        id
    }

    /// Find the first entity whose `name` matches exactly, returning its id.
    pub fn find_entity_by_name(&self, name: &str) -> Option<EntityId> {
        self.entities
            .values()
            .find(|e| e.name == name)
            .map(|e| e.id)
    }

    /// Return all outgoing neighbours of `id` as `(target_id, relation_type, weight)`.
    pub fn neighbors(&self, id: EntityId) -> Vec<(EntityId, &RelationType, f64)> {
        self.relations
            .iter()
            .filter(|r| r.from_entity == id)
            .map(|r| (r.to_entity, &r.relation_type, r.weight))
            .collect()
    }

    /// BFS shortest path from `from` to `to`; returns the sequence of entity
    /// ids (inclusive of endpoints), or `None` if unreachable.
    pub fn bfs_path(&self, from: EntityId, to: EntityId) -> Option<Vec<EntityId>> {
        if from == to {
            return Some(vec![from]);
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<Vec<EntityId>> = VecDeque::new();
        queue.push_back(vec![from]);
        visited.insert(from);

        while let Some(path) = queue.pop_front() {
            let current = *path.last()?;
            for (neighbour, _, _) in self.neighbors(current) {
                if neighbour == to {
                    let mut full = path.clone();
                    full.push(neighbour);
                    return Some(full);
                }
                if visited.insert(neighbour) {
                    let mut new_path = path.clone();
                    new_path.push(neighbour);
                    queue.push_back(new_path);
                }
            }
        }

        None
    }

    /// Return all entity ids reachable from `from` by following edges of type
    /// `rel_type` (transitive closure, not including `from` itself).
    pub fn transitive_closure(&self, from: EntityId, rel_type: &RelationType) -> Vec<EntityId> {
        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<EntityId> = VecDeque::new();
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            for r in &self.relations {
                if r.from_entity == current && &r.relation_type == rel_type {
                    if visited.insert(r.to_entity) {
                        queue.push_back(r.to_entity);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Return all ancestors of `entity` reachable via [`RelationType::IsA`]
    /// edges (i.e. all super-types / super-classes).
    pub fn infer_is_a(&self, entity: EntityId) -> Vec<EntityId> {
        self.transitive_closure(entity, &RelationType::IsA)
    }

    /// Return all entities whose `entity_type` field equals `entity_type`.
    pub fn entities_of_type(&self, entity_type: &str) -> Vec<EntityId> {
        self.entities
            .values()
            .filter(|e| e.entity_type == entity_type)
            .map(|e| e.id)
            .collect()
    }

    /// Extract a subgraph containing only the given entity ids and the
    /// relations between them.
    pub fn subgraph(&self, entity_ids: &[EntityId]) -> KnowledgeGraph {
        let id_set: HashSet<EntityId> = entity_ids.iter().copied().collect();
        let mut sub = KnowledgeGraph::new();

        // Copy matching entities preserving their original ids.
        for &id in &id_set {
            if let Some(e) = self.entities.get(&id) {
                sub.entities.insert(
                    id,
                    Entity {
                        id: e.id,
                        entity_type: e.entity_type.clone(),
                        name: e.name.clone(),
                        properties: e.properties.clone(),
                    },
                );
            }
        }

        // Keep track of the highest entity id so next_entity_id is safe.
        sub.next_entity_id = id_set.iter().copied().max().unwrap_or(0) + 1;

        // Copy relations whose both endpoints are in the subgraph.
        for r in &self.relations {
            if id_set.contains(&r.from_entity) && id_set.contains(&r.to_entity) {
                sub.relations.push(r.clone());
                if r.id >= sub.next_relation_id {
                    sub.next_relation_id = r.id + 1;
                }
            }
        }

        sub
    }

    /// Return the common ancestors of `a` and `b` via the [`RelationType::IsA`]
    /// chain (set intersection of their respective ancestor sets).
    pub fn common_ancestors(&self, a: EntityId, b: EntityId) -> Vec<EntityId> {
        let ancestors_a: HashSet<EntityId> = self.infer_is_a(a).into_iter().collect();
        let ancestors_b: HashSet<EntityId> = self.infer_is_a(b).into_iter().collect();
        ancestors_a.intersection(&ancestors_b).copied().collect()
    }

    /// Compute aggregate graph statistics.
    pub fn stats(&self) -> KGStats {
        let entity_count = self.entities.len();
        let relation_count = self.relations.len();

        let avg_degree = if entity_count == 0 {
            0.0
        } else {
            relation_count as f64 / entity_count as f64
        };

        let connected_components = self.count_connected_components();

        KGStats {
            entity_count,
            relation_count,
            avg_degree,
            connected_components,
        }
    }

    /// Count connected components using union-find (treating all edges as
    /// undirected for this purpose).
    fn count_connected_components(&self) -> usize {
        let ids: Vec<EntityId> = self.entities.keys().copied().collect();
        if ids.is_empty() {
            return 0;
        }

        // Map entity ids to contiguous indices.
        let index: HashMap<EntityId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n = ids.len();
        let mut parent: Vec<usize> = (0..n).collect();

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

        for r in &self.relations {
            if let (Some(&ia), Some(&ib)) =
                (index.get(&r.from_entity), index.get(&r.to_entity))
            {
                union(&mut parent, ia, ib);
            }
        }

        // Count distinct roots.
        (0..n).filter(|&i| find(&mut parent, i) == i).count()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── KGStats ───────────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`KnowledgeGraph`].
#[derive(Debug, Clone)]
pub struct KGStats {
    /// Total number of entities.
    pub entity_count: usize,
    /// Total number of relations.
    pub relation_count: usize,
    /// Average out-degree (relations / entities).
    pub avg_degree: f64,
    /// Number of weakly connected components (treating edges as undirected).
    pub connected_components: usize,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn add_entity_and_relation() {
        let mut kg = KnowledgeGraph::new();
        let a = kg.add_entity("Type", "A");
        let b = kg.add_entity("Type", "B");
        let rel_id = kg.add_relation(a, b, RelationType::RelatedTo, 1.0);
        assert_eq!(rel_id, 1);
        assert_eq!(kg.relations.len(), 1);
        assert_eq!(kg.entities.len(), 2);
    }

    #[test]
    fn bfs_finds_path() {
        let mut kg = KnowledgeGraph::new();
        let a = kg.add_entity("N", "A");
        let b = kg.add_entity("N", "B");
        let c = kg.add_entity("N", "C");
        kg.add_relation(a, b, RelationType::RelatedTo, 1.0);
        kg.add_relation(b, c, RelationType::RelatedTo, 1.0);

        let path = kg.bfs_path(a, c).unwrap();
        assert_eq!(path, vec![a, b, c]);
    }

    #[test]
    fn bfs_returns_none_when_unreachable() {
        let mut kg = KnowledgeGraph::new();
        let a = kg.add_entity("N", "A");
        let b = kg.add_entity("N", "B");
        assert!(kg.bfs_path(a, b).is_none());
    }

    #[test]
    fn transitive_closure_follows_chain() {
        let mut kg = KnowledgeGraph::new();
        let dog = kg.add_entity("Animal", "Dog");
        let mammal = kg.add_entity("Class", "Mammal");
        let animal = kg.add_entity("Class", "Animal");
        kg.add_relation(dog, mammal, RelationType::IsA, 1.0);
        kg.add_relation(mammal, animal, RelationType::IsA, 1.0);

        let closure = kg.transitive_closure(dog, &RelationType::IsA);
        assert!(closure.contains(&mammal));
        assert!(closure.contains(&animal));
        assert!(!closure.contains(&dog));
    }

    #[test]
    fn infer_is_a_returns_ancestors() {
        let mut kg = KnowledgeGraph::new();
        let sparrow = kg.add_entity("Animal", "Sparrow");
        let bird = kg.add_entity("Class", "Bird");
        let vertebrate = kg.add_entity("Class", "Vertebrate");
        kg.add_relation(sparrow, bird, RelationType::IsA, 1.0);
        kg.add_relation(bird, vertebrate, RelationType::IsA, 1.0);

        let ancestors = kg.infer_is_a(sparrow);
        assert!(ancestors.contains(&bird));
        assert!(ancestors.contains(&vertebrate));
    }

    #[test]
    fn common_ancestors_returned() {
        let mut kg = KnowledgeGraph::new();
        let cat = kg.add_entity("Animal", "Cat");
        let dog = kg.add_entity("Animal", "Dog");
        let mammal = kg.add_entity("Class", "Mammal");
        kg.add_relation(cat, mammal, RelationType::IsA, 1.0);
        kg.add_relation(dog, mammal, RelationType::IsA, 1.0);

        let common = kg.common_ancestors(cat, dog);
        assert!(common.contains(&mammal));
    }

    #[test]
    fn stats_count() {
        let mut kg = KnowledgeGraph::new();
        let a = kg.add_entity("T", "A");
        let b = kg.add_entity("T", "B");
        let c = kg.add_entity("T", "C");
        kg.add_relation(a, b, RelationType::RelatedTo, 1.0);
        // c is isolated → 2 connected components.

        let s = kg.stats();
        assert_eq!(s.entity_count, 3);
        assert_eq!(s.relation_count, 1);
        assert_eq!(s.connected_components, 2);
        let _ = c; // used implicitly via add_entity
    }
}
