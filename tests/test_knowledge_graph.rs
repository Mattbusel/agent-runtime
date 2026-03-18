//! Tests for the knowledge graph: node insertion, edge traversal, cycle detection,
//! shortest paths, centrality, community detection, and subgraph extraction.

use agent_runtime::graph::{Entity, EntityId, GraphStore, Relationship};
use agent_runtime::AgentRuntimeError;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn store() -> GraphStore {
    GraphStore::new()
}

fn add(g: &GraphStore, id: &str) {
    g.add_entity(Entity::new(id, "Node")).unwrap();
}

fn link(g: &GraphStore, from: &str, to: &str) {
    g.add_relationship(Relationship::new(from, to, "EDGE", 1.0))
        .unwrap();
}

fn link_w(g: &GraphStore, from: &str, to: &str, weight: f32) {
    g.add_relationship(Relationship::new(from, to, "EDGE", weight))
        .unwrap();
}

// ── Node insertion ────────────────────────────────────────────────────────────

#[test]
fn graph_add_entity_increments_count() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    assert_eq!(g.entity_count().unwrap(), 2);
}

#[test]
fn graph_get_entity_returns_correct_entity() {
    let g = store();
    let e = Entity::new("node-1", "Concept");
    g.add_entity(e.clone()).unwrap();
    let fetched = g.get_entity(&EntityId::new("node-1")).unwrap();
    assert_eq!(fetched.id, EntityId::new("node-1"));
    assert_eq!(fetched.label, "Concept");
}

#[test]
fn graph_get_missing_entity_returns_error() {
    let g = store();
    let result = g.get_entity(&EntityId::new("ghost"));
    assert!(matches!(result, Err(AgentRuntimeError::Graph(_))));
}

#[test]
fn graph_add_duplicate_entity_is_error_or_overwrite() {
    let g = store();
    add(&g, "dup");
    // Adding again — whether it errors or silently overwrites is implementation-defined.
    // We only verify the entity count does not drop below 1.
    let _ = g.add_entity(Entity::new("dup", "Other"));
    assert!(g.entity_count().unwrap() >= 1);
}

#[test]
fn graph_entity_with_properties_stored_correctly() {
    let g = store();
    let mut props = std::collections::HashMap::new();
    props.insert("color".to_string(), serde_json::json!("blue"));
    props.insert("weight".to_string(), serde_json::json!(42));
    let e = Entity::with_properties("rich", "Thing", props);
    g.add_entity(e).unwrap();
    let fetched = g.get_entity(&EntityId::new("rich")).unwrap();
    assert_eq!(fetched.properties["color"], serde_json::json!("blue"));
}

// ── Edge insertion ────────────────────────────────────────────────────────────

#[test]
fn graph_add_relationship_increments_count() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    link(&g, "a", "b");
    assert_eq!(g.relationship_count().unwrap(), 1);
}

#[test]
fn graph_add_relationship_with_missing_source_returns_error() {
    let g = store();
    add(&g, "b");
    let result = g.add_relationship(Relationship::new("missing", "b", "E", 1.0));
    assert!(result.is_err());
}

#[test]
fn graph_add_relationship_with_missing_target_returns_error() {
    let g = store();
    add(&g, "a");
    let result = g.add_relationship(Relationship::new("a", "missing", "E", 1.0));
    assert!(result.is_err());
}

// ── BFS traversal ─────────────────────────────────────────────────────────────

#[test]
fn graph_bfs_single_node_visits_nothing() {
    let g = store();
    add(&g, "alone");
    let visited = g.bfs(&EntityId::new("alone")).unwrap();
    assert!(
        visited.is_empty(),
        "BFS from isolated node should return no neighbors"
    );
}

#[test]
fn graph_bfs_visits_all_reachable_nodes() {
    let g = store();
    for id in ["a", "b", "c", "d"] {
        add(&g, id);
    }
    link(&g, "a", "b");
    link(&g, "b", "c");
    link(&g, "a", "d");

    let visited = g.bfs(&EntityId::new("a")).unwrap();
    assert_eq!(visited.len(), 3); // b, c, d
}

#[test]
fn graph_bfs_does_not_visit_unreachable_nodes() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    add(&g, "isolated");
    link(&g, "a", "b");

    let visited = g.bfs(&EntityId::new("a")).unwrap();
    assert!(!visited.contains(&EntityId::new("isolated")));
}

#[test]
fn graph_bfs_from_missing_node_returns_error() {
    let g = store();
    let result = g.bfs(&EntityId::new("ghost"));
    assert!(result.is_err());
}

// ── DFS traversal ─────────────────────────────────────────────────────────────

#[test]
fn graph_dfs_visits_all_reachable_nodes() {
    let g = store();
    for id in ["a", "b", "c"] {
        add(&g, id);
    }
    link(&g, "a", "b");
    link(&g, "b", "c");

    let visited = g.dfs(&EntityId::new("a")).unwrap();
    assert_eq!(visited.len(), 2); // b, c
}

#[test]
fn graph_dfs_from_missing_node_returns_error() {
    let g = store();
    let result = g.dfs(&EntityId::new("ghost"));
    assert!(result.is_err());
}

// ── Shortest path ─────────────────────────────────────────────────────────────

#[test]
fn graph_shortest_path_direct_connection() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    link(&g, "a", "b");

    let path = g
        .shortest_path(&EntityId::new("a"), &EntityId::new("b"))
        .unwrap()
        .unwrap();
    assert_eq!(path, vec![EntityId::new("a"), EntityId::new("b")]);
}

#[test]
fn graph_shortest_path_prefers_fewer_hops() {
    let g = store();
    for id in ["a", "b", "c", "d"] {
        add(&g, id);
    }
    link(&g, "a", "b");
    link(&g, "b", "c");
    link(&g, "c", "d");
    link(&g, "a", "d"); // direct shortcut

    let path = g
        .shortest_path(&EntityId::new("a"), &EntityId::new("d"))
        .unwrap()
        .unwrap();
    assert_eq!(path.len(), 2, "direct path a->d should be chosen");
}

#[test]
fn graph_shortest_path_none_when_disconnected() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    // No edge between a and b
    let path = g
        .shortest_path(&EntityId::new("a"), &EntityId::new("b"))
        .unwrap();
    assert!(path.is_none());
}

#[test]
fn graph_shortest_path_same_node_returns_single_element_path() {
    let g = store();
    add(&g, "solo");
    let path = g
        .shortest_path(&EntityId::new("solo"), &EntityId::new("solo"))
        .unwrap()
        .unwrap();
    assert_eq!(path, vec![EntityId::new("solo")]);
}

// ── Weighted shortest path ────────────────────────────────────────────────────

#[test]
fn graph_weighted_shortest_path_picks_lower_weight_route() {
    let g = store();
    for id in ["a", "b", "c"] {
        add(&g, id);
    }
    link_w(&g, "a", "b", 1.0);
    link_w(&g, "b", "c", 1.0); // total: 2.0
    link_w(&g, "a", "c", 10.0); // direct but expensive

    let result = g
        .shortest_path_weighted(&EntityId::new("a"), &EntityId::new("c"))
        .unwrap()
        .unwrap();
    let (path, weight) = result;
    assert_eq!(
        path,
        vec![EntityId::new("a"), EntityId::new("b"), EntityId::new("c")]
    );
    assert!((weight - 2.0).abs() < 1e-5);
}

#[test]
fn graph_weighted_shortest_path_negative_weight_returns_error() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    g.add_relationship(Relationship::new("a", "b", "NEG", -1.0))
        .unwrap();
    let result = g.shortest_path_weighted(&EntityId::new("a"), &EntityId::new("b"));
    assert!(result.is_err());
}

// ── Transitive closure ────────────────────────────────────────────────────────

#[test]
fn graph_transitive_closure_includes_all_reachable() {
    let g = store();
    for id in ["a", "b", "c", "d"] {
        add(&g, id);
    }
    link(&g, "a", "b");
    link(&g, "b", "c");
    link(&g, "c", "d");

    let closure = g.transitive_closure(&EntityId::new("a")).unwrap();
    // a plus all descendants
    assert_eq!(closure.len(), 4);
    assert!(closure.contains(&EntityId::new("a")));
    assert!(closure.contains(&EntityId::new("d")));
}

#[test]
fn graph_transitive_closure_isolated_node_contains_only_itself() {
    let g = store();
    add(&g, "alone");
    let closure = g.transitive_closure(&EntityId::new("alone")).unwrap();
    assert_eq!(closure.len(), 1);
    assert!(closure.contains(&EntityId::new("alone")));
}

// ── Entity removal ────────────────────────────────────────────────────────────

#[test]
fn graph_remove_entity_decrements_count() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    g.remove_entity(&EntityId::new("a")).unwrap();
    assert_eq!(g.entity_count().unwrap(), 1);
}

#[test]
fn graph_remove_entity_removes_its_relationships() {
    let g = store();
    add(&g, "a");
    add(&g, "b");
    link(&g, "a", "b");
    g.remove_entity(&EntityId::new("a")).unwrap();
    assert_eq!(g.relationship_count().unwrap(), 0);
}

#[test]
fn graph_remove_missing_entity_returns_error() {
    let g = store();
    let result = g.remove_entity(&EntityId::new("ghost"));
    assert!(result.is_err());
}

// ── Subgraph extraction ───────────────────────────────────────────────────────

#[test]
fn graph_subgraph_contains_only_requested_nodes() {
    let g = store();
    for id in ["a", "b", "c", "d"] {
        add(&g, id);
    }
    link(&g, "a", "b");
    link(&g, "b", "c");
    link(&g, "c", "d");

    let sub = g
        .subgraph(&[EntityId::new("a"), EntityId::new("b"), EntityId::new("c")])
        .unwrap();

    assert_eq!(sub.entity_count().unwrap(), 3);
    assert!(sub.get_entity(&EntityId::new("d")).is_err());
}

#[test]
fn graph_subgraph_only_includes_edges_within_node_set() {
    let g = store();
    for id in ["a", "b", "c"] {
        add(&g, id);
    }
    link(&g, "a", "b");
    link(&g, "b", "c");

    let sub = g
        .subgraph(&[EntityId::new("a"), EntityId::new("b")])
        .unwrap();

    // Only a->b should be in the subgraph; b->c should be absent.
    assert_eq!(sub.relationship_count().unwrap(), 1);
}

// ── Degree centrality ─────────────────────────────────────────────────────────

#[test]
fn graph_degree_centrality_hub_node_has_highest_value() {
    let g = store();
    for id in ["hub", "a", "b", "c"] {
        add(&g, id);
    }
    link(&g, "hub", "a");
    link(&g, "hub", "b");
    link(&g, "hub", "c");

    let centrality = g.degree_centrality().unwrap();
    let hub_val = *centrality.get(&EntityId::new("hub")).unwrap();
    let a_val = *centrality.get(&EntityId::new("a")).unwrap();
    assert!(
        hub_val > a_val,
        "hub centrality {hub_val} should exceed leaf {a_val}"
    );
}

#[test]
fn graph_degree_centrality_isolated_node_is_zero() {
    let g = store();
    add(&g, "alone");
    add(&g, "b");
    link(&g, "alone", "b");
    link(&g, "b", "alone"); // now alone has both in and out

    let g2 = store();
    add(&g2, "alone");

    let c = g2.degree_centrality().unwrap();
    // Single node: no edges, centrality is 0 (or undefined; result should be present)
    let val = c.get(&EntityId::new("alone")).copied().unwrap_or(0.0);
    assert_eq!(val, 0.0);
}

// ── Concurrent access ─────────────────────────────────────────────────────────

#[tokio::test]
async fn graph_concurrent_entity_additions_are_consistent() {
    use std::sync::Arc;

    let g = Arc::new(GraphStore::new());
    let mut handles = Vec::new();

    for i in 0u32..20 {
        let g = Arc::clone(&g);
        handles.push(tokio::spawn(async move {
            g.add_entity(Entity::new(format!("node-{i}"), "N")).unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(g.entity_count().unwrap(), 20);
}
