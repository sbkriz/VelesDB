//! BDD tests for VelesQL v3.5 Phase 5: SELECT EDGES and INSERT NODE.
//!
//! All tests exercise the full pipeline: SQL string -> parse -> execute -> verify.
//! Requires `persistence` feature (gated in `bdd.rs`).

use super::helpers::{create_test_db, execute_sql};

// =========================================================================
// Helper: create a graph collection and populate it with edges/nodes.
// =========================================================================

fn setup_graph_collection(db: &velesdb_core::Database) {
    execute_sql(
        db,
        "CREATE GRAPH COLLECTION kg (dimension = 4, metric = 'cosine') SCHEMALESS;",
    )
    .expect("test: CREATE GRAPH COLLECTION should succeed");
}

fn populate_graph_edges(db: &velesdb_core::Database) {
    execute_sql(
        db,
        "INSERT EDGE INTO kg (source = 1, target = 2, label = 'KNOWS');",
    )
    .expect("test: INSERT EDGE 1->2 KNOWS");
    execute_sql(
        db,
        "INSERT EDGE INTO kg (source = 1, target = 3, label = 'WORKS_WITH');",
    )
    .expect("test: INSERT EDGE 1->3 WORKS_WITH");
    execute_sql(
        db,
        "INSERT EDGE INTO kg (source = 2, target = 3, label = 'KNOWS');",
    )
    .expect("test: INSERT EDGE 2->3 KNOWS");
}

// =========================================================================
// A. SELECT EDGES — nominal
// =========================================================================

#[test]
fn test_select_all_edges() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);
    populate_graph_edges(&db);

    let results = execute_sql(&db, "SELECT EDGES FROM kg;").expect("SELECT EDGES should work");
    assert_eq!(results.len(), 3, "Graph has 3 edges");
}

#[test]
fn test_select_edges_by_source() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);
    populate_graph_edges(&db);

    let results =
        execute_sql(&db, "SELECT EDGES FROM kg WHERE source = 1;").expect("filter by source");
    assert_eq!(results.len(), 2, "Node 1 has 2 outgoing edges");
}

#[test]
fn test_select_edges_by_label() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);
    populate_graph_edges(&db);

    let results =
        execute_sql(&db, "SELECT EDGES FROM kg WHERE label = 'KNOWS';").expect("filter by label");
    assert_eq!(results.len(), 2, "Two KNOWS edges exist");
}

// =========================================================================
// B. INSERT NODE — nominal
// =========================================================================

#[test]
fn test_insert_node_creates_payload() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);

    execute_sql(
        &db,
        "INSERT NODE INTO kg (id = 42, payload = '{\"name\": \"Alice\"}');",
    )
    .expect("INSERT NODE should succeed");

    // Verify via API.
    let graph = db
        .get_graph_collection("kg")
        .expect("kg should be a graph collection");
    let payload = graph
        .get_node_payload(42)
        .expect("payload retrieval should succeed")
        .expect("payload should exist");
    assert_eq!(payload["name"], "Alice");
}

// =========================================================================
// C. Edge cases
// =========================================================================

#[test]
fn test_select_edges_empty_graph() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);

    let results = execute_sql(&db, "SELECT EDGES FROM kg;").expect("empty graph should work");
    assert_eq!(results.len(), 0, "No edges in empty graph");
}

#[test]
fn test_select_edges_with_limit() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);
    populate_graph_edges(&db);

    let results =
        execute_sql(&db, "SELECT EDGES FROM kg LIMIT 2;").expect("LIMIT should be respected");
    assert_eq!(results.len(), 2, "LIMIT 2 should return at most 2");
}

#[test]
fn test_insert_node_updates_existing() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);

    execute_sql(
        &db,
        "INSERT NODE INTO kg (id = 10, payload = '{\"name\": \"Bob\"}');",
    )
    .expect("first INSERT NODE");

    execute_sql(
        &db,
        "INSERT NODE INTO kg (id = 10, payload = '{\"name\": \"Robert\"}');",
    )
    .expect("second INSERT NODE should update");

    let graph = db
        .get_graph_collection("kg")
        .expect("kg should be a graph collection");
    let payload = graph
        .get_node_payload(10)
        .expect("retrieval should succeed")
        .expect("payload should exist");
    assert_eq!(payload["name"], "Robert", "Payload should be updated");
}

// =========================================================================
// D. Complex lifecycle
// =========================================================================

#[test]
fn test_graph_node_edge_lifecycle() {
    let (_dir, db) = create_test_db();
    setup_graph_collection(&db);

    // Insert nodes.
    execute_sql(
        &db,
        "INSERT NODE INTO kg (id = 1, payload = '{\"name\": \"Alice\"}');",
    )
    .expect("insert node 1");
    execute_sql(
        &db,
        "INSERT NODE INTO kg (id = 2, payload = '{\"name\": \"Bob\"}');",
    )
    .expect("insert node 2");

    // Insert edge.
    execute_sql(
        &db,
        "INSERT EDGE INTO kg (source = 1, target = 2, label = 'KNOWS');",
    )
    .expect("insert edge");

    // SELECT EDGES should show 1 edge.
    let results = execute_sql(&db, "SELECT EDGES FROM kg;").expect("select edges");
    assert_eq!(results.len(), 1);

    // Delete the edge (use a known auto-generated ID).
    let edge_id = results[0].point.id;
    let delete_sql = format!("DELETE EDGE {edge_id} FROM kg;");
    execute_sql(&db, &delete_sql).expect("delete edge");

    // SELECT EDGES should now return 0.
    let results_after = execute_sql(&db, "SELECT EDGES FROM kg;").expect("select after delete");
    assert_eq!(results_after.len(), 0);
}

// =========================================================================
// E. Negative
// =========================================================================

#[test]
fn test_select_edges_from_vector_collection_fails() {
    let (_dir, db) = create_test_db();

    // Create a vector collection (not a graph).
    execute_sql(
        &db,
        "CREATE COLLECTION docs (dimension = 4, metric = 'cosine');",
    )
    .expect("create vector collection");

    let err = execute_sql(&db, "SELECT EDGES FROM docs;").expect_err("should fail on non-graph");
    let msg = format!("{err}");
    assert!(
        msg.contains("not found") || msg.contains("NotFound"),
        "Error should indicate collection not found as graph: {msg}"
    );
}
