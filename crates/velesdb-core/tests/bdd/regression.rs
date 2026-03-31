//! BDD regression tests for bugs found by Devin Review.
//!
//! Each test documents the original bug and proves the fix holds.
//! Tests exercise the full pipeline: SQL string -> parse -> execute -> verify.

use super::helpers::{create_test_db, execute_sql, execute_sql_with_params, payload_str};
use std::collections::HashMap;
use velesdb_core::Point;

// ============================================================================
// Bug 1: INSERT INTO must return the inserted point (not empty results)
// ============================================================================

/// The server previously discarded INSERT INTO results by routing through a
/// mutation path that returned `Ok(())` instead of the inserted points.
/// At core level, `execute_insert()` builds `SearchResult` items and returns
/// them. This test ensures that contract is never broken.
#[test]
fn test_regression_insert_into_returns_results() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE METADATA COLLECTION reg").expect("create metadata collection");

    let results =
        execute_sql(&db, "INSERT INTO reg (id, title) VALUES (1, 'test')").expect("insert");

    assert!(
        !results.is_empty(),
        "INSERT INTO must return the inserted point, not empty results"
    );
    assert_eq!(
        results.len(),
        1,
        "single-row INSERT returns exactly 1 result"
    );
    assert_eq!(
        results[0].point.id, 1,
        "returned point must have correct id"
    );
}

/// Same bug, but for a vector collection (requires vector column).
#[test]
fn test_regression_insert_into_vector_returns_results() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION vins (dimension = 4, metric = 'cosine')",
    )
    .expect("create vector collection");

    let mut params = HashMap::new();
    params.insert("v".to_string(), serde_json::json!([1.0, 0.0, 0.0, 0.0]));

    let results = execute_sql_with_params(
        &db,
        "INSERT INTO vins (id, vector, title) VALUES (1, $v, 'test')",
        &params,
    )
    .expect("insert with vector");

    assert!(
        !results.is_empty(),
        "INSERT INTO vector collection must return the inserted point"
    );
    assert_eq!(results[0].point.id, 1);
}

// ============================================================================
// Bug 2: UPDATE must return updated points (not empty results)
// ============================================================================

/// UPDATE was routed through a mutation path that discarded results.
/// Core `upsert_and_collect()` builds `SearchResult` items from the updated
/// points. This test verifies both the return value and the state change.
#[test]
fn test_regression_update_returns_results_and_mutates_state() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION upd (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let vc = db.get_vector_collection("upd").expect("get collection");
    vc.upsert(vec![Point::new(
        1,
        vec![1.0, 0.0, 0.0, 0.0],
        Some(serde_json::json!({"status": "old"})),
    )])
    .expect("seed data");

    let results = execute_sql(&db, "UPDATE upd SET status = 'new' WHERE id = 1").expect("update");

    assert_eq!(results.len(), 1, "UPDATE should return the updated point");
    assert_eq!(results[0].point.id, 1, "updated point has correct id");

    // Verify state mutation persisted.
    let fetched = vc.get(&[1]);
    let payload = fetched[0]
        .as_ref()
        .expect("point should exist")
        .payload
        .as_ref()
        .expect("payload should exist");
    assert_eq!(
        payload.get("status").and_then(|v| v.as_str()),
        Some("new"),
        "payload field must reflect the UPDATE"
    );
}

// ============================================================================
// Bug 3: UPSERT must return results (same root cause as INSERT INTO)
// ============================================================================

/// UPSERT INTO shares the same `execute_insert` path as INSERT INTO.
/// This test ensures the returned results are non-empty.
#[test]
fn test_regression_upsert_returns_results() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE METADATA COLLECTION ups").expect("create metadata collection");

    let results =
        execute_sql(&db, "UPSERT INTO ups (id, title) VALUES (1, 'test')").expect("upsert");

    assert!(!results.is_empty(), "UPSERT must return the upserted point");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].point.id, 1);
}

/// UPSERT on a vector collection with bind-param vector.
#[test]
fn test_regression_upsert_vector_returns_results() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION vups (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let mut params = HashMap::new();
    params.insert("v".to_string(), serde_json::json!([0.0, 1.0, 0.0, 0.0]));

    let results = execute_sql_with_params(
        &db,
        "UPSERT INTO vups (id, vector, title) VALUES (1, $v, 'test')",
        &params,
    )
    .expect("upsert with vector");

    assert!(
        !results.is_empty(),
        "UPSERT INTO vector collection must return the upserted point"
    );
    assert_eq!(results[0].point.id, 1);
}

// ============================================================================
// Bug 4: Introspection queries work through Database::execute_query
//        without requiring a FROM clause
// ============================================================================

/// Introspection queries (SHOW, DESCRIBE, EXPLAIN) have an empty `from`
/// field in the AST. A naive implementation that resolves the collection
/// from `query.select.from` before checking for introspection would fail
/// with `CollectionNotFound("")`.
#[test]
fn test_regression_show_collections_no_from_clause() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION test_col (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let results = execute_sql(&db, "SHOW COLLECTIONS").expect("SHOW must work without FROM clause");

    assert!(
        !results.is_empty(),
        "SHOW COLLECTIONS must return at least the created collection"
    );
    let names: Vec<&str> = results
        .iter()
        .filter_map(|r| payload_str(r, "name"))
        .collect();
    assert!(
        names.contains(&"test_col"),
        "SHOW must include the created collection"
    );
}

#[test]
fn test_regression_describe_no_from_clause() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION desc_col (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let results = execute_sql(&db, "DESCRIBE desc_col")
        .expect("DESCRIBE must work without standard FROM clause");

    assert_eq!(results.len(), 1, "DESCRIBE returns exactly 1 result");
    assert_eq!(payload_str(&results[0], "name"), Some("desc_col"));
}

#[test]
fn test_regression_explain_no_from_clause() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION exp_col (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let results = execute_sql(&db, "EXPLAIN SELECT * FROM exp_col LIMIT 1")
        .expect("EXPLAIN must work through Database::execute_query");

    assert_eq!(results.len(), 1, "EXPLAIN returns exactly 1 plan result");

    let has_plan = results[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("plan"))
        .is_some();
    assert!(has_plan, "EXPLAIN result must contain a 'plan' field");
}

// ============================================================================
// Bug 5: FLUSH works through Database::execute_query without FROM clause
// ============================================================================

/// FLUSH is an admin statement, not a SELECT. It must bypass the
/// collection-resolution path entirely and delegate to `execute_admin`.
#[test]
fn test_regression_flush_no_from_clause() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION flush_col (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let results = execute_sql(&db, "FLUSH").expect("FLUSH must work without FROM clause");

    assert_eq!(results.len(), 1, "FLUSH returns one status result");

    let status = results[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("status"))
        .and_then(serde_json::Value::as_str);
    assert_eq!(
        status,
        Some("flushed"),
        "FLUSH should return status='flushed'"
    );
}

// ============================================================================
// Bug 6: All DDL variants extract collection name correctly
// ============================================================================

/// CREATE INDEX must correctly extract the collection name from the SQL
/// statement, not from the (possibly empty) `query.select.from` field.
/// A broken implementation would fail with `CollectionNotFound("")`.
#[test]
fn test_regression_create_index_collection_name_extracted() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION idx_col (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let result = execute_sql(&db, "CREATE INDEX ON idx_col (field1)");

    assert!(
        result.is_ok(),
        "CREATE INDEX must correctly extract collection name from SQL: {:?}",
        result.err()
    );
}

/// DROP INDEX must also resolve the collection name from the DDL statement.
#[test]
fn test_regression_drop_index_collection_name_extracted() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION didx (dimension = 4, metric = 'cosine')",
    )
    .expect("create");
    execute_sql(&db, "CREATE INDEX ON didx (tag)").expect("create index");

    let result = execute_sql(&db, "DROP INDEX ON didx (tag)");

    assert!(
        result.is_ok(),
        "DROP INDEX must correctly extract collection name: {:?}",
        result.err()
    );
}

// ============================================================================
// Bug 7: SELECT EDGES works as DML without a standard FROM clause
// ============================================================================

/// SELECT EDGES is dispatched as a DML statement, not a standard SELECT.
/// The collection name comes from the `SelectEdgesStatement.collection`
/// field, not from `query.select.from`. A broken dispatch would try to
/// resolve an empty collection name.
#[test]
fn test_regression_select_edges_no_standard_from() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE GRAPH COLLECTION edge_col (dimension = 4, metric = 'cosine') SCHEMALESS",
    )
    .expect("create graph collection");

    let results = execute_sql(&db, "SELECT EDGES FROM edge_col LIMIT 10")
        .expect("SELECT EDGES must work through DML dispatch");

    assert_eq!(
        results.len(),
        0,
        "empty graph collection should return 0 edges"
    );
}

/// SELECT EDGES with actual edges, verifying the full pipeline.
#[test]
fn test_regression_select_edges_returns_inserted_edges() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE GRAPH COLLECTION g (dimension = 4, metric = 'cosine') SCHEMALESS",
    )
    .expect("create");

    execute_sql(
        &db,
        "INSERT EDGE INTO g (source = 1, target = 2, label = 'KNOWS')",
    )
    .expect("insert edge");

    let results =
        execute_sql(&db, "SELECT EDGES FROM g LIMIT 10").expect("SELECT EDGES after insert");

    assert_eq!(results.len(), 1, "should return the inserted edge");

    let label = results[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("label"))
        .and_then(serde_json::Value::as_str);
    assert_eq!(label, Some("KNOWS"), "edge label must match");
}
