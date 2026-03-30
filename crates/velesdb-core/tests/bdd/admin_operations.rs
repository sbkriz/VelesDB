//! BDD integration tests for `VelesQL` admin statements: ANALYZE, TRUNCATE, ALTER COLLECTION.
//!
//! Tests the full pipeline: parse -> validate -> execute -> verify state.

use super::helpers::{create_test_db, execute_sql};
use velesdb_core::Point;

// ============================================================================
// ANALYZE — nominal
// ============================================================================

#[test]
fn test_analyze_returns_stats() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION docs (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let vc = db.get_vector_collection("docs").expect("get collection");
    vc.upsert(vec![
        Point::new(1, vec![1.0, 0.0, 0.0, 0.0], None),
        Point::new(2, vec![0.0, 1.0, 0.0, 0.0], None),
        Point::new(3, vec![0.0, 0.0, 1.0, 0.0], None),
    ])
    .expect("upsert");

    let results = execute_sql(&db, "ANALYZE docs").expect("ANALYZE should succeed");
    assert_eq!(results.len(), 1, "ANALYZE should return one result");

    let total = results[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("total_points"))
        .and_then(serde_json::Value::as_u64);
    assert_eq!(total, Some(3), "should report 3 total points");
}

#[test]
fn test_analyze_empty_collection() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION empty (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let results = execute_sql(&db, "ANALYZE COLLECTION empty")
        .expect("ANALYZE empty collection should succeed");
    assert_eq!(results.len(), 1);

    let total = results[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("total_points"))
        .and_then(serde_json::Value::as_u64);
    assert_eq!(
        total,
        Some(0),
        "empty collection should have 0 total points"
    );
}

// ============================================================================
// TRUNCATE — nominal
// ============================================================================

#[test]
fn test_truncate_deletes_all_data() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION trunc (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let vc = db.get_vector_collection("trunc").expect("get collection");
    vc.upsert(vec![
        Point::new(1, vec![1.0, 0.0, 0.0, 0.0], None),
        Point::new(2, vec![0.0, 1.0, 0.0, 0.0], None),
        Point::new(3, vec![0.0, 0.0, 1.0, 0.0], None),
        Point::new(4, vec![0.0, 0.0, 0.0, 1.0], None),
        Point::new(5, vec![1.0, 1.0, 0.0, 0.0], None),
    ])
    .expect("upsert");
    assert_eq!(vc.len(), 5);

    let results = execute_sql(&db, "TRUNCATE trunc").expect("TRUNCATE should succeed");
    assert_eq!(results.len(), 1, "TRUNCATE returns one result");

    let deleted = results[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("deleted_count"))
        .and_then(serde_json::Value::as_u64);
    assert_eq!(deleted, Some(5), "should report 5 deleted");

    assert_eq!(vc.len(), 0, "collection should be empty after TRUNCATE");
}

#[test]
fn test_truncate_empty_collection() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION empty_trunc (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let results = execute_sql(&db, "TRUNCATE empty_trunc").expect("TRUNCATE empty should succeed");
    assert_eq!(results.len(), 1);

    let deleted = results[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("deleted_count"))
        .and_then(serde_json::Value::as_u64);
    assert_eq!(
        deleted,
        Some(0),
        "TRUNCATE on empty collection reports 0 deleted"
    );
}

#[test]
fn test_truncate_then_insert() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION reinsert (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let vc = db
        .get_vector_collection("reinsert")
        .expect("get collection");
    vc.upsert(vec![Point::new(1, vec![1.0, 0.0, 0.0, 0.0], None)])
        .expect("upsert");

    execute_sql(&db, "TRUNCATE reinsert").expect("TRUNCATE");
    assert_eq!(vc.len(), 0, "should be empty after TRUNCATE");

    // Re-insert works normally
    vc.upsert(vec![Point::new(10, vec![0.0, 1.0, 0.0, 0.0], None)])
        .expect("re-insert");
    assert_eq!(vc.len(), 1, "re-insert should work after TRUNCATE");
}

// ============================================================================
// ALTER COLLECTION — nominal
// ============================================================================

#[test]
fn test_alter_collection_set_option() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION alter_test (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let results = execute_sql(&db, "ALTER COLLECTION alter_test SET (auto_reindex = true)")
        .expect("ALTER COLLECTION should succeed");

    assert!(
        results.is_empty(),
        "ALTER returns empty result set on success"
    );
}

// ============================================================================
// Complex / lifecycle scenarios
// ============================================================================

#[test]
fn test_analyze_truncate_lifecycle() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION lifecycle (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let vc = db
        .get_vector_collection("lifecycle")
        .expect("get collection");
    vc.upsert(vec![
        Point::new(1, vec![1.0, 0.0, 0.0, 0.0], None),
        Point::new(2, vec![0.0, 1.0, 0.0, 0.0], None),
        Point::new(3, vec![0.0, 0.0, 1.0, 0.0], None),
    ])
    .expect("upsert");

    // ANALYZE with data
    let stats1 = execute_sql(&db, "ANALYZE lifecycle").expect("first ANALYZE");
    let count1 = stats1[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("total_points"))
        .and_then(serde_json::Value::as_u64);
    assert_eq!(count1, Some(3));

    // TRUNCATE
    execute_sql(&db, "TRUNCATE lifecycle").expect("TRUNCATE");
    assert_eq!(vc.len(), 0);

    // ANALYZE after TRUNCATE
    let stats2 = execute_sql(&db, "ANALYZE lifecycle").expect("second ANALYZE");
    let count2 = stats2[0]
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get("total_points"))
        .and_then(serde_json::Value::as_u64);
    assert_eq!(count2, Some(0));
}

#[test]
fn test_truncate_preserves_collection() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION preserved (dimension = 4, metric = 'cosine')",
    )
    .expect("create");

    let vc = db
        .get_vector_collection("preserved")
        .expect("get collection");
    vc.upsert(vec![Point::new(1, vec![1.0, 0.0, 0.0, 0.0], None)])
        .expect("upsert");

    execute_sql(&db, "TRUNCATE COLLECTION preserved").expect("TRUNCATE");

    // Collection still shows in SHOW COLLECTIONS
    let show = execute_sql(&db, "SHOW COLLECTIONS").expect("SHOW");
    let names: Vec<&str> = show
        .iter()
        .filter_map(|r| {
            r.point
                .payload
                .as_ref()
                .and_then(|p| p.get("name"))
                .and_then(serde_json::Value::as_str)
        })
        .collect();
    assert!(
        names.contains(&"preserved"),
        "collection should still exist after TRUNCATE"
    );

    // Can re-insert
    vc.upsert(vec![Point::new(99, vec![0.0, 0.0, 0.0, 1.0], None)])
        .expect("re-insert after TRUNCATE");
    assert_eq!(vc.len(), 1, "re-insert works after TRUNCATE");
}

// ============================================================================
// Negative — nonexistent collections
// ============================================================================

#[test]
fn test_analyze_nonexistent_fails() {
    let (_dir, db) = create_test_db();

    let result = execute_sql(&db, "ANALYZE ghost");
    assert!(result.is_err(), "ANALYZE nonexistent should fail");
    let msg = result.expect_err("error").to_string();
    assert!(
        msg.contains("ghost") || msg.contains("not found") || msg.contains("NotFound"),
        "error should mention missing collection, got: {msg}"
    );
}

#[test]
fn test_truncate_nonexistent_fails() {
    let (_dir, db) = create_test_db();

    let result = execute_sql(&db, "TRUNCATE ghost");
    assert!(result.is_err(), "TRUNCATE nonexistent should fail");
    let msg = result.expect_err("error").to_string();
    assert!(
        msg.contains("ghost") || msg.contains("not found") || msg.contains("NotFound"),
        "error should mention missing collection, got: {msg}"
    );
}

#[test]
fn test_alter_nonexistent_fails() {
    let (_dir, db) = create_test_db();

    let result = execute_sql(&db, "ALTER COLLECTION ghost SET (auto_reindex = true)");
    assert!(result.is_err(), "ALTER nonexistent should fail");
    let msg = result.expect_err("error").to_string();
    assert!(
        msg.contains("ghost") || msg.contains("not found") || msg.contains("NotFound"),
        "error should mention missing collection, got: {msg}"
    );
}
