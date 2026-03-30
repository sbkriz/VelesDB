//! BDD-style end-to-end tests for `VelesQL` operators and statements that are
//! **not** covered by the `where_filters` or `vector_search` suites.
//!
//! Coverage: UPDATE, comparison operators (`<`, `<=`, `>=`, `!=`, `<>`),
//! NOT operator, ILIKE, LIKE edge cases, string escaping, and SQL comments.
//!
//! Each scenario follows GIVEN (setup data) -> WHEN (execute SQL) -> THEN
//! (verify results). Tests exercise the **full pipeline**: SQL string ->
//! `Parser::parse()` -> `Database::execute_query()` -> verify state.

use std::collections::HashSet;

use serde_json::json;
use velesdb_core::{Database, Point};

use super::helpers::{create_test_db, execute_sql, result_ids};

// =========================================================================
// Module-specific setup
// =========================================================================

/// Populate an `items` collection with diverse test data.
///
/// | id | name      | price | status   | category |
/// |----|-----------|-------|----------|----------|
/// | 1  | O'Brien   | 10    | active   | A        |
/// | 2  | Alice     | 20    | active   | B        |
/// | 3  | Bob       | 30    | deleted  | A        |
/// | 4  | CHARLIE   | 40    | pending  | B        |
/// | 5  | alice     | 50    | active   | C        |
fn setup_items(db: &Database) {
    execute_sql(
        db,
        "CREATE COLLECTION items (dimension = 4, metric = 'cosine');",
    )
    .expect("test: CREATE items");

    let vc = db
        .get_vector_collection("items")
        .expect("test: get items collection");

    vc.upsert(vec![
        Point::new(
            1,
            vec![1.0, 0.0, 0.0, 0.0],
            Some(json!({"name": "O'Brien", "price": 10, "status": "active", "category": "A"})),
        ),
        Point::new(
            2,
            vec![0.0, 1.0, 0.0, 0.0],
            Some(json!({"name": "Alice", "price": 20, "status": "active", "category": "B"})),
        ),
        Point::new(
            3,
            vec![0.0, 0.0, 1.0, 0.0],
            Some(json!({"name": "Bob", "price": 30, "status": "deleted", "category": "A"})),
        ),
        Point::new(
            4,
            vec![0.0, 0.0, 0.0, 1.0],
            Some(json!({"name": "CHARLIE", "price": 40, "status": "pending", "category": "B"})),
        ),
        Point::new(
            5,
            vec![0.5, 0.5, 0.0, 0.0],
            Some(json!({"name": "alice", "price": 50, "status": "active", "category": "C"})),
        ),
    ])
    .expect("test: upsert items");
}

// =========================================================================
// UPDATE: single field
// =========================================================================

/// GIVEN items, WHEN UPDATE sets status to 'archived' WHERE id = 1,
/// THEN direct get confirms the field changed.
#[test]
fn test_update_changes_field_value() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    execute_sql(&db, "UPDATE items SET status = 'archived' WHERE id = 1;")
        .expect("test: UPDATE single field");

    let vc = db.get_vector_collection("items").expect("test: get items");
    let points = vc.get(&[1]);
    let point = points[0].as_ref().expect("test: point 1 should exist");
    let status = point
        .payload
        .as_ref()
        .and_then(|p| p.get("status"))
        .and_then(serde_json::Value::as_str);
    assert_eq!(
        status,
        Some("archived"),
        "status should be 'archived' after UPDATE"
    );
}

// =========================================================================
// UPDATE: multiple fields
// =========================================================================

/// GIVEN items, WHEN UPDATE sets status='sold' and price=0 WHERE id = 3,
/// THEN both fields reflect the new values.
#[test]
fn test_update_multiple_fields() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    execute_sql(
        &db,
        "UPDATE items SET status = 'sold', price = 0 WHERE id = 3;",
    )
    .expect("test: UPDATE multiple fields");

    let vc = db.get_vector_collection("items").expect("test: get items");
    let points = vc.get(&[3]);
    let point = points[0].as_ref().expect("test: point 3 should exist");
    let payload = point.payload.as_ref().expect("test: payload should exist");

    assert_eq!(
        payload.get("status").and_then(serde_json::Value::as_str),
        Some("sold"),
        "status should be 'sold'"
    );
    assert_eq!(
        payload.get("price").and_then(serde_json::Value::as_i64),
        Some(0),
        "price should be 0"
    );
}

// =========================================================================
// UPDATE: no matching rows (no-op)
// =========================================================================

/// GIVEN items, WHEN UPDATE targets a nonexistent id,
/// THEN no error occurs and other items are unchanged.
#[test]
fn test_update_with_no_matching_rows() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let result = execute_sql(&db, "UPDATE items SET status = 'x' WHERE id = 999;");
    assert!(result.is_ok(), "UPDATE with no matches should not error");

    // Verify no item was touched
    let all = execute_sql(&db, "SELECT * FROM items LIMIT 10;")
        .expect("test: SELECT all after no-op UPDATE");
    assert_eq!(all.len(), 5, "All 5 items should still exist");

    for r in &all {
        let status = r
            .point
            .payload
            .as_ref()
            .and_then(|p| p.get("status"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        assert_ne!(
            status, "x",
            "No item should have status='x', but id={} does",
            r.point.id
        );
    }
}

// =========================================================================
// UPDATE: without WHERE updates all rows
// =========================================================================

/// GIVEN items, WHEN UPDATE omits WHERE, THEN all 5 items are updated.
#[test]
fn test_update_without_where_updates_all() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    execute_sql(&db, "UPDATE items SET category = 'X';").expect("test: UPDATE all rows");

    let all = execute_sql(&db, "SELECT * FROM items LIMIT 10;")
        .expect("test: SELECT all after blanket UPDATE");
    assert_eq!(all.len(), 5, "All 5 items should exist");

    for r in &all {
        let cat = r
            .point
            .payload
            .as_ref()
            .and_then(|p| p.get("category"))
            .and_then(serde_json::Value::as_str);
        assert_eq!(
            cat,
            Some("X"),
            "Every item should have category='X', but id={} has {:?}",
            r.point.id,
            cat
        );
    }
}

// =========================================================================
// Comparison: less than (<)
// =========================================================================

/// GIVEN items, WHEN WHERE price < 25, THEN ids {1,2} match.
#[test]
fn test_where_less_than() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items WHERE price < 25 LIMIT 10;")
        .expect("test: less-than filter");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [1, 2].into_iter().collect();
    assert_eq!(ids, expected, "price < 25 should match ids {{1, 2}}");
}

// =========================================================================
// Comparison: less than or equal (<=)
// =========================================================================

/// GIVEN items, WHEN WHERE price <= 20, THEN ids {1,2} match.
#[test]
fn test_where_less_than_or_equal() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items WHERE price <= 20 LIMIT 10;")
        .expect("test: less-than-or-equal filter");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [1, 2].into_iter().collect();
    assert_eq!(ids, expected, "price <= 20 should match ids {{1, 2}}");
}

// =========================================================================
// Comparison: greater than or equal (>=)
// =========================================================================

/// GIVEN items, WHEN WHERE price >= 40, THEN ids {4,5} match.
#[test]
fn test_where_greater_than_or_equal() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items WHERE price >= 40 LIMIT 10;")
        .expect("test: greater-than-or-equal filter");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [4, 5].into_iter().collect();
    assert_eq!(ids, expected, "price >= 40 should match ids {{4, 5}}");
}

// =========================================================================
// Comparison: not equal (!=)
// =========================================================================

/// GIVEN items, WHEN WHERE status != 'active', THEN ids {3,4} match.
#[test]
fn test_where_not_equal() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(
        &db,
        "SELECT * FROM items WHERE status != 'active' LIMIT 10;",
    )
    .expect("test: not-equal filter");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [3, 4].into_iter().collect();
    assert_eq!(
        ids, expected,
        "status != 'active' should match ids {{3, 4}} (deleted, pending)"
    );
}

// =========================================================================
// NOT operator: negates condition
// =========================================================================

/// GIVEN items, WHEN WHERE NOT (status = 'active'), THEN ids {3,4} match.
#[test]
fn test_where_not_negates_condition() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(
        &db,
        "SELECT * FROM items WHERE NOT (status = 'active') LIMIT 10;",
    )
    .expect("test: NOT condition filter");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [3, 4].into_iter().collect();
    assert_eq!(
        ids, expected,
        "NOT (status = 'active') should match ids {{3, 4}}"
    );
}

// =========================================================================
// NOT operator: with IN
// =========================================================================

/// GIVEN items, WHEN WHERE NOT (category IN ('A', 'B')), THEN id {5} matches.
#[test]
fn test_where_not_with_in() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(
        &db,
        "SELECT * FROM items WHERE NOT (category IN ('A', 'B')) LIMIT 10;",
    )
    .expect("test: NOT IN filter");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [5].into_iter().collect();
    assert_eq!(
        ids, expected,
        "NOT (category IN ('A', 'B')) should match only id {{5}} (category C)"
    );
}

// =========================================================================
// ILIKE: case-insensitive exact match
// =========================================================================

/// GIVEN items, WHEN WHERE name ILIKE 'alice', THEN ids {2,5} match.
#[test]
fn test_ilike_case_insensitive_match() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(
        &db,
        "SELECT * FROM items WHERE name ILIKE 'alice' LIMIT 10;",
    )
    .expect("test: ILIKE exact case-insensitive");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [2, 5].into_iter().collect();
    assert_eq!(
        ids, expected,
        "ILIKE 'alice' should match both Alice (id=2) and alice (id=5)"
    );
}

// =========================================================================
// ILIKE: pattern match with wildcards
// =========================================================================

/// GIVEN items, WHEN WHERE name ILIKE '%LI%', THEN ids {2,4,5} match.
#[test]
fn test_ilike_pattern_match() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items WHERE name ILIKE '%LI%' LIMIT 10;")
        .expect("test: ILIKE pattern case-insensitive");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [2, 4, 5].into_iter().collect();
    assert_eq!(
        ids, expected,
        "ILIKE '%LI%' should match Alice (id=2), CHARLIE (id=4), alice (id=5)"
    );
}

// =========================================================================
// LIKE: case-sensitive exact match
// =========================================================================

/// GIVEN items, WHEN WHERE name LIKE 'alice', THEN only id 5 matches.
#[test]
fn test_like_is_case_sensitive() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items WHERE name LIKE 'alice' LIMIT 10;")
        .expect("test: LIKE case-sensitive exact");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [5].into_iter().collect();
    assert_eq!(
        ids, expected,
        "LIKE 'alice' (case-sensitive) should match only lowercase alice (id=5)"
    );
}

// =========================================================================
// LIKE: underscore wildcard
// =========================================================================

/// GIVEN items, WHEN WHERE name LIKE 'Bo_', THEN id 3 (Bob) matches.
#[test]
fn test_like_wildcard_underscore() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items WHERE name LIKE 'Bo_' LIMIT 10;")
        .expect("test: LIKE underscore wildcard");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [3].into_iter().collect();
    assert_eq!(ids, expected, "LIKE 'Bo_' should match only Bob (id=3)");
}

// =========================================================================
// String escaping: single-quote in value
// =========================================================================

/// GIVEN items with name "O'Brien" (id=1),
/// WHEN WHERE name = 'O''Brien', THEN id 1 matches.
#[test]
fn test_string_escaping_obrien() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items WHERE name = 'O''Brien' LIMIT 10;")
        .expect("test: string escaping with doubled single quote");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [1].into_iter().collect();
    assert_eq!(ids, expected, "Escaped O''Brien should match id 1");
}

// =========================================================================
// Comments: line comment is ignored
// =========================================================================

/// GIVEN items, WHEN query contains a line comment (--),
/// THEN the comment is ignored and all 5 items are returned.
#[test]
fn test_comment_ignored_in_query() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(&db, "SELECT * FROM items -- get all\nLIMIT 10;")
        .expect("test: line comment should be ignored by parser");

    assert_eq!(
        results.len(),
        5,
        "Comment should be stripped; all 5 items should be returned"
    );
}

// =========================================================================
// Negative: UPDATE nonexistent collection
// =========================================================================

/// GIVEN no collection named 'ghost',
/// WHEN UPDATE ghost SET x = 1 WHERE id = 1,
/// THEN an error mentioning the collection is returned.
#[test]
fn test_update_nonexistent_collection_fails() {
    let (_dir, db) = create_test_db();

    let err = execute_sql(&db, "UPDATE ghost SET x = 1 WHERE id = 1;")
        .expect_err("UPDATE on nonexistent collection should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("ghost") || msg.contains("not found") || msg.contains("NotFound"),
        "Error should mention the missing collection, got: {msg}"
    );
}

// =========================================================================
// Comparison: diamond not-equal (<>)
// =========================================================================

/// GIVEN items, WHEN WHERE status <> 'active',
/// THEN same result as != (ids {3,4}).
#[test]
fn test_where_not_equal_diamond_syntax() {
    let (_dir, db) = create_test_db();
    setup_items(&db);

    let results = execute_sql(
        &db,
        "SELECT * FROM items WHERE status <> 'active' LIMIT 10;",
    )
    .expect("test: diamond not-equal filter");

    let ids = result_ids(&results);
    let expected: HashSet<u64> = [3, 4].into_iter().collect();
    assert_eq!(
        ids, expected,
        "status <> 'active' should match ids {{3, 4}}, same as !="
    );
}
