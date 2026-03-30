#![cfg(feature = "persistence")]
//! BDD-style end-to-end tests for `VelesQL` WHERE clause behaviors.
//!
//! Each scenario follows GIVEN (setup data) -> WHEN (execute SQL) -> THEN (verify
//! results).  Tests exercise the **full pipeline**: SQL string -> `Parser::parse()`
//! -> `Database::execute_query()` -> verify returned `SearchResult` values.
//!
//! Run with: `cargo test -p velesdb-core --features persistence -- velesql_where_bdd --test-threads=1`

#![allow(clippy::cast_precision_loss, clippy::uninlined_format_args)]

use std::collections::{HashMap, HashSet};

use serde_json::json;
use tempfile::TempDir;
use velesdb_core::{velesql::Parser, Database, Point, SearchResult};

// =========================================================================
// Helpers
// =========================================================================

/// Execute a `VelesQL` SQL string through the full pipeline: parse -> execute.
fn execute_sql(db: &Database, sql: &str) -> velesdb_core::Result<Vec<SearchResult>> {
    let query = Parser::parse(sql).map_err(|e| velesdb_core::Error::Query(e.to_string()))?;
    db.execute_query(&query, &HashMap::new())
}

/// Execute a `VelesQL` SQL string with bind parameters (e.g. `$v` for NEAR).
fn execute_sql_with_params(
    db: &Database,
    sql: &str,
    params: &HashMap<String, serde_json::Value>,
) -> velesdb_core::Result<Vec<SearchResult>> {
    let query = Parser::parse(sql).map_err(|e| velesdb_core::Error::Query(e.to_string()))?;
    db.execute_query(&query, params)
}

/// Create a fresh database in a temp directory.
fn create_test_db() -> (TempDir, Database) {
    let dir = TempDir::new().expect("test: create temp dir");
    let db = Database::open(dir.path()).expect("test: open database");
    (dir, db)
}

/// Populate a `products` collection with diverse test data for WHERE filtering.
///
/// | id | category    | price  | name             | brand  | stock | active |
/// |----|-------------|--------|------------------|--------|-------|--------|
/// | 1  | electronics | 299.99 | Laptop           | Acme   | 50    | true   |
/// | 2  | electronics | 99.99  | Mouse            | Acme   | 200   | true   |
/// | 3  | books       | 19.99  | Rust Programming | null   | 30    | true   |
/// | 4  | books       | 29.99  | Python Cookbook   | null   | 0     | false  |
/// | 5  | clothing    | 49.99  | T-Shirt          | Style  | 100   | true   |
fn setup_products_collection(db: &Database) {
    execute_sql(
        db,
        "CREATE COLLECTION products (dimension = 4, metric = 'cosine');",
    )
    .expect("test: CREATE products");

    let vc = db
        .get_vector_collection("products")
        .expect("test: get products collection");

    vc.upsert(vec![
        Point::new(
            1,
            vec![1.0, 0.0, 0.0, 0.0],
            Some(json!({"category": "electronics", "price": 299.99, "name": "Laptop", "brand": "Acme", "stock": 50, "active": true})),
        ),
        Point::new(
            2,
            vec![0.0, 1.0, 0.0, 0.0],
            Some(json!({"category": "electronics", "price": 99.99, "name": "Mouse", "brand": "Acme", "stock": 200, "active": true})),
        ),
        Point::new(
            3,
            vec![0.0, 0.0, 1.0, 0.0],
            Some(json!({"category": "books", "price": 19.99, "name": "Rust Programming", "brand": null, "stock": 30, "active": true})),
        ),
        Point::new(
            4,
            vec![0.0, 0.0, 0.0, 1.0],
            Some(json!({"category": "books", "price": 29.99, "name": "Python Cookbook", "brand": null, "stock": 0, "active": false})),
        ),
        Point::new(
            5,
            vec![0.5, 0.5, 0.0, 0.0],
            Some(json!({"category": "clothing", "price": 49.99, "name": "T-Shirt", "brand": "Style", "stock": 100, "active": true})),
        ),
    ])
    .expect("test: upsert products");
}

/// Collect result IDs into a `HashSet` for order-independent comparison.
fn result_ids(results: &[SearchResult]) -> HashSet<u64> {
    results.iter().map(|r| r.point.id).collect()
}

// =========================================================================
// Scenario 1: Equality filter returns matching rows
// =========================================================================

#[test]
fn test_where_equality_returns_matching_rows() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN executing an equality filter on category
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE category = 'electronics' LIMIT 10;",
    )
    .expect("test: equality filter should succeed");

    // THEN exactly 2 results are returned (ids 1 and 2)
    assert_eq!(
        results.len(),
        2,
        "Should return exactly 2 electronics items"
    );
    let ids = result_ids(&results);
    assert!(ids.contains(&1), "Should contain Laptop (id=1)");
    assert!(ids.contains(&2), "Should contain Mouse (id=2)");
}

// =========================================================================
// Scenario 2: Greater-than filter
// =========================================================================

#[test]
fn test_where_greater_than_filters_correctly() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering for price > 50
    let results = execute_sql(&db, "SELECT * FROM products WHERE price > 50 LIMIT 10;")
        .expect("test: greater-than filter should succeed");

    // THEN returns items where price > 50 (ids 1 at 299.99, 2 at 99.99)
    let ids = result_ids(&results);
    assert_eq!(ids.len(), 2, "Should return 2 items with price > 50");
    assert!(ids.contains(&1), "Laptop (299.99) should match price > 50");
    assert!(ids.contains(&2), "Mouse (99.99) should match price > 50");
}

// =========================================================================
// Scenario 3: AND combines filters
// =========================================================================

#[test]
fn test_where_and_combines_filters() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN combining category = 'electronics' AND price < 200
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE category = 'electronics' AND price < 200 LIMIT 10;",
    )
    .expect("test: AND filter should succeed");

    // THEN only Mouse (id 2, price 99.99) matches both conditions
    assert_eq!(
        results.len(),
        1,
        "Only Mouse (99.99) is electronics under 200"
    );
    assert_eq!(results[0].point.id, 2, "Result should be Mouse (id=2)");
}

// =========================================================================
// Scenario 4: OR widens results
// =========================================================================

#[test]
fn test_where_or_widens_results() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering category = 'electronics' OR category = 'clothing'
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE category = 'electronics' OR category = 'clothing' LIMIT 10;",
    )
    .expect("test: OR filter should succeed");

    // THEN returns 3 results (ids 1, 2, 5)
    let ids = result_ids(&results);
    assert_eq!(
        ids.len(),
        3,
        "Should return electronics + clothing = 3 items"
    );
    let expected: HashSet<u64> = [1, 2, 5].into_iter().collect();
    assert_eq!(ids, expected, "Should contain ids 1, 2, 5");
}

// =========================================================================
// Scenario 5: IN matches set membership
// =========================================================================

#[test]
fn test_where_in_matches_set_membership() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering category IN ('books', 'clothing')
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE category IN ('books', 'clothing') LIMIT 10;",
    )
    .expect("test: IN filter should succeed");

    // THEN returns 3 results (ids 3, 4, 5)
    let ids = result_ids(&results);
    assert_eq!(ids.len(), 3, "Should return books + clothing = 3 items");
    let expected: HashSet<u64> = [3, 4, 5].into_iter().collect();
    assert_eq!(ids, expected, "Should contain ids 3, 4, 5");
}

// =========================================================================
// Scenario 6: NOT IN excludes values
// =========================================================================

#[test]
fn test_where_not_in_excludes_values() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering category NOT IN ('electronics')
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE category NOT IN ('electronics') LIMIT 10;",
    )
    .expect("test: NOT IN filter should succeed");

    // THEN returns 3 results (books + clothing = ids 3, 4, 5)
    let ids = result_ids(&results);
    assert_eq!(ids.len(), 3, "Should exclude electronics, leaving 3 items");
    let expected: HashSet<u64> = [3, 4, 5].into_iter().collect();
    assert_eq!(ids, expected, "Should contain ids 3, 4, 5");
}

// =========================================================================
// Scenario 7: BETWEEN filters range inclusively
// =========================================================================

#[test]
fn test_where_between_filters_range_inclusively() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering price BETWEEN 20 AND 100
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE price BETWEEN 20 AND 100 LIMIT 10;",
    )
    .expect("test: BETWEEN filter should succeed");

    // THEN returns ids where 20 <= price <= 100 (ids 2=99.99, 4=29.99, 5=49.99)
    let ids = result_ids(&results);
    assert_eq!(ids.len(), 3, "Three items have price in [20, 100]");
    let expected: HashSet<u64> = [2, 4, 5].into_iter().collect();
    assert_eq!(ids, expected, "Should contain ids 2, 4, 5");
}

// =========================================================================
// Scenario 8: IS NULL finds null values
// =========================================================================

#[test]
fn test_where_is_null_finds_null_values() {
    // GIVEN a products collection with 5 items (books have null brand)
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering brand IS NULL
    let results = execute_sql(&db, "SELECT * FROM products WHERE brand IS NULL LIMIT 10;")
        .expect("test: IS NULL filter should succeed");

    // THEN returns ids 3 and 4 (books with null brand)
    let ids = result_ids(&results);
    assert_eq!(ids.len(), 2, "Two items have null brand");
    let expected: HashSet<u64> = [3, 4].into_iter().collect();
    assert_eq!(ids, expected, "Should contain ids 3, 4 (books)");
}

// =========================================================================
// Scenario 9: IS NOT NULL finds non-null values
// =========================================================================

#[test]
fn test_where_is_not_null_finds_non_null_values() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering brand IS NOT NULL
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE brand IS NOT NULL LIMIT 10;",
    )
    .expect("test: IS NOT NULL filter should succeed");

    // THEN returns ids 1, 2, 5 (items with brand set)
    let ids = result_ids(&results);
    assert_eq!(ids.len(), 3, "Three items have non-null brand");
    let expected: HashSet<u64> = [1, 2, 5].into_iter().collect();
    assert_eq!(ids, expected, "Should contain ids 1, 2, 5");
}

// =========================================================================
// Scenario 10: LIKE pattern matching (substring)
// =========================================================================

#[test]
fn test_where_like_substring_match() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering name LIKE '%Shirt%'
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE name LIKE '%Shirt%' LIMIT 10;",
    )
    .expect("test: LIKE substring filter should succeed");

    // THEN returns id 5 (T-Shirt)
    assert_eq!(results.len(), 1, "Only T-Shirt contains 'Shirt'");
    assert_eq!(results[0].point.id, 5, "Result should be T-Shirt (id=5)");
}

// =========================================================================
// Scenario 11: LIKE prefix match
// =========================================================================

#[test]
fn test_where_like_prefix_match() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering name LIKE 'Rust%'
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE name LIKE 'Rust%' LIMIT 10;",
    )
    .expect("test: LIKE prefix filter should succeed");

    // THEN returns id 3 (Rust Programming)
    assert_eq!(
        results.len(),
        1,
        "Only 'Rust Programming' starts with 'Rust'"
    );
    assert_eq!(
        results[0].point.id, 3,
        "Result should be Rust Programming (id=3)"
    );
}

// =========================================================================
// Scenario 12: No match returns empty results
// =========================================================================

#[test]
fn test_where_no_match_returns_empty() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering for a category that does not exist
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE category = 'automotive' LIMIT 10;",
    )
    .expect("test: non-matching filter should succeed");

    // THEN returns 0 results
    assert!(
        results.is_empty(),
        "No items match category='automotive', should return empty"
    );
}

// =========================================================================
// Scenario 13: Complex AND + OR with grouping
// =========================================================================

#[test]
fn test_where_complex_and_or_grouping() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN filtering (category = 'electronics' OR category = 'books') AND price < 30
    let results = execute_sql(
        &db,
        "SELECT * FROM products WHERE (category = 'electronics' OR category = 'books') AND price < 30 LIMIT 10;",
    )
    .expect("test: complex AND+OR filter should succeed");

    // THEN returns ids 3 (books, 19.99) and 4 (books, 29.99) — electronics are all >= 99.99
    let ids = result_ids(&results);
    assert_eq!(ids.len(), 2, "Only two items match the compound condition");
    let expected: HashSet<u64> = [3, 4].into_iter().collect();
    assert_eq!(ids, expected, "Should contain ids 3, 4 (books under 30)");
}

// =========================================================================
// Scenario 14: Vector NEAR with metadata filter
// =========================================================================

#[test]
fn test_where_vector_near_with_filter() {
    // GIVEN a products collection with 5 items
    let (_dir, db) = create_test_db();
    setup_products_collection(&db);

    // WHEN querying with vector NEAR $v AND category = 'electronics' LIMIT 2
    let mut params = HashMap::new();
    params.insert("v".to_string(), json!([1.0_f32, 0.0, 0.0, 0.0]));

    let results = execute_sql_with_params(
        &db,
        "SELECT * FROM products WHERE vector NEAR $v AND category = 'electronics' LIMIT 2;",
        &params,
    )
    .expect("test: NEAR + filter should succeed");

    // THEN returns at most 2 results, all from the electronics category
    assert!(
        results.len() <= 2,
        "LIMIT 2 should cap at 2 results, got {}",
        results.len()
    );
    assert!(
        !results.is_empty(),
        "Should return at least 1 electronics result"
    );

    for r in &results {
        let cat = r
            .point
            .payload
            .as_ref()
            .and_then(|p| p.get("category"))
            .and_then(serde_json::Value::as_str);
        assert_eq!(
            cat,
            Some("electronics"),
            "All NEAR results must pass the category filter, got {:?} for id={}",
            cat,
            r.point.id
        );
    }

    // The result IDs must be from the electronics set {1, 2}
    let ids = result_ids(&results);
    let electronics: HashSet<u64> = [1, 2].into_iter().collect();
    assert!(
        ids.is_subset(&electronics),
        "NEAR + filter results should be within electronics ids, got {:?}",
        ids
    );
}
