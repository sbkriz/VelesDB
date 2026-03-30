#![cfg(feature = "persistence")]
//! BDD-style end-to-end tests for advanced `VelesQL` behaviors.
//!
//! Covers features MISSING from other BDD suites: `similarity()` threshold,
//! `ORDER BY similarity()`, `MATCH` text (BM25), `MATCH` graph patterns,
//! and temporal expressions (`NOW() - INTERVAL '...'`).
//!
//! Each scenario follows Given-When-Then structure:
//! - **Given**: a collection with known, deterministic data
//! - **When**: a `VelesQL` query is executed through the full pipeline
//! - **Then**: results match expected behavior

#![allow(clippy::cast_precision_loss, clippy::uninlined_format_args)]

use std::collections::HashMap;

use serde_json::json;
use tempfile::TempDir;
use velesdb_core::{velesql::Parser, Database, Point, SearchResult};

// =========================================================================
// Helpers
// =========================================================================

/// Execute a `VelesQL` SQL string through the full pipeline: parse -> validate -> execute.
fn execute_sql(db: &Database, sql: &str) -> velesdb_core::Result<Vec<SearchResult>> {
    let query = Parser::parse(sql).map_err(|e| velesdb_core::Error::Query(e.to_string()))?;
    db.execute_query(&query, &HashMap::new())
}

/// Execute a `VelesQL` SQL string with named parameters.
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

/// Build a param map with a single vector parameter named `$v`.
fn vector_param(v: &[f32]) -> HashMap<String, serde_json::Value> {
    let mut params = HashMap::new();
    params.insert("v".to_string(), json!(v));
    params
}

/// Populate a "docs" collection with known vectors for similarity testing.
///
/// - Science (ids 1-3): vectors near `[1,0,0,0]`
/// - Tech (ids 4-5): vectors near `[0,1,0,0]`
/// - History (ids 6-7): vectors near `[0,0,1,0]`
/// - Misc outlier (id 8): uniform `[0.25, 0.25, 0.25, 0.25]`
fn setup_similarity_collection(db: &Database) {
    execute_sql(
        db,
        "CREATE COLLECTION docs (dimension = 4, metric = 'cosine');",
    )
    .expect("test: CREATE docs");

    let vc = db
        .get_vector_collection("docs")
        .expect("test: docs collection must exist");

    vc.upsert(vec![
        Point::new(1, vec![1.0, 0.0, 0.0, 0.0], Some(json!({"title":"Physics 101","category":"science"}))),
        Point::new(2, vec![0.95, 0.05, 0.0, 0.0], Some(json!({"title":"Chemistry Basics","category":"science"}))),
        Point::new(3, vec![0.9, 0.1, 0.0, 0.0], Some(json!({"title":"Biology Today","category":"science"}))),
        Point::new(4, vec![0.0, 1.0, 0.0, 0.0], Some(json!({"title":"Rust Handbook","category":"tech"}))),
        Point::new(5, vec![0.05, 0.95, 0.0, 0.0], Some(json!({"title":"Python Guide","category":"tech"}))),
        Point::new(6, vec![0.0, 0.0, 1.0, 0.0], Some(json!({"title":"World History","category":"history"}))),
        Point::new(7, vec![0.0, 0.0, 0.9, 0.1], Some(json!({"title":"Ancient Rome","category":"history"}))),
        Point::new(8, vec![0.25, 0.25, 0.25, 0.25], Some(json!({"title":"General Knowledge","category":"misc"}))),
    ])
    .expect("test: upsert search corpus");
}

/// Populate a collection with text-rich payloads for BM25 testing.
///
/// BM25 index is populated automatically during `upsert` when payloads
/// contain string fields.
fn setup_text_collection(db: &Database) {
    execute_sql(
        db,
        "CREATE COLLECTION articles (dimension = 4, metric = 'cosine');",
    )
    .expect("test: CREATE articles");

    let vc = db
        .get_vector_collection("articles")
        .expect("test: articles collection must exist");

    vc.upsert(vec![
        Point::new(1, vec![1.0, 0.0, 0.0, 0.0], Some(json!({
            "content": "rust programming language systems",
            "category": "tech"
        }))),
        Point::new(2, vec![0.0, 1.0, 0.0, 0.0], Some(json!({
            "content": "python programming tutorial beginner",
            "category": "tech"
        }))),
        Point::new(3, vec![0.0, 0.0, 1.0, 0.0], Some(json!({
            "content": "football world cup sports results",
            "category": "sports"
        }))),
        Point::new(4, vec![0.5, 0.5, 0.0, 0.0], Some(json!({
            "content": "database systems and rust web framework",
            "category": "tech"
        }))),
    ])
    .expect("test: upsert text corpus");
}

/// Populate a collection with numeric timestamp payloads for temporal testing.
fn setup_temporal_collection(db: &Database) {
    execute_sql(
        db,
        "CREATE COLLECTION events (dimension = 4, metric = 'cosine');",
    )
    .expect("test: CREATE events");

    let vc = db
        .get_vector_collection("events")
        .expect("test: events collection must exist");

    // Use epoch seconds: all items have timestamps firmly in the past.
    let now_approx = 1_743_300_000_i64; // ~2025-03-29 (well in the past for our tests)
    vc.upsert(vec![
        Point::new(1, vec![1.0, 0.0, 0.0, 0.0], Some(json!({"timestamp": now_approx - 100}))),
        Point::new(2, vec![0.0, 1.0, 0.0, 0.0], Some(json!({"timestamp": now_approx - 200}))),
        Point::new(3, vec![0.0, 0.0, 1.0, 0.0], Some(json!({"timestamp": now_approx - 300}))),
    ])
    .expect("test: upsert temporal corpus");
}

// =========================================================================
// similarity() threshold
// =========================================================================

/// GIVEN a docs collection with clustered vectors
/// WHEN `similarity(vector, $v) > 0.9` with `$v=[1,0,0,0]`
/// THEN only vectors very close to the query are returned (cosine > 0.9).
#[test]
fn test_similarity_threshold_filters_by_score() {
    let (_dir, db) = create_test_db();
    setup_similarity_collection(&db);

    let sql = "SELECT * FROM docs WHERE similarity(vector, $v) > 0.9 LIMIT 10";
    let params = vector_param(&[1.0, 0.0, 0.0, 0.0]);
    let results =
        execute_sql_with_params(&db, sql, &params).expect("test: similarity threshold query");

    assert!(
        !results.is_empty(),
        "Should return at least the exact match (id=1)"
    );

    // All returned results must have cosine similarity > 0.9 to [1,0,0,0].
    for r in &results {
        assert!(
            r.score > 0.9,
            "Score {} for id={} must be > 0.9",
            r.score,
            r.point.id
        );
    }

    // The exact match (id=1) must be present.
    let ids: Vec<u64> = results.iter().map(|r| r.point.id).collect();
    assert!(ids.contains(&1), "Exact match (id=1) must be in results");

    // Orthogonal vectors (ids 4-7) must NOT be present.
    for excluded_id in [4, 5, 6, 7] {
        assert!(
            !ids.contains(&excluded_id),
            "Orthogonal vector id={} must not pass threshold 0.9",
            excluded_id
        );
    }
}

/// GIVEN a docs collection
/// WHEN `similarity(vector, $v) > 0.0` (very permissive threshold)
/// THEN returns all non-orthogonal results (essentially all vectors).
#[test]
fn test_similarity_threshold_zero_returns_all() {
    let (_dir, db) = create_test_db();
    setup_similarity_collection(&db);

    let sql = "SELECT * FROM docs WHERE similarity(vector, $v) > 0.0 LIMIT 20";
    let params = vector_param(&[1.0, 0.0, 0.0, 0.0]);
    let results =
        execute_sql_with_params(&db, sql, &params).expect("test: similarity > 0.0 query");

    // With cosine on unit-ish vectors, [1,0,0,0] has sim 0.0 against [0,1,0,0] etc.
    // The exact zero-similarity vectors may or may not pass (depends on > vs >=).
    // At minimum, the science cluster (ids 1,2,3) and the misc outlier (id 8) should pass.
    assert!(
        results.len() >= 3,
        "Permissive threshold (> 0.0) should return at least the 3 science docs, got {}",
        results.len()
    );

    for r in &results {
        assert!(
            r.score > 0.0,
            "All results must have positive score, got {} for id={}",
            r.score,
            r.point.id
        );
    }
}

/// GIVEN a docs collection
/// WHEN `similarity(vector, $v) > 0.99` (very strict threshold)
/// THEN returns only exact or near-exact matches.
#[test]
fn test_similarity_threshold_one_returns_near_exact() {
    let (_dir, db) = create_test_db();
    setup_similarity_collection(&db);

    let sql = "SELECT * FROM docs WHERE similarity(vector, $v) > 0.99 LIMIT 10";
    let params = vector_param(&[1.0, 0.0, 0.0, 0.0]);
    let results =
        execute_sql_with_params(&db, sql, &params).expect("test: similarity > 0.99 query");

    // Only id=1 ([1,0,0,0]) should have cosine > 0.99 against [1,0,0,0].
    // id=2 ([0.95,0.05,0,0]) has cosine ~0.998 so it may or may not pass.
    assert!(
        !results.is_empty(),
        "At least the exact match (id=1) must pass > 0.99"
    );
    assert!(
        results.len() <= 3,
        "Strict threshold should return very few results, got {}",
        results.len()
    );

    for r in &results {
        assert!(
            r.score > 0.99,
            "Score {} for id={} must be > 0.99",
            r.score,
            r.point.id
        );
    }
}

// =========================================================================
// ORDER BY similarity()
// =========================================================================

/// GIVEN a docs collection
/// WHEN `ORDER BY similarity() DESC LIMIT 5`
/// THEN scores are in non-increasing order.
#[test]
fn test_order_by_similarity_desc() {
    let (_dir, db) = create_test_db();
    setup_similarity_collection(&db);

    let sql = "SELECT * FROM docs WHERE vector NEAR $v ORDER BY similarity() DESC LIMIT 5";
    let params = vector_param(&[1.0, 0.0, 0.0, 0.0]);
    let results =
        execute_sql_with_params(&db, sql, &params).expect("test: ORDER BY similarity() DESC");

    assert_eq!(results.len(), 5, "LIMIT 5 should return exactly 5 results");

    // Scores must be in non-increasing (descending) order.
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "Scores must be non-increasing: {} (id={}) should be >= {} (id={})",
            w[0].score,
            w[0].point.id,
            w[1].score,
            w[1].point.id
        );
    }
}

/// GIVEN a docs collection
/// WHEN `ORDER BY similarity() ASC LIMIT 5`
/// THEN scores are in non-decreasing order.
#[test]
fn test_order_by_similarity_asc() {
    let (_dir, db) = create_test_db();
    setup_similarity_collection(&db);

    let sql = "SELECT * FROM docs WHERE vector NEAR $v ORDER BY similarity() ASC LIMIT 5";
    let params = vector_param(&[1.0, 0.0, 0.0, 0.0]);
    let results =
        execute_sql_with_params(&db, sql, &params).expect("test: ORDER BY similarity() ASC");

    assert_eq!(results.len(), 5, "LIMIT 5 should return exactly 5 results");

    // Scores must be in non-decreasing (ascending) order.
    for w in results.windows(2) {
        assert!(
            w[0].score <= w[1].score,
            "Scores must be non-decreasing: {} (id={}) should be <= {} (id={})",
            w[0].score,
            w[0].point.id,
            w[1].score,
            w[1].point.id
        );
    }
}

// =========================================================================
// MATCH text (BM25)
// =========================================================================

/// GIVEN a collection with text content in payloads
/// WHEN `content MATCH 'rust'` (BM25 full-text search)
/// THEN returns results containing "rust" in their content field.
#[test]
fn test_match_bm25_returns_relevant_results() {
    let (_dir, db) = create_test_db();
    setup_text_collection(&db);

    let sql = "SELECT * FROM articles WHERE content MATCH 'rust' LIMIT 5";
    let results = execute_sql(&db, sql).expect("test: BM25 text search");

    // Points 1 and 4 contain "rust" in their content.
    assert!(
        !results.is_empty(),
        "BM25 search for 'rust' should find matching documents"
    );

    let ids: Vec<u64> = results.iter().map(|r| r.point.id).collect();
    assert!(
        ids.contains(&1),
        "id=1 ('rust programming language') should match 'rust', got {:?}",
        ids
    );
}

/// GIVEN a collection with text content
/// WHEN `vector NEAR $v AND content MATCH 'programming'`
/// THEN returns hybrid (vector + BM25 RRF) fused results.
#[test]
fn test_match_hybrid_near_and_text() {
    let (_dir, db) = create_test_db();
    setup_text_collection(&db);

    let sql =
        "SELECT * FROM articles WHERE vector NEAR $v AND content MATCH 'programming' LIMIT 5";
    let params = vector_param(&[1.0, 0.0, 0.0, 0.0]);
    let results =
        execute_sql_with_params(&db, sql, &params).expect("test: hybrid NEAR + BM25 query");

    // "programming" appears in ids 1 and 2; NEAR [1,0,0,0] favors id=1.
    assert!(
        !results.is_empty(),
        "Hybrid search should return at least one result"
    );

    // All returned results should have non-zero scores (fused).
    for r in &results {
        assert!(
            r.score > 0.0,
            "Hybrid result must have positive fused score, got {} for id={}",
            r.score,
            r.point.id
        );
    }
}

// =========================================================================
// MATCH graph patterns
// =========================================================================

/// GIVEN a graph collection with nodes and edges
/// WHEN a MATCH graph query is executed via `Database::execute_query`
/// THEN the database correctly returns an error indicating MATCH queries must
///      go through the collection-level API.
///
/// NOTE: `Database::execute_query` deliberately rejects top-level MATCH queries.
/// Graph MATCH execution requires routing through `GraphCollection::execute_match()`
/// or `Collection::execute_query_with_client()`.
#[test]
fn test_graph_match_via_database_returns_informative_error() {
    let (_dir, db) = create_test_db();

    // Create a graph collection with edges via SQL DDL.
    execute_sql(
        &db,
        "CREATE GRAPH COLLECTION kg (dimension = 4, metric = 'cosine') SCHEMALESS;",
    )
    .expect("test: CREATE GRAPH kg");

    let gc = db
        .get_graph_collection("kg")
        .expect("test: kg graph collection must exist");

    // Add nodes with labels.
    gc.upsert_node_payload(1, &json!({"_labels": ["Person"], "name": "Alice"}))
        .expect("test: node 1");
    gc.upsert_node_payload(2, &json!({"_labels": ["Person"], "name": "Bob"}))
        .expect("test: node 2");

    // Add edge: Alice -> Bob.
    let edge = velesdb_core::GraphEdge::new(1, 1, 2, "KNOWS").expect("test: create edge");
    gc.add_edge(edge).expect("test: add edge");

    // Database-level MATCH query is explicitly unsupported (by design).
    let sql = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b LIMIT 10";
    let err = execute_sql(&db, sql).expect_err("test: database MATCH should error");

    let msg = err.to_string();
    assert!(
        msg.contains("MATCH") || msg.contains("match"),
        "Error should mention MATCH queries, got: {}",
        msg
    );
}

/// GIVEN a graph collection with nodes and edges
/// WHEN a MATCH query is executed through `GraphCollection::execute_match()`
/// THEN matched patterns are returned correctly.
#[test]
fn test_graph_match_basic_traversal_via_collection() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE GRAPH COLLECTION kg (dimension = 4, metric = 'cosine') SCHEMALESS;",
    )
    .expect("test: CREATE GRAPH kg");

    let gc = db
        .get_graph_collection("kg")
        .expect("test: kg graph collection must exist");

    gc.upsert_node_payload(1, &json!({"_labels": ["Person"], "name": "Alice"}))
        .expect("test: node 1");
    gc.upsert_node_payload(2, &json!({"_labels": ["Person"], "name": "Bob"}))
        .expect("test: node 2");
    gc.upsert_node_payload(3, &json!({"_labels": ["Company"], "name": "Acme"}))
        .expect("test: node 3");

    let edge1 = velesdb_core::GraphEdge::new(1, 1, 2, "KNOWS").expect("test: edge 1");
    gc.add_edge(edge1).expect("test: add edge 1");
    let edge2 = velesdb_core::GraphEdge::new(2, 1, 3, "WORKS_AT").expect("test: edge 2");
    gc.add_edge(edge2).expect("test: add edge 2");

    // Execute MATCH via the GraphCollection API.
    let match_clause = velesdb_core::velesql::MatchClause {
        patterns: vec![velesdb_core::velesql::GraphPattern {
            name: None,
            nodes: vec![
                velesdb_core::velesql::NodePattern::new().with_alias("a"),
                velesdb_core::velesql::NodePattern::new().with_alias("b"),
            ],
            relationships: vec![velesdb_core::velesql::RelationshipPattern::new(
                velesdb_core::velesql::Direction::Outgoing,
            )],
        }],
        where_clause: None,
        return_clause: velesdb_core::velesql::ReturnClause {
            items: vec![velesdb_core::velesql::ReturnItem {
                expression: "*".to_string(),
                alias: None,
            }],
            order_by: None,
            limit: Some(100),
        },
    };

    let results = gc
        .execute_match(&match_clause, &HashMap::new())
        .expect("test: graph MATCH execution");

    // Alice (1) has two outgoing edges, so we expect at least 2 traversal results.
    assert!(
        results.len() >= 2,
        "MATCH should return at least 2 results for Alice's outgoing edges, got {}",
        results.len()
    );
}

/// GIVEN a graph collection with labeled edges
/// WHEN MATCH filters by edge label `[:KNOWS]`
/// THEN only KNOWS-typed traversals are returned.
#[test]
fn test_graph_match_with_label_filter() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE GRAPH COLLECTION social (dimension = 4, metric = 'cosine') SCHEMALESS;",
    )
    .expect("test: CREATE GRAPH social");

    let gc = db
        .get_graph_collection("social")
        .expect("test: social graph collection must exist");

    gc.upsert_node_payload(1, &json!({"_labels": ["Person"], "name": "Alice"}))
        .expect("test: node 1");
    gc.upsert_node_payload(2, &json!({"_labels": ["Person"], "name": "Bob"}))
        .expect("test: node 2");
    gc.upsert_node_payload(3, &json!({"_labels": ["Company"], "name": "Acme"}))
        .expect("test: node 3");

    let edge1 = velesdb_core::GraphEdge::new(1, 1, 2, "KNOWS").expect("test: edge KNOWS");
    gc.add_edge(edge1).expect("test: add KNOWS edge");
    let edge2 = velesdb_core::GraphEdge::new(2, 1, 3, "WORKS_AT").expect("test: edge WORKS_AT");
    gc.add_edge(edge2).expect("test: add WORKS_AT edge");

    // MATCH with specific edge label filter.
    let match_clause = velesdb_core::velesql::MatchClause {
        patterns: vec![velesdb_core::velesql::GraphPattern {
            name: None,
            nodes: vec![
                velesdb_core::velesql::NodePattern::new().with_alias("p"),
                velesdb_core::velesql::NodePattern::new().with_alias("q"),
            ],
            relationships: vec![{
                let mut rel = velesdb_core::velesql::RelationshipPattern::new(
                    velesdb_core::velesql::Direction::Outgoing,
                );
                rel.types = vec!["KNOWS".to_string()];
                rel
            }],
        }],
        where_clause: None,
        return_clause: velesdb_core::velesql::ReturnClause {
            items: vec![velesdb_core::velesql::ReturnItem {
                expression: "*".to_string(),
                alias: None,
            }],
            order_by: None,
            limit: Some(100),
        },
    };

    let results = gc
        .execute_match(&match_clause, &HashMap::new())
        .expect("test: MATCH [:KNOWS] execution");

    // Only Alice->Bob via KNOWS should match.
    assert!(
        !results.is_empty(),
        "MATCH [:KNOWS] should find at least one traversal"
    );

    // All results should be reachable via KNOWS edges only.
    // From Alice (1), KNOWS leads only to Bob (2).
    for r in &results {
        let target = r.node_id;
        assert_eq!(
            target, 2,
            "KNOWS edge from Alice should reach only Bob (id=2), got id={}",
            target
        );
    }
}

// =========================================================================
// Temporal expressions
// =========================================================================

/// GIVEN items with numeric timestamps in their payload
/// WHEN `WHERE timestamp > NOW() - INTERVAL '999999 days'` (very long ago)
/// THEN all items are returned because their timestamps are recent.
#[test]
fn test_temporal_now_minus_interval_returns_all() {
    let (_dir, db) = create_test_db();
    setup_temporal_collection(&db);

    // 999999 days ago is well before any of our epoch timestamps.
    let sql =
        "SELECT * FROM events WHERE timestamp > NOW() - INTERVAL '999999 days' LIMIT 10";
    let results = execute_sql(&db, sql).expect("test: temporal far-past query");

    assert_eq!(
        results.len(),
        3,
        "All 3 events should have timestamps > (now - 999999 days), got {}",
        results.len()
    );
}

/// GIVEN items with numeric timestamps in their payload
/// WHEN `WHERE timestamp > NOW() + INTERVAL '1 day'` (future threshold)
/// THEN returns 0 items because all timestamps are in the past.
#[test]
fn test_temporal_future_threshold_returns_none() {
    let (_dir, db) = create_test_db();
    setup_temporal_collection(&db);

    // NOW() + 1 day is in the future; all our timestamps are in the past.
    let sql =
        "SELECT * FROM events WHERE timestamp > NOW() + INTERVAL '1 day' LIMIT 10";
    let results = execute_sql(&db, sql).expect("test: temporal future query");

    assert_eq!(
        results.len(),
        0,
        "No events should have timestamps > (now + 1 day), got {}",
        results.len()
    );
}

// =========================================================================
// Negative / edge cases
// =========================================================================

/// GIVEN no collection named "nonexistent"
/// WHEN a similarity query is run against it
/// THEN an error mentioning the missing collection is returned.
#[test]
fn test_similarity_on_nonexistent_collection_fails() {
    let (_dir, db) = create_test_db();

    let sql = "SELECT * FROM nonexistent WHERE similarity(vector, $v) > 0.5 LIMIT 5";
    let params = vector_param(&[1.0, 0.0, 0.0, 0.0]);
    let err = execute_sql_with_params(&db, sql, &params)
        .expect_err("test: query on missing collection should error");

    let msg = err.to_string();
    assert!(
        msg.contains("nonexistent") || msg.contains("not found") || msg.contains("NotFound"),
        "Error should mention the missing collection, got: {}",
        msg
    );
}

/// GIVEN an empty collection (no points)
/// WHEN `content MATCH 'anything'` is executed
/// THEN returns 0 results without error.
#[test]
fn test_match_text_on_empty_collection() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE COLLECTION empty_col (dimension = 4, metric = 'cosine');",
    )
    .expect("test: CREATE empty_col");

    let sql = "SELECT * FROM empty_col WHERE content MATCH 'anything' LIMIT 5";
    let results = execute_sql(&db, sql).expect("test: BM25 on empty collection");

    assert_eq!(
        results.len(),
        0,
        "MATCH on empty collection should return 0 results"
    );
}
