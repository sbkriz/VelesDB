//! Shared helpers for all `VelesQL` BDD test modules.
//!
//! Each helper is small and reusable across multiple BDD scenarios.
//! Module-specific setup functions (e.g. `setup_products_collection`) live in
//! their respective module files, not here.

use std::collections::{HashMap, HashSet};

use tempfile::TempDir;
use velesdb_core::{velesql::Parser, Database, SearchResult};

/// Execute a `VelesQL` SQL string through the full pipeline: parse -> validate -> execute.
pub fn execute_sql(db: &Database, sql: &str) -> velesdb_core::Result<Vec<SearchResult>> {
    let query = Parser::parse(sql).map_err(|e| velesdb_core::Error::Query(e.to_string()))?;
    db.execute_query(&query, &HashMap::new())
}

/// Execute a `VelesQL` SQL string with bind parameters (e.g. `$v` for NEAR).
pub fn execute_sql_with_params(
    db: &Database,
    sql: &str,
    params: &HashMap<String, serde_json::Value>,
) -> velesdb_core::Result<Vec<SearchResult>> {
    let query = Parser::parse(sql).map_err(|e| velesdb_core::Error::Query(e.to_string()))?;
    db.execute_query(&query, params)
}

/// Create a fresh database in a temp directory.
pub fn create_test_db() -> (TempDir, Database) {
    let dir = TempDir::new().expect("test: create temp dir");
    let db = Database::open(dir.path()).expect("test: open database");
    (dir, db)
}

/// Collect result IDs into a `HashSet` for order-independent comparison.
pub fn result_ids(results: &[SearchResult]) -> HashSet<u64> {
    results.iter().map(|r| r.point.id).collect()
}

/// Build a param map with a single vector parameter named `$v`.
pub fn vector_param(v: &[f32]) -> HashMap<String, serde_json::Value> {
    let mut params = HashMap::new();
    params.insert("v".to_string(), serde_json::json!(v));
    params
}

/// Floating-point equality within epsilon.
pub fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

/// Extract a numeric payload field from a `SearchResult`.
pub fn payload_f64(result: &SearchResult, field: &str) -> Option<f64> {
    result
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get(field))
        .and_then(serde_json::Value::as_f64)
}

/// Extract a string payload field from a `SearchResult`.
pub fn payload_str<'a>(result: &'a SearchResult, field: &str) -> Option<&'a str> {
    result
        .point
        .payload
        .as_ref()
        .and_then(|p| p.get(field))
        .and_then(serde_json::Value::as_str)
}
