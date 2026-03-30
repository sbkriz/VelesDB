//! Unit tests for SELECT EDGES and INSERT NODE parsing (VelesQL v3.5 Phase 5).

use crate::velesql::{CompareOp, Condition, DmlStatement, Parser, Value};

// ============================================================================
// A. SELECT EDGES — parse nominal
// ============================================================================

#[test]
fn test_parse_select_edges_basic() {
    let query = Parser::parse("SELECT EDGES FROM kg LIMIT 100").expect("should parse");
    assert!(query.is_dml_query());
    assert!(query.is_select_edges_query());

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::SelectEdges(stmt) = dml else {
        panic!("Expected SelectEdges variant, got {dml:?}");
    };

    assert_eq!(stmt.collection, "kg");
    assert!(stmt.where_clause.is_none());
    assert_eq!(stmt.limit, Some(100));
}

#[test]
fn test_parse_select_edges_where_source() {
    let query = Parser::parse("SELECT EDGES FROM kg WHERE source = 1").expect("should parse");

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::SelectEdges(stmt) = dml else {
        panic!("Expected SelectEdges variant");
    };

    assert_eq!(stmt.collection, "kg");
    let where_clause = stmt.where_clause.expect("Expected WHERE clause");
    let Condition::Comparison(cmp) = where_clause else {
        panic!("Expected Comparison condition, got {where_clause:?}");
    };
    assert_eq!(cmp.column, "source");
    assert_eq!(cmp.operator, CompareOp::Eq);
    assert_eq!(cmp.value, Value::Integer(1));
}

#[test]
fn test_parse_select_edges_where_target() {
    let query = Parser::parse("SELECT EDGES FROM kg WHERE target = 2").expect("should parse");

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::SelectEdges(stmt) = dml else {
        panic!("Expected SelectEdges variant");
    };

    let where_clause = stmt.where_clause.expect("Expected WHERE clause");
    let Condition::Comparison(cmp) = where_clause else {
        panic!("Expected Comparison condition");
    };
    assert_eq!(cmp.column, "target");
    assert_eq!(cmp.value, Value::Integer(2));
}

#[test]
fn test_parse_select_edges_where_label() {
    let query = Parser::parse("SELECT EDGES FROM kg WHERE label = 'KNOWS'").expect("should parse");

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::SelectEdges(stmt) = dml else {
        panic!("Expected SelectEdges variant");
    };

    let where_clause = stmt.where_clause.expect("Expected WHERE clause");
    let Condition::Comparison(cmp) = where_clause else {
        panic!("Expected Comparison condition");
    };
    assert_eq!(cmp.column, "label");
    assert_eq!(cmp.value, Value::String("KNOWS".to_string()));
}

#[test]
fn test_parse_select_edges_with_limit() {
    let query = Parser::parse("SELECT EDGES FROM kg WHERE source = 1 LIMIT 5")
        .expect("should parse with LIMIT");

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::SelectEdges(stmt) = dml else {
        panic!("Expected SelectEdges variant");
    };

    assert_eq!(stmt.collection, "kg");
    assert!(stmt.where_clause.is_some());
    assert_eq!(stmt.limit, Some(5));
}

#[test]
fn test_parse_select_edges_no_where_no_limit() {
    let query = Parser::parse("SELECT EDGES FROM kg").expect("should parse without WHERE/LIMIT");

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::SelectEdges(stmt) = dml else {
        panic!("Expected SelectEdges variant");
    };

    assert_eq!(stmt.collection, "kg");
    assert!(stmt.where_clause.is_none());
    assert!(stmt.limit.is_none());
}

#[test]
fn test_parse_select_edges_combined_source_and_label() {
    let query = Parser::parse("SELECT EDGES FROM kg WHERE source = 1 AND label = 'KNOWS'")
        .expect("should parse combined WHERE");

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::SelectEdges(stmt) = dml else {
        panic!("Expected SelectEdges variant");
    };

    let where_clause = stmt.where_clause.expect("Expected WHERE clause");
    let Condition::And(left, right) = where_clause else {
        panic!("Expected And condition, got {where_clause:?}");
    };

    let Condition::Comparison(left_cmp) = *left else {
        panic!("Expected left Comparison");
    };
    assert_eq!(left_cmp.column, "source");

    let Condition::Comparison(right_cmp) = *right else {
        panic!("Expected right Comparison");
    };
    assert_eq!(right_cmp.column, "label");
    assert_eq!(right_cmp.value, Value::String("KNOWS".to_string()));
}

// ============================================================================
// B. INSERT NODE — parse nominal
// ============================================================================

#[test]
fn test_parse_insert_node_basic() {
    let query = Parser::parse(
        "INSERT NODE INTO kg (id = 42, payload = '{\"name\": \"Alice\", \"_labels\": [\"Person\"]}')",
    )
    .expect("should parse INSERT NODE");

    assert!(query.is_dml_query());
    assert!(query.is_insert_node_query());

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::InsertNode(stmt) = dml else {
        panic!("Expected InsertNode variant, got {dml:?}");
    };

    assert_eq!(stmt.collection, "kg");
    assert_eq!(stmt.node_id, 42);
    assert_eq!(stmt.payload["name"], "Alice");
}

#[test]
fn test_parse_insert_node_without_explicit_payload() {
    let query = Parser::parse("INSERT NODE INTO kg (id = 10, name = 'Bob', age = 30)")
        .expect("should parse INSERT NODE with inline fields");

    let dml = query.dml.expect("Expected DML");
    let DmlStatement::InsertNode(stmt) = dml else {
        panic!("Expected InsertNode variant");
    };

    assert_eq!(stmt.collection, "kg");
    assert_eq!(stmt.node_id, 10);
    assert_eq!(stmt.payload["name"], "Bob");
    assert_eq!(stmt.payload["age"], 30);
}

// ============================================================================
// C. Negative cases
// ============================================================================

#[test]
fn test_select_edges_without_from_fails() {
    let result = Parser::parse("SELECT EDGES");
    assert!(result.is_err(), "SELECT EDGES without FROM should fail");
}

#[test]
fn test_insert_node_without_into_fails() {
    let result = Parser::parse("INSERT NODE kg (id = 1)");
    assert!(result.is_err(), "INSERT NODE without INTO should fail");
}

// ============================================================================
// D. Query type flags
// ============================================================================

#[test]
fn test_select_edges_query_type_flags() {
    let query = Parser::parse("SELECT EDGES FROM kg").expect("should parse");
    assert!(query.is_dml_query(), "SELECT EDGES is a DML query");
    assert!(query.is_select_edges_query());
    assert!(!query.is_select_query(), "Not a regular SELECT");
    assert!(!query.is_ddl_query());
    assert!(!query.is_match_query());
    assert!(!query.is_introspection_query());
}

#[test]
fn test_insert_node_query_type_flags() {
    let query =
        Parser::parse("INSERT NODE INTO kg (id = 1, name = 'Alice')").expect("should parse");
    assert!(query.is_dml_query());
    assert!(query.is_insert_node_query());
    assert!(!query.is_select_query());
    assert!(!query.is_ddl_query());
}

// ============================================================================
// E. No conflict with regular SELECT
// ============================================================================

#[test]
fn test_regular_select_still_works() {
    let query = Parser::parse("SELECT * FROM docs LIMIT 10").expect("regular SELECT should parse");
    assert!(query.is_select_query(), "Regular SELECT should work");
    assert!(!query.is_select_edges_query());
}

#[test]
fn test_select_edges_with_semicolon() {
    let query = Parser::parse("SELECT EDGES FROM kg;").expect("should parse with semicolon");
    assert!(query.is_select_edges_query());
}
