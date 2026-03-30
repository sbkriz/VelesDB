//! Unit tests for VelesQL introspection statement parsing (SHOW/DESCRIBE/EXPLAIN).
//!
//! Covers nominal parsing, edge cases, and negative cases for all three
//! introspection statement types.

use crate::velesql::ast::IntrospectionStatement;
use crate::velesql::Parser;

// ============================================================================
// SHOW COLLECTIONS — nominal cases
// ============================================================================

#[test]
fn test_parse_show_collections() {
    let query = Parser::parse("SHOW COLLECTIONS").expect("SHOW COLLECTIONS should parse");
    assert!(query.is_introspection_query());
    assert!(!query.is_select_query());
    assert!(!query.is_ddl_query());

    let intro = query
        .introspection
        .expect("Expected introspection statement");
    assert_eq!(intro, IntrospectionStatement::ShowCollections);
}

#[test]
fn test_parse_show_collections_with_semicolon() {
    let query = Parser::parse("SHOW COLLECTIONS;").expect("trailing semicolon should parse");

    let intro = query
        .introspection
        .expect("Expected introspection statement");
    assert_eq!(intro, IntrospectionStatement::ShowCollections);
}

#[test]
fn test_parse_show_collections_case_insensitive() {
    let query = Parser::parse("show collections").expect("lowercase should parse");
    assert!(query.is_introspection_query());

    let query2 = Parser::parse("Show Collections").expect("mixed case should parse");
    assert!(query2.is_introspection_query());
}

// ============================================================================
// DESCRIBE COLLECTION — nominal cases
// ============================================================================

#[test]
fn test_parse_describe_collection() {
    let query =
        Parser::parse("DESCRIBE COLLECTION docs").expect("DESCRIBE COLLECTION should parse");
    assert!(query.is_introspection_query());

    let intro = query
        .introspection
        .expect("Expected introspection statement");
    match intro {
        IntrospectionStatement::DescribeCollection(desc) => {
            assert_eq!(desc.name, "docs");
        }
        other => panic!("Expected DescribeCollection, got {other:?}"),
    }
}

#[test]
fn test_parse_describe_without_collection_keyword() {
    let query = Parser::parse("DESCRIBE docs").expect("DESCRIBE without COLLECTION should parse");
    assert!(query.is_introspection_query());

    let intro = query
        .introspection
        .expect("Expected introspection statement");
    match intro {
        IntrospectionStatement::DescribeCollection(desc) => {
            assert_eq!(desc.name, "docs");
        }
        other => panic!("Expected DescribeCollection, got {other:?}"),
    }
}

#[test]
fn test_parse_describe_collection_with_semicolon() {
    let query =
        Parser::parse("DESCRIBE COLLECTION docs;").expect("trailing semicolon should parse");
    assert!(query.is_introspection_query());

    let intro = query
        .introspection
        .expect("Expected introspection statement");
    match intro {
        IntrospectionStatement::DescribeCollection(desc) => {
            assert_eq!(desc.name, "docs");
        }
        other => panic!("Expected DescribeCollection, got {other:?}"),
    }
}

#[test]
fn test_parse_describe_case_insensitive() {
    let query = Parser::parse("describe collection my_coll").expect("lowercase should parse");
    let intro = query
        .introspection
        .expect("Expected introspection statement");
    match intro {
        IntrospectionStatement::DescribeCollection(desc) => {
            assert_eq!(desc.name, "my_coll");
        }
        other => panic!("Expected DescribeCollection, got {other:?}"),
    }
}

// ============================================================================
// EXPLAIN — nominal cases
// ============================================================================

#[test]
fn test_parse_explain_select() {
    let query =
        Parser::parse("EXPLAIN SELECT * FROM docs LIMIT 10").expect("EXPLAIN SELECT should parse");
    assert!(query.is_introspection_query());

    let intro = query
        .introspection
        .expect("Expected introspection statement");
    match intro {
        IntrospectionStatement::Explain(inner) => {
            assert!(inner.is_select_query());
            assert_eq!(inner.select.from, "docs");
            assert_eq!(inner.select.limit, Some(10));
        }
        other => panic!("Expected Explain, got {other:?}"),
    }
}

#[test]
fn test_parse_explain_with_semicolon() {
    let query = Parser::parse("EXPLAIN SELECT * FROM docs LIMIT 5;")
        .expect("trailing semicolon should parse");
    assert!(query.is_introspection_query());
}

#[test]
fn test_parse_explain_case_insensitive() {
    let query = Parser::parse("explain select * from docs limit 5")
        .expect("lowercase EXPLAIN should parse");
    assert!(query.is_introspection_query());
}

// ============================================================================
// Negative cases — parse failures
// ============================================================================

#[test]
fn test_parse_show_reject_unknown() {
    // SHOW TABLES is not valid VelesQL.
    let result = Parser::parse("SHOW TABLES");
    assert!(result.is_err(), "SHOW TABLES should fail to parse");
}

#[test]
fn test_parse_show_alone_fails() {
    // SHOW alone is not a valid statement.
    let result = Parser::parse("SHOW");
    assert!(result.is_err(), "SHOW alone should fail to parse");
}

#[test]
fn test_parse_describe_missing_name() {
    // DESCRIBE COLLECTION without a name is not valid.
    let result = Parser::parse("DESCRIBE COLLECTION");
    assert!(
        result.is_err(),
        "DESCRIBE COLLECTION without name should fail"
    );
}

#[test]
fn test_parse_explain_without_query() {
    // EXPLAIN alone is not valid.
    let result = Parser::parse("EXPLAIN");
    assert!(result.is_err(), "EXPLAIN alone should fail to parse");
}

// ============================================================================
// Query type predicates
// ============================================================================

#[test]
fn test_introspection_query_type_flags() {
    let query = Parser::parse("SHOW COLLECTIONS").expect("should parse");
    assert!(query.is_introspection_query());
    assert!(!query.is_select_query());
    assert!(!query.is_match_query());
    assert!(!query.is_dml_query());
    assert!(!query.is_train());
    assert!(!query.is_ddl_query());
}

#[test]
fn test_describe_query_type_flags() {
    let query = Parser::parse("DESCRIBE docs").expect("should parse");
    assert!(query.is_introspection_query());
    assert!(!query.is_select_query());
}

#[test]
fn test_explain_query_type_flags() {
    let query = Parser::parse("EXPLAIN SELECT * FROM docs LIMIT 1").expect("should parse");
    assert!(query.is_introspection_query());
    assert!(!query.is_select_query());
}

// ============================================================================
// LET clause rejection for introspection (at validation time)
// ============================================================================

#[test]
fn test_let_with_introspection_parses_but_validation_rejects() {
    use crate::velesql::validation::QueryValidator;

    // Grammar allows LET before any statement, so parsing succeeds.
    let query = Parser::parse("LET x = 1 SHOW COLLECTIONS").expect("grammar allows LET + SHOW");
    assert!(query.is_introspection_query());
    assert!(!query.let_bindings.is_empty());

    // Validation rejects LET with introspection statements.
    let result = QueryValidator::validate(&query);
    assert!(
        result.is_err(),
        "Validator should reject LET + introspection"
    );
}

// ============================================================================
// Additional coverage: complex inner queries, quoted identifiers
// ============================================================================

#[test]
fn test_explain_with_vector_near_in_inner_query() {
    let sql = "EXPLAIN SELECT * FROM docs WHERE vector NEAR $v AND category = 'tech' LIMIT 10";
    let query = Parser::parse(sql).expect("EXPLAIN with NEAR should parse");
    assert!(query.is_introspection_query());

    let IntrospectionStatement::Explain(inner) = query
        .introspection
        .as_ref()
        .expect("introspection should be present")
    else {
        panic!("Expected Explain variant");
    };
    assert!(inner.select.where_clause.is_some(), "inner WHERE present");
    assert_eq!(inner.select.limit, Some(10));
}

#[test]
fn test_describe_with_quoted_identifier() {
    let sql = "DESCRIBE COLLECTION `select`";
    let query = Parser::parse(sql).expect("DESCRIBE with quoted keyword should parse");

    let IntrospectionStatement::DescribeCollection(desc) = query
        .introspection
        .as_ref()
        .expect("introspection should be present")
    else {
        panic!("Expected DescribeCollection variant");
    };
    assert_eq!(desc.name, "select", "quoted keyword should be unquoted in AST");
}

#[test]
fn test_explain_match_is_rejected() {
    let result = Parser::parse("EXPLAIN MATCH (a)-[:REL]->(b) RETURN a");
    assert!(
        result.is_err(),
        "EXPLAIN MATCH should fail (only compound_query supported)"
    );
}
