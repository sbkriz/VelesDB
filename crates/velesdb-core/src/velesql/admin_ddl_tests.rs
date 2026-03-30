//! Tests for ANALYZE, TRUNCATE, and ALTER COLLECTION parsing (VelesQL v3.5).
//!
//! Covers nominal parsing, case insensitivity, semicolons, optional COLLECTION
//! keyword, edge cases (missing names, unknown options), and query type flags.

use crate::velesql::ast::DdlStatement;
use crate::velesql::Parser;

// ============================================================================
// ANALYZE — nominal cases
// ============================================================================

#[test]
fn test_parse_analyze() {
    let query = Parser::parse("ANALYZE docs").expect("ANALYZE should parse");

    let ddl = query.ddl.expect("Expected DDL statement");
    let DdlStatement::Analyze(stmt) = ddl else {
        panic!("Expected Analyze variant, got {ddl:?}");
    };

    assert_eq!(stmt.collection, "docs");
}

#[test]
fn test_parse_analyze_with_collection_keyword() {
    let query = Parser::parse("ANALYZE COLLECTION docs").expect("ANALYZE COLLECTION should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::Analyze(stmt) = ddl else {
        panic!("Expected Analyze variant");
    };
    assert_eq!(stmt.collection, "docs");
}

#[test]
fn test_parse_analyze_with_semicolon() {
    let query = Parser::parse("ANALYZE docs;").expect("trailing semicolon should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::Analyze(stmt) = ddl else {
        panic!("Expected Analyze variant");
    };
    assert_eq!(stmt.collection, "docs");
}

#[test]
fn test_parse_analyze_case_insensitive() {
    let query = Parser::parse("analyze collection Docs").expect("lowercase should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::Analyze(stmt) = ddl else {
        panic!("Expected Analyze variant");
    };
    assert_eq!(stmt.collection, "Docs");
}

// ============================================================================
// TRUNCATE — nominal cases
// ============================================================================

#[test]
fn test_parse_truncate() {
    let query = Parser::parse("TRUNCATE docs").expect("TRUNCATE should parse");

    let ddl = query.ddl.expect("Expected DDL statement");
    let DdlStatement::Truncate(stmt) = ddl else {
        panic!("Expected Truncate variant, got {ddl:?}");
    };

    assert_eq!(stmt.collection, "docs");
}

#[test]
fn test_parse_truncate_with_collection_keyword() {
    let query =
        Parser::parse("TRUNCATE COLLECTION docs").expect("TRUNCATE COLLECTION should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::Truncate(stmt) = ddl else {
        panic!("Expected Truncate variant");
    };
    assert_eq!(stmt.collection, "docs");
}

#[test]
fn test_parse_truncate_with_semicolon() {
    let query = Parser::parse("TRUNCATE docs;").expect("trailing semicolon should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::Truncate(stmt) = ddl else {
        panic!("Expected Truncate variant");
    };
    assert_eq!(stmt.collection, "docs");
}

#[test]
fn test_parse_truncate_case_insensitive() {
    let query = Parser::parse("truncate collection My_Coll").expect("lowercase should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::Truncate(stmt) = ddl else {
        panic!("Expected Truncate variant");
    };
    assert_eq!(stmt.collection, "My_Coll");
}

// ============================================================================
// ALTER COLLECTION — nominal cases
// ============================================================================

#[test]
fn test_parse_alter_collection_set() {
    let query = Parser::parse("ALTER COLLECTION docs SET (auto_reindex = true)")
        .expect("ALTER COLLECTION SET should parse");

    let ddl = query.ddl.expect("Expected DDL statement");
    let DdlStatement::AlterCollection(stmt) = ddl else {
        panic!("Expected AlterCollection variant, got {ddl:?}");
    };

    assert_eq!(stmt.collection, "docs");
    assert_eq!(stmt.options.len(), 1);
    assert_eq!(stmt.options[0].0, "auto_reindex");
    assert_eq!(stmt.options[0].1, "true");
}

#[test]
fn test_parse_alter_collection_with_semicolon() {
    let query = Parser::parse("ALTER COLLECTION docs SET (auto_reindex = false);")
        .expect("trailing semicolon should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::AlterCollection(stmt) = ddl else {
        panic!("Expected AlterCollection variant");
    };
    assert_eq!(stmt.collection, "docs");
    assert_eq!(stmt.options[0].1, "false");
}

#[test]
fn test_parse_alter_collection_case_insensitive() {
    let query = Parser::parse("alter collection Docs set (auto_reindex = true)")
        .expect("lowercase should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::AlterCollection(stmt) = ddl else {
        panic!("Expected AlterCollection variant");
    };
    assert_eq!(stmt.collection, "Docs");
}

// ============================================================================
// Negative cases — parse failures
// ============================================================================

#[test]
fn test_parse_analyze_missing_name_fails() {
    let result = Parser::parse("ANALYZE");
    assert!(result.is_err(), "ANALYZE without name should fail");
}

#[test]
fn test_parse_truncate_missing_name_fails() {
    let result = Parser::parse("TRUNCATE");
    assert!(result.is_err(), "TRUNCATE without name should fail");
}

#[test]
fn test_parse_alter_missing_set_fails() {
    let result = Parser::parse("ALTER COLLECTION docs");
    assert!(result.is_err(), "ALTER COLLECTION without SET should fail");
}

#[test]
fn test_parse_alter_empty_options_fails() {
    let result = Parser::parse("ALTER COLLECTION docs SET ()");
    assert!(
        result.is_err(),
        "ALTER COLLECTION SET () with empty options should fail"
    );
}

// ============================================================================
// Query type predicates
// ============================================================================

#[test]
fn test_analyze_query_type_flags() {
    let query = Parser::parse("ANALYZE docs").expect("should parse");
    assert!(query.is_ddl_query(), "ANALYZE should be DDL");
    assert!(!query.is_select_query());
    assert!(!query.is_dml_query());
    assert!(!query.is_match_query());
    assert!(!query.is_train());
    assert!(!query.is_introspection_query());
}

#[test]
fn test_truncate_query_type_flags() {
    let query = Parser::parse("TRUNCATE docs").expect("should parse");
    assert!(query.is_ddl_query(), "TRUNCATE should be DDL");
    assert!(!query.is_select_query());
    assert!(!query.is_dml_query());
}

#[test]
fn test_alter_collection_query_type_flags() {
    let query =
        Parser::parse("ALTER COLLECTION docs SET (auto_reindex = true)").expect("should parse");
    assert!(query.is_ddl_query(), "ALTER COLLECTION should be DDL");
    assert!(!query.is_select_query());
    assert!(!query.is_dml_query());
}
