//! Tests for CREATE INDEX / DROP INDEX parsing (VelesQL v3.5).
//!
//! Covers nominal parsing, case insensitivity, semicolons, edge cases
//! (missing fields, missing collections), query type predicates, and
//! non-interference with CREATE COLLECTION / DROP COLLECTION.

use crate::velesql::ast::DdlStatement;
use crate::velesql::Parser;

// ============================================================================
// CREATE INDEX — nominal cases
// ============================================================================

#[test]
fn test_parse_create_index() {
    let query =
        Parser::parse("CREATE INDEX ON docs (category)").expect("CREATE INDEX should parse");

    let ddl = query.ddl.expect("Expected DDL statement");
    let DdlStatement::CreateIndex(stmt) = ddl else {
        panic!("Expected CreateIndex variant, got {ddl:?}");
    };

    assert_eq!(stmt.collection, "docs");
    assert_eq!(stmt.field, "category");
}

#[test]
fn test_parse_create_index_with_semicolon() {
    let query =
        Parser::parse("CREATE INDEX ON docs (category);").expect("trailing semicolon should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::CreateIndex(stmt) = ddl else {
        panic!("Expected CreateIndex variant");
    };
    assert_eq!(stmt.collection, "docs");
    assert_eq!(stmt.field, "category");
}

#[test]
fn test_parse_create_index_case_insensitive() {
    let query = Parser::parse("create index on Docs (Category)")
        .expect("case-insensitive keywords should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::CreateIndex(stmt) = ddl else {
        panic!("Expected CreateIndex variant");
    };
    assert_eq!(stmt.collection, "Docs");
    assert_eq!(stmt.field, "Category");
}

// ============================================================================
// DROP INDEX — nominal cases
// ============================================================================

#[test]
fn test_parse_drop_index() {
    let query = Parser::parse("DROP INDEX ON docs (category)").expect("DROP INDEX should parse");

    let ddl = query.ddl.expect("Expected DDL statement");
    let DdlStatement::DropIndex(stmt) = ddl else {
        panic!("Expected DropIndex variant, got {ddl:?}");
    };

    assert_eq!(stmt.collection, "docs");
    assert_eq!(stmt.field, "category");
}

#[test]
fn test_parse_drop_index_with_semicolon() {
    let query =
        Parser::parse("DROP INDEX ON docs (category);").expect("trailing semicolon should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::DropIndex(stmt) = ddl else {
        panic!("Expected DropIndex variant");
    };
    assert_eq!(stmt.collection, "docs");
    assert_eq!(stmt.field, "category");
}

// ============================================================================
// Edge cases — missing fields should fail
// ============================================================================

#[test]
fn test_parse_create_index_missing_field_fails() {
    let result = Parser::parse("CREATE INDEX ON docs");
    assert!(
        result.is_err(),
        "CREATE INDEX without (field) should fail at grammar level"
    );
}

#[test]
fn test_parse_create_index_missing_collection_fails() {
    let result = Parser::parse("CREATE INDEX ON (field)");
    assert!(
        result.is_err(),
        "CREATE INDEX ON (field) without collection should fail"
    );
}

#[test]
fn test_parse_drop_index_missing_field_fails() {
    let result = Parser::parse("DROP INDEX ON docs");
    assert!(
        result.is_err(),
        "DROP INDEX without (field) should fail at grammar level"
    );
}

// ============================================================================
// Query type predicates
// ============================================================================

#[test]
fn test_create_index_query_type_flags() {
    let query =
        Parser::parse("CREATE INDEX ON docs (category)").expect("CREATE INDEX should parse");

    assert!(query.is_ddl_query(), "CREATE INDEX should be DDL");
    assert!(!query.is_select_query());
    assert!(!query.is_dml_query());
    assert!(!query.is_match_query());
    assert!(!query.is_train());
    assert!(!query.is_introspection_query());
}

#[test]
fn test_drop_index_query_type_flags() {
    let query = Parser::parse("DROP INDEX ON docs (category)").expect("DROP INDEX should parse");

    assert!(query.is_ddl_query(), "DROP INDEX should be DDL");
    assert!(!query.is_select_query());
    assert!(!query.is_dml_query());
}

// ============================================================================
// Non-interference with CREATE COLLECTION / DROP COLLECTION
// ============================================================================

#[test]
fn test_create_index_does_not_conflict_with_create_collection() {
    // CREATE COLLECTION must still parse correctly after adding CREATE INDEX
    let query = Parser::parse("CREATE COLLECTION docs (dimension = 4, metric = 'cosine')")
        .expect("CREATE COLLECTION should still parse");

    assert!(query.is_ddl_query());
    let ddl = query.ddl.expect("Expected DDL");
    assert!(
        matches!(ddl, DdlStatement::CreateCollection(_)),
        "Should be CreateCollection, got {ddl:?}"
    );
}

#[test]
fn test_drop_index_does_not_conflict_with_drop_collection() {
    // DROP COLLECTION must still parse correctly after adding DROP INDEX
    let query = Parser::parse("DROP COLLECTION docs").expect("DROP COLLECTION should still parse");

    assert!(query.is_ddl_query());
    let ddl = query.ddl.expect("Expected DDL");
    assert!(
        matches!(ddl, DdlStatement::DropCollection(_)),
        "Should be DropCollection, got {ddl:?}"
    );
}

#[test]
fn test_create_collection_if_exists_still_works() {
    let query = Parser::parse("DROP COLLECTION IF EXISTS docs")
        .expect("DROP COLLECTION IF EXISTS should still parse");

    assert!(query.is_ddl_query());
    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::DropCollection(drop) = ddl else {
        panic!("Expected DropCollection variant");
    };
    assert!(drop.if_exists);
    assert_eq!(drop.name, "docs");
}

// ============================================================================
// Quoted identifiers
// ============================================================================

#[test]
fn test_create_index_quoted_collection() {
    let query = Parser::parse("CREATE INDEX ON `my-docs` (category)")
        .expect("quoted collection should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::CreateIndex(stmt) = ddl else {
        panic!("Expected CreateIndex variant");
    };
    assert_eq!(stmt.collection, "my-docs");
    assert_eq!(stmt.field, "category");
}

#[test]
fn test_create_index_quoted_field() {
    let query =
        Parser::parse("CREATE INDEX ON docs (`field-name`)").expect("quoted field should parse");

    let ddl = query.ddl.expect("Expected DDL");
    let DdlStatement::CreateIndex(stmt) = ddl else {
        panic!("Expected CreateIndex variant");
    };
    assert_eq!(stmt.collection, "docs");
    assert_eq!(stmt.field, "field-name");
}
