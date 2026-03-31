//! Tests for FLUSH statement parsing (VelesQL v3.6).
//!
//! Covers all four syntax variants, case insensitivity, trailing semicolons,
//! and query type flag predicates.

use crate::velesql::ast::AdminStatement;
use crate::velesql::Parser;

// ============================================================================
// Nominal — bare FLUSH (all collections, fast)
// ============================================================================

#[test]
fn test_parse_flush_bare() {
    let query = Parser::parse("FLUSH").expect("FLUSH should parse");

    let admin = query.admin.expect("Expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(!stmt.full, "bare FLUSH should not be full");
    assert!(stmt.collection.is_none(), "bare FLUSH flushes all");
}

#[test]
fn test_parse_flush_with_semicolon() {
    let query = Parser::parse("FLUSH;").expect("FLUSH; should parse");

    let admin = query.admin.expect("Expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(!stmt.full);
    assert!(stmt.collection.is_none());
}

// ============================================================================
// Nominal — FLUSH FULL (all collections, full)
// ============================================================================

#[test]
fn test_parse_flush_full() {
    let query = Parser::parse("FLUSH FULL").expect("FLUSH FULL should parse");

    let admin = query.admin.expect("Expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(stmt.full, "FLUSH FULL should be full");
    assert!(stmt.collection.is_none(), "FLUSH FULL flushes all");
}

// ============================================================================
// Nominal — FLUSH <collection> (specific, fast)
// ============================================================================

#[test]
fn test_parse_flush_named_collection() {
    let query = Parser::parse("FLUSH docs").expect("FLUSH docs should parse");

    let admin = query.admin.expect("Expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(!stmt.full);
    assert_eq!(stmt.collection.as_deref(), Some("docs"));
}

// ============================================================================
// Nominal — FLUSH FULL <collection> (specific, full)
// ============================================================================

#[test]
fn test_parse_flush_full_named_collection() {
    let query = Parser::parse("FLUSH FULL docs").expect("FLUSH FULL docs should parse");

    let admin = query.admin.expect("Expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(stmt.full, "should be full flush");
    assert_eq!(stmt.collection.as_deref(), Some("docs"));
}

// ============================================================================
// Case insensitivity
// ============================================================================

#[test]
fn test_parse_flush_case_insensitive() {
    let query = Parser::parse("flush full Docs").expect("lowercase should parse");

    let admin = query.admin.expect("Expected admin");
    let AdminStatement::Flush(stmt) = admin;

    assert!(stmt.full);
    assert_eq!(stmt.collection.as_deref(), Some("Docs"));
}

// ============================================================================
// Query type predicates
// ============================================================================

#[test]
fn test_flush_query_type_flags() {
    let query = Parser::parse("FLUSH").expect("should parse");

    assert!(query.is_admin_query(), "FLUSH should be admin");
    assert!(!query.is_select_query());
    assert!(!query.is_dml_query());
    assert!(!query.is_ddl_query());
    assert!(!query.is_match_query());
    assert!(!query.is_train());
    assert!(!query.is_introspection_query());
}

#[test]
fn test_flush_full_query_type_flags() {
    let query = Parser::parse("FLUSH FULL docs").expect("should parse");

    assert!(query.is_admin_query(), "FLUSH FULL docs should be admin");
    assert!(!query.is_select_query());
    assert!(!query.is_dml_query());
}

// ============================================================================
// Regression — 'full' as reserved word vs collection name (Devin review)
// ============================================================================

/// `FLUSH full` must parse as a FULL flush of all collections, not as a
/// fast flush of a collection named "full". The keyword takes precedence.
#[test]
fn test_flush_full_is_reserved_word() {
    let query = Parser::parse("FLUSH full").expect("test: FLUSH full should parse");

    let admin = query.admin.expect("test: expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(stmt.full, "FLUSH full should set full=true (keyword consumed)");
    assert!(
        stmt.collection.is_none(),
        "FLUSH full should not produce a collection name"
    );
}

/// `` FLUSH `full` `` uses backtick escaping to treat "full" as a literal
/// collection name, bypassing the reserved word.
#[test]
fn test_flush_backtick_full_is_collection() {
    let query = Parser::parse("FLUSH `full`").expect("test: FLUSH `full` should parse");

    let admin = query.admin.expect("test: expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(
        !stmt.full,
        "backtick-escaped `full` should NOT activate the FULL keyword"
    );
    assert_eq!(
        stmt.collection.as_deref(),
        Some("full"),
        "backtick-escaped `full` should be the collection name"
    );
}

/// `FLUSH FULL full` uses the FULL keyword followed by a collection literally
/// named "full". Both the keyword and collection must be recognised.
#[test]
fn test_flush_full_full_is_collection() {
    let query = Parser::parse("FLUSH FULL full").expect("test: FLUSH FULL full should parse");

    let admin = query.admin.expect("test: expected admin statement");
    let AdminStatement::Flush(stmt) = admin;

    assert!(stmt.full, "FLUSH FULL full should set full=true");
    assert_eq!(
        stmt.collection.as_deref(),
        Some("full"),
        "the trailing 'full' after the FULL keyword should be the collection name"
    );
}
