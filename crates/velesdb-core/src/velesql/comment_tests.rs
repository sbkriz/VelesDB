//! Parser tests for SQL comment support (`--` single-line comments).
//!
//! VelesQL grammar treats `--` as a single-line comment (stripped as WHITESPACE
//! by pest). These tests verify that comments are correctly ignored in various
//! positions and that `--` inside string literals is NOT treated as a comment.

use crate::velesql::Parser;

// ============================================================================
// Nominal — comments in valid positions
// ============================================================================

#[test]
fn test_comment_at_end_of_query() {
    let query = "SELECT * FROM docs LIMIT 10 -- get all docs";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    assert_eq!(parsed.select.from, "docs");
    assert_eq!(parsed.select.limit, Some(10));
}

#[test]
fn test_comment_on_own_line_before_query() {
    let query = "-- fetch documents\nSELECT * FROM docs LIMIT 5";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    assert_eq!(parsed.select.from, "docs");
    assert_eq!(parsed.select.limit, Some(5));
}

#[test]
fn test_comment_between_clauses() {
    let query = "SELECT * FROM docs -- source collection\nWHERE id = 1";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    assert_eq!(parsed.select.from, "docs");
    assert!(parsed.select.where_clause.is_some());
}

#[test]
fn test_multiple_comments() {
    let query = "\
-- first comment\n\
SELECT * FROM docs -- inline comment\n\
WHERE id = 1 -- filter by id\n\
LIMIT 10 -- cap results";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    assert_eq!(parsed.select.from, "docs");
    assert_eq!(parsed.select.limit, Some(10));
}

#[test]
fn test_comment_with_special_characters() {
    let query = "SELECT * FROM docs LIMIT 1 -- this is a test! #$%&";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    assert_eq!(parsed.select.from, "docs");
    assert_eq!(parsed.select.limit, Some(1));
}

#[test]
fn test_empty_comment_at_end() {
    let query = "SELECT * FROM docs --";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    assert_eq!(parsed.select.from, "docs");
}

// ============================================================================
// String literal — `--` inside strings must NOT be a comment
// ============================================================================

#[test]
fn test_double_dash_inside_string_literal_not_comment() {
    let query = "SELECT * FROM docs WHERE name = 'test--value'";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    assert_eq!(parsed.select.from, "docs");
    assert!(parsed.select.where_clause.is_some());
}

// ============================================================================
// Negative — comment-only input must fail
// ============================================================================

#[test]
fn test_comment_only_fails() {
    let result = Parser::parse("-- just a comment");
    assert!(
        result.is_err(),
        "Comment-only input should not parse as a valid query"
    );
}

#[test]
fn test_multiple_comment_lines_only_fails() {
    let result = Parser::parse("-- line one\n-- line two\n-- line three");
    assert!(
        result.is_err(),
        "Multiple comment-only lines should not parse"
    );
}
