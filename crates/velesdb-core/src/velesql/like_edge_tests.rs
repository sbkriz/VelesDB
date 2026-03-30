//! Edge-case tests for LIKE and ILIKE parsing in `VelesQL`.
//!
//! Covers wildcard patterns (`%`, `_`, mixed), special content (spaces,
//! numbers, unicode, escaped quotes), ILIKE specifics, combined conditions
//! (AND, OR, NOT, vector NEAR), AST field validation, and negative cases.
//!
//! Existing coverage (in `parser_tests.rs` and `negative_edge_tests.rs`):
//! - basic LIKE, basic ILIKE, ILIKE lowercase, quoted identifiers, LIKE
//!   without pattern.

use crate::velesql::{Condition, Parser};

// =============================================================================
// Group 1 — LIKE wildcard patterns
// =============================================================================

/// `%` matches everything (zero or more characters).
#[test]
fn test_like_percent_only() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE '%'")
        .expect("percent-only pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.column, "name");
    assert_eq!(like.pattern, "%");
    assert!(!like.case_insensitive);
}

/// `_` matches exactly one character.
#[test]
fn test_like_underscore_only() {
    let q = Parser::parse("SELECT * FROM docs WHERE code LIKE '_'")
        .expect("underscore-only pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "_");
}

/// Multiple underscores match exactly that many characters.
#[test]
fn test_like_multiple_underscores() {
    let q = Parser::parse("SELECT * FROM docs WHERE code LIKE '___'")
        .expect("triple underscore pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "___");
}

/// `%` at start: suffix matching.
#[test]
fn test_like_percent_at_start() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE '%suffix'")
        .expect("suffix pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "%suffix");
}

/// `%` at end: prefix matching.
#[test]
fn test_like_percent_at_end() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE 'prefix%'")
        .expect("prefix pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "prefix%");
}

/// `%` at both ends: contains matching.
#[test]
fn test_like_percent_both_ends() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE '%middle%'")
        .expect("contains pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "%middle%");
}

/// Mixed `_` and `%` wildcards.
#[test]
fn test_like_mixed_wildcards() {
    let q = Parser::parse("SELECT * FROM docs WHERE code LIKE 'A_%_B%'")
        .expect("mixed wildcard pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "A_%_B%");
}

/// No wildcards: exact-match semantics (LIKE still parses).
#[test]
fn test_like_no_wildcards() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE 'exact'")
        .expect("exact-match LIKE should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "exact");
}

/// Empty pattern: `''` is a valid SQL string.
#[test]
fn test_like_empty_pattern() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE ''")
        .expect("empty LIKE pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "");
}

// =============================================================================
// Group 2 — LIKE with special content
// =============================================================================

/// Pattern containing spaces.
#[test]
fn test_like_pattern_with_spaces() {
    let q = Parser::parse("SELECT * FROM docs WHERE title LIKE 'hello world%'")
        .expect("pattern with spaces should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "hello world%");
}

/// Pattern containing numbers and punctuation.
#[test]
fn test_like_pattern_with_numbers() {
    let q = Parser::parse("SELECT * FROM docs WHERE sku LIKE 'SKU-2024-%'")
        .expect("pattern with numbers should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "SKU-2024-%");
}

/// Pattern containing unicode characters.
#[test]
fn test_like_pattern_with_unicode() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE '\u{65E5}\u{672C}%'")
        .expect("unicode pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "\u{65E5}\u{672C}%");
}

/// Escaped single quote inside pattern (`''` -> `'`).
#[test]
fn test_like_pattern_with_escaped_quote() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE 'O''Brien%'")
        .expect("escaped-quote pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "O'Brien%");
}

// =============================================================================
// Group 3 — ILIKE edge cases
// =============================================================================

/// ILIKE with `%` only.
#[test]
fn test_ilike_percent_only() {
    let q = Parser::parse("SELECT * FROM docs WHERE name ILIKE '%'")
        .expect("ILIKE percent-only should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "%");
    assert!(like.case_insensitive);
}

/// ILIKE preserves original pattern casing (matching is case-insensitive
/// at runtime, but the AST stores the literal pattern as written).
#[test]
fn test_ilike_mixed_case_pattern() {
    let q = Parser::parse("SELECT * FROM docs WHERE name ILIKE 'JoHn%'")
        .expect("ILIKE mixed-case pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "JoHn%");
    assert!(like.case_insensitive);
}

/// ILIKE with underscore wildcard.
#[test]
fn test_ilike_with_underscore() {
    let q = Parser::parse("SELECT * FROM docs WHERE code ILIKE 'a_c'")
        .expect("ILIKE underscore should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "a_c");
    assert!(like.case_insensitive);
}

/// ILIKE with empty pattern.
#[test]
fn test_ilike_empty_pattern() {
    let q = Parser::parse("SELECT * FROM docs WHERE name ILIKE ''")
        .expect("ILIKE empty pattern should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.pattern, "");
    assert!(like.case_insensitive);
}

// =============================================================================
// Group 4 — LIKE combined with other operators
// =============================================================================

/// LIKE + AND: both branches are accessible in the condition tree.
#[test]
fn test_like_and_comparison() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE '%test%' AND active = true")
        .expect("LIKE + AND should parse");
    let Condition::And(left, right) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected And condition");
    };
    assert!(
        matches!(*left, Condition::Like(_)),
        "Left branch should be Like, got: {left:?}"
    );
    assert!(
        matches!(*right, Condition::Comparison(_)),
        "Right branch should be Comparison, got: {right:?}"
    );
}

/// LIKE + OR: two LIKE branches.
#[test]
fn test_like_or_like() {
    let q = Parser::parse("SELECT * FROM docs WHERE name LIKE '%a%' OR name LIKE '%b%'")
        .expect("LIKE OR LIKE should parse");
    let Condition::Or(left, right) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Or condition");
    };
    let Condition::Like(left_like) = *left else {
        panic!("Left branch should be Like");
    };
    assert_eq!(left_like.pattern, "%a%");

    let Condition::Like(right_like) = *right else {
        panic!("Right branch should be Like");
    };
    assert_eq!(right_like.pattern, "%b%");
}

/// LIKE + vector NEAR in a single WHERE clause.
#[test]
fn test_like_and_vector_near() {
    let sql = "SELECT * FROM docs WHERE vector NEAR $v AND title LIKE '%AI%' LIMIT 10";
    let q = Parser::parse(sql).expect("LIKE + vector NEAR should parse");

    let Condition::And(left, right) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected And condition");
    };
    assert!(
        matches!(*left, Condition::VectorSearch(_)),
        "Left should be VectorSearch, got: {left:?}"
    );
    let Condition::Like(like) = *right else {
        panic!("Right should be Like, got: {right:?}");
    };
    assert_eq!(like.pattern, "%AI%");
    assert_eq!(q.select.limit, Some(10));
}

/// NOT + LIKE: the parsed tree wraps LIKE inside Group(Not(...)).
#[test]
fn test_not_like() {
    let q = Parser::parse("SELECT * FROM docs WHERE NOT (name LIKE '%spam%')")
        .expect("NOT LIKE should parse");

    let Condition::Not(inner) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Not condition");
    };
    // NOT wraps a Group which wraps the Like
    let Condition::Group(grouped) = *inner else {
        panic!("Expected Group inside Not, got: {inner:?}");
    };
    let Condition::Like(like) = *grouped else {
        panic!("Expected Like inside Group, got: {grouped:?}");
    };
    assert_eq!(like.pattern, "%spam%");
}

/// ILIKE + AND + comparison with LIMIT.
#[test]
fn test_ilike_and_comparison_with_limit() {
    let sql = "SELECT * FROM docs WHERE title ILIKE '%db%' AND price > 50 LIMIT 10";
    let q = Parser::parse(sql).expect("ILIKE + AND + comparison should parse");

    let Condition::And(left, right) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected And condition");
    };
    let Condition::Like(like) = *left else {
        panic!("Left should be Like (ILIKE), got: {left:?}");
    };
    assert!(like.case_insensitive);
    assert_eq!(like.pattern, "%db%");

    assert!(
        matches!(*right, Condition::Comparison(_)),
        "Right should be Comparison, got: {right:?}"
    );
    assert_eq!(q.select.limit, Some(10));
}

// =============================================================================
// Group 5 — AST field validation
// =============================================================================

/// Verify all `LikeCondition` fields for a LIKE query.
#[test]
fn test_like_ast_fields() {
    let q =
        Parser::parse("SELECT * FROM docs WHERE name LIKE '%test%'").expect("should parse LIKE");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.column, "name");
    assert_eq!(like.pattern, "%test%");
    assert!(!like.case_insensitive, "LIKE must be case-sensitive");
}

/// Verify all `LikeCondition` fields for an ILIKE query.
#[test]
fn test_ilike_ast_fields() {
    let q =
        Parser::parse("SELECT * FROM docs WHERE title ILIKE 'Hello%'").expect("should parse ILIKE");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.column, "title");
    assert_eq!(like.pattern, "Hello%");
    assert!(like.case_insensitive, "ILIKE must be case-insensitive");
}

/// Nested column reference (dot-separated path) in LIKE.
#[test]
fn test_like_nested_column() {
    let q = Parser::parse("SELECT * FROM docs WHERE payload.title LIKE '%test%'")
        .expect("nested column LIKE should parse");
    let Condition::Like(like) = q.select.where_clause.expect("should have WHERE") else {
        panic!("Expected Like condition");
    };
    assert_eq!(like.column, "payload.title");
    assert_eq!(like.pattern, "%test%");
}

// =============================================================================
// Group 6 — Negative cases (must fail)
// =============================================================================

/// LIKE with an integer literal instead of a string pattern.
#[test]
fn test_reject_like_integer_pattern() {
    let query = "SELECT * FROM docs WHERE name LIKE 123";
    assert!(
        Parser::parse(query).is_err(),
        "LIKE with integer pattern should fail: {query}"
    );
}

/// LIKE with a bind parameter instead of a string literal.
#[test]
fn test_reject_like_parameter_pattern() {
    let query = "SELECT * FROM docs WHERE name LIKE $pattern";
    assert!(
        Parser::parse(query).is_err(),
        "LIKE with parameter pattern should fail: {query}"
    );
}

/// LIKE without a preceding column name.
#[test]
fn test_reject_like_missing_column() {
    let query = "SELECT * FROM docs WHERE LIKE '%test%'";
    assert!(
        Parser::parse(query).is_err(),
        "LIKE without column should fail: {query}"
    );
}
