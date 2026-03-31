//! Comprehensive negative and edge-case parser tests.
//!
//! Validates that the `VelesQL` parser correctly rejects malformed input and
//! gracefully handles boundary conditions. Organized into groups:
//!
//! - **Group 1**: Invalid statement structure
//! - **Group 2**: Invalid WHERE conditions
//! - **Group 3**: Invalid vector operations
//! - **Group 4**: Invalid DDL
//! - **Group 5**: Invalid DML
//! - **Group 6**: Invalid LET bindings
//! - **Group 7**: Invalid TRAIN statements
//! - **Group 8**: Edge cases that SHOULD parse

use crate::velesql::Parser;

// ============================================================================
// Group 1 — Invalid statement structure
// ============================================================================

/// Empty string is not a valid query.
#[test]
fn test_reject_empty_string() {
    let query = "";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// Whitespace-only input is not a valid query.
#[test]
fn test_reject_whitespace_only() {
    let query = "   ";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// Random English text is not a valid query.
#[test]
fn test_reject_random_text() {
    let query = "hello world";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// SELECT without FROM clause is invalid.
#[test]
fn test_reject_select_without_from() {
    let query = "SELECT * WHERE id = 1";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// Missing SELECT keyword (bare column list with FROM).
#[test]
fn test_reject_missing_select_keyword() {
    let query = "* FROM docs";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// Double semicolons should fail (second `;` is garbage after EOI).
#[test]
fn test_reject_double_semicolons() {
    let query = "SELECT * FROM docs;;";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// CREATE TABLE is not valid VelesQL; only CREATE COLLECTION is.
#[test]
fn test_reject_create_table() {
    let query = "CREATE TABLE docs (dimension = 768)";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// DROP TABLE is not valid VelesQL; only DROP COLLECTION is.
#[test]
fn test_reject_drop_table() {
    let query = "DROP TABLE docs";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

// ============================================================================
// Group 2 — Invalid WHERE conditions
// ============================================================================

/// Comparison operator without right-hand value.
#[test]
fn test_reject_comparison_without_value() {
    let query = "SELECT * FROM docs WHERE id >";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// Double comparison operators are not valid.
#[test]
fn test_reject_double_operator() {
    let query = "SELECT * FROM docs WHERE id >= <= 5";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// IN with empty parenthesized list is not valid (grammar requires at least one value).
#[test]
fn test_reject_in_with_empty_list() {
    let query = "SELECT * FROM docs WHERE id IN ()";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// BETWEEN requires AND between the two bounds.
#[test]
fn test_reject_between_missing_and() {
    let query = "SELECT * FROM docs WHERE id BETWEEN 1 5";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// LIKE requires a string pattern argument.
#[test]
fn test_reject_like_without_pattern() {
    let query = "SELECT * FROM docs WHERE name LIKE";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// IS must be followed by NULL or NOT NULL.
#[test]
fn test_reject_is_without_null() {
    let query = "SELECT * FROM docs WHERE name IS";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

// ============================================================================
// Group 3 — Invalid vector operations
// ============================================================================

/// NEAR without a vector value (literal or parameter).
#[test]
fn test_reject_near_without_vector() {
    let query = "SELECT * FROM docs WHERE vector NEAR LIMIT 10";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// `SPARSE_NEAR {}` — empty sparse literal is invalid (grammar requires at least one entry).
#[test]
fn test_reject_sparse_near_empty_literal() {
    let query = "SELECT * FROM docs WHERE vector SPARSE_NEAR {} LIMIT 5";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// `NEAR_FUSED []` — empty vector array is invalid (grammar requires at least one vector).
#[test]
fn test_reject_near_fused_empty_array() {
    let query = "SELECT * FROM docs WHERE vector NEAR_FUSED [] LIMIT 5";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

// ============================================================================
// Group 4 — Invalid DDL
// ============================================================================

/// CREATE COLLECTION without a name.
#[test]
fn test_reject_create_collection_without_name() {
    let query = "CREATE COLLECTION";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// DROP without COLLECTION keyword.
#[test]
fn test_reject_drop_without_collection_keyword() {
    let query = "DROP docs";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// CREATE VECTOR COLLECTION — "VECTOR" is not a valid collection kind
/// (grammar only accepts GRAPH and METADATA as `collection_kind_kw`).
#[test]
fn test_reject_create_vector_collection_explicit_kind() {
    let query = "CREATE VECTOR COLLECTION docs (dimension = 768)";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

// ============================================================================
// Group 5 — Invalid DML
// ============================================================================

/// DELETE FROM without WHERE is invalid (grammar mandates WHERE clause).
#[test]
fn test_reject_delete_without_where() {
    let query = "DELETE FROM docs";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// DELETE EDGE without an edge-id value before FROM.
#[test]
fn test_reject_delete_edge_without_id() {
    let query = "DELETE EDGE FROM kg";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// INSERT EDGE INTO — missing the rest of the statement.
#[test]
fn test_reject_insert_edge_incomplete() {
    let query = "INSERT EDGE INTO";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// UPDATE without SET clause.
#[test]
fn test_reject_update_without_set() {
    let query = "UPDATE docs WHERE id = 1";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

// ============================================================================
// Group 6 — Invalid LET bindings
// ============================================================================

/// LET with missing binding name (starts with `=`).
#[test]
fn test_reject_let_missing_name() {
    let query = "LET = 5 SELECT * FROM docs LIMIT 5";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// LET without the `=` assignment operator.
#[test]
fn test_reject_let_missing_assignment() {
    let query = "LET x SELECT * FROM docs LIMIT 5";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

// ============================================================================
// Group 7 — Invalid TRAIN statements
// ============================================================================

/// TRAIN without the QUANTIZER keyword.
#[test]
fn test_reject_train_without_quantizer() {
    let query = "TRAIN my_collection WITH (m=8)";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// TRAIN QUANTIZER missing ON keyword.
#[test]
fn test_reject_train_quantizer_without_on() {
    let query = "TRAIN QUANTIZER my_collection WITH (m=8)";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

/// TRAIN QUANTIZER ON collection without WITH clause.
#[test]
fn test_reject_train_quantizer_without_with() {
    let query = "TRAIN QUANTIZER ON my_collection";
    assert!(Parser::parse(query).is_err(), "Should have failed: {query}");
}

// ============================================================================
// Group 8 — Edge cases that SHOULD parse
// ============================================================================

/// A 100-character collection name should be accepted.
#[test]
fn test_accept_long_collection_name() {
    let long_name = "a".repeat(100);
    let query = format!("SELECT * FROM {long_name} LIMIT 10");
    assert!(
        Parser::parse(&query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(&query).err()
    );
}

/// Deeply nested AND/OR conditions (5 levels).
#[test]
fn test_accept_deeply_nested_conditions() {
    let query =
        "SELECT * FROM docs WHERE (a = 1 AND (b = 2 OR (c = 3 AND (d = 4 OR e = 5)))) LIMIT 10";
    assert!(
        Parser::parse(query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(query).err()
    );
}

/// Trailing semicolon should be consumed by the grammar.
#[test]
fn test_accept_trailing_semicolon() {
    let query = "SELECT * FROM docs LIMIT 10;";
    assert!(
        Parser::parse(query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(query).err()
    );
}

/// All-uppercase keywords should parse (case-insensitive grammar).
#[test]
fn test_accept_all_caps_keywords() {
    let query = "SELECT * FROM docs WHERE id = 1 LIMIT 10";
    assert!(
        Parser::parse(query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(query).err()
    );
}

/// Mixed-case keywords should parse (PEG `^` case-insensitive matching).
#[test]
fn test_accept_mixed_case_keywords() {
    let query = "SeLeCt * FrOm docs LiMiT 10";
    assert!(
        Parser::parse(query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(query).err()
    );
}

/// Unicode characters inside string literal values.
#[test]
fn test_accept_unicode_in_string_values() {
    let query = "SELECT * FROM docs WHERE name = '\u{65E5}\u{672C}\u{8A9E}'";
    assert!(
        Parser::parse(query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(query).err()
    );
}

/// Very large integer literal should be accepted by the grammar.
#[test]
fn test_accept_very_large_integer() {
    let query = "SELECT * FROM docs WHERE id = 999999999999";
    assert!(
        Parser::parse(query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(query).err()
    );
}

/// Negative float literal in WHERE condition.
#[test]
fn test_accept_negative_float() {
    let query = "SELECT * FROM docs WHERE score > -0.5";
    assert!(
        Parser::parse(query).is_ok(),
        "Should have parsed: {:?}",
        Parser::parse(query).err()
    );
}
