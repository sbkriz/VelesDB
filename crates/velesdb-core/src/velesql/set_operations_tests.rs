//! Tests for SQL set operations (EPIC-040 US-006).
//!
//! Covers:
//! - UNION / UNION ALL
//! - INTERSECT
//! - EXCEPT
//! - Chained N-ary operations

use crate::velesql::Parser;

#[test]
fn test_union_basic() {
    let sql = "SELECT * FROM products WHERE category = 'electronics' UNION SELECT * FROM products WHERE price > 100";
    let result = Parser::parse(sql);
    assert!(result.is_ok(), "Failed to parse UNION: {:?}", result.err());

    let query = result.unwrap();
    let compound = query
        .compound
        .as_ref()
        .expect("Compound query should be present");

    assert_eq!(compound.operations.len(), 1);
    assert_eq!(compound.operations[0].0, crate::velesql::SetOperator::Union);
}

#[test]
fn test_union_all() {
    let sql = "SELECT id FROM docs WHERE author = 'Alice' UNION ALL SELECT id FROM docs WHERE topic = 'AI'";
    let result = Parser::parse(sql);
    assert!(
        result.is_ok(),
        "Failed to parse UNION ALL: {:?}",
        result.err()
    );

    let query = result.unwrap();
    let compound = query
        .compound
        .as_ref()
        .expect("Compound query should be present");

    assert_eq!(compound.operations.len(), 1);
    assert_eq!(
        compound.operations[0].0,
        crate::velesql::SetOperator::UnionAll
    );
}

#[test]
fn test_intersect() {
    let sql = "SELECT id FROM active_users INTERSECT SELECT id FROM premium_users";
    let result = Parser::parse(sql);
    assert!(
        result.is_ok(),
        "Failed to parse INTERSECT: {:?}",
        result.err()
    );

    let query = result.unwrap();
    let compound = query
        .compound
        .as_ref()
        .expect("Compound query should be present");

    assert_eq!(compound.operations.len(), 1);
    assert_eq!(
        compound.operations[0].0,
        crate::velesql::SetOperator::Intersect
    );
}

#[test]
fn test_except() {
    let sql = "SELECT id FROM all_users EXCEPT SELECT id FROM banned_users";
    let result = Parser::parse(sql);
    assert!(result.is_ok(), "Failed to parse EXCEPT: {:?}", result.err());

    let query = result.unwrap();
    let compound = query
        .compound
        .as_ref()
        .expect("Compound query should be present");

    assert_eq!(compound.operations.len(), 1);
    assert_eq!(
        compound.operations[0].0,
        crate::velesql::SetOperator::Except
    );
}

#[test]
fn test_simple_select_no_compound() {
    let sql = "SELECT * FROM docs";
    let result = Parser::parse(sql);
    assert!(result.is_ok());

    let query = result.unwrap();
    assert!(query.compound.is_none());
}

// =========================================================================
// N-ary compound queries (Bug 2, issue #383)
// =========================================================================

#[test]
fn test_chained_three_way_union() {
    let sql = "SELECT * FROM a UNION SELECT * FROM b UNION SELECT * FROM c";
    let query = Parser::parse(sql).expect("should parse 3-way UNION");

    let compound = query.compound.as_ref().expect("compound should be present");
    assert_eq!(compound.operations.len(), 2);
    assert_eq!(compound.operations[0].0, crate::velesql::SetOperator::Union);
    assert_eq!(compound.operations[1].0, crate::velesql::SetOperator::Union);
    assert_eq!(compound.operations[0].1.from, "b");
    assert_eq!(compound.operations[1].1.from, "c");
}

#[test]
fn test_chained_mixed_operators() {
    let sql = "SELECT * FROM a UNION SELECT * FROM b INTERSECT SELECT * FROM c";
    let query = Parser::parse(sql).expect("should parse mixed operators");

    let compound = query.compound.as_ref().expect("compound should be present");
    assert_eq!(compound.operations.len(), 2);
    assert_eq!(compound.operations[0].0, crate::velesql::SetOperator::Union);
    assert_eq!(
        compound.operations[1].0,
        crate::velesql::SetOperator::Intersect
    );
}
