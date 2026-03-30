//! Edge-case tests for HAVING, Set Operations, OFFSET, DISTINCT, and WITH
//! clauses in the VelesQL parser.
//!
//! These tests exercise boundary conditions and combinations that sit outside
//! the nominal happy-path covered by the per-clause test modules.

use crate::velesql::{
    AggregateArg, AggregateType, CompareOp, DistinctMode, LogicalOp, Parser, SetOperator, Value,
    WithValue,
};

// ========== HAVING — compound conditions ==========

#[test]
fn test_having_and_two_aggregates() {
    let sql = "SELECT cat, COUNT(*) FROM p \
               GROUP BY cat \
               HAVING COUNT(*) > 5 AND AVG(price) < 100";
    let query = Parser::parse(sql).expect("HAVING with AND should parse");

    let having = query
        .select
        .having
        .as_ref()
        .expect("HAVING should be present");
    assert_eq!(having.conditions.len(), 2, "two conditions expected");
    assert_eq!(having.operators.len(), 1, "one operator expected");
    assert_eq!(having.operators[0], LogicalOp::And);

    // First condition: COUNT(*) > 5
    assert_eq!(
        having.conditions[0].aggregate.function_type,
        AggregateType::Count
    );
    assert_eq!(
        having.conditions[0].aggregate.argument,
        AggregateArg::Wildcard
    );
    assert_eq!(having.conditions[0].operator, CompareOp::Gt);
    assert_eq!(having.conditions[0].value, Value::Integer(5));

    // Second condition: AVG(price) < 100
    assert_eq!(
        having.conditions[1].aggregate.function_type,
        AggregateType::Avg
    );
    assert_eq!(
        having.conditions[1].aggregate.argument,
        AggregateArg::Column("price".to_string())
    );
    assert_eq!(having.conditions[1].operator, CompareOp::Lt);
    assert_eq!(having.conditions[1].value, Value::Integer(100));
}

#[test]
fn test_having_or_operator() {
    let sql = "SELECT r, SUM(a) FROM s \
               GROUP BY r \
               HAVING SUM(a) > 1000 OR COUNT(*) > 50";
    let query = Parser::parse(sql).expect("HAVING with OR should parse");

    let having = query
        .select
        .having
        .as_ref()
        .expect("HAVING should be present");
    assert_eq!(having.conditions.len(), 2);
    assert_eq!(having.operators, vec![LogicalOp::Or]);

    assert_eq!(
        having.conditions[0].aggregate.function_type,
        AggregateType::Sum
    );
    assert_eq!(
        having.conditions[1].aggregate.function_type,
        AggregateType::Count
    );
}

#[test]
fn test_having_min_max_aggregates() {
    let sql = "SELECT cat, MIN(price), MAX(price) FROM items \
               GROUP BY cat \
               HAVING MIN(price) > 0 AND MAX(price) < 10000";
    let query = Parser::parse(sql).expect("HAVING with MIN/MAX should parse");

    let having = query
        .select
        .having
        .as_ref()
        .expect("HAVING should be present");
    assert_eq!(having.conditions.len(), 2);
    assert_eq!(
        having.conditions[0].aggregate.function_type,
        AggregateType::Min
    );
    assert_eq!(having.conditions[0].operator, CompareOp::Gt);
    assert_eq!(
        having.conditions[1].aggregate.function_type,
        AggregateType::Max
    );
    assert_eq!(having.conditions[1].operator, CompareOp::Lt);
}

#[test]
fn test_having_sum_float_threshold() {
    let sql = "SELECT dept, SUM(amount) FROM expenses \
               GROUP BY dept \
               HAVING SUM(amount) > 99.99";
    let query = Parser::parse(sql).expect("HAVING with float threshold should parse");

    let having = query
        .select
        .having
        .as_ref()
        .expect("HAVING should be present");
    assert_eq!(having.conditions.len(), 1);
    assert_eq!(
        having.conditions[0].aggregate.function_type,
        AggregateType::Sum
    );

    // The grammar parses 99.99 as a float.
    let threshold = match &having.conditions[0].value {
        Value::Float(f) => *f,
        other => panic!("Expected Float value, got: {other:?}"),
    };
    assert!((threshold - 99.99).abs() < f64::EPSILON);
}

#[test]
fn test_having_without_group_by_parses() {
    // Grammar allows HAVING without GROUP BY — semantic validation rejects it.
    let sql = "SELECT COUNT(*) FROM docs HAVING COUNT(*) > 0";
    let query = Parser::parse(sql).expect("HAVING without GROUP BY should still parse");

    assert!(query.select.group_by.is_none(), "no GROUP BY in query");
    assert!(query.select.having.is_some(), "HAVING should be present");
}

// ========== HAVING — negative ==========

#[test]
fn test_having_without_aggregate_fails() {
    // Grammar rule: having_term = { aggregate_function ~ compare_op ~ value }
    // A bare column reference does not match aggregate_function.
    let sql = "SELECT * FROM items GROUP BY cat HAVING price > 10";
    let result = Parser::parse(sql);
    assert!(
        result.is_err(),
        "HAVING without aggregate should fail to parse"
    );
}

// ========== Set Operations — edge cases ==========

#[test]
fn test_three_way_union_operators() {
    let sql = "SELECT * FROM a UNION SELECT * FROM b UNION SELECT * FROM c";
    let query = Parser::parse(sql).expect("three-way UNION should parse");

    let compound = query.compound.as_ref().expect("compound should be present");
    assert_eq!(compound.operations.len(), 2);
    assert_eq!(compound.operations[0].0, SetOperator::Union);
    assert_eq!(compound.operations[1].0, SetOperator::Union);
    assert_eq!(compound.operations[0].1.from, "b");
    assert_eq!(compound.operations[1].1.from, "c");
}

#[test]
fn test_union_all_operator() {
    let sql = "SELECT * FROM t1 UNION ALL SELECT * FROM t2";
    let query = Parser::parse(sql).expect("UNION ALL should parse");

    let compound = query.compound.as_ref().expect("compound should be present");
    assert_eq!(compound.operations.len(), 1);
    assert_eq!(compound.operations[0].0, SetOperator::UnionAll);
}

#[test]
fn test_mixed_union_intersect_left_to_right() {
    let sql = "SELECT * FROM a UNION SELECT * FROM b INTERSECT SELECT * FROM c";
    let query = Parser::parse(sql).expect("mixed set ops should parse");

    let compound = query.compound.as_ref().expect("compound should be present");
    assert_eq!(compound.operations.len(), 2);
    // Left-to-right evaluation: first UNION, then INTERSECT.
    assert_eq!(compound.operations[0].0, SetOperator::Union);
    assert_eq!(compound.operations[1].0, SetOperator::Intersect);
}

#[test]
fn test_except_operator() {
    let sql = "SELECT id FROM all_items EXCEPT SELECT id FROM deleted";
    let query = Parser::parse(sql).expect("EXCEPT should parse");

    let compound = query.compound.as_ref().expect("compound should be present");
    assert_eq!(compound.operations.len(), 1);
    assert_eq!(compound.operations[0].0, SetOperator::Except);
    assert_eq!(compound.operations[0].1.from, "deleted");
}

#[test]
fn test_set_operation_limit_on_last_select() {
    // LIMIT is part of the last SELECT's clauses, not the compound.
    let sql = "SELECT * FROM a UNION SELECT * FROM b LIMIT 10";
    let query = Parser::parse(sql).expect("UNION + LIMIT on last SELECT should parse");

    let compound = query.compound.as_ref().expect("compound should be present");
    assert_eq!(compound.operations.len(), 1);
    // LIMIT belongs to the second SELECT statement.
    assert_eq!(compound.operations[0].1.limit, Some(10));
}

#[test]
fn test_set_operation_where_on_each() {
    let sql = "SELECT * FROM a WHERE x = 1 UNION SELECT * FROM b WHERE y = 2";
    let query = Parser::parse(sql).expect("UNION with WHERE on each should parse");

    // First SELECT has WHERE
    assert!(
        query.select.where_clause.is_some(),
        "first SELECT should have WHERE"
    );
    // Second SELECT (inside compound) also has WHERE
    let compound = query.compound.as_ref().expect("compound should be present");
    assert!(
        compound.operations[0].1.where_clause.is_some(),
        "second SELECT should have WHERE"
    );
}

#[test]
fn test_set_operation_mismatched_columns_parses() {
    // Column mismatch is a semantic error, not a parse error.
    let sql = "SELECT id FROM a UNION SELECT name, age FROM b";
    let result = Parser::parse(sql);
    assert!(
        result.is_ok(),
        "mismatched columns should still parse successfully"
    );
}

// ========== OFFSET — edge cases ==========

#[test]
fn test_offset_zero() {
    let sql = "SELECT * FROM docs LIMIT 10 OFFSET 0";
    let query = Parser::parse(sql).expect("OFFSET 0 should parse");
    assert_eq!(query.select.limit, Some(10));
    assert_eq!(query.select.offset, Some(0));
}

#[test]
fn test_offset_without_limit() {
    // Grammar defines offset_clause independently of limit_clause.
    let sql = "SELECT * FROM docs OFFSET 10";
    let query = Parser::parse(sql).expect("OFFSET without LIMIT should parse");
    assert!(query.select.limit.is_none(), "LIMIT should be absent");
    assert_eq!(query.select.offset, Some(10));
}

#[test]
fn test_large_offset() {
    let sql = "SELECT * FROM docs LIMIT 10 OFFSET 1000000";
    let query = Parser::parse(sql).expect("large OFFSET should parse");
    assert_eq!(query.select.offset, Some(1_000_000));
}

#[test]
fn test_offset_with_vector_near() {
    let sql = "SELECT * FROM docs WHERE vector NEAR $v LIMIT 10 OFFSET 5";
    let query = Parser::parse(sql).expect("OFFSET with vector NEAR should parse");
    assert!(query.select.where_clause.is_some());
    assert_eq!(query.select.limit, Some(10));
    assert_eq!(query.select.offset, Some(5));
}

// ========== OFFSET — negative ==========

#[test]
fn test_offset_negative_fails() {
    // Grammar allows negative integer token, but parse_u64_clause rejects it.
    let sql = "SELECT * FROM docs OFFSET -1";
    let result = Parser::parse(sql);
    assert!(
        result.is_err(),
        "negative OFFSET should fail to parse as u64"
    );
}

// ========== DISTINCT — edge cases ==========

#[test]
fn test_distinct_with_aggregation() {
    let sql = "SELECT DISTINCT category, COUNT(*) FROM p GROUP BY category";
    let query = Parser::parse(sql).expect("DISTINCT with aggregation should parse");
    assert_eq!(query.select.distinct, DistinctMode::All);
    assert!(query.select.group_by.is_some());
}

#[test]
fn test_distinct_with_order_by_non_selected() {
    // Parse-level: DISTINCT + ORDER BY on a column not in SELECT list.
    // Semantic validation may reject this, but parsing should succeed.
    let sql = "SELECT DISTINCT category FROM p ORDER BY price ASC LIMIT 10";
    let query = Parser::parse(sql).expect("DISTINCT + ORDER BY non-selected should parse");
    assert_eq!(query.select.distinct, DistinctMode::All);
    assert!(query.select.order_by.is_some());
    assert_eq!(query.select.limit, Some(10));
}

#[test]
fn test_distinct_with_join() {
    let sql = "SELECT DISTINCT d.category FROM docs AS d \
               JOIN tags AS t ON d.id = t.doc_id";
    let query = Parser::parse(sql).expect("DISTINCT with JOIN should parse");
    assert_eq!(query.select.distinct, DistinctMode::All);
    assert!(!query.select.joins.is_empty(), "JOIN should be present");
}

#[test]
fn test_distinct_with_vector_near() {
    let sql = "SELECT DISTINCT category FROM docs WHERE vector NEAR $v LIMIT 10";
    let query = Parser::parse(sql).expect("DISTINCT + vector NEAR should parse");
    assert_eq!(query.select.distinct, DistinctMode::All);
    assert!(query.select.where_clause.is_some());
    assert_eq!(query.select.limit, Some(10));
}

// ========== WITH clause — edge cases ==========

#[test]
fn test_with_all_known_options() {
    let sql = "SELECT * FROM docs \
               WITH (mode = 'accurate', ef_search = 512, timeout_ms = 5000, rerank = true)";
    let query = Parser::parse(sql).expect("WITH all options should parse");

    let with = query
        .select
        .with_clause
        .as_ref()
        .expect("WITH should be present");
    assert_eq!(with.options.len(), 4);

    let mode = with
        .options
        .iter()
        .find(|o| o.key == "mode")
        .expect("mode option");
    assert_eq!(mode.value, WithValue::String("accurate".to_string()));

    let ef = with
        .options
        .iter()
        .find(|o| o.key == "ef_search")
        .expect("ef_search option");
    assert_eq!(ef.value, WithValue::Integer(512));

    let timeout = with
        .options
        .iter()
        .find(|o| o.key == "timeout_ms")
        .expect("timeout_ms option");
    assert_eq!(timeout.value, WithValue::Integer(5000));

    let rerank = with
        .options
        .iter()
        .find(|o| o.key == "rerank")
        .expect("rerank option");
    assert_eq!(rerank.value, WithValue::Boolean(true));
}

#[test]
fn test_with_quantization_dual_and_oversampling() {
    let sql = "SELECT * FROM docs WITH (quantization = 'dual', oversampling = 2.0)";
    let query = Parser::parse(sql).expect("WITH quantization + oversampling should parse");

    let with = query
        .select
        .with_clause
        .as_ref()
        .expect("WITH should be present");
    assert_eq!(with.options.len(), 2);

    let quant = with
        .options
        .iter()
        .find(|o| o.key == "quantization")
        .expect("quantization option");
    assert_eq!(quant.value, WithValue::String("dual".to_string()));

    let os = with
        .options
        .iter()
        .find(|o| o.key == "oversampling")
        .expect("oversampling option");
    assert_eq!(os.value, WithValue::Float(2.0));
}

#[test]
fn test_with_unknown_key_parses() {
    // Parser does not validate WITH option keys — that is semantic.
    let sql = "SELECT * FROM docs WITH (custom_key = 'value')";
    let query = Parser::parse(sql).expect("WITH unknown key should parse");

    let with = query
        .select
        .with_clause
        .as_ref()
        .expect("WITH should be present");
    assert_eq!(with.options.len(), 1);
    assert_eq!(with.options[0].key, "custom_key");
    assert_eq!(
        with.options[0].value,
        WithValue::String("value".to_string())
    );
}

#[test]
fn test_with_single_option() {
    let sql = "SELECT * FROM docs WITH (mode = 'fast')";
    let query = Parser::parse(sql).expect("WITH single option should parse");

    let with = query
        .select
        .with_clause
        .as_ref()
        .expect("WITH should be present");
    assert_eq!(with.options.len(), 1);
    assert_eq!(with.options[0].key, "mode");
    assert_eq!(with.options[0].value, WithValue::String("fast".to_string()));
}

#[test]
fn test_with_integer_value() {
    let sql = "SELECT * FROM docs WITH (ef_search = 128)";
    let query = Parser::parse(sql).expect("WITH integer value should parse");

    let with = query
        .select
        .with_clause
        .as_ref()
        .expect("WITH should be present");
    assert_eq!(with.options.len(), 1);
    assert_eq!(with.options[0].key, "ef_search");
    assert_eq!(with.options[0].value, WithValue::Integer(128));
}

// ========== WITH clause — negative ==========

#[test]
fn test_with_empty_parens_fails() {
    // Grammar: with_option_list = { with_option ~ ("," ~ with_option)* }
    // Requires at least one option.
    let sql = "SELECT * FROM docs WITH ()";
    let result = Parser::parse(sql);
    assert!(
        result.is_err(),
        "WITH () with empty parens should fail to parse"
    );
}
