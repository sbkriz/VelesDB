//! Edge-case parser tests for JOIN, ORDER BY, IN/NOT IN, BETWEEN, IS NULL,
//! and Subquery operators.
//!
//! These tests complement the existing coverage in `join_extended_tests`,
//! `orderby_multi_tests`, `parser_tests`, and `negative_edge_tests` by
//! exercising boundary conditions, multi-clause compositions, and
//! malformed-input rejection for each operator family.

use crate::velesql::{
    AggregateType, ArithmeticExpr, ArithmeticOp, CompareOp, Condition, JoinType, OrderByExpr,
    Parser, SelectColumns, Value,
};

// ============================================================================
// Group 1 — JOIN edge cases
// ============================================================================

/// JOIN combined with vector search in WHERE clause.
#[test]
fn test_join_with_vector_search() {
    let sql = "SELECT d.title FROM docs AS d \
               JOIN categories AS c ON d.cat_id = c.id \
               WHERE vector NEAR $v LIMIT 10";
    let query = Parser::parse(sql).expect("JOIN + vector NEAR should parse");

    assert_eq!(query.select.joins.len(), 1);
    assert_eq!(query.select.joins[0].table, "categories");
    assert_eq!(query.select.joins[0].alias, Some("c".to_string()));
    assert!(
        query.select.where_clause.is_some(),
        "WHERE clause with vector search must be present"
    );
    assert_eq!(query.select.limit, Some(10));
}

/// Multiple JOINs chaining three tables.
#[test]
fn test_multiple_joins_three_tables() {
    let sql = "SELECT a.x, b.y, c.z FROM a \
               JOIN b ON a.id = b.a_id \
               JOIN c ON b.id = c.b_id";
    let query = Parser::parse(sql).expect("triple JOIN should parse");

    assert_eq!(query.select.joins.len(), 2, "should have exactly 2 JOINs");
    assert_eq!(query.select.joins[0].table, "b");
    assert_eq!(query.select.joins[1].table, "c");
    assert_eq!(query.select.joins[0].join_type, JoinType::Inner);
    assert_eq!(query.select.joins[1].join_type, JoinType::Inner);
}

/// JOIN combined with GROUP BY and aggregate in projection.
#[test]
fn test_join_with_group_by() {
    let sql = "SELECT c.name, COUNT(*) FROM orders AS o \
               JOIN customers AS c ON o.cid = c.id \
               GROUP BY c.name";
    let query = Parser::parse(sql).expect("JOIN + GROUP BY should parse");

    assert_eq!(query.select.joins.len(), 1);
    assert_eq!(query.select.joins[0].table, "customers");
    assert!(
        query.select.group_by.is_some(),
        "GROUP BY clause must be present"
    );
}

/// LEFT JOIN is accepted (runtime may error, but parser must succeed).
#[test]
fn test_left_join_parse() {
    let sql = "SELECT * FROM a LEFT JOIN b ON a.id = b.aid";
    let query = Parser::parse(sql).expect("LEFT JOIN should parse");

    assert_eq!(query.select.joins.len(), 1);
    assert_eq!(query.select.joins[0].join_type, JoinType::Left);
    assert_eq!(query.select.joins[0].table, "b");
}

/// RIGHT JOIN is accepted.
#[test]
fn test_right_join_parse() {
    let sql = "SELECT * FROM a RIGHT JOIN b ON a.id = b.aid";
    let query = Parser::parse(sql).expect("RIGHT JOIN should parse");

    assert_eq!(query.select.joins[0].join_type, JoinType::Right);
}

/// FULL OUTER JOIN is accepted.
#[test]
fn test_full_outer_join_parse() {
    let sql = "SELECT * FROM a FULL OUTER JOIN b ON a.id = b.aid";
    let query = Parser::parse(sql).expect("FULL OUTER JOIN should parse");

    assert_eq!(query.select.joins[0].join_type, JoinType::Full);
}

/// JOIN with alias and qualified wildcard `d.*` in projection.
#[test]
fn test_join_with_alias_and_qualified_wildcard() {
    let sql = "SELECT d.* FROM docs AS d JOIN tags AS t ON d.id = t.doc_id";
    let query = Parser::parse(sql).expect("qualified wildcard with JOIN should parse");

    assert_eq!(query.select.joins.len(), 1);
    assert_eq!(query.select.joins[0].table, "tags");
    assert_eq!(query.select.joins[0].alias, Some("t".to_string()));

    // Projection should be a qualified wildcard for alias "d"
    match &query.select.columns {
        SelectColumns::QualifiedWildcard(alias) => assert_eq!(alias, "d"),
        other => panic!("Expected QualifiedWildcard(\"d\"), got {other:?}"),
    }
}

// ============================================================================
// Group 2 — ORDER BY edge cases
// ============================================================================

/// ORDER BY three fields with mixed directions.
#[test]
fn test_orderby_three_fields_mixed_directions() {
    let sql = "SELECT * FROM products ORDER BY category ASC, price DESC, created_at ASC";
    let query = Parser::parse(sql).expect("triple ORDER BY should parse");

    let order_by = query.select.order_by.expect("ORDER BY must be present");
    assert_eq!(order_by.len(), 3, "should have 3 ORDER BY items");

    assert!(matches!(&order_by[0].expr, OrderByExpr::Field(f) if f == "category"));
    assert!(!order_by[0].descending);
    assert!(matches!(&order_by[1].expr, OrderByExpr::Field(f) if f == "price"));
    assert!(order_by[1].descending);
    assert!(matches!(&order_by[2].expr, OrderByExpr::Field(f) if f == "created_at"));
    assert!(!order_by[2].descending);
}

/// ORDER BY without explicit direction defaults to ASC.
#[test]
fn test_orderby_default_direction_single() {
    let sql = "SELECT * FROM docs ORDER BY category";
    let query = Parser::parse(sql).expect("ORDER BY without direction should parse");

    let order_by = query.select.order_by.expect("ORDER BY must be present");
    assert_eq!(order_by.len(), 1);
    assert!(matches!(&order_by[0].expr, OrderByExpr::Field(f) if f == "category"));
    assert!(!order_by[0].descending, "default direction must be ASC");
}

/// ORDER BY aggregate expression: COUNT(*) DESC.
#[test]
fn test_orderby_aggregate_count() {
    let sql = "SELECT cat, COUNT(*) FROM p GROUP BY cat ORDER BY COUNT(*) DESC";
    let query = Parser::parse(sql).expect("ORDER BY COUNT(*) should parse");

    let order_by = query.select.order_by.expect("ORDER BY must be present");
    assert_eq!(order_by.len(), 1);
    assert!(order_by[0].descending);
    match &order_by[0].expr {
        OrderByExpr::Aggregate(agg) => {
            assert_eq!(agg.function_type, AggregateType::Count);
        }
        other => panic!("Expected Aggregate(Count), got {other:?}"),
    }
}

/// ORDER BY parenthesized arithmetic expression.
#[test]
fn test_orderby_parenthesized_arithmetic() {
    let sql = "SELECT * FROM docs ORDER BY (0.5 * vector_score + 0.5 * bm25_score) DESC";
    let query = Parser::parse(sql).expect("parenthesized arithmetic ORDER BY should parse");

    let order_by = query.select.order_by.expect("ORDER BY must be present");
    assert_eq!(order_by.len(), 1);
    assert!(order_by[0].descending);
    assert!(
        matches!(&order_by[0].expr, OrderByExpr::Arithmetic(_)),
        "should be Arithmetic variant, got {:?}",
        order_by[0].expr
    );
}

/// ORDER BY with division operator.
#[test]
fn test_orderby_division() {
    let sql = "SELECT * FROM docs ORDER BY score / 2 ASC";
    let query = Parser::parse(sql).expect("ORDER BY with division should parse");

    let order_by = query.select.order_by.expect("ORDER BY must be present");
    assert_eq!(order_by.len(), 1);
    assert!(!order_by[0].descending, "direction should be ASC");

    match &order_by[0].expr {
        OrderByExpr::Arithmetic(ArithmeticExpr::BinaryOp { op, .. }) => {
            assert_eq!(*op, ArithmeticOp::Div);
        }
        other => panic!("Expected Arithmetic(Div), got {other:?}"),
    }
}

/// ORDER BY similarity() * 1.0 + 0.0 — verifies negative-free arithmetic.
#[test]
fn test_orderby_similarity_arithmetic_identity() {
    let sql = "SELECT * FROM docs WHERE vector NEAR $v ORDER BY similarity() * 1.0 + 0.0 DESC";
    let query = Parser::parse(sql).expect("similarity arithmetic ORDER BY should parse");

    let order_by = query.select.order_by.expect("ORDER BY must be present");
    assert_eq!(order_by.len(), 1);
    assert!(order_by[0].descending);
    assert!(
        matches!(&order_by[0].expr, OrderByExpr::Arithmetic(_)),
        "should be Arithmetic, got {:?}",
        order_by[0].expr
    );
}

/// ORDER BY similarity() DESC followed by a field ASC.
#[test]
fn test_orderby_similarity_bare_then_field() {
    let sql = "SELECT * FROM docs WHERE vector NEAR $v \
               ORDER BY similarity() DESC, created_at ASC";
    let query = Parser::parse(sql).expect("similarity + field ORDER BY should parse");

    let order_by = query.select.order_by.expect("ORDER BY must be present");
    assert_eq!(order_by.len(), 2);

    assert_eq!(order_by[0].expr, OrderByExpr::SimilarityBare);
    assert!(order_by[0].descending);
    assert!(matches!(&order_by[1].expr, OrderByExpr::Field(f) if f == "created_at"));
    assert!(!order_by[1].descending);
}

// ============================================================================
// Group 3 — IN / NOT IN edge cases
// ============================================================================

/// IN with a single value.
#[test]
fn test_in_single_value() {
    let sql = "SELECT * FROM docs WHERE id IN (1)";
    let query = Parser::parse(sql).expect("IN with single value should parse");

    match &query.select.where_clause {
        Some(Condition::In(inc)) => {
            assert_eq!(inc.column, "id");
            assert_eq!(inc.values.len(), 1);
            assert!(!inc.negated);
            assert_eq!(inc.values[0], Value::Integer(1));
        }
        other => panic!("Expected In condition, got {other:?}"),
    }
}

/// IN with many values (12 items).
#[test]
fn test_in_many_values() {
    let sql = "SELECT * FROM docs WHERE id IN (1,2,3,4,5,6,7,8,9,10,11,12)";
    let query = Parser::parse(sql).expect("IN with 12 values should parse");

    match &query.select.where_clause {
        Some(Condition::In(inc)) => {
            assert_eq!(inc.values.len(), 12);
            assert_eq!(inc.values[0], Value::Integer(1));
            assert_eq!(inc.values[11], Value::Integer(12));
        }
        other => panic!("Expected In condition, got {other:?}"),
    }
}

/// IN with string values.
#[test]
fn test_in_string_values() {
    let sql = "SELECT * FROM docs WHERE category IN ('tech', 'science', 'ai', 'ml')";
    let query = Parser::parse(sql).expect("IN with strings should parse");

    match &query.select.where_clause {
        Some(Condition::In(inc)) => {
            assert_eq!(inc.column, "category");
            assert_eq!(inc.values.len(), 4);
            assert!(!inc.negated);
            assert_eq!(inc.values[0], Value::String("tech".to_string()));
            assert_eq!(inc.values[3], Value::String("ml".to_string()));
        }
        other => panic!("Expected In condition, got {other:?}"),
    }
}

/// IN with mixed types (int + string). Parser accepts; semantic analysis
/// may reject later.
#[test]
fn test_in_mixed_types() {
    let sql = "SELECT * FROM docs WHERE x IN (1, 'two', 3)";
    let query = Parser::parse(sql).expect("IN with mixed types should parse");

    match &query.select.where_clause {
        Some(Condition::In(inc)) => {
            assert_eq!(inc.values.len(), 3);
            assert_eq!(inc.values[0], Value::Integer(1));
            assert_eq!(inc.values[1], Value::String("two".to_string()));
            assert_eq!(inc.values[2], Value::Integer(3));
        }
        other => panic!("Expected In condition, got {other:?}"),
    }
}

/// NOT IN with string values.
#[test]
fn test_not_in_strings() {
    let sql = "SELECT * FROM docs WHERE status NOT IN ('deleted', 'archived', 'spam')";
    let query = Parser::parse(sql).expect("NOT IN with strings should parse");

    match &query.select.where_clause {
        Some(Condition::In(inc)) => {
            assert_eq!(inc.column, "status");
            assert!(inc.negated, "NOT IN must set negated=true");
            assert_eq!(inc.values.len(), 3);
            assert_eq!(inc.values[0], Value::String("deleted".to_string()));
        }
        other => panic!("Expected In(negated) condition, got {other:?}"),
    }
}

/// NOT IN with a single value.
#[test]
fn test_not_in_single_value() {
    let sql = "SELECT * FROM docs WHERE id NOT IN (42)";
    let query = Parser::parse(sql).expect("NOT IN with single value should parse");

    match &query.select.where_clause {
        Some(Condition::In(inc)) => {
            assert!(inc.negated);
            assert_eq!(inc.values.len(), 1);
            assert_eq!(inc.values[0], Value::Integer(42));
        }
        other => panic!("Expected In(negated) condition, got {other:?}"),
    }
}

/// IN with a nested (dotted) field name.
#[test]
fn test_in_nested_field() {
    let sql = "SELECT * FROM docs WHERE payload.category IN ('a', 'b')";
    let query = Parser::parse(sql).expect("IN with nested field should parse");

    match &query.select.where_clause {
        Some(Condition::In(inc)) => {
            assert_eq!(inc.column, "payload.category");
            assert_eq!(inc.values.len(), 2);
        }
        other => panic!("Expected In condition, got {other:?}"),
    }
}

// ============================================================================
// Group 4 — BETWEEN edge cases
// ============================================================================

/// BETWEEN with float bounds.
#[test]
fn test_between_floats() {
    let sql = "SELECT * FROM docs WHERE score BETWEEN 0.5 AND 1.0";
    let query = Parser::parse(sql).expect("BETWEEN with floats should parse");

    match &query.select.where_clause {
        Some(Condition::Between(btw)) => {
            assert_eq!(btw.column, "score");
            assert_eq!(btw.low, Value::Float(0.5));
            assert_eq!(btw.high, Value::Float(1.0));
        }
        other => panic!("Expected Between condition, got {other:?}"),
    }
}

/// BETWEEN with string bounds.
#[test]
fn test_between_strings() {
    let sql = "SELECT * FROM docs WHERE name BETWEEN 'A' AND 'M'";
    let query = Parser::parse(sql).expect("BETWEEN with strings should parse");

    match &query.select.where_clause {
        Some(Condition::Between(btw)) => {
            assert_eq!(btw.column, "name");
            assert_eq!(btw.low, Value::String("A".to_string()));
            assert_eq!(btw.high, Value::String("M".to_string()));
        }
        other => panic!("Expected Between condition, got {other:?}"),
    }
}

/// BETWEEN with identical low and high (degenerate range).
#[test]
fn test_between_same_values() {
    let sql = "SELECT * FROM docs WHERE id BETWEEN 5 AND 5";
    let query = Parser::parse(sql).expect("BETWEEN with equal bounds should parse");

    match &query.select.where_clause {
        Some(Condition::Between(btw)) => {
            assert_eq!(btw.low, Value::Integer(5));
            assert_eq!(btw.high, Value::Integer(5));
        }
        other => panic!("Expected Between condition, got {other:?}"),
    }
}

/// BETWEEN with a nested (dotted) field.
#[test]
fn test_between_nested_field() {
    let sql = "SELECT * FROM docs WHERE payload.price BETWEEN 10 AND 100";
    let query = Parser::parse(sql).expect("BETWEEN with nested field should parse");

    match &query.select.where_clause {
        Some(Condition::Between(btw)) => {
            assert_eq!(btw.column, "payload.price");
            assert_eq!(btw.low, Value::Integer(10));
            assert_eq!(btw.high, Value::Integer(100));
        }
        other => panic!("Expected Between condition, got {other:?}"),
    }
}

/// BETWEEN combined with AND (additional comparison).
#[test]
fn test_between_combined_with_and() {
    let sql = "SELECT * FROM docs WHERE price BETWEEN 10 AND 100 AND category = 'tech'";
    let query = Parser::parse(sql).expect("BETWEEN + AND condition should parse");

    // The top-level condition should be And(Between, Comparison)
    match &query.select.where_clause {
        Some(Condition::And(left, right)) => {
            assert!(
                matches!(left.as_ref(), Condition::Between(btw) if btw.column == "price"),
                "left side should be BETWEEN, got {left:?}"
            );
            assert!(
                matches!(right.as_ref(), Condition::Comparison(c) if c.column == "category"),
                "right side should be Comparison, got {right:?}"
            );
        }
        other => panic!("Expected And(Between, Comparison), got {other:?}"),
    }
}

// ============================================================================
// Group 5 — IS NULL / IS NOT NULL edge cases
// ============================================================================

/// IS NULL with a deeply nested (two-dot) field.
#[test]
fn test_is_null_nested_field() {
    let sql = "SELECT * FROM docs WHERE payload.metadata.author IS NULL";
    let query = Parser::parse(sql).expect("IS NULL with nested field should parse");

    match &query.select.where_clause {
        Some(Condition::IsNull(isnull)) => {
            assert_eq!(isnull.column, "payload.metadata.author");
            assert!(isnull.is_null, "IS NULL must set is_null=true");
        }
        other => panic!("Expected IsNull condition, got {other:?}"),
    }
}

/// IS NOT NULL combined with vector search via AND.
#[test]
fn test_is_not_null_with_vector_search() {
    let sql = "SELECT * FROM docs \
               WHERE vector NEAR $v AND category IS NOT NULL LIMIT 10";
    let query = Parser::parse(sql).expect("vector NEAR + IS NOT NULL should parse");

    assert_eq!(query.select.limit, Some(10));
    // Top-level should be And(VectorSearch, IsNull{is_null:false})
    match &query.select.where_clause {
        Some(Condition::And(left, right)) => {
            assert!(
                matches!(left.as_ref(), Condition::VectorSearch(_)),
                "left should be VectorSearch, got {left:?}"
            );
            match right.as_ref() {
                Condition::IsNull(isnull) => {
                    assert_eq!(isnull.column, "category");
                    assert!(!isnull.is_null, "IS NOT NULL must set is_null=false");
                }
                other => panic!("right should be IsNull, got {other:?}"),
            }
        }
        other => panic!("Expected And(VectorSearch, IsNull), got {other:?}"),
    }
}

/// Multiple null checks: a IS NULL AND b IS NOT NULL AND c IS NULL.
#[test]
fn test_multiple_null_checks() {
    let sql = "SELECT * FROM docs WHERE a IS NULL AND b IS NOT NULL AND c IS NULL";
    let query = Parser::parse(sql).expect("multiple null checks should parse");

    // Walk the AND tree to extract all IS NULL conditions.
    let mut null_checks = Vec::new();
    collect_null_conditions(query.select.where_clause.as_ref(), &mut null_checks);

    assert_eq!(null_checks.len(), 3, "should find 3 null checks");

    // Verify each by column name
    let a_check = null_checks.iter().find(|(col, _)| col == "a");
    let b_check = null_checks.iter().find(|(col, _)| col == "b");
    let c_check = null_checks.iter().find(|(col, _)| col == "c");

    assert!(matches!(a_check, Some((_, true))), "a should be IS NULL");
    assert!(
        matches!(b_check, Some((_, false))),
        "b should be IS NOT NULL"
    );
    assert!(matches!(c_check, Some((_, true))), "c should be IS NULL");
}

/// Helper: recursively collects `(column, is_null)` pairs from a condition
/// tree. Only used in test code.
fn collect_null_conditions(cond: Option<&Condition>, out: &mut Vec<(String, bool)>) {
    match cond {
        Some(Condition::IsNull(isnull)) => {
            out.push((isnull.column.clone(), isnull.is_null));
        }
        Some(Condition::And(left, right)) => {
            collect_null_conditions(Some(left), out);
            collect_null_conditions(Some(right), out);
        }
        _ => {}
    }
}

// ============================================================================
// Group 6 — Subquery edge cases
// ============================================================================

/// Subquery with COUNT aggregate.
#[test]
fn test_subquery_with_count() {
    let sql = "SELECT * FROM docs WHERE total > (SELECT COUNT(*) FROM other)";
    let query = Parser::parse(sql).expect("subquery with COUNT should parse");

    let (comp, sub) = extract_comparison_subquery(query.select.where_clause.as_ref());
    assert_eq!(comp.column, "total");
    assert_eq!(comp.operator, CompareOp::Gt);

    // The inner SELECT should target "other" with COUNT(*) aggregate
    assert_eq!(sub.select.from, "other");
}

/// Subquery with MIN aggregate.
#[test]
fn test_subquery_with_min() {
    let sql = "SELECT * FROM docs WHERE price < (SELECT MIN(price) FROM deals)";
    let query = Parser::parse(sql).expect("subquery with MIN should parse");

    let (comp, sub) = extract_comparison_subquery(query.select.where_clause.as_ref());
    assert_eq!(comp.column, "price");
    assert_eq!(comp.operator, CompareOp::Lt);
    assert_eq!(sub.select.from, "deals");
}

/// Subquery with LIMIT.
#[test]
fn test_subquery_with_limit() {
    let sql = "SELECT * FROM docs WHERE score > (SELECT AVG(score) FROM baseline LIMIT 1)";
    let query = Parser::parse(sql).expect("subquery with LIMIT should parse");

    let (comp, sub) = extract_comparison_subquery(query.select.where_clause.as_ref());
    assert_eq!(comp.column, "score");
    assert_eq!(comp.operator, CompareOp::Gt);
    assert_eq!(sub.select.from, "baseline");
    assert_eq!(sub.select.limit, Some(1));
}

/// Subquery with equality and inner WHERE clause.
#[test]
fn test_subquery_equality_with_inner_where() {
    let sql = "SELECT * FROM docs WHERE amount = \
               (SELECT MAX(amount) FROM orders WHERE status = 'paid')";
    let query = Parser::parse(sql).expect("subquery with inner WHERE should parse");

    let (comp, sub) = extract_comparison_subquery(query.select.where_clause.as_ref());
    assert_eq!(comp.column, "amount");
    assert_eq!(comp.operator, CompareOp::Eq);
    assert_eq!(sub.select.from, "orders");
    assert!(
        sub.select.where_clause.is_some(),
        "inner subquery must have its own WHERE clause"
    );
}

/// Helper: extracts a `(Comparison, Subquery)` pair from a top-level
/// `Condition::Comparison` whose value is `Value::Subquery`.
fn extract_comparison_subquery(
    cond: Option<&Condition>,
) -> (&crate::velesql::Comparison, &crate::velesql::Subquery) {
    match cond {
        Some(Condition::Comparison(comp)) => match &comp.value {
            Value::Subquery(sub) => (comp, sub),
            other => panic!("Expected Value::Subquery, got {other:?}"),
        },
        other => panic!("Expected Comparison with subquery, got {other:?}"),
    }
}

// ============================================================================
// Group 7 — Negative cases (must fail)
// ============================================================================

/// JOIN without ON or USING clause should fail.
#[test]
fn test_reject_join_without_condition() {
    let sql = "SELECT * FROM a JOIN b";
    assert!(
        Parser::parse(sql).is_err(),
        "JOIN without ON or USING should fail"
    );
}

/// ORDER BY with no field expression should fail.
#[test]
fn test_reject_orderby_without_field() {
    let sql = "SELECT * FROM docs ORDER BY";
    assert!(
        Parser::parse(sql).is_err(),
        "ORDER BY without expression should fail"
    );
}

/// IN with unclosed parenthesis should fail.
#[test]
fn test_reject_in_unclosed_paren() {
    let sql = "SELECT * FROM docs WHERE id IN (1, 2, 3";
    assert!(
        Parser::parse(sql).is_err(),
        "IN with unclosed paren should fail"
    );
}

/// BETWEEN without second bound should fail.
#[test]
fn test_reject_between_missing_upper_bound() {
    let sql = "SELECT * FROM docs WHERE id BETWEEN 1 AND";
    assert!(
        Parser::parse(sql).is_err(),
        "BETWEEN without upper bound should fail"
    );
}
