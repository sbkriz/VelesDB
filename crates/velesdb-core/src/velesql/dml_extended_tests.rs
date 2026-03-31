//! Extended DML parser tests for INSERT, UPDATE, and DELETE statements.
//!
//! Covers value types, multi-column inserts, conditional updates/deletes,
//! logical operators (AND/OR), IN, BETWEEN, NULL, and negative cases
//! (missing keywords, column count mismatches).

use crate::velesql::{CompareOp, Condition, DmlStatement, Parser, Value};

// ============================================================================
// INSERT — nominal cases
// ============================================================================

#[test]
fn test_insert_all_value_types() {
    let query = "INSERT INTO docs (a, b, c, d, e) VALUES (42, 2.75, 'hello', true, NULL)";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Insert(insert) = dml else {
        panic!("Expected Insert variant");
    };

    assert_eq!(insert.table, "docs");
    assert_eq!(insert.columns, vec!["a", "b", "c", "d", "e"]);
    assert_eq!(insert.rows.len(), 1);
    let row = &insert.rows[0];
    assert_eq!(row.len(), 5);
    assert_eq!(row[0], Value::Integer(42));
    assert_eq!(row[1], Value::Float(2.75));
    assert_eq!(row[2], Value::String("hello".to_string()));
    assert_eq!(row[3], Value::Boolean(true));
    assert_eq!(row[4], Value::Null);
}

#[test]
fn test_insert_with_vector_parameter() {
    let query = "INSERT INTO docs (id, vector, title) VALUES (1, $v, 'test')";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Insert(insert) = dml else {
        panic!("Expected Insert variant");
    };

    assert_eq!(insert.table, "docs");
    assert_eq!(insert.columns, vec!["id", "vector", "title"]);
    assert_eq!(insert.rows.len(), 1);
    assert_eq!(insert.rows[0][0], Value::Integer(1));
    assert_eq!(insert.rows[0][1], Value::Parameter("v".to_string()));
    assert_eq!(insert.rows[0][2], Value::String("test".to_string()));
}

#[test]
fn test_insert_many_columns() {
    let query = "INSERT INTO metrics (a, b, c, d, e, f) VALUES (1, 2, 3, 4, 5, 6)";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Insert(insert) = dml else {
        panic!("Expected Insert variant");
    };

    assert_eq!(insert.columns.len(), 6);
    assert_eq!(insert.rows.len(), 1);
    assert_eq!(insert.rows[0].len(), 6);
    for (i, val) in insert.rows[0].iter().enumerate() {
        let expected = i64::try_from(i + 1).expect("test index fits in i64");
        assert_eq!(*val, Value::Integer(expected));
    }
}

// ============================================================================
// INSERT — negative cases
// ============================================================================

#[test]
fn test_insert_missing_values_keyword_fails() {
    let result = Parser::parse("INSERT INTO docs (id, name) (1, 'x')");
    assert!(result.is_err(), "Missing VALUES keyword should fail");
}

#[test]
fn test_insert_column_count_mismatch_fails() {
    // The parser validates column/value count parity at parse time.
    let result = Parser::parse("INSERT INTO docs (id, name, extra) VALUES (1, 'x')");
    assert!(
        result.is_err(),
        "Column/value count mismatch should fail at parse time"
    );
}

#[test]
fn test_insert_missing_into_keyword_fails() {
    let result = Parser::parse("INSERT docs (id) VALUES (1)");
    assert!(result.is_err(), "Missing INTO keyword should fail");
}

// ============================================================================
// UPDATE — nominal cases
// ============================================================================

#[test]
fn test_update_single_field() {
    let query = "UPDATE docs SET status = 'active' WHERE id = 1";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Update(update) = dml else {
        panic!("Expected Update variant");
    };

    assert_eq!(update.table, "docs");
    assert_eq!(update.assignments.len(), 1);
    assert_eq!(update.assignments[0].column, "status");
    assert_eq!(
        update.assignments[0].value,
        Value::String("active".to_string())
    );

    let where_clause = update.where_clause.expect("Expected WHERE clause");
    let Condition::Comparison(cmp) = where_clause else {
        panic!("Expected Comparison condition");
    };
    assert_eq!(cmp.column, "id");
    assert_eq!(cmp.operator, CompareOp::Eq);
    assert_eq!(cmp.value, Value::Integer(1));
}

#[test]
fn test_update_multiple_fields() {
    let query = "UPDATE docs SET a = 1, b = 'x', c = true WHERE id = 5";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Update(update) = dml else {
        panic!("Expected Update variant");
    };

    assert_eq!(update.table, "docs");
    assert_eq!(update.assignments.len(), 3);
    assert_eq!(update.assignments[0].column, "a");
    assert_eq!(update.assignments[0].value, Value::Integer(1));
    assert_eq!(update.assignments[1].column, "b");
    assert_eq!(update.assignments[1].value, Value::String("x".to_string()));
    assert_eq!(update.assignments[2].column, "c");
    assert_eq!(update.assignments[2].value, Value::Boolean(true));
}

#[test]
fn test_update_without_where_clause() {
    let query = "UPDATE docs SET active = false";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Update(update) = dml else {
        panic!("Expected Update variant");
    };

    assert_eq!(update.table, "docs");
    assert_eq!(update.assignments[0].column, "active");
    assert_eq!(update.assignments[0].value, Value::Boolean(false));
    assert!(update.where_clause.is_none());
}

#[test]
fn test_update_with_in_clause() {
    let query = "UPDATE docs SET status = 'archived' WHERE id IN (1, 2, 3)";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Update(update) = dml else {
        panic!("Expected Update variant");
    };

    assert_eq!(update.table, "docs");
    let where_clause = update.where_clause.expect("Expected WHERE clause");
    let Condition::In(in_cond) = where_clause else {
        panic!("Expected In condition, got: {where_clause:?}");
    };
    assert_eq!(in_cond.column, "id");
    assert_eq!(in_cond.values.len(), 3);
}

#[test]
fn test_update_with_null_value() {
    let query = "UPDATE docs SET deleted_at = NULL WHERE id = 1";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Update(update) = dml else {
        panic!("Expected Update variant");
    };

    assert_eq!(update.assignments[0].column, "deleted_at");
    assert_eq!(update.assignments[0].value, Value::Null);
}

#[test]
fn test_update_with_and_condition() {
    let query = "UPDATE docs SET verified = true WHERE status = 'pending' AND age > 30";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Update(update) = dml else {
        panic!("Expected Update variant");
    };

    let where_clause = update.where_clause.expect("Expected WHERE clause");
    let Condition::And(left, right) = where_clause else {
        panic!("Expected And condition, got: {where_clause:?}");
    };

    let Condition::Comparison(left_cmp) = *left else {
        panic!("Expected left Comparison");
    };
    assert_eq!(left_cmp.column, "status");
    assert_eq!(left_cmp.operator, CompareOp::Eq);

    let Condition::Comparison(right_cmp) = *right else {
        panic!("Expected right Comparison");
    };
    assert_eq!(right_cmp.column, "age");
    assert_eq!(right_cmp.operator, CompareOp::Gt);
    assert_eq!(right_cmp.value, Value::Integer(30));
}

// ============================================================================
// UPDATE — negative cases
// ============================================================================

#[test]
fn test_update_missing_set_keyword_fails() {
    let result = Parser::parse("UPDATE docs status = 'active' WHERE id = 1");
    assert!(result.is_err(), "Missing SET keyword should fail");
}

#[test]
fn test_update_missing_table_name_fails() {
    let result = Parser::parse("UPDATE SET status = 'active'");
    assert!(result.is_err(), "Missing table name should fail");
}

// ============================================================================
// DELETE — nominal cases
// ============================================================================

#[test]
fn test_delete_by_id() {
    let query = "DELETE FROM docs WHERE id = 42";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Delete(delete) = dml else {
        panic!("Expected Delete variant");
    };

    assert_eq!(delete.table, "docs");
    let Condition::Comparison(cmp) = &delete.where_clause else {
        panic!("Expected Comparison condition");
    };
    assert_eq!(cmp.column, "id");
    assert_eq!(cmp.operator, CompareOp::Eq);
    assert_eq!(cmp.value, Value::Integer(42));
}

#[test]
fn test_delete_with_in_clause() {
    let query = "DELETE FROM docs WHERE id IN (1, 2, 3)";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Delete(delete) = dml else {
        panic!("Expected Delete variant");
    };

    assert_eq!(delete.table, "docs");
    let Condition::In(in_cond) = &delete.where_clause else {
        panic!("Expected In condition");
    };
    assert_eq!(in_cond.column, "id");
    assert_eq!(in_cond.values.len(), 3);
}

#[test]
fn test_delete_with_string_comparison() {
    let query = "DELETE FROM logs WHERE level = 'debug'";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Delete(delete) = dml else {
        panic!("Expected Delete variant");
    };

    assert_eq!(delete.table, "logs");
    let Condition::Comparison(cmp) = &delete.where_clause else {
        panic!("Expected Comparison condition");
    };
    assert_eq!(cmp.column, "level");
    assert_eq!(cmp.value, Value::String("debug".to_string()));
}

#[test]
fn test_delete_with_and_condition() {
    let query = "DELETE FROM events WHERE status = 'expired' AND ts < 1000";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Delete(delete) = dml else {
        panic!("Expected Delete variant");
    };

    assert_eq!(delete.table, "events");
    let Condition::And(left, right) = &delete.where_clause else {
        panic!("Expected And condition");
    };

    let Condition::Comparison(left_cmp) = left.as_ref() else {
        panic!("Expected left Comparison");
    };
    assert_eq!(left_cmp.column, "status");
    assert_eq!(left_cmp.value, Value::String("expired".to_string()));

    let Condition::Comparison(right_cmp) = right.as_ref() else {
        panic!("Expected right Comparison");
    };
    assert_eq!(right_cmp.column, "ts");
    assert_eq!(right_cmp.operator, CompareOp::Lt);
    assert_eq!(right_cmp.value, Value::Integer(1000));
}

#[test]
fn test_delete_with_or_condition() {
    let query = "DELETE FROM events WHERE status = 'expired' OR status = 'cancelled'";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Delete(delete) = dml else {
        panic!("Expected Delete variant");
    };

    assert_eq!(delete.table, "events");
    let Condition::Or(left, right) = &delete.where_clause else {
        panic!("Expected Or condition, got: {:?}", delete.where_clause);
    };

    let Condition::Comparison(left_cmp) = left.as_ref() else {
        panic!("Expected left Comparison");
    };
    assert_eq!(left_cmp.value, Value::String("expired".to_string()));

    let Condition::Comparison(right_cmp) = right.as_ref() else {
        panic!("Expected right Comparison");
    };
    assert_eq!(right_cmp.value, Value::String("cancelled".to_string()));
}

#[test]
fn test_delete_with_between() {
    let query = "DELETE FROM logs WHERE ts BETWEEN 0 AND 1000";
    let result = Parser::parse(query);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

    let parsed = result.expect("already checked is_ok");
    let dml = parsed.dml.expect("Expected DML statement");
    let DmlStatement::Delete(delete) = dml else {
        panic!("Expected Delete variant");
    };

    assert_eq!(delete.table, "logs");
    let Condition::Between(between) = &delete.where_clause else {
        panic!("Expected Between condition");
    };
    assert_eq!(between.column, "ts");
    assert_eq!(between.low, Value::Integer(0));
    assert_eq!(between.high, Value::Integer(1000));
}

// ============================================================================
// DELETE — negative cases
// ============================================================================

#[test]
fn test_delete_without_where_clause_fails() {
    let result = Parser::parse("DELETE FROM docs");
    assert!(
        result.is_err(),
        "DELETE without WHERE should fail (mandatory to prevent accidental full deletion)"
    );
}

#[test]
fn test_delete_missing_from_keyword_fails() {
    let result = Parser::parse("DELETE docs WHERE id = 1");
    assert!(result.is_err(), "Missing FROM keyword should fail");
}
