//! Robustness regression tests for parser panic-prone paths.

use crate::velesql::Parser;

#[test]
fn parse_join_condition_handles_quoted_identifiers_with_dots() {
    let query = r#"SELECT * FROM users AS u JOIN orders AS o ON `tenant.users`.id = "order.items"."user.id""#;

    let parsed = Parser::parse(query);

    assert!(
        parsed.is_ok(),
        "expected quoted JOIN column references to parse"
    );
}
