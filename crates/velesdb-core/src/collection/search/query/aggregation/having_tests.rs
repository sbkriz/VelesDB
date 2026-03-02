#![cfg(all(test, feature = "persistence"))]

use super::*;
use crate::velesql::{AggregateArg, AggregateFunction, AggregateType, OrderByExpr, SelectOrderBy};

#[test]
fn test_sort_aggregation_results_order_by_count_desc_sorts_rows() {
    // ARRANGE
    let mut rows = vec![
        serde_json::json!({"category": "science", "count": 2}),
        serde_json::json!({"category": "tech", "count": 5}),
        serde_json::json!({"category": "history", "count": 3}),
    ];
    let order_by = vec![SelectOrderBy {
        expr: OrderByExpr::Aggregate(AggregateFunction {
            function_type: AggregateType::Count,
            argument: AggregateArg::Wildcard,
            alias: None,
        }),
        descending: true,
    }];

    // ACT
    Collection::sort_aggregation_results(&mut rows, &order_by);

    // ASSERT
    let ordered_categories: Vec<&str> = rows
        .iter()
        .map(|row| {
            row.get("category")
                .and_then(serde_json::Value::as_str)
                .expect("category should be a string")
        })
        .collect();
    assert_eq!(ordered_categories, vec!["tech", "history", "science"]);
}
