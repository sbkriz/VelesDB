//! WHERE clause evaluation for MATCH queries (EPIC-045 US-002).
//!
//! Handles condition evaluation, parameter resolution, and comparison operations.

use crate::collection::types::Collection;
use crate::error::{Error, Result};
use crate::storage::{PayloadStorage, VectorStorage};
use std::collections::HashMap;

/// Applies an ordering comparison operator to an `Ord` pair.
fn apply_ord_op<T: PartialOrd>(op: crate::velesql::CompareOp, a: &T, b: &T) -> bool {
    use crate::velesql::CompareOp;
    match op {
        CompareOp::Eq => a == b,
        CompareOp::NotEq => a != b,
        CompareOp::Lt => a < b,
        CompareOp::Gt => a > b,
        CompareOp::Lte => a <= b,
        CompareOp::Gte => a >= b,
    }
}

/// Compares two floats with epsilon tolerance for equality.
fn compare_floats(op: crate::velesql::CompareOp, actual: f64, expected: f64) -> bool {
    use crate::velesql::CompareOp;
    match op {
        CompareOp::Eq => (actual - expected).abs() < 0.001,
        CompareOp::NotEq => (actual - expected).abs() >= 0.001,
        _ => apply_ord_op(op, &actual, &expected),
    }
}

/// Compares a similarity score against a threshold, inverting for distance metrics.
fn compare_score(
    op: crate::velesql::CompareOp,
    score: f32,
    threshold: f32,
    higher_is_better: bool,
) -> bool {
    use crate::velesql::CompareOp;
    match op {
        CompareOp::Eq => (score - threshold).abs() < f32::EPSILON,
        CompareOp::NotEq => (score - threshold).abs() >= f32::EPSILON,
        CompareOp::Gt | CompareOp::Gte | CompareOp::Lt | CompareOp::Lte => {
            if higher_is_better {
                apply_ord_op(op, &score, &threshold)
            } else {
                // Invert: "similarity > X" ↔ "distance < X"
                apply_ord_op(invert_order(op), &score, &threshold)
            }
        }
    }
}

/// Flips a relational operator (Gt↔Lt, Gte↔Lte). Eq/NotEq pass through.
const fn invert_order(op: crate::velesql::CompareOp) -> crate::velesql::CompareOp {
    use crate::velesql::CompareOp;
    match op {
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Gte => CompareOp::Lte,
        CompareOp::Lte => CompareOp::Gte,
        other => other,
    }
}

/// Resolves a query vector from a `VectorExpr`, looking up parameters as needed.
fn resolve_query_vector(
    vector: &crate::velesql::VectorExpr,
    params: &HashMap<String, serde_json::Value>,
) -> Result<Vec<f32>> {
    use crate::velesql::VectorExpr;

    match vector {
        VectorExpr::Literal(v) => Ok(v.clone()),
        VectorExpr::Parameter(name) => {
            let param_value = params
                .get(name)
                .ok_or_else(|| Error::Config(format!("Missing vector parameter: ${name}")))?;

            match param_value {
                serde_json::Value::Array(arr) => Ok(arr
                    .iter()
                    .filter_map(|v| {
                        v.as_f64().map(|f| {
                            #[allow(clippy::cast_possible_truncation)]
                            let r = f as f32;
                            r
                        })
                    })
                    .collect()),
                _ => Err(Error::Config(format!(
                    "Parameter ${name} must be a vector array"
                ))),
            }
        }
    }
}

impl Collection {
    /// Evaluates a WHERE condition against a node's payload (EPIC-045 US-002).
    ///
    /// Supports basic comparisons: =, <>, <, >, <=, >=
    /// Parameters are resolved from the `params` map.
    pub(crate) fn evaluate_where_condition(
        &self,
        node_id: u64,
        bindings: Option<&HashMap<String, u64>>,
        condition: &crate::velesql::Condition,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<bool> {
        use crate::velesql::Condition;

        match condition {
            Condition::Comparison(cmp) => {
                self.evaluate_comparison_condition(node_id, bindings, cmp, params)
            }
            Condition::And(left, right) => Ok(self
                .evaluate_where_condition(node_id, bindings, left, params)?
                && self.evaluate_where_condition(node_id, bindings, right, params)?),
            Condition::Or(left, right) => Ok(self
                .evaluate_where_condition(node_id, bindings, left, params)?
                || self.evaluate_where_condition(node_id, bindings, right, params)?),
            Condition::Not(inner) => {
                Ok(!self.evaluate_where_condition(node_id, bindings, inner, params)?)
            }
            Condition::Group(inner) => {
                self.evaluate_where_condition(node_id, bindings, inner, params)
            }
            Condition::Similarity(sim) => self.evaluate_similarity_condition(node_id, sim, params),
            // Other condition types (VectorSearch, VectorFusedSearch, etc.)
            // handled separately in `execute_match_with_similarity`.
            _ => Ok(true),
        }
    }

    /// Evaluates a single comparison condition against a node's payload.
    fn evaluate_comparison_condition(
        &self,
        node_id: u64,
        bindings: Option<&HashMap<String, u64>>,
        cmp: &crate::velesql::Comparison,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<bool> {
        let target_id = resolve_target_id(&cmp.column, bindings, node_id);

        let payload_storage = self.payload_storage.read();
        let Some(target_payload) = payload_storage.retrieve(target_id).ok().flatten() else {
            return Ok(false);
        };

        let column_path = strip_alias(&cmp.column, bindings);
        let Some(actual) = Self::json_get_path(&target_payload, column_path) else {
            return Ok(false);
        };

        let resolved_value = Self::resolve_where_param(&cmp.value, params)?;
        Self::evaluate_comparison(cmp.operator, actual, &resolved_value)
    }

    /// Evaluates a similarity condition against a node's vector (EPIC-052 US-007).
    fn evaluate_similarity_condition(
        &self,
        node_id: u64,
        sim: &crate::velesql::SimilarityCondition,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<bool> {
        let query_vector = resolve_query_vector(&sim.vector, params)?;
        if query_vector.is_empty() {
            return Ok(false);
        }

        let vector_storage = self.vector_storage.read();
        let Some(node_vector) = vector_storage.retrieve(node_id)? else {
            return Ok(false);
        };

        if node_vector.len() != query_vector.len() {
            return Ok(false);
        }

        let config = self.config.read();
        let metric = config.metric;
        let higher_is_better = metric.higher_is_better();
        drop(config);

        let score = metric.calculate(&node_vector, &query_vector);

        #[allow(clippy::cast_possible_truncation)]
        let threshold = sim.threshold as f32;

        Ok(compare_score(
            sim.operator,
            score,
            threshold,
            higher_is_better,
        ))
    }

    /// Resolves a Value for WHERE clause, substituting parameters from the params map.
    ///
    /// If the value is a Parameter, looks it up in params and converts to appropriate Value type.
    /// Otherwise, returns the value unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if a required parameter is missing.
    pub(crate) fn resolve_where_param(
        value: &crate::velesql::Value,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<crate::velesql::Value> {
        use crate::velesql::Value;

        match value {
            Value::Parameter(name) => {
                let param_value = params
                    .get(name)
                    .ok_or_else(|| Error::Config(format!("Missing parameter: ${name}")))?;

                Ok(match param_value {
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            Value::Integer(i)
                        } else if let Some(f) = n.as_f64() {
                            Value::Float(f)
                        } else {
                            Value::Null
                        }
                    }
                    serde_json::Value::String(s) => Value::String(s.clone()),
                    serde_json::Value::Bool(b) => Value::Boolean(*b),
                    serde_json::Value::Null => Value::Null,
                    _ => {
                        return Err(Error::Config(format!(
                            "Unsupported parameter type for ${name}: {param_value:?}",
                        )));
                    }
                })
            }
            other => Ok(other.clone()),
        }
    }

    /// Evaluates a comparison operation.
    #[allow(clippy::unnecessary_wraps)] // Consistent with other evaluation methods
    pub(crate) fn evaluate_comparison(
        operator: crate::velesql::CompareOp,
        actual: &serde_json::Value,
        expected: &crate::velesql::Value,
    ) -> Result<bool> {
        use crate::velesql::Value;

        Ok(match (actual, expected) {
            (serde_json::Value::Number(n), Value::Integer(i)) => n
                .as_i64()
                .is_some_and(|actual_i| apply_ord_op(operator, &actual_i, i)),
            (serde_json::Value::Number(n), Value::Float(f)) => n
                .as_f64()
                .is_some_and(|actual_f| compare_floats(operator, actual_f, *f)),
            (serde_json::Value::String(s), Value::String(expected_s)) => {
                apply_ord_op(operator, &s.as_str(), &expected_s.as_str())
            }
            (serde_json::Value::Bool(b), Value::Boolean(expected_b)) => {
                matches!(
                    (operator, b == expected_b),
                    (crate::velesql::CompareOp::Eq, true)
                        | (crate::velesql::CompareOp::NotEq, false)
                )
            }
            (serde_json::Value::Null, Value::Null) => {
                matches!(operator, crate::velesql::CompareOp::Eq)
            }
            (_, Value::Null) => matches!(operator, crate::velesql::CompareOp::NotEq),
            _ => false,
        })
    }

    fn json_get_path<'a>(root: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
        if path.is_empty() {
            return Some(root);
        }

        let mut current = root;
        for part in path.split('.') {
            current = current.get(part)?;
        }
        Some(current)
    }
}

/// Resolves the target node ID from a column reference, using alias bindings if present.
fn resolve_target_id(
    column: &str,
    bindings: Option<&HashMap<String, u64>>,
    default_id: u64,
) -> u64 {
    column
        .split_once('.')
        .and_then(|(alias, _)| bindings?.get(alias).copied())
        .unwrap_or(default_id)
}

/// Strips the alias prefix from a column path when the alias exists in bindings.
fn strip_alias<'a>(column: &'a str, bindings: Option<&HashMap<String, u64>>) -> &'a str {
    match column.split_once('.') {
        Some((alias, rest)) if bindings.and_then(|b| b.get(alias)).is_some() => rest,
        _ => column,
    }
}
