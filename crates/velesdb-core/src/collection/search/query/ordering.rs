//! ORDER BY clause execution for VelesQL queries.
//!
//! Handles multi-column sorting with support for:
//! - Metadata field sorting (ASC/DESC)
//! - similarity() function sorting
//! - Arithmetic expression sorting (EPIC-042)
//! - Mixed type JSON value comparison with total ordering

use crate::collection::types::Collection;
use crate::error::Result;
use crate::point::SearchResult;
use crate::velesql::{ArithmeticExpr, ArithmeticOp};
use std::cmp::Ordering;

/// Compare two JSON values for sorting with total ordering.
///
/// Ordering priority (ascending): Null < Bool < Number < String < Array < Object
/// This ensures deterministic sorting even with mixed types.
#[must_use]
pub fn compare_json_values(
    a: Option<&serde_json::Value>,
    b: Option<&serde_json::Value>,
) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(va), Some(vb)) => {
            // BUG FIX: Define total ordering for mixed JSON types
            // Type priority: Null(0) < Bool(1) < Number(2) < String(3) < Array(4) < Object(5)
            let type_rank = |v: &serde_json::Value| -> u8 {
                match v {
                    serde_json::Value::Null => 0,
                    serde_json::Value::Bool(_) => 1,
                    serde_json::Value::Number(_) => 2,
                    serde_json::Value::String(_) => 3,
                    serde_json::Value::Array(_) => 4,
                    serde_json::Value::Object(_) => 5,
                }
            };

            let rank_a = type_rank(va);
            let rank_b = type_rank(vb);

            // First compare by type rank
            if rank_a != rank_b {
                return rank_a.cmp(&rank_b);
            }

            // Same type: compare values
            match (va, vb) {
                (serde_json::Value::Number(na), serde_json::Value::Number(nb)) => {
                    let fa = na.as_f64().unwrap_or(0.0);
                    let fb = nb.as_f64().unwrap_or(0.0);
                    fa.total_cmp(&fb) // Use total_cmp for NaN safety
                }
                (serde_json::Value::String(sa), serde_json::Value::String(sb)) => sa.cmp(sb),
                (serde_json::Value::Bool(ba), serde_json::Value::Bool(bb)) => ba.cmp(bb),
                // Null vs Null, Array vs Array, Object vs Object: treat as equal
                // (comparing array/object contents would be complex and rarely needed)
                _ => Ordering::Equal,
            }
        }
    }
}

impl Collection {
    /// Apply ORDER BY clause to results.
    ///
    /// Supports multiple ORDER BY columns with stable sorting.
    /// Each column is compared in order; ties are broken by subsequent columns.
    ///
    /// # Examples
    ///
    /// ```sql
    /// SELECT * FROM collection ORDER BY category ASC, priority DESC
    /// SELECT * FROM collection ORDER BY similarity() DESC, timestamp ASC
    /// ```
    pub(crate) fn apply_order_by(
        &self,
        results: &mut [SearchResult],
        order_by: &[crate::velesql::SelectOrderBy],
        params: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        if order_by.is_empty() {
            return Ok(());
        }

        let similarity_scores_map = self.precompute_similarity_scores(results, order_by, params)?;
        let higher_is_better = self.config.read().metric.higher_is_better();

        let mut indices: Vec<usize> = (0..results.len()).collect();
        indices.sort_by(|&i, &j| {
            Self::compare_by_order_columns(
                i,
                j,
                results,
                order_by,
                &similarity_scores_map,
                higher_is_better,
            )
        });

        let sorted_results: Vec<SearchResult> =
            indices.iter().map(|&i| results[i].clone()).collect();
        results.clone_from_slice(&sorted_results);

        // Write back the score from the first similarity column (any position).
        let first_sim_idx = order_by
            .iter()
            .enumerate()
            .find(|(_, ob)| {
                matches!(
                    ob.expr,
                    crate::velesql::OrderByExpr::Similarity(_)
                        | crate::velesql::OrderByExpr::SimilarityBare
                )
            })
            .map(|(idx, _)| idx);
        if let Some(sim_idx) = first_sim_idx {
            if let Some(scores) = similarity_scores_map.get(&sim_idx) {
                for (i, result) in results.iter_mut().enumerate() {
                    result.score = scores[indices[i]];
                }
            }
        }

        Ok(())
    }

    /// Pre-computes similarity scores for all ORDER BY similarity() columns.
    fn precompute_similarity_scores(
        &self,
        results: &[SearchResult],
        order_by: &[crate::velesql::SelectOrderBy],
        params: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<std::collections::HashMap<usize, Vec<f32>>> {
        use crate::velesql::OrderByExpr;
        let mut map = std::collections::HashMap::new();
        for (idx, ob) in order_by.iter().enumerate() {
            match &ob.expr {
                OrderByExpr::Similarity(sim) => {
                    let order_vec = Self::resolve_vector(&sim.vector, params)?;
                    let scores: Vec<f32> = results
                        .iter()
                        .map(|r| self.compute_metric_score(&r.point.vector, &order_vec))
                        .collect();
                    map.insert(idx, scores);
                }
                OrderByExpr::SimilarityBare => {
                    // Zero-arg similarity(): use existing search scores (no recompute).
                    let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
                    map.insert(idx, scores);
                }
                OrderByExpr::Field(_) | OrderByExpr::Aggregate(_) | OrderByExpr::Arithmetic(_) => {}
            }
        }
        Ok(map)
    }

    /// Compares two result indices across all ORDER BY columns.
    fn compare_by_order_columns(
        i: usize,
        j: usize,
        results: &[SearchResult],
        order_by: &[crate::velesql::SelectOrderBy],
        similarity_scores: &std::collections::HashMap<usize, Vec<f32>>,
        higher_is_better: bool,
    ) -> Ordering {
        use crate::velesql::OrderByExpr;
        for (idx, ob) in order_by.iter().enumerate() {
            let cmp = match &ob.expr {
                OrderByExpr::Similarity(_) | OrderByExpr::SimilarityBare => similarity_scores
                    .get(&idx)
                    .map_or(Ordering::Equal, |scores| scores[i].total_cmp(&scores[j])),
                OrderByExpr::Field(field_name) => {
                    let val_i = results[i]
                        .point
                        .payload
                        .as_ref()
                        .and_then(|p| p.get(field_name));
                    let val_j = results[j]
                        .point
                        .payload
                        .as_ref()
                        .and_then(|p| p.get(field_name));
                    compare_json_values(val_i, val_j)
                }
                OrderByExpr::Aggregate(_) => Ordering::Equal,
                OrderByExpr::Arithmetic(expr) => {
                    let ctx_i =
                        ScoreContext::new(results[i].score, results[i].point.payload.as_ref());
                    let ctx_j =
                        ScoreContext::new(results[j].score, results[j].point.payload.as_ref());
                    let val_i = evaluate_arithmetic(expr, &ctx_i);
                    let val_j = evaluate_arithmetic(expr, &ctx_j);
                    val_i.total_cmp(&val_j)
                }
            };

            let is_similarity = matches!(
                &ob.expr,
                OrderByExpr::Similarity(_) | OrderByExpr::SimilarityBare
            );
            let directed_cmp =
                Self::apply_sort_direction(cmp, ob.descending, is_similarity, higher_is_better);
            if directed_cmp != Ordering::Equal {
                return directed_cmp;
            }
        }
        Ordering::Equal
    }

    /// Applies ASC/DESC direction, accounting for distance metric inversion.
    fn apply_sort_direction(
        cmp: Ordering,
        descending: bool,
        is_similarity: bool,
        higher_is_better: bool,
    ) -> Ordering {
        if descending {
            if is_similarity && !higher_is_better {
                cmp
            } else {
                cmp.reverse()
            }
        } else if is_similarity && !higher_is_better {
            cmp.reverse()
        } else {
            cmp
        }
    }
}

/// Context for evaluating arithmetic ORDER BY expressions (EPIC-042).
///
/// Holds the pre-computed search score and optional payload for variable resolution.
pub(crate) struct ScoreContext<'a> {
    /// Pre-computed search score (vector similarity or fused score).
    search_score: f32,
    /// Payload fields for variable resolution.
    payload: Option<&'a serde_json::Value>,
}

impl<'a> ScoreContext<'a> {
    /// Creates a new score context from a search result.
    pub(crate) fn new(search_score: f32, payload: Option<&'a serde_json::Value>) -> Self {
        Self {
            search_score,
            payload,
        }
    }

    /// Resolves a variable name to a numeric value.
    ///
    /// Built-in names (`vector_score`, `fused_score`, `similarity`) map to the
    /// pre-computed search score. Other names are looked up in the payload.
    fn resolve_variable(&self, name: &str) -> f32 {
        match name {
            "vector_score" | "fused_score" | "similarity" => self.search_score,
            _ => self
                .payload
                .and_then(|p| p.get(name))
                .and_then(serde_json::Value::as_f64)
                .map_or(0.0, |v| {
                    #[allow(clippy::cast_possible_truncation)]
                    // Reason: payload values are user-defined scores; f64→f32 precision loss is acceptable.
                    {
                        v as f32
                    }
                }),
        }
    }
}

/// Evaluates an arithmetic expression against a score context (EPIC-042).
///
/// Division by zero returns `0.0` (safe default for sorting).
pub(crate) fn evaluate_arithmetic(expr: &ArithmeticExpr, ctx: &ScoreContext<'_>) -> f32 {
    match expr {
        ArithmeticExpr::Literal(v) => {
            #[allow(clippy::cast_possible_truncation)]
            // Reason: arithmetic literals are user-defined weights; f64→f32 precision loss is acceptable.
            {
                *v as f32
            }
        }
        ArithmeticExpr::Variable(name) => ctx.resolve_variable(name),
        ArithmeticExpr::Similarity(_) => ctx.search_score,
        ArithmeticExpr::BinaryOp { left, op, right } => {
            let l = evaluate_arithmetic(left, ctx);
            let r = evaluate_arithmetic(right, ctx);
            match op {
                ArithmeticOp::Add => l + r,
                ArithmeticOp::Sub => l - r,
                ArithmeticOp::Mul => l * r,
                ArithmeticOp::Div => {
                    if r.abs() > f32::EPSILON {
                        l / r
                    } else {
                        0.0
                    }
                }
            }
        }
    }
}
