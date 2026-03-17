//! Aggregation query execution for VelesQL (EPIC-017 US-002, US-003, US-006).
//!
//! Implements streaming aggregation with O(1) memory complexity.
//! Supports GROUP BY for grouped aggregations (US-003).
//! Supports HAVING for filtering groups (US-006).
//! Supports parallel aggregation with rayon (EPIC-018 US-001).

// SAFETY: Numeric casts in aggregation are intentional:
// - All casts are for computing aggregate statistics (sum, avg, count)
// - f64/u64 casts for maintaining precision in intermediate calculations
// - i64->usize for group limits: limits bounded by MAX_GROUPS (1M)
// - Values bounded by result set size and field cardinality
// - Precision loss acceptable for aggregation results
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

mod grouped;
mod having;
#[cfg(test)]
mod having_tests;

use super::where_eval::GraphMatchEvalCache;
use crate::collection::types::Collection;
use crate::error::Result;
use crate::storage::{PayloadStorage, VectorStorage};
use crate::velesql::{
    AggregateArg, AggregateFunction, AggregateType, Aggregator, Query, SelectColumns,
};
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Group key for GROUP BY operations with pre-computed hash.
/// Avoids JSON serialization overhead by using direct value hashing.
#[derive(Clone)]
pub(crate) struct GroupKey {
    /// Original values for result construction
    pub(crate) values: Vec<serde_json::Value>,
    /// Pre-computed hash for fast HashMap lookup
    hash: u64,
}

impl GroupKey {
    pub(crate) fn new(values: Vec<serde_json::Value>) -> Self {
        let hash = Self::compute_hash(&values);
        Self { values, hash }
    }

    fn compute_hash(values: &[serde_json::Value]) -> u64 {
        let mut hasher = FxHasher::default();
        for v in values {
            Self::hash_value(v, &mut hasher);
        }
        hasher.finish()
    }

    fn hash_value(value: &serde_json::Value, hasher: &mut FxHasher) {
        match value {
            serde_json::Value::Null => 0u8.hash(hasher),
            serde_json::Value::Bool(b) => {
                1u8.hash(hasher);
                b.hash(hasher);
            }
            serde_json::Value::Number(n) => {
                2u8.hash(hasher);
                // Use bits for consistent hashing of floats
                if let Some(f) = n.as_f64() {
                    f.to_bits().hash(hasher);
                }
            }
            serde_json::Value::String(s) => {
                3u8.hash(hasher);
                s.hash(hasher);
            }
            _ => {
                // Arrays and objects: fallback to string representation
                4u8.hash(hasher);
                value.to_string().hash(hasher);
            }
        }
    }
}

impl Hash for GroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl PartialEq for GroupKey {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: different hash means definitely different
        self.hash == other.hash && self.values == other.values
    }
}

impl Eq for GroupKey {}

/// Threshold for switching to parallel aggregation.
/// Below this, sequential is faster due to overhead.
const PARALLEL_THRESHOLD: usize = 10_000;

/// Chunk size for parallel processing.
const CHUNK_SIZE: usize = 1000;

impl Collection {
    /// Execute an aggregation query and return results as JSON.
    ///
    /// Supports COUNT(*), COUNT(column), SUM, AVG, MIN, MAX.
    /// Uses streaming aggregation - O(1) memory, single pass over data.
    ///
    /// # Arguments
    ///
    /// * `query` - Parsed VelesQL query with aggregation functions
    /// * `params` - Query parameters for placeholders
    ///
    /// # Returns
    ///
    /// JSON object with aggregation results, e.g.:
    /// ```json
    /// {"count": 100, "sum_price": 5000.0, "avg_rating": 4.5}
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when SELECT does not contain aggregations, when HAVING is
    /// used without GROUP BY, or when underlying scan/filter/aggregation operations fail.
    pub fn execute_aggregate(
        &self,
        query: &Query,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let stmt = &query.select;

        let aggregations: &[AggregateFunction] = match &stmt.columns {
            SelectColumns::Aggregations(aggs) => aggs,
            SelectColumns::Mixed { aggregations, .. } => aggregations,
            _ => {
                return Err(crate::error::Error::Config(
                    "execute_aggregate requires aggregation functions in SELECT".to_string(),
                ))
            }
        };

        if let Some(ref group_by) = stmt.group_by {
            return self.execute_grouped_aggregate(
                query, aggregations, &group_by.columns, stmt.having.as_ref(), params,
            );
        }

        if stmt.having.is_some() {
            return Err(crate::error::Error::Config(
                "HAVING clause requires GROUP BY clause".to_string(),
            ));
        }

        let agg_result = self.run_ungrouped_aggregation(stmt, aggregations, params)?;
        Ok(Self::build_aggregate_result(aggregations, &agg_result))
    }

    /// Runs the ungrouped aggregation scan (parallel or sequential).
    fn run_ungrouped_aggregation(
        &self,
        stmt: &crate::velesql::SelectStatement,
        aggregations: &[AggregateFunction],
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<crate::velesql::AggregateResult> {
        let where_clause = stmt.where_clause.as_ref();
        let use_runtime_where_eval = where_clause.is_some_and(|cond| {
            Self::condition_contains_graph_match(cond) || Self::condition_requires_vector_eval(cond)
        });

        let filter = if use_runtime_where_eval {
            None
        } else {
            where_clause.as_ref().map(|cond| {
                let resolved = Self::resolve_condition_params(cond, params);
                crate::filter::Filter::new(crate::filter::Condition::from(resolved))
            })
        };

        let columns_to_aggregate: std::collections::HashSet<&str> = aggregations
            .iter()
            .filter_map(|agg| match &agg.argument {
                AggregateArg::Column(col) => Some(col.as_str()),
                AggregateArg::Wildcard => None,
            })
            .collect();

        let has_count_star = aggregations
            .iter()
            .any(|agg| matches!(agg.argument, AggregateArg::Wildcard));

        let payload_storage = self.payload_storage.read();
        let vector_storage = self.vector_storage.read();
        let ids: Vec<u64> = vector_storage.ids();
        let total_count = ids.len();

        if total_count >= PARALLEL_THRESHOLD && !use_runtime_where_eval {
            let payloads: Vec<Option<serde_json::Value>> = ids
                .iter()
                .map(|&id| payload_storage.retrieve(id).ok().flatten())
                .collect();
            drop(payload_storage);
            drop(vector_storage);

            Ok(Self::aggregate_parallel(
                &payloads, filter.as_ref(), &columns_to_aggregate, has_count_star,
            ))
        } else {
            self.aggregate_sequential(
                &ids, &*payload_storage, &*vector_storage, stmt, params,
                filter.as_ref(), &columns_to_aggregate, has_count_star,
                use_runtime_where_eval,
            )
        }
    }

    /// Parallel aggregation on pre-fetched payloads.
    fn aggregate_parallel(
        payloads: &[Option<serde_json::Value>],
        filter: Option<&crate::filter::Filter>,
        columns_to_aggregate: &std::collections::HashSet<&str>,
        has_count_star: bool,
    ) -> crate::velesql::AggregateResult {
        let columns_vec: Vec<String> = columns_to_aggregate
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        let partial_aggregators: Vec<Aggregator> = payloads
            .par_chunks(CHUNK_SIZE)
            .map(|chunk| {
                let mut chunk_agg = Aggregator::new();
                for payload in chunk {
                    if let Some(f) = filter {
                        let matches = match payload {
                            Some(ref p) => f.matches(p),
                            None => f.matches(&serde_json::Value::Null),
                        };
                        if !matches {
                            continue;
                        }
                    }
                    if has_count_star {
                        chunk_agg.process_count();
                    }
                    if let Some(ref p) = payload {
                        for col in &columns_vec {
                            if let Some(value) = Self::get_nested_value(p, col) {
                                chunk_agg.process_value(col, value);
                            }
                        }
                    }
                }
                chunk_agg
            })
            .collect();

        let mut final_agg = Aggregator::new();
        for partial in partial_aggregators {
            final_agg.merge(partial);
        }
        final_agg.finalize()
    }

    /// Sequential aggregation with optional runtime WHERE evaluation.
    #[allow(clippy::too_many_arguments)]
    fn aggregate_sequential(
        &self,
        ids: &[u64],
        payload_storage: &dyn PayloadStorage,
        vector_storage: &dyn VectorStorage,
        stmt: &crate::velesql::SelectStatement,
        params: &HashMap<String, serde_json::Value>,
        filter: Option<&crate::filter::Filter>,
        columns_to_aggregate: &std::collections::HashSet<&str>,
        has_count_star: bool,
        use_runtime_where_eval: bool,
    ) -> Result<crate::velesql::AggregateResult> {
        let needs_vector_eval = stmt.where_clause.as_ref().is_some_and(Self::condition_requires_vector_eval);
        let mut aggregator = Aggregator::new();
        let mut graph_cache = GraphMatchEvalCache::default();

        for &id in ids {
            let payload = payload_storage.retrieve(id).ok().flatten();

            if use_runtime_where_eval {
                let vector = if needs_vector_eval {
                    vector_storage.retrieve(id).ok().flatten()
                } else {
                    None
                };
                if let Some(cond) = stmt.where_clause.as_ref() {
                    if !self.evaluate_where_condition_for_record(
                        cond, id, payload.as_ref(), vector.as_deref(),
                        params, &stmt.from_alias, &mut graph_cache,
                    )? {
                        continue;
                    }
                }
            } else if let Some(f) = filter {
                let matches = match payload {
                    Some(ref p) => f.matches(p),
                    None => f.matches(&serde_json::Value::Null),
                };
                if !matches {
                    continue;
                }
            }

            if has_count_star {
                aggregator.process_count();
            }
            if let Some(ref p) = payload {
                for col in columns_to_aggregate {
                    if let Some(value) = Self::get_nested_value(p, col) {
                        aggregator.process_value(col, value);
                    }
                }
            }
        }
        Ok(aggregator.finalize())
    }

    /// Builds the JSON result object from aggregation results.
    fn build_aggregate_result(
        aggregations: &[AggregateFunction],
        agg_result: &crate::velesql::AggregateResult,
    ) -> serde_json::Value {
        let mut result = serde_json::Map::new();

        for agg in aggregations {
            let key = Self::aggregation_result_key_from_fn(agg);
            let value = Self::aggregation_result_value_from_fn(agg, agg_result);
            result.insert(key, value);
        }

        serde_json::Value::Object(result)
    }

    /// Computes the result key for an aggregation function (used by ungrouped path).
    fn aggregation_result_key_from_fn(agg: &AggregateFunction) -> String {
        if let Some(ref alias) = agg.alias {
            alias.clone()
        } else {
            match &agg.argument {
                AggregateArg::Wildcard => "count".to_string(),
                AggregateArg::Column(col) => {
                    let prefix = match agg.function_type {
                        AggregateType::Count => "count",
                        AggregateType::Sum => "sum",
                        AggregateType::Avg => "avg",
                        AggregateType::Min => "min",
                        AggregateType::Max => "max",
                    };
                    format!("{prefix}_{col}")
                }
            }
        }
    }

    /// Computes the result value for an aggregation function (used by ungrouped path).
    fn aggregation_result_value_from_fn(
        agg: &AggregateFunction,
        agg_result: &crate::velesql::AggregateResult,
    ) -> serde_json::Value {
        match (&agg.function_type, &agg.argument) {
            (AggregateType::Count, AggregateArg::Wildcard) => {
                serde_json::json!(agg_result.count)
            }
            (AggregateType::Count, AggregateArg::Column(col)) => {
                let count = agg_result.counts.get(col.as_str()).copied().unwrap_or(0);
                serde_json::json!(count)
            }
            (AggregateType::Sum, AggregateArg::Column(col)) => agg_result
                .sums.get(col.as_str())
                .map_or(serde_json::Value::Null, |v| serde_json::json!(v)),
            (AggregateType::Avg, AggregateArg::Column(col)) => agg_result
                .avgs.get(col.as_str())
                .map_or(serde_json::Value::Null, |v| serde_json::json!(v)),
            (AggregateType::Min, AggregateArg::Column(col)) => agg_result
                .mins.get(col.as_str())
                .map_or(serde_json::Value::Null, |v| serde_json::json!(v)),
            (AggregateType::Max, AggregateArg::Column(col)) => agg_result
                .maxs.get(col.as_str())
                .map_or(serde_json::Value::Null, |v| serde_json::json!(v)),
            _ => serde_json::Value::Null,
        }
    }

    /// Get a nested value from JSON payload using dot notation.
    pub(crate) fn get_nested_value<'a>(
        payload: &'a serde_json::Value,
        path: &str,
    ) -> Option<&'a serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = payload;

        for part in parts {
            match current {
                serde_json::Value::Object(map) => {
                    current = map.get(part)?;
                }
                _ => return None,
            }
        }

        Some(current)
    }
}
