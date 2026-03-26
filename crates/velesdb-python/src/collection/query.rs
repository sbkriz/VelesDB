//! VelesQL query, match, and explain methods for Collection.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::collections::HashMap;

use crate::collection_helpers::id_score_pairs_to_dicts;
use crate::utils::{extract_vector, json_to_python, python_to_json, to_pyobject};

use super::Collection;

/// Convert a `MatchResult` to a Python dict.
///
/// Extracts node_id, depth, path, bindings, score, and projected fields
/// into a flat Python dict with interned keys.
fn match_result_to_dict(
    py: Python<'_>,
    r: velesdb_core::collection::search::query::match_exec::MatchResult,
) -> PyObject {
    let dict = PyDict::new(py);
    let _ = dict.set_item(PyString::intern(py, "node_id"), r.node_id);
    let _ = dict.set_item(PyString::intern(py, "depth"), r.depth);
    let _ = dict.set_item(PyString::intern(py, "path"), to_pyobject(py, r.path));
    let _ = dict.set_item(
        PyString::intern(py, "bindings"),
        to_pyobject(py, r.bindings),
    );
    let _ = dict.set_item(PyString::intern(py, "score"), r.score);
    let projected = PyDict::new(py);
    for (k, v) in r.projected {
        let _ = projected.set_item(k, json_to_python(py, &v));
    }
    let _ = dict.set_item(PyString::intern(py, "projected"), projected);
    dict.into_any().unbind()
}

/// Parses a VelesQL string into an AST, mapping parse errors to `PyValueError`.
fn parse_velesql(query_str: &str) -> PyResult<velesdb_core::velesql::Query> {
    velesdb_core::velesql::Parser::parse(query_str).map_err(|e| {
        PyValueError::new_err(format!(
            "VelesQL parse error at position {}: {}",
            e.position, e.message
        ))
    })
}

/// Converts Python params dict to Rust `HashMap<String, serde_json::Value>`.
fn convert_params(
    py: Python<'_>,
    params: Option<HashMap<String, PyObject>>,
) -> PyResult<HashMap<String, serde_json::Value>> {
    params
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| python_to_json(py, &v).map(|json_val| (k, json_val)))
        .collect()
}

#[pymethods]
impl Collection {
    /// Execute a VelesQL query (EPIC-031 US-008).
    ///
    /// Args:
    ///     query_str: VelesQL SELECT query string
    ///     params: Query parameters (vectors as lists/numpy arrays, scalars)
    ///
    /// Returns:
    ///     List of query results with node_id, vector_score, graph_score,
    ///     fused_score, bindings (payload), and column_data
    ///
    /// Example:
    ///     >>> results = collection.query(
    ///     ...     "SELECT * FROM docs WHERE vector NEAR $q LIMIT 20",
    ///     ...     params={"q": query_embedding}
    ///     ... )
    #[pyo3(signature = (query_str, params=None))]
    fn query(
        &self,
        query_str: &str,
        params: Option<HashMap<String, PyObject>>,
    ) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let parsed = parse_velesql(query_str)?;
            let rust_params = convert_params(py, params)?;

            let results = self
                .inner
                .execute_query(&parsed, &rust_params)
                .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {e}")))?;

            Ok(crate::collection_helpers::search_results_to_multimodel_dicts(py, results))
        })
    }

    /// Execute a MATCH graph traversal query.
    ///
    /// Args:
    ///     query_str: VelesQL MATCH query string
    ///     params: Query parameters (default: empty dict)
    ///     vector: Optional query vector for similarity scoring
    ///     threshold: Similarity threshold (default: 0.0)
    ///
    /// Returns:
    ///     List of dicts with keys: node_id, depth, path, bindings, score, projected
    #[pyo3(signature = (query_str, params = None, vector = None, threshold = 0.0))]
    fn match_query(
        &self,
        query_str: &str,
        params: Option<HashMap<String, PyObject>>,
        vector: Option<PyObject>,
        threshold: f32,
    ) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let parsed = parse_velesql(query_str)?;
            let match_clause = parsed
                .match_clause
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("Query is not a MATCH query"))?;

            let rust_params = convert_params(py, params)?;

            let results = if let Some(vector_obj) = vector {
                let query_vector = extract_vector(py, &vector_obj)?;
                self.inner
                    .execute_match_with_similarity(
                        match_clause,
                        &query_vector,
                        threshold,
                        &rust_params,
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("MATCH query failed: {e}")))?
            } else {
                self.inner
                    .execute_match(match_clause, &rust_params)
                    .map_err(|e| PyRuntimeError::new_err(format!("MATCH query failed: {e}")))?
            };

            let py_results: Vec<PyObject> = results
                .into_iter()
                .map(|r| match_result_to_dict(py, r))
                .collect();
            Ok(py_results)
        })
    }

    /// Return query execution plan (EXPLAIN).
    #[pyo3(signature = (query_str))]
    fn explain(&self, query_str: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let parsed = parse_velesql(query_str)?;

            let plan = if let Some(match_clause) = parsed.match_clause.as_ref() {
                let stats = velesdb_core::collection::search::query::match_planner::CollectionStats::default();
                velesdb_core::velesql::QueryPlan::from_match(match_clause, &stats)
            } else {
                velesdb_core::velesql::QueryPlan::from_select(&parsed.select)
            };

            let dict = PyDict::new(py);
            let _ = dict.set_item(
                PyString::intern(py, "tree"),
                to_pyobject(py, plan.to_tree()),
            );
            let _ = dict.set_item(
                PyString::intern(py, "estimated_cost_ms"),
                plan.estimated_cost_ms,
            );
            let _ = dict.set_item(
                PyString::intern(py, "filter_strategy"),
                plan.filter_strategy.as_str(),
            );
            let _ = dict.set_item(
                PyString::intern(py, "index_used"),
                plan.index_used.map(|i| i.as_str().to_string()),
            );
            Ok(dict.into_any().unbind())
        })
    }

    /// Execute a VelesQL query returning only IDs and scores (no payload).
    ///
    /// More efficient than `query()` when payload is not needed.
    ///
    /// Args:
    ///     velesql: VelesQL query string
    ///     params: Optional dict of query parameters
    ///
    /// Returns:
    ///     List of dicts with 'id' and 'score' fields
    ///
    /// Example:
    ///     >>> ids = collection.query_ids("SELECT * FROM docs WHERE price > 100 LIMIT 5")
    #[pyo3(signature = (velesql, params = None))]
    fn query_ids(
        &self,
        velesql: &str,
        params: Option<HashMap<String, PyObject>>,
    ) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let parsed_query = parse_velesql(velesql)?;
            let json_params = convert_params(py, params)?;

            let results = self
                .inner
                .execute_query(&parsed_query, &json_params)
                .map_err(|e| PyRuntimeError::new_err(format!("Query execution failed: {e}")))?;

            let tuples: Vec<(u64, f32)> =
                results.into_iter().map(|r| (r.point.id, r.score)).collect();
            Ok(id_score_pairs_to_dicts(py, tuples))
        })
    }
}
