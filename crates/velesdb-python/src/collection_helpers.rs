//! Helper functions for Collection result conversions.
//!
//! Extracted from collection.rs to reduce file size and improve maintainability.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::collections::{BTreeMap, HashMap};

use crate::utils::json_to_python;
use velesdb_core::sparse_index::SparseVector;
use velesdb_core::{Filter, Point, SearchResult};

/// Parse a Python filter object into a VelesDB Filter.
///
/// Converts the Python dict directly to `serde_json::Value` via [`python_to_json`],
/// then deserializes into [`Filter`]. This avoids the Python `json.dumps` round-trip.
pub fn parse_filter(py: Python<'_>, filter: &PyObject) -> PyResult<Filter> {
    let json_value = crate::utils::python_to_json(py, filter)?;
    serde_json::from_value(json_value)
        .map_err(|e| PyValueError::new_err(format!("Invalid filter: {e}")))
}

/// Parse an optional Python filter object.
pub fn parse_optional_filter(py: Python<'_>, filter: Option<PyObject>) -> PyResult<Option<Filter>> {
    match filter {
        Some(f) => Ok(Some(parse_filter(py, &f)?)),
        None => Ok(None),
    }
}

/// Convert payload to Python object (shared helper to avoid duplication).
#[inline]
fn payload_to_python(py: Python<'_>, payload: &Option<serde_json::Value>) -> PyObject {
    match payload {
        Some(p) => json_to_python(py, p),
        None => py.None(),
    }
}

/// Convert a `SearchResult` to a Python dict, bypassing `HashMap` intermediary.
///
/// Uses `PyDict::new()` + `set_item()` directly and `PyString::intern()` for
/// static keys to avoid repeated string allocation.
pub fn search_result_to_dict(py: Python<'_>, result: &SearchResult) -> PyObject {
    let dict = PyDict::new(py);
    // PyString::intern reuses the same Python string object across calls
    let _ = dict.set_item(PyString::intern(py, "id"), result.point.id);
    let _ = dict.set_item(PyString::intern(py, "score"), result.score);
    let _ = dict.set_item(
        PyString::intern(py, "payload"),
        payload_to_python(py, &result.point.payload),
    );
    dict.into_any().unbind()
}

/// Convert a `SearchResult` to a multi-model Python dict (EPIC-031).
pub fn search_result_to_multimodel_dict(py: Python<'_>, result: &SearchResult) -> PyObject {
    let dict = PyDict::new(py);
    let none = py.None();
    let payload_py = payload_to_python(py, &result.point.payload);

    // Multi-model fields
    let _ = dict.set_item(PyString::intern(py, "node_id"), result.point.id);
    let _ = dict.set_item(PyString::intern(py, "vector_score"), result.score);
    let _ = dict.set_item(PyString::intern(py, "graph_score"), &none);
    let _ = dict.set_item(PyString::intern(py, "fused_score"), result.score);
    let _ = dict.set_item(PyString::intern(py, "bindings"), payload_py.clone_ref(py));
    let _ = dict.set_item(PyString::intern(py, "column_data"), &none);

    // Legacy fields for compatibility
    let _ = dict.set_item(PyString::intern(py, "id"), result.point.id);
    let _ = dict.set_item(PyString::intern(py, "score"), result.score);
    let _ = dict.set_item(PyString::intern(py, "payload"), payload_py);

    dict.into_any().unbind()
}

/// Convert a `Point` to a Python dict.
pub fn point_to_dict(py: Python<'_>, point: &Point) -> PyObject {
    let dict = PyDict::new(py);
    let _ = dict.set_item(PyString::intern(py, "id"), point.id);
    let np_vector = numpy::PyArray1::from_slice(py, &point.vector);
    let _ = dict.set_item(PyString::intern(py, "vector"), np_vector);
    let _ = dict.set_item(
        PyString::intern(py, "payload"),
        payload_to_python(py, &point.payload),
    );
    dict.into_any().unbind()
}

/// Convert a list of `SearchResult`s to Python dicts.
pub fn search_results_to_dicts(py: Python<'_>, results: Vec<SearchResult>) -> Vec<PyObject> {
    results
        .into_iter()
        .map(|r| search_result_to_dict(py, &r))
        .collect()
}

/// Convert a list of `SearchResult`s to multi-model Python dicts.
pub fn search_results_to_multimodel_dicts(
    py: Python<'_>,
    results: Vec<SearchResult>,
) -> Vec<PyObject> {
    results
        .into_iter()
        .map(|r| search_result_to_multimodel_dict(py, &r))
        .collect()
}

/// Convert a list of (id, score) pairs to Python dicts.
pub fn id_score_pairs_to_dicts(py: Python<'_>, results: Vec<(u64, f32)>) -> Vec<PyObject> {
    results
        .into_iter()
        .map(|(id, score)| {
            let dict = PyDict::new(py);
            let _ = dict.set_item(PyString::intern(py, "id"), id);
            let _ = dict.set_item(PyString::intern(py, "score"), score);
            dict.into_any().unbind()
        })
        .collect()
}

/// Parse a Python object into a `SparseVector`.
///
/// Accepts:
/// - A Python `dict[int, float]` mapping dimension indices to weights.
/// - A scipy sparse object with a `.toarray()` method (COO/CSR/CSC).
///
/// Returns `PyValueError` if the object is neither format.
pub fn parse_sparse_vector(py: Python<'_>, obj: &PyObject) -> PyResult<SparseVector> {
    // Try dict[int, float] first (most common usage pattern).
    if let Ok(dict) = obj.extract::<HashMap<u32, f32>>(py) {
        let pairs: Vec<(u32, f32)> = dict.into_iter().collect();
        return Ok(SparseVector::new(pairs));
    }

    // Try scipy.sparse via duck typing: check for `.toarray()` method.
    // First confirm the attribute exists without calling it, so we can distinguish
    // "not a scipy object" (AttributeError → try next format) from runtime errors.
    let has_toarray_attr = obj
        .getattr(py, "toarray")
        .map(|attr| !attr.is_none(py))
        .unwrap_or(false);

    if has_toarray_attr {
        // `.toarray()` returns a dense numpy 2D array; flatten to 1D.
        let array = obj
            .call_method0(py, "toarray")
            .map_err(|e| PyValueError::new_err(format!("scipy toarray() failed: {e}")))?;
        let flat = array
            .call_method0(py, "flatten")
            .map_err(|e| PyValueError::new_err(format!("scipy flatten() failed: {e}")))?;
        let values: Vec<f32> = flat.extract(py).map_err(|e| {
            PyValueError::new_err(format!("Failed to extract floats from scipy array: {e}"))
        })?;
        let pairs: Vec<(u32, f32)> = values
            .into_iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > f32::EPSILON)
            .map(|(i, v)| u32::try_from(i).map(|idx| (idx, v)))
            .collect::<Result<_, _>>()
            .map_err(|_| {
                PyValueError::new_err("Sparse vector has more than u32::MAX dimensions")
            })?;
        return Ok(SparseVector::new(pairs));
    }

    Err(PyValueError::new_err(
        "sparse_vector must be a dict[int, float] or a scipy sparse object with .toarray()",
    ))
}

/// Parse the `sparse_vector` field from a point dict into named sparse vectors.
///
/// Accepts:
/// - `dict[int, float]`: treated as the default (unnamed) sparse vector.
/// - `dict[str, dict[int, float]]`: named sparse vectors.
///
/// Returns `None` if the key is absent from the point dict.
pub fn parse_sparse_vectors_from_point(
    py: Python<'_>,
    point_dict: &HashMap<String, PyObject>,
) -> PyResult<Option<BTreeMap<String, SparseVector>>> {
    let Some(obj) = point_dict.get("sparse_vector") else {
        return Ok(None);
    };

    // Try as dict[int, float] -> default sparse vector (key "").
    if let Ok(dict) = obj.extract::<HashMap<u32, f32>>(py) {
        let sv = SparseVector::new(dict.into_iter().collect());
        let mut map = BTreeMap::new();
        map.insert(String::new(), sv);
        return Ok(Some(map));
    }

    // Try as dict[str, dict[int, float]] -> named sparse vectors.
    if let Ok(named) = obj.extract::<HashMap<String, PyObject>>(py) {
        let mut map = BTreeMap::new();
        for (name, inner_obj) in named {
            let sv = parse_sparse_vector(py, &inner_obj)?;
            map.insert(name, sv);
        }
        return Ok(Some(map));
    }

    Err(PyValueError::new_err(
        "sparse_vector must be dict[int, float] or dict[str, dict[int, float]]",
    ))
}
