//! Mutation methods for Collection (upsert, delete, flush, stream_insert).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use velesdb_core::Point;

use crate::collection_helpers::{
    dict_to_json_map, extract_point_id, extract_point_vector, parse_payload, parse_point_dicts,
};

use super::Collection;

#[pymethods]
impl Collection {
    /// Insert or update vectors in the collection.
    #[pyo3(signature = (points))]
    fn upsert(&self, points: Vec<HashMap<String, PyObject>>) -> PyResult<usize> {
        Python::with_gil(|py| {
            let core_points = parse_point_dicts(py, &points)?;
            let count = core_points.len();
            self.inner
                .upsert(core_points)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to upsert: {e}")))?;
            Ok(count)
        })
    }

    /// Insert or update metadata-only points (no vectors).
    #[pyo3(signature = (points))]
    fn upsert_metadata(&self, points: Vec<HashMap<String, PyObject>>) -> PyResult<usize> {
        Python::with_gil(|py| {
            let mut core_points = Vec::with_capacity(points.len());

            for point_dict in &points {
                let id = extract_point_id(py, point_dict)?;
                let payload = parse_payload(py, point_dict.get("payload"))?.ok_or_else(|| {
                    PyValueError::new_err("Metadata-only point must have 'payload' field")
                })?;
                core_points.push(Point::metadata_only(id, payload));
            }

            let count = core_points.len();
            self.inner
                .upsert_metadata(core_points)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to upsert_metadata: {e}")))?;
            Ok(count)
        })
    }

    /// Bulk insert optimized for high-throughput import.
    #[pyo3(signature = (points))]
    fn upsert_bulk(&self, points: Vec<HashMap<String, PyObject>>) -> PyResult<usize> {
        Python::with_gil(|py| {
            let mut core_points = Vec::with_capacity(points.len());

            for point_dict in &points {
                let id = extract_point_id(py, point_dict)?;
                let vector = extract_point_vector(py, point_dict)?;
                let payload = parse_payload(py, point_dict.get("payload"))?;
                core_points.push(Point::new(id, vector, payload));
            }

            self.inner
                .upsert_bulk(&core_points)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to upsert_bulk: {e}")))
        })
    }

    /// Bulk insert from numpy arrays for maximum throughput.
    ///
    /// Bypasses Python dict parsing overhead by accepting vectors as a 2D
    /// numpy array and IDs as a 1D array. Payloads are optional.
    ///
    /// Args:
    ///     vectors: numpy.ndarray of shape (n, dimension), dtype float32
    ///     ids: numpy.ndarray of shape (n,), dtype uint64 (or list of int)
    ///     payloads: Optional list of payload dicts (same length as ids)
    ///
    /// Returns:
    ///     Number of inserted points
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> vectors = np.random.randn(1000, 384).astype(np.float32)
    ///     >>> ids = np.arange(1000, dtype=np.uint64)
    ///     >>> count = collection.upsert_bulk_numpy(vectors, ids)
    #[pyo3(signature = (vectors, ids, payloads=None))]
    fn upsert_bulk_numpy(
        &self,
        vectors: numpy::PyReadonlyArray2<f32>,
        ids: Vec<u64>,
        payloads: Option<Vec<Option<HashMap<String, PyObject>>>>,
    ) -> PyResult<usize> {
        Python::with_gil(|py| {
            let array = vectors.as_array();
            let n = array.nrows();
            validate_numpy_lengths(n, &ids, &payloads)?;

            let mut core_points = Vec::with_capacity(n);
            for i in 0..n {
                let row = array.row(i);
                let vec_slice = row
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("numpy array must be C-contiguous"))?;

                let payload = match payloads.as_ref().and_then(|p| p[i].as_ref()) {
                    Some(dict) => Some(dict_to_json_map(py, dict)?),
                    None => None,
                };
                core_points.push(Point::new(ids[i], vec_slice.to_vec(), payload));
            }

            self.inner
                .upsert_bulk(&core_points)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to upsert_bulk: {e}")))
        })
    }

    /// Get points by their IDs.
    #[pyo3(signature = (ids))]
    fn get(&self, ids: Vec<u64>) -> PyResult<Vec<Option<PyObject>>> {
        Python::with_gil(|py| {
            let points = self.inner.get(&ids);
            let py_points = points
                .into_iter()
                .map(|opt_point| {
                    opt_point.map(|p| crate::collection_helpers::point_to_dict(py, &p))
                })
                .collect();
            Ok(py_points)
        })
    }

    /// Delete points by their IDs.
    #[pyo3(signature = (ids))]
    fn delete(&self, ids: Vec<u64>) -> PyResult<()> {
        self.inner
            .delete(&ids)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete: {e}")))
    }

    /// Flush all pending changes to disk.
    fn flush(&self) -> PyResult<()> {
        self.inner
            .flush()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to flush: {e}")))
    }

    /// Insert points via the streaming ingestion channel.
    ///
    /// Points are buffered and merged asynchronously into the HNSW index.
    /// This is faster than `upsert` for high-throughput workloads but offers
    /// eventual consistency (points appear in search after buffer merge).
    ///
    /// Args:
    ///     points: List of point dicts (same format as upsert).
    ///
    /// Returns:
    ///     Number of points successfully queued.
    ///
    /// Raises:
    ///     RuntimeError: If the streaming buffer is full or not configured.
    ///
    /// Example:
    ///     >>> count = collection.stream_insert([
    ///     ...     {"id": 1, "vector": [...], "payload": {"key": "value"}}
    ///     ... ])
    #[pyo3(signature = (points))]
    fn stream_insert(&self, points: Vec<HashMap<String, PyObject>>) -> PyResult<usize> {
        Python::with_gil(|py| {
            // Phase 1: Parse all points while holding the GIL (required for PyObject access)
            let parsed = parse_point_dicts(py, &points)?;

            // Phase 2: Send entire batch in one call (single lock acquisition)
            self.inner.stream_insert_batch(parsed).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Stream insert failed (buffer full or not configured): {e}"
                ))
            })
        })
    }
}

/// Validate that ids and optional payloads lengths match the vector row count.
fn validate_numpy_lengths(
    n: usize,
    ids: &[u64],
    payloads: &Option<Vec<Option<HashMap<String, PyObject>>>>,
) -> PyResult<()> {
    if ids.len() != n {
        return Err(PyValueError::new_err(format!(
            "ids length ({}) must match vectors row count ({n})",
            ids.len()
        )));
    }
    if let Some(ref p) = *payloads {
        if p.len() != n {
            return Err(PyValueError::new_err(format!(
                "payloads length ({}) must match vectors row count ({n})",
                p.len()
            )));
        }
    }
    Ok(())
}
