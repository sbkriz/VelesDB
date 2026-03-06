//! Tests for `point` module

use super::point::*;
use crate::index::sparse::SparseVector;
use serde_json::json;

#[test]
fn test_point_creation() {
    let point = Point::new(1, vec![0.1, 0.2, 0.3], Some(json!({"title": "Test"})));

    assert_eq!(point.id, 1);
    assert_eq!(point.dimension(), 3);
    assert!(point.payload.is_some());
}

#[test]
fn test_point_without_payload() {
    let point = Point::without_payload(1, vec![0.1, 0.2, 0.3]);

    assert_eq!(point.id, 1);
    assert!(point.payload.is_none());
}

#[test]
fn test_point_serialization() {
    let point = Point::new(1, vec![0.1, 0.2], Some(json!({"key": "value"})));
    let json = serde_json::to_string(&point).unwrap();
    let deserialized: Point = serde_json::from_str(&json).unwrap();

    assert_eq!(point.id, deserialized.id);
    assert_eq!(point.vector, deserialized.vector);
}

#[test]
fn test_point_metadata_only() {
    let point = Point::metadata_only(42, json!({"category": "test"}));

    assert_eq!(point.id, 42);
    assert!(point.is_metadata_only());
    assert_eq!(point.dimension(), 0);
    assert!(point.payload.is_some());
}

#[test]
fn test_search_result_serialization() {
    let point = Point::new(1, vec![0.1, 0.2], None);
    let result = SearchResult::new(point, 0.85);

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: SearchResult = serde_json::from_str(&json).unwrap();

    assert_eq!(result.point.id, deserialized.point.id);
    assert!((result.score - deserialized.score).abs() < 1e-5);
}

#[test]
fn test_point_backward_compat_no_sparse_vector() {
    // Deserialize JSON without sparse_vector field -> None
    let json_str = r#"{"id": 1, "vector": [0.1, 0.2], "payload": null}"#;
    let point: Point = serde_json::from_str(json_str).unwrap();
    assert_eq!(point.id, 1);
    assert!(!point.has_sparse_vector());
    assert!(point.sparse_vector.is_none());
}

#[test]
fn test_point_sparse_vector_round_trip() {
    let sv = SparseVector::new(vec![(1, 0.5), (3, 1.5)]);
    let point = Point::with_sparse(42, vec![0.1], Some(json!({"k": "v"})), Some(sv));
    assert!(point.has_sparse_vector());

    let json = serde_json::to_string(&point).unwrap();
    let deserialized: Point = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.sparse_vector, point.sparse_vector);
    assert_eq!(deserialized.id, 42);
}

#[test]
fn test_point_sparse_only() {
    let sv = SparseVector::new(vec![(1, 1.0), (2, 2.0)]);
    let point = Point::sparse_only(10, sv.clone(), None);
    assert!(point.is_metadata_only()); // No dense vector
    assert!(point.has_sparse_vector());
    assert_eq!(point.sparse_vector.unwrap().nnz(), 2);
}

#[test]
fn test_point_new_has_no_sparse_vector() {
    let point = Point::new(1, vec![0.1], None);
    assert!(!point.has_sparse_vector());
}

#[test]
fn test_point_sparse_vector_not_serialized_when_none() {
    let point = Point::new(1, vec![0.1], None);
    let json = serde_json::to_string(&point).unwrap();
    assert!(!json.contains("sparse_vector"));
}
