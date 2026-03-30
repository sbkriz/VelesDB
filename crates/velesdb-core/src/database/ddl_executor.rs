//! DDL and extended DML executor for `VelesQL` v3.3.
//!
//! Handles CREATE/DROP COLLECTION, INSERT EDGE, DELETE, and DELETE EDGE
//! by delegating to existing [`Database`] APIs.

use crate::collection::graph::{EdgeType, GraphSchema, NodeType, ValueType};
use crate::velesql::{CreateCollectionKind, DdlStatement, GraphSchemaMode, SchemaDefinition};
use crate::{Error, Result, SearchResult};

use super::Database;

impl Database {
    /// Dispatches a DDL statement to the appropriate executor.
    ///
    /// # Errors
    ///
    /// Returns an error if the observer rejects the operation (RBAC)
    /// or if the collection operation itself fails.
    pub(super) fn execute_ddl(&self, ddl: &DdlStatement) -> Result<Vec<SearchResult>> {
        // RBAC hook — allows premium extensions to reject DDL.
        if let Some(ref observer) = self.observer {
            let (operation, name) = ddl_operation_info(ddl);
            observer.on_ddl_request(operation, &name)?;
        }

        match ddl {
            DdlStatement::CreateCollection(stmt) => self.execute_create_collection(stmt),
            DdlStatement::DropCollection(stmt) => self.execute_drop_collection(stmt),
        }
    }

    /// Executes a CREATE COLLECTION statement.
    ///
    /// Delegates to the appropriate typed creation API based on the
    /// collection kind (Vector, Graph, or Metadata).
    ///
    /// # Errors
    ///
    /// Returns an error if the collection already exists or parameters are invalid.
    fn execute_create_collection(
        &self,
        stmt: &crate::velesql::CreateCollectionStatement,
    ) -> Result<Vec<SearchResult>> {
        match &stmt.kind {
            CreateCollectionKind::Vector(params) => self.create_vector_from_ddl(&stmt.name, params),
            CreateCollectionKind::Graph(params) => self.create_graph_from_ddl(&stmt.name, params),
            CreateCollectionKind::Metadata => {
                self.create_metadata_collection(&stmt.name)?;
                Ok(Vec::new())
            }
        }
    }

    /// Creates a vector collection from DDL parameters.
    fn create_vector_from_ddl(
        &self,
        name: &str,
        params: &crate::velesql::VectorCollectionParams,
    ) -> Result<Vec<SearchResult>> {
        let metric = resolve_metric(&params.metric)?;
        let storage = resolve_storage_mode(params.storage.as_deref())?;

        if params.m.is_some() || params.ef_construction.is_some() {
            self.create_vector_collection_with_hnsw(
                name,
                params.dimension,
                metric,
                storage,
                params.m,
                params.ef_construction,
            )?;
        } else {
            self.create_vector_collection_with_options(name, params.dimension, metric, storage)?;
        }
        Ok(Vec::new())
    }

    /// Creates a graph collection from DDL parameters.
    fn create_graph_from_ddl(
        &self,
        name: &str,
        params: &crate::velesql::GraphCollectionParams,
    ) -> Result<Vec<SearchResult>> {
        let schema = build_graph_schema(&params.schema_mode);

        if let Some(dim) = params.dimension {
            let metric_str = params.metric.as_deref().unwrap_or("cosine");
            let metric = resolve_metric(metric_str)?;
            self.create_graph_collection_with_embeddings(name, schema, dim, metric)?;
        } else {
            self.create_graph_collection(name, schema)?;
        }
        Ok(Vec::new())
    }

    /// Executes a DROP COLLECTION statement.
    ///
    /// When `IF EXISTS` is specified, silently succeeds if the collection
    /// does not exist instead of returning an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection does not exist (without IF EXISTS)
    /// or if the deletion itself fails.
    fn execute_drop_collection(
        &self,
        stmt: &crate::velesql::DropCollectionStatement,
    ) -> Result<Vec<SearchResult>> {
        match self.delete_collection(&stmt.name) {
            Ok(()) => Ok(Vec::new()),
            Err(Error::CollectionNotFound(_)) if stmt.if_exists => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// Executes an INSERT EDGE statement.
    ///
    /// Resolves the target graph collection and builds a `GraphEdge`
    /// from the statement fields.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not a graph collection,
    /// or if the edge insertion fails.
    pub(super) fn execute_insert_edge(
        &self,
        stmt: &crate::velesql::InsertEdgeStatement,
    ) -> Result<Vec<SearchResult>> {
        self.check_dml_mutation("INSERT_EDGE", &stmt.collection)?;
        let graph = self.resolve_graph_collection(&stmt.collection)?;
        let edge = build_graph_edge(stmt)?;
        graph.add_edge(edge)?;
        Ok(Vec::new())
    }

    /// Executes a DELETE FROM statement.
    ///
    /// Extracts point IDs from the WHERE clause and deletes them
    /// from the resolved collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not found, or the WHERE
    /// clause does not match a supported ID pattern.
    pub(super) fn execute_delete(
        &self,
        stmt: &crate::velesql::DeleteStatement,
    ) -> Result<Vec<SearchResult>> {
        self.check_dml_mutation("DELETE", &stmt.table)?;
        let ids = extract_delete_ids(&stmt.where_clause)?;
        let collection = self.resolve_writable_collection(&stmt.table)?;
        collection.delete(&ids)?;
        Ok(Vec::new())
    }

    /// Executes a DELETE EDGE statement.
    ///
    /// Resolves the target graph collection and removes the edge by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not a graph collection.
    pub(super) fn execute_delete_edge(
        &self,
        stmt: &crate::velesql::DeleteEdgeStatement,
    ) -> Result<Vec<SearchResult>> {
        self.check_dml_mutation("DELETE_EDGE", &stmt.collection)?;
        let graph = self.resolve_graph_collection(&stmt.collection)?;
        let _ = graph.remove_edge(stmt.edge_id);
        Ok(Vec::new())
    }

    /// Checks the RBAC observer for DML mutation permission.
    ///
    /// Returns `Ok(())` if no observer is configured or if the observer
    /// allows the operation. Propagates the observer's error otherwise.
    fn check_dml_mutation(&self, operation: &str, collection: &str) -> Result<()> {
        if let Some(ref observer) = self.observer {
            observer.on_dml_mutation_request(operation, collection)?;
        }
        Ok(())
    }

    /// Resolves a graph collection by name.
    ///
    /// Returns `CollectionNotFound` if no graph collection exists with the given name.
    fn resolve_graph_collection(&self, name: &str) -> Result<crate::collection::GraphCollection> {
        self.get_graph_collection(name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Private helper functions
// ---------------------------------------------------------------------------

/// Builds a `GraphEdge` from an `InsertEdgeStatement`.
///
/// Resolves the edge ID (explicit or hashed), creates the edge, and
/// attaches properties if any are present.
fn build_graph_edge(
    stmt: &crate::velesql::InsertEdgeStatement,
) -> Result<crate::collection::GraphEdge> {
    let edge_id = stmt
        .edge_id
        .unwrap_or_else(|| hash_edge_id(stmt.source, stmt.target, &stmt.label));
    let edge = crate::collection::GraphEdge::new(edge_id, stmt.source, stmt.target, &stmt.label)?;

    if stmt.properties.is_empty() {
        return Ok(edge);
    }
    let props = resolve_edge_properties(&stmt.properties)?;
    Ok(edge.with_properties(props))
}

/// Extracts the operation name and collection name from a DDL statement.
///
/// Used by the RBAC hook to identify the operation being requested.
fn ddl_operation_info(ddl: &DdlStatement) -> (&str, String) {
    match ddl {
        DdlStatement::CreateCollection(stmt) => ("CREATE", stmt.name.clone()),
        DdlStatement::DropCollection(stmt) => ("DROP", stmt.name.clone()),
    }
}

/// Resolves a metric name string to a `DistanceMetric` enum.
///
/// # Errors
///
/// Returns a query error if the metric name is unrecognized.
fn resolve_metric(s: &str) -> Result<crate::DistanceMetric> {
    crate::DistanceMetric::parse_alias(s).ok_or_else(|| {
        Error::Query(format!(
            "Unknown metric '{s}'. Use: cosine, euclidean, dot, hamming, jaccard"
        ))
    })
}

/// Resolves an optional storage mode string to a `StorageMode` enum.
///
/// Defaults to `StorageMode::Full` when `None` is provided.
///
/// # Errors
///
/// Returns a query error if the storage mode name is unrecognized.
fn resolve_storage_mode(s: Option<&str>) -> Result<crate::StorageMode> {
    let Some(name) = s else {
        return Ok(crate::StorageMode::default());
    };
    crate::StorageMode::parse_alias(name).ok_or_else(|| {
        Error::Query(format!(
            "Unknown storage mode '{name}'. Use: full, sq8, binary, pq, rabitq"
        ))
    })
}

/// Maps a `VelesQL` type name string to a `ValueType`.
fn resolve_value_type(s: &str) -> ValueType {
    match s.to_uppercase().as_str() {
        "INTEGER" | "INT" => ValueType::Integer,
        "FLOAT" | "DOUBLE" => ValueType::Float,
        "BOOLEAN" | "BOOL" => ValueType::Boolean,
        "VECTOR" | "EMBEDDING" => ValueType::Vector,
        // "STRING", "TEXT", and any unrecognized type default to String.
        _ => ValueType::String,
    }
}

/// Builds a `GraphSchema` from the AST `GraphSchemaMode`.
fn build_graph_schema(mode: &GraphSchemaMode) -> GraphSchema {
    match mode {
        GraphSchemaMode::Schemaless => GraphSchema::schemaless(),
        GraphSchemaMode::Typed(definitions) => build_typed_schema(definitions),
    }
}

/// Builds a typed graph schema from a list of schema definitions.
fn build_typed_schema(definitions: &[SchemaDefinition]) -> GraphSchema {
    let mut schema = GraphSchema::new();

    for def in definitions {
        match def {
            SchemaDefinition::Node { name, properties } => {
                let props: std::collections::HashMap<String, ValueType> = properties
                    .iter()
                    .map(|(k, v)| (k.clone(), resolve_value_type(v)))
                    .collect();
                schema = schema.with_node_type(NodeType::new(name).with_properties(props));
            }
            SchemaDefinition::Edge {
                name,
                from_type,
                to_type,
            } => {
                schema = schema.with_edge_type(EdgeType::new(name, from_type, to_type));
            }
        }
    }

    schema
}

/// Generates a deterministic edge ID from source, target, and label.
///
/// Uses FNV-1a-inspired mixing for collision resistance. The same
/// (source, target, label) triple always produces the same ID.
pub(super) fn hash_edge_id(source: u64, target: u64, label: &str) -> u64 {
    // FNV-1a offset basis and prime for u64
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    // Mix source bytes
    for byte in source.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Mix target bytes
    for byte in target.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Mix label bytes
    for byte in label.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Resolves AST `Value` properties into a `HashMap<String, serde_json::Value>`.
///
/// # Errors
///
/// Returns an error if a property value type is unsupported.
fn resolve_edge_properties(
    properties: &[(String, crate::velesql::Value)],
) -> Result<std::collections::HashMap<String, serde_json::Value>> {
    properties
        .iter()
        .map(|(key, val)| {
            let json_val = value_to_json(val)?;
            Ok((key.clone(), json_val))
        })
        .collect()
}

/// Converts a `VelesQL` `Value` to `serde_json::Value`.
///
/// # Errors
///
/// Returns an error for unsupported value types (subqueries, temporal).
fn value_to_json(val: &crate::velesql::Value) -> Result<serde_json::Value> {
    match val {
        crate::velesql::Value::Integer(v) => Ok(serde_json::json!(v)),
        crate::velesql::Value::Float(v) => Ok(serde_json::json!(v)),
        crate::velesql::Value::String(v) => Ok(serde_json::json!(v)),
        crate::velesql::Value::Boolean(v) => Ok(serde_json::json!(v)),
        crate::velesql::Value::Null => Ok(serde_json::Value::Null),
        crate::velesql::Value::Parameter(_)
        | crate::velesql::Value::Temporal(_)
        | crate::velesql::Value::Subquery(_) => Err(Error::Query(
            "Edge properties must be literal values".to_string(),
        )),
    }
}

/// Extracts point IDs from a DELETE WHERE clause.
///
/// Supports two patterns:
/// - `id = N` (single ID)
/// - `id IN (N1, N2, ...)` (multi-ID)
///
/// # Errors
///
/// Returns an error if the WHERE clause does not match a supported pattern.
fn extract_delete_ids(condition: &crate::velesql::Condition) -> Result<Vec<u64>> {
    match condition {
        crate::velesql::Condition::Comparison(cmp) if cmp.column == "id" => extract_single_id(cmp),
        crate::velesql::Condition::In(in_cond) if in_cond.column == "id" && !in_cond.negated => {
            extract_in_ids(in_cond)
        }
        _ => Err(Error::Query(
            "DELETE WHERE must use 'id = N' or 'id IN (N1, N2, ...)'".to_string(),
        )),
    }
}

/// Extracts a single ID from a comparison `id = N`.
fn extract_single_id(cmp: &crate::velesql::Comparison) -> Result<Vec<u64>> {
    if cmp.operator != crate::velesql::CompareOp::Eq {
        return Err(Error::Query(
            "DELETE WHERE id must use '=' operator".to_string(),
        ));
    }
    let id = value_to_u64(&cmp.value)?;
    Ok(vec![id])
}

/// Extracts multiple IDs from an IN condition `id IN (N1, N2, ...)`.
fn extract_in_ids(in_cond: &crate::velesql::InCondition) -> Result<Vec<u64>> {
    in_cond.values.iter().map(value_to_u64).collect()
}

/// Converts a `VelesQL` `Value` to a `u64` point ID.
///
/// # Errors
///
/// Returns an error if the value is not a non-negative integer.
fn value_to_u64(val: &crate::velesql::Value) -> Result<u64> {
    match val {
        crate::velesql::Value::Integer(v) => {
            u64::try_from(*v).map_err(|_| Error::Query(format!("ID must be non-negative, got {v}")))
        }
        _ => Err(Error::Query("ID values must be integers".to_string())),
    }
}
