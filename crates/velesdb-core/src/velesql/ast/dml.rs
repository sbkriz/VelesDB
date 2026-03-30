//! DML statement types for VelesQL.
//!
//! This module defines INSERT/UPDATE/DELETE and graph mutation AST nodes.

use serde::{Deserialize, Serialize};

use super::{Condition, Value};

/// INSERT statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InsertStatement {
    /// Target collection/table name.
    pub table: String,
    /// Target columns.
    pub columns: Vec<String>,
    /// Values corresponding to `columns`.
    pub values: Vec<Value>,
}

/// UPDATE assignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateAssignment {
    /// Column name to update.
    pub column: String,
    /// Assigned value expression.
    pub value: Value,
}

/// UPDATE statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateStatement {
    /// Target collection/table name.
    pub table: String,
    /// SET assignments.
    pub assignments: Vec<UpdateAssignment>,
    /// Optional WHERE clause.
    pub where_clause: Option<Condition>,
}

/// INSERT EDGE statement (VelesQL v3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InsertEdgeStatement {
    /// Target graph collection name.
    pub collection: String,
    /// Optional explicit edge ID.
    pub edge_id: Option<u64>,
    /// Source node ID.
    pub source: u64,
    /// Target node ID.
    pub target: u64,
    /// Edge label/type.
    pub label: String,
    /// Optional edge properties.
    pub properties: Vec<(String, Value)>,
}

/// DELETE FROM statement (VelesQL v3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeleteStatement {
    /// Target collection/table name.
    pub table: String,
    /// WHERE clause (mandatory — prevents accidental full deletion).
    pub where_clause: Condition,
}

/// DELETE EDGE statement (VelesQL v3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeleteEdgeStatement {
    /// Target graph collection name.
    pub collection: String,
    /// Edge ID to delete.
    pub edge_id: u64,
}

/// DML statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DmlStatement {
    /// INSERT statement.
    Insert(InsertStatement),
    /// UPDATE statement.
    Update(UpdateStatement),
    /// INSERT EDGE statement (VelesQL v3.3).
    InsertEdge(InsertEdgeStatement),
    /// DELETE FROM statement (VelesQL v3.3).
    Delete(DeleteStatement),
    /// DELETE EDGE statement (VelesQL v3.3).
    DeleteEdge(DeleteEdgeStatement),
}
