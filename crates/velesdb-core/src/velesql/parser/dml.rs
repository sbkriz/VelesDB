//! DML statement parsing (INSERT, UPDATE, INSERT EDGE, DELETE, DELETE EDGE).

use super::{extract_identifier, Rule};
use crate::velesql::ast::{
    DeleteEdgeStatement, DeleteStatement, DmlStatement, InsertEdgeStatement, InsertStatement,
    Query, UpdateAssignment, UpdateStatement, Value,
};
use crate::velesql::error::ParseError;
use crate::velesql::Parser;

impl Parser {
    pub(crate) fn parse_insert_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        let mut table = None;
        let mut columns = Vec::new();
        let mut values = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier => {
                    if table.is_none() {
                        table = Some(extract_identifier(&inner));
                    } else {
                        columns.push(extract_identifier(&inner));
                    }
                }
                Rule::value => values.push(Self::parse_value(inner)?),
                _ => {}
            }
        }

        let table =
            table.ok_or_else(|| ParseError::syntax(0, "", "INSERT requires target table"))?;
        Self::validate_insert_columns_values(&columns, &values)?;

        Ok(Query::new_dml(DmlStatement::Insert(InsertStatement {
            table,
            columns,
            values,
        })))
    }

    /// Validates INSERT column and value lists: non-empty and matching lengths.
    fn validate_insert_columns_values(
        columns: &[String],
        values: &[crate::velesql::ast::Value],
    ) -> Result<(), ParseError> {
        if columns.is_empty() {
            return Err(ParseError::syntax(
                0,
                "",
                "INSERT requires at least one target column",
            ));
        }
        if columns.len() != values.len() {
            return Err(ParseError::syntax(
                0,
                "",
                "INSERT columns/value count mismatch",
            ));
        }
        Ok(())
    }

    pub(crate) fn parse_update_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        let mut table = None;
        let mut assignments = Vec::new();
        let mut where_clause = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier if table.is_none() => {
                    table = Some(extract_identifier(&inner));
                }
                Rule::assignment => {
                    assignments.push(Self::parse_assignment(inner)?);
                }
                Rule::where_clause => where_clause = Some(Self::parse_where_clause(inner)?),
                _ => {}
            }
        }

        let table =
            table.ok_or_else(|| ParseError::syntax(0, "", "UPDATE requires target table"))?;
        if assignments.is_empty() {
            return Err(ParseError::syntax(
                0,
                "",
                "UPDATE requires at least one assignment",
            ));
        }

        Ok(Query::new_dml(DmlStatement::Update(UpdateStatement {
            table,
            assignments,
            where_clause,
        })))
    }

    /// Parses a single `column = value` assignment from an UPDATE statement.
    fn parse_assignment(pair: pest::iterators::Pair<Rule>) -> Result<UpdateAssignment, ParseError> {
        let mut inner = pair.into_inner();
        let column = inner
            .next()
            .map(|p| extract_identifier(&p))
            .ok_or_else(|| ParseError::syntax(0, "", "UPDATE assignment missing column"))?;
        let value_pair = inner
            .next()
            .ok_or_else(|| ParseError::syntax(0, "", "UPDATE assignment missing value"))?;
        let value = Self::parse_value(value_pair)?;
        Ok(UpdateAssignment { column, value })
    }

    /// Parses an `INSERT EDGE` statement.
    ///
    /// Grammar:
    /// ```text
    /// INSERT EDGE INTO collection (fields) [WITH PROPERTIES (options)]
    /// ```
    pub(crate) fn parse_insert_edge_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        let mut collection: Option<String> = None;
        let mut fields: Vec<(String, Value)> = Vec::new();
        let mut properties: Vec<(String, Value)> = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier if collection.is_none() => {
                    collection = Some(extract_identifier(&inner));
                }
                Rule::edge_field_list => fields = parse_edge_fields(inner)?,
                Rule::edge_properties_clause => {
                    properties = parse_edge_properties(inner)?;
                }
                _ => {}
            }
        }

        let collection = collection
            .ok_or_else(|| ParseError::syntax(0, "", "INSERT EDGE requires a collection name"))?;

        build_insert_edge(collection, &fields, properties)
    }

    /// Parses a `DELETE FROM` statement.
    ///
    /// Grammar:
    /// ```text
    /// DELETE FROM collection WHERE condition
    /// ```
    pub(crate) fn parse_delete_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        let mut table: Option<String> = None;
        let mut where_clause = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier if table.is_none() => {
                    table = Some(extract_identifier(&inner));
                }
                Rule::where_clause => {
                    where_clause = Some(Self::parse_where_clause(inner)?);
                }
                _ => {}
            }
        }

        let table =
            table.ok_or_else(|| ParseError::syntax(0, "", "DELETE requires a target table"))?;
        let where_clause = where_clause
            .ok_or_else(|| ParseError::syntax(0, "", "DELETE requires a WHERE clause"))?;

        Ok(Query::new_dml(DmlStatement::Delete(DeleteStatement {
            table,
            where_clause,
        })))
    }

    /// Parses a `DELETE EDGE` statement.
    ///
    /// Grammar:
    /// ```text
    /// DELETE EDGE edge_id FROM collection
    /// ```
    pub(crate) fn parse_delete_edge_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        let mut edge_id: Option<u64> = None;
        let mut collection: Option<String> = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::value if edge_id.is_none() => {
                    let v = Self::parse_value(inner)?;
                    edge_id = Some(extract_edge_id(&v)?);
                }
                Rule::identifier if collection.is_none() => {
                    collection = Some(extract_identifier(&inner));
                }
                _ => {}
            }
        }

        let edge_id =
            edge_id.ok_or_else(|| ParseError::syntax(0, "", "DELETE EDGE requires an edge ID"))?;
        let collection = collection
            .ok_or_else(|| ParseError::syntax(0, "", "DELETE EDGE requires a collection name"))?;

        Ok(Query::new_dml(DmlStatement::DeleteEdge(
            DeleteEdgeStatement {
                collection,
                edge_id,
            },
        )))
    }
}

// ---------------------------------------------------------------------------
// Private helpers for DML extensions
// ---------------------------------------------------------------------------

/// Parses edge field list: `identifier = value` pairs.
fn parse_edge_fields(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Vec<(String, Value)>, ParseError> {
    let mut fields = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::edge_field {
            fields.push(parse_single_edge_field(inner)?);
        }
    }

    Ok(fields)
}

/// Parses a single `edge_field`: `identifier = value`.
fn parse_single_edge_field(
    pair: pest::iterators::Pair<Rule>,
) -> Result<(String, Value), ParseError> {
    let mut key = String::new();
    let mut value = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier if key.is_empty() => {
                key = extract_identifier(&inner).to_ascii_lowercase();
            }
            Rule::value => value = Some(Parser::parse_value(inner)?),
            _ => {}
        }
    }

    let value = value.ok_or_else(|| ParseError::syntax(0, "", "Edge field requires a value"))?;
    Ok((key, value))
}

/// Parses the `edge_properties_clause`: `WITH PROPERTIES (options)`.
fn parse_edge_properties(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Vec<(String, Value)>, ParseError> {
    let mut properties = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::create_option_list {
            for opt_pair in inner.into_inner() {
                if opt_pair.as_rule() == Rule::create_option {
                    properties.push(parse_option_as_value(opt_pair)?);
                }
            }
        }
    }

    Ok(properties)
}

/// Parses a `create_option` into a `(String, Value)` pair.
fn parse_option_as_value(pair: pest::iterators::Pair<Rule>) -> Result<(String, Value), ParseError> {
    let mut key = String::new();
    let mut value = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier if key.is_empty() => {
                key = extract_identifier(&inner).to_ascii_lowercase();
            }
            Rule::create_option_value => {
                value = Some(parse_create_option_value_as_value(inner)?);
            }
            _ => {}
        }
    }

    let value =
        value.ok_or_else(|| ParseError::syntax(0, "", "Property option requires a value"))?;
    Ok((key, value))
}

/// Converts a `create_option_value` into a `Value`.
fn parse_create_option_value_as_value(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Value, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::syntax(0, "", "Expected option value"))?;

    match inner.as_rule() {
        Rule::identifier => Ok(Value::String(extract_identifier(&inner))),
        _ => crate::velesql::parser::helpers::parse_scalar_from_rule(&inner),
    }
}

/// Builds an `InsertEdgeStatement` from parsed fields and properties.
fn build_insert_edge(
    collection: String,
    fields: &[(String, Value)],
    properties: Vec<(String, Value)>,
) -> Result<Query, ParseError> {
    let source = extract_required_u64(fields, "source", "INSERT EDGE")?;
    let target = extract_required_u64(fields, "target", "INSERT EDGE")?;
    let label = extract_required_string(fields, "label", "INSERT EDGE")?;
    let edge_id = extract_optional_u64(fields, "id");

    Ok(Query::new_dml(DmlStatement::InsertEdge(
        InsertEdgeStatement {
            collection,
            edge_id,
            source,
            target,
            label,
            properties,
        },
    )))
}

/// Extracts a required `u64` field from edge field pairs.
fn extract_required_u64(
    fields: &[(String, Value)],
    key: &str,
    context: &str,
) -> Result<u64, ParseError> {
    let value = fields
        .iter()
        .find(|(k, _)| k == key)
        .ok_or_else(|| ParseError::syntax(0, "", format!("{context} requires '{key}' field")))?;

    extract_edge_id(&value.1)
}

/// Extracts a required `String` field from edge field pairs.
fn extract_required_string(
    fields: &[(String, Value)],
    key: &str,
    context: &str,
) -> Result<String, ParseError> {
    let value = fields
        .iter()
        .find(|(k, _)| k == key)
        .ok_or_else(|| ParseError::syntax(0, "", format!("{context} requires '{key}' field")))?;

    match &value.1 {
        Value::String(s) => Ok(s.clone()),
        _ => Err(ParseError::syntax(
            0,
            "",
            format!("'{key}' must be a string"),
        )),
    }
}

/// Extracts an optional `u64` field from edge field pairs.
fn extract_optional_u64(fields: &[(String, Value)], key: &str) -> Option<u64> {
    fields
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            Value::Integer(i) => u64::try_from(*i).ok(),
            _ => None,
        })
}

/// Converts a `Value` reference to a `u64` edge ID.
fn extract_edge_id(value: &Value) -> Result<u64, ParseError> {
    match value {
        Value::Integer(i) => u64::try_from(*i)
            .map_err(|_| ParseError::syntax(0, "", "Edge ID must be a non-negative integer")),
        _ => Err(ParseError::syntax(0, "", "Edge ID must be an integer")),
    }
}
