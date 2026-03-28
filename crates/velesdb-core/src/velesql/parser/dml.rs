//! DML statement parsing (INSERT, UPDATE).

use super::{extract_identifier, Rule};
use crate::velesql::ast::{
    DmlStatement, InsertStatement, Query, UpdateAssignment, UpdateStatement,
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
}
