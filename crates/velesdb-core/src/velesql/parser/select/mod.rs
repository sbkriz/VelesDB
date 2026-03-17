//! SELECT statement parsing - facade module.

mod clause_compound;
mod clause_from_join;
mod clause_group_order;
mod clause_limit_with;
mod clause_projection;
pub(crate) mod validation;

use super::Rule;
use crate::velesql::ast::{Query, SelectColumns, SelectStatement};
use crate::velesql::error::ParseError;
use crate::velesql::Parser;

impl Parser {
    pub(crate) fn parse_query(pair: pest::iterators::Pair<Rule>) -> Result<Query, ParseError> {
        let inner = pair.into_inner();
        for p in inner {
            match p.as_rule() {
                Rule::match_query => return Self::parse_match_query(p),
                Rule::compound_query => return Self::parse_compound_query(p),
                Rule::train_stmt => return Self::parse_train_stmt(p),
                Rule::insert_stmt => return Self::parse_insert_stmt(p),
                Rule::update_stmt => return Self::parse_update_stmt(p),
                _ => {}
            }
        }
        Err(ParseError::syntax(
            0,
            "",
            "Expected MATCH, SELECT, INSERT, UPDATE, or TRAIN query",
        ))
    }

    pub(crate) fn parse_select_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<SelectStatement, ParseError> {
        let mut stmt = SelectStmtBuilder::default();

        for inner_pair in pair.into_inner() {
            Self::dispatch_select_clause(inner_pair, &mut stmt)?;
        }

        Ok(stmt.build())
    }

    /// Dispatches a single clause within a SELECT statement to the appropriate parser.
    fn dispatch_select_clause(
        pair: pest::iterators::Pair<Rule>,
        stmt: &mut SelectStmtBuilder,
    ) -> Result<(), ParseError> {
        match pair.as_rule() {
            Rule::distinct_modifier => stmt.distinct = crate::velesql::DistinctMode::All,
            Rule::select_list => stmt.columns = Self::parse_select_list(pair)?,
            Rule::from_clause => Self::dispatch_from_clause(pair, stmt),
            Rule::join_clause => Self::dispatch_join_clause(pair, stmt)?,
            _ => Self::dispatch_optional_clause(pair, stmt)?,
        }
        Ok(())
    }

    fn dispatch_from_clause(pair: pest::iterators::Pair<Rule>, stmt: &mut SelectStmtBuilder) {
        let (table, aliases) = Self::parse_from_clause(pair);
        stmt.from = table;
        stmt.from_alias = aliases;
    }

    fn dispatch_join_clause(
        pair: pest::iterators::Pair<Rule>,
        stmt: &mut SelectStmtBuilder,
    ) -> Result<(), ParseError> {
        let join = Self::parse_join_clause(pair)?;
        // BUG-8: Collect JOIN aliases into from_alias so all
        // aliases visible in scope are available to the executor.
        if let Some(ref alias) = join.alias {
            stmt.from_alias.push(alias.clone());
        }
        stmt.joins.push(join);
        Ok(())
    }

    fn dispatch_optional_clause(
        pair: pest::iterators::Pair<Rule>,
        stmt: &mut SelectStmtBuilder,
    ) -> Result<(), ParseError> {
        match pair.as_rule() {
            Rule::where_clause => stmt.where_clause = Some(Self::parse_where_clause(pair)?),
            Rule::group_by_clause => stmt.group_by = Some(Self::parse_group_by_clause(pair)),
            Rule::having_clause => stmt.having = Some(Self::parse_having_clause(pair)?),
            Rule::order_by_clause => stmt.order_by = Some(Self::parse_order_by_clause(pair)?),
            Rule::limit_clause => stmt.limit = Some(Self::parse_limit_clause(pair)?),
            Rule::offset_clause => stmt.offset = Some(Self::parse_offset_clause(pair)?),
            Rule::with_clause => stmt.with_clause = Some(Self::parse_with_clause(pair)?),
            Rule::using_fusion_clause => {
                stmt.fusion_clause = Some(Self::parse_using_fusion_clause(pair));
            }
            _ => {}
        }
        Ok(())
    }
}

/// Builder accumulator for `SelectStatement` fields during parsing.
struct SelectStmtBuilder {
    distinct: crate::velesql::DistinctMode,
    columns: SelectColumns,
    from: String,
    from_alias: Vec<String>,
    joins: Vec<crate::velesql::JoinClause>,
    where_clause: Option<crate::velesql::Condition>,
    order_by: Option<Vec<crate::velesql::SelectOrderBy>>,
    limit: Option<u64>,
    offset: Option<u64>,
    with_clause: Option<crate::velesql::WithClause>,
    group_by: Option<crate::velesql::GroupByClause>,
    having: Option<crate::velesql::HavingClause>,
    fusion_clause: Option<crate::velesql::FusionClause>,
}

impl Default for SelectStmtBuilder {
    fn default() -> Self {
        Self {
            distinct: crate::velesql::DistinctMode::None,
            columns: SelectColumns::All,
            from: String::new(),
            from_alias: Vec::new(),
            joins: Vec::new(),
            where_clause: None,
            order_by: None,
            limit: None,
            offset: None,
            with_clause: None,
            group_by: None,
            having: None,
            fusion_clause: None,
        }
    }
}

impl SelectStmtBuilder {
    fn build(self) -> SelectStatement {
        SelectStatement {
            distinct: self.distinct,
            columns: self.columns,
            from: self.from,
            from_alias: self.from_alias,
            joins: self.joins,
            where_clause: self.where_clause,
            order_by: self.order_by,
            limit: self.limit,
            offset: self.offset,
            with_clause: self.with_clause,
            group_by: self.group_by,
            having: self.having,
            fusion_clause: self.fusion_clause,
        }
    }
}
