//! Introspection statement parsing (SHOW COLLECTIONS, DESCRIBE, EXPLAIN).

use super::{extract_identifier, Rule};
use crate::velesql::ast::{DescribeCollectionStatement, IntrospectionStatement, Query};
use crate::velesql::error::ParseError;
use crate::velesql::Parser;

impl Parser {
    /// Parses a `SHOW COLLECTIONS` statement.
    ///
    /// Grammar:
    /// ```text
    /// SHOW COLLECTIONS
    /// ```
    ///
    /// Returns `Result` for `dispatch_statement()` uniformity even though
    /// this particular variant never fails.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn parse_show_collections_stmt(
        _pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        Ok(Query::new_introspection(
            IntrospectionStatement::ShowCollections,
        ))
    }

    /// Parses a `DESCRIBE [COLLECTION] <name>` statement.
    ///
    /// Grammar:
    /// ```text
    /// DESCRIBE [COLLECTION] identifier
    /// ```
    pub(crate) fn parse_describe_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        let name = pair
            .into_inner()
            .find(|p| p.as_rule() == Rule::identifier)
            .map(|p| extract_identifier(&p))
            .ok_or_else(|| ParseError::syntax(0, "", "DESCRIBE requires a collection name"))?;

        Ok(Query::new_introspection(
            IntrospectionStatement::DescribeCollection(DescribeCollectionStatement { name }),
        ))
    }

    /// Parses an `EXPLAIN <query>` statement.
    ///
    /// Grammar:
    /// ```text
    /// EXPLAIN compound_query
    /// ```
    pub(crate) fn parse_explain_stmt(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Query, ParseError> {
        let compound_pair = pair
            .into_inner()
            .find(|p| p.as_rule() == Rule::compound_query)
            .ok_or_else(|| {
                ParseError::syntax(0, "", "EXPLAIN requires a query (e.g. EXPLAIN SELECT ...)")
            })?;

        let inner_query = Self::parse_compound_query(compound_pair)?;

        Ok(Query::new_introspection(IntrospectionStatement::Explain(
            Box::new(inner_query),
        )))
    }
}
