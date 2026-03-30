//! Admin statement parsing (FLUSH).

use super::{extract_identifier, Rule};
use crate::velesql::ast::{AdminStatement, FlushStatement, Query};
use crate::velesql::error::ParseError;
use crate::velesql::Parser;

impl Parser {
    /// Parses a `FLUSH [FULL] [collection]` statement.
    ///
    /// Grammar:
    /// ```text
    /// FLUSH [FULL] [identifier]
    /// ```
    ///
    /// The `flush_full_kw` named rule emits a token when FULL is present,
    /// allowing reliable detection via `as_rule()`. A negative lookahead
    /// in the grammar prevents "fulltext" from matching as `FULL` + `text`.
    ///
    /// Returns `Result` for `dispatch_statement()` uniformity even though
    /// this variant never fails in practice.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn parse_flush_stmt(pair: pest::iterators::Pair<Rule>) -> Result<Query, ParseError> {
        let mut full = false;
        let mut collection: Option<String> = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::flush_full_kw => full = true,
                Rule::identifier => collection = Some(extract_identifier(&inner)),
                _ => {}
            }
        }

        Ok(Query::new_admin(AdminStatement::Flush(FlushStatement {
            full,
            collection,
        })))
    }
}
