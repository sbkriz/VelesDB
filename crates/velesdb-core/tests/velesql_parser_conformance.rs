#![allow(clippy::doc_markdown)]

use serde::Deserialize;
use std::path::PathBuf;
use velesdb_core::velesql::Parser;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<ParserCase>,
}

#[derive(Debug, Deserialize)]
struct ParserCase {
    id: String,
    query: String,
    should_parse: bool,
    /// Expected from_alias Vec (BUG-8 conformance). When present, the test
    /// verifies the parsed AST contains exactly these aliases in order.
    #[serde(default)]
    from_alias: Option<Vec<String>>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/velesql_parser_cases.json")
}

#[test]
fn test_velesql_parser_conformance_fixture_cases() {
    let content = std::fs::read_to_string(fixture_path()).expect("read parser fixture");
    let fixture: Fixture = serde_json::from_str(&content).expect("parse parser fixture");

    for case in &fixture.cases {
        let parsed = Parser::parse(&case.query);
        assert_eq!(
            parsed.is_ok(),
            case.should_parse,
            "parser conformance failed for case {}",
            case.id
        );

        // BUG-8: Validate from_alias Vec when expected aliases are specified.
        if case.should_parse {
            if let Some(ref expected_aliases) = case.from_alias {
                let query = parsed.unwrap();
                assert_eq!(
                    &query.select.from_alias, expected_aliases,
                    "from_alias mismatch for case {}: expected {:?}, got {:?}",
                    case.id, expected_aliases, query.select.from_alias
                );
            }
        }
    }
}
