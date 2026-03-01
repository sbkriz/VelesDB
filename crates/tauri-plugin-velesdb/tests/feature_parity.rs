use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use toml::Value;

fn cargo_features(path: &Path) -> BTreeMap<String, Vec<String>> {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    let parsed: Value = toml::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()));

    let feature_table = parsed
        .get("features")
        .and_then(Value::as_table)
        .unwrap_or_else(|| panic!("missing [features] table in {}", path.display()));

    feature_table
        .iter()
        .map(|(feature, values)| {
            let values = values
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToOwned::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            (feature.clone(), values)
        })
        .collect()
}

fn core_features_without_default(
    core_features: &BTreeMap<String, Vec<String>>,
) -> BTreeSet<String> {
    core_features
        .keys()
        .filter(|name| name.as_str() != "default")
        .cloned()
        .collect()
}

#[test]
fn tauri_plugin_exposes_all_velesdb_core_features() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let core_manifest = manifest_dir.join("../velesdb-core/Cargo.toml");
    let plugin_manifest = manifest_dir.join("Cargo.toml");

    let core_features = cargo_features(&core_manifest);
    let plugin_features = cargo_features(&plugin_manifest);

    let expected = core_features_without_default(&core_features);
    let actual = plugin_features
        .keys()
        .filter(|name| name.as_str() != "default")
        .cloned()
        .collect::<BTreeSet<_>>();

    assert_eq!(
        actual, expected,
        "tauri-plugin-velesdb must expose every non-default velesdb-core feature"
    );
}

#[test]
fn tauri_plugin_features_forward_to_velesdb_core() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let core_manifest = manifest_dir.join("../velesdb-core/Cargo.toml");
    let plugin_manifest = manifest_dir.join("Cargo.toml");

    let core_features = cargo_features(&core_manifest);
    let plugin_features = cargo_features(&plugin_manifest);

    for feature in core_features.keys() {
        if feature == "default" {
            continue;
        }

        let forwarded = plugin_features
            .get(feature)
            .unwrap_or_else(|| panic!("missing feature '{feature}' in tauri plugin"));

        assert!(
            forwarded.contains(&format!("velesdb-core/{feature}")),
            "feature '{feature}' in tauri plugin must forward to velesdb-core/{feature}"
        );
    }
}

#[test]
fn tauri_plugin_default_feature_forwards_to_velesdb_core_default() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let plugin_manifest = manifest_dir.join("Cargo.toml");
    let plugin_features = cargo_features(&plugin_manifest);

    let default_features = plugin_features
        .get("default")
        .unwrap_or_else(|| panic!("missing default feature in tauri plugin"));

    assert!(
        default_features.contains(&"velesdb-core/default".to_string()),
        "tauri plugin default feature must include velesdb-core/default"
    );
}
