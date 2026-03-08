#![allow(clippy::pedantic)]

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use tempfile::{NamedTempFile, TempDir};
use velesdb_core::Database;
use velesdb_migrate::config::{
    DestinationConfig, DistanceMetric, MigrationConfig, MigrationOptions, SourceConfig, StorageMode,
};
use velesdb_migrate::connectors::{csv_file::CsvFileConfig, json_file::JsonFileConfig};
use velesdb_migrate::Pipeline;

fn write_json_source(file: &mut NamedTempFile, rows: &[serde_json::Value]) {
    file.as_file_mut().set_len(0).expect("truncate json source");
    file.as_file_mut()
        .seek(SeekFrom::Start(0))
        .expect("seek json source");
    let json = serde_json::to_string(rows).expect("serialize json source");
    file.write_all(json.as_bytes()).expect("write json source");
    file.flush().expect("flush json source");
}

fn write_csv_source(file: &mut NamedTempFile, content: &str) {
    file.as_file_mut().set_len(0).expect("truncate csv source");
    file.as_file_mut()
        .seek(SeekFrom::Start(0))
        .expect("seek csv source");
    file.write_all(content.as_bytes())
        .expect("write csv source");
    file.flush().expect("flush csv source");
}

fn json_config(source_path: &Path, destination_path: &Path, collection: &str) -> MigrationConfig {
    MigrationConfig {
        source: SourceConfig::JsonFile(JsonFileConfig {
            path: source_path.to_path_buf(),
            array_path: String::new(),
            id_field: "id".to_string(),
            vector_field: "vector".to_string(),
            payload_fields: Vec::new(),
        }),
        destination: DestinationConfig {
            path: destination_path.to_path_buf(),
            collection: collection.to_string(),
            dimension: 2,
            metric: DistanceMetric::Cosine,
            storage_mode: StorageMode::Full,
        },
        options: MigrationOptions {
            batch_size: 2,
            workers: 1,
            ..MigrationOptions::default()
        },
    }
}

fn csv_config(source_path: &Path, destination_path: &Path, collection: &str) -> MigrationConfig {
    MigrationConfig {
        source: SourceConfig::CsvFile(CsvFileConfig {
            path: source_path.to_path_buf(),
            id_column: "id".to_string(),
            vector_column: "vector".to_string(),
            vector_spread: false,
            dim_prefix: "dim_".to_string(),
            delimiter: ',',
            has_header: true,
        }),
        destination: DestinationConfig {
            path: destination_path.to_path_buf(),
            collection: collection.to_string(),
            dimension: 2,
            metric: DistanceMetric::Cosine,
            storage_mode: StorageMode::Full,
        },
        options: MigrationOptions {
            batch_size: 2,
            workers: 1,
            ..MigrationOptions::default()
        },
    }
}

fn open_collection(db_path: &Path, name: &str) -> velesdb_core::Collection {
    Database::open(db_path)
        .expect("open database")
        .get_collection(name)
        .expect("collection exists")
}

#[tokio::test]
async fn test_pipeline_json_e2e_persists_and_reads_back() {
    let source = TempDir::new().expect("source dir");
    let destination = TempDir::new().expect("destination dir");
    let mut file = NamedTempFile::new_in(source.path()).expect("json source file");
    write_json_source(
        &mut file,
        &[
            serde_json::json!({"id": 1, "vector": [1.0, 0.0], "title": "Doc 1"}),
            serde_json::json!({"id": 2, "vector": [0.0, 1.0], "title": "Doc 2"}),
        ],
    );

    let config = json_config(file.path(), destination.path(), "json_docs");
    let mut pipeline = Pipeline::new(config).expect("pipeline");
    let stats = pipeline.run().await.expect("pipeline run");

    assert_eq!(stats.extracted, 2);
    assert_eq!(stats.loaded, 2);
    assert_eq!(stats.failed, 0);

    let collection = open_collection(destination.path(), "json_docs");
    assert_eq!(collection.len(), 2);

    let points = collection.get(&[1, 2]);
    assert_eq!(
        points[0]
            .as_ref()
            .and_then(|point| point.payload.clone())
            .expect("payload")["title"],
        "Doc 1"
    );
    assert_eq!(
        points[1]
            .as_ref()
            .and_then(|point| point.payload.clone())
            .expect("payload")["title"],
        "Doc 2"
    );

    let mut ids = collection.all_ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2]);
}

#[tokio::test]
async fn test_pipeline_csv_e2e_persists_and_reads_back() {
    let source = TempDir::new().expect("source dir");
    let destination = TempDir::new().expect("destination dir");
    let mut file = NamedTempFile::new_in(source.path()).expect("csv source file");
    write_csv_source(
        &mut file,
        "id,vector,title\n1,[0.1, 0.2],CSV Doc 1\n2,[0.2, 0.1],CSV Doc 2",
    );

    let config = csv_config(file.path(), destination.path(), "csv_docs");
    let mut pipeline = Pipeline::new(config).expect("pipeline");
    let stats = pipeline.run().await.expect("pipeline run");

    assert_eq!(stats.loaded, 2);

    let collection = open_collection(destination.path(), "csv_docs");
    assert_eq!(collection.len(), 2);
    let points = collection.get(&[1, 2]);
    assert_eq!(
        points[0]
            .as_ref()
            .and_then(|point| point.payload.clone())
            .expect("payload")["title"],
        "CSV Doc 1"
    );
}

#[tokio::test]
async fn test_pipeline_dry_run_does_not_create_collection() {
    let source = TempDir::new().expect("source dir");
    let destination = TempDir::new().expect("destination dir");
    let mut file = NamedTempFile::new_in(source.path()).expect("json source file");
    write_json_source(
        &mut file,
        &[serde_json::json!({"id": 1, "vector": [0.1, 0.2], "title": "Dry Run"})],
    );

    let mut config = json_config(file.path(), destination.path(), "dry_run_docs");
    config.options.dry_run = true;

    let mut pipeline = Pipeline::new(config).expect("pipeline");
    let stats = pipeline.run().await.expect("dry run");

    assert_eq!(stats.loaded, 1);
    assert!(Database::open(destination.path())
        .expect("open database")
        .get_collection("dry_run_docs")
        .is_none());
}

#[tokio::test]
async fn test_pipeline_continue_on_error_skips_invalid_point() {
    let source = TempDir::new().expect("source dir");
    let destination = TempDir::new().expect("destination dir");
    let mut file = NamedTempFile::new_in(source.path()).expect("json source file");
    write_json_source(
        &mut file,
        &[
            serde_json::json!({"id": 1, "vector": [0.1, 0.2], "title": "Valid"}),
            serde_json::json!({"id": 2, "vector": [0.1, 0.2, 0.3], "title": "Invalid"}),
        ],
    );

    let mut config = json_config(file.path(), destination.path(), "fallback_docs");
    config.options.continue_on_error = true;

    let mut pipeline = Pipeline::new(config).expect("pipeline");
    let stats = pipeline.run().await.expect("pipeline run");

    assert_eq!(stats.loaded, 1);
    assert_eq!(stats.failed, 1);

    let collection = open_collection(destination.path(), "fallback_docs");
    assert_eq!(collection.len(), 1);
    assert!(collection.get(&[1])[0].is_some());
    assert!(collection.get(&[2])[0].is_none());
}

#[tokio::test]
async fn test_pipeline_checkpoint_resumes_from_last_offset() {
    let source = TempDir::new().expect("source dir");
    let destination = TempDir::new().expect("destination dir");
    let mut file = NamedTempFile::new_in(source.path()).expect("json source file");
    let checkpoint_path = destination.path().join("resume-checkpoint.json");

    write_json_source(
        &mut file,
        &[
            serde_json::json!({"id": 1, "vector": [0.1, 0.2], "title": "First"}),
            serde_json::json!({"id": 2, "vector": [0.1, 0.2, 0.3], "title": "Broken"}),
            serde_json::json!({"id": 3, "vector": [0.2, 0.1], "title": "Third"}),
        ],
    );

    let mut config = json_config(file.path(), destination.path(), "resume_docs");
    config.options.batch_size = 1;
    config.options.checkpoint_path = Some(checkpoint_path.clone());

    let mut first_run = Pipeline::new(config.clone()).expect("pipeline");
    let error = first_run.run().await.expect_err("resume should fail");
    assert!(matches!(error, velesdb_migrate::Error::Loading(_)));
    assert!(checkpoint_path.exists());

    write_json_source(
        &mut file,
        &[
            serde_json::json!({"id": 1, "vector": [0.1, 0.2], "title": "First"}),
            serde_json::json!({"id": 2, "vector": [0.2, 0.2], "title": "Second"}),
            serde_json::json!({"id": 3, "vector": [0.2, 0.1], "title": "Third"}),
        ],
    );

    let mut resumed_run = Pipeline::new(config).expect("pipeline");
    let stats = resumed_run.run().await.expect("resume succeeds");

    assert_eq!(stats.loaded, 3);
    assert_eq!(stats.failed, 0);
    assert!(!checkpoint_path.exists());

    let collection = open_collection(destination.path(), "resume_docs");
    assert_eq!(collection.len(), 3);
}

#[tokio::test]
async fn test_pipeline_workers_match_sequential_results_and_text_ids_stable() {
    let source = TempDir::new().expect("source dir");
    let destination_one = TempDir::new().expect("destination dir one");
    let destination_two = TempDir::new().expect("destination dir two");
    let mut file = NamedTempFile::new_in(source.path()).expect("json source file");
    write_json_source(
        &mut file,
        &[
            serde_json::json!({"id": "alpha", "vector": [1.0, 0.0], "title": "Alpha"}),
            serde_json::json!({"id": "beta", "vector": [0.0, 1.0], "title": "Beta"}),
            serde_json::json!({"id": "gamma", "vector": [0.5, 0.5], "title": "Gamma"}),
        ],
    );

    let mut sequential_config = json_config(file.path(), destination_one.path(), "worker_docs");
    sequential_config.options.workers = 1;

    let mut parallel_config = json_config(file.path(), destination_two.path(), "worker_docs");
    parallel_config.options.workers = 4;

    Pipeline::new(sequential_config)
        .expect("pipeline")
        .run()
        .await
        .expect("sequential run");
    Pipeline::new(parallel_config)
        .expect("pipeline")
        .run()
        .await
        .expect("parallel run");

    let sequential = open_collection(destination_one.path(), "worker_docs");
    let parallel = open_collection(destination_two.path(), "worker_docs");

    let mut sequential_ids = sequential.all_ids();
    let mut parallel_ids = parallel.all_ids();
    sequential_ids.sort_unstable();
    parallel_ids.sort_unstable();

    assert_eq!(sequential_ids, parallel_ids);

    assert_eq!(sequential_ids.len(), 3);
}
