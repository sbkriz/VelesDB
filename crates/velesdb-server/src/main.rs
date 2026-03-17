#![allow(clippy::doc_markdown)]
//! `VelesDB` Server - REST API for the `VelesDB` vector database.

use axum::{
    extract::DefaultBodyLimit,
    routing::{delete, get, post},
    Router,
};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(feature = "swagger-ui")]
use utoipa::OpenApi;
#[cfg(feature = "swagger-ui")]
use utoipa_swagger_ui::SwaggerUi;
use velesdb_core::Database;
#[cfg(feature = "swagger-ui")]
use velesdb_server::ApiDoc;
use velesdb_server::{
    add_edge, aggregate, analyze_collection, batch_search, collection_sanity,
    config::{parse_api_keys_env, CliOverrides, ServerConfig},
    create_collection, create_index, delete_collection, delete_index, delete_point, explain,
    flush_collection, get_collection, get_collection_config, get_collection_stats, get_edges,
    get_guardrails, get_node_degree, get_point, health_check, hybrid_search, is_empty,
    list_collections, list_indexes, match_query, multi_query_search, query, search, search_ids,
    stream_insert, stream_traverse, stream_upsert_points, text_search, traverse_graph,
    update_guardrails, upsert_points, AppState, OnboardingMetrics,
};

/// VelesDB Server - A high-performance vector database
#[derive(Parser, Debug)]
#[command(name = "velesdb-server")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to velesdb.toml configuration file
    #[arg(short, long, env = "VELESDB_CONFIG")]
    config: Option<PathBuf>,

    /// Data directory for persistent storage
    #[arg(short, long, env = "VELESDB_DATA_DIR")]
    data_dir: Option<String>,

    /// Host address to bind to
    #[arg(long, env = "VELESDB_HOST")]
    host: Option<String>,

    /// Port to listen on
    #[arg(short, long, env = "VELESDB_PORT")]
    port: Option<u16>,

    /// TLS certificate file (PEM)
    #[arg(long, env = "VELESDB_TLS_CERT")]
    tls_cert: Option<String>,

    /// TLS private key file (PEM)
    #[arg(long, env = "VELESDB_TLS_KEY")]
    tls_key: Option<String>,
}

fn configure_tracing() {
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .try_init();
}

fn log_startup(cfg: &ServerConfig) {
    tracing::info!("Starting VelesDB server...");
    tracing::info!("Data directory: {}", cfg.data_dir);
    tracing::info!("Bind address: {}:{}", cfg.host, cfg.port);
    if cfg.auth_enabled() {
        tracing::info!(
            "API key authentication enabled ({} key(s))",
            cfg.api_keys.len()
        );
    } else {
        tracing::info!("API key authentication disabled (local dev mode)");
    }
    if cfg.tls_enabled() {
        tracing::info!("TLS enabled");
    }
}

fn init_app_state(data_dir: &str) -> anyhow::Result<Arc<AppState>> {
    let db = Database::open(data_dir)?;
    Ok(Arc::new(AppState {
        db,
        onboarding_metrics: OnboardingMetrics::default(),
        query_limits: parking_lot::RwLock::new(velesdb_core::guardrails::QueryLimits::default()),
    }))
}

fn build_router(state: Arc<AppState>) -> Router {
    let api_router = Router::new()
        .route("/health", get(health_check))
        .route(
            "/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/collections/{name}",
            get(get_collection).delete(delete_collection),
        )
        .route("/collections/{name}/empty", get(is_empty))
        .route("/collections/{name}/config", get(get_collection_config))
        .route("/collections/{name}/sanity", get(collection_sanity))
        .route("/collections/{name}/flush", post(flush_collection))
        .route("/collections/{name}/analyze", post(analyze_collection))
        .route("/collections/{name}/stats", get(get_collection_stats))
        .route("/guardrails", get(get_guardrails).put(update_guardrails))
        // 100MB limit scoped to batch vector upload routes only
        // (1000 vectors × 768D × 4 bytes = ~3MB typical; 100MB covers extreme cases)
        .merge(
            Router::new()
                .route("/collections/{name}/points", post(upsert_points))
                .route(
                    "/collections/{name}/points/stream",
                    post(stream_upsert_points),
                )
                .layer(DefaultBodyLimit::max(100 * 1024 * 1024)),
        )
        .route("/collections/{name}/stream/insert", post(stream_insert))
        .route(
            "/collections/{name}/points/{id}",
            get(get_point).delete(delete_point),
        )
        .route("/collections/{name}/search", post(search))
        .route("/collections/{name}/search/batch", post(batch_search))
        .route("/collections/{name}/search/multi", post(multi_query_search))
        .route("/collections/{name}/search/text", post(text_search))
        .route("/collections/{name}/search/hybrid", post(hybrid_search))
        .route("/collections/{name}/search/ids", post(search_ids))
        .route(
            "/collections/{name}/indexes",
            get(list_indexes).post(create_index),
        )
        .route(
            "/collections/{name}/indexes/{label}/{property}",
            delete(delete_index),
        )
        .route("/query", post(query))
        .route("/aggregate", post(aggregate))
        .route("/query/explain", post(explain))
        .route("/collections/{name}/match", post(match_query))
        .route(
            "/collections/{name}/graph/edges",
            get(get_edges).post(add_edge),
        )
        .route("/collections/{name}/graph/traverse", post(traverse_graph))
        .route(
            "/collections/{name}/graph/traverse/stream",
            get(stream_traverse),
        )
        .route(
            "/collections/{name}/graph/nodes/{node_id}/degree",
            get(get_node_degree),
        );

    #[cfg(feature = "prometheus")]
    let api_router = {
        use velesdb_server::prometheus_metrics;
        api_router.route("/metrics", get(prometheus_metrics))
    };

    let api_router = api_router.with_state(state);

    #[cfg(feature = "swagger-ui")]
    let api_router = {
        let swagger_ui =
            SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi());
        api_router.merge(Router::<()>::new().merge(swagger_ui))
    };

    api_router
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

async fn serve(host: &str, port: u16, app: Router) -> anyhow::Result<()> {
    if host != "127.0.0.1" && host != "localhost" {
        tracing::warn!(
            "VelesDB server exposed on network ({host}). \
             Consider using 127.0.0.1 for local-first usage."
        );
    }
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("VelesDB server listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_cli_overrides(args: Args) -> CliOverrides {
    CliOverrides {
        config_path: args.config,
        host: args.host,
        port: args.port,
        data_dir: args.data_dir,
        api_keys: parse_api_keys_env(),
        tls_cert: args.tls_cert,
        tls_key: args.tls_key,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    configure_tracing();

    let args = Args::parse();
    let cli = build_cli_overrides(args);
    let cfg = ServerConfig::load(cli)?;
    cfg.validate()?;

    log_startup(&cfg);

    let state = init_app_state(&cfg.data_dir)?;
    let app = build_router(state);

    serve(&cfg.host, cfg.port, app).await
}
