//! `ex-index` server: versioned JSON REST API (`/v1`), server-rendered admin
//! UI (`/ui`), embedded assets, `/healthz` and Prometheus `/metrics`, all
//! backed by the `ex-index-core` store.

pub mod api;
pub mod assets;
pub mod auth;
pub mod config;
pub mod error;
pub mod telemetry;
pub mod ui;

use axum::extract::DefaultBodyLimit;
use axum::routing::get;
use axum::Router;
use ex_index_core::store::{Catalog, StoreOptions};
use std::future::Future;
use std::sync::Arc;
use tower_http::compression::CompressionLayer;
use tower_http::trace::TraceLayer;

pub use config::{Cli, Command, ServeArgs};

/// Max request body (64 MiB).
pub const BODY_LIMIT: usize = 64 * 1024 * 1024;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Shared handler state.
#[derive(Clone)]
pub struct AppState {
    pub catalog: Arc<Catalog>,
    pub auth: Arc<auth::Auth>,
    pub metrics: metrics_exporter_prometheus::PrometheusHandle,
}

/// Build the full router.
pub fn router(state: AppState) -> Router {
    let v1 = api::routes().route_layer(axum::middleware::from_fn_with_state(
        state.clone(),
        auth::require_api,
    ));
    let metrics = Router::new()
        .route("/metrics", get(telemetry::render))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_api,
        ));
    let ui = ui::routes().route_layer(axum::middleware::from_fn_with_state(
        state.clone(),
        auth::require_ui,
    ));
    Router::new()
        .merge(v1)
        .merge(metrics)
        .merge(ui)
        .merge(ui::public_routes())
        .route("/healthz", get(api::healthz))
        .route("/assets/{*path}", get(assets::serve))
        .route("/", get(|| async { axum::response::Redirect::to("/ui") }))
        .fallback(api::not_found)
        .layer(axum::middleware::from_fn(telemetry::track))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Open the catalog and serve on `listener` until `shutdown` resolves; then
/// drain in-flight requests, fsync WALs and close every index.
pub async fn serve(
    args: ServeArgs,
    listener: tokio::net::TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), BoxError> {
    let opts = StoreOptions {
        small_index_threshold: args.small_index_threshold,
        delta_threshold: args.delta_threshold,
        ..StoreOptions::default()
    };
    let data_dir = args.data_dir.clone();
    let catalog =
        Arc::new(tokio::task::spawn_blocking(move || Catalog::open(data_dir, opts)).await??);
    let state = AppState {
        catalog: catalog.clone(),
        auth: Arc::new(auth::Auth::new(args.api_key.clone())?),
        metrics: telemetry::handle(),
    };
    tracing::info!(
        addr = %listener.local_addr()?,
        data_dir = %args.data_dir.display(),
        auth = state.auth.enabled(),
        indexes = catalog.list().len(),
        "ex-index {VERSION} listening"
    );
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await?;
    tracing::info!("shutting down: flushing indexes");
    tokio::task::spawn_blocking(move || catalog.shutdown()).await?;
    Ok(())
}

/// Resolves on SIGINT (Ctrl-C) or SIGTERM.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("shutdown signal received");
}
