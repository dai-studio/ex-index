//! Prometheus metrics: request counts / latency and per-index vector counts.

use crate::AppState;
use axum::extract::{MatchedPath, Request, State};
use axum::http::header;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;
use std::time::Instant;

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Process-wide recorder (installed once, shared by every server instance).
pub fn handle() -> PrometheusHandle {
    HANDLE
        .get_or_init(|| {
            let recorder = PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            if metrics::set_global_recorder(recorder).is_err() {
                tracing::warn!("a metrics recorder was already installed");
            }
            handle
        })
        .clone()
}

/// Middleware recording `http_requests_total` and `http_request_duration_seconds`.
pub async fn track(req: Request, next: Next) -> Response {
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "unmatched".to_string());
    let method = req.method().to_string();
    let t0 = Instant::now();
    let resp = next.run(req).await;
    let status = resp.status().as_u16().to_string();
    metrics::counter!("http_requests_total", "method" => method.clone(), "route" => route.clone(), "status" => status)
        .increment(1);
    metrics::histogram!("http_request_duration_seconds", "method" => method, "route" => route)
        .record(t0.elapsed().as_secs_f64());
    resp
}

pub async fn render(State(s): State<AppState>) -> impl IntoResponse {
    for idx in s.catalog.list() {
        let info = idx.info();
        let name = info.config.name.clone();
        metrics::gauge!("ex_index_vectors", "index" => name.clone()).set(info.count as f64);
        metrics::gauge!("ex_index_deleted_vectors", "index" => name.clone())
            .set(info.deleted as f64);
        metrics::gauge!("ex_index_delta_vectors", "index" => name.clone())
            .set(info.delta_count as f64);
        metrics::gauge!("ex_index_size_bytes", "index" => name.clone()).set(info.size_bytes as f64);
        metrics::gauge!("ex_index_building", "index" => name).set(if idx.is_building() {
            1.0
        } else {
            0.0
        });
    }
    metrics::gauge!("ex_index_indexes").set(s.catalog.list().len() as f64);
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        s.metrics.render(),
    )
}
