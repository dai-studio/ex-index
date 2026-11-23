//! `/v1` JSON endpoints and `/healthz`.

use crate::error::{ApiError, ApiJson, ApiResult};
use crate::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use ex_index_core::models::{
    CreateIndexRequest, IndexInfo, IndexList, SearchRequest, SearchResponse, UpsertRequest,
    UpsertResponse, VectorRecord, MAX_UPSERT_BATCH,
};
use serde_json::json;
use std::time::Instant;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/indexes", post(create_index).get(list_indexes))
        .route("/v1/indexes/{name}", get(get_index).delete(delete_index))
        .route("/v1/indexes/{name}/vectors", post(upsert))
        .route(
            "/v1/indexes/{name}/vectors/{id}",
            get(get_vector).delete(delete_vector),
        )
        .route("/v1/indexes/{name}/search", post(search))
        .route("/v1/indexes/{name}/build", post(build))
}

/// Run blocking store work off the async executor.
pub async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, ex_index_core::Error> + Send + 'static,
) -> ApiResult<T> {
    Ok(tokio::task::spawn_blocking(f).await??)
}

async fn create_index(
    State(s): State<AppState>,
    ApiJson(req): ApiJson<CreateIndexRequest>,
) -> ApiResult<(StatusCode, Json<IndexInfo>)> {
    let cat = s.catalog.clone();
    let info = blocking(move || cat.create(req).map(|i| i.info())).await?;
    Ok((StatusCode::CREATED, Json(info)))
}

async fn list_indexes(State(s): State<AppState>) -> ApiResult<Json<IndexList>> {
    let cat = s.catalog.clone();
    let indexes = blocking(move || Ok(cat.list().iter().map(|i| i.info()).collect())).await?;
    Ok(Json(IndexList { indexes }))
}

async fn get_index(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<IndexInfo>> {
    let cat = s.catalog.clone();
    Ok(Json(
        blocking(move || cat.get(&name).map(|i| i.info())).await?,
    ))
}

async fn delete_index(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
    let cat = s.catalog.clone();
    blocking(move || cat.delete(&name)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn upsert(
    State(s): State<AppState>,
    Path(name): Path<String>,
    ApiJson(req): ApiJson<UpsertRequest>,
) -> ApiResult<Json<UpsertResponse>> {
    if req.vectors.len() > MAX_UPSERT_BATCH {
        return Err(ApiError::invalid(format!(
            "at most {MAX_UPSERT_BATCH} vectors per request"
        )));
    }
    let idx = s.catalog.get(&name)?;
    let upserted = blocking(move || idx.upsert(req.vectors)).await?;
    metrics::counter!("ex_index_vectors_upserted_total", "index" => name)
        .increment(upserted as u64);
    Ok(Json(UpsertResponse { upserted }))
}

async fn get_vector(
    State(s): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> ApiResult<Json<VectorRecord>> {
    let idx = s.catalog.get(&name)?;
    Ok(Json(blocking(move || idx.get(&id)).await?))
}

async fn delete_vector(
    State(s): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let idx = s.catalog.get(&name)?;
    blocking(move || idx.delete(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn search(
    State(s): State<AppState>,
    Path(name): Path<String>,
    ApiJson(req): ApiJson<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let idx = s.catalog.get(&name)?;
    let t0 = Instant::now();
    let hits = blocking(move || idx.search(&req)).await?;
    let took_ms = t0.elapsed().as_secs_f64() * 1000.0;
    metrics::histogram!("ex_index_search_seconds", "index" => name).record(took_ms / 1000.0);
    Ok(Json(SearchResponse { hits, took_ms }))
}

async fn build(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<(StatusCode, Json<IndexInfo>)> {
    let idx = s.catalog.get(&name)?;
    let info = blocking(move || {
        idx.start_build()?;
        Ok(idx.info())
    })
    .await?;
    Ok((StatusCode::ACCEPTED, Json(info)))
}

pub async fn healthz() -> impl IntoResponse {
    Json(json!({"status": "ok", "version": crate::VERSION}))
}

pub async fn not_found() -> ApiError {
    ApiError::not_found("no such route")
}
