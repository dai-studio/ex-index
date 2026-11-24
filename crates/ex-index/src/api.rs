//! `/v1` JSON endpoints and `/healthz`.

use crate::embed::Purpose;
use crate::error::{ApiError, ApiJson, ApiResult};
use crate::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use ex_index_core::embedding::embedding_model;
use ex_index_core::models::{
    validate_id, CreateIndexRequest, IndexConfig, IndexInfo, IndexList, SearchRequest,
    SearchResponse, UpsertRequest, UpsertResponse, VectorRecord, MAX_METADATA_BYTES,
    MAX_UPSERT_BATCH,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::time::Instant;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/indexes", post(create_index).get(list_indexes))
        .route("/v1/indexes/{name}", get(get_index).delete(delete_index))
        .route("/v1/indexes/{name}/vectors", post(upsert))
        .route("/v1/indexes/{name}/documents", post(upsert_documents))
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

/// Max documents per `POST /documents` request.
pub const MAX_DOCUMENT_BATCH: usize = 1000;

/// A text to embed with the index's embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentRecord {
    pub id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpsertDocumentsRequest {
    pub documents: Vec<DocumentRecord>,
}

fn model_for(
    cfg: &IndexConfig,
) -> Result<&'static ex_index_core::embedding::EmbeddingModel, ApiError> {
    let id = cfg.embedding_model.as_deref().ok_or_else(|| {
        ApiError::invalid(format!(
            "index {} has no embedding_model; send vectors instead, or create an index with \"embedding_model\"",
            cfg.name
        ))
    })?;
    embedding_model(id).ok_or_else(|| ApiError::internal(format!("unknown embedding model {id}")))
}

/// Validate documents, embed their texts and return vector records. Validation
/// runs first so a bad batch never costs an embedding call.
pub async fn embed_documents(
    s: &AppState,
    cfg: &IndexConfig,
    docs: Vec<DocumentRecord>,
) -> ApiResult<Vec<VectorRecord>> {
    let model = model_for(cfg)?;
    if docs.len() > MAX_DOCUMENT_BATCH {
        return Err(ApiError::invalid(format!(
            "at most {MAX_DOCUMENT_BATCH} documents per request"
        )));
    }
    for d in &docs {
        validate_id(&d.id).map_err(|e| ApiError::invalid(format!("{e} (id {:?})", d.id)))?;
        if d.text.trim().is_empty() {
            return Err(ApiError::invalid(format!(
                "text must not be empty (id {:?})",
                d.id
            )));
        }
        if let Some(m) = &d.metadata {
            if serde_json::to_vec(m).map(|b| b.len()).unwrap_or(usize::MAX) > MAX_METADATA_BYTES {
                return Err(ApiError::invalid(format!(
                    "metadata exceeds {MAX_METADATA_BYTES} bytes (id {:?})",
                    d.id
                )));
            }
        }
    }
    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let vectors = s.embedder.embed(model, &texts, Purpose::Document).await?;
    Ok(docs
        .into_iter()
        .zip(vectors)
        .map(|(d, vector)| VectorRecord {
            id: d.id,
            vector,
            metadata: d.metadata,
        })
        .collect())
}

/// If the request carries `text`, embed it as a query and put it in `vector`.
pub async fn resolve_query(
    s: &AppState,
    cfg: &IndexConfig,
    req: &mut SearchRequest,
) -> ApiResult<()> {
    let Some(text) = req.text.take() else {
        return Ok(());
    };
    if !req.vector.is_empty() {
        return Err(ApiError::invalid("give either vector or text, not both"));
    }
    if text.trim().is_empty() {
        return Err(ApiError::invalid("text must not be empty"));
    }
    let model = model_for(cfg)?;
    let mut v = s.embedder.embed(model, &[text], Purpose::Query).await?;
    req.vector = v.pop().unwrap_or_default();
    Ok(())
}

async fn upsert_documents(
    State(s): State<AppState>,
    Path(name): Path<String>,
    ApiJson(req): ApiJson<UpsertDocumentsRequest>,
) -> ApiResult<Json<UpsertResponse>> {
    let idx = s.catalog.get(&name)?;
    let records = embed_documents(&s, idx.config(), req.documents).await?;
    let upserted = blocking(move || idx.upsert(records)).await?;
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
    ApiJson(mut req): ApiJson<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let idx = s.catalog.get(&name)?;
    let t0 = Instant::now();
    resolve_query(&s, idx.config(), &mut req).await?;
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
