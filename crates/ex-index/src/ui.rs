//! Server-rendered admin UI (askama + shadcn-html). Form posts call the same
//! store functions as the REST API, then redirect (PRG) with a flash message.

use crate::api::{
    embed_documents, ms_since, resolve_query, DocumentRecord, UpsertDocumentsRequest,
};
use crate::auth::Auth;
use crate::AppState;
use askama::Template;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use ex_index_core::embedding::EMBEDDING_MODELS;
use ex_index_core::models::{
    CreateIndexRequest, IndexParamsInput, SearchRequest, UpsertRequest, VectorRecord,
};
use ex_index_core::Metric;
use std::collections::HashMap;
use std::time::Instant;

type Params = HashMap<String, String>;

/// Routes behind the UI auth check.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/ui", get(indexes_page))
        .route("/ui/new", get(new_page).post(new_submit))
        .route("/ui/indexes/{name}", get(detail_page))
        .route("/ui/indexes/{name}/upsert", post(upsert_submit))
        .route("/ui/indexes/{name}/documents", post(documents_submit))
        .route(
            "/ui/indexes/{name}/delete-vector",
            post(delete_vector_submit),
        )
        .route("/ui/indexes/{name}/delete", post(delete_index_submit))
        .route("/ui/indexes/{name}/build", post(build_submit))
        .route(
            "/ui/indexes/{name}/search",
            get(search_page).post(search_submit),
        )
        .route("/ui/logout", post(logout))
}

/// Routes reachable without a session (login form).
pub fn public_routes() -> Router<AppState> {
    Router::new().route("/ui/login", get(login_page).post(login_submit))
}

/// Fields shared by every page (rendered by base.html).
pub struct Layout {
    pub version: &'static str,
    pub csrf: String,
    pub auth_enabled: bool,
    pub flash_msg: String,
    pub flash_err: String,
}

fn layout(s: &AppState, headers: &HeaderMap, q: &Params) -> Layout {
    Layout {
        version: crate::VERSION,
        csrf: s.auth.csrf_token(headers),
        auth_enabled: s.auth.enabled() && s.auth.check_cookie(headers),
        flash_msg: q.get("msg").cloned().unwrap_or_default(),
        flash_err: q.get("err").cloned().unwrap_or_default(),
    }
}

fn render(t: impl Template) -> Response {
    match t.render() {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "template render failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "template error").into_response()
        }
    }
}

/// Percent-encode a query-string value.
pub fn qenc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn back(path: &str, msg: Result<String, String>) -> Response {
    let (k, v) = match msg {
        Ok(m) => ("msg", m),
        Err(e) => ("err", e),
    };
    let sep = if path.contains('?') { '&' } else { '?' };
    Redirect::to(&format!("{path}{sep}{k}={}", qenc(&v))).into_response()
}

fn csrf_ok(s: &AppState, headers: &HeaderMap, form: &Params) -> bool {
    form.get("csrf")
        .map(|t| s.auth.check_csrf(headers, t))
        .unwrap_or(false)
}

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        "invalid or missing CSRF token; reload the page and try again",
    )
        .into_response()
}

async fn run<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, ex_index_core::Error> + Send + 'static,
) -> Result<T, String> {
    match tokio::task::spawn_blocking(f).await {
        Ok(r) => r.map_err(|e| e.to_string()),
        Err(e) => Err(format!("internal error: {e}")),
    }
}

fn human_bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}

// ---------------------------------------------------------------- indexes

pub struct IndexRow {
    pub name: String,
    pub metric: &'static str,
    pub dimension: u32,
    pub count: u64,
    pub building: bool,
}

#[derive(Template)]
#[template(path = "indexes.html")]
struct IndexesPage {
    layout: Layout,
    rows: Vec<IndexRow>,
}

async fn indexes_page(
    State(s): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<Params>,
) -> Response {
    let cat = s.catalog.clone();
    let rows = run(move || {
        Ok(cat
            .list()
            .iter()
            .map(|i| {
                let info = i.info();
                IndexRow {
                    name: info.config.name,
                    metric: info.config.metric.as_str(),
                    dimension: info.config.dimension,
                    count: info.count,
                    building: i.is_building(),
                }
            })
            .collect())
    })
    .await
    .unwrap_or_default();
    render(IndexesPage {
        layout: layout(&s, &headers, &q),
        rows,
    })
}

// ---------------------------------------------------------------- create

struct NewPage;

pub struct ModelOpt {
    pub id: &'static str,
    pub dimension: u32,
}

fn model_opts() -> Vec<ModelOpt> {
    EMBEDDING_MODELS
        .iter()
        .map(|m| ModelOpt {
            id: m.id,
            dimension: m.dimension,
        })
        .collect()
}

impl NewPage {
    fn with(layout: Layout, f: &Params) -> NewPageView {
        let g = |k: &str| f.get(k).cloned().unwrap_or_default();
        NewPageView {
            layout,
            name: g("name"),
            dimension: g("dimension"),
            metric: f.get("metric").cloned().unwrap_or_else(|| "cosine".into()),
            embedding_model: g("embedding_model"),
            models: model_opts(),
            r: g("r"),
            l_build: g("l_build"),
            alpha: g("alpha"),
            pq_subvectors: g("pq_subvectors"),
            l_search: g("l_search"),
        }
    }
}

#[derive(Template)]
#[template(path = "new.html")]
struct NewPageView {
    layout: Layout,
    name: String,
    dimension: String,
    metric: String,
    embedding_model: String,
    models: Vec<ModelOpt>,
    r: String,
    l_build: String,
    alpha: String,
    pq_subvectors: String,
    l_search: String,
}

async fn new_page(
    State(s): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<Params>,
) -> Response {
    render(NewPage::with(layout(&s, &headers, &q), &q))
}

fn opt_num<T: std::str::FromStr>(f: &Params, k: &str) -> Result<Option<T>, String> {
    match f.get(k).map(|v| v.trim()).filter(|v| !v.is_empty()) {
        None => Ok(None),
        Some(v) => v
            .parse()
            .map(Some)
            .map_err(|_| format!("{k} must be a number")),
    }
}

fn parse_create(f: &Params) -> Result<CreateIndexRequest, String> {
    let metric = f
        .get("metric")
        .and_then(|m| Metric::parse(m))
        .ok_or("metric must be cosine, euclidean or dot")?;
    let embedding_model = f
        .get("embedding_model")
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty());
    let dimension = opt_num::<u32>(f, "dimension")?;
    if dimension.is_none() && embedding_model.is_none() {
        return Err("dimension is required unless an embedding model is chosen".into());
    }
    Ok(CreateIndexRequest {
        name: f
            .get("name")
            .cloned()
            .unwrap_or_default()
            .trim()
            .to_string(),
        dimension,
        metric,
        params: Some(IndexParamsInput {
            r: opt_num(f, "r")?,
            l_build: opt_num(f, "l_build")?,
            alpha: opt_num(f, "alpha")?,
            pq_subvectors: opt_num(f, "pq_subvectors")?,
            l_search: opt_num(f, "l_search")?,
        }),
        embedding_model,
    })
}

async fn new_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let req = match parse_create(&f) {
        Ok(r) => r,
        Err(e) => {
            let mut l = layout(&s, &headers, &Params::new());
            l.flash_err = e;
            return (StatusCode::BAD_REQUEST, render(NewPage::with(l, &f))).into_response();
        }
    };
    let name = req.name.clone();
    let cat = s.catalog.clone();
    match run(move || cat.create(req).map(|_| ())).await {
        Ok(()) => back(
            &format!("/ui/indexes/{name}"),
            Ok(format!("Index {name} created.")),
        ),
        Err(e) => {
            let mut l = layout(&s, &headers, &Params::new());
            l.flash_err = e;
            (StatusCode::BAD_REQUEST, render(NewPage::with(l, &f))).into_response()
        }
    }
}

// ---------------------------------------------------------------- detail

#[derive(Template)]
#[template(path = "detail.html")]
struct DetailPage {
    layout: Layout,
    name: String,
    metric: &'static str,
    dimension: u32,
    building: bool,
    count: u64,
    delta_count: u64,
    deleted: u64,
    size: String,
    built_at: String,
    cfg_r: u32,
    cfg_l_build: u32,
    cfg_alpha: f32,
    cfg_pq: u32,
    cfg_l_search: u32,
    embedding_model: String,
    embed_ready: bool,
    example_vector: String,
    lookup_id: String,
    lookup_json: String,
}

async fn detail_page(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Query(q): Query<Params>,
) -> Response {
    let idx = match s.catalog.get(&name) {
        Ok(i) => i,
        Err(e) => return back("/ui", Err(e.to_string())),
    };
    let mut l = layout(&s, &headers, &q);
    let lookup_id = q.get("id").cloned().unwrap_or_default();
    let (info, building, lookup) = {
        let idx2 = idx.clone();
        let id = lookup_id.clone();
        run(move || {
            let lookup = if id.is_empty() {
                None
            } else {
                Some(idx2.get(&id))
            };
            Ok((idx2.info(), idx2.is_building(), lookup))
        })
        .await
        .unwrap_or_else(|_| (idx.info(), idx.is_building(), None))
    };
    let lookup_json = match lookup {
        Some(Ok(rec)) => serde_json::to_string_pretty(&rec).unwrap_or_default(),
        Some(Err(e)) => {
            l.flash_err = e.to_string();
            String::new()
        }
        None => String::new(),
    };
    let c = &info.config;
    let ex: Vec<String> = (0..c.dimension.min(3))
        .map(|i| format!("0.{}", i + 1))
        .collect();
    render(DetailPage {
        layout: l,
        name: c.name.clone(),
        metric: c.metric.as_str(),
        dimension: c.dimension,
        building,
        count: info.count,
        delta_count: info.delta_count,
        deleted: info.deleted,
        size: human_bytes(info.size_bytes),
        built_at: info.built_at.clone().unwrap_or_else(|| "never".into()),
        cfg_r: c.params.r,
        cfg_l_build: c.params.l_build,
        cfg_alpha: c.params.alpha,
        cfg_pq: c.params.pq_subvectors,
        cfg_l_search: c.params.l_search,
        embedding_model: c.embedding_model.clone().unwrap_or_default(),
        embed_ready: s.embedder.configured(),
        example_vector: if c.dimension > 3 {
            format!("{}, …", ex.join(", "))
        } else {
            ex.join(", ")
        },
        lookup_id,
        lookup_json,
    })
}

/// Accepts `[records]`, `{"vectors": [records]}` or a single record.
pub fn parse_records(text: &str) -> Result<Vec<VectorRecord>, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let res = if v.is_array() {
        serde_json::from_value::<Vec<VectorRecord>>(v)
    } else if v.get("vectors").is_some() {
        serde_json::from_value::<UpsertRequest>(v).map(|r| r.vectors)
    } else {
        serde_json::from_value::<VectorRecord>(v).map(|r| vec![r])
    };
    res.map_err(|e| format!("invalid records: {e}"))
}

async fn upsert_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let path = format!("/ui/indexes/{name}");
    let records = match parse_records(f.get("records").map(String::as_str).unwrap_or("")) {
        Ok(r) => r,
        Err(e) => return back(&path, Err(e)),
    };
    let idx = match s.catalog.get(&name) {
        Ok(i) => i,
        Err(e) => return back("/ui", Err(e.to_string())),
    };
    match run(move || idx.upsert(records)).await {
        Ok(n) => back(
            &path,
            Ok(format!(
                "Upserted {n} vector{}.",
                if n == 1 { "" } else { "s" }
            )),
        ),
        Err(e) => back(&path, Err(e)),
    }
}

/// Accepts `[docs]`, `{"documents": [docs]}` or a single doc.
pub fn parse_documents(text: &str) -> Result<Vec<DocumentRecord>, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let res = if v.is_array() {
        serde_json::from_value::<Vec<DocumentRecord>>(v)
    } else if v.get("documents").is_some() {
        serde_json::from_value::<UpsertDocumentsRequest>(v).map(|r| r.documents)
    } else {
        serde_json::from_value::<DocumentRecord>(v).map(|r| vec![r])
    };
    res.map_err(|e| format!("invalid documents: {e}"))
}

async fn documents_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let path = format!("/ui/indexes/{name}");
    let docs = match parse_documents(f.get("documents").map(String::as_str).unwrap_or("")) {
        Ok(d) => d,
        Err(e) => return back(&path, Err(e)),
    };
    let idx = match s.catalog.get(&name) {
        Ok(i) => i,
        Err(e) => return back("/ui", Err(e.to_string())),
    };
    let records = match embed_documents(&s, idx.config(), docs).await {
        Ok(r) => r,
        Err(e) => return back(&path, Err(e.message)),
    };
    match run(move || idx.upsert(records)).await {
        Ok(n) => back(
            &path,
            Ok(format!(
                "Embedded and upserted {n} document{}.",
                if n == 1 { "" } else { "s" }
            )),
        ),
        Err(e) => back(&path, Err(e)),
    }
}

async fn delete_vector_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let path = format!("/ui/indexes/{name}");
    let id = f.get("id").cloned().unwrap_or_default();
    let idx = match s.catalog.get(&name) {
        Ok(i) => i,
        Err(e) => return back("/ui", Err(e.to_string())),
    };
    let id2 = id.clone();
    match run(move || idx.delete(&id2)).await {
        Ok(()) => back(&path, Ok(format!("Deleted vector {id}."))),
        Err(e) => back(&path, Err(e)),
    }
}

async fn delete_index_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let cat = s.catalog.clone();
    let n = name.clone();
    match run(move || cat.delete(&n)).await {
        Ok(()) => back("/ui", Ok(format!("Index {name} deleted."))),
        Err(e) => back("/ui", Err(e)),
    }
}

async fn build_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let idx = match s.catalog.get(&name) {
        Ok(i) => i,
        Err(e) => return back("/ui", Err(e.to_string())),
    };
    let to = headers
        .get(header::REFERER)
        .and_then(|r| r.to_str().ok())
        .filter(|r| r.contains(&format!("/ui/indexes/{name}")))
        .map(|_| format!("/ui/indexes/{name}"))
        .unwrap_or_else(|| "/ui".to_string());
    match run(move || idx.start_build()).await {
        Ok(()) => back(
            &to,
            Ok(format!("Rebuild of {name} started in the background.")),
        ),
        Err(e) => back(&to, Err(e)),
    }
}

// ---------------------------------------------------------------- search

pub struct HitRow {
    pub rank: usize,
    pub id: String,
    pub id_q: String,
    pub score: String,
    pub metadata: String,
}

#[derive(Template)]
#[template(path = "search.html")]
struct SearchPage {
    layout: Layout,
    name: String,
    metric: &'static str,
    dimension: u32,
    score_hint: &'static str,
    embedding_model: String,
    text: String,
    vector: String,
    k: String,
    l_search: String,
    filter: String,
    searched: bool,
    hits: Vec<HitRow>,
    took: String,
    embed_took: String,
    search_took: String,
}

fn search_page_for(
    s: &AppState,
    headers: &HeaderMap,
    q: &Params,
    cfg: &ex_index_core::models::IndexConfig,
    f: &Params,
) -> SearchPage {
    SearchPage {
        layout: layout(s, headers, q),
        name: cfg.name.clone(),
        metric: cfg.metric.as_str(),
        dimension: cfg.dimension,
        score_hint: match cfg.metric {
            Metric::Cosine => "score = cosine similarity (higher is better)",
            Metric::Euclidean => "score = L2 distance (lower is better)",
            Metric::Dot => "score = inner product (higher is better)",
        },
        embedding_model: cfg.embedding_model.clone().unwrap_or_default(),
        text: f.get("text").cloned().unwrap_or_default(),
        vector: f.get("vector").cloned().unwrap_or_default(),
        k: f.get("k").cloned().unwrap_or_else(|| "10".into()),
        l_search: f.get("l_search").cloned().unwrap_or_default(),
        filter: f.get("filter").cloned().unwrap_or_default(),
        searched: false,
        hits: Vec::new(),
        took: String::new(),
        embed_took: String::new(),
        search_took: String::new(),
    }
}

async fn search_page(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Query(q): Query<Params>,
) -> Response {
    match s.catalog.get(&name) {
        Ok(idx) => render(search_page_for(&s, &headers, &q, idx.config(), &q)),
        Err(e) => back("/ui", Err(e.to_string())),
    }
}

fn parse_search(f: &Params) -> Result<SearchRequest, String> {
    let text = f
        .get("text")
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    let raw = f.get("vector").map(|v| v.trim()).unwrap_or("");
    let vector: Vec<f32> = if raw.is_empty() && text.is_some() {
        Vec::new()
    } else {
        serde_json::from_str(raw)
            .map_err(|e| format!("vector must be a JSON array of numbers: {e}"))?
    };
    let mut req = SearchRequest::new(vector, opt_num::<u32>(f, "k")?.unwrap_or(10));
    req.text = text;
    req.l_search = opt_num(f, "l_search")?;
    if let Some(filt) = f.get("filter").map(|s| s.trim()).filter(|s| !s.is_empty()) {
        req.filter = Some(
            serde_json::from_str(filt).map_err(|e| format!("filter must be a JSON object: {e}"))?,
        );
    }
    Ok(req)
}

async fn search_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let idx = match s.catalog.get(&name) {
        Ok(i) => i,
        Err(e) => return back("/ui", Err(e.to_string())),
    };
    let mut page = search_page_for(&s, &headers, &Params::new(), idx.config(), &f);
    let mut req = match parse_search(&f) {
        Ok(r) => r,
        Err(e) => {
            page.layout.flash_err = e;
            return (StatusCode::BAD_REQUEST, render(page)).into_response();
        }
    };
    let embed_ms = match resolve_query(&s, idx.config(), &mut req).await {
        Ok(ms) => ms,
        Err(e) => {
            page.layout.flash_err = e.message;
            return (e.status, render(page)).into_response();
        }
    };
    let t0 = Instant::now();
    match run(move || idx.search(&req)).await {
        Ok(hits) => {
            let search_ms = ms_since(t0);
            page.took = format!("{:.2}", embed_ms.unwrap_or(0.0) + search_ms);
            page.embed_took = embed_ms.map(|ms| format!("{ms:.2}")).unwrap_or_default();
            page.search_took = format!("{search_ms:.2}");
            page.searched = true;
            page.hits = hits
                .into_iter()
                .enumerate()
                .map(|(i, h)| HitRow {
                    rank: i + 1,
                    id_q: qenc(&h.id),
                    id: h.id,
                    score: format!("{:.6}", h.score),
                    metadata: h
                        .metadata
                        .map(|m| serde_json::Value::Object(m).to_string())
                        .unwrap_or_default(),
                })
                .collect();
            render(page)
        }
        Err(e) => {
            page.layout.flash_err = e;
            (StatusCode::BAD_REQUEST, render(page)).into_response()
        }
    }
}

// ---------------------------------------------------------------- login

#[derive(Template)]
#[template(path = "login.html")]
struct LoginPage {
    layout: Layout,
}

async fn login_page(
    State(s): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<Params>,
) -> Response {
    if !s.auth.enabled() {
        return Redirect::to("/ui").into_response();
    }
    render(LoginPage {
        layout: layout(&s, &headers, &q),
    })
}

async fn login_submit(
    State(s): State<AppState>,
    headers: HeaderMap,
    Form(f): Form<Params>,
) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let key = f.get("key").map(String::as_str).unwrap_or("");
    if !s.auth.check_key(key) {
        let mut l = layout(&s, &headers, &Params::new());
        l.flash_err = "Invalid API key.".into();
        return (StatusCode::UNAUTHORIZED, render(LoginPage { layout: l })).into_response();
    }
    let mut resp = Redirect::to("/ui").into_response();
    if let Some(c) = s.auth.login_cookie() {
        if let Ok(v) = c.parse() {
            resp.headers_mut().insert(header::SET_COOKIE, v);
        }
    }
    resp
}

async fn logout(State(s): State<AppState>, headers: HeaderMap, Form(f): Form<Params>) -> Response {
    if !csrf_ok(&s, &headers, &f) {
        return forbidden();
    }
    let mut resp = Redirect::to("/ui/login").into_response();
    if let Ok(v) = Auth::logout_cookie().parse() {
        resp.headers_mut().insert(header::SET_COOKIE, v);
    }
    resp
}
