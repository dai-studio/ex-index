//! Cloudflare Workers AI embeddings, against a local mock of the REST API.
mod common;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use common::{client, csrf, TestServer};
use ex_index::ServeArgs;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// (model, request field, batch size) per call.
type Calls = Arc<Mutex<Vec<(String, String, usize)>>>;

fn dims(model: &str) -> usize {
    ex_index_core::embedding::embedding_model(model)
        .map(|m| m.dimension as usize)
        .unwrap_or(0)
}

/// Bag-of-words embedding: texts sharing words get similar vectors.
fn fake_embed(text: &str, d: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; d];
    for w in text.split_whitespace() {
        let h = w
            .to_lowercase()
            .bytes()
            .fold(1469598103934665603u64, |h, b| {
                (h ^ b as u64).wrapping_mul(1099511628211)
            });
        v[(h % d as u64) as usize] += 1.0;
    }
    v
}

async fn mock_run(
    State(calls): State<Calls>,
    Path((account, model)): Path<(String, String)>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some("Bearer good-token")
        || account != "acct"
    {
        return (
            StatusCode::FORBIDDEN,
            Json(
                json!({"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}]}),
            ),
        );
    }
    let d = dims(&model);
    let (field, texts) = body
        .as_object()
        .and_then(|o| o.iter().next())
        .map(|(k, v)| (k.clone(), v.as_array().cloned().unwrap_or_default()))
        .unwrap();
    calls
        .lock()
        .unwrap()
        .push((model.clone(), field, texts.len()));
    let data: Vec<Vec<f32>> = texts
        .iter()
        .map(|t| fake_embed(t.as_str().unwrap(), d))
        .collect();
    (
        StatusCode::OK,
        Json(
            json!({"result":{"shape":[texts.len(), d],"data":data},"success":true,"errors":[],"messages":[]}),
        ),
    )
}

async fn start_mock() -> (String, Calls) {
    let calls: Calls = Arc::default();
    let app = Router::new()
        .route(
            "/client/v4/accounts/{account}/ai/run/{*model}",
            post(mock_run),
        )
        .with_state(calls.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}/client/v4"), calls)
}

async fn start(dir: &std::path::Path, base: &str, token: Option<&str>) -> TestServer {
    let mut args = ServeArgs::for_data_dir(dir);
    args.cloudflare_api_base = base.to_string();
    args.cloudflare_account_id = Some("acct".into());
    args.cloudflare_api_token = token.map(String::from);
    TestServer::start_with(args).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn documents_and_text_search() {
    let (base, calls) = start_mock().await;
    let dir = tempfile::tempdir().unwrap();
    let s = start(dir.path(), &base, Some("good-token")).await;
    let c = client();

    // create: dimension comes from the model
    let r = c
        .post(s.url("/v1/indexes"))
        .json(&json!({"name":"docs","metric":"cosine","embedding_model":"@cf/baai/bge-small-en-v1.5"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);
    let info: Value = r.json().await.unwrap();
    assert_eq!(info["config"]["dimension"], 384);
    assert_eq!(
        info["config"]["embedding_model"],
        "@cf/baai/bge-small-en-v1.5"
    );

    // mismatched dimension / unknown model
    for bad in [
        json!({"name":"x","dimension":768,"metric":"cosine","embedding_model":"@cf/baai/bge-small-en-v1.5"}),
        json!({"name":"x","metric":"cosine","embedding_model":"@cf/nope"}),
        json!({"name":"x","metric":"cosine"}),
    ] {
        let r = c
            .post(s.url("/v1/indexes"))
            .json(&bad)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "{bad}");
    }

    // 150 documents -> two calls of 100 + 50 using the `text` field
    let mut docs: Vec<Value> = (0..148)
        .map(|i| json!({"id": format!("filler-{i}"), "text": format!("filler number {i} lorem ipsum")}))
        .collect();
    docs.push(json!({"id":"cloud","text":"This is a story about an orange cloud","metadata":{"animal":false}}));
    docs.push(
        json!({"id":"llama","text":"This is a story about a llama","metadata":{"animal":true}}),
    );
    let r = c
        .post(s.url("/v1/indexes/docs/documents"))
        .json(&json!({ "documents": docs }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["upserted"], 150);
    {
        let calls = calls.lock().unwrap();
        let sizes: Vec<usize> = calls.iter().map(|c| c.2).collect();
        assert_eq!(sizes, vec![100, 50]);
        assert!(calls.iter().all(|c| c.1 == "text"));
    }

    // text search
    let r: Value = c
        .post(s.url("/v1/indexes/docs/search"))
        .json(&json!({"text":"orange cloud story","k":3}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["hits"][0]["id"], "cloud", "{r}");
    let (embed, search, took) = (
        r["embed_ms"].as_f64().unwrap(),
        r["search_ms"].as_f64().unwrap(),
        r["took_ms"].as_f64().unwrap(),
    );
    assert!((took - (embed + search)).abs() < 1e-9, "{r}");
    let r: Value = c
        .post(s.url("/v1/indexes/docs/search"))
        .json(&json!({"text":"story about a llama","k":1,"filter":{"animal":true}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["hits"][0]["id"], "llama");

    // both vector and text
    let r = c
        .post(s.url("/v1/indexes/docs/search"))
        .json(&json!({"text":"x","vector":vec![1.0; 384]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);

    // invalid documents are rejected before any embedding call
    let before = calls.lock().unwrap().len();
    for bad in [
        json!({"documents":[{"id":"","text":"a"}]}),
        json!({"documents":[{"id":"a","text":"  "}]}),
        json!({"documents":[{"id":"a","vector":[1]}]}),
    ] {
        let r = c
            .post(s.url("/v1/indexes/docs/documents"))
            .json(&bad)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "{bad}");
    }
    assert_eq!(calls.lock().unwrap().len(), before);

    // qwen3 embeds search queries through the `queries` field
    c.post(s.url("/v1/indexes"))
        .json(&json!({"name":"qwen","metric":"cosine","embedding_model":"@cf/qwen/qwen3-embedding-0.6b"}))
        .send()
        .await
        .unwrap();
    let r = c
        .post(s.url("/v1/indexes/qwen/documents"))
        .json(&json!({"documents":[{"id":"a","text":"hello world"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r: Value = c
        .post(s.url("/v1/indexes/qwen/search"))
        .json(&json!({"text":"hello"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["hits"][0]["id"], "a");
    {
        let calls = calls.lock().unwrap();
        let last: Vec<&str> = calls.iter().rev().take(2).map(|c| c.1.as_str()).collect();
        assert_eq!(last, vec!["queries", "text"]);
    }

    // index without a model
    c.post(s.url("/v1/indexes"))
        .json(&json!({"name":"plain","dimension":3,"metric":"cosine"}))
        .send()
        .await
        .unwrap();
    let r = c
        .post(s.url("/v1/indexes/plain/documents"))
        .json(&json!({"documents":[{"id":"a","text":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);

    // admin UI: add documents and search by text
    let html = c
        .get(s.url("/ui/indexes/docs"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("Add documents"));
    let token = csrf(&html);
    let r = c
        .post(s.url("/ui/indexes/docs/documents"))
        .form(&[
            ("csrf", token.as_str()),
            (
                "documents",
                r#"[{"id":"emoji","text":"a hugging emoji tale"}]"#,
            ),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    let loc = r.headers()["location"].to_str().unwrap();
    assert!(loc.contains("msg="), "{loc}");
    let html = c
        .post(s.url("/ui/indexes/docs/search"))
        .form(&[
            ("csrf", token.as_str()),
            ("text", "hugging emoji"),
            ("vector", ""),
            ("k", "1"),
        ])
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains(">emoji</a>"), "{html}");
    assert!(html.contains("(embed "), "{html}");

    // the model survives a restart
    s.stop().await;
    let s = start(dir.path(), &base, Some("good-token")).await;
    let info: Value = c
        .get(s.url("/v1/indexes/docs"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        info["config"]["embedding_model"],
        "@cf/baai/bge-small-en-v1.5"
    );
    assert_eq!(info["count"], 151);
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn credential_errors() {
    let (base, _calls) = start_mock().await;
    let dir = tempfile::tempdir().unwrap();
    let body = json!({"documents":[{"id":"a","text":"hi"}]});

    // no token configured -> 503
    let s = start(dir.path(), &base, None).await;
    let c = client();
    c.post(s.url("/v1/indexes"))
        .json(&json!({"name":"docs","metric":"cosine","embedding_model":"@cf/baai/bge-m3"}))
        .send()
        .await
        .unwrap();
    let r = c
        .post(s.url("/v1/indexes/docs/documents"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
    assert_eq!(
        r.json::<Value>().await.unwrap()["error"]["code"],
        "unavailable"
    );
    s.stop().await;

    // rejected token -> 502 with Cloudflare's message
    let s = start(dir.path(), &base, Some("bad-token")).await;
    let r = c
        .post(s.url("/v1/indexes/docs/documents"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "upstream");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Authentication error"));
    s.stop().await;
}
