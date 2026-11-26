//! Design test "REST CRUD + search": every endpoint incl. error cases.
mod common;
use common::{client, TestServer};
use serde_json::{json, Value};

async fn err_code(r: reqwest::Response) -> (u16, String) {
    let status = r.status().as_u16();
    let v: Value = r.json().await.unwrap();
    (status, v["error"]["code"].as_str().unwrap().to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rest_crud_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let s = TestServer::start(dir.path(), None).await;
    let c = client();

    // health
    let h: Value = c
        .get(s.url("/healthz"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(h["status"], "ok");
    assert_eq!(h["version"], env!("CARGO_PKG_VERSION"));

    // create
    let r = c
        .post(s.url("/v1/indexes"))
        .json(&json!({"name":"docs","dimension":3,"metric":"euclidean"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);
    let info: Value = r.json().await.unwrap();
    assert_eq!(info["config"]["params"]["pq_subvectors"], 1);
    assert_eq!(info["status"], "ready");
    // duplicate → 409, invalid → 400
    let r = c
        .post(s.url("/v1/indexes"))
        .json(&json!({"name":"docs","dimension":3,"metric":"euclidean"}))
        .send()
        .await
        .unwrap();
    assert_eq!(err_code(r).await, (409, "conflict".into()));
    for bad in [
        json!({"name":"Docs","dimension":3,"metric":"euclidean"}),
        json!({"name":"x","dimension":0,"metric":"euclidean"}),
        json!({"name":"x","dimension":3,"metric":"hamming"}),
        json!({"name":"x","dimension":3}),
        json!({"name":"x","dimension":3,"metric":"dot","params":{"alpha":9}}),
    ] {
        let r = c
            .post(s.url("/v1/indexes"))
            .json(&bad)
            .send()
            .await
            .unwrap();
        assert_eq!(err_code(r).await, (400, "invalid_argument".into()), "{bad}");
    }
    let r = c
        .post(s.url("/v1/indexes"))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(err_code(r).await.0, 400);

    // list / get / 404
    let l: Value = c
        .get(s.url("/v1/indexes"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(l["indexes"].as_array().unwrap().len(), 1);
    assert_eq!(
        c.get(s.url("/v1/indexes/docs"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let r = c.get(s.url("/v1/indexes/nope")).send().await.unwrap();
    assert_eq!(err_code(r).await, (404, "not_found".into()));

    // upsert
    let body = json!({"vectors":[
        {"id":"a","vector":[0,0,0],"metadata":{"lang":"en"}},
        {"id":"b","vector":[1,0,0],"metadata":{"lang":"de"}},
        {"id":"c","vector":[3,0,0]}
    ]});
    let r: Value = c
        .post(s.url("/v1/indexes/docs/vectors"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["upserted"], 3);
    // whole batch rejected on a bad vector
    let r = c
        .post(s.url("/v1/indexes/docs/vectors"))
        .json(&json!({"vectors":[{"id":"d","vector":[1,1,1]},{"id":"e","vector":[1,1]}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(err_code(r).await, (400, "invalid_argument".into()));
    assert_eq!(
        c.get(s.url("/v1/indexes/docs/vectors/d"))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    let r = c
        .post(s.url("/v1/indexes/nope/vectors"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(err_code(r).await.0, 404);
    // > 10k vectors per request
    let many: Vec<Value> = (0..10_001)
        .map(|i| json!({"id": i.to_string(), "vector":[0,0,1]}))
        .collect();
    let r = c
        .post(s.url("/v1/indexes/docs/vectors"))
        .json(&json!({"vectors": many}))
        .send()
        .await
        .unwrap();
    assert_eq!(err_code(r).await.0, 400);

    // get vector
    let v: Value = c
        .get(s.url("/v1/indexes/docs/vectors/a"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        v,
        json!({"id":"a","vector":[0.0,0.0,0.0],"metadata":{"lang":"en"}})
    );

    // search: euclidean scores are true distances, best (lowest) first
    let r: Value = c
        .post(s.url("/v1/indexes/docs/search"))
        .json(&json!({"vector":[2.9,0,0],"k":3,"include_vector":true}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let hits = r["hits"].as_array().unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["c", "b", "a"]);
    assert!((hits[0]["score"].as_f64().unwrap() - 0.1).abs() < 1e-5);
    assert!((hits[2]["score"].as_f64().unwrap() - 2.9).abs() < 1e-5);
    assert_eq!(hits[0]["vector"], json!([3.0, 0.0, 0.0]));
    assert!(r["took_ms"].as_f64().unwrap() >= 0.0);
    assert!(r["search_ms"].as_f64().unwrap() >= 0.0);
    assert!(r.get("embed_ms").is_none());
    // filter
    let r: Value = c
        .post(s.url("/v1/indexes/docs/search"))
        .json(&json!({"vector":[2.9,0,0],"filter":{"lang":"en"},"include_metadata":true}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        r["hits"],
        json!([{"id":"a","score":2.9,"metadata":{"lang":"en"}}])
    );
    // search errors
    for bad in [
        json!({"vector":[1,2]}),
        json!({"vector":[1,2,3],"k":0}),
        json!({"vector":[1,2,3],"k":5,"l_search":2}),
    ] {
        let r = c
            .post(s.url("/v1/indexes/docs/search"))
            .json(&bad)
            .send()
            .await
            .unwrap();
        assert_eq!(err_code(r).await, (400, "invalid_argument".into()), "{bad}");
    }

    // re-upsert replaces; delete hides
    c.post(s.url("/v1/indexes/docs/vectors"))
        .json(&json!({"vectors":[{"id":"a","vector":[10,0,0],"metadata":{"v":2}}]}))
        .send()
        .await
        .unwrap();
    let v: Value = c
        .get(s.url("/v1/indexes/docs/vectors/a"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["metadata"], json!({"v":2}));
    assert_eq!(
        c.delete(s.url("/v1/indexes/docs/vectors/b"))
            .send()
            .await
            .unwrap()
            .status(),
        204
    );
    let r = c
        .delete(s.url("/v1/indexes/docs/vectors/b"))
        .send()
        .await
        .unwrap();
    assert_eq!(err_code(r).await, (404, "not_found".into()));
    let r: Value = c
        .post(s.url("/v1/indexes/docs/search"))
        .json(&json!({"vector":[1,0,0]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = r["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["c", "a"]);

    // build: 202 then 409 while running
    let r = c
        .post(s.url("/v1/indexes/docs/build"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 202);
    let info: Value = r.json().await.unwrap();
    assert_eq!(info["status"], "building");
    // wait until ready
    for _ in 0..500 {
        let i: Value = c
            .get(s.url("/v1/indexes/docs"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if i["status"] == "ready" {
            assert_eq!(i["count"], 2);
            assert!(i["built_at"].is_string());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let r = c
        .post(s.url("/v1/indexes/nope/build"))
        .send()
        .await
        .unwrap();
    assert_eq!(err_code(r).await.0, 404);

    // metrics
    let m = c
        .get(s.url("/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("http_requests_total"));
    assert!(m.contains("ex_index_vectors{index=\"docs\"} 2"));

    // unknown route → JSON 404
    let r = c.get(s.url("/v1/nothing")).send().await.unwrap();
    assert_eq!(err_code(r).await.0, 404);

    // delete index
    assert_eq!(
        c.delete(s.url("/v1/indexes/docs"))
            .send()
            .await
            .unwrap()
            .status(),
        204
    );
    assert_eq!(
        c.get(s.url("/v1/indexes/docs"))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert!(!dir.path().join("docs").exists());
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn conflict_when_build_running() {
    let dir = tempfile::tempdir().unwrap();
    let s = TestServer::start(dir.path(), None).await;
    let c = client();
    c.post(s.url("/v1/indexes"))
        .json(&json!({"name":"big","dimension":32,"metric":"cosine"}))
        .send()
        .await
        .unwrap();
    let vecs: Vec<Value> = (0..5000)
        .map(|i| json!({"id": format!("v{i}"), "vector": (0..32).map(|j| ((i * 31 + j * 17) % 97) as f32 / 97.0 + 0.01).collect::<Vec<_>>()}))
        .collect();
    c.post(s.url("/v1/indexes/big/vectors"))
        .json(&json!({"vectors": vecs}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        c.post(s.url("/v1/indexes/big/build"))
            .send()
            .await
            .unwrap()
            .status(),
        202
    );
    let r = c.post(s.url("/v1/indexes/big/build")).send().await.unwrap();
    assert_eq!(err_code(r).await, (409, "conflict".into()));
    s.stop().await;
}

/// Requirement "Ingest vectors + metadata": one batch of 10k × 768-d vectors
/// with metadata succeeds (compact float formatting keeps it under 64 MiB).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingest_10k_x_768_batch() {
    let dir = tempfile::tempdir().unwrap();
    let s = TestServer::start(dir.path(), None).await;
    let c = client();
    c.post(s.url("/v1/indexes"))
        .json(&json!({"name":"e","dimension":768,"metric":"cosine"}))
        .send()
        .await
        .unwrap();
    let mut body = String::with_capacity(60 << 20);
    body.push_str("{\"vectors\":[");
    let mut x: u64 = 1;
    for i in 0..10_000 {
        if i > 0 {
            body.push(',');
        }
        body.push_str(&format!(
            "{{\"id\":\"doc-{i}\",\"metadata\":{{\"i\":{i},\"lang\":\"en\"}},\"vector\":["
        ));
        for j in 0..768 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if j > 0 {
                body.push(',');
            }
            body.push_str(&format!(
                "{:.3}",
                ((x >> 40) as f64 / (1u64 << 24) as f64) - 0.5
            ));
        }
        body.push_str("]}");
    }
    body.push_str("]}");
    assert!(body.len() < ex_index::BODY_LIMIT, "{} bytes", body.len());
    let r = c
        .post(s.url("/v1/indexes/e/vectors"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let v: Value = c
        .get(s.url("/v1/indexes/e/vectors/doc-1234"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["metadata"], json!({"i":1234,"lang":"en"}));
    assert_eq!(v["vector"].as_array().unwrap().len(), 768);
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn body_over_64mib_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let s = TestServer::start(dir.path(), None).await;
    let c = client();
    c.post(s.url("/v1/indexes"))
        .json(&json!({"name":"x","dimension":2,"metric":"dot"}))
        .send()
        .await
        .unwrap();
    let body = format!(
        "{{\"vectors\":[], \"pad\":\"{}\"}}",
        "x".repeat(ex_index::BODY_LIMIT)
    );
    let r = c
        .post(s.url("/v1/indexes/x/vectors"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 413);
    s.stop().await;
}
