//! Design test "Restart persistence" and requirement "Durability", against the
//! real binary: upsert, delete some, rebuild, upsert more (delta), kill -9,
//! restart, search.
use serde_json::{json, Value};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn spawn(data: &std::path::Path, port: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_ex-index"))
        .args([
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--data-dir",
        ])
        .arg(data)
        .args(["--small-index-threshold", "500", "--log-level", "warn"])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap()
}

async fn wait_healthy(c: &reqwest::Client, base: &str) {
    for _ in 0..300 {
        if let Ok(r) = c.get(format!("{base}/healthz")).send().await {
            if r.status() == 200 {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("server did not become healthy");
}

fn vec_for(i: usize) -> Vec<f32> {
    let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..16)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

async fn search_all(c: &reqwest::Client, base: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for q in 0..20 {
        let r: Value = c
            .post(format!("{base}/v1/indexes/p/search"))
            .json(&json!({"vector": vec_for(100_000 + q), "k": 10, "l_search": 200}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        out.push(
            r["hits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|h| h["id"].as_str().unwrap().to_string())
                .collect(),
        );
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kill9_restart_keeps_acknowledged_writes() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let port = free_port();
    let base = format!("http://127.0.0.1:{port}");
    ex_index::embed::install_crypto_provider();
    let c = reqwest::Client::new();

    let mut child = spawn(&data, port);
    wait_healthy(&c, &base).await;
    c.post(format!("{base}/v1/indexes"))
        .json(&json!({"name":"p","dimension":16,"metric":"euclidean"}))
        .send()
        .await
        .unwrap();
    let batch = |r: std::ops::Range<usize>| json!({"vectors": r.map(|i| json!({"id": format!("v{i}"), "vector": vec_for(i), "metadata": {"i": i}})).collect::<Vec<_>>()});
    assert_eq!(
        c.post(format!("{base}/v1/indexes/p/vectors"))
            .json(&batch(0..2000))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    for i in (0..2000).step_by(9) {
        assert_eq!(
            c.delete(format!("{base}/v1/indexes/p/vectors/v{i}"))
                .send()
                .await
                .unwrap()
                .status(),
            204
        );
    }
    // explicit rebuild (an automatic one may already be running → 202 or 409)
    let st = c
        .post(format!("{base}/v1/indexes/p/build"))
        .send()
        .await
        .unwrap()
        .status();
    assert!(st == 202 || st == 409, "{st}");
    for _ in 0..1000 {
        let i: Value = c
            .get(format!("{base}/v1/indexes/p"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if i["status"] == "ready" && i["built_at"].is_string() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // more upserts + deletes land in the delta
    assert_eq!(
        c.post(format!("{base}/v1/indexes/p/vectors"))
            .json(&batch(2000..2300))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        c.delete(format!("{base}/v1/indexes/p/vectors/v2001"))
            .send()
            .await
            .unwrap()
            .status(),
        204
    );
    assert_eq!(
        c.delete(format!("{base}/v1/indexes/p/vectors/v1"))
            .send()
            .await
            .unwrap()
            .status(),
        204
    );
    let before = search_all(&c, &base).await;
    let info: Value = c
        .get(format!("{base}/v1/indexes/p"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let count = info["count"].as_u64().unwrap();
    assert_eq!(count as usize, 2300 - 2000usize.div_ceil(9) - 2);

    child.kill().unwrap(); // SIGKILL on unix, TerminateProcess on windows
    child.wait().unwrap();

    let mut child = spawn(&data, port);
    wait_healthy(&c, &base).await;
    let info: Value = c
        .get(format!("{base}/v1/indexes/p"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(info["count"].as_u64().unwrap(), count);
    assert_eq!(search_all(&c, &base).await, before);
    for id in ["v0", "v9", "v1", "v2001"] {
        assert_eq!(
            c.get(format!("{base}/v1/indexes/p/vectors/{id}"))
                .send()
                .await
                .unwrap()
                .status(),
            404,
            "{id}"
        );
    }
    let v: Value = c
        .get(format!("{base}/v1/indexes/p/vectors/v2299"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["metadata"]["i"], 2299);
    child.kill().unwrap();
    child.wait().unwrap();
}
