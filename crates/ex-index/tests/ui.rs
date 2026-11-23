//! Design test "UI smoke".
mod common;
use common::{client, csrf, TestServer};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ui_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let s = TestServer::start(dir.path(), None).await;
    let c = client();

    let home = c.get(s.url("/ui")).send().await.unwrap();
    assert_eq!(home.status(), 200);
    let html = home.text().await.unwrap();
    assert!(html.contains("No indexes yet"));
    // every stylesheet referenced by the layout is served from /assets
    for href in html
        .split("href=\"/assets/")
        .skip(1)
        .map(|r| &r[..r.find('"').unwrap()])
    {
        let r = c
            .get(s.url(&format!("/assets/{href}")))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{href}");
        assert!(r.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/css"));
    }
    assert_eq!(
        c.get(s.url("/assets/missing.css"))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(c.get(s.url("/")).send().await.unwrap().status(), 303);

    // create form
    let page = c
        .get(s.url("/ui/new"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(page.contains("Create index"));
    let t = csrf(&page);
    let r = c
        .post(s.url("/ui/new"))
        .form(&[
            ("csrf", t.as_str()),
            ("name", "BAD NAME"),
            ("dimension", "3"),
            ("metric", "cosine"),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    assert!(r.text().await.unwrap().contains("name must match"));
    let r = c
        .post(s.url("/ui/new"))
        .form(&[
            ("csrf", t.as_str()),
            ("name", "films"),
            ("dimension", "3"),
            ("metric", "cosine"),
            ("r", "32"),
            ("alpha", ""),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    assert!(r.headers()["location"]
        .to_str()
        .unwrap()
        .starts_with("/ui/indexes/films"));
    let html = c
        .get(s.url("/ui"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("/ui/indexes/films") && html.contains("cosine"));

    // detail + upsert form
    let page = c
        .get(s.url("/ui/indexes/films"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(page.contains("Upsert vectors") && page.contains(">32<"));
    let t = csrf(&page);
    let records = r#"[{"id":"alien","vector":[1,0,0],"metadata":{"genre":"scifi"}},{"id":"heat","vector":[0,1,0],"metadata":{"genre":"crime"}}]"#;
    let r = c
        .post(s.url("/ui/indexes/films/upsert"))
        .form(&[("csrf", t.as_str()), ("records", records)])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    let loc = r.headers()["location"].to_str().unwrap().to_string();
    let page = c
        .get(s.url(&loc))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(page.contains("Upserted 2 vectors."));
    let r = c
        .post(s.url("/ui/indexes/films/upsert"))
        .form(&[("csrf", t.as_str()), ("records", "[{")])
        .send()
        .await
        .unwrap();
    assert!(r.headers()["location"].to_str().unwrap().contains("err="));

    // lookup by id
    let page = c
        .get(s.url("/ui/indexes/films?id=alien"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(page.contains("scifi"));

    // search form
    let page = c
        .get(s.url("/ui/indexes/films/search"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let t = csrf(&page);
    let r = c
        .post(s.url("/ui/indexes/films/search"))
        .form(&[
            ("csrf", t.as_str()),
            ("vector", "[0.9, 0.1, 0]"),
            ("k", "5"),
            ("l_search", ""),
            ("filter", ""),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let html = r.text().await.unwrap();
    let alien = html.find(">alien<").expect("alien row");
    let heat = html.find(">heat<").expect("heat row");
    assert!(alien < heat, "results ranked best-first");
    let r = c
        .post(s.url("/ui/indexes/films/search"))
        .form(&[
            ("csrf", t.as_str()),
            ("vector", "[0.9, 0.1, 0]"),
            ("k", "5"),
            ("filter", r#"{"genre":"crime"}"#),
        ])
        .send()
        .await
        .unwrap();
    let html = r.text().await.unwrap();
    assert!(html.contains(">heat<") && !html.contains(">alien<"));
    let r = c
        .post(s.url("/ui/indexes/films/search"))
        .form(&[("csrf", t.as_str()), ("vector", "[1]")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);

    // delete vector, rebuild, delete index
    let r = c
        .post(s.url("/ui/indexes/films/delete-vector"))
        .form(&[("csrf", t.as_str()), ("id", "heat")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    let r = c
        .post(s.url("/ui/indexes/films/build"))
        .form(&[("csrf", t.as_str())])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    // CSRF required on every post
    let r = c
        .post(s.url("/ui/indexes/films/delete"))
        .form(&[("csrf", "nope")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    for _ in 0..200 {
        let i: serde_json::Value = c
            .get(s.url("/v1/indexes/films"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if i["status"] == "ready" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let r = c
        .post(s.url("/ui/indexes/films/delete"))
        .form(&[("csrf", t.as_str())])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    let loc = r.headers()["location"].to_str().unwrap().to_string();
    let html = c
        .get(s.url(&loc))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("Index films deleted."));
    assert!(!html.contains("/ui/indexes/films\""));
    s.stop().await;
}
