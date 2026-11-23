//! Design test "Auth on/off".
mod common;
use common::{client, csrf, TestServer};
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_enabled() {
    let dir = tempfile::tempdir().unwrap();
    let s = TestServer::start(dir.path(), Some("s3cret")).await;
    let c = client();
    // healthz + assets always open
    assert_eq!(c.get(s.url("/healthz")).send().await.unwrap().status(), 200);
    assert_eq!(
        c.get(s.url("/assets/app.css"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    for path in ["/v1/indexes", "/metrics", "/ui"] {
        let r = c.get(s.url(path)).send().await.unwrap();
        assert_eq!(r.status(), 401, "{path} without token");
        if path != "/ui" {
            let v: serde_json::Value = r.json().await.unwrap();
            assert_eq!(v["error"]["code"], "unauthorized");
        }
        let r = c
            .get(s.url(path))
            .bearer_auth("wrong")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 401, "{path} wrong token");
        let r = c
            .get(s.url(path))
            .bearer_auth("s3cret")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{path} correct token");
    }
    let r = c
        .post(s.url("/v1/indexes"))
        .json(&json!({"name":"a","dimension":2,"metric":"dot"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);

    // browser flow: redirect to login, log in, cookie grants access
    let r = c
        .get(s.url("/ui"))
        .header("accept", "text/html")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    assert_eq!(r.headers()["location"], "/ui/login");
    let page = c
        .get(s.url("/ui/login"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let token = csrf(&page);
    let r = c
        .post(s.url("/ui/login"))
        .form(&[("csrf", token.as_str()), ("key", "nope")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    let r = c
        .post(s.url("/ui/login"))
        .form(&[("csrf", "forged"), ("key", "s3cret")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let r = c
        .post(s.url("/ui/login"))
        .form(&[("csrf", token.as_str()), ("key", "s3cret")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    let cookie = r.headers()["set-cookie"].to_str().unwrap().to_string();
    assert!(cookie.contains("HttpOnly") && cookie.contains("SameSite=Strict"));
    assert!(
        !cookie.contains("s3cret"),
        "cookie must not contain the raw key"
    );
    let session = cookie.split(';').next().unwrap().to_string();
    let r = c
        .get(s.url("/ui"))
        .header("cookie", &session)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    // the cookie does not unlock the JSON API
    assert_eq!(
        c.get(s.url("/v1/indexes"))
            .header("cookie", &session)
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    // a forged cookie does not work
    let r = c
        .get(s.url("/ui"))
        .header("cookie", "exidx_session=deadbeef")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    // CSRF is bound to the session
    let page = c
        .get(s.url("/ui/new"))
        .header("cookie", &session)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let t = csrf(&page);
    let r = c
        .post(s.url("/ui/new"))
        .header("cookie", &session)
        .form(&[
            ("csrf", "x"),
            ("name", "a"),
            ("dimension", "2"),
            ("metric", "dot"),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let r = c
        .post(s.url("/ui/new"))
        .header("cookie", &session)
        .form(&[
            ("csrf", t.as_str()),
            ("name", "a"),
            ("dimension", "2"),
            ("metric", "dot"),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 303);
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let s = TestServer::start(dir.path(), None).await;
    let c = client();
    for path in ["/healthz", "/v1/indexes", "/metrics", "/ui"] {
        assert_eq!(
            c.get(s.url(path)).send().await.unwrap().status(),
            200,
            "{path}"
        );
        assert_eq!(
            c.get(s.url(path))
                .bearer_auth("anything")
                .send()
                .await
                .unwrap()
                .status(),
            200,
            "{path}"
        );
    }
    // login page just redirects when auth is off
    assert_eq!(
        c.get(s.url("/ui/login")).send().await.unwrap().status(),
        303
    );
    s.stop().await;
}
