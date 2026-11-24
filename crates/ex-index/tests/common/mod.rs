#![allow(dead_code)]
use ex_index::ServeArgs;
use std::net::SocketAddr;
use std::path::Path;

pub struct TestServer {
    pub addr: SocketAddr,
    pub base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl TestServer {
    pub async fn start(data_dir: &Path, api_key: Option<&str>) -> TestServer {
        let mut args = ServeArgs::for_data_dir(data_dir);
        args.api_key = api_key.map(String::from);
        Self::start_with(args).await
    }

    pub async fn start_with(args: ServeArgs) -> TestServer {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            ex_index::serve(args, listener, async move {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        TestServer {
            addr,
            base: format!("http://{addr}"),
            shutdown: Some(tx),
            handle: Some(handle),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            h.await.unwrap();
        }
    }
}

pub fn client() -> reqwest::Client {
    ex_index::embed::install_crypto_provider();
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap()
}

/// Extract the CSRF token from a rendered UI page.
pub fn csrf(html: &str) -> String {
    let i = html.find("name=\"csrf\" value=\"").expect("csrf field");
    let rest = &html[i + 19..];
    rest[..rest.find('"').unwrap()].to_string()
}
