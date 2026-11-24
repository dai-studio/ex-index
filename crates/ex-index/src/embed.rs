//! Cloudflare Workers AI embeddings client.
//!
//! `POST {base}/accounts/{account_id}/ai/run/{model}` with
//! `{"text": [...]}` (documents) or `{"queries": [...]}` (qwen3 queries);
//! the response is `{"success", "errors", "result": {"shape", "data"}}`.

use crate::error::ApiError;
use axum::http::StatusCode;
use ex_index_core::embedding::EmbeddingModel;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

pub const DEFAULT_API_BASE: &str = "https://api.cloudflare.com/client/v4";

/// What a text is embedded for. Some models encode queries differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    Document,
    Query,
}

/// Install rustls' ring provider process-wide (idempotent). Must run before
/// any `reqwest::Client` is built.
pub fn install_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

pub struct Embedder {
    client: reqwest::Client,
    base: String,
    creds: Option<(String, String)>,
}

impl Embedder {
    /// `account_id` and `api_token` must both be set for embedding calls to work.
    pub fn new(
        base: &str,
        account_id: Option<String>,
        api_token: Option<String>,
    ) -> Result<Self, crate::BoxError> {
        install_crypto_provider();
        let roots = rustls::RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        };
        let mut tls = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        tls.alpn_protocols = vec![b"http/1.1".to_vec()];
        let client = reqwest::Client::builder()
            .tls_backend_preconfigured(tls)
            .timeout(Duration::from_secs(60))
            .user_agent(concat!("ex-index/", env!("CARGO_PKG_VERSION")))
            .build()?;
        let nonempty =
            |s: Option<String>| s.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let creds = match (nonempty(account_id), nonempty(api_token)) {
            (Some(a), Some(t)) => Some((a, t)),
            _ => None,
        };
        Ok(Embedder {
            client,
            base: base.trim_end_matches('/').to_string(),
            creds,
        })
    }

    pub fn configured(&self) -> bool {
        self.creds.is_some()
    }

    /// Embed `texts` in batches of `model.max_batch`, preserving order.
    pub async fn embed(
        &self,
        model: &EmbeddingModel,
        texts: &[String],
        purpose: Purpose,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        let (account, token) = self.creds.as_ref().ok_or_else(|| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable",
                "Cloudflare embeddings are not configured: set CLOUDFLARE_ACCOUNT_ID and \
                 CLOUDFLARE_API_TOKEN (or --cloudflare-account-id / --cloudflare-api-token)",
            )
        })?;
        let url = format!("{}/accounts/{account}/ai/run/{}", self.base, model.id);
        let field = match purpose {
            Purpose::Document => "text",
            Purpose::Query => model.query_field,
        };
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(model.max_batch.max(1)) {
            let t0 = std::time::Instant::now();
            let resp = self
                .client
                .post(&url)
                .bearer_auth(token)
                .json(&json!({ field: chunk }))
                .send()
                .await
                .map_err(|e| upstream(format!("request to Cloudflare failed: {e}")))?;
            let status = resp.status();
            let body = resp
                .bytes()
                .await
                .map_err(|e| upstream(format!("reading Cloudflare response failed: {e}")))?;
            metrics::histogram!("ex_index_embed_seconds", "model" => model.id)
                .record(t0.elapsed().as_secs_f64());
            let vectors = parse_response(status.as_u16(), &body)?;
            if vectors.len() != chunk.len() {
                return Err(upstream(format!(
                    "Cloudflare returned {} embeddings for {} inputs",
                    vectors.len(),
                    chunk.len()
                )));
            }
            if let Some(v) = vectors.iter().find(|v| v.len() != model.dimension as usize) {
                return Err(upstream(format!(
                    "{} returned a {}-dimensional vector, expected {}",
                    model.id,
                    v.len(),
                    model.dimension
                )));
            }
            out.extend(vectors);
        }
        metrics::counter!("ex_index_embedded_texts_total", "model" => model.id)
            .increment(texts.len() as u64);
        Ok(out)
    }
}

pub type SharedEmbedder = Arc<Embedder>;

fn upstream(msg: impl Into<String>) -> ApiError {
    ApiError::new(StatusCode::BAD_GATEWAY, "upstream", msg)
}

#[derive(Deserialize)]
struct Envelope {
    #[serde(default)]
    success: Option<bool>,
    #[serde(default)]
    errors: Vec<Value>,
    #[serde(default)]
    result: Option<Value>,
}

/// Extract `result.data` (array of vectors) from a Workers AI response.
fn parse_response(status: u16, body: &[u8]) -> Result<Vec<Vec<f32>>, ApiError> {
    let env: Option<Envelope> = serde_json::from_slice(body).ok();
    let errors = env
        .as_ref()
        .map(|e| {
            e.errors
                .iter()
                .map(|x| match (x.get("code"), x.get("message")) {
                    (Some(c), Some(Value::String(m))) => format!("{m} (code {c})"),
                    (_, Some(Value::String(m))) => m.clone(),
                    _ => x.to_string(),
                })
                .collect::<Vec<_>>()
                .join("; ")
        })
        .unwrap_or_default();
    let failed =
        !(200..300).contains(&status) || env.as_ref().and_then(|e| e.success) == Some(false);
    if failed {
        let detail = if errors.is_empty() {
            String::from_utf8_lossy(&body[..body.len().min(300)]).into_owned()
        } else {
            errors
        };
        return Err(upstream(format!(
            "Cloudflare returned HTTP {status}: {detail}"
        )));
    }
    let data = env
        .and_then(|e| e.result)
        .and_then(|mut r| r.get_mut("data").map(Value::take))
        .ok_or_else(|| upstream("Cloudflare response has no result.data"))?;
    serde_json::from_value::<Vec<Vec<f32>>>(data).map_err(|e| {
        upstream(format!(
            "Cloudflare result.data is not a list of vectors: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_success() {
        let body = br#"{"result":{"shape":[2,3],"data":[[1,2,3],[4,5,6]],"pooling":"mean"},"success":true,"errors":[],"messages":[]}"#;
        let v = parse_response(200, body).unwrap();
        assert_eq!(v, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn reports_errors() {
        let body = br#"{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}]}"#;
        let e = parse_response(403, body).unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_GATEWAY);
        assert!(
            e.message.contains("Authentication error (code 10000)"),
            "{}",
            e.message
        );
        let e = parse_response(500, b"oops").unwrap_err();
        assert!(e.message.contains("oops"));
        let e = parse_response(200, br#"{"success":true,"result":{}}"#).unwrap_err();
        assert!(e.message.contains("result.data"));
    }
}
