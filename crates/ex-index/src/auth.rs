//! Optional API key (design decision "Optional API key").
//!
//! * `/v1` and `/metrics`: `Authorization: Bearer <key>`, compared in
//!   constant time. Missing / wrong → 401 JSON.
//! * `/ui`: bearer header, or the login cookie holding
//!   `HMAC-SHA256(key, "ex-index-ui-session")` (HttpOnly, SameSite=Strict).
//!   Unauthenticated browsers are redirected to `/ui/login`.
//! * Every UI form post carries a CSRF token `HMAC(secret, session)`.
//! * `/healthz` and `/assets` are always open; with no key everything is open.

use crate::error::ApiError;
use crate::AppState;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

pub const SESSION_COOKIE: &str = "exidx_session";

pub struct Auth {
    key: Option<String>,
    session_token: Option<String>,
    csrf_secret: [u8; 32],
}

/// HMAC-SHA256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(k.map(|b| b ^ 0x36));
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(k.map(|b| b ^ 0x5c));
    outer.update(&ih[..]);
    let mut out = [0u8; 32];
    out.copy_from_slice(&outer.finalize()[..]);
    out
}

/// Constant-time equality (hashes both sides so lengths don't leak).
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let (ha, hb) = (Sha256::digest(a), Sha256::digest(b));
    bool::from(ha[..].ct_eq(&hb[..]))
}

impl Auth {
    pub fn new(key: Option<String>) -> Result<Auth, crate::BoxError> {
        let key = key.filter(|k| !k.is_empty());
        let mut csrf_secret = [0u8; 32];
        getrandom::fill(&mut csrf_secret).map_err(|e| format!("random source unavailable: {e}"))?;
        let session_token = key
            .as_ref()
            .map(|k| hex::encode(hmac_sha256(k.as_bytes(), b"ex-index-ui-session")));
        Ok(Auth {
            key,
            session_token,
            csrf_secret,
        })
    }

    pub fn enabled(&self) -> bool {
        self.key.is_some()
    }

    pub fn check_bearer(&self, headers: &HeaderMap) -> bool {
        let Some(key) = &self.key else { return true };
        headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| {
                v.strip_prefix("Bearer ")
                    .or_else(|| v.strip_prefix("bearer "))
            })
            .map(|t| ct_eq(t.trim().as_bytes(), key.as_bytes()))
            .unwrap_or(false)
    }

    /// Value of the session cookie, if present.
    pub fn session_cookie(headers: &HeaderMap) -> Option<String> {
        headers
            .get_all(header::COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .find_map(|v| {
                v.split(';').find_map(|kv| {
                    let (k, val) = kv.trim().split_once('=')?;
                    (k == SESSION_COOKIE).then(|| val.to_string())
                })
            })
    }

    pub fn check_cookie(&self, headers: &HeaderMap) -> bool {
        let Some(tok) = &self.session_token else {
            return true;
        };
        Self::session_cookie(headers)
            .map(|c| ct_eq(c.as_bytes(), tok.as_bytes()))
            .unwrap_or(false)
    }

    /// Validate a submitted API key (login form).
    pub fn check_key(&self, submitted: &str) -> bool {
        match &self.key {
            None => true,
            Some(k) => ct_eq(submitted.as_bytes(), k.as_bytes()),
        }
    }

    /// `Set-Cookie` value for a successful login.
    pub fn login_cookie(&self) -> Option<String> {
        self.session_token.as_ref().map(|t| {
            format!("{SESSION_COOKIE}={t}; Path=/ui; HttpOnly; SameSite=Strict; Max-Age=2592000")
        })
    }

    pub fn logout_cookie() -> String {
        format!("{SESSION_COOKIE}=; Path=/ui; HttpOnly; SameSite=Strict; Max-Age=0")
    }

    /// CSRF token bound to the caller's session (or "anon" when auth is off).
    pub fn csrf_token(&self, headers: &HeaderMap) -> String {
        let session = Self::session_cookie(headers).unwrap_or_else(|| "anon".to_string());
        hex::encode(hmac_sha256(&self.csrf_secret, session.as_bytes()))
    }

    pub fn check_csrf(&self, headers: &HeaderMap, token: &str) -> bool {
        ct_eq(self.csrf_token(headers).as_bytes(), token.as_bytes())
    }
}

/// Bearer check for `/v1` and `/metrics`.
pub async fn require_api(State(s): State<AppState>, req: Request, next: Next) -> Response {
    if s.auth.check_bearer(req.headers()) {
        next.run(req).await
    } else {
        ApiError::unauthorized().into_response()
    }
}

/// Bearer-or-cookie check for `/ui` (redirects browsers to the login page).
pub async fn require_ui(State(s): State<AppState>, req: Request, next: Next) -> Response {
    if !s.auth.enabled() || s.auth.check_bearer(req.headers()) || s.auth.check_cookie(req.headers())
    {
        return next.run(req).await;
    }
    let wants_html = req
        .headers()
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|a| a.contains("text/html"))
        .unwrap_or(false);
    if wants_html && req.method() == axum::http::Method::GET {
        Redirect::to("/ui/login").into_response()
    } else {
        ApiError::unauthorized().into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hmac_rfc4231_case_2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex::encode(mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }
    #[test]
    fn ct_eq_works() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"abcd"));
    }
}
