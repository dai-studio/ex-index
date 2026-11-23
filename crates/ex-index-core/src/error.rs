//! Shared error type. Each variant maps to one public API error code.

/// Result alias used across the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Bad input from the caller (HTTP 400, code `invalid_argument`).
    #[error("{0}")]
    InvalidArgument(String),
    /// Missing index or vector (HTTP 404, code `not_found`).
    #[error("{0}")]
    NotFound(String),
    /// Name already exists / build already running (HTTP 409, code `conflict`).
    #[error("{0}")]
    Conflict(String),
    /// On-disk data failed validation.
    #[error("corrupt data: {0}")]
    Corrupt(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Internal(String),
}

impl Error {
    /// Stable machine-readable error code used in `{"error":{"code",..}}`.
    pub fn code(&self) -> &'static str {
        match self {
            Error::InvalidArgument(_) => "invalid_argument",
            Error::NotFound(_) => "not_found",
            Error::Conflict(_) => "conflict",
            Error::Corrupt(_) | Error::Io(_) | Error::Internal(_) => "internal",
        }
    }

    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::InvalidArgument(msg.into())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Corrupt(e.to_string())
    }
}
