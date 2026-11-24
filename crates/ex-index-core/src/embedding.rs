//! Catalog of hosted embedding models an index can be bound to.
//!
//! The core crate only knows the static facts (id, output dimension, batch
//! limit). The HTTP calls live in the server crate.

/// A Cloudflare Workers AI text-embedding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingModel {
    /// Workers AI model id, e.g. `@cf/baai/bge-base-en-v1.5`.
    pub id: &'static str,
    /// Output vector length.
    pub dimension: u32,
    /// Max number of input strings per API call.
    pub max_batch: usize,
    /// Request field used for search queries (documents always use `text`).
    pub query_field: &'static str,
}

pub const EMBEDDING_MODELS: &[EmbeddingModel] = &[
    EmbeddingModel {
        id: "@cf/qwen/qwen3-embedding-0.6b",
        dimension: 1024,
        max_batch: 32,
        query_field: "queries",
    },
    EmbeddingModel {
        id: "@cf/baai/bge-m3",
        dimension: 1024,
        max_batch: 32,
        query_field: "text",
    },
    EmbeddingModel {
        id: "@cf/baai/bge-small-en-v1.5",
        dimension: 384,
        max_batch: 100,
        query_field: "text",
    },
    EmbeddingModel {
        id: "@cf/baai/bge-base-en-v1.5",
        dimension: 768,
        max_batch: 100,
        query_field: "text",
    },
    EmbeddingModel {
        id: "@cf/baai/bge-large-en-v1.5",
        dimension: 1024,
        max_batch: 100,
        query_field: "text",
    },
];

pub fn embedding_model(id: &str) -> Option<&'static EmbeddingModel> {
    EMBEDDING_MODELS.iter().find(|m| m.id == id)
}

pub fn supported_ids() -> String {
    EMBEDDING_MODELS
        .iter()
        .map(|m| m.id)
        .collect::<Vec<_>>()
        .join(", ")
}
