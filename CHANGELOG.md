# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Cloudflare Workers AI embeddings. An index can be created with an `embedding_model`
  (`@cf/qwen/qwen3-embedding-0.6b`, `@cf/baai/bge-m3`, `@cf/baai/bge-small-en-v1.5`,
  `@cf/baai/bge-base-en-v1.5`, `@cf/baai/bge-large-en-v1.5`); `dimension` then defaults to the
  model's.
- `POST /v1/indexes/{name}/documents` embeds `{id, text, metadata?}` records and upserts them.
- `search` accepts `"text"` instead of `"vector"` on indexes with an embedding model.
- Admin UI: embedding-model picker, "Add documents" form, and text search.
- `--cloudflare-account-id` / `CLOUDFLARE_ACCOUNT_ID` and `--cloudflare-api-token` /
  `CLOUDFLARE_API_TOKEN`.
- Metrics `ex_index_embed_seconds` and `ex_index_embedded_texts_total`.
- Search responses include `search_ms` and, for text queries, `embed_ms`; `took_ms` remains the
  total. The admin UI shows the same split.

### Changed
- `ex_index_search_seconds` now measures only the index search, not query embedding (which is
  in `ex_index_embed_seconds`).

### Fixed
- Admin UI: link-styled buttons (e.g. "Create index") rendered dark text on a dark background.

## [0.1.0] - 2026-11-23

### Added
- In-house DiskANN engine (`ex-index-core`):
  - Vamana graph with a parallel two-pass build.
  - Product quantization with ADC tables. Dot indexes use inner-product ADC.
  - 4 KiB-sector on-disk layout read through mmap, with PQ-navigated beam search and
    full-vector re-ranking.
- Cosine, euclidean and dot-product metrics with raw per-metric scores. Dot uses the MIPS→L2
  transform, with the 64 largest-norm vectors as extra entry points.
- Storage: fsynced WAL, in-memory Vamana delta, tombstone deletes, background rebuild with
  atomic generation swap, crash recovery, and exact brute-force search below 10k vectors.
- Best-effort top-level metadata equality filter with adaptive over-fetch.
- `ex-index serve`: versioned JSON REST API (`/v1`), `/healthz`, and Prometheus `/metrics`.
- Optional bearer API key with constant-time comparison.
- Server-rendered admin UI (`/ui`) built with askama and shadcn-html, embedded in the binary,
  with login cookie and CSRF protection.
- Recall and latency benchmark, plus a CI matrix (Linux, macOS, Windows) and a cross-compiled
  release workflow.
