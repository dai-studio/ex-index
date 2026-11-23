# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
