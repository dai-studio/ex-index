# ex-index

A vector search server in a single static binary. It runs an in-house Rust implementation of
**DiskANN**: a Vamana graph, product quantization, and an SSD-friendly on-disk layout. You create
indexes, add embedding vectors with JSON metadata, and search by **cosine**, **euclidean** or
**dot-product** similarity, either through a versioned JSON REST API or a server-rendered admin
UI.

- One self-contained executable. Templates, CSS and TLS root certificates are embedded.
- Optional text embedding through Cloudflare Workers AI: add documents as text and search by
  text.
- All state lives under `--data-dir`. Writes are fsynced to a WAL before they are acknowledged
  and become searchable immediately.
- Optional bearer API key covering `/v1`, `/ui` and `/metrics`.
- Prometheus metrics at `/metrics`, health at `/healthz`.

## Quickstart

```sh
cargo build --release
./target/release/ex-index serve --port 7700 --data-dir ./data
```

```sh
# create an index
curl -s -X POST localhost:7700/v1/indexes -H 'content-type: application/json' \
  -d '{"name":"docs","dimension":3,"metric":"cosine"}'

# upsert vectors with metadata
curl -s -X POST localhost:7700/v1/indexes/docs/vectors -H 'content-type: application/json' -d '{
  "vectors": [
    {"id": "a", "vector": [1, 0, 0], "metadata": {"lang": "en"}},
    {"id": "b", "vector": [0, 1, 0], "metadata": {"lang": "de"}}
  ]}'

# search
curl -s -X POST localhost:7700/v1/indexes/docs/search -H 'content-type: application/json' \
  -d '{"vector": [0.9, 0.1, 0], "k": 5, "filter": {"lang": "en"}}'
# {"hits":[{"id":"a","score":0.99388373,"metadata":{"lang":"en"}}],"took_ms":0.05}
```

The admin UI is at <http://localhost:7700/ui>.

## Configuration

| Flag | Env | Default | |
|---|---|---|---|
| `--host` | `EX_INDEX_HOST` | `0.0.0.0` | Interface to bind |
| `--port` | `EX_INDEX_PORT` | `7700` | TCP port |
| `--data-dir` | `EX_INDEX_DATA_DIR` | `./data` | Holds every index (one sub-directory each) |
| `--api-key` | `EX_INDEX_API_KEY` | unset (auth off) | Require `Authorization: Bearer <key>` |
| `--delta-threshold` | `EX_INDEX_DELTA_THRESHOLD` | `50000` | Rebuild in the background once the in-memory delta holds this many vectors |
| `--log-level` | `EX_INDEX_LOG_LEVEL` | `info` | `tracing` filter, e.g. `debug` or `ex_index=debug` |
| `--cloudflare-account-id` | `CLOUDFLARE_ACCOUNT_ID` | unset | Cloudflare account for Workers AI embeddings |
| `--cloudflare-api-token` | `CLOUDFLARE_API_TOKEN` | unset | API token with Workers AI access |

On SIGINT or SIGTERM the server stops accepting connections, drains in-flight requests, fsyncs
every WAL and closes the index files.

## Concepts

**Index.** An index has a name (`^[a-z0-9_-]{1,64}$`), a fixed `dimension` (1–4096) and a fixed
`metric`. Both are immutable after creation. Optional `params`:

| param | default | meaning |
|---|---|---|
| `r` | 64 | max out-degree of the Vamana graph |
| `l_build` | 100 | candidate list size while building |
| `alpha` | 1.2 | RobustPrune slack |
| `pq_subvectors` | `ceil(dimension/4)` | PQ bytes per vector held in RAM |
| `l_search` | 100 | default search candidate list size |

**Embedding model.** Set `embedding_model` when you create an index to have ex-index embed
text for you with Cloudflare Workers AI. `dimension` can then be omitted (it is taken from the
model) and must match the model if given. The model is fixed for the life of the index.

| `embedding_model` | dimension | max texts per Cloudflare call |
|---|---|---|
| `@cf/qwen/qwen3-embedding-0.6b` | 1024 | 32 |
| `@cf/baai/bge-m3` | 1024 | 32 |
| `@cf/baai/bge-small-en-v1.5` | 384 | 100 |
| `@cf/baai/bge-base-en-v1.5` | 768 | 100 |
| `@cf/baai/bge-large-en-v1.5` | 1024 | 100 |

Such an index accepts `POST /documents` with text, and `search` with `"text"` instead of
`"vector"`. Larger requests are split into batches automatically. Documents are sent in the
`text` field; search queries for qwen3 go in the `queries` field (which applies its retrieval
instruction), for the other models in `text`. The bge `en` models read at most 512 tokens per
text. Plain vectors can still be upserted into these indexes. The server needs
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`; without them these calls return 503.
The text itself is not stored; put it in `metadata` if you want it back in search results.

```sh
export CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=...
./target/release/ex-index serve

curl -s -X POST localhost:7700/v1/indexes -H 'content-type: application/json' \
  -d '{"name":"stories","metric":"cosine","embedding_model":"@cf/baai/bge-base-en-v1.5"}'
curl -s -X POST localhost:7700/v1/indexes/stories/documents -H 'content-type: application/json' -d '{
  "documents": [
    {"id": "cloud", "text": "This is a story about an orange cloud", "metadata": {"title": "Cloud"}},
    {"id": "llama", "text": "This is a story about a llama"}
  ]}'
curl -s -X POST localhost:7700/v1/indexes/stories/search -H 'content-type: application/json' \
  -d '{"text": "orange sky", "k": 1}'
```

**Vectors.** Each vector has a string `id` (1–256 chars), `vector` (exactly `dimension` finite
floats) and optional `metadata` (a JSON object, ≤ 64 KiB). Upserting an existing id replaces it.
A batch is rejected as a whole (400) if any record is invalid. Cosine indexes reject zero-norm
vectors and store vectors L2-normalized, so `GET` returns the normalized vector.

**Scores.** Results are always sorted best-first:

| metric | `score` | better |
|---|---|---|
| `cosine` | cosine similarity | higher |
| `euclidean` | true L2 distance (with the square root applied) | lower |
| `dot` | raw inner product of the original vectors | higher |

**Metadata filter.** `filter` is a JSON object of top-level `key: value` pairs, AND-ed, compared
by exact JSON equality. Below 10,000 live vectors the search is exact brute force and the filter
result is exact. Above that the search over-fetches adaptively, starting at `k×4` candidates and
doubling up to `l_search×8`. This is **best-effort**: a very selective filter may return fewer
than `k` hits.

**Lifecycle.** Writes are appended to a WAL (fsync before ack) and inserted into an in-memory
delta (itself a Vamana graph once the index has a disk generation). Search covers the disk index
and the delta together, then merges the results and drops tombstones. Deletes are tombstones.
Rebuilds run in the background:

- the first one starts automatically when an index reaches 10,000 live vectors;
- later ones start when the delta reaches `--delta-threshold`, or on `POST /build`.

A rebuild merges everything into a new on-disk generation, drops tombstoned vectors, and swaps
the new generation in atomically. Writes that arrive during a rebuild are kept.

## REST API

Every route under `/v1` takes and returns JSON. When an API key is set, add
`-H 'authorization: Bearer <key>'`. Errors look like this:

```json
{"error": {"code": "invalid_argument | not_found | conflict | unauthorized | upstream | unavailable | internal", "message": "..."}}
```

| Method & path | Body | Success | Errors |
|---|---|---|---|
| `POST /v1/indexes` | `{name, dimension?, metric, params?, embedding_model?}` | 201 `IndexInfo` | 400, 409 |
| `GET /v1/indexes` | | 200 `{indexes: [IndexInfo]}` | |
| `GET /v1/indexes/{name}` | | 200 `IndexInfo` | 404 |
| `DELETE /v1/indexes/{name}` | | 204 | 404 |
| `POST /v1/indexes/{name}/vectors` | `{vectors: [VectorRecord]}` (≤ 10k) | 200 `{upserted}` | 400, 404, 413 |
| `POST /v1/indexes/{name}/documents` | `{documents: [{id, text, metadata?}]}` (≤ 1000) | 200 `{upserted}` | 400, 404, 502 Cloudflare error, 503 not configured |
| `GET /v1/indexes/{name}/vectors/{id}` | | 200 `VectorRecord` | 404 |
| `DELETE /v1/indexes/{name}/vectors/{id}` | | 204 | 404 |
| `POST /v1/indexes/{name}/search` | `SearchRequest` | 200 `{hits, took_ms, embed_ms?, search_ms}` | 400, 404, 502, 503 |
| `POST /v1/indexes/{name}/build` | | 202 `IndexInfo` (`status: building`) | 404, 409 if already building |
| `GET /healthz` | | 200 `{status: "ok", version}` | (never needs auth) |
| `GET /metrics` | | 200 Prometheus text | 401 |

`SearchRequest`: `{"vector": [...], "k": 10, "l_search": null, "filter": null, "include_vector": false, "include_metadata": true}`.
Instead of `vector`, an index with an `embedding_model` accepts `"text": "..."`.
`k` must be 1–1000. If you set `l_search`, it must be ≥ `k`. A larger `l_search` gives better
recall at the cost of latency.

`IndexInfo`: `{config, count, deleted, delta_count, status: "ready"|"building", built_at?, size_bytes}`.

Request bodies are limited to 64 MiB. A full batch of 10,000 × 768-d vectors fits only with
compact float formatting (about 7 characters per number). Split larger payloads into several
requests.

```sh
curl -s localhost:7700/v1/indexes/docs
curl -s localhost:7700/v1/indexes/docs/vectors/a
curl -s -X DELETE localhost:7700/v1/indexes/docs/vectors/a
curl -s -X POST localhost:7700/v1/indexes/docs/build
curl -s -X DELETE localhost:7700/v1/indexes/docs
```

## Admin UI

The UI is plain server-rendered HTML (askama templates styled with
[shadcn-html](https://github.com/codylindley/shadcn-html)) and needs no JavaScript. Pages:

- `/ui`: all indexes, with open, rebuild and delete actions.
- `/ui/new`: create an index.
- `/ui/indexes/{name}`: stats, configuration, JSON upsert, and lookup or delete by id. Indexes
  with an embedding model also get an "Add documents" form for text.
- `/ui/indexes/{name}/search`: search playground (by text when the index has a model).

When an API key is set, `/ui/login` exchanges the key for an `HttpOnly; SameSite=Strict` cookie.
The cookie holds an HMAC of the key, never the key itself. Every form post carries a CSRF token.

## Data directory

```
<data-dir>/<index>/
  manifest.json      config, current generation, first live WAL (atomic temp+rename = commit point)
  main.<gen>/        immutable disk generation
    graph.bin        4 KiB header sector + node records [f32 × d'][u32 count][u32 × R]
    pq.bin           PQ codebook + codes
    ids.bin          slot → string id + metadata offsets
    metadata.log     metadata JSON, append-only
    tombstones.bin   tombstone bitset checkpoint
  wal.<gen>.log      un-merged upserts/deletes (CRC-framed; torn tails truncated on start)
```

All files are little-endian, so a directory can be copied between hosts and operating systems.
In `graph.bin`, records smaller than 4 KiB are packed into sectors without crossing a sector
boundary. Larger records span whole sectors. `d'` is `dimension`, or `dimension + 1` for dot
indexes because of the MIPS→L2 transform.

## Memory sizing

Rough RAM needs per **1M vectors at 128-d** with default params:

| | RAM |
|---|---|
| serving (PQ codes 32 B + id map ≈ 60–100 B per vector) | ≈ 100–150 MB, plus the OS page cache for `graph.bin` (≈ 0.8 GB on disk) |
| rebuild (full vectors + graph in RAM while building) | ≈ 1.0–1.5 GB transient |
| in-memory delta | ≈ (4·d + 4·R + metadata) bytes per un-merged vector |

## Performance

Run the recall and latency benchmark locally. It is deliberately not part of CI because timings
depend on the machine:

```sh
cargo bench -p ex-index-core --bench recall
# knobs: EX_BENCH_N, EX_BENCH_DIM, EX_BENCH_QUERIES, EX_BENCH_METRICS, EX_BENCH_DATA=iid,lowrank, EX_BENCH_L
```

Results on an Apple M4 (10 cores) with 100k × 128-d vectors, 1,000 single queries, k = 10,
default params (`l_search` = 100), disk index with PQ:

| data | metric | build | recall@10 | p50 | p99 |
|---|---|---|---|---|---|
| low-rank (16-d latent, ≈ real embeddings) | cosine | 10 s | 1.000 | 0.18 ms | 0.24 ms |
| low-rank | euclidean | 9 s | 1.000 | 0.21 ms | 0.35 ms |
| low-rank | dot | 8 s | 0.958 | 0.10 ms | 0.16 ms |
| i.i.d. Gaussian (full rank) | cosine | 26 s | 0.578 | 0.25 ms | 0.35 ms |
| i.i.d. Gaussian | euclidean | 31 s | 0.745 | 0.21 ms | 0.26 ms |
| i.i.d. Gaussian | dot | 9 s | 0.036 | 0.06 ms | 0.09 ms |

Full-rank i.i.d. Gaussian data has no neighborhood structure, which makes it the worst case for
every graph-based ANN method. On it, recall depends mostly on `l_search`. At `l_search` = 1000,
cosine reaches 0.994 and euclidean 0.963, with p99 latency of 2.7 ms and 1.5 ms. Dot product is a special case. The MIPS→L2 reduction
puts every query far outside the data sphere, and on full-rank data with widely varying norms
greedy search cannot reach the top inner-product items. If inner-product search on such data
matters, normalize the vectors and use cosine. Real embedding models produce data much closer to
the low-rank rows.

## Development

```sh
cargo test --workspace          # unit + integration tests (spawns servers on random ports)
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p ex-index-core    # recall/latency + distance micro-benchmarks
```

The workspace has two crates:

- `crates/ex-index-core`: distance, Vamana, PQ, disk format, WAL and store.
- `crates/ex-index`: the axum server, auth, UI and the CLI.

CI (`.github/workflows/ci.yml`) runs fmt, clippy and tests on Linux, macOS and Windows. Pushing a
`v*` tag runs `.github/workflows/release.yml`, which builds static binaries for these targets:

- x86_64 and aarch64 Linux (musl)
- x86_64 and aarch64 macOS
- x86_64 and aarch64 Windows

It smoke-tests `/healthz` on each runnable target and attaches the archives to a GitHub release.

## License

MIT OR Apache-2.0. The vendored shadcn-html CSS is MIT-licensed; see
`crates/ex-index/assets/shadcn/LICENSE`.
