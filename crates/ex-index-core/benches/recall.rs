//! Design test "Recall & latency bench" — run locally, not in CI:
//!
//! ```sh
//! cargo bench -p ex-index-core --bench recall
//! # knobs: EX_BENCH_N (100000), EX_BENCH_DIM (128), EX_BENCH_QUERIES (1000),
//! #        EX_BENCH_METRICS (cosine,euclidean,dot), EX_BENCH_DATA (iid,lowrank),
//! #        EX_BENCH_L (unset = index default l_search 100)
//! ```
//!
//! For each dataset × metric: create an index through the store, upsert
//! N×DIM vectors, build the on-disk DiskANN index (Vamana + PQ), then run
//! single queries (warm cache, default build params) and report recall@10
//! against brute force plus p50 / p99 latency. Targets: recall@10 ≥ 0.95,
//! p99 < 10 ms.
//!
//! Datasets:
//! * `iid` — i.i.d. Gaussian (normalized for cosine, varying norms for dot).
//!   Full-rank random data is the worst case for every graph ANN method.
//! * `lowrank` — 16-dimensional latent structure projected to DIM plus noise,
//!   a proxy for real embedding models.

use ex_index_core::brute::BruteForce;
use ex_index_core::distance::normalize;
use ex_index_core::harness::{gaussian_vectors, low_rank_vectors, percentile, recall_at_k};
use ex_index_core::models::{CreateIndexRequest, SearchRequest, VectorRecord};
use ex_index_core::store::{Catalog, StoreOptions};
use ex_index_core::Metric;
use rayon::prelude::*;
use std::time::{Duration, Instant};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // `cargo bench` passes `--bench`; ignore arguments.
    let n = env("EX_BENCH_N", 100_000);
    let d = env("EX_BENCH_DIM", 128);
    let nq = env("EX_BENCH_QUERIES", 1000);
    let metrics: Vec<Metric> = std::env::var("EX_BENCH_METRICS")
        .unwrap_or_else(|_| "cosine,euclidean,dot".into())
        .split(',')
        .filter_map(|m| Metric::parse(m.trim()))
        .collect();
    let datasets: Vec<String> = std::env::var("EX_BENCH_DATA")
        .unwrap_or_else(|_| "iid,lowrank".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let l_override: Option<u32> = std::env::var("EX_BENCH_L")
        .ok()
        .and_then(|v| v.parse().ok());
    let l_label = l_override
        .map(|l| l.to_string())
        .unwrap_or_else(|| "100 (default)".into());
    println!("ex-index recall & latency bench: n={n} dim={d} queries={nq} k=10 l_search={l_label}");
    println!(
        "{:<8} {:<10} {:>8} {:>9} {:>8} {:>8} {:>8} {:>7}",
        "data", "metric", "build_s", "recall10", "p50_ms", "p99_ms", "mean_ms", "result"
    );
    let mut all_ok = true;
    for dataset in &datasets {
        for &metric in &metrics {
            // queries come from the same distribution as the data
            let (mut data, queries) = {
                let all = match dataset.as_str() {
                    "lowrank" => low_rank_vectors(n + nq, d, 16, 7),
                    _ => gaussian_vectors(n + nq, d, 2),
                };
                (all[..n * d].to_vec(), all[n * d..].to_vec())
            };
            if metric == Metric::Dot {
                // varying norms so inner product differs from cosine
                let mut s = 0x2545_F491_4F6C_DD1Du64;
                for row in data.chunks_exact_mut(d) {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let f = 0.25 + 1.5 * ((s >> 40) as f32 / (1u64 << 24) as f32);
                    row.iter_mut().for_each(|x| *x *= f);
                }
            }

            let dir = tempfile::tempdir().expect("tempdir");
            let opts = StoreOptions {
                auto_build: false,
                ..StoreOptions::default()
            };
            let cat = Catalog::open(dir.path(), opts).expect("catalog");
            let idx = cat
                .create(CreateIndexRequest {
                    name: "bench".into(),
                    dimension: Some(d as u32),
                    metric,
                    params: None,
                    embedding_model: None,
                })
                .expect("create");
            let recs: Vec<VectorRecord> = data
                .chunks_exact(d)
                .enumerate()
                .map(|(i, v)| VectorRecord {
                    id: i.to_string(),
                    vector: v.to_vec(),
                    metadata: None,
                })
                .collect();
            for chunk in recs.chunks(10_000) {
                idx.upsert(chunk.to_vec()).expect("upsert");
            }
            let t0 = Instant::now();
            idx.start_build().expect("build");
            assert!(
                idx.wait_for_build(Duration::from_secs(3600)),
                "build timed out"
            );
            if let Some(e) = idx.last_build_error() {
                panic!("build failed: {e}");
            }
            let build_s = t0.elapsed().as_secs_f64();

            // ground truth (brute force, prepared vectors)
            let mut stored = data.clone();
            let mut qs = queries.clone();
            if metric == Metric::Cosine {
                stored.chunks_exact_mut(d).for_each(|r| {
                    normalize(r);
                });
                qs.chunks_exact_mut(d).for_each(|r| {
                    normalize(r);
                });
            }
            let bf = BruteForce::new(metric, d, stored);
            let truth: Vec<Vec<u32>> = qs
                .par_chunks_exact(d)
                .map(|q| bf.search(q, 10).into_iter().map(|(i, _)| i).collect())
                .collect();

            // warm-up, then timed single queries
            for q in queries.chunks_exact(d).take(50) {
                let mut req = SearchRequest::new(q.to_vec(), 10);
                req.l_search = l_override;
                idx.search(&req).unwrap();
            }
            let mut lat = Vec::with_capacity(nq);
            let mut got = Vec::with_capacity(nq);
            for q in queries.chunks_exact(d) {
                let mut req = SearchRequest::new(q.to_vec(), 10);
                req.l_search = l_override;
                let t = Instant::now();
                let hits = idx.search(&req).unwrap();
                lat.push(t.elapsed().as_secs_f64() * 1000.0);
                got.push(
                    hits.iter()
                        .map(|h| h.id.parse::<u32>().unwrap())
                        .collect::<Vec<_>>(),
                );
            }
            let recall = recall_at_k(&truth, &got, 10);
            let mean = lat.iter().sum::<f64>() / lat.len() as f64;
            let p50 = percentile(&mut lat, 50.0);
            let p99 = percentile(&mut lat, 99.0);
            let ok = recall >= 0.95 && p99 < 10.0;
            all_ok &= ok;
            println!(
                "{:<8} {:<10} {:>8.1} {:>9.4} {:>8.3} {:>8.3} {:>8.3} {:>7}",
                dataset,
                metric.as_str(),
                build_s,
                recall,
                p50,
                p99,
                mean,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    println!(
        "{}",
        if all_ok {
            "ALL TARGETS MET"
        } else {
            "SOME TARGETS MISSED"
        }
    );
}
