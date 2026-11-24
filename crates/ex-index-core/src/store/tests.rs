use super::*;
use crate::brute::BruteForce;
use crate::harness::{gaussian_vectors, varying_norm_vectors};
use crate::models::{CreateIndexRequest, IndexParamsInput, SearchRequest, VectorRecord};
use serde_json::json;
use std::time::Duration;

fn opts(small: usize) -> StoreOptions {
    StoreOptions {
        small_index_threshold: small,
        delta_threshold: 1_000_000,
        beam_width: 4,
        auto_build: false,
    }
}

fn create(cat: &Catalog, name: &str, dim: u32, metric: Metric) -> Arc<Index> {
    cat.create(CreateIndexRequest {
        name: name.into(),
        dimension: Some(dim),
        metric,
        params: Some(IndexParamsInput {
            r: Some(32),
            l_build: Some(64),
            ..Default::default()
        }),
        embedding_model: None,
    })
    .unwrap()
}

fn records(data: &[f32], d: usize, prefix: &str, offset: usize) -> Vec<VectorRecord> {
    data.chunks_exact(d)
        .enumerate()
        .map(|(i, v)| VectorRecord {
            id: format!("{prefix}{}", i + offset),
            vector: v.to_vec(),
            metadata: Some(
                json!({"i": i + offset, "parity": (i + offset) % 2})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        })
        .collect()
}

fn ids(hits: &[SearchHit]) -> Vec<String> {
    hits.iter().map(|h| h.id.clone()).collect()
}

fn build_and_wait(idx: &Arc<Index>) {
    idx.start_build().unwrap();
    assert!(idx.wait_for_build(Duration::from_secs(120)));
    assert_eq!(idx.last_build_error(), None);
}

#[test]
fn crud_small_mode() {
    let dir = tempfile::tempdir().unwrap();
    let cat = Catalog::open(dir.path(), opts(10_000)).unwrap();
    let idx = create(&cat, "docs", 4, Metric::Cosine);
    assert!(matches!(
        cat.create(CreateIndexRequest {
            name: "docs".into(),
            dimension: Some(4),
            metric: Metric::Cosine,
            params: None,
            embedding_model: None,
        }),
        Err(Error::Conflict(_))
    ));
    let recs = vec![
        VectorRecord {
            id: "a".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            metadata: Some(json!({"lang":"en"}).as_object().unwrap().clone()),
        },
        VectorRecord {
            id: "b".into(),
            vector: vec![0.0, 2.0, 0.0, 0.0],
            metadata: None,
        },
    ];
    assert_eq!(idx.upsert(recs).unwrap(), 2);
    let a = idx.get("a").unwrap();
    assert_eq!(a.metadata.unwrap()["lang"], "en");
    assert_eq!(
        idx.get("b").unwrap().vector,
        vec![0.0, 1.0, 0.0, 0.0],
        "cosine vectors stored normalized"
    );
    // invalid batch rejected atomically
    let bad = vec![
        VectorRecord {
            id: "c".into(),
            vector: vec![1.0, 1.0, 0.0, 0.0],
            metadata: None,
        },
        VectorRecord {
            id: "d".into(),
            vector: vec![0.0; 4],
            metadata: None,
        },
    ];
    assert!(matches!(idx.upsert(bad), Err(Error::InvalidArgument(_))));
    assert!(idx.get("c").is_err());
    // re-upsert replaces
    idx.upsert(vec![VectorRecord {
        id: "a".into(),
        vector: vec![0.0, 0.0, 1.0, 0.0],
        metadata: None,
    }])
    .unwrap();
    let a = idx.get("a").unwrap();
    assert_eq!(a.vector, vec![0.0, 0.0, 1.0, 0.0]);
    assert!(a.metadata.is_none());
    let hits = idx
        .search(&SearchRequest::new(vec![0.0, 0.0, 3.0, 0.0], 5))
        .unwrap();
    assert_eq!(ids(&hits), vec!["a", "b"]);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    // delete hides
    idx.delete("a").unwrap();
    assert!(matches!(idx.delete("a"), Err(Error::NotFound(_))));
    assert_eq!(
        ids(&idx
            .search(&SearchRequest::new(vec![0.0, 0.0, 3.0, 0.0], 5))
            .unwrap()),
        vec!["b"]
    );
    let info = idx.info();
    assert_eq!((info.count, info.deleted), (1, 2));
    // delete index
    cat.delete("docs").unwrap();
    assert!(cat.get("docs").is_err());
    assert!(!dir.path().join("docs").exists());
}

/// Requirement "Similarity search (3 metrics)": top-k equals brute-force
/// ordering on a 1k fixture with l_search >= n, both in brute-force mode and
/// through the disk index.
#[test]
fn three_metrics_match_brute_force() {
    let (n, d) = (1000usize, 16usize);
    for metric in [Metric::Cosine, Metric::Euclidean, Metric::Dot] {
        for small in [10_000usize, 0] {
            let dir = tempfile::tempdir().unwrap();
            let cat = Catalog::open(dir.path(), opts(small)).unwrap();
            let idx = create(&cat, "m", d as u32, metric);
            let data = varying_norm_vectors(n, d, 4);
            idx.upsert(records(&data, d, "v", 0)).unwrap();
            if small == 0 {
                build_and_wait(&idx);
            }
            let mut stored = data.clone();
            if metric == Metric::Cosine {
                for r in stored.chunks_exact_mut(d) {
                    crate::distance::normalize(r);
                }
            }
            let bf = BruteForce::new(metric, d, stored);
            for q in gaussian_vectors(20, d, 77).chunks_exact(d) {
                let mut qp = q.to_vec();
                if metric == Metric::Cosine {
                    crate::distance::normalize(&mut qp);
                }
                let truth: Vec<String> = bf
                    .search(&qp, 10)
                    .iter()
                    .map(|(i, _)| format!("v{i}"))
                    .collect();
                let mut req = SearchRequest::new(q.to_vec(), 10);
                req.l_search = Some(n as u32);
                let hits = idx.search(&req).unwrap();
                assert_eq!(ids(&hits), truth, "metric {metric:?} small={small}");
                let best = bf.search(&qp, 1)[0].1;
                assert!((hits[0].score - metric.to_score(best)).abs() < 1e-4);
                // best-first ordering of public scores
                for w in hits.windows(2) {
                    if metric.higher_is_better() {
                        assert!(w[0].score >= w[1].score);
                    } else {
                        assert!(w[0].score <= w[1].score);
                    }
                }
            }
        }
    }
}

/// Design test "Restart persistence": upsert, delete some, rebuild, upsert
/// more (delta), crash (no flush), restart, search.
#[test]
fn restart_persistence() {
    let (n, d) = (3000usize, 12usize);
    let dir = tempfile::tempdir().unwrap();
    let data = gaussian_vectors(n, d, 1);
    let more = gaussian_vectors(500, d, 2);
    let queries = gaussian_vectors(25, d, 3);
    let run = |idx: &Arc<Index>| -> Vec<Vec<String>> {
        queries
            .chunks_exact(d)
            .map(|q| {
                let mut r = SearchRequest::new(q.to_vec(), 10);
                r.l_search = Some(200);
                ids(&idx.search(&r).unwrap())
            })
            .collect()
    };
    let before;
    {
        let cat = Catalog::open(dir.path(), opts(1000)).unwrap();
        let idx = create(&cat, "p", d as u32, Metric::Euclidean);
        for chunk in records(&data, d, "v", 0).chunks(1000) {
            idx.upsert(chunk.to_vec()).unwrap();
        }
        for i in (0..n).step_by(7) {
            idx.delete(&format!("v{i}")).unwrap();
        }
        build_and_wait(&idx);
        idx.upsert(records(&more, d, "w", 0)).unwrap();
        for i in (0..500).step_by(5) {
            idx.delete(&format!("w{i}")).unwrap();
        }
        idx.delete("v1").unwrap(); // tombstone in main after build
        before = run(&idx);
        assert_eq!(idx.info().count as usize, n - n.div_ceil(7) - 1 + 400);
        // simulated crash: no flush / graceful shutdown
    }
    let cat = Catalog::open(dir.path(), opts(1000)).unwrap();
    let idx = cat.get("p").unwrap();
    assert_eq!(run(&idx), before);
    for i in (0..n).step_by(7) {
        assert!(idx.get(&format!("v{i}")).is_err());
    }
    assert!(idx.get("v1").is_err() && idx.get("w0").is_err() && idx.get("w1").is_ok());
    assert_eq!(idx.info().count as usize, n - n.div_ceil(7) - 1 + 400);
    // second restart after a graceful flush keeps the same answers
    cat.shutdown();
    drop(cat);
    let cat = Catalog::open(dir.path(), opts(1000)).unwrap();
    assert_eq!(run(&cat.get("p").unwrap()), before);
    // the WAL was truncated to the new generation, old gens removed
    let wals: Vec<_> = crate::wal::list_generations(&dir.path().join("p")).unwrap();
    assert_eq!(wals, vec![1]);
}

#[test]
fn writes_during_build_are_kept() {
    let (n, d) = (4000usize, 8usize);
    let dir = tempfile::tempdir().unwrap();
    let cat = Catalog::open(dir.path(), opts(100)).unwrap();
    let idx = create(&cat, "b", d as u32, Metric::Euclidean);
    let data = gaussian_vectors(n, d, 5);
    idx.upsert(records(&data, d, "v", 0)).unwrap();
    idx.start_build().unwrap();
    assert!(matches!(idx.start_build(), Err(Error::Conflict(_))));
    assert_eq!(idx.info().status, IndexStatus::Building);
    // during the build: replace v0, delete v1, add x
    idx.upsert(vec![VectorRecord {
        id: "v0".into(),
        vector: vec![100.0; d],
        metadata: None,
    }])
    .unwrap();
    idx.delete("v1").unwrap();
    idx.upsert(vec![VectorRecord {
        id: "x".into(),
        vector: vec![-100.0; d],
        metadata: None,
    }])
    .unwrap();
    assert!(idx.wait_for_build(Duration::from_secs(60)));
    assert_eq!(idx.info().status, IndexStatus::Ready);
    assert_eq!(idx.get("v0").unwrap().vector, vec![100.0; d]);
    assert!(idx.get("v1").is_err());
    let hits = idx.search(&SearchRequest::new(vec![100.0; d], 1)).unwrap();
    assert_eq!(ids(&hits), vec!["v0"]);
    let hits = idx.search(&SearchRequest::new(vec![-100.0; d], 1)).unwrap();
    assert_eq!(ids(&hits), vec!["x"]);
    assert_eq!(idx.info().count as usize, n);
    // and it all survives a restart
    drop(idx);
    drop(cat);
    let cat = Catalog::open(dir.path(), opts(100)).unwrap();
    let idx = cat.get("b").unwrap();
    assert_eq!(idx.get("v0").unwrap().vector, vec![100.0; d]);
    assert!(idx.get("v1").is_err());
    assert_eq!(idx.info().count as usize, n);
}

#[test]
fn metadata_filter_exact_and_ann() {
    let (n, d) = (2000usize, 8usize);
    for small in [10_000usize, 0] {
        let dir = tempfile::tempdir().unwrap();
        let cat = Catalog::open(dir.path(), opts(small)).unwrap();
        let idx = create(&cat, "f", d as u32, Metric::Euclidean);
        let data = gaussian_vectors(n, d, 8);
        let mut recs = records(&data, d, "v", 0);
        for (i, r) in recs.iter_mut().enumerate() {
            let lang = if i % 10 == 0 { "en" } else { "de" };
            r.metadata
                .as_mut()
                .unwrap()
                .insert("lang".into(), json!(lang));
        }
        idx.upsert(recs).unwrap();
        if small == 0 {
            build_and_wait(&idx);
        }
        let mut req = SearchRequest::new(data[..d].to_vec(), 10);
        req.filter = Some(json!({"lang":"en"}).as_object().unwrap().clone());
        let hits = idx.search(&req).unwrap();
        assert_eq!(hits.len(), 10, "small={small}");
        assert!(hits
            .iter()
            .all(|h| h.metadata.as_ref().unwrap()["lang"] == "en"));
        assert_eq!(hits[0].id, "v0");
        // filter matching nothing → empty (best-effort)
        req.filter = Some(json!({"lang":"fr"}).as_object().unwrap().clone());
        assert!(idx.search(&req).unwrap().is_empty());
        // include_vector / include_metadata flags
        let mut req = SearchRequest::new(data[..d].to_vec(), 1);
        req.include_vector = true;
        req.include_metadata = false;
        let h = &idx.search(&req).unwrap()[0];
        assert_eq!(h.vector.as_deref(), Some(&data[..d]));
        assert!(h.metadata.is_none());
    }
}

#[test]
fn small_index_auto_build_at_threshold() {
    let d = 8usize;
    let dir = tempfile::tempdir().unwrap();
    let o = StoreOptions {
        small_index_threshold: 500,
        delta_threshold: 300,
        beam_width: 4,
        auto_build: true,
    };
    let cat = Catalog::open(dir.path(), o).unwrap();
    let idx = create(&cat, "s", d as u32, Metric::Euclidean);
    let data = gaussian_vectors(1000, d, 9);
    idx.upsert(records(&data[..499 * d], d, "v", 0)).unwrap();
    assert!(!idx.is_building());
    assert!(!dir.path().join("s").join("main.1").exists());
    idx.upsert(records(&data[499 * d..500 * d], d, "v", 499))
        .unwrap();
    assert!(idx.wait_for_build(Duration::from_secs(60)));
    assert!(
        dir.path().join("s").join("main.1").exists(),
        "first disk build at threshold"
    );
    assert_eq!(idx.info().delta_count, 0);
    // delta threshold triggers a rebuild
    idx.upsert(records(&data[500 * d..], d, "v", 500)).unwrap();
    assert!(idx.wait_for_build(Duration::from_secs(60)));
    assert!(dir.path().join("s").join("main.2").exists());
    assert!(!dir.path().join("s").join("main.1").exists());
    assert_eq!(idx.info().count, 1000);
    let hits = idx
        .search(&SearchRequest::new(data[700 * d..701 * d].to_vec(), 1))
        .unwrap();
    assert_eq!(hits[0].id, "v700");
}

#[test]
fn dot_vectors_above_max_norm_stay_exact_in_delta() {
    let (n, d) = (3000usize, 8usize);
    let dir = tempfile::tempdir().unwrap();
    let cat = Catalog::open(dir.path(), opts(100)).unwrap();
    let idx = create(&cat, "dot", d as u32, Metric::Dot);
    let data = varying_norm_vectors(n, d, 10);
    idx.upsert(records(&data, d, "v", 0)).unwrap();
    build_and_wait(&idx);
    // delta graph needs > 2048 slots to be used; add many normal + one huge vector
    let extra = varying_norm_vectors(2500, d, 11);
    idx.upsert(records(&extra, d, "e", 0)).unwrap();
    let mut huge = vec![0.0f32; d];
    huge[0] = 1000.0;
    idx.upsert(vec![VectorRecord {
        id: "huge".into(),
        vector: huge,
        metadata: None,
    }])
    .unwrap();
    let mut q = vec![0.0f32; d];
    q[0] = 1.0;
    let hits = idx.search(&SearchRequest::new(q, 3)).unwrap();
    assert_eq!(hits[0].id, "huge");
    assert!((hits[0].score - 1000.0).abs() < 1e-3);
}
