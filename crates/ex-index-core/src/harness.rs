//! Recall harness shared by tests and benches: synthetic data generators,
//! brute-force ground truth and recall@k.

use crate::brute::BruteForce;
use crate::distance::{normalize, Metric};
use crate::rng::Rng;
use rayon::prelude::*;

/// `n × dim` i.i.d. standard normal vectors (flat, row-major).
pub fn gaussian_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n * dim).map(|_| rng.gaussian()).collect()
}

/// `n × dim` uniform `[0,1)` vectors.
pub fn uniform_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n * dim).map(|_| rng.next_f32()).collect()
}

/// `n × dim` unit-norm vectors.
pub fn normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = gaussian_vectors(n, dim, seed);
    for row in v.chunks_exact_mut(dim) {
        normalize(row);
    }
    v
}

/// Gaussian vectors with per-row random norms in `[0.1, 3.1)` — exercises the
/// MIPS transform for dot-product indexes.
pub fn varying_norm_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed ^ 0xABCD);
    let mut v = normalized_vectors(n, dim, seed);
    for row in v.chunks_exact_mut(dim) {
        let s = 0.1 + 3.0 * rng.next_f32();
        row.iter_mut().for_each(|x| *x *= s);
    }
    v
}

/// Clustered data (mixture of gaussians), closer to real embeddings.
pub fn clustered_vectors(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let centers: Vec<f32> = (0..clusters * dim).map(|_| rng.gaussian() * 3.0).collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = rng.below(clusters);
        for j in 0..dim {
            out.push(centers[c * dim + j] + rng.gaussian());
        }
    }
    out
}

/// Low intrinsic-dimension data, closer to real embeddings: `x = A·z + ε`
/// with `z ~ N(0, I_intrinsic)`, a fixed random `dim × intrinsic` matrix `A`
/// and small isotropic noise.
pub fn low_rank_vectors(n: usize, dim: usize, intrinsic: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed ^ 0x10F7);
    let scale = 1.0 / (intrinsic as f32).sqrt();
    let a: Vec<f32> = (0..dim * intrinsic)
        .map(|_| rng.gaussian() * scale)
        .collect();
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n * dim);
    let mut z = vec![0f32; intrinsic];
    for _ in 0..n {
        z.iter_mut().for_each(|v| *v = rng.gaussian());
        for row in a.chunks_exact(intrinsic) {
            out.push(crate::distance::dot(row, &z) + 0.05 * rng.gaussian());
        }
    }
    out
}

/// Exact top-k ids for each query (queries/data already prepared for `metric`).
pub fn ground_truth(
    metric: Metric,
    dim: usize,
    data: &[f32],
    queries: &[f32],
    k: usize,
) -> Vec<Vec<u32>> {
    let bf = BruteForce::new(metric, dim, data.to_vec());
    queries
        .par_chunks_exact(dim)
        .map(|q| bf.search(q, k).into_iter().map(|(i, _)| i).collect())
        .collect()
}

/// Mean recall@k of `got` against `truth` (both best-first id lists).
pub fn recall_at_k(truth: &[Vec<u32>], got: &[Vec<u32>], k: usize) -> f64 {
    assert_eq!(truth.len(), got.len());
    if truth.is_empty() {
        return 1.0;
    }
    let mut hits = 0usize;
    let mut total = 0usize;
    for (t, g) in truth.iter().zip(got) {
        let t: std::collections::HashSet<u32> = t.iter().take(k).copied().collect();
        hits += g.iter().take(k).filter(|i| t.contains(i)).count();
        total += t.len();
    }
    hits as f64 / total.max(1) as f64
}

/// Latency percentile (`p` in 0..=100) of a list of durations in milliseconds.
pub fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let idx = ((p / 100.0) * (samples.len() - 1) as f64).round() as usize;
    samples[idx.min(samples.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recall_math() {
        let t = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let g = vec![vec![1, 2, 9], vec![6, 5, 4]];
        assert!((recall_at_k(&t, &g, 3) - 5.0 / 6.0).abs() < 1e-12);
        let mut s = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&mut s, 50.0), 3.0);
        assert_eq!(percentile(&mut s, 100.0), 5.0);
    }
    #[test]
    fn ground_truth_is_self_for_euclidean() {
        let d = 8;
        let data = gaussian_vectors(100, d, 1);
        let gt = ground_truth(Metric::Euclidean, d, &data, &data[..10 * d], 1);
        for (i, g) in gt.iter().enumerate() {
            assert_eq!(g[0], i as u32);
        }
    }
}
