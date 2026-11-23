//! Distance kernels and metric semantics.
//!
//! Kernels are written as 8-lane chunked loops so LLVM auto-vectorizes them on
//! every target; the optional `simd` feature (nightly) uses `std::simd`.

use serde::{Deserialize, Serialize};

/// Similarity metric chosen at index creation (immutable afterwards).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    Cosine,
    Euclidean,
    Dot,
}

impl Metric {
    pub fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::Dot => "dot",
        }
    }

    pub fn parse(s: &str) -> Option<Metric> {
        match s {
            "cosine" => Some(Metric::Cosine),
            "euclidean" => Some(Metric::Euclidean),
            "dot" => Some(Metric::Dot),
            _ => None,
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Metric::Cosine => 0,
            Metric::Euclidean => 1,
            Metric::Dot => 2,
        }
    }

    pub fn from_u32(v: u32) -> Option<Metric> {
        match v {
            0 => Some(Metric::Cosine),
            1 => Some(Metric::Euclidean),
            2 => Some(Metric::Dot),
            _ => None,
        }
    }

    /// Dimension of the build space (`d + 1` for dot because of MIPS→L2).
    pub fn build_dim(self, dim: usize) -> usize {
        match self {
            Metric::Dot => dim + 1,
            _ => dim,
        }
    }

    /// Exact internal distance (lower = closer) between a prepared query and a
    /// stored vector (normalized for cosine, original for euclidean / dot).
    #[inline]
    pub fn distance(self, q: &[f32], x: &[f32]) -> f32 {
        match self {
            Metric::Cosine => 1.0 - dot(q, x),
            Metric::Euclidean => l2sq(q, x),
            Metric::Dot => -dot(q, x),
        }
    }

    /// Convert an internal distance into the public score
    /// (cosine similarity / true L2 distance / raw inner product).
    #[inline]
    pub fn to_score(self, internal: f32) -> f32 {
        match self {
            Metric::Cosine => 1.0 - internal,
            Metric::Euclidean => internal.max(0.0).sqrt(),
            Metric::Dot => -internal,
        }
    }

    /// Whether a larger public score is better (results are always best-first).
    pub fn higher_is_better(self) -> bool {
        !matches!(self, Metric::Euclidean)
    }
}

/// The `Distance` trait from the design: a stateless distance function.
pub trait Distance {
    fn dist(a: &[f32], b: &[f32]) -> f32;
}

/// Squared euclidean distance.
pub struct L2Sq;
impl Distance for L2Sq {
    #[inline]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        l2sq(a, b)
    }
}

/// Negative inner product (lower = more similar).
pub struct NegDot;
impl Distance for NegDot {
    #[inline]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        -dot(a, b)
    }
}

/// Squared L2 distance.
#[cfg(not(feature = "simd"))]
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// Inner product.
#[cfg(not(feature = "simd"))]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[cfg(feature = "simd")]
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    use std::simd::{f32x8, num::SimdFloat};
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = f32x8::splat(0.0);
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        let d = f32x8::from_slice(x) - f32x8::from_slice(y);
        acc += d * d;
    }
    let mut s = acc.reduce_sum();
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        s += d * d;
    }
    s
}

#[cfg(feature = "simd")]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    use std::simd::{f32x8, num::SimdFloat};
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = f32x8::splat(0.0);
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        acc += f32x8::from_slice(x) * f32x8::from_slice(y);
    }
    let mut s = acc.reduce_sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[inline]
pub fn norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

/// L2-normalize in place. Returns `false` (and leaves `v` untouched) for a
/// zero-norm or non-finite-norm vector.
pub fn normalize(v: &mut [f32]) -> bool {
    let n = (norm_sq(v) as f64).sqrt();
    if n == 0.0 || !n.is_finite() {
        return false;
    }
    let inv = (1.0 / n) as f32;
    for x in v.iter_mut() {
        *x *= inv;
    }
    true
}

/// Append the MIPS→L2 lifted copy of `x` to `out`: `[x, sqrt(max(M² − ‖x‖², 0))]`.
pub fn mips_augment_into(x: &[f32], max_norm: f32, out: &mut Vec<f32>) {
    out.extend_from_slice(x);
    let extra = (max_norm * max_norm - norm_sq(x)).max(0.0).sqrt();
    out.push(extra);
}

/// Indices of the (up to) `k` largest-norm rows of `data` (`n × dim`),
/// used as extra search entry points for dot-product (MIPS) graphs.
pub fn top_norm_indices(data: &[f32], dim: usize, k: usize) -> Vec<u32> {
    let mut norms: Vec<(f32, u32)> = data
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, x)| (norm_sq(x), i as u32))
        .collect();
    let k = k.min(norms.len());
    if k == 0 {
        return Vec::new();
    }
    norms.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
    norms.truncate(k);
    norms.sort_by(|a, b| b.0.total_cmp(&a.0));
    norms.into_iter().map(|x| x.1).collect()
}

/// Number of max-norm entry points used for dot-product indexes.
pub const MIPS_ENTRY_POINTS: usize = 64;

/// Map a prepared query into build space (`[q, 0]` for dot).
pub fn query_to_build_space(metric: Metric, q: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(q.len() + 1);
    v.extend_from_slice(q);
    if metric == Metric::Dot {
        v.push(0.0);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn naive_l2(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2))
            .sum()
    }
    fn naive_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum()
    }

    fn rel_close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + b.abs())
    }

    #[test]
    fn kernels_match_naive_reference() {
        let mut rng = Rng::new(7);
        for &d in &[1usize, 3, 7, 8, 9, 31, 128, 768, 4096] {
            for _ in 0..20 {
                let a: Vec<f32> = (0..d).map(|_| rng.gaussian()).collect();
                let b: Vec<f32> = (0..d).map(|_| rng.gaussian()).collect();
                assert!(rel_close(l2sq(&a, &b) as f64, naive_l2(&a, &b)), "l2 d={d}");
                assert!(
                    rel_close(dot(&a, &b) as f64, naive_dot(&a, &b)),
                    "dot d={d}"
                );
            }
        }
    }

    #[test]
    fn edge_cases_zero_and_large_norm() {
        let z = vec![0.0f32; 16];
        let big: Vec<f32> = (0..16).map(|i| 1e4 * (i as f32 + 1.0)).collect();
        assert_eq!(l2sq(&z, &z), 0.0);
        assert_eq!(dot(&z, &big), 0.0);
        assert!(rel_close(l2sq(&z, &big) as f64, naive_l2(&z, &big)));
        assert!(rel_close(dot(&big, &big) as f64, naive_dot(&big, &big)));
        let mut zz = z.clone();
        assert!(!normalize(&mut zz), "zero vector must be rejected");
    }

    #[test]
    fn cosine_normalization_is_idempotent() {
        let mut rng = Rng::new(3);
        for _ in 0..50 {
            let mut v: Vec<f32> = (0..64).map(|_| rng.gaussian() * 100.0).collect();
            assert!(normalize(&mut v));
            let once = v.clone();
            assert!(normalize(&mut v));
            for (a, b) in once.iter().zip(&v) {
                assert!((a - b).abs() < 1e-6);
            }
            assert!((norm_sq(&v) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn metric_scores() {
        let mut rng = Rng::new(11);
        let mut a: Vec<f32> = (0..32).map(|_| rng.gaussian()).collect();
        let mut b: Vec<f32> = (0..32).map(|_| rng.gaussian()).collect();
        // euclidean: score is the true (sqrt) distance
        let e = Metric::Euclidean.to_score(Metric::Euclidean.distance(&a, &b));
        assert!(rel_close(e as f64, naive_l2(&a, &b).sqrt()));
        // dot: raw inner product
        let d = Metric::Dot.to_score(Metric::Dot.distance(&a, &b));
        assert!(rel_close(d as f64, naive_dot(&a, &b)));
        // cosine: cosine similarity of normalized vectors
        let cos_ref = naive_dot(&a, &b) / (naive_dot(&a, &a).sqrt() * naive_dot(&b, &b).sqrt());
        normalize(&mut a);
        normalize(&mut b);
        let c = Metric::Cosine.to_score(Metric::Cosine.distance(&a, &b));
        assert!((c as f64 - cos_ref).abs() < 1e-5);
        // cosine in build space: ‖a−b‖² = 2 − 2cos
        assert!(((l2sq(&a, &b) as f64) - (2.0 - 2.0 * cos_ref)).abs() < 1e-5);
    }

    #[test]
    fn mips_transform_preserves_dot_ranking() {
        let mut rng = Rng::new(5);
        let xs: Vec<Vec<f32>> = (0..200)
            .map(|_| {
                (0..16)
                    .map(|_| rng.gaussian() * rng.next_f32() * 3.0)
                    .collect()
            })
            .collect();
        let m = xs.iter().map(|x| norm_sq(x).sqrt()).fold(0.0f32, f32::max);
        let q: Vec<f32> = (0..16).map(|_| rng.gaussian()).collect();
        let qb = query_to_build_space(Metric::Dot, &q);
        let mut by_dot: Vec<usize> = (0..xs.len()).collect();
        by_dot.sort_by(|&i, &j| dot(&q, &xs[j]).total_cmp(&dot(&q, &xs[i])));
        let mut by_l2: Vec<usize> = (0..xs.len()).collect();
        let aug: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| {
                let mut v = Vec::new();
                mips_augment_into(x, m, &mut v);
                v
            })
            .collect();
        by_l2.sort_by(|&i, &j| l2sq(&qb, &aug[i]).total_cmp(&l2sq(&qb, &aug[j])));
        assert_eq!(&by_dot[..10], &by_l2[..10]);
    }
}
