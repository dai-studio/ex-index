//! Product quantization: k-means (≤ 256 centroids) per sub-space trained on a
//! sample; codes are `m` bytes per vector and drive graph navigation through
//! asymmetric distance (ADC) lookup tables.
//!
//! The input dimension is split into `m` contiguous sub-vectors whose sizes
//! differ by at most one (`floor` or `ceil` of `dim / m`), so no sub-vector is
//! pure padding; shorter sub-vectors are zero-padded to `ceil(dim / m)`.

use crate::distance::l2sq;
use crate::error::{Error, Result};
use crate::rng::Rng;
use rayon::prelude::*;
use std::io::{Read, Write};

pub const KS: usize = 256;
const TRAIN_SAMPLE: usize = 50_000;
const KMEANS_ITERS: usize = 12;
const PQ_MAGIC: [u8; 8] = *b"EXIDXPQ2";

#[derive(Debug, Clone, PartialEq)]
pub struct PqCodebook {
    /// Input (build-space) dimension.
    pub dim: usize,
    /// Number of sub-vectors (= bytes per code).
    pub m: usize,
    /// Coordinates per sub-vector (`ceil(dim / m)`).
    pub sub_dim: usize,
    /// Centroids per sub-space (≤ 256; fewer when trained on < 256 vectors).
    pub ks: usize,
    /// `m × ks × sub_dim` centroid coordinates.
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Train on `data` (`n × dim`), sampling at most 50k vectors.
    pub fn train(data: &[f32], dim: usize, m: usize, seed: u64) -> PqCodebook {
        assert!(dim > 0 && m > 0 && m <= dim);
        let n = data.len() / dim;
        let sub_dim = dim.div_ceil(m);
        let mut rng = Rng::new(seed);
        let sample_idx = rng.sample_indices(n, TRAIN_SAMPLE.min(n));
        let ns = sample_idx.len();
        let ks = KS.min(ns.max(1));
        let mut centroids = vec![0f32; m * ks * sub_dim];
        if ns == 0 {
            return PqCodebook {
                dim,
                m,
                sub_dim,
                ks,
                centroids,
            };
        }
        let cb_tmp = PqCodebook {
            dim,
            m,
            sub_dim,
            ks,
            centroids: Vec::new(),
        };
        for j in 0..m {
            // gather sub-vectors of the sample
            let mut pts = vec![0f32; ns * sub_dim];
            for (r, &i) in sample_idx.iter().enumerate() {
                cb_tmp.sub_vector(
                    &data[i * dim..(i + 1) * dim],
                    j,
                    &mut pts[r * sub_dim..(r + 1) * sub_dim],
                );
            }
            let c = kmeans(&pts, sub_dim, ks, KMEANS_ITERS, &mut rng);
            centroids[j * ks * sub_dim..(j + 1) * ks * sub_dim].copy_from_slice(&c);
        }
        PqCodebook {
            dim,
            m,
            sub_dim,
            ks,
            centroids,
        }
    }

    /// `(start, len)` of sub-vector `j` in the input vector.
    #[inline]
    pub fn sub_range(&self, j: usize) -> (usize, usize) {
        let base = self.dim / self.m;
        let extra = self.dim % self.m;
        (j * base + j.min(extra), base + usize::from(j < extra))
    }

    /// Copy sub-vector `j` of `v` into `out` (zero-padded to `sub_dim`).
    #[inline]
    fn sub_vector(&self, v: &[f32], j: usize, out: &mut [f32]) {
        let (start, len) = self.sub_range(j);
        out[..len].copy_from_slice(&v[start..start + len]);
        out[len..].iter_mut().for_each(|o| *o = 0.0);
    }

    #[inline]
    fn centroid(&self, j: usize, c: usize) -> &[f32] {
        let off = (j * self.ks + c) * self.sub_dim;
        &self.centroids[off..off + self.sub_dim]
    }

    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        let mut buf = vec![0f32; self.sub_dim];
        for (j, o) in out.iter_mut().enumerate().take(self.m) {
            self.sub_vector(v, j, &mut buf);
            let mut best = (f32::INFINITY, 0usize);
            for c in 0..self.ks {
                let d = l2sq(&buf, self.centroid(j, c));
                if d < best.0 {
                    best = (d, c);
                }
            }
            *o = best.1 as u8;
        }
    }

    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.m];
        self.encode_into(v, &mut out);
        out
    }

    /// Encode all vectors (`n × dim`) in parallel → `n × m` bytes.
    pub fn encode_all(&self, data: &[f32]) -> Vec<u8> {
        let n = data.len() / self.dim;
        let mut codes = vec![0u8; n * self.m];
        codes
            .par_chunks_mut(self.m.max(1))
            .zip(data.par_chunks(self.dim))
            .for_each(|(c, v)| self.encode_into(v, c));
        codes
    }

    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        for (j, &c) in code.iter().enumerate().take(self.m) {
            let (_, len) = self.sub_range(j);
            out.extend_from_slice(&self.centroid(j, c as usize)[..len]);
        }
        out
    }

    /// ADC table: `m × ks` squared distances from each query sub-vector to
    /// every centroid.
    pub fn adc_table(&self, q: &[f32]) -> Vec<f32> {
        let mut table = vec![0f32; self.m * self.ks];
        let mut buf = vec![0f32; self.sub_dim];
        for j in 0..self.m {
            self.sub_vector(q, j, &mut buf);
            for c in 0..self.ks {
                table[j * self.ks + c] = l2sq(&buf, self.centroid(j, c));
            }
        }
        table
    }

    /// Inner-product ADC table: `m × ks` values `−q_j · c_j`, so that the
    /// summed distance ranks codes by descending approximate `q · x` (used for
    /// MIPS, where it avoids the `‖c(x)‖²` quantization noise of L2 ADC).
    pub fn adc_table_ip(&self, q: &[f32]) -> Vec<f32> {
        let mut table = vec![0f32; self.m * self.ks];
        let mut buf = vec![0f32; self.sub_dim];
        for j in 0..self.m {
            self.sub_vector(q, j, &mut buf);
            for c in 0..self.ks {
                table[j * self.ks + c] = -crate::distance::dot(&buf, self.centroid(j, c));
            }
        }
        table
    }

    /// Sum of table entries for a code (approximate squared L2 for
    /// [`Self::adc_table`], approximate `−q·x` for [`Self::adc_table_ip`]).
    #[inline]
    pub fn adc_distance(&self, table: &[f32], code: &[u8]) -> f32 {
        let ks = self.ks;
        code.iter()
            .enumerate()
            .map(|(j, &c)| table[j * ks + c as usize])
            .sum()
    }

    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        w.write_all(&PQ_MAGIC)?;
        for v in [self.dim, self.m, self.sub_dim, self.ks] {
            w.write_all(&(v as u32).to_le_bytes())?;
        }
        for c in &self.centroids {
            w.write_all(&c.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> Result<PqCodebook> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if magic != PQ_MAGIC {
            return Err(Error::Corrupt("bad pq magic".into()));
        }
        let mut u = [0u8; 4];
        let mut hdr = [0usize; 4];
        for h in hdr.iter_mut() {
            r.read_exact(&mut u)?;
            *h = u32::from_le_bytes(u) as usize;
        }
        let [dim, m, sub_dim, ks] = hdr;
        if dim == 0 || m == 0 || ks == 0 || ks > KS || sub_dim != dim.div_ceil(m) {
            return Err(Error::Corrupt("bad pq header".into()));
        }
        let mut bytes = vec![0u8; m * ks * sub_dim * 4];
        r.read_exact(&mut bytes)?;
        let centroids = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(PqCodebook {
            dim,
            m,
            sub_dim,
            ks,
            centroids,
        })
    }
}

/// Lloyd's k-means on `pts` (`n × d`). Returns `k × d` centroids.
fn kmeans(pts: &[f32], d: usize, k: usize, iters: usize, rng: &mut Rng) -> Vec<f32> {
    let n = pts.len() / d;
    let mut cent = vec![0f32; k * d];
    for (c, i) in rng.sample_indices(n, k).into_iter().enumerate() {
        cent[c * d..(c + 1) * d].copy_from_slice(&pts[i * d..(i + 1) * d]);
    }
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        assign
            .par_iter_mut()
            .zip(pts.par_chunks_exact(d))
            .for_each(|(a, p)| {
                let mut best = (f32::INFINITY, 0u32);
                for (c, cv) in cent.chunks_exact(d).enumerate() {
                    let dd = l2sq(p, cv);
                    if dd < best.0 {
                        best = (dd, c as u32);
                    }
                }
                *a = best.1;
            });
        let mut sums = vec![0f64; k * d];
        let mut counts = vec![0usize; k];
        for (i, &a) in assign.iter().enumerate() {
            let a = a as usize;
            counts[a] += 1;
            for t in 0..d {
                sums[a * d + t] += pts[i * d + t] as f64;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // re-seed an empty cluster with a random point
                let i = rng.below(n);
                cent[c * d..(c + 1) * d].copy_from_slice(&pts[i * d..(i + 1) * d]);
            } else {
                for t in 0..d {
                    cent[c * d + t] = (sums[c * d + t] / counts[c] as f64) as f32;
                }
            }
        }
    }
    cent
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::clustered_vectors;

    fn spearman(a: &[f32], b: &[f32]) -> f64 {
        fn ranks(x: &[f32]) -> Vec<f64> {
            let mut idx: Vec<usize> = (0..x.len()).collect();
            idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
            let mut r = vec![0f64; x.len()];
            for (rank, &i) in idx.iter().enumerate() {
                r[i] = rank as f64;
            }
            r
        }
        let (ra, rb) = (ranks(a), ranks(b));
        let n = a.len() as f64;
        let d2: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - y).powi(2)).sum();
        1.0 - 6.0 * d2 / (n * (n * n - 1.0))
    }

    #[test]
    fn pq_round_trip_and_adc_ranking() {
        let (n, d) = (5000, 32);
        let data = clustered_vectors(n, d, 50, 21);
        let cb = PqCodebook::train(&data, d, d / 4, 5);
        assert_eq!(cb.ks, 256);
        let codes = cb.encode_all(&data);
        assert_eq!(codes.len(), n * cb.m);
        // reconstruction MSE well below the data variance
        let var: f64 = {
            let mean: Vec<f64> = (0..d)
                .map(|j| data.chunks_exact(d).map(|r| r[j] as f64).sum::<f64>() / n as f64)
                .collect();
            data.chunks_exact(d)
                .map(|r| {
                    r.iter()
                        .zip(&mean)
                        .map(|(x, m)| (*x as f64 - m).powi(2))
                        .sum::<f64>()
                })
                .sum::<f64>()
                / n as f64
        };
        let mse: f64 = data
            .chunks_exact(d)
            .zip(codes.chunks_exact(cb.m))
            .map(|(v, c)| l2sq(v, &cb.decode(c)) as f64)
            .sum::<f64>()
            / n as f64;
        assert!(mse < 0.15 * var, "mse {mse} vs variance {var}");
        // ADC distance ranking correlates with the exact ranking
        let q = &data[123 * d..124 * d];
        let table = cb.adc_table(q);
        let exact: Vec<f32> = data.chunks_exact(d).map(|v| l2sq(q, v)).collect();
        let approx: Vec<f32> = codes
            .chunks_exact(cb.m)
            .map(|c| cb.adc_distance(&table, c))
            .collect();
        let rho = spearman(&exact, &approx);
        assert!(rho > 0.9, "spearman {rho}");
    }

    #[test]
    fn padding_and_small_training_sets() {
        // dim not divisible by m → last sub-vector zero-padded
        let d = 10;
        let data = clustered_vectors(100, d, 5, 1);
        let cb = PqCodebook::train(&data, d, 3, 1);
        assert_eq!(cb.sub_dim, 4);
        assert_eq!(
            (cb.sub_range(0), cb.sub_range(1), cb.sub_range(2)),
            ((0, 4), (4, 3), (7, 3))
        );
        // 129 dims into 32 sub-vectors: no sub-vector is pure padding
        let cb2 = PqCodebook {
            dim: 129,
            m: 32,
            sub_dim: 5,
            ks: 1,
            centroids: vec![0.0; 32 * 5],
        };
        assert_eq!(cb2.sub_range(0), (0, 5));
        assert_eq!(cb2.sub_range(31), (125, 4));
        assert_eq!(cb.ks, 100);
        let code = cb.encode(&data[..d]);
        assert_eq!(code.len(), 3);
        assert_eq!(cb.decode(&code).len(), d);
        let mut buf = Vec::new();
        cb.write_to(&mut buf).unwrap();
        let back = PqCodebook::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(back, cb);
    }
}
