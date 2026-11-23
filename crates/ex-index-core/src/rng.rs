//! Small deterministic PRNG (SplitMix64) so builds are reproducible and the
//! core crate needs no RNG dependency.

#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[0, n)`; `n` must be > 0.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Standard normal via Box–Muller.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }

    /// `k` distinct indices from `0..n` (k ≤ n).
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let k = k.min(n);
        if k * 4 >= n {
            let mut all: Vec<usize> = (0..n).collect();
            self.shuffle(&mut all);
            all.truncate(k);
            return all;
        }
        let mut seen = std::collections::HashSet::with_capacity(k * 2);
        let mut out = Vec::with_capacity(k);
        while out.len() < k {
            let i = self.below(n);
            if seen.insert(i) {
                out.push(i);
            }
        }
        out
    }
}
