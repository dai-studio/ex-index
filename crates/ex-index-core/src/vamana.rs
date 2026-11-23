//! Vamana graph (DiskANN, Subramanya et al. 2019): medoid entry point,
//! random R-regular initialization, two passes (α = 1, then α) of
//! GreedySearch + RobustPrune, built in parallel with rayon and per-node
//! locks. The same structure (via [`Graph::insert_batch`]) backs the
//! in-memory delta index.
//!
//! All distances are squared L2 in build space.

use crate::distance::l2sq;
use crate::rng::Rng;
use crate::util::FxHashSet;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Cap on the RobustPrune candidate pool (DiskANN's `C`).
const MAX_POOL: usize = 750;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VamanaParams {
    pub r: u32,
    pub l_build: u32,
    pub alpha: f32,
}

impl Default for VamanaParams {
    fn default() -> Self {
        VamanaParams {
            r: 64,
            l_build: 100,
            alpha: 1.2,
        }
    }
}

/// In-memory Vamana graph over build-space vectors.
pub struct Graph {
    dim: usize,
    data: Vec<f32>,
    adj: Vec<RwLock<Vec<u32>>>,
    medoid: u32,
    params: VamanaParams,
}

#[derive(Clone, Copy)]
struct Cand {
    dist: f32,
    id: u32,
    expanded: bool,
}

/// Sorted, bounded candidate list used by greedy / beam search.
pub(crate) struct CandList {
    cap: usize,
    items: Vec<Cand>,
    cursor: usize,
}

impl CandList {
    pub(crate) fn new(cap: usize) -> Self {
        CandList {
            cap: cap.max(1),
            items: Vec::with_capacity(cap + 1),
            cursor: 0,
        }
    }

    /// Insert keeping ascending order; returns false if rejected.
    #[inline]
    pub(crate) fn insert(&mut self, id: u32, dist: f32) -> bool {
        if self.items.len() >= self.cap && dist >= self.items[self.items.len() - 1].dist {
            return false;
        }
        let pos = self.items.partition_point(|c| c.dist <= dist);
        self.items.insert(
            pos,
            Cand {
                dist,
                id,
                expanded: false,
            },
        );
        if self.items.len() > self.cap {
            self.items.pop();
        }
        if pos < self.cursor {
            self.cursor = pos;
        }
        true
    }

    /// Next closest unexpanded candidate (marks it expanded).
    #[inline]
    pub(crate) fn pop_unexpanded(&mut self) -> Option<(u32, f32)> {
        while self.cursor < self.items.len() {
            let c = &mut self.items[self.cursor];
            self.cursor += 1;
            if !c.expanded {
                c.expanded = true;
                return Some((c.id, c.dist));
            }
        }
        None
    }

    pub(crate) fn into_vec(self) -> Vec<(u32, f32)> {
        self.items.into_iter().map(|c| (c.id, c.dist)).collect()
    }
}

impl Graph {
    /// Empty graph (used for the delta index; grows via [`Graph::insert_batch`]).
    pub fn new_empty(dim: usize, params: VamanaParams) -> Self {
        Graph {
            dim,
            data: Vec::new(),
            adj: Vec::new(),
            medoid: 0,
            params,
        }
    }

    pub fn len(&self) -> usize {
        self.adj.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adj.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn medoid(&self) -> u32 {
        self.medoid
    }

    pub fn params(&self) -> VamanaParams {
        self.params
    }

    #[inline]
    pub fn vector(&self, i: u32) -> &[f32] {
        let i = i as usize;
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn neighbors(&self, i: u32) -> Vec<u32> {
        self.adj[i as usize].read().clone()
    }

    /// Consume into `(vectors, adjacency lists, medoid)` for writing to disk.
    pub fn into_parts(self) -> (Vec<f32>, Vec<Vec<u32>>, u32) {
        let adj = self.adj.into_iter().map(|l| l.into_inner()).collect();
        (self.data, adj, self.medoid)
    }

    /// Build a graph over `data` (`n × dim`, build space).
    pub fn build(data: Vec<f32>, dim: usize, params: VamanaParams, seed: u64) -> Self {
        assert!(dim > 0 && data.len() % dim == 0);
        let n = data.len() / dim;
        let r = params.r as usize;
        let mut g = Graph {
            dim,
            data,
            adj: Vec::with_capacity(n),
            medoid: 0,
            params,
        };
        if n == 0 {
            return g;
        }
        g.medoid = g.compute_medoid();

        // Random R-regular initialization.
        let deg = r.min(n - 1);
        let mut rng = Rng::new(seed);
        for i in 0..n {
            let mut nb = Vec::with_capacity(r);
            if deg > 0 {
                let mut seen = FxHashSet::default();
                while nb.len() < deg {
                    let j = rng.below(n);
                    if j != i && seen.insert(j) {
                        nb.push(j as u32);
                    }
                }
            }
            g.adj.push(RwLock::new(nb));
        }

        // Two passes: α = 1 then α.
        let mut order: Vec<u32> = (0..n as u32).collect();
        for (pass, alpha) in [1.0f32, params.alpha.max(1.0)].into_iter().enumerate() {
            let mut rng = Rng::new(seed.wrapping_add(pass as u64 + 1));
            rng.shuffle(&mut order);
            order.par_iter().for_each(|&p| g.link_node(p, alpha));
        }
        g
    }

    /// Approximate medoid: the point closest to the centroid.
    fn compute_medoid(&self) -> u32 {
        let n = self.len_by_data();
        let mut centroid = vec![0f64; self.dim];
        for row in self.data.chunks_exact(self.dim) {
            for (c, x) in centroid.iter_mut().zip(row) {
                *c += *x as f64;
            }
        }
        let centroid: Vec<f32> = centroid.iter().map(|c| (*c / n as f64) as f32).collect();
        self.data
            .par_chunks_exact(self.dim)
            .enumerate()
            .map(|(i, row)| (l2sq(row, &centroid), i as u32))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|x| x.1)
            .unwrap_or(0)
    }

    fn len_by_data(&self) -> usize {
        self.data.len() / self.dim
    }

    /// GreedySearch from the medoid: returns the `l` closest candidates
    /// (best-first) as `(node, squared L2)`.
    pub fn greedy_search(&self, q: &[f32], l: usize) -> Vec<(u32, f32)> {
        self.search_inner(q, l, None, &[])
    }

    /// GreedySearch seeded from the medoid plus `extra_starts`.
    pub fn greedy_search_from(&self, q: &[f32], l: usize, extra_starts: &[u32]) -> Vec<(u32, f32)> {
        self.search_inner(q, l, None, extra_starts)
    }

    fn search_inner(
        &self,
        q: &[f32],
        l: usize,
        mut expanded: Option<&mut Vec<(u32, f32)>>,
        extra_starts: &[u32],
    ) -> Vec<(u32, f32)> {
        if self.adj.is_empty() {
            return Vec::new();
        }
        let mut cands = CandList::new(l);
        let mut visited = FxHashSet::default();
        visited.reserve(l * 4);
        for &start in std::iter::once(&self.medoid).chain(extra_starts) {
            if (start as usize) < self.adj.len() && visited.insert(start) {
                cands.insert(start, l2sq(q, self.vector(start)));
            }
        }
        let mut nbuf: Vec<u32> = Vec::with_capacity(self.params.r as usize * 2);
        while let Some((id, dist)) = cands.pop_unexpanded() {
            if let Some(e) = expanded.as_deref_mut() {
                e.push((id, dist));
            }
            nbuf.clear();
            nbuf.extend_from_slice(&self.adj[id as usize].read());
            for &nb in &nbuf {
                if visited.insert(nb) {
                    cands.insert(nb, l2sq(q, self.vector(nb)));
                }
            }
        }
        cands.into_vec()
    }

    /// RobustPrune(p, pool, α, R) → new out-neighbors of p.
    fn robust_prune(&self, p: u32, pool: &mut Vec<(u32, f32)>, alpha: f32) -> Vec<u32> {
        let r = self.params.r as usize;
        pool.retain(|&(id, _)| id != p);
        pool.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        pool.dedup_by_key(|c| c.0);
        pool.truncate(MAX_POOL);
        let mut alive = vec![true; pool.len()];
        let mut out = Vec::with_capacity(r);
        for i in 0..pool.len() {
            if !alive[i] {
                continue;
            }
            let (star, _) = pool[i];
            out.push(star);
            if out.len() >= r {
                break;
            }
            let vs = self.vector(star);
            for j in (i + 1)..pool.len() {
                if alive[j] {
                    let (cand, d_pc) = pool[j];
                    if alpha * l2sq(vs, self.vector(cand)) <= d_pc {
                        alive[j] = false;
                    }
                }
            }
        }
        out
    }

    /// One Vamana insertion step for node `p` (search, prune, back-links).
    fn link_node(&self, p: u32, alpha: f32) {
        let r = self.params.r as usize;
        let l = (self.params.l_build as usize).max(r);
        let xp = self.vector(p);
        let mut pool = Vec::with_capacity(l * 2);
        self.search_inner(xp, l, Some(&mut pool), &[]);
        for nb in self.adj[p as usize].read().iter() {
            pool.push((*nb, l2sq(xp, self.vector(*nb))));
        }
        let new_nbrs = self.robust_prune(p, &mut pool, alpha);
        *self.adj[p as usize].write() = new_nbrs.clone();
        for j in new_nbrs {
            let mut nj = self.adj[j as usize].write();
            if nj.contains(&p) {
                continue;
            }
            if nj.len() < r {
                nj.push(p);
            } else {
                let xj = self.vector(j);
                let mut pool: Vec<(u32, f32)> =
                    nj.iter().map(|&k| (k, l2sq(xj, self.vector(k)))).collect();
                pool.push((p, l2sq(xj, xp)));
                *nj = self.robust_prune(j, &mut pool, alpha);
            }
        }
    }

    /// Append `vectors` (`m × dim`) and link them into the graph in parallel
    /// (single α pass, FreshDiskANN-style). Returns the new node ids.
    pub fn insert_batch(&mut self, vectors: &[f32]) -> std::ops::Range<u32> {
        assert_eq!(vectors.len() % self.dim, 0);
        let start = self.adj.len() as u32;
        let m = vectors.len() / self.dim;
        self.data.extend_from_slice(vectors);
        for _ in 0..m {
            self.adj.push(RwLock::new(Vec::new()));
        }
        let end = start + m as u32;
        if start == 0 && m > 0 {
            self.medoid = 0;
        }
        let alpha = self.params.alpha.max(1.0);
        let first = if start == 0 { 1 } else { start };
        let this = &*self;
        (first..end)
            .into_par_iter()
            .for_each(|p| this.link_node(p, alpha));
        start..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::harness::{gaussian_vectors, ground_truth, recall_at_k};

    fn check_invariants(g: &Graph) {
        let n = g.len();
        let r = g.params().r as usize;
        for i in 0..n as u32 {
            let nb = g.neighbors(i);
            assert!(nb.len() <= r, "out-degree {} > R", nb.len());
            assert!(!nb.contains(&i), "self loop at {i}");
            let mut s = nb.clone();
            s.sort();
            s.dedup();
            assert_eq!(s.len(), nb.len(), "duplicate edges at {i}");
        }
        // reachability from medoid
        let mut seen = vec![false; n];
        let mut stack = vec![g.medoid()];
        seen[g.medoid() as usize] = true;
        while let Some(u) = stack.pop() {
            for v in g.neighbors(u) {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    stack.push(v);
                }
            }
        }
        let reached = seen.iter().filter(|x| **x).count();
        assert_eq!(reached, n, "only {reached}/{n} nodes reachable from medoid");
    }

    #[test]
    fn vamana_invariants_and_recall_5k() {
        let (n, d) = (5000, 32);
        let data = gaussian_vectors(n, d, 42);
        let g = Graph::build(data.clone(), d, VamanaParams::default(), 1);
        check_invariants(&g);
        let queries = gaussian_vectors(200, d, 4242);
        let truth = ground_truth(Metric::Euclidean, d, &data, &queries, 10);
        let got: Vec<Vec<u32>> = queries
            .chunks_exact(d)
            .map(|q| g.greedy_search(q, 100).iter().map(|x| x.0).collect())
            .collect();
        let recall = recall_at_k(&truth, &got, 10);
        assert!(recall >= 0.98, "recall@10 = {recall}");
    }

    #[test]
    fn tiny_graphs() {
        let g = Graph::build(vec![1.0, 2.0], 2, VamanaParams::default(), 1);
        assert_eq!(g.len(), 1);
        assert_eq!(g.greedy_search(&[0.0, 0.0], 5), vec![(0, 5.0)]);
        let g = Graph::build(gaussian_vectors(3, 4, 1), 4, VamanaParams::default(), 1);
        check_invariants(&g);
        assert_eq!(g.greedy_search(&[0.0; 4], 10).len(), 3);
    }

    #[test]
    fn incremental_insert_matches_quality() {
        let (n, d) = (3000, 16);
        let data = gaussian_vectors(n, d, 9);
        let mut g = Graph::new_empty(d, VamanaParams::default());
        for chunk in data.chunks(d * 500) {
            g.insert_batch(chunk);
        }
        assert_eq!(g.len(), n);
        check_invariants(&g);
        let queries = gaussian_vectors(100, d, 99);
        let truth = ground_truth(Metric::Euclidean, d, &data, &queries, 10);
        let got: Vec<Vec<u32>> = queries
            .chunks_exact(d)
            .map(|q| g.greedy_search(q, 100).iter().map(|x| x.0).collect())
            .collect();
        let recall = recall_at_k(&truth, &got, 10);
        assert!(recall >= 0.95, "incremental recall@10 = {recall}");
    }

    #[test]
    fn dot_via_mips_recall_on_unnormalized_data() {
        use crate::distance::{mips_augment_into, norm_sq, query_to_build_space};
        use crate::harness::varying_norm_vectors;
        let (n, d) = (4000, 24);
        let data = varying_norm_vectors(n, d, 17);
        let m = data
            .chunks_exact(d)
            .map(|x| norm_sq(x).sqrt())
            .fold(0.0f32, f32::max);
        let mut aug = Vec::with_capacity(n * (d + 1));
        for x in data.chunks_exact(d) {
            mips_augment_into(x, m, &mut aug);
        }
        let g = Graph::build(aug, d + 1, VamanaParams::default(), 3);
        check_invariants(&g);
        let queries = gaussian_vectors(200, d, 777);
        let truth = ground_truth(Metric::Dot, d, &data, &queries, 10);
        let got: Vec<Vec<u32>> = queries
            .chunks_exact(d)
            .map(|q| {
                let qb = query_to_build_space(Metric::Dot, q);
                // re-rank the L candidates by the exact raw dot of the originals
                let mut c: Vec<(u32, f32)> = g
                    .greedy_search(&qb, 100)
                    .into_iter()
                    .map(|(i, _)| {
                        (
                            i,
                            Metric::Dot.distance(q, &data[i as usize * d..(i as usize + 1) * d]),
                        )
                    })
                    .collect();
                c.sort_by(|a, b| a.1.total_cmp(&b.1));
                c.into_iter().map(|x| x.0).collect()
            })
            .collect();
        let recall = recall_at_k(&truth, &got, 10);
        assert!(recall >= 0.95, "dot recall@10 = {recall}");
    }
}
