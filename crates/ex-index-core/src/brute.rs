//! Flat exact top-k search: ground truth for every recall check and the
//! serving path for small indexes.

use crate::distance::Metric;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A candidate ordered by distance (max-heap on distance for top-k pruning).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored<T> {
    pub dist: f32,
    pub item: T,
}

impl<T: PartialEq> Eq for Scored<T> {}
impl<T: PartialEq> PartialOrd for Scored<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: PartialEq> Ord for Scored<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

/// Bounded collector keeping the `k` smallest distances.
pub struct TopK<T> {
    k: usize,
    heap: BinaryHeap<Scored<T>>,
}

impl<T: PartialEq> TopK<T> {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// Current admission threshold (distance of the worst kept item).
    #[inline]
    pub fn worst(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().map(|s| s.dist).unwrap_or(f32::INFINITY)
        }
    }

    #[inline]
    pub fn push(&mut self, dist: f32, item: T) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(Scored { dist, item });
        } else if dist < self.worst() {
            self.heap.pop();
            self.heap.push(Scored { dist, item });
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Sorted ascending by distance (best first).
    pub fn into_sorted(self) -> Vec<Scored<T>> {
        let mut v = self.heap.into_vec();
        v.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        v
    }
}

/// Exact flat index over prepared (normalized for cosine) vectors.
pub struct BruteForce {
    metric: Metric,
    dim: usize,
    data: Vec<f32>,
}

impl BruteForce {
    pub fn new(metric: Metric, dim: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len() % dim.max(1), 0);
        BruteForce { metric, dim, data }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim.max(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn vector(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Top-k `(index, internal distance)` best-first.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut top = TopK::new(k);
        for (i, x) in self.data.chunks_exact(self.dim).enumerate() {
            top.push(self.metric.distance(q, x), i as u32);
        }
        top.into_sorted()
            .into_iter()
            .map(|s| (s.item, s.dist))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn topk_orders_best_first() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 2.0, 0.0];
        let bf = BruteForce::new(Metric::Euclidean, 2, data);
        let r = bf.search(&[1.9, 0.0], 3);
        let ids: Vec<u32> = r.iter().map(|x| x.0).collect();
        assert_eq!(ids, vec![3, 1, 0]);
        let r = BruteForce::new(Metric::Dot, 2, vec![1.0, 0.0, 3.0, 0.0, -2.0, 0.0])
            .search(&[1.0, 0.0], 2);
        assert_eq!(r[0].0, 1);
        assert_eq!(Metric::Dot.to_score(r[0].1), 3.0);
    }
}
