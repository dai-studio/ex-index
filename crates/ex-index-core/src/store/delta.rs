//! In-memory delta buffer: vectors upserted since the last build. Indexed by
//! an incrementally-built Vamana graph once a main index exists; vectors that
//! cannot be placed in the graph (dot vectors above the graph's max norm, or
//! anything before linking) are scanned exactly.

use crate::distance::{mips_augment_into, norm_sq, Metric, MIPS_ENTRY_POINTS};
use crate::vamana::{Graph, VamanaParams};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Deltas at or below this many slots are always scanned exactly.
const EXACT_DELTA_MAX: usize = 2048;

pub(crate) struct Delta {
    metric: Metric,
    dim: usize,
    params: VamanaParams,
    pub ids: Vec<String>,
    vectors: Vec<f32>,
    pub metadata: Vec<Option<Arc<Map<String, Value>>>>,
    pub id_map: HashMap<String, u32>,
    graph: Option<Graph>,
    graph_slots: Vec<u32>,
    unlinked: Vec<u32>,
    mips_m: f32,
    /// Dot: graph nodes with the largest norms (extra search entry points).
    entries: Vec<u32>,
}

impl Delta {
    pub fn new(metric: Metric, dim: usize, params: VamanaParams) -> Self {
        Delta {
            metric,
            dim,
            params,
            ids: Vec::new(),
            vectors: Vec::new(),
            metadata: Vec::new(),
            id_map: HashMap::new(),
            graph: None,
            graph_slots: Vec::new(),
            unlinked: Vec::new(),
            mips_m: 0.0,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    pub fn vector(&self, slot: u32) -> &[f32] {
        let s = slot as usize;
        &self.vectors[s * self.dim..(s + 1) * self.dim]
    }

    pub fn push(
        &mut self,
        id: String,
        vector: &[f32],
        meta: Option<Arc<Map<String, Value>>>,
    ) -> u32 {
        let slot = self.ids.len() as u32;
        self.vectors.extend_from_slice(vector);
        self.metadata.push(meta);
        self.id_map.insert(id.clone(), slot);
        self.ids.push(id);
        self.unlinked.push(slot);
        slot
    }

    /// Link every pending slot into the delta graph (called once a main index
    /// exists). For dot indexes the graph's MIPS max norm is fixed when the
    /// graph is created; larger vectors stay in the exact-scan list.
    pub fn link_pending(&mut self, main_mips_m: f32) {
        if self.unlinked.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.unlinked);
        let bd = self.metric.build_dim(self.dim);
        let mut linkable = Vec::with_capacity(pending.len());
        let mut buf = Vec::with_capacity(pending.len() * bd);
        if self.metric == Metric::Dot {
            if self.graph.is_none() {
                let pend_max = pending
                    .iter()
                    .map(|&s| norm_sq(self.vector(s)).sqrt())
                    .fold(0.0f32, f32::max);
                self.mips_m = main_mips_m.max(pend_max);
            }
            let limit = self.mips_m * (1.0 + 1e-6);
            for s in pending {
                if norm_sq(self.vector(s)).sqrt() <= limit {
                    mips_augment_into(self.vector(s), self.mips_m, &mut buf);
                    linkable.push(s);
                } else {
                    self.unlinked.push(s);
                }
            }
        } else {
            for s in pending {
                buf.extend_from_slice(self.vector(s));
                linkable.push(s);
            }
        }
        if linkable.is_empty() {
            return;
        }
        let params = self.params;
        let g = self
            .graph
            .get_or_insert_with(|| Graph::new_empty(bd, params));
        g.insert_batch(&buf);
        self.graph_slots.extend(linkable);
        if self.metric == Metric::Dot {
            let mut nodes: Vec<(f32, u32)> = self
                .graph_slots
                .iter()
                .enumerate()
                .map(|(node, &slot)| (norm_sq(self.vector(slot)), node as u32))
                .collect();
            let k = MIPS_ENTRY_POINTS.min(nodes.len());
            nodes.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            self.entries = nodes[..k].iter().map(|x| x.1).collect();
        }
    }

    /// Candidate slots for a query (`q_build` is in build space). Tombstones
    /// are filtered by the caller.
    pub fn candidates(&self, q_build: &[f32], l: usize) -> Vec<u32> {
        match &self.graph {
            Some(g) if self.len() > EXACT_DELTA_MAX => {
                let mut out: Vec<u32> = g
                    .greedy_search(q_build, l)
                    .into_iter()
                    .map(|(n, _)| self.graph_slots[n as usize])
                    .collect();
                out.extend_from_slice(&self.unlinked);
                out
            }
            _ => (0..self.len() as u32).collect(),
        }
    }
}
