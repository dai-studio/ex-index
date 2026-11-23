//! The immutable "main" index generation: `<index>/main.<gen>/` holding
//! `graph.bin`, `pq.bin`, `ids.bin`, `metadata.log` and `tombstones.bin`.

use crate::bitset::Bitset;
use crate::disk::{write_graph, write_pq, DiskHeader, DiskIndex};
use crate::distance::{mips_augment_into, norm_sq, top_norm_indices, Metric, MIPS_ENTRY_POINTS};
use crate::error::{Error, Result};
use crate::models::IndexConfig;
use crate::pq::PqCodebook;
use crate::util::{fsync_dir, read_exact_at, write_atomic};
use crate::vamana::{Graph, VamanaParams};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const IDS_MAGIC: [u8; 8] = *b"EXIDXIDS";

pub(crate) struct MainIndex {
    pub dir: PathBuf,
    pub disk: DiskIndex,
    pub ids: Vec<String>,
    pub id_map: HashMap<String, u32>,
    meta_index: Vec<(u64, u32)>,
    meta_file: File,
    metric: Metric,
    dim: usize,
}

/// Live records gathered for a (re)build.
#[derive(Default)]
pub(crate) struct BuildInput {
    pub ids: Vec<String>,
    /// Stored vectors (`n × d`, normalized for cosine).
    pub vectors: Vec<f32>,
    pub metas: Vec<Option<Vec<u8>>>,
}

impl MainIndex {
    pub fn open(dir: &Path, cfg: &IndexConfig) -> Result<MainIndex> {
        let disk = DiskIndex::open(dir)?;
        let h = *disk.graph.header();
        if h.metric != cfg.metric || h.dim as usize != cfg.metric.build_dim(cfg.dimension as usize)
        {
            return Err(Error::Corrupt(format!(
                "{}: graph header does not match index config",
                dir.display()
            )));
        }
        let (ids, meta_index) = read_ids(&dir.join("ids.bin"))?;
        if ids.len() != disk.len() {
            return Err(Error::Corrupt(
                "ids.bin count does not match graph.bin".into(),
            ));
        }
        let id_map = ids
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let meta_file = File::open(dir.join("metadata.log"))?;
        Ok(MainIndex {
            dir: dir.to_path_buf(),
            disk,
            ids,
            id_map,
            meta_index,
            meta_file,
            metric: cfg.metric,
            dim: cfg.dimension as usize,
        })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn mips_m(&self) -> f32 {
        self.disk.graph.header().mips_m
    }

    /// Stored (original-space) vector of `slot` (drops the MIPS coordinate).
    pub fn vector_into(&self, slot: u32, out: &mut Vec<f32>) {
        self.disk.graph.vector_into(slot, out);
        out.truncate(self.dim);
    }

    pub fn metadata_raw(&self, slot: u32) -> Result<Option<Vec<u8>>> {
        let (off, len) = self.meta_index[slot as usize];
        if len == 0 {
            return Ok(None);
        }
        let mut buf = vec![0u8; len as usize];
        read_exact_at(&self.meta_file, &mut buf, off)?;
        Ok(Some(buf))
    }

    pub fn metadata(&self, slot: u32) -> Result<Option<Map<String, Value>>> {
        match self.metadata_raw(slot)? {
            None => Ok(None),
            Some(b) => Ok(Some(serde_json::from_slice(&b)?)),
        }
    }

    pub fn load_tombstones(&self) -> Bitset {
        let n = self.len();
        match std::fs::read(self.dir.join("tombstones.bin"))
            .ok()
            .and_then(|b| Bitset::from_bytes(&b))
        {
            Some(mut b) if b.len() <= n => {
                b.grow(n);
                b
            }
            _ => Bitset::new(n),
        }
    }

    pub fn save_tombstones(&self, t: &Bitset) -> Result<()> {
        write_atomic(&self.dir.join("tombstones.bin"), &t.to_bytes())?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn metric(&self) -> Metric {
        self.metric
    }
}

/// Slot ids and `(offset, len)` metadata locations from `ids.bin`.
type IdsFile = (Vec<String>, Vec<(u64, u32)>);

fn read_ids(path: &Path) -> Result<IdsFile> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != IDS_MAGIC {
        return Err(Error::Corrupt("ids.bin: bad magic".into()));
    }
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8)?;
    let n = u64::from_le_bytes(b8) as usize;
    let mut ids = Vec::with_capacity(n);
    let mut metas = Vec::with_capacity(n);
    let mut b2 = [0u8; 2];
    let mut b4 = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut b2)?;
        let mut s = vec![0u8; u16::from_le_bytes(b2) as usize];
        r.read_exact(&mut s)?;
        ids.push(String::from_utf8(s)?);
        r.read_exact(&mut b8)?;
        r.read_exact(&mut b4)?;
        metas.push((u64::from_le_bytes(b8), u32::from_le_bytes(b4)));
    }
    Ok((ids, metas))
}

/// Build a complete main generation into `dir` (which must already exist).
pub(crate) fn write_main(
    dir: &Path,
    cfg: &IndexConfig,
    input: BuildInput,
    seed: u64,
    should_abort: &dyn Fn() -> bool,
) -> Result<()> {
    let metric = cfg.metric;
    let d = cfg.dimension as usize;
    let bd = metric.build_dim(d);
    let n = input.ids.len();
    assert_eq!(input.vectors.len(), n * d);
    let abort = || -> Result<()> {
        if should_abort() {
            Err(Error::Internal("build aborted".into()))
        } else {
            Ok(())
        }
    };

    // metadata.log + ids.bin
    {
        let mut meta_w = BufWriter::new(File::create(dir.join("metadata.log"))?);
        let mut ids_w = BufWriter::new(File::create(dir.join("ids.bin"))?);
        ids_w.write_all(&IDS_MAGIC)?;
        ids_w.write_all(&(n as u64).to_le_bytes())?;
        let mut off = 0u64;
        for (id, meta) in input.ids.iter().zip(&input.metas) {
            ids_w.write_all(&(id.len() as u16).to_le_bytes())?;
            ids_w.write_all(id.as_bytes())?;
            let len = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            if let Some(m) = meta {
                meta_w.write_all(m)?;
                meta_w.write_all(b"\n")?;
            }
            ids_w.write_all(&off.to_le_bytes())?;
            ids_w.write_all(&(len as u32).to_le_bytes())?;
            if len > 0 {
                off += len as u64 + 1;
            }
        }
        meta_w
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        ids_w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    abort()?;

    let entries = if metric == Metric::Dot {
        top_norm_indices(&input.vectors, d, MIPS_ENTRY_POINTS)
    } else {
        Vec::new()
    };

    // PQ: dot indexes quantize the original vectors (inner-product ADC);
    // other metrics quantize the build-space vectors (L2 ADC).
    let m = (cfg.params.pq_subvectors as usize).clamp(1, d);
    let (pq, codes) = if metric == Metric::Dot {
        let pq = PqCodebook::train(&input.vectors, d, m, seed);
        let codes = pq.encode_all(&input.vectors);
        (Some(pq), Some(codes))
    } else {
        (None, None)
    };
    abort()?;

    // build-space vectors
    let (build_vecs, mips_m) = if metric == Metric::Dot {
        let m = input
            .vectors
            .chunks_exact(d)
            .map(|x| norm_sq(x).sqrt())
            .fold(0.0f32, f32::max);
        let mut v = Vec::with_capacity(n * bd);
        for x in input.vectors.chunks_exact(d) {
            mips_augment_into(x, m, &mut v);
        }
        (v, m)
    } else {
        (input.vectors, 0.0)
    };

    let (pq, codes) = match (pq, codes) {
        (Some(p), Some(c)) => (p, c),
        _ => {
            let pq = PqCodebook::train(&build_vecs, bd, m, seed);
            let codes = pq.encode_all(&build_vecs);
            (pq, codes)
        }
    };
    abort()?;

    // Vamana graph
    let params = VamanaParams {
        r: cfg.params.r,
        l_build: cfg.params.l_build,
        alpha: cfg.params.alpha,
    };
    let graph = Graph::build(build_vecs, bd, params, seed);
    abort()?;
    let (data, adj, medoid) = graph.into_parts();
    let header = DiskHeader::new(metric, bd as u32, cfg.params.r, n as u64, medoid, mips_m);
    write_graph(&dir.join("graph.bin"), &header, &data, &adj, &entries)?;
    write_pq(&dir.join("pq.bin"), &pq, &codes)?;
    write_atomic(&dir.join("tombstones.bin"), &Bitset::new(n).to_bytes())?;
    fsync_dir(dir)?;
    Ok(())
}
