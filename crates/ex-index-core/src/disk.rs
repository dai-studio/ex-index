//! On-disk DiskANN layout and beam search.
//!
//! `graph.bin`:
//! * sector 0 — header: magic `EXIDXGRF`, format version, metric, d', R,
//!   node_size, nodes_per_sector, sectors_per_node, count, medoid, MIPS max
//!   norm, then `u32 n_entries` + extra entry points (dot indexes: the
//!   largest-norm vectors, which seed beam search alongside the medoid). All
//!   integers / floats little-endian.
//! * sectors 1.. — node records `[f32 × d'] [u32 neighbor_count] [u32 × R]`.
//!   When a record fits in a 4 KiB sector, `floor(4096 / node_size)` records
//!   are packed per sector (never straddling a sector boundary); otherwise a
//!   record spans `ceil(node_size / 4096)` whole sectors (up to 4096-d).
//!
//! `pq.bin`: PQ codebook followed by `count` and the `count × m` codes, which
//! are loaded into RAM to navigate; full vectors are read from the mmap and
//! used to re-rank. For dot indexes the codebook covers the original `d`
//! dimensions (not the MIPS coordinate) and navigation uses inner-product
//! ADC; other metrics use L2 ADC over the build-space vectors.

use crate::distance::{l2sq, Metric};
use crate::error::{Error, Result};
use crate::pq::PqCodebook;
use crate::util::{fsync_dir, FxHashSet};
use crate::vamana::CandList;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const SECTOR: usize = 4096;
pub const GRAPH_MAGIC: [u8; 8] = *b"EXIDXGRF";
pub const FORMAT_VERSION: u32 = 1;
/// Max extra entry points stored in the header sector.
pub const MAX_ENTRY_POINTS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskHeader {
    pub metric: Metric,
    /// Build-space dimension d' (`d + 1` for dot).
    pub dim: u32,
    pub r: u32,
    pub node_size: u32,
    pub nodes_per_sector: u32,
    pub sectors_per_node: u32,
    pub count: u64,
    pub medoid: u32,
    /// MIPS→L2 max norm M (0 for non-dot indexes).
    pub mips_m: f32,
}

impl DiskHeader {
    pub fn new(metric: Metric, dim: u32, r: u32, count: u64, medoid: u32, mips_m: f32) -> Self {
        let node_size = dim * 4 + 4 + r * 4;
        let (nps, spn) = if node_size as usize <= SECTOR {
            (SECTOR as u32 / node_size, 1)
        } else {
            (0, (node_size as usize).div_ceil(SECTOR) as u32)
        };
        DiskHeader {
            metric,
            dim,
            r,
            node_size,
            nodes_per_sector: nps,
            sectors_per_node: spn,
            count,
            medoid,
            mips_m,
        }
    }

    /// Byte offset of node `i`'s record.
    #[inline]
    pub fn node_offset(&self, i: u64) -> usize {
        if self.nodes_per_sector > 0 {
            let nps = self.nodes_per_sector as u64;
            (SECTOR as u64 * (1 + i / nps) + (i % nps) * self.node_size as u64) as usize
        } else {
            (SECTOR as u64 * (1 + i * self.sectors_per_node as u64)) as usize
        }
    }

    pub fn file_len(&self) -> usize {
        let data_sectors = if self.nodes_per_sector > 0 {
            self.count.div_ceil(self.nodes_per_sector as u64)
        } else {
            self.count * self.sectors_per_node as u64
        };
        SECTOR * (1 + data_sectors as usize)
    }

    fn to_sector(self, entries: &[u32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(SECTOR);
        b.extend_from_slice(&GRAPH_MAGIC);
        for v in [
            FORMAT_VERSION,
            self.metric.to_u32(),
            self.dim,
            self.r,
            self.node_size,
            self.nodes_per_sector,
            self.sectors_per_node,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&self.count.to_le_bytes());
        b.extend_from_slice(&self.medoid.to_le_bytes());
        b.extend_from_slice(&self.mips_m.to_le_bytes());
        let entries = &entries[..entries.len().min(MAX_ENTRY_POINTS)];
        b.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in entries {
            b.extend_from_slice(&e.to_le_bytes());
        }
        b.resize(SECTOR, 0);
        b
    }

    fn parse_entries(b: &[u8], count: u64) -> Result<Vec<u32>> {
        let n = u32::from_le_bytes(b[52..56].try_into().unwrap()) as usize;
        if n > MAX_ENTRY_POINTS {
            return Err(Error::Corrupt("graph.bin: too many entry points".into()));
        }
        let e: Vec<u32> = b[56..56 + n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if e.iter().any(|&x| x as u64 >= count) {
            return Err(Error::Corrupt("graph.bin: entry point out of range".into()));
        }
        Ok(e)
    }

    fn parse(b: &[u8]) -> Result<DiskHeader> {
        if b.len() < SECTOR || b[..8] != GRAPH_MAGIC {
            return Err(Error::Corrupt("graph.bin: bad magic".into()));
        }
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        if u(8) != FORMAT_VERSION {
            return Err(Error::Corrupt(format!(
                "graph.bin: unsupported format version {}",
                u(8)
            )));
        }
        let metric = Metric::from_u32(u(12))
            .ok_or_else(|| Error::Corrupt("graph.bin: bad metric".into()))?;
        let count = u64::from_le_bytes(b[36..44].try_into().unwrap());
        let h = DiskHeader::new(
            metric,
            u(16),
            u(20),
            count,
            u(44),
            f32::from_le_bytes(b[48..52].try_into().unwrap()),
        );
        if h.node_size != u(24) || h.nodes_per_sector != u(28) || h.sectors_per_node != u(32) {
            return Err(Error::Corrupt("graph.bin: inconsistent node layout".into()));
        }
        if count > 0 && h.medoid as u64 >= count {
            return Err(Error::Corrupt("graph.bin: medoid out of range".into()));
        }
        Ok(h)
    }
}

/// Write `graph.bin` via temp file + fsync + rename.
pub fn write_graph(
    path: &Path,
    header: &DiskHeader,
    vectors: &[f32],
    adj: &[Vec<u32>],
    entries: &[u32],
) -> Result<()> {
    let d = header.dim as usize;
    let r = header.r as usize;
    assert_eq!(vectors.len(), adj.len() * d);
    assert_eq!(header.count as usize, adj.len());
    let tmp = path.with_extension("bin.tmp");
    {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        w.write_all(&header.to_sector(entries))?;
        let mut sector = vec![0u8; SECTOR * header.sectors_per_node as usize];
        let mut rec = Vec::with_capacity(header.node_size as usize);
        let mut in_sector = 0usize;
        for (i, nb) in adj.iter().enumerate() {
            rec.clear();
            for x in &vectors[i * d..(i + 1) * d] {
                rec.extend_from_slice(&x.to_le_bytes());
            }
            let cnt = nb.len().min(r);
            rec.extend_from_slice(&(cnt as u32).to_le_bytes());
            for k in 0..r {
                let v = nb.get(k).filter(|_| k < cnt).copied().unwrap_or(0);
                rec.extend_from_slice(&v.to_le_bytes());
            }
            if header.nodes_per_sector > 0 {
                let off = in_sector * header.node_size as usize;
                sector[off..off + rec.len()].copy_from_slice(&rec);
                in_sector += 1;
                if in_sector == header.nodes_per_sector as usize {
                    w.write_all(&sector)?;
                    sector.iter_mut().for_each(|b| *b = 0);
                    in_sector = 0;
                }
            } else {
                sector.iter_mut().for_each(|b| *b = 0);
                sector[..rec.len()].copy_from_slice(&rec);
                w.write_all(&sector)?;
            }
        }
        if header.nodes_per_sector > 0 && in_sector > 0 {
            w.write_all(&sector)?;
        }
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        fsync_dir(dir)?;
    }
    Ok(())
}

/// Write `pq.bin` (codebook + codes) via temp file + rename.
pub fn write_pq(path: &Path, pq: &PqCodebook, codes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("bin.tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        pq.write_to(&mut w)?;
        let count = (codes.len() / pq.m.max(1)) as u64;
        w.write_all(&count.to_le_bytes())?;
        w.write_all(codes)?;
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn read_pq(path: &Path) -> Result<(PqCodebook, Vec<u8>)> {
    let mut r = BufReader::new(File::open(path)?);
    let pq = PqCodebook::read_from(&mut r)?;
    let mut c = [0u8; 8];
    r.read_exact(&mut c)?;
    let count = u64::from_le_bytes(c) as usize;
    let mut codes = vec![0u8; count * pq.m];
    r.read_exact(&mut codes)?;
    Ok((pq, codes))
}

/// Memory-mapped `graph.bin`.
pub struct DiskGraph {
    mmap: Mmap,
    header: DiskHeader,
    entries: Vec<u32>,
}

impl DiskGraph {
    pub fn open(path: &Path) -> Result<DiskGraph> {
        let file = File::open(path)?;
        // SAFETY: the file is written once (temp + rename) and never modified
        // in place while mapped; index directories are owned by this process.
        let mmap = unsafe { Mmap::map(&file)? };
        let header = DiskHeader::parse(&mmap)?;
        let entries = DiskHeader::parse_entries(&mmap, header.count)?;
        if mmap.len() < header.file_len() {
            return Err(Error::Corrupt(format!(
                "graph.bin truncated: {} < {} bytes",
                mmap.len(),
                header.file_len()
            )));
        }
        Ok(DiskGraph {
            mmap,
            header,
            entries,
        })
    }

    pub fn header(&self) -> &DiskHeader {
        &self.header
    }

    /// Extra entry points (besides the medoid).
    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.header.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    #[inline]
    fn record(&self, i: u32) -> &[u8] {
        let off = self.header.node_offset(i as u64);
        &self.mmap[off..off + self.header.node_size as usize]
    }

    /// Read node `i`'s full build-space vector into `out`.
    #[inline]
    pub fn vector_into(&self, i: u32, out: &mut Vec<f32>) {
        let d = self.header.dim as usize;
        let rec = &self.record(i)[..d * 4];
        out.clear();
        out.extend(
            rec.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
    }

    /// Read node `i`'s neighbor list into `out`.
    #[inline]
    pub fn neighbors_into(&self, i: u32, out: &mut Vec<u32>) {
        let d = self.header.dim as usize;
        let rec = &self.record(i)[d * 4..];
        let cnt =
            (u32::from_le_bytes(rec[..4].try_into().unwrap()) as usize).min(self.header.r as usize);
        out.clear();
        out.extend(
            rec[4..4 + cnt * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
    }
}

/// Disk index: mmapped graph + full vectors, PQ codes in RAM.
pub struct DiskIndex {
    pub graph: DiskGraph,
    pub pq: PqCodebook,
    pub codes: Vec<u8>,
}

impl DiskIndex {
    pub fn open(dir: &Path) -> Result<DiskIndex> {
        let graph = DiskGraph::open(&dir.join("graph.bin"))?;
        let (pq, codes) = read_pq(&dir.join("pq.bin"))?;
        let h = graph.header;
        let expected_pq_dim = if h.metric == Metric::Dot {
            h.dim as usize - 1
        } else {
            h.dim as usize
        };
        if codes.len() != graph.len() * pq.m || pq.dim != expected_pq_dim {
            return Err(Error::Corrupt("pq.bin does not match graph.bin".into()));
        }
        Ok(DiskIndex { graph, pq, codes })
    }

    pub fn len(&self) -> usize {
        self.graph.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Beam search: navigate with PQ (ADC) distances, expand up to
    /// `beam_width` nodes per round reading their sectors from the mmap, and
    /// re-rank every expanded node with its full-precision vector.
    ///
    /// `q` is in build space. Returns up to `k` `(node, full squared L2)`
    /// best-first.
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        l_search: usize,
        beam_width: usize,
    ) -> Vec<(u32, f32)> {
        if self.graph.is_empty() {
            return Vec::new();
        }
        let m = self.pq.m;
        let table = if self.graph.header.metric == Metric::Dot {
            self.pq.adc_table_ip(&q[..self.pq.dim])
        } else {
            self.pq.adc_table(q)
        };
        let l = l_search.max(k).max(1);
        let mut cands = CandList::new(l);
        let mut visited = FxHashSet::default();
        visited.reserve(l * 8);
        let code = |i: u32| &self.codes[i as usize * m..(i as usize + 1) * m];
        let medoid = self.graph.header.medoid;
        for &start in std::iter::once(&medoid).chain(self.graph.entries()) {
            if visited.insert(start) {
                cands.insert(start, self.pq.adc_distance(&table, code(start)));
            }
        }
        let mut full: Vec<(u32, f32)> = Vec::with_capacity(l * 2);
        let mut vbuf = Vec::with_capacity(self.graph.header.dim as usize);
        let mut nbuf = Vec::with_capacity(self.graph.header.r as usize);
        let mut beam = Vec::with_capacity(beam_width.max(1));
        loop {
            beam.clear();
            while beam.len() < beam_width.max(1) {
                match cands.pop_unexpanded() {
                    Some((id, _)) => beam.push(id),
                    None => break,
                }
            }
            if beam.is_empty() {
                break;
            }
            for &id in &beam {
                self.graph.vector_into(id, &mut vbuf);
                full.push((id, l2sq(q, &vbuf)));
                self.graph.neighbors_into(id, &mut nbuf);
                for &nb in &nbuf {
                    if visited.insert(nb) {
                        cands.insert(nb, self.pq.adc_distance(&table, code(nb)));
                    }
                }
            }
        }
        full.sort_by(|a, b| a.1.total_cmp(&b.1));
        full.truncate(k);
        full
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{gaussian_vectors, ground_truth, recall_at_k};
    use crate::vamana::{Graph, VamanaParams};

    fn build_disk(dir: &Path, data: &[f32], d: usize, m: usize) -> (DiskIndex, Graph) {
        let g = Graph::build(data.to_vec(), d, VamanaParams::default(), 7);
        let pq = PqCodebook::train(data, d, m, 7);
        let codes = pq.encode_all(data);
        let n = data.len() / d;
        let adj: Vec<Vec<u32>> = (0..n as u32).map(|i| g.neighbors(i)).collect();
        let h = DiskHeader::new(Metric::Euclidean, d as u32, 64, n as u64, g.medoid(), 0.0);
        write_graph(&dir.join("graph.bin"), &h, data, &adj, &[]).unwrap();
        write_pq(&dir.join("pq.bin"), &pq, &codes).unwrap();
        (DiskIndex::open(dir).unwrap(), g)
    }

    #[test]
    fn header_layout() {
        let h = DiskHeader::new(Metric::Cosine, 128, 64, 10, 3, 0.0);
        assert_eq!(h.node_size, 128 * 4 + 4 + 256);
        assert_eq!(h.nodes_per_sector, 5);
        assert_eq!(h.node_offset(0), 4096);
        assert_eq!(h.node_offset(5), 8192);
        let h = DiskHeader::new(Metric::Dot, 4097, 64, 10, 3, 1.0);
        assert_eq!(h.nodes_per_sector, 0);
        assert_eq!(h.sectors_per_node, 5);
        assert_eq!(h.node_offset(2), 4096 * 11);
        let sec = h.to_sector(&[1, 7]);
        assert_eq!(DiskHeader::parse(&sec).unwrap(), h);
        assert_eq!(DiskHeader::parse_entries(&sec, 10).unwrap(), vec![1, 7]);
        assert!(DiskHeader::parse_entries(&h.to_sector(&[10]), 10).is_err());
    }

    #[test]
    fn disk_round_trip_and_recall() {
        let dir = tempfile::tempdir().unwrap();
        let (n, d) = (4000, 32);
        let data = gaussian_vectors(n, d, 3);
        let (disk, g) = build_disk(dir.path(), &data, d, 8);
        let mut v = Vec::new();
        let mut nb = Vec::new();
        for i in [0u32, 1, 1234, 3999] {
            disk.graph.vector_into(i, &mut v);
            assert_eq!(&v[..], g.vector(i));
            disk.graph.neighbors_into(i, &mut nb);
            assert_eq!(nb, g.neighbors(i));
        }
        let queries = gaussian_vectors(200, d, 33);
        let truth = ground_truth(Metric::Euclidean, d, &data, &queries, 10);
        let got: Vec<Vec<u32>> = queries
            .chunks_exact(d)
            .map(|q| {
                disk.search(q, 10, 100, 4)
                    .into_iter()
                    .map(|x| x.0)
                    .collect()
            })
            .collect();
        let r_disk = recall_at_k(&truth, &got, 10);
        let got_mem: Vec<Vec<u32>> = queries
            .chunks_exact(d)
            .map(|q| g.greedy_search(q, 100).into_iter().map(|x| x.0).collect())
            .collect();
        let r_mem = recall_at_k(&truth, &got_mem, 10);
        assert!(
            r_disk >= 0.95,
            "disk recall@10 {r_disk} (in-memory {r_mem})"
        );
        assert!(r_disk >= r_mem - 0.03, "disk {r_disk} vs in-memory {r_mem}");
    }

    #[test]
    fn multi_sector_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let (n, d) = (300, 1100); // node_size > 4096 → multi-sector records
        let data = gaussian_vectors(n, d, 5);
        let (disk, _) = build_disk(dir.path(), &data, d, 64);
        assert_eq!(disk.graph.header().nodes_per_sector, 0);
        assert_eq!(disk.graph.header().sectors_per_node, 2);
        let mut v = Vec::new();
        disk.graph.vector_into(299, &mut v);
        assert_eq!(&v[..], &data[299 * d..300 * d]);
        let q = &data[17 * d..18 * d];
        assert_eq!(disk.search(q, 1, 50, 4)[0].0, 17);
    }

    #[test]
    fn corrupt_header_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("graph.bin"), vec![0u8; 4096]).unwrap();
        assert!(DiskGraph::open(&dir.path().join("graph.bin")).is_err());
    }
}
