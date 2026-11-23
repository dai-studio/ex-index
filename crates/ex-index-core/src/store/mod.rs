//! Index catalog and lifecycle: WAL append + fsync, string id ↔ slot maps,
//! metadata, tombstones, delta buffer, background rebuild with atomic swap,
//! and recovery on startup.
//!
//! Directory layout per index (`<data-dir>/<name>/`):
//! * `manifest.json` — config + current main generation + first live WAL gen
//! * `main.<gen>/` — immutable main index (graph.bin, pq.bin, ids.bin,
//!   metadata.log, tombstones.bin)
//! * `wal.<gen>.log` — un-merged upserts / deletes (delta is rebuilt from them)
//!
//! A rebuild writes `main.<gen>.tmp/`, renames it to `main.<gen>/` and then
//! atomically replaces `manifest.json` (temp + rename), which is the commit
//! point. Stale generations and consumed WALs are removed afterwards (and on
//! the next start if removal failed, e.g. due to Windows file locks).

mod delta;
mod main_index;

use crate::bitset::Bitset;
use crate::brute::TopK;
use crate::distance::{query_to_build_space, Metric};
use crate::error::{Error, Result};
use crate::models::{
    filter_matches, CreateIndexRequest, IndexConfig, IndexInfo, IndexStatus, SearchHit,
    SearchRequest, VectorRecord, MAX_UPSERT_BATCH,
};
use crate::util::{dir_size, fsync_dir, now_unix, rfc3339, write_atomic};
use crate::vamana::VamanaParams;
use crate::wal::{list_generations, replay, wal_path, WalOp, WalWriter};
use delta::Delta;
use main_index::{write_main, BuildInput, MainIndex};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MANIFEST: &str = "manifest.json";
const DELETED_MARKER: &str = "DELETED";
const MANIFEST_VERSION: u32 = 1;

/// Store tuning knobs.
#[derive(Debug, Clone)]
pub struct StoreOptions {
    /// Below this many live vectors, searches are exact brute force and no
    /// disk index is built automatically.
    pub small_index_threshold: usize,
    /// Auto-rebuild once the delta holds this many vectors.
    pub delta_threshold: usize,
    /// Nodes expanded per beam-search round on the disk index.
    pub beam_width: usize,
    /// Trigger builds automatically (first build + delta threshold).
    pub auto_build: bool,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions {
            small_index_threshold: 10_000,
            delta_threshold: 50_000,
            beam_width: 4,
            auto_build: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    format_version: u32,
    config: IndexConfig,
    main_gen: Option<u64>,
    wal_start: u64,
    built_at: Option<u64>,
    main_count: u64,
}

fn read_manifest(dir: &Path) -> Result<Manifest> {
    let m: Manifest = serde_json::from_slice(&std::fs::read(dir.join(MANIFEST))?)?;
    if m.format_version != MANIFEST_VERSION {
        return Err(Error::Corrupt(format!(
            "unsupported manifest version {}",
            m.format_version
        )));
    }
    Ok(m)
}

fn write_manifest(dir: &Path, m: &Manifest) -> Result<()> {
    write_atomic(&dir.join(MANIFEST), &serde_json::to_vec_pretty(m)?)?;
    Ok(())
}

fn main_dir(dir: &Path, gen: u64) -> PathBuf {
    dir.join(format!("main.{gen}"))
}

/// Where a live copy of an id lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Loc {
    Main(u32),
    Frozen(usize, u32),
    Active(u32),
}

struct FrozenDelta {
    data: Arc<Delta>,
    tomb: Bitset,
}

struct State {
    main: Option<Arc<MainIndex>>,
    main_tomb: Bitset,
    /// Deltas being (or waiting to be) merged by a build.
    frozen: Vec<FrozenDelta>,
    active: Delta,
    active_tomb: Bitset,
    /// Ids upserted / deleted since the running build froze its snapshot.
    touched: HashSet<String>,
    main_gen: Option<u64>,
    wal_start: u64,
    built_at: Option<u64>,
}

impl State {
    fn locate(&self, id: &str) -> Option<Loc> {
        if let Some(&s) = self.active.id_map.get(id) {
            if !self.active_tomb.get(s as usize) {
                return Some(Loc::Active(s));
            }
        }
        for (i, f) in self.frozen.iter().enumerate().rev() {
            if let Some(&s) = f.data.id_map.get(id) {
                if !f.tomb.get(s as usize) {
                    return Some(Loc::Frozen(i, s));
                }
            }
        }
        if let Some(main) = &self.main {
            if let Some(&s) = main.id_map.get(id) {
                if !self.main_tomb.get(s as usize) {
                    return Some(Loc::Main(s));
                }
            }
        }
        None
    }

    fn tombstone(&mut self, loc: Loc) {
        match loc {
            Loc::Main(s) => self.main_tomb.set(s as usize),
            Loc::Frozen(i, s) => self.frozen[i].tomb.set(s as usize),
            Loc::Active(s) => self.active_tomb.set(s as usize),
        };
    }

    /// Tombstone every live copy of `id`; returns whether one existed.
    fn remove_id(&mut self, id: &str) -> bool {
        let mut found = false;
        while let Some(loc) = self.locate(id) {
            self.tombstone(loc);
            found = true;
        }
        found
    }

    fn apply(&mut self, op: WalOp, building: bool) {
        match op {
            WalOp::Upsert {
                id,
                vector,
                metadata,
            } => {
                let meta = metadata
                    .and_then(|b| serde_json::from_slice::<Map<String, Value>>(&b).ok())
                    .map(Arc::new);
                self.upsert_one(id, &vector, meta, building);
            }
            WalOp::Delete { id } => {
                self.remove_id(&id);
                if building {
                    self.touched.insert(id);
                }
            }
        }
    }

    fn upsert_one(
        &mut self,
        id: String,
        vector: &[f32],
        meta: Option<Arc<Map<String, Value>>>,
        building: bool,
    ) {
        self.remove_id(&id);
        if building {
            self.touched.insert(id.clone());
        }
        self.active.push(id, vector, meta);
    }

    fn link_active(&mut self) {
        if let Some(main) = &self.main {
            let m = main.mips_m();
            self.active.link_pending(m);
        }
    }

    fn main_live(&self) -> usize {
        self.main
            .as_ref()
            .map(|m| m.len() - self.main_tomb.count_ones())
            .unwrap_or(0)
    }

    fn delta_live(&self) -> usize {
        let f: usize = self
            .frozen
            .iter()
            .map(|f| f.data.len() - f.tomb.count_ones())
            .sum();
        f + self.active.len() - self.active_tomb.count_ones()
    }

    fn deleted(&self) -> usize {
        self.main_tomb.count_ones()
            + self
                .frozen
                .iter()
                .map(|f| f.tomb.count_ones())
                .sum::<usize>()
            + self.active_tomb.count_ones()
    }

    fn live(&self) -> usize {
        self.main_live() + self.delta_live()
    }

    fn id_of(&self, loc: Loc) -> &str {
        match loc {
            Loc::Main(s) => &self.main.as_ref().unwrap().ids[s as usize],
            Loc::Frozen(i, s) => &self.frozen[i].data.ids[s as usize],
            Loc::Active(s) => &self.active.ids[s as usize],
        }
    }

    fn vector_of(&self, loc: Loc) -> Vec<f32> {
        match loc {
            Loc::Main(s) => {
                let mut v = Vec::new();
                self.main.as_ref().unwrap().vector_into(s, &mut v);
                v
            }
            Loc::Frozen(i, s) => self.frozen[i].data.vector(s).to_vec(),
            Loc::Active(s) => self.active.vector(s).to_vec(),
        }
    }

    fn metadata_of(&self, loc: Loc) -> Result<Option<Map<String, Value>>> {
        Ok(match loc {
            Loc::Main(s) => self.main.as_ref().unwrap().metadata(s)?,
            Loc::Frozen(i, s) => self.frozen[i].data.metadata[s as usize].as_deref().cloned(),
            Loc::Active(s) => self.active.metadata[s as usize].as_deref().cloned(),
        })
    }

    /// Does the record at `loc` satisfy `filter`?
    fn passes(&self, loc: Loc, filter: Option<&Map<String, Value>>) -> Result<bool> {
        let Some(f) = filter else { return Ok(true) };
        Ok(match loc {
            Loc::Main(_) => filter_matches(f, self.metadata_of(loc)?.as_ref()),
            Loc::Frozen(i, s) => {
                filter_matches(f, self.frozen[i].data.metadata[s as usize].as_deref())
            }
            Loc::Active(s) => filter_matches(f, self.active.metadata[s as usize].as_deref()),
        })
    }

    /// All deltas (frozen then active) with their tombstones and location tags.
    fn deltas(&self) -> Vec<(&Delta, &Bitset, Option<usize>)> {
        let mut v: Vec<(&Delta, &Bitset, Option<usize>)> = self
            .frozen
            .iter()
            .enumerate()
            .map(|(i, f)| (&*f.data, &f.tomb, Some(i)))
            .collect();
        v.push((&self.active, &self.active_tomb, None));
        v
    }

    /// Exact search over every live vector.
    fn exact_search(
        &self,
        metric: Metric,
        q: &[f32],
        k: usize,
        filter: Option<&Map<String, Value>>,
    ) -> Result<Vec<(f32, Loc)>> {
        let mut top = TopK::new(k);
        if let Some(main) = &self.main {
            let mut buf = Vec::new();
            for s in 0..main.len() as u32 {
                if self.main_tomb.get(s as usize) {
                    continue;
                }
                main.vector_into(s, &mut buf);
                let d = metric.distance(q, &buf);
                if d < top.worst() && self.passes(Loc::Main(s), filter)? {
                    top.push(d, Loc::Main(s));
                }
            }
        }
        for (delta, tomb, fi) in self.deltas() {
            for s in 0..delta.len() as u32 {
                if tomb.get(s as usize) {
                    continue;
                }
                let loc = fi.map(|i| Loc::Frozen(i, s)).unwrap_or(Loc::Active(s));
                let d = metric.distance(q, delta.vector(s));
                if d < top.worst() && self.passes(loc, filter)? {
                    top.push(d, loc);
                }
            }
        }
        Ok(top
            .into_sorted()
            .into_iter()
            .map(|s| (s.dist, s.item))
            .collect())
    }

    /// Approximate candidates from main (disk beam search) + deltas, exactly
    /// re-scored, tombstones dropped, sorted best-first.
    fn ann_candidates(
        &self,
        metric: Metric,
        q: &[f32],
        qb: &[f32],
        l: usize,
        beam: usize,
    ) -> Vec<(f32, Loc)> {
        let mut cands = Vec::with_capacity(l * 2);
        if let Some(main) = &self.main {
            let mut buf = Vec::new();
            for (s, _) in main.disk.search(qb, l, l, beam) {
                if !self.main_tomb.get(s as usize) {
                    main.vector_into(s, &mut buf);
                    cands.push((metric.distance(q, &buf), Loc::Main(s)));
                }
            }
        }
        for (delta, tomb, fi) in self.deltas() {
            for s in delta.candidates(qb, l) {
                if !tomb.get(s as usize) {
                    let loc = fi.map(|i| Loc::Frozen(i, s)).unwrap_or(Loc::Active(s));
                    cands.push((metric.distance(q, delta.vector(s)), loc));
                }
            }
        }
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        cands.dedup_by(|a, b| a.1 == b.1);
        cands
    }
}

struct Snapshot {
    main: Option<Arc<MainIndex>>,
    main_tomb: Bitset,
    frozen: Vec<(Arc<Delta>, Bitset)>,
    wal_start: u64,
}

/// One vector index.
pub struct Index {
    dir: PathBuf,
    config: IndexConfig,
    opts: StoreOptions,
    state: RwLock<State>,
    wal: Mutex<WalWriter>,
    building: AtomicBool,
    closed: AtomicBool,
    last_error: Mutex<Option<String>>,
}

impl Index {
    fn new_delta(cfg: &IndexConfig) -> Delta {
        Delta::new(
            cfg.metric,
            cfg.dimension as usize,
            VamanaParams {
                r: cfg.params.r,
                l_build: cfg.params.l_build,
                alpha: cfg.params.alpha,
            },
        )
    }

    fn create(dir: PathBuf, config: IndexConfig, opts: StoreOptions) -> Result<Arc<Index>> {
        std::fs::create_dir(&dir).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                Error::Conflict(format!("index {:?} already exists", config.name))
            } else {
                e.into()
            }
        })?;
        let manifest = Manifest {
            format_version: MANIFEST_VERSION,
            config: config.clone(),
            main_gen: None,
            wal_start: 0,
            built_at: None,
            main_count: 0,
        };
        let res = (|| {
            write_manifest(&dir, &manifest)?;
            WalWriter::open(&dir, 0)
        })();
        let wal = match res {
            Ok(w) => w,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(e);
            }
        };
        if let Some(parent) = dir.parent() {
            fsync_dir(parent)?;
        }
        let state = State {
            main: None,
            main_tomb: Bitset::default(),
            frozen: Vec::new(),
            active: Self::new_delta(&config),
            active_tomb: Bitset::default(),
            touched: HashSet::new(),
            main_gen: None,
            wal_start: 0,
            built_at: None,
        };
        Ok(Arc::new(Index {
            dir,
            config,
            opts,
            state: RwLock::new(state),
            wal: Mutex::new(wal),
            building: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }))
    }

    /// Open an existing index directory and recover its state.
    fn open(dir: PathBuf, opts: StoreOptions) -> Result<Arc<Index>> {
        let manifest = read_manifest(&dir)?;
        let config = manifest.config.clone();
        config.validate()?;
        let main = match manifest.main_gen {
            Some(g) => Some(Arc::new(MainIndex::open(&main_dir(&dir, g), &config)?)),
            None => None,
        };
        let main_tomb = main
            .as_ref()
            .map(|m| m.load_tombstones())
            .unwrap_or_default();

        // Remove stale generations / temp dirs left by interrupted builds.
        for e in std::fs::read_dir(&dir)?.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("main.")
                && Some(e.path()) != manifest.main_gen.map(|g| main_dir(&dir, g))
            {
                tracing::info!(index = %config.name, %name, "removing stale build directory");
                let _ = std::fs::remove_dir_all(e.path());
            }
        }

        let mut state = State {
            main,
            main_tomb,
            frozen: Vec::new(),
            active: Self::new_delta(&config),
            active_tomb: Bitset::default(),
            touched: HashSet::new(),
            main_gen: manifest.main_gen,
            wal_start: manifest.wal_start,
            built_at: manifest.built_at,
        };
        let mut last_gen = manifest.wal_start;
        let mut replayed = 0usize;
        for g in list_generations(&dir)? {
            if g < manifest.wal_start {
                let _ = std::fs::remove_file(wal_path(&dir, g));
                continue;
            }
            for op in replay(&wal_path(&dir, g))? {
                state.apply(op, false);
                replayed += 1;
            }
            last_gen = g;
        }
        state.link_active();
        let wal = WalWriter::open(&dir, last_gen)?;
        tracing::info!(
            index = %config.name,
            main = state.main_live(),
            delta = state.delta_live(),
            replayed,
            "index recovered"
        );
        Ok(Arc::new(Index {
            dir,
            config,
            opts,
            state: RwLock::new(state),
            wal: Mutex::new(wal),
            building: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }))
    }

    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn is_building(&self) -> bool {
        self.building.load(Ordering::SeqCst)
    }

    /// Error message of the last failed build, if any.
    pub fn last_build_error(&self) -> Option<String> {
        self.last_error.lock().clone()
    }

    fn check_open(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            Err(Error::NotFound(format!(
                "index {:?} not found",
                self.config.name
            )))
        } else {
            Ok(())
        }
    }

    /// Insert or replace a batch. Durable (WAL fsync) and searchable on return.
    /// The whole batch is rejected if any record is invalid.
    pub fn upsert(self: &Arc<Self>, records: Vec<VectorRecord>) -> Result<usize> {
        if records.len() > MAX_UPSERT_BATCH {
            return Err(Error::invalid(format!(
                "at most {MAX_UPSERT_BATCH} vectors per request"
            )));
        }
        let mut prepared = Vec::with_capacity(records.len());
        let mut ops = Vec::with_capacity(records.len());
        for r in records {
            let (v, meta_bytes) = r.prepare(&self.config)?;
            ops.push(WalOp::Upsert {
                id: r.id.clone(),
                vector: v.clone(),
                metadata: meta_bytes,
            });
            prepared.push((r.id, v, r.metadata.map(Arc::new)));
        }
        let n = prepared.len();
        if n == 0 {
            return Ok(0);
        }
        self.check_open()?;
        {
            let mut wal = self.wal.lock();
            wal.append(&ops)?;
            let mut st = self.state.write();
            let building = self.is_building();
            for (id, v, meta) in prepared {
                st.upsert_one(id, &v, meta, building);
            }
            st.link_active();
        }
        self.maybe_auto_build();
        Ok(n)
    }

    /// Delete by id (tombstone). `NotFound` if no live copy exists.
    pub fn delete(&self, id: &str) -> Result<()> {
        self.check_open()?;
        let mut wal = self.wal.lock();
        if self.state.read().locate(id).is_none() {
            return Err(Error::NotFound(format!("vector {id:?} not found")));
        }
        wal.append(&[WalOp::Delete { id: id.to_string() }])?;
        let mut st = self.state.write();
        st.remove_id(id);
        if self.is_building() {
            st.touched.insert(id.to_string());
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<VectorRecord> {
        self.check_open()?;
        let st = self.state.read();
        let loc = st
            .locate(id)
            .ok_or_else(|| Error::NotFound(format!("vector {id:?} not found")))?;
        Ok(VectorRecord {
            id: id.to_string(),
            vector: st.vector_of(loc),
            metadata: st.metadata_of(loc)?,
        })
    }

    /// Top-k search, best-first, with raw per-metric scores.
    pub fn search(&self, req: &SearchRequest) -> Result<Vec<SearchHit>> {
        let l_search = req.effective_l_search(&self.config)?;
        let q = self.config.prepare_vector(&req.vector)?;
        self.check_open()?;
        let metric = self.config.metric;
        let k = req.k as usize;
        let filter = req.filter.as_ref().filter(|f| !f.is_empty());
        let st = self.state.read();
        let exact = st.main.is_none() || st.live() < self.opts.small_index_threshold;
        let results: Vec<(f32, Loc)> = if exact {
            st.exact_search(metric, &q, k, filter)?
        } else {
            let qb = query_to_build_space(metric, &q);
            let cap = l_search.saturating_mul(8).max(k);
            let mut fetch = if filter.is_some() {
                (k * 4).min(cap)
            } else {
                k
            };
            loop {
                let l = l_search.max(fetch);
                let cands = st.ann_candidates(metric, &q, &qb, l, self.opts.beam_width);
                let mut hits = Vec::with_capacity(k);
                for (d, loc) in cands {
                    if st.passes(loc, filter)? {
                        hits.push((d, loc));
                        if hits.len() == k {
                            break;
                        }
                    }
                }
                if hits.len() >= k || filter.is_none() || fetch >= cap {
                    break hits;
                }
                fetch = (fetch * 2).min(cap);
            }
        };
        let mut out = Vec::with_capacity(results.len());
        for (d, loc) in results {
            out.push(SearchHit {
                id: st.id_of(loc).to_string(),
                score: metric.to_score(d),
                metadata: if req.include_metadata {
                    st.metadata_of(loc)?
                } else {
                    None
                },
                vector: if req.include_vector {
                    Some(st.vector_of(loc))
                } else {
                    None
                },
            });
        }
        Ok(out)
    }

    pub fn info(&self) -> IndexInfo {
        let st = self.state.read();
        IndexInfo {
            config: self.config.clone(),
            count: st.live() as u64,
            deleted: st.deleted() as u64,
            delta_count: st.delta_live() as u64,
            status: if self.is_building() {
                IndexStatus::Building
            } else {
                IndexStatus::Ready
            },
            built_at: st.built_at.map(rfc3339),
            size_bytes: dir_size(&self.dir),
        }
    }

    /// Live vector count (cheap).
    pub fn count(&self) -> usize {
        self.state.read().live()
    }

    fn maybe_auto_build(self: &Arc<Self>) {
        if !self.opts.auto_build || self.is_building() || self.closed.load(Ordering::SeqCst) {
            return;
        }
        let need = {
            let st = self.state.read();
            if st.main.is_none() {
                st.live() >= self.opts.small_index_threshold
            } else {
                st.active.len() >= self.opts.delta_threshold
            }
        };
        if need {
            match self.start_build() {
                Ok(()) => tracing::info!(index = %self.config.name, "automatic rebuild started"),
                Err(Error::Conflict(_)) => {}
                Err(e) => {
                    tracing::warn!(index = %self.config.name, error = %e, "could not start automatic rebuild")
                }
            }
        }
    }

    /// Start a background merge of main + delta into a new disk generation.
    /// `Conflict` if a build is already running.
    pub fn start_build(self: &Arc<Self>) -> Result<()> {
        self.check_open()?;
        if self
            .building
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(Error::Conflict(format!(
                "a build is already running for {:?}",
                self.config.name
            )));
        }
        let snapshot = match self.freeze() {
            Ok(s) => s,
            Err(e) => {
                self.building.store(false, Ordering::SeqCst);
                return Err(e);
            }
        };
        let this = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name(format!("build-{}", self.config.name))
            .spawn(move || {
                let started = Instant::now();
                let wal_start = snapshot.wal_start;
                let n_frozen = snapshot.frozen.len();
                let res = this.run_build(snapshot);
                this.finish_build(res, wal_start, n_frozen, started);
            });
        if let Err(e) = spawned {
            self.building.store(false, Ordering::SeqCst);
            return Err(e.into());
        }
        Ok(())
    }

    /// Rotate the WAL and freeze the active delta (writes during the build go
    /// to a fresh delta generation).
    fn freeze(&self) -> Result<Snapshot> {
        let mut wal = self.wal.lock();
        let mut st = self.state.write();
        let new_gen = wal.gen() + 1;
        let new_wal = WalWriter::open(&self.dir, new_gen)?;
        wal.sync()?;
        *wal = new_wal;
        let active = std::mem::replace(&mut st.active, Self::new_delta(&self.config));
        let tomb = std::mem::take(&mut st.active_tomb);
        st.frozen.push(FrozenDelta {
            data: Arc::new(active),
            tomb,
        });
        st.touched.clear();
        Ok(Snapshot {
            main: st.main.clone(),
            main_tomb: st.main_tomb.clone(),
            frozen: st
                .frozen
                .iter()
                .map(|f| (f.data.clone(), f.tomb.clone()))
                .collect(),
            wal_start: new_gen,
        })
    }

    fn run_build(&self, snap: Snapshot) -> Result<Option<(Arc<MainIndex>, u64)>> {
        let d = self.config.dimension as usize;
        let mut input = BuildInput::default();
        if let Some(main) = &snap.main {
            let mut buf = Vec::with_capacity(d);
            for s in 0..main.len() as u32 {
                if snap.main_tomb.get(s as usize) {
                    continue;
                }
                main.vector_into(s, &mut buf);
                input.vectors.extend_from_slice(&buf);
                input.ids.push(main.ids[s as usize].clone());
                input.metas.push(main.metadata_raw(s)?);
            }
        }
        for (delta, tomb) in &snap.frozen {
            for s in 0..delta.len() as u32 {
                if tomb.get(s as usize) {
                    continue;
                }
                input.vectors.extend_from_slice(delta.vector(s));
                input.ids.push(delta.ids[s as usize].clone());
                input.metas.push(match &delta.metadata[s as usize] {
                    Some(m) => Some(serde_json::to_vec(&**m)?),
                    None => None,
                });
            }
        }
        drop(snap);
        if input.ids.is_empty() {
            return Ok(None);
        }
        let gen = {
            // generation numbers only grow; skip any leftover directories
            let st = self.state.read();
            st.main_gen.map_or(1, |g| g + 1)
        };
        let tmp = self.dir.join(format!("main.{gen}.tmp"));
        let fin = main_dir(&self.dir, gen);
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        // create_dir (not _all): fails if the index directory was deleted.
        std::fs::create_dir(&tmp)?;
        let closed = || self.closed.load(Ordering::SeqCst);
        let res = write_main(&tmp, &self.config, input, gen ^ 0x5EED, &closed);
        if let Err(e) = res {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(e);
        }
        if fin.exists() {
            std::fs::remove_dir_all(&fin)?;
        }
        std::fs::rename(&tmp, &fin)?;
        fsync_dir(&self.dir)?;
        Ok(Some((Arc::new(MainIndex::open(&fin, &self.config)?), gen)))
    }

    fn finish_build(
        &self,
        res: Result<Option<(Arc<MainIndex>, u64)>>,
        wal_start: u64,
        n_frozen: usize,
        t0: Instant,
    ) {
        let mut old_main_dir = None;
        match res {
            Ok(new) => {
                let mut st = self.state.write();
                if self.closed.load(Ordering::SeqCst) {
                    drop(st);
                    if let Some((m, _)) = new {
                        let d = m.dir.clone();
                        drop(m);
                        let _ = std::fs::remove_dir_all(d);
                    }
                    self.building.store(false, Ordering::SeqCst);
                    return;
                }
                let (new_main, new_gen) = match new {
                    Some((m, g)) => (Some(m), Some(g)),
                    None => (None, None),
                };
                let built_at = now_unix();
                let manifest = Manifest {
                    format_version: MANIFEST_VERSION,
                    config: self.config.clone(),
                    main_gen: new_gen,
                    wal_start,
                    built_at: Some(built_at),
                    main_count: new_main.as_ref().map(|m| m.len() as u64).unwrap_or(0),
                };
                if let Err(e) = write_manifest(&self.dir, &manifest) {
                    tracing::error!(index = %self.config.name, error = %e, "build commit failed");
                    *self.last_error.lock() = Some(e.to_string());
                    drop(st);
                    if let Some(m) = new_main {
                        let d = m.dir.clone();
                        drop(m);
                        let _ = std::fs::remove_dir_all(d);
                    }
                    self.building.store(false, Ordering::SeqCst);
                    return;
                }
                // Writes that happened during the build supersede the snapshot.
                let mut tomb = Bitset::new(new_main.as_ref().map(|m| m.len()).unwrap_or(0));
                if let Some(m) = &new_main {
                    for id in &st.touched {
                        if let Some(&s) = m.id_map.get(id) {
                            tomb.set(s as usize);
                        }
                    }
                }
                old_main_dir = st.main.as_ref().map(|m| m.dir.clone());
                let old_main = std::mem::replace(&mut st.main, new_main);
                st.main_tomb = tomb;
                let nf = n_frozen.min(st.frozen.len());
                st.frozen.drain(..nf);
                st.main_gen = new_gen;
                st.wal_start = wal_start;
                st.built_at = Some(built_at);
                st.touched.clear();
                st.link_active();
                drop(st);
                drop(old_main); // closes the old mmap before its files are removed
                *self.last_error.lock() = None;
                tracing::info!(
                    index = %self.config.name,
                    gen = ?new_gen,
                    secs = t0.elapsed().as_secs_f64(),
                    "build committed"
                );
            }
            Err(e) => {
                if !self.closed.load(Ordering::SeqCst) {
                    tracing::error!(index = %self.config.name, error = %e, "build failed");
                    *self.last_error.lock() = Some(e.to_string());
                }
            }
        }
        // Best-effort cleanup (retried at next start if it fails).
        if let Some(d) = old_main_dir {
            if let Err(e) = std::fs::remove_dir_all(&d) {
                tracing::warn!(dir = %d.display(), error = %e, "could not remove old generation");
            }
        }
        if self.last_error.lock().is_none() {
            if let Ok(gens) = list_generations(&self.dir) {
                for g in gens.into_iter().filter(|g| *g < wal_start) {
                    let _ = std::fs::remove_file(wal_path(&self.dir, g));
                }
            }
        }
        self.building.store(false, Ordering::SeqCst);
    }

    /// Block until no build is running (or `timeout` elapses). Returns true
    /// when idle.
    pub fn wait_for_build(&self, timeout: Duration) -> bool {
        let t0 = Instant::now();
        while self.is_building() {
            if t0.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    }

    /// fsync the WAL and checkpoint main tombstones (graceful shutdown).
    pub fn flush(&self) -> Result<()> {
        let wal = self.wal.lock();
        wal.sync()?;
        let st = self.state.read();
        if let Some(m) = &st.main {
            m.save_tombstones(&st.main_tomb)?;
        }
        Ok(())
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

/// All indexes under the data directory.
pub struct Catalog {
    dir: PathBuf,
    opts: StoreOptions,
    indexes: RwLock<BTreeMap<String, Arc<Index>>>,
}

impl Catalog {
    /// Open (creating if needed) the data directory and recover every index.
    pub fn open(dir: impl Into<PathBuf>, opts: StoreOptions) -> Result<Catalog> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let mut map = BTreeMap::new();
        for e in std::fs::read_dir(&dir)? {
            let e = e?;
            let path = e.path();
            if !path.is_dir() {
                continue;
            }
            if path.join(DELETED_MARKER).exists() {
                let _ = std::fs::remove_dir_all(&path);
                continue;
            }
            if !path.join(MANIFEST).exists() {
                continue;
            }
            match Index::open(path.clone(), opts.clone()) {
                Ok(idx) => {
                    map.insert(idx.name().to_string(), idx);
                }
                Err(e) => {
                    return Err(Error::Internal(format!(
                        "failed to open index at {}: {e}",
                        path.display()
                    )));
                }
            }
        }
        let cat = Catalog {
            dir,
            opts,
            indexes: RwLock::new(map),
        };
        for idx in cat.list() {
            idx.maybe_auto_build();
        }
        Ok(cat)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn create(&self, req: CreateIndexRequest) -> Result<Arc<Index>> {
        let cfg = req.into_config()?;
        let mut map = self.indexes.write();
        if map.contains_key(&cfg.name) {
            return Err(Error::Conflict(format!(
                "index {:?} already exists",
                cfg.name
            )));
        }
        let idx = Index::create(self.dir.join(&cfg.name), cfg.clone(), self.opts.clone())?;
        map.insert(cfg.name, idx.clone());
        Ok(idx)
    }

    pub fn get(&self, name: &str) -> Result<Arc<Index>> {
        self.indexes
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("index {name:?} not found")))
    }

    pub fn list(&self) -> Vec<Arc<Index>> {
        self.indexes.read().values().cloned().collect()
    }

    /// Remove an index and its directory.
    pub fn delete(&self, name: &str) -> Result<()> {
        let idx = self
            .indexes
            .write()
            .remove(name)
            .ok_or_else(|| Error::NotFound(format!("index {name:?} not found")))?;
        idx.close();
        // Let an in-flight build notice the close flag before removing files.
        idx.wait_for_build(Duration::from_secs(5));
        let dir = idx.dir.clone();
        drop(idx);
        let mut last = None;
        for _ in 0..20 {
            match std::fs::remove_dir_all(&dir) {
                Ok(()) => return Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(e) => last = Some(e),
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        tracing::warn!(dir = %dir.display(), error = ?last, "index directory removal deferred to next start");
        let _ = std::fs::write(dir.join(DELETED_MARKER), b"");
        Ok(())
    }

    /// Flush every index (graceful shutdown).
    pub fn shutdown(&self) {
        for idx in self.list() {
            if let Err(e) = idx.flush() {
                tracing::error!(index = %idx.name(), error = %e, "flush failed");
            }
        }
    }
}

#[cfg(test)]
mod tests;
