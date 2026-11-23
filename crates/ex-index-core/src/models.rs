//! Wire / persisted data shapes (serde) and their validation rules.

use crate::distance::{normalize, Metric};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const MAX_DIMENSION: u32 = 4096;
pub const MAX_ID_CHARS: usize = 256;
pub const MAX_METADATA_BYTES: usize = 64 * 1024;
pub const MAX_K: u32 = 1000;
pub const MAX_UPSERT_BATCH: usize = 10_000;

/// Graph / PQ / search parameters of an index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexParams {
    /// Max out-degree R of the Vamana graph.
    pub r: u32,
    /// Candidate list size L used while building.
    pub l_build: u32,
    /// RobustPrune slack α (≥ 1).
    pub alpha: f32,
    /// Number of PQ sub-vectors m (bytes per compressed vector).
    pub pq_subvectors: u32,
    /// Default search candidate list size.
    pub l_search: u32,
}

impl IndexParams {
    pub fn defaults_for(dimension: u32) -> Self {
        IndexParams {
            r: 64,
            l_build: 100,
            alpha: 1.2,
            pq_subvectors: dimension.div_ceil(4).max(1),
            l_search: 100,
        }
    }
}

/// Optional user-supplied params (missing fields take defaults).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexParamsInput {
    pub r: Option<u32>,
    pub l_build: Option<u32>,
    pub alpha: Option<f32>,
    pub pq_subvectors: Option<u32>,
    pub l_search: Option<u32>,
}

/// Fully-resolved index configuration (immutable after creation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexConfig {
    pub name: String,
    pub dimension: u32,
    pub metric: Metric,
    pub params: IndexParams,
}

/// Body of `POST /v1/indexes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateIndexRequest {
    pub name: String,
    pub dimension: u32,
    pub metric: Metric,
    #[serde(default)]
    pub params: Option<IndexParamsInput>,
}

/// `^[a-z0-9_-]{1,64}$`
pub fn validate_index_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
    if ok {
        Ok(())
    } else {
        Err(Error::invalid("name must match ^[a-z0-9_-]{1,64}$"))
    }
}

impl CreateIndexRequest {
    /// Apply defaults and validate every field.
    pub fn into_config(self) -> Result<IndexConfig> {
        validate_index_name(&self.name)?;
        if self.dimension == 0 || self.dimension > MAX_DIMENSION {
            return Err(Error::invalid(format!(
                "dimension must be in 1..={MAX_DIMENSION}"
            )));
        }
        let d = IndexParams::defaults_for(self.dimension);
        let p = self.params.unwrap_or_default();
        let params = IndexParams {
            r: p.r.unwrap_or(d.r),
            l_build: p.l_build.unwrap_or(d.l_build),
            alpha: p.alpha.unwrap_or(d.alpha),
            pq_subvectors: p.pq_subvectors.unwrap_or(d.pq_subvectors),
            l_search: p.l_search.unwrap_or(d.l_search),
        };
        let cfg = IndexConfig {
            name: self.name,
            dimension: self.dimension,
            metric: self.metric,
            params,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

impl IndexConfig {
    pub fn validate(&self) -> Result<()> {
        validate_index_name(&self.name)?;
        let p = &self.params;
        if self.dimension == 0 || self.dimension > MAX_DIMENSION {
            return Err(Error::invalid(format!(
                "dimension must be in 1..={MAX_DIMENSION}"
            )));
        }
        if !(2..=512).contains(&p.r) {
            return Err(Error::invalid("params.r must be in 2..=512"));
        }
        if !(1..=5000).contains(&p.l_build) {
            return Err(Error::invalid("params.l_build must be in 1..=5000"));
        }
        if !(p.alpha.is_finite() && (1.0..=4.0).contains(&p.alpha)) {
            return Err(Error::invalid("params.alpha must be in 1.0..=4.0"));
        }
        if p.pq_subvectors == 0 || p.pq_subvectors > self.dimension {
            return Err(Error::invalid(
                "params.pq_subvectors must be in 1..=dimension",
            ));
        }
        if !(1..=5000).contains(&p.l_search) {
            return Err(Error::invalid("params.l_search must be in 1..=5000"));
        }
        Ok(())
    }

    /// Validate a vector against the index's fixed dimension and metric, and
    /// return its stored form (L2-normalized for cosine).
    pub fn prepare_vector(&self, v: &[f32]) -> Result<Vec<f32>> {
        if v.len() != self.dimension as usize {
            return Err(Error::invalid(format!(
                "vector has length {}, index dimension is {}",
                v.len(),
                self.dimension
            )));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid("vector contains NaN or infinite values"));
        }
        let mut out = v.to_vec();
        if self.metric == Metric::Cosine && !normalize(&mut out) {
            return Err(Error::invalid(
                "zero-norm vector is not allowed in a cosine index",
            ));
        }
        Ok(out)
    }
}

/// A vector with its string id and optional JSON metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

pub fn validate_id(id: &str) -> Result<()> {
    let n = id.chars().count();
    if n == 0 || n > MAX_ID_CHARS {
        return Err(Error::invalid(format!(
            "id must be 1..={MAX_ID_CHARS} characters"
        )));
    }
    Ok(())
}

impl VectorRecord {
    /// Validates id + metadata size and returns `(stored vector, metadata JSON bytes)`.
    pub fn prepare(&self, cfg: &IndexConfig) -> Result<(Vec<f32>, Option<Vec<u8>>)> {
        validate_id(&self.id)
            .map_err(|e| Error::invalid(format!("{e} (id {:?})", truncate(&self.id))))?;
        let v = cfg
            .prepare_vector(&self.vector)
            .map_err(|e| Error::invalid(format!("{e} (id {:?})", truncate(&self.id))))?;
        let meta = match &self.metadata {
            None => None,
            Some(m) => {
                let bytes = serde_json::to_vec(m).map_err(|e| Error::Internal(e.to_string()))?;
                if bytes.len() > MAX_METADATA_BYTES {
                    return Err(Error::invalid(format!(
                        "metadata exceeds {} bytes (id {:?})",
                        MAX_METADATA_BYTES,
                        truncate(&self.id)
                    )));
                }
                Some(bytes)
            }
        };
        Ok((v, meta))
    }
}

fn truncate(s: &str) -> String {
    s.chars().take(64).collect()
}

/// Body of `POST /v1/indexes/{name}/vectors`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpsertRequest {
    pub vectors: Vec<VectorRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    pub upserted: usize,
}

fn default_k() -> u32 {
    10
}
fn default_true() -> bool {
    true
}

/// Body of `POST /v1/indexes/{name}/search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: u32,
    #[serde(default)]
    pub l_search: Option<u32>,
    #[serde(default)]
    pub filter: Option<Map<String, Value>>,
    #[serde(default)]
    pub include_vector: bool,
    #[serde(default = "default_true")]
    pub include_metadata: bool,
}

impl SearchRequest {
    pub fn new(vector: Vec<f32>, k: u32) -> Self {
        SearchRequest {
            vector,
            k,
            l_search: None,
            filter: None,
            include_vector: false,
            include_metadata: true,
        }
    }

    /// Validate k / l_search and return the effective l_search.
    pub fn effective_l_search(&self, cfg: &IndexConfig) -> Result<usize> {
        if self.k == 0 || self.k > MAX_K {
            return Err(Error::invalid(format!("k must be in 1..={MAX_K}")));
        }
        match self.l_search {
            Some(l) if l < self.k => Err(Error::invalid("l_search must be >= k")),
            Some(l) if l > 10_000 => Err(Error::invalid("l_search must be <= 10000")),
            Some(l) => Ok(l as usize),
            None => Ok(cfg.params.l_search.max(self.k) as usize),
        }
    }
}

/// One search result. `score` follows the "Score semantics" rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub took_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexStatus {
    Ready,
    Building,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub config: IndexConfig,
    /// Live (searchable) vectors.
    pub count: u64,
    /// Tombstoned slots awaiting consolidation by the next rebuild.
    pub deleted: u64,
    /// Live vectors held in the in-memory delta (not yet merged on disk).
    pub delta_count: u64,
    pub status: IndexStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub built_at: Option<String>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexList {
    pub indexes: Vec<IndexInfo>,
}

/// Top-level metadata equality filter (all keys AND-ed).
pub fn filter_matches(filter: &Map<String, Value>, meta: Option<&Map<String, Value>>) -> bool {
    if filter.is_empty() {
        return true;
    }
    match meta {
        None => false,
        Some(m) => filter.iter().all(|(k, v)| m.get(k) == Some(v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(v: Value) -> Result<IndexConfig> {
        let r: CreateIndexRequest =
            serde_json::from_value(v).map_err(|e| Error::invalid(e.to_string()))?;
        r.into_config()
    }

    #[test]
    fn defaults_applied() {
        let c = req(json!({"name":"docs","dimension":10,"metric":"cosine"})).unwrap();
        assert_eq!(
            c.params,
            IndexParams {
                r: 64,
                l_build: 100,
                alpha: 1.2,
                pq_subvectors: 3,
                l_search: 100
            }
        );
        let c = req(json!({"name":"a","dimension":768,"metric":"dot","params":{"r":32}})).unwrap();
        assert_eq!(c.params.r, 32);
        assert_eq!(c.params.pq_subvectors, 192);
    }

    #[test]
    fn invalid_configs_rejected() {
        for bad in [
            json!({"name":"Bad","dimension":4,"metric":"cosine"}),
            json!({"name":"","dimension":4,"metric":"cosine"}),
            json!({"name":"x".repeat(65),"dimension":4,"metric":"cosine"}),
            json!({"name":"ok","dimension":0,"metric":"cosine"}),
            json!({"name":"ok","dimension":4097,"metric":"cosine"}),
            json!({"name":"ok","dimension":4,"metric":"manhattan"}),
            json!({"name":"ok","dimension":4,"metric":"cosine","params":{"pq_subvectors":5}}),
            json!({"name":"ok","dimension":4,"metric":"cosine","params":{"alpha":0.5}}),
            json!({"name":"ok","dimension":4,"metric":"cosine","params":{"r":1}}),
            json!({"name":"ok","dimension":4,"metric":"cosine","params":{"bogus":1}}),
        ] {
            assert!(req(bad.clone()).is_err(), "{bad}");
        }
        assert!(req(json!({"name":"a-b_9","dimension":4096,"metric":"euclidean"})).is_ok());
    }

    #[test]
    fn vector_validation() {
        let c = req(json!({"name":"c","dimension":3,"metric":"cosine"})).unwrap();
        assert!(c.prepare_vector(&[1.0, 2.0]).is_err());
        assert!(c.prepare_vector(&[1.0, f32::NAN, 0.0]).is_err());
        assert!(c.prepare_vector(&[1.0, f32::INFINITY, 0.0]).is_err());
        assert!(c.prepare_vector(&[0.0, 0.0, 0.0]).is_err());
        let v = c.prepare_vector(&[3.0, 0.0, 4.0]).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[2] - 0.8).abs() < 1e-6);
        let e = req(json!({"name":"e","dimension":3,"metric":"euclidean"})).unwrap();
        assert_eq!(
            e.prepare_vector(&[0.0, 0.0, 0.0]).unwrap(),
            vec![0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn record_validation() {
        let c = req(json!({"name":"c","dimension":2,"metric":"dot"})).unwrap();
        let ok = VectorRecord {
            id: "a".into(),
            vector: vec![1.0, 2.0],
            metadata: Some(json!({"x":1}).as_object().unwrap().clone()),
        };
        let (v, m) = ok.prepare(&c).unwrap();
        assert_eq!(v, vec![1.0, 2.0]);
        assert_eq!(m.unwrap(), br#"{"x":1}"#);
        let mut bad = ok.clone();
        bad.id = String::new();
        assert!(bad.prepare(&c).is_err());
        bad.id = "é".repeat(257);
        assert!(bad.prepare(&c).is_err());
        bad.id = "é".repeat(256);
        assert!(bad.prepare(&c).is_ok());
        let mut big = ok.clone();
        big.metadata = Some(
            json!({"s": "x".repeat(MAX_METADATA_BYTES)})
                .as_object()
                .unwrap()
                .clone(),
        );
        assert!(big.prepare(&c).is_err());
        // metadata must be an object
        assert!(serde_json::from_value::<VectorRecord>(
            json!({"id":"a","vector":[1,2],"metadata":[1]})
        )
        .is_err());
    }

    #[test]
    fn search_request_defaults_and_ranges() {
        let c = req(json!({"name":"c","dimension":2,"metric":"dot"})).unwrap();
        let s: SearchRequest = serde_json::from_value(json!({"vector":[1,2]})).unwrap();
        assert_eq!(s.k, 10);
        assert!(s.include_metadata && !s.include_vector);
        assert_eq!(s.effective_l_search(&c).unwrap(), 100);
        let mut s2 = s.clone();
        s2.k = 0;
        assert!(s2.effective_l_search(&c).is_err());
        s2.k = 1001;
        assert!(s2.effective_l_search(&c).is_err());
        s2.k = 200;
        assert_eq!(s2.effective_l_search(&c).unwrap(), 200);
        s2.l_search = Some(50);
        assert!(s2.effective_l_search(&c).is_err());
    }

    #[test]
    fn filter_semantics() {
        let f = json!({"lang":"en","n":1}).as_object().unwrap().clone();
        let m1 = json!({"lang":"en","n":1,"x":true})
            .as_object()
            .unwrap()
            .clone();
        let m2 = json!({"lang":"de","n":1}).as_object().unwrap().clone();
        assert!(filter_matches(&f, Some(&m1)));
        assert!(!filter_matches(&f, Some(&m2)));
        assert!(!filter_matches(&f, None));
        assert!(filter_matches(&Map::new(), None));
    }
}
