//! `ex-index-core`: an in-house DiskANN implementation (Vamana graph, product
//! quantization and an SSD-friendly on-disk layout) plus the index store that
//! the `ex-index` server exposes over REST.
//!
//! Internally every graph, PQ codebook and disk index works in squared-L2
//! "build space":
//! * cosine vectors are L2-normalized on ingest (`‖x−q‖² = 2 − 2·cos`),
//! * euclidean vectors are used as-is,
//! * dot-product vectors are lifted with the MIPS→L2 transform (one extra
//!   coordinate `sqrt(M² − ‖x‖²)`, queries get `0`).
//!
//! Public scores are only produced at output time (see [`distance::Metric`]).
#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod bitset;
pub mod brute;
pub mod disk;
pub mod distance;
pub mod embedding;
pub mod error;
pub mod harness;
pub mod models;
pub mod pq;
pub mod rng;
pub mod store;
pub mod util;
pub mod vamana;
pub mod wal;

pub use distance::Metric;
pub use error::{Error, Result};
