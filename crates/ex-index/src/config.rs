//! CLI flags with environment-variable fallbacks (design "Server config").

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "ex-index",
    version,
    about = "Single-binary DiskANN vector search server"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the HTTP server.
    Serve(ServeArgs),
}

#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
    /// Interface to bind.
    #[arg(long, env = "EX_INDEX_HOST", default_value = "0.0.0.0")]
    pub host: String,
    /// TCP port.
    #[arg(long, env = "EX_INDEX_PORT", default_value_t = 7700)]
    pub port: u16,
    /// Directory holding every index (all persisted state lives here).
    #[arg(long, env = "EX_INDEX_DATA_DIR", default_value = "./data")]
    pub data_dir: PathBuf,
    /// Require `Authorization: Bearer <key>` on /v1, /ui and /metrics. Unset = auth off.
    #[arg(long, env = "EX_INDEX_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Rebuild an index in the background once its delta holds this many vectors.
    #[arg(long, env = "EX_INDEX_DELTA_THRESHOLD", default_value_t = 50_000)]
    pub delta_threshold: usize,
    /// Log level / tracing filter (e.g. `info`, `debug`, `ex_index=debug`).
    #[arg(long, env = "EX_INDEX_LOG_LEVEL", default_value = "info")]
    pub log_level: String,
    /// Live-vector count below which search is exact brute force (testing knob).
    #[arg(
        long,
        env = "EX_INDEX_SMALL_INDEX_THRESHOLD",
        default_value_t = 10_000,
        hide = true
    )]
    pub small_index_threshold: usize,
}

impl ServeArgs {
    /// Defaults for embedding the server (tests).
    pub fn for_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        ServeArgs {
            host: "127.0.0.1".into(),
            port: 0,
            data_dir: data_dir.into(),
            api_key: None,
            delta_threshold: 50_000,
            log_level: "info".into(),
            small_index_threshold: 10_000,
        }
    }
}
