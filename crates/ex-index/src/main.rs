use clap::Parser;
use ex_index::{Cli, Command};
use tracing_subscriber::EnvFilter;

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => {
            let filter =
                EnvFilter::try_new(&args.log_level).unwrap_or_else(|_| EnvFilter::new("info"));
            let ansi = std::io::IsTerminal::is_terminal(&std::io::stdout());
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(false)
                .with_ansi(ansi)
                .init();
            if args.api_key.as_deref() == Some("") {
                eprintln!("error: --api-key must not be empty");
                return std::process::ExitCode::FAILURE;
            }
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("error: failed to start runtime: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            let res: Result<(), ex_index::BoxError> = rt.block_on(async move {
                let listener =
                    tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
                ex_index::serve(args, listener, ex_index::shutdown_signal()).await
            });
            match res {
                Ok(()) => std::process::ExitCode::SUCCESS,
                Err(e) => {
                    tracing::error!("fatal: {e}");
                    eprintln!("error: {e}");
                    std::process::ExitCode::FAILURE
                }
            }
        }
    }
}
