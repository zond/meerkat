use clap::Parser;
use meerkat_tui::{Args, run};
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    // Parse before touching the terminal so `--help` and errors render plainly.
    let args = Args::parse();
    match run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("rkat-tui: {err}");
            ExitCode::from(u8::try_from(err.exit_code()).unwrap_or(1))
        }
    }
}
