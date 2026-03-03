mod auth;
mod cli;
mod compat;
mod core;
mod ui;

use anyhow::Result;
use clap::Parser;
use cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Spawn a non-blocking background update check (at most once per 24h).
    // This never blocks the main command — we just read the result at the end.
    let update_handle = core::update_check::spawn_check();

    let result = cli::run(cli).await;

    // Wait briefly for the background check to finish (it's usually instant
    // since most calls hit a fresh cache). Then print a hint if applicable.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), update_handle).await;
    core::update_check::print_if_update_available();

    result
}
