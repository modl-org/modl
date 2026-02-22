mod auth;
mod cli;
mod compat;
mod core;

use anyhow::Result;
use clap::Parser;
use cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    cli::run(cli).await
}
