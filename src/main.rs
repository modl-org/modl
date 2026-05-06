mod auth;
mod cli;
mod compat;
mod core;
mod ui;

use anyhow::Result;
use clap::Parser;
use cli::Cli;

fn main() {
    // Windows default stack is 1 MB which is too small for our large CLI enum
    // and async state machines. Spawn the real entry point with 8 MB stack.
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .name("modl-main".into())
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime")
                .block_on(async_main())
        })
        .expect("failed to spawn main thread")
        .join()
        .expect("main thread panicked");

    if let Err(e) = result {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();

    // Spawn a non-blocking background update check (at most once per 24h).
    // This never blocks the main command — we just read the result at the end.
    let update_handle = core::update_check::spawn_check();

    // If the runtime is stale (pinned dep versions changed since last install),
    // warn the user once. Suppress for 'upgrade' and 'runtime' commands to
    // avoid noise during the update/reinstall workflow itself.
    if !matches!(
        &cli.command,
        cli::Commands::Upgrade | cli::Commands::Runtime { .. }
    ) {
        warn_if_runtime_stale();
    }

    let result = cli::run(cli).await;

    // Wait briefly for the background check to finish (it's usually instant
    // since most calls hit a fresh cache). Then print a hint if applicable.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), update_handle).await;
    core::update_check::print_if_update_available();

    result
}

fn warn_if_runtime_stale() {
    match core::runtime::stale_runtime_profile() {
        Ok(Some(profile)) => {
            eprintln!(
                "{} {} Runtime dependencies have changed. Run {} to update.",
                console::style("!"),
                console::style("warning:").yellow().bold(),
                console::style("modl runtime install").cyan(),
            );
            let _ = profile;
        }
        Err(e) => {
            // Best-effort — never fail the command because of a marker read.
            let _ = e;
        }
        _ => {}
    }
}
