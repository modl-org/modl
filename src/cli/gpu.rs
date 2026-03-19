use anyhow::{Result, bail};
use console::style;

pub async fn attach(spec: &str, idle: &str) -> Result<()> {
    println!(
        "\n  {} GPU sessions are not yet available.\n",
        style("!").yellow()
    );
    println!(
        "  Requested: {} GPU with {} idle timeout",
        style(spec).bold(),
        idle
    );
    println!("  This feature is coming soon — see `modl gpu --help` for the planned commands.");
    println!();
    bail!("modl gpu attach is not yet implemented");
}

pub async fn detach() -> Result<()> {
    bail!("modl gpu detach is not yet implemented — no active GPU session");
}

pub async fn status() -> Result<()> {
    println!("No active GPU sessions.");
    Ok(())
}

pub async fn ssh() -> Result<()> {
    bail!("modl gpu ssh is not yet implemented — no active GPU session");
}
