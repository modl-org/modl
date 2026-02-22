use anyhow::Result;

pub async fn run() -> Result<()> {
    println!("Fetching latest registry index...");
    // TODO: Download index.json from modshq/mods-registry releases
    // TODO: Show if installed items have newer versions
    Ok(())
}
