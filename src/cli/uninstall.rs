use anyhow::Result;

pub async fn run(id: &str, _force: bool) -> Result<()> {
    println!("Uninstalling '{}'...", id);
    // TODO: Check if other items depend on this
    // TODO: Remove symlinks
    // TODO: Mark as uninstalled in DB
    Ok(())
}
