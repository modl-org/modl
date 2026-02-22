use anyhow::Result;

pub async fn run(id: &str) -> Result<()> {
    println!("Info for '{}':", id);
    // TODO: Look up in registry index
    // TODO: Show variants, deps, VRAM reqs, description
    // TODO: If installed, show location + disk usage
    Ok(())
}
