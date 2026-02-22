use anyhow::Result;

pub async fn run(id: &str, variant: Option<&str>, dry_run: bool) -> Result<()> {
    if dry_run {
        println!("Dry run: would install '{}'", id);
    } else {
        println!("Installing '{}'...", id);
    }
    if let Some(v) = variant {
        println!("  variant: {}", v);
    }
    // TODO: Resolve dependencies
    // TODO: Auto-select variant based on GPU VRAM
    // TODO: Check auth requirements
    // TODO: Download with progress bars
    // TODO: Verify SHA256
    // TODO: Store in content-addressed store
    // TODO: Create symlinks
    // TODO: Update SQLite DB
    Ok(())
}
