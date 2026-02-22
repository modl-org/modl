use anyhow::Result;

pub async fn run() -> Result<()> {
    println!("Garbage collecting unreferenced files...");
    // TODO: Find store files not referenced by any installed entry
    // TODO: Show space to recover
    // TODO: Require confirmation
    Ok(())
}
