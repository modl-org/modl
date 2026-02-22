use anyhow::Result;

pub async fn run() -> Result<()> {
    println!("Running diagnostics...");
    // TODO: Check for broken symlinks
    // TODO: Verify hashes
    // TODO: Check LoRA/base model compat
    // TODO: Check missing deps
    Ok(())
}
