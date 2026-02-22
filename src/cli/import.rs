use anyhow::Result;

pub async fn run(path: &str) -> Result<()> {
    println!("Importing from '{}'...", path);
    // TODO: Parse mods.lock
    // TODO: Install all items
    Ok(())
}
