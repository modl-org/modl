use anyhow::Result;

pub async fn run() -> Result<()> {
    println!("Exporting to mods.lock...");
    // TODO: Query all installed items from DB
    // TODO: Write mods.lock YAML
    Ok(())
}
