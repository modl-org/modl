use anyhow::Result;

pub async fn run(provider: &str) -> Result<()> {
    println!("Configuring auth for '{}'...", provider);
    // TODO: Prompt for token/key
    // TODO: Verify token
    // TODO: Store in ~/.mods/auth.yaml
    Ok(())
}
