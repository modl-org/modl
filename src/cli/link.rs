use anyhow::Result;

pub async fn run(comfyui: Option<&str>, a1111: Option<&str>) -> Result<()> {
    if let Some(path) = comfyui {
        println!("Linking ComfyUI at '{}'...", path);
    }
    if let Some(path) = a1111 {
        println!("Linking A1111 at '{}'...", path);
    }
    // TODO: Scan model folders
    // TODO: Hash files, match against registry
    // TODO: Register in DB
    // TODO: Set up symlink config
    Ok(())
}
