use anyhow::Result;

pub async fn run(type_filter: Option<&str>) -> Result<()> {
    if let Some(t) = type_filter {
        println!("Listing installed models (type: {})...", t);
    } else {
        println!("Listing all installed models...");
    }
    // TODO: Query SQLite DB
    // TODO: Render table with comfy-table
    Ok(())
}
