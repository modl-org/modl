use anyhow::Result;

pub async fn run(
    _type_filter: Option<&str>,
    _for_model: Option<&str>,
    _period: &str,
) -> Result<()> {
    println!("Fetching popular models...");
    // TODO: Query registry index for trending items
    // TODO: Render results
    Ok(())
}
