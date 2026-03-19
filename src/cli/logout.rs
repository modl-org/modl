use anyhow::Result;
use console::style;

use crate::core::config::Config;

pub async fn run() -> Result<()> {
    let mut config = Config::load()?;
    let Some(ref mut cloud) = config.cloud else {
        println!("No cloud login found.");
        return Ok(());
    };

    if cloud.api_key.is_none() {
        println!("No cloud login found.");
        return Ok(());
    }

    cloud.api_key = None;
    cloud.username = None;
    config.save()?;

    println!("{} Logged out", style("✓").green().bold());
    Ok(())
}
