use anyhow::{Context, Result};
use console::style;
use serde::Deserialize;

#[derive(Deserialize)]
struct LockFile {
    #[allow(dead_code)]
    generated: Option<String>,
    #[allow(dead_code)]
    modl_version: Option<String>,
    items: Vec<LockItem>,
}

#[derive(Deserialize)]
struct LockItem {
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    asset_type: String,
    variant: Option<String>,
    #[allow(dead_code)]
    sha256: String,
}

pub async fn run(path: &str) -> Result<()> {
    let contents =
        std::fs::read_to_string(path).with_context(|| format!("Failed to read '{}'", path))?;
    let lock: LockFile = serde_yaml::from_str(&contents).context("Failed to parse lock file")?;

    println!(
        "{} Importing {} models from {}",
        style("→").cyan(),
        style(lock.items.len()).bold(),
        style(path).dim()
    );

    for item in &lock.items {
        let variant_arg = item.variant.as_deref();
        println!();
        match super::install::run(&item.id, variant_arg, false, false).await {
            Ok(()) => {}
            Err(e) => {
                eprintln!(
                    "  {} Failed to install '{}': {}",
                    style("✗").red(),
                    item.id,
                    e
                );
            }
        }
    }

    println!();
    println!("{} Import complete.", style("✓").green().bold());

    Ok(())
}
