use anyhow::Result;
use console::style;
use serde::Serialize;

use crate::core::db::Database;

#[derive(Serialize)]
struct LockFile {
    generated: String,
    modl_version: String,
    items: Vec<LockItem>,
}

#[derive(Serialize)]
struct LockItem {
    id: String,
    #[serde(rename = "type")]
    asset_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    variant: Option<String>,
    sha256: String,
}

pub async fn run() -> Result<()> {
    let db = Database::open()?;
    let models = db.list_installed(None)?;

    if models.is_empty() {
        println!("No models installed. Nothing to export.");
        return Ok(());
    }

    let lock = LockFile {
        generated: chrono::Utc::now().to_rfc3339(),
        modl_version: env!("CARGO_PKG_VERSION").to_string(),
        items: models
            .iter()
            .map(|m| LockItem {
                id: m.id.clone(),
                asset_type: m.asset_type.clone(),
                variant: m.variant.clone(),
                sha256: m.sha256.clone(),
            })
            .collect(),
    };

    let yaml = serde_yaml::to_string(&lock)?;
    std::fs::write("modl.lock", &yaml)?;

    println!(
        "{} Exported {} models to {}",
        style("✓").green(),
        style(lock.items.len()).bold(),
        style("modl.lock").cyan()
    );

    Ok(())
}
