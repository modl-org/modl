use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::db::Database;
use crate::core::job::RemoveBgJobSpec;

/// Resolve the store path for an installed segmentation model.
fn resolve_segmentation_model_path(model_id: &str, db: &Database) -> Option<String> {
    let installed = db.list_installed(None).ok()?;
    for model in &installed {
        if (model.id == model_id || model.name == model_id) && model.asset_type == "segmentation" {
            return Some(model.store_path.clone());
        }
    }
    None
}

pub async fn run(paths: &[String], output_dir: Option<&str>, json: bool) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl remove-bg <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    // Resolve BiRefNet model path from modl store
    let db = Database::open()?;
    let model_path = resolve_segmentation_model_path("birefnet-dis", &db);
    if model_path.is_none() {
        anyhow::bail!("BiRefNet model not installed. Run `modl pull birefnet-dis` first.");
    }

    let out_dir = output_dir.map(String::from).unwrap_or_else(|| {
        let date = chrono::Local::now().format("%Y-%m-%d");
        dirs::home_dir()
            .expect("home dir")
            .join(".modl")
            .join("outputs")
            .join(date.to_string())
            .to_string_lossy()
            .to_string()
    });

    std::fs::create_dir_all(&out_dir)?;

    let spec = RemoveBgJobSpec {
        image_paths: paths.to_vec(),
        output_dir: out_dir.clone(),
        model_path,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize remove-bg spec")?;

    if !json {
        println!(
            "{} Removing background from {} image(s)...",
            style("→").cyan(),
            paths.len()
        );
    }

    let result = super::analysis::spawn_analysis_worker("remove-bg", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        let processed = data.get("processed").and_then(|v| v.as_u64()).unwrap_or(0);
        let errors = data.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
        if processed > 0 {
            println!("  Output: {}", style(&out_dir).bold());
        }
        if errors > 0 {
            println!("  {} {errors} image(s) failed", style("⚠").yellow());
        }
    }

    if !result.success {
        anyhow::bail!("Background removal failed");
    }

    Ok(())
}
