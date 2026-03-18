use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::db::Database;
use crate::core::job::UpscaleJobSpec;

/// Resolve the store path for an installed upscaler model.
fn resolve_upscaler_path(model_id: &str, db: &Database) -> Option<String> {
    let installed = db.list_installed(None).ok()?;
    for model in &installed {
        if (model.id == model_id || model.name == model_id) && model.asset_type == "upscaler" {
            return Some(model.store_path.clone());
        }
    }
    None
}

pub async fn run(
    paths: &[String],
    output_dir: Option<&str>,
    scale: u32,
    model: &str,
    json: bool,
) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl upscale <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    // Resolve model path from modl store
    let db = Database::open()?;
    let model_path = resolve_upscaler_path(model, &db);
    if model_path.is_none() {
        anyhow::bail!(
            "Upscaler model not installed: {model}. Run `modl pull {model}` first.\n\
             Available upscalers: modl ls --type upscaler"
        );
    }

    let out_dir = output_dir.map(String::from).unwrap_or_else(|| {
        let date = chrono::Local::now().format("%Y-%m-%d");
        crate::core::paths::modl_root()
            .join("outputs")
            .join(date.to_string())
            .to_string_lossy()
            .to_string()
    });

    std::fs::create_dir_all(&out_dir)?;

    let spec = UpscaleJobSpec {
        image_paths: paths.to_vec(),
        output_dir: out_dir.clone(),
        scale,
        model_path,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize upscale spec")?;

    if !json {
        println!(
            "{} Upscaling {} image(s) ({}x, model: {})...",
            style("→").cyan(),
            paths.len(),
            scale,
            model
        );
    }

    let result = super::analysis::spawn_analysis_worker("upscale", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        let upscaled = data.get("upscaled").and_then(|v| v.as_u64()).unwrap_or(0);
        let errors = data.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
        if upscaled > 0 {
            println!("  Output: {}", style(&out_dir).bold());
        }
        if errors > 0 {
            println!("  {} {errors} image(s) failed", style("⚠").yellow());
        }
    }

    if !result.success {
        anyhow::bail!(
            "Upscaling failed. Check that the Python runtime has spandrel installed:\n  \
             modl train setup"
        );
    }

    Ok(())
}
