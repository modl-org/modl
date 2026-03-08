use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::FaceRestoreJobSpec;

pub async fn run(
    paths: &[String],
    output_dir: Option<&str>,
    fidelity: f32,
    json: bool,
) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl face-restore <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
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

    let spec = FaceRestoreJobSpec {
        image_paths: paths.to_vec(),
        output_dir: out_dir.clone(),
        model: "codeformer".to_string(),
        model_path: None,
        fidelity,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize face-restore spec")?;

    if !json {
        println!(
            "{} Restoring faces (fidelity: {:.1})...",
            style("→").cyan(),
            fidelity
        );
    }

    let result = super::analysis::spawn_analysis_worker("face-restore", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        let restored = data.get("restored").and_then(|v| v.as_u64()).unwrap_or(0);
        let errors = data.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
        if restored > 0 {
            println!("  Output: {}", style(&out_dir).bold());
        }
        if errors > 0 {
            println!("  {} {errors} image(s) failed", style("⚠").yellow());
        }
    }

    if !result.success {
        anyhow::bail!("Face restoration failed");
    }

    Ok(())
}
