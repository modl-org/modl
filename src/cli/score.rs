use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::ScoreJobSpec;

pub async fn run(paths: &[String], json: bool) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl score <image_or_dir> [...]");
    }

    // Verify paths exist
    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    let spec = ScoreJobSpec {
        image_paths: paths.to_vec(),
        model: "laion-aesthetic-v2".to_string(),
        clip_model_path: None,
        predictor_path: None,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize score spec")?;

    if !json {
        println!("{} Scoring image(s)...", style("→").cyan());
    }

    let result = super::analysis::spawn_analysis_worker("score", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        // Print table of scores
        println!();
        if let Some(scores) = data.get("scores").and_then(|s| s.as_array()) {
            println!("  {:<50} {}", style("Image").bold(), style("Score").bold());
            println!(
                "  {}",
                style("──────────────────────────────────────────────────────────").dim()
            );
            for entry in scores {
                let image = entry.get("image").and_then(|v| v.as_str()).unwrap_or("?");
                let filename = PathBuf::from(image)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                if let Some(score) = entry.get("score").and_then(|v| v.as_f64()) {
                    let color = if score >= 6.0 {
                        console::Color::Green
                    } else if score >= 4.5 {
                        console::Color::Yellow
                    } else {
                        console::Color::Red
                    };
                    println!(
                        "  {:<50} {}",
                        filename,
                        style(format!("{score:.2}")).fg(color).bold()
                    );
                } else {
                    println!("  {:<50} {}", filename, style("error").red());
                }
            }
        }

        if let Some(mean) = data.get("mean_score").and_then(|v| v.as_f64()) {
            println!();
            println!("  Mean score: {}", style(format!("{mean:.2}")).bold());
        }
    }

    if !result.success {
        anyhow::bail!("Scoring failed");
    }

    Ok(())
}
