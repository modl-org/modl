use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::DescribeJobSpec;

pub async fn run(paths: &[String], detail: &str, json: bool) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl describe <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    let spec = DescribeJobSpec {
        image_paths: paths.to_vec(),
        model: "qwen25-vl-3b".to_string(),
        detail: detail.to_string(),
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize describe spec")?;

    if !json {
        println!(
            "{} Describing image(s) (detail: {})...",
            style("->").cyan(),
            detail
        );
    }

    let result = super::analysis::spawn_analysis_worker("describe", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        println!();
        if let Some(descriptions) = data.get("descriptions").and_then(|d| d.as_array()) {
            for entry in descriptions {
                let image = entry.get("image").and_then(|v| v.as_str()).unwrap_or("?");
                let filename = PathBuf::from(image)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let text = entry
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no description)");

                println!("  {}:", filename);
                println!("    {}", text);
                println!();
            }
        }
    }

    if !result.success {
        anyhow::bail!("Describe failed");
    }

    Ok(())
}
