use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::VlTagJobSpec;

pub async fn run(
    paths: &[String],
    max_tags: Option<usize>,
    model: Option<&str>,
    json: bool,
) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl vl-tag <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    let model_id = model.unwrap_or("qwen3-vl-2b").to_string();

    let spec = VlTagJobSpec {
        image_paths: paths.to_vec(),
        model: model_id.clone(),
        max_tags,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize vl-tag spec")?;

    if !json {
        println!("{} Tagging image(s) [{}]...", style("→").cyan(), model_id);
    }

    let result = super::analysis::spawn_analysis_worker("vl-tag", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        println!();
        if let Some(results) = data.get("results").and_then(|r| r.as_array()) {
            for entry in results {
                let image = entry.get("image").and_then(|v| v.as_str()).unwrap_or("?");
                let filename = PathBuf::from(image)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let tags = entry.get("tags").and_then(|t| t.as_array());

                if let Some(tags) = tags {
                    let tag_strs: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
                    println!("  {}: {}", style(filename).bold(), tag_strs.join(", "));
                } else {
                    println!("  {}: {}", filename, style("(no tags)").dim());
                }
            }
        }
    }

    if !result.success {
        anyhow::bail!("Tagging failed");
    }

    Ok(())
}
