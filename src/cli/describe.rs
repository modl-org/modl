use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::DescribeJobSpec;

pub async fn run(paths: &[String], detail: &str, model: Option<&str>, json: bool) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl describe <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    let model_id = model.unwrap_or("qwen3-vl-8b").to_string();

    let spec = DescribeJobSpec {
        image_paths: paths.to_vec(),
        model: model_id.clone(),
        detail: detail.to_string(),
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize describe spec")?;

    if !json {
        println!(
            "{} Describing image(s) [{}, {}]...",
            style("→").cyan(),
            model_id,
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
                let caption = entry
                    .get("caption")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no description)");

                println!("  {}:", style(filename).bold());
                // Wrap long captions at ~80 chars
                for line in textwrap(caption, 76) {
                    println!("    {}", line);
                }
                println!();
            }
        }
    }

    if !result.success {
        anyhow::bail!("Describe failed");
    }

    Ok(())
}

/// Simple word-wrap for terminal display.
fn textwrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in s.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if line.len() + word.len() + 1 > width && !line.is_empty() {
                lines.push(line);
                line = String::new();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    lines
}
