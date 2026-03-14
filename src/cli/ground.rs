use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::GroundJobSpec;

pub async fn run(query: &str, paths: &[String], threshold: Option<f64>, json: bool) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl ground <query> <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    let spec = GroundJobSpec {
        image_paths: paths.to_vec(),
        query: query.to_string(),
        model: "qwen25-vl-3b".to_string(),
        threshold,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize ground spec")?;

    if !json {
        println!(
            "{} Finding \"{}\" in image(s)...",
            style("->").cyan(),
            query
        );
    }

    let result = super::analysis::spawn_analysis_worker("ground", &yaml, json).await?;

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
                let objects = entry.get("objects").and_then(|o| o.as_array());
                let count = objects.map(|o| o.len()).unwrap_or(0);

                if count == 0 {
                    println!(
                        "  {} {} -- no objects matching \"{}\"",
                        style("o").dim(),
                        filename,
                        query
                    );
                } else {
                    println!(
                        "  {} {} -- {} object(s) matching \"{}\"",
                        style("*").green(),
                        filename,
                        count,
                        query
                    );
                    if let Some(objs) = objects {
                        for (j, obj) in objs.iter().enumerate() {
                            let bbox = obj.get("bbox").and_then(|v| v.as_array());
                            let bbox_str = if let Some(b) = bbox {
                                let vals: Vec<String> = b
                                    .iter()
                                    .filter_map(|v| v.as_f64().map(|f| format!("{:.0}", f)))
                                    .collect();
                                format!("[{}]", vals.join(", "))
                            } else {
                                "?".to_string()
                            };
                            println!("    object {}: bbox {}", j + 1, style(bbox_str).dim());
                        }
                    }
                }
            }
        }
    }

    if !result.success {
        anyhow::bail!("Grounding failed");
    }

    Ok(())
}
