use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::GroundJobSpec;

pub async fn run(
    query: &str,
    paths: &[String],
    threshold: Option<f64>,
    model: Option<&str>,
    json: bool,
) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl ground <query> <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    let model_id = model.unwrap_or("qwen3-vl-2b").to_string();

    let spec = GroundJobSpec {
        image_paths: paths.to_vec(),
        query: query.to_string(),
        model: model_id.clone(),
        threshold,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize ground spec")?;

    if !json {
        println!(
            "{} Finding \"{}\" in image(s) [{}]...",
            style("→").cyan(),
            query,
            model_id
        );
    }

    let result = super::analysis::spawn_analysis_worker("ground", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        println!();
        if let Some(detections) = data.get("detections").and_then(|d| d.as_array()) {
            for det in detections {
                let image = det.get("image").and_then(|v| v.as_str()).unwrap_or("?");
                let filename = PathBuf::from(image)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let objects = det.get("objects").and_then(|o| o.as_array());
                let count = objects.map(|o| o.len()).unwrap_or(0);

                if count == 0 {
                    println!(
                        "  {} {} — no objects matching \"{}\"",
                        style("○").dim(),
                        filename,
                        query
                    );
                } else {
                    println!(
                        "  {} {} — {} object(s) matching \"{}\"",
                        style("●").green(),
                        filename,
                        count,
                        query
                    );
                    if let Some(objs) = objects {
                        for (j, obj) in objs.iter().enumerate() {
                            let label = obj.get("label").and_then(|v| v.as_str()).unwrap_or(query);
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
                            println!("    {} {}: bbox {}", j + 1, label, style(bbox_str).dim());
                        }
                    }
                }
            }
        }

        if let Some(total) = data.get("total_objects").and_then(|v| v.as_u64()) {
            println!();
            println!("  Total: {} object(s)", style(total).bold());
        }
    }

    if !result.success {
        anyhow::bail!("Grounding failed");
    }

    Ok(())
}
