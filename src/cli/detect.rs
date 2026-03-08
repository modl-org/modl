use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::DetectJobSpec;

pub async fn run(paths: &[String], detect_type: &str, embeddings: bool, json: bool) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!("No image paths provided. Usage: modl detect <image_or_dir> [...]");
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    let spec = DetectJobSpec {
        image_paths: paths.to_vec(),
        detect_type: detect_type.to_string(),
        model: "insightface-buffalo-l".to_string(),
        model_path: None,
        return_embeddings: embeddings,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize detect spec")?;

    if !json {
        println!(
            "{} Detecting {}s in image(s)...",
            style("→").cyan(),
            detect_type
        );
    }

    let result = super::analysis::spawn_analysis_worker("detect", &yaml, json).await?;

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
                let face_count = det.get("face_count").and_then(|v| v.as_u64()).unwrap_or(0);

                if face_count == 0 {
                    println!("  {} {} — no faces", style("○").dim(), filename);
                } else {
                    println!(
                        "  {} {} — {} face(s)",
                        style("●").green(),
                        filename,
                        face_count
                    );
                    if let Some(faces) = det.get("faces").and_then(|f| f.as_array()) {
                        for (j, face) in faces.iter().enumerate() {
                            let conf = face
                                .get("confidence")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            let bbox = face.get("bbox").and_then(|v| v.as_array());
                            let bbox_str = if let Some(b) = bbox {
                                let vals: Vec<String> = b
                                    .iter()
                                    .filter_map(|v| v.as_f64().map(|f| format!("{:.0}", f)))
                                    .collect();
                                vals.join(", ")
                            } else {
                                "?".to_string()
                            };
                            let conf_color = if conf >= 0.8 {
                                console::Color::Green
                            } else if conf >= 0.5 {
                                console::Color::Yellow
                            } else {
                                console::Color::Red
                            };
                            println!(
                                "    face {}: confidence {} bbox [{}]",
                                j + 1,
                                style(format!("{conf:.2}")).fg(conf_color),
                                style(bbox_str).dim()
                            );
                        }
                    }
                }
            }
        }

        if let Some(total) = data.get("total_faces").and_then(|v| v.as_u64()) {
            println!();
            println!("  Total faces: {}", style(total).bold());
        }
    }

    if !result.success {
        anyhow::bail!("Detection failed");
    }

    Ok(())
}
