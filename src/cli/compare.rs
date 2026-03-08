use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::CompareJobSpec;

pub async fn run(paths: &[String], reference: Option<&str>, json: bool) -> Result<()> {
    if paths.is_empty() {
        anyhow::bail!(
            "No image paths provided. Usage: modl compare <image1> <image2> or modl compare <dir>"
        );
    }

    for p in paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }
    if let Some(r) = reference
        && !PathBuf::from(r).exists()
    {
        anyhow::bail!("Reference image not found: {r}");
    }

    let spec = CompareJobSpec {
        image_paths: paths.to_vec(),
        reference_path: reference.map(String::from),
        model: "clip-vit-large-patch14".to_string(),
        clip_model_path: None,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize compare spec")?;

    if !json {
        println!("{} Comparing image(s)...", style("→").cyan());
    }

    let result = super::analysis::spawn_analysis_worker("compare", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        let mode = data
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        println!();

        match mode {
            "pairwise" => {
                if let Some(sim) = data.get("similarity").and_then(|v| v.as_f64()) {
                    let images = data.get("images").and_then(|v| v.as_array());
                    let names: Vec<&str> = images
                        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
                        .unwrap_or_default();
                    println!(
                        "  {} vs {} — similarity: {}",
                        names.first().unwrap_or(&"?"),
                        names.get(1).unwrap_or(&"?"),
                        style(format!("{sim:.4}")).bold()
                    );
                }
            }
            "reference" => {
                let ref_name = data
                    .get("reference")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                println!("  Reference: {}", style(ref_name).bold());
                println!();
                if let (Some(images), Some(sims)) = (
                    data.get("images").and_then(|v| v.as_array()),
                    data.get("similarities").and_then(|v| v.as_array()),
                ) {
                    for (name, sim) in images.iter().zip(sims.iter()) {
                        let name = name.as_str().unwrap_or("?");
                        if let Some(s) = sim.as_f64() {
                            let color = if s >= 0.8 {
                                console::Color::Green
                            } else if s >= 0.6 {
                                console::Color::Yellow
                            } else {
                                console::Color::Red
                            };
                            println!("  {:<40} {}", name, style(format!("{s:.4}")).fg(color));
                        }
                    }
                }
            }
            "matrix" => {
                if let Some(images) = data.get("images").and_then(|v| v.as_array()) {
                    let names: Vec<&str> = images.iter().filter_map(|v| v.as_str()).collect();
                    let _n = names.len();

                    // Print header
                    print!("  {:<20}", "");
                    for name in &names {
                        print!("{:>10}", &name[..name.len().min(9)]);
                    }
                    println!();

                    if let Some(matrix) = data.get("similarities").and_then(|v| v.as_array()) {
                        for (i, row) in matrix.iter().enumerate() {
                            print!("  {:<20}", &names[i][..names[i].len().min(19)]);
                            if let Some(cols) = row.as_array() {
                                for (j, val) in cols.iter().enumerate() {
                                    if let Some(s) = val.as_f64() {
                                        if i == j {
                                            print!("{:>10}", style(format!("{s:.2}")).dim());
                                        } else {
                                            let color = if s >= 0.8 {
                                                console::Color::Green
                                            } else if s >= 0.6 {
                                                console::Color::Yellow
                                            } else {
                                                console::Color::Red
                                            };
                                            print!("{:>10}", style(format!("{s:.3}")).fg(color));
                                        }
                                    }
                                }
                            }
                            println!();
                        }
                    }
                }
            }
            _ => {}
        }

        if let Some(mean) = data.get("mean_similarity").and_then(|v| v.as_f64()) {
            println!();
            println!("  Mean similarity: {}", style(format!("{mean:.4}")).bold());
        }
    }

    if !result.success {
        anyhow::bail!("Comparison failed");
    }

    Ok(())
}
