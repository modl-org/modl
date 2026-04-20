use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::{ComposeJobSpec, ComposeLayer};

/// Parse a "0.5,0.7" string into [f64; 2], enforcing 0.0–1.0 range.
fn parse_position(s: &str) -> Result<[f64; 2]> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        anyhow::bail!("Position must be x,y (e.g. 0.5,0.7)");
    }
    let x: f64 = parts[0].trim().parse().context("Invalid x position")?;
    let y: f64 = parts[1].trim().parse().context("Invalid y position")?;
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
        anyhow::bail!(
            "Position values must be between 0.0 and 1.0 (fractional canvas coordinates), got {x},{y}"
        );
    }
    Ok([x, y])
}

/// Parse a "1024x768" or "1024,768" string into [u32; 2].
fn parse_canvas_size(s: &str) -> Result<[u32; 2]> {
    let sep = if s.contains('x') { 'x' } else { ',' };
    let parts: Vec<&str> = s.split(sep).collect();
    if parts.len() != 2 {
        anyhow::bail!("Canvas size must be WxH or W,H (e.g. 1024x768)");
    }
    let w: u32 = parts[0].trim().parse().context("Invalid width")?;
    let h: u32 = parts[1].trim().parse().context("Invalid height")?;
    Ok([w, h])
}

pub struct ComposeArgs<'a> {
    pub background: &'a str,
    pub layers: &'a [String],
    pub positions: &'a [String],
    pub scales: &'a [f64],
    pub opacities: &'a [f64],
    pub canvas_size: Option<&'a str>,
    pub output_dir: Option<&'a str>,
    pub json: bool,
}

pub async fn run(args: ComposeArgs<'_>) -> Result<()> {
    let ComposeArgs {
        background,
        layers,
        positions,
        scales,
        opacities,
        canvas_size,
        output_dir,
        json,
    } = args;
    if layers.is_empty() {
        anyhow::bail!(
            "At least one --layer is required. Usage: modl process compose --background bg.png --layer subject.png"
        );
    }

    // Validate background
    if !["transparent", "white", "black"].contains(&background) {
        let bg_path = PathBuf::from(background);
        if !bg_path.exists() {
            anyhow::bail!("Background image not found: {background}");
        }
    }

    // Validate layer paths
    for (i, layer_path) in layers.iter().enumerate() {
        let p = PathBuf::from(layer_path);
        if !p.exists() {
            anyhow::bail!("Layer {} not found: {}", i + 1, layer_path);
        }
    }

    // Validate positions up front
    for (i, pos) in positions.iter().enumerate() {
        parse_position(pos)
            .with_context(|| format!("Invalid --position for layer {}: {pos:?}", i + 1))?;
    }

    // Validate scale values
    for (i, &s) in scales.iter().enumerate() {
        if s <= 0.0 {
            anyhow::bail!("--scale for layer {} must be > 0, got {s}", i + 1);
        }
    }

    // Validate opacity values
    for (i, &o) in opacities.iter().enumerate() {
        if !(0.0..=1.0).contains(&o) {
            anyhow::bail!(
                "--opacity for layer {} must be between 0.0 and 1.0, got {o}",
                i + 1
            );
        }
    }

    // Warn on parameter count mismatches
    if !positions.is_empty() && positions.len() != layers.len() {
        eprintln!(
            "{} {} --position value(s) for {} layer(s); unmatched layers default to center (0.5,0.5)",
            style("warning:").yellow(),
            positions.len(),
            layers.len(),
        );
    }
    if !scales.is_empty() && scales.len() != layers.len() {
        eprintln!(
            "{} {} --scale value(s) for {} layer(s); unmatched layers default to 1.0",
            style("warning:").yellow(),
            scales.len(),
            layers.len(),
        );
    }
    if !opacities.is_empty() && opacities.len() != layers.len() {
        eprintln!(
            "{} {} --opacity value(s) for {} layer(s); unmatched layers default to 1.0",
            style("warning:").yellow(),
            opacities.len(),
            layers.len(),
        );
    }

    // Build layers with position/scale/opacity
    let compose_layers: Vec<ComposeLayer> = layers
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let position = if i < positions.len() {
                parse_position(&positions[i]).expect("already validated")
            } else {
                [0.5, 0.5]
            };
            let scale = if i < scales.len() { scales[i] } else { 1.0 };
            let opacity = if i < opacities.len() {
                opacities[i]
            } else {
                1.0
            };

            ComposeLayer {
                path: std::fs::canonicalize(path)
                    .unwrap_or_else(|_| PathBuf::from(path))
                    .to_string_lossy()
                    .to_string(),
                position,
                scale,
                opacity,
            }
        })
        .collect();

    // Resolve canvas size
    let canvas_size_vec = canvas_size.map(parse_canvas_size).transpose()?;

    // Resolve background path
    let bg_resolved = if ["transparent", "white", "black"].contains(&background) {
        if canvas_size_vec.is_none() {
            anyhow::bail!(
                "--canvas-size is required when using a solid color background (transparent/white/black)"
            );
        }
        background.to_string()
    } else {
        std::fs::canonicalize(background)
            .unwrap_or_else(|_| PathBuf::from(background))
            .to_string_lossy()
            .to_string()
    };

    let out_dir = output_dir.map(String::from).unwrap_or_else(|| {
        let date = chrono::Local::now().format("%Y-%m-%d");
        crate::core::paths::modl_root()
            .join("outputs")
            .join(date.to_string())
            .to_string_lossy()
            .to_string()
    });

    std::fs::create_dir_all(&out_dir)?;

    let spec = ComposeJobSpec {
        background: bg_resolved,
        layers: compose_layers,
        output_dir: out_dir.clone(),
        canvas_size: canvas_size_vec,
    };

    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize compose spec")?;

    if !json {
        println!(
            "{} Compositing {} layer(s) onto {}...",
            style("→").cyan(),
            layers.len(),
            if ["transparent", "white", "black"].contains(&background) {
                background.to_string()
            } else {
                PathBuf::from(background)
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_else(|| background.to_string())
            }
        );
    }

    let result = super::analysis::spawn_analysis_worker("compose", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data
        && let Some(output) = data.get("output").and_then(|v| v.as_str())
    {
        println!("  Output: {}", style(output).bold());
    }

    if !result.success {
        anyhow::bail!("Composition failed");
    }

    Ok(())
}
