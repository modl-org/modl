use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::job::SegmentJobSpec;

pub async fn run(
    image: &str,
    output: Option<&str>,
    method: &str,
    bbox: Option<&str>,
    point: Option<&str>,
    expand_px: u32,
    json: bool,
) -> Result<()> {
    let image_path = PathBuf::from(image);
    if !image_path.is_file() {
        anyhow::bail!("Image not found: {image}");
    }

    // Parse bbox: "x1,y1,x2,y2"
    let bbox_arr = if let Some(bbox_str) = bbox {
        let parts: Vec<f32> = bbox_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Invalid bbox format. Expected: x1,y1,x2,y2")?;
        if parts.len() != 4 {
            anyhow::bail!("bbox must have 4 values: x1,y1,x2,y2");
        }
        Some([parts[0], parts[1], parts[2], parts[3]])
    } else {
        None
    };

    // Parse point: "x,y"
    let point_arr = if let Some(point_str) = point {
        let parts: Vec<f32> = point_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Invalid point format. Expected: x,y")?;
        if parts.len() != 2 {
            anyhow::bail!("point must have 2 values: x,y");
        }
        Some([parts[0], parts[1]])
    } else {
        None
    };

    // Default output path
    let output_mask = output.map(String::from).unwrap_or_else(|| {
        let stem = image_path.file_stem().unwrap_or_default().to_string_lossy();
        image_path
            .with_file_name(format!("{stem}_mask.png"))
            .to_string_lossy()
            .to_string()
    });

    let spec = SegmentJobSpec {
        image_path: image.to_string(),
        output_mask_path: output_mask.clone(),
        method: method.to_string(),
        bbox: bbox_arr,
        point: point_arr,
        model: "sam-vit-base".to_string(),
        model_path: None,
        expand_px,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize segment spec")?;

    if !json {
        println!(
            "{} Segmenting {} (method: {})",
            style("→").cyan(),
            image_path.file_name().unwrap_or_default().to_string_lossy(),
            method
        );
    }

    let result = super::analysis::spawn_analysis_worker("segment", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if result.success {
        println!("  Mask saved to: {}", style(&output_mask).bold());
    }

    if !result.success {
        anyhow::bail!("Segmentation failed");
    }

    Ok(())
}
