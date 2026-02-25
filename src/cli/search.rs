use anyhow::Result;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::HumanBytes;

use crate::core::manifest::AssetType;
use crate::core::registry::RegistryIndex;

pub async fn run(
    query: &str,
    type_filter: Option<&str>,
    for_model: Option<&str>,
    tag: Option<&str>,
    min_rating: Option<f32>,
) -> Result<()> {
    let index = RegistryIndex::load_or_fetch().await?;

    let mut results = index.search(query);

    // Apply filters
    if let Some(t) = type_filter {
        let parsed_type = parse_asset_type(t);
        results.retain(|m| {
            if let Some(ref at) = parsed_type {
                m.asset_type == *at
            } else {
                m.asset_type.to_string() == t
            }
        });
    }

    if let Some(base) = for_model {
        results.retain(|m| m.base_models.iter().any(|b| b == base));
    }

    if let Some(t) = tag {
        let t_lower = t.to_lowercase();
        results.retain(|m| m.tags.iter().any(|tag| tag.to_lowercase() == t_lower));
    }

    if let Some(min) = min_rating {
        results.retain(|m| m.rating.unwrap_or(0.0) >= min);
    }

    if results.is_empty() {
        println!("No results for '{}'.", query);
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("Name").fg(Color::Cyan),
        Cell::new("Type").fg(Color::Cyan),
        Cell::new("Size").fg(Color::Cyan),
        Cell::new("Tags").fg(Color::Cyan),
        Cell::new("ID").fg(Color::Cyan),
    ]);

    for m in &results {
        let size = if !m.variants.is_empty() {
            // Show range if multiple variants
            let sizes: Vec<u64> = m.variants.iter().map(|v| v.size).collect();
            let min_s = sizes.iter().min().unwrap_or(&0);
            let max_s = sizes.iter().max().unwrap_or(&0);
            if min_s == max_s {
                HumanBytes(*min_s).to_string()
            } else {
                format!("{} \u{2013} {}", HumanBytes(*min_s), HumanBytes(*max_s))
            }
        } else if let Some(ref f) = m.file {
            HumanBytes(f.size).to_string()
        } else {
            "\u{2014}".to_string()
        };

        // Build name with variants on a second line
        let name_display = if m.variants.len() > 1 {
            let variant_list = m
                .variants
                .iter()
                .map(|v| v.id.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
            format!("{}\n  {}", m.name, style(variant_list).dim())
        } else {
            m.name.clone()
        };

        let tags = m
            .tags
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");

        table.add_row(vec![
            Cell::new(&name_display),
            Cell::new(m.asset_type.to_string()),
            Cell::new(size),
            Cell::new(tags),
            Cell::new(&m.id).fg(Color::DarkGrey),
        ]);
    }

    println!("{table}");
    println!("\n  {} results", style(results.len()).bold());

    Ok(())
}

fn parse_asset_type(s: &str) -> Option<AssetType> {
    match s.to_lowercase().as_str() {
        "checkpoint" => Some(AssetType::Checkpoint),
        "lora" => Some(AssetType::Lora),
        "vae" => Some(AssetType::Vae),
        "text_encoder" | "textencoder" => Some(AssetType::TextEncoder),
        "controlnet" => Some(AssetType::Controlnet),
        "upscaler" => Some(AssetType::Upscaler),
        "embedding" => Some(AssetType::Embedding),
        "ipadapter" => Some(AssetType::Ipadapter),
        _ => None,
    }
}
