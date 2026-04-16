use anyhow::Result;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::HumanBytes;

use crate::core::manifest::AssetType;
use crate::core::registry::RegistryIndex;

use super::fmt::format_downloads;

pub async fn run(type_filter: Option<AssetType>, for_model: Option<&str>) -> Result<()> {
    let index = RegistryIndex::load_or_fetch().await?;

    let mut items: Vec<_> = index.items.iter().collect();

    // Apply filters
    if let Some(ref t) = type_filter {
        items.retain(|m| m.asset_type == *t);
    }
    if let Some(base) = for_model {
        items.retain(|m| m.base_models.iter().any(|b| b == base));
    }

    // Sort by downloads (descending)
    items.sort_by_key(|b| std::cmp::Reverse(b.downloads.unwrap_or(0)));

    // Take top 20
    items.truncate(20);

    if items.is_empty() {
        println!("No items found.");
        return Ok(());
    }

    println!("{} Popular models", style("★").yellow());
    println!();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("#").fg(Color::Cyan),
        Cell::new("Name").fg(Color::Cyan),
        Cell::new("Type").fg(Color::Cyan),
        Cell::new("Downloads").fg(Color::Cyan),
        Cell::new("Rating").fg(Color::Cyan),
        Cell::new("Size").fg(Color::Cyan),
        Cell::new("ID").fg(Color::Cyan),
    ]);

    for (i, m) in items.iter().enumerate() {
        let size = if !m.variants.is_empty() {
            let s = m.variants.first().map(|v| v.size).unwrap_or(0);
            HumanBytes(s).to_string()
        } else if let Some(ref f) = m.file {
            HumanBytes(f.size).to_string()
        } else {
            "—".to_string()
        };

        table.add_row(vec![
            Cell::new((i + 1).to_string()),
            Cell::new(&m.name),
            Cell::new(m.asset_type.to_string()),
            Cell::new(
                m.downloads
                    .map(format_downloads)
                    .unwrap_or_else(|| "—".to_string()),
            ),
            Cell::new(
                m.rating
                    .map(|r| format!("{:.1}", r))
                    .unwrap_or_else(|| "—".to_string()),
            ),
            Cell::new(size),
            Cell::new(&m.id).fg(Color::DarkGrey),
        ]);
    }

    println!("{table}");

    Ok(())
}
