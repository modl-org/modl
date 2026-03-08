use anyhow::Result;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::HumanBytes;

use crate::auth::AuthStore;
use crate::core::huggingface;
use crate::core::manifest::AssetType;
use crate::core::registry::RegistryIndex;

use super::fmt::format_downloads;

pub async fn run(
    query: &str,
    type_filter: Option<AssetType>,
    for_model: Option<&str>,
    tag: Option<&str>,
    min_rating: Option<f32>,
) -> Result<()> {
    let index = RegistryIndex::load_or_fetch().await?;

    let mut results = index.search(query);

    // Apply filters
    if let Some(ref t) = type_filter {
        results.retain(|m| m.asset_type == *t);
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

    // ── Registry results ────────────────────────────────────────────────

    if !results.is_empty() {
        println!(
            "\n{} {} registry results for '{}':\n",
            style("→").cyan(),
            results.len(),
            query
        );

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
    }

    // ── HuggingFace results ─────────────────────────────────────────────
    // Skip HF search if user used registry-only filters
    let skip_hf = for_model.is_some() || min_rating.is_some();

    if !skip_hf {
        let auth_store = AuthStore::load().unwrap_or_default();
        let hf_token = auth_store.token_for("huggingface");

        match huggingface::search(query, hf_token.as_deref(), 10).await {
            Ok(hf_results) if !hf_results.is_empty() => {
                println!("\n{} HuggingFace results:\n", style("→").cyan(),);

                let mut hf_table = Table::new();
                hf_table.load_preset(UTF8_FULL_CONDENSED);
                hf_table.set_header(vec![
                    Cell::new("Model").fg(Color::Cyan),
                    Cell::new("Downloads").fg(Color::Cyan),
                    Cell::new("Pipeline").fg(Color::Cyan),
                    Cell::new("Pull command").fg(Color::Cyan),
                ]);

                for m in &hf_results {
                    let downloads = m
                        .downloads
                        .map(format_downloads)
                        .unwrap_or_else(|| "\u{2014}".to_string());

                    let pipeline = m.pipeline_tag.as_deref().unwrap_or("\u{2014}").to_string();

                    hf_table.add_row(vec![
                        Cell::new(&m.id),
                        Cell::new(&downloads),
                        Cell::new(&pipeline),
                        Cell::new(format!("modl pull hf:{}", m.id)).fg(Color::DarkGrey),
                    ]);
                }

                println!("{hf_table}");
            }
            Ok(_) => {} // No HF results, just skip
            Err(e) => {
                // Don't fail the whole search if HF is unreachable
                if results.is_empty() {
                    println!("No results for '{}'.", query);
                    eprintln!("  {} HuggingFace search failed: {}", style("!").yellow(), e);
                }
            }
        }
    }

    if results.is_empty() {
        // Already printed HF results above if any, so only print "no results"
        // if we haven't printed anything at all
    }

    Ok(())
}
