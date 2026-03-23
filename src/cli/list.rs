use anyhow::Result;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::HumanBytes;

use crate::core::db::Database;
use crate::core::manifest::AssetType;

/// Display sections for user-visible grouping
struct Section {
    label: &'static str,
    types: &'static [&'static str],
}

const USER_SECTIONS: &[Section] = &[
    Section {
        label: "Models",
        types: &["checkpoint", "diffusion_model"],
    },
    Section {
        label: "LoRAs",
        types: &["lora"],
    },
    Section {
        label: "ControlNets",
        types: &["controlnet"],
    },
    Section {
        label: "Recipes",
        types: &["recipe"],
    },
];

const INTERNAL_TYPES: &[&str] = &[
    "vae",
    "text_encoder",
    "upscaler",
    "vision_language",
    "analysis",
    "segmentation",
    "ipadapter",
];

pub async fn run(type_filter: Option<AssetType>, show_all: bool) -> Result<()> {
    let db = Database::open()?;
    let filter_str = type_filter.as_ref().map(|t| t.to_string());
    let models = db.list_installed(filter_str.as_deref())?;

    if models.is_empty() {
        if let Some(t) = type_filter {
            println!("No installed models of type '{}'.", t);
        } else {
            println!("No models installed yet.");
            println!(
                "  Run {} to get started.",
                style("modl install flux-dev").cyan()
            );
        }
        return Ok(());
    }

    // If filtering by type, show a flat table (legacy behavior)
    if type_filter.is_some() {
        return print_flat_table(&models);
    }

    let mut total_size: u64 = 0;
    let mut printed_any = false;

    // Print user-visible sections
    for section in USER_SECTIONS {
        let items: Vec<_> = models
            .iter()
            .filter(|m| section.types.contains(&m.asset_type.as_str()))
            .collect();

        if items.is_empty() {
            continue;
        }

        if printed_any {
            println!();
        }
        println!("{}", style(section.label).bold().cyan());

        let mut table = Table::new();
        table.load_preset(UTF8_FULL_CONDENSED);
        table.set_header(vec![
            Cell::new("Name").fg(Color::Cyan),
            Cell::new("Variant").fg(Color::Cyan),
            Cell::new("Size").fg(Color::Cyan),
            Cell::new("ID").fg(Color::DarkGrey),
        ]);

        for model in &items {
            total_size += model.size;
            table.add_row(vec![
                Cell::new(&model.name),
                Cell::new(model.variant.as_deref().unwrap_or("—")),
                Cell::new(HumanBytes(model.size).to_string()),
                Cell::new(&model.id).fg(Color::DarkGrey),
            ]);
        }

        println!("{table}");
        printed_any = true;
    }

    // If --all, also show internal types
    if show_all {
        let internal: Vec<_> = models
            .iter()
            .filter(|m| INTERNAL_TYPES.contains(&m.asset_type.as_str()))
            .collect();

        if !internal.is_empty() {
            if printed_any {
                println!();
            }
            println!("{}", style("Internal Dependencies").bold().dim());

            let mut table = Table::new();
            table.load_preset(UTF8_FULL_CONDENSED);
            table.set_header(vec![
                Cell::new("Name").fg(Color::Cyan),
                Cell::new("Type").fg(Color::Cyan),
                Cell::new("Size").fg(Color::Cyan),
                Cell::new("ID").fg(Color::DarkGrey),
            ]);

            for model in &internal {
                total_size += model.size;
                table.add_row(vec![
                    Cell::new(&model.name),
                    Cell::new(&model.asset_type),
                    Cell::new(HumanBytes(model.size).to_string()),
                    Cell::new(&model.id).fg(Color::DarkGrey),
                ]);
            }

            println!("{table}");
        }
    } else {
        // Count hidden internal items
        let internal_count = models
            .iter()
            .filter(|m| INTERNAL_TYPES.contains(&m.asset_type.as_str()))
            .count();
        let internal_size: u64 = models
            .iter()
            .filter(|m| INTERNAL_TYPES.contains(&m.asset_type.as_str()))
            .map(|m| m.size)
            .sum();
        total_size += internal_size;

        if internal_count > 0 {
            println!();
            println!(
                "  {} internal dependencies hidden ({}) — use {} to show",
                style(internal_count).dim(),
                style(HumanBytes(internal_size)).dim(),
                style("--all").cyan(),
            );
        }
    }

    println!();
    println!("  {} total", style(HumanBytes(total_size)).bold());

    Ok(())
}

fn print_flat_table(models: &[crate::core::db::InstalledModel]) -> Result<()> {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("Name").fg(Color::Cyan),
        Cell::new("Type").fg(Color::Cyan),
        Cell::new("Variant").fg(Color::Cyan),
        Cell::new("Size").fg(Color::Cyan),
        Cell::new("ID").fg(Color::Cyan),
    ]);

    let mut total_size: u64 = 0;

    for model in models {
        total_size += model.size;
        table.add_row(vec![
            Cell::new(&model.name),
            Cell::new(&model.asset_type),
            Cell::new(model.variant.as_deref().unwrap_or("—")),
            Cell::new(HumanBytes(model.size).to_string()),
            Cell::new(&model.id).fg(Color::DarkGrey),
        ]);
    }

    println!("{table}");
    println!();
    println!(
        "  {} models, {} total",
        style(models.len()).bold(),
        style(HumanBytes(total_size)).bold()
    );

    Ok(())
}
