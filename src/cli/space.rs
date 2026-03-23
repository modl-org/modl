use anyhow::Result;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::HumanBytes;

use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::registry::RegistryIndex;

const USER_TYPES: &[&str] = &[
    "checkpoint",
    "diffusion_model",
    "lora",
    "controlnet",
    "recipe",
];

pub async fn run(show_all: bool) -> Result<()> {
    let config = Config::load()?;
    let db = Database::open()?;
    let models = db.list_installed(None)?;

    if models.is_empty() {
        println!("No models installed. Nothing to report.");
        return Ok(());
    }

    // Try to load registry for dependency info
    let registry = RegistryIndex::load().ok();

    println!("{}", style("Disk usage summary:").bold().cyan());
    println!();

    // Separate user-visible and internal models
    let user_models: Vec<_> = models
        .iter()
        .filter(|m| USER_TYPES.contains(&m.asset_type.as_str()))
        .collect();
    let internal_models: Vec<_> = models
        .iter()
        .filter(|m| !USER_TYPES.contains(&m.asset_type.as_str()))
        .collect();

    // Build a map of installed model sizes by ID for dep accounting
    let installed_sizes: std::collections::HashMap<&str, u64> =
        models.iter().map(|m| (m.id.as_str(), m.size)).collect();

    // Track which deps are shared
    let mut dep_usage: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    // Print user models with their dep sizes
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("Name").fg(Color::Cyan),
        Cell::new("Type").fg(Color::Cyan),
        Cell::new("Own Size").fg(Color::Cyan),
        Cell::new("+ Deps").fg(Color::Cyan),
        Cell::new("Total").fg(Color::Cyan),
    ]);

    for model in &user_models {
        let own_size = model.size;
        let mut dep_size: u64 = 0;

        // Look up deps from registry
        if let Some(ref reg) = registry
            && let Some(manifest) = reg.find(&model.id)
        {
            for dep in &manifest.requires {
                if dep.optional {
                    continue;
                }
                if let Some(&size) = installed_sizes.get(dep.id.as_str()) {
                    dep_size += size;
                    dep_usage
                        .entry(dep.id.clone())
                        .or_default()
                        .push(model.id.clone());
                }
            }
        }

        let total = own_size + dep_size;
        let dep_str = if dep_size > 0 {
            HumanBytes(dep_size).to_string()
        } else {
            "—".to_string()
        };

        table.add_row(vec![
            Cell::new(&model.name),
            Cell::new(&model.asset_type),
            Cell::new(HumanBytes(own_size).to_string()),
            Cell::new(dep_str),
            Cell::new(HumanBytes(total).to_string()),
        ]);
    }

    println!("{table}");

    if show_all && !internal_models.is_empty() {
        println!();
        println!("{}", style("Internal Dependencies:").bold().dim());

        let mut int_table = Table::new();
        int_table.load_preset(UTF8_FULL_CONDENSED);
        int_table.set_header(vec![
            Cell::new("Name").fg(Color::Cyan),
            Cell::new("Type").fg(Color::Cyan),
            Cell::new("Size").fg(Color::Cyan),
            Cell::new("Used By").fg(Color::Cyan),
        ]);

        for model in &internal_models {
            let used_by = dep_usage
                .get(&model.id)
                .map(|ids| ids.join(", "))
                .unwrap_or_default();

            int_table.add_row(vec![
                Cell::new(&model.name),
                Cell::new(&model.asset_type),
                Cell::new(HumanBytes(model.size).to_string()),
                Cell::new(used_by).fg(Color::DarkGrey),
            ]);
        }

        println!("{int_table}");
    }

    println!();

    // Deduplicated total (actual disk usage)
    let db_total: u64 = models.iter().map(|m| m.size).sum();

    // Store directory size (actual disk usage)
    let store_root = config.store_root().join("store");
    let store_size = if store_root.exists() {
        dir_size(&store_root).unwrap_or(0)
    } else {
        0
    };

    // Count shared deps
    let shared_count = dep_usage.values().filter(|v| v.len() > 1).count();
    if shared_count > 0 {
        let shared_size: u64 = dep_usage
            .iter()
            .filter(|(_, users)| users.len() > 1)
            .filter_map(|(id, _)| installed_sizes.get(id.as_str()))
            .sum();
        println!(
            "  Shared dependencies: {} ({} shared across models)",
            style(shared_count).bold(),
            HumanBytes(shared_size)
        );
    }

    println!(
        "  Total (deduplicated): {}",
        style(HumanBytes(db_total)).bold()
    );
    println!("  Store directory:      {}", HumanBytes(store_size));
    println!("  Models installed:     {}", style(models.len()).bold());

    Ok(())
}

fn dir_size(path: &std::path::Path) -> Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                size += dir_size(&entry.path())?;
            } else {
                size += meta.len();
            }
        }
    }
    Ok(size)
}
