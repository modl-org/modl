use anyhow::Result;
use console::style;

use crate::core::install;
use crate::core::manifest::AssetType;

pub fn run(
    path: &str,
    name: &str,
    id: Option<&str>,
    asset_type: &AssetType,
    sha256: Option<&str>,
) -> Result<()> {
    let config = crate::core::config::Config::load()?;
    let db = crate::core::db::Database::open()?;
    let result = install::register_file(
        std::path::Path::new(path),
        name,
        id,
        asset_type,
        sha256,
        &db,
        &config.store_root(),
    )?;

    if result.already_registered {
        println!(
            "{} {} already registered as {} ({})",
            style("i").dim(),
            name,
            style(&result.id).bold(),
            style(&result.sha256[..16]).dim(),
        );
    } else {
        println!(
            "{} Registered {} as {} ({})",
            style("✓").green().bold(),
            style(name).bold(),
            result.id,
            style(&result.sha256[..16]).dim(),
        );
        println!("  {}", result.store_path.display());
    }
    Ok(())
}
