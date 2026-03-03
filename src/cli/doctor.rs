use anyhow::Result;
use console::style;

use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::store::Store;
use crate::core::symlink;

pub async fn run(verify_hashes: bool) -> Result<()> {
    println!(
        "{}",
        style("modl doctor — running diagnostics").bold().cyan()
    );
    println!();

    let config = Config::load()?;
    let db = Database::open()?;
    let models = db.list_installed(None)?;
    let mut issues = 0;

    // 1. Check for broken symlinks in target directories
    println!("{} Checking symlinks...", style("→").cyan());
    for target in &config.targets {
        let broken = symlink::find_broken(&target.path)?;
        if broken.is_empty() {
            println!(
                "  {} {} — all symlinks valid",
                style("✓").green(),
                target.path.display()
            );
        } else {
            for b in &broken {
                println!("  {} Broken symlink: {}", style("✗").red(), b.display());
                issues += 1;
            }
        }
    }

    // 2. Verify store files exist with correct size
    println!();
    println!("{} Checking store files...", style("→").cyan());
    let mut store_ok = true;
    for m in &models {
        let path = std::path::Path::new(&m.store_path);
        if !path.exists() {
            println!(
                "  {} Missing store file for '{}': {}",
                style("✗").red(),
                m.name,
                m.store_path
            );
            issues += 1;
            store_ok = false;
        } else if let Ok(meta) = std::fs::metadata(path)
            && meta.len() != m.size
        {
            println!(
                "  {} Size mismatch for '{}' — expected {} bytes, got {}",
                style("✗").red(),
                m.name,
                m.size,
                meta.len()
            );
            println!(
                "    Fix: {}",
                style(format!("modl uninstall {} && modl install {}", m.id, m.id)).cyan()
            );
            issues += 1;
            store_ok = false;
        }
    }
    if store_ok {
        println!(
            "  {} All store files present and sizes match",
            style("✓").green()
        );
    }

    // 3. Verify hashes (opt-in, slow for large files)
    if verify_hashes {
        println!();
        println!("{} Verifying file hashes...", style("→").cyan());
        for m in &models {
            let path = std::path::Path::new(&m.store_path);
            if path.exists() {
                match Store::verify_hash(path, &m.sha256) {
                    Ok(true) => {}
                    Ok(false) => {
                        println!(
                            "  {} Hash mismatch for '{}' — file may be corrupted",
                            style("✗").red(),
                            m.name
                        );
                        println!(
                            "    Fix: {}",
                            style(format!("modl uninstall {} && modl install {}", m.id, m.id))
                                .cyan()
                        );
                        issues += 1;
                    }
                    Err(e) => {
                        println!(
                            "  {} Could not verify '{}': {}",
                            style("!").yellow(),
                            m.name,
                            e
                        );
                    }
                }
            }
        }
        if issues == 0 {
            println!("  {} All hashes verified", style("✓").green());
        }
    } else {
        println!();
        println!(
            "  {} Hash verification skipped (use {} for full check)",
            style("i").dim(),
            style("--verify-hashes").cyan()
        );
    }

    // Summary
    println!();
    if issues == 0 {
        println!(
            "{} No issues found. Everything looks good!",
            style("✓").green().bold()
        );
    } else {
        println!(
            "{} Found {} issue{}.",
            style("!").yellow().bold(),
            issues,
            if issues == 1 { "" } else { "s" }
        );
    }

    Ok(())
}
