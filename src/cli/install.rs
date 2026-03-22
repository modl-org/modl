use anyhow::{Context, Result};
use console::style;
use dialoguer::Select;
use indicatif::HumanBytes;

use crate::auth::AuthStore;
use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::install::{self, resolve_file_info};
use crate::core::manifest::Manifest;
use crate::core::registry::RegistryIndex;
use crate::core::store::Store;

pub async fn run(id: &str, variant: Option<&str>, dry_run: bool, force: bool) -> Result<()> {
    // Handle hf:owner/repo prefix — direct pull from HuggingFace
    if let Some(repo_id) = id.strip_prefix("hf:") {
        return run_hf_pull(repo_id, dry_run, force).await;
    }

    let config = Config::load()?;
    let index = RegistryIndex::load_or_fetch().await?;
    let db = Database::open()?;

    // Resolve family aliases (e.g. "sdxl" → "sdxl-base-1.0") if the ID
    // isn't found directly in the registry.
    let resolved_id = if index.find(id).is_none() {
        if let Some(info) = crate::core::model_family::resolve_model(id) {
            // Try the family ID as a registry lookup (e.g. "flux-dev" works as-is)
            // then try common manifest patterns (e.g. "sdxl" → "sdxl-base-1.0")
            let candidates = [info.id.to_string(), format!("{}-base-1.0", info.id)];
            candidates
                .into_iter()
                .find(|c| index.find(c).is_some())
                .unwrap_or_else(|| id.to_string())
        } else {
            id.to_string()
        }
    } else {
        id.to_string()
    };
    let id = resolved_id.as_str();

    let (mut plan, vram) = install::resolve_plan(id, variant, &index, &db)?;

    // Interactive variant selection for the primary model (if it has multiple variants
    // and the user didn't specify --variant)
    if variant.is_none() {
        for item in &mut plan.items {
            if item.manifest.id == id && !item.already_installed && item.manifest.variants.len() > 1
            {
                let selected = prompt_variant_selection(&item.manifest, vram)?;
                item.variant_id = Some(selected);
            }
        }
    }

    // Display the plan
    println!(
        "{} Install plan for {}:",
        style("→").cyan(),
        style(id).bold()
    );
    println!();

    let mut total_download: u64 = 0;
    for item in &plan.items {
        let effective_variant = if item.manifest.id == id {
            item.variant_id.as_deref().or(variant)
        } else {
            item.variant_id.as_deref()
        };
        let info = resolve_file_info(&item.manifest, effective_variant, vram);
        let status = if item.already_installed {
            style("installed").green().to_string()
        } else {
            total_download += info.size;
            style(HumanBytes(info.size).to_string())
                .yellow()
                .to_string()
        };
        println!(
            "  {} {} {} {}",
            if item.already_installed {
                style("✓").green()
            } else {
                style("↓").cyan()
            },
            style(&item.manifest.name).bold(),
            style(format!("({})", item.manifest.asset_type)).dim(),
            if let Some(ref v) = info.variant_label {
                format!("[{}] {}", v, status)
            } else {
                status
            }
        );
    }

    println!();
    if total_download > 0 {
        println!(
            "  Total download: {}",
            style(HumanBytes(total_download)).bold()
        );
    }

    if dry_run {
        println!();
        println!("{}", style("Dry run — nothing downloaded.").dim());
        return Ok(());
    }

    let items_to_install: Vec<_> = plan.items.iter().filter(|i| !i.already_installed).collect();

    if items_to_install.is_empty() {
        println!("{} Everything is already installed.", style("✓").green());
        return Ok(());
    }

    println!();

    let auth_store = AuthStore::load().unwrap_or_default();
    let store = Store::new(config.store_root());

    // Download each item
    for item in &items_to_install {
        let effective_variant = if item.manifest.id == id {
            item.variant_id.as_deref().or(variant)
        } else {
            item.variant_id.as_deref()
        };

        match install::install_item(
            &item.manifest,
            effective_variant,
            vram,
            &config,
            &store,
            &auth_store,
            &db,
            force,
        )
        .await
        {
            Ok(result) => {
                if result.adopted {
                    println!(
                        "  {} {} (adopted from existing file)",
                        style("✓").green(),
                        style(&result.name).bold()
                    );
                } else {
                    println!("  {} {}", style("✓").green(), style(&result.name).bold());
                }
            }
            Err(e) => {
                // Check for 401 and provide helpful auth message
                let err_msg = format!("{e}");
                if err_msg.contains("401") || err_msg.contains("Unauthorized") {
                    println!();
                    println!(
                        "  {} This model requires authentication.",
                        style("\u{2717}").red()
                    );
                    let provider = item
                        .manifest
                        .auth
                        .as_ref()
                        .map(|a| a.provider.as_str())
                        .unwrap_or("huggingface");
                    println!(
                        "    Run: {}",
                        style(format!("modl auth {}", provider)).cyan()
                    );
                    if let Some(ref auth) = item.manifest.auth
                        && let Some(ref terms) = auth.terms_url
                    {
                        println!("    Accept terms at: {}", style(terms).underlined());
                    }
                    println!();
                }
                return Err(e).with_context(|| format!("Failed to install {}", item.manifest.name));
            }
        }
    }

    println!();
    println!(
        "{} Installed {} successfully.",
        style("✓").green().bold(),
        style(id).bold()
    );

    Ok(())
}

// ── HuggingFace direct pull ─────────────────────────────────────────────

async fn run_hf_pull(repo_id: &str, dry_run: bool, force: bool) -> Result<()> {
    println!(
        "{} Resolving {} on HuggingFace...",
        style("→").cyan(),
        style(format!("hf:{}", repo_id)).bold()
    );

    // For dry run, just show info
    if dry_run {
        let auth_store = AuthStore::load().unwrap_or_default();
        let hf_token = auth_store.token_for("huggingface");
        let model = crate::core::huggingface::get_model(repo_id, hf_token.as_deref()).await?;
        let resolved =
            crate::core::huggingface::resolve_download(repo_id, None, hf_token.as_deref()).await?;
        let asset_type_str = crate::core::huggingface::guess_asset_type(&model, &resolved.filename);
        let display_name = repo_id.split('/').next_back().unwrap_or(repo_id);
        println!();
        println!(
            "  {} {} {}",
            style("↓").cyan(),
            style(display_name).bold(),
            style(format!(
                "({}) [{}]",
                asset_type_str,
                HumanBytes(resolved.size)
            ))
            .dim(),
        );
        println!();
        println!("{}", style("Dry run — nothing downloaded.").dim());
        return Ok(());
    }

    let (result, _final_path) = install::hf_pull(repo_id, None, force).await?;

    if result.already_installed {
        println!();
        println!(
            "{} {} is already installed. Use --force to re-download.",
            style("✓").green(),
            style(&result.display_name).bold()
        );
    } else {
        println!();
        println!(
            "{} Installed {} (hf:{}) successfully.",
            style("✓").green().bold(),
            style(&result.display_name).bold(),
            repo_id,
        );
    }

    Ok(())
}

/// Show an interactive menu for the user to pick a variant
fn prompt_variant_selection(manifest: &Manifest, vram: Option<u64>) -> Result<String> {
    let auto_selected = vram.and_then(|vram_mb| {
        let variant_info: Vec<(String, u64)> = manifest
            .variants
            .iter()
            .map(|v| (v.id.clone(), v.vram_required.unwrap_or(0)))
            .collect();
        crate::core::gpu::select_variant(vram_mb, &variant_info)
    });

    let items: Vec<String> = manifest
        .variants
        .iter()
        .map(|v| {
            let recommended = auto_selected.as_ref().map(|s| s == &v.id).unwrap_or(false);
            let precision = v
                .precision
                .as_ref()
                .map(|p| format!(", {}", p))
                .unwrap_or_default();
            let note = v
                .note
                .as_ref()
                .map(|n| format!(" - {}", n))
                .unwrap_or_default();
            format!(
                "{}  ({}{}){}{}",
                v.id,
                HumanBytes(v.size),
                precision,
                note,
                if recommended {
                    format!("  {}", style("<- recommended for your GPU").dim())
                } else {
                    String::new()
                },
            )
        })
        .collect();

    let default_idx = auto_selected
        .as_ref()
        .and_then(|s| manifest.variants.iter().position(|v| &v.id == s))
        .unwrap_or(0);

    println!(
        "\n  {} {} has multiple variants:",
        style("?").yellow(),
        style(&manifest.name).bold()
    );

    let selection = Select::new()
        .items(&items)
        .default(default_idx)
        .interact()?;

    Ok(manifest.variants[selection].id.clone())
}
