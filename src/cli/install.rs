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

    // Try exact match first, then fuzzy-resolve the ID if not found
    let resolved_id = if index.find(id).is_some() {
        id.to_string()
    } else {
        try_resolve_alias(id, &index)
    };

    let (mut plan, vram) = install::resolve_plan(&resolved_id, variant, &index, &db)?;

    // Show alias resolution if we resolved to a different ID
    if resolved_id != id {
        println!(
            "{} {} → {}",
            style("→").cyan(),
            style(id).dim(),
            style(&resolved_id).bold()
        );
    }

    // Interactive variant selection for the primary model (if it has multiple variants
    // and the user didn't specify --variant)
    if variant.is_none() {
        for item in &mut plan.items {
            if item.manifest.id == resolved_id
                && !item.already_installed
                && item.manifest.variants.len() > 1
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
        style(&resolved_id).bold()
    );
    println!();

    let mut total_download: u64 = 0;
    for item in &plan.items {
        let effective_variant = if item.manifest.id == resolved_id {
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
        let effective_variant = if item.manifest.id == resolved_id {
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
        style(&resolved_id).bold()
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
    let is_mps = crate::core::gpu::detect()
        .map(|g| g.device == crate::core::gpu::DeviceType::Mps)
        .unwrap_or(false);

    // Filter out MPS-incompatible variants (fp8 needs CUDA float8 dtype).
    // GGUF is fine on MPS: weights are dequantized to float16/bfloat16 at load time.
    let visible_variants: Vec<_> = manifest
        .variants
        .iter()
        .filter(|v| {
            if is_mps {
                let id = v.id.to_lowercase();
                let prec = v.precision.as_deref().unwrap_or("").to_lowercase();
                !id.contains("fp8") && !prec.contains("fp8")
            } else {
                true
            }
        })
        .collect();

    let variants_for_selection = if visible_variants.is_empty() {
        // Fallback: show all if filtering removed everything
        manifest.variants.iter().collect::<Vec<_>>()
    } else {
        visible_variants
    };

    let auto_selected = vram.and_then(|vram_mb| {
        let variant_info: Vec<(String, u64)> = variants_for_selection
            .iter()
            .map(|v| (v.id.clone(), v.vram_required.unwrap_or(0)))
            .collect();
        crate::core::gpu::select_variant(vram_mb, &variant_info)
    });

    let items: Vec<String> = variants_for_selection
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
        .and_then(|s| variants_for_selection.iter().position(|v| &v.id == s))
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

    Ok(variants_for_selection[selection].id.clone())
}

/// Try to resolve a user-typed ID to a registry ID.
///
/// Handles common aliases: "sdxl" → "sdxl-base-1.0", "klein" → "flux2-klein-4b", etc.
/// Falls back to substring search on the registry index.
/// Returns the original ID if no match is found (the resolver will then produce
/// a nice error with suggestions).
fn try_resolve_alias(id: &str, index: &RegistryIndex) -> String {
    let lower = id.to_lowercase();

    // 1. Try single-result substring match on registry IDs
    //    e.g. "sdxl" matches "sdxl-base-1.0", "klein-4b" matches "flux2-klein-4b"
    let matches: Vec<_> = index
        .items
        .iter()
        .filter(|m| {
            let mid = m.id.to_lowercase();
            mid.contains(&lower) || lower.contains(&mid)
        })
        .collect();

    if matches.len() == 1 {
        return matches[0].id.clone();
    }

    // 2. Try model_family fuzzy resolve → map family ID to registry ID
    if let Some(info) = crate::core::model_family::resolve_model(id) {
        // The family ID (e.g. "sdxl") may differ from registry ID (e.g. "sdxl-base-1.0")
        // Try exact match first
        if index.find(info.id).is_some() {
            return info.id.to_string();
        }
        // Then try substring match with the family ID
        let family_matches: Vec<_> = index
            .items
            .iter()
            .filter(|m| m.id.to_lowercase().contains(info.id))
            .collect();
        if family_matches.len() == 1 {
            return family_matches[0].id.clone();
        }
    }

    // No confident match — return original and let resolver show suggestions
    id.to_string()
}
