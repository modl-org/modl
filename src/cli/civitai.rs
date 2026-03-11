use anyhow::{Context, Result};
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};

use crate::auth::AuthStore;
use crate::core::civitai;
use crate::core::config::Config;
use crate::core::db::{Database, InstalledModelRecord};
use crate::core::download;
use crate::core::store::Store;

use super::fmt::format_downloads;

// ---------------------------------------------------------------------------
// modl search --civitai "query"
// ---------------------------------------------------------------------------

pub async fn search(query: &str, base_model: Option<&str>, sort: Option<&str>) -> Result<()> {
    println!(
        "\n{} Searching CivitAI for LoRAs matching '{}'...\n",
        style("→").cyan(),
        query
    );

    let resp = civitai::search_loras(query, base_model, sort, 1, 20, false)
        .await
        .context("CivitAI search failed")?;

    if resp.items.is_empty() {
        println!("No LoRAs found on CivitAI for '{}'.", query);
        return Ok(());
    }

    let db = Database::open().ok();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("Name").fg(Color::Cyan),
        Cell::new("Base").fg(Color::Cyan),
        Cell::new("Size").fg(Color::Cyan),
        Cell::new("Downloads").fg(Color::Cyan),
        Cell::new("Trigger").fg(Color::Cyan),
        Cell::new("Pull command").fg(Color::Cyan),
    ]);

    for model in &resp.items {
        let version = match model.model_versions.first() {
            Some(v) => v,
            None => continue,
        };

        let civitai_id = format!("civitai:{}", version.id);
        let installed = db
            .as_ref()
            .and_then(|d| d.is_installed(&civitai_id).ok())
            .unwrap_or(false);

        let file = version.files.first();
        let size = file
            .and_then(|f| f.size_kb)
            .map(|kb| HumanBytes((kb * 1024.0) as u64).to_string())
            .unwrap_or_else(|| "—".to_string());

        let downloads = model
            .stats
            .as_ref()
            .and_then(|s| s.download_count)
            .map(format_downloads)
            .unwrap_or_else(|| "—".to_string());

        let trigger = version
            .trained_words
            .as_ref()
            .and_then(|w| w.first())
            .cloned()
            .unwrap_or_default();

        let base = version.base_model.as_deref().unwrap_or("—").to_string();

        let name_display = if installed {
            format!("{} {}", model.name, style("(installed)").green())
        } else {
            model.name.clone()
        };

        table.add_row(vec![
            Cell::new(&name_display),
            Cell::new(&base),
            Cell::new(&size),
            Cell::new(&downloads),
            Cell::new(&trigger),
            Cell::new(format!("modl pull civitai:{}", version.id)).fg(Color::DarkGrey),
        ]);
    }

    println!("{table}");

    let total = resp.metadata.total_items.unwrap_or(0);
    if total > 20 {
        println!("\n  {} Showing 20 of {} results.", style("ℹ").blue(), total);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// modl pull civitai:<version_id>
// ---------------------------------------------------------------------------

pub async fn install(version_id_str: &str, force: bool) -> Result<()> {
    let version_id: u64 = version_id_str
        .parse()
        .context("Invalid CivitAI version ID — expected a number")?;

    let lora_id = format!("civitai:{version_id}");
    let db = Database::open()?;

    // Already installed?
    if !force && db.is_installed(&lora_id).unwrap_or(false) {
        println!(
            "{} CivitAI LoRA {} is already installed.",
            style("✓").green(),
            style(&lora_id).bold()
        );
        return Ok(());
    }

    println!(
        "{} Fetching info for CivitAI version {}...",
        style("→").cyan(),
        style(version_id).bold()
    );

    // Look up version details from CivitAI to get metadata
    let info = fetch_version_info(version_id).await?;

    let config = Config::load()?;
    let auth_store = AuthStore::load().unwrap_or_default();
    let store = Store::new(config.store_root());
    let api_key = auth_store.token_for("civitai");

    // Build download URL
    let url = civitai::download_url(version_id, api_key.as_deref());

    let file_name = info
        .file_name
        .clone()
        .unwrap_or_else(|| format!("{}.safetensors", info.name.replace(' ', "_")));
    let sha256 = info.sha256.as_deref().unwrap_or("").to_lowercase();
    let expected_size = info.file_size_kb.map(|kb| (kb * 1024.0) as u64);

    // Compute store path
    let hash_for_path = if sha256.is_empty() {
        format!("civitai-{version_id}")
    } else {
        sha256.clone()
    };
    let store_path = store.path_for(
        &crate::core::manifest::AssetType::Lora,
        &hash_for_path,
        &file_name,
    );
    store
        .ensure_dir(&store_path)
        .context("Failed to create store directory")?;

    // Display plan
    let size_display = expected_size
        .map(|s| HumanBytes(s).to_string())
        .unwrap_or_else(|| "unknown size".to_string());

    println!(
        "\n  {} {} ({})",
        style("↓").cyan(),
        style(&info.name).bold(),
        size_display
    );
    if let Some(ref base) = info.base_model {
        println!("    Base model: {}", style(base).dim());
    }
    if !info.trigger_words.is_empty() {
        println!(
            "    Trigger: {}",
            style(info.trigger_words.join(", ")).dim()
        );
    }
    println!();

    // Download with progress
    let pb = ProgressBar::new(expected_size.unwrap_or(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {spinner:.cyan} [{bar:30.cyan/dim}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("━╸─"),
    );
    if expected_size.is_none() {
        pb.set_length(0);
    }

    download::download_file(&url, &store_path, expected_size, api_key.as_deref())
        .await
        .context("Download failed")?;

    pb.finish_and_clear();

    // Verify hash
    if !sha256.is_empty() {
        match Store::verify_hash(&store_path, &sha256) {
            Ok(true) => {}
            Ok(false) => {
                let _ = std::fs::remove_file(&store_path);
                anyhow::bail!("SHA256 hash mismatch after download");
            }
            Err(e) => {
                eprintln!("  {} Hash verification failed: {e}", style("!").yellow());
            }
        }
    }

    let actual_size = std::fs::metadata(&store_path).map(|m| m.len()).unwrap_or(0);

    // Record in DB
    db.insert_installed(&InstalledModelRecord {
        id: &lora_id,
        name: &info.name,
        asset_type: "lora",
        variant: None,
        sha256: &sha256,
        size: actual_size,
        file_name: &file_name,
        store_path: store_path.to_str().unwrap_or(""),
    })
    .context("Failed to record in database")?;

    // Store artifact metadata (trigger words, base model, etc.)
    let metadata = serde_json::json!({
        "trigger_word": info.trigger_words.first().unwrap_or(&String::new()),
        "trigger_words": info.trigger_words,
        "base_model": info.base_model.clone().unwrap_or_default(),
        "source": "civitai",
        "civitai_version_id": version_id,
    });

    db.insert_artifact(
        &lora_id,
        None,
        "lora",
        store_path.to_str().unwrap_or(""),
        &sha256,
        actual_size,
        Some(&metadata.to_string()),
    )
    .context("Failed to store artifact metadata")?;

    println!(
        "{} Installed {} (civitai:{}) successfully.",
        style("✓").green().bold(),
        style(&info.name).bold(),
        version_id,
    );

    if !info.trigger_words.is_empty() {
        println!(
            "  Trigger word: {}",
            style(info.trigger_words.join(", ")).cyan()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Fetch version info from CivitAI
// ---------------------------------------------------------------------------

struct VersionInfo {
    name: String,
    base_model: Option<String>,
    trigger_words: Vec<String>,
    file_name: Option<String>,
    file_size_kb: Option<f64>,
    sha256: Option<String>,
}

async fn fetch_version_info(version_id: u64) -> Result<VersionInfo> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let url = format!("https://civitai.com/api/v1/model-versions/{version_id}");
    let resp = client
        .get(&url)
        .header("User-Agent", "modl/1.0")
        .send()
        .await
        .context("Failed to reach CivitAI API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("CivitAI API returned {status}: {body}");
    }

    let version: civitai::ModelVersion = resp
        .json()
        .await
        .context("Failed to parse CivitAI version response")?;

    let file = version.files.first();

    let name = version.name.clone();

    Ok(VersionInfo {
        name,
        base_model: version.base_model,
        trigger_words: version.trained_words.unwrap_or_default(),
        file_name: file.map(|f| f.name.clone()),
        file_size_kb: file.and_then(|f| f.size_kb),
        sha256: file.and_then(|f| f.hashes.as_ref().and_then(|h| h.sha256.clone())),
    })
}
