use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use console::style;

use crate::core::config::Config;
use crate::core::db::{Database, InstalledModelRecord};
use crate::core::download;
use crate::core::hub::{HubClient, parse_hub_ref};
use crate::core::manifest::AssetType;
use crate::core::paths::modl_root;
use crate::core::store::Store;

pub async fn run(reference: &str) -> Result<()> {
    let href = parse_hub_ref(reference)
        .context("Invalid hub reference. Expected <user>/<slug> or <user>/<slug>@<version>.")?;

    // Public hub pulls should work without authentication; private assets will
    // still require a key and be rejected by the API.
    let client = HubClient::from_config(false)?;
    let item = client.get_item(&href.username, &href.slug).await?;
    let pull = client
        .pull(&href.username, &href.slug, href.version)
        .await
        .with_context(|| format!("Failed to pull {reference}"))?;

    match item.item.item_type.as_str() {
        "lora" => pull_lora(&href.username, &href.slug, &item, &pull).await,
        "dataset" => pull_dataset(&href.username, &href.slug, &item, &pull).await,
        other => bail!("Unsupported hub item type: {other}"),
    }
}

async fn pull_lora(
    username: &str,
    slug: &str,
    item: &crate::core::hub::HubItemDetail,
    pull: &crate::core::hub::PullResponse,
) -> Result<()> {
    let tmp_dir = modl_root().join("tmp").join("hub");
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp_path = tmp_dir.join(format!("{username}-{slug}-v{}.safetensors", pull.version));

    println!(
        "{} Downloading {}/{}@v{}",
        style("↓").cyan(),
        username,
        slug,
        pull.version
    );
    download::download_file(&pull.download_url, &tmp_path, pull.size_bytes, None).await?;

    if let Some(ref expected_sha) = pull.sha256
        && !Store::verify_hash(&tmp_path, expected_sha)?
    {
        let _ = std::fs::remove_file(&tmp_path);
        bail!("SHA256 mismatch for downloaded LoRA");
    }

    let config = Config::load()?;
    let store = Store::new(config.store_root());
    let db = Database::open()?;

    let filename = format!("{slug}-v{}.safetensors", pull.version);
    let sha_for_path = pull
        .sha256
        .clone()
        .unwrap_or_else(|| format!("hub-{}-{}-v{}", username, slug, pull.version));
    let store_path = store.path_for(&AssetType::Lora, &sha_for_path, &filename);
    store.ensure_dir(&store_path)?;
    if !store_path.exists() {
        std::fs::copy(&tmp_path, &store_path).with_context(|| {
            format!(
                "Failed to copy {} -> {}",
                tmp_path.display(),
                store_path.display()
            )
        })?;
    }

    let size = std::fs::metadata(&store_path)?.len();
    let model_id = format!("hub:{}/{}@{}", username, slug, pull.version);
    let model_name = format!("{username}/{slug}");
    let sha_record = pull.sha256.clone().unwrap_or_default();

    db.insert_installed(&InstalledModelRecord {
        id: &model_id,
        name: &model_name,
        asset_type: "lora",
        variant: None,
        sha256: &sha_record,
        size,
        file_name: &filename,
        store_path: &store_path.to_string_lossy(),
    })?;

    let metadata = serde_json::json!({
        "source": "hub",
        "username": username,
        "slug": slug,
        "version": pull.version,
        "r2_key": pull.r2_key,
        "visibility": item.item.visibility,
        "base_model": item.item.base_model,
        "trigger_words": item.item.trigger_words,
    });

    db.insert_artifact(
        &model_id,
        None,
        "lora",
        &store_path.to_string_lossy(),
        &sha_record,
        size,
        Some(&metadata.to_string()),
    )?;

    let _ = std::fs::remove_file(&tmp_path);
    println!(
        "{} Installed {} (v{})",
        style("✓").green().bold(),
        style(format!("{}/{}", username, slug)).bold(),
        pull.version
    );
    Ok(())
}

async fn pull_dataset(
    username: &str,
    slug: &str,
    item: &crate::core::hub::HubItemDetail,
    pull: &crate::core::hub::PullResponse,
) -> Result<()> {
    let tmp_dir = modl_root().join("tmp").join("hub");
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp_zip = tmp_dir.join(format!("{username}-{slug}-v{}.zip", pull.version));

    println!(
        "{} Downloading dataset {}/{}@v{}",
        style("↓").cyan(),
        username,
        slug,
        pull.version
    );
    download::download_file(&pull.download_url, &tmp_zip, pull.size_bytes, None).await?;

    if let Some(ref expected_sha) = pull.sha256
        && !Store::verify_hash(&tmp_zip, expected_sha)?
    {
        let _ = std::fs::remove_file(&tmp_zip);
        bail!("SHA256 mismatch for downloaded dataset archive");
    }

    let datasets_root = modl_root().join("datasets");
    std::fs::create_dir_all(&datasets_root)?;
    let dest_dir = datasets_root.join(format!("{}_{}_v{}", username, slug, pull.version));
    if dest_dir.exists() {
        bail!("Dataset destination already exists: {}", dest_dir.display());
    }

    extract_zip(&tmp_zip, &dest_dir)?;

    let db = Database::open()?;
    let size = std::fs::metadata(&tmp_zip)?.len();
    let artifact_id = format!("hub-dataset:{}/{}@{}", username, slug, pull.version);
    let sha_record = pull.sha256.clone().unwrap_or_default();
    let metadata = serde_json::json!({
        "source": "hub",
        "username": username,
        "slug": slug,
        "version": pull.version,
        "r2_key": pull.r2_key,
        "visibility": item.item.visibility,
    });
    db.insert_artifact(
        &artifact_id,
        None,
        "dataset",
        &dest_dir.to_string_lossy(),
        &sha_record,
        size,
        Some(&metadata.to_string()),
    )?;

    let _ = std::fs::remove_file(&tmp_zip);
    println!(
        "{} Dataset extracted to {}",
        style("✓").green().bold(),
        dest_dir.display()
    );
    Ok(())
}

fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = std::fs::File::open(zip_path)
        .with_context(|| format!("Failed to open {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("Failed to read zip archive")?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).context("Failed to read zip entry")?;
        let Some(enclosed) = entry.enclosed_name().map(|p| p.to_path_buf()) else {
            continue;
        };
        let out_path: PathBuf = dest_dir.join(enclosed);

        if entry.name().ends_with('/') {
            std::fs::create_dir_all(&out_path)
                .with_context(|| format!("Failed to create {}", out_path.display()))?;
            continue;
        }

        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let mut out_file = std::fs::File::create(&out_path)
            .with_context(|| format!("Failed to create {}", out_path.display()))?;
        std::io::copy(&mut entry, &mut out_file)
            .with_context(|| format!("Failed to extract {}", out_path.display()))?;
        out_file.flush().ok();
    }

    Ok(())
}
