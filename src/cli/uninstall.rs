use anyhow::Result;
use console::style;
use indicatif::HumanBytes;

use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::manifest::AssetType;

pub async fn run(id: &str, force: bool) -> Result<()> {
    let db = Database::open()?;
    let config = Config::load()?;

    // Check if it's a trained artifact (LoRA) first
    if let Some(artifact) = db.find_artifact(id)? {
        return remove_trained_artifact(&db, &artifact, force);
    }

    if !db.is_installed(id)? {
        anyhow::bail!("'{}' is not installed.", id);
    }

    // Check for dependents (items that depend on this one)
    // For now, simple check — a future improvement would query the dependencies table
    if !force {
        println!("{} Checking for dependent models...", style("→").cyan());
        // TODO: Query dependencies table for items that require this id
        // For now, allow uninstall freely
    }

    // Get installed model info before removing
    let models = db.list_installed(None)?;
    let model = models.iter().find(|m| m.id == id);

    if let Some(m) = model {
        // Remove symlinks from all targets
        for target in &config.targets {
            if target.symlink {
                let link_path = crate::compat::symlink_path(
                    &target.path,
                    &target.tool_type,
                    &m.asset_type
                        .parse::<AssetType>()
                        .unwrap_or(AssetType::Checkpoint),
                    &m.file_name,
                );
                if link_path.is_symlink() {
                    std::fs::remove_file(&link_path).ok();
                    println!(
                        "  {} Removed symlink: {}",
                        style("×").red(),
                        link_path.display()
                    );
                }
            }
        }

        println!(
            "  {} Marked {} as uninstalled",
            style("×").red(),
            style(&m.name).bold()
        );
        println!(
            "  {} Store file kept — run {} to reclaim space",
            style("i").dim(),
            style("modl gc").cyan()
        );
    }

    db.remove_installed(id)?;

    println!();
    println!("{} Uninstalled '{}'.", style("✓").green(), id);

    Ok(())
}

fn remove_trained_artifact(
    db: &Database,
    artifact: &crate::core::db::ArtifactRecord,
    _force: bool,
) -> Result<()> {
    let meta: serde_json::Value = artifact
        .metadata
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Null);

    let lora_name = meta
        .get("lora_name")
        .and_then(|v| v.as_str())
        .unwrap_or(&artifact.artifact_id);

    println!(
        "{} Removing trained {} {}",
        style("→").cyan(),
        &artifact.kind,
        style(lora_name).bold()
    );

    // Remove the LoRA symlink from ~/.modl/loras/
    let loras_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".modl")
        .join("loras");
    let symlink_path = loras_dir.join(format!("{}.safetensors", lora_name));
    if symlink_path.is_symlink() {
        std::fs::remove_file(&symlink_path).ok();
        println!(
            "  {} Removed symlink: {}",
            style("×").red(),
            symlink_path.display()
        );
    }

    // Remove the store file
    let store_path = std::path::Path::new(&artifact.path);
    if store_path.exists() {
        std::fs::remove_file(store_path).ok();
        println!(
            "  {} Deleted {} ({})",
            style("×").red(),
            store_path.display(),
            HumanBytes(artifact.size_bytes)
        );
        // Remove the parent dir if empty
        if let Some(parent) = store_path.parent() {
            std::fs::remove_dir(parent).ok();
        }
    }

    // Remove training output directory (samples, logs, config)
    let training_output_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".modl")
        .join("training_output")
        .join(lora_name);
    if training_output_dir.is_dir() {
        std::fs::remove_dir_all(&training_output_dir)?;
        println!(
            "  {} Deleted training output: {}",
            style("×").red(),
            training_output_dir.display()
        );
    }
    // Also clean up any stale config/log files at the training_output level
    let training_output_parent = training_output_dir.parent().unwrap();
    for suffix in &["-config.yaml", ".log"] {
        let stale = training_output_parent.join(format!("{}{}", lora_name, suffix));
        if stale.exists() {
            std::fs::remove_file(&stale).ok();
        }
    }

    // Also remove from the installed table if it was registered there
    db.remove_installed(&artifact.artifact_id)?;

    // Remove the artifact record from DB
    db.delete_artifact(&artifact.artifact_id)?;

    // Remove associated job records from DB
    db.delete_jobs_by_lora_name(lora_name)?;

    println!();
    println!(
        "{} Removed trained {} '{}'.",
        style("✓").green(),
        &artifact.kind,
        lora_name
    );

    Ok(())
}
