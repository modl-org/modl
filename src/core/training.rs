use anyhow::{Result, bail};
use std::env;
use std::path::PathBuf;

use super::db::Database;
use super::training_status;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn modl_root() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
}

/// Resolve the path to the Python worker package root.
///
/// Checks `MODL_WORKER_PYTHON_ROOT` env var first, then falls back to
/// `CARGO_MANIFEST_DIR/python`.
pub fn resolve_worker_python_root() -> Result<PathBuf> {
    if let Ok(custom) = env::var("MODL_WORKER_PYTHON_ROOT") {
        let path = PathBuf::from(custom);
        if path.exists() {
            return Ok(path);
        }
        bail!(
            "MODL_WORKER_PYTHON_ROOT points to missing path: {}",
            path.display()
        );
    }

    let default_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    if default_path.exists() {
        Ok(default_path)
    } else {
        bail!(
            "Worker python package not found at {}. Set MODL_WORKER_PYTHON_ROOT to a valid path.",
            default_path.display()
        )
    }
}

// ---------------------------------------------------------------------------
// Training run management
// ---------------------------------------------------------------------------

/// List training run names (directories under training_output/).
pub fn list_training_runs() -> Result<Vec<String>> {
    let output_dir = modl_root().join("training_output");
    let mut runs = Vec::new();

    if output_dir.exists() {
        for entry in std::fs::read_dir(&output_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                runs.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    runs.sort();
    Ok(runs)
}

/// Delete a training run: output directory, log file, LoRA symlink, and DB records.
///
/// Refuses to delete if training is currently running.
pub fn delete_training_run(name: &str) -> Result<()> {
    // Safety: refuse to delete if training is currently running
    let running = training_status::get_status(name)
        .map(|s| s.is_running)
        .unwrap_or(false);
    if running {
        bail!("Cannot delete '{name}': training is still running. Pause it first.");
    }

    let root = modl_root();
    let run_dir = root.join("training_output").join(name);
    let log_file = root.join("training_output").join(format!("{name}.log"));

    if !run_dir.exists() && !log_file.exists() {
        bail!("Training run '{name}' not found");
    }

    // Remove training output directory (samples, checkpoints, config)
    if run_dir.exists() {
        std::fs::remove_dir_all(&run_dir)?;
    }
    // Remove log file
    if log_file.exists() {
        std::fs::remove_file(&log_file)?;
    }

    // Remove LoRA symlink from loras/ directory
    let lora_link = root.join("loras").join(format!("{name}.safetensors"));
    if lora_link.symlink_metadata().is_ok() {
        std::fs::remove_file(&lora_link)?;
    }

    // Clean up DB job records
    if let Ok(db) = Database::open() {
        let _ = db.delete_jobs_by_lora_name(name);
    }

    Ok(())
}
