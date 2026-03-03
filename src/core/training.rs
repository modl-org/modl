use anyhow::{Result, bail};
use std::env;
use std::path::PathBuf;

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
