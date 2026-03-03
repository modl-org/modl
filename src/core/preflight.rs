//! Pre-flight checks that run before training or generation.
//!
//! Each check returns a clear, actionable error if something is wrong.
//! The goal is to fail fast with a fix hint, not fail cryptically mid-run.

use anyhow::{Result, bail};

use crate::core::db::Database;
use crate::core::runtime;

/// Check that the training runtime is installed and bootstrapped.
///
/// This is a fast check (file existence only, no subprocess).
pub fn check_runtime() -> Result<()> {
    let status = runtime::status()?;
    if !status.installed {
        bail!(
            "Training runtime is not installed.\n\n\
             Run this first:\n\n  \
             modl runtime install\n  \
             modl train setup\n\n\
             This installs a managed Python environment with PyTorch and ai-toolkit."
        );
    }
    Ok(())
}

/// Check that a base model is installed locally.
///
/// Skips the check if the model ID looks like a local path (contains `/`).
pub fn check_base_model(base_model_id: &str) -> Result<()> {
    // If the user passed a filesystem path, skip the registry check
    if base_model_id.contains('/') || base_model_id.contains('.') {
        return Ok(());
    }

    let db = Database::open()?;
    if !db.is_installed(base_model_id)? {
        bail!(
            "Base model '{}' is not installed.\n\n\
             Pull it first:\n\n  \
             modl model pull {}\n\n\
             Or see available models:\n\n  \
             modl model search \"{}\"",
            base_model_id,
            base_model_id,
            base_model_id.split('-').next().unwrap_or(base_model_id)
        );
    }
    Ok(())
}

/// Check that auth is configured for providers that require it.
///
/// Currently checks HuggingFace token presence. Civitai is checked at
/// download time since not all Civitai models require auth.
pub fn check_auth_if_gated(base_model_id: &str) -> Result<()> {
    // Models known to be gated on HuggingFace
    const GATED_MODELS: &[&str] = &["flux-dev", "sd-3.5-large", "sd-3.5-medium"];

    if !GATED_MODELS.contains(&base_model_id) {
        return Ok(());
    }

    let auth = crate::auth::AuthStore::load()?;
    if auth.huggingface.is_none() {
        bail!(
            "Model '{}' requires HuggingFace authentication.\n\n\
             Set up auth first:\n\n  \
             modl auth huggingface\n\n\
             You'll need to accept the model's terms at huggingface.co \
             and provide an access token.",
            base_model_id
        );
    }
    Ok(())
}

/// Run all pre-flight checks for training.
pub fn for_training(base_model_id: &str) -> Result<()> {
    check_runtime()?;
    check_base_model(base_model_id)?;
    check_auth_if_gated(base_model_id)?;
    Ok(())
}

/// Run all pre-flight checks for generation.
///
/// Same as training but could diverge in the future (e.g. generation
/// might not need the full ai-toolkit bootstrap).
pub fn for_generation(base_model_id: &str) -> Result<()> {
    check_runtime()?;
    check_base_model(base_model_id)?;
    check_auth_if_gated(base_model_id)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_like_model_skips_check() {
        // Paths with / or . should skip the DB lookup
        assert!(check_base_model("/home/user/models/custom.safetensors").is_ok());
        assert!(check_base_model("./my-model.safetensors").is_ok());
    }
}
