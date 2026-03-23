//! Pre-flight checks that run before training or generation.
//!
//! Each check returns a clear, actionable error if something is wrong.
//! The goal is to fail fast with a fix hint, not fail cryptically mid-run.

use anyhow::{Result, bail};

use crate::core::db::Database;
use crate::core::model_family;
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

/// Check that a generation-capable runtime is available.
///
/// Accepts either the lightweight generator profile or the full trainer
/// profile (which is a superset). If neither is bootstrapped but the
/// runtime lock exists, we pass — `setup_generation()` will finish it.
pub fn check_generation_runtime() -> Result<()> {
    // Either profile is sufficient for generation
    if runtime::is_profile_ready("generator").unwrap_or(false) {
        return Ok(());
    }
    if runtime::is_profile_ready("trainer-cu124").unwrap_or(false) {
        return Ok(());
    }

    // Lock file exists with a compatible profile — setup_generation will
    // complete the bootstrap
    let status = runtime::status()?;
    if status.installed {
        let lock_profile = status.profile.as_deref().unwrap_or("");
        if lock_profile == "generator" || lock_profile == "trainer-cu124" {
            return Ok(());
        }
    }

    bail!(
        "Python runtime is not installed.\n\n\
         Run this first:\n\n  \
         modl runtime install\n\n\
         This installs a managed Python environment with PyTorch and diffusers."
    );
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

    // Try exact match first
    if db.is_installed(base_model_id)? {
        return Ok(());
    }

    // Try model_family alias resolution (e.g. "sdxl" → "sdxl-base-1.0")
    if let Some(family_info) = model_family::resolve_model(base_model_id) {
        let installed = db.list_installed(None)?;
        let gen_types = ["checkpoint", "diffusion_model"];
        let found = installed.iter().any(|m| {
            gen_types.contains(&m.asset_type.as_str())
                && (m.id == family_info.id
                    || m.id.contains(family_info.id)
                    || family_info.id.contains(&*m.id))
        });
        if found {
            return Ok(());
        }
    }

    // Base variant (e.g. "flux2-klein-base-4b") shares components with
    // its distilled parent ("flux2-klein-4b"). The Python worker resolves
    // the transformer via MODEL_REGISTRY / HF hub, so allow training if
    // the parent model is installed.
    let parent_id = base_model_id.replace("-base", "");
    if parent_id != base_model_id && db.is_installed(&parent_id)? {
        return Ok(());
    }

    bail!(
        "Base model '{}' is not installed.\n\n\
         Pull it first:\n\n  \
         modl pull {}\n\n\
         Or see available models:\n\n  \
         modl search \"{}\"",
        base_model_id,
        base_model_id,
        base_model_id.split('-').next().unwrap_or(base_model_id)
    )
}

/// Check that auth is configured for providers that require it.
///
/// Currently checks HuggingFace token presence. Civitai is checked at
/// download time since not all Civitai models require auth.
pub fn check_auth_if_gated(base_model_id: &str) -> Result<()> {
    // Models known to be gated on HuggingFace
    const GATED_MODELS: &[&str] = &["flux-dev", "flux2-dev", "sd-3.5-large", "sd-3.5-medium"];

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

/// Check that all required dependencies (text encoders, VAEs, etc.) are installed.
///
/// Looks up the model in the registry index and verifies each `requires`
/// entry is present in the installed models DB.
pub fn check_dependencies(base_model_id: &str) -> Result<()> {
    // Skip for filesystem paths
    if base_model_id.contains('/') || base_model_id.contains('.') {
        return Ok(());
    }

    let index = match crate::core::registry::RegistryIndex::load() {
        Ok(idx) => idx,
        Err(_) => return Ok(()), // No registry index — skip check
    };

    let manifest = match index.find(base_model_id) {
        Some(m) => m,
        None => return Ok(()), // Not in registry — skip check
    };

    if manifest.requires.is_empty() {
        return Ok(());
    }

    let db = Database::open()?;
    let mut missing = Vec::new();

    for dep in &manifest.requires {
        let installed = db.is_installed(&dep.id)?
            || dep
                .optional_variant
                .as_deref()
                .map(|v| db.is_installed(v))
                .transpose()?
                .unwrap_or(false);
        if !installed {
            let reason = dep.reason.as_deref().unwrap_or("");
            missing.push(format!("  - {} ({}): {}", dep.id, dep.dep_type, reason));
        }
    }

    if !missing.is_empty() {
        bail!(
            "Model '{}' requires components that are not installed:\n\n{}\n\n\
             Pull the base model to install all dependencies:\n\n  \
             modl pull {}",
            base_model_id,
            missing.join("\n"),
            base_model_id
        );
    }

    Ok(())
}

/// Check that the current device supports training.
///
/// Training requires CUDA — MPS (Apple Silicon) is not supported because
/// ai-toolkit and bitsandbytes depend on CUDA-specific kernels.
pub fn check_device_for_training() -> Result<()> {
    if let Some(info) = crate::core::gpu::detect()
        && info.device == crate::core::gpu::DeviceType::Mps
    {
        bail!(
            "Training requires a CUDA GPU and is not supported on Apple Silicon (MPS).\n\n\
             Use cloud training instead:\n\n  \
             modl train --cloud\n\n\
             This runs training on a cloud GPU."
        );
    }
    Ok(())
}

/// Run all pre-flight checks for training.
pub fn for_training(base_model_id: &str) -> Result<()> {
    check_device_for_training()?;
    check_runtime()?;
    check_base_model(base_model_id)?;
    check_dependencies(base_model_id)?;
    check_auth_if_gated(base_model_id)?;
    Ok(())
}

/// Run all pre-flight checks for generation.
///
/// Uses the lightweight generation runtime check — does not require
/// ai-toolkit, only Python + PyTorch + diffusers.
pub fn for_generation(base_model_id: &str) -> Result<()> {
    check_generation_runtime()?;
    check_base_model(base_model_id)?;
    check_dependencies(base_model_id)?;
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
