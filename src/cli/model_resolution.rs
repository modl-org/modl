use console::style;

use crate::core::cloud::CloudProvider;
use crate::core::db::Database;
use crate::core::model_family;
use crate::core::model_resolve;

/// Pick the best installed generation model, falling back to flux-schnell.
///
/// If only one generation model is installed, uses it. With multiple installed,
/// prefers flux-schnell. If nothing is installed (or DB error), returns
/// "flux-schnell" as the hardcoded default so the error surfaces at execution.
pub fn default_generation_model(db: &Database) -> String {
    let gen_types = ["checkpoint", "diffusion_model"];
    if let Ok(installed) = db.list_installed(None) {
        let gen_models: Vec<_> = installed
            .iter()
            .filter(|m| gen_types.contains(&m.asset_type.as_str()))
            .collect();
        if gen_models.len() == 1 {
            return gen_models[0].id.clone();
        }
        if !gen_models.is_empty() {
            if let Some(m) = gen_models.iter().find(|m| m.id == "flux-schnell") {
                return m.id.clone();
            }
            return gen_models[0].id.clone();
        }
    }
    "flux-schnell".to_string()
}

/// Resolve the cloud provider: explicit flag → config default → Modal.
pub fn resolve_cloud_provider(provider: Option<CloudProvider>) -> CloudProvider {
    if let Some(p) = provider {
        return p;
    }
    if let Ok(config) = crate::core::config::Config::load()
        && let Some(ref cloud) = config.cloud
        && let Some(ref default) = cloud.default_provider
        && let Ok(p) = default.parse()
    {
        return p;
    }
    CloudProvider::Modal
}

/// Smart inpaint routing: auto-upgrade Flux 1 → Flux Fill when available.
///
/// When the user requests inpainting with a plain Flux 1 model, tries to
/// route to an installed Flux Fill model instead. Fill models have 384 input
/// channels and produce cleaner inpainting results.
///
/// Returns `(effective_model_id, Option<base_path>)`.
pub fn resolve_inpaint_model(base_model: &str, db: &Database) -> (String, Option<String>) {
    let info = model_family::resolve_model(base_model);
    let is_flux1 = info.is_some_and(|m| m.arch_key == "flux" || m.arch_key == "flux_schnell");
    if !is_flux1 {
        return (
            base_model.to_string(),
            model_resolve::resolve_base_model_path(base_model, db),
        );
    }

    for candidate in ["flux-fill-dev-onereward", "flux-fill-dev"] {
        if model_resolve::resolve_base_model_path(candidate, db).is_some() {
            println!(
                "  {} Using {} for inpainting",
                style("↳").dim(),
                style(candidate).bold()
            );
            return (
                candidate.to_string(),
                model_resolve::resolve_base_model_path(candidate, db),
            );
        }
    }

    (
        base_model.to_string(),
        model_resolve::resolve_base_model_path(base_model, db),
    )
}
