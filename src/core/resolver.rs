use anyhow::Result;
use std::collections::HashSet;

use super::manifest::Manifest;
use super::registry::RegistryIndex;

/// Resolved install plan — what needs to be downloaded
#[derive(Debug)]
pub struct InstallPlan {
    /// Items to install, in dependency order (deps first)
    pub items: Vec<ResolvedItem>,
}

#[derive(Debug)]
pub struct ResolvedItem {
    pub manifest: Manifest,
    pub variant_id: Option<String>,
    /// Whether this item is already installed (skip download)
    pub already_installed: bool,
}

/// Resolve all dependencies for a given model ID
pub fn resolve(
    id: &str,
    variant: Option<&str>,
    index: &RegistryIndex,
    installed: &HashSet<String>,
) -> Result<InstallPlan> {
    let mut plan = Vec::new();
    let mut visited = HashSet::new();
    resolve_recursive(id, variant, index, installed, &mut visited, &mut plan)?;
    Ok(InstallPlan { items: plan })
}

fn resolve_recursive(
    id: &str,
    variant: Option<&str>,
    index: &RegistryIndex,
    installed: &HashSet<String>,
    visited: &mut HashSet<String>,
    plan: &mut Vec<ResolvedItem>,
) -> Result<()> {
    if visited.contains(id) {
        return Ok(()); // Already processed (handles circular deps)
    }
    visited.insert(id.to_string());

    let manifest = index
        .find(id)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", id))?;

    // Resolve dependencies first (depth-first)
    for dep in &manifest.requires {
        // `optional_variant` is an alternative model ID (e.g. "t5-xxl-fp8" as
        // a lighter substitute for "t5-xxl-fp16"), NOT a variant within the dep.
        // If the alternative model is already installed, skip this dep.
        // Otherwise, install the primary dep and let VRAM auto-select pick the
        // right variant within that model.
        let dep_id = &dep.id;

        // If user already installed the optional alternative, skip the primary
        if let Some(ref alt_id) = dep.optional_variant
            && installed.contains(alt_id)
        {
            visited.insert(dep_id.to_string());
            continue;
        }

        resolve_recursive(
            dep_id, None, // variant auto-selected by VRAM in install.rs
            index, installed, visited, plan,
        )?;
    }

    // Add this item
    plan.push(ResolvedItem {
        manifest: manifest.clone(),
        variant_id: variant.map(String::from),
        already_installed: installed.contains(id),
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::manifest::{AssetType, Dependency, Manifest};

    fn make_index(items: Vec<Manifest>) -> RegistryIndex {
        RegistryIndex {
            version: 2,
            generated_at: None,
            total_count: None,
            type_counts: None,
            cloud_available_count: None,
            schema_url: None,
            items,
        }
    }

    fn simple_manifest(id: &str, deps: Vec<(&str, AssetType)>) -> Manifest {
        Manifest {
            id: id.to_string(),
            name: id.to_string(),
            asset_type: AssetType::Checkpoint,
            architecture: None,
            author: None,
            license: None,
            homepage: None,
            description: None,
            variants: vec![],
            file: None,
            requires: deps
                .into_iter()
                .map(|(dep_id, dep_type)| Dependency {
                    id: dep_id.to_string(),
                    dep_type,
                    reason: None,
                    optional_variant: None,
                })
                .collect(),
            auth: None,
            defaults: None,
            base_models: vec![],
            trigger_words: vec![],
            recommended_weight: None,
            weight_range: None,
            preprocessor: None,
            scale_factor: None,
            clip_vision_model: None,
            cloud_available: false,
            cloud_training: None,
            cloud_inference: None,
            huggingface_repo: None,
            category: None,
            training_details: None,
            sample_images: vec![],
            recipe: None,
            publisher: None,
            preview_images: vec![],
            tags: vec![],
            rating: None,
            downloads: None,
            added: None,
            updated: None,
        }
    }

    #[test]
    fn test_resolve_no_deps() {
        let index = make_index(vec![simple_manifest("model-a", vec![])]);
        let installed = HashSet::new();
        let plan = resolve("model-a", None, &index, &installed).unwrap();
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].manifest.id, "model-a");
    }

    #[test]
    fn test_resolve_with_deps() {
        let index = make_index(vec![
            simple_manifest("vae-1", vec![]),
            simple_manifest("model-a", vec![("vae-1", AssetType::Vae)]),
        ]);
        let installed = HashSet::new();
        let plan = resolve("model-a", None, &index, &installed).unwrap();
        assert_eq!(plan.items.len(), 2);
        assert_eq!(plan.items[0].manifest.id, "vae-1"); // Dep first
        assert_eq!(plan.items[1].manifest.id, "model-a");
    }

    #[test]
    fn test_resolve_skips_installed() {
        let index = make_index(vec![
            simple_manifest("vae-1", vec![]),
            simple_manifest("model-a", vec![("vae-1", AssetType::Vae)]),
        ]);
        let installed: HashSet<String> = ["vae-1".to_string()].into();
        let plan = resolve("model-a", None, &index, &installed).unwrap();
        assert_eq!(plan.items.len(), 2);
        assert!(plan.items[0].already_installed); // vae-1 marked as installed
        assert!(!plan.items[1].already_installed);
    }

    #[test]
    fn test_resolve_propagates_optional_variant() {
        // optional_variant is an alternative model ID (e.g. "t5-xxl-fp8"),
        // NOT a variant within the dep. When the alternative is NOT installed,
        // the primary dep is resolved with variant_id = None (auto-selected
        // by VRAM in install.rs).
        let encoder = Manifest {
            id: "t5-xxl".to_string(),
            name: "T5-XXL".to_string(),
            asset_type: AssetType::TextEncoder,
            variants: vec![],
            requires: vec![],
            ..simple_manifest("t5-xxl", vec![])
        };

        let checkpoint = Manifest {
            id: "flux-dev".to_string(),
            name: "Flux Dev".to_string(),
            asset_type: AssetType::Checkpoint,
            variants: vec![],
            requires: vec![Dependency {
                id: "t5-xxl".to_string(),
                dep_type: AssetType::TextEncoder,
                reason: Some("Text encoder".to_string()),
                optional_variant: Some("t5-xxl-fp8".to_string()),
            }],
            ..simple_manifest("flux-dev", vec![])
        };

        let index = make_index(vec![encoder, checkpoint]);
        let installed = HashSet::new();
        let plan = resolve("flux-dev", None, &index, &installed).unwrap();

        assert_eq!(plan.items.len(), 2);
        assert_eq!(plan.items[0].manifest.id, "t5-xxl");
        // variant auto-selected later — resolver does not propagate optional_variant
        assert_eq!(plan.items[0].variant_id, None);
        assert_eq!(plan.items[1].manifest.id, "flux-dev");
    }

    #[test]
    fn test_resolve_skips_dep_when_optional_variant_installed() {
        // When the alternative model (optional_variant) is already installed,
        // the primary dep should be skipped entirely.
        let encoder = Manifest {
            id: "t5-xxl".to_string(),
            name: "T5-XXL".to_string(),
            asset_type: AssetType::TextEncoder,
            variants: vec![],
            requires: vec![],
            ..simple_manifest("t5-xxl", vec![])
        };

        let checkpoint = Manifest {
            id: "flux-dev".to_string(),
            name: "Flux Dev".to_string(),
            asset_type: AssetType::Checkpoint,
            variants: vec![],
            requires: vec![Dependency {
                id: "t5-xxl".to_string(),
                dep_type: AssetType::TextEncoder,
                reason: Some("Text encoder".to_string()),
                optional_variant: Some("t5-xxl-fp8".to_string()),
            }],
            ..simple_manifest("flux-dev", vec![])
        };

        let index = make_index(vec![encoder, checkpoint]);
        // The alternative model is installed — dep should be skipped
        let installed: HashSet<String> = ["t5-xxl-fp8".to_string()].into();
        let plan = resolve("flux-dev", None, &index, &installed).unwrap();

        // Only flux-dev should be in the plan (t5-xxl skipped)
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].manifest.id, "flux-dev");
    }

    #[test]
    fn test_resolve_not_found() {
        let index = make_index(vec![]);
        let installed = HashSet::new();
        let result = resolve("nonexistent", None, &index, &installed);
        assert!(result.is_err());
    }
}
