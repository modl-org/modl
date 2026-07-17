use anyhow::Result;
use std::path::PathBuf;

use super::db::Database;
use super::job::LoraRef;
use super::model_family;

/// Resolve a base model path from installed models (match by ID, display name, or family alias).
pub fn resolve_base_model_path(base_model: &str, db: &Database) -> Option<String> {
    let installed = db.list_installed(None).ok()?;
    let gen_types = ["checkpoint", "diffusion_model"];

    // Exact match by ID or display name
    for model in &installed {
        if (model.name == base_model || model.id == base_model)
            && gen_types.contains(&model.asset_type.as_str())
        {
            return Some(model.store_path.clone());
        }
    }

    // Fuzzy match via model_family (e.g. "sdxl" → "sdxl-base-1.0")
    if let Some(family_info) = model_family::resolve_model(base_model) {
        let family_id = &family_info.id;
        for model in &installed {
            if gen_types.contains(&model.asset_type.as_str())
                && (model.id == *family_id
                    || model.id.contains(family_id.as_str())
                    || family_id.contains(&*model.id))
            {
                return Some(model.store_path.clone());
            }
        }
    }

    None
}

/// Resolve a LoRA name to its store path by looking in the DB.
///
/// Accepts either a direct path to a `.safetensors` file or a name/ID lookup.
pub fn resolve_lora(name: &str, weight: f32, db: &Database) -> Result<Option<LoraRef>> {
    // Check if the name is a direct path to a .safetensors file
    let path = PathBuf::from(name);
    if path.exists() && path.extension().is_some_and(|e| e == "safetensors") {
        return Ok(Some(LoraRef {
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            path: path.to_string_lossy().to_string(),
            weight,
        }));
    }

    // Look up in installed models (match by ID or display name)
    let installed = db.list_installed(None)?;
    for model in &installed {
        if (model.name == name || model.id == name) && model.asset_type == "lora" {
            return Ok(Some(LoraRef {
                name: model.name.clone(),
                path: model.store_path.clone(),
                weight,
            }));
        }
    }

    Ok(None)
}

/// Size presets: aspect ratio → (width, height)
pub const SIZE_PRESETS: &[(&str, u32, u32)] = &[
    ("1:1", 1024, 1024),
    ("16:9", 1344, 768),
    ("9:16", 768, 1344),
    ("4:3", 1152, 896),
    ("3:4", 896, 1152),
];

/// Resolve a size preset string to (width, height), scaled to a model's
/// native resolution. Presets are defined at 1024-native; a 512-native model
/// (SD 1.5) gets 1:1 → 512×512, 16:9 → 704×384, etc. Explicit WxH is used
/// as-is — the user asked for those exact dimensions.
pub fn resolve_size_for_model(size: &str, native_resolution: u32) -> Result<(u32, u32)> {
    for &(name, w, h) in SIZE_PRESETS {
        if size == name {
            return Ok((
                scale_to_native(w, native_resolution),
                scale_to_native(h, native_resolution),
            ));
        }
    }
    resolve_size(size)
}

/// Scale a 1024-reference dimension to a model's native resolution,
/// snapped to the nearest multiple of 64 (UNet/DiT dimension constraint).
fn scale_to_native(dim: u32, native: u32) -> u32 {
    (((dim * native / 1024) + 32) / 64).max(1) * 64
}

/// Resolve a size preset string to (width, height).
pub fn resolve_size(size: &str) -> Result<(u32, u32)> {
    for &(name, w, h) in SIZE_PRESETS {
        if size == name {
            return Ok((w, h));
        }
    }

    if let Some((w, h)) = size.split_once('x') {
        let w: u32 = w
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid width in size"))?;
        let h: u32 = h
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid height in size"))?;
        return Ok((w, h));
    }

    anyhow::bail!(
        "Unknown size: {size}. Use a preset (1:1, 16:9, 9:16, 4:3, 3:4) or WxH (e.g. 1024x1024)"
    );
}
