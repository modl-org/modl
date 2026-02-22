pub mod layouts;

use std::path::{Path, PathBuf};

use crate::core::config::ToolType;
use crate::core::manifest::AssetType;

/// Get the subfolder path for a given asset type within a tool's directory
pub fn asset_folder(tool_type: &ToolType, asset_type: &AssetType) -> PathBuf {
    match tool_type {
        ToolType::Comfyui => layouts::comfyui(asset_type),
        ToolType::A1111 => layouts::a1111(asset_type),
        ToolType::Invokeai => layouts::invokeai(asset_type),
        ToolType::Custom => PathBuf::from("models"),
    }
}

/// Build the full symlink path for a model file within a tool installation
pub fn symlink_path(
    tool_root: &Path,
    tool_type: &ToolType,
    asset_type: &AssetType,
    file_name: &str,
) -> PathBuf {
    tool_root
        .join(asset_folder(tool_type, asset_type))
        .join(file_name)
}

/// Auto-detect installed tools by checking common locations
pub fn detect_tools() -> Vec<(ToolType, PathBuf)> {
    let mut found = Vec::new();
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return found,
    };

    // Common ComfyUI locations
    for dir in &["ComfyUI", "comfyui", ".comfyui"] {
        let path = home.join(dir);
        if path.join("models").exists() {
            found.push((ToolType::Comfyui, path));
            break;
        }
    }

    // Common A1111 locations
    for dir in &[
        "stable-diffusion-webui",
        "sd-webui",
        "automatic1111",
    ] {
        let path = home.join(dir);
        if path.join("models").exists() {
            found.push((ToolType::A1111, path));
            break;
        }
    }

    found
}
