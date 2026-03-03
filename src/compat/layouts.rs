use std::path::PathBuf;

use crate::core::manifest::AssetType;

/// ComfyUI folder layout
pub fn comfyui(asset_type: &AssetType) -> PathBuf {
    match asset_type {
        AssetType::Checkpoint => PathBuf::from("models/checkpoints"),
        AssetType::DiffusionModel => PathBuf::from("models/diffusion_models"),
        AssetType::Lora => PathBuf::from("models/loras"),
        AssetType::Vae => PathBuf::from("models/vae"),
        AssetType::TextEncoder => PathBuf::from("models/text_encoders"),
        AssetType::Controlnet => PathBuf::from("models/controlnet"),
        AssetType::Upscaler => PathBuf::from("models/upscale_models"),
        AssetType::Embedding => PathBuf::from("models/embeddings"),
        AssetType::Ipadapter => PathBuf::from("models/ipadapter"),
        AssetType::Segmentation => PathBuf::from("models/BiRefNet"),
        AssetType::VisionLanguage => PathBuf::from("models/vision_language"),
        AssetType::Recipe => PathBuf::from("recipes"),
    }
}

/// A1111 / SD WebUI folder layout
pub fn a1111(asset_type: &AssetType) -> PathBuf {
    match asset_type {
        AssetType::Checkpoint => PathBuf::from("models/Stable-diffusion"),
        AssetType::DiffusionModel => PathBuf::from("models/diffusion_models"),
        AssetType::Lora => PathBuf::from("models/Lora"),
        AssetType::Vae => PathBuf::from("models/VAE"),
        AssetType::TextEncoder => PathBuf::from("models/text_encoder"),
        AssetType::Controlnet => PathBuf::from("extensions/sd-webui-controlnet/models"),
        AssetType::Upscaler => PathBuf::from("models/ESRGAN"),
        AssetType::Embedding => PathBuf::from("embeddings"),
        AssetType::Ipadapter => PathBuf::from("models/ipadapter"),
        AssetType::Segmentation => PathBuf::from("models/BiRefNet"),
        AssetType::VisionLanguage => PathBuf::from("models/vision_language"),
        AssetType::Recipe => PathBuf::from("recipes"),
    }
}

/// InvokeAI folder layout
pub fn invokeai(asset_type: &AssetType) -> PathBuf {
    match asset_type {
        AssetType::Checkpoint => PathBuf::from("models/sd"),
        AssetType::DiffusionModel => PathBuf::from("models/diffusion_models"),
        AssetType::Lora => PathBuf::from("models/lora"),
        AssetType::Vae => PathBuf::from("models/vae"),
        AssetType::TextEncoder => PathBuf::from("models/text_encoder"),
        AssetType::Controlnet => PathBuf::from("models/controlnet"),
        AssetType::Upscaler => PathBuf::from("models/upscaler"),
        AssetType::Embedding => PathBuf::from("models/embedding"),
        AssetType::Ipadapter => PathBuf::from("models/ip_adapter"),
        AssetType::Segmentation => PathBuf::from("models/segmentation"),
        AssetType::VisionLanguage => PathBuf::from("models/vision_language"),
        AssetType::Recipe => PathBuf::from("recipes"),
    }
}
