use anyhow::{Result, bail};

use crate::core::job::{LoraType, Optimizer, Preset, TrainingParams};

/// Dataset statistics needed for preset resolution
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub image_count: u32,
    #[allow(dead_code)]
    pub caption_coverage: f32,
}

/// GPU context for auto-settings
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub vram_mb: u64,
}

/// Which base model family (affects resolution)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseModelFamily {
    Flux,
    FluxSchnell,
    ZImage,
    Chroma,
    QwenImage,
    Sdxl,
    Sd15,
}

impl BaseModelFamily {
    /// Infer family from a base model ID string
    pub fn from_model_id(id: &str) -> Result<Self> {
        let lower = id.to_lowercase();
        if lower.contains("z-image") || lower.contains("z_image") || lower.contains("zimage") {
            Ok(Self::ZImage)
        } else if lower.contains("chroma") {
            Ok(Self::Chroma)
        } else if lower.contains("qwen-image") || lower.contains("qwen_image") {
            Ok(Self::QwenImage)
        } else if lower.contains("flux") && lower.contains("schnell") {
            Ok(Self::FluxSchnell)
        } else if lower.contains("flux") {
            Ok(Self::Flux)
        } else if lower.contains("sdxl") || lower.contains("xl") {
            Ok(Self::Sdxl)
        } else if lower.contains("sd-1.5") || lower.contains("sd15") || lower.contains("1.5") {
            Ok(Self::Sd15)
        } else {
            bail!(
                "Unknown base model family for '{id}'. Supported families include:\n\
                 - flux-dev / flux-schnell\n\
                 - z-image / z-image-turbo\n\
                 - chroma\n\
                 - qwen-image\n\
                 - sdxl-base-1.0\n\
                 - sd-1.5\n\n\
                 If this is a custom model path, include a known family token in the base id \
                 (for example: Qwen/Qwen-Image, stabilityai/stable-diffusion-xl-base-1.0)."
            )
        }
    }

    pub fn default_resolution(&self) -> u32 {
        match self {
            Self::Flux
            | Self::FluxSchnell
            | Self::Sdxl
            | Self::ZImage
            | Self::Chroma
            | Self::QwenImage => 1024,
            Self::Sd15 => 512,
        }
    }
}

/// Pure logic: resolve training parameters from preset + context.
/// No I/O — only takes values and returns computed params.
pub fn resolve_params(
    preset: Preset,
    lora_type: LoraType,
    dataset: &DatasetStats,
    gpu: Option<&GpuContext>,
    base_model: &str,
    trigger_word: &str,
) -> Result<TrainingParams> {
    let family = BaseModelFamily::from_model_id(base_model)?;
    let resolution = family.default_resolution();
    let vram_mb = gpu.map(|g| g.vram_mb).unwrap_or(0);
    let quantize = vram_mb > 0 && vram_mb < 40_000;
    let img_count = dataset.image_count;

    // Z-Image trains significantly faster than Flux/SDXL (~1.3s/iter on 5090).
    // For distilled models (ZImage turbo), keep LR at 1e-4 max to avoid
    // breaking distillation. Style LoRAs need 3000-5000 steps per Ostris.
    let is_zimage = matches!(family, BaseModelFamily::ZImage);

    let (steps, rank, learning_rate) = match (preset, lora_type) {
        // --- Style presets: high rank, many more steps ---
        // Style requires much longer training than character/subject.
        // Reference: ~15 epochs with repeats for proper style transfer.
        (Preset::Quick, LoraType::Style) if is_zimage => {
            // Z-Image style trains fast; 3000 steps is a good starting point
            let steps = compute_steps(img_count, 25, 3000, 5000);
            (steps, 16, 1e-4)
        }
        (Preset::Standard, LoraType::Style) if is_zimage => {
            let steps = compute_steps(img_count, 50, 3500, 5000);
            (steps, 16, 1e-4)
        }
        (Preset::Advanced, LoraType::Style) if is_zimage => {
            let steps = compute_steps(img_count, 75, 4000, 6000);
            (steps, 32, 1e-4)
        }
        (Preset::Quick, LoraType::Style) => {
            // Quick style: ~5 epochs worth. batch=2, repeats=10
            // epoch ≈ img_count * 10 / 2 steps, so 5 epochs ≈ img * 25
            let steps = compute_steps(img_count, 25, 4000, 15000);
            (steps, 32, 1e-4)
        }
        (Preset::Standard, LoraType::Style) => {
            // Standard style: ~15 epochs
            let steps = compute_steps(img_count, 75, 8000, 40000);
            (steps, 64, 1e-4)
        }
        (Preset::Advanced, LoraType::Style) => {
            let steps = compute_steps(img_count, 75, 8000, 50000);
            (steps, 128, 1e-4)
        }

        // --- Character / Object presets ---
        (Preset::Quick, _) if is_zimage => {
            let steps = compute_steps(img_count, 100, 1000, 1500);
            (steps, 8, 1e-4)
        }
        (Preset::Standard, _) if is_zimage => {
            let steps = compute_steps(img_count, 150, 1500, 3000);
            let rank = if img_count < 20 { 16 } else { 32 };
            (steps, rank, 1e-4)
        }
        (Preset::Quick, _) => {
            let steps = compute_steps(img_count, 150, 1000, 1500);
            (steps, 8, 1e-4)
        }
        (Preset::Standard, _) => {
            let steps = compute_steps(img_count, 200, 2000, 4000);
            let rank = if img_count < 20 { 16 } else { 32 };
            let lr = if img_count < 10 { 5e-5 } else { 1e-4 };
            (steps, rank, lr)
        }
        (Preset::Advanced, _) => {
            let steps = compute_steps(img_count, 200, 2000, 4000);
            (steps, 16, 1e-4)
        }
    };

    Ok(TrainingParams {
        preset,
        lora_type,
        trigger_word: trigger_word.to_string(),
        steps,
        rank,
        learning_rate,
        optimizer: Optimizer::Adamw8bit,
        resolution,
        seed: None,
        quantize,
        batch_size: 0,              // 0 = let adapter choose per lora_type
        num_repeats: 0,             // 0 = let adapter choose per lora_type
        caption_dropout_rate: -1.0, // negative = let adapter choose
        resume_from: None,
    })
}

/// steps_per_image * images, clamped to [min, max]
fn compute_steps(image_count: u32, per_image: u32, min: u32, max: u32) -> u32 {
    let raw = image_count.saturating_mul(per_image);
    raw.clamp(min, max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(vram_mb: u64) -> GpuContext {
        GpuContext { vram_mb }
    }

    fn dataset(image_count: u32) -> DatasetStats {
        DatasetStats {
            image_count,
            caption_coverage: 1.0,
        }
    }

    fn resolve(
        preset: Preset,
        lora_type: LoraType,
        dataset: &DatasetStats,
        gpu: Option<&GpuContext>,
        base_model: &str,
        trigger_word: &str,
    ) -> TrainingParams {
        resolve_params(preset, lora_type, dataset, gpu, base_model, trigger_word)
            .expect("preset resolution should succeed")
    }

    // --- Quick preset ---

    #[test]
    fn quick_min_steps() {
        let p = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(3),
            Some(&gpu(24576)),
            "flux-schnell",
            "OHWX",
        );
        // 3 * 150 = 450, clamped to min 1000
        assert_eq!(p.steps, 1000);
        assert_eq!(p.rank, 8);
        assert!((p.learning_rate - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn quick_max_steps() {
        let p = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(50),
            Some(&gpu(24576)),
            "flux-schnell",
            "OHWX",
        );
        // 50 * 150 = 7500, clamped to max 1500
        assert_eq!(p.steps, 1500);
    }

    #[test]
    fn quick_normal_scaling() {
        let p = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(8),
            Some(&gpu(24576)),
            "flux-schnell",
            "OHWX",
        );
        // 8 * 150 = 1200
        assert_eq!(p.steps, 1200);
    }

    // --- Standard preset ---

    #[test]
    fn standard_small_dataset() {
        let p = resolve(
            Preset::Standard,
            LoraType::Character,
            &dataset(8),
            Some(&gpu(24576)),
            "flux-dev",
            "OHWX",
        );
        // 8 * 200 = 1600, min 2000 → 2000
        assert_eq!(p.steps, 2000);
        assert_eq!(p.rank, 16); // < 20 images → 16
        assert!((p.learning_rate - 5e-5).abs() < 1e-10); // < 10 images → 5e-5
    }

    #[test]
    fn standard_large_dataset() {
        let p = resolve(
            Preset::Standard,
            LoraType::Character,
            &dataset(25),
            Some(&gpu(24576)),
            "flux-dev",
            "OHWX",
        );
        // 25 * 200 = 5000, max 4000
        assert_eq!(p.steps, 4000);
        assert_eq!(p.rank, 32); // >= 20 images → 32
        assert!((p.learning_rate - 1e-4).abs() < 1e-10); // >= 10 images → 1e-4
    }

    #[test]
    fn standard_medium_dataset() {
        let p = resolve(
            Preset::Standard,
            LoraType::Character,
            &dataset(15),
            Some(&gpu(24576)),
            "flux-dev",
            "OHWX",
        );
        // 15 * 200 = 3000
        assert_eq!(p.steps, 3000);
        assert_eq!(p.rank, 16); // < 20
        assert!((p.learning_rate - 1e-4).abs() < 1e-10); // >= 10
    }

    // --- Advanced preset ---

    #[test]
    fn advanced_defaults() {
        let p = resolve(
            Preset::Advanced,
            LoraType::Character,
            &dataset(10),
            Some(&gpu(24576)),
            "flux-schnell",
            "OHWX",
        );
        assert_eq!(p.rank, 16);
        assert!((p.learning_rate - 1e-4).abs() < 1e-10);
    }

    // --- Auto settings ---

    #[test]
    fn quantize_when_low_vram() {
        let p = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            Some(&gpu(24576)),
            "flux-schnell",
            "OHWX",
        );
        assert!(p.quantize); // 24GB < 40GB
    }

    #[test]
    fn no_quantize_when_high_vram() {
        let p = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            Some(&gpu(49152)),
            "flux-schnell",
            "OHWX",
        );
        assert!(!p.quantize); // 48GB >= 40GB
    }

    #[test]
    fn no_quantize_when_no_gpu_info() {
        let p = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            None,
            "flux-schnell",
            "OHWX",
        );
        assert!(!p.quantize); // no GPU info → don't quantize
    }

    #[test]
    fn resolution_from_base_model() {
        let flux = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            None,
            "flux-dev",
            "OHWX",
        );
        assert_eq!(flux.resolution, 1024);

        let sdxl = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            None,
            "stable-diffusion-xl",
            "OHWX",
        );
        assert_eq!(sdxl.resolution, 1024);

        let sd15 = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            None,
            "sd-1.5",
            "OHWX",
        );
        assert_eq!(sd15.resolution, 512);
    }

    #[test]
    fn optimizer_always_adamw8bit() {
        let p = resolve(
            Preset::Standard,
            LoraType::Character,
            &dataset(10),
            Some(&gpu(24576)),
            "flux-schnell",
            "OHWX",
        );
        assert_eq!(p.optimizer, Optimizer::Adamw8bit);
    }

    #[test]
    fn trigger_word_passthrough() {
        let p = resolve(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            None,
            "flux-schnell",
            "MY_TRIGGER",
        );
        assert_eq!(p.trigger_word, "MY_TRIGGER");
    }

    #[test]
    fn unknown_family_returns_helpful_error() {
        let err = resolve_params(
            Preset::Quick,
            LoraType::Character,
            &dataset(10),
            None,
            "totally-unknown-model",
            "OHWX",
        )
        .expect_err("unknown base family should error");

        let msg = err.to_string();
        assert!(msg.contains("Unknown base model family"));
        assert!(msg.contains("qwen-image"));
        assert!(msg.contains("sdxl-base-1.0"));
    }
}
