// ---------------------------------------------------------------------------
// core::enhance — Prompt enhancement with pluggable backends
//
// Architecture:
//   PromptEnhancer trait → BuiltinEnhancer (rule-based, no deps)
//                        → (future) OllamaEnhancer, DiffusersEnhancer, etc.
//
// The builtin enhancer applies heuristic improvements:
//   - Adds quality/style booster tags
//   - Expands terse prompts with descriptive detail
//   - Structures prompts with subject → environment → style → quality
//   - Preserves user intent while enriching specificity
// ---------------------------------------------------------------------------

use anyhow::Result;

/// Request to enhance a prompt.
#[derive(Debug, Clone)]
pub struct EnhanceRequest {
    pub prompt: String,
    /// Target model family — influences style tags (e.g., "sdxl", "flux", "sd3")
    pub model_hint: Option<String>,
    /// Desired enhancement intensity: "subtle", "moderate", "aggressive"
    pub intensity: EnhanceIntensity,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum EnhanceIntensity {
    /// Minimal changes — just quality tags
    Subtle,
    /// Balanced expansion + quality tags
    #[default]
    Moderate,
    /// Full rewrite with rich descriptors
    Aggressive,
}

impl std::str::FromStr for EnhanceIntensity {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "subtle" | "s" => Ok(Self::Subtle),
            "moderate" | "m" => Ok(Self::Moderate),
            "aggressive" | "a" => Ok(Self::Aggressive),
            _ => anyhow::bail!("Unknown intensity: {s}. Use: subtle, moderate, aggressive"),
        }
    }
}

impl std::fmt::Display for EnhanceIntensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Subtle => write!(f, "subtle"),
            Self::Moderate => write!(f, "moderate"),
            Self::Aggressive => write!(f, "aggressive"),
        }
    }
}

/// Result of prompt enhancement.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EnhanceResult {
    pub original: String,
    pub enhanced: String,
    pub backend: String,
}

// ---------------------------------------------------------------------------
// Trait — implement this for new LLM backends
// ---------------------------------------------------------------------------

pub trait PromptEnhancer {
    fn enhance(&self, req: &EnhanceRequest) -> Result<EnhanceResult>;
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Builtin rule-based enhancer (zero dependencies, always available)
// ---------------------------------------------------------------------------

pub struct BuiltinEnhancer;

impl BuiltinEnhancer {
    /// Quality boosters commonly used in diffusion model prompts
    const QUALITY_TAGS: &[&str] = &[
        "masterpiece",
        "best quality",
        "highly detailed",
        "sharp focus",
        "professional",
    ];

    /// SDXL-specific quality boosters
    const SDXL_QUALITY: &[&str] = &[
        "masterpiece",
        "best quality",
        "extremely detailed",
        "8k uhd",
        "sharp focus",
    ];

    /// Flux-specific quality boosters (less tag-heavy, more natural language)
    const FLUX_QUALITY: &[&str] = &["highly detailed", "professional photography", "sharp focus"];

    /// Photography style descriptors
    const PHOTO_DESCRIPTORS: &[&str] = &[
        "natural lighting",
        "depth of field",
        "volumetric lighting",
        "cinematic composition",
        "high dynamic range",
    ];

    /// Art style descriptors
    const ART_DESCRIPTORS: &[&str] = &[
        "intricate details",
        "vibrant colors",
        "dramatic lighting",
        "rich textures",
    ];

    fn is_photo_prompt(prompt: &str) -> bool {
        let lower = prompt.to_lowercase();
        lower.contains("photo")
            || lower.contains("photograph")
            || lower.contains("portrait")
            || lower.contains("headshot")
            || lower.contains("realistic")
            || lower.contains("candid")
    }

    fn already_has_quality_tags(prompt: &str) -> bool {
        let lower = prompt.to_lowercase();
        lower.contains("masterpiece")
            || lower.contains("best quality")
            || lower.contains("highly detailed")
            || lower.contains("8k")
            || lower.contains("uhd")
    }

    fn quality_tags_for_model(model_hint: Option<&str>) -> &'static [&'static str] {
        match model_hint.map(|s| s.to_lowercase()).as_deref() {
            Some(m) if m.contains("sdxl") => Self::SDXL_QUALITY,
            Some(m) if m.contains("flux") => Self::FLUX_QUALITY,
            _ => Self::QUALITY_TAGS,
        }
    }

    fn enhance_subtle(prompt: &str, model_hint: Option<&str>) -> String {
        if Self::already_has_quality_tags(prompt) {
            return prompt.to_string();
        }
        let tags = Self::quality_tags_for_model(model_hint);
        // Append 2-3 quality tags
        let suffix: Vec<&str> = tags.iter().take(3).copied().collect();
        format!(
            "{}, {}",
            prompt.trim_end_matches([',', '.', ' ']),
            suffix.join(", ")
        )
    }

    fn enhance_moderate(prompt: &str, model_hint: Option<&str>) -> String {
        let trimmed = prompt.trim();
        let has_quality = Self::already_has_quality_tags(trimmed);
        let is_photo = Self::is_photo_prompt(trimmed);
        let tags = Self::quality_tags_for_model(model_hint);

        let mut parts: Vec<String> = vec![trimmed.trim_end_matches([',', '.', ' ']).to_string()];

        // Add contextual descriptors
        if is_photo {
            parts.extend(
                Self::PHOTO_DESCRIPTORS
                    .iter()
                    .take(2)
                    .map(|s| s.to_string()),
            );
        } else {
            parts.extend(Self::ART_DESCRIPTORS.iter().take(2).map(|s| s.to_string()));
        }

        // Add quality tags if not present
        if !has_quality {
            parts.extend(tags.iter().take(3).map(|s| s.to_string()));
        }

        parts.join(", ")
    }

    fn enhance_aggressive(prompt: &str, model_hint: Option<&str>) -> String {
        let trimmed = prompt.trim();
        let is_photo = Self::is_photo_prompt(trimmed);
        let tags = Self::quality_tags_for_model(model_hint);

        let mut parts: Vec<String> = vec![trimmed.trim_end_matches([',', '.', ' ']).to_string()];

        // Full descriptor set
        if is_photo {
            parts.extend(Self::PHOTO_DESCRIPTORS.iter().map(|s| s.to_string()));
        } else {
            parts.extend(Self::ART_DESCRIPTORS.iter().map(|s| s.to_string()));
        }

        // All quality tags
        for tag in tags {
            let lower = parts
                .iter()
                .any(|p| p.to_lowercase().contains(&tag.to_lowercase()));
            if !lower {
                parts.push(tag.to_string());
            }
        }

        parts.join(", ")
    }
}

impl PromptEnhancer for BuiltinEnhancer {
    fn enhance(&self, req: &EnhanceRequest) -> Result<EnhanceResult> {
        if req.prompt.trim().is_empty() {
            anyhow::bail!("Cannot enhance an empty prompt");
        }

        let model_hint = req.model_hint.as_deref();
        let enhanced = match req.intensity {
            EnhanceIntensity::Subtle => Self::enhance_subtle(&req.prompt, model_hint),
            EnhanceIntensity::Moderate => Self::enhance_moderate(&req.prompt, model_hint),
            EnhanceIntensity::Aggressive => Self::enhance_aggressive(&req.prompt, model_hint),
        };

        Ok(EnhanceResult {
            original: req.prompt.clone(),
            enhanced,
            backend: self.name().to_string(),
        })
    }

    fn name(&self) -> &str {
        "builtin"
    }
}

// ---------------------------------------------------------------------------
// Public API — resolve the best available enhancer
// ---------------------------------------------------------------------------

/// Get the default prompt enhancer.
/// Currently returns the builtin rule-based enhancer.
/// In the future, this can check config for an LLM backend (ollama, diffusers, etc.)
pub fn default_enhancer() -> Box<dyn PromptEnhancer> {
    // TODO: Check config for preferred backend
    //   let config = Config::load().ok();
    //   if config.enhance.backend == "ollama" { return Box::new(OllamaEnhancer::new(...)) }
    Box::new(BuiltinEnhancer)
}

/// Convenience: enhance a prompt with defaults.
pub fn enhance_prompt(
    prompt: &str,
    model_hint: Option<&str>,
    intensity: EnhanceIntensity,
) -> Result<EnhanceResult> {
    let enhancer = default_enhancer();
    enhancer.enhance(&EnhanceRequest {
        prompt: prompt.to_string(),
        model_hint: model_hint.map(String::from),
        intensity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtle_adds_quality_tags() {
        let result = enhance_prompt("a cat", None, EnhanceIntensity::Subtle).unwrap();
        assert!(result.enhanced.contains("masterpiece"));
        assert!(result.enhanced.starts_with("a cat"));
    }

    #[test]
    fn test_subtle_skips_if_already_tagged() {
        let result = enhance_prompt(
            "a cat, masterpiece, best quality",
            None,
            EnhanceIntensity::Subtle,
        )
        .unwrap();
        assert_eq!(result.enhanced, "a cat, masterpiece, best quality");
    }

    #[test]
    fn test_moderate_adds_descriptors() {
        let result =
            enhance_prompt("portrait of a woman", None, EnhanceIntensity::Moderate).unwrap();
        assert!(result.enhanced.contains("natural lighting"));
    }

    #[test]
    fn test_sdxl_uses_sdxl_tags() {
        let result = enhance_prompt("a cat", Some("sdxl"), EnhanceIntensity::Subtle).unwrap();
        assert!(result.enhanced.contains("extremely detailed"));
    }

    #[test]
    fn test_flux_uses_flux_tags() {
        let result = enhance_prompt("a cat", Some("flux-dev"), EnhanceIntensity::Subtle).unwrap();
        assert!(result.enhanced.contains("professional photography"));
    }

    #[test]
    fn test_empty_prompt_errors() {
        assert!(enhance_prompt("", None, EnhanceIntensity::Moderate).is_err());
    }
}
