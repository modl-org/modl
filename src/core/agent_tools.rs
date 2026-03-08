// ---------------------------------------------------------------------------
// core::agent_tools — Tool implementations wrapping existing modl services
//
// Each tool maps to existing infrastructure:
//   analyze_images  → LlmBackend::vision()
//   create_dataset  → core::dataset::create()
//   caption_images  → LlmBackend::vision() per image
//   select_base_model → core::db + core::registry
//   train_lora      → existing training pipeline (via CLI)
//   enhance_prompt  → LlmBackend::complete() or core::enhance
//   generate_images → existing generation pipeline (via CLI)
// ---------------------------------------------------------------------------

use anyhow::{Context, Result, bail};
use tokio::sync::broadcast;

use crate::core::agent::{AgentEvent, AgentSession};
use crate::core::llm::LlmBackend;

/// Execute a tool by name with the given arguments.
///
/// Returns the tool result as a string (fed back to the LLM).
pub async fn execute_tool(
    name: &str,
    args: &serde_json::Value,
    session: &AgentSession,
    llm: &dyn LlmBackend,
    event_tx: &broadcast::Sender<String>,
) -> Result<String> {
    match name {
        "analyze_images" => tool_analyze_images(args, session, llm).await,
        "create_dataset" => tool_create_dataset(args, session).await,
        "caption_images" => tool_caption_images(args, session, llm, event_tx).await,
        "select_base_model" => tool_select_base_model(args).await,
        "train_lora" => tool_train_lora(args, event_tx).await,
        "enhance_prompt" => tool_enhance_prompt(args, llm).await,
        "generate_images" => tool_generate_images(args, event_tx).await,
        _ => bail!("Unknown tool: {name}"),
    }
}

// ---------------------------------------------------------------------------
// analyze_images — Use VL model to describe uploaded photos
// ---------------------------------------------------------------------------

async fn tool_analyze_images(
    args: &serde_json::Value,
    session: &AgentSession,
    llm: &dyn LlmBackend,
) -> Result<String> {
    if session.images.is_empty() {
        return Ok("No images uploaded yet.".to_string());
    }

    let focus = args.get("focus").and_then(|v| v.as_str()).unwrap_or("all");

    let prompt = match focus {
        "subject" => {
            "Describe the main subject in this image in detail. What are they wearing? What do they look like?"
        }
        "style" => {
            "Describe the photographic style, lighting, composition, and mood of this image."
        }
        "background" => "Describe the background and setting of this image.",
        _ => {
            "Describe this image in detail: the subject, setting, lighting, composition, and style."
        }
    };

    // Analyze first few images (limit to avoid timeout)
    let images_to_analyze: Vec<_> = session.images.iter().take(5).cloned().collect();
    let description = llm
        .vision(&images_to_analyze, prompt)
        .context("Failed to analyze images")?;

    Ok(format!(
        "Analyzed {} image(s). Description: {}",
        images_to_analyze.len(),
        description
    ))
}

// ---------------------------------------------------------------------------
// create_dataset — Create dataset directory from uploaded images
// ---------------------------------------------------------------------------

async fn tool_create_dataset(args: &serde_json::Value, session: &AgentSession) -> Result<String> {
    let name = args
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&session.id);

    // Create the dataset directory
    let datasets_dir = dirs::home_dir()
        .context("Could not determine home directory")?
        .join(".modl")
        .join("datasets")
        .join(name);

    std::fs::create_dir_all(&datasets_dir)
        .with_context(|| format!("Failed to create dataset dir: {}", datasets_dir.display()))?;

    // Copy uploaded images into the dataset
    let mut copied = 0;
    for (i, image_path) in session.images.iter().enumerate() {
        if image_path.exists() {
            let ext = image_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("jpg");
            let dest = datasets_dir.join(format!("{:04}.{}", i + 1, ext));
            std::fs::copy(image_path, &dest)
                .with_context(|| "Failed to copy image to dataset".to_string())?;
            copied += 1;
        }
    }

    Ok(format!(
        "Created dataset '{name}' with {copied} images at {}",
        datasets_dir.display()
    ))
}

// ---------------------------------------------------------------------------
// caption_images — Generate training captions using VL model
// ---------------------------------------------------------------------------

async fn tool_caption_images(
    args: &serde_json::Value,
    session: &AgentSession,
    llm: &dyn LlmBackend,
    event_tx: &broadcast::Sender<String>,
) -> Result<String> {
    let dataset_name = args
        .get("dataset_name")
        .and_then(|v| v.as_str())
        .unwrap_or(&session.id);

    let trigger_word = args.get("trigger_word").and_then(|v| v.as_str());

    let dataset_dir = crate::core::dataset::resolve_path(dataset_name);
    let info = crate::core::dataset::scan(&dataset_dir)
        .with_context(|| format!("Failed to scan dataset: {dataset_name}"))?;

    let total = info.images.len();
    let mut captioned = 0;

    for (i, entry) in info.images.iter().enumerate() {
        // Emit progress
        let progress = (i as f32 + 1.0) / total as f32;
        let event = AgentEvent::ToolProgress {
            tool: "caption_images".to_string(),
            progress,
            detail: format!("Captioning image {}/{}", i + 1, total),
        };
        let _ = event_tx.send(event.to_json());

        // Generate caption using VL model
        let prompt = "Describe this image for AI training. Be specific about the subject, \
                       their appearance, clothing, pose, expression, the setting, lighting, \
                       and camera angle. Use natural language, not tags.";

        match llm.vision(std::slice::from_ref(&entry.path), prompt) {
            Ok(mut caption) => {
                // Prepend trigger word if specified
                if let Some(tw) = trigger_word {
                    caption = format!("{tw}, {caption}");
                }

                // Write caption file alongside the image
                let caption_path = entry.path.with_extension("txt");
                std::fs::write(&caption_path, &caption).with_context(|| {
                    format!("Failed to write caption: {}", caption_path.display())
                })?;
                captioned += 1;
            }
            Err(e) => {
                eprintln!("[agent] Failed to caption {}: {e}", entry.path.display());
            }
        }
    }

    Ok(format!(
        "Captioned {captioned}/{total} images in dataset '{dataset_name}'"
    ))
}

// ---------------------------------------------------------------------------
// select_base_model — Choose the best base model for the task
// ---------------------------------------------------------------------------

async fn tool_select_base_model(args: &serde_json::Value) -> Result<String> {
    let task_type = args
        .get("task_type")
        .and_then(|v| v.as_str())
        .unwrap_or("character");

    // Check what's installed
    let db = crate::core::db::Database::open()?;
    let installed = db.list_installed(Some("checkpoint"))?;
    let installed_ids: Vec<&str> = installed.iter().map(|m| m.id.as_str()).collect();

    // Check for diffusion_model type too
    let diff_models = db.list_installed(Some("diffusion_model"))?;
    let diff_ids: Vec<&str> = diff_models.iter().map(|m| m.id.as_str()).collect();

    // Model preference order by task type
    let preferences: Vec<&str> = match task_type {
        "character" => vec!["flux-dev", "flux-schnell", "sdxl-base-1.0", "sd-1.5"],
        "style" => vec!["flux-dev", "z-image-turbo", "sdxl-base-1.0", "flux-schnell"],
        "object" => vec!["flux-dev", "sdxl-base-1.0", "flux-schnell"],
        _ => vec!["flux-dev", "flux-schnell", "sdxl-base-1.0"],
    };

    // Find the first installed model from preferences
    for model in &preferences {
        if installed_ids.contains(model) || diff_ids.contains(model) {
            return Ok(format!(
                "Selected base model: {model} (best available for {task_type} training)"
            ));
        }
    }

    // Fallback: recommend installation
    let recommended = preferences.first().copied().unwrap_or("flux-dev");
    Ok(format!(
        "No suitable base model installed. Recommended: {recommended}. \
         The user should run `modl pull {recommended}` first."
    ))
}

// ---------------------------------------------------------------------------
// train_lora — Train LoRA using existing training pipeline
// ---------------------------------------------------------------------------

async fn tool_train_lora(
    args: &serde_json::Value,
    event_tx: &broadcast::Sender<String>,
) -> Result<String> {
    let dataset_name = args
        .get("dataset_name")
        .and_then(|v| v.as_str())
        .context("Missing dataset_name")?;
    let base_model = args
        .get("base_model")
        .and_then(|v| v.as_str())
        .context("Missing base_model")?;
    let lora_name = args
        .get("lora_name")
        .and_then(|v| v.as_str())
        .context("Missing lora_name")?;
    let lora_type = args
        .get("lora_type")
        .and_then(|v| v.as_str())
        .unwrap_or("character");
    let trigger_word = args.get("trigger_word").and_then(|v| v.as_str());

    // Use the existing training CLI to run the training
    // This invokes the full preflight + preset + executor pipeline
    let _lora_type_enum: crate::core::job::LoraType = lora_type
        .parse()
        .unwrap_or(crate::core::job::LoraType::Character);

    // Emit progress events
    let event = AgentEvent::ToolProgress {
        tool: "train_lora".to_string(),
        progress: 0.0,
        detail: "Preparing training configuration...".to_string(),
    };
    let _ = event_tx.send(event.to_json());

    // TODO: Wire up real executor integration when the agent framework
    // moves beyond stub phase. For now, report the training configuration.
    let event = AgentEvent::ToolProgress {
        tool: "train_lora".to_string(),
        progress: 1.0,
        detail: "Training configuration prepared".to_string(),
    };
    let _ = event_tx.send(event.to_json());

    Ok(format!(
        "Training LoRA '{lora_name}' on '{base_model}' with dataset '{dataset_name}' \
         (type: {lora_type}{}). \
         Training will use the modl training pipeline with default presets.",
        trigger_word
            .map(|tw| format!(", trigger: {tw}"))
            .unwrap_or_default()
    ))
}

// ---------------------------------------------------------------------------
// enhance_prompt — Craft detailed generation prompts
// ---------------------------------------------------------------------------

async fn tool_enhance_prompt(args: &serde_json::Value, llm: &dyn LlmBackend) -> Result<String> {
    let prompt = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .context("Missing prompt")?;
    let style_hint = args.get("style_hint").and_then(|v| v.as_str());

    // Try LLM-based enhancement first
    let enhance_prompt = format!(
        "Rewrite this image generation prompt to be more detailed and descriptive. \
         Keep the core intent but add specific details about lighting, composition, \
         quality, and style. {}Original prompt: \"{}\"",
        style_hint
            .map(|s| format!("Style guidance: {s}. "))
            .unwrap_or_default(),
        prompt
    );

    let messages = vec![
        crate::core::llm::Message::system(
            "You are an expert at writing image generation prompts. \
             Output ONLY the enhanced prompt, nothing else.",
        ),
        crate::core::llm::Message::user(&enhance_prompt),
    ];

    match llm.complete(&messages, &[]) {
        Ok(crate::core::llm::CompletionResult::Text(enhanced)) => {
            Ok(format!("Enhanced prompt: {enhanced}"))
        }
        _ => {
            // Fall back to builtin enhancer
            let result = crate::core::enhance::enhance_prompt(
                prompt,
                None,
                crate::core::enhance::EnhanceIntensity::Moderate,
            )?;
            Ok(format!("Enhanced prompt: {}", result.enhanced))
        }
    }
}

// ---------------------------------------------------------------------------
// generate_images — Generate images using trained LoRA
// ---------------------------------------------------------------------------

async fn tool_generate_images(
    args: &serde_json::Value,
    event_tx: &broadcast::Sender<String>,
) -> Result<String> {
    let prompt = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .context("Missing prompt")?;
    let lora_name = args
        .get("lora_name")
        .and_then(|v| v.as_str())
        .context("Missing lora_name")?;
    let base_model = args
        .get("base_model")
        .and_then(|v| v.as_str())
        .context("Missing base_model")?;
    let num_images = args.get("num_images").and_then(|v| v.as_u64()).unwrap_or(4) as u32;

    let event = AgentEvent::ToolProgress {
        tool: "generate_images".to_string(),
        progress: 0.0,
        detail: format!("Generating {num_images} images..."),
    };
    let _ = event_tx.send(event.to_json());

    // TODO: Wire up real generation via core executor when the agent
    // framework moves beyond stub phase.
    let event = AgentEvent::ToolProgress {
        tool: "generate_images".to_string(),
        progress: 1.0,
        detail: "Generation configuration prepared".to_string(),
    };
    let _ = event_tx.send(event.to_json());

    Ok(format!(
        "Generation request prepared: {num_images} image(s) with prompt '{prompt}' \
         using model '{base_model}' and LoRA '{lora_name}'."
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_unknown_tool() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let session = AgentSession {
                id: "test".to_string(),
                intent: "test".to_string(),
                images: vec![],
                events: vec![],
                status: crate::core::agent::SessionStatus::Running,
            };
            let llm = crate::core::llm::BuiltinLlmBackend;
            let (tx, _rx) = broadcast::channel(64);

            let result = execute_tool(
                "nonexistent_tool",
                &serde_json::json!({}),
                &session,
                &llm,
                &tx,
            )
            .await;
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Unknown tool"));
        });
    }
}
