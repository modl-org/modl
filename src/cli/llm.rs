use anyhow::{Result, bail};
use console::style;

use crate::core::llm::{self, CompletionResult, LlmBackend, Message};

/// `modl llm pull <model_id>` — Download an LLM model to the store.
pub async fn pull(model_id: &str) -> Result<()> {
    println!(
        "\n  {} Pulling LLM model: {}\n",
        style("→").cyan(),
        style(model_id).bold()
    );

    let store_dir = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("store")
        .join("llm")
        .join(model_id);

    std::fs::create_dir_all(&store_dir)?;

    // Check registry for the model
    let index = crate::core::registry::RegistryIndex::load_or_fetch().await?;
    if let Some(manifest) = index.find(model_id) {
        // Use existing download infrastructure
        println!(
            "  {} Found {} in registry",
            style("✓").green(),
            style(&manifest.name).bold()
        );

        // Extract download info from variant or file
        let variant = manifest.variants.first();
        let download_info: Option<(&str, &str, u64)> = variant
            .map(|v| (v.url.as_str(), v.file.as_str(), v.size))
            .or_else(|| {
                manifest
                    .file
                    .as_ref()
                    .map(|f| (f.url.as_str(), model_id, f.size))
            });

        if let Some((url, filename, size)) = download_info {
            let dest = store_dir.join(filename);
            if dest.exists() {
                println!(
                    "  {} Already downloaded: {}",
                    style("✓").green(),
                    dest.display()
                );
                return Ok(());
            }
            println!(
                "  {} Downloading {} ({:.1} GB)",
                style("↓").cyan(),
                filename,
                size as f64 / 1_073_741_824.0
            );
            crate::core::download::download_file(url, &dest, Some(size), None).await?;
            println!("\n  {} Model ready: {}", style("✓").green(), dest.display());
        } else {
            bail!("No downloadable file found for model {model_id}");
        }
    } else {
        // Not in registry — provide guidance
        println!(
            "  {} Model '{}' not found in registry.",
            style("!").yellow(),
            model_id
        );
        println!();
        println!("  Available LLM models:");
        println!(
            "    {}  ~3GB  Text reasoning (recommended)",
            style("qwen3.5-4b-instruct-q4").bold()
        );
        println!(
            "    {}   ~5GB  Vision-language (image understanding)",
            style("qwen3-vl-8b-instruct-q4").bold()
        );
        println!();
        println!(
            "  To add a model manually, place a .gguf file in:\n    {}",
            store_dir.display()
        );
    }

    Ok(())
}

/// `modl llm chat "prompt"` — Run text completion (or VL inference with --image).
pub async fn chat(
    prompt: &str,
    image: Option<&str>,
    cloud: bool,
    model: Option<&str>,
) -> Result<()> {
    let backend: Box<dyn LlmBackend> = if let Some(model_id) = model {
        llm::resolve_model(model_id, !cloud)?
    } else {
        llm::resolve_backend(cloud)?
    };

    println!(
        "  {} Using backend: {}\n",
        style("→").dim(),
        style(backend.name()).cyan()
    );

    if let Some(image_path) = image {
        // Vision-language mode
        let path = std::path::PathBuf::from(image_path);
        if !path.exists() {
            bail!("Image not found: {image_path}");
        }
        let result = backend.vision(&[path], prompt)?;
        println!("{result}");
    } else {
        // Text completion mode
        let messages = vec![
            Message::system(
                "You are a helpful AI assistant. Be concise and direct in your responses.",
            ),
            Message::user(prompt),
        ];
        let result = backend.complete(&messages, &[])?;
        match result {
            CompletionResult::Text(text) => println!("{text}"),
            CompletionResult::ToolCall { name, args, .. } => {
                println!(
                    "  {} Tool call: {}({})",
                    style("⚡").yellow(),
                    style(&name).bold(),
                    serde_json::to_string_pretty(&args)?
                );
            }
        }
    }

    Ok(())
}

/// `modl llm ls` — List available LLM models.
pub async fn list() -> Result<()> {
    let store_dir = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("store")
        .join("llm");

    println!("\n  {} Installed LLM models:\n", style("LLM").bold());

    if !store_dir.exists() {
        println!(
            "  {} No models installed. Run `modl llm pull <model>` to get started.",
            style("(empty)").dim()
        );
        println!();
        return Ok(());
    }

    let mut found = false;
    if let Ok(entries) = std::fs::read_dir(&store_dir) {
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                let name = entry.file_name().to_string_lossy().to_string();
                // Check for .gguf files inside
                let model_dir = entry.path();
                let mut size_total: u64 = 0;
                let mut file_count = 0;
                if let Ok(files) = std::fs::read_dir(&model_dir) {
                    for f in files.flatten() {
                        if f.path().extension().is_some_and(|e| e == "gguf") {
                            size_total += f.metadata().map(|m| m.len()).unwrap_or(0);
                            file_count += 1;
                        }
                    }
                }
                if file_count > 0 {
                    found = true;
                    println!(
                        "    {}  {:.1} GB",
                        style(&name).bold(),
                        size_total as f64 / 1_073_741_824.0
                    );
                }
            }
        }
    }

    if !found {
        println!(
            "  {} No models installed. Run `modl llm pull <model>` to get started.",
            style("(empty)").dim()
        );
    }

    println!();
    Ok(())
}
