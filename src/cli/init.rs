use anyhow::{Context, Result};
use console::style;
use dialoguer::{Confirm, MultiSelect, Select};
use std::path::PathBuf;

use crate::compat::{check_windows_dev_mode, detect_tools};
use crate::core::config::{Config, GpuOverride, StorageConfig, TargetConfig, ToolType};
use crate::core::gpu;

pub async fn run() -> Result<()> {
    println!(
        "{}",
        style("modl init — setting up your environment")
            .bold()
            .cyan()
    );
    println!();

    // Check if config already exists
    let config_path = Config::default_path();
    if config_path.exists() {
        let overwrite = Confirm::new()
            .with_prompt("Config already exists at ~/.modl/config.yaml. Overwrite?")
            .default(false)
            .interact()?;
        if !overwrite {
            println!("Keeping existing config.");
            return Ok(());
        }
    }

    // 1. Storage root
    let default_root = dirs::home_dir()
        .expect("Could not determine home directory")
        .join("modl");
    let root_display = default_root.display().to_string();

    println!("{} Where should modl store model files?", style("→").cyan());
    println!("  Default: {}", style(&root_display).dim());

    let use_default = Confirm::new()
        .with_prompt(format!("Use {}?", root_display))
        .default(true)
        .interact()?;

    let storage_root = if use_default {
        default_root
    } else {
        let custom: String = dialoguer::Input::new()
            .with_prompt("Storage path")
            .interact_text()?;
        PathBuf::from(shellexpand::tilde(&custom).to_string())
    };

    // Create store directory
    std::fs::create_dir_all(&storage_root).context("Failed to create storage directory")?;

    println!(
        "  {} Storage: {}",
        style("✓").green(),
        storage_root.display()
    );

    // Windows: check Developer Mode for symlink support
    if let Some(dev_mode_enabled) = check_windows_dev_mode() {
        if !dev_mode_enabled {
            println!();
            println!(
                "  {} Windows Developer Mode is not enabled.",
                style("⚠").yellow()
            );
            println!("  Without it, modl will use hard links instead of symlinks.");
            println!("  Hard links work fine, but require store and tools on the same drive.");
            println!();
            println!(
                "  To enable symlinks (recommended): {} → {} → {}",
                style("Settings").bold(),
                style("Privacy & security").bold(),
                style("For developers → Developer Mode").bold()
            );
            println!();
        } else {
            println!(
                "  {} Windows Developer Mode enabled (symlinks available)",
                style("✓").green()
            );
        }
    }

    // 2. Detect tools
    println!();
    println!(
        "{} Scanning for AI image generation tools...",
        style("→").cyan()
    );

    let detected = detect_tools();
    let mut targets: Vec<TargetConfig> = Vec::new();

    if detected.is_empty() {
        println!("  {} No tools auto-detected.", style("!").yellow());
        println!("  Can't find your install? Add it manually with:");
        println!(
            "    {}",
            style("modl link --comfyui /path/to/ComfyUI").cyan()
        );
        println!(
            "    {}",
            style("modl link --a1111 /path/to/stable-diffusion-webui").cyan()
        );
    } else {
        let labels: Vec<String> = detected
            .iter()
            .map(|(tool_type, path)| {
                format!(
                    "{} ({})",
                    match tool_type {
                        ToolType::Comfyui => "ComfyUI",
                        ToolType::A1111 => "A1111 / SD WebUI",
                        ToolType::Invokeai => "InvokeAI",
                        ToolType::Custom => "Custom",
                    },
                    path.display()
                )
            })
            .collect();

        let selected = MultiSelect::new()
            .with_prompt("Found these tools — select which to target")
            .items(&labels)
            .defaults(&vec![true; labels.len()])
            .interact()?;

        for idx in selected {
            let (tool_type, path) = &detected[idx];
            targets.push(TargetConfig {
                path: path.clone(),
                tool_type: tool_type.clone(),
                symlink: true,
            });
            println!("  {} {} → symlink mode", style("✓").green(), labels[idx]);
        }
    }

    // Allow manual additions
    if Confirm::new()
        .with_prompt("Add a tool path manually?")
        .default(false)
        .interact()?
    {
        let path_str: String = dialoguer::Input::new()
            .with_prompt("Path to tool installation")
            .interact_text()?;
        let tool_idx = Select::new()
            .with_prompt("Tool type")
            .items(&["ComfyUI", "A1111 / SD WebUI", "InvokeAI", "Custom"])
            .default(0)
            .interact()?;
        let tool_type = match tool_idx {
            0 => ToolType::Comfyui,
            1 => ToolType::A1111,
            2 => ToolType::Invokeai,
            _ => ToolType::Custom,
        };
        targets.push(TargetConfig {
            path: PathBuf::from(shellexpand::tilde(&path_str).to_string()),
            tool_type,
            symlink: true,
        });
    }

    // 3. GPU detection
    println!();
    println!("{} Detecting GPU...", style("→").cyan());

    let gpu_override = match gpu::detect() {
        Some(info) => {
            println!(
                "  {} {} — {} MB VRAM",
                style("✓").green(),
                info.name,
                info.vram_mb
            );
            Some(GpuOverride {
                vram_mb: info.vram_mb,
            })
        }
        None => {
            println!(
                "  {} No NVIDIA GPU detected. Variant auto-selection will default to smallest.",
                style("!").yellow()
            );
            None
        }
    };

    // 4. Write config
    let config = Config {
        storage: StorageConfig { root: storage_root },
        targets,
        gpu: gpu_override,
        cloud: None,
    };

    config.save().context("Failed to save config")?;

    println!();
    println!(
        "{} Config saved to {}",
        style("✓").green().bold(),
        Config::default_path().display()
    );
    println!();
    println!("Next steps:");
    println!("  {} Fetch the model registry", style("modl update").cyan());
    println!(
        "  {} Install your first model",
        style("modl install flux-dev").cyan()
    );

    Ok(())
}
