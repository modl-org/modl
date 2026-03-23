use anyhow::{Context, Result};
use console::style;
use dialoguer::{Confirm, MultiSelect, Select};
use std::path::{Path, PathBuf};

use crate::compat::{check_windows_dev_mode, detect_tools};
use crate::core::config::{Config, GpuOverride, StorageConfig, TargetConfig, ToolType};
use crate::core::gpu;
use crate::core::runtime;

/// Starter models offered during interactive setup.
/// Hardcoded top picks — sorted by VRAM fit at display time.
struct StarterModel {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    /// Approximate total download size in bytes (model + dependencies)
    size_bytes: u64,
    /// Minimum VRAM in MB to run this model
    vram_min: u64,
}

const STARTER_MODELS: &[StarterModel] = &[
    StarterModel {
        id: "flux2-klein-4b",
        label: "flux2-klein-4b",
        description: "4-step, small & fast, Apache 2.0",
        size_bytes: 4_200_000_000,
        vram_min: 8_000,
    },
    StarterModel {
        id: "z-image-turbo",
        label: "z-image-turbo",
        description: "1-step real-time, Apache 2.0",
        size_bytes: 8_500_000_000,
        vram_min: 12_000,
    },
    StarterModel {
        id: "flux-schnell",
        label: "flux-schnell",
        description: "4-step, fast",
        size_bytes: 4_600_000_000,
        vram_min: 8_000,
    },
    StarterModel {
        id: "flux-dev",
        label: "flux-dev",
        description: "28-step, highest quality",
        size_bytes: 12_000_000_000,
        vram_min: 12_000,
    },
    StarterModel {
        id: "sdxl-base-1.0",
        label: "sdxl",
        description: "Classic, wide LoRA ecosystem",
        size_bytes: 6_500_000_000,
        vram_min: 8_000,
    },
];

fn format_size(bytes: u64) -> String {
    let gb = bytes as f64 / 1_000_000_000.0;
    format!("{:.1} GB", gb)
}

pub async fn run(defaults: bool, root_override: Option<&str>) -> Result<()> {
    println!(
        "{}",
        style("modl init — setting up your environment")
            .bold()
            .cyan()
    );
    println!();

    // Determine storage root
    let default_root = dirs::home_dir()
        .expect("Could not determine home directory")
        .join("modl");

    let storage_root = if let Some(r) = root_override {
        PathBuf::from(shellexpand::tilde(r).to_string())
    } else if defaults {
        default_root.clone()
    } else {
        pick_storage_root(&default_root)?
    };

    std::fs::create_dir_all(&storage_root).context("Failed to create storage directory")?;
    println!(
        "  {} Storage: {}",
        style("✓").green(),
        storage_root.display()
    );

    // Windows Developer Mode check
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

    // Tool detection (skip in defaults mode)
    let targets = if defaults {
        Vec::new()
    } else {
        detect_tool_targets()?
    };

    // GPU detection
    println!();
    println!("{} Detecting GPU...", style("→").cyan());
    let gpu_info = gpu::detect();
    let gpu_override = match &gpu_info {
        Some(info) => {
            let device_label = match info.device {
                gpu::DeviceType::Cuda => "CUDA",
                gpu::DeviceType::Mps => "MPS (Apple Silicon)",
            };
            println!(
                "  {} {} — {} MB VRAM ({})",
                style("✓").green(),
                info.name,
                info.vram_mb,
                device_label,
            );
            if info.device == gpu::DeviceType::Mps {
                println!(
                    "  {} Training is not available on MPS. Use {} for cloud training.",
                    style("!").yellow(),
                    style("modl train --cloud").cyan(),
                );
            }
            Some(GpuOverride {
                vram_mb: info.vram_mb,
            })
        }
        None => {
            println!(
                "  {} No GPU detected. Variant auto-selection will default to smallest.",
                style("!").yellow()
            );
            None
        }
    };

    // Save config
    let config = Config {
        storage: StorageConfig {
            root: storage_root.clone(),
        },
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

    // Runtime install — auto-install on defaults, prompt otherwise
    let runtime_installed = if defaults {
        install_runtime(&gpu_info).await.unwrap_or(false)
    } else {
        offer_runtime_install(&gpu_info).await?
    };

    // Model download offer
    let mut model_installed = false;
    if !defaults {
        model_installed = offer_starter_model(&gpu_info).await?;
    }

    // Service install offer
    if !defaults {
        offer_service_install().await?;
    }

    println!();
    println!("Next steps:");
    if !runtime_installed {
        println!("  {} Install runtime", style("modl runtime install").cyan());
    }
    if model_installed {
        println!(
            "  {} Generate an image",
            style("modl generate \"a cat in space\"").cyan()
        );
        println!("  {} Launch the web UI", style("modl serve").cyan());
        println!("  {} Browse all models", style("modl search flux").cyan());
    } else {
        println!(
            "  {} Browse available models",
            style("modl search flux").cyan()
        );
        println!(
            "  {} Install a model",
            style("modl pull flux2-klein-4b").cyan()
        );
        println!("  {} Launch the web UI", style("modl serve").cyan());
    }

    Ok(())
}

fn pick_storage_root(default_root: &Path) -> Result<PathBuf> {
    // Check if config already exists
    let config_path = Config::default_path();
    if config_path.exists() {
        let overwrite = Confirm::new()
            .with_prompt("Config already exists at ~/.modl/config.yaml. Overwrite?")
            .default(false)
            .interact()?;
        if !overwrite {
            println!("Keeping existing config.");
            std::process::exit(0);
        }
    }

    let root_display = default_root.display().to_string();
    println!("{} Where should modl store model files?", style("→").cyan());
    println!("  Default: {}", style(&root_display).dim());

    let use_default = Confirm::new()
        .with_prompt(format!("Use {}?", root_display))
        .default(true)
        .interact()?;

    if use_default {
        Ok(default_root.to_path_buf())
    } else {
        let custom: String = dialoguer::Input::new()
            .with_prompt("Storage path")
            .interact_text()?;
        Ok(PathBuf::from(shellexpand::tilde(&custom).to_string()))
    }
}

fn detect_tool_targets() -> Result<Vec<TargetConfig>> {
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

    Ok(targets)
}

async fn offer_starter_model(gpu_info: &Option<gpu::GpuInfo>) -> Result<bool> {
    println!();
    let download = Confirm::new()
        .with_prompt("Would you like to download a starter model?")
        .default(true)
        .interact()?;

    if !download {
        return Ok(false);
    }

    let vram_mb = gpu_info.as_ref().map(|g| g.vram_mb).unwrap_or(0);

    // Build menu items, marking which fit the GPU
    let mut items: Vec<String> = Vec::new();
    let mut recommended_idx: Option<usize> = None;

    for model in STARTER_MODELS {
        let fits = vram_mb == 0 || vram_mb >= model.vram_min;
        let size = format_size(model.size_bytes);
        let mut label = format!("{:<18} {} ({})", model.label, model.description, size);
        if fits && recommended_idx.is_none() && vram_mb > 0 {
            label.push_str("  ← recommended for your GPU");
            recommended_idx = Some(items.len());
        } else if !fits {
            label.push_str("  (may not fit your VRAM)");
        }
        items.push(label);
    }
    items.push("Skip".to_string());

    let default_idx = recommended_idx.unwrap_or(0);
    let selected = Select::new()
        .with_prompt("Select a model")
        .items(&items)
        .default(default_idx)
        .interact()?;

    if selected >= STARTER_MODELS.len() {
        // User chose "Skip"
        return Ok(false);
    }

    let model_id = STARTER_MODELS[selected].id;
    println!();
    println!(
        "{} Downloading {}...",
        style("→").cyan(),
        style(model_id).bold()
    );

    // Ensure registry is available
    let index = crate::core::registry::RegistryIndex::load_or_fetch().await?;
    let db = crate::core::db::Database::open()?;
    let config = Config::load()?;
    let store = crate::core::store::Store::new(config.store_root());
    let auth_store = crate::auth::AuthStore::load().unwrap_or_default();

    let (plan, vram) = crate::core::install::resolve_plan(model_id, None, &index, &db)?;

    for item in &plan.items {
        if item.already_installed {
            println!(
                "  {} {} already installed",
                style("✓").green(),
                item.manifest.name
            );
            continue;
        }

        let effective_variant = item.variant_id.as_deref();
        let info = crate::core::install::resolve_file_info(&item.manifest, effective_variant, vram);

        println!(
            "  {} {} ({})",
            style("↓").cyan(),
            item.manifest.name,
            format_size(info.size)
        );

        crate::core::install::install_item(
            &item.manifest,
            effective_variant,
            vram,
            &config,
            &store,
            &auth_store,
            &db,
            false,
        )
        .await?;

        println!("  {} {}", style("✓").green(), item.manifest.name);
    }

    println!(
        "\n  {} {} is ready to use!",
        style("✓").green().bold(),
        model_id
    );

    Ok(true)
}

async fn offer_service_install() -> Result<()> {
    println!();
    let install = Confirm::new()
        .with_prompt("Install the web UI as a background service? (starts on boot)")
        .default(false)
        .interact()?;

    if !install {
        return Ok(());
    }

    super::serve::install_service(3333).await?;
    Ok(())
}

/// Install runtime without prompting (used for --defaults mode).
async fn install_runtime(gpu_info: &Option<gpu::GpuInfo>) -> Result<bool> {
    let profile = runtime::default_profile();
    println!();
    println!(
        "{} Installing runtime (profile: {})...",
        style("→").cyan(),
        style(profile).bold()
    );
    runtime::install(Some(profile), None)?;
    runtime::bootstrap(Some(profile), None).await?;

    println!("  {} Runtime installed ({})", style("✓").green(), profile);

    if gpu_info.as_ref().map(|g| g.device) == Some(gpu::DeviceType::Mps) {
        println!(
            "  {} Generation works on MPS. Training requires {} (cloud GPU).",
            style("i").dim(),
            style("modl train --cloud").cyan(),
        );
    }

    Ok(true)
}

/// Prompt user to install the Python runtime during init.
async fn offer_runtime_install(gpu_info: &Option<gpu::GpuInfo>) -> Result<bool> {
    let profile = runtime::default_profile();
    let profile_desc = if profile == "generator" {
        "generation-only (MPS)"
    } else {
        "generation + training (CUDA)"
    };

    println!();
    let install = Confirm::new()
        .with_prompt(format!("Install the Python runtime? ({profile_desc})"))
        .default(true)
        .interact()?;

    if !install {
        println!(
            "  {} You can install later with {}",
            style("i").dim(),
            style("modl runtime install").cyan(),
        );
        return Ok(false);
    }

    install_runtime(gpu_info).await
}
