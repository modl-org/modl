mod auth;
mod config;
mod datasets;
mod doctor;
mod export;
mod gc;
mod generate;
mod import;
mod info;
mod init;
mod install;
mod link;
mod list;
mod outputs;
mod popular;
mod preview;
mod runtime;
mod search;
mod space;
mod train;
mod train_setup;
mod train_status;
mod uninstall;
mod update;
mod upgrade;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use console::style;

use crate::core::cloud::CloudProvider;
use crate::core::job::{LoraType, Optimizer, Preset};
use crate::core::manifest::AssetType;

/// Extended help text for `modl train` with recommended settings per model.
const TRAIN_HELP_EXTRA: &str = "\
\x1b[1mRecommended settings by model:\x1b[0m

  SDXL (sdxl-base-1.0):
    Style:     --rank 32  --lr 1e-4   --steps 10000+  --batch-size 2
    Character: --rank 16  --lr 1e-4   --steps 1500    --batch-size 1
    Object:    --rank 8   --lr 1e-4   --steps 1000    --batch-size 1

  Flux (flux-dev):
    Style:     --rank 16  --lr 1e-4   --steps 5000+   --batch-size 1
    Character: --rank 16  --lr 4e-4   --steps 1500    --batch-size 1
    Object:    --rank 8   --lr 1e-4   --steps 1000    --batch-size 1

  SD 1.5 (sd-1.5):
    Style:     --rank 32  --lr 1e-4   --steps 5000+   --batch-size 2
    Character: --rank 8   --lr 1e-4   --steps 1500    --batch-size 1

  Z-Image-Turbo (z-image-turbo):
    Style:     --rank 16  --lr 1e-4   --steps 3000-5000  --batch-size 1
    Character: --rank 16  --lr 1e-4   --steps 1500       --batch-size 1
    ⚡ Trains very fast (~1.3s/step). Uses training adapter automatically.
    ⚠  Do not exceed --lr 1e-4 — higher LR breaks distillation.
    For style: caption images literally (what's depicted, not the style).
    Inference: 8 steps, CFG 1.0, euler. Remove training adapter for inference.

  Qwen-Image (qwen-image):
    Style:     --rank 16  --lr 2e-4   --steps 3000    --batch-size 1
    Character: --rank 16  --lr 1e-4   --steps 3000+   --batch-size 1
    ⚠  20B param model. Style uses 3-bit + ARA (~23GB, fits 24GB cards).
    ⚠  Character/object uses uint6 (~30GB, needs 32GB GPU like RTX 5090).
    ❌ Character on 24GB NOT recommended (int4 = severe quality loss).
    For style: use literal captions, usually no trigger word needed.

  Chroma (chroma):
    Style:     --rank 16  --lr 1e-4   --steps 5000+   --batch-size 1
    Character: --rank 16  --lr 1e-4   --steps 2000    --batch-size 1

\x1b[1mOptimizer guide:\x1b[0m
    adamw8bit  Best default. Low VRAM, stable training.
    prodigy    Auto-tunes LR. Set --lr 1.0 and let it adapt.
    adamw      Full precision. More VRAM but sometimes better quality.
    adafactor  Memory-efficient. Good for VRAM-constrained setups.

\x1b[1mStyle training tips:\x1b[0m
    - Use 50-150 images of consistent style (fewer is often better)
    - Higher --caption-dropout (0.3-0.5) helps learn style over content
    - More --steps (10k-40k) needed for large datasets
    - Higher --rank (32-64) gives more capacity for complex styles
    - Use --repeats 10 for small datasets (<50 images)
";

const TRAIN_EXAMPLES: &str = "\
\x1b[1mExamples:\x1b[0m
  # Train a character LoRA on Flux (interactive prompts fill in the rest)
  modl train --base flux-dev --lora-type character

  # Quick style LoRA with all flags specified
  modl train --dataset paintings --base flux-dev --lora-type style \\
    --name impressionist-v1 --trigger \"in the style of MYPAINT\" --preset quick

  # Resume from a checkpoint
  modl train --base flux-dev --lora-type character --resume ./checkpoint-500.safetensors

  # Dry-run: see the generated spec without running
  modl train --dataset headshots --base flux-dev --lora-type character --dry-run
";

const GENERATE_EXAMPLES: &str = "\
\x1b[1mExamples:\x1b[0m
  # Simple generation with default model (flux-schnell)
  modl generate \"a cat astronaut on the moon\"

  # Use a specific base model + LoRA
  modl generate \"photo of OHWX person at the beach\" --base flux-dev --lora pedro-v1

  # Generate multiple images with a fixed seed
  modl generate \"product photo on marble\" --count 4 --seed 42 --size 16:9

  # Landscape format with more steps
  modl generate \"sunset over mountains\" --size 16:9 --steps 30 --guidance 4.0
";

const DATASET_EXAMPLES: &str = "\
\x1b[1mExamples:\x1b[0m
  # Full pipeline: create → resize → caption in one command
  modl dataset prepare my-photos --from ~/photos/headshots/

  # Step-by-step: create, then caption, then validate
  modl dataset create products --from ~/product-photos/
  modl dataset caption products
  modl dataset validate products

  # Resize to 512px (SD 1.5) instead of default 1024px
  modl dataset resize my-dataset --resolution 512
";

const MODEL_PULL_EXAMPLES: &str = "\
\x1b[1mExamples:\x1b[0m
  # Pull a model (auto-selects variant for your GPU)
  modl model pull flux-dev

  # Force a specific variant
  modl model pull flux-dev --variant fp8

  # Preview what would be downloaded
  modl model pull sdxl-base-1.0 --dry-run
";

/// Auth provider for `modl auth` command.
#[derive(Clone, Debug, ValueEnum)]
pub enum AuthProvider {
    #[value(alias = "hf")]
    Huggingface,
    Civitai,
}

#[derive(Parser)]
#[command(
    name = "modl",
    about = "Model manager for the AI image generation ecosystem",
    version,
    propagate_version = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum ModelCommands {
    /// Download a model, LoRA, VAE, or other asset (with dependency resolution)
    #[command(after_help = MODEL_PULL_EXAMPLES)]
    Pull {
        /// Model ID from the registry (e.g., flux-dev, realistic-skin-v3)
        id: String,
        /// Force a specific variant (e.g., fp16, fp8, gguf-q4)
        #[arg(long)]
        variant: Option<String>,
        /// Show what would be installed without doing it
        #[arg(long)]
        dry_run: bool,
        /// Force re-download even if files already exist
        #[arg(long)]
        force: bool,
    },

    /// Remove an installed model
    Rm {
        /// Model ID to remove
        id: String,
        /// Force removal even if other items depend on this
        #[arg(long)]
        force: bool,
    },

    /// List installed models
    Ls {
        /// Filter by asset type (checkpoint, lora, vae, text_encoder, etc.)
        #[arg(long, short = 't', value_enum)]
        r#type: Option<AssetType>,
    },

    /// Show detailed info about a model
    Info {
        /// Model ID to inspect
        id: String,
    },

    /// Search the registry
    Search {
        /// Search query
        query: String,
        /// Filter by asset type
        #[arg(long, short = 't', value_enum)]
        r#type: Option<AssetType>,
        /// Filter by compatible base model
        #[arg(long)]
        r#for: Option<String>,
        /// Filter by tag
        #[arg(long)]
        tag: Option<String>,
        /// Minimum rating
        #[arg(long)]
        min_rating: Option<f32>,
    },

    /// Show popular/trending models
    Popular {
        /// Filter by asset type
        #[arg(long, short = 't', value_enum)]
        r#type: Option<AssetType>,
        /// Filter by compatible base model
        #[arg(long)]
        r#for: Option<String>,
    },

    /// Link an existing tool's model folder into modl
    Link {
        /// Path to ComfyUI installation
        #[arg(long)]
        comfyui: Option<String>,
        /// Path to A1111 installation
        #[arg(long)]
        a1111: Option<String>,
    },

    /// Fetch latest registry index
    Update,

    /// Show disk usage breakdown
    Space,

    /// Garbage collect — remove unreferenced files from the store
    Gc,

    /// Export installed state to a lock file
    Export,

    /// Import and install from a lock file
    Import {
        /// Path to modl.lock file
        path: String,
    },
}

#[derive(Subcommand)]
pub enum TrainSubcommands {
    /// Prepare managed training dependencies (ai-toolkit + torch stack)
    Setup {
        /// Force re-install of training dependencies
        #[arg(long)]
        reinstall: bool,
    },

    /// Show live training progress (parses log files)
    Status {
        /// Show status for a specific run name only
        name: Option<String>,
        /// Watch mode: refresh every 2 seconds
        #[arg(long, short = 'w')]
        watch: bool,
    },
}

#[derive(Subcommand)]
pub enum Commands {
    /// Interactive first-run setup — detect tools, configure storage
    Init,

    /// Manage models, LoRAs, VAEs, and other assets
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },

    /// Train a LoRA with managed runtime
    #[command(args_conflicts_with_subcommands = true)]
    #[command(after_long_help = TRAIN_HELP_EXTRA)]
    #[command(after_help = TRAIN_EXAMPLES)]
    Train {
        #[command(subcommand)]
        command: Option<TrainSubcommands>,
        /// Dataset name or directory path
        #[arg(long)]
        dataset: Option<String>,
        /// Base model id (e.g. flux-dev, sdxl-base-1.0)
        #[arg(long)]
        base: Option<String>,
        /// Output LoRA name
        #[arg(long)]
        name: Option<String>,
        /// Trigger word used during training
        #[arg(long)]
        trigger: Option<String>,
        /// LoRA type: style, character, object
        #[arg(long, value_enum, rename_all = "snake_case")]
        lora_type: Option<LoraType>,
        /// Training preset: quick, standard, advanced
        #[arg(long, value_enum)]
        preset: Option<Preset>,
        /// Override training steps
        #[arg(long)]
        steps: Option<u32>,
        /// LoRA rank (capacity). Higher = more expressive but larger file
        #[arg(long)]
        rank: Option<u32>,
        /// Learning rate (e.g. 1e-4, 2e-4, 5e-5)
        #[arg(long)]
        lr: Option<f64>,
        /// Batch size per step (higher = faster but more VRAM)
        #[arg(long)]
        batch_size: Option<u32>,
        /// Image resolution for training
        #[arg(long)]
        resolution: Option<u32>,
        /// Optimizer: adamw8bit, prodigy, adamw, adafactor, sgd
        #[arg(long, value_enum)]
        optimizer: Option<Optimizer>,
        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
        /// Dataset repetitions per epoch
        #[arg(long)]
        repeats: Option<u32>,
        /// Caption dropout rate (0.0-1.0, higher = learn style over content)
        #[arg(long)]
        caption_dropout: Option<f64>,
        /// Resume from a checkpoint .safetensors file
        #[arg(long)]
        resume: Option<String>,
        /// Load a full TrainJobSpec YAML (escape hatch)
        #[arg(long)]
        config: Option<String>,
        /// Generate spec and print it without executing
        #[arg(long)]
        dry_run: bool,
        /// Run training on a cloud provider instead of locally
        #[arg(long)]
        cloud: bool,
        /// Cloud provider to use (modal, replicate, runpod)
        #[arg(long, value_enum)]
        provider: Option<CloudProvider>,
    },

    /// Generate images using diffusers
    #[command(after_help = GENERATE_EXAMPLES)]
    Generate {
        /// Text prompt for image generation
        prompt: String,
        /// Base model to use (default: flux-schnell)
        #[arg(long)]
        base: Option<String>,
        /// LoRA name or path to apply
        #[arg(long)]
        lora: Option<String>,
        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
        /// Image size preset (1:1, 16:9, 9:16, 4:3, 3:4) or WxH
        #[arg(long, default_value = "1:1")]
        size: String,
        /// Number of inference steps
        #[arg(long)]
        steps: Option<u32>,
        /// Guidance scale
        #[arg(long)]
        guidance: Option<f32>,
        /// Number of images to generate
        #[arg(long, default_value = "1")]
        count: u32,
        /// Run generation on a cloud provider instead of locally
        #[arg(long)]
        cloud: bool,
        /// Cloud provider to use (modal, replicate, runpod)
        #[arg(long, value_enum)]
        provider: Option<CloudProvider>,
    },

    /// Manage datasets for training
    #[command(after_help = DATASET_EXAMPLES)]
    Dataset {
        #[command(subcommand)]
        command: datasets::DatasetCommands,
    },

    /// Manage embedded Python runtime
    Runtime {
        #[command(subcommand)]
        command: runtime::RuntimeCommands,
    },

    /// Check for broken symlinks, missing deps, corrupt files
    Doctor {
        /// Also verify SHA256 hashes (slow for large files)
        #[arg(long)]
        verify_hashes: bool,
    },

    /// View or update configuration (e.g., storage.root, gpu.vram_mb)
    Config {
        /// Config key to view or set (e.g., storage.root)
        key: Option<String>,
        /// New value (required when setting a key)
        value: Option<String>,
    },

    /// Configure authentication (HuggingFace, Civitai)
    Auth {
        /// Auth provider: huggingface or civitai
        #[arg(value_enum)]
        provider: AuthProvider,
    },

    /// Browse and manage generated outputs and training artifacts
    Outputs {
        #[command(subcommand)]
        command: outputs::OutputCommands,
    },

    /// Launch web UI to preview training samples, datasets, and configs
    Preview {
        /// Port to bind the preview server on
        #[arg(long, default_value = "3333")]
        port: u16,
        /// Don't auto-open the browser
        #[arg(long)]
        no_open: bool,
        /// Run in foreground (blocks terminal; default is background/daemon)
        #[arg(long)]
        foreground: bool,
    },

    /// Update modl CLI to the latest release
    Upgrade,

    /// Dump CLI schema as JSON (for docs generation)
    #[command(hide = true)]
    CliSchema,
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Init => init::run().await,
        Commands::Model { command } => run_model(command).await,
        Commands::Train {
            command,
            dataset,
            base,
            name,
            trigger,
            lora_type,
            preset,
            steps,
            rank,
            lr,
            batch_size,
            resolution,
            optimizer,
            seed,
            repeats,
            caption_dropout,
            resume,
            config,
            dry_run,
            cloud,
            provider,
        } => match command {
            Some(TrainSubcommands::Setup { reinstall }) => train_setup::run(reinstall).await,
            Some(TrainSubcommands::Status { name, watch }) => {
                train_status::run(name.as_deref(), watch)?;
                Ok(())
            }
            None if base.is_none() || lora_type.is_none() => {
                print_train_info();
                Ok(())
            }
            None => {
                let base_val = base.as_deref().unwrap();
                let lora_type_val = lora_type.unwrap();
                train::run(
                    dataset.as_deref(),
                    base_val,
                    name.as_deref(),
                    trigger.as_deref(),
                    lora_type_val,
                    preset,
                    train::TrainOverrides {
                        steps,
                        rank,
                        lr,
                        batch_size,
                        resolution,
                        optimizer,
                        seed,
                        repeats,
                        caption_dropout,
                        resume,
                    },
                    config.as_deref(),
                    dry_run,
                    cloud,
                    provider,
                )
                .await
            }
        },
        Commands::Generate {
            prompt,
            base,
            lora,
            seed,
            size,
            steps,
            guidance,
            count,
            cloud,
            provider,
        } => {
            generate::run(
                &prompt,
                base.as_deref(),
                lora.as_deref(),
                seed,
                &size,
                steps,
                guidance,
                count,
                cloud,
                provider,
            )
            .await
        }
        Commands::Dataset { command } => datasets::run(command).await,
        Commands::Runtime { command } => runtime::run(command).await,
        Commands::Doctor { verify_hashes } => doctor::run(verify_hashes).await,
        Commands::Config { key, value } => config::run(key.as_deref(), value.as_deref()).await,
        Commands::Auth { provider } => auth::run(provider).await,
        Commands::Outputs { command } => outputs::run(command).await,
        Commands::Preview {
            port,
            no_open,
            foreground,
        } => preview::run(port, no_open, foreground).await,
        Commands::Upgrade => upgrade::run().await,
        Commands::CliSchema => {
            dump_cli_schema();
            Ok(())
        }
    }
}

async fn run_model(command: ModelCommands) -> Result<()> {
    match command {
        ModelCommands::Pull {
            id,
            variant,
            dry_run,
            force,
        } => install::run(&id, variant.as_deref(), dry_run, force).await,
        ModelCommands::Rm { id, force } => uninstall::run(&id, force).await,
        ModelCommands::Ls { r#type } => list::run(r#type).await,
        ModelCommands::Info { id } => info::run(&id).await,
        ModelCommands::Search {
            query,
            r#type,
            r#for,
            tag,
            min_rating,
        } => search::run(&query, r#type, r#for.as_deref(), tag.as_deref(), min_rating).await,
        ModelCommands::Popular { r#type, r#for } => popular::run(r#type, r#for.as_deref()).await,
        ModelCommands::Link { comfyui, a1111 } => {
            link::run(comfyui.as_deref(), a1111.as_deref()).await
        }
        ModelCommands::Update => update::run().await,
        ModelCommands::Space => space::run().await,
        ModelCommands::Gc => gc::run().await,
        ModelCommands::Export => export::run().await,
        ModelCommands::Import { path } => import::run(&path).await,
    }
}

fn print_train_info() {
    println!(
        "\n  {} — Train LoRAs on your GPU\n",
        style("modl train").bold().cyan()
    );

    println!(
        "  {} --base <MODEL> --lora-type <TYPE> [OPTIONS]\n",
        style("Usage:").bold()
    );

    // ── Supported models ──────────────────────────────────────────
    println!("  {}", style("Supported models:").bold());
    println!();
    println!(
        "    {:<22} {:<10} {}",
        style("Model").bold(),
        style("VRAM").bold(),
        style("Notes").bold()
    );
    println!(
        "    {}",
        style("─────────────────────────────────────────────────────────────────").dim()
    );
    println!(
        "    {:<22} {:<10} Best quality/speed balance. Quantized by default.",
        "flux-dev",
        style("~12 GB").green()
    );
    println!(
        "    {:<22} {:<10} Fast inference (4 steps). Uses training adapter.",
        "flux-schnell",
        style("~12 GB").green()
    );
    println!(
        "    {:<22} {:<10} {}",
        "z-image-turbo",
        style("~12 GB").green(),
        style("Fast training (~1.3s/step). LR capped at 1e-4.").dim()
    );
    println!(
        "    {:<22} {:<10} {}",
        "z-image",
        style("~12 GB").green(),
        style("Non-turbo variant. 30 inference steps.").dim()
    );
    println!(
        "    {:<22} {:<10} Quantized flow matching model.",
        "chroma",
        style("~14 GB").green()
    );
    println!(
        "    {:<22} {:<10} 20B params. Style via 3-bit + ARA. Fits 24GB.",
        "qwen-image (style)",
        style("~23 GB").yellow()
    );
    println!(
        "    {:<22} {:<10} Needs 32GB GPU (e.g. RTX 5090). uint6 quant.",
        "qwen-image (char/obj)",
        style("~30 GB").red()
    );
    println!(
        "    {:<22} {:<10} {}",
        "sdxl-base-1.0",
        style("~10 GB").green(),
        style("Stable classic. Good for style + character.").dim()
    );
    println!(
        "    {:<22} {:<10} {}",
        "sd-1.5",
        style("~6 GB").green(),
        style("Legacy. 512px resolution.").dim()
    );

    // ── LoRA types ────────────────────────────────────────────────
    println!();
    println!("  {}", style("LoRA types:").bold());
    println!(
        "    {}    Learn the visual identity of a person (trigger word = their name)",
        style("character").bold()
    );
    println!(
        "    {}        Learn an artistic style (literal captions, no trigger word for Qwen)",
        style("style").bold()
    );
    println!(
        "    {}       Learn a specific object or product",
        style("object").bold()
    );

    // ── Quick start ───────────────────────────────────────────────
    println!();
    println!("  {}", style("Quick start:").bold());
    println!(
        "    {} modl train --base flux-dev --lora-type character",
        style("$").dim()
    );
    println!(
        "    {} modl train --base flux-dev --lora-type style --dataset paintings",
        style("$").dim()
    );
    println!(
        "    {} modl train --base qwen-image --lora-type style",
        style("$").dim()
    );

    // ── Subcommands ───────────────────────────────────────────────
    println!();
    println!("  {}", style("Subcommands:").bold());
    println!(
        "    {}    Install training dependencies (ai-toolkit + torch)",
        style("modl train setup").bold()
    );
    println!(
        "    {}   Show live training progress",
        style("modl train status").bold()
    );

    // ── More info ─────────────────────────────────────────────────
    println!();
    println!(
        "  Run {} for all flags, presets, and optimizer guide.",
        style("modl train --help").bold()
    );
    println!();
}

fn dump_cli_schema() {
    let cmd = Cli::command();
    let mut commands = Vec::new();
    collect_schema_commands(&cmd, "", &mut commands);

    let schema = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commands": commands,
    });

    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
}

fn collect_schema_commands(
    cmd: &clap::Command,
    prefix: &str,
    commands: &mut Vec<serde_json::Value>,
) {
    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }

        let full_name = if prefix.is_empty() {
            sub.get_name().to_string()
        } else {
            format!("{prefix} {}", sub.get_name())
        };

        let has_subcommands = sub.get_subcommands().any(|s| !s.is_hide_set());

        // Recurse into nested subcommands
        if has_subcommands {
            collect_schema_commands(sub, &full_name, commands);

            // If the command also has its own args (e.g. `train`), emit it too
            let has_own_args = sub
                .get_arguments()
                .any(|a| a.get_id() != "help" && a.get_id() != "version");
            if !has_own_args {
                continue;
            }
        }

        let description = sub.get_about().map(|s| s.to_string()).unwrap_or_default();

        let mut args = Vec::new();
        let mut flags = Vec::new();

        for arg in sub.get_arguments() {
            if arg.get_id() == "help" || arg.get_id() == "version" {
                continue;
            }

            let id = arg.get_id().to_string();
            let help = arg.get_help().map(|s| s.to_string()).unwrap_or_default();
            let required = arg.is_required_set();

            let short = arg.get_short().map(|c| format!("-{c}"));
            let long = arg.get_long().map(|l| format!("--{l}"));

            let default = arg
                .get_default_values()
                .first()
                .and_then(|v| v.to_str())
                .map(String::from);

            let is_bool = matches!(
                arg.get_action(),
                clap::ArgAction::SetTrue | clap::ArgAction::SetFalse | clap::ArgAction::Count
            );

            if arg.is_positional() {
                args.push(serde_json::json!({
                    "name": id,
                    "description": help,
                    "required": required,
                }));
            } else {
                let mut flag = serde_json::json!({
                    "name": long.as_deref().or(short.as_deref()).unwrap_or(&id),
                    "description": help,
                    "is_bool": is_bool,
                });
                if let Some(s) = &short {
                    flag.as_object_mut()
                        .unwrap()
                        .insert("short".into(), serde_json::json!(s));
                }
                if let Some(d) = &default {
                    flag.as_object_mut()
                        .unwrap()
                        .insert("default".into(), serde_json::json!(d));
                }
                flags.push(flag);
            }
        }

        // Build usage string
        let usage = {
            let mut parts = vec![format!("modl {full_name}")];
            for a in &args {
                let n = a["name"].as_str().unwrap();
                if a["required"].as_bool().unwrap_or(false) {
                    parts.push(format!("<{n}>"));
                } else {
                    parts.push(format!("[{n}]"));
                }
            }
            if !flags.is_empty() {
                parts.push("[flags]".into());
            }
            parts.join(" ")
        };

        commands.push(serde_json::json!({
            "name": full_name,
            "description": description,
            "usage": usage,
            "args": args,
            "flags": flags,
        }));
    }
}
