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
mod popular;
mod runtime;
mod search;
mod space;
mod train;
mod train_setup;
mod uninstall;
mod update;
mod upgrade;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "mods",
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
        #[arg(long, short = 't')]
        r#type: Option<String>,
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
        #[arg(long, short = 't')]
        r#type: Option<String>,
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
        r#type: Option<String>,
        /// Filter by compatible base model
        #[arg(long)]
        r#for: Option<String>,
        /// Time period: day, week, month
        #[arg(long, default_value = "week")]
        period: String,
    },

    /// Link an existing tool's model folder into mods
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
        /// Path to mods.lock file
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
    Train {
        #[command(subcommand)]
        command: Option<TrainSubcommands>,
        /// Dataset name or directory path
        #[arg(long)]
        dataset: Option<String>,
        /// Base model id (e.g. flux-schnell, flux-dev)
        #[arg(long)]
        base: Option<String>,
        /// Output LoRA name
        #[arg(long)]
        name: Option<String>,
        /// Trigger word used during training
        #[arg(long)]
        trigger: Option<String>,
        /// Training preset: quick, standard, advanced
        #[arg(long)]
        preset: Option<String>,
        /// Override training steps
        #[arg(long)]
        steps: Option<u32>,
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
        #[arg(long)]
        provider: Option<String>,
    },

    /// Generate images using diffusers
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
        #[arg(long)]
        provider: Option<String>,
    },

    /// Manage datasets for training
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
        provider: String,
    },

    /// Update mods CLI to the latest release
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
            preset,
            steps,
            config,
            dry_run,
            cloud,
            provider,
        } => match command {
            Some(TrainSubcommands::Setup { reinstall }) => train_setup::run(reinstall).await,
            None => {
                train::run(
                    dataset.as_deref(),
                    base.as_deref(),
                    name.as_deref(),
                    trigger.as_deref(),
                    preset.as_deref(),
                    steps,
                    config.as_deref(),
                    dry_run,
                    cloud,
                    provider.as_deref(),
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
                provider.as_deref(),
            )
            .await
        }
        Commands::Dataset { command } => datasets::run(command).await,
        Commands::Runtime { command } => runtime::run(command).await,
        Commands::Doctor { verify_hashes } => doctor::run(verify_hashes).await,
        Commands::Config { key, value } => config::run(key.as_deref(), value.as_deref()).await,
        Commands::Auth { provider } => auth::run(&provider).await,
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
        ModelCommands::Ls { r#type } => list::run(r#type.as_deref()).await,
        ModelCommands::Info { id } => info::run(&id).await,
        ModelCommands::Search {
            query,
            r#type,
            r#for,
            tag,
            min_rating,
        } => {
            search::run(
                &query,
                r#type.as_deref(),
                r#for.as_deref(),
                tag.as_deref(),
                min_rating,
            )
            .await
        }
        ModelCommands::Popular {
            r#type,
            r#for,
            period,
        } => popular::run(r#type.as_deref(), r#for.as_deref(), &period).await,
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
            let mut parts = vec![format!("mods {full_name}")];
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
