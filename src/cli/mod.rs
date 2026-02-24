mod auth;
mod config;
mod doctor;
mod export;
mod gc;
mod import;
mod info;
mod init;
mod install;
mod link;
mod list;
mod popular;
mod search;
mod space;
mod uninstall;
mod update;

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
pub enum Commands {
    /// Interactive first-run setup — detect tools, configure storage
    Init,

    /// Install a model, LoRA, VAE, or other asset (with dependency resolution)
    Install {
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
    Uninstall {
        /// Model ID to uninstall
        id: String,
        /// Force removal even if other items depend on this
        #[arg(long)]
        force: bool,
    },

    /// List installed models
    List {
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

    /// Show disk usage breakdown
    Space,

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

    /// Garbage collect — remove unreferenced files from the store
    Gc,

    /// Link an existing tool's model folder into mods
    Link {
        /// Path to ComfyUI installation
        #[arg(long)]
        comfyui: Option<String>,
        /// Path to A1111 installation
        #[arg(long)]
        a1111: Option<String>,
    },

    /// Configure authentication (HuggingFace, Civitai)
    Auth {
        /// Auth provider: huggingface or civitai
        provider: String,
    },

    /// Fetch latest registry index
    Update,

    /// Export installed state to a lock file
    Export,

    /// Import and install from a lock file
    Import {
        /// Path to mods.lock file
        path: String,
    },

    /// Dump CLI schema as JSON (for docs generation)
    #[command(hide = true)]
    CliSchema,

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
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Init => init::run().await,
        Commands::Install {
            id,
            variant,
            dry_run,
            force,
        } => install::run(&id, variant.as_deref(), dry_run, force).await,
        Commands::Uninstall { id, force } => uninstall::run(&id, force).await,
        Commands::List { r#type } => list::run(r#type.as_deref()).await,
        Commands::Info { id } => info::run(&id).await,
        Commands::Search {
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
        Commands::Config { key, value } => {
            config::run(key.as_deref(), value.as_deref()).await
        }
        Commands::Space => space::run().await,
        Commands::Doctor { verify_hashes } => doctor::run(verify_hashes).await,
        Commands::Gc => gc::run().await,
        Commands::Link { comfyui, a1111 } => link::run(comfyui.as_deref(), a1111.as_deref()).await,
        Commands::Auth { provider } => auth::run(&provider).await,
        Commands::Update => update::run().await,
        Commands::Export => export::run().await,
        Commands::Import { path } => import::run(&path).await,
        Commands::CliSchema => {
            dump_cli_schema();
            Ok(())
        }
        Commands::Popular {
            r#type,
            r#for,
            period,
        } => popular::run(r#type.as_deref(), r#for.as_deref(), &period).await,
    }
}

fn dump_cli_schema() {
    let cmd = Cli::command();
    let mut commands = Vec::new();

    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }

        let name = sub.get_name().to_string();
        let description = sub
            .get_about()
            .map(|s| s.to_string())
            .unwrap_or_default();

        let mut args = Vec::new();
        let mut flags = Vec::new();

        for arg in sub.get_arguments() {
            if arg.get_id() == "help" || arg.get_id() == "version" {
                continue;
            }

            let id = arg.get_id().to_string();
            let help = arg
                .get_help()
                .map(|s| s.to_string())
                .unwrap_or_default();
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
                    flag.as_object_mut().unwrap().insert("short".into(), serde_json::json!(s));
                }
                if let Some(d) = &default {
                    flag.as_object_mut().unwrap().insert("default".into(), serde_json::json!(d));
                }
                flags.push(flag);
            }
        }

        // Build usage string
        let usage = {
            let mut parts = vec![format!("mods {name}")];
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
            "name": name,
            "description": description,
            "usage": usage,
            "args": args,
            "flags": flags,
        }));
    }

    let schema = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commands": commands,
    });

    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
}
