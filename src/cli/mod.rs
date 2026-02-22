mod init;
mod install;
mod uninstall;
mod list;
mod info;
mod search;
mod space;
mod doctor;
mod gc;
mod link;
mod auth;
mod update;
mod export;
mod import;
mod popular;

use anyhow::Result;
use clap::{Parser, Subcommand};

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
    Doctor,

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
        Commands::Install { id, variant, dry_run } => {
            install::run(&id, variant.as_deref(), dry_run).await
        }
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
        Commands::Space => space::run().await,
        Commands::Doctor => doctor::run().await,
        Commands::Gc => gc::run().await,
        Commands::Link { comfyui, a1111 } => {
            link::run(comfyui.as_deref(), a1111.as_deref()).await
        }
        Commands::Auth { provider } => auth::run(&provider).await,
        Commands::Update => update::run().await,
        Commands::Export => export::run().await,
        Commands::Import { path } => import::run(&path).await,
        Commands::Popular {
            r#type,
            r#for,
            period,
        } => popular::run(r#type.as_deref(), r#for.as_deref(), &period).await,
    }
}
