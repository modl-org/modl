pub(crate) mod analysis;
mod auth;
mod civitai;
mod compare;
mod config;
mod datasets;
mod describe;
mod detect;
mod doctor;
pub(crate) mod edit;
mod enhance;
mod export_import;
mod fmt;
mod gc;
pub(crate) mod generate;
mod gpu;
mod ground;
mod hub_pull;
mod info;
mod init;
mod install;
mod link;
mod list;
mod login;
mod logout;
mod mcp;
mod outputs;
mod popular;
mod preprocess;
mod push;
mod remove_bg;
mod run;
mod runtime;
mod score;
mod search;
mod segment;
mod serve;
mod space;
mod train;
mod train_setup;
mod train_status;
mod uninstall;
mod update;
mod upgrade;
mod upscale;
mod whoami;
pub(crate) mod worker;

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
    Style:     --rank 16  --lr 1e-4   --steps 3000-3500  --batch-size 1
    Character: --rank 16  --lr 1e-4   --steps 1500       --batch-size 1
    ⚡ Only 6B params — trains very fast (~2 it/s on 4090, ~1.3s on 5090).
    ⚡ No quantization needed on 24GB+ cards (~17GB VRAM without quantize).
    ⚠  Do NOT exceed --lr 1e-4 — higher LR breaks turbo distillation.
    ⚠  Uses a DD (de-distillation) training adapter automatically.
       The adapter prevents breaking the 8-step turbo during training.
       Works for ~5k-10k steps; beyond ~20k distillation degrades.
    Style presets auto-apply these settings (per Ostris / ai-toolkit):
      • Differential guidance (scale=3): overshoots target to converge.
      • Literal captions: describe content, not style. E.g. \"a bear\"
        not \"a child's drawing of a bear\". Style is learned implicitly.
      • Trigger word optional (modl still asks for one as a fallback).
      • cache_text_embeddings=true: unloads text encoder after caching.
    For extreme style (e.g. children's art), Ostris recommends resuming
      from ~2000 steps with high-noise timestep bias to rebuild composition.
      This is a two-phase approach: balanced first, then high-noise.
    Character LoRAs use balanced timesteps, trigger word required.
    Inference: 8 steps, guidance 1.0 (no CFG). Adapter removed automatically.

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

const ENHANCE_EXAMPLES: &str = "\
\x1b[1mExamples:\x1b[0m
  # Enhance a prompt with moderate intensity (default)
  modl enhance \"a cat on the moon\"

  # Subtle enhancement — just quality tags
  modl enhance \"portrait of a woman\" --intensity subtle

  # Aggressive enhancement with model-specific tags
  modl enhance \"sunset over mountains\" --intensity aggressive --model sdxl

  # Output as JSON for scripting
  modl enhance \"product photo\" --json
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

  # img2img: re-style an existing image (lower strength = closer to original)
  modl generate \"watercolor painting\" --init-image photo.png --strength 0.6

  # Inpainting: regenerate masked region (white = edit, black = keep)
  modl generate \"a garden with roses\" --init-image photo.png --mask mask.png

  # Inpainting auto-routes to Flux Fill if installed (best quality)
  modl generate \"wooden table\" --base flux-dev --init-image room.png --mask table_mask.png
";

const EDIT_EXAMPLES: &str = "\
\x1b[1mExamples:\x1b[0m
  # Edit with default model (qwen-image-edit-2511)
  modl edit \"make the sky sunset orange\" --image photo.png

  # Use a faster/smaller model
  modl edit \"replace the chair with a sofa\" --image room.png --base klein-4b

  # Generate multiple edit variants
  modl edit \"add sunglasses\" --image portrait.png --count 3

  # Output as JSON for scripting
  modl edit \"remove the text\" --image screenshot.png --json
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
  # Pull a model from the registry
  modl pull flux-dev

  # Pull directly from HuggingFace
  modl pull hf:stabilityai/stable-diffusion-xl-base-1.0

  # Force a specific variant
  modl pull flux-dev --variant fp8

  # Preview what would be downloaded
  modl pull sdxl-base-1.0 --dry-run
";

/// Inpainting method for `modl generate --mask`.
#[derive(Clone, Debug, ValueEnum)]
pub enum InpaintMethod {
    /// Auto-select: LanPaint for supported models, standard for others
    Auto,
    /// LanPaint training-free inpainting (supports Klein, Z-Image)
    Lanpaint,
    /// Standard diffusers inpainting or Flux Fill
    Standard,
}

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
    about = "AI image generation toolkit",
    version,
    propagate_version = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

// ── Subcommand groups ────────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum WorkerSubcommands {
    /// Start the persistent worker daemon (keeps models in VRAM)
    Start {
        /// Idle timeout in seconds (worker shuts down after this long without requests)
        #[arg(long, default_value = "600")]
        timeout: u32,
    },

    /// Stop the persistent worker daemon
    Stop,

    /// Show worker status (loaded models, VRAM, uptime)
    Status,
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
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Delete a training run (output, logs, LoRA, and DB records)
    Rm {
        /// Training run name to delete
        name: String,
    },

    /// List training runs
    Ls,
}

#[derive(Subcommand)]
pub enum VisionCommands {
    /// Describe image content using vision-language AI (detailed captioning)
    Describe {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Detail level: brief, detailed, verbose
        #[arg(long, default_value = "detailed")]
        detail: String,
        /// VL model: qwen3-vl-8b (default, quality, 16GB) or qwen3-vl-2b (fast, 4GB)
        #[arg(long)]
        model: Option<String>,
        /// Use smaller/faster VL model (qwen3-vl-2b, 4GB) — less accurate
        #[arg(long)]
        fast: bool,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Score image aesthetic quality on a 1-10 scale using AI
    Score {
        /// Image file(s) or directory to score
        #[arg(required = true)]
        paths: Vec<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Detect faces in images
    Detect {
        /// Image file(s) or directory to analyze
        #[arg(required = true)]
        paths: Vec<String>,
        /// Detection type (currently: face)
        #[arg(long, default_value = "face")]
        r#type: String,
        /// Include face embeddings for identity matching
        #[arg(long)]
        embeddings: bool,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Find objects in images by text description
    Ground {
        /// Text query -- what to find (e.g. "coffee cup", "person")
        query: String,
        /// Image file(s) or directory to search
        #[arg(required = true)]
        paths: Vec<String>,
        /// Minimum confidence threshold
        #[arg(long)]
        threshold: Option<f64>,
        /// VL model: qwen3-vl-8b (default, quality, 16GB) or qwen3-vl-2b (fast, 4GB)
        #[arg(long)]
        model: Option<String>,
        /// Use smaller/faster VL model (qwen3-vl-2b, 4GB) — less accurate
        #[arg(long)]
        fast: bool,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compare images using CLIP similarity
    Compare {
        /// Image file(s) or directory to compare
        #[arg(required = true)]
        paths: Vec<String>,
        /// Reference image (compare all others against this)
        #[arg(long)]
        reference: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ProcessCommands {
    /// Upscale images 2x or 4x using Real-ESRGAN super-resolution
    Upscale {
        /// Image file(s) or directory to upscale
        #[arg(required = true)]
        paths: Vec<String>,
        /// Scale factor (2 or 4)
        #[arg(long, default_value = "4")]
        scale: u32,
        /// Upscaler model ID (default: realesrgan-x4plus)
        #[arg(long, default_value = "realesrgan-x4plus")]
        model: String,
        /// Output directory (default: ~/.modl/outputs/<date>/)
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Remove image background, output transparent PNG
    #[command(name = "remove-bg")]
    RemoveBg {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Output directory (default: ~/.modl/outputs/<date>/)
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Generate a segmentation mask for use with generate --mask (inpainting)
    Segment {
        /// Input image
        image: String,
        /// Output mask path (default: <image>_mask.png)
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Segmentation method: bbox, background, sam
        #[arg(long, default_value = "bbox")]
        method: String,
        /// Bounding box: x1,y1,x2,y2 (for bbox/sam methods)
        #[arg(long)]
        bbox: Option<String>,
        /// Point prompt: x,y (for sam method)
        #[arg(long)]
        point: Option<String>,
        /// Expand mask by N pixels (feathering)
        #[arg(long, default_value = "10")]
        expand: u32,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Extract control images from an image (canny, depth, pose, softedge, scribble)
    Preprocess {
        /// Preprocessing method
        #[command(subcommand)]
        command: preprocess::PreprocessMethod,
    },
}

#[derive(Subcommand)]
pub enum SystemCommands {
    /// Remove unreferenced files from the store
    Gc,

    /// Fetch latest registry index
    Update,

    /// Link a tool's model folder (ComfyUI, A1111)
    Link {
        /// Path to model directory (assumes ComfyUI layout)
        path: Option<String>,
        /// Path to ComfyUI installation
        #[arg(long)]
        comfyui: Option<String>,
        /// Path to A1111 installation
        #[arg(long)]
        a1111: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum AuthCommands {
    /// Login to modl hub
    Login,
    /// Logout from modl hub
    Logout,
    /// Show hub account info
    Whoami,
    /// Configure source credentials (HuggingFace, CivitAI) for gated model downloads
    Add {
        /// Auth provider: huggingface or civitai
        #[arg(value_enum)]
        provider: AuthProvider,
    },
}

#[derive(Subcommand)]
pub enum GpuCommands {
    /// Provision a remote GPU instance
    Attach {
        /// GPU spec (e.g. a100, a10g, h100)
        spec: String,
        /// Idle timeout before auto-shutdown (e.g. 30m, 1h)
        #[arg(long, default_value = "30m")]
        idle: String,
    },

    /// Shut down the attached GPU instance
    Detach,

    /// Show running GPU instances
    Status,

    /// Open a shell to the attached GPU instance
    Ssh,

    /// Run as a GPU agent on a remote instance (hidden — used by Vast.ai entrypoint)
    #[command(hide = true)]
    Agent {
        /// Session token for authenticating with the orchestrator
        #[arg(long)]
        session_token: String,
        /// Orchestrator API base URL
        #[arg(long, default_value = "https://api.modl.run")]
        api_base: String,
    },
}

// ── Top-level commands ───────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum Commands {
    // ── Primary Actions (flat) ───────────────────────────────────────
    /// Generate images from text prompts (txt2img, img2img, inpainting)
    ///
    /// Supports multiple modes:
    ///   txt2img:    modl generate "prompt"
    ///   img2img:    modl generate "prompt" --init-image photo.png --strength 0.6
    ///   inpainting: modl generate "prompt" --init-image photo.png --mask mask.png
    ///
    /// Models:
    ///   flux-schnell (default)   4 steps, 12B, fastest
    ///   flux-dev                28 steps, 12B, best quality
    ///   z-image-turbo            8 steps, 6B, fast
    ///   z-image                 30 steps, 6B, quality
    ///   qwen-image              40 steps, 20B, text rendering
    ///   chroma                  45 steps, 12B, artistic
    ///   sdxl                    30 steps, 3.5B, huge LoRA ecosystem
    ///
    /// Use --lora to apply a trained LoRA. Use --controlnet for structural guidance.
    #[command(verbatim_doc_comment, after_help = GENERATE_EXAMPLES)]
    Generate {
        /// Text prompt for image generation
        prompt: String,
        /// Base model to use (default: flux-schnell)
        #[arg(long)]
        base: Option<String>,
        /// LoRA name or path to apply
        #[arg(long)]
        lora: Option<String>,
        /// LoRA strength/weight (0.0 = no effect, 1.0 = full strength)
        #[arg(long, default_value = "1.0")]
        lora_strength: f32,
        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
        /// Image size preset (1:1, 16:9, 9:16, 4:3, 3:4) or WxH [default: 1:1, or init-image dimensions]
        #[arg(long)]
        size: Option<String>,
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
        #[arg(long, hide = true)]
        cloud: bool,
        /// Cloud provider to use (modal, replicate, runpod)
        #[arg(long, value_enum, hide = true)]
        provider: Option<CloudProvider>,
        /// Source image for img2img or inpainting (use with --mask for inpainting)
        #[arg(long)]
        init_image: Option<String>,
        /// Mask image for inpainting: white pixels = regenerate, black = preserve. Requires --init-image
        #[arg(long)]
        mask: Option<String>,
        /// Denoising strength for img2img (0.0 = identical to input, 1.0 = fully new). Default: 0.75
        #[arg(long)]
        strength: Option<f32>,
        /// Inpainting method: auto (default), lanpaint (training-free), standard (diffusers/Fill)
        #[arg(long, value_enum, default_value = "auto")]
        inpaint: InpaintMethod,
        /// Control image for ControlNet conditioning (can be repeated up to 2x)
        #[arg(long)]
        controlnet: Vec<String>,
        /// ControlNet conditioning strength (comma-separated if multiple)
        #[arg(long, default_value = "0.75")]
        cn_strength: String,
        /// Stop applying ControlNet at this fraction of total steps (comma-separated)
        #[arg(long, default_value = "0.8")]
        cn_end: String,
        /// ControlNet type: canny, depth, pose, softedge, scribble, hed, mlsd, gray, normal (auto-detected from filename if omitted)
        #[arg(long)]
        cn_type: Option<String>,
        /// Style reference image (can be repeated; backend varies by model)
        #[arg(long)]
        style_ref: Vec<String>,
        /// Style reference strength (0.0-1.0)
        #[arg(long, default_value = "0.6")]
        style_strength: f32,
        /// Style type: style, face, content (SDXL IP-Adapter variants only)
        #[arg(long)]
        style_type: Option<String>,
        /// Lightning LoRA for ~10x faster generation (4 or 8 steps instead of 40-50).
        /// Use --fast for 4-step (fastest) or --fast 8 for 8-step (higher quality).
        /// Auto-applies a model-specific distillation LoRA. Cannot combine with --lora.
        /// Supported: qwen-image, qwen-image-edit.
        #[arg(long, num_args = 0..=1, default_missing_value = "4", value_name = "STEPS")]
        fast: Option<u32>,
        /// Force one-shot mode (skip persistent worker, cold start every time)
        #[arg(long)]
        no_worker: bool,
        /// Run on a remote GPU instance (auto-provisions via Vast.ai if no active session)
        #[arg(long)]
        attach_gpu: bool,
        /// GPU type for remote execution (e.g. a100, a10g, h100, rtx4090)
        #[arg(long, default_value = "a100")]
        gpu_type: String,
        /// Output result as JSON (suppresses progress output)
        #[arg(long)]
        json: bool,
    },

    /// Edit images using natural language instructions (no mask needed)
    ///
    /// Unlike generate --mask (pixel-level inpainting), edit uses instruction-following
    /// models that understand "change X to Y" without needing a mask.
    ///
    /// Models:
    ///   qwen-image-edit-2511 (default)  40 steps, 20B, best quality
    ///   klein-4b                         4 steps, 4B, fastest
    ///   klein-9b                         4 steps, 9B, balanced
    ///   flux2-dev                       28 steps, 24B, flux-based
    ///
    /// Examples:
    ///   modl edit "make the sky sunset orange" --image photo.png
    ///   modl edit "replace the chair with a sofa" --image room.png --base klein-4b
    ///   modl edit "add sunglasses" --image portrait.png --count 3
    #[command(verbatim_doc_comment, after_help = EDIT_EXAMPLES)]
    Edit {
        /// Natural language edit instruction (e.g. "make the sky sunset orange")
        prompt: String,
        /// Source image(s) — local path or URL (can be repeated)
        #[arg(long, required = true)]
        image: Vec<String>,
        /// LoRA name or path to apply (combine with reference images for multi-character scenes)
        #[arg(long)]
        lora: Option<String>,
        /// LoRA strength/weight (0.0 = no effect, 1.0 = full strength)
        #[arg(long, default_value = "1.0")]
        lora_strength: f32,
        /// Base model to use (default: qwen-image-edit)
        #[arg(long)]
        base: Option<String>,
        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
        /// Number of inference steps
        #[arg(long)]
        steps: Option<u32>,
        /// Guidance scale
        #[arg(long)]
        guidance: Option<f32>,
        /// Number of output images
        #[arg(long, default_value = "1")]
        count: u32,
        /// Output size (e.g. "16:9", "1820x1024") — larger than source for outpainting
        #[arg(long)]
        size: Option<String>,
        /// Lightning LoRA for ~10x faster editing (4 or 8 steps instead of 40-50).
        /// Use --fast for 4-step (fastest) or --fast 8 for 8-step (higher quality).
        /// Supported: qwen-image-edit.
        #[arg(long, num_args = 0..=1, default_missing_value = "4", value_name = "STEPS")]
        fast: Option<u32>,
        /// Run on cloud
        #[arg(long, hide = true)]
        cloud: bool,
        /// Cloud provider
        #[arg(long, value_enum, hide = true)]
        provider: Option<CloudProvider>,
        /// Force one-shot mode
        #[arg(long)]
        no_worker: bool,
        /// Run on a remote GPU instance (auto-provisions via Vast.ai if no active session)
        #[arg(long)]
        attach_gpu: bool,
        /// GPU type for remote execution (e.g. a100, a10g, h100, rtx4090)
        #[arg(long, default_value = "a100")]
        gpu_type: String,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Train LoRA models
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
        /// Class word for character/object (e.g. "man", "woman", "dog")
        #[arg(long, alias = "class")]
        class_word: Option<String>,
        /// Resume from a checkpoint .safetensors file
        #[arg(long)]
        resume: Option<String>,
        /// Sample image frequency (steps). 0 = only at the end. Default: auto (steps/10)
        #[arg(long)]
        sample_every: Option<u32>,
        /// Load a full TrainJobSpec YAML (escape hatch)
        #[arg(long)]
        config: Option<String>,
        /// Generate spec and print it without executing
        #[arg(long)]
        dry_run: bool,
        /// Run training on a cloud provider instead of locally
        #[arg(long, hide = true)]
        cloud: bool,
        /// Cloud provider to use (modal, replicate, runpod)
        #[arg(long, value_enum, hide = true)]
        provider: Option<CloudProvider>,
        /// Run on a remote GPU instance (auto-provisions via Vast.ai if no active session)
        #[arg(long)]
        attach_gpu: bool,
        /// GPU type for remote execution (e.g. a100, a10g, h100, rtx4090)
        #[arg(long, default_value = "a100")]
        gpu_type: String,
    },

    /// Enhance prompts with AI quality tags and descriptors for better generation results
    #[command(after_help = ENHANCE_EXAMPLES, hide = true)]
    Enhance {
        /// Text prompt to enhance
        prompt: String,
        /// Target model family hint (e.g., sdxl, flux, sd3) for model-specific tags
        #[arg(long)]
        model: Option<String>,
        /// Enhancement intensity: subtle, moderate, aggressive
        #[arg(long, default_value = "moderate")]
        intensity: String,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    // ── Model Management ─────────────────────────────────────────────
    /// Download models from registry, HuggingFace (hf:), CivitAI (civitai:), or hub (user/slug)
    #[command(after_help = MODEL_PULL_EXAMPLES)]
    Pull {
        /// Registry ID (e.g. flux-dev) or HuggingFace repo (hf:owner/model)
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

    /// Push LoRAs or datasets to modl hub.
    ///
    /// By default, pushes only the final checkpoint + sample images. The sample
    /// evolution grid in the hub UI works from the samples alone — intermediate
    /// checkpoint weights are not needed for visualization.
    ///
    /// Use --checkpoints to upload all intermediate checkpoint files (useful if
    /// you want others to download and test specific training steps).
    Push {
        /// Asset kind: lora or dataset
        #[arg(value_parser = ["lora", "dataset"])]
        kind: String,
        /// Source: file path, directory, or training run name
        source: String,
        /// Hub slug/name for the item
        #[arg(long)]
        name: String,
        /// Visibility: public (discoverable), unlisted (link-only), or private (owner-only)
        #[arg(long, default_value = "public", value_parser = ["public", "private", "unlisted"])]
        visibility: String,
        /// Optional description
        #[arg(long)]
        description: Option<String>,
        /// Base model (inferred from training manifest if omitted)
        #[arg(long)]
        base: Option<String>,
        /// Trigger word(s) (repeat flag for multiple)
        #[arg(long = "trigger")]
        trigger_words: Vec<String>,
        /// Upload all intermediate checkpoints, not just the final one
        #[arg(long)]
        checkpoints: bool,
        /// Optional owner username override
        #[arg(long)]
        owner: Option<String>,
    },

    /// List installed models
    Ls {
        /// Filter by asset type (checkpoint, lora, vae, text_encoder, etc.)
        #[arg(long, short = 't', value_enum)]
        r#type: Option<AssetType>,
        /// Show disk usage summary grouped by type
        #[arg(long)]
        summary: bool,
        /// Show all items including internal dependencies (VAEs, text encoders, etc.)
        #[arg(long, short = 'a')]
        all: bool,
    },

    /// Remove an installed model
    Rm {
        /// Model ID to remove
        id: String,
        /// Force removal even if other items depend on this
        #[arg(long)]
        force: bool,
    },

    /// Search the model registry
    Search {
        /// Search query (optional with --popular)
        query: Option<String>,
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
        /// Show popular/trending models (ignores query)
        #[arg(long)]
        popular: bool,
        /// Search CivitAI for LoRAs instead of the modl registry
        #[arg(long)]
        civitai: bool,
        /// Base model filter for CivitAI search (e.g., "SDXL 1.0", "Flux.1 D")
        #[arg(long)]
        base_model: Option<String>,
        /// Sort order for CivitAI search (Most Downloaded, Highest Rated, Newest)
        #[arg(long)]
        sort: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
        /// Show all items including internal dependencies (VAEs, text encoders, etc.)
        #[arg(long, short = 'a')]
        all: bool,
    },

    /// Show model details
    Info {
        /// Model ID to inspect
        id: String,
    },

    // ── Vision & Processing ─────────────────────────────────────────
    /// Image understanding tools (describe, score, detect, ground, compare)
    Vision {
        #[command(subcommand)]
        command: VisionCommands,
    },

    /// Image processing tools (upscale, remove-bg, segment, preprocess)
    Process {
        #[command(subcommand)]
        command: ProcessCommands,
    },

    // ── Remote GPU ───────────────────────────────────────────────────
    /// Manage remote GPU sessions
    #[command(hide = true)]
    Gpu {
        #[command(subcommand)]
        command: GpuCommands,
    },

    // ── Auth ──────────────────────────────────────────────────────────
    /// Authentication: hub login/logout and source credentials (HuggingFace, CivitAI)
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },

    // ── Data Management ──────────────────────────────────────────────
    /// Manage training datasets
    #[command(after_help = DATASET_EXAMPLES)]
    Dataset {
        #[command(subcommand)]
        command: datasets::DatasetCommands,
    },

    /// Browse and manage generated outputs
    Outputs {
        #[command(subcommand)]
        command: outputs::OutputCommands,
    },

    // ── Services ─────────────────────────────────────────────────────
    /// Launch the web UI
    Serve {
        /// Port to bind the preview server on
        #[arg(long, default_value = "3939")]
        port: u16,
        /// Don't auto-open the browser
        #[arg(long)]
        no_open: bool,
        /// Run in foreground (blocks terminal; default is background/daemon)
        #[arg(long)]
        foreground: bool,
        /// Install modl serve as a system service (systemd on Linux, launchd on macOS)
        #[arg(long)]
        install_service: bool,
        /// Remove the modl system service
        #[arg(long)]
        remove_service: bool,
    },

    /// Manage the persistent GPU worker (keeps models in VRAM)
    Worker {
        #[command(subcommand)]
        command: WorkerSubcommands,
    },

    /// Start MCP server for AI agents
    ///
    /// Exposes modl tools (generate, list_models, pull_model, describe, score,
    /// upscale, remove_bg, enhance) over the MCP stdio transport.
    ///
    ///   { "mcpServers": { "modl": { "command": "modl", "args": ["mcp"] } } }
    Mcp,

    // ── System ───────────────────────────────────────────────────────
    /// View or update configuration (e.g., storage.root, gpu.vram_mb)
    Config {
        /// Config key to view or set (e.g., storage.root)
        key: Option<String>,
        /// New value (required when setting a key)
        value: Option<String>,
    },

    /// Check for broken symlinks, missing deps, corrupt files
    Doctor {
        /// Also verify SHA256 hashes (slow for large files)
        #[arg(long)]
        verify_hashes: bool,
        /// Re-populate database from orphaned store files
        #[arg(long)]
        repair: bool,
    },

    /// Update modl CLI to the latest release
    Upgrade,

    /// System maintenance (gc, update, link)
    System {
        #[command(subcommand)]
        command: SystemCommands,
    },

    // ── Workflow ─────────────────────────────────────────────────────
    /// Execute a workflow from a YAML spec file
    #[command(long_about = "\
Execute a batch workflow of generate/edit steps from a YAML spec.

A workflow declares a model, an optional LoRA, shared defaults, and an ordered \
list of steps. Each step is a generate or edit job. Outputs of earlier steps \
can be referenced by later steps via $step-id.outputs[N].

All outputs land in ~/.modl/outputs/<date>/ alongside regular `modl generate` \
results and are visible in `modl serve`. Each step is registered as a job \
row tagged with the workflow name for later filtering.

EXAMPLE

    name: book-chapter-3
    model: flux2-klein-4b
    lora: my-son-v2

    defaults:
      width: 1024
      height: 1024
      steps: 4
      guidance: 1.0

    steps:
      - id: scene-1
        generate: \"OHWX reading a book under a tree\"
        seeds: [42, 7, 99, 333]     # 4 variations for winner selection

      - id: scene-1-rain
        edit: \"$scene-1.outputs[0]\"  # reference first seed variation
        prompt: \"add heavy rain and puddles\"
        seed: 42

Run with:  modl run book-chapter-3.yaml

See `modl/docs/guides/workflows.md` for the full reference.")]
    Run {
        /// Workflow spec file (.yaml)
        spec: String,
        /// Auto-pull missing models before running (not yet implemented)
        #[arg(long)]
        auto_pull: bool,
    },

    // ── Hidden ───────────────────────────────────────────────────────
    /// Login to modl hub (alias for `modl auth login`)
    #[command(hide = true)]
    Login,
    /// Interactive first-run setup
    #[command(hide = true)]
    Init {
        /// Skip all prompts (use ~/modl, auto-detect GPU, no tool targets)
        #[arg(long)]
        defaults: bool,
        /// Override storage root (default: ~/modl)
        #[arg(long)]
        root: Option<String>,
    },

    /// Export data (outputs, trained LoRAs, DB) to a backup archive
    #[command(hide = true)]
    Export {
        /// Output archive path (.tar.zst)
        output: String,
        /// Exclude generation outputs (DB + LoRAs only)
        #[arg(long)]
        no_outputs: bool,
        /// Only include outputs after this date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,
    },

    /// Import data from a backup archive
    #[command(hide = true)]
    Import {
        /// Path to .tar.zst backup archive
        path: String,
        /// Preview what would be restored without making changes
        #[arg(long)]
        dry_run: bool,
        /// Overwrite existing files (default: skip)
        #[arg(long)]
        overwrite: bool,
    },

    /// Manage embedded Python runtime
    #[command(hide = true)]
    Runtime {
        #[command(subcommand)]
        command: runtime::RuntimeCommands,
    },

    /// Dump CLI schema as JSON
    #[command(hide = true)]
    CliSchema,
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        // ── Primary Actions ──────────────────────────────────────────
        Commands::Generate {
            prompt,
            base,
            lora,
            lora_strength,
            seed,
            size,
            steps,
            guidance,
            count,
            init_image,
            mask,
            strength,
            inpaint,
            controlnet,
            cn_strength,
            cn_end,
            cn_type,
            style_ref,
            style_strength,
            style_type,
            fast,
            cloud,
            provider,
            no_worker,
            attach_gpu,
            gpu_type,
            json,
        } => {
            generate::run(generate::GenerateArgs {
                prompt: &prompt,
                base: base.as_deref(),
                lora: lora.as_deref(),
                lora_strength,
                seed,
                size: size.as_deref(),
                steps,
                guidance,
                count,
                init_image: init_image.as_deref(),
                mask: mask.as_deref(),
                strength,
                inpaint,
                controlnet: &controlnet,
                cn_strength: &cn_strength,
                cn_end: &cn_end,
                cn_type: cn_type.as_deref(),
                style_ref: &style_ref,
                style_strength,
                style_type: style_type.as_deref(),
                fast,
                cloud,
                provider,
                no_worker,
                attach_gpu,
                gpu_type: &gpu_type,
                json,
            })
            .await
        }
        Commands::Edit {
            prompt,
            image,
            lora,
            lora_strength,
            base,
            seed,
            steps,
            guidance,
            count,
            size,
            fast,
            cloud,
            provider,
            no_worker,
            attach_gpu,
            gpu_type,
            json,
        } => {
            edit::run(edit::EditArgs {
                prompt: &prompt,
                images: &image,
                lora: lora.as_deref(),
                lora_strength,
                base: base.as_deref(),
                seed,
                steps,
                guidance,
                count,
                size: size.as_deref(),
                fast,
                cloud,
                provider,
                no_worker,
                attach_gpu,
                gpu_type: &gpu_type,
                json,
            })
            .await
        }
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
            class_word,
            resume,
            sample_every,
            config,
            dry_run,
            cloud,
            provider,
            attach_gpu,
            gpu_type,
        } => match command {
            Some(TrainSubcommands::Setup { reinstall }) => train_setup::run(reinstall).await,
            Some(TrainSubcommands::Status { name, watch, json }) => {
                train_status::run(name.as_deref(), watch, json)?;
                Ok(())
            }
            Some(TrainSubcommands::Rm { name }) => {
                use console::style;
                crate::core::training::delete_training_run(&name)?;
                println!("{} Deleted training run '{}'", style("✓").green(), name);
                Ok(())
            }
            Some(TrainSubcommands::Ls) => {
                let runs = crate::core::training::list_training_runs()?;
                if runs.is_empty() {
                    println!("No training runs found.");
                } else {
                    for name in &runs {
                        println!("  {name}");
                    }
                    println!("\n{} training run(s)", runs.len());
                }
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
                        class_word,
                        resume,
                        sample_every,
                    },
                    config.as_deref(),
                    dry_run,
                    cloud,
                    provider,
                    attach_gpu,
                    &gpu_type,
                )
                .await
            }
        },
        Commands::Enhance {
            prompt,
            model,
            intensity,
            json,
        } => enhance::run(&prompt, model.as_deref(), &intensity, json).await,

        // ── Model Management ─────────────────────────────────────────
        Commands::Pull {
            id,
            variant,
            dry_run,
            force,
        } => {
            if crate::core::hub::parse_hub_ref(&id).is_some() {
                if dry_run {
                    anyhow::bail!("--dry-run is not supported for hub pulls");
                }
                if variant.is_some() {
                    anyhow::bail!("--variant is not supported for hub pulls");
                }
                if force {
                    anyhow::bail!("--force is not supported for hub pulls");
                }
                hub_pull::run(&id).await
            } else if let Some(version_id) = id.strip_prefix("civitai:") {
                if dry_run {
                    anyhow::bail!("--dry-run is not supported for CivitAI installs");
                }
                civitai::install(version_id, force).await
            } else {
                install::run(&id, variant.as_deref(), dry_run, force).await
            }
        }
        Commands::Push {
            kind,
            source,
            name,
            visibility,
            description,
            base,
            trigger_words,
            checkpoints,
            owner,
        } => {
            push::run(
                &kind,
                &source,
                &name,
                &visibility,
                description.as_deref(),
                base.as_deref(),
                &trigger_words,
                checkpoints,
                owner.as_deref(),
            )
            .await
        }
        Commands::Ls {
            r#type,
            summary,
            all,
        } => {
            if summary {
                space::run(all).await
            } else {
                list::run(r#type, all).await
            }
        }
        Commands::Rm { id, force } => uninstall::run(&id, force).await,
        Commands::Search {
            query,
            r#type,
            r#for,
            tag,
            min_rating,
            popular,
            civitai: civitai_flag,
            base_model,
            sort,
            json,
            all,
        } => {
            if civitai_flag {
                let q = query.as_deref().unwrap_or("");
                if q.is_empty() {
                    anyhow::bail!("Search query required for --civitai");
                }
                civitai::search(q, base_model.as_deref(), sort.as_deref()).await
            } else if popular {
                popular::run(r#type, r#for.as_deref()).await
            } else {
                let q = query.as_deref().unwrap_or("");
                if q.is_empty() {
                    anyhow::bail!("Search query required (or use --popular)");
                }
                search::run(
                    q,
                    r#type,
                    r#for.as_deref(),
                    tag.as_deref(),
                    min_rating,
                    json,
                    all,
                )
                .await
            }
        }
        Commands::Info { id } => info::run(&id).await,

        // ── Vision (image → text/data) ─────────────────────────────
        Commands::Vision { command } => match command {
            VisionCommands::Describe {
                paths,
                detail,
                model,
                fast,
                json,
            } => {
                let effective_model = if fast && model.is_none() {
                    Some("qwen3-vl-2b".to_string())
                } else {
                    model
                };
                describe::run(&paths, &detail, effective_model.as_deref(), json).await
            }
            VisionCommands::Score { paths, json } => score::run(&paths, json).await,
            VisionCommands::Detect {
                paths,
                r#type,
                embeddings,
                json,
            } => detect::run(&paths, &r#type, embeddings, json).await,
            VisionCommands::Ground {
                query,
                paths,
                threshold,
                model,
                fast,
                json,
            } => {
                let effective_model = if fast && model.is_none() {
                    Some("qwen3-vl-2b".to_string())
                } else {
                    model
                };
                ground::run(&query, &paths, threshold, effective_model.as_deref(), json).await
            }
            VisionCommands::Compare {
                paths,
                reference,
                json,
            } => compare::run(&paths, reference.as_deref(), json).await,
        },

        // ── Process (image → image) ─────────────────────────────────
        Commands::Process { command } => match command {
            ProcessCommands::Upscale {
                paths,
                scale,
                model,
                output,
                json,
            } => upscale::run(&paths, output.as_deref(), scale, &model, json).await,
            ProcessCommands::RemoveBg {
                paths,
                output,
                json,
            } => remove_bg::run(&paths, output.as_deref(), json).await,
            ProcessCommands::Segment {
                image,
                output,
                method,
                bbox,
                point,
                expand,
                json,
            } => {
                segment::run(
                    &image,
                    output.as_deref(),
                    &method,
                    bbox.as_deref(),
                    point.as_deref(),
                    expand,
                    json,
                )
                .await
            }
            ProcessCommands::Preprocess { command } => preprocess::run(command).await,
        },

        // ── Remote GPU ───────────────────────────────────────────────
        Commands::Gpu { command } => match command {
            GpuCommands::Attach { spec, idle } => gpu::attach(&spec, &idle).await,
            GpuCommands::Detach => gpu::detach().await,
            GpuCommands::Status => gpu::status().await,
            GpuCommands::Ssh => gpu::ssh().await,
            GpuCommands::Agent {
                session_token,
                api_base,
            } => gpu::agent(&session_token, &api_base).await,
        },

        // ── Auth ─────────────────────────────────────────────────────
        Commands::Auth { command } => match command {
            AuthCommands::Login => login::run().await,
            AuthCommands::Logout => logout::run().await,
            AuthCommands::Whoami => whoami::run().await,
            AuthCommands::Add { provider } => auth::run(provider).await,
        },
        Commands::Login => login::run().await,

        // ── Data Management ──────────────────────────────────────────
        Commands::Dataset { command } => datasets::run(command).await,
        Commands::Outputs { command } => outputs::run(command).await,

        // ── Services ─────────────────────────────────────────────────
        Commands::Serve {
            port,
            no_open,
            foreground,
            install_service,
            remove_service,
        } => {
            if install_service {
                serve::install_service(port).await
            } else if remove_service {
                serve::remove_service().await
            } else {
                serve::run(port, no_open, foreground).await
            }
        }
        Commands::Worker { command } => match command {
            WorkerSubcommands::Start { timeout } => worker::start(timeout).await,
            WorkerSubcommands::Stop => worker::stop().await,
            WorkerSubcommands::Status => worker::status().await,
        },
        Commands::Mcp => mcp::run().await,

        // ── System ───────────────────────────────────────────────────
        Commands::Config { key, value } => config::run(key.as_deref(), value.as_deref()).await,
        Commands::Doctor {
            verify_hashes,
            repair,
        } => doctor::run(verify_hashes, repair).await,
        Commands::Upgrade => upgrade::run().await,
        Commands::System { command } => match command {
            SystemCommands::Gc => gc::run().await,
            SystemCommands::Update => update::run().await,
            SystemCommands::Link {
                path,
                comfyui,
                a1111,
            } => {
                let comfy = comfyui.or(path);
                link::run(comfy.as_deref(), a1111.as_deref()).await
            }
        },

        // ── Workflow ────────────────────────────────────────────────
        Commands::Run { spec, auto_pull } => run::run(&spec, auto_pull).await,

        // ── Hidden ───────────────────────────────────────────────────
        Commands::Init { defaults, root } => init::run(defaults, root.as_deref()).await,
        Commands::Export {
            output,
            no_outputs,
            since,
        } => export_import::run_export(&output, no_outputs, since.as_deref()),
        Commands::Import {
            path,
            dry_run,
            overwrite,
        } => export_import::run_import(&path, dry_run, overwrite),
        Commands::Runtime { command } => runtime::run(command).await,
        Commands::CliSchema => {
            dump_cli_schema();
            Ok(())
        }
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
    use crate::core::model_family;

    let cmd = Cli::command();
    let mut commands = Vec::new();
    collect_schema_commands(&cmd, "", &mut commands);

    // Build model capability matrix from models.toml
    let cn_list = model_family::controlnet_support_list();
    let sr_list = model_family::style_ref_support_list();
    let models: Vec<serde_json::Value> = model_family::families()
        .iter()
        .flat_map(|f| {
            f.models.iter().map(move |m| {
                let has_controlnet = cn_list.iter().any(|c| c.base_model_id == m.id);
                let controlnet_types: Vec<String> = cn_list
                    .iter()
                    .find(|c| c.base_model_id == m.id)
                    .map(|c| c.supported_types.to_vec())
                    .unwrap_or_default();
                let has_style_ref = sr_list.iter().any(|s| s.base_model_id == m.id);
                let style_ref_mechanism = sr_list
                    .iter()
                    .find(|s| s.base_model_id == m.id)
                    .map(|s| s.mechanism.as_str());

                serde_json::json!({
                    "id": m.id,
                    "name": m.name,
                    "family": f.name,
                    "vendor": f.vendor,
                    "description": m.description,
                    "params_b": m.total_b,
                    "transformer_b": m.transformer_b,
                    "vram_bf16_gb": m.vram_bf16_gb,
                    "vram_fp8_gb": m.vram_fp8_gb,
                    "capabilities": {
                        "txt2img": m.capabilities.txt2img,
                        "img2img": m.capabilities.img2img,
                        "inpaint": m.capabilities.inpaint,
                        "edit": m.capabilities.edit,
                        "training": m.capabilities.training,
                        "controlnet": has_controlnet,
                        "controlnet_types": controlnet_types,
                        "style_ref": has_style_ref,
                        "style_ref_mechanism": style_ref_mechanism,
                        "text_rendering": m.text_rendering,
                    },
                    "defaults": {
                        "steps": m.default_steps,
                        "guidance": m.default_guidance,
                        "resolution": m.default_resolution,
                    },
                    "quality": m.quality,
                    "speed": m.speed,
                })
            })
        })
        .collect();

    let schema = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commands": commands,
        "models": models,
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
