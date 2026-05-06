//! MCP (Model Context Protocol) server for modl.
//!
//! Implements MCP stdio transport (Content-Length framing + JSON-RPC 2.0)
//! by wrapping modl CLI commands directly. No dependency on `modl serve` —
//! each tool call spawns the appropriate `modl` subcommand.

use anyhow::{Context, Result};
use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::process::Command;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    method: String,
    #[serde(default)]
    params: Option<Value>,
    #[serde(default)]
    id: Option<Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

fn tool_definitions() -> Value {
    json!([
        {
            "name": "generate",
            "description": "Generate images from a text prompt using AI models. Returns file paths of generated images.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt describing the image to generate"
                    },
                    "base": {
                        "type": "string",
                        "description": "Base model (e.g. flux-schnell, flux-dev, z-image, z-image-turbo, sdxl, qwen-image, chroma). Default: flux-schnell"
                    },
                    "size": {
                        "type": "string",
                        "description": "Image size: 1:1, 16:9, 9:16, 4:3, 3:4, or WxH (e.g. 1280x720). Default: 1:1"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of inference steps (model-dependent default)"
                    },
                    "guidance": {
                        "type": "number",
                        "description": "Guidance scale (model-dependent default)"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of images to generate. Default: 1"
                    },
                    "lora": {
                        "type": "string",
                        "description": "LoRA name or file path to apply"
                    },
                    "lora_strength": {
                        "type": "number",
                        "description": "LoRA strength (0.0-1.0). Default: 1.0"
                    },
                    "init_image": {
                        "type": "string",
                        "description": "Path to source image for img2img"
                    },
                    "strength": {
                        "type": "number",
                        "description": "Denoising strength for img2img (0.0-1.0). Default: 0.75"
                    },
                    "mask": {
                        "type": "string",
                        "description": "Path to mask image for inpainting (white = regenerate)"
                    },
                    "controlnet": {
                        "type": "string",
                        "description": "Path to control image for ControlNet conditioning"
                    },
                    "cn_type": {
                        "type": "string",
                        "description": "ControlNet type: canny, depth, pose, softedge, scribble, hed, mlsd, gray, normal"
                    },
                    "cn_strength": {
                        "type": "number",
                        "description": "ControlNet conditioning strength. Default: 0.75"
                    },
                    "fast": {
                        "type": "boolean",
                        "description": "Use Lightning LoRA for faster generation (fewer steps)"
                    }
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "edit",
            "description": "Edit an existing image using AI models guided by a text prompt. Returns file paths of edited images.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt describing the desired edit"
                    },
                    "image": {
                        "type": "string",
                        "description": "Path to the source image to edit"
                    },
                    "base": {
                        "type": "string",
                        "description": "Base model (e.g. klein-4b, klein-9b, qwen-image-edit, flux-2-dev). Default: auto-selected"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of inference steps (model-dependent default)"
                    },
                    "guidance": {
                        "type": "number",
                        "description": "Guidance scale (model-dependent default)"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of edited images to generate. Default: 1"
                    },
                    "fast": {
                        "type": "boolean",
                        "description": "Use Lightning LoRA for faster generation (fewer steps)"
                    }
                },
                "required": ["prompt", "image"]
            }
        },
        {
            "name": "train",
            "description": "Preview a LoRA training configuration (dry-run). Returns the resolved training spec without starting training.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "base": {
                        "type": "string",
                        "description": "Base model to train on (e.g. flux-dev, flux-schnell, z-image, sdxl)"
                    },
                    "lora_type": {
                        "type": "string",
                        "description": "Type of LoRA to train: style, character, or object",
                        "enum": ["style", "character", "object"]
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Path to training dataset directory"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the training run"
                    },
                    "trigger": {
                        "type": "string",
                        "description": "Trigger word for the LoRA"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of training steps"
                    },
                    "rank": {
                        "type": "integer",
                        "description": "LoRA rank (dimensionality)"
                    },
                    "lr": {
                        "type": "number",
                        "description": "Learning rate"
                    },
                    "preset": {
                        "type": "string",
                        "description": "Training preset: quick, standard, or advanced",
                        "enum": ["quick", "standard", "advanced"]
                    }
                },
                "required": ["base", "lora_type"]
            }
        },
        {
            "name": "train_status",
            "description": "Check the status of LoRA training runs. Shows progress, loss, and completion status.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Specific training run name to check. If omitted, shows all recent runs."
                    }
                }
            }
        },
        {
            "name": "list_models",
            "description": "List all installed models with their type, variant, size, and ID.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "pull_model",
            "description": "Download a model from the modl registry or HuggingFace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID to pull (e.g. flux-dev, z-image, sdxl, or hf:owner/model)"
                    },
                    "variant": {
                        "type": "string",
                        "description": "Force a specific variant (e.g. fp16, fp8, bf16, gguf-q4)"
                    }
                },
                "required": ["model_id"]
            }
        },
        {
            "name": "search_models",
            "description": "Search for models in the modl registry and optionally CivitAI.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'flux', 'anime', 'realistic')"
                    },
                    "type": {
                        "type": "string",
                        "description": "Filter by model type (e.g. checkpoint, lora, vae)"
                    },
                    "popular": {
                        "type": "boolean",
                        "description": "Sort by popularity"
                    },
                    "civitai": {
                        "type": "boolean",
                        "description": "Include CivitAI results"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "describe",
            "description": "Describe/caption an image using AI vision models.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the image file to describe"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "score",
            "description": "Score image quality and aesthetics using AI. Returns quality metrics for one or more images.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to an image file or directory of images to score"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "upscale",
            "description": "Upscale an image to higher resolution using AI super-resolution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the image file to upscale"
                    },
                    "scale": {
                        "type": "integer",
                        "description": "Upscale factor: 2 or 4. Default: 4",
                        "enum": [2, 4]
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "remove_bg",
            "description": "Remove the background from an image, producing a transparent PNG.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the image file to remove background from"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "enhance",
            "description": "Enhance a text prompt for better image generation results using AI rewriting.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to enhance"
                    },
                    "model": {
                        "type": "string",
                        "description": "Target model to optimize the prompt for"
                    },
                    "intensity": {
                        "type": "string",
                        "description": "Enhancement intensity: subtle, moderate, or aggressive",
                        "enum": ["subtle", "moderate", "aggressive"]
                    }
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "run_workflow",
            "description": "Submit a batch workflow YAML to modl run. Returns immediately with a run_id — the job continues in the background. Use job_status to poll for completion. Useful for fire-and-forget batch runs: submit from laptop, close lid, check back later.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_yaml": {
                        "type": "string",
                        "description": "Full YAML content of the workflow spec (the contents of a .yaml file)"
                    }
                },
                "required": ["spec_yaml"]
            }
        },
        {
            "name": "job_status",
            "description": "Check the status of a workflow run submitted via run_workflow. Returns aggregate status (pending/running/completed/partial_failure) and artifact URLs. Set MODL_BASE_URL env var on the server for HTTP URLs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID returned by run_workflow"
                    }
                },
                "required": ["run_id"]
            }
        },
        {
            "name": "list_run_outputs",
            "description": "List artifact paths or URLs for a completed workflow run. If MODL_BASE_URL is set on the server, returns HTTP URLs downloadable over Tailscale.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to list outputs for"
                    }
                },
                "required": ["run_id"]
            }
        }
    ])
}

// ---------------------------------------------------------------------------
// Tool execution — wraps modl CLI commands
// ---------------------------------------------------------------------------

/// Find the modl binary path (we ARE modl, so use current_exe).
fn modl_bin() -> Result<std::path::PathBuf> {
    std::env::current_exe().context("Failed to find modl binary path")
}

/// Run a modl subcommand and capture stdout + stderr.
fn run_modl(args: &[&str]) -> Result<(String, String, bool), String> {
    let bin = modl_bin().map_err(|e| e.to_string())?;
    let output = Command::new(bin)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("Failed to execute modl: {}", e))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    Ok((stdout, stderr, output.status.success()))
}

fn tool_generate(args: &Value) -> Result<Value, (i32, String)> {
    let prompt = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: prompt".to_string()))?;

    let mut cmd_args: Vec<String> = vec!["generate".into(), "--json".into(), prompt.into()];

    // Map JSON params to CLI flags
    if let Some(v) = args.get("base").and_then(|v| v.as_str()) {
        cmd_args.extend(["--base".into(), v.into()]);
    }
    if let Some(v) = args.get("size").and_then(|v| v.as_str()) {
        cmd_args.extend(["--size".into(), v.into()]);
    }
    if let Some(v) = args.get("steps").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--steps".into(), v.to_string()]);
    }
    if let Some(v) = args.get("guidance").and_then(|v| v.as_f64()) {
        cmd_args.extend(["--guidance".into(), v.to_string()]);
    }
    if let Some(v) = args.get("seed").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--seed".into(), v.to_string()]);
    }
    if let Some(v) = args.get("count").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--count".into(), v.to_string()]);
    }
    if let Some(v) = args.get("lora").and_then(|v| v.as_str()) {
        cmd_args.extend(["--lora".into(), v.into()]);
    }
    if let Some(v) = args.get("lora_strength").and_then(|v| v.as_f64()) {
        cmd_args.extend(["--lora-strength".into(), v.to_string()]);
    }
    if let Some(v) = args.get("init_image").and_then(|v| v.as_str()) {
        cmd_args.extend(["--init-image".into(), v.into()]);
    }
    if let Some(v) = args.get("strength").and_then(|v| v.as_f64()) {
        cmd_args.extend(["--strength".into(), v.to_string()]);
    }
    if let Some(v) = args.get("mask").and_then(|v| v.as_str()) {
        cmd_args.extend(["--mask".into(), v.into()]);
    }
    if let Some(v) = args.get("controlnet").and_then(|v| v.as_str()) {
        cmd_args.extend(["--controlnet".into(), v.into()]);
    }
    if let Some(v) = args.get("cn_type").and_then(|v| v.as_str()) {
        cmd_args.extend(["--cn-type".into(), v.into()]);
    }
    if let Some(v) = args.get("cn_strength").and_then(|v| v.as_f64()) {
        cmd_args.extend(["--cn-strength".into(), v.to_string()]);
    }
    if args.get("fast").and_then(|v| v.as_bool()).unwrap_or(false) {
        cmd_args.push("--fast".into());
    }

    let args_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    let (stdout, stderr, success) = run_modl(&args_ref).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Generation failed: {}", msg.trim())));
    }

    // Parse the --json output from modl generate
    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        let images = result.get("images").cloned().unwrap_or_else(|| json!([]));
        let status = result
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("completed");

        let mut text = format!("Status: {}\n", status);
        if let Some(arr) = images.as_array() {
            for path in arr {
                if let Some(p) = path.as_str() {
                    text.push_str(&format!("Image: {}\n", p));
                }
            }
        }

        Ok(json!({
            "content": [{"type": "text", "text": text.trim()}]
        }))
    } else {
        // Fallback: return raw stdout
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_edit(args: &Value) -> Result<Value, (i32, String)> {
    let prompt = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: prompt".to_string()))?;
    let image = args
        .get("image")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: image".to_string()))?;

    let mut cmd_args: Vec<String> = vec![
        "edit".into(),
        "--json".into(),
        prompt.into(),
        "--image".into(),
        image.into(),
    ];

    if let Some(v) = args.get("base").and_then(|v| v.as_str()) {
        cmd_args.extend(["--base".into(), v.into()]);
    }
    if let Some(v) = args.get("steps").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--steps".into(), v.to_string()]);
    }
    if let Some(v) = args.get("guidance").and_then(|v| v.as_f64()) {
        cmd_args.extend(["--guidance".into(), v.to_string()]);
    }
    if let Some(v) = args.get("seed").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--seed".into(), v.to_string()]);
    }
    if let Some(v) = args.get("count").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--count".into(), v.to_string()]);
    }
    if args.get("fast").and_then(|v| v.as_bool()).unwrap_or(false) {
        cmd_args.push("--fast".into());
    }

    let args_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    let (stdout, stderr, success) = run_modl(&args_ref).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Edit failed: {}", msg.trim())));
    }

    // Parse the --json output from modl edit
    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        let images = result.get("images").cloned().unwrap_or_else(|| json!([]));
        let status = result
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("completed");

        let mut text = format!("Status: {}\n", status);
        if let Some(arr) = images.as_array() {
            for path in arr {
                if let Some(p) = path.as_str() {
                    text.push_str(&format!("Image: {}\n", p));
                }
            }
        }

        Ok(json!({
            "content": [{"type": "text", "text": text.trim()}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_train(args: &Value) -> Result<Value, (i32, String)> {
    let base = args
        .get("base")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: base".to_string()))?;
    let lora_type = args
        .get("lora_type")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: lora_type".to_string()))?;

    let mut cmd_args: Vec<String> = vec![
        "train".into(),
        "--dry-run".into(),
        "--base".into(),
        base.into(),
        "--lora-type".into(),
        lora_type.into(),
    ];

    if let Some(v) = args.get("dataset").and_then(|v| v.as_str()) {
        cmd_args.extend(["--dataset".into(), v.into()]);
    }
    if let Some(v) = args.get("name").and_then(|v| v.as_str()) {
        cmd_args.extend(["--name".into(), v.into()]);
    }
    if let Some(v) = args.get("trigger").and_then(|v| v.as_str()) {
        cmd_args.extend(["--trigger".into(), v.into()]);
    }
    if let Some(v) = args.get("steps").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--steps".into(), v.to_string()]);
    }
    if let Some(v) = args.get("rank").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--rank".into(), v.to_string()]);
    }
    if let Some(v) = args.get("lr").and_then(|v| v.as_f64()) {
        cmd_args.extend(["--lr".into(), v.to_string()]);
    }
    if let Some(v) = args.get("preset").and_then(|v| v.as_str()) {
        cmd_args.extend(["--preset".into(), v.into()]);
    }

    let args_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    let (stdout, stderr, success) = run_modl(&args_ref).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Training failed: {}", msg.trim())));
    }

    // dry-run outputs the training spec as YAML
    let clean = strip_ansi(&stdout);
    Ok(json!({
        "content": [{"type": "text", "text": clean.trim()}]
    }))
}

fn tool_train_status(args: &Value) -> Result<Value, (i32, String)> {
    let mut cmd_args: Vec<String> = vec!["train".into(), "status".into(), "--json".into()];

    if let Some(v) = args.get("name").and_then(|v| v.as_str()) {
        cmd_args.push(v.into());
    }

    let args_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    let (stdout, stderr, success) = run_modl(&args_ref).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Train status failed: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        Ok(json!({
            "content": [{"type": "text", "text": serde_json::to_string_pretty(&result).unwrap_or_else(|_| stdout.clone())}]
        }))
    } else {
        let clean = strip_ansi(&stdout);
        Ok(json!({
            "content": [{"type": "text", "text": clean.trim()}]
        }))
    }
}

fn tool_list_models(_args: &Value) -> Result<Value, (i32, String)> {
    let (stdout, stderr, success) = run_modl(&["ls"]).map_err(|e| (-32603, e))?;

    if !success {
        return Err((-32603, format!("Failed to list models: {}", stderr.trim())));
    }

    // Strip ANSI codes for clean text output
    let clean = strip_ansi(&stdout);
    Ok(json!({
        "content": [{"type": "text", "text": clean.trim()}]
    }))
}

fn tool_pull_model(args: &Value) -> Result<Value, (i32, String)> {
    let model_id = args
        .get("model_id")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: model_id".to_string()))?;

    let mut cmd_args = vec!["pull", model_id];

    let variant_owned;
    if let Some(v) = args.get("variant").and_then(|v| v.as_str()) {
        variant_owned = v.to_string();
        cmd_args.extend(["--variant", &variant_owned]);
    }

    let (stdout, stderr, success) = run_modl(&cmd_args).map_err(|e| (-32603, e))?;

    let output = strip_ansi(if success { &stdout } else { &stderr });
    let text = if success {
        format!(
            "Model '{}' pulled successfully.\n{}",
            model_id,
            output.trim()
        )
    } else {
        format!("Failed to pull '{}': {}", model_id, output.trim())
    };

    if !success {
        return Err((-32603, text));
    }

    Ok(json!({
        "content": [{"type": "text", "text": text.trim()}]
    }))
}

fn tool_search_models(args: &Value) -> Result<Value, (i32, String)> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: query".to_string()))?;

    let mut cmd_args: Vec<String> = vec!["search".into(), "--json".into(), query.into()];

    if let Some(v) = args.get("type").and_then(|v| v.as_str()) {
        cmd_args.extend(["--type".into(), v.into()]);
    }
    if args
        .get("popular")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        cmd_args.push("--popular".into());
    }
    if args
        .get("civitai")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        cmd_args.push("--civitai".into());
    }

    let args_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    let (stdout, stderr, success) = run_modl(&args_ref).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Search failed: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        Ok(json!({
            "content": [{"type": "text", "text": serde_json::to_string_pretty(&result).unwrap_or_else(|_| stdout.clone())}]
        }))
    } else {
        let clean = strip_ansi(&stdout);
        Ok(json!({
            "content": [{"type": "text", "text": clean.trim()}]
        }))
    }
}

fn tool_describe(args: &Value) -> Result<Value, (i32, String)> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: path".to_string()))?;

    let (stdout, stderr, success) =
        run_modl(&["vision", "describe", "--json", path]).map_err(|e| (-32603, e))?;

    if !success {
        return Err((-32603, format!("Describe failed: {}", stderr.trim())));
    }

    // Try to parse JSON output
    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        let caption = result
            .get("caption")
            .and_then(|v| v.as_str())
            .unwrap_or(&stdout);
        Ok(json!({
            "content": [{"type": "text", "text": caption}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_score(args: &Value) -> Result<Value, (i32, String)> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: path".to_string()))?;

    let (stdout, stderr, success) =
        run_modl(&["vision", "score", "--json", path]).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Score failed: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        Ok(json!({
            "content": [{"type": "text", "text": serde_json::to_string_pretty(&result).unwrap_or_else(|_| stdout.clone())}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_upscale(args: &Value) -> Result<Value, (i32, String)> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: path".to_string()))?;

    let mut cmd_args: Vec<String> = vec![
        "process".into(),
        "upscale".into(),
        "--json".into(),
        path.into(),
    ];

    if let Some(v) = args.get("scale").and_then(|v| v.as_u64()) {
        cmd_args.extend(["--scale".into(), v.to_string()]);
    }

    let args_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    let (stdout, stderr, success) = run_modl(&args_ref).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Upscale failed: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        let output_path = result
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let text = format!("Upscaled image: {}", output_path);
        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_remove_bg(args: &Value) -> Result<Value, (i32, String)> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: path".to_string()))?;

    let (stdout, stderr, success) =
        run_modl(&["process", "remove-bg", "--json", path]).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Remove background failed: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        let output_path = result
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let text = format!("Background removed: {}", output_path);
        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_run_workflow(args: &Value) -> Result<Value, (i32, String)> {
    let spec_yaml = args
        .get("spec_yaml")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: spec_yaml".to_string()))?;

    // Pre-generate the run_id. Includes a short random suffix to prevent
    // collisions when two workflows are submitted within the same second.
    let run_id = format!(
        "mcp-{}-{}",
        Local::now().format("%Y%m%d-%H%M%S"),
        &uuid::Uuid::new_v4().to_string()[..6]
    );

    // Write YAML to a stable path alongside the stderr log — no temp file
    // race (child must open before parent deletes). Also useful for debugging.
    let run_log_dir = crate::core::paths::modl_root().join("run-logs");
    let _ = std::fs::create_dir_all(&run_log_dir);
    let spec_path = run_log_dir.join(format!("{run_id}.yaml"));
    let log_path = run_log_dir.join(format!("{run_id}.log"));
    std::fs::write(&spec_path, spec_yaml)
        .map_err(|e| (-32603, format!("Failed to write workflow spec: {e}")))?;

    let bin = modl_bin().map_err(|e| (-32603, e.to_string()))?;

    // Capture stderr via pipe for a short window so we can detect immediate
    // failures (bad YAML, missing model, etc.) before returning run_id.
    let mut child = std::process::Command::new(&bin)
        .args(["run", spec_path.to_str().unwrap_or(""), "--run-id", &run_id])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| (-32603, format!("Failed to spawn modl run: {e}")))?;

    // Wait up to 500 ms — enough to catch YAML parse errors and missing files
    // without blocking for the actual GPU work.
    std::thread::sleep(std::time::Duration::from_millis(500));
    if let Ok(Some(status)) = child.try_wait() {
        // Process already exited — read stderr to report what went wrong.
        let stderr_text = child
            .stderr
            .take()
            .and_then(|mut r| {
                let mut buf = String::new();
                std::io::Read::read_to_string(&mut r, &mut buf).ok()?;
                Some(buf)
            })
            .unwrap_or_default();
        if !status.success() {
            return Err((
                -32603,
                format!("modl run failed immediately: {}", stderr_text.trim()),
            ));
        }
    }

    // Forward remaining stderr to the log file (best-effort).
    if let Some(mut stderr_pipe) = child.stderr.take() {
        let log_path_clone = log_path.clone();
        std::thread::spawn(move || {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path_clone)
            {
                let _ = std::io::copy(&mut stderr_pipe, &mut f);
            }
        });
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "run_id": run_id,
                "status": "submitted",
                "spec": spec_path.to_string_lossy(),
                "log": log_path.to_string_lossy(),
            })).unwrap_or_default()
        }]
    }))
}

fn tool_job_status(args: &Value) -> Result<Value, (i32, String)> {
    let run_id = args
        .get("run_id")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: run_id".to_string()))?;

    let (stdout, stderr, success) =
        run_modl(&["status", "--json", run_id]).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Status check failed: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        Ok(json!({
            "content": [{"type": "text", "text": serde_json::to_string_pretty(&result).unwrap_or_else(|_| stdout.clone())}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_list_run_outputs(args: &Value) -> Result<Value, (i32, String)> {
    let run_id = args
        .get("run_id")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: run_id".to_string()))?;

    let (stdout, stderr, success) =
        run_modl(&["status", "--json", run_id]).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Failed to list outputs: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        let artifacts = result
            .get("artifacts")
            .cloned()
            .unwrap_or_else(|| json!([]));
        let count = artifacts.as_array().map(|a| a.len()).unwrap_or(0);
        let text = format!(
            "{count} artifact(s) for run {run_id}:\n{}",
            serde_json::to_string_pretty(&artifacts).unwrap_or_default()
        );
        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

fn tool_enhance(args: &Value) -> Result<Value, (i32, String)> {
    let prompt = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: prompt".to_string()))?;

    let mut cmd_args: Vec<String> = vec!["enhance".into(), "--json".into(), prompt.into()];

    if let Some(v) = args.get("model").and_then(|v| v.as_str()) {
        cmd_args.extend(["--model".into(), v.into()]);
    }
    if let Some(v) = args.get("intensity").and_then(|v| v.as_str()) {
        cmd_args.extend(["--intensity".into(), v.into()]);
    }

    let args_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    let (stdout, stderr, success) = run_modl(&args_ref).map_err(|e| (-32603, e))?;

    if !success {
        let msg = if stderr.is_empty() { &stdout } else { &stderr };
        return Err((-32603, format!("Enhance failed: {}", msg.trim())));
    }

    if let Ok(result) = serde_json::from_str::<Value>(&stdout) {
        let enhanced = result
            .get("enhanced")
            .and_then(|v| v.as_str())
            .unwrap_or(&stdout);
        Ok(json!({
            "content": [{"type": "text", "text": enhanced}]
        }))
    } else {
        Ok(json!({
            "content": [{"type": "text", "text": stdout.trim()}]
        }))
    }
}

// ---------------------------------------------------------------------------
// MCP protocol handling
// ---------------------------------------------------------------------------

fn handle_request(request: &JsonRpcRequest) -> Option<JsonRpcResponse> {
    // Notifications (no id) don't get responses
    request.id.as_ref()?;

    let result = match request.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "modl",
                "version": env!("CARGO_PKG_VERSION")
            }
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let params = request.params.as_ref();
            let name = params
                .and_then(|p| p.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let arguments = params
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or_else(|| json!({}));

            match name {
                "generate" => tool_generate(&arguments),
                "edit" => tool_edit(&arguments),
                "train" => tool_train(&arguments),
                "train_status" => tool_train_status(&arguments),
                "list_models" => tool_list_models(&arguments),
                "pull_model" => tool_pull_model(&arguments),
                "search_models" => tool_search_models(&arguments),
                "describe" => tool_describe(&arguments),
                "score" => tool_score(&arguments),
                "upscale" => tool_upscale(&arguments),
                "remove_bg" => tool_remove_bg(&arguments),
                "enhance" => tool_enhance(&arguments),
                "run_workflow" => tool_run_workflow(&arguments),
                "job_status" => tool_job_status(&arguments),
                "list_run_outputs" => tool_list_run_outputs(&arguments),
                _ => Err((-32601, format!("Unknown tool: {}", name))),
            }
        }
        _ => Err((-32601, format!("Method not found: {}", request.method))),
    };

    Some(match result {
        Ok(value) => JsonRpcResponse::success(request.id.clone(), value),
        Err((code, msg)) => JsonRpcResponse::error(request.id.clone(), code, msg),
    })
}

// ---------------------------------------------------------------------------
// Content-Length framed stdio transport
// ---------------------------------------------------------------------------

/// Read a single Content-Length framed message from stdin.
fn read_message(reader: &mut impl BufRead) -> Result<Option<String>> {
    // Read headers until blank line
    let mut content_length: Option<usize> = None;
    loop {
        let mut header = String::new();
        let bytes_read = reader
            .read_line(&mut header)
            .context("Failed to read header")?;
        if bytes_read == 0 {
            return Ok(None); // EOF
        }

        let trimmed = header.trim();
        if trimmed.is_empty() {
            break; // End of headers
        }

        if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(
                len_str
                    .trim()
                    .parse()
                    .context("Invalid Content-Length value")?,
            );
        }
    }

    let length = content_length.context("Missing Content-Length header")?;

    // Read exactly `length` bytes
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .context("Failed to read message body")?;

    String::from_utf8(body)
        .context("Invalid UTF-8 in message body")
        .map(Some)
}

/// Write a Content-Length framed message to stdout.
fn write_message(writer: &mut impl Write, body: &str) -> Result<()> {
    write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
    writer.flush()?;
    Ok(())
}

/// Strip ANSI escape codes from a string.
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until we hit a letter (end of ANSI sequence)
            while let Some(&next) = chars.peek() {
                chars.next();
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run() -> Result<()> {
    // Log to stderr (stdout is the MCP transport)
    eprintln!("modl MCP server v{} starting", env!("CARGO_PKG_VERSION"));

    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut stdout = io::stdout();

    loop {
        match read_message(&mut reader) {
            Ok(Some(body)) => {
                let request: JsonRpcRequest = match serde_json::from_str(&body) {
                    Ok(req) => req,
                    Err(e) => {
                        let resp =
                            JsonRpcResponse::error(None, -32700, format!("Parse error: {}", e));
                        let json = serde_json::to_string(&resp)?;
                        write_message(&mut stdout, &json)?;
                        continue;
                    }
                };

                if let Some(response) = handle_request(&request) {
                    let json = serde_json::to_string(&response)?;
                    write_message(&mut stdout, &json)?;
                }
            }
            Ok(None) => break, // EOF — client disconnected
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "initialize".into(),
            params: Some(json!({"protocolVersion": "2024-11-05"})),
            id: Some(json!(1)),
        };
        let resp = handle_request(&req).unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "modl");
    }

    #[test]
    fn test_tools_list() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/list".into(),
            params: None,
            id: Some(json!(2)),
        };
        let resp = handle_request(&req).unwrap();
        let tools = resp.result.unwrap();
        let tool_list = tools["tools"].as_array().unwrap();
        let names: Vec<&str> = tool_list
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"generate"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"train"));
        assert!(names.contains(&"train_status"));
        assert!(names.contains(&"list_models"));
        assert!(names.contains(&"pull_model"));
        assert!(names.contains(&"search_models"));
        assert!(names.contains(&"describe"));
        assert!(names.contains(&"score"));
        assert!(names.contains(&"upscale"));
        assert!(names.contains(&"remove_bg"));
        assert!(names.contains(&"enhance"));
        assert!(names.contains(&"run_workflow"));
        assert!(names.contains(&"job_status"));
        assert!(names.contains(&"list_run_outputs"));
        assert_eq!(tool_list.len(), 15);
    }

    #[test]
    fn test_ping() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "ping".into(),
            params: None,
            id: Some(json!(3)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_unknown_method() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "foo/bar".into(),
            params: None,
            id: Some(json!(4)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_notification_returns_none() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "initialized".into(),
            params: None,
            id: None,
        };
        assert!(handle_request(&req).is_none());
    }

    #[test]
    fn test_generate_missing_prompt() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "generate", "arguments": {}})),
            id: Some(json!(5)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("prompt"));
    }

    #[test]
    fn test_edit_missing_params() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "edit", "arguments": {"prompt": "test"}})),
            id: Some(json!(7)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("image"));
    }

    #[test]
    fn test_train_missing_params() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "train", "arguments": {"base": "flux-dev"}})),
            id: Some(json!(8)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("lora_type"));
    }

    #[test]
    fn test_search_models_missing_query() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "search_models", "arguments": {}})),
            id: Some(json!(9)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("query"));
    }

    #[test]
    fn test_score_missing_path() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "score", "arguments": {}})),
            id: Some(json!(10)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("path"));
    }

    #[test]
    fn test_upscale_missing_path() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "upscale", "arguments": {}})),
            id: Some(json!(11)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("path"));
    }

    #[test]
    fn test_remove_bg_missing_path() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "remove_bg", "arguments": {}})),
            id: Some(json!(12)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("path"));
    }

    #[test]
    fn test_enhance_missing_prompt() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "enhance", "arguments": {}})),
            id: Some(json!(13)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("prompt"));
    }

    #[test]
    fn test_unknown_tool() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/call".into(),
            params: Some(json!({"name": "nonexistent", "arguments": {}})),
            id: Some(json!(6)),
        };
        let resp = handle_request(&req).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("\x1b[32mhello\x1b[0m"), "hello");
        assert_eq!(strip_ansi("no codes here"), "no codes here");
        assert_eq!(strip_ansi("\x1b[1;34mblue\x1b[0m"), "blue");
    }

    #[test]
    fn test_content_length_framing() {
        let body = r#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
        let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);

        let mut reader = io::BufReader::new(framed.as_bytes());
        let message = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(message, body);
    }

    #[test]
    fn test_content_length_eof() {
        let mut reader = io::BufReader::new(&b""[..]);
        assert!(read_message(&mut reader).unwrap().is_none());
    }
}
