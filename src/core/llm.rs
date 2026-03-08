// ---------------------------------------------------------------------------
// core::llm — LLM inference with pluggable backends
//
// Architecture:
//   LlmBackend trait → LocalLlmBackend (llama.cpp via llama-cpp-2)
//                     → CloudLlmBackend (HTTP API)
//                     → BuiltinLlmBackend (rule-based fallback, zero deps)
//
// Same trait pattern as Executor and PromptEnhancer — the agent doesn't
// know or care where inference runs. Users without a GPU get `--cloud`.
// ---------------------------------------------------------------------------

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Message types (shared across all backends)
// ---------------------------------------------------------------------------

/// Role in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// For tool-result messages: which tool call this responds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            tool_call_id: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_call_id: None,
        }
    }

    pub fn tool_result(call_id: &str, content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(call_id.to_string()),
        }
    }
}

/// Tool definition for function-calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

/// Result of a completion call.
#[derive(Debug, Clone)]
pub enum CompletionResult {
    /// Plain text response.
    Text(String),
    /// The model wants to call a tool.
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// Trait — implement this for new LLM backends
// ---------------------------------------------------------------------------

pub trait LlmBackend: Send + Sync {
    /// Text completion with optional tool-use support (for agent loop).
    fn complete(&self, messages: &[Message], tools: &[ToolDef]) -> Result<CompletionResult>;

    /// Vision-language: describe/analyze images.
    fn vision(&self, images: &[PathBuf], prompt: &str) -> Result<String>;

    /// Backend name for logging/UI ("local-gpu", "local-cpu", "cloud", "builtin").
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Local backend (llama.cpp via llama-cpp-2 crate)
// ---------------------------------------------------------------------------

/// Device preference for local inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

/// Local LLM backend using llama.cpp compiled into the binary.
///
/// Behind the `llm` cargo feature flag — when disabled, `resolve_backend()`
/// skips local and falls through to cloud or builtin.
#[allow(dead_code)] // Placeholder fields — will be used when llama-cpp-2 integration is complete
pub struct LocalLlmBackend {
    model_id: String,
    model_path: PathBuf,
    is_vl: bool,
    device: Device,
}

impl LocalLlmBackend {
    /// Load a GGUF model from the store. Keeps it resident for reuse.
    pub fn load(model_id: &str, prefer_gpu: bool) -> Result<Self> {
        let store_dir = dirs::home_dir()
            .context("Could not determine home directory")?
            .join(".modl")
            .join("store")
            .join("llm");

        // Look for the model in the store
        let model_path = find_model_in_store(&store_dir, model_id)?;
        let is_vl = model_id.contains("vl") || model_id.contains("vision");
        let device = if prefer_gpu { Device::Gpu } else { Device::Cpu };

        // TODO(llm feature): Initialize llama-cpp-2 model and context here.
        // For now, validate the file exists so we fail early.
        if !model_path.exists() {
            bail!(
                "Model file not found: {}. Run `modl llm pull {model_id}` first.",
                model_path.display()
            );
        }

        Ok(Self {
            model_id: model_id.to_string(),
            model_path,
            is_vl,
            device,
        })
    }

    /// Unload model, free VRAM.
    #[allow(dead_code)] // Will be called by agent VRAM management
    pub fn unload(&mut self) {
        // TODO(llm feature): Drop llama-cpp-2 context and model to free VRAM.
        eprintln!(
            "[llm] unloading model {} (device: {:?})",
            self.model_id, self.device
        );
    }
}

impl LlmBackend for LocalLlmBackend {
    fn complete(&self, messages: &[Message], tools: &[ToolDef]) -> Result<CompletionResult> {
        // TODO(llm feature): Run llama-cpp-2 inference with chat template + tool grammar.
        let _ = (messages, tools);
        bail!(
            "Local LLM backend not yet available (model: {}, device: {:?}). \
             Rebuild with `--features llm` once llama-cpp-2 integration is complete.",
            self.model_id,
            self.device
        )
    }

    fn vision(&self, images: &[PathBuf], prompt: &str) -> Result<String> {
        if !self.is_vl {
            bail!(
                "Model {} is not a vision-language model. Use a VL model like qwen3-vl.",
                self.model_id
            );
        }
        // TODO(llm feature): Run VL inference via llama-cpp-2 clip API.
        let _ = (images, prompt);
        bail!(
            "Local VL backend not yet available (model: {}). \
             Rebuild with `--features llm` once llama-cpp-2 VL integration is complete.",
            self.model_id
        )
    }

    fn name(&self) -> &str {
        match self.device {
            Device::Gpu => "local-gpu",
            Device::Cpu => "local-cpu",
        }
    }
}

/// Find a model GGUF file in the LLM store.
fn find_model_in_store(store_dir: &std::path::Path, model_id: &str) -> Result<PathBuf> {
    // Convention: ~/.modl/store/llm/<model_id>/<file>.gguf
    let model_dir = store_dir.join(model_id);
    if model_dir.is_dir() {
        // Find the first .gguf file
        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "gguf") {
                    return Ok(path);
                }
            }
        }
    }
    // Return the expected path even if not found (caller checks existence)
    Ok(model_dir.join(format!("{model_id}.gguf")))
}

// ---------------------------------------------------------------------------
// Cloud backend (HTTP API to modl-managed endpoint)
// ---------------------------------------------------------------------------

/// Cloud LLM backend — calls a modl-managed API.
///
/// Contract:
///   POST {api_base}/v1/chat/completions  (OpenAI-compatible + tools)
///   POST {api_base}/v1/vision            (images as base64 + prompt)
pub struct CloudLlmBackend {
    api_base: String,
    auth_token: String,
    client: reqwest::blocking::Client,
}

impl CloudLlmBackend {
    /// Create from ~/.modl/auth.yaml cloud config.
    pub fn from_config() -> Result<Self> {
        let auth_path = dirs::home_dir()
            .context("Could not determine home directory")?
            .join(".modl")
            .join("auth.yaml");

        if !auth_path.exists() {
            bail!(
                "No auth config found at ~/.modl/auth.yaml. Run `modl auth` to configure cloud access."
            );
        }

        let yaml = std::fs::read_to_string(&auth_path)
            .with_context(|| format!("Failed to read {}", auth_path.display()))?;
        let config: serde_yaml::Value =
            serde_yaml::from_str(&yaml).context("Failed to parse auth.yaml")?;

        let cloud = config
            .get("cloud")
            .context("No 'cloud' section in auth.yaml")?;

        let api_base = cloud
            .get("api_base")
            .and_then(|v| v.as_str())
            .context("Missing cloud.api_base in auth.yaml")?
            .trim_end_matches('/')
            .to_string();

        let auth_token = cloud
            .get("token")
            .and_then(|v| v.as_str())
            .context("Missing cloud.token in auth.yaml")?
            .to_string();

        Ok(Self {
            api_base,
            auth_token,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .context("Failed to create HTTP client")?,
        })
    }
}

impl LlmBackend for CloudLlmBackend {
    fn complete(&self, messages: &[Message], tools: &[ToolDef]) -> Result<CompletionResult> {
        let mut body = serde_json::json!({
            "messages": messages,
            "model": "default",
        });

        if !tools.is_empty() {
            let tool_defs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(tool_defs);
        }

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.api_base))
            .bearer_auth(&self.auth_token)
            .json(&body)
            .send()
            .context("Cloud LLM request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            bail!("Cloud LLM returned {status}: {text}");
        }

        let result: serde_json::Value = resp.json().context("Failed to parse cloud response")?;

        // Parse OpenAI-compatible response
        let choice = result
            .get("choices")
            .and_then(|c| c.get(0))
            .context("No choices in cloud response")?;

        let message = choice.get("message").context("No message in choice")?;

        // Check for tool calls first
        if let Some(tool_calls) = message.get("tool_calls")
            && let Some(tc) = tool_calls.get(0)
        {
            let function = tc.get("function").context("No function in tool_call")?;
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("call_0")
                .to_string();
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .context("No function name")?
                .to_string();
            let args_str = function
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let args: serde_json::Value =
                serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));

            return Ok(CompletionResult::ToolCall { id, name, args });
        }

        // Plain text response
        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(CompletionResult::Text(content))
    }

    fn vision(&self, images: &[PathBuf], prompt: &str) -> Result<String> {
        // Encode images as base64
        let encoded: Vec<String> = images
            .iter()
            .map(|p| {
                let bytes = std::fs::read(p)
                    .with_context(|| format!("Failed to read image: {}", p.display()))?;
                Ok(base64_encode(&bytes))
            })
            .collect::<Result<Vec<_>>>()?;

        let body = serde_json::json!({
            "images": encoded,
            "prompt": prompt,
            "model": "default-vl",
        });

        let resp = self
            .client
            .post(format!("{}/v1/vision", self.api_base))
            .bearer_auth(&self.auth_token)
            .json(&body)
            .send()
            .context("Cloud vision request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            bail!("Cloud vision returned {status}: {text}");
        }

        let result: serde_json::Value = resp.json().context("Failed to parse vision response")?;
        let description = result
            .get("description")
            .or_else(|| {
                result
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
            })
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(description)
    }

    fn name(&self) -> &str {
        "cloud"
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity(4 * (bytes.len() / 3 + 1));
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARSET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARSET[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Builtin fallback (rule-based, zero deps, always available)
// ---------------------------------------------------------------------------

/// Builtin LLM backend — rule-based heuristics, no model needed.
///
/// Used as the final fallback when no local model is available and cloud
/// is not configured. Provides basic functionality for the agent to operate
/// at a reduced capability level.
pub struct BuiltinLlmBackend;

impl LlmBackend for BuiltinLlmBackend {
    fn complete(&self, messages: &[Message], tools: &[ToolDef]) -> Result<CompletionResult> {
        // Extract the last user message for context
        let last_user = messages
            .iter()
            .rfind(|m| matches!(m.role, Role::User))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // If tools are available and this looks like a studio session,
        // return a structured default workflow
        if !tools.is_empty() {
            // Find the first available tool and suggest it
            if let Some(tool) = tools.first() {
                return Ok(CompletionResult::ToolCall {
                    id: "builtin_0".to_string(),
                    name: tool.name.clone(),
                    args: serde_json::json!({}),
                });
            }
        }

        // Simple response — the builtin can't really reason
        Ok(CompletionResult::Text(format!(
            "I'm running in builtin mode (no LLM model loaded). \
             Your request: \"{}\". \
             For better results, run `modl llm pull qwen3.5-4b-instruct-q4` or configure cloud access.",
            &last_user[..last_user.len().min(100)]
        )))
    }

    fn vision(&self, images: &[PathBuf], _prompt: &str) -> Result<String> {
        // Return a generic description — can't actually analyze images
        Ok(format!(
            "Received {} image(s). Running in builtin mode — unable to analyze images. \
             Install a VL model with `modl llm pull qwen3-vl-8b-instruct-q4` for image understanding.",
            images.len()
        ))
    }

    fn name(&self) -> &str {
        "builtin"
    }
}

// ---------------------------------------------------------------------------
// Backend resolution — graceful degradation chain
// ---------------------------------------------------------------------------

/// Resolve the best available LLM backend.
///
/// Degradation chain:
///   local GPU → local CPU → cloud → builtin (always works)
///
/// With `prefer_cloud = true`, the chain starts at cloud instead.
pub fn resolve_backend(prefer_cloud: bool) -> Result<Box<dyn LlmBackend>> {
    if prefer_cloud && let Ok(cloud) = CloudLlmBackend::from_config() {
        return Ok(Box::new(cloud));
    }

    // Try local GPU
    if let Ok(local) = LocalLlmBackend::load("default-text", true) {
        return Ok(Box::new(local));
    }

    // Try local CPU
    if let Ok(local) = LocalLlmBackend::load("default-text", false) {
        return Ok(Box::new(local));
    }

    // Try cloud (if not already tried)
    if !prefer_cloud && let Ok(cloud) = CloudLlmBackend::from_config() {
        return Ok(Box::new(cloud));
    }

    // Builtin rules — always works
    Ok(Box::new(BuiltinLlmBackend))
}

/// Resolve a specific model by ID, with device preference.
pub fn resolve_model(model_id: &str, prefer_gpu: bool) -> Result<Box<dyn LlmBackend>> {
    if let Ok(local) = LocalLlmBackend::load(model_id, prefer_gpu) {
        return Ok(Box::new(local));
    }
    if prefer_gpu && let Ok(local) = LocalLlmBackend::load(model_id, false) {
        return Ok(Box::new(local));
    }
    if let Ok(cloud) = CloudLlmBackend::from_config() {
        return Ok(Box::new(cloud));
    }
    bail!(
        "Model '{model_id}' not available locally and cloud not configured. \
         Run `modl llm pull {model_id}` or configure cloud access in ~/.modl/auth.yaml."
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_constructors() {
        let m = Message::system("You are helpful");
        assert!(matches!(m.role, Role::System));
        assert_eq!(m.content, "You are helpful");
        assert!(m.tool_call_id.is_none());

        let m = Message::user("hello");
        assert!(matches!(m.role, Role::User));

        let m = Message::assistant("hi");
        assert!(matches!(m.role, Role::Assistant));

        let m = Message::tool_result("call_1", "done");
        assert!(matches!(m.role, Role::Tool));
        assert_eq!(m.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_builtin_backend_text() {
        let backend = BuiltinLlmBackend;
        assert_eq!(backend.name(), "builtin");

        let messages = vec![Message::user("hello world")];
        let result = backend.complete(&messages, &[]).unwrap();
        match result {
            CompletionResult::Text(text) => {
                assert!(text.contains("builtin mode"));
                assert!(text.contains("hello world"));
            }
            _ => panic!("Expected Text result"),
        }
    }

    #[test]
    fn test_builtin_backend_with_tools() {
        let backend = BuiltinLlmBackend;
        let tools = vec![ToolDef {
            name: "analyze_images".to_string(),
            description: "Analyze uploaded images".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }];

        let messages = vec![Message::user("analyze my photos")];
        let result = backend.complete(&messages, &tools).unwrap();
        match result {
            CompletionResult::ToolCall { name, .. } => {
                assert_eq!(name, "analyze_images");
            }
            _ => panic!("Expected ToolCall result"),
        }
    }

    #[test]
    fn test_builtin_backend_vision() {
        let backend = BuiltinLlmBackend;
        let images = vec![PathBuf::from("/tmp/test.jpg")];
        let result = backend.vision(&images, "describe this").unwrap();
        assert!(result.contains("1 image(s)"));
        assert!(result.contains("builtin mode"));
    }

    #[test]
    fn test_resolve_backend_always_succeeds() {
        // resolve_backend should always return something (at minimum, builtin)
        let backend = resolve_backend(false).unwrap();
        // On CI/test machines without models or cloud, this should be builtin
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
    }
}
