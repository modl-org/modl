// ---------------------------------------------------------------------------
// core::agent — Tool-use agent loop for Studio sessions
//
// The agent receives user intent + images and executes a multi-step workflow:
//   1. Analyze uploaded photos (VL model)
//   2. Create + caption dataset
//   3. Select base model + train LoRA
//   4. Generate output images
//
// Fully decoupled from both LLM backend (trait) and execution backend (trait).
// ---------------------------------------------------------------------------

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::broadcast;

use crate::core::llm::{CompletionResult, LlmBackend, Message, ToolDef};

// ---------------------------------------------------------------------------
// Session types
// ---------------------------------------------------------------------------

/// A Studio session — one end-to-end "upload photos → get results" workflow.
pub struct AgentSession {
    pub id: String,
    pub intent: String,
    pub images: Vec<PathBuf>,
    pub events: Vec<AgentEvent>,
    pub status: SessionStatus,
}

/// Session lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Events emitted by the agent for UI consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent is thinking / planning next step.
    Thinking { message: String },
    /// Agent is executing a tool.
    ToolStart { tool: String, description: String },
    /// Tool progress update (e.g., training 45%).
    ToolProgress {
        tool: String,
        progress: f32,
        detail: String,
    },
    /// Tool completed.
    ToolComplete { tool: String, result: String },
    /// Agent produced final output.
    OutputReady { images: Vec<String> },
    /// Error occurred.
    Error { message: String },
}

impl AgentEvent {
    /// Serialize to JSON for SSE transmission.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are a studio photographer AI assistant. The user has uploaded photos and \
described what they want. You have tools to prepare datasets, train custom \
models, and generate images.

Analyze the uploaded photos to understand the subject, then create a training \
dataset, caption the images, train a LoRA, and generate the requested images.

Always explain what you're doing in simple terms. Follow this workflow:
1. analyze_images — Understand the uploaded photos
2. create_dataset — Prepare a training dataset
3. caption_images — Generate captions for training
4. select_base_model — Choose the best base model
5. train_lora — Train a custom LoRA
6. enhance_prompt — Create detailed generation prompts
7. generate_images — Generate the final images

Call tools one at a time. After each tool completes, decide the next step.";

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

fn agent_tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "analyze_images".to_string(),
            description: "Analyze uploaded photos to understand the subject, style, and background. Returns a text description.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to focus on: 'subject', 'style', 'background', or 'all'"
                    }
                }
            }),
        },
        ToolDef {
            name: "create_dataset".to_string(),
            description: "Create a training dataset from the uploaded images.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the dataset"
                    }
                },
                "required": ["name"]
            }),
        },
        ToolDef {
            name: "caption_images".to_string(),
            description: "Generate training captions for all images in the dataset using the VL model.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to caption"
                    },
                    "trigger_word": {
                        "type": "string",
                        "description": "Trigger word to prepend to each caption (e.g., 'OHWX person')"
                    }
                },
                "required": ["dataset_name"]
            }),
        },
        ToolDef {
            name: "select_base_model".to_string(),
            description: "Choose the best base model for training based on the task.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["character", "style", "object"],
                        "description": "Type of LoRA to train"
                    }
                },
                "required": ["task_type"]
            }),
        },
        ToolDef {
            name: "train_lora".to_string(),
            description: "Train a LoRA using the prepared dataset and selected base model.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the training dataset"
                    },
                    "base_model": {
                        "type": "string",
                        "description": "Base model ID (e.g., 'flux-dev')"
                    },
                    "lora_name": {
                        "type": "string",
                        "description": "Name for the output LoRA"
                    },
                    "trigger_word": {
                        "type": "string",
                        "description": "Trigger word for the LoRA"
                    },
                    "lora_type": {
                        "type": "string",
                        "enum": ["character", "style", "object"],
                        "description": "Type of LoRA"
                    }
                },
                "required": ["dataset_name", "base_model", "lora_name", "lora_type"]
            }),
        },
        ToolDef {
            name: "enhance_prompt".to_string(),
            description: "Craft a detailed image generation prompt from the user's intent.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user's raw intent to enhance"
                    },
                    "style_hint": {
                        "type": "string",
                        "description": "Style guidance from image analysis"
                    }
                },
                "required": ["prompt"]
            }),
        },
        ToolDef {
            name: "generate_images".to_string(),
            description: "Generate images using the trained LoRA.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The generation prompt"
                    },
                    "lora_name": {
                        "type": "string",
                        "description": "Name of the trained LoRA to apply"
                    },
                    "base_model": {
                        "type": "string",
                        "description": "Base model ID"
                    },
                    "num_images": {
                        "type": "integer",
                        "description": "Number of images to generate",
                        "default": 4
                    }
                },
                "required": ["prompt", "lora_name", "base_model"]
            }),
        },
    ]
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

/// Maximum number of agent turns before giving up.
const MAX_TURNS: usize = 20;

/// Run the agent loop for a Studio session.
///
/// Takes trait objects — works identically whether LLM is local (llama.cpp)
/// or cloud (API), and whether execution is local or cloud.
pub async fn run_session(
    session: &mut AgentSession,
    llm: &dyn LlmBackend,
    event_tx: broadcast::Sender<String>,
) -> Result<()> {
    session.status = SessionStatus::Running;

    let tools = agent_tools();
    let mut messages = vec![
        Message::system(SYSTEM_PROMPT),
        Message::user(&format!(
            "I've uploaded {} photo(s). My request: {}",
            session.images.len(),
            session.intent
        )),
    ];

    let emit = |event: &AgentEvent| {
        let json = event.to_json();
        let _ = event_tx.send(json);
    };

    emit(&AgentEvent::Thinking {
        message: "Understanding your request...".to_string(),
    });

    for turn in 0..MAX_TURNS {
        let result = match llm.complete(&messages, &tools) {
            Ok(r) => r,
            Err(e) => {
                let event = AgentEvent::Error {
                    message: format!("LLM error: {e}"),
                };
                emit(&event);
                session.events.push(event);
                session.status = SessionStatus::Failed;
                return Err(e);
            }
        };

        match result {
            CompletionResult::Text(text) => {
                // Agent is done reasoning — check if we should continue or finish
                messages.push(Message::assistant(&text));

                // If this is a final message after tools have run, we're done
                if turn > 0 {
                    emit(&AgentEvent::Thinking { message: text });
                    break;
                }

                emit(&AgentEvent::Thinking {
                    message: text.clone(),
                });
            }

            CompletionResult::ToolCall { id, name, args } => {
                let description = describe_tool_call(&name, &args);
                emit(&AgentEvent::ToolStart {
                    tool: name.clone(),
                    description: description.clone(),
                });
                session.events.push(AgentEvent::ToolStart {
                    tool: name.clone(),
                    description,
                });

                messages.push(Message::assistant(&format!(
                    "Calling tool: {name}({})",
                    serde_json::to_string(&args).unwrap_or_default()
                )));

                // Execute the tool
                let tool_result =
                    crate::core::agent_tools::execute_tool(&name, &args, session, llm, &event_tx)
                        .await;

                let result_text = match &tool_result {
                    Ok(text) => text.clone(),
                    Err(e) => format!("Error: {e}"),
                };

                emit(&AgentEvent::ToolComplete {
                    tool: name.clone(),
                    result: result_text.clone(),
                });
                session.events.push(AgentEvent::ToolComplete {
                    tool: name.clone(),
                    result: result_text.clone(),
                });

                // Feed tool result back into conversation
                messages.push(Message::tool_result(&id, &result_text));

                // Check if this was the final step (generate_images)
                if name == "generate_images" {
                    if let Ok(ref text) = tool_result {
                        // Parse output image paths
                        if let Ok(paths) = serde_json::from_str::<Vec<String>>(text) {
                            emit(&AgentEvent::OutputReady {
                                images: paths.clone(),
                            });
                            session
                                .events
                                .push(AgentEvent::OutputReady { images: paths });
                        }
                    }
                    session.status = SessionStatus::Completed;
                    return Ok(());
                }

                if tool_result.is_err() {
                    // Let the agent know about the error so it can adapt
                    emit(&AgentEvent::Thinking {
                        message: "Adjusting plan due to error...".to_string(),
                    });
                }
            }
        }
    }

    if session.status != SessionStatus::Completed {
        session.status = SessionStatus::Completed;
    }

    Ok(())
}

/// Generate a human-readable description of a tool call.
fn describe_tool_call(name: &str, args: &serde_json::Value) -> String {
    match name {
        "analyze_images" => "Understanding your photos...".to_string(),
        "create_dataset" => {
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("dataset");
            format!("Creating dataset '{name}'...")
        }
        "caption_images" => "Generating captions for training...".to_string(),
        "select_base_model" => "Choosing the best model for your task...".to_string(),
        "train_lora" => {
            let name = args
                .get("lora_name")
                .and_then(|v| v.as_str())
                .unwrap_or("model");
            format!("Training custom model '{name}'...")
        }
        "enhance_prompt" => "Crafting generation prompts...".to_string(),
        "generate_images" => {
            let count = args.get("num_images").and_then(|v| v.as_u64()).unwrap_or(4);
            format!("Generating {count} images...")
        }
        _ => format!("Running {name}..."),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_status_display() {
        assert_eq!(SessionStatus::Pending.to_string(), "pending");
        assert_eq!(SessionStatus::Running.to_string(), "running");
        assert_eq!(SessionStatus::Completed.to_string(), "completed");
        assert_eq!(SessionStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn test_agent_event_serialization() {
        let event = AgentEvent::Thinking {
            message: "planning...".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("thinking"));
        assert!(json.contains("planning..."));

        let event = AgentEvent::ToolProgress {
            tool: "train_lora".to_string(),
            progress: 0.5,
            detail: "step 250/500".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("tool_progress"));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn test_agent_tools_defined() {
        let tools = agent_tools();
        assert_eq!(tools.len(), 7);
        assert_eq!(tools[0].name, "analyze_images");
        assert_eq!(tools[6].name, "generate_images");
    }

    #[test]
    fn test_describe_tool_call() {
        assert!(
            describe_tool_call("analyze_images", &serde_json::json!({})).contains("Understanding")
        );
        assert!(
            describe_tool_call("train_lora", &serde_json::json!({"lora_name": "test"}))
                .contains("test")
        );
        assert!(
            describe_tool_call("generate_images", &serde_json::json!({"num_images": 8}))
                .contains("8")
        );
    }

    struct MockLlm;

    impl LlmBackend for MockLlm {
        fn complete(&self, _messages: &[Message], _tools: &[ToolDef]) -> Result<CompletionResult> {
            // Return a text response to end the loop
            Ok(CompletionResult::Text("Done!".to_string()))
        }
        fn vision(&self, _images: &[PathBuf], _prompt: &str) -> Result<String> {
            Ok("A golden retriever dog".to_string())
        }
        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_agent_session_with_mock() {
        let mock = MockLlm;

        let mut session = AgentSession {
            id: "test-session".to_string(),
            intent: "Studio photoshoot of my dog".to_string(),
            images: vec![],
            events: vec![],
            status: SessionStatus::Pending,
        };

        let (tx, _rx) = broadcast::channel(64);
        let result = run_session(&mut session, &mock, tx).await;
        assert!(result.is_ok());
        assert_eq!(session.status, SessionStatus::Completed);
    }
}
