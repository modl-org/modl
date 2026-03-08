# LLM / VL Runtime Architecture

> **STATUS: PHASE 1 IMPLEMENTED.** Rust-native `LlmBackend` trait with three backends (Local, Cloud, Builtin). Agent framework with tool-use loop. Studio UI wired up.

---

## Runtime decision: Rust-native trait

The LLM runtime is a **Rust trait** (`LlmBackend`) with pluggable backends, following the same pattern as `Executor` and `PromptEnhancer`. The agent doesn't know or care where inference runs.

### Why Rust-native (not llama-cpp-python worker)

The original plan called for a separate Python worker on a Unix socket (like the diffusion worker). We went with Rust-native instead:

| Concern | Rust-native (chosen) | Python worker (original plan) |
|---------|---------------------|-------------------------------|
| Dependencies | Zero Python deps for LLM | Needs llama-cpp-python in managed runtime |
| Startup | Instant (compiled in) | 2-4s (Python import) |
| VRAM management | Direct control | Socket coordination |
| Distribution | Single binary | Binary + Python runtime |
| Cloud backend | Just HTTP client | Would still need HTTP client |
| VL support | Via llama-cpp-2 crate (TODO) | Via llama-cpp-python |

The local backend uses the `llama-cpp-2` Rust crate (behind `--features llm` cargo flag). When the feature is disabled, `resolve_backend()` skips local and falls through to cloud or builtin.

---

## Architecture

```
LlmBackend trait
    ├── LocalLlmBackend (llama-cpp-2, GGUF models from store)
    │   ├── GPU mode (fast, ~3-5GB VRAM)
    │   └── CPU mode (slow, fallback)
    │
    ├── CloudLlmBackend (HTTP → modl API, OpenAI-compatible)
    │   ├── POST /v1/chat/completions (text + tool use)
    │   └── POST /v1/vision (images + prompt)
    │
    └── BuiltinLlmBackend (rule-based, zero deps, always works)
```

### Graceful degradation chain

```
resolve_backend(prefer_cloud: bool)
    if prefer_cloud → try Cloud
    → try Local GPU
    → try Local CPU
    → try Cloud (if not already tried)
    → Builtin (always succeeds)
```

With `--cloud` flag, starts at cloud. The enhance button / Studio agent **always works** — quality scales with available resources.

---

## Key files

| File | Role |
|------|------|
| `src/core/llm.rs` | `LlmBackend` trait, all three backends, resolution logic |
| `src/core/agent.rs` | Agent loop, session state, tool definitions, system prompt |
| `src/core/agent_tools.rs` | Tool implementations wrapping existing modl services |
| `src/cli/llm.rs` | `modl llm pull/chat/ls` CLI commands |
| `src/ui/server.rs` | Studio API endpoints (session CRUD, SSE streaming) |

---

## Model management

GGUF models stored in `~/.modl/store/llm/<model_id>/`. Registry manifests for:

| Model | Size | Use case |
|-------|------|----------|
| `qwen3.5-4b-instruct-q4` | ~3GB | Text reasoning (agent, enhance) |
| `qwen3-vl-8b-instruct-q4` | ~5GB | Vision-language (captioning, image analysis) |

Auto-download on first Studio use if not installed.

### VRAM coexistence targets

| Config | Diffusion | LLM | Total | GPU |
|--------|-----------|-----|-------|-----|
| Flux fp8 + Qwen 4B Q4 | ~12GB | ~3GB | ~15GB | 24GB OK |
| Z-Image bf16 + Qwen 4B Q4 | ~14GB | ~3GB | ~17GB | 24GB OK |

When training is active, the agent should unload the LLM or fall back to cloud/builtin.

---

## Cloud backend contract

The cloud backend calls a modl-managed API (same endpoint used by Tauri desktop app for non-GPU users):

```
POST {api_base}/v1/chat/completions
  Authorization: Bearer {auth_token}
  Body: { messages, tools, model }

POST {api_base}/v1/vision
  Authorization: Bearer {auth_token}
  Body: { images (base64), prompt, model }
```

Config in `~/.modl/auth.yaml`:
```yaml
cloud:
  api_base: https://api.modl.run
  token: modl_key_...
```

---

## Agent framework

The agent uses the LLM with tool-use to orchestrate Studio sessions:

1. Analyze uploaded photos (VL model via `llm.vision()`)
2. Create + caption dataset
3. Select base model + train LoRA
4. Generate output images

### Tools

| Tool | Maps to | Description |
|------|---------|-------------|
| `analyze_images` | `llm.vision()` | VL model describes uploaded photos |
| `create_dataset` | `core::dataset` | Create dataset from uploads |
| `caption_images` | `llm.vision()` per image | Generate training captions |
| `select_base_model` | DB query | Choose best base model |
| `train_lora` | executor pipeline | Train LoRA |
| `generate_images` | `cli::generate::run()` | Generate images |
| `enhance_prompt` | `llm.complete()` | Craft detailed prompts |

### DB schema

```sql
studio_sessions (id, intent, status, created_at, completed_at)
session_events (session_id, sequence, event_json, timestamp)
session_images (session_id, image_path, role)
```

---

## Implementation status

| Component | Status |
|-----------|--------|
| `LlmBackend` trait + 3 backends | ✅ Done |
| `LocalLlmBackend` (placeholder, needs llama-cpp-2 wiring) | 🟡 Stub |
| `CloudLlmBackend` (OpenAI-compat HTTP) | ✅ Done |
| `BuiltinLlmBackend` (rule-based fallback) | ✅ Done |
| Agent loop + 7 tools | ✅ Done |
| Studio API (Axum endpoints, SSE) | ✅ Done |
| Studio UI (React) | ✅ Done |
| `modl llm pull/chat/ls` CLI | ✅ Done |
| llama-cpp-2 actual inference | ❌ Not started |
| VRAM coexistence with gpu.lock | ❌ Not started |

## Next steps

1. Wire `llama-cpp-2` crate into `LocalLlmBackend` (complete/vision methods)
2. Test VL inference with Qwen3-VL GGUF
3. Integrate `GpuLock` (see gpu-resource-management.md) for VRAM coordination
4. Connect Studio agent to actual executor for real training/generation
