# Modl Runtime — Architecture Proposal

> **STATUS: ASPIRATIONAL / NOT IMPLEMENTED**
> This document describes a future product vision (ComfyUI as sidecar engine,
> YAML workflow format, Next.js web UI, node registry, cloud deploy platform).
> None of this is built. The current product is a CLI toolkit for models,
> training, and generation — see [../PLAN.md](../PLAN.md) for actual status.

## The Thesis

ComfyUI solves the hard problem (execution engine, model management, node ecosystem) but wraps it in the wrong UX for 2026: spaghetti node graphs that are impossible for LLMs to generate, impossible to version-control, and impossible to deploy. 

**Modl Runtime** keeps ComfyUI's Python execution core (the hard part) but replaces everything above it:

```
┌─────────────────────────────────────────────────────┐
│  NEW: AI Chat UI / YAML Editor / Deploy Platform    │  ← You build this
├─────────────────────────────────────────────────────┤
│  NEW: Modl Runtime Server (Rust)                    │  ← Orchestration layer
│  - YAML workflow parsing + validation               │
│  - modl.lock dependency resolution                  │
│  - Job queue + WebSocket streaming                  │
│  - Deploy packaging                                 │
├─────────────────────────────────────────────────────┤
│  REUSED: ComfyUI Python execution engine            │  ← Fork/embed, not rewrite
│  - Node system (NODE_CLASS_MAPPINGS)                │
│  - Graph execution (topological sort + caching)     │
│  - Model management (VRAM, partial loading)         │
│  - All existing nodes + custom nodes                │
├─────────────────────────────────────────────────────┤
│  REUSED: modl (model manager)                       │  ← Already built
│  - Content-addressed store                          │
│  - Dependency resolution                            │
│  - GPU-aware variant selection                      │
└─────────────────────────────────────────────────────┘
```

The key insight: **don't rewrite the Python ML runtime** — it's 50K+ lines of battle-tested GPU memory management, model loading, and sampling code. Instead, control it from a better layer.

---

## Part 1: YAML Workflow Format

### Why YAML

1. **LLM-native** — GPT-4/Claude can generate valid YAML trivially; generating ComfyUI's nested JSON-with-positional-arrays is error-prone
2. **Version-controllable** — clean diffs, mergeable, reviewable
3. **Human-readable** — a designer can read and tweak parameters without a node editor
4. **Lockfile-friendly** — YAML composes naturally with `modl.lock`

### Format Design

```yaml
# flux-portrait.flow.yaml
name: "Flux Portrait Generator"
version: 1
description: "High-quality portrait generation with Flux Dev"

# Dependencies — resolved by modl
requires:
  - flux-dev          # checkpoint (modl resolves VAE, text encoders automatically)
  - realistic-skin-v3 # lora

# Input parameters — what the user/API caller provides
inputs:
  prompt:
    type: string
    description: "What to generate"
    default: "portrait photo of a woman, natural lighting"
  negative:
    type: string
    default: "blurry, low quality, bad anatomy"
  seed:
    type: int
    default: -1        # -1 = random
  width:
    type: int
    default: 1024
    range: [512, 2048]
  height:
    type: int
    default: 1024
    range: [512, 2048]
  steps:
    type: int
    default: 25
    range: [1, 100]
  cfg:
    type: float
    default: 3.5
    range: [1.0, 20.0]
  denoise:
    type: float
    default: 1.0
    range: [0.0, 1.0]

# Optional: input images for img2img, controlnet, etc.
# image_inputs:
#   source_image:
#     type: image
#     optional: true
#     description: "Reference image for img2img"

# The pipeline — a linear(ish) DAG, not a sprawling graph
# Each step has an implicit ID (its key). Outputs flow forward by reference.
steps:
  load_model:
    node: CheckpointLoaderSimple
    params:
      ckpt_name: flux-dev        # resolved by modl to actual filename

  apply_lora:
    node: LoraLoader
    params:
      model: $load_model.model
      clip: $load_model.clip
      lora_name: realistic-skin-v3
      strength_model: 0.7
      strength_clip: 0.7

  encode_positive:
    node: CLIPTextEncode
    params:
      clip: $apply_lora.clip
      text: $inputs.prompt

  encode_negative:
    node: CLIPTextEncode
    params:
      clip: $apply_lora.clip
      text: $inputs.negative

  empty_latent:
    node: EmptyLatentImage
    params:
      width: $inputs.width
      height: $inputs.height
      batch_size: 1

  sample:
    node: KSampler
    params:
      model: $apply_lora.model
      positive: $encode_positive.conditioning
      negative: $encode_negative.conditioning
      latent_image: $empty_latent.latent
      seed: $inputs.seed
      steps: $inputs.steps
      cfg: $inputs.cfg
      sampler_name: euler
      scheduler: normal
      denoise: $inputs.denoise

  decode:
    node: VAEDecode
    params:
      samples: $sample.latent
      vae: $load_model.vae

  save:
    node: SaveImage
    params:
      images: $decode.image
      filename_prefix: "modl_output"

# Outputs — what gets returned to the caller
outputs:
  images: $save.images
```

### Reference Syntax

```
$step_name.output_name    → reference a step's named output
$inputs.param_name        → reference a workflow input parameter
```

**Named outputs** instead of positional indices. The runtime maps `$load_model.model` → ComfyUI's `["load_model", 0]` using the node's `RETURN_TYPES` and `RETURN_NAMES`. This is what makes YAML human-readable — no one remembers that output slot 2 of `CheckpointLoaderSimple` is the VAE.

### Compilation: YAML → ComfyUI Prompt JSON

The runtime compiles `flow.yaml` into ComfyUI's native prompt format:

```yaml
# flux-portrait.flow.yaml steps.sample
sample:
  node: KSampler
  params:
    model: $apply_lora.model
    seed: $inputs.seed
    ...
```

Compiles to:

```json
{
  "sample": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["apply_lora", 0],
      "seed": 42,
      ...
    }
  }
}
```

The compiler:
1. Resolves `$inputs.*` references to concrete values
2. Resolves `$step.output` references to `[step_id, output_index]` tuples
3. Resolves model names (`flux-dev`) to actual filenames via modl registry
4. Validates types match (using `/object_info` from ComfyUI)
5. Emits standard ComfyUI prompt JSON

This means **every existing ComfyUI node works day one** — no porting needed.

---

## Part 2: System Architecture

### Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (Web)                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │  Chat UI     │ │ YAML Editor  │ │ Gallery/History   │ │
│  │  (LLM agent) │ │ (Monaco)     │ │                   │ │
│  └──────┬───────┘ └──────┬───────┘ └────────┬──────────┘ │
│         │                │                   │            │
│         └────────────────┴───────────────────┘            │
│                        │ HTTP + WebSocket                 │
└────────────────────────┼──────────────────────────────────┘
                         │
┌────────────────────────┼──────────────────────────────────┐
│              Modl Runtime Server (Rust)                    │
│                        │                                   │
│  ┌─────────────────────┴──────────────────────────┐       │
│  │              API Gateway (axum)                  │       │
│  │  POST /run    — execute a workflow              │       │
│  │  POST /chat   — LLM-assisted workflow creation  │       │
│  │  GET  /ws     — streaming progress/results      │       │
│  │  GET  /flows  — list/manage saved flows         │       │
│  │  GET  /nodes  — available node catalog          │       │
│  │  POST /deploy — package for cloud deployment    │       │
│  └─────────┬──────────────────────────────────────┘       │
│            │                                               │
│  ┌─────────┴──────────┐  ┌────────────────────────────┐   │
│  │  Workflow Compiler  │  │  Job Queue + Scheduler     │   │
│  │  YAML → ComfyUI    │  │  Priority, cancellation,   │   │
│  │  prompt JSON        │  │  retry, batching           │   │
│  └─────────┬──────────┘  └─────────────┬──────────────┘   │
│            │                            │                   │
│  ┌─────────┴────────────────────────────┴──────────────┐   │
│  │              Python Bridge                           │   │
│  │  Embedded Python (PyO3) or HTTP to ComfyUI process   │   │
│  │  - Submit compiled prompts                           │   │
│  │  - Stream progress events                            │   │
│  │  - Query node definitions (/object_info)             │   │
│  │  - Model loading coordination                        │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                               │
│  ┌─────────────────────────┴───────────────────────────┐   │
│  │              Modl Core (existing)                    │   │
│  │  - Model resolution (name → file path)              │   │
│  │  - Dependency checking                              │   │
│  │  - Variant selection                                │   │
│  │  - Content-addressed store                          │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┼──────────────────────────────────┐
│         ComfyUI Python Process (embedded/sidecar)         │
│                        │                                   │
│  ┌─────────────────────┴──────────────────────────────┐   │
│  │  PromptExecutor (unmodified ComfyUI core)           │   │
│  │  - Topological sort + execution                     │   │
│  │  - Node instantiation + caching                     │   │
│  │  - Subgraph expansion                               │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Model Management (unmodified)                      │   │
│  │  - VRAM management, partial loading                 │   │
│  │  - Model patching (LoRA, hooks)                     │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  All Nodes (built-in + custom)                      │   │
│  │  - KSampler, VAEDecode, CLIPTextEncode, etc.        │   │
│  │  - Custom nodes from community                      │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Python Bridge: Sidecar vs Embedded

**Recommended: Sidecar process (Phase 1)**

Run ComfyUI as a headless subprocess. The Rust server communicates via HTTP + WebSocket:

```
Rust Server ──HTTP POST /prompt──→ ComfyUI :8188
             ←──WebSocket events──
```

Why sidecar first:
- **Zero ComfyUI modifications needed** — use it as-is
- **Custom nodes just work** — they load normally in the Python process
- **Simpler debugging** — two separate processes, standard tools
- **Can run ComfyUI on a different machine** (remote GPU)

```rust
// src/runtime/bridge.rs
pub struct ComfyBridge {
    base_url: String,        // http://localhost:8188
    ws_url: String,          // ws://localhost:8188/ws
    client_id: String,
}

impl ComfyBridge {
    /// Submit a compiled prompt for execution
    pub async fn execute(&self, prompt_json: Value) -> Result<JobHandle> { ... }
    
    /// Stream execution events (progress, previews, completion)
    pub fn subscribe(&self) -> impl Stream<Item = ExecutionEvent> { ... }
    
    /// Get all available node definitions
    pub async fn get_node_catalog(&self) -> Result<NodeCatalog> { ... }
    
    /// Check system status (VRAM, queue depth)
    pub async fn system_stats(&self) -> Result<SystemStats> { ... }
}
```

**Future: PyO3 embedding (Phase 2+)**

For tighter integration and single-binary distribution, embed Python via PyO3:

```rust
// Future: direct Python embedding
use pyo3::prelude::*;

fn execute_prompt(prompt: &str) -> PyResult<()> {
    Python::with_gil(|py| {
        let comfy = py.import("comfy_execution")?;
        // ...
    })
}
```

This is significantly more complex (Python environment management, venv isolation, GPU driver compatibility) — defer it.

---

## Part 3: Frontend Architecture

### Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **Next.js 15 (App Router)** | RSC for fast initial loads, server actions for LLM calls, great DX |
| AI Chat | **Vercel AI SDK** | Streaming, tool calling, multi-provider (OpenAI, Anthropic, local) |
| Editor | **Monaco Editor** | YAML editing with IntelliSense, same editor as VS Code |
| Styling | **Tailwind CSS + shadcn/ui** | Fast iteration, great component library |
| State | **Zustand** | Lightweight, no boilerplate |
| Real-time | **Native WebSocket** | Stream execution progress, previews |
| Gallery | **Masonry grid** | Image results with metadata |

### Three-Panel Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ⌂ Modl Runtime          [flow: flux-portrait.flow.yaml ▾]  │
├──────────────┬─────────────────────┬────────────────────────┤
│              │                     │                        │
│  CHAT        │  WORKFLOW           │  OUTPUT                │
│              │                     │                        │
│  ┌────────┐  │  ┌───────────────┐  │  ┌──────────────────┐ │
│  │ You:   │  │  │ name: "..."   │  │  │                  │ │
│  │ make a │  │  │               │  │  │   [Generated     │ │
│  │ flux   │  │  │ requires:     │  │  │    Image]        │ │
│  │ flow   │  │  │   - flux-dev  │  │  │                  │ │
│  │ for    │  │  │               │  │  │                  │ │
│  │ portra │  │  │ inputs:       │  │  └──────────────────┘ │
│  │ its    │  │  │   prompt:     │  │                        │
│  │        │  │  │     type: ... │  │  Seed: 8566257        │
│  │ Agent: │  │  │               │  │  Steps: 25 / 25  ✓    │
│  │ I've   │  │  │ steps:        │  │  Time: 12.3s          │
│  │ created│  │  │   load_model: │  │                        │
│  │ a flow │  │  │     node: ... │  │  ┌──────────────────┐ │
│  │ with   │  │  │               │  │  │ Run ▶  [params]  │ │
│  │ ...    │  │  │   sample:     │  │  │                  │ │
│  │        │  │  │     node: ... │  │  │ prompt: [      ] │ │
│  │        │  │  │     params:   │  │  │ steps:  [25   ] │ │
│  │        │  │  │       model:  │  │  │ cfg:    [3.5  ] │ │
│  │ ░░░░░░ │  │  │       ...     │  │  │ seed:   [rand ] │ │
│  └────────┘  │  └───────────────┘  │  └──────────────────┘ │
│              │                     │                        │
├──────────────┴─────────────────────┴────────────────────────┤
│  [Models: flux-dev ✓ realistic-skin-v3 ✓]  [Deploy ▸]      │
└─────────────────────────────────────────────────────────────┘
```

### Panel Details

#### Left: Chat Panel (LLM Agent)

The LLM has **tools** to manipulate the workflow:

```typescript
const tools = {
  create_flow: {
    description: "Create a new workflow YAML",
    parameters: { name: "string", description: "string", base_model: "string" },
    // LLM generates the full YAML
  },
  modify_flow: {
    description: "Modify the current workflow YAML",
    parameters: { changes: "string" },  // description of changes
    // LLM edits the YAML directly
  },
  add_step: {
    description: "Add a step to the workflow",
    parameters: { step_name: "string", node_type: "string", params: "object" },
  },
  run_flow: {
    description: "Execute the current workflow with given inputs",
    parameters: { inputs: "object" },
  },
  search_nodes: {
    description: "Search available ComfyUI nodes",
    parameters: { query: "string" },
  },
  search_models: {
    description: "Search for models in the modl registry",
    parameters: { query: "string", type: "string" },
  },
  install_model: {
    description: "Install a model via modl",
    parameters: { model_id: "string" },
  },
  explain_flow: {
    description: "Explain what the current workflow does",
    parameters: {},
  },
  suggest_params: {
    description: "Suggest parameter adjustments based on the output",
    parameters: { feedback: "string" },  // e.g. "too dark", "more detail"
  },
};
```

**System prompt** gives the LLM:
- Full YAML format spec
- Available node catalog (from `/object_info`)
- Current workflow context
- Installed models list
- Parameter ranges and best practices per model

The LLM can **see the generated image** (multimodal) and iterate:
> "The skin looks too smooth. Let me increase the LoRA strength and add a detail-enhancing step..."

#### Center: YAML Editor

Monaco editor with custom language support:

```typescript
// YAML language features
const modlYamlLanguage = {
  // Autocomplete for node types
  completionProvider: {
    triggerCharacters: ['.', '$', ':'],
    async provideCompletionItems(position) {
      // After "node: " → suggest all available node types
      // After "$" → suggest step names and their outputs
      // After "params:" → suggest the node's INPUT_TYPES
    }
  },
  
  // Hover for documentation
  hoverProvider: {
    async provideHover(position) {
      // Hover over node type → show description, inputs, outputs
      // Hover over $reference → show what it resolves to
      // Hover over model name → show modl info
    }
  },
  
  // Validation (red squiggles)
  diagnosticsProvider: {
    // Type mismatches between connected steps
    // Unknown node types
    // Missing required params
    // Invalid $references
    // Uninstalled model dependencies
  },
  
  // Go-to-definition for $references
  definitionProvider: {
    // Click $step_name.output → jump to that step
  }
};
```

**Live validation**: As the user types, the runtime validates the YAML and shows errors inline. The LLM sees these errors too and can fix them.

#### Right: Output + Run Panel

- **Parameter form**: Auto-generated from `inputs:` section. Sliders for ranges, text fields for strings, image upload for image inputs.
- **Run button**: Submits to runtime, shows progress bar with step-by-step status.
- **Image gallery**: Results with metadata (seed, params used). Click to compare. Drag to reuse as input.
- **History**: Previous runs with all parameters, rerunnable.

### Responsive Design

- **Desktop**: Three-panel as shown
- **Tablet**: Two panels (chat + output, editor as overlay)
- **Mobile**: Single panel with tabs. Chat-first — the AI generates everything.

---

## Part 4: LLM Integration Details

### Context Assembly

When the LLM needs to help, the system prompt includes:

```
You are a Modl Runtime assistant. You create and modify image generation workflows.

AVAILABLE NODES:
{node_catalog_summary}  // Compact summary of all nodes with types

INSTALLED MODELS:
{modl_list_output}      // What's available locally

CURRENT WORKFLOW:
{current_yaml}          // The workflow being edited

LAST RUN RESULT:
{last_run_metadata}     // Parameters, timing, any errors

FORMAT RULES:
- Use $step_name.output_name for references
- Always include a `requires:` section with modl model IDs
- Always include `inputs:` with sensible defaults
- Prefer named outputs from RETURN_NAMES when available
...
```

### Node Catalog Compression

ComfyUI's `/object_info` returns ~500KB of JSON for all nodes. For the LLM context, compress to essentials:

```yaml
# Compressed node catalog (~20KB)
CheckpointLoaderSimple:
  inputs: { ckpt_name: COMBO[checkpoints] }
  outputs: [model: MODEL, clip: CLIP, vae: VAE]
  
KSampler:
  inputs:
    model: MODEL
    seed: INT[0..MAX]
    steps: INT[1..10000, default=20]
    cfg: FLOAT[0..100, default=8]
    sampler_name: COMBO[euler, euler_ancestral, ...]
    scheduler: COMBO[normal, karras, ...]
    positive: CONDITIONING
    negative: CONDITIONING
    latent_image: LATENT
    denoise: FLOAT[0..1, default=1]
  outputs: [latent: LATENT]

CLIPTextEncode:
  inputs: { clip: CLIP, text: STRING }
  outputs: [conditioning: CONDITIONING]
# ... ~200 most-used nodes
```

### Multi-Turn Workflow Iteration

```
User: "make me a flux workflow for product photos on white background"

Agent: [calls create_flow tool]
→ Generates YAML with flux-dev, CLIPTextEncode, KSampler, appropriate params
→ YAML appears in editor

User: "add controlnet for depth, I want to keep the shape of my product"

Agent: [calls modify_flow tool]  
→ Adds ControlNetLoader + ControlNetApply steps
→ Adds depth_image to inputs
→ Updates requires: to include flux-depth-controlnet

User: "run it" (drags product photo into image input)

Agent: [calls run_flow tool]
→ Runtime compiles YAML → ComfyUI prompt → executes
→ Progress streams to output panel
→ Result image appears

User: "the edges are too soft, make the controlnet stronger"

Agent: [calls modify_flow tool]
→ Changes controlnet strength from 0.7 to 0.9
→ [calls run_flow tool]
→ New result with sharper edges
```

---

## Part 5: modl.lock Integration

### Flow Lock File

Every `*.flow.yaml` can have a companion `*.flow.lock`:

```yaml
# flux-portrait.flow.lock
# Auto-generated by `modl runtime lock`
# Pins exact model versions for reproducibility

generated: 2026-02-25T10:30:00Z
modl_version: 0.2.0
runtime_version: 0.1.0

models:
  flux-dev:
    type: checkpoint
    variant: fp16
    sha256: "a1b2c3d4e5f6..."
    file: flux1-dev.safetensors
    size: 23800000000
  
  flux-vae:
    type: vae
    sha256: "f6e5d4c3b2a1..."
    file: ae.safetensors
    # Implicit dependency of flux-dev
  
  t5-xxl-fp16:
    type: text_encoder
    sha256: "..."
    file: t5xxl_fp16.safetensors
  
  clip-l:
    type: text_encoder
    sha256: "..."
    file: clip_l.safetensors
  
  realistic-skin-v3:
    type: lora
    sha256: "..."
    file: realistic-skin-v3.safetensors

nodes:
  # Built-in nodes included in ComfyUI — pin ComfyUI version
  comfyui: "0.3.10"
  
  # If custom nodes are used:
  # custom_nodes:
  #   - repo: https://github.com/user/comfyui-node-pack
  #     commit: "abc123"
  #     sha256: "..."   # hash of the node pack

# Computed from the flow's `requires:` + transitive dependencies
# `modl runtime install` uses this to set up the exact environment
```

### Commands

```bash
# Resolve and lock all dependencies for a flow
modl runtime lock flux-portrait.flow.yaml

# Install everything needed to run a locked flow  
modl runtime install flux-portrait.flow.yaml

# Run a flow
modl runtime run flux-portrait.flow.yaml --prompt "a woman in a garden"

# Package for deployment (flow + lock + all model references)
modl runtime pack flux-portrait.flow.yaml -o flux-portrait.tar.gz
```

---

## Part 6: Node Registry (Future Extension)

### The Problem

ComfyUI custom nodes are:
- Installed via `git clone` into `custom_nodes/`
- No versioning, no dependency declarations, no compatibility metadata
- ComfyUI Manager helps but it's a patchwork

### Modl Node Registry

Extend the existing modl-registry with a `nodes/` section:

```yaml
# modl-registry/manifests/nodes/comfyui-impact-pack.yaml
id: comfyui-impact-pack
name: "Impact Pack"
type: node_pack
author: ltdrdata
repository: https://github.com/ltdrdata/ComfyUI-Impact-Pack
license: GPL-3.0

versions:
  - version: "7.2.0"
    commit: "abc123def456"
    sha256: "..."  # hash of the repo at this commit
    comfyui_compat: ">=0.3.0"
    python_requires: ">=3.10"
    pip_dependencies:
      - ultralytics>=8.0
      - opencv-python>=4.7
    
    # Nodes provided by this pack
    provides_nodes:
      - SAMLoader
      - SAMDetectorCombined  
      - DetailerForEach
      - FaceDetailer
      # ...

  - version: "7.1.0"
    commit: "previous..."
    # ...

tags: [face, detail, segmentation, SAM]
```

### In Flow Files

```yaml
requires:
  - flux-dev                    # model (existing)
  - node:comfyui-impact-pack    # node pack (new)

steps:
  # Now you can use nodes from the pack
  face_detail:
    node: FaceDetailer
    params: ...
```

### Lock File Includes Nodes

```yaml
# .flow.lock
nodes:
  comfyui: "0.3.10"
  custom:
    comfyui-impact-pack:
      version: "7.2.0"
      commit: "abc123def456"
      sha256: "..."
```

---

## Part 7: Deploy / Cloud Platform

### Deploy Packaging

`modl runtime pack` creates a self-contained deployment bundle:

```
flux-portrait.bundle/
  flow.yaml           # The workflow
  flow.lock           # Pinned dependencies
  manifest.json       # Bundle metadata
  # Models are NOT included (too large) — referenced by hash
  # The deploy platform pulls from modl CDN or HuggingFace
```

### Modal.com Integration

```python
# Auto-generated by `modl runtime deploy --target modal`
import modal

app = modal.App("flux-portrait")

# Volume with models (pre-populated or downloaded on first run)
model_volume = modal.Volume.from_name("modl-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "safetensors", "transformers", "accelerate")
    .pip_install("comfyui-core")  # Hypothetical headless ComfyUI package
    .run_commands("modl runtime install flow.lock --target /models")
)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/models": model_volume},
    timeout=300,
)
def generate(prompt: str, seed: int = -1, **kwargs):
    from modl_runtime import execute_flow
    return execute_flow("flow.yaml", inputs={"prompt": prompt, "seed": seed, **kwargs})

@app.local_entrypoint()
def main(prompt: str):
    result = generate.remote(prompt)
    # Save result images locally
```

### Modl Cloud (Future — The "Vercel for AI Flows" Vision)

```bash
# Deploy to Modl Cloud
modl deploy flux-portrait.flow.yaml

# Returns:
# ✓ Deployed to https://flux-portrait.modl.run
# ✓ API: POST https://flux-portrait.modl.run/api/generate
# ✓ Dashboard: https://app.modl.run/flows/flux-portrait
```

What Modl Cloud does:
- **Serverless GPU** — spin up A100/H100 on demand (backed by Modal/RunPod)
- **Model caching** — popular models pre-loaded on warm instances
- **CDN** — results served from edge
- **Billing** — per-second GPU billing, transparent pricing
- **Monitoring** — latency, success rate, cost per generation
- **Scaling** — auto-scale based on request volume

This is the long-term monetization play — free tier for hobbyists, paid tiers for production.

---

## Part 8: Implementation Phases

### Phase 1: YAML Runtime (4-6 weeks)

**Goal**: `modl runtime run flow.yaml` works locally.

```
modl/src/
  runtime/
    mod.rs              # Module root
    yaml_schema.rs      # Flow YAML serde types
    compiler.rs         # YAML → ComfyUI prompt JSON
    bridge.rs           # HTTP/WS client to ComfyUI
    runner.rs           # Orchestrates compile → submit → stream results
    validator.rs        # Type-check the YAML against node catalog
    lock.rs             # Lock file generation and resolution
```

Deliverables:
- [ ] YAML schema definition (serde structs)
- [ ] Compiler: YAML → ComfyUI prompt JSON
- [ ] Bridge: talk to ComfyUI via HTTP + WebSocket
- [ ] CLI: `modl runtime run`, `modl runtime lock`, `modl runtime install`
- [ ] Node catalog fetcher (from ComfyUI `/object_info`)
- [ ] Validation (type checking, reference resolution)
- [ ] Integration tests with actual ComfyUI

### Phase 2: Web UI — Chat + Editor (6-8 weeks)

**Goal**: Web interface with LLM-assisted workflow creation.

```
modl-app/                     # New repo or monorepo
  src/
    app/
      page.tsx                # Main three-panel layout
      api/
        chat/route.ts         # LLM chat endpoint (Vercel AI SDK)
        run/route.ts          # Workflow execution proxy
        nodes/route.ts        # Node catalog endpoint
    components/
      chat/
        ChatPanel.tsx
        MessageBubble.tsx
        ToolCallDisplay.tsx   # Show what the LLM is doing
      editor/
        YamlEditor.tsx        # Monaco wrapper
        YamlLanguage.ts       # Custom language features
        FlowValidation.ts     # Real-time validation
      output/
        OutputPanel.tsx
        ImageGallery.tsx
        ParameterForm.tsx     # Auto-generated from inputs:
        ProgressBar.tsx
      shared/
        ModelBadge.tsx        # Shows installed model status
        NodeSearch.tsx        # Search available nodes
    lib/
      runtime-client.ts      # HTTP client to Modl Runtime
      websocket.ts           # WebSocket connection manager
      llm-tools.ts           # Tool definitions for the LLM
      yaml-utils.ts          # YAML parsing/manipulation
    stores/
      flow-store.ts          # Current workflow state (Zustand)
      execution-store.ts     # Run history and current progress
```

Deliverables:
- [ ] Three-panel layout (responsive)
- [ ] Monaco YAML editor with autocomplete
- [ ] LLM chat with tool calling (create/modify/run flows)
- [ ] Real-time execution progress via WebSocket
- [ ] Parameter form auto-generation
- [ ] Image gallery with metadata
- [ ] Flow save/load
- [ ] Model status badges (installed/missing)

### Phase 3: Node Registry + Lock Files (3-4 weeks)

**Goal**: Custom nodes are versioned and lockable.

- [ ] Node manifest schema in modl-registry
- [ ] `modl runtime lock` includes node versions
- [ ] `modl runtime install` sets up custom nodes
- [ ] Node search in the UI

### Phase 4: Deploy Platform (6-8 weeks)

**Goal**: One-click deployment to Modal.com, then Modl Cloud.

- [ ] `modl runtime pack` — bundle creation
- [ ] `modl deploy --target modal` — Modal.com codegen
- [ ] Modl Cloud MVP — serverless GPU execution
- [ ] Public API for deployed flows
- [ ] Usage dashboard

---

## Part 9: Key Design Decisions

### Why Reuse ComfyUI vs. Rewrite?

| Consideration | Reuse (recommended) | Rewrite |
|--------------|---------------------|---------|
| Time to first working demo | **2-3 weeks** | 6-12 months |
| Node ecosystem (1000+ custom nodes) | **Day one** | Never (or years of porting) |
| Model support (Flux, SD3, SDXL, Wan, etc.) | **All, immediately** | One at a time |
| GPU memory management | **Battle-tested** | Must reimplement |
| Sampling algorithms | **35+ samplers** | Port one by one |
| Maintenance burden | Track upstream | Own everything |
| Performance | Good enough | Potentially faster (Rust/candle) |
| Distribution | Requires Python | Single binary |

**Verdict**: Reuse for Phase 1-3. Consider selective Rust rewrite of hot paths (sampling, LoRA patching) in Phase 5+ if performance matters.

### Why Rust Server + Python Runtime (Not All-Python)?

The Rust server handles:
- HTTP/WebSocket serving (faster, lower memory)
- YAML parsing and compilation (no Python needed)
- Job scheduling and queue management
- Lock file resolution
- Model dependency resolution (already in modl)
- Static file serving for the web UI
- Deploy packaging

Python handles only:
- Actual ML inference (PyTorch, CUDA)
- Node execution
- Model loading

This split means the "control plane" is fast, single-binary Rust, while the "data plane" is Python+PyTorch on the GPU. Same architecture as TensorFlow Serving, Triton, vLLM, etc.

### Why Not Just a ComfyUI Frontend?

We could build "just a frontend" that generates ComfyUI JSON. But:
- We'd be limited by ComfyUI's API (no lock files, no deployment)
- Users would still need to install/manage ComfyUI separately
- No path to cloud deployment without owning the runtime
- Can't control the node loading or model management

By wrapping ComfyUI as a sidecar, we own the user experience end-to-end while still leveraging its execution engine.

### Alternative: Comfy-Org's API

Comfy-Org now offers a cloud API. We could use that instead of running ComfyUI locally. Good for cloud deployment, but:
- Not open source
- Vendor lock-in
- Can't use custom nodes
- Per-generation pricing instead of per-GPU-second

Consider supporting it as a backend option alongside local ComfyUI.

---

## Part 10: Competitive Landscape

| Tool | Strength | Weakness | Modl Differentiator |
|------|----------|----------|-------------------|
| ComfyUI | Node ecosystem, flexibility | UX, no deployment, no versioning | LLM-first UX, lock files, deployment |
| A1111 | Simple UI, extensions | Stale, limited pipelines | Modern, composable, deployable |
| InvokeAI | Polish, good UX | Small node ecosystem, slow dev | AI-assisted, YAML portability |
| Replicate | Easy deployment | Vendor lock-in, expensive | Open source, portable, cheaper |
| fal.ai | Fast, good API | Limited to their models/pipelines | Bring your own pipeline, any model |
| RunComfy | ComfyUI in the cloud | Just hosted ComfyUI | Better UX, lock files, modl integration |

**Modl' unique position**: The only tool that combines LLM-native workflow authoring, reproducible environments (lock files), and a path from local GPU to cloud deployment — all while maintaining compatibility with ComfyUI's massive node ecosystem.

---

## Appendix A: YAML Format — Advanced Features

### Conditionals

```yaml
steps:
  upscale:
    node: UpscaleImage
    when: $inputs.upscale_enabled    # Only execute if input is true
    params:
      image: $decode.image
      model: 4x-UltraSharp
```

### Loops / Batch

```yaml
steps:
  generate_batch:
    node: KSampler
    repeat: $inputs.batch_count       # Run N times with different seeds
    params:
      seed: $auto_increment           # Auto-increment seed per iteration
      ...
```

### Subflows (Composition)

```yaml
# face-enhance.subflow.yaml
name: "Face Enhancement"
inputs:
  image: { type: image }
  strength: { type: float, default: 0.5 }
steps:
  detect: { node: FaceDetailer, params: { image: $inputs.image, ... } }
outputs:
  image: $detect.image
```

```yaml
# main.flow.yaml
steps:
  generate: { ... }
  enhance:
    subflow: face-enhance
    params:
      image: $generate.image
      strength: 0.7
```

### Templates

```yaml
# flux-base.template.yaml
# Reusable base for Flux workflows
requires:
  - flux-dev

steps:
  load_model:
    node: CheckpointLoaderSimple
    params: { ckpt_name: flux-dev }
  
  encode_positive:
    node: CLIPTextEncode
    params: { clip: $load_model.clip, text: $inputs.prompt }
  
  encode_negative:
    node: CLIPTextEncode
    params: { clip: $load_model.clip, text: $inputs.negative }
```

```yaml
# my-flow.flow.yaml
extends: flux-base

inputs:
  prompt: { type: string }
  negative: { type: string, default: "blurry" }

steps:
  # Inherited: load_model, encode_positive, encode_negative
  # Add your own:
  sample:
    node: KSampler
    params:
      model: $load_model.model
      positive: $encode_positive.conditioning
      ...
```

---

## Appendix B: Directory Structure (Full Vision)

```
modl/
  modl/                    # CLI model manager (existing, Rust)
  modl-registry/           # Model manifests (existing)
  modl-runtime/            # Could live in modl repo as a feature, or separate
    src/runtime/           # Rust: YAML compiler, bridge, job queue
  modl-app/                # Web UI (new repo)
    src/                   # Next.js app
  modl-cloud/              # Deploy platform (future)
    infrastructure/        # Terraform/Pulumi for Modal.com, AWS, etc.
    api/                   # Public API gateway
```

Or, keep it simpler — everything in the `modl` monorepo:

```
modl/
  src/
    cli/                   # Existing CLI commands
    core/                  # Existing core logic
    auth/                  # Existing auth
    compat/                # Existing tool compat
    runtime/               # NEW: YAML runtime
  app/                     # NEW: Web UI (Next.js)
  tests/
```
