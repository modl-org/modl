# GPU Resource Management

> Spec for cross-process GPU coordination in modl.

## Problem

Training, generation, captioning, and prompt enhancement all compete for the
same GPU. These run as separate OS processes (CLI invocations), so in-process
mutexes don't work. Today:

- Training has no awareness of other GPU tasks
- `modl generate` from CLI has zero GPU checks
- `modl dataset caption` blindly calls `.to("cuda")`
- The web UI disables generate during training, but only in the frontend
- Server-side only prevents concurrent generation (AtomicBool), not generation
  during training
- Python workers crash with unhelpful PyTorch OOM errors when VRAM is exhausted

Users have wildly different GPUs (8GB–48GB+). A static "don't do X during Y"
rule is too conservative for a 4090 and too optimistic for a 3060.

## Design: Activity Lock + VRAM Budget

### Core: `~/.modl/gpu.lock`

A JSON file written atomically, read before any GPU operation:

```json
{
  "activities": [
    {
      "pid": 12345,
      "task": "training",
      "model": "sdxl",
      "vram_est_mb": 22000,
      "exclusive": true,
      "started_at": "2026-03-04T10:00:00Z"
    }
  ]
}
```

- **Atomic writes**: write to `gpu.lock.tmp`, `rename()` over `gpu.lock`
- **Stale detection**: on read, check each PID with `kill(pid, 0)` — prune dead entries
- **RAII guard**: `GpuGuard` releases the activity on drop (including panic unwind)

### Task VRAM Profiles

Each GPU operation declares its expected VRAM footprint:

| Task | Model example | Est. VRAM (MB) | Exclusive? |
|------|---------------|----------------|------------|
| Training | SDXL | ~22,000 | Yes |
| Training | Flux (fp8) | ~24,000 | Yes |
| Generation | SDXL | ~8,000 | Soft* |
| Generation | Flux (fp8) | ~12,000 | Soft* |
| Captioning | Florence-2 | ~2,000 | No |
| Captioning | Qwen3-VL-8B (fp8) | ~5,000 | No |
| LLM Enhance | Qwen3.5-4B (Q4) | ~3,500 | No |
| LLM Enhance | Builtin (rules) | 0 | No |
| LLM Enhance | Remote API | 0 | No |

*"Soft exclusive" means it blocks other pipelines (loading two diffusion models
thrashes VRAM) but could coexist with small resident models on large GPUs.

### Compatibility Rules

Simple priority system, not a full matrix:

1. **Training is king** — when training runs, nothing else touches the GPU.
   Training uses nearly all VRAM and runs for hours. Don't risk OOM corruption.

2. **One pipeline at a time** — generation and captioning load different models
   into VRAM. Running both means loading, unloading, reloading. Just serialize.

3. **Small models can coexist** — a 3.5GB Q4 LLM fits alongside a 12GB Flux
   pipeline on a 24GB GPU. The lock system checks `vram_free ≥ new_task_estimate`.

4. **Zero-GPU tasks are always allowed** — builtin rule-based enhance, remote
   API calls, CPU-only operations bypass the lock entirely.

### Decision logic

```
fn can_acquire(existing: &[Activity], new: &GpuTask, gpu: &GpuInfo) -> Verdict {
    // Training blocks everything
    if existing.iter().any(|a| a.exclusive) {
        return Verdict::Blocked("GPU busy — training active")
    }

    // New training blocks if anything is running
    if new.is_exclusive() && !existing.is_empty() {
        return Verdict::Blocked("GPU busy — wait for current task to finish")
    }

    // Zero-VRAM tasks always pass
    if new.vram_est_mb() == 0 {
        return Verdict::Allowed
    }

    // Check VRAM budget
    let used: u64 = existing.iter().map(|a| a.vram_est_mb).sum();
    let available = gpu.vram_mb.saturating_sub(used);

    if new.vram_est_mb() > available {
        return Verdict::InsufficientVram { needed: new.vram_est_mb(), available }
    }

    // Pipeline conflict: two diffusion pipelines can't coexist
    if new.is_pipeline() && existing.iter().any(|a| a.is_pipeline()) {
        return Verdict::Blocked("Another pipeline is already loaded")
    }

    Verdict::Allowed
}
```

### CLI UX

**Normal operation:**
```
$ modl generate "a cat" --model flux-dev
→ GPU: RTX 4090 (24576 MB total, 22012 MB free)
→ Estimated VRAM: ~12000 MB
Generating...
```

**Blocked by training:**
```
$ modl generate "a cat" --model flux-dev
✗ GPU busy: training sdxl (PID 12345, started 2h 14m ago)
  Cannot generate while training is active.
  Options:
    --force    Override lock (OOM risk)
    --cloud    Use cloud GPU (requires modl auth)
    --wait     Wait for current task to finish
```

**Marginal VRAM:**
```
$ modl dataset caption photos/ --model qwen
⚠ Low VRAM: need ~5000 MB, only 4200 MB estimated free
  Proceeding — PyTorch may offload to system RAM (slower).
  Use --force to skip this warning, or --cpu to run on CPU.
```

**`modl gpu` command:**
```
$ modl gpu
GPU: NVIDIA RTX 4090
VRAM: 24576 MB total, 2564 MB free

Active tasks:
  training sdxl  PID 12345  22012 MB est.  2h 14m  ███████████░ 82%

$ modl gpu --release
Released stale GPU lock (PID 99999 no longer running)
```

### `--force` escape hatch

Power users who know their GPU can handle it:

```
$ modl generate "a cat" --force
⚠ Overriding GPU lock. OOM errors possible.
```

`--force` bypasses all lock checks. Logged so `modl doctor` can report
"you've been force-overriding — if you're seeing OOM, this is why."

## Rust API

```rust
// src/core/gpu_lock.rs

pub enum GpuTask {
    Training { model: String, vram_est_mb: u64 },
    Generation { model: String, vram_est_mb: u64 },
    Captioning { model: String, vram_est_mb: u64 },
    LlmEnhance { model: String, vram_est_mb: u64 },
    Other { label: String, vram_est_mb: u64 },
}

pub struct GpuGuard { /* releases on drop */ }

pub struct GpuLock;

impl GpuLock {
    /// Acquire GPU for a task. Returns error if blocked.
    pub fn acquire(task: GpuTask) -> Result<GpuGuard>;

    /// Non-blocking check. Returns None if can't acquire.
    pub fn try_acquire(task: GpuTask) -> Result<Option<GpuGuard>>;

    /// Check current GPU activity (prunes stale PIDs).
    pub fn current() -> Result<Vec<GpuActivity>>;

    /// Force-release all locks.
    pub fn release_all() -> Result<()>;
}
```

Every CLI command that touches the GPU wraps its work in a guard:

```rust
// modl generate
let _guard = GpuLock::acquire(GpuTask::Generation {
    model: model_id.clone(),
    vram_est_mb: estimate_gen_vram(&model_id),
})?;
// ... spawn Python worker, generate images ...
// guard drops → lock released automatically
```

The web server replaces its `AtomicBool` with `GpuLock`:

```rust
// POST /api/generate
let guard = match GpuLock::try_acquire(task) {
    Ok(Some(g)) => g,
    Ok(None) => return (StatusCode::CONFLICT, Json(busy_response)),
    Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, ...),
};
// ... guard lives until generation completes
```

## VRAM Estimation

Estimates come from a static table, not runtime measurement. Good enough for
gating (±20% is fine — the goal is preventing obvious conflicts, not precise
accounting):

```rust
fn estimate_vram(model_id: &str, task: TaskType) -> u64 {
    // Look up model in registry → get architecture + variant
    // Apply task-specific multiplier:
    //   training: base_vram * 2.5 (optimizer states, gradients)
    //   generation: base_vram * 1.0
    //   captioning: depends on VL model
    //   enhance: depends on LLM model + quantization
}
```

The registry already has `vram_required_mb` per variant. Reuse this.

## Enhance Graceful Degradation

The prompt enhance feature has a natural fallback chain — it should **never**
be blocked:

```
1. GPU available + LLM model installed → local LLM on GPU (fast)
2. GPU busy + LLM model installed → local LLM on CPU (slow, ~5-10s)
3. No LLM model → builtin rule-based enhancer (instant)
4. Config has remote API → call remote endpoint (fast, needs network)
```

Auto-negotiation in `enhance_prompt()`:
```rust
pub fn enhance_prompt(...) -> Result<EnhanceResult> {
    let enhancer = resolve_enhancer(); // checks GPU lock, config, installed models
    enhancer.enhance(req)
}

fn resolve_enhancer() -> Box<dyn PromptEnhancer> {
    if let Some(remote) = config.enhance_api_url {
        return Box::new(RemoteEnhancer::new(remote));
    }
    if llm_model_installed() {
        if gpu_available_for(GpuTask::LlmEnhance { .. }) {
            return Box::new(LocalLlmEnhancer::new(Device::Cuda));
        }
        return Box::new(LocalLlmEnhancer::new(Device::Cpu));
    }
    Box::new(BuiltinEnhancer) // always works, zero deps
}
```

This means the ✨ Enhance button in the UI **always works** — it just gets faster
when you have a local LLM and GPU headroom.

## Implementation Phases

| Phase | What | Effort |
|-------|------|--------|
| **Now (Phase 4)** | `GpuLock` file-based lock + `GpuGuard` RAII + `modl gpu` command | ~2h |
| **Now (Phase 4)** | Gate `modl generate`, `modl train`, `modl dataset caption` | ~1h |
| **Phase 5** | Persistent worker manages GPU internally (model cache = explicit VRAM tracking) | Part of worker spec |
| **Phase 6** | Replace `AtomicBool` in server.rs with `GpuLock` | ~30min |
| **Phase 6** | `/api/gpu` returns lock activity, not just `training_active` | ~30min |
| **Phase 9** | Enhance auto-negotiation (GPU → CPU → builtin fallback) | ~1h |

## What about multi-GPU?

Not now. `gpu.lock` tracks device index 0. When someone actually has multi-GPU
and asks for it, extend the lock file with a `device` field. The architecture
supports it — each activity already has all the metadata needed.

## What about `modl doctor`?

Add GPU diagnostics:
```
$ modl doctor
...
GPU: NVIDIA RTX 4090 (24576 MB)
  ✓ CUDA available (driver 560.35.03)
  ✓ NVML accessible
  ✓ No stale GPU locks
  ⚠ VRAM: only 2564 MB free (training active?)
```
