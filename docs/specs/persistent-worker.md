# Persistent Worker & Model Caching

> **STATUS: PLANNED — Phase 2.** Current one-shot subprocess model works, but every `modl gen` pays a 20-45s cold start loading ~24GB of model weights into VRAM. This spec describes the persistent worker architecture that eliminates that overhead.

---

## Problem

Every `modl gen` invocation currently:

1. Spawns a new Python process (~2-4s: import torch + diffusers)
2. Loads model from disk (~15-30s: deserialize 24GB of safetensors)
3. Transfers to GPU (~3-8s: CPU → VRAM)
4. Loads LoRA + fuse (~1-3s)
5. Runs inference (~8-15s on a 4090 at 28 steps, 1024×1024)
6. Process exits, GPU memory freed

**Cold start overhead is 2-5× the actual inference time.** For interactive use (one image at a time), 70-80% of wall time is wasted on loading.

ComfyUI solves this by being a long-running server. The model stays in VRAM between requests. Second generation of the same model takes ~10s instead of ~40s.

---

## Current Architecture (one-shot)

```
modl gen "prompt"
  │
  ▼
LocalExecutor::submit_generate()
  │  write spec → /tmp/jobs/{id}.yaml
  │  spawn: python -m modl_worker.main generate --config <spec.yaml>
  ▼
Python process (modl_worker/main.py)
  │  parse args → run_generate(config_path, emitter)
  │  load pipeline from scratch (from_pretrained / from_single_file)
  │  generate N images → emit artifact events
  │  exit(0)
  ▼
Process dies, VRAM freed, model gone
```

`--count N` does reuse the loaded model within one invocation (1 cold load + N inference passes). But the next `modl gen` starts from zero.

---

## Proposed Architecture (persistent worker)

```
modl worker start             # or: spawned automatically on first gen
  │
  ▼
Python daemon (modl_worker/main.py serve)
  │  listen on Unix socket: ~/.modl/worker.sock
  │  model cache: dict[CacheKey, Pipeline]
  │  idle timeout: 10 min (configurable)
  ▼
modl gen "prompt"
  │
  ▼
LocalExecutor::submit_generate()
  │  check: is worker alive? (connect to ~/.modl/worker.sock)
  │  YES → send spec over socket, read JSONL events back
  │  NO  → spawn worker, wait for ready, then send spec
  ▼
Worker receives spec:
  │  cache_key = (model_id, lora_id, dtype)
  │  HIT  → reuse loaded pipeline (~0s)
  │  MISS → load pipeline, evict LRU if needed, cache it
  │  run inference → emit events over socket
  │  stay alive, wait for next job
```

### Spawning

Two modes, both using the same worker process:

| Mode | How | When |
|------|-----|------|
| **CLI flag** | `modl worker start` / `modl worker stop` / `modl worker status` | Manual control, power users |
| **Auto-spawn** | `modl gen` detects no worker → spawns one in background | Default UX, zero friction |
| **UI** | `modl serve` starts the worker as part of the web server | Future web UI |

Auto-spawn detail:
1. `LocalExecutor::submit_generate()` tries to connect to `~/.modl/worker.sock`
2. Connection refused → spawn `python -m modl_worker.main serve` as a background process
3. Write PID to `~/.modl/worker.pid`
4. Retry socket connection with backoff (up to 30s for first model load)
5. If spawn fails → fall back to one-shot mode (current behavior, always works)

### Worker lifecycle

```
modl worker start [--timeout 600] [--model flux-dev]
  │  bind ~/.modl/worker.sock
  │  optionally pre-load a model (--model flag)
  │  emit ready signal
  ▼
  Serving loop:
    │  accept connection → read spec JSON → run job → write JSONL events → close conn
    │  reset idle timer after each job
    │  if idle_timeout exceeded → shutdown gracefully
  ▼
modl worker stop
  │  send SIGTERM → worker emits cancelled, flushes, exits
  │  or: idle timeout → self-shutdown
```

---

## CLI Interface

```bash
# Manual worker management
modl worker start                  # start persistent worker (background)
modl worker start --timeout 1800   # 30 min idle timeout (default: 600s)
modl worker start --model flux-dev # pre-load model into VRAM on startup
modl worker stop                   # graceful shutdown
modl worker status                 # show: running/stopped, loaded models, VRAM usage, uptime

# Generation (auto-spawns worker if not running)
modl gen "a cat"                   # connects to worker or spawns one
modl gen "a cat" --no-worker       # force one-shot mode (legacy behavior)
```

### `modl worker status` output

```
Worker: running (PID 12345, uptime 8m)
Socket: ~/.modl/worker.sock
Models loaded:
  flux-dev (bf16, 23.8 GB VRAM)    last used: 2m ago
  └─ LoRA: brutalist-v1            fused, weight=1.0
Idle timeout: 10m (8m remaining)
```

---

## Protocol Extension

The existing JSONL worker protocol (`schema_version: v1`) works as-is over a Unix socket instead of stdout/stdin. One addition:

### New transport: Unix socket

| Aspect | One-shot (current) | Persistent (new) |
|--------|-------------------|-------------------|
| **Transport** | stdout (events), stdin (unused) | Unix socket (bidirectional) |
| **Request** | CLI args + spec YAML file | JSON envelope over socket |
| **Response** | JSONL on stdout | JSONL on socket |
| **Lifecycle** | process-per-job | connection-per-job, process persists |

### Socket request format

```json
{
  "action": "generate",
  "job_id": "gen-01JT...",
  "spec": { ... GenerateJobSpec ... }
}
```

### Socket response

Same JSONL events as current stdout protocol. Connection closed after terminal event.

### Control commands (on the same socket)

```json
{"action": "status"}
{"action": "preload", "model_id": "flux-dev"}
{"action": "evict", "model_id": "flux-dev"}
{"action": "shutdown"}
```

---

## Model Cache Design

```python
# In modl_worker/serve.py (new file)

@dataclass
class CacheKey:
    model_id: str
    dtype: str  # "bfloat16", "float16"

@dataclass
class CachedPipeline:
    pipeline: Any           # diffusers Pipeline object
    loaded_at: float        # time.time()
    last_used: float
    vram_estimate_mb: int
    lora_id: str | None     # currently fused LoRA (if any)
    lora_weight: float

class ModelCache:
    def __init__(self, max_models: int = 2):
        self._cache: dict[CacheKey, CachedPipeline] = {}
        self._max_models = max_models

    def get_or_load(self, spec: dict, emitter: EventEmitter) -> Pipeline:
        key = CacheKey(model_id=spec["model"]["base_model_id"], dtype="bfloat16")

        if key in self._cache:
            cached = self._cache[key]
            cached.last_used = time.time()
            # Handle LoRA changes
            self._reconcile_lora(cached, spec, emitter)
            return cached.pipeline

        # Evict LRU if at capacity
        if len(self._cache) >= self._max_models:
            self._evict_lru()

        # Load fresh pipeline
        pipeline = self._load_pipeline(spec, emitter)
        self._cache[key] = CachedPipeline(
            pipeline=pipeline,
            loaded_at=time.time(),
            last_used=time.time(),
            vram_estimate_mb=self._estimate_vram(pipeline),
            lora_id=None,
            lora_weight=0.0,
        )
        return pipeline
```

### LoRA hot-swap

When the cached base model matches but the LoRA changed:

```python
def _reconcile_lora(self, cached: CachedPipeline, spec: dict, emitter: EventEmitter):
    requested_lora = spec.get("lora", {}).get("path")
    if cached.lora_id == requested_lora:
        return  # already fused

    # Unfuse current LoRA
    if cached.lora_id:
        cached.pipeline.unfuse_lora()
        cached.pipeline.unload_lora_weights()
        emitter.info(f"Unloaded LoRA: {cached.lora_id}")

    # Load new LoRA
    if requested_lora:
        weight = spec.get("lora", {}).get("weight", 1.0)
        cached.pipeline.load_lora_weights(requested_lora)
        cached.pipeline.fuse_lora(lora_scale=weight)
        cached.lora_id = requested_lora
        cached.lora_weight = weight
        emitter.info(f"Hot-swapped LoRA: {requested_lora} (weight={weight})")
```

This avoids the 20-30s base model reload when only the LoRA changes.

---

## Incremental Optimizations (independent of persistent worker)

These can be done before or alongside the persistent worker:

### 1. Batch inference for `--count` (trivial, ~2h)

Current code runs a for-loop with `num_images_per_prompt=1`. Change to:

```python
# gen_adapter.py — in the generation loop
batch_size = min(count, 4)  # cap by VRAM
gen_kwargs["num_images_per_prompt"] = batch_size
result = pipe(**gen_kwargs)
# result.images is a list of batch_size images
```

Parallelizes the denoising loop across images. Only helps for `--count > 1`.

### 2. LoRA hot-swap without persistent worker (~half day)

Even in one-shot mode, if we detect a `--lora` change with the same base model, we can skip the base model reload:

- Cache the last-used pipeline in a global variable within the process
- Only useful for `--count` with multiple LoRAs (future feature)

Won't help much without a persistent worker since the process exits after one job.

### 3. `torch.compile()` for repeat inference (~1 day)

```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
# or for Flux:
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
```

30-60s compile cost on first run, ~20-30% faster inference after. **Only worthwhile with a persistent worker** (compile result is lost when process exits).

---

## Phasing

### Phase 1 (current): One-shot, works today

- Every `modl gen` is a cold start
- `--count N` reuses the model within one invocation
- Acceptable for training (inherently long-running) and occasional generation

### Phase 2: Persistent worker

**Prerequisites:**
- E2E generation flow validated on GPU ← **do this first**
- gen_adapter.py is stable and tested

**Implementation order:**
1. Add `serve` command to `modl_worker/main.py` with socket listener
2. Add `ModelCache` class with LRU eviction
3. Add `modl worker start/stop/status` CLI commands (Rust side)
4. Modify `LocalExecutor::submit_generate()` to prefer socket, fall back to one-shot
5. Add auto-spawn logic (detect dead worker, spawn on first gen)
6. Add idle timeout + graceful shutdown
7. Add LoRA hot-swap (unfuse/load/fuse without reloading base)
8. Add `torch.compile()` behind a `--compile` flag

**Estimated effort:** ~3-5 days for the core (steps 1-6), +1 day for LoRA hot-swap, +1 day for torch.compile.

### Phase 3: UI integration

When `modl serve` launches the web UI, it also manages the worker:

```
modl serve
  ├── HTTP server (Rust, serves UI)
  ├── WebSocket bridge (Rust → worker socket → UI)
  └── Worker process (Python, persistent, auto-managed)
```

The UI can show:
- Worker status (loaded models, VRAM, uptime)
- Start/stop/preload controls
- Real-time generation streaming via WebSocket

---

## Comparison: Current vs Persistent

| Metric | One-shot (now) | Persistent worker |
|--------|---------------|-------------------|
| First generation | ~35-50s | ~35-50s (same cold start) |
| Second gen (same model) | ~35-50s | **~8-15s** |
| Second gen (different LoRA) | ~35-50s | **~10-18s** (hot-swap) |
| Second gen (different model) | ~35-50s | ~35-50s (cache miss, evict LRU) |
| `--count 4` | ~35s + 3×10s = ~65s | ~35s + 3×10s = ~65s (same) |
| Idle VRAM | 0 | ~24GB (Flux-dev cached) |
| Process overhead | New Python per job | One long-lived process |
| Failure mode | Clean (process exits) | Needs health checks + restart |

The persistent worker transforms interactive generation from "wait 40s, get image" to "wait 10s, get image" — which is the difference between usable and frustrating.

---

## File Map (new/modified files)

| File | Change |
|------|--------|
| `python/modl_worker/main.py` | Add `serve` subcommand |
| `python/modl_worker/serve.py` | **New.** Socket listener, ModelCache, serve loop |
| `python/modl_worker/adapters/gen_adapter.py` | Accept optional pre-loaded pipeline arg |
| `src/cli/mod.rs` | Add `worker` subcommand group |
| `src/cli/worker.rs` | **New.** `start`, `stop`, `status` commands |
| `src/core/executor.rs` | `submit_generate()` try socket first, fall back to one-shot |

No changes to:
- `src/core/job.rs` (spec format unchanged)
- `python/modl_worker/protocol.py` (JSONL format unchanged)
- `src/cli/train.rs` (training stays one-shot — inherently long-running)
