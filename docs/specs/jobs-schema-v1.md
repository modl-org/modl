# Jobs Schema v1

> **STATUS: IMPLEMENTED** — These schemas match the types in `src/core/job.rs`. Canonical reference.

## Purpose

Define transport-agnostic, backend-neutral schemas for all executable jobs in `modl`.

Design goals:
- One schema for local and cloud
- Strongly typed events/artifacts
- Composable for future multi-step pipelines (train → generate → evaluate)

---

## Versioning Rules

- Every top-level schema includes `schema_version`.
- `v1` additions must be backward-compatible (new optional fields only).
- Breaking changes require `v2`.

---

## Common Types

## `JobId`

- Format: ULID string
- Example: `01JT8Q5J9Q2Y3X5F8R6G2M7A1K`

## `Target`

- Enum: `local | cloud`

## `JobKind`

- Enum: `train | generate | evaluate` (`evaluate` reserved, not implemented in phase 1)

## `Provider`

- Nullable string
- Examples: `modal`, `runpod`

## `ArtifactKind`

- Enum:
  - `lora`
  - `image`
  - `sample_image`
  - `log`
  - `metrics`
  - `manifest`

---

## `TrainJobSpec v1`

```json
{
  "schema_version": "v1",
  "job_kind": "train",
  "target": "local",
  "provider": null,
  "idempotency_key": "optional-string",
  "project_id": "optional-string",
  "correlation_id": "optional-string",
  "dataset": {
    "name": "products",
    "path": "/home/user/.modl/datasets/products",
    "image_count": 12,
    "caption_coverage": 0.83
  },
  "model": {
    "base_model_id": "flux-schnell",
    "base_model_path": "/home/user/.modl/store/.../flux-schnell.safetensors"
  },
  "output": {
    "lora_name": "product-v1",
    "destination_dir": "/home/user/.modl/outputs/training"
  },
  "training": {
    "preset": "standard",
    "trigger_word": "OHWX",
    "steps": 2000,
    "rank": 16,
    "learning_rate": 0.0001,
    "optimizer": "adamw8bit",
    "resolution": [1024, 1024],
    "seed": -1,
    "advanced_config_path": null
  },
  "runtime": {
    "profile": "trainer-cu124",
    "python_version": "3.11.11",
    "timeout_seconds": 0
  },
  "labels": {
    "env": "dev"
  }
}
```

Validation rules:
- `steps` range: `1..=100000`
- `rank` range: `1..=256`
- `learning_rate` range: `(0, 1)`
- `resolution` both dimensions `>= 256`
- `target=cloud` requires `provider`

---

## `GenerateJobSpec v1`

```json
{
  "schema_version": "v1",
  "job_kind": "generate",
  "target": "local",
  "provider": null,
  "idempotency_key": "optional-string",
  "project_id": "optional-string",
  "correlation_id": "optional-string",
  "model": {
    "base_model_id": "flux-schnell",
    "lora": {
      "id": "product-v1",
      "path": "/home/user/.modl/store/.../product-v1.safetensors",
      "strength": 1.0
    }
  },
  "prompt": {
    "positive": "a photo of OHWX on marble countertop",
    "negative": "blurry, low quality"
  },
  "sampling": {
    "seed": 42,
    "steps": 28,
    "guidance": 3.5,
    "size": [1024, 1024],
    "count": 1
  },
  "runtime": {
    "profile": "inference-cu124",
    "python_version": "3.11.11",
    "timeout_seconds": 0
  },
  "labels": {
    "env": "dev"
  }
}
```

Validation rules:
- `count` range: `1..=64`
- `steps` range: `1..=200`
- `guidance` range: `0..=30`

---

## `JobEvent v1`

Envelope:

```json
{
  "schema_version": "v1",
  "job_id": "01JT8Q...",
  "sequence": 17,
  "timestamp": "2026-02-26T18:00:00Z",
  "source": "worker",
  "event": { "type": "progress", "step": 120, "total_steps": 2000, "loss": 0.0821 }
}
```

Event payload variants:
- `job_accepted`
- `job_started`
- `progress`
- `artifact`
- `log`
- `warning`
- `completed`
- `cancelled`
- `error`
- `heartbeat`

Ordering guarantees:
- `sequence` strictly monotonic per `job_id`
- `completed|error|cancelled` are terminal

---

## `Artifact v1`

```json
{
  "schema_version": "v1",
  "job_id": "01JT8Q...",
  "artifact_id": "01JT8R...",
  "kind": "lora",
  "path": "/home/user/.modl/store/.../product-v1.safetensors",
  "uri": null,
  "sha256": "...",
  "size_bytes": 123456789,
  "metadata": {
    "step": 2000,
    "seed": 42
  },
  "created_at": "2026-02-26T18:40:00Z"
}
```

---

## Composability Extension (Reserved in v1)

Include optional `upstream_job_ids` in all specs:

```json
"upstream_job_ids": ["01JT8Q...", "01JT8R..."]
```

This enables future `modl pipeline` orchestration without changing the core schema family.

---

## Storage Mapping (SQLite)

- `jobs`: one row per submitted job
- `job_events`: append-only stream with `sequence`
- `artifacts`: normalized artifact rows

The schemas above are canonical; DB projections may denormalize for query performance.
