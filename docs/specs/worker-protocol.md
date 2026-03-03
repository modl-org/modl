# modl_worker Protocol (JSONL)

> **STATUS: IMPLEMENTED** — Matches `python/modl_worker/protocol.py` and event parsing in `src/core/executor.rs`. Canonical reference.

## Purpose

Define the canonical JSONL protocol between `modl` (Rust) and `modl_worker` (Python).

This protocol is the long-term contract for local execution and the template for cloud event normalization.

---

## Transport

- One JSON object per line (UTF-8)
- Emitted on stdout
- stderr reserved for non-protocol diagnostics
- Each line must be independently parseable

---

## Envelope

All events use this envelope:

```json
{
  "schema_version": "v1",
  "job_id": "01JT8Q...",
  "sequence": 1,
  "timestamp": "2026-02-26T18:00:00Z",
  "source": "modl_worker",
  "event": {
    "type": "job_started"
  }
}
```

Rules:
- `sequence` strictly monotonic per job
- `timestamp` ISO-8601 UTC
- exactly one terminal event per job

---

## Event Types

## `job_accepted`

```json
{"type":"job_accepted","worker_pid":12345}
```

## `job_started`

```json
{"type":"job_started","stage":"init"}
```

## `progress`

```json
{"type":"progress","stage":"train","step":120,"total_steps":2000,"loss":0.0821,"eta_seconds":842}
```

## `artifact`

```json
{"type":"artifact","kind":"sample_image","path":"/tmp/sample-0120.png","sha256":"...","size_bytes":12345,"metadata":{"step":120}}
```

## `log`

```json
{"type":"log","level":"info","message":"Loading dataset"}
```

## `warning`

```json
{"type":"warning","code":"LOW_DISK_SPACE","message":"Less than 10GB free"}
```

## `heartbeat`

```json
{"type":"heartbeat","uptime_seconds":300}
```

## `completed`

```json
{"type":"completed","duration_seconds":2450,"artifacts":[{"kind":"lora","path":"/tmp/product-v1.safetensors"}]}
```

## `cancelled`

```json
{"type":"cancelled","reason":"user_requested"}
```

## `error`

```json
{"type":"error","code":"CUDA_MISMATCH","message":"Detected CUDA 11.8 but profile requires CUDA 12.x","recoverable":false,"details":{"detected":"11.8","required":"12.x"}}
```

---

## Error Taxonomy v1

Runtime/setup:
- `RUNTIME_NOT_INSTALLED`
- `PYTHON_EXEC_MISSING`
- `DEPENDENCY_IMPORT_FAILED`
- `PROFILE_INCOMPATIBLE`

Environment/GPU:
- `GPU_NOT_FOUND`
- `CUDA_NOT_FOUND`
- `CUDA_MISMATCH`
- `CUDA_OOM`

Input/config:
- `SPEC_VALIDATION_FAILED`
- `DATASET_INVALID`
- `MODEL_NOT_FOUND`
- `ADAPTER_CONFIG_FAILED`

Execution:
- `TRAINING_FAILED`
- `GENERATION_FAILED`
- `WORKER_INTERNAL_ERROR`
- `INTERRUPTED`

Protocol:
- `PROTOCOL_VIOLATION`

Each error event requires:
- `code`
- human-readable `message`
- `recoverable` boolean
- optional `details` object

---

## Worker Module Split (Required)

`modl_worker` structure:

```text
modl_worker/
  main.py
  protocol.py
  adapters/
    train_adapter.py
    generate_adapter.py
```

Responsibilities:
- `protocol.py`: event emitters, sequence tracking, error mapping helpers
- `adapters/train_adapter.py`: ai-toolkit train integration only
- `adapters/generate_adapter.py`: generation integration only
- `main.py`: command dispatch + lifecycle

---

## Lifecycle and Exit Codes

- Exit `0` only after `completed` or `cancelled`
- Exit non-zero after `error`
- If non-zero exit happens without an `error` event, Rust synthesizes `WORKER_EXIT_NONZERO`

Signal handling:
- SIGTERM -> emit `cancelled` when possible then exit 0
- SIGINT -> same behavior

---

## Handshake Sequence

Minimum expected startup sequence:

1. `job_accepted`
2. `job_started`
3. zero or more (`progress`, `artifact`, `log`, `warning`, `heartbeat`)
4. terminal event (`completed` | `error` | `cancelled`)

Any deviation may trigger `PROTOCOL_VIOLATION` in host runtime.

---

## Compatibility Rules

- Unknown optional fields must be ignored by host
- Unknown event types should map to warning + preserved raw payload
- `schema_version` mismatch triggers hard failure unless compatible mapping is explicitly defined

---

## Testing Requirements

Contract tests required for:
- happy-path train and generate
- each terminal event kind
- representative errors (`CUDA_MISMATCH`, `CUDA_OOM`, `SPEC_VALIDATION_FAILED`)
- malformed JSON line handling
- missing terminal event handling
