# Execution Target Interface

> **STATUS:** `LocalExecutor` fully implemented. `CloudExecutor` stubbed (provider enum + cred resolution done, `submit` returns not-implemented). See [cloud plan](../cloud/plan.md) for cloud roadmap.

## Purpose

Define the stable execution contract used by CLI commands and higher-level orchestration.

The CLI must depend only on this interface, never on Python process details or cloud provider APIs.

---

## Core Interface

```rust
pub trait ExecutionTarget {
    fn submit_train(&self, spec: TrainJobSpec) -> Result<JobHandle>;
    fn submit_generate(&self, spec: GenerateJobSpec) -> Result<JobHandle>;
    fn stream_events(&self, job_id: &str) -> Result<EventStream>;
    fn cancel(&self, job_id: &str) -> Result<()>;
    fn fetch_artifacts(&self, job_id: &str) -> Result<Vec<Artifact>>;
    fn get_status(&self, job_id: &str) -> Result<JobStatus>;
}
```

Implementations:
- `LocalExecutor`
- `CloudExecutor`

---

## Semantics

## `submit_train`

- Validates `TrainJobSpec`
- Persists job row with `status=queued`
- Returns `JobHandle { job_id }`
- Must be idempotent when `idempotency_key` provided

## `submit_generate`

- Same semantics as `submit_train`

## `stream_events`

- Returns ordered `JobEvent v1` stream
- Must include terminal event (`completed`, `error`, or `cancelled`)
- Reconnect behavior:
  - caller may provide `from_sequence`
  - target must replay from that sequence when available

## `cancel`

- Best-effort cancellation
- Emits `cancelled` event on success
- Returns success if job already terminal

## `fetch_artifacts`

- Returns normalized artifacts
- For cloud target, must ensure local sync before returning local file paths

## `get_status`

- Returns latest known status snapshot
- Must not replace event-stream consumption; snapshot only

---

## Job Lifecycle State Machine

```text
queued -> accepted -> running -> completed
queued -> accepted -> running -> error
queued -> cancelled
accepted -> cancelled
running -> cancelled
```

Rules:
- terminal states: `completed | error | cancelled`
- no transitions out of terminal states

---

## Local Executor Contract

## Phase 1

- Spawn-per-job process model for training
- Worker launched via managed runtime Python entrypoint
- Event transport: worker stdout JSONL
- stderr reserved for diagnostics; not protocol

Failure mapping requirements:
- process spawn failure -> `EXECUTOR_START_FAILED`
- protocol decode failure -> `PROTOCOL_DECODE_ERROR`
- non-zero exit without terminal event -> synthesize `WORKER_EXIT_NONZERO`

## Phase 2+

- Optional warm worker for generation if startup overhead warrants it
- Must preserve same event and status semantics

---

## Cloud Executor Contract

- Delegates execution to `CloudProvider`
- Converts provider events to canonical `JobEvent v1`
- Persists canonical events before exposing to CLI

Cloud executor must hide provider specifics from command handlers.

---

## Timeout and Retry Policy

- `submit_*` operations are retried on transient transport failures
- job execution retries are policy-based and explicit in spec
- retries must emit `warning` events with retry attempt metadata

---

## Configuration Resolution

Target selection precedence:
1. CLI flags (`--target`, `--cloud`, `--provider`)
2. environment variables
3. project config (future)
4. user config
5. defaults

Resolver output:
- `ResolvedTarget { target, provider, profile }`

---

## Telemetry and Auditing

Each execution call must include:
- `correlation_id`
- `job_id`
- `target`
- `provider` (nullable)

These identifiers must be present in logs and persisted events for traceability.

---

## Compatibility Policy

This interface is considered stable once Phase 1 ships.
Additive method changes require default impl or versioned trait migration.
Breaking changes require explicit migration plan.
