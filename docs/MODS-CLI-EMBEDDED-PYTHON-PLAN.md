# Mods CLI Plan — Embedded Python Runtime + Local/Cloud Extensible Execution

## 0) Goal (Product + DX)

Deliver a **single-command install** for `mods` where users can immediately run:

```bash
mods train --dataset products --base flux-schnell --name product-v1
```

…without manually installing Python, ai-toolkit, CUDA Python packages, or extra setup scripts.

The architecture must:
- Work great locally first
- Add `--cloud` execution later (Modal/other providers) with minimal code churn
- Keep CLI UX stable regardless of local/cloud backend

---

## 1) Non-Negotiables

1. **CLI is the source of truth**
   - Same commands for local and cloud (`mods train`, `mods generate`)
   - `--cloud` is an execution target, not a separate product

2. **Managed Python runtime by mods**
   - No dependency on system Python for core workflows
   - `mods` owns Python version + package set + compatibility checks

3. **Strict interface boundaries**
   - Rust defines job spec + events + state model
   - Python workers execute jobs and emit structured events
   - Cloud/local both implement the same execution contract

4. **Extensibility before optimization**
   - Design for pluggable executors (`local`, `cloud`) from day one
   - Avoid hard-coding provider-specific assumptions in CLI commands

---

## 2) UX Spec (Target Experience)

## Installation UX

### A) One-line install

Examples:

```bash
curl -fsSL https://mods.sh/install | bash
# or
brew install modshq/tap/mods
```

Installs:
- `mods` Rust binary
- Minimal runtime bootstrap metadata (no giant payload during installer)

### A.1) What "minimal bootstrap" explicitly means

Installer payload target: **< 80MB**.

Included in installer/bootstrap:
- `mods` binary
- runtime manifest index (profiles + hashes + compatibility metadata)
- runtime resolver and health-check logic

Not included in installer/bootstrap:
- PyTorch wheels
- CUDA runtime wheels
- ai-toolkit/diffusers heavy dependencies

Heavy dependencies are downloaded lazily on first command that needs them.

### B) First training/generation command auto-bootstraps runtime

First time user runs:

```bash
mods train ...
```

`mods` automatically:
1. Detects GPU / environment
2. Downloads correct managed Python runtime bundle
3. Creates isolated runtime dir under `~/.mods/runtime/`
4. Installs pinned ML packages for selected backend profile
5. Persists runtime lock metadata

Expected first-run UX for training:

```text
Preparing mods training runtime (one-time): profile trainer-cu124
Downloading dependencies (~6GB)...
This happens only once per profile/version.
```

All transparent, with progress + clear status messages.

### C) Explicit control commands

```bash
mods runtime status
mods runtime install
mods runtime doctor
mods runtime upgrade
mods runtime reset
```

This gives power users control while preserving zero-config defaults.

---

## 3) Runtime Packaging Strategy ("Embedded" Python)

Use **managed embedded Python distributions**, not system Python.

## Recommended approach (Phase 1)

1. Package per-platform Python runtime artifacts (**Linux only in Phase 1**):
   - CPython standalone distribution (or equivalent reproducible build)
   - Prebuilt wheelhouse index for pinned package sets

2. Store runtime in:

```text
~/.mods/runtime/
  python/
    3.11.11/
      bin/python
      ...
  envs/
    trainer-cu124/
    inference-cu124/
  wheelhouse/
  manifests/
    runtime.lock.json
```

3. `mods` resolves and installs a profile:
   - `trainer-cu124`
  - `inference-cu124` (Phase 2)

Phase 1 intentionally excludes:
- CUDA 11.x
- ROCm
- Apple Silicon/MPS
- Windows/macOS

These are explicit future profiles after Linux + CUDA 12.x is stable.

4. Runtime lock file pins:
   - python version
   - package versions
   - wheel hashes
   - build compatibility constraints

## Why this path

- True one-line UX
- Reproducibility across machines
- Safe upgrades/rollbacks
- Keeps CLI in control of compatibility matrix

---

## 4) Architecture Contracts (No-Refactor Future)

## Core abstraction: `ExecutionTarget`

Define once and keep stable:

```rust
trait ExecutionTarget {
    fn submit_train(&self, spec: TrainJobSpec) -> Result<JobHandle>;
    fn submit_generate(&self, spec: GenerateJobSpec) -> Result<JobHandle>;
    fn stream_events(&self, job_id: &str) -> EventStream;
    fn cancel(&self, job_id: &str) -> Result<()>;
    fn fetch_artifacts(&self, job_id: &str) -> Result<Vec<Artifact>>;
}
```

Implementations:
- `LocalExecutor` (spawns managed Python worker)
- `CloudExecutor` (queues remote job, streams remote events)

CLI never talks directly to Python or cloud APIs; it only talks to `ExecutionTarget`.

## Stable job schemas

Use versioned JSON schemas:
- `TrainJobSpec v1`
- `GenerateJobSpec v1`
- `JobEvent v1`

Design rule: specs must be composable so future orchestration can chain jobs (train → generate → evaluate) without introducing a second abstraction.

These must be transport-agnostic (local process/stdout, websocket, HTTP).

## Event protocol (JSONL)

Python worker emits structured JSON lines:

```json
{"type":"job_started","job_id":"..."}
{"type":"progress","step":120,"total_steps":2000,"loss":0.0821}
{"type":"artifact","kind":"sample_image","path":"..."}
{"type":"completed","artifacts":[...]}
{"type":"error","code":"CUDA_MISMATCH","message":"...","recoverable":false}
```

Same event model is used by cloud backend.

Error events are first-class protocol messages, not ad-hoc stderr parsing.

---

## 5) CLI Command Surface (Future-proof)

## Training

```bash
mods train --dataset products --base flux-dev --name product-v1
mods train ... --target local
mods train ... --cloud
mods train ... --provider modal
mods train ... --subscription pro
```

Rules:
- `--cloud` is shorthand for `--target cloud`
- provider defaults from config
- command output format is identical across local/cloud

## Generation

```bash
mods generate "a photo of OHWX" --lora product-v1
mods generate "..." --target local
mods generate "..." --cloud
```

## Runtime/admin

```bash
mods runtime status
mods runtime profiles ls
mods runtime install --profile trainer-cu124
mods runtime doctor
```

## Cloud/admin

```bash
mods cloud login
mods cloud providers ls
mods cloud subscription status
mods cloud quotas
```

---

## 6) Data Model + State

Extend existing SQLite with target-agnostic tables:

- `jobs`
  - `id`, `kind` (`train|generate`), `target` (`local|cloud`), `provider`, `status`, `created_at`, `finished_at`
- `job_events`
  - append-only normalized events
- `artifacts`
  - output images, lora files, logs, metadata
- `runtime_installations`
  - python/runtime profiles and health info
- `subscriptions`
  - provider entitlements/cached capabilities

This allows:
- Same `mods jobs` and `mods outputs` UX for both local/cloud
- Resume/history independent of backend

---

## 7) Python Worker Encapsulation

Ship a `mods_worker` Python package managed by `mods`.

## Python boundary responsibilities

`mods_worker` does:
- `protocol.py`: canonical event emitter + schema helpers + error mapping
- `adapters/train_adapter.py`: `TrainJobSpec` → ai-toolkit config + train execution
- `adapters/generate_adapter.py`: generation backend adapter + execution
- `main.py`: worker entrypoint + lifecycle hooks + command routing

`mods` Rust does:
- Job planning + validation
- Model resolution from mods store
- Runtime lifecycle
- Progress rendering
- Persistence in SQLite
- Local/cloud routing

## Anti-coupling rule

Do not leak ai-toolkit-specific fields into top-level CLI flags unless required.
Keep ai-toolkit details in adapter layer + advanced config escape hatch.

Worker split is mandatory to keep backend swaps (ai-toolkit → other trainer) low-risk.

---

## 8) Cloud Extensibility Plan (No Big Refactor)

## Provider adapter interface

```rust
trait CloudProvider {
    fn submit_job(&self, spec: JobSpec) -> Result<RemoteJobId>;
    fn stream_events(&self, remote_job_id: &str) -> EventStream;
    fn get_artifacts(&self, remote_job_id: &str) -> Result<Vec<Artifact>>;
    fn capabilities(&self) -> Result<ProviderCapabilities>;
}
```

First provider: Modal.
Future providers: RunPod, Lambda Labs, internal workers.

## Subscription/capability gating

Before submission:
- resolve user subscription tier
- verify requested workload is allowed (`gpu_type`, `max_steps`, concurrency)
- downgrade/fail fast with actionable message

Capability model sits above provider-specific APIs.

## Artifact sync model

All cloud artifacts normalized to local mods store:
- downloaded + content-addressed
- registered in same DB tables as local outputs

User still sees one unified history.

---

## 9) Configuration Design

Extend `~/.mods/config.yaml`:

```yaml
runtime:
  auto_install: true
  python_version: "3.11.11"
  profile: "trainer-cu124"
  channel: "stable"

execution:
  default_target: "local"   # local | cloud
  default_provider: "modal"

cloud:
  modal:
    token: "..."
```

Rules:
- `--cloud` overrides `default_target`
- missing cloud token → guided login flow
- runtime profile auto-selected but overridable
- subscription tier is server-authoritative (never configured locally)

## Configuration precedence (explicit)

Highest to lowest:
1. CLI flags
2. Environment variables
3. Project config (`./.mods/config.yaml`) (future)
4. User config (`~/.mods/config.yaml`)
5. Built-in defaults

This precedence must be consistent for runtime, execution target, and cloud provider settings.

For cloud entitlement checks, local config may cache non-authoritative metadata only; server/provider response is source of truth.

---

## 10) Phased Delivery Plan

## Phase 1 — Local Embedded Runtime Foundation

Deliver:
- `mods runtime install/status/doctor`
- Managed Python distribution install
- Pinned runtime profile install
- `LocalExecutor` + JSONL worker events
- `mods train` local path via worker
- Process lifecycle policy: **spawn-per-job** for training

Exit criteria:
- Fresh Linux machine can train with no manual Python setup
- Scope locked to Linux + CUDA 12.x + ai-toolkit + local target only

Out of scope in Phase 1:
- long-running local generation worker
- cloud execution
- non-Linux platforms

## Phase 2 — Generate + Shared Execution Core

Deliver:
- `mods generate` on same execution/event pipeline
- Unified job/event/artifact persistence
- `mods jobs` and `mods outputs` read unified DB
- Local generation lifecycle policy decision:
  - start with spawn-per-job
  - add optional warm worker mode only if startup overhead is proven significant

Exit criteria:
- Local train + generate both run through shared job model

## Phase 3 — Cloud Adapter (Modal MVP)

Deliver:
- `CloudExecutor` + `ModalProvider`
- `mods train --cloud` and `mods generate --cloud`
- auth + quota checks (launch)
- artifact sync to local store

Exit criteria:
- identical CLI UX for local and cloud execution

## Phase 4 — Multi-provider + Policy Engine

Deliver:
- provider registry
- smarter scheduling (`auto`: local if idle, cloud if queue/backpressure)
- full capability matrix, budget limits, org/team policy hooks

Exit criteria:
- backend choice becomes a config/policy concern, not CLI refactor

---

## 11) Risks + Mitigations

1. **Python/CUDA dependency drift**
   - Mitigation: pinned runtime profiles + lock manifests + staged channels (`stable`, `beta`)

2. **Large runtime payloads**
   - Mitigation: bootstrap minimal, lazy-install per profile, shared wheel cache

3. **ai-toolkit API churn**
   - Mitigation: strict adapter layer in `mods_worker`; contract tests for `JobEvent` schema

4. **Cloud provider lock-in**
   - Mitigation: provider trait + neutral job/event schema + artifact normalization

---

## 12) Suggested Repository Layout Changes

```text
mods/
  src/
    runtime/
      manager.rs          # install/upgrade/doctor embedded Python runtime
      profiles.rs         # runtime profile resolution
    execution/
      mod.rs
      target.rs           # ExecutionTarget trait
      local.rs            # LocalExecutor
      cloud.rs            # CloudExecutor facade
    cloud/
      mod.rs
      provider.rs         # CloudProvider trait
      modal.rs            # ModalProvider implementation
    jobs/
      schema.rs           # JobSpec/Event/Artifact v1
      store.rs            # DB persistence helpers

  python/
    mods_worker/
      __init__.py
      main.py             # entrypoint for local worker jobs
      protocol.py         # event schema mirror + error mapping
      adapters/
        train_adapter.py    # ai-toolkit wrapper
        generate_adapter.py
    requirements/
      trainer-cu124.txt
      trainer-cpu.txt
      inference-cu124.txt

  runtime-manifests/
    profiles/
      trainer-cu124.json
      trainer-cpu.json
```

---

## 13) Acceptance Criteria (Product-level)

A new user should be able to:

1. Install in one line
2. Run `mods train ...` immediately
3. See automatic runtime bootstrap with clear progress
4. Re-run without reinstall overhead
5. Switch to cloud using `--cloud` without changing command semantics
6. Retrieve all outputs/history in one place (`mods outputs`, `mods jobs`)

If all six are true, the UX promise is met.

---

## 14) What Not To Do (to avoid refactor debt)

- Do not make CLI command trees diverge for local vs cloud
- Do not parse unstructured Python logs as long-term contract
- Do not persist provider-native event formats directly in DB
- Do not rely on system Python for core workflows
- Do not expose provider-specific flags on core commands unless behind namespaced options

---

## 15) Immediate Next Planning Outputs

After approving this plan, produce these specs before coding:

1. `jobs-schema-v1.md` (JSON schema definitions)
2. `runtime-profile-spec.md` (how embedded runtime manifests are described)
3. `execution-target-interface.md` (Rust trait methods + semantics)
4. `cloud-capability-model.md` (subscription/quota/capability checks)
5. `mods-worker-protocol.md` (JSONL event and error contract)

These five docs will make implementation parallelizable and minimize architecture drift.
