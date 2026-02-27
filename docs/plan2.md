Plan to implement                                              │
│                                                                │
│ Plan: Shippable mods train with Pluggable Executor             │
│                                                                │
│ Context                                                        │
│                                                                │
│ The feat/train-command branch has infrastructure plumbing      │
│ (runtime management, Python worker, subprocess orchestration)  │
│ but none of the user-facing training UX from PLAN.md. The goal │
│  is to ship the local training MVP while keeping a clean seam  │
│ for --cloud later — adding cloud should mean implementing one  │
│ new struct, not refactoring the pipeline.                      │
│                                                                │
│ The key insight: the job spec is the contract. Same            │
│ TrainJobSpec struct gets built by presets, persisted to DB,    │
│ and handed to whichever executor runs it. Everything above the │
│  executor (interactive prompts, presets, dataset validation)   │
│ and below it (artifact collection, DB tracking, symlinks) is   │
│ shared code.                                                   │
│                                                                │
│ ---                                                            │
│ Architecture                                                   │
│                                                                │
│ CLI layer (interactive prompts, progress display)              │
│     │                                                          │
│     ├── presets::resolve_params()  ── pure logic, no I/O       │
│     ├── dataset::validate()        ── filesystem scan          │
│     ├── gpu::detect()              ── existing code            │
│     │                                                          │
│     ▼                                                          │
│ TrainJobSpec  ◄── the serialization boundary                   │
│     │                                                          │
│     ▼                                                          │
│ ┌─────────────────────┐                                        │
│ │  dyn Executor       │  ◄── trait: submit / events / cancel   │
│ ├─────────────────────┤                                        │
│ │  LocalExecutor      │  ◄── Phase 1 (this plan)               │
│ │  CloudExecutor      │  ◄── Phase 3 (future, just impl the    │
│ trait)                                                         │
│ └─────────────────────┘                                        │
│     │                                                          │
│     ▼                                                          │
│ artifacts::collect_lora()  ── hash, store, register, symlink   │
│                                                                │
│ ---                                                            │
│ Implementation Steps                                           │
│                                                                │
│ Step 1: Core types — src/core/job.rs (new)                     │
│                                                                │
│ All shared types that form the contract between CLI, executor, │
│  and DB:                                                       │
│                                                                │
│ - TrainJobSpec — dataset ref, model ref, output ref, training  │
│ params, runtime ref, labels                                    │
│ - ExecutionTarget enum — Local | Cloud                         │
│ - DatasetRef — name, path, image_count, caption_coverage       │
│ - ModelRef — base_model_id, optional base_model_path           │
│ - OutputRef — lora_name, destination_dir                       │
│ - TrainingParams — preset, trigger_word, steps, rank,          │
│ learning_rate, optimizer, resolution, seed, quantize           │
│ - RuntimeRef — profile, python_version                         │
│ - JobEvent — envelope with schema_version, job_id, sequence,   │
│ timestamp, source, event payload                               │
│ - EventPayload — tagged enum: JobAccepted, JobStarted,         │
│ Progress, Artifact, Log, Warning, Completed, Error, Cancelled, │
│  Heartbeat                                                     │
│ - JobStatus enum — Queued, Accepted, Running, Completed,       │
│ Error, Cancelled                                               │
│                                                                │
│ Step 2: Presets — src/core/presets.rs (new)                    │
│                                                                │
│ Pure logic, no I/O. Takes dataset stats + VRAM + base model,   │
│ returns TrainingParams.                                        │
│                                                                │
│ ┌──────────┬───────────────┬─────────┬──────────┬──────────┐   │
│ │  Preset  │     Steps     │  Rank   │    LR    │  Notes   │   │
│ ├──────────┼───────────────┼─────────┼──────────┼──────────┤   │
│ │          │ 150/img, min  │         │          │ ~20 min  │   │
│ │ Quick    │ 1000, max     │ 8       │ 1e-4     │ on 4090  │   │
│ │          │ 1500          │         │          │          │   │
│ ├──────────┼───────────────┼─────────┼──────────┼──────────┤   │
│ │          │ 200/img, min  │ 16 (<20 │ 5e-5     │          │   │
│ │ Standard │ 2000, max     │  img)   │ (<10     │ ~45 min  │   │
│ │          │ 4000          │ or 32   │ img) or  │ on 4090  │   │
│ │          │               │         │ 1e-4     │          │   │
│ ├──────────┼───────────────┼─────────┼──────────┼──────────┤   │
│ │          │ defaults,     │         │          │ Opens    │   │
│ │ Advanced │ user edits    │ 16      │ 1e-4     │ $EDITOR  │   │
│ │          │ YAML          │         │          │          │   │
│ └──────────┴───────────────┴─────────┴──────────┴──────────┘   │
│                                                                │
│ Auto-settings:                                                 │
│ - quantize = true if VRAM < 40GB                               │
│ - Resolution from base model (flux/sdxl → 1024, sd-1.5 → 512)  │
│ - Optimizer always adamw8bit                                   │
│                                                                │
│ Inline #[cfg(test)] module with tests for each preset tier and │
│  scaling rules.                                                │
│                                                                │
│ Step 3: Dataset management — src/core/dataset.rs (new)         │
│                                                                │
│ - create(name, from_dir) — copies/symlinks images to           │
│ ~/.mods/datasets/<name>/, validates extensions (jpg/jpeg/png   │
│ only), finds paired .txt caption files                         │
│ - scan(path) — returns DatasetInfo { name, path, image_count,  │
│ captioned_count, caption_coverage, images }                    │
│ - validate(path) — calls scan, fails if 0 images, warns if < 5 │
│ - list() — scans ~/.mods/datasets/                             │
│                                                                │
│ Step 4: Dataset CLI — src/cli/datasets.rs (new)                │
│                                                                │
│ Subcommand: mods datasets <create|list|validate>               │
│                                                                │
│ - create <name> --from <dir> — calls dataset::create, prints   │
│ summary table                                                  │
│ - list — calls dataset::list, prints table (name, images,      │
│ captions, path)                                                │
│ - validate <name> — calls dataset::validate, prints report     │
│                                                                │
│ Register as Datasets variant in Commands enum.                 │
│                                                                │
│ Step 5: Executor trait — src/core/executor.rs (new)            │
│                                                                │
│ Three methods, that's it:                                      │
│                                                                │
│ pub trait Executor {                                           │
│     fn submit(&mut self, spec: TrainJobSpec) ->                │
│ Result<JobHandle>;                                             │
│     fn events(&mut self, job_id: &str) ->                      │
│ Result<mpsc::Receiver<JobEvent>>;                              │
│     fn cancel(&self, job_id: &str) -> Result<()>;              │
│ }                                                              │
│                                                                │
│ LocalExecutor — refactors the guts of                          │
│ training.rs::run_python_training_proxy:                        │
│ - submit: writes spec YAML to                                  │
│ ~/.mods/runtime/jobs/<job_id>.yaml, spawns Python worker on a  │
│ background thread, worker stdout parsed into JobEvent values   │
│ sent through mpsc::Sender                                      │
│ - events: returns the receiver half                            │
│ - cancel: sends SIGTERM to stored child PID                    │
│                                                                │
│ Uses std::sync::mpsc (not async) — matches the sync stdout     │
│ reading pattern. Future CloudExecutor can spawn a tokio task   │
│ that polls HTTP and sends into the same channel type.          │
│                                                                │
│ Step 6: Job tracking — src/core/db.rs (modify)                 │
│                                                                │
│ Add 3 tables to existing CREATE TABLE IF NOT EXISTS migration  │
│ block:                                                         │
│                                                                │
│ - jobs — job_id PK, kind, status, spec_json, target, provider, │
│  created_at, started_at, completed_at                          │
│ - job_events — job_id + sequence (unique), event_json,         │
│ timestamp                                                      │
│ - artifacts — artifact_id PK, job_id FK, kind, path, sha256,   │
│ size_bytes, metadata JSON, created_at                          │
│                                                                │
│ New methods: insert_job, update_job_status, list_jobs,         │
│ insert_job_event, insert_artifact, list_artifacts.             │
│                                                                │
│ Step 7: Artifact collection — src/core/artifacts.rs (new)      │
│                                                                │
│ collect_lora(output_path, lora_name, base_model, trigger_word, │
│  job_id, db, config):                                          │
│                                                                │
│ 1. Hash the .safetensors file (reuse Store::hash_file)         │
│ 2. Move to content-addressed store (reuse Store::path_for +    │
│ Store::ensure_dir)                                             │
│ 3. Register in installed table as AssetType::Lora (reuse       │
│ existing db.insert_installed)                                  │
│ 4. Register in artifacts table (link to job)                   │
│ 5. Create symlinks to configured targets (reuse existing       │
│ symlink module)                                                │
│                                                                │
│ Step 8: Python worker updates                                  │
│                                                                │
│ python/mods_worker/adapters/train_adapter.py (modify):         │
│ - Add spec_to_aitoolkit_config(spec) — translates TrainJobSpec │
│  fields to ai-toolkit's YAML format (single function, one      │
│ place to maintain)                                             │
│ - Add scan_output_artifacts(output_dir, emitter) — after       │
│ training, glob for *.safetensors, emit artifact events with    │
│ path, sha256, size                                             │
│ - Emit job_accepted with worker_pid as the first event         │
│                                                                │
│ python/mods_worker/protocol.py (modify):                       │
│ - Add convenience methods: job_accepted(), job_started(),      │
│ progress(), artifact(), completed()                            │
│ - Accept job_id in constructor, include in every envelope      │
│                                                                │
│ python/mods_worker/main.py (modify):                           │
│ - Accept --job-id argument, pass to EventEmitter               │
│ - Read full TrainJobSpec YAML (not minimal config)             │
│                                                                │
│ Step 9: Rewrite CLI train — src/cli/train.rs (rewrite)         │
│                                                                │
│ Update Commands::Train args: all optional (dataset, base,      │
│ name, trigger, preset, steps, config, dry_run).                │
│                                                                │
│ Flow:                                                          │
│ 1. --config <yaml> → escape hatch, load spec directly          │
│ 2. Missing args → interactive mode with dialoguer:             │
│   - Dataset: Select from dataset::list() or enter path         │
│   - Base model: Select from hardcoded list (flux-dev,          │
│ flux-schnell)                                                  │
│   - Trigger word: Input with default "OHWX"                    │
│   - Name: Input with default <dataset>-v1                      │
│   - Preset: Select from Quick/Standard/Advanced                │
│ 3. dataset::validate() the chosen dataset                      │
│ 4. gpu::detect() for VRAM                                      │
│ 5. presets::resolve_params() → TrainingParams                  │
│ 6. If Advanced → write YAML to temp, open $EDITOR, re-parse    │
│ 7. Assemble TrainJobSpec                                       │
│ 8. If --dry-run → print spec YAML, exit                        │
│ 9. db.insert_job(status=queued)                                │
│ 10. LocalExecutor::from_runtime_setup() (bootstraps runtime if │
│  needed)                                                       │
│ 11. executor.submit(spec) → JobHandle                          │
│ 12. executor.events(job_id) → event loop with                  │
│ indicatif::ProgressBar                                         │
│   - Progress → update bar (step/total, loss, ETA)              │
│   - Artifact → store path for later collection                 │
│   - Completed → break                                          │
│   - Error → print message, break                               │
│ 13. db.update_job_completed()                                  │
│ 14. artifacts::collect_lora() for each artifact event          │
│ 15. Print summary: LoRA name, path, SHA256, symlink locations  │
│                                                                │
│ Step 10: Module registration                                   │
│                                                                │
│ - src/core/mod.rs — add: pub mod artifacts, pub mod dataset,   │
│ pub mod executor, pub mod job, pub mod presets                 │
│ - src/cli/mod.rs — add: mod datasets, update Commands enum     │
│ with Datasets subcommand, update Train variant args, add       │
│ dispatch arms                                                  │
│ - Clean up old src/core/training.rs — keep                     │
│ resolve_worker_python_root() helper, remove the rest (moved to │
│  executor)                                                     │
│ - Clean up old src/cli/train_setup.rs — keep as-is (still      │
│ useful for explicit runtime bootstrap)                         │
│                                                                │
│ ---                                                            │
│ Files Summary                                                  │
│                                                                │
│ File: src/core/job.rs                                          │
│ Action: New                                                    │
│ Purpose: Shared types: TrainJobSpec, JobEvent, EventPayload,   │
│   JobStatus                                                    │
│ ────────────────────────────────────────                       │
│ File: src/core/presets.rs                                      │
│ Action: New                                                    │
│ Purpose: Preset logic: Quick/Standard/Advanced param           │
│ resolution                                                     │
│ ────────────────────────────────────────                       │
│ File: src/core/dataset.rs                                      │
│ Action: New                                                    │
│ Purpose: Dataset create/scan/validate/list                     │
│ ────────────────────────────────────────                       │
│ File: src/core/executor.rs                                     │
│ Action: New                                                    │
│ Purpose: Executor trait + LocalExecutor                        │
│ ────────────────────────────────────────                       │
│ File: src/core/artifacts.rs                                    │
│ Action: New                                                    │
│ Purpose: LoRA collection: hash, store, register, symlink       │
│ ────────────────────────────────────────                       │
│ File: src/core/db.rs                                           │
│ Action: Modify                                                 │
│ Purpose: Add jobs/job_events/artifacts tables + CRUD           │
│ ────────────────────────────────────────                       │
│ File: src/core/mod.rs                                          │
│ Action: Modify                                                 │
│ Purpose: Register new modules                                  │
│ ────────────────────────────────────────                       │
│ File: src/core/training.rs                                     │
│ Action: Trim                                                   │
│ Purpose: Keep worker root helper, remove proxy (moved to       │
│   executor)                                                    │
│ ────────────────────────────────────────                       │
│ File: src/cli/train.rs                                         │
│ Action: Rewrite                                                │
│ Purpose: Interactive prompts, executor dispatch, progress      │
│ display                                                        │
│ ────────────────────────────────────────                       │
│ File: src/cli/datasets.rs                                      │
│ Action: New                                                    │
│ Purpose: Dataset subcommands                                   │
│ ────────────────────────────────────────                       │
│ File: src/cli/mod.rs                                           │
│ Action: Modify                                                 │
│ Purpose: Register datasets, update Train args                  │
│ ────────────────────────────────────────                       │
│ File: python/mods_worker/main.py                               │
│ Action: Modify                                                 │
│ Purpose: Accept --job-id, read full spec                       │
│ ────────────────────────────────────────                       │
│ File: python/mods_worker/protocol.py                           │
│ Action: Modify                                                 │
│ Purpose: Add convenience emitters, job_id support              │
│ ────────────────────────────────────────                       │
│ File: python/mods_worker/adapters/train_adapter.py             │
│ Action: Modify                                                 │
│ Purpose: Spec translation, artifact scanning                   │
│                                                                │
│ ---                                                            │
│ Cloud-Readiness: What Adding --cloud Looks Like Later          │
│                                                                │
│ New code:                                                      │
│ - src/core/cloud_executor.rs — implements Executor trait       │
│ (dataset upload, API calls, event polling, artifact download)  │
│ - Add --cloud / --provider flags to Commands::Train            │
│                                                                │
│ Untouched code (everything else):                              │
│ - TrainJobSpec — same struct serialized as JSON in API call    │
│ - presets.rs — same preset logic                               │
│ - dataset.rs — same validation (cloud executor handles upload) │
│ - db.rs — same tables                                          │
│ - artifacts.rs — same collection (cloud executor downloads     │
│ artifact first)                                                │
│ - cli/train.rs — same flow, one branch:                        │
│ let executor: Box<dyn Executor> = if cloud {                   │
│     Box::new(CloudExecutor::new(provider)?)                    │
│ } else {                                                       │
│     Box::new(LocalExecutor::from_runtime_setup().await?)       │
│ };                                                             │
│                                                                │
│ ---                                                            │
│ Verification                                                   │
│                                                                │
│ 1. Unit tests: Preset scaling rules, dataset scanning with     │
│ tempfile fixtures, job spec serialization round-trips, DB CRUD │
│  with in-memory SQLite                                         │
│ 2. Integration test: mods datasets create test --from          │
│ ./fixtures → mods train --dataset test --base flux-schnell     │
│ --name test-v1 --preset quick --dry-run → verify spec YAML     │
│ output                                                         │
│ 3. Manual E2E: Run mods train interactively on a real dataset  │
│ with a GPU, verify LoRA appears in mods list --type lora  