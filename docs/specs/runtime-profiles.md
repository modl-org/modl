# Runtime Profile Specification

> **STATUS: PARTIALLY IMPLEMENTED / ASPIRATIONAL**
> The actual runtime (`src/core/runtime.rs`, 803 LOC) uses a simpler approach:
> Python venv creation + pip install from requirements. The manifest index,
> profile JSON schemas, and lockfile format described below are a *target
> architecture* not yet built. The current system works but doesn't have the
> full reproducibility guarantees this spec envisions.

## Purpose

Define how `modl` describes, resolves, installs, verifies, and upgrades managed embedded Python runtimes.

Phase 1 scope:
- Linux only
- CUDA 12.x family only
- Local execution only

---

## Concepts

## Runtime Channel

- `stable` (default)
- `beta`

Channel selects which profile manifests are eligible.

## Runtime Profile

A profile is an installable compatibility unit, e.g.:
- `trainer-cu124`
- `inference-cu124` (phase 2)

A profile defines:
- Python runtime version
- Required packages and hashes
- Platform constraints
- GPU/runtime constraints

---

## Directory Layout

```text
~/.modl/runtime/
  python/
    3.11.11/
      bin/python
  envs/
    trainer-cu124/
      pyvenv.cfg
      bin/python
  wheelhouse/
    linux_x86_64/cu124/
  manifests/
    index.json
    profiles/
      trainer-cu124.json
  locks/
    runtime.lock.json
```

---

## Manifest Index Schema

File: `~/.modl/runtime/manifests/index.json`

```json
{
  "schema_version": "v1",
  "channel": "stable",
  "generated_at": "2026-02-26T18:00:00Z",
  "profiles": [
    {
      "id": "trainer-cu124",
      "version": "2026.02.1",
      "manifest_uri": "https://github.com/modl/modl-runtime-manifests/releases/download/v2026.02.1/trainer-cu124.json",
      "sha256": "..."
    }
  ]
}
```

---

## Profile Manifest Schema

```json
{
  "schema_version": "v1",
  "id": "trainer-cu124",
  "version": "2026.02.1",
  "channel": "stable",
  "description": "Training profile for Linux CUDA 12.4",
  "constraints": {
    "os": ["linux"],
    "arch": ["x86_64", "aarch64"],
    "gpu": {
      "required": true,
      "cuda_major": 12,
      "cuda_min_minor": 4,
      "cuda_max_minor": 9
    }
  },
  "python": {
    "version": "3.11.11",
    "artifact_uri": "https://github.com/modl/modl-runtime-manifests/releases/download/v2026.02.1/cpython-3.11.11-linux-x86_64.tar.zst",
    "sha256": "..."
  },
  "packages": [
    {
      "name": "torch",
      "version": "2.5.1",
      "wheel_uri": "https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-linux_x86_64.whl",
      "sha256": "..."
    },
    {
      "name": "ai-toolkit",
      "version": "0.9.0",
      "wheel_uri": "https://files.pythonhosted.org/packages/.../ai_toolkit-0.9.0-py3-none-any.whl",
      "sha256": "..."
    }
  ],
  "entrypoints": {
    "worker": "modl_worker.main"
  }
}
```

---

## Runtime Lock Schema

File: `~/.modl/runtime/locks/runtime.lock.json`

```json
{
  "schema_version": "v1",
  "installed_at": "2026-02-26T18:10:00Z",
  "channel": "stable",
  "profile": {
    "id": "trainer-cu124",
    "version": "2026.02.1",
    "sha256": "..."
  },
  "python": {
    "version": "3.11.11",
    "sha256": "..."
  },
  "packages": [
    {"name": "torch", "version": "2.5.1", "sha256": "..."},
    {"name": "ai-toolkit", "version": "0.9.0", "sha256": "..."}
  ]
}
```

---

## Resolution Algorithm

1. Resolve requested profile (CLI flag or config default)
2. Fetch/validate manifest index signature/hash
3. Select profile by `id + channel`
4. Validate platform constraints
5. Validate GPU/CUDA constraints
6. Download python artifact + wheels (if missing)
7. Verify hashes for all artifacts
8. Install venv and packages
9. Write `runtime.lock.json`
10. Run post-install health checks (`modl runtime doctor`)

Failure at any step is atomic for lock creation.

## Artifact Sources Policy

Launch sources:
- Runtime/index/profile artifacts: GitHub Releases (versioned and checksummed)
- Python ecosystem packages: PyPI (or direct files.pythonhosted.org artifacts)
- PyTorch GPU wheels: official PyTorch wheel index

No `cdn.modl.sh` dependency in launch path.

---

## Commands Contract

## `modl runtime install [--profile ID] [--channel stable|beta]`

- Installs or updates profile
- Prints expected download size before confirmation unless `--yes`

## `modl runtime status`

- Prints installed profile/version, lock status, health summary

## `modl runtime doctor`

Checks:
- Python executable health
- package import sanity
- CUDA availability for selected profile
- worker entrypoint viability

## `modl runtime upgrade`

- Moves to latest compatible version in channel
- Preserves previous lock for rollback

## `modl runtime reset`

- Deletes env + lock for selected profile
- Keeps manifests/wheel cache unless `--purge-cache`

---

## Bootstrap Budget and UX

Install-time payload target:
- bootstrap: < 80MB

First-command lazy payload:
- trainer profile may be multi-GB

Required user messaging:
- show profile selected
- show estimated download size
- show one-time nature of heavy install
- show resumable download behavior

---

## Out of Scope (Phase 1)

- ROCm profiles
- MPS profiles
- CUDA 11.x variants
- Windows/macOS installers
- Multi-profile concurrent scheduling
