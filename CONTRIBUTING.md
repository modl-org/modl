# Contributing to modl

## Adding Support for a New Tool

Modl uses folder layout mappings to know where each tool expects its model files. Adding a new tool is a small, focused PR.

### What you need to add

**1. Add a `ToolType` variant** in `src/core/config.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolType {
    Comfyui,
    A1111,
    Invokeai,
    YourTool,  // <-- add here
}
```

**2. Add a layout function** in `src/compat/layouts.rs`:

```rust
pub fn yourtool(asset_type: &AssetType) -> PathBuf {
    match asset_type {
        AssetType::Checkpoint => PathBuf::from("models/checkpoints"),
        AssetType::Lora => PathBuf::from("models/loras"),
        AssetType::Vae => PathBuf::from("models/vae"),
        // ... map each asset type to the folder your tool expects
    }
}
```

Each `AssetType` must have a mapping. Check the existing layouts in `layouts.rs` for reference — ComfyUI and A1111 are fully implemented.

**3. Wire it up** in `src/compat/mod.rs`:

```rust
pub fn asset_folder(tool_type: &ToolType, asset_type: &AssetType) -> PathBuf {
    match tool_type {
        // ...existing entries...
        ToolType::YourTool => layouts::yourtool(asset_type),
    }
}
```

**4. Add a CLI flag** to `modl link` in `src/cli/mod.rs` and `src/cli/link.rs`:

Add `--yourtool <path>` alongside the existing `--comfyui` and `--a1111` flags.

**5. (Optional) Add auto-detection** in `src/compat/mod.rs` `detect_tools()`:

If your tool has a predictable install location, you can add it to the auto-detection scan. This is nice to have but not required — explicit `modl link --yourtool /path` always works.

### How to verify

```bash
# Install git hooks (runs fmt + clippy before each commit)
cp hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

# Or run manually:
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```

All three must pass. CI runs these on every PR across Linux, macOS, and Windows.

### What makes a good layout mapping

- Verify the paths against the tool's actual source code or documentation, not just community convention
- Include all asset types, even uncommon ones (fall back to a reasonable `models/<type>` path)
- If the tool uses a database or API for model management (like InvokeAI), note that in a comment — symlink-based integration may not be the right fit

## Adding Models to the Registry

Model manifests live in a separate repo: [modl-registry](https://github.com/modl-org/modl-registry). See that repo's CONTRIBUTING.md for how to add models.

## Developing the UI (`modl serve`)

The web UI lives in `src/ui/web/` (React + Vite + Tailwind). Assets are embedded into the Rust binary via `include_str!` at compile time, so the production build is fully self-contained.

### Quick start (recommended)

```bash
./scripts/dev.sh
```

This single script builds the CLI, starts the persistent GPU worker (model caching), the backend API on `:3939`, and the Vite dev server on `:5173`. Open **http://localhost:5173**. Ctrl+C stops everything.

Use `./scripts/dev.sh --no-worker` if you don't have a GPU or don't need generation.

### Manual setup (if you prefer separate terminals)

```bash
# Terminal 1: run the Rust backend (API + file server on :3939)
modl serve

# Terminal 2: run Vite dev server with hot-reload on :5173
cd src/ui/web && npm run dev

# Terminal 3 (optional): start the persistent GPU worker for fast generation
modl worker start
```

### Persistent worker

The persistent worker keeps diffusers models loaded in GPU VRAM between generation requests, reducing inference from ~40s (cold start) to ~10s (warm).

```bash
modl worker start              # start daemon (auto-shuts down after 10 min idle)
modl worker status             # show loaded models, VRAM usage, uptime
modl worker stop               # graceful shutdown, frees VRAM
```

When the worker is running, `modl gen` and the web UI automatically use it. Use `modl gen --no-worker` to force a cold start.

Open **http://localhost:5173** (not :3939). Vite proxies `/api` and `/files` requests to the backend automatically.

When you're ready to commit, build the production bundle and recompile:

```bash
cd src/ui/web && npx vite build   # outputs to src/ui/dist/
cd ../../../ && cargo build --release
```

`src/ui/dist/` is checked into git so that `cargo build` works without requiring Node.js.

## General Guidelines

- Run `cargo fmt` and `cargo clippy -- -D warnings` before submitting
- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality where practical
- The CLI uses `anyhow` for error handling, `console` for styled output, `indicatif` for progress bars

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
