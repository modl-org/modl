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

## General Guidelines

- Run `cargo fmt` and `cargo clippy -- -D warnings` before submitting
- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality where practical
- The CLI uses `anyhow` for error handling, `console` for styled output, `indicatif` for progress bars

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
