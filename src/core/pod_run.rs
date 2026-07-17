//! Run modl workflows ON a pod — the pod is an ephemeral remote modl server.
//!
//! Instead of rewriting job specs and intercepting artifacts per step (the
//! PodExecutor spike), the pod gets a full modl install: the release tarball
//! already ships the binary with `python/modl_worker` beside it, so the pod
//! runs the exact same code path as a local machine — real store (`modl pull`
//! resolves models pod-side, no storeless special-casing), the real workflow
//! engine (step chaining never leaves the pod), and the fire-and-forget
//! surface already proven by the MCP tools:
//!
//! ```text
//! ensure_modl        install modl vX (GH release, or local musl build in dev)
//! modl auth add      HF token, shipped over stdin — never in argv
//! modl pull <model>  once per referenced model (no-op when warm)
//! nohup modl run --run-id X   > ~/.modl/run-logs/X.log
//! modl status X --json        authoritative completion (poll)
//! modl outputs export X       stage artifacts → rsync home
//! ```
//!
//! The run log tail is cosmetic; `modl status --json` polled from the pod's
//! own DB is what decides completion, so a dropped SSH link never corrupts a
//! run — reconnect and keep polling.

use anyhow::{Context, Result, bail};
use console::style;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::core::pod::{
    Pod, REMOTE_ROOT, rsync_from, rsync_to, run_ssh_capture, run_ssh_quiet, run_ssh_stdin,
    run_ssh_streaming, shell_quote,
};

/// Where the modl install lives on the pod (binary + `python/modl_worker`,
/// same layout as the release tarball so the worker resolves next to the
/// binary).
pub(crate) const REMOTE_MODL_DIR: &str = "/root/modl-bin";
pub(crate) const REMOTE_MODL: &str = "/root/modl-bin/modl";
/// How often to ask the pod's DB for the run status.
const STATUS_POLL_SECS: u64 = 15;
/// Consecutive failed status polls tolerated before giving up (flaky links
/// are expected; the run itself is unaffected).
const MAX_POLL_FAILURES: u32 = 20;

pub struct RemoteRunOutcome {
    pub run_id: String,
    /// Aggregate status: completed | partial_failure | cancelled
    pub status: String,
    /// Local directory the run's artifacts were exported into.
    pub local_dir: PathBuf,
    pub artifacts: Vec<PathBuf>,
}

/// Caller-controlled knobs for a remote workflow run. Everything here maps
/// 1:1 onto the pod-side `modl run` invocation so local and pod flags behave
/// identically.
#[derive(Default)]
pub struct RemoteRunOptions {
    /// Use this run id instead of generating a `pod-…` one (`run --run-id`).
    pub run_id: Option<String>,
    /// Pod-side `modl run --force` — regenerate even when matching sidecars
    /// exist in the pod's outputs.
    pub force: bool,
    /// Pod-side `modl run --in-order` — YAML declaration order, no scheduler.
    pub in_order: bool,
}

/// Ensure the pod runs the same modl version as this CLI.
///
/// Fast-path: remote `modl --version` already matches. Otherwise install the
/// matching GH release (musl static, ships the python worker beside the
/// binary). For unreleased dev builds, fall back to uploading a locally-built
/// musl binary + the local worker tree.
pub fn ensure_modl(pod: &Pod) -> Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    if let Ok(v) = run_ssh_capture(&pod.ssh, &format!("{REMOTE_MODL} --version 2>/dev/null"))
        && v.contains(version)
    {
        // Dev builds share a version string across commits, so when a local
        // pod binary exists, verify the actual bytes match — otherwise a
        // reused pod keeps running yesterday's code.
        match local_pod_binary() {
            Err(_) => return Ok(()), // release-installed CLI: version match is enough
            Ok(local) => {
                let local_sha = sha256_file(&local)?;
                let remote_sha = run_ssh_capture(
                    &pod.ssh,
                    &format!("sha256sum {REMOTE_MODL} 2>/dev/null | cut -d' ' -f1"),
                )
                .unwrap_or_default();
                if remote_sha.trim() == local_sha {
                    return Ok(());
                }
                // fall through and reinstall
            }
        }
    }

    let url = format!(
        "https://github.com/modl-org/modl/releases/download/v{version}/modl-v{version}-x86_64-unknown-linux-musl.tar.gz"
    );
    eprintln!("{} Installing modl v{version} on pod...", style("→").cyan());
    let install = format!(
        "set -e; mkdir -p {REMOTE_MODL_DIR}; curl -fsSL {} | tar xz -C {REMOTE_MODL_DIR}",
        shell_quote(&url)
    );
    if run_ssh_streaming(&pod.ssh, &install).is_err() {
        // Dev build with no published asset — upload a locally-built binary.
        let local = local_pod_binary()?;
        eprintln!(
            "{} No release asset for v{version} — uploading local build ({})...",
            style("!").yellow(),
            local.display()
        );
        run_ssh_quiet(&pod.ssh, &format!("mkdir -p {REMOTE_MODL_DIR}/python"))?;
        rsync_to(&pod.ssh, &local, REMOTE_MODL)?;
        rsync_to(
            &pod.ssh,
            &crate::core::training::resolve_worker_python_root()?.join("modl_worker"),
            &format!("{REMOTE_MODL_DIR}/python/"),
        )?;
        run_ssh_quiet(&pod.ssh, &format!("chmod +x {REMOTE_MODL}"))?;
    }

    // 2>&1: the failure detail (e.g. a GLIBC version error from a non-static
    // dev binary on an older-libc image) arrives on stderr.
    let v = run_ssh_capture(&pod.ssh, &format!("{REMOTE_MODL} --version 2>&1"))?;
    if !v.contains(version) {
        bail!(
            "modl on the pod reports '{}' but this CLI is v{version} — install failed?{}",
            v.trim(),
            if v.contains("GLIBC") {
                "\n  The uploaded dev binary is dynamically linked against a newer glibc than \
                 the pod image ships. Build a static one:\n    \
                 cargo build --release --target x86_64-unknown-linux-musl"
            } else {
                ""
            }
        );
    }
    Ok(())
}

/// Dev fallback: a locally-built binary that can run on the pod
/// (x86_64-linux). Preference order: `MODL_POD_BINARY` override, the
/// workspace musl build (static, always works), then the host gnu build —
/// Vast images ship a recent glibc, so a gnu binary from a dev box works,
/// it's just not guaranteed static.
fn local_pod_binary() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("MODL_POD_BINARY") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Ok(p);
        }
        bail!("MODL_POD_BINARY points to missing path: {}", p.display());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let musl = root.join("target/x86_64-unknown-linux-musl/release/modl");
    if musl.exists() {
        return Ok(musl);
    }
    let gnu = root.join("target/release/modl");
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) && gnu.exists() {
        return Ok(gnu);
    }
    bail!(
        "modl v{} has no published release asset and no local x86_64-linux build.\n  \
         Build one: cargo build --release --target x86_64-unknown-linux-musl\n  \
         (or `cargo build --release` on x86_64 Linux, or set MODL_POD_BINARY, \
         or publish the release so pods can download it)",
        env!("CARGO_PKG_VERSION")
    )
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Ship the HF token (stdin → 600-perm file, never argv) and register it in
/// the pod's auth store so `modl pull` can fetch gated models.
fn configure_auth(pod: &Pod) -> Result<()> {
    let Some(token) = crate::core::pod::huggingface_token() else {
        eprintln!(
            "{} No HuggingFace token found — gated models (Flux, Klein) will fail to pull on the pod.",
            style("⚠").yellow()
        );
        return Ok(());
    };
    run_ssh_stdin(
        &pod.ssh,
        &format!("umask 077 && cat > {REMOTE_ROOT}/.hf_token"),
        &token,
    )?;
    // `modl auth add huggingface` reads HF_TOKEN from the environment
    // non-interactively; the shell prefix assignment keeps it out of argv.
    run_ssh_quiet(
        &pod.ssh,
        &format!("HF_TOKEN=\"$(cat {REMOTE_ROOT}/.hf_token)\" {REMOTE_MODL} auth add huggingface"),
    )?;
    Ok(())
}

/// Install modl, register auth, and warm the pod store. Idempotent — safe to
/// call before every run.
pub fn prepare(pod: &Pod, models: &[String]) -> Result<()> {
    ensure_modl(pod)?;
    configure_auth(pod)?;
    pull_models(pod, models)
}

/// `modl pull` each model on the pod. No-ops when already in the pod store;
/// variant selection (fp8/GGUF by VRAM) happens pod-side.
pub fn pull_models(pod: &Pod, models: &[String]) -> Result<()> {
    for m in models {
        pull_model(pod, m, None)?;
    }
    Ok(())
}

/// `modl pull` a single registry item on the pod, optionally pinning a
/// variant (Lightning LoRAs resolve their variant from the `fast:` step
/// count instead of pod VRAM).
pub fn pull_model(pod: &Pod, id: &str, variant: Option<&str>) -> Result<()> {
    eprintln!(
        "{} Ensuring {} on pod...",
        style("→").cyan(),
        style(id).bold()
    );
    let mut cmd = format!("{REMOTE_MODL} pull {}", shell_quote(id));
    if let Some(v) = variant {
        cmd.push_str(&format!(" --variant {}", shell_quote(v)));
    }
    run_ssh_streaming(&pod.ssh, &cmd).with_context(|| format!("`modl pull {id}` failed on the pod"))
}

/// Model IDs referenced by a workflow YAML (workflow-level + per-step
/// overrides). Cheap serde_yaml read — no full parse, no image-ref side
/// effects.
pub fn workflow_models(yaml: &str) -> Result<Vec<String>> {
    let v: serde_yaml::Value = serde_yaml::from_str(yaml).context("Invalid workflow YAML")?;
    let mut models: Vec<String> = Vec::new();
    let mut push = |m: Option<&serde_yaml::Value>| {
        if let Some(s) = m.and_then(|v| v.as_str())
            && !models.iter().any(|x| x == s)
        {
            models.push(s.to_string());
        }
    };
    push(v.get("model"));
    if let Some(steps) = v.get("steps").and_then(|s| s.as_sequence()) {
        for step in steps {
            push(step.get("model"));
        }
    }
    if models.is_empty() {
        bail!("Workflow declares no model (neither top-level nor per-step).");
    }
    Ok(models)
}

/// All `lora:` references in a workflow YAML (workflow-level + per-step),
/// deduplicated, in declaration order.
fn workflow_lora_refs(v: &serde_yaml::Value) -> Vec<String> {
    let mut loras: Vec<String> = Vec::new();
    let mut push = |l: Option<&serde_yaml::Value>| {
        if let Some(s) = l.and_then(|v| v.as_str())
            && !loras.iter().any(|x| x == s)
        {
            loras.push(s.to_string());
        }
    };
    push(v.get("lora"));
    if let Some(steps) = v.get("steps").and_then(|s| s.as_sequence()) {
        for step in steps {
            push(step.get("lora"));
        }
    }
    loras
}

/// A locally-stored LoRA (trained or registered) that must be shipped into
/// the pod's store before the workflow can reference it by name.
#[derive(Debug)]
pub struct LoraPush {
    /// Stable id to register under on the pod (kept identical to the local
    /// id so the same reference resolves on both machines).
    pub id: String,
    pub name: String,
    pub sha256: String,
    pub size: u64,
    pub file_name: String,
    pub local_path: PathBuf,
}

/// How each LoRA referenced by a workflow reaches the pod.
#[derive(Debug, Default)]
pub struct WorkflowLoras {
    /// Registry items — pulled pod-side like models: (id, optional variant).
    /// Includes the Lightning LoRAs required by `fast:` steps.
    pub pulls: Vec<(String, Option<String>)>,
    /// Local LoRAs — rsynced into the pod store and registered.
    pub pushes: Vec<LoraPush>,
}

/// Classify every LoRA a workflow needs (explicit `lora:` refs + the
/// Lightning LoRAs implied by `fast:` steps) into pod-side pulls vs local
/// pushes. Registry items are pulled on the pod (its own bandwidth, correct
/// variant selection); anything only known locally is pushed. Unresolvable
/// references fail here — before any pod time is spent.
pub fn classify_workflow_loras(
    yaml: &str,
    db: &crate::core::db::Database,
) -> Result<WorkflowLoras> {
    let v: serde_yaml::Value = serde_yaml::from_str(yaml).context("Invalid workflow YAML")?;
    let registry_ids: std::collections::HashSet<String> =
        crate::core::registry::RegistryIndex::load()
            .map(|idx| idx.items.iter().map(|m| m.id.clone()).collect())
            .unwrap_or_default();
    let installed_loras = db.list_installed(Some("lora"))?;

    let mut result = WorkflowLoras::default();

    for lora in workflow_lora_refs(&v) {
        if registry_ids.contains(&lora) {
            result.pulls.push((lora, None));
        } else if let Some(m) = installed_loras
            .iter()
            .find(|m| m.name == lora || m.id == lora)
        {
            result.pushes.push(LoraPush {
                id: m.id.clone(),
                name: m.name.clone(),
                sha256: m.sha256.clone(),
                size: m.size,
                file_name: m.file_name.clone(),
                local_path: PathBuf::from(&m.store_path),
            });
        } else {
            bail!(
                "LoRA `{lora}` not found — not installed locally and not a registry item.\n  \
                 Train it (`modl train`), pull it (`modl pull {lora}`), or register a file \
                 (`modl register <file> --name {lora}`)."
            );
        }
    }

    // `fast:` steps need the model's Lightning LoRA on the pod. It's always
    // a registry item — resolve the variant from the requested step count.
    let wf_model = v.get("model").and_then(|m| m.as_str());
    for step in v
        .get("steps")
        .and_then(|s| s.as_sequence())
        .into_iter()
        .flatten()
    {
        let Some(fast) = step.get("fast").and_then(|f| f.as_u64()) else {
            continue;
        };
        let model = step
            .get("model")
            .and_then(|m| m.as_str())
            .or(wf_model)
            .context("`fast` step has no model (neither per-step nor workflow-level)")?;
        let Some(cfg) = crate::core::model_family::lightning_config(model) else {
            let id = step.get("id").and_then(|i| i.as_str()).unwrap_or("?");
            bail!("step `{id}`: `fast` is not supported for `{model}` (no Lightning config).");
        };
        let (variant, _) = cfg.resolve(fast as u32);
        let pull = (cfg.lora_registry_id.clone(), Some(variant.to_string()));
        if !result.pulls.contains(&pull) {
            result.pulls.push(pull);
        }
    }

    Ok(result)
}

/// Ship a local LoRA into the pod's store: rsync the file to its
/// content-addressed path (skipped when already there — content-addressed
/// means existence implies identity), then `modl register` pod-side, which
/// re-hashes (verifying the transfer) and records it in the pod's SQLite +
/// store index. After this, the workflow's `lora: <name>` resolves on the
/// pod exactly like it does locally.
pub fn push_lora(pod: &Pod, lora: &LoraPush) -> Result<()> {
    // Mirror Store::path_for: <store_root>/store/<type>/<sha16>/<file>. Pods
    // run modl as root with stock config, whose store root is ~/modl. If a
    // pod ever diverges, `modl register` below copies the file into the
    // right place — this path is just the zero-copy fast path.
    let prefix = &lora.sha256[..lora.sha256.len().min(16)];
    let remote_dir = format!("/root/modl/store/lora/{prefix}");
    let remote_path = format!("{remote_dir}/{}", lora.file_name);

    let exists = run_ssh_capture(
        &pod.ssh,
        &format!(
            "test -f {} && echo yes || echo no",
            shell_quote(&remote_path)
        ),
    )
    .unwrap_or_default();
    if !exists.contains("yes") {
        eprintln!(
            "{} Pushing LoRA {} to pod ({:.1} MB)...",
            style("→").cyan(),
            style(&lora.name).bold(),
            lora.size as f64 / 1_048_576.0
        );
        run_ssh_quiet(&pod.ssh, &format!("mkdir -p {}", shell_quote(&remote_dir)))?;
        rsync_to(&pod.ssh, &lora.local_path, &remote_path)
            .with_context(|| format!("Failed to push LoRA {} to the pod", lora.name))?;
    }

    let mut cmd = format!(
        "{REMOTE_MODL} register {} --name {} --id {} --type lora",
        shell_quote(&remote_path),
        shell_quote(&lora.name),
        shell_quote(&lora.id),
    );
    // Only a full hash can be verified — reconciled entries may carry a
    // 16-char prefix, which `register` would reject as a mismatch.
    if lora.sha256.len() == 64 {
        cmd.push_str(&format!(" --sha256 {}", shell_quote(&lora.sha256)));
    }
    run_ssh_streaming(&pod.ssh, &cmd)
        .with_context(|| format!("`modl register` failed on the pod for LoRA {}", lora.name))?;
    Ok(())
}

/// Reject specs the pod can't satisfy yet, with a useful message.
pub fn check_pod_supported(yaml: &str) -> Result<()> {
    let v: serde_yaml::Value = serde_yaml::from_str(yaml).context("Invalid workflow YAML")?;
    let steps = v.get("steps").and_then(|s| s.as_sequence());

    // LoRA refs by name resolve pod-side after push/pull. A file path names
    // a file on THIS machine that won't exist there — catch it at submit
    // time with the fix.
    for lora in workflow_lora_refs(&v) {
        if lora.contains('/') || lora.ends_with(".safetensors") {
            bail!(
                "`lora: {lora}` is a file path — it won't exist on the pod.\n  \
                 Register it locally first:\n    \
                 modl register {lora} --name <name>\n  \
                 then reference it as `lora: <name>`."
            );
        }
    }

    // Image refs must resolve on the pod: `$step.outputs[N]` and `$name`
    // variables do, data URIs are decoded pod-side, but a bare path names a
    // file on THIS machine that won't exist there — the run would only fail
    // minutes later, pod-side, with a confusing message.
    if let Some(images) = v.get("images").and_then(|m| m.as_mapping()) {
        for (name, value) in images {
            let (Some(name), Some(value)) = (name.as_str(), value.as_str()) else {
                continue;
            };
            if !value.starts_with("data:image/") {
                bail!(
                    "images.{name} is a file path — it won't exist on the pod.\n  \
                     Inline it as a base64 data URI:\n    \
                     {name}: \"data:image/png;base64,...\"  (base64 -w0 <file>, or `base64 -i <file>` on macOS)"
                );
            }
        }
    }
    for step in steps.into_iter().flatten() {
        let id = step.get("id").and_then(|i| i.as_str()).unwrap_or("?");
        // (field label, ref string) pairs across every image slot.
        let mut refs: Vec<(&str, &str)> = Vec::new();
        match step.get("edit") {
            Some(serde_yaml::Value::String(s)) => refs.push(("edit source", s)),
            Some(serde_yaml::Value::Sequence(seq)) => {
                refs.extend(
                    seq.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| ("edit source", s)),
                );
            }
            _ => {}
        }
        if let Some(s) = step.get("init_image").and_then(|v| v.as_str()) {
            refs.push(("init_image", s));
        }
        if let Some(s) = step.get("mask").and_then(|v| v.as_str())
            && s != "auto"
            && s != "from-alpha"
        {
            refs.push(("mask", s));
        }
        for key in ["controlnet", "style_ref"] {
            for entry in step
                .get(key)
                .and_then(|v| v.as_sequence())
                .into_iter()
                .flatten()
            {
                if let Some(s) = entry.get("image").and_then(|v| v.as_str()) {
                    refs.push((key, s));
                }
            }
        }
        for (what, src) in refs {
            if !src.starts_with('$') && !src.starts_with("data:image/") {
                bail!(
                    "step `{id}`: {what} `{src}` is a local file path — it won't exist on the pod.\n  \
                     Add it to the `images:` map as a base64 data URI and reference it as `$name`."
                );
            }
        }
    }

    // The checks above are pod-specific messaging; the pod runs the real
    // parser, so run it here too — schema errors, unknown `$name`/`$step`
    // refs, and undecodable base64 must fail at submit time, not minutes
    // later on the pod after install/auth/model-pull time is already paid.
    // Everything left is data URIs and `$refs` (bare paths bailed above), so
    // data URIs materialize into a throwaway dir and a parse success here is
    // a parse success on the pod.
    let tmp = tempfile::tempdir().context("Failed to create a scratch dir for validation")?;
    crate::core::workflow::parse_str(yaml, tmp.path())?;
    Ok(())
}

/// Run a workflow YAML on the pod end-to-end: install modl, pull models,
/// fire-and-forget `modl run`, poll to completion, export artifacts home.
pub async fn run_workflow_on_pod(
    pod: &Pod,
    spec_yaml: &str,
    local_dest: Option<&Path>,
    opts: RemoteRunOptions,
) -> Result<RemoteRunOutcome> {
    check_pod_supported(spec_yaml)?;
    let models = workflow_models(spec_yaml)?;
    // Resolve LoRA references against the local DB *before* touching the pod
    // so unresolvable refs fail without spending pod time.
    let loras = {
        let db = crate::core::db::Database::open()?;
        classify_workflow_loras(spec_yaml, &db)?
    };
    prepare(pod, &models)?;
    for (id, variant) in &loras.pulls {
        pull_model(pod, id, variant.as_deref())?;
    }
    for lora in &loras.pushes {
        push_lora(pod, lora)?;
    }

    let run_id = opts.run_id.unwrap_or_else(|| {
        format!(
            "pod-{}-{:04x}",
            chrono::Local::now().format("%Y%m%d-%H%M%S"),
            std::process::id() & 0xffff
        )
    });

    // Ship the spec.
    let remote_spec = format!("{REMOTE_ROOT}/runs/{run_id}.yaml");
    run_ssh_quiet(&pod.ssh, &format!("mkdir -p {REMOTE_ROOT}/runs"))?;
    run_ssh_stdin(
        &pod.ssh,
        &format!("cat > {}", shell_quote(&remote_spec)),
        spec_yaml,
    )?;

    // Fire and forget — the run survives a dropped link or a closed laptop.
    let log = format!("{REMOTE_ROOT}/runs/{run_id}.log");
    let pidfile = format!("{REMOTE_ROOT}/runs/{run_id}.pid");
    let mut run_flags = String::new();
    if opts.force {
        run_flags.push_str(" --force");
    }
    if opts.in_order {
        run_flags.push_str(" --in-order");
    }
    let launch = format!(
        "nohup {REMOTE_MODL} run {} --run-id {}{run_flags} > {} 2>&1 & echo $! > {}; echo ok",
        shell_quote(&remote_spec),
        shell_quote(&run_id),
        shell_quote(&log),
        shell_quote(&pidfile),
    );
    run_ssh_quiet(&pod.ssh, &launch)?;
    eprintln!(
        "{} Workflow running on pod {} — {}",
        style("→").cyan(),
        pod.instance_id,
        style(&run_id).dim()
    );

    let status = wait_for_run(pod, &run_id, &log, &pidfile)?;
    if status != "completed" && status != "partial_failure" {
        // cancelled (or anything unexpected): nothing worth exporting.
        bail!("Pod run {run_id} ended with status: {status}");
    }
    if status == "partial_failure" {
        eprintln!(
            "{} Some steps failed — exporting the artifacts that did complete.",
            style("⚠").yellow()
        );
    }

    let (local_dir, artifacts) = export_run(pod, &run_id, local_dest)?;

    Ok(RemoteRunOutcome {
        run_id,
        status,
        local_dir,
        artifacts,
    })
}

/// Export a finished run's artifacts on the pod and rsync them home
/// (directly over SSH — no intermediary storage). Also the re-attach path:
/// `modl pod pull <run-id>` fetches a run whose watcher died with the
/// laptop lid.
///
/// The pod-side export carries each image's sidecar YAML. With an explicit
/// `local_dest` the files land there verbatim; without one they are imported
/// into `~/.modl/outputs/<date>/` and registered from their sidecars, so pod
/// results show up in `modl outputs` and the web UI like local generations.
pub fn export_run(
    pod: &Pod,
    run_id: &str,
    local_dest: Option<&Path>,
) -> Result<(PathBuf, Vec<PathBuf>)> {
    let remote_export = format!("{REMOTE_ROOT}/export/{run_id}");
    run_ssh_streaming(
        &pod.ssh,
        &format!(
            "{REMOTE_MODL} outputs export {} --dest {}",
            shell_quote(run_id),
            shell_quote(&remote_export)
        ),
    )
    .context("Artifact export failed on the pod")?;

    // Explicit destination: sync verbatim (images + sidecars), no DB writes.
    if let Some(dest) = local_dest {
        std::fs::create_dir_all(dest)
            .with_context(|| format!("Failed to create {}", dest.display()))?;
        rsync_from(&pod.ssh, &format!("{remote_export}/"), dest)
            .context("Failed to sync artifacts back from the pod")?;
        let mut artifacts = walk_files(dest);
        artifacts.sort();
        if artifacts.is_empty() {
            bail!(
                "Run {run_id} exported nothing — inspect the pod: \
                 modl pod exec -- {REMOTE_MODL} status {run_id}"
            );
        }
        eprintln!(
            "{} {} file(s) → {}",
            style("✓").green().bold(),
            artifacts.len(),
            dest.display()
        );
        for a in &artifacts {
            eprintln!("  {}", a.display());
        }
        return Ok((dest.to_path_buf(), artifacts));
    }

    // Default: stage locally, then import into the outputs tree + register
    // from sidecars — SQLite stays a rebuildable cache over the files.
    let staging = tempfile::tempdir().context("Failed to create staging directory")?;
    rsync_from(&pod.ssh, &format!("{remote_export}/"), staging.path())
        .context("Failed to sync artifacts back from the pod")?;

    let db = crate::core::db::Database::open()?;
    let artifacts = crate::core::outputs::import_run_dir(staging.path(), &db)?;
    if artifacts.is_empty() {
        bail!(
            "Run {run_id} exported nothing — inspect the pod: \
             modl pod exec -- {REMOTE_MODL} status {run_id}"
        );
    }

    eprintln!(
        "{} {} artifact(s) imported into ~/.modl/outputs (visible in `modl outputs` and the UI)",
        style("✓").green().bold(),
        artifacts.len(),
    );
    for a in &artifacts {
        eprintln!("  {}", a.display());
    }
    let local_dir = crate::core::paths::modl_root().join("outputs");
    Ok((local_dir, artifacts))
}

/// Poll `modl status --json` on the pod until the run reaches a terminal
/// state, streaming the run log for feedback. The status poll is
/// authoritative; the log tail is cosmetic and may drop/reconnect freely.
fn wait_for_run(pod: &Pod, run_id: &str, log: &str, pidfile: &str) -> Result<String> {
    let mut tail = spawn_log_tail(pod, log);
    let mut poll_failures = 0u32;
    // Liveness probe for the detached runner. `run_ssh_capture` returns Ok("")
    // on transport failure, which matches neither "alive" nor "dead" — so a
    // flaky link never masquerades as a verdict.
    let probe_cmd = format!(
        "if [ -f {pf} ] && kill -0 \"$(cat {pf})\" 2>/dev/null; \
         then echo alive; else echo dead; fi",
        pf = shell_quote(pidfile)
    );
    let dead_runner_err = |status: &str| -> anyhow::Error {
        let tail_txt = run_ssh_capture(&pod.ssh, &format!("tail -n 30 {}", shell_quote(log)))
            .unwrap_or_default();
        anyhow::anyhow!(
            "The workflow runner on the pod died before finishing (status: {status}).\n\nLast run log:\n{tail_txt}"
        )
    };

    loop {
        // Drain whatever the tail has produced since the last poll.
        if let Some((_, rx)) = &tail {
            while let Ok(line) = rx.try_recv() {
                eprintln!("  {}", style(line.trim_end()).dim());
            }
            // Tail children die with flaky links — respawn lazily.
            if let Some((child, _)) = &mut tail
                && child.try_wait().ok().flatten().is_some()
            {
                tail = spawn_log_tail(pod, log);
            }
        }

        match run_status(pod, run_id) {
            Ok(status) => {
                poll_failures = 0;
                if status == "completed" || status == "partial_failure" || status == "cancelled" {
                    if let Some((mut child, rx)) = tail.take() {
                        // Print any final log lines before killing the tail.
                        while let Ok(line) = rx.try_recv() {
                            eprintln!("  {}", style(line.trim_end()).dim());
                        }
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    return Ok(status);
                }
                // Not terminal — if the runner process died the DB will never
                // reach a terminal state; catch that instead of hanging.
                if (status == "running" || status == "pending")
                    && let Ok(out) = run_ssh_capture(&pod.ssh, &probe_cmd)
                    && out.contains("dead")
                {
                    return Err(dead_runner_err(&format!("still '{status}'")));
                }
            }
            Err(e) => {
                // A failed status poll is either a flaky link (run unaffected,
                // keep trying) or a runner that died before the run was ever
                // registered in the pod's DB — a state the status query alone
                // can't distinguish, and one that would otherwise burn the
                // full poll budget before bailing with a wrong "lost contact"
                // diagnosis. Probe the runner: a reachable pod with a dead
                // pid means the run is never coming.
                if let Ok(out) = run_ssh_capture(&pod.ssh, &probe_cmd)
                    && out.contains("dead")
                {
                    return Err(dead_runner_err(
                        "never registered — it likely failed at startup",
                    ));
                }
                poll_failures += 1;
                if poll_failures >= MAX_POLL_FAILURES {
                    bail!(
                        "Lost contact with the pod for {} consecutive polls — the run may still be going.\n  \
                         Check later: modl pod exec -- {REMOTE_MODL} status {run_id} --json\n  ({e})",
                        poll_failures
                    );
                }
                if poll_failures == 1 {
                    eprintln!(
                        "{} Status poll failed — retrying (run continues on the pod)...",
                        style("!").yellow()
                    );
                }
            }
        }

        std::thread::sleep(std::time::Duration::from_secs(STATUS_POLL_SECS));
    }
}

pub(crate) fn run_status(pod: &Pod, run_id: &str) -> Result<String> {
    let v = run_status_json(pod, run_id)?;
    v.get("status")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string())
        .context("modl status JSON missing 'status'")
}

/// The pod's own `modl status --json` for a run, verbatim. Pod runs live in
/// the pod's DB — the local DB only learns about them when artifacts are
/// pulled home — so status queries for in-flight runs must ask the pod.
pub fn run_status_json(pod: &Pod, run_id: &str) -> Result<serde_json::Value> {
    let out = run_ssh_capture(
        &pod.ssh,
        &format!("{REMOTE_MODL} status {} --json", shell_quote(run_id)),
    )?;
    serde_json::from_str(out.trim()).with_context(|| {
        format!(
            "Pod did not return status JSON for run {run_id} — the run may not exist there{}",
            if out.trim().is_empty() {
                String::new()
            } else {
                format!(" (got: {})", out.trim())
            }
        )
    })
}

/// Best-effort `tail -F` of the remote run log through a reader thread.
/// Returns None when the tail can't be spawned — polling still works.
#[allow(clippy::type_complexity)]
fn spawn_log_tail(
    pod: &Pod,
    log: &str,
) -> Option<(std::process::Child, std::sync::mpsc::Receiver<String>)> {
    let mut args = pod.ssh.base_args();
    args.push(format!(
        "touch {log} && tail -n +1 -F {log}",
        log = shell_quote(log)
    ));
    let mut child = Command::new("ssh")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    Some((child, rx))
}

fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk_files(&path));
        } else {
            found.push(path);
        }
    }
    found
}

/// Read a local image file and encode it as a `data:image/...;base64,...` URI
/// (the MCP-safe image-ref form — client paths don't exist on pods).
fn image_data_uri(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    };
    use base64::Engine as _;
    Ok(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

fn insert_seeds(step: &mut serde_yaml::Mapping, seeds: &[u64]) {
    match seeds {
        [] => {}
        [one] => {
            step.insert("seed".into(), (*one).into());
        }
        many => {
            step.insert(
                "seeds".into(),
                serde_yaml::Value::Sequence(many.iter().map(|s| (*s).into()).collect()),
            );
        }
    }
}

/// Image inputs for a `generate --pod` spec. All paths are local files that
/// get inlined into the workflow's `images:` map as base64 data URIs.
#[derive(Default)]
pub struct GenImageInputs<'a> {
    pub init_image: Option<&'a Path>,
    pub mask: Option<&'a Path>,
    pub strength: Option<f32>,
    /// Fully-resolved ControlNet inputs (`image` is a local path).
    pub controlnet: &'a [crate::core::job::ControlNetInput],
    /// Fully-resolved style-ref inputs (`image` is a local path).
    pub style_ref: &'a [crate::core::job::StyleRefInput],
}

/// Build a one-step generate workflow so `modl generate --pod` funnels
/// through the same remote surface as `modl run --pod`.
#[allow(clippy::too_many_arguments)]
pub fn single_generate_spec(
    model: &str,
    prompt: &str,
    lora: Option<&str>,
    fast: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seeds: &[u64],
    image_inputs: &GenImageInputs<'_>,
) -> Result<String> {
    let mut images = serde_yaml::Mapping::new();
    let mut inline = |name: &str, path: &Path| -> Result<()> {
        images.insert(name.into(), image_data_uri(path)?.into());
        Ok(())
    };

    let mut step = serde_yaml::Mapping::new();
    step.insert("id".into(), "gen".into());
    step.insert("generate".into(), prompt.into());
    if let Some(p) = image_inputs.init_image {
        inline("init", p)?;
        step.insert("init_image".into(), "$init".into());
    }
    if let Some(p) = image_inputs.mask {
        inline("mask", p)?;
        step.insert("mask".into(), "$mask".into());
    }
    insert_opt(
        &mut step,
        "strength",
        image_inputs.strength.map(|v| f64::from(v).into()),
    );
    if !image_inputs.controlnet.is_empty() {
        let mut entries = Vec::with_capacity(image_inputs.controlnet.len());
        for (i, cn) in image_inputs.controlnet.iter().enumerate() {
            let name = format!("cn-{}", i + 1);
            inline(&name, Path::new(&cn.image))?;
            let mut entry = serde_yaml::Mapping::new();
            entry.insert("image".into(), format!("${name}").into());
            entry.insert("type".into(), cn.control_type.as_str().into());
            entry.insert("strength".into(), f64::from(cn.strength).into());
            entry.insert("end".into(), f64::from(cn.control_end).into());
            entries.push(serde_yaml::Value::Mapping(entry));
        }
        step.insert("controlnet".into(), serde_yaml::Value::Sequence(entries));
    }
    if !image_inputs.style_ref.is_empty() {
        let mut entries = Vec::with_capacity(image_inputs.style_ref.len());
        for (i, sr) in image_inputs.style_ref.iter().enumerate() {
            let name = format!("style-{}", i + 1);
            inline(&name, Path::new(&sr.image))?;
            let mut entry = serde_yaml::Mapping::new();
            entry.insert("image".into(), format!("${name}").into());
            entry.insert("strength".into(), f64::from(sr.strength).into());
            entry.insert("type".into(), sr.style_type.as_str().into());
            entries.push(serde_yaml::Value::Mapping(entry));
        }
        step.insert("style_ref".into(), serde_yaml::Value::Sequence(entries));
    }
    insert_opt(&mut step, "lora", lora.map(|v| v.into()));
    insert_opt(&mut step, "fast", fast.map(|v| v.into()));
    insert_opt(&mut step, "width", width.map(|v| v.into()));
    insert_opt(&mut step, "height", height.map(|v| v.into()));
    insert_opt(&mut step, "steps", steps.map(|v| v.into()));
    insert_opt(&mut step, "guidance", guidance.map(|v| v.into()));
    insert_seeds(&mut step, seeds);

    let mut root = serde_yaml::Mapping::new();
    root.insert("name".into(), "pod-generate".into());
    root.insert("model".into(), model.into());
    if !images.is_empty() {
        root.insert("images".into(), serde_yaml::Value::Mapping(images));
    }
    root.insert(
        "steps".into(),
        serde_yaml::Value::Sequence(vec![serde_yaml::Value::Mapping(step)]),
    );
    serde_yaml::to_string(&serde_yaml::Value::Mapping(root)).context("Failed to build spec")
}

/// Build a one-step edit workflow with every input image inlined as a base64
/// data URI. `mask` is a local path, or the `auto`/`from-alpha` sentinels
/// which pass through as-is.
#[allow(clippy::too_many_arguments)]
pub fn single_edit_spec(
    model: &str,
    prompt: &str,
    image_paths: &[PathBuf],
    mask: Option<&str>,
    blend: Option<crate::core::job::BlendMode>,
    lora: Option<&str>,
    fast: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seeds: &[u64],
) -> Result<String> {
    let mut images = serde_yaml::Mapping::new();

    // Single image keeps the historical `$input` name; multi-image edits get
    // `$input-1`, `$input-2`, … in order.
    let source_refs: Vec<String> = match image_paths {
        [one] => {
            images.insert("input".into(), image_data_uri(one)?.into());
            vec!["$input".to_string()]
        }
        many => many
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let name = format!("input-{}", i + 1);
                images.insert(name.as_str().into(), image_data_uri(p)?.into());
                Ok(format!("${name}"))
            })
            .collect::<Result<_>>()?,
    };

    let mut step = serde_yaml::Mapping::new();
    step.insert("id".into(), "edit".into());
    match source_refs.as_slice() {
        [one] => {
            step.insert("edit".into(), one.as_str().into());
        }
        many => {
            step.insert(
                "edit".into(),
                serde_yaml::Value::Sequence(many.iter().map(|s| s.as_str().into()).collect()),
            );
        }
    }
    step.insert("prompt".into(), prompt.into());
    match mask {
        None => {}
        Some(sentinel @ ("auto" | "from-alpha")) => {
            step.insert("mask".into(), sentinel.into());
        }
        Some(path) => {
            images.insert("mask".into(), image_data_uri(Path::new(path))?.into());
            step.insert("mask".into(), "$mask".into());
        }
    }
    if let Some(b) = blend
        && b != crate::core::job::BlendMode::Pixel
    {
        step.insert("blend".into(), "latent".into());
    }
    insert_opt(&mut step, "lora", lora.map(|v| v.into()));
    insert_opt(&mut step, "fast", fast.map(|v| v.into()));
    insert_opt(&mut step, "width", width.map(|v| v.into()));
    insert_opt(&mut step, "height", height.map(|v| v.into()));
    insert_opt(&mut step, "steps", steps.map(|v| v.into()));
    insert_opt(&mut step, "guidance", guidance.map(|v| v.into()));
    insert_seeds(&mut step, seeds);

    let mut root = serde_yaml::Mapping::new();
    root.insert("name".into(), "pod-edit".into());
    root.insert("model".into(), model.into());
    root.insert("images".into(), serde_yaml::Value::Mapping(images));
    root.insert(
        "steps".into(),
        serde_yaml::Value::Sequence(vec![serde_yaml::Value::Mapping(step)]),
    );
    serde_yaml::to_string(&serde_yaml::Value::Mapping(root)).context("Failed to build spec")
}

fn insert_opt(map: &mut serde_yaml::Mapping, key: &str, val: Option<serde_yaml::Value>) {
    if let Some(v) = val {
        map.insert(key.into(), v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_models_from_workflow_and_steps() {
        let yaml = r#"
name: t
model: flux-schnell
steps:
  - id: a
    generate: "x"
  - id: b
    generate: "y"
    model: z-image
  - id: c
    generate: "z"
    model: flux-schnell
"#;
        let models = workflow_models(yaml).unwrap();
        assert_eq!(models, vec!["flux-schnell".to_string(), "z-image".into()]);
    }

    #[test]
    fn lora_names_supported_but_path_refs_rejected() {
        // Name-form LoRA refs are fine — they're pushed/pulled before the run.
        let by_name = "name: t\nmodel: m\nlora: my-lora\nsteps:\n  - id: a\n    generate: x\n";
        assert!(check_pod_supported(by_name).is_ok());
        let per_step = "name: t\nmodel: m\nsteps:\n  - id: a\n    generate: x\n    lora: l\n";
        assert!(check_pod_supported(per_step).is_ok());

        // Path-form refs name files on this machine — rejected with the fix.
        let by_path =
            "name: t\nmodel: m\nlora: /home/me/x.safetensors\nsteps:\n  - id: a\n    generate: x\n";
        let err = check_pod_supported(by_path).unwrap_err().to_string();
        assert!(err.contains("modl register"), "{err}");
        let step_path =
            "name: t\nmodel: m\nsteps:\n  - id: a\n    generate: x\n    lora: x.safetensors\n";
        assert!(check_pod_supported(step_path).is_err());
    }

    #[test]
    fn extracts_lora_refs_dedup() {
        let yaml = r#"
name: t
model: m
lora: alice
steps:
  - id: a
    generate: x
  - id: b
    generate: y
    lora: bob
  - id: c
    generate: z
    lora: alice
"#;
        let v: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            workflow_lora_refs(&v),
            vec!["alice".to_string(), "bob".into()]
        );
    }

    #[test]
    fn classifies_local_lora_as_push_and_unknown_as_error() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db = crate::core::db::Database::open_at(tmp.path()).unwrap();
        db.insert_installed(&crate::core::db::InstalledModelRecord {
            id: "train:alice:abcd",
            name: "alice",
            asset_type: "lora",
            variant: None,
            sha256: &"a".repeat(64),
            size: 42,
            file_name: "alice.safetensors",
            store_path: "/store/lora/aaaa/alice.safetensors",
        })
        .unwrap();

        let yaml = "name: t\nmodel: m\nlora: alice\nsteps:\n  - id: a\n    generate: x\n";
        let loras = classify_workflow_loras(yaml, &db).unwrap();
        assert!(loras.pulls.is_empty());
        assert_eq!(loras.pushes.len(), 1);
        assert_eq!(loras.pushes[0].id, "train:alice:abcd");
        assert_eq!(loras.pushes[0].file_name, "alice.safetensors");

        let unknown = "name: t\nmodel: m\nlora: nope\nsteps:\n  - id: a\n    generate: x\n";
        let err = classify_workflow_loras(unknown, &db)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn fast_step_adds_lightning_pull() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db = crate::core::db::Database::open_at(tmp.path()).unwrap();

        // qwen-image has a Lightning config in models.toml.
        let yaml = "name: t\nmodel: qwen-image\nsteps:\n  - id: a\n    generate: x\n    fast: 4\n";
        let loras = classify_workflow_loras(yaml, &db).unwrap();
        assert_eq!(loras.pulls.len(), 1);
        assert_eq!(loras.pulls[0].0, "qwen-image-2512-lightning");
        assert_eq!(loras.pulls[0].1.as_deref(), Some("4step-bf16"));

        // A model without a Lightning config fails at classification time.
        let bad = "name: t\nmodel: sdxl\nsteps:\n  - id: a\n    generate: x\n    fast: 4\n";
        let err = classify_workflow_loras(bad, &db).unwrap_err().to_string();
        assert!(err.contains("not supported"), "{err}");
    }

    #[test]
    fn generate_spec_roundtrips() {
        let spec = single_generate_spec(
            "flux-schnell",
            "a red apple",
            None,
            None,
            Some(1024),
            None,
            Some(4),
            None,
            &[7],
            &Default::default(),
        )
        .unwrap();
        let models = workflow_models(&spec).unwrap();
        assert_eq!(models, vec!["flux-schnell".to_string()]);
        assert!(spec.contains("a red apple"));
        assert!(spec.contains("width: 1024"));
        assert!(spec.contains("seed: 7"));
        assert!(!spec.contains("height"));
        assert!(!spec.contains("lora"));
        assert!(!spec.contains("fast"));
    }

    #[test]
    fn generate_spec_carries_lora_and_fast() {
        let spec = single_generate_spec(
            "qwen-image",
            "p",
            Some("alice"),
            None,
            None,
            None,
            None,
            None,
            &[],
            &Default::default(),
        )
        .unwrap();
        assert!(spec.contains("lora: alice"));

        let spec = single_generate_spec(
            "qwen-image",
            "p",
            None,
            Some(4),
            None,
            None,
            None,
            None,
            &[],
            &Default::default(),
        )
        .unwrap();
        assert!(spec.contains("fast: 4"));
    }

    #[test]
    fn edit_spec_inlines_image_as_data_uri() {
        let tmp = std::env::temp_dir().join(format!("modl-podrun-test-{}.png", std::process::id()));
        std::fs::write(&tmp, b"fakepng").unwrap();
        let spec = single_edit_spec(
            "klein-9b",
            "make it golden",
            std::slice::from_ref(&tmp),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &[],
        )
        .unwrap();
        std::fs::remove_file(&tmp).unwrap();
        assert!(spec.contains("data:image/png;base64,"));
        assert!(spec.contains("$input"));
        assert!(spec.contains("make it golden"));
        // No optional params passed — none serialized (`steps:` here is the
        // workflow's step list, not an inference-steps field).
        assert!(!spec.contains("guidance"));
        assert!(!spec.contains("seed"));
        assert!(!spec.contains("width"));
    }

    #[test]
    fn edit_spec_carries_params_and_seeds() {
        let tmp =
            std::env::temp_dir().join(format!("modl-podrun-test2-{}.png", std::process::id()));
        std::fs::write(&tmp, b"fakepng").unwrap();
        let spec = single_edit_spec(
            "klein-9b",
            "p",
            std::slice::from_ref(&tmp),
            None,
            None,
            None,
            None,
            Some(1820),
            Some(1024),
            Some(28),
            Some(4.5),
            &[7, 8],
        )
        .unwrap();
        std::fs::remove_file(&tmp).unwrap();
        assert!(spec.contains("width: 1820"));
        assert!(spec.contains("height: 1024"));
        assert!(spec.contains("steps: 28"));
        assert!(spec.contains("guidance: 4.5"));
        assert!(spec.contains("seeds:"));
        assert!(spec.contains("- 7"));
        assert!(spec.contains("- 8"));
    }

    #[test]
    fn generate_spec_inlines_image_inputs() {
        let dir = std::env::temp_dir().join(format!("modl-podrun-gen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let init = dir.join("init.png");
        let mask = dir.join("mask.png");
        let pose = dir.join("ref_pose.png");
        for p in [&init, &mask, &pose] {
            std::fs::write(p, b"fakepng").unwrap();
        }
        let cn = [crate::core::job::ControlNetInput {
            image: pose.to_string_lossy().to_string(),
            control_type: "pose".to_string(),
            strength: 0.9,
            control_end: 0.7,
        }];
        let inputs = GenImageInputs {
            init_image: Some(&init),
            mask: Some(&mask),
            strength: Some(0.6),
            controlnet: &cn,
            style_ref: &[],
        };
        let spec = single_generate_spec(
            "flux-dev",
            "p",
            None,
            None,
            None,
            None,
            None,
            None,
            &[],
            &inputs,
        )
        .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(spec.contains("init: data:image/png;base64,"));
        assert!(spec.contains("init_image: $init"));
        assert!(spec.contains("mask: $mask"));
        assert!(spec.contains("strength: 0.6"));
        assert!(spec.contains("cn-1: data:image/png;base64,"));
        assert!(spec.contains("image: $cn-1"));
        assert!(spec.contains("type: pose"));
        // The whole spec must pass the pod-support check (no bare paths).
        check_pod_supported(&spec).unwrap();
    }

    #[test]
    fn edit_spec_inlines_multiple_images_and_mask() {
        let dir = std::env::temp_dir().join(format!("modl-podrun-edit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.png");
        let b = dir.join("b.jpg");
        let m = dir.join("m.png");
        for p in [&a, &b, &m] {
            std::fs::write(p, b"fakepng").unwrap();
        }
        let spec = single_edit_spec(
            "klein-9b",
            "blend them",
            &[a.clone(), b.clone()],
            Some(m.to_string_lossy().as_ref()),
            Some(crate::core::job::BlendMode::Latent),
            None,
            None,
            None,
            None,
            None,
            None,
            &[],
        )
        .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(spec.contains("input-1: data:image/png;base64,"));
        assert!(spec.contains("input-2: data:image/jpeg;base64,"));
        assert!(spec.contains("- $input-1"));
        assert!(spec.contains("- $input-2"));
        assert!(spec.contains("mask: $mask"));
        assert!(spec.contains("blend: latent"));
        check_pod_supported(&spec).unwrap();
    }

    #[test]
    fn edit_spec_passes_mask_sentinel_through() {
        let tmp =
            std::env::temp_dir().join(format!("modl-podrun-test3-{}.png", std::process::id()));
        std::fs::write(&tmp, b"fakepng").unwrap();
        let spec = single_edit_spec(
            "klein-9b",
            "p",
            std::slice::from_ref(&tmp),
            Some("from-alpha"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &[],
        )
        .unwrap();
        std::fs::remove_file(&tmp).unwrap();
        assert!(spec.contains("mask: from-alpha"));
        check_pod_supported(&spec).unwrap();
    }

    #[test]
    fn rejects_bare_paths_in_new_image_slots() {
        for yaml in [
            "name: t\nmodel: m\nsteps:\n  - id: a\n    generate: p\n    init_image: photo.png\n",
            "name: t\nmodel: m\nsteps:\n  - id: a\n    generate: p\n    init_image: $x\n    mask: m.png\n",
            "name: t\nmodel: m\nsteps:\n  - id: a\n    generate: p\n    controlnet:\n      - image: pose.png\n        type: pose\n",
            "name: t\nmodel: m\nsteps:\n  - id: a\n    edit: [\"$x\", \"b.png\"]\n    prompt: p\n",
        ] {
            let err = check_pod_supported(yaml).unwrap_err().to_string();
            assert!(err.contains("local file path"), "{yaml}: {err}");
        }
        // Sentinel masks are fine.
        let ok = "name: t\nmodel: m\nimages:\n  x: \"data:image/png;base64,aWs=\"\nsteps:\n  - id: a\n    edit: \"$x\"\n    prompt: p\n    mask: from-alpha\n";
        check_pod_supported(ok).unwrap();
    }

    #[test]
    fn runs_the_full_parser_at_submit_time() {
        // An undefined $name passed the path-only checks before — now the
        // real parser runs locally, so it fails at submit instead of minutes
        // later pod-side.
        let undefined =
            "name: t\nmodel: m\nsteps:\n  - id: a\n    edit: \"$nope\"\n    prompt: p\n";
        let err = check_pod_supported(undefined).unwrap_err().to_string();
        assert!(err.contains("not defined"), "{err}");

        // Undecodable base64 in the images map fails the same way.
        let bad_b64 = "name: t\nmodel: m\nimages:\n  x: \"data:image/png;base64,%%%\"\nsteps:\n  - id: a\n    edit: \"$x\"\n    prompt: p\n";
        assert!(check_pod_supported(bad_b64).is_err());

        // A $step ref into a step id that doesn't exist is caught too.
        let bad_step = "name: t\nmodel: m\nsteps:\n  - id: a\n    edit: \"$ghost.outputs[0]\"\n    prompt: p\n";
        assert!(check_pod_supported(bad_step).is_err());
    }

    #[test]
    fn rejects_bare_local_image_paths() {
        // Bare path in an edit step — would resolve pod-side and fail there.
        let step_path =
            "name: t\nmodel: m\nsteps:\n  - id: a\n    edit: photo.png\n    prompt: p\n";
        let err = check_pod_supported(step_path).unwrap_err().to_string();
        assert!(err.contains("local file path"), "{err}");

        // File path in the images: map — same problem.
        let images_path = "name: t\nmodel: m\nimages:\n  ref: /home/me/ref.png\nsteps:\n  - id: a\n    generate: x\n";
        let err = check_pod_supported(images_path).unwrap_err().to_string();
        assert!(err.contains("file path"), "{err}");

        // Data URIs, $variables and $step refs are all fine.
        let ok = "name: t\nmodel: m\nimages:\n  ref: \"data:image/png;base64,aWs=\"\nsteps:\n  - id: a\n    generate: x\n  - id: b\n    edit: \"$ref\"\n    prompt: p\n  - id: c\n    edit: \"$a.outputs[0]\"\n    prompt: p\n";
        assert!(check_pod_supported(ok).is_ok());
    }
}
