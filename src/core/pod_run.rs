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

    let v = run_ssh_capture(&pod.ssh, &format!("{REMOTE_MODL} --version"))?;
    if !v.contains(version) {
        bail!(
            "modl on the pod reports '{}' but this CLI is v{version} — install failed?",
            v.trim()
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
        eprintln!(
            "{} Ensuring {} on pod...",
            style("→").cyan(),
            style(m).bold()
        );
        run_ssh_streaming(&pod.ssh, &format!("{REMOTE_MODL} pull {}", shell_quote(m)))
            .with_context(|| format!("`modl pull {m}` failed on the pod"))?;
    }
    Ok(())
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

/// Reject specs the pod can't satisfy yet, with a useful message.
pub fn check_pod_supported(yaml: &str) -> Result<()> {
    let v: serde_yaml::Value = serde_yaml::from_str(yaml).context("Invalid workflow YAML")?;
    let steps = v.get("steps").and_then(|s| s.as_sequence());
    let has_lora = v.get("lora").is_some()
        || steps
            .map(|steps| steps.iter().any(|s| s.get("lora").is_some()))
            .unwrap_or(false);
    if has_lora {
        bail!(
            "LoRA references aren't supported on pods yet — the pod's store has no local LoRAs.\n  \
             Run this workflow locally, or drop the `lora:` field."
        );
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
        let Some(src) = step.get("edit").and_then(|e| e.as_str()) else {
            continue;
        };
        if !src.starts_with('$') && !src.starts_with("data:image/") {
            let id = step.get("id").and_then(|i| i.as_str()).unwrap_or("?");
            bail!(
                "step `{id}`: edit source `{src}` is a local file path — it won't exist on the pod.\n  \
                 Add it to the `images:` map as a base64 data URI and reference it as `$name`."
            );
        }
    }
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
    prepare(pod, &models)?;

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

    let local_dir = local_dest
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(format!("./pod-outputs/{run_id}")));
    std::fs::create_dir_all(&local_dir)
        .with_context(|| format!("Failed to create {}", local_dir.display()))?;
    rsync_from(&pod.ssh, &format!("{remote_export}/"), &local_dir)
        .context("Failed to sync artifacts back from the pod")?;

    let mut artifacts = walk_files(&local_dir);
    artifacts.sort();
    if artifacts.is_empty() {
        bail!(
            "Run {run_id} exported nothing — inspect the pod: \
             modl pod exec -- {REMOTE_MODL} status {run_id}"
        );
    }

    eprintln!(
        "{} {} artifact(s) → {}",
        style("✓").green().bold(),
        artifacts.len(),
        local_dir.display()
    );
    for a in &artifacts {
        eprintln!("  {}", a.display());
    }
    Ok((local_dir, artifacts))
}

/// Poll `modl status --json` on the pod until the run reaches a terminal
/// state, streaming the run log for feedback. The status poll is
/// authoritative; the log tail is cosmetic and may drop/reconnect freely.
fn wait_for_run(pod: &Pod, run_id: &str, log: &str, pidfile: &str) -> Result<String> {
    let mut tail = spawn_log_tail(pod, log);
    let mut poll_failures = 0u32;

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
                if status == "running" || status == "pending" {
                    let probe = run_ssh_capture(
                        &pod.ssh,
                        &format!(
                            "if [ -f {pf} ] && kill -0 \"$(cat {pf})\" 2>/dev/null; \
                             then echo alive; else echo dead; fi",
                            pf = shell_quote(pidfile)
                        ),
                    );
                    if let Ok(out) = probe
                        && out.contains("dead")
                    {
                        let tail_txt =
                            run_ssh_capture(&pod.ssh, &format!("tail -n 30 {}", shell_quote(log)))
                                .unwrap_or_default();
                        bail!(
                            "The workflow runner on the pod died before finishing (status still '{status}').\n\nLast run log:\n{tail_txt}"
                        );
                    }
                }
            }
            Err(e) => {
                // Flaky link — the run is unaffected, keep trying.
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
    let out = run_ssh_capture(
        &pod.ssh,
        &format!("{REMOTE_MODL} status {} --json", shell_quote(run_id)),
    )?;
    let v: serde_json::Value =
        serde_json::from_str(out.trim()).context("modl status returned invalid JSON")?;
    v.get("status")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string())
        .context("modl status JSON missing 'status'")
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

/// Build a one-step generate workflow so `modl generate --pod` funnels
/// through the same remote surface as `modl run --pod`.
pub fn single_generate_spec(
    model: &str,
    prompt: &str,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seeds: &[u64],
) -> Result<String> {
    let mut step = serde_yaml::Mapping::new();
    step.insert("id".into(), "gen".into());
    step.insert("generate".into(), prompt.into());
    insert_opt(&mut step, "width", width.map(|v| v.into()));
    insert_opt(&mut step, "height", height.map(|v| v.into()));
    insert_opt(&mut step, "steps", steps.map(|v| v.into()));
    insert_opt(&mut step, "guidance", guidance.map(|v| v.into()));
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

    let mut root = serde_yaml::Mapping::new();
    root.insert("name".into(), "pod-generate".into());
    root.insert("model".into(), model.into());
    root.insert(
        "steps".into(),
        serde_yaml::Value::Sequence(vec![serde_yaml::Value::Mapping(step)]),
    );
    serde_yaml::to_string(&serde_yaml::Value::Mapping(root)).context("Failed to build spec")
}

/// Build a one-step edit workflow with the input image inlined as a base64
/// data URI (the MCP-safe image-ref form — client paths don't exist on pods).
#[allow(clippy::too_many_arguments)]
pub fn single_edit_spec(
    model: &str,
    prompt: &str,
    image_path: &Path,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seeds: &[u64],
) -> Result<String> {
    let bytes = std::fs::read(image_path)
        .with_context(|| format!("Failed to read {}", image_path.display()))?;
    let mime = match image_path
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
    let data_uri = format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    );

    let mut images = serde_yaml::Mapping::new();
    images.insert("input".into(), data_uri.into());

    let mut step = serde_yaml::Mapping::new();
    step.insert("id".into(), "edit".into());
    step.insert("edit".into(), "$input".into());
    step.insert("prompt".into(), prompt.into());
    insert_opt(&mut step, "width", width.map(|v| v.into()));
    insert_opt(&mut step, "height", height.map(|v| v.into()));
    insert_opt(&mut step, "steps", steps.map(|v| v.into()));
    insert_opt(&mut step, "guidance", guidance.map(|v| v.into()));
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
    fn rejects_lora_specs() {
        let yaml = "name: t\nmodel: m\nlora: my-lora\nsteps:\n  - id: a\n    generate: x\n";
        assert!(check_pod_supported(yaml).is_err());
        let yaml2 = "name: t\nmodel: m\nsteps:\n  - id: a\n    generate: x\n    lora: l\n";
        assert!(check_pod_supported(yaml2).is_err());
        let ok = "name: t\nmodel: m\nsteps:\n  - id: a\n    generate: x\n";
        assert!(check_pod_supported(ok).is_ok());
    }

    #[test]
    fn generate_spec_roundtrips() {
        let spec = single_generate_spec(
            "flux-schnell",
            "a red apple",
            Some(1024),
            None,
            Some(4),
            None,
            &[7],
        )
        .unwrap();
        let models = workflow_models(&spec).unwrap();
        assert_eq!(models, vec!["flux-schnell".to_string()]);
        assert!(spec.contains("a red apple"));
        assert!(spec.contains("width: 1024"));
        assert!(spec.contains("seed: 7"));
        assert!(!spec.contains("height"));
    }

    #[test]
    fn edit_spec_inlines_image_as_data_uri() {
        let tmp = std::env::temp_dir().join(format!("modl-podrun-test-{}.png", std::process::id()));
        std::fs::write(&tmp, b"fakepng").unwrap();
        let spec = single_edit_spec(
            "klein-9b",
            "make it golden",
            &tmp,
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
            &tmp,
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
