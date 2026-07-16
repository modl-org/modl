//! BYO-pod GPU lifecycle — rent a Vast.ai GPU with the user's own key, ship
//! `python/modl_worker` + job specs, run jobs, sync artifacts back.
//!
//! The pod never needs the modl binary, the hosted orchestrator, or the Vast
//! API key:
//! - `python/modl_worker` + a uv-managed Python 3.11 venv + job specs ship
//!   over rsync/ssh
//! - the worker resolves the base model to a HuggingFace repo (no local store)
//! - jobs run under nohup; JSONL events stream back via `ssh tail -F`
//! - finished artifacts rsync home and register like a local run
//!
//! ## Composable stages (docs/plans/pod-lifecycle.md §1.1)
//!
//! `provision → bootstrap → run_train_job → teardown`. The one-shot
//! `run_pod_training()` chains all four; `modl pod up` stops after `bootstrap`
//! and persists a [`crate::core::pod_state::PodRecord`] so later jobs reuse
//! the warm instance (bootstrap fast-paths on a fingerprint match).

use anyhow::{Context, Result, bail};
use console::style;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::core::db::Database;
use crate::core::executor::parse_worker_event;
use crate::core::job::{EventPayload, TrainJobSpec};
use crate::core::pod_state::{self, PodRecord};
use crate::core::training::resolve_worker_python_root;
use crate::core::vast;

pub(crate) const REMOTE_ROOT: &str = "/root/modl-pod";
/// Pods are always CUDA, and the trainer profile is a superset that
/// `setup_generation()` reuses — so a pod carries exactly one PyTorch
/// environment for both training and generation.
const POD_RUNTIME_PROFILE: &str = "trainer-cu124";
/// Managed-runtime paths on the pod (root user, default modl_root). These
/// mirror `runtime_root()/envs/<profile>` and `runtime_root()/ai-toolkit`.
const POD_RUNTIME_PY: &str = "/root/.modl/runtime/envs/trainer-cu124/bin/python";
const POD_AITOOLKIT: &str = "/root/.modl/runtime/ai-toolkit";
/// Default disk for one-shot train pods. `pod up` raises this (persistent
/// pods accumulate an HF cache across jobs — that's the point).
pub const POD_DISK_GB: f64 = 80.0;
/// Per-host boot budget. Duds get destroyed and the next offer is tried,
/// so this can be tight — good hosts come up in 1-3 minutes.
const PROVISION_TIMEOUT_SECS: u64 = 8 * 60;
/// Give up if SSH never accepts a connection after the instance reports running.
const SSH_TIMEOUT_SECS: u64 = 6 * 60;
/// Seconds of event silence before probing whether the worker is still alive.
const LIVENESS_CHECK_SECS: u64 = 60;

/// Rental confirmation injected by the caller. Receives a human-readable
/// price line; returns whether to proceed. Core never talks to a TTY itself —
/// the CLI injects a dialoguer prompt, non-interactive callers (MCP, web UI)
/// inject nothing and must pass `yes` explicitly to rent.
pub type ConfirmRent = Box<dyn Fn(&str) -> Result<bool> + Send + Sync>;

/// Rental configuration shared by `train --pod` (one-shot) and `pod up`.
pub struct PodOptions {
    pub gpu_type: String,
    pub max_price_per_hour: f64,
    pub disk_gb: f64,
    pub yes: bool,
    pub keep_pod: bool,
    /// Vast instance label (shows in `pod ls` / the Vast console).
    pub label: String,
    /// Interactive rent confirmation. Only consulted when `yes` is false;
    /// `None` + `yes: false` refuses to rent rather than block on a TTY.
    pub confirm_rent: Option<ConfirmRent>,
}

/// SSH target for a booted pod. Fields are `pub(crate)` so `pod_executor`
/// can reuse the transport helpers.
#[derive(Clone)]
pub(crate) struct SshTarget {
    pub(crate) host: String,
    pub(crate) port: u16,
}

impl SshTarget {
    pub(crate) fn base_args(&self) -> Vec<String> {
        vec![
            "-o".into(),
            "StrictHostKeyChecking=accept-new".into(),
            "-o".into(),
            "ConnectTimeout=15".into(),
            "-o".into(),
            "ServerAliveInterval=30".into(),
            "-p".into(),
            self.port.to_string(),
            format!("root@{}", self.host),
        ]
    }

    pub(crate) fn rsync_transport(&self) -> String {
        format!(
            "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p {}",
            self.port
        )
    }
}

/// A booted, SSH-reachable pod. Returned by [`provision`], consumed by the
/// other stages. Convertible to/from a persisted [`PodRecord`].
#[derive(Clone)]
pub struct Pod {
    pub instance_id: u64,
    pub gpu_name: String,
    pub dph_total: f64,
    pub(crate) ssh: SshTarget,
    /// RFC3339 — set when the pod was first provisioned.
    pub created_at: String,
    pub bootstrap_fingerprint: Option<String>,
}

impl Pod {
    /// Build a persistable record for `pods.json`.
    pub fn to_record(&self, label: &str) -> PodRecord {
        PodRecord {
            instance_id: self.instance_id,
            gpu_name: self.gpu_name.clone(),
            dph_total: self.dph_total,
            ssh_host: self.ssh.host.clone(),
            ssh_port: self.ssh.port,
            created_at: self.created_at.clone(),
            bootstrap_fingerprint: self.bootstrap_fingerprint.clone(),
            label: label.to_string(),
        }
    }
}

impl From<PodRecord> for Pod {
    fn from(r: PodRecord) -> Self {
        Pod {
            instance_id: r.instance_id,
            gpu_name: r.gpu_name,
            dph_total: r.dph_total,
            ssh: SshTarget {
                host: r.ssh_host,
                port: r.ssh_port,
            },
            created_at: r.created_at,
            bootstrap_fingerprint: r.bootstrap_fingerprint,
        }
    }
}

// ===========================================================================
// Stage 1 — provision (search → rent w/ boot-failover → wait for SSH)
// ===========================================================================

/// Search the marketplace, rent an offer (retrying across offers/machines on
/// boot failures), and wait until SSH accepts connections. The returned
/// [`Pod`] is ready for [`bootstrap`].
pub async fn provision(opts: &PodOptions, min_vram_gb: Option<u32>) -> Result<Pod> {
    vast::api_key()?;
    check_local_tools()?;

    // ---------------------------------------------------------------
    // 1. Find and confirm an offer
    // ---------------------------------------------------------------
    eprintln!(
        "{} Searching Vast.ai for {} offers (max ${:.2}/hr, ranked by perf-per-dollar)...",
        style("→").cyan(),
        opts.gpu_type,
        opts.max_price_per_hour
    );
    let offers = vast::search_offers(&opts.gpu_type, opts.max_price_per_hour, min_vram_gb).await?;
    if offers.is_empty() {
        bail!(
            "No rentable {} offers under ${:.2}/hr{}. Try a bigger GPU type or raise --max-price.",
            opts.gpu_type,
            opts.max_price_per_hour,
            min_vram_gb
                .map(|g| format!(" with ≥{g}GB VRAM"))
                .unwrap_or_default()
        );
    }

    let best = &offers[0];
    eprintln!(
        "  {} — {:.0}GB VRAM, {:.0}GB disk, {:.0} Mbps down, ${:.3}/hr (reliability {:.1}%, value {:.0} dlperf/$)",
        style(&best.gpu_name).bold(),
        best.gpu_ram_mb as f64 / 1024.0,
        best.disk_gb,
        best.inet_down,
        best.dph_total,
        best.reliability * 100.0,
        vast::offer_value(best)
    );

    if !opts.yes {
        let prompt = format!(
            "Rent this pod on your Vast.ai account (~${:.3}/hr, billed until destroyed)?",
            best.dph_total
        );
        let ok = match &opts.confirm_rent {
            Some(confirm) => confirm(&prompt)?,
            None => bail!(
                "Refusing to rent a pod (~${:.3}/hr) without confirmation — pass --yes / yes: true to opt in.",
                best.dph_total
            ),
        };
        if !ok {
            bail!("Aborted before renting a pod.");
        }
    }

    // ---------------------------------------------------------------
    // 2. Rent + boot — a marketplace host can be taken (rent fails) or a
    //    dud (rents but never boots). Both fall through to the next offer.
    // ---------------------------------------------------------------
    // Disable Vast's auto-tmux so our SSH commands run in a plain shell.
    let onstart = "touch /root/.no_auto_tmux\n";
    let env = vec![("HF_HUB_OFFLINE".to_string(), "0".to_string())];

    // The user confirmed offers[0]'s price; fallback offers are ranked by
    // perf-per-dollar, not price, so without a cap a failed first rent could
    // silently land on anything up to --max-price.
    let price_cap = best.dph_total * 1.25;

    let mut booted: Option<(u64, vast::Offer, SshTarget)> = None;
    let mut bad_machines: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for offer in offers.iter().take(8) {
        if offer.machine_id != 0 && bad_machines.contains(&offer.machine_id) {
            continue; // same physical box as one that just failed to boot
        }
        if offer.dph_total > price_cap {
            eprintln!(
                "{} Skipping fallback offer {} at ${:.3}/hr — more than 25% over the confirmed ${:.3}/hr",
                style("!").yellow(),
                offer.id,
                offer.dph_total,
                best.dph_total
            );
            continue;
        }
        let id = match vast::create_instance(
            offer.id,
            vast::POD_IMAGE,
            onstart,
            &env,
            opts.disk_gb,
            &opts.label,
        )
        .await
        {
            Ok(id) => id,
            Err(e) => {
                eprintln!(
                    "{} Offer {} unavailable ({e}), trying next...",
                    style("!").yellow(),
                    offer.id
                );
                continue;
            }
        };

        eprintln!(
            "{} Rented instance {} (${:.3}/hr). If anything goes wrong: modl pod rm {}",
            style("✓").green(),
            style(id).bold(),
            offer.dph_total,
            id
        );

        // Track the instance from the moment it bills. If the boot wait, the
        // bootstrap, or the user's Ctrl-C kills this flow, `pod ls` and the
        // stale-pod nag must still know about it. SSH details land after
        // boot; the fingerprint after bootstrap.
        if let Err(e) = pod_state::upsert(PodRecord {
            instance_id: id,
            gpu_name: offer.gpu_name.clone(),
            dph_total: offer.dph_total,
            ssh_host: String::new(),
            ssh_port: 0,
            created_at: pod_state::now_rfc3339(),
            bootstrap_fingerprint: None,
            label: opts.label.clone(),
        }) {
            eprintln!(
                "{} Could not record pod {id} in pods.json: {e}",
                style("⚠").yellow()
            );
        }

        match wait_for_instance(id).await {
            Ok(ssh) => {
                booted = Some((id, offer.clone(), ssh));
                break;
            }
            Err(e) => {
                eprintln!("{} {e}", style("!").yellow());
                eprintln!(
                    "{} Host never booted — destroying instance {} and trying the next offer...",
                    style("!").yellow(),
                    id
                );
                bad_machines.insert(offer.machine_id);
                match vast::destroy_instance(id).await {
                    // Only forget a dud we actually destroyed — a record for
                    // a still-billing instance must survive.
                    Ok(()) => {
                        let _ = pod_state::remove(id);
                    }
                    Err(e) => eprintln!(
                        "{} Could not destroy {id}: {e} — still billing, check with: modl pod ls",
                        style("✗").red()
                    ),
                }
            }
        }
    }
    let (instance_id, rented_offer, ssh) =
        booted.context("No offer produced a working pod — try again shortly")?;

    let pod = Pod {
        instance_id,
        gpu_name: rented_offer.gpu_name,
        dph_total: rented_offer.dph_total,
        ssh,
        created_at: pod_state::now_rfc3339(),
        bootstrap_fingerprint: None,
    };
    // Refresh the record with the SSH target now that we have one.
    if let Err(e) = pod_state::upsert(pod.to_record(&opts.label)) {
        eprintln!(
            "{} Could not update pods.json for {instance_id}: {e}",
            style("⚠").yellow()
        );
    }

    // ---------------------------------------------------------------
    // 3. Wait for sshd to accept connections (running != ssh-ready)
    // ---------------------------------------------------------------
    wait_for_ssh(&pod.ssh)?;

    Ok(pod)
}

// ===========================================================================
// Stage 2 — bootstrap (rsync worker + ai-toolkit/deps, fingerprinted)
// ===========================================================================

/// Ship `modl_worker`, install modl, and ensure the managed runtime.
///
/// The pod runs the same managed runtime as a local install — modl's trainer
/// profile (torch + diffusers + ai-toolkit at the pinned SHA). Previously the
/// pod built a second, duplicate PyTorch venv via a bespoke bootstrap script;
/// now `modl runtime install` owns the environment and its own idempotency,
/// so both training and generation share one env.
pub fn bootstrap(pod: &Pod) -> Result<()> {
    let ssh = &pod.ssh;

    // Always refresh the worker — small, and it picks up local worker edits
    // between jobs on a persistent pod.
    eprintln!("{} Uploading worker...", style("→").cyan());
    run_ssh_quiet(ssh, &format!("mkdir -p {REMOTE_ROOT}/worker"))?;
    rsync_to(
        ssh,
        &resolve_worker_python_root()?.join("modl_worker"),
        &format!("{REMOTE_ROOT}/worker/"),
    )?;

    crate::core::pod_run::ensure_modl(pod)?;

    eprintln!(
        "{} Ensuring managed runtime ({POD_RUNTIME_PROFILE}) on pod...",
        style("→").cyan()
    );
    // `train setup` = runtime install + bootstrap (deps, ai-toolkit clone)
    // in one shot, idempotent via the runtime's own bootstrap marker.
    run_ssh_streaming(
        ssh,
        &format!("{} train setup", crate::core::pod_run::REMOTE_MODL),
    )
    .context("`modl train setup` failed on the pod")?;

    let _ = pod_state::set_fingerprint(pod.instance_id, &bootstrap_fingerprint());
    Ok(())
}

/// Marker recorded on the pod record after a successful bootstrap. The
/// runtime install owns real idempotency; this only says which modl version
/// and profile last set the pod up.
pub fn bootstrap_fingerprint() -> String {
    format!("modl-{}:{}", env!("CARGO_PKG_VERSION"), POD_RUNTIME_PROFILE)
}

// ===========================================================================
// Stage 3 — run_train_job (upload dataset/spec → train → sync back)
// ===========================================================================

/// Upload the dataset + rewritten spec, launch the trainer under nohup, stream
/// events, then rsync the LoRA home and register it. Assumes [`bootstrap`] has
/// run. Does NOT tear the pod down.
pub fn run_train_job(pod: &Pod, spec: &TrainJobSpec) -> Result<()> {
    let ssh = &pod.ssh;

    let ds = Path::new(&spec.dataset.path);
    if !ds.is_dir() {
        bail!("Dataset directory not found: {}", spec.dataset.path);
    }

    let hf_token = huggingface_token();
    if hf_token.is_none() {
        eprintln!(
            "{} No HuggingFace token found — gated base models (Flux, Klein) will fail to download on the pod. Add one with: modl auth add huggingface",
            style("⚠").yellow()
        );
    }

    // A worker from a previous run may still be alive: nohup survives a
    // dropped link, and proceeding would wipe its dataset and events file
    // out from under it and launch a second trainer on the same GPU.
    if matches!(probe_worker(ssh), WorkerProbe::Alive) {
        bail!(
            "A job is already running on pod {} — wait for it to finish, or destroy the pod: modl pod rm {}",
            pod.instance_id,
            pod.instance_id
        );
    }

    // ---------------------------------------------------------------
    // 4. Ship dataset + spec
    // ---------------------------------------------------------------
    let dataset_path = PathBuf::from(&spec.dataset.path);
    eprintln!("{} Uploading dataset...", style("→").cyan());
    run_ssh_quiet(ssh, &format!("mkdir -p {REMOTE_ROOT}/dataset"))?;
    rsync_to(ssh, &dataset_path, &format!("{REMOTE_ROOT}/dataset-up/"))?;
    // rsync of a dir path (no trailing slash) nests it: dataset-up/<basename>.
    // Normalize so the spec can always point at REMOTE_ROOT/dataset.
    let ds_name = dataset_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .context("Dataset path has no directory name")?;
    run_ssh_quiet(
        ssh,
        &format!(
            "rm -rf {REMOTE_ROOT}/dataset && mv {REMOTE_ROOT}/dataset-up/{} {REMOTE_ROOT}/dataset",
            shell_quote(&ds_name)
        ),
    )?;

    let remote_spec = build_remote_spec(spec);
    let spec_yaml = serde_yaml::to_string(&remote_spec)?;
    let tmp_spec = std::env::temp_dir().join(format!("modl-pod-spec-{}.yaml", pod.instance_id));
    std::fs::write(&tmp_spec, &spec_yaml)?;
    rsync_to(ssh, &tmp_spec, &format!("{REMOTE_ROOT}/spec.yaml"))?;
    let _ = std::fs::remove_file(&tmp_spec);

    // ---------------------------------------------------------------
    // 5. Launch training under nohup, stream events
    // ---------------------------------------------------------------
    let job_id = format!("pod-{}", chrono::Local::now().format("%Y%m%d-%H%M%S"));
    let db = Database::open()?;
    db.insert_job(
        &job_id,
        "train",
        "running",
        &serde_json::to_string(spec)?,
        "pod",
        Some("vast"),
        None,
    )?;

    // The HF token must not land in any argv: local `ps` shows the full ssh
    // command line, and remote `env` would show it in the worker's argv too.
    // Ship it over stdin to a 600-perm file and export it via a shell prefix
    // assignment, which appears in no process's arguments.
    let token_prefix = match hf_token {
        Some(t) => {
            run_ssh_stdin(
                ssh,
                &format!("umask 077 && cat > {REMOTE_ROOT}/.hf_token"),
                &t,
            )?;
            format!("HF_TOKEN=\"$(cat {REMOTE_ROOT}/.hf_token)\" ")
        }
        None => String::new(),
    };
    let launch = format!(
        "test -x {POD_RUNTIME_PY} || {{ echo 'managed runtime missing — rerun modl pod up' >&2; exit 9; }}; \
         cd {REMOTE_ROOT} && rm -f events.jsonl && \
         {token_prefix}HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
         PYTHONPATH={REMOTE_ROOT}/worker MODL_AITOOLKIT_ROOT={POD_AITOOLKIT} \
         nohup {POD_RUNTIME_PY} -m modl_worker.main train --config {REMOTE_ROOT}/spec.yaml --job-id {job_id} \
         >> {REMOTE_ROOT}/events.jsonl 2>> {REMOTE_ROOT}/stderr.log & \
         echo $! > {REMOTE_ROOT}/worker.pid; echo launched"
    );

    // From here on the job row exists — no error may leave it stuck at
    // "running" (phantom running jobs in `modl ls` / the UI training tab).
    let result = (|| -> Result<()> {
        run_ssh_quiet(ssh, &launch)?;
        eprintln!(
            "{} Training started on pod — {}",
            style("→").cyan(),
            style(&job_id).dim()
        );

        match stream_events(ssh, &job_id, &db)? {
            TrainOutcome::Completed => {}
            TrainOutcome::Failed(msg) => {
                let tail = run_ssh_capture(ssh, &format!("tail -n 30 {REMOTE_ROOT}/stderr.log"))
                    .unwrap_or_default();
                bail!("Pod training failed: {msg}\n\nLast worker stderr:\n{tail}");
            }
        }

        // ---------------------------------------------------------------
        // 6. Sync artifacts back + register locally
        // ---------------------------------------------------------------
        eprintln!("{} Syncing artifacts back...", style("→").cyan());
        let local_out = PathBuf::from(&spec.output.destination_dir);
        std::fs::create_dir_all(&local_out)?;
        rsync_from(ssh, &format!("{REMOTE_ROOT}/output/"), &local_out)?;

        let final_lora = find_final_lora(&local_out, &spec.output.lora_name)?;
        let store_root = crate::core::paths::modl_root();
        let collected = crate::core::artifacts::collect_lora(
            &final_lora,
            &spec.output.lora_name,
            &spec.model.base_model_id,
            &spec.params.trigger_word,
            &job_id,
            &db,
            &store_root,
        )?;
        db.update_job_status(&job_id, "completed")?;

        eprintln!(
            "{} LoRA registered: {}",
            style("✓").green().bold(),
            collected.store_path.display()
        );
        eprintln!(
            "  Try it: modl generate \"{} ...\" --lora {} --base {}",
            spec.params.trigger_word, spec.output.lora_name, spec.model.base_model_id
        );
        Ok(())
    })();

    if result.is_err() {
        let _ = db.update_job_status(&job_id, "error");
    }
    result
}

// ===========================================================================
// Stage 4 — teardown (destroy + prune pods.json)
// ===========================================================================

/// Destroy the instance (billing stops) and remove it from `pods.json`.
///
/// The record is only pruned on a successful destroy — a still-billing
/// instance must stay visible to `pod ls` and the stale-pod nag. A failed
/// destroy is an error so callers (e.g. `pod up --fresh`) don't proceed to
/// rent a second pod while the first one still bills.
pub async fn teardown(pod: &Pod) -> Result<()> {
    eprintln!(
        "{} Destroying instance {}...",
        style("→").cyan(),
        pod.instance_id
    );
    match vast::destroy_instance(pod.instance_id).await {
        Ok(()) => {
            eprintln!("{} Pod destroyed — billing stopped.", style("✓").green());
            let _ = pod_state::remove(pod.instance_id);
            Ok(())
        }
        Err(e) => bail!(
            "Could not destroy instance {}: {e}\n  It is still billing — destroy it manually: modl pod rm {}",
            pod.instance_id,
            pod.instance_id
        ),
    }
}

// ===========================================================================
// One-shot orchestration — provision → bootstrap → train → teardown
// ===========================================================================

/// One-shot `modl train --pod` (no active pod reused): provision a fresh pod,
/// bootstrap, train, and — unless `--keep-pod` — destroy it. `--keep-pod`
/// persists a `pods.json` record so the instance becomes a reusable pod.
pub async fn run_pod_training(spec: TrainJobSpec, opts: PodOptions) -> Result<()> {
    let min_vram_gb = min_vram_for_train(&spec);
    if let Some(gb) = min_vram_gb {
        eprintln!(
            "{} {} needs ≥{}GB VRAM for training ({})",
            style("→").cyan(),
            spec.model.base_model_id,
            gb,
            if spec.params.quantize {
                "quantized"
            } else {
                "bf16"
            }
        );
    }

    let pod = provision(&opts, min_vram_gb).await?;
    let started_at = std::time::Instant::now();

    // Everything after boot runs inside a fallible block so the pod is
    // destroyed on error unless --keep-pod was passed.
    let result = bootstrap(&pod).and_then(|()| run_train_job(&pod, &spec));

    if opts.keep_pod {
        // --keep-pod is the manual bridge into persistent mode: record the
        // instance so `train --pod` / `pod run` reuse it (and `pod ls` shows it).
        let mut kept = pod.clone();
        kept.bootstrap_fingerprint = Some(bootstrap_fingerprint());
        if let Err(e) = pod_state::upsert(kept.to_record(&opts.label)) {
            eprintln!("{} Could not record kept pod: {e}", style("⚠").yellow());
        }
        eprintln!(
            "{} --keep-pod: instance {} still running & billing.\n  Reuse:  modl train --pod {} …  /  modl pod run …\n  Kill:   modl pod rm {}",
            style("⚠").yellow(),
            pod.instance_id,
            opts.gpu_type,
            pod.instance_id
        );
    } else {
        if result.is_err() {
            // Best-effort artifact rescue before the pod (and its disk) goes
            // away — a run that died at step 900/1000 still holds valuable
            // checkpoints.
            let local_out = PathBuf::from(&spec.output.destination_dir);
            if std::fs::create_dir_all(&local_out).is_ok()
                && rsync_from(&pod.ssh, &format!("{REMOTE_ROOT}/output/"), &local_out).is_ok()
            {
                eprintln!(
                    "{} Salvaged partial artifacts → {}",
                    style("→").cyan(),
                    local_out.display()
                );
            }
        }
        if let Err(e) = teardown(&pod).await {
            // Don't clobber the training result — the LoRA may already be
            // registered locally. The pod stays tracked and nagged about.
            eprintln!("{} {e}", style("✗").red());
        }
    }

    let hours = started_at.elapsed().as_secs_f64() / 3600.0;
    eprintln!(
        "  Pod time: {:.0}m — estimated cost ${:.2}",
        hours * 60.0,
        hours * pod.dph_total
    );

    result
}

/// VRAM floor for a training job, from models.toml. Presets enable
/// quantization, so fp8 is the realistic floor; bf16 otherwise.
pub fn min_vram_for_train(spec: &TrainJobSpec) -> Option<u32> {
    crate::core::model_family::find_model(&spec.model.base_model_id).map(|m| {
        if spec.params.quantize {
            m.vram_fp8_gb
        } else {
            m.vram_bf16_gb
        }
    })
}

// ===========================================================================
// Shared helpers
// ===========================================================================

fn check_local_tools() -> Result<()> {
    for tool in ["ssh", "rsync"] {
        let ok = Command::new(tool)
            .arg("-V")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
            || Command::new(tool)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
        if !ok {
            bail!("`{tool}` not found — pod jobs need OpenSSH and rsync installed locally.");
        }
    }
    Ok(())
}

pub(crate) fn huggingface_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN")
        && !t.trim().is_empty()
    {
        return Some(t.trim().to_string());
    }
    crate::auth::AuthStore::load()
        .ok()
        .and_then(|s| s.token_for("huggingface"))
}

/// Remote copy of the spec: no local store paths, remote dataset/output dirs.
/// The worker falls back to HuggingFace repo IDs when base_model_path is unset.
fn build_remote_spec(spec: &TrainJobSpec) -> TrainJobSpec {
    let mut remote = spec.clone();
    remote.model.base_model_path = None;
    remote.dataset.path = format!("{REMOTE_ROOT}/dataset");
    remote.output.destination_dir = format!("{REMOTE_ROOT}/output");
    remote
}

enum TrainOutcome {
    Completed,
    Failed(String),
}

/// Follow the remote events file, printing progress and persisting events.
/// Reconnects if the SSH tail drops (pods have flaky links).
///
/// A worker that dies without a terminal event (kernel OOM-kill, a crash
/// before the first event is emitted) leaves a healthy `tail -F` following a
/// file that will never grow — without a liveness probe that hangs forever
/// on a billing pod. So when the stream goes quiet we ask the pod whether
/// the worker PID is still alive, and only a definitive dead answer fails
/// the job (SSH errors during the probe count as alive so a flaky link
/// can't kill a healthy run).
fn stream_events(ssh: &SshTarget, job_id: &str, db: &Database) -> Result<TrainOutcome> {
    let mut reconnects = 0u32;
    let mut last_seq: u64 = 0;

    loop {
        let mut args = ssh.base_args();
        args.push(format!("tail -n +1 -F {REMOTE_ROOT}/events.jsonl"));
        let mut child = Command::new("ssh")
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn ssh tail for event streaming")?;

        let stdout = child.stdout.take().context("ssh tail: no stdout")?;
        let (line_tx, line_rx) = std::sync::mpsc::channel::<String>();
        let reader_thread = std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                if line_tx.send(line).is_err() {
                    break;
                }
            }
        });

        // Some(..) = terminal outcome; None = tail dropped, reconnect.
        let mut outcome: Option<TrainOutcome> = None;
        loop {
            use std::sync::mpsc::RecvTimeoutError;
            let line = match line_rx
                .recv_timeout(std::time::Duration::from_secs(LIVENESS_CHECK_SECS))
            {
                Ok(line) => line,
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    // Quiet stream — normal during long steps or model
                    // downloads, as long as the worker is still running.
                    if !matches!(probe_worker(ssh), WorkerProbe::Dead) {
                        continue;
                    }
                    // Drain any in-flight lines: the worker may have just
                    // finished and its terminal event still be in the pipe.
                    while let Ok(line) = line_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                        if let Some(o) = handle_event_line(&line, job_id, db, &mut last_seq) {
                            outcome = Some(o);
                            break;
                        }
                    }
                    if outcome.is_none() {
                        outcome = Some(TrainOutcome::Failed(
                            "worker process died without reporting an error \
                             (OOM-killed or crashed before the first event)"
                                .to_string(),
                        ));
                    }
                    break;
                }
            };
            if let Some(o) = handle_event_line(&line, job_id, db, &mut last_seq) {
                outcome = Some(o);
                break;
            }
        }

        let _ = child.kill();
        let _ = child.wait();
        drop(line_rx);
        let _ = reader_thread.join();

        if let Some(outcome) = outcome {
            return Ok(outcome);
        }

        // Tail ended without a terminal event — connection dropped.
        reconnects += 1;
        if reconnects > 20 {
            bail!(
                "Lost connection to pod {} times — training may still be running.\n\
                 Reconnect manually: modl pod ls / ssh, events at {REMOTE_ROOT}/events.jsonl",
                reconnects
            );
        }
        eprintln!(
            "{} SSH stream dropped — reconnecting ({reconnects})...",
            style("!").yellow()
        );
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}

/// Parse, persist, and print one JSONL event line. Returns Some(..) for a
/// terminal event.
fn handle_event_line(
    line: &str,
    job_id: &str,
    db: &Database,
    last_seq: &mut u64,
) -> Option<TrainOutcome> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let raw = serde_json::from_str::<serde_json::Value>(line).ok()?;
    let event = parse_worker_event(&raw, job_id)?;
    if event.sequence != 0 && event.sequence <= *last_seq {
        return None; // already seen before a reconnect
    }
    *last_seq = event.sequence.max(*last_seq);
    let _ = db.insert_job_event(
        job_id,
        event.sequence,
        &serde_json::to_string(&event).unwrap_or_default(),
    );

    match &event.event {
        EventPayload::Progress {
            stage,
            step,
            total_steps,
            loss,
            ..
        } => {
            let loss_str = loss.map(|l| format!("  loss {l:.4}")).unwrap_or_default();
            eprintln!("  [{stage}] step {step}/{total_steps}{loss_str}");
        }
        EventPayload::Log { level, message } if level != "debug" => {
            eprintln!("  {}", style(message).dim());
        }
        EventPayload::Warning { message, .. } => {
            eprintln!("  {} {message}", style("⚠").yellow());
        }
        EventPayload::Artifact { path, .. } => {
            eprintln!("  {} checkpoint: {path}", style("•").cyan());
        }
        EventPayload::Completed { .. } => return Some(TrainOutcome::Completed),
        EventPayload::Error { message, .. } => {
            return Some(TrainOutcome::Failed(message.clone()));
        }
        _ => {}
    }
    None
}

enum WorkerProbe {
    Alive,
    Dead,
    Unknown,
}

/// Ask the pod whether the worker process (from this or a previous job) is
/// running. `Unknown` on SSH failure — callers pick the safe direction:
/// the liveness loop only acts on `Dead`, the relaunch guard only on `Alive`.
fn probe_worker(ssh: &SshTarget) -> WorkerProbe {
    let cmd = format!(
        "if [ -f {REMOTE_ROOT}/worker.pid ] && kill -0 \"$(cat {REMOTE_ROOT}/worker.pid)\" 2>/dev/null; \
         then echo alive; else echo dead; fi"
    );
    match run_ssh_capture(ssh, &cmd) {
        Ok(out) if out.contains("alive") => WorkerProbe::Alive,
        Ok(out) if out.contains("dead") => WorkerProbe::Dead,
        _ => WorkerProbe::Unknown,
    }
}

async fn wait_for_instance(instance_id: u64) -> Result<SshTarget> {
    eprintln!(
        "{} Waiting for instance to boot (usually 1-3 minutes)...",
        style("→").cyan()
    );
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(PROVISION_TIMEOUT_SECS);
    let mut last_status = String::new();
    let mut poll_failures = 0u32;

    loop {
        // Vast's API throws intermittent 5xx/timeouts — a single transient
        // error must not get a healthy booting pod destroyed and its machine
        // blacklisted. Only give up after several consecutive failures.
        let inst = match vast::get_instance(instance_id).await {
            Ok(inst) => {
                poll_failures = 0;
                inst
            }
            Err(e) => {
                poll_failures += 1;
                if poll_failures >= 5 {
                    return Err(e.context(format!(
                        "Vast API failed {poll_failures} consecutive status polls for instance {instance_id}"
                    )));
                }
                eprintln!(
                    "{} Vast API error while polling ({poll_failures}/5), retrying...",
                    style("!").yellow()
                );
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                continue;
            }
        };
        if inst.actual_status != last_status {
            eprintln!("  status: {}", inst.actual_status);
            last_status = inst.actual_status.clone();
        }
        if inst.actual_status == "running"
            && let (Some(host), Some(port)) = (inst.ssh_host.clone(), inst.ssh_port)
        {
            return Ok(SshTarget { host, port });
        }
        // Hosts sometimes fail at container init (runc/kernel mismatches,
        // broken nvidia runtime). The daemon error is terminal — don't
        // burn the whole boot budget waiting for it to change.
        if let Some(msg) = &inst.status_msg
            && (msg.contains("Error response from daemon")
                || msg.contains("OCI runtime")
                || msg.contains("failed to start containers"))
        {
            bail!(
                "Host failed to start the container: {}",
                msg.lines().next().unwrap_or(msg)
            );
        }
        if std::time::Instant::now() > deadline {
            bail!(
                "Instance {instance_id} did not reach running state within {} minutes.",
                PROVISION_TIMEOUT_SECS / 60
            );
        }
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    }
}

fn wait_for_ssh(ssh: &SshTarget) -> Result<()> {
    eprintln!(
        "{} Waiting for SSH at {}:{}...",
        style("→").cyan(),
        ssh.host,
        ssh.port
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(SSH_TIMEOUT_SECS);
    loop {
        let ok = Command::new("ssh")
            .args(ssh.base_args())
            .arg("true")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            bail!(
                "SSH to root@{}:{} never came up. Is an SSH key registered on your \
                 Vast.ai account (https://cloud.vast.ai/account)?",
                ssh.host,
                ssh.port
            );
        }
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}

pub(crate) fn run_ssh_quiet(ssh: &SshTarget, cmd: &str) -> Result<()> {
    let out = Command::new("ssh")
        .args(ssh.base_args())
        .arg(cmd)
        .output()
        .context("ssh command failed to spawn")?;
    if !out.status.success() {
        bail!(
            "Remote command failed: {cmd}\n{}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// Run a command on a pod over SSH with inherited stdio (for `modl pod exec`).
/// Returns the remote command's exit status so the caller can propagate it.
pub fn ssh_exec(host: &str, port: u16, command: &str) -> Result<std::process::ExitStatus> {
    let ssh = SshTarget {
        host: host.to_string(),
        port,
    };
    Command::new("ssh")
        .args(ssh.base_args())
        .arg(command)
        .status()
        .context("ssh command failed to spawn")
}

pub(crate) fn run_ssh_capture(ssh: &SshTarget, cmd: &str) -> Result<String> {
    let out = Command::new("ssh")
        .args(ssh.base_args())
        .arg(cmd)
        .output()
        .context("ssh command failed to spawn")?;
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Run a remote command feeding `input` on stdin — for secrets that must not
/// appear in any argv.
pub(crate) fn run_ssh_stdin(ssh: &SshTarget, cmd: &str, input: &str) -> Result<()> {
    use std::io::Write;
    let mut child = Command::new("ssh")
        .args(ssh.base_args())
        .arg(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("ssh command failed to spawn")?;
    child
        .stdin
        .take()
        .context("ssh: no stdin handle")?
        .write_all(input.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "Remote command failed: {cmd}\n{}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// Run a remote script with live output (bootstrap etc.). The remote output
/// is progress narration, so it goes to OUR stderr — stdout stays reserved
/// for results (`--json` consumers pipe stdout).
pub(crate) fn run_ssh_streaming(ssh: &SshTarget, script: &str) -> Result<()> {
    let status = Command::new("ssh")
        .args(ssh.base_args())
        .arg(format!("bash -c {}", shell_quote(script)))
        .stdout(stderr_stdio())
        .status()
        .context("ssh command failed to spawn")?;
    if !status.success() {
        bail!("Remote bootstrap failed (exit {status}).");
    }
    Ok(())
}

/// A `Stdio` handle pointing at this process's stderr (falls back to
/// inherited stdout if the fd can't be cloned).
fn stderr_stdio() -> Stdio {
    #[cfg(unix)]
    {
        use std::os::fd::AsFd;
        if let Ok(fd) = std::io::stderr().as_fd().try_clone_to_owned() {
            return Stdio::from(fd);
        }
    }
    Stdio::inherit()
}

pub(crate) fn rsync_to(ssh: &SshTarget, local: &Path, remote: &str) -> Result<()> {
    let status = Command::new("rsync")
        .args(["-az", "--delete", "-e", &ssh.rsync_transport()])
        .arg(local)
        .arg(format!("root@{}:{}", ssh.host, remote))
        .status()
        .context("rsync failed to spawn")?;
    if !status.success() {
        bail!("rsync upload failed for {}", local.display());
    }
    Ok(())
}

pub(crate) fn rsync_from(ssh: &SshTarget, remote: &str, local: &Path) -> Result<()> {
    let status = Command::new("rsync")
        .args(["-az", "-e", &ssh.rsync_transport()])
        .arg(format!("root@{}:{}", ssh.host, remote))
        .arg(local)
        .status()
        .context("rsync failed to spawn")?;
    if !status.success() {
        bail!("rsync download failed for {remote}");
    }
    Ok(())
}

/// Locate the final trained LoRA in the synced output directory.
fn find_final_lora(out_dir: &Path, lora_name: &str) -> Result<PathBuf> {
    // ai-toolkit writes <name>/<name>.safetensors for the final checkpoint.
    let preferred = out_dir
        .join(lora_name)
        .join(format!("{lora_name}.safetensors"));
    if preferred.exists() {
        return Ok(preferred);
    }
    let direct = out_dir.join(format!("{lora_name}.safetensors"));
    if direct.exists() {
        return Ok(direct);
    }

    // Fallback: newest .safetensors anywhere under the output dir.
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in walk_safetensors(out_dir) {
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if newest.as_ref().map(|(t, _)| modified > *t).unwrap_or(true) {
            newest = Some((modified, entry));
        }
    }
    newest
        .map(|(_, p)| p)
        .with_context(|| format!("No .safetensors found in {}", out_dir.display()))
}

fn walk_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk_safetensors(&path));
        } else if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            found.push(path);
        }
    }
    found
}

pub(crate) fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::job::*;

    fn sample_spec() -> TrainJobSpec {
        TrainJobSpec {
            dataset: DatasetRef {
                name: "my-ds".into(),
                path: "/home/user/datasets/my-ds".into(),
                image_count: 12,
                caption_coverage: 1.0,
            },
            model: ModelRef {
                base_model_id: "flux-dev".into(),
                base_model_path: Some("/home/user/modl/store/x".into()),
                arch_key: Some("flux".into()),
            },
            output: OutputRef {
                lora_name: "my-lora".into(),
                destination_dir: "/home/user/.modl/training_output/my-lora".into(),
            },
            params: TrainingParams {
                preset: Preset::Quick,
                lora_type: LoraType::Style,
                trigger_word: "ohwx".into(),
                steps: 500,
                rank: 16,
                learning_rate: 1e-4,
                optimizer: Optimizer::Adamw8bit,
                resolution: 1024,
                seed: None,
                quantize: true,
                batch_size: 1,
                num_repeats: 1,
                caption_dropout_rate: 0.05,
                resume_from: None,
                class_word: None,
                sample_every: None,
            },
            runtime: RuntimeRef {
                profile: "trainer-cu124".into(),
                python_version: None,
            },
            target: ExecutionTarget::Local,
            labels: Default::default(),
        }
    }

    #[test]
    fn remote_spec_strips_local_paths() {
        let spec = sample_spec();
        let remote = build_remote_spec(&spec);
        assert_eq!(remote.model.base_model_path, None);
        assert_eq!(remote.dataset.path, format!("{REMOTE_ROOT}/dataset"));
        assert_eq!(
            remote.output.destination_dir,
            format!("{REMOTE_ROOT}/output")
        );
        // Original untouched
        assert!(spec.model.base_model_path.is_some());
    }

    #[test]
    fn fingerprint_tracks_version_and_profile() {
        let fp = bootstrap_fingerprint();
        assert!(fp.contains(env!("CARGO_PKG_VERSION")));
        assert!(fp.contains(POD_RUNTIME_PROFILE));
        assert_eq!(fp, bootstrap_fingerprint());
    }

    #[test]
    fn runtime_paths_agree_with_profile() {
        // The launch command hardcodes the pod-side runtime layout; keep the
        // pieces consistent with each other.
        assert!(POD_RUNTIME_PY.contains(POD_RUNTIME_PROFILE));
        assert!(POD_RUNTIME_PY.starts_with("/root/.modl/runtime/envs/"));
        assert!(POD_AITOOLKIT.starts_with("/root/.modl/runtime/"));
    }

    #[test]
    fn pod_record_roundtrip() {
        let pod = Pod {
            instance_id: 999,
            gpu_name: "RTX 3090".into(),
            dph_total: 0.19,
            ssh: SshTarget {
                host: "ssh5.vast.ai".into(),
                port: 4242,
            },
            created_at: "2026-07-15T00:00:00+00:00".into(),
            bootstrap_fingerprint: Some("deadbeef".into()),
        };
        let rec = pod.to_record("modl-pod-test");
        assert_eq!(rec.instance_id, 999);
        assert_eq!(rec.ssh_port, 4242);
        assert_eq!(rec.label, "modl-pod-test");
        let back = Pod::from(rec);
        assert_eq!(back.instance_id, 999);
        assert_eq!(back.ssh.host, "ssh5.vast.ai");
        assert_eq!(back.dph_total, 0.19);
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }

    #[test]
    fn finds_final_lora_in_nested_dir() {
        let tmp = std::env::temp_dir().join(format!("modl-pod-test-{}", std::process::id()));
        let nested = tmp.join("my-lora");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("my-lora.safetensors"), b"x").unwrap();
        std::fs::write(nested.join("my-lora_000000500.safetensors"), b"x").unwrap();
        let found = find_final_lora(&tmp, "my-lora").unwrap();
        assert!(found.ends_with("my-lora/my-lora.safetensors"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
