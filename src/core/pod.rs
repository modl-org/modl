//! BYO-pod training — rent a Vast.ai GPU with the user's own key, train,
//! sync artifacts back into the local store, destroy the pod.
//!
//! The pod never needs the modl binary or the hosted orchestrator:
//! - `python/modl_worker` + dataset + spec ship over rsync
//! - the worker resolves the base model to a HuggingFace repo (no store)
//! - training runs under nohup; JSONL events stream back via `ssh tail -F`
//! - the finished LoRA rsyncs home and registers like a local training run

use anyhow::{Context, Result, bail};
use console::style;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::core::db::Database;
use crate::core::executor::parse_worker_event;
use crate::core::job::{EventPayload, TrainJobSpec};
use crate::core::runtime::{
    AITOOLKIT_PIN_SHA, AITOOLKIT_REPO_URL, TRAINER_TORCH_VERSION, TRAINER_TORCHVISION_VERSION,
};
use crate::core::training::resolve_worker_python_root;
use crate::core::vast;

const REMOTE_ROOT: &str = "/root/modl-pod";
const POD_DISK_GB: f64 = 80.0;
/// Per-host boot budget. Duds get destroyed and the next offer is tried,
/// so this can be tight — good hosts come up in 1-3 minutes.
const PROVISION_TIMEOUT_SECS: u64 = 8 * 60;
/// Give up if SSH never accepts a connection after the instance reports running.
const SSH_TIMEOUT_SECS: u64 = 6 * 60;

pub struct PodOptions {
    pub gpu_type: String,
    pub max_price_per_hour: f64,
    pub yes: bool,
    pub keep_pod: bool,
}

#[derive(Clone)]
struct SshTarget {
    host: String,
    port: u16,
}

impl SshTarget {
    fn base_args(&self) -> Vec<String> {
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

    fn rsync_transport(&self) -> String {
        format!(
            "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p {}",
            self.port
        )
    }
}

pub async fn run_pod_training(spec: TrainJobSpec, opts: PodOptions) -> Result<()> {
    preflight(&spec)?;

    let hf_token = huggingface_token();
    if hf_token.is_none() {
        println!(
            "{} No HuggingFace token found — gated base models (Flux, Klein) will fail to download on the pod. Add one with: modl auth add huggingface",
            style("⚠").yellow()
        );
    }

    // ---------------------------------------------------------------
    // 1. Find and confirm an offer
    // ---------------------------------------------------------------
    // VRAM the training job actually needs, from models.toml. Presets enable
    // quantization, so fp8 is the realistic floor; bf16 otherwise.
    let min_vram_gb = crate::core::model_family::find_model(&spec.model.base_model_id).map(|m| {
        if spec.params.quantize {
            m.vram_fp8_gb
        } else {
            m.vram_bf16_gb
        }
    });
    if let Some(gb) = min_vram_gb {
        println!(
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

    println!(
        "{} Searching Vast.ai for {} offers (max ${:.2}/hr, ranked by perf-per-dollar)...",
        style("→").cyan(),
        opts.gpu_type,
        opts.max_price_per_hour
    );
    let offers = vast::search_offers(&opts.gpu_type, opts.max_price_per_hour, min_vram_gb).await?;
    if offers.is_empty() {
        bail!(
            "No rentable {} offers under ${:.2}/hr{}. Try a bigger --pod type or raise --max-price.",
            opts.gpu_type,
            opts.max_price_per_hour,
            min_vram_gb
                .map(|g| format!(" with ≥{g}GB VRAM"))
                .unwrap_or_default()
        );
    }

    let best = &offers[0];
    println!(
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
        let ok = dialoguer::Confirm::new()
            .with_prompt(format!(
                "Rent this pod on your Vast.ai account (~${:.3}/hr, billed until destroyed)?",
                best.dph_total
            ))
            .default(true)
            .interact()
            .context("Cancelled")?;
        if !ok {
            bail!("Aborted before renting a pod.");
        }
    }

    // ---------------------------------------------------------------
    // 2. Rent + boot — a marketplace host can be taken (rent fails) or a
    //    dud (rents but never boots). Both fall through to the next offer.
    // ---------------------------------------------------------------
    let label = format!("modl-pod-{}", spec.output.lora_name);
    // Disable Vast's auto-tmux so our SSH commands run in a plain shell.
    let onstart = "touch /root/.no_auto_tmux\n";
    let env = vec![("HF_HUB_OFFLINE".to_string(), "0".to_string())];

    let mut booted: Option<(u64, vast::Offer, SshTarget)> = None;
    for offer in offers.iter().take(5) {
        let id = match vast::create_instance(
            offer.id,
            vast::POD_IMAGE,
            onstart,
            &env,
            POD_DISK_GB,
            &label,
        )
        .await
        {
            Ok(id) => id,
            Err(e) => {
                println!(
                    "{} Offer {} unavailable ({e}), trying next...",
                    style("!").yellow(),
                    offer.id
                );
                continue;
            }
        };

        println!(
            "{} Rented instance {} (${:.3}/hr). If anything goes wrong: modl pod rm {}",
            style("✓").green(),
            style(id).bold(),
            offer.dph_total,
            id
        );

        match wait_for_instance(id).await {
            Ok(ssh) => {
                booted = Some((id, offer.clone(), ssh));
                break;
            }
            Err(e) => {
                println!("{} {e}", style("!").yellow());
                println!(
                    "{} Host never booted — destroying instance {} and trying the next offer...",
                    style("!").yellow(),
                    id
                );
                if let Err(e) = vast::destroy_instance(id).await {
                    println!(
                        "{} Could not destroy {id}: {e} — check with: modl pod ls",
                        style("✗").red()
                    );
                }
            }
        }
    }
    let (instance_id, rented_offer, ssh) =
        booted.context("No offer produced a working pod — try again shortly")?;
    let started_at = std::time::Instant::now();

    // Everything after boot runs inside a fallible block so the pod is
    // destroyed on error unless --keep-pod was passed.
    let result = drive_pod(&spec, &ssh, instance_id, hf_token.as_deref()).await;

    if opts.keep_pod {
        println!(
            "{} --keep-pod: instance {} is still running and billing. Destroy with: modl pod rm {}",
            style("⚠").yellow(),
            instance_id,
            instance_id
        );
    } else {
        println!(
            "{} Destroying instance {}...",
            style("→").cyan(),
            instance_id
        );
        match vast::destroy_instance(instance_id).await {
            Ok(()) => println!("{} Pod destroyed — billing stopped.", style("✓").green()),
            Err(e) => println!(
                "{} Could not destroy instance {instance_id}: {e}\n  Destroy it manually: modl pod rm {instance_id}",
                style("✗").red()
            ),
        }
    }

    let hours = started_at.elapsed().as_secs_f64() / 3600.0;
    println!(
        "  Pod time: {:.0}m — estimated cost ${:.2}",
        hours * 60.0,
        hours * rented_offer.dph_total
    );

    result
}

/// Bootstrap → train → sync back. Assumes the instance is booted;
/// the caller handles teardown.
async fn drive_pod(
    spec: &TrainJobSpec,
    ssh: &SshTarget,
    instance_id: u64,
    hf_token: Option<&str>,
) -> Result<()> {
    // ---------------------------------------------------------------
    // 3. Wait for sshd to accept connections (running != ssh-ready)
    // ---------------------------------------------------------------
    wait_for_ssh(ssh)?;
    let ssh = ssh.clone();

    // ---------------------------------------------------------------
    // 4. Ship worker + dataset + spec
    // ---------------------------------------------------------------
    let worker_root = resolve_worker_python_root()?;
    let dataset_path = PathBuf::from(&spec.dataset.path);

    println!("{} Uploading worker + dataset...", style("→").cyan());
    run_ssh_quiet(
        &ssh,
        &format!("mkdir -p {REMOTE_ROOT}/worker {REMOTE_ROOT}/dataset"),
    )?;
    rsync_to(
        &ssh,
        &worker_root.join("modl_worker"),
        &format!("{REMOTE_ROOT}/worker/"),
    )?;
    rsync_to(&ssh, &dataset_path, &format!("{REMOTE_ROOT}/dataset-up/"))?;
    // rsync of a dir path (no trailing slash) nests it: dataset-up/<basename>.
    // Normalize so the spec can always point at REMOTE_ROOT/dataset.
    let ds_name = dataset_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .context("Dataset path has no directory name")?;
    run_ssh_quiet(
        &ssh,
        &format!(
            "rm -rf {REMOTE_ROOT}/dataset && mv {REMOTE_ROOT}/dataset-up/{} {REMOTE_ROOT}/dataset",
            shell_quote(&ds_name)
        ),
    )?;

    let remote_spec = build_remote_spec(spec);
    let spec_yaml = serde_yaml::to_string(&remote_spec)?;
    let tmp_spec = std::env::temp_dir().join(format!("modl-pod-spec-{instance_id}.yaml"));
    std::fs::write(&tmp_spec, &spec_yaml)?;
    rsync_to(&ssh, &tmp_spec, &format!("{REMOTE_ROOT}/spec.yaml"))?;
    let _ = std::fs::remove_file(&tmp_spec);

    // ---------------------------------------------------------------
    // 5. Bootstrap: ai-toolkit @ pinned SHA + Python deps
    // ---------------------------------------------------------------
    println!(
        "{} Bootstrapping pod (ai-toolkit @ {} + deps, a few minutes)...",
        style("→").cyan(),
        &AITOOLKIT_PIN_SHA[..8]
    );
    run_ssh_streaming(&ssh, &bootstrap_script())?;

    // ---------------------------------------------------------------
    // 6. Launch training under nohup, stream events
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

    let token_env = hf_token
        .map(|t| format!("HF_TOKEN={} ", shell_quote(t)))
        .unwrap_or_default();
    let launch = format!(
        "cd {REMOTE_ROOT} && rm -f events.jsonl && \
         nohup env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 {token_env}\
         PYTHONPATH={REMOTE_ROOT}/worker MODL_AITOOLKIT_ROOT={REMOTE_ROOT}/ai-toolkit \
         {REMOTE_ROOT}/venv/bin/python -m modl_worker.main train --config {REMOTE_ROOT}/spec.yaml --job-id {job_id} \
         >> {REMOTE_ROOT}/events.jsonl 2>> {REMOTE_ROOT}/stderr.log & echo launched"
    );
    run_ssh_quiet(&ssh, &launch)?;
    println!(
        "{} Training started on pod — {}",
        style("→").cyan(),
        style(&job_id).dim()
    );

    let outcome = stream_events(&ssh, &job_id, &db)?;

    match outcome {
        TrainOutcome::Completed => {}
        TrainOutcome::Failed(msg) => {
            db.update_job_status(&job_id, "error")?;
            let tail = run_ssh_capture(&ssh, &format!("tail -n 30 {REMOTE_ROOT}/stderr.log"))
                .unwrap_or_default();
            bail!("Pod training failed: {msg}\n\nLast worker stderr:\n{tail}");
        }
    }

    // ---------------------------------------------------------------
    // 7. Sync artifacts back + register locally
    // ---------------------------------------------------------------
    println!("{} Syncing artifacts back...", style("→").cyan());
    let local_out = PathBuf::from(&spec.output.destination_dir);
    std::fs::create_dir_all(&local_out)?;
    rsync_from(&ssh, &format!("{REMOTE_ROOT}/output/"), &local_out)?;

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

    println!(
        "{} LoRA registered: {}",
        style("✓").green().bold(),
        collected.store_path.display()
    );
    println!(
        "  Try it: modl generate \"{} ...\" --lora {} --base {}",
        spec.params.trigger_word, spec.output.lora_name, spec.model.base_model_id
    );
    Ok(())
}

fn preflight(spec: &TrainJobSpec) -> Result<()> {
    vast::api_key()?;

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
            bail!("`{tool}` not found — pod training needs OpenSSH and rsync installed locally.");
        }
    }

    let ds = Path::new(&spec.dataset.path);
    if !ds.is_dir() {
        bail!("Dataset directory not found: {}", spec.dataset.path);
    }
    Ok(())
}

fn huggingface_token() -> Option<String> {
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

fn bootstrap_script() -> String {
    format!(
        r#"set -e
export DEBIAN_FRONTEND=noninteractive
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p {REMOTE_ROOT}
if [ ! -d {REMOTE_ROOT}/ai-toolkit/.git ]; then
  echo "[pod] cloning ai-toolkit..."
  git clone --quiet {AITOOLKIT_REPO_URL} {REMOTE_ROOT}/ai-toolkit
fi
cd {REMOTE_ROOT}/ai-toolkit
git fetch --quiet origin {AITOOLKIT_PIN_SHA} || git fetch --quiet origin
git checkout --quiet {AITOOLKIT_PIN_SHA}
echo "[pod] installing uv (image python/pip vary too much to trust)..."
python3 -m pip install --quiet uv 2>/dev/null || pip install --quiet uv
echo "[pod] creating managed python 3.11 venv (host-independent, like the local runtime)..."
python3 -m uv venv --quiet --python 3.11 {REMOTE_ROOT}/venv
echo "[pod] installing torch {TRAINER_TORCH_VERSION} + ai-toolkit + worker deps (uv, parallel)..."
# ai-toolkit pins diffusers to a git commit that predates what the worker
# needs — strip its diffusers pin (throwaway clone, edit in place so the
# `-r requirements_base.txt` include keeps working) and pin ours instead.
sed -i -E '/^(diffusers|git\+.*diffusers)/d' requirements*.txt
python3 -m uv pip install --quiet --python {REMOTE_ROOT}/venv/bin/python \
  "torch=={TRAINER_TORCH_VERSION}" "torchvision=={TRAINER_TORCHVISION_VERSION}" \
  -r requirements.txt \
  "diffusers>=0.38.0" "transformers>=4.51" accelerate "safetensors>=0.5" \
  pillow "gguf>=0.10.0" pyyaml
echo "[pod] bootstrap complete"
"#
    )
}

enum TrainOutcome {
    Completed,
    Failed(String),
}

/// Follow the remote events file, printing progress and persisting events.
/// Reconnects if the SSH tail drops (pods have flaky links).
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
        let reader = std::io::BufReader::new(stdout);

        for line in reader.lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(raw) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let Some(event) = parse_worker_event(&raw, job_id) else {
                continue;
            };
            if event.sequence != 0 && event.sequence <= last_seq {
                continue; // already seen before a reconnect
            }
            last_seq = event.sequence.max(last_seq);
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
                    println!("  [{stage}] step {step}/{total_steps}{loss_str}");
                }
                EventPayload::Log { level, message } if level != "debug" => {
                    println!("  {}", style(message).dim());
                }
                EventPayload::Warning { message, .. } => {
                    println!("  {} {message}", style("⚠").yellow());
                }
                EventPayload::Artifact { path, .. } => {
                    println!("  {} checkpoint: {path}", style("•").cyan());
                }
                EventPayload::Completed { .. } => {
                    let _ = child.kill();
                    return Ok(TrainOutcome::Completed);
                }
                EventPayload::Error { message, .. } => {
                    let _ = child.kill();
                    return Ok(TrainOutcome::Failed(message.clone()));
                }
                _ => {}
            }
        }

        // Tail ended without a terminal event — connection dropped.
        let _ = child.kill();
        reconnects += 1;
        if reconnects > 20 {
            bail!(
                "Lost connection to pod {} times — training may still be running.\n\
                 Reconnect manually: modl pod ls / ssh, events at {REMOTE_ROOT}/events.jsonl",
                reconnects
            );
        }
        println!(
            "{} SSH stream dropped — reconnecting ({reconnects})...",
            style("!").yellow()
        );
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}

async fn wait_for_instance(instance_id: u64) -> Result<SshTarget> {
    println!(
        "{} Waiting for instance to boot (usually 1-3 minutes)...",
        style("→").cyan()
    );
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(PROVISION_TIMEOUT_SECS);
    let mut last_status = String::new();

    loop {
        let inst = vast::get_instance(instance_id).await?;
        if inst.actual_status != last_status {
            println!("  status: {}", inst.actual_status);
            last_status = inst.actual_status.clone();
        }
        if inst.actual_status == "running"
            && let (Some(host), Some(port)) = (inst.ssh_host.clone(), inst.ssh_port)
        {
            return Ok(SshTarget { host, port });
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
    println!(
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

fn run_ssh_quiet(ssh: &SshTarget, cmd: &str) -> Result<()> {
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

fn run_ssh_capture(ssh: &SshTarget, cmd: &str) -> Result<String> {
    let out = Command::new("ssh")
        .args(ssh.base_args())
        .arg(cmd)
        .output()
        .context("ssh command failed to spawn")?;
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Run a remote script with live output (bootstrap etc.).
fn run_ssh_streaming(ssh: &SshTarget, script: &str) -> Result<()> {
    let status = Command::new("ssh")
        .args(ssh.base_args())
        .arg(format!("bash -c {}", shell_quote(script)))
        .status()
        .context("ssh command failed to spawn")?;
    if !status.success() {
        bail!("Remote bootstrap failed (exit {status}).");
    }
    Ok(())
}

fn rsync_to(ssh: &SshTarget, local: &Path, remote: &str) -> Result<()> {
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

fn rsync_from(ssh: &SshTarget, remote: &str, local: &Path) -> Result<()> {
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

fn shell_quote(s: &str) -> String {
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
    fn bootstrap_script_pins_ai_toolkit() {
        let script = bootstrap_script();
        assert!(script.contains(AITOOLKIT_PIN_SHA));
        assert!(script.contains("requirements.txt"));
        assert!(script.contains("unset HF_HUB_OFFLINE"));
        assert!(script.contains("uv venv --quiet --python 3.11"));
        assert!(script.contains(TRAINER_TORCH_VERSION));
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
