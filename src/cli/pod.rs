//! `modl pod` — manage Vast.ai instances rented with your own API key.
//!
//! `pod up` rents + bootstraps a persistent pod that later `train --pod` /
//! `pod run` jobs reuse; `pod ls`/`rm`/`ssh`/`exec` are the safety net and
//! debugging tools. Instances bill until destroyed — `pod rm` is the habit.

use anyhow::{Context, Result, bail};
use console::style;

use crate::core::pod::{self, PodOptions};
use crate::core::pod_state::{self, PodRecord};
use crate::core::vast;

/// The CLI's interactive rent confirmation — a dialoguer prompt injected into
/// [`PodOptions`] so `core::pod` never touches a TTY itself.
pub(crate) fn dialoguer_confirm() -> crate::core::pod::ConfirmRent {
    Box::new(|prompt: &str| {
        dialoguer::Confirm::new()
            .with_prompt(prompt.to_string())
            .default(true)
            .interact()
            .context("Cancelled")
    })
}

pub async fn ls(json: bool) -> Result<()> {
    let instances = vast::list_instances().await?;

    // Local records let us mark the active pod and show its age/label.
    let records = pod_state::load();
    let rec_by_id: std::collections::HashMap<u64, &PodRecord> =
        records.iter().map(|r| (r.instance_id, r)).collect();

    if json {
        let out: Vec<serde_json::Value> = instances
            .iter()
            .map(|i| {
                let rec = rec_by_id.get(&i.id);
                serde_json::json!({
                    "instance_id": i.id,
                    "gpu_name": i.gpu_name,
                    "status": i.actual_status,
                    "dph_total": i.dph_total,
                    "ssh_host": i.ssh_host,
                    "ssh_port": i.ssh_port,
                    // Tracked in pods.json — the pod train/run/pull reuse.
                    "active": rec.is_some(),
                    "label": rec.map(|r| r.label.clone()),
                    // All-in (gpu + storage) — dph_total above is Vast's live
                    // GPU rate and excludes the storage line.
                    "dph_all_in": rec.map(|r| r.dph_all_in()),
                    "cost_so_far": rec.map(|r| r.cost_so_far()),
                    // Bootstrapped and ready for jobs (fingerprint recorded).
                    "ready": rec.map(|r| r.bootstrap_fingerprint.is_some()).unwrap_or(false),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    if instances.is_empty() {
        println!("No Vast.ai instances on this account.");
        return Ok(());
    }

    println!(
        "{:<3}{:<12} {:<14} {:<10} {:>8} {:>8}  {}",
        "",
        style("ID").bold(),
        style("GPU").bold(),
        style("STATUS").bold(),
        style("$/HR").bold(),
        style("~COST").bold(),
        style("SSH").bold()
    );
    for i in &instances {
        let ssh = match (&i.ssh_host, i.ssh_port) {
            (Some(h), Some(p)) => format!("ssh -p {p} root@{h}"),
            _ => "-".to_string(),
        };
        // Active pod (in pods.json) gets a marker + running-cost estimate.
        let (marker, cost) = match rec_by_id.get(&i.id) {
            Some(r) => (
                style("*").green().to_string(),
                format!("${:.2}", r.cost_so_far()),
            ),
            None => (" ".to_string(), "-".to_string()),
        };
        println!(
            "{:<3}{:<12} {:<14} {:<10} {:>8.3} {:>8}  {}",
            marker, i.id, i.gpu_name, i.actual_status, i.dph_total, cost, ssh
        );
    }
    if !records.is_empty() {
        println!(
            "\n{} = active modl pod (reused by train --pod / pod run)",
            style("*").green()
        );
    }
    println!(
        "{} Instances bill until destroyed: modl pod rm <id>",
        style("!").yellow()
    );
    Ok(())
}

/// `modl pod up <gpu|auto>` — rent + bootstrap a persistent pod.
#[allow(clippy::too_many_arguments)]
pub async fn up(
    gpu: String,
    max_price: f64,
    disk: f64,
    min_vram: Option<u32>,
    fresh: bool,
    models: Vec<String>,
    yes: bool,
    json: bool,
) -> Result<()> {
    // 1. Reuse is automatic — an existing pod short-circuits unless --fresh.
    let mut replacing: Option<PodRecord> = None;
    if let Some(rec) = pod_state::active_pod().await? {
        if !fresh {
            eprintln!(
                "{} Pod {} already up — {}, ${:.3}/hr all-in. Reuse is automatic; pass {} to replace it.",
                style("✓").green(),
                rec.instance_id,
                rec.gpu_name,
                rec.dph_all_in(),
                style("--fresh").bold()
            );
            if !models.is_empty() {
                crate::core::pod_run::prepare(&pod::Pod::from(rec.clone()), &models)?;
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&up_result_json(&rec, true))?
                );
            } else {
                print_summary(&rec);
            }
            return Ok(());
        }
        // Don't destroy the working pod yet: if no offer fits the price cap,
        // the rent is declined, or every host duds out, we'd end with zero
        // pods and a lost warm cache. Provision the replacement first — a few
        // minutes of double billing is strictly cheaper than losing the old
        // pod for nothing.
        eprintln!(
            "{} --fresh: provisioning a replacement for pod {} — the old pod is destroyed only once the new one is up…",
            style("→").cyan(),
            rec.instance_id
        );
        replacing = Some(rec);
    }

    // 2. VRAM floor: for an explicit GPU the user chose deliberately (none);
    //    for "auto" require a floor so we don't rent a too-small card.
    let auto = matches!(gpu.to_lowercase().as_str(), "auto" | "any");
    let min_vram_gb = if auto {
        Some(min_vram.unwrap_or(24))
    } else {
        min_vram
    };
    if auto {
        eprintln!(
            "{} auto: picking the best-value GPU with ≥{}GB VRAM (override with --min-vram).",
            style("→").cyan(),
            min_vram_gb.unwrap()
        );
    }

    let opts = PodOptions {
        gpu_type: gpu.clone(),
        max_price_per_hour: max_price,
        disk_gb: disk,
        yes,
        keep_pod: true, // pod up is persistent by definition — never auto-destroys
        label: format!("modl-pod-{}", gpu),
        confirm_rent: Some(dialoguer_confirm()),
    };

    // 3. Provision + bootstrap.
    let pod_obj = pod::provision(&opts, min_vram_gb).await?;
    pod::bootstrap(&pod_obj)?;

    // 4. Persist the record (with the fingerprint we just bootstrapped to).
    let mut recorded = pod_obj.clone();
    recorded.bootstrap_fingerprint = Some(pod::bootstrap_fingerprint());
    let rec = recorded.to_record(&opts.label);
    pod_state::upsert(rec.clone())?;

    // 4b. --fresh: the replacement is up and bootstrapped — now destroy the
    //     old pod. A failed teardown doesn't fail the command (the new pod is
    //     live); the old record stays tracked and nagged about.
    if let Some(old) = replacing
        && old.instance_id != rec.instance_id
        && let Err(e) = pod::teardown(&pod::Pod::from(old)).await
    {
        eprintln!("{} {e}", style("✗").red());
    }

    // 5. Warm the store: install modl + pull the requested models so the
    //    first job doesn't pay the download.
    if !models.is_empty() {
        crate::core::pod_run::prepare(&pod_obj, &models)?;
    }

    // 6. Summary.
    eprintln!(
        "{} Pod {} up — {}, ${:.3}/hr all-in, billing until destroyed",
        style("✓").green().bold(),
        pod_obj.instance_id,
        pod_obj.gpu_name,
        pod_obj.dph_total + pod_obj.dph_storage
    );
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&up_result_json(&rec, false))?
        );
    } else {
        print_summary(&rec);
    }
    Ok(())
}

/// Machine-readable `pod up` result — everything an agent needs to reuse,
/// monitor, and eventually destroy the pod.
fn up_result_json(rec: &PodRecord, reused: bool) -> serde_json::Value {
    serde_json::json!({
        "instance_id": rec.instance_id,
        "gpu_name": rec.gpu_name,
        "dph_total": rec.dph_total,
        "dph_storage": rec.dph_storage,
        "dph_all_in": rec.dph_all_in(),
        "ssh_host": rec.ssh_host,
        "ssh_port": rec.ssh_port,
        "label": rec.label,
        "created_at": rec.created_at,
        "ready": rec.bootstrap_fingerprint.is_some(),
        "reused": reused,
    })
}

/// `modl pod exec [--id N] -- <cmd…>` — ssh passthrough for debugging.
pub async fn exec(id: Option<u64>, cmd: Vec<String>) -> Result<()> {
    if cmd.is_empty() {
        bail!("Provide a command after `--`, e.g. `modl pod exec -- nvidia-smi`.");
    }

    let (host, port) = match id {
        Some(i) => {
            let inst = vast::get_instance(i).await?;
            match (inst.ssh_host, inst.ssh_port) {
                (Some(h), Some(p)) => (h, p),
                _ => bail!(
                    "Instance {i} has no SSH details yet (status: {}).",
                    inst.actual_status
                ),
            }
        }
        None => {
            let rec = pod_state::active_pod().await?.context(
                "No active pod — run `modl pod up <gpu>` first, or pass --id <instance>.",
            )?;
            (rec.ssh_host, rec.ssh_port)
        }
    };

    let joined = cmd.join(" ");
    let status = pod::ssh_exec(&host, port, &joined)?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

/// `modl pod pull <run-id>` — fetch a finished run's artifacts from the
/// active pod. The re-attach path: runs are fire-and-forget on the pod, so
/// if the watching command died (closed laptop, dropped link), this brings
/// the results home directly over SSH.
pub async fn pull(run_id: String, dest: Option<std::path::PathBuf>, json: bool) -> Result<()> {
    let rec = pod_state::active_pod().await?.context(
        "No active pod — the run's pod may already be destroyed (artifacts die with it).",
    )?;
    let pod_obj = pod::Pod::from(rec.clone());

    let status = crate::core::pod_run::run_status(&pod_obj, &run_id)?;
    match status.as_str() {
        "completed" => {}
        "partial_failure" => eprintln!(
            "{} Run ended with partial failures — pulling the artifacts that completed.",
            style("⚠").yellow()
        ),
        "running" | "pending" => bail!(
            "Run {run_id} is still {status} on pod {} — try again when it finishes.",
            rec.instance_id
        ),
        other => bail!(
            "Run {run_id} on pod {} has status '{other}' — nothing to pull.",
            rec.instance_id
        ),
    }

    let (local_dir, artifacts) =
        crate::core::pod_run::export_run(&pod_obj, &run_id, dest.as_deref())?;
    // Same shape as `run --pod --json` so agents parse one schema.
    let result = serde_json::json!({
        "status": status,
        "run_id": run_id,
        "output_dir": local_dir,
        "images": artifacts,
    });
    // Record the synced-home marker the MCP tools key on. Without it, a
    // pulled run still answers job_status/list_run_outputs with "not synced
    // yet — use pod_pull" (circular), and after `pod rm` becomes a dead end.
    // Deliberately overwrites a reaper "failed" marker — the artifacts being
    // home is the truer, terminal answer.
    let result_path = crate::core::pod_run::run_result_path(&run_id);
    if let Some(parent) = result_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&result_path, result.to_string()) {
        eprintln!(
            "{} Could not record the synced-home marker ({e}) — job_status may still report the run as not synced.",
            style("⚠").yellow()
        );
    }
    if json {
        println!("{}", serde_json::to_string(&result)?);
    }
    Ok(())
}

/// `modl pod logs <run-id> [-f]` — tail a fire-and-forget run's log on the
/// active pod. The re-attach story for watching: `pod pull` brings artifacts
/// home, this brings the log stream back after the submitting watcher died.
pub async fn logs(run_id: String, follow: bool, lines: u32) -> Result<()> {
    let rec = pod_state::active_pod()
        .await?
        .context("No active pod — run logs live on the pod and die with it.")?;

    let log = format!("{}/runs/{}.log", pod::REMOTE_ROOT, run_id);
    let quoted = crate::core::pod::shell_quote(&log);
    let tail_flags = if follow {
        format!("-n {lines} -F")
    } else {
        format!("-n {lines}")
    };
    // One round-trip: a missing log gets a clear error instead of tail noise.
    let cmd = format!(
        "if [ -f {quoted} ]; then tail {tail_flags} {quoted}; \
         else echo 'No log for run {run_id} on this pod — check the run id (modl pod exec -- ls {}/runs/)' >&2; exit 2; fi",
        pod::REMOTE_ROOT
    );
    let status = pod::ssh_exec(&rec.ssh_host, rec.ssh_port, &cmd)?;
    if !status.success() {
        // Ctrl-C on -f lands here too — only propagate real failures.
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

pub async fn rm(instance_id: u64, yes: bool) -> Result<()> {
    if !yes {
        let ok = dialoguer::Confirm::new()
            .with_prompt(format!(
                "Destroy Vast.ai instance {instance_id}? (billing stops)"
            ))
            .default(true)
            .interact()?;
        if !ok {
            return Ok(());
        }
    }
    let outcome = vast::destroy_instance(instance_id).await?;
    // Drop it from pods.json too, whether or not it was a tracked modl pod.
    let _ = pod_state::remove(instance_id);
    match outcome {
        vast::DestroyOutcome::Destroyed => println!(
            "{} Instance {instance_id} destroyed — billing stopped.",
            style("✓").green()
        ),
        // Don't confirm "billing stopped" for an id Vast never knew: on a
        // typo the REAL pod is still billing — send the user to pod ls.
        vast::DestroyOutcome::AlreadyGone => println!(
            "{} Instance {instance_id} not found on Vast (already destroyed, or a mistyped id).\n  Check what is still running: modl pod ls",
            style("!").yellow()
        ),
    }
    Ok(())
}

pub async fn ssh(instance_id: u64) -> Result<()> {
    let inst = vast::get_instance(instance_id).await?;
    match (inst.ssh_host, inst.ssh_port) {
        (Some(host), Some(port)) => {
            println!("ssh -p {port} root@{host}");
            Ok(())
        }
        _ => anyhow::bail!(
            "Instance {instance_id} has no SSH details yet (status: {})",
            inst.actual_status
        ),
    }
}

fn print_summary(rec: &PodRecord) {
    println!("  SSH:    ssh -p {} root@{}", rec.ssh_port, rec.ssh_host);
    println!("  Reuse:  modl train --pod … / modl pod run … (automatic)");
    println!(
        "  Kill:   {}",
        style(format!("modl pod rm {}", rec.instance_id)).bold()
    );
}
