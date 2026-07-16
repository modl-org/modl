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

pub async fn ls() -> Result<()> {
    let instances = vast::list_instances().await?;
    if instances.is_empty() {
        println!("No Vast.ai instances on this account.");
        return Ok(());
    }

    // Local records let us mark the active pod and show its age/label.
    let records = pod_state::load();
    let rec_by_id: std::collections::HashMap<u64, &PodRecord> =
        records.iter().map(|r| (r.instance_id, r)).collect();

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
pub async fn up(
    gpu: String,
    max_price: f64,
    disk: f64,
    min_vram: Option<u32>,
    fresh: bool,
    models: Vec<String>,
    yes: bool,
) -> Result<()> {
    // 1. Reuse is automatic — an existing pod short-circuits unless --fresh.
    if let Some(rec) = pod_state::active_pod().await? {
        if !fresh {
            println!(
                "{} Pod {} already up — {}, ${:.3}/hr. Reuse is automatic; pass {} to replace it.",
                style("✓").green(),
                rec.instance_id,
                rec.gpu_name,
                rec.dph_total,
                style("--fresh").bold()
            );
            if !models.is_empty() {
                crate::core::pod_run::prepare(&pod::Pod::from(rec.clone()), &models)?;
            }
            print_summary(&rec);
            return Ok(());
        }
        println!(
            "{} --fresh: destroying active pod {} before provisioning a new one…",
            style("→").cyan(),
            rec.instance_id
        );
        pod::teardown(&pod::Pod::from(rec)).await?;
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
        println!(
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
    };

    // 3. Provision + bootstrap.
    let pod_obj = pod::provision(&opts, min_vram_gb).await?;
    pod::bootstrap(&pod_obj)?;

    // 4. Persist the record (with the fingerprint we just bootstrapped to).
    let mut recorded = pod_obj.clone();
    recorded.bootstrap_fingerprint = Some(pod::bootstrap_fingerprint());
    let rec = recorded.to_record(&opts.label);
    pod_state::upsert(rec.clone())?;

    // 5. Warm the store: install modl + pull the requested models so the
    //    first job doesn't pay the download.
    if !models.is_empty() {
        crate::core::pod_run::prepare(&pod_obj, &models)?;
    }

    // 6. Summary.
    println!(
        "{} Pod {} up — {}, ${:.3}/hr, billing until destroyed",
        style("✓").green().bold(),
        pod_obj.instance_id,
        pod_obj.gpu_name,
        pod_obj.dph_total
    );
    print_summary(&rec);
    Ok(())
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
pub async fn pull(run_id: String, dest: Option<std::path::PathBuf>) -> Result<()> {
    let rec = pod_state::active_pod().await?.context(
        "No active pod — the run's pod may already be destroyed (artifacts die with it).",
    )?;
    let pod_obj = pod::Pod::from(rec.clone());

    let status = crate::core::pod_run::run_status(&pod_obj, &run_id)?;
    match status.as_str() {
        "completed" => {}
        "partial_failure" => println!(
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

    crate::core::pod_run::export_run(&pod_obj, &run_id, dest.as_deref())?;
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
    vast::destroy_instance(instance_id).await?;
    // Drop it from pods.json too, whether or not it was a tracked modl pod.
    let _ = pod_state::remove(instance_id);
    println!(
        "{} Instance {instance_id} destroyed — billing stopped.",
        style("✓").green()
    );
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
