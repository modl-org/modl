use anyhow::{Context, Result, bail};
use console::style;

use crate::core::db::Database;

pub async fn run(run_id: &str, json: bool, pod: bool) -> Result<()> {
    if pod {
        return run_on_pod(run_id, json).await;
    }
    let db = Database::open()?;

    let jobs = db.list_jobs_by_run_id(run_id)?;

    if jobs.is_empty() {
        bail!("No workflow run found with ID '{run_id}'");
    }

    // Aggregate status across step-jobs.
    let total = jobs.len();
    let completed = jobs.iter().filter(|j| j.status == "completed").count();
    let failed = jobs.iter().filter(|j| j.status == "error").count();
    let running = jobs.iter().filter(|j| j.status == "running").count();

    let cancelled = jobs.iter().filter(|j| j.status == "cancelled").count();
    // partial_failure / cancelled: all steps terminated, none still running.
    // Runs with pending steps and failures stay "pending" until all settle.
    let is_terminal = running == 0 && completed + failed + cancelled == total;
    let aggregate = if failed > 0 && is_terminal {
        "partial_failure"
    } else if running > 0 {
        "running"
    } else if completed == total {
        "completed"
    } else if cancelled > 0 && is_terminal {
        "cancelled"
    } else {
        "pending"
    };

    // Collect artifacts for all step-jobs.
    let mut artifacts: Vec<(String, String)> = Vec::new(); // (path, url)
    let base_url = std::env::var("MODL_BASE_URL").ok();

    for job in &jobs {
        let arts = db.list_artifacts(Some(&job.job_id)).unwrap_or_default();
        for a in arts {
            let url = match &base_url {
                Some(base) => {
                    // Strip the local root prefix and build a URL.
                    let local_root = crate::core::paths::modl_root();
                    let rel = std::path::Path::new(&a.path)
                        .strip_prefix(&local_root)
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|_| a.path.clone());
                    format!("{base}/files/{rel}")
                }
                None => a.path.clone(),
            };
            artifacts.push((a.path.clone(), url));
        }
    }

    if json {
        let out = serde_json::json!({
            "run_id": run_id,
            "status": aggregate,
            "steps": {
                "total": total,
                "completed": completed,
                "failed": failed,
                "running": running,
                "cancelled": cancelled,
            },
            "artifacts": artifacts.iter().map(|(path, url)| serde_json::json!({
                "path": path,
                "url": url,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // Human output
    println!("{}", style("Workflow run status").bold().cyan());
    println!();
    println!("  {}  {}", style("Run ID:").dim(), run_id);
    println!(
        "  {}  {}",
        style("Status:").dim(),
        match aggregate {
            "completed" => style(aggregate).green().to_string(),
            "running" => style(aggregate).yellow().to_string(),
            "partial_failure" => style(aggregate).red().to_string(),
            "cancelled" => style(aggregate).yellow().to_string(),
            _ => style(aggregate).dim().to_string(),
        }
    );
    println!(
        "  {}  {}/{} steps complete{}",
        style("Steps:").dim(),
        completed,
        total,
        if failed > 0 {
            format!(", {} failed", failed)
        } else {
            String::new()
        }
    );

    if !artifacts.is_empty() {
        println!();
        println!(
            "  {} {} artifact{}",
            style("Outputs:").dim(),
            artifacts.len(),
            if artifacts.len() == 1 { "" } else { "s" }
        );
        for (path, url) in &artifacts {
            if base_url.is_some() {
                println!("    {}", style(url).underlined().cyan());
            } else {
                println!("    {}", style(path).dim());
            }
        }
    }

    if aggregate != "completed" && aggregate != "partial_failure" && aggregate != "cancelled" {
        println!();
        println!(
            "  {} Re-run {} to refresh.",
            style("ℹ").dim(),
            style(format!("modl status {run_id}")).cyan()
        );
    } else if base_url.is_none() && !artifacts.is_empty() {
        println!();
        println!(
            "  {} Set {} to get HTTP URLs for remote download.",
            style("ℹ").dim(),
            style("MODL_BASE_URL=http://server:PORT").cyan()
        );
    }

    Ok(())
}

/// `modl status <run-id> --pod` — proxy the active pod's own status. A pod
/// run's step-jobs live in the pod's DB; locally it only becomes visible
/// once its artifacts are pulled home. The JSON is the pod's status payload
/// verbatim plus `"target": "pod"` (artifact paths in it are pod-side).
async fn run_on_pod(run_id: &str, json: bool) -> Result<()> {
    let rec = crate::core::pod_state::active_pod()
        .await?
        .context("No active pod — pod run state dies with the pod (modl pod ls).")?;
    let pod = crate::core::pod::Pod::from(rec.clone());

    let mut v = crate::core::pod_run::run_status_json(&pod, run_id)?;
    if let Some(obj) = v.as_object_mut() {
        obj.insert("target".into(), serde_json::json!("pod"));
        obj.insert("instance_id".into(), serde_json::json!(rec.instance_id));
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("?");
    let steps = v.get("steps").cloned().unwrap_or_default();
    println!("{}", style("Workflow run status (on pod)").bold().cyan());
    println!();
    println!("  {}  {}", style("Run ID:").dim(), run_id);
    println!(
        "  {}  {} (pod {})",
        style("Status:").dim(),
        status,
        rec.instance_id
    );
    println!(
        "  {}  {}/{} steps complete{}",
        style("Steps:").dim(),
        steps.get("completed").and_then(|x| x.as_u64()).unwrap_or(0),
        steps.get("total").and_then(|x| x.as_u64()).unwrap_or(0),
        match steps.get("failed").and_then(|x| x.as_u64()).unwrap_or(0) {
            0 => String::new(),
            n => format!(", {n} failed"),
        }
    );
    if status == "completed" || status == "partial_failure" {
        println!();
        println!(
            "  {} Bring artifacts home: {}",
            style("ℹ").dim(),
            style(format!("modl pod pull {run_id}")).cyan()
        );
    }
    Ok(())
}
