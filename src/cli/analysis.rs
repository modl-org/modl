use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

use crate::core::executor::read_worker_stdout;
use crate::core::job::{EventPayload, JobEvent};
use crate::core::runtime;
use crate::core::training::resolve_worker_python_root;

/// Result of an analysis worker run.
pub struct AnalysisResult {
    pub result_data: Option<serde_json::Value>,
    pub success: bool,
}

/// Try to submit an analysis job via the persistent worker Unix socket.
///
/// Returns `Ok(Some(receiver))` if the worker accepted the job,
/// `Ok(None)` if no worker is running (socket doesn't exist or refused).
#[cfg(unix)]
fn try_analysis_via_socket(
    action: &str,
    job_id: &str,
    spec_yaml: &str,
) -> Result<Option<mpsc::Receiver<JobEvent>>> {
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    let sock_path = crate::core::paths::modl_root().join("worker.sock");

    // Try to connect — if socket doesn't exist or daemon isn't running, return None
    let mut stream = match UnixStream::connect(&sock_path) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };

    // Parse the YAML spec into a JSON value so the worker gets structured data
    let spec_value: serde_json::Value = serde_yaml::from_str(spec_yaml)
        .context("Failed to parse spec YAML for socket submission")?;

    let request = serde_json::json!({
        "action": action,
        "job_id": job_id,
        "spec": spec_value,
    });

    let request_line = format!("{}\n", serde_json::to_string(&request)?);

    stream
        .write_all(request_line.as_bytes())
        .context("Failed to write to worker socket")?;
    stream
        .shutdown(std::net::Shutdown::Write)
        .context("Failed to shutdown socket write")?;

    // Read JSONL events from socket (same protocol as subprocess stdout)
    let (tx, rx) = mpsc::channel();
    let job_id_owned = job_id.to_string();
    thread::spawn(move || {
        read_worker_stdout(stream, &job_id_owned, tx);
    });

    Ok(Some(rx))
}

#[cfg(not(unix))]
fn try_analysis_via_socket(
    _action: &str,
    _job_id: &str,
    _spec_yaml: &str,
) -> Result<Option<mpsc::Receiver<JobEvent>>> {
    Ok(None)
}

/// Spawn a Python worker for an analysis operation and stream progress.
///
/// `worker_command` is the subcommand name (e.g. "score", "detect", "compare").
/// `spec_yaml` is the serialized job spec.
/// `quiet` suppresses progress output (for --json mode).
///
/// Tries the persistent worker socket first (`~/.modl/worker.sock`). If the
/// worker is not running, falls back to spawning a one-shot Python subprocess.
pub async fn spawn_analysis_worker(
    worker_command: &str,
    spec_yaml: &str,
    quiet: bool,
) -> Result<AnalysisResult> {
    let runtime_root = crate::core::paths::modl_root().join("runtime");
    let jobs_dir = runtime_root.join("jobs");
    std::fs::create_dir_all(&jobs_dir)
        .with_context(|| format!("Failed to create jobs dir: {}", jobs_dir.display()))?;

    let job_id = format!(
        "{}-{}",
        worker_command,
        chrono::Utc::now().format("%Y%m%d-%H%M%S")
    );
    let spec_path = jobs_dir.join(format!("{job_id}.yaml"));
    std::fs::write(&spec_path, spec_yaml)
        .with_context(|| format!("Failed to write spec: {}", spec_path.display()))?;

    // -----------------------------------------------------------------------
    // Try persistent worker socket first (warm path, no cold start)
    // -----------------------------------------------------------------------
    let (rx, mut child) =
        if let Some(socket_rx) = try_analysis_via_socket(worker_command, &job_id, spec_yaml)? {
            (socket_rx, None)
        } else {
            // -------------------------------------------------------------------
            // Cold-start fallback: spawn a one-shot Python subprocess
            // -------------------------------------------------------------------

            // Resolve Python
            let setup = runtime::setup_training(false).await?;
            if !setup.ready {
                anyhow::bail!("Python runtime is not ready. Run `modl train setup` first.");
            }

            let worker_root = resolve_worker_python_root()?;
            let mut py_path = worker_root.to_string_lossy().to_string();
            if let Ok(Some(aitk_dir)) = runtime::aitoolkit_path() {
                py_path = format!("{}:{}", py_path, aitk_dir.display());
            }
            if let Ok(current) = std::env::var("PYTHONPATH")
                && !current.trim().is_empty()
            {
                py_path = format!("{}:{}", py_path, current);
            }

            let mut spawned = Command::new(&setup.python_path)
                .arg("-m")
                .arg("modl_worker.main")
                .arg(worker_command)
                .arg("--config")
                .arg(&spec_path)
                .arg("--job-id")
                .arg(&job_id)
                .env("PYTHONPATH", py_path)
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .with_context(|| {
                    format!(
                        "Failed to start {} worker using {}",
                        worker_command,
                        setup.python_path.display()
                    )
                })?;

            let stdout = spawned
                .stdout
                .take()
                .context("Failed to capture worker stdout")?;

            let (tx, subprocess_rx) = mpsc::channel();
            let job_id_clone = job_id.clone();
            thread::spawn(move || {
                read_worker_stdout(stdout, &job_id_clone, tx);
            });

            (subprocess_rx, Some(spawned))
        };

    // -----------------------------------------------------------------------
    // Event processing loop — identical regardless of event source
    // -----------------------------------------------------------------------

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
        pb.set_message("loading model...");
        pb
    };

    let mut result_data: Option<serde_json::Value> = None;
    let mut had_error = false;

    for event in rx {
        match event.event {
            EventPayload::Progress {
                step, total_steps, ..
            } => {
                if total_steps > 0 {
                    pb.set_length(total_steps as u64);
                    pb.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.green} [{bar:30.cyan/dim}] {pos}/{len} {msg}",
                        )
                        .unwrap()
                        .progress_chars("━╸─"),
                    );
                }
                pb.set_position(step as u64);
                pb.set_message(format!("{}...", worker_command));
            }
            EventPayload::Result { data, .. } => {
                result_data = Some(data);
            }
            EventPayload::Log { message, .. }
                if message.contains("Loading")
                    || message.contains("loaded")
                    || message.contains("Downloading") =>
            {
                pb.set_message(message);
            }
            EventPayload::Warning { message, .. } => {
                pb.println(format!("  {} {}", style("⚠").yellow(), message));
            }
            EventPayload::Error {
                message,
                recoverable,
                ..
            } => {
                pb.println(format!("  {} {}", style("✗").red(), message));
                if !recoverable {
                    had_error = true;
                }
            }
            EventPayload::Completed { message } => {
                pb.finish_and_clear();
                if !quiet {
                    let msg = message.unwrap_or_else(|| format!("{} completed", worker_command));
                    println!("{} {}", style("✓").green().bold(), msg);
                }
            }
            _ => {}
        }
    }

    // Only wait for the child process if we spawned one (cold-start path).
    // The socket path has no child to wait for.
    let process_success = if let Some(ref mut c) = child {
        c.wait().context("Failed to wait for worker")?.success()
    } else {
        true
    };

    Ok(AnalysisResult {
        result_data,
        success: process_success && !had_error,
    })
}
