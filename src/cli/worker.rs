use anyhow::{Context, Result, bail};
use console::style;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use crate::core::runtime;
use crate::core::training::resolve_worker_python_root;

/// Path to the worker Unix socket.
fn socket_path() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("worker.sock")
}

/// Path to the worker PID file.
fn pid_path() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("worker.pid")
}

/// Check if a process with the given PID is alive.
fn is_pid_alive(pid: u32) -> bool {
    // kill -0 checks process existence without sending a signal
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Read the PID from the PID file, if it exists and the process is alive.
fn read_worker_pid() -> Option<u32> {
    let path = pid_path();
    let content = std::fs::read_to_string(&path).ok()?;
    let pid: u32 = content.trim().parse().ok()?;
    if is_pid_alive(pid) {
        Some(pid)
    } else {
        // Stale PID file — clean up
        let _ = std::fs::remove_file(&path);
        None
    }
}

/// Try to connect to the worker socket.
pub fn try_connect() -> Option<UnixStream> {
    let sock = socket_path();
    UnixStream::connect(&sock).ok()
}

/// Check if the worker is running (PID alive + socket connectable).
pub fn is_worker_running() -> bool {
    read_worker_pid().is_some() && try_connect().is_some()
}

/// Start the persistent worker daemon in the background.
pub async fn start(timeout: u32) -> Result<()> {
    // Check if already running
    if is_worker_running() {
        let pid = read_worker_pid().unwrap_or(0);
        println!(
            "{} Worker already running (PID {})",
            style("●").green(),
            pid
        );
        return Ok(());
    }

    // Clean up stale socket
    let sock = socket_path();
    if sock.exists() {
        let _ = std::fs::remove_file(&sock);
    }

    // Resolve Python path
    let setup = runtime::setup_training(false).await?;
    if !setup.ready {
        bail!("Training runtime is not ready. Run `modl train setup` first.");
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

    // Redirect worker stderr to a log file so diffusers/torch can write
    // freely without hitting a broken pipe when the parent process exits.
    let modl_dir = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl");
    let log_path = modl_dir.join("worker.log");
    let log_file = std::fs::File::create(&log_path)
        .with_context(|| format!("Failed to create worker log: {}", log_path.display()))?;

    // Spawn worker as a background process
    let mut command = Command::new(&setup.python_path);
    command
        .arg("-m")
        .arg("modl_worker.main")
        .arg("serve")
        .arg("--timeout")
        .arg(timeout.to_string())
        .env("PYTHONPATH", py_path)
        .stdout(std::process::Stdio::null())
        .stderr(log_file);

    // Detach from parent process group so it survives terminal close
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }

    let child = command.spawn().context("Failed to start worker process")?;
    let child_pid = child.id();

    println!(
        "{} Starting worker (PID {})...",
        style("→").cyan(),
        child_pid
    );

    // Wait for the socket to appear (up to 30s)
    let start = std::time::Instant::now();
    let max_wait = Duration::from_secs(30);
    loop {
        if start.elapsed() > max_wait {
            bail!("Worker did not start within 30s. Check logs or run `modl worker status`.");
        }

        if sock.exists()
            && let Ok(mut stream) = UnixStream::connect(&sock)
        {
            // Send a ping to verify it's responsive
            let _ = stream.write_all(b"{\"action\":\"ping\"}\n");
            stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut buf = [0u8; 1024];
            if stream.read(&mut buf).is_ok() {
                println!(
                    "{} Worker started (PID {}, socket: {})",
                    style("✓").green().bold(),
                    child_pid,
                    sock.display()
                );
                return Ok(());
            }
        }

        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Stop the running worker daemon.
pub async fn stop() -> Result<()> {
    // Try graceful shutdown via socket first
    if let Some(mut stream) = try_connect() {
        let _ = stream.write_all(b"{\"action\":\"shutdown\"}\n");
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        let mut buf = [0u8; 1024];
        let _ = stream.read(&mut buf);
        // Give it a moment to clean up
        std::thread::sleep(Duration::from_millis(500));
    }

    // If still alive, send SIGTERM
    if let Some(pid) = read_worker_pid() {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .status();
        std::thread::sleep(Duration::from_millis(500));
    }

    // Clean up files
    let sock = socket_path();
    if sock.exists() {
        let _ = std::fs::remove_file(&sock);
    }
    let pid_file = pid_path();
    if pid_file.exists() {
        let _ = std::fs::remove_file(&pid_file);
    }

    println!("{} Worker stopped", style("✓").green().bold());
    Ok(())
}

/// Show worker status.
pub async fn status() -> Result<()> {
    let pid = read_worker_pid();

    if pid.is_none() {
        println!("{} Worker: {}", style("●").red(), style("stopped").dim());
        return Ok(());
    }

    let pid = pid.unwrap();

    // Connect and get status
    let mut stream = match try_connect() {
        Some(s) => s,
        None => {
            println!(
                "{} Worker: {} (PID {} but socket not responding)",
                style("●").yellow(),
                style("stale").yellow(),
                pid
            );
            return Ok(());
        }
    };

    stream.write_all(b"{\"action\":\"status\"}\n")?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();

    let mut response = String::new();
    let mut reader = BufReader::new(&stream);
    reader.read_line(&mut response)?;

    let status: serde_json::Value =
        serde_json::from_str(response.trim()).context("Failed to parse worker status response")?;

    let uptime = status
        .get("uptime_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let idle = status
        .get("idle_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let idle_timeout = status
        .get("idle_timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let jobs_served = status
        .get("jobs_served")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!(
        "{} Worker: {} (PID {}, uptime {})",
        style("●").green(),
        style("running").green(),
        pid,
        format_duration(uptime)
    );
    println!("  Socket: {}", socket_path().display());
    println!("  Jobs served: {}", jobs_served);

    // Show loaded models
    if let Some(cache) = status.get("cache") {
        let models = cache
            .get("models")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let max = cache
            .get("max_models")
            .and_then(|v| v.as_u64())
            .unwrap_or(2);

        if models.is_empty() {
            println!("  Models loaded: none (0/{})", max);
        } else {
            println!("  Models loaded ({}/{}):", models.len(), max);
            for model in &models {
                let model_id = model
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let vram = model
                    .get("vram_estimate_mb")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let last_used = model
                    .get("last_used")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let ago = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()
                    - last_used) as u64;
                let lora_id = model.get("lora_id").and_then(|v| v.as_str());

                print!(
                    "    {} ({}, ~{} MB VRAM)    last used: {}",
                    style(model_id).bold(),
                    model
                        .get("dtype")
                        .and_then(|v| v.as_str())
                        .unwrap_or("bf16"),
                    vram,
                    format_duration(ago)
                );
                if let Some(lora) = lora_id {
                    let weight = model
                        .get("lora_weight")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);
                    print!(
                        "\n    {} LoRA: {} (weight={:.1})",
                        style("└─").dim(),
                        lora,
                        weight
                    );
                }
                println!();
            }
        }
    }

    println!(
        "  Idle timeout: {} ({} remaining)",
        format_duration(idle_timeout),
        format_duration(idle_timeout.saturating_sub(idle))
    );

    Ok(())
}

/// Spawn the worker automatically if not running. Returns true if worker is
/// available after this call. Used by the executor before attempting socket
/// connection.
#[allow(dead_code)]
pub async fn auto_spawn_if_needed() -> bool {
    if is_worker_running() {
        return true;
    }

    // Try to start with default timeout
    match start(600).await {
        Ok(()) => true,
        Err(e) => {
            eprintln!(
                "{} Auto-spawn worker failed: {}. Falling back to one-shot mode.",
                style("⚠").yellow(),
                e
            );
            false
        }
    }
}

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else {
        format!("{:.1}h", secs as f64 / 3600.0)
    }
}
