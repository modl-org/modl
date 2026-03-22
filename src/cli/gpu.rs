use anyhow::{Result, bail};
use console::style;

use crate::core::gpu_session::{self, GpuClient, SessionState};

pub async fn attach(spec: &str, idle: &str) -> Result<()> {
    println!(
        "{} Provisioning {} GPU (idle timeout: {})...",
        style("→").cyan(),
        style(spec).bold(),
        idle
    );

    let session = gpu_session::provision_session(spec, idle, &[]).await?;

    println!();
    println!(
        "  {} GPU session {} is {}",
        style("✓").green().bold(),
        style(&session.session_id).bold(),
        style(&session.state).green()
    );
    println!("  GPU type: {}", session.gpu_type);
    if let Some(price) = session.price_per_hour {
        println!("  Cost:     ${:.2}/hr", price);
    }
    if let Some(ref host) = session.instance_host {
        println!("  Host:     {}", host);
    }
    println!();
    println!(
        "  Run {} to use it, or {} to shut down.",
        style("modl generate \"...\" --attach-gpu").bold(),
        style("modl gpu detach").bold()
    );
    println!();

    Ok(())
}

pub async fn detach() -> Result<()> {
    let session = match gpu_session::load_session()? {
        Some(s) => s,
        None => bail!("No active GPU session. Nothing to detach."),
    };

    if session.state == SessionState::Destroyed {
        gpu_session::remove_session()?;
        println!("Session was already destroyed. Cleaned up local state.");
        return Ok(());
    }

    println!(
        "{} Destroying GPU session {}...",
        style("→").cyan(),
        style(&session.session_id).bold()
    );

    let client = GpuClient::from_session(&session)?;
    client.destroy_session(&session.session_id).await?;
    gpu_session::remove_session()?;

    println!("  {} GPU session destroyed.", style("✓").green().bold());

    Ok(())
}

pub async fn status() -> Result<()> {
    let session = match gpu_session::load_session()? {
        Some(s) => s,
        None => {
            println!("No active GPU sessions.");
            return Ok(());
        }
    };

    // Fetch live status from orchestrator
    let client = GpuClient::from_session(&session)?;
    match client.get_session(&session.session_id).await {
        Ok(status) => {
            println!("{} GPU Session", style("●").cyan().bold());
            println!("  ID:       {}", style(&status.session_id).bold());
            println!("  State:    {}", format_state(&status.state));
            println!("  GPU:      {}", status.gpu_type);
            println!("  Timeout:  {}", status.idle_timeout);
            println!("  Created:  {}", status.created_at);

            if let Some(price) = status.price_per_hour {
                println!("  Rate:     ${:.2}/hr", price);
            }
            if let Some(cost) = status.total_cost {
                println!("  Total:    ${:.2}", cost);
            }
            if let Some(runtime) = status.runtime_seconds {
                let hours = runtime / 3600;
                let mins = (runtime % 3600) / 60;
                println!("  Runtime:  {}h {}m", hours, mins);
            }
            if let Some(ref host) = status.instance_host {
                println!("  Host:     {}", host);
            }
            if let Some(ref msg) = status.error_message {
                println!("  Error:    {}", style(msg).red());
            }

            // Update local cache
            let updated = gpu_session::GpuSession {
                session_id: status.session_id,
                gpu_type: status.gpu_type,
                state: status.state,
                idle_timeout: status.idle_timeout,
                created_at: status.created_at,
                api_base: session.api_base,
                price_per_hour: status.price_per_hour,
                instance_host: status.instance_host,
                ssh_port: status.ssh_port,
            };

            if updated.state == SessionState::Destroyed {
                gpu_session::remove_session()?;
            } else {
                gpu_session::save_session(&updated)?;
            }
        }
        Err(e) => {
            println!(
                "{} GPU Session (cached — cannot reach orchestrator)",
                style("●").yellow()
            );
            println!("  ID:       {}", style(&session.session_id).bold());
            println!("  State:    {} (last known)", format_state(&session.state));
            println!("  GPU:      {}", session.gpu_type);
            println!("  Error:    {}", style(format!("{e:#}")).dim());
        }
    }

    Ok(())
}

pub async fn ssh() -> Result<()> {
    let session = match gpu_session::load_session()? {
        Some(s) => s,
        None => bail!("No active GPU session. Run `modl gpu attach` first."),
    };

    let host = session
        .instance_host
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Session has no SSH host. It may still be provisioning."))?;

    let port = session.ssh_port.unwrap_or(22);

    println!(
        "{} Connecting to {} (port {})...",
        style("→").cyan(),
        style(host).bold(),
        port
    );

    let status = std::process::Command::new("ssh")
        .arg("-p")
        .arg(port.to_string())
        .arg(format!("root@{host}"))
        .status()
        .context("Failed to launch ssh")?;

    if !status.success() {
        bail!("SSH exited with status: {}", status);
    }

    Ok(())
}

/// GPU agent: runs on a Vast.ai instance, polls orchestrator for jobs, executes locally.
///
/// This is a hidden subcommand invoked by the Docker entrypoint. It:
/// 1. Polls GET /gpu/agent/jobs/{session_id} for queued jobs
/// 2. Runs them locally via LocalExecutor
/// 3. Reports events back via POST /gpu/agent/events
/// 4. Reports final status via POST /gpu/agent/jobs/{job_id}/status
pub async fn agent(session_token: &str, api_base: &str) -> Result<()> {
    use crate::core::executor::LocalExecutor;
    use crate::core::job::{EventPayload, JobEvent};

    println!("[agent] Starting GPU agent, polling {api_base}");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")?;

    let auth_header = format!("Bearer {session_token}");

    // We need to discover our session_id. Poll until we get a job or do a simple
    // approach: extract session_id from the first successful poll.
    // The agent endpoint is /gpu/agent/jobs/{session_id} — but we don't know the
    // session_id. The orchestrator can look it up by token. Let's use a self-identify
    // endpoint pattern: try polling with a placeholder and let the server match by token.
    //
    // Actually, the server already validates the token → session mapping. We need to
    // know the session_id to form the URL. The entrypoint could pass it, but for
    // simplicity we'll add a small identify step.

    // Step 1: Identify our session via the token
    let session_id = identify_session(&client, api_base, &auth_header).await?;
    println!("[agent] Session ID: {session_id}");

    // Step 2: Set up executor. On Vast.ai, torch is pre-installed globally —
    // no venv needed. Try the runtime setup first, fall back to system Python.
    println!("[agent] Setting up local executor...");
    let mut executor = match LocalExecutor::from_runtime_setup().await {
        Ok(e) => e,
        Err(_) => {
            // Fall back to system Python (vastai/pytorch has torch globally)
            let python = std::path::PathBuf::from("/usr/bin/python3");
            let runtime_root = crate::core::paths::modl_root().join("runtime");
            std::fs::create_dir_all(&runtime_root).ok();
            println!("[agent] Using system Python: {}", python.display());
            LocalExecutor::new(python, runtime_root)
        }
    };

    // Step 3: Poll loop — look for jobs, run them, report back
    let poll_interval = std::time::Duration::from_secs(2);

    loop {
        tokio::time::sleep(poll_interval).await;

        // Poll for next job
        let poll_url = format!("{api_base}/gpu/agent/jobs/{session_id}");
        let resp = client
            .get(&poll_url)
            .header(reqwest::header::AUTHORIZATION, &auth_header)
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[agent] Poll error (retrying): {e}");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        if resp.status().as_u16() == 204 {
            // No jobs — keep polling
            continue;
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            eprintln!("[agent] Poll returned {status}: {text}");
            if status.as_u16() == 401 {
                // Token revoked — session destroyed, exit
                println!("[agent] Session token revoked, shutting down.");
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            continue;
        }

        // Parse the job
        let body: serde_json::Value = resp.json().await.context("Failed to parse job response")?;
        let job_id = body["job_id"].as_str().unwrap_or("").to_string();
        let job_type = body["job_type"].as_str().unwrap_or("").to_string();
        let spec = &body["spec"];

        println!("[agent] Received job {job_id} (type: {job_type})");

        // Ensure the model is available locally (pull if needed).
        // Use --variant fp16 to avoid interactive prompts in non-TTY.
        if let Some(model_id) = spec
            .get("model")
            .and_then(|m| m.get("base_model_id"))
            .and_then(|v| v.as_str())
        {
            println!("[agent] Ensuring model {model_id} is available...");
            let pull_status = std::process::Command::new("modl")
                .args(["pull", model_id, "--variant", "fp16"])
                .status();
            match pull_status {
                Ok(s) if s.success() => println!("[agent] Model {model_id} ready"),
                Ok(s) => eprintln!("[agent] Warning: modl pull {model_id} exited with {s}"),
                Err(e) => eprintln!("[agent] Warning: failed to run modl pull: {e}"),
            }
        }

        // Execute the job locally
        let result = run_agent_job(&mut executor, &job_type, spec, &job_id).await;

        // Collect events and report them
        match result {
            Ok(events) => {
                // Log all events for debugging
                for event in &events {
                    match &event.event {
                        EventPayload::Error { code, message, .. } => {
                            eprintln!("[agent] Event ERROR [{code}]: {message}");
                        }
                        EventPayload::Log { level, message } => {
                            eprintln!("[agent] Event LOG [{level}]: {message}");
                        }
                        EventPayload::Warning { code, message } => {
                            eprintln!("[agent] Event WARN [{code}]: {message}");
                        }
                        _ => {}
                    }
                }

                // Upload any artifact files to R2 before reporting events
                upload_artifacts(&client, api_base, &auth_header, &job_id, &events).await;

                // Report events to orchestrator
                report_events(&client, api_base, &auth_header, &job_id, &events).await;

                // Check if job completed or failed
                let final_status = events
                    .iter()
                    .rev()
                    .find_map(|e| match &e.event {
                        EventPayload::Completed { .. } => Some("completed"),
                        EventPayload::Error { .. } => Some("failed"),
                        _ => None,
                    })
                    .unwrap_or("completed");

                report_job_status(&client, api_base, &auth_header, &job_id, final_status).await;
                println!(
                    "[agent] Job {job_id} finished: {final_status} ({} events)",
                    events.len()
                );
            }
            Err(e) => {
                eprintln!("[agent] Job {job_id} failed: {e:#}");

                // Report error event
                let error_event = JobEvent {
                    schema_version: "v1".into(),
                    job_id: job_id.clone(),
                    sequence: 1,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    source: "modl_agent".into(),
                    event: EventPayload::Error {
                        code: "AGENT_ERROR".into(),
                        message: format!("{e:#}"),
                        recoverable: false,
                        details: None,
                    },
                };
                report_events(&client, api_base, &auth_header, &job_id, &[error_event]).await;
                report_job_status(&client, api_base, &auth_header, &job_id, "failed").await;
            }
        }
    }
}

/// Identify our session by token via the orchestrator.
/// Tries to poll for jobs — if we get 204 or 200, the session_id is valid.
/// Falls back to trying the /gpu/agent/whoami endpoint if available.
async fn identify_session(
    _client: &reqwest::Client,
    _api_base: &str,
    _auth_header: &str,
) -> Result<String> {
    // The orchestrator sets MODL_SESSION_ID env var in the onstart command
    if let Ok(id) = std::env::var("MODL_SESSION_ID") {
        return Ok(id);
    }

    // Try the whoami-like pattern: POST to agent/events with empty body to get 403
    // with session info. This is hacky — better to just have the entrypoint pass it.
    //
    // For now, require MODL_SESSION_ID env var.
    bail!(
        "MODL_SESSION_ID not set. The GPU agent requires the session ID to be passed \
         via the MODL_SESSION_ID environment variable."
    )
}

/// Fix up a job spec for remote execution on the agent.
///
/// The CLI sends `base_model_path` from the user's local store, which doesn't
/// exist on the Vast.ai instance. We resolve it to the *agent's* local store
/// path (after `modl pull`) so the Python worker can load from disk with
/// HF_HUB_OFFLINE=1. If the model isn't in the agent's store, we clear the
/// path and let the worker fall back to resolve_model_path (HF repo ID).
fn fixup_spec_for_remote(spec: &serde_json::Value) -> serde_json::Value {
    let mut spec = spec.clone();
    if let Some(model) = spec.get_mut("model").and_then(|m| m.as_object_mut()) {
        // Try to resolve base_model_path on the agent's store
        let resolved = model
            .get("base_model_id")
            .and_then(|v| v.as_str())
            .and_then(resolve_agent_model_path);
        match resolved {
            Some(local_path) => {
                model.insert("base_model_path".to_string(), serde_json::json!(local_path));
            }
            None => {
                model.remove("base_model_path");
            }
        }
    }
    // Also fix lora paths — the remote instance won't have local LoRA files
    if let Some(lora) = spec.get_mut("lora").and_then(|l| l.as_object_mut()) {
        lora.remove("path");
    }
    // Fix output dir to use the remote instance's temp dir
    if let Some(output) = spec.get_mut("output").and_then(|o| o.as_object_mut()) {
        output.insert(
            "output_dir".to_string(),
            serde_json::json!("/tmp/modl-output"),
        );
    }
    spec
}

/// Resolve a model ID to the agent's local store path (after `modl pull`).
/// Returns None if the model is not installed locally.
fn resolve_agent_model_path(model_id: &str) -> Option<String> {
    let db = crate::core::db::Database::open().ok()?;
    let installed = db.list_installed(None).ok()?;
    let gen_types = ["checkpoint", "diffusion_model"];

    for model in &installed {
        if (model.name == model_id || model.id == model_id)
            && gen_types.contains(&model.asset_type.as_str())
        {
            return Some(model.store_path.clone());
        }
    }

    None
}

/// Run a job locally and collect all events.
async fn run_agent_job(
    executor: &mut crate::core::executor::LocalExecutor,
    job_type: &str,
    spec: &serde_json::Value,
    job_id: &str,
) -> Result<Vec<crate::core::job::JobEvent>> {
    use crate::core::executor::Executor;
    use crate::core::job::{EditJobSpec, GenerateJobSpec, TrainJobSpec};

    let spec = fixup_spec_for_remote(spec);

    let handle = match job_type {
        "generate" => {
            let gen_spec: GenerateJobSpec =
                serde_json::from_value(spec.clone()).context("Failed to parse generate spec")?;
            executor.submit_generate(&gen_spec)?
        }
        "train" => {
            let train_spec: TrainJobSpec =
                serde_json::from_value(spec.clone()).context("Failed to parse train spec")?;
            executor.submit(&train_spec)?
        }
        "edit" => {
            let edit_spec: EditJobSpec =
                serde_json::from_value(spec.clone()).context("Failed to parse edit spec")?;
            executor.submit_edit(&edit_spec)?
        }
        other => bail!("Unknown job type: {other}"),
    };

    // Collect events from the executor channel
    let rx = executor.events(&handle.job_id)?;
    let mut events = Vec::new();
    let mut seq: u64 = 0;

    for event in rx {
        seq += 1;
        let mut event = event;
        event.sequence = seq;
        event.job_id = job_id.to_string(); // Use the orchestrator's job_id
        events.push(event);
    }

    Ok(events)
}

/// Upload artifact files to R2 via presigned URLs.
/// For each Artifact event with a local file path, gets a presigned URL from the
/// orchestrator and uploads the file directly to R2.
async fn upload_artifacts(
    client: &reqwest::Client,
    api_base: &str,
    auth_header: &str,
    _job_id: &str,
    events: &[crate::core::job::JobEvent],
) {
    use crate::core::job::EventPayload;

    for event in events {
        if let EventPayload::Artifact {
            path,
            sha256: artifact_sha,
            ..
        } = &event.event
        {
            let local_path = std::path::Path::new(path);
            if !local_path.exists() {
                eprintln!("[agent] Artifact file not found: {path}");
                continue;
            }

            let filename = local_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let content_type = if filename.ends_with(".png") {
                "image/png"
            } else {
                "application/octet-stream"
            };

            // Get presigned upload URL (agent endpoint — session-token auth)
            let presign_url = format!("{api_base}/gpu/agent/upload/presign");
            let presign_resp = client
                .post(&presign_url)
                .header(reqwest::header::AUTHORIZATION, auth_header)
                .json(&serde_json::json!({
                    "filename": filename,
                    "content_type": content_type,
                }))
                .send()
                .await;

            let presign_data = match presign_resp {
                Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("[agent] Failed to parse presign response: {e}");
                        continue;
                    }
                },
                Ok(r) => {
                    eprintln!("[agent] Presign failed: {}", r.status());
                    continue;
                }
                Err(e) => {
                    eprintln!("[agent] Presign request failed: {e}");
                    continue;
                }
            };

            let upload_url = match presign_data["upload_url"].as_str() {
                Some(u) => u.to_string(),
                None => {
                    eprintln!("[agent] No upload_url in presign response");
                    continue;
                }
            };
            let r2_key = presign_data["r2_key"].as_str().unwrap_or("").to_string();

            // Upload the file
            let file_bytes = match std::fs::read(local_path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[agent] Failed to read {path}: {e}");
                    continue;
                }
            };
            let file_size = file_bytes.len() as i64;

            match client
                .put(&upload_url)
                .header(reqwest::header::CONTENT_TYPE, content_type)
                .body(file_bytes)
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => {
                    println!("[agent] Uploaded artifact: {filename}");

                    // Register artifact in the DB so CLI can discover it
                    let artifact_type = if filename.ends_with(".png") || filename.ends_with(".jpg")
                    {
                        "image"
                    } else if filename.ends_with(".safetensors") {
                        "lora"
                    } else {
                        "file"
                    };

                    let sha256 = artifact_sha.clone().unwrap_or_default();

                    let register_url = format!("{api_base}/gpu/agent/artifacts");
                    let _ = client
                        .post(&register_url)
                        .header(reqwest::header::AUTHORIZATION, auth_header)
                        .json(&serde_json::json!({
                            "job_id": _job_id,
                            "artifact_type": artifact_type,
                            "r2_key": r2_key,
                            "sha256": sha256,
                            "size_bytes": file_size,
                        }))
                        .send()
                        .await;
                }
                Ok(r) => {
                    eprintln!("[agent] Upload failed for {filename}: {}", r.status());
                }
                Err(e) => {
                    eprintln!("[agent] Upload request failed for {filename}: {e}");
                }
            }
        }
    }
}

/// Report collected events to the orchestrator.
async fn report_events(
    client: &reqwest::Client,
    api_base: &str,
    auth_header: &str,
    job_id: &str,
    events: &[crate::core::job::JobEvent],
) {
    // Serialize events to JSON values
    let event_values: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| serde_json::to_value(e).ok())
        .collect();

    if event_values.is_empty() {
        return;
    }

    // Send in batches of 50
    for chunk in event_values.chunks(50) {
        let url = format!("{api_base}/gpu/agent/events");
        let body = serde_json::json!({
            "job_id": job_id,
            "events": chunk,
        });

        if let Err(e) = client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, auth_header)
            .json(&body)
            .send()
            .await
        {
            eprintln!("[agent] Failed to report events: {e}");
        }
    }
}

/// Report final job status to the orchestrator.
async fn report_job_status(
    client: &reqwest::Client,
    api_base: &str,
    auth_header: &str,
    job_id: &str,
    status: &str,
) {
    let url = format!("{api_base}/gpu/agent/jobs/{job_id}/status");
    let body = serde_json::json!({"status": status});

    if let Err(e) = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, auth_header)
        .json(&body)
        .send()
        .await
    {
        eprintln!("[agent] Failed to report job status: {e}");
    }
}

fn format_state(state: &SessionState) -> String {
    match state {
        SessionState::Ready | SessionState::Idle => style(state.to_string()).green().to_string(),
        SessionState::Busy => style(state.to_string()).yellow().to_string(),
        SessionState::Provisioning | SessionState::Installing => {
            style(state.to_string()).cyan().to_string()
        }
        SessionState::Error => style(state.to_string()).red().to_string(),
        SessionState::Destroying | SessionState::Destroyed => {
            style(state.to_string()).dim().to_string()
        }
    }
}

use anyhow::Context;
