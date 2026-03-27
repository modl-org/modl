use anyhow::{Context, Result, bail};
use console::style;
use serde::{Deserialize, Serialize};

use crate::core::executor::{Executor, LocalExecutor};
use crate::core::job::{EventPayload, TrainJobSpec};

// ---------------------------------------------------------------------------
// Stubs for future interactive GPU commands
// ---------------------------------------------------------------------------

pub async fn attach(spec: &str, idle: &str) -> Result<()> {
    println!(
        "\n  {} GPU sessions are not yet available.\n",
        style("!").yellow()
    );
    println!(
        "  Requested: {} GPU with {} idle timeout",
        style(spec).bold(),
        idle
    );
    println!("  This feature is coming soon — see `modl gpu --help` for the planned commands.");
    println!();
    bail!("modl gpu attach is not yet implemented");
}

pub async fn detach() -> Result<()> {
    bail!("modl gpu detach is not yet implemented — no active GPU session");
}

pub async fn status() -> Result<()> {
    println!("No active GPU sessions.");
    Ok(())
}

pub async fn ssh() -> Result<()> {
    bail!("modl gpu ssh is not yet implemented — no active GPU session");
}

// ---------------------------------------------------------------------------
// GPU Agent — runs on Vast.ai instances, polls for jobs, executes locally
// ---------------------------------------------------------------------------

/// API response types for the agent endpoints.
#[derive(Debug, Deserialize)]
struct AgentJobResponse {
    job_id: String,
    job_type: String,
    spec: serde_json::Value,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct PresignResponse {
    upload_url: String,
    r2_key: String,
}

#[derive(Debug, Serialize)]
struct AgentEventsPayload {
    job_id: String,
    events: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct AgentStatusPayload {
    status: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct AgentArtifactPayload {
    job_id: String,
    artifact_type: String,
    r2_key: String,
    sha256: String,
    size_bytes: u64,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct PresignRequest {
    filename: String,
    content_type: String,
}

/// Run as a GPU agent on a remote Vast.ai instance.
///
/// This is an internal command called by the onstart script after the instance
/// boots. It polls the orchestrator API for queued jobs, executes them locally
/// via the Python worker, and reports results back.
pub async fn agent(session_token: &str, api_base: &str) -> Result<()> {
    let api_base = api_base.trim_end_matches('/');
    let session_id = std::env::var("MODL_SESSION_ID")
        .context("MODL_SESSION_ID env var required for agent mode")?;

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let auth_header = format!("Bearer {session_token}");

    // Exchange session token for a scoped hub API key
    let hub_key = exchange_for_hub_key(&client, api_base, &auth_header).await?;
    write_hub_config(api_base, &hub_key)?;

    eprintln!(
        "{} GPU agent started (session {})",
        style("→").cyan(),
        &session_id[..8.min(session_id.len())]
    );

    // Agent loop: poll for jobs, execute, repeat
    loop {
        match poll_for_job(&client, api_base, &auth_header, &session_id).await {
            Ok(Some(job)) => {
                eprintln!(
                    "{} Got job: {} (type: {})",
                    style("→").cyan(),
                    &job.job_id,
                    &job.job_type
                );

                if let Err(e) = execute_agent_job(&client, api_base, &auth_header, &job).await {
                    eprintln!("{} Job {} failed: {e:#}", style("✗").red(), &job.job_id);
                    // Report failure to API
                    let _ =
                        update_job_status(&client, api_base, &auth_header, &job.job_id, "failed")
                            .await;
                }
            }
            Ok(None) => {
                // No job available — wait and poll again
            }
            Err(e) => {
                eprintln!("{} Poll error: {e:#}", style("⚠").yellow());
            }
        }

        // Poll interval: 3 seconds
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
}

/// Poll the orchestrator for the next queued job.
async fn poll_for_job(
    client: &reqwest::Client,
    api_base: &str,
    auth: &str,
    session_id: &str,
) -> Result<Option<AgentJobResponse>> {
    let url = format!("{api_base}/gpu/agent/jobs/{session_id}");
    let resp = client
        .get(&url)
        .header(reqwest::header::AUTHORIZATION, auth)
        .send()
        .await
        .context("Failed to poll for jobs")?;

    if resp.status().as_u16() == 204 {
        return Ok(None);
    }

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("Poll failed: HTTP {status} {text}");
    }

    let job: AgentJobResponse = resp.json().await.context("Failed to parse job response")?;
    Ok(Some(job))
}

/// Execute a job received from the orchestrator.
async fn execute_agent_job(
    client: &reqwest::Client,
    api_base: &str,
    auth: &str,
    job: &AgentJobResponse,
) -> Result<()> {
    match job.job_type.as_str() {
        "train" => execute_train_job(client, api_base, auth, job).await,
        "generate" => {
            eprintln!("  generate jobs not yet supported in agent mode");
            bail!("generate not implemented in agent");
        }
        other => bail!("Unknown job type: {other}"),
    }
}

/// Execute a training job on the local GPU.
async fn execute_train_job(
    client: &reqwest::Client,
    api_base: &str,
    auth: &str,
    job: &AgentJobResponse,
) -> Result<()> {
    // Parse the spec
    let mut spec: TrainJobSpec =
        serde_json::from_value(job.spec.clone()).context("Failed to parse TrainJobSpec")?;

    // Pull dataset from hub if a hub ref is provided
    let dataset_hub_ref = job
        .spec
        .get("_dataset_hub_ref")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if !dataset_hub_ref.is_empty() {
        let dataset_path = pull_dataset_from_hub(dataset_hub_ref).await?;
        spec.dataset.path = dataset_path;
    }

    // Ensure output directory exists
    std::fs::create_dir_all(&spec.output.destination_dir)?;

    // Bootstrap local executor and submit job
    eprintln!("  Preparing training runtime...");
    let mut executor = LocalExecutor::from_runtime_setup().await?;
    let handle = executor.submit(&spec)?;
    let job_id_local = &handle.job_id;

    eprintln!("  Training started (local job: {job_id_local})");

    // Stream events from local executor → report to API
    let rx = executor.events(job_id_local)?;

    let mut event_batch: Vec<serde_json::Value> = Vec::new();
    let mut sequence: u64 = 0;
    let mut final_status = "completed";
    let mut last_hub_ref: Option<String> = None;

    for event in rx {
        sequence += 1;

        // Push each checkpoint to hub immediately as it's produced
        if let EventPayload::Artifact { ref path, .. } = event.event {
            let local_path = std::path::Path::new(path);
            if local_path.exists() && local_path.extension().is_some_and(|e| e == "safetensors") {
                let step = parse_step_from_filename(path);
                let is_final = step.is_none(); // no step suffix = final LoRA
                eprintln!(
                    "  Pushing checkpoint to hub: {}{}",
                    local_path.display(),
                    step.map(|s| format!(" (step {s})")).unwrap_or_default()
                );

                match hub_push_checkpoint(&spec.output.lora_name, local_path, &spec, step, is_final)
                    .await
                {
                    Ok(hub_ref) => {
                        eprintln!("  {} Published: {}", style("✓").green(), hub_ref);
                        last_hub_ref = Some(hub_ref);
                    }
                    Err(e) => {
                        eprintln!("{} Hub push failed: {e:#}", style("⚠").yellow());
                        // Report the error as an event so the CLI user can see it
                        sequence += 1;
                        let err_event = serde_json::json!({
                            "schema_version": "v1",
                            "job_id": job.job_id,
                            "sequence": sequence,
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                            "source": "modl_agent",
                            "event": {
                                "type": "log",
                                "level": "warning",
                                "message": format!("Hub push failed: {e:#}"),
                            },
                        });
                        let _ =
                            report_events(client, api_base, auth, &job.job_id, &[err_event]).await;
                    }
                }
            }
        }

        if matches!(
            event.event,
            EventPayload::Error { .. } | EventPayload::Cancelled
        ) {
            final_status = "failed";
        }

        let is_terminal = matches!(
            event.event,
            EventPayload::Completed { .. } | EventPayload::Error { .. } | EventPayload::Cancelled
        );

        // Hold back the Completed event — we'll send it after hub push
        // so the CLI sees hub_registered before completed.
        if matches!(event.event, EventPayload::Completed { .. }) {
            // Don't add to batch yet — will be sent after hub push
            break;
        }

        // Build cloud event envelope with the remote job_id
        let cloud_event = serde_json::json!({
            "schema_version": "v1",
            "job_id": job.job_id,
            "sequence": sequence,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "source": "modl_agent",
            "event": event.event,
        });
        event_batch.push(cloud_event);

        // Flush batch every 5 events or on terminal events
        if event_batch.len() >= 5 || is_terminal {
            let _ = report_events(client, api_base, auth, &job.job_id, &event_batch).await;
            event_batch.clear();
        }

        if is_terminal {
            break;
        }
    }

    // Flush remaining events
    if !event_batch.is_empty() {
        let _ = report_events(client, api_base, auth, &job.job_id, &event_batch).await;
    }

    // Report hub registration event (for the last pushed checkpoint)
    if final_status == "completed"
        && let Some(ref hub_ref) = last_hub_ref
    {
        sequence += 1;
        let hub_event = serde_json::json!({
            "schema_version": "v1",
            "job_id": job.job_id,
            "sequence": sequence,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "source": "modl_agent",
            "event": {
                "type": "result",
                "result_type": "hub_registered",
                "data": { "hub_ref": hub_ref }
            },
        });
        let _ = report_events(client, api_base, auth, &job.job_id, &[hub_event]).await;
    }

    // Now send the completed event (held back until after hub push)
    if final_status == "completed" {
        sequence += 1;
        let completed_event = serde_json::json!({
            "schema_version": "v1",
            "job_id": job.job_id,
            "sequence": sequence,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "source": "modl_agent",
            "event": { "type": "completed", "message": "Training completed" },
        });
        let _ = report_events(client, api_base, auth, &job.job_id, &[completed_event]).await;
    }

    // Update job status
    update_job_status(client, api_base, auth, &job.job_id, final_status).await?;

    eprintln!(
        "{} Job {} {}",
        if final_status == "completed" {
            style("✓").green()
        } else {
            style("✗").red()
        },
        &job.job_id,
        final_status
    );

    Ok(())
}

/// Pull a dataset from the hub and extract it. Returns the local path.
async fn pull_dataset_from_hub(hub_ref: &str) -> Result<String> {
    use crate::core::hub::HubClient;

    eprintln!("  Pulling dataset from hub ({hub_ref})...");

    let (username, slug) = hub_ref
        .split_once('/')
        .context("Invalid hub ref — expected username/slug")?;

    let hub = HubClient::from_config(true)?;
    let pull_resp = hub.pull(username, slug, None).await?;

    // Download the zip
    let bytes = reqwest::get(&pull_resp.download_url).await?.bytes().await?;

    // Extract to a dataset directory
    let dest = crate::core::paths::modl_root().join("datasets").join(slug);
    std::fs::create_dir_all(&dest)?;

    let cursor = std::io::Cursor::new(&bytes);
    let mut archive = zip::ZipArchive::new(cursor).context("Failed to open dataset zip")?;
    archive
        .extract(&dest)
        .context("Failed to extract dataset zip")?;

    eprintln!(
        "  {} Dataset ready: {} ({} files)",
        style("✓").green(),
        dest.display(),
        archive.len()
    );

    Ok(dest.to_string_lossy().to_string())
}

/// Report events back to the orchestrator API.
async fn report_events(
    client: &reqwest::Client,
    api_base: &str,
    auth: &str,
    job_id: &str,
    events: &[serde_json::Value],
) -> Result<()> {
    let url = format!("{api_base}/gpu/agent/events");
    let payload = AgentEventsPayload {
        job_id: job_id.to_string(),
        events: events.to_vec(),
    };

    let resp = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, auth)
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        eprintln!("  Warning: event report failed: {text}");
    }
    Ok(())
}

/// Update job status on the orchestrator.
async fn update_job_status(
    client: &reqwest::Client,
    api_base: &str,
    auth: &str,
    job_id: &str,
    status: &str,
) -> Result<()> {
    let url = format!("{api_base}/gpu/agent/jobs/{job_id}/status");
    let payload = AgentStatusPayload {
        status: status.to_string(),
    };

    let resp = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, auth)
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        eprintln!("  Warning: status update failed: {text}");
    }
    Ok(())
}

/// Upload a training artifact to R2 and register it in the DB.
#[allow(dead_code)]
async fn upload_artifact(
    client: &reqwest::Client,
    api_base: &str,
    auth: &str,
    job_id: &str,
    local_path: &std::path::Path,
    sha256: &str,
    size_bytes: u64,
) -> Result<()> {
    let filename = local_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("artifact.safetensors");

    // Get presigned upload URL
    let presign_url = format!("{api_base}/gpu/agent/upload/presign");
    let presign_req = PresignRequest {
        filename: filename.to_string(),
        content_type: "application/octet-stream".to_string(),
    };

    let resp = client
        .post(&presign_url)
        .header(reqwest::header::AUTHORIZATION, auth)
        .json(&presign_req)
        .send()
        .await
        .context("Failed to get presigned upload URL")?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        bail!("Presign failed: {text}");
    }

    let presign: PresignResponse = resp.json().await?;

    // Upload file to R2
    let file_bytes = tokio::fs::read(local_path).await?;
    let actual_size = file_bytes.len() as u64;

    let upload_resp = client
        .put(&presign.upload_url)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .body(file_bytes)
        .send()
        .await
        .context("Failed to upload artifact to R2")?;

    if !upload_resp.status().is_success() {
        bail!("R2 upload failed: HTTP {}", upload_resp.status());
    }

    // Register artifact in DB
    let register_url = format!("{api_base}/gpu/agent/artifacts");
    let register_payload = AgentArtifactPayload {
        job_id: job_id.to_string(),
        artifact_type: "lora".to_string(),
        r2_key: presign.r2_key.clone(),
        sha256: sha256.to_string(),
        size_bytes: if size_bytes > 0 {
            size_bytes
        } else {
            actual_size
        },
    };

    let resp = client
        .post(&register_url)
        .header(reqwest::header::AUTHORIZATION, auth)
        .json(&register_payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        eprintln!("  Warning: artifact registration failed: {text}");
    }

    eprintln!(
        "  Artifact uploaded: {} ({:.1} MB)",
        presign.r2_key,
        actual_size as f64 / 1_048_576.0
    );

    Ok(())
}

/// Exchange session token for a scoped hub API key via the orchestrator.
async fn exchange_for_hub_key(
    client: &reqwest::Client,
    api_base: &str,
    auth: &str,
) -> Result<String> {
    let url = format!("{api_base}/gpu/agent/exchange");
    let resp = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, auth)
        .send()
        .await
        .context("Failed to exchange session token for hub key")?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        bail!("Token exchange failed: {text}");
    }

    #[derive(Deserialize)]
    struct ExchangeResponse {
        api_key: String,
    }

    let exchange: ExchangeResponse = resp.json().await?;
    eprintln!(
        "  {} Hub credentials obtained (scoped, expiring)",
        style("✓").green()
    );
    Ok(exchange.api_key)
}

/// Write hub credentials to ~/.modl/config.yaml so hub push/pull works.
fn write_hub_config(api_base: &str, api_key: &str) -> Result<()> {
    let config_path = crate::core::paths::modl_root().join("config.yaml");

    // Read existing config or start from empty mapping
    let mut config: serde_yaml::Value = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        serde_yaml::from_str(&content).unwrap_or(serde_yaml::Value::Mapping(Default::default()))
    } else {
        // Create parent dirs if needed
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        serde_yaml::Value::Mapping(Default::default())
    };

    if let serde_yaml::Value::Mapping(ref mut map) = config {
        let mut cloud = serde_yaml::Mapping::new();
        cloud.insert(
            serde_yaml::Value::String("api_base".into()),
            serde_yaml::Value::String(api_base.to_string()),
        );
        cloud.insert(
            serde_yaml::Value::String("api_key".into()),
            serde_yaml::Value::String(api_key.to_string()),
        );
        map.insert(
            serde_yaml::Value::String("cloud".into()),
            serde_yaml::Value::Mapping(cloud),
        );
    }

    let yaml = serde_yaml::to_string(&config)?;
    std::fs::write(&config_path, yaml)?;
    eprintln!("  {} Hub credentials configured", style("✓").green());
    Ok(())
}

/// Parse step number from checkpoint filename.
/// Format: "{name}_{step:09d}.safetensors" → Some(step)
/// Final LoRA "{name}.safetensors" → None
fn parse_step_from_filename(path: &str) -> Option<u32> {
    let stem = std::path::Path::new(path).file_stem()?.to_str()?;
    let step_str = stem.rsplit('_').next()?;
    if step_str.chars().all(|c| c.is_ascii_digit()) && step_str.len() >= 6 {
        step_str.parse().ok()
    } else {
        None
    }
}

/// Zip sample images from the training output directory up to a given step.
fn zip_samples(
    output_dir: &str,
    up_to_step: Option<u32>,
) -> Option<(std::path::PathBuf, Vec<String>, usize)> {
    let samples_dir = std::path::Path::new(output_dir)
        .join(std::path::Path::new(output_dir).file_name()?)
        .join("samples");

    if !samples_dir.exists() {
        return None;
    }

    let mut image_files: Vec<_> = std::fs::read_dir(&samples_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (name.ends_with(".jpg") || name.ends_with(".png"))
                && if let Some(max_step) = up_to_step {
                    // Parse step from filename: {timestamp}__{step}_{idx}.jpg
                    name.split("__")
                        .nth(1)
                        .and_then(|s| s.split('_').next())
                        .and_then(|s| s.parse::<u32>().ok())
                        .is_some_and(|s| s <= max_step)
                } else {
                    true
                }
        })
        .collect();

    if image_files.is_empty() {
        return None;
    }

    image_files.sort_by_key(|a| a.file_name());

    // Collect unique prompts (by prompt index)
    let mut max_prompt_idx = 0u32;
    for f in &image_files {
        let name = f.file_name().to_string_lossy().to_string();
        if let Some(idx_str) = name.split("__").nth(1).and_then(|s| s.split('_').nth(1))
            && let Some(idx) = idx_str
                .split('.')
                .next()
                .and_then(|s| s.parse::<u32>().ok())
        {
            max_prompt_idx = max_prompt_idx.max(idx);
        }
    }

    let count = image_files.len();

    // Zip them
    let tmp = std::env::temp_dir().join(format!("modl-samples-{}.zip", std::process::id()));
    let file = std::fs::File::create(&tmp).ok()?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    for entry in &image_files {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Ok(bytes) = std::fs::read(entry.path()) {
            let _ = zip.start_file(&name, options);
            let _ = std::io::Write::write_all(&mut zip, &bytes);
        }
    }
    let _ = zip.finish();

    // We don't have prompt text here — return empty prompts (metadata has them)
    Some((tmp, vec![], count))
}

/// Push a checkpoint to the hub with metadata and samples.
async fn hub_push_checkpoint(
    lora_name: &str,
    lora_path: &std::path::Path,
    spec: &TrainJobSpec,
    step: Option<u32>,
    is_final: bool,
) -> Result<String> {
    use crate::core::hub::{CreateItemRequest, HubClient};

    let hub = HubClient::from_config(true)?;
    let me = hub.me().await.context("Failed to get hub account")?;
    let username = me.username.as_deref().unwrap_or("unknown");

    let slug = lora_name
        .to_lowercase()
        .replace(' ', "-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect::<String>();

    // Create hub item if not exists
    let _ = hub
        .create_item(&CreateItemRequest {
            slug: slug.clone(),
            item_type: "lora".to_string(),
            visibility: "private".to_string(),
            description: Some(format!("Cloud-trained LoRA: {lora_name}")),
            tags: vec!["cloud-trained".to_string()],
            base_model: Some(spec.model.base_model_id.clone()),
            trigger_words: if spec.params.trigger_word.is_empty() {
                vec![]
            } else {
                vec![spec.params.trigger_word.clone()]
            },
        })
        .await;

    // Start push
    let push_resp = hub.push_start(username, &slug).await?;

    // Upload LoRA
    crate::core::hub::upload_file_presigned(
        &push_resp.upload_url,
        lora_path,
        "application/octet-stream",
    )
    .await?;

    // Zip and upload samples if available
    let mut samples_count = 0usize;
    if let Some(ref samples_url) = push_resp.samples_upload_url
        && let Some((zip_path, _, count)) = zip_samples(&spec.output.destination_dir, step)
    {
        samples_count = count;
        match crate::core::hub::upload_file_presigned(samples_url, &zip_path, "application/zip")
            .await
        {
            Ok(_) => {
                eprintln!("  {} Uploaded {count} sample images", style("✓").green());
            }
            Err(e) => {
                eprintln!("{} Sample upload failed: {e:#}", style("⚠").yellow());
                samples_count = 0; // Don't claim samples in metadata
            }
        }
        let _ = std::fs::remove_file(&zip_path);
    }

    // Compute SHA256
    let file_bytes = std::fs::read(lora_path)?;
    let sha256 = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        format!("{:x}", hasher.finalize())
    };

    // Rich metadata
    let metadata = serde_json::json!({
        "source": "modl-cloud-agent",
        "base_model": spec.model.base_model_id,
        "trigger_words": [spec.params.trigger_word],
        "lora_type": format!("{:?}", spec.params.lora_type).to_lowercase(),
        "file_name": lora_path.file_name().and_then(|n| n.to_str()).unwrap_or("lora.safetensors"),
        "checkpoint_step": step,
        "is_checkpoint": step.is_some(),
        "is_final": is_final,
        "training": {
            "steps": spec.params.steps,
            "learning_rate": spec.params.learning_rate,
            "optimizer": format!("{:?}", spec.params.optimizer).to_lowercase(),
            "rank": spec.params.rank,
            "resolution": spec.params.resolution,
            "batch_size": spec.params.batch_size,
        },
        "dataset": {
            "name": spec.dataset.name,
            "image_count": spec.dataset.image_count,
        },
        "samples_r2_key": push_resp.samples_r2_key,
        "samples_count": samples_count,
    });

    hub.push_complete(
        username,
        &slug,
        &push_resp.version_id,
        file_bytes.len() as u64,
        &sha256,
        Some(metadata),
    )
    .await?;

    Ok(format!("{username}/{slug}"))
}

/// Push a LoRA artifact to the hub. Returns the hub reference (username/slug).
#[allow(dead_code)]
async fn hub_push_artifact(
    lora_name: &str,
    lora_path: &std::path::Path,
    base_model: &str,
    trigger_word: &str,
) -> Result<String> {
    use crate::core::hub::{CreateItemRequest, HubClient};

    let hub = HubClient::from_config(true)?;
    let me = hub.me().await.context("Failed to get hub account")?;
    let username = me.username.as_deref().unwrap_or("unknown");

    // Slugify the lora name
    let slug = lora_name
        .to_lowercase()
        .replace(' ', "-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect::<String>();

    // Create hub item if it doesn't exist (ignore 409 conflict)
    let create_req = CreateItemRequest {
        slug: slug.clone(),
        item_type: "lora".to_string(),
        visibility: "private".to_string(),
        description: Some(format!("Cloud-trained LoRA: {lora_name}")),
        tags: vec!["cloud-trained".to_string()],
        base_model: Some(base_model.to_string()),
        trigger_words: if trigger_word.is_empty() {
            vec![]
        } else {
            vec![trigger_word.to_string()]
        },
    };
    let _ = hub.create_item(&create_req).await; // ignore if already exists

    // Start push (get presigned upload URL)
    let push_resp = hub
        .push_start(username, &slug)
        .await
        .context("Failed to start hub push")?;

    // Upload the LoRA file
    crate::core::hub::upload_file_presigned(
        &push_resp.upload_url,
        lora_path,
        "application/octet-stream",
    )
    .await
    .context("Failed to upload LoRA to hub")?;

    // Compute SHA256
    let file_bytes = std::fs::read(lora_path)?;
    let sha256 = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        format!("{:x}", hasher.finalize())
    };

    // Complete push with metadata
    let metadata = serde_json::json!({
        "source": "modl-cloud-agent",
        "base_model": base_model,
        "trigger_words": [trigger_word],
        "file_name": lora_path.file_name().and_then(|n| n.to_str()).unwrap_or("lora.safetensors"),
    });

    hub.push_complete(
        username,
        &slug,
        &push_resp.version_id,
        file_bytes.len() as u64,
        &sha256,
        Some(metadata),
    )
    .await
    .context("Failed to complete hub push")?;

    Ok(format!("{username}/{slug}"))
}
