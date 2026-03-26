use anyhow::{Context, Result, bail};
use console::style;
use serde::Deserialize;
use std::sync::mpsc;

use crate::core::config::Config;
use crate::core::executor::{Executor, JobHandle, parse_worker_event};
use crate::core::hub::DEFAULT_API_BASE;
use crate::core::job::{EditJobSpec, EventPayload, GenerateJobSpec, JobEvent, TrainJobSpec};

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// Supported cloud providers for remote training/generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CloudProvider {
    Modal,
    Replicate,
    #[value(name = "runpod")]
    RunPod,
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Modal => write!(f, "modal"),
            Self::Replicate => write!(f, "replicate"),
            Self::RunPod => write!(f, "runpod"),
        }
    }
}

impl std::str::FromStr for CloudProvider {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "modal" => Ok(Self::Modal),
            "replicate" => Ok(Self::Replicate),
            "runpod" => Ok(Self::RunPod),
            _ => bail!("Unknown cloud provider: {s}. Supported: modal, replicate, runpod"),
        }
    }
}

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CreateSessionResponse {
    session_id: String,
    #[allow(dead_code)]
    state: String,
    #[allow(dead_code)]
    price_per_hour: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SessionStatusResponse {
    #[allow(dead_code)]
    session_id: String,
    state: String,
    price_per_hour: Option<f64>,
    #[allow(dead_code)]
    instance_host: Option<String>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SubmitJobResponse {
    job_id: String,
}

#[derive(Debug, Deserialize)]
struct EventsResponse {
    events: Vec<serde_json::Value>,
    job_status: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ArtifactEntry {
    artifact_type: Option<String>,
    r2_key: String,
    sha256: Option<String>,
    size_bytes: Option<u64>,
    download_url: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ArtifactsResponse {
    artifacts: Vec<ArtifactEntry>,
}

// ---------------------------------------------------------------------------
// CloudExecutor
// ---------------------------------------------------------------------------

/// Cloud executor — provisions a remote GPU via the modl-cloud API,
/// submits jobs, polls events, and downloads artifacts.
///
/// Uses the GPU sessions API (Vast.ai orchestrator) under the hood.
/// The CLI event loop doesn't know or care — same Executor trait.
pub struct CloudExecutor {
    #[allow(dead_code)]
    provider: CloudProvider,
    api_base: String,
    api_key: String,
    client: reqwest::Client,
    /// Active GPU session + job (set after submit)
    session_id: Option<String>,
    gpu_job_id: Option<String>,
}

impl CloudExecutor {
    /// Create a cloud executor. Reads API key from config or env.
    pub fn new(provider: CloudProvider) -> Result<Self> {
        let config = Config::load().ok();
        let cloud = config.as_ref().and_then(|c| c.cloud.as_ref());

        let api_base = cloud
            .and_then(|c| c.api_base.as_deref())
            .unwrap_or(DEFAULT_API_BASE)
            .trim_end_matches('/')
            .to_string();

        let api_key = cloud
            .and_then(|c| c.api_key.as_deref())
            .filter(|k| !k.trim().is_empty())
            .map(String::from)
            .or_else(|| std::env::var("MODL_API_KEY").ok())
            .context(
                "No cloud API key. Run `modl login` first, or set cloud.api_key in ~/.modl/config.yaml",
            )?;

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(300))
            .build()?;

        Ok(Self {
            provider,
            api_base,
            api_key,
            client,
            session_id: None,
            gpu_job_id: None,
        })
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    /// Resolve which GPU type to provision based on the base model.
    fn gpu_type_for_model(base_model: &str) -> &'static str {
        match base_model {
            // Small models can run on consumer GPUs
            "flux2-klein-4b" | "sdxl-base-1.0" | "sd-1.5" => "rtx3090",
            // Medium models need more VRAM
            "flux-schnell" | "flux-dev" | "z-image" | "z-image-turbo" | "chroma" => "rtx4090",
            // Large models need A100
            _ => "a100",
        }
    }
}

impl CloudExecutor {
    /// Destroy the GPU session (cleanup after training).
    pub async fn destroy_session(&self) -> Result<()> {
        let session_id = match &self.session_id {
            Some(id) => id,
            None => return Ok(()),
        };

        let resp = self
            .client
            .delete(format!("{}/gpu/sessions/{}", self.api_base, session_id))
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() || r.status().as_u16() == 204 => {
                eprintln!("  {} GPU session destroyed", style("✓").green());
            }
            Ok(r) => {
                eprintln!(
                    "  {} Could not destroy GPU session: HTTP {}",
                    style("⚠").yellow(),
                    r.status()
                );
            }
            Err(e) => {
                eprintln!(
                    "  {} Could not destroy GPU session: {e}",
                    style("⚠").yellow()
                );
            }
        }
        Ok(())
    }
}

impl Executor for CloudExecutor {
    fn submit(&mut self, spec: &TrainJobSpec) -> Result<JobHandle> {
        // Run async provisioning inside the existing tokio runtime.
        // block_in_place allows blocking the current thread without deadlocking.
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.submit_train_async(spec))
        })?;
        Ok(result)
    }

    fn submit_generate(&mut self, _spec: &GenerateJobSpec) -> Result<JobHandle> {
        bail!("Cloud generation is not yet available. Use local generation (remove --cloud flag).");
    }

    fn submit_edit(&mut self, _spec: &EditJobSpec) -> Result<JobHandle> {
        bail!("Cloud editing is not yet available. Use local editing (remove --cloud flag).");
    }

    fn cleanup(&mut self) -> Result<()> {
        if self.session_id.is_some() {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(self.destroy_session())
            })?;
        }
        Ok(())
    }

    fn events(&mut self, _job_id: &str) -> Result<mpsc::Receiver<JobEvent>> {
        let session_id = self.session_id.clone().context("No active GPU session")?;
        let gpu_job_id = self.gpu_job_id.clone().context("No active GPU job")?;

        let api_base = self.api_base.clone();
        let auth = self.auth_header();
        let client = self.client.clone();

        let (tx, rx) = mpsc::channel::<JobEvent>();

        // Spawn a tokio task that polls the API and sends events through the channel
        tokio::spawn(async move {
            poll_events_loop(client, api_base, auth, session_id, gpu_job_id, tx).await;
        });

        Ok(rx)
    }
}

impl CloudExecutor {
    /// Async implementation of training submission.
    async fn submit_train_async(&mut self, spec: &TrainJobSpec) -> Result<JobHandle> {
        let base_model = &spec.model.base_model_id;
        let gpu_type = Self::gpu_type_for_model(base_model);

        // 1. Create GPU session
        eprintln!(
            "  {} Provisioning {} GPU for {}...",
            style("☁").cyan(),
            gpu_type,
            base_model,
        );

        let session = self
            .client
            .post(format!("{}/gpu/sessions", self.api_base))
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .json(&serde_json::json!({
                "gpu_type": gpu_type,
                "idle_timeout": "30m",
                "models": [base_model],
            }))
            .send()
            .await
            .context("Failed to create GPU session")?;

        if !session.status().is_success() {
            let text = session.text().await.unwrap_or_default();
            bail!("Failed to create GPU session: {text}");
        }

        let session_resp: CreateSessionResponse = session.json().await?;
        self.session_id = Some(session_resp.session_id.clone());

        eprintln!(
            "  {} Session {} created, waiting for instance...",
            style("☁").cyan(),
            &session_resp.session_id[..8.min(session_resp.session_id.len())],
        );

        // 2. Poll until session is ready (up to 15 minutes)
        let price = self
            .wait_for_session_ready(&session_resp.session_id)
            .await?;

        if let Some(p) = price {
            eprintln!("  {} GPU ready (${:.2}/hr)", style("✓").green(), p,);
        } else {
            eprintln!("  {} GPU ready", style("✓").green());
        }

        // 3. Push dataset to hub
        eprintln!(
            "  {} Pushing dataset to hub ({} images)...",
            style("☁").cyan(),
            spec.dataset.image_count,
        );

        let dataset_hub_ref = self.push_dataset_to_hub(&spec.dataset).await?;
        eprintln!(
            "  {} Dataset pushed ({})",
            style("✓").green(),
            dataset_hub_ref
        );

        // 4. Submit training job to the GPU session
        let mut job_spec = serde_json::to_value(spec)?;
        // Inject the dataset hub ref so the agent can pull it
        job_spec["_dataset_hub_ref"] = serde_json::Value::String(dataset_hub_ref);

        let submit_resp = self
            .client
            .post(format!(
                "{}/gpu/sessions/{}/jobs",
                self.api_base, session_resp.session_id
            ))
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .json(&serde_json::json!({
                "job_type": "train",
                "spec": job_spec,
            }))
            .send()
            .await
            .context("Failed to submit training job")?;

        if !submit_resp.status().is_success() {
            let text = submit_resp.text().await.unwrap_or_default();
            bail!("Failed to submit training job: {text}");
        }

        let job_resp: SubmitJobResponse = submit_resp.json().await?;
        self.gpu_job_id = Some(job_resp.job_id.clone());

        eprintln!(
            "  {} Training job submitted ({})",
            style("✓").green(),
            &job_resp.job_id[..8.min(job_resp.job_id.len())],
        );

        Ok(JobHandle {
            job_id: job_resp.job_id,
            child_pid: None,
        })
    }

    /// Poll session status until it's ready or fails.
    async fn wait_for_session_ready(&self, session_id: &str) -> Result<Option<f64>> {
        let url = format!("{}/gpu/sessions/{}", self.api_base, session_id);

        for i in 0..180 {
            // 15 min at 5s intervals
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            let resp = self
                .client
                .get(&url)
                .header(reqwest::header::AUTHORIZATION, self.auth_header())
                .send()
                .await?;

            if !resp.status().is_success() {
                continue;
            }

            let status: SessionStatusResponse = resp.json().await?;

            match status.state.as_str() {
                "ready" | "idle" => return Ok(status.price_per_hour),
                "error" => {
                    let msg = status
                        .error_message
                        .unwrap_or_else(|| "Unknown error".to_string());
                    bail!("GPU session failed: {msg}");
                }
                "destroyed" => bail!("GPU session was destroyed unexpectedly"),
                state => {
                    if i % 6 == 0 {
                        // Print status every 30s
                        eprintln!("  {} Instance state: {state}...", style("·").dim());
                    }
                }
            }
        }

        bail!("GPU session did not become ready within 15 minutes");
    }

    /// Push dataset to hub via `modl hub push`. Returns hub ref (username/slug).
    async fn push_dataset_to_hub(&self, dataset: &crate::core::job::DatasetRef) -> Result<String> {
        use crate::core::hub::{CreateItemRequest, HubClient};

        let ds_path = std::path::Path::new(&dataset.path);
        if !ds_path.exists() {
            bail!("Dataset path does not exist: {}", dataset.path);
        }

        let hub = HubClient::from_config(true)?;
        let me = hub.me().await.context("Failed to get hub account")?;
        let username = me
            .username
            .as_deref()
            .context("No hub username — run `modl login` first")?;

        let slug = dataset
            .name
            .to_lowercase()
            .replace(' ', "-")
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect::<String>();

        // Create hub item if not exists
        let _ = hub
            .create_item(&CreateItemRequest {
                slug: slug.clone(),
                item_type: "dataset".to_string(),
                visibility: "private".to_string(),
                description: Some(format!("{} ({} images)", dataset.name, dataset.image_count)),
                tags: vec![],
                base_model: None,
                trigger_words: vec![],
            })
            .await;

        // Zip dataset (images + captions only, exclude caches)
        let tmp = std::env::temp_dir().join(format!("modl-dataset-{}.zip", std::process::id()));
        {
            let file = std::fs::File::create(&tmp)?;
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);

            for entry in walkdir::WalkDir::new(ds_path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let rel_path = entry.path().strip_prefix(ds_path)?;
                let name = rel_path.to_string_lossy();
                // Skip caches, hidden files, and non-training files
                if name.starts_with('.')
                    || name.starts_with("_latent_cache")
                    || name.contains("__pycache__")
                {
                    continue;
                }
                zip.start_file(name.as_ref(), options)?;
                let bytes = std::fs::read(entry.path())?;
                std::io::Write::write_all(&mut zip, &bytes)?;
            }
            zip.finish()?;
        }

        let zip_size = std::fs::metadata(&tmp)?.len();
        eprintln!(
            "    Dataset zipped: {:.1} MB (images + captions only)",
            zip_size as f64 / 1_048_576.0
        );

        // Push to hub
        let push_resp = hub.push_start(username, &slug).await?;

        crate::core::hub::upload_file_presigned(&push_resp.upload_url, &tmp, "application/zip")
            .await
            .context("Failed to upload dataset to hub")?;

        // Compute SHA256
        let file_bytes = std::fs::read(&tmp)?;
        let sha256 = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&file_bytes);
            format!("{:x}", hasher.finalize())
        };

        let metadata = serde_json::json!({
            "source": "modl-cloud-training",
            "dataset_name": dataset.name,
            "image_count": dataset.image_count,
            "caption_coverage": dataset.caption_coverage,
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

        let _ = std::fs::remove_file(&tmp);
        Ok(format!("{username}/{slug}"))
    }
}

/// Background task: poll GPU session events and forward to mpsc channel.
async fn poll_events_loop(
    client: reqwest::Client,
    api_base: String,
    auth: String,
    session_id: String,
    job_id: String,
    tx: mpsc::Sender<JobEvent>,
) {
    let mut after: u64 = 0;
    let mut consecutive_errors = 0u32;

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let url = format!(
            "{}/gpu/sessions/{}/jobs/{}/events?after={}",
            api_base, session_id, job_id, after
        );

        let resp = match client
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, &auth)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                consecutive_errors += 1;
                // 300 errors * 2s = 10 min tolerance for agent boot
                if consecutive_errors > 300 {
                    let _ = tx.send(JobEvent {
                        schema_version: "v1".into(),
                        job_id: job_id.clone(),
                        sequence: after + 1,
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        source: "modl_cloud".into(),
                        event: EventPayload::Error {
                            code: "POLL_FAILED".into(),
                            message: format!("Lost connection to cloud API: {e}"),
                            recoverable: false,
                            details: None,
                        },
                    });
                    break;
                }
                continue;
            }
        };

        if !resp.status().is_success() {
            consecutive_errors += 1;
            if consecutive_errors > 300 {
                break;
            }
            continue;
        }

        consecutive_errors = 0;

        let events_resp: EventsResponse = match resp.json().await {
            Ok(e) => e,
            Err(_) => continue,
        };

        for raw_event in &events_resp.events {
            let seq = raw_event
                .get("sequence")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if seq > after {
                after = seq;
            }

            // Parse the event using the same parser as local executor
            if let Some(event) = parse_worker_event(raw_event, &job_id) {
                let is_terminal = matches!(
                    event.event,
                    EventPayload::Completed { .. }
                        | EventPayload::Error { .. }
                        | EventPayload::Cancelled
                );

                if tx.send(event).is_err() {
                    return; // receiver dropped
                }

                if is_terminal {
                    return;
                }
            }
        }

        // Also check job status in case we missed the terminal event
        if events_resp.job_status != "queued" && events_resp.job_status != "running" {
            eprintln!(
                "  [poll] job_status={}, events={}",
                events_resp.job_status,
                events_resp.events.len()
            );
        }
        if events_resp.job_status == "completed" || events_resp.job_status == "failed" {
            // Give a moment for any final events to arrive
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;

            // Poll once more for any remaining events
            if let Ok(r) = client
                .get(format!(
                    "{}/gpu/sessions/{}/jobs/{}/events?after={}",
                    api_base, session_id, job_id, after
                ))
                .header(reqwest::header::AUTHORIZATION, &auth)
                .send()
                .await
                && let Ok(final_resp) = r.json::<EventsResponse>().await
            {
                for raw_event in &final_resp.events {
                    if let Some(event) = parse_worker_event(raw_event, &job_id) {
                        let _ = tx.send(event);
                    }
                }
            }

            // If job completed but we never got a Completed event, synthesize one
            if events_resp.job_status == "completed" {
                let _ = tx.send(JobEvent {
                    schema_version: "v1".into(),
                    job_id: job_id.clone(),
                    sequence: after + 1,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    source: "modl_cloud".into(),
                    event: EventPayload::Completed {
                        message: Some("Cloud training completed".to_string()),
                    },
                });
            } else {
                let _ = tx.send(JobEvent {
                    schema_version: "v1".into(),
                    job_id: job_id.clone(),
                    sequence: after + 1,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    source: "modl_cloud".into(),
                    event: EventPayload::Error {
                        code: "CLOUD_JOB_FAILED".into(),
                        message: "Cloud training job failed".to_string(),
                        recoverable: false,
                        details: None,
                    },
                });
            }

            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Credential resolution (legacy — kept for backward compat with provider keys)
// ---------------------------------------------------------------------------

/// Resolve API key for a provider from environment or config.
#[allow(dead_code)]
fn resolve_api_key(provider: CloudProvider) -> Result<String> {
    let env_vars: &[&str] = match provider {
        CloudProvider::Modal => &["MODAL_TOKEN_ID"],
        CloudProvider::Replicate => &["REPLICATE_API_TOKEN"],
        CloudProvider::RunPod => &["RUNPOD_API_KEY"],
    };

    for var in env_vars {
        if let Ok(val) = std::env::var(var)
            && !val.trim().is_empty()
        {
            return Ok(val);
        }
    }

    if let Ok(config) = crate::core::config::Config::load()
        && let Some(ref cloud) = config.cloud
    {
        let key = match provider {
            CloudProvider::Modal => cloud.modal_token.as_deref(),
            CloudProvider::Replicate => cloud.replicate_token.as_deref(),
            CloudProvider::RunPod => cloud.runpod_key.as_deref(),
        };
        if let Some(k) = key
            && !k.trim().is_empty()
        {
            return Ok(k.to_string());
        }
    }

    bail!(
        "No API key found for {}. Set {} or add it to ~/.modl/config.yaml under cloud.",
        provider,
        env_vars.join(" / ")
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_str() {
        assert_eq!(
            "modal".parse::<CloudProvider>().unwrap(),
            CloudProvider::Modal
        );
        assert_eq!(
            "Replicate".parse::<CloudProvider>().unwrap(),
            CloudProvider::Replicate
        );
        assert_eq!(
            "RUNPOD".parse::<CloudProvider>().unwrap(),
            CloudProvider::RunPod
        );
        assert!("unknown".parse::<CloudProvider>().is_err());
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(CloudProvider::Modal.to_string(), "modal");
        assert_eq!(CloudProvider::Replicate.to_string(), "replicate");
        assert_eq!(CloudProvider::RunPod.to_string(), "runpod");
    }

    #[test]
    fn test_gpu_type_for_model() {
        assert_eq!(
            CloudExecutor::gpu_type_for_model("flux2-klein-4b"),
            "rtx3090"
        );
        assert_eq!(CloudExecutor::gpu_type_for_model("flux-dev"), "rtx4090");
        assert_eq!(CloudExecutor::gpu_type_for_model("qwen-image"), "a100");
    }
}
