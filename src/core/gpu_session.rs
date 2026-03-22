use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::core::hub::HubClient;

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/// Possible states of a remote GPU session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Provisioning,
    Installing,
    Ready,
    Busy,
    Idle,
    Destroying,
    Destroyed,
    Error,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provisioning => write!(f, "provisioning"),
            Self::Installing => write!(f, "installing"),
            Self::Ready => write!(f, "ready"),
            Self::Busy => write!(f, "busy"),
            Self::Idle => write!(f, "idle"),
            Self::Destroying => write!(f, "destroying"),
            Self::Destroyed => write!(f, "destroyed"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ---------------------------------------------------------------------------
// Local session file (persisted to ~/.modl/gpu_session.json)
// ---------------------------------------------------------------------------

/// Locally persisted GPU session info. Stored at `~/.modl/gpu_session.json`
/// so subsequent commands can find the active session without re-querying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSession {
    pub session_id: String,
    pub gpu_type: String,
    pub state: SessionState,
    pub idle_timeout: String,
    pub created_at: String,
    /// Orchestrator API base URL
    pub api_base: String,
    /// Price per hour in USD (from orchestrator)
    #[serde(default)]
    pub price_per_hour: Option<f64>,
    /// Instance IP/hostname (for SSH)
    #[serde(default)]
    pub instance_host: Option<String>,
    /// SSH port
    #[serde(default)]
    pub ssh_port: Option<u16>,
}

/// Path to the local session file.
fn session_path() -> PathBuf {
    crate::core::paths::modl_root().join("gpu_session.json")
}

/// Load the active GPU session from disk, if any.
pub fn load_session() -> Result<Option<GpuSession>> {
    let path = session_path();
    if !path.exists() {
        return Ok(None);
    }
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let session: GpuSession =
        serde_json::from_str(&contents).context("Failed to parse gpu_session.json")?;
    Ok(Some(session))
}

/// Save a GPU session to disk.
pub fn save_session(session: &GpuSession) -> Result<()> {
    let path = session_path();
    let json = serde_json::to_string_pretty(session).context("Failed to serialize session")?;
    std::fs::write(&path, json).with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

/// Remove the local session file.
pub fn remove_session() -> Result<()> {
    let path = session_path();
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("Failed to remove {}", path.display()))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Orchestrator API types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct CreateSessionRequest {
    pub gpu_type: String,
    pub idle_timeout: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub state: SessionState,
    #[serde(default)]
    pub price_per_hour: Option<f64>,
    #[serde(default)]
    pub instance_host: Option<String>,
    #[serde(default)]
    pub ssh_port: Option<u16>,
}

#[derive(Debug, Deserialize)]
pub struct SessionStatusResponse {
    pub session_id: String,
    pub state: SessionState,
    pub gpu_type: String,
    pub idle_timeout: String,
    pub created_at: String,
    #[serde(default)]
    pub price_per_hour: Option<f64>,
    #[serde(default)]
    pub total_cost: Option<f64>,
    #[serde(default)]
    pub runtime_seconds: Option<u64>,
    #[serde(default)]
    pub instance_host: Option<String>,
    #[serde(default)]
    pub ssh_port: Option<u16>,
    #[serde(default)]
    pub error_message: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SubmitJobRequest {
    pub job_type: String,
    pub spec: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct SubmitJobResponse {
    pub job_id: String,
    #[allow(dead_code)]
    pub state: String,
}

#[derive(Debug, Deserialize)]
pub struct JobArtifact {
    pub r2_key: String,
    pub download_url: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub artifact_type: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

impl JobArtifact {
    /// Extract the filename from the R2 key (last path segment).
    pub fn filename(&self) -> String {
        self.r2_key
            .rsplit('/')
            .next()
            .unwrap_or("artifact")
            .to_string()
    }
}

#[derive(Debug, Deserialize)]
struct JobArtifactsResponse {
    #[allow(dead_code)]
    job_id: String,
    artifacts: Vec<JobArtifact>,
}

/// Wrapper for the poll_job_events endpoint response.
#[derive(Debug, Deserialize)]
struct PollEventsResponse {
    #[allow(dead_code)]
    job_id: String,
    events: Vec<serde_json::Value>,
    #[allow(dead_code)]
    #[serde(default)]
    job_status: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct PresignUploadResponse {
    pub upload_url: String,
    pub r2_key: String,
}

// ---------------------------------------------------------------------------
// Orchestrator client
// ---------------------------------------------------------------------------

/// Client for the GPU orchestrator API on the Dokku VPS.
/// Reuses the hub API key for authentication.
pub struct GpuClient {
    api_base: String,
    api_key: String,
    client: reqwest::Client,
}

impl GpuClient {
    /// Create from config (same auth as hub).
    pub fn from_config() -> Result<Self> {
        let hub = HubClient::from_config(true)?;
        let api_base = hub.api_base.clone();
        let api_key = hub
            .api_key
            .clone()
            .context("No API key configured. Run `modl auth login` first.")?;

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            api_base,
            api_key,
            client,
        })
    }

    /// Create from an existing session's api_base and config key.
    pub fn from_session(session: &GpuSession) -> Result<Self> {
        let hub = HubClient::from_config(true)?;
        let api_key = hub
            .api_key
            .clone()
            .context("No API key configured. Run `modl auth login` first.")?;

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            api_base: session.api_base.clone(),
            api_key,
            client,
        })
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    /// POST /gpu/sessions — provision a new remote GPU instance.
    pub async fn create_session(
        &self,
        req: &CreateSessionRequest,
    ) -> Result<CreateSessionResponse> {
        let url = format!("{}/gpu/sessions", self.api_base);
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .json(req)
            .send()
            .await
            .context("Failed to create GPU session")?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                && let Some(detail) = v.get("detail").and_then(|d| d.as_str())
            {
                bail!("GPU session creation failed: {detail}");
            }
            bail!(
                "GPU session creation failed (HTTP {}): {}",
                status.as_u16(),
                text
            );
        }

        serde_json::from_str(&text).context("Failed to parse create session response")
    }

    /// GET /gpu/sessions/{id} — poll session status.
    pub async fn get_session(&self, session_id: &str) -> Result<SessionStatusResponse> {
        let url = format!("{}/gpu/sessions/{}", self.api_base, session_id);
        let resp = self
            .client
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .send()
            .await
            .context("Failed to get GPU session status")?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!(
                "Failed to get session status (HTTP {}): {}",
                status.as_u16(),
                text
            );
        }

        serde_json::from_str(&text).context("Failed to parse session status")
    }

    /// DELETE /gpu/sessions/{id} — destroy instance.
    pub async fn destroy_session(&self, session_id: &str) -> Result<()> {
        let url = format!("{}/gpu/sessions/{}", self.api_base, session_id);
        let resp = self
            .client
            .delete(&url)
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .send()
            .await
            .context("Failed to destroy GPU session")?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("Failed to destroy session: {}", text);
        }
        Ok(())
    }

    /// POST /gpu/sessions/{id}/jobs — submit a job to the remote instance.
    pub async fn submit_job(
        &self,
        session_id: &str,
        req: &SubmitJobRequest,
    ) -> Result<SubmitJobResponse> {
        let url = format!("{}/gpu/sessions/{}/jobs", self.api_base, session_id);
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .json(req)
            .send()
            .await
            .context("Failed to submit job to remote GPU")?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                && let Some(detail) = v.get("detail").and_then(|d| d.as_str())
            {
                bail!("Remote job submission failed: {detail}");
            }
            bail!(
                "Remote job submission failed (HTTP {}): {}",
                status.as_u16(),
                text
            );
        }

        serde_json::from_str(&text).context("Failed to parse job submission response")
    }

    /// GET /gpu/sessions/{id}/jobs/{jid}/events — poll for job events.
    /// Returns raw JSON events array.
    pub async fn poll_job_events(
        &self,
        session_id: &str,
        job_id: &str,
        after_seq: u64,
    ) -> Result<Vec<serde_json::Value>> {
        let url = format!(
            "{}/gpu/sessions/{}/jobs/{}/events?after={}",
            self.api_base, session_id, job_id, after_seq
        );
        let resp = self
            .client
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .send()
            .await
            .context("Failed to poll job events")?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("Failed to poll events (HTTP {}): {}", status.as_u16(), text);
        }

        // Server returns {"job_id": ..., "events": [...], "job_status": ...}
        let wrapper: PollEventsResponse =
            serde_json::from_str(&text).context("Failed to parse job events response")?;
        Ok(wrapper.events)
    }

    /// GET /gpu/sessions/{id}/jobs/{jid}/artifacts — get download URLs for outputs.
    pub async fn get_job_artifacts(
        &self,
        session_id: &str,
        job_id: &str,
    ) -> Result<Vec<JobArtifact>> {
        let url = format!(
            "{}/gpu/sessions/{}/jobs/{}/artifacts",
            self.api_base, session_id, job_id
        );
        let resp = self
            .client
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .send()
            .await
            .context("Failed to get job artifacts")?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!(
                "Failed to get artifacts (HTTP {}): {}",
                status.as_u16(),
                text
            );
        }

        // Server returns {"job_id": ..., "artifacts": [...]}
        let wrapper: JobArtifactsResponse =
            serde_json::from_str(&text).context("Failed to parse artifacts response")?;
        Ok(wrapper.artifacts)
    }

    /// POST /gpu/upload/presign — get a presigned R2 URL for uploading assets.
    #[allow(dead_code)]
    pub async fn presign_upload(
        &self,
        filename: &str,
        content_type: &str,
    ) -> Result<PresignUploadResponse> {
        let url = format!("{}/gpu/upload/presign", self.api_base);
        let body = serde_json::json!({
            "filename": filename,
            "content_type": content_type,
        });
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.auth_header())
            .json(&body)
            .send()
            .await
            .context("Failed to get upload presign URL")?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!(
                "Failed to get presign URL (HTTP {}): {}",
                status.as_u16(),
                text
            );
        }

        serde_json::from_str(&text).context("Failed to parse presign response")
    }

    /// Download a file from a URL to a local path.
    pub async fn download_artifact(&self, url: &str, dest: &std::path::Path) -> Result<()> {
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .with_context(|| format!("Failed to download artifact from {url}"))?;

        if !resp.status().is_success() {
            bail!("Download failed (HTTP {})", resp.status().as_u16());
        }

        let bytes = resp.bytes().await.context("Failed to read artifact body")?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(dest, &bytes)
            .with_context(|| format!("Failed to write artifact to {}", dest.display()))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Session lifecycle helpers
// ---------------------------------------------------------------------------

/// Ensure there is an active (ready) GPU session, provisioning if necessary.
/// Returns the session. Used by `--attach-gpu` auto-provision flow.
pub async fn ensure_session(
    gpu_type: &str,
    idle_timeout: &str,
    models: &[String],
) -> Result<GpuSession> {
    // Check for existing active session
    if let Some(session) = load_session()? {
        if session.state == SessionState::Ready
            || session.state == SessionState::Idle
            || session.state == SessionState::Busy
        {
            return Ok(session);
        }
        // If provisioning/installing, poll until ready
        if session.state == SessionState::Provisioning || session.state == SessionState::Installing
        {
            return wait_for_ready(&session).await;
        }
        // Destroyed/error: remove stale file and provision fresh
        remove_session()?;
    }

    // Provision a new session
    provision_session(gpu_type, idle_timeout, models).await
}

/// Provision a new GPU session via the orchestrator.
pub async fn provision_session(
    gpu_type: &str,
    idle_timeout: &str,
    models: &[String],
) -> Result<GpuSession> {
    let client = GpuClient::from_config()?;

    let resp = client
        .create_session(&CreateSessionRequest {
            gpu_type: gpu_type.to_string(),
            idle_timeout: idle_timeout.to_string(),
            models: models.to_vec(),
        })
        .await?;

    let session = GpuSession {
        session_id: resp.session_id,
        gpu_type: gpu_type.to_string(),
        state: resp.state,
        idle_timeout: idle_timeout.to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        api_base: client.api_base.clone(),
        price_per_hour: resp.price_per_hour,
        instance_host: resp.instance_host,
        ssh_port: resp.ssh_port,
    };

    save_session(&session)?;

    // Wait for the session to become ready
    wait_for_ready(&session).await
}

/// Poll the orchestrator until the session reaches `ready` state.
async fn wait_for_ready(session: &GpuSession) -> Result<GpuSession> {
    let client = GpuClient::from_session(session)?;

    let max_wait = std::time::Duration::from_secs(1200); // 20 minutes
    let poll_interval = std::time::Duration::from_secs(5);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > max_wait {
            bail!(
                "GPU session {} did not become ready within 10 minutes. \
                 Check `modl gpu status` for details.",
                session.session_id
            );
        }

        tokio::time::sleep(poll_interval).await;

        let status = client.get_session(&session.session_id).await?;

        match status.state {
            SessionState::Ready | SessionState::Idle => {
                let updated = GpuSession {
                    session_id: status.session_id,
                    gpu_type: status.gpu_type,
                    state: status.state,
                    idle_timeout: status.idle_timeout,
                    created_at: status.created_at,
                    api_base: session.api_base.clone(),
                    price_per_hour: status.price_per_hour,
                    instance_host: status.instance_host,
                    ssh_port: status.ssh_port,
                };
                save_session(&updated)?;
                return Ok(updated);
            }
            SessionState::Error => {
                let msg = status
                    .error_message
                    .unwrap_or_else(|| "unknown error".to_string());
                remove_session()?;
                bail!("GPU session failed: {msg}");
            }
            SessionState::Destroyed | SessionState::Destroying => {
                remove_session()?;
                bail!("GPU session was destroyed before becoming ready");
            }
            // Still provisioning/installing — keep polling
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_display() {
        assert_eq!(SessionState::Ready.to_string(), "ready");
        assert_eq!(SessionState::Provisioning.to_string(), "provisioning");
        assert_eq!(SessionState::Busy.to_string(), "busy");
    }

    #[test]
    fn test_session_roundtrip() {
        let session = GpuSession {
            session_id: "sess-123".to_string(),
            gpu_type: "a100".to_string(),
            state: SessionState::Ready,
            idle_timeout: "30m".to_string(),
            created_at: "2026-03-19T00:00:00Z".to_string(),
            api_base: "https://hub.modl.run".to_string(),
            price_per_hour: Some(2.50),
            instance_host: Some("1.2.3.4".to_string()),
            ssh_port: Some(22),
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: GpuSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "sess-123");
        assert_eq!(back.state, SessionState::Ready);
        assert_eq!(back.price_per_hour, Some(2.50));
    }
}
