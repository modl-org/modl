use anyhow::{Context, Result};
use std::sync::mpsc;
use std::thread;

use crate::core::executor::{Executor, JobHandle, parse_worker_event};
use crate::core::gpu_session::{GpuClient, GpuSession, SubmitJobRequest};
use crate::core::job::{EditJobSpec, EventPayload, GenerateJobSpec, JobEvent, TrainJobSpec};

// ---------------------------------------------------------------------------
// RemoteExecutor
// ---------------------------------------------------------------------------

/// Executor that submits jobs to a remote GPU instance via the orchestrator API.
///
/// Implements the same `Executor` trait as `LocalExecutor` and `CloudExecutor`.
/// The orchestrator proxies the job to a Vast.ai instance running the modl
/// agent, then relays events back.
pub struct RemoteExecutor {
    session: GpuSession,
    /// Map from job_id to receiver
    jobs: std::collections::HashMap<String, Option<mpsc::Receiver<JobEvent>>>,
}

impl RemoteExecutor {
    pub fn new(session: GpuSession) -> Self {
        Self {
            session,
            jobs: std::collections::HashMap::new(),
        }
    }

    /// Submit a job to the remote instance and start polling for events.
    fn submit_remote(&mut self, job_type: &str, spec: &impl serde::Serialize) -> Result<JobHandle> {
        let spec_value = serde_json::to_value(spec).context("Failed to serialize job spec")?;

        let session = self.session.clone();
        let job_type_owned = job_type.to_string();

        // Submit the job via a blocking HTTP call. We use block_in_place
        // to allow blocking inside the tokio async runtime without panicking.
        let rt = tokio::runtime::Handle::try_current()
            .context("RemoteExecutor requires a tokio runtime")?;

        let submit_result = tokio::task::block_in_place(|| {
            rt.block_on(async {
                let client = GpuClient::from_session(&session)?;
                client
                    .submit_job(
                        &session.session_id,
                        &SubmitJobRequest {
                            job_type: job_type_owned,
                            spec: spec_value,
                        },
                    )
                    .await
            })
        })?;

        let job_id = submit_result.job_id.clone();

        // Set up event polling in a background thread
        let (tx, rx) = mpsc::channel::<JobEvent>();
        let session_clone = self.session.clone();
        let job_id_clone = job_id.clone();

        thread::spawn(move || {
            poll_remote_events(session_clone, &job_id_clone, tx);
        });

        self.jobs.insert(job_id.clone(), Some(rx));

        Ok(JobHandle {
            job_id,
            child_pid: None,
        })
    }
}

impl Executor for RemoteExecutor {
    fn submit(&mut self, spec: &TrainJobSpec) -> Result<JobHandle> {
        self.submit_remote("train", spec)
    }

    fn submit_generate(&mut self, spec: &GenerateJobSpec) -> Result<JobHandle> {
        self.submit_remote("generate", spec)
    }

    fn submit_edit(&mut self, spec: &EditJobSpec) -> Result<JobHandle> {
        self.submit_remote("edit", spec)
    }

    fn events(&mut self, job_id: &str) -> Result<mpsc::Receiver<JobEvent>> {
        let rx = self
            .jobs
            .get_mut(job_id)
            .with_context(|| format!("No remote job found with id: {job_id}"))?
            .take()
            .with_context(|| format!("Event receiver already consumed for job: {job_id}"))?;
        Ok(rx)
    }
}

// ---------------------------------------------------------------------------
// Event polling
// ---------------------------------------------------------------------------

/// Poll the orchestrator for job events and forward them through the channel.
/// Runs in a background thread.
fn poll_remote_events(session: GpuSession, job_id: &str, tx: mpsc::Sender<JobEvent>) {
    let poll_interval = std::time::Duration::from_millis(500);
    let mut last_seq: u64 = 0;

    // Build a new single-threaded runtime for this polling thread
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            let _ = tx.send(JobEvent {
                schema_version: "v1".into(),
                job_id: job_id.into(),
                sequence: 0,
                timestamp: chrono::Utc::now().to_rfc3339(),
                source: "remote_executor".into(),
                event: EventPayload::Error {
                    code: "POLL_SETUP_FAILED".into(),
                    message: format!("Failed to create polling runtime: {e}"),
                    recoverable: false,
                    details: None,
                },
            });
            return;
        }
    };

    rt.block_on(async {
        let client = match GpuClient::from_session(&session) {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(JobEvent {
                    schema_version: "v1".into(),
                    job_id: job_id.into(),
                    sequence: 0,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    source: "remote_executor".into(),
                    event: EventPayload::Error {
                        code: "CLIENT_INIT_FAILED".into(),
                        message: format!("Failed to create GPU client: {e}"),
                        recoverable: false,
                        details: None,
                    },
                });
                return;
            }
        };

        loop {
            tokio::time::sleep(poll_interval).await;

            let events = match client
                .poll_job_events(&session.session_id, job_id, last_seq)
                .await
            {
                Ok(events) => events,
                Err(e) => {
                    // Transient error — log and retry
                    let _ = tx.send(JobEvent {
                        schema_version: "v1".into(),
                        job_id: job_id.into(),
                        sequence: 0,
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        source: "remote_executor".into(),
                        event: EventPayload::Log {
                            level: "warn".into(),
                            message: format!("Event poll failed (retrying): {e}"),
                        },
                    });
                    continue;
                }
            };

            let mut is_terminal = false;

            for raw_event in &events {
                // Try to parse as a typed JobEvent
                if let Some(event) = parse_worker_event(raw_event, job_id) {
                    if event.sequence > last_seq {
                        last_seq = event.sequence;
                    }

                    // Check if this is a terminal event
                    match &event.event {
                        EventPayload::Completed { .. }
                        | EventPayload::Error { .. }
                        | EventPayload::Cancelled => {
                            is_terminal = true;
                        }
                        _ => {}
                    }

                    if tx.send(event).is_err() {
                        return; // receiver dropped
                    }
                } else {
                    // Try direct deserialization
                    if let Ok(event) = serde_json::from_value::<JobEvent>(raw_event.clone()) {
                        if event.sequence > last_seq {
                            last_seq = event.sequence;
                        }

                        match &event.event {
                            EventPayload::Completed { .. }
                            | EventPayload::Error { .. }
                            | EventPayload::Cancelled => {
                                is_terminal = true;
                            }
                            _ => {}
                        }

                        if tx.send(event).is_err() {
                            return;
                        }
                    }
                }
            }

            if is_terminal {
                return;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu_session::SessionState;

    #[test]
    fn test_remote_executor_creation() {
        let session = GpuSession {
            session_id: "sess-test".to_string(),
            gpu_type: "a100".to_string(),
            state: SessionState::Ready,
            idle_timeout: "30m".to_string(),
            created_at: "2026-03-19T00:00:00Z".to_string(),
            api_base: "https://hub.modl.run".to_string(),
            price_per_hour: Some(2.50),
            instance_host: None,
            ssh_port: None,
        };

        let executor = RemoteExecutor::new(session);
        assert!(executor.jobs.is_empty());
    }
}
