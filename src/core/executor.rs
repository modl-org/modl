use anyhow::{Context, Result, bail};
use std::env;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;

use crate::core::job::{EventPayload, GenerateJobSpec, JobEvent, TrainJobSpec};
use crate::core::runtime;
use crate::core::training::resolve_worker_python_root;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Handle returned after submitting a job
#[derive(Debug)]
pub struct JobHandle {
    pub job_id: String,
    #[allow(dead_code)]
    pub child_pid: Option<u32>,
}

/// Executor runs a job spec and produces a stream of JobEvents.
///
/// Uses std::sync::mpsc (not async) — matches the sync stdout-reading
/// pattern. A future CloudExecutor can spawn a tokio task that polls HTTP
/// and sends into the same channel type.
pub trait Executor {
    fn submit(&mut self, spec: &TrainJobSpec) -> Result<JobHandle>;
    fn submit_generate(&mut self, spec: &GenerateJobSpec) -> Result<JobHandle>;
    fn events(&mut self, job_id: &str) -> Result<mpsc::Receiver<JobEvent>>;
    #[allow(dead_code)]
    fn cancel(&self, job_id: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// LocalExecutor
// ---------------------------------------------------------------------------

pub struct LocalExecutor {
    python_path: PathBuf,
    runtime_root: PathBuf,
    /// Map from job_id to (child, receiver)
    jobs: std::collections::HashMap<String, JobState>,
}

struct JobState {
    #[allow(dead_code)]
    child: Option<Child>,
    receiver: Option<mpsc::Receiver<JobEvent>>,
}

impl LocalExecutor {
    /// Create an executor from an already-set-up runtime.
    pub fn new(python_path: PathBuf, runtime_root: PathBuf) -> Self {
        Self {
            python_path,
            runtime_root,
            jobs: std::collections::HashMap::new(),
        }
    }

    /// Bootstrap the training runtime if needed, then create executor.
    pub async fn from_runtime_setup() -> Result<Self> {
        let setup = runtime::setup_training(false).await?;
        if !setup.ready {
            bail!(
                "Training runtime is not ready. ai-toolkit command could not be detected. \
                 Run `modl train-setup --reinstall` or set MODL_AITOOLKIT_TRAIN_CMD."
            );
        }

        let runtime_root = dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".modl")
            .join("runtime");

        Ok(Self::new(setup.python_path, runtime_root))
    }
}

impl Executor for LocalExecutor {
    fn submit(&mut self, spec: &TrainJobSpec) -> Result<JobHandle> {
        let job_id = format!(
            "job-{}-{}",
            spec.output.lora_name,
            chrono::Utc::now().format("%Y%m%d-%H%M%S")
        );

        // Write spec YAML to jobs dir
        let jobs_dir = self.runtime_root.join("jobs");
        std::fs::create_dir_all(&jobs_dir)
            .with_context(|| format!("Failed to create jobs dir: {}", jobs_dir.display()))?;

        let spec_path = jobs_dir.join(format!("{}.yaml", job_id));
        let yaml = serde_yaml::to_string(spec).context("Failed to serialize job spec")?;
        std::fs::write(&spec_path, &yaml)
            .with_context(|| format!("Failed to write spec: {}", spec_path.display()))?;

        // Set up Python worker command
        let worker_root = resolve_worker_python_root()?;
        let mut py_path = worker_root.to_string_lossy().to_string();
        if let Ok(Some(aitk_dir)) = runtime::aitoolkit_path() {
            py_path = format!("{}:{}", py_path, aitk_dir.display());
        }
        if let Ok(current) = env::var("PYTHONPATH")
            && !current.trim().is_empty()
        {
            py_path = format!("{}:{}", py_path, current);
        }

        let mut command = Command::new(&self.python_path);
        command
            .arg("-m")
            .arg("modl_worker.main")
            .arg("train")
            .arg("--config")
            .arg(&spec_path)
            .arg("--job-id")
            .arg(&job_id)
            .env("PYTHONPATH", &py_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Tell the Python worker where ai-toolkit lives so it can invoke run.py
        if let Ok(Some(ref aitk_dir)) = runtime::aitoolkit_path() {
            command.env("MODL_AITOOLKIT_ROOT", aitk_dir);
        }

        if let Ok(Some(template)) = runtime::train_command_template() {
            command.env("MODL_AITOOLKIT_TRAIN_CMD", template);
        }

        let mut child = command.spawn().with_context(|| {
            format!(
                "Failed to start training worker using {}",
                self.python_path.display()
            )
        })?;

        let child_pid = child.id();

        // Set up event channel
        let (tx, rx) = mpsc::channel::<JobEvent>();
        let job_id_clone = job_id.clone();

        let stdout = child
            .stdout
            .take()
            .context("Failed to capture worker stdout")?;

        let stderr = child
            .stderr
            .take()
            .context("Failed to capture worker stderr")?;

        // Spawn reader threads for stdout and stderr
        thread::spawn(move || {
            read_worker_stdout(stdout, &job_id_clone, tx.clone());
            // After stdout closes, read remaining stderr
            read_worker_stderr(stderr, &job_id_clone, tx);
        });

        self.jobs.insert(
            job_id.clone(),
            JobState {
                child: Some(child),
                receiver: Some(rx),
            },
        );

        Ok(JobHandle {
            job_id,
            child_pid: Some(child_pid),
        })
    }

    fn submit_generate(&mut self, spec: &GenerateJobSpec) -> Result<JobHandle> {
        let job_id = format!("gen-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));

        // Write spec YAML to jobs dir
        let jobs_dir = self.runtime_root.join("jobs");
        std::fs::create_dir_all(&jobs_dir)
            .with_context(|| format!("Failed to create jobs dir: {}", jobs_dir.display()))?;

        let spec_path = jobs_dir.join(format!("{}.yaml", job_id));
        let yaml = serde_yaml::to_string(spec).context("Failed to serialize generate spec")?;
        std::fs::write(&spec_path, &yaml)
            .with_context(|| format!("Failed to write spec: {}", spec_path.display()))?;

        // Set up Python worker command
        let worker_root = resolve_worker_python_root()?;
        let mut py_path = worker_root.to_string_lossy().to_string();
        if let Ok(Some(aitk_dir)) = runtime::aitoolkit_path() {
            py_path = format!("{}:{}", py_path, aitk_dir.display());
        }
        if let Ok(current) = env::var("PYTHONPATH")
            && !current.trim().is_empty()
        {
            py_path = format!("{}:{}", py_path, current);
        }

        let mut command = Command::new(&self.python_path);
        command
            .arg("-m")
            .arg("modl_worker.main")
            .arg("generate")
            .arg("--config")
            .arg(&spec_path)
            .arg("--job-id")
            .arg(&job_id)
            .env("PYTHONPATH", py_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command.spawn().with_context(|| {
            format!(
                "Failed to start generate worker using {}",
                self.python_path.display()
            )
        })?;

        let child_pid = child.id();

        // Set up event channel
        let (tx, rx) = mpsc::channel::<JobEvent>();
        let job_id_clone = job_id.clone();

        let stdout = child
            .stdout
            .take()
            .context("Failed to capture worker stdout")?;

        let stderr = child
            .stderr
            .take()
            .context("Failed to capture worker stderr")?;

        // Spawn reader threads for stdout and stderr
        thread::spawn(move || {
            read_worker_stdout(stdout, &job_id_clone, tx.clone());
            read_worker_stderr(stderr, &job_id_clone, tx);
        });

        self.jobs.insert(
            job_id.clone(),
            JobState {
                child: Some(child),
                receiver: Some(rx),
            },
        );

        Ok(JobHandle {
            job_id,
            child_pid: Some(child_pid),
        })
    }

    fn events(&mut self, job_id: &str) -> Result<mpsc::Receiver<JobEvent>> {
        let state = self
            .jobs
            .get_mut(job_id)
            .with_context(|| format!("No job found with id: {job_id}"))?;

        state
            .receiver
            .take()
            .with_context(|| format!("Event receiver already consumed for job: {job_id}"))
    }

    fn cancel(&self, job_id: &str) -> Result<()> {
        if let Some(state) = self.jobs.get(job_id)
            && let Some(ref child) = state.child
        {
            let pid = child.id();
            // Use kill command to send SIGTERM
            let status = Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .status()
                .context("Failed to send SIGTERM to worker process")?;
            if !status.success() {
                bail!("Failed to kill process {pid}");
            }
            return Ok(());
        }
        bail!("No running process found for job: {job_id}");
    }
}

/// Read stderr from the worker process, wrap each line as a Log event.
pub(crate) fn read_worker_stderr(
    stderr: impl std::io::Read,
    job_id: &str,
    tx: mpsc::Sender<JobEvent>,
) {
    let reader = BufReader::new(stderr);
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let event = JobEvent {
            schema_version: "v1".into(),
            job_id: job_id.into(),
            sequence: 0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            source: "modl_worker".into(),
            event: EventPayload::Log {
                level: "stderr".into(),
                message: line,
            },
        };
        if tx.send(event).is_err() {
            break;
        }
    }
}

/// Read stdout from the Python worker, parse JSON events, send through channel.
pub(crate) fn read_worker_stdout(
    stdout: impl std::io::Read,
    job_id: &str,
    tx: mpsc::Sender<JobEvent>,
) {
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as a worker protocol envelope
        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(raw) => {
                if let Some(event) = parse_worker_event(&raw, job_id) {
                    if tx.send(event).is_err() {
                        break; // receiver dropped
                    }
                } else {
                    // Unstructured log line — wrap as Log event
                    let event = JobEvent {
                        schema_version: "v1".into(),
                        job_id: job_id.into(),
                        sequence: 0,
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        source: "modl_worker".into(),
                        event: EventPayload::Log {
                            level: "info".into(),
                            message: line,
                        },
                    };
                    if tx.send(event).is_err() {
                        break;
                    }
                }
            }
            Err(_) => {
                // Raw non-JSON line → Log event
                let event = JobEvent {
                    schema_version: "v1".into(),
                    job_id: job_id.into(),
                    sequence: 0,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    source: "modl_worker".into(),
                    event: EventPayload::Log {
                        level: "info".into(),
                        message: line,
                    },
                };
                if tx.send(event).is_err() {
                    break;
                }
            }
        }
    }
}

/// Parse a JSON value from the Python worker protocol into a typed JobEvent.
pub(crate) fn parse_worker_event(raw: &serde_json::Value, job_id: &str) -> Option<JobEvent> {
    let event_val = raw.get("event")?;
    let event_type = event_val.get("type")?.as_str()?;

    let schema_version = raw
        .get("schema_version")
        .and_then(|v| v.as_str())
        .unwrap_or("v1")
        .to_string();
    let sequence = raw.get("sequence").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp = raw
        .get("timestamp")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let source = raw
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("modl_worker")
        .to_string();
    let raw_job_id = raw
        .get("job_id")
        .and_then(|v| v.as_str())
        .unwrap_or(job_id)
        .to_string();

    let payload = match event_type {
        "job_accepted" => EventPayload::JobAccepted {
            worker_pid: event_val
                .get("worker_pid")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        },
        "job_started" => EventPayload::JobStarted {
            config: event_val
                .get("config")
                .and_then(|v| v.as_str())
                .map(String::from),
            command: event_val.get("command").and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|i| i.as_str().map(String::from))
                        .collect()
                })
            }),
        },
        "progress" => EventPayload::Progress {
            stage: event_val
                .get("stage")
                .and_then(|v| v.as_str())
                .unwrap_or("train")
                .to_string(),
            step: event_val.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            total_steps: event_val
                .get("total_steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            loss: event_val.get("loss").and_then(|v| v.as_f64()),
            eta_seconds: event_val.get("eta_seconds").and_then(|v| v.as_f64()),
        },
        "artifact" => EventPayload::Artifact {
            path: event_val
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            sha256: event_val
                .get("sha256")
                .and_then(|v| v.as_str())
                .map(String::from),
            size_bytes: event_val.get("size_bytes").and_then(|v| v.as_u64()),
        },
        "log" => EventPayload::Log {
            level: event_val
                .get("level")
                .and_then(|v| v.as_str())
                .unwrap_or("info")
                .to_string(),
            message: event_val
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        },
        "warning" => EventPayload::Warning {
            code: event_val
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            message: event_val
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        },
        "completed" => EventPayload::Completed {
            message: event_val
                .get("message")
                .and_then(|v| v.as_str())
                .map(String::from),
        },
        "error" => EventPayload::Error {
            code: event_val
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN")
                .to_string(),
            message: event_val
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            recoverable: event_val
                .get("recoverable")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            details: event_val.get("details").cloned(),
        },
        "cancelled" => EventPayload::Cancelled,
        "heartbeat" => EventPayload::Heartbeat,
        _ => {
            // Unknown event type — wrap as log
            EventPayload::Log {
                level: "debug".into(),
                message: format!("Unknown event type: {}", event_type),
            }
        }
    };

    Some(JobEvent {
        schema_version,
        job_id: raw_job_id,
        sequence,
        timestamp,
        source,
        event: payload,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_progress_event() {
        let raw: serde_json::Value = serde_json::json!({
            "schema_version": "v1",
            "sequence": 5,
            "timestamp": "2026-01-01T00:00:00Z",
            "source": "modl_worker",
            "event": {
                "type": "progress",
                "stage": "train",
                "step": 100,
                "total_steps": 2000,
                "loss": 0.0345
            }
        });

        let event = parse_worker_event(&raw, "job-123").unwrap();
        assert_eq!(event.sequence, 5);
        if let EventPayload::Progress {
            step,
            total_steps,
            loss,
            ..
        } = event.event
        {
            assert_eq!(step, 100);
            assert_eq!(total_steps, 2000);
            assert!((loss.unwrap() - 0.0345).abs() < 1e-6);
        } else {
            panic!("Expected Progress event");
        }
    }

    #[test]
    fn test_parse_completed_event() {
        let raw: serde_json::Value = serde_json::json!({
            "schema_version": "v1",
            "sequence": 10,
            "timestamp": "2026-01-01T01:00:00Z",
            "source": "modl_worker",
            "event": {
                "type": "completed",
                "message": "Training finished"
            }
        });

        let event = parse_worker_event(&raw, "job-123").unwrap();
        if let EventPayload::Completed { message } = event.event {
            assert_eq!(message.unwrap(), "Training finished");
        } else {
            panic!("Expected Completed event");
        }
    }

    #[test]
    fn test_parse_error_event() {
        let raw: serde_json::Value = serde_json::json!({
            "schema_version": "v1",
            "sequence": 3,
            "timestamp": "2026-01-01T00:05:00Z",
            "source": "modl_worker",
            "event": {
                "type": "error",
                "code": "TRAINING_FAILED",
                "message": "Process exited with code 1",
                "recoverable": false,
                "details": {"exit_code": 1}
            }
        });

        let event = parse_worker_event(&raw, "job-123").unwrap();
        if let EventPayload::Error {
            code, recoverable, ..
        } = event.event
        {
            assert_eq!(code, "TRAINING_FAILED");
            assert!(!recoverable);
        } else {
            panic!("Expected Error event");
        }
    }

    #[test]
    fn test_parse_artifact_event() {
        let raw: serde_json::Value = serde_json::json!({
            "schema_version": "v1",
            "sequence": 8,
            "timestamp": "2026-01-01T00:50:00Z",
            "source": "modl_worker",
            "event": {
                "type": "artifact",
                "path": "/tmp/output/lora.safetensors",
                "sha256": "abc123",
                "size_bytes": 1024000
            }
        });

        let event = parse_worker_event(&raw, "job-123").unwrap();
        if let EventPayload::Artifact {
            path,
            sha256,
            size_bytes,
        } = event.event
        {
            assert_eq!(path, "/tmp/output/lora.safetensors");
            assert_eq!(sha256.unwrap(), "abc123");
            assert_eq!(size_bytes.unwrap(), 1024000);
        } else {
            panic!("Expected Artifact event");
        }
    }

    #[test]
    fn test_parse_unknown_event_type() {
        let raw: serde_json::Value = serde_json::json!({
            "schema_version": "v1",
            "sequence": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "source": "modl_worker",
            "event": {
                "type": "some_future_event",
                "data": "foo"
            }
        });

        let event = parse_worker_event(&raw, "job-123").unwrap();
        if let EventPayload::Log { level, .. } = event.event {
            assert_eq!(level, "debug");
        } else {
            panic!("Expected Log fallback for unknown event");
        }
    }

    #[test]
    fn test_parse_non_protocol_json() {
        let raw: serde_json::Value = serde_json::json!({"some": "random json"});
        let event = parse_worker_event(&raw, "job-123");
        assert!(event.is_none()); // no "event" key → None
    }
}
