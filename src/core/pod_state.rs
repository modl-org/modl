//! Persistent pod state — `~/.modl/pods.json`.
//!
//! Vast.ai is the source of truth; this file is a local cache so `modl train
//! --pod`, `modl generate --pod`, and the `modl pod` commands can find and
//! reuse a running instance without re-searching the marketplace.
//!
//! Design (from docs/plans/pod-lifecycle.md §0):
//! - **Plural data model, singular UX.** The file holds an array; the MVP
//!   enforces one *active* pod (newest wins, with a warning).
//! - Every read that matters reconciles against `GET /instances/` and prunes
//!   records whose instance is gone. SSH host/port are refreshed (they change
//!   on host restarts).
//! - The Vast API key never lands here — only instance IDs and SSH targets.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::core::vast;

/// How long a pod can run before `warn_if_stale` nags on every pod command.
const STALE_AFTER_SECS: i64 = 2 * 60 * 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PodRecord {
    pub instance_id: u64,
    pub gpu_name: String,
    pub dph_total: f64,
    /// Hourly storage cost for the provisioned disk — bills alongside the GPU
    /// rate from the moment of rent, and can rival it on cheap cards.
    /// Recorded at rent time (the instances API doesn't report it); defaults
    /// to 0 for records written before this field existed.
    #[serde(default)]
    pub dph_storage: f64,
    pub ssh_host: String,
    pub ssh_port: u16,
    /// RFC3339 timestamp of when the record was first created.
    pub created_at: String,
    /// sha256 of the bootstrap script the pod was last bootstrapped with.
    /// `None` until the first successful bootstrap.
    pub bootstrap_fingerprint: Option<String>,
    pub label: String,
}

impl PodRecord {
    /// Age in seconds since `created_at`, or 0 if unparseable.
    pub fn age_secs(&self) -> i64 {
        parse_rfc3339(&self.created_at)
            .map(|t| (now() - t).num_seconds().max(0))
            .unwrap_or(0)
    }

    /// All-in hourly rate: GPU + storage. Every user-facing cost figure must
    /// use this — the rent quote deliberately prices all-in, and a stale nag
    /// showing half the true burn defeats its purpose.
    pub fn dph_all_in(&self) -> f64 {
        self.dph_total + self.dph_storage
    }

    /// Running-cost estimate so far: age × all-in hourly rate.
    pub fn cost_so_far(&self) -> f64 {
        (self.age_secs() as f64 / 3600.0) * self.dph_all_in()
    }
}

fn state_path() -> PathBuf {
    crate::core::paths::modl_root().join("pods.json")
}

/// Guard for cross-process read-modify-write cycles on pods.json. The atomic
/// rename in `save` prevents torn files, not lost updates — and concurrency
/// is designed-in: MCP's detached `modl pod up` upserts SSH details while
/// `pod ls`/`active_pod` reconcile-save from other processes. A lost update
/// can drop the record of a billing instance, silently disabling the stale
/// nag. Advisory flock on a sidecar file; released when the guard drops.
/// Writes are tiny, so blocking on the lock is momentary.
struct StateLock(#[allow(dead_code)] std::fs::File);

fn lock_state() -> Result<StateLock> {
    let path = state_path().with_extension("json.lock");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("Failed to lock {}", path.display()));
        }
    }
    Ok(StateLock(file))
}

fn now() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

/// Current time as an RFC3339 string, for stamping new records.
pub fn now_rfc3339() -> String {
    now().to_rfc3339()
}

fn parse_rfc3339(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|t| t.with_timezone(&chrono::Utc))
}

/// Load all pod records. Returns `[]` on a missing or unparseable file —
/// a corrupt cache should never wedge the CLI.
pub fn load() -> Vec<PodRecord> {
    let path = state_path();
    let Ok(data) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    serde_json::from_str(&data).unwrap_or_default()
}

/// Write records atomically (tmp file + rename) so a crash mid-write can't
/// leave a truncated `pods.json`.
pub fn save(pods: &[PodRecord]) -> Result<()> {
    let path = state_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(pods).context("Failed to serialize pods.json")?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).with_context(|| format!("Failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("Failed to finalize {}", path.display()))?;
    Ok(())
}

/// Insert or replace a record by instance ID.
pub fn upsert(rec: PodRecord) -> Result<()> {
    let _lock = lock_state()?;
    let mut pods = load();
    if let Some(existing) = pods.iter_mut().find(|p| p.instance_id == rec.instance_id) {
        *existing = rec;
    } else {
        pods.push(rec);
    }
    save(&pods)
}

/// Update the bootstrap fingerprint on an existing record. No-op if the
/// instance isn't tracked.
pub fn set_fingerprint(instance_id: u64, fingerprint: &str) -> Result<()> {
    let _lock = lock_state()?;
    let mut pods = load();
    if let Some(rec) = pods.iter_mut().find(|p| p.instance_id == instance_id) {
        rec.bootstrap_fingerprint = Some(fingerprint.to_string());
        save(&pods)?;
    }
    Ok(())
}

/// Remove a record by instance ID. No-op if absent.
pub fn remove(instance_id: u64) -> Result<()> {
    let _lock = lock_state()?;
    let mut pods = load();
    let before = pods.len();
    pods.retain(|p| p.instance_id != instance_id);
    if pods.len() != before {
        save(&pods)?;
    }
    Ok(())
}

/// Reconcile the cache against Vast and return the single active pod (MVP).
///
/// - Prunes records whose instance no longer exists on the account.
/// - Refreshes `ssh_host`/`ssh_port`/`gpu_name`/`dph_total` from live data.
/// - Returns the newest *running* record (warns if more than one is running).
pub async fn active_pod() -> Result<Option<PodRecord>> {
    // Peek without the lock — an empty cache skips the network round-trip.
    if load().is_empty() {
        return Ok(None);
    }

    let live = vast::list_instances().await?;
    let by_id: std::collections::HashMap<u64, &vast::Instance> =
        live.iter().map(|i| (i.id, i)).collect();

    // Reconcile under the lock against a FRESH load: the Vast call above
    // takes seconds, and a concurrent writer (MCP's detached `pod up`
    // recording SSH details) may have changed the file since the peek —
    // saving that stale snapshot would clobber its update. The lock is held
    // only for the local read-modify-write, never across the network call.
    let lock = lock_state()?;
    let mut pods = load();

    // Prune dead records; refresh survivors from live data.
    pods.retain(|rec| by_id.contains_key(&rec.instance_id));
    for rec in pods.iter_mut() {
        if let Some(inst) = by_id.get(&rec.instance_id) {
            if let (Some(host), Some(port)) = (&inst.ssh_host, inst.ssh_port) {
                rec.ssh_host = host.clone();
                rec.ssh_port = port;
            }
            if !inst.gpu_name.is_empty() {
                rec.gpu_name = inst.gpu_name.clone();
            }
            // Refresh the GPU rate from `dph_base`, never `dph_total`: the
            // instances endpoint's dph_total is ALL-IN (base + storage), so
            // writing it into rec.dph_total would double-count storage in
            // every dph_all_in() display after the first reconcile
            // (observed live: $0.169 offer became a $0.247 "all-in" line).
            if inst.dph_base > 0.0 {
                rec.dph_total = inst.dph_base;
            }
        }
    }
    save(&pods)?;
    drop(lock);

    // Active = running with a usable SSH target.
    let mut running: Vec<PodRecord> = pods
        .into_iter()
        .filter(|rec| {
            by_id
                .get(&rec.instance_id)
                .map(|i| i.actual_status == "running")
                .unwrap_or(false)
                && !rec.ssh_host.is_empty()
        })
        .collect();

    if running.is_empty() {
        return Ok(None);
    }

    // Newest first (by created_at).
    running.sort_by(|a, b| {
        parse_rfc3339(&b.created_at)
            .unwrap_or_else(now)
            .cmp(&parse_rfc3339(&a.created_at).unwrap_or_else(now))
    });

    if running.len() > 1 {
        use console::style;
        eprintln!(
            "{} {} pods running — using the newest ({}). Others: {}",
            style("⚠").yellow(),
            running.len(),
            running[0].instance_id,
            running[1..]
                .iter()
                .map(|p| p.instance_id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    Ok(Some(running.into_iter().next().unwrap()))
}

/// Print one stderr line per pod that has been running longer than 2h.
///
/// Zero network cost — uses only the recorded `created_at` + `dph_total`, so
/// it's cheap enough to call from the same place as the update-check nag.
/// Idle safety: pods can't self-destruct (we refuse to ship them the API
/// key), so this nag is the mitigation for a forgotten billing instance.
pub fn warn_if_stale() {
    use console::style;
    for rec in load() {
        if rec.age_secs() >= STALE_AFTER_SECS {
            let hrs = rec.age_secs() as f64 / 3600.0;
            eprintln!(
                "{} Pod {} has been running {:.0}h (${:.2} so far) — {} when done.",
                style("⚠").yellow(),
                rec.instance_id,
                hrs,
                rec.cost_so_far(),
                style(format!("modl pod rm {}", rec.instance_id)).bold(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: u64) -> PodRecord {
        PodRecord {
            instance_id: id,
            gpu_name: "RTX 3090".into(),
            dph_total: 0.20,
            dph_storage: 0.0,
            ssh_host: "ssh5.vast.ai".into(),
            ssh_port: 12345,
            created_at: now_rfc3339(),
            bootstrap_fingerprint: Some("abc123".into()),
            label: "modl-pod".into(),
        }
    }

    #[test]
    fn record_roundtrips_through_json() {
        let r = rec(42);
        let json = serde_json::to_string(&r).unwrap();
        let back: PodRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn age_and_cost_from_created_at() {
        let mut r = rec(1);
        // 3 hours ago
        r.created_at = (now() - chrono::Duration::hours(3)).to_rfc3339();
        r.dph_total = 0.30;
        assert!(r.age_secs() >= 3 * 3600 - 5);
        assert!((r.cost_so_far() - 0.90).abs() < 0.01);
    }

    #[test]
    fn cost_so_far_includes_storage() {
        let mut r = rec(1);
        r.created_at = (now() - chrono::Duration::hours(10)).to_rfc3339();
        r.dph_total = 0.12;
        r.dph_storage = 0.10; // storage can rival the GPU rate on cheap cards
        assert!((r.cost_so_far() - 2.20).abs() < 0.02);
    }

    #[test]
    fn old_records_without_storage_field_still_parse() {
        let json = r#"{"instance_id": 7, "gpu_name": "RTX 3090", "dph_total": 0.2,
                       "ssh_host": "h", "ssh_port": 1, "created_at": "x",
                       "bootstrap_fingerprint": null, "label": "modl-pod"}"#;
        let r: PodRecord = serde_json::from_str(json).unwrap();
        assert_eq!(r.dph_storage, 0.0);
    }

    #[test]
    fn age_is_zero_for_unparseable_timestamp() {
        let mut r = rec(1);
        r.created_at = "not-a-date".into();
        assert_eq!(r.age_secs(), 0);
        assert_eq!(r.cost_so_far(), 0.0);
    }
}
