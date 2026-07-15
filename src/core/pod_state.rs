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

    /// Running-cost estimate so far: age × hourly rate.
    pub fn cost_so_far(&self) -> f64 {
        (self.age_secs() as f64 / 3600.0) * self.dph_total
    }
}

fn state_path() -> PathBuf {
    crate::core::paths::modl_root().join("pods.json")
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
    let mut pods = load();
    if let Some(rec) = pods.iter_mut().find(|p| p.instance_id == instance_id) {
        rec.bootstrap_fingerprint = Some(fingerprint.to_string());
        save(&pods)?;
    }
    Ok(())
}

/// Remove a record by instance ID. No-op if absent.
pub fn remove(instance_id: u64) -> Result<()> {
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
    let mut pods = load();
    if pods.is_empty() {
        return Ok(None);
    }

    let live = vast::list_instances().await?;
    let by_id: std::collections::HashMap<u64, &vast::Instance> =
        live.iter().map(|i| (i.id, i)).collect();

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
            if inst.dph_total > 0.0 {
                rec.dph_total = inst.dph_total;
            }
        }
    }
    save(&pods)?;

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
    fn age_is_zero_for_unparseable_timestamp() {
        let mut r = rec(1);
        r.created_at = "not-a-date".into();
        assert_eq!(r.age_secs(), 0);
        assert_eq!(r.cost_so_far(), 0.0);
    }
}
