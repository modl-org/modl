//! Background update check — runs at most once per 24 hours.
//!
//! On every CLI invocation, we spawn a non-blocking check that:
//! 1. Reads a cached timestamp from `~/.modl/update-check.json`
//! 2. If < 24h old, skips the network call entirely
//! 3. Otherwise, hits the GitHub releases API (15s timeout)
//! 4. Writes back the result (latest version + timestamp)
//!
//! After the main command finishes, `print_if_update_available()` reads the
//! cache and prints a one-liner if a newer version exists.
//!
//! This never blocks, never panics, and never prints errors. If anything
//! goes wrong the check is silently skipped.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const REPO: &str = "modl-org/modl";
const CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60; // 24 hours

#[derive(Debug, Serialize, Deserialize)]
struct UpdateCache {
    /// Latest version seen (without 'v' prefix)
    latest_version: String,
    /// Unix timestamp of last check
    checked_at: u64,
}

fn cache_path() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("update-check.json")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Compare semver strings: returns true if `latest` is newer than `current`.
fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |s: &str| -> (u64, u64, u64) {
        let parts: Vec<u64> = s.split('.').filter_map(|p| p.parse().ok()).collect();
        (
            parts.first().copied().unwrap_or(0),
            parts.get(1).copied().unwrap_or(0),
            parts.get(2).copied().unwrap_or(0),
        )
    };
    parse(latest) > parse(current)
}

/// Read the cached update check result. Returns None if missing or stale.
fn read_cache() -> Option<UpdateCache> {
    let data = std::fs::read_to_string(cache_path()).ok()?;
    serde_json::from_str(&data).ok()
}

/// Write cache to disk (best-effort, errors swallowed).
fn write_cache(cache: &UpdateCache) {
    let path = cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(json) = serde_json::to_string(cache) {
        std::fs::write(path, json).ok();
    }
}

/// Spawn a non-blocking background task that checks for updates.
///
/// This returns a `JoinHandle` that the caller can optionally await
/// after the main command completes. The task never fails — all errors
/// are swallowed.
pub fn spawn_check() -> tokio::task::JoinHandle<()> {
    tokio::spawn(async {
        let _ = check_and_cache().await;
    })
}

/// The actual check: skip if recently checked, otherwise hit GitHub API.
async fn check_and_cache() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // If cache is fresh, skip the network call
    if let Some(cache) = read_cache() {
        let age = now_unix().saturating_sub(cache.checked_at);
        if age < CHECK_INTERVAL_SECS {
            return Ok(());
        }
    }

    // Fetch latest release from GitHub
    let client = reqwest::Client::builder()
        .user_agent("modl-cli")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    #[derive(Deserialize)]
    struct Release {
        tag_name: String,
    }

    let release: Release = client
        .get(format!(
            "https://api.github.com/repos/{}/releases/latest",
            REPO
        ))
        .send()
        .await?
        .json()
        .await?;

    let latest = release.tag_name.trim_start_matches('v').to_string();

    write_cache(&UpdateCache {
        latest_version: latest,
        checked_at: now_unix(),
    });

    Ok(())
}

/// Print a one-liner to stderr if a newer version is available.
///
/// Call this after the main command completes. Reads from cache only
/// (no network). If no cache exists or versions match, prints nothing.
pub fn print_if_update_available() {
    let cache = match read_cache() {
        Some(c) => c,
        None => return,
    };

    if !is_newer(&cache.latest_version, CURRENT_VERSION) {
        return;
    }

    use console::style;
    eprintln!();
    eprintln!(
        "  {} Update available: {} {} {}",
        style("!").yellow().bold(),
        style(format!("v{}", CURRENT_VERSION)).dim(),
        style("→").dim(),
        style(format!("v{}", cache.latest_version)).green().bold(),
    );
    eprintln!("  Run {} to update", style("modl upgrade").bold(),);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_newer() {
        assert!(is_newer("0.2.0", "0.1.0"));
        assert!(is_newer("0.1.1", "0.1.0"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.0.9", "0.1.0"));
    }

    #[test]
    fn test_cache_roundtrip() {
        // Just verify serde works
        let cache = UpdateCache {
            latest_version: "0.2.0".to_string(),
            checked_at: 1700000000,
        };
        let json = serde_json::to_string(&cache).unwrap();
        let parsed: UpdateCache = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.latest_version, "0.2.0");
        assert_eq!(parsed.checked_at, 1700000000);
    }
}
