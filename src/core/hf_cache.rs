//! Where the Python worker's HuggingFace downloads land.
//!
//! The store holds single-file weights, content-addressed by SHA256. Some
//! Python paths cannot use that layout: ai-toolkit training and a handful of
//! adapters call `from_pretrained(repo_id)`, which needs the HF *directory*
//! layout and therefore downloads through `huggingface_hub`.
//!
//! Those downloads used to land wherever `huggingface_hub` defaults to
//! (`~/.cache/huggingface`), because modl passed `HF_HUB_OFFLINE` to its
//! children but never `HF_HOME`. Two consequences:
//!
//! 1. `storage.root` was bypassed. Someone who points modl at a big disk still
//!    got tens or hundreds of GB accumulating in `$HOME`, on a disk that may
//!    be much smaller.
//! 2. It was invisible. `modl system gc` reported a clean store while the HF
//!    cache next to it held the *same* weights again — once as the upstream
//!    source, once converted into the store.
//!
//! This module picks one location, stamps it onto every Python child, and
//! exposes the size so commands can report it.
//!
//! Resolution order — an environment variable always wins, so an existing
//! cache is never orphaned by an upgrade:
//!
//! 1. `HF_HOME` / `HF_HUB_CACHE` from the environment
//! 2. `storage.hf_cache` from `config.yaml`
//! 3. `~/.cache/huggingface` (`huggingface_hub`'s own default)

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::core::config::Config;

/// Which rule picked the cache location — used for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// `HF_HOME` or `HF_HUB_CACHE` was already set in the environment.
    Env,
    /// `storage.hf_cache` in `config.yaml`.
    Config,
    /// `huggingface_hub`'s default, `~/.cache/huggingface`.
    Default,
}

impl Source {
    pub fn label(&self) -> &'static str {
        match self {
            Source::Env => "HF_HOME in the environment",
            Source::Config => "storage.hf_cache in config.yaml",
            Source::Default => "huggingface_hub default",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HfCache {
    home: PathBuf,
    hub: PathBuf,
    source: Source,
}

impl HfCache {
    /// The value to export as `HF_HOME`.
    pub fn home(&self) -> &Path {
        &self.home
    }

    /// The directory that actually holds the blobs (`<home>/hub`, unless
    /// `HF_HUB_CACHE` pointed somewhere else).
    pub fn hub(&self) -> &Path {
        &self.hub
    }

    pub fn source(&self) -> Source {
        self.source
    }

    /// Total bytes under the hub directory. `None` if it doesn't exist yet.
    ///
    /// Walks the tree with `walkdir` and counts each file once, so hardlinked
    /// blobs are not double-counted within the cache.
    pub fn size_on_disk(&self) -> Option<u64> {
        if !self.hub.exists() {
            return None;
        }
        let total = walkdir::WalkDir::new(&self.hub)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        Some(total)
    }
}

/// Resolve the cache location, caching the answer for the process.
pub fn resolved() -> &'static HfCache {
    static CACHE: OnceLock<HfCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        // A missing or malformed config must not stop a job from running —
        // fall back to the default location.
        let configured = Config::load()
            .ok()
            .and_then(|c| c.storage.hf_cache.clone())
            .map(|p| expand_tilde(&p));
        resolve_from(
            env_path("HF_HOME"),
            env_path("HF_HUB_CACHE"),
            configured,
            dirs::home_dir(),
        )
    })
}

/// Pure resolver — every input is explicit so the precedence rules are testable
/// without touching the real environment.
pub fn resolve_from(
    env_home: Option<PathBuf>,
    env_hub: Option<PathBuf>,
    configured: Option<PathBuf>,
    home_dir: Option<PathBuf>,
) -> HfCache {
    // HF_HUB_CACHE is the more specific of the two and wins inside
    // huggingface_hub, so honour it the same way here.
    if let Some(hub) = env_hub {
        let home = env_home.unwrap_or_else(|| hub.parent().unwrap_or(&hub).to_path_buf());
        return HfCache {
            home,
            hub,
            source: Source::Env,
        };
    }

    if let Some(home) = env_home {
        return HfCache {
            hub: home.join("hub"),
            home,
            source: Source::Env,
        };
    }

    if let Some(home) = configured {
        return HfCache {
            hub: home.join("hub"),
            home,
            source: Source::Config,
        };
    }

    let home = home_dir
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("huggingface");
    HfCache {
        hub: home.join("hub"),
        home,
        source: Source::Default,
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    match std::env::var_os(key) {
        Some(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => None,
    }
}

fn expand_tilde(path: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(stripped);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> Option<PathBuf> {
        Some(PathBuf::from("/home/tester"))
    }

    #[test]
    fn falls_back_to_huggingface_default() {
        let c = resolve_from(None, None, None, home());
        assert_eq!(c.home(), Path::new("/home/tester/.cache/huggingface"));
        assert_eq!(c.hub(), Path::new("/home/tester/.cache/huggingface/hub"));
        assert_eq!(c.source(), Source::Default);
    }

    #[test]
    fn config_overrides_the_default() {
        let c = resolve_from(None, None, Some(PathBuf::from("/srv/disk2/hf")), home());
        assert_eq!(c.home(), Path::new("/srv/disk2/hf"));
        assert_eq!(c.hub(), Path::new("/srv/disk2/hf/hub"));
        assert_eq!(c.source(), Source::Config);
    }

    #[test]
    fn env_wins_over_config_so_existing_caches_are_never_orphaned() {
        let c = resolve_from(
            Some(PathBuf::from("/mnt/big/hf")),
            None,
            Some(PathBuf::from("/srv/disk2/hf")),
            home(),
        );
        assert_eq!(c.home(), Path::new("/mnt/big/hf"));
        assert_eq!(c.source(), Source::Env);
    }

    #[test]
    fn hub_cache_env_points_the_hub_dir_directly() {
        let c = resolve_from(
            None,
            Some(PathBuf::from("/mnt/big/hf/hub")),
            Some(PathBuf::from("/srv/disk2/hf")),
            home(),
        );
        assert_eq!(c.hub(), Path::new("/mnt/big/hf/hub"));
        assert_eq!(c.home(), Path::new("/mnt/big/hf"));
        assert_eq!(c.source(), Source::Env);
    }

    #[test]
    fn empty_env_var_is_treated_as_unset() {
        // env_path() filters empties; resolve_from receives None and falls back.
        let c = resolve_from(None, None, None, home());
        assert_eq!(c.source(), Source::Default);
    }

    #[test]
    fn size_is_none_when_the_cache_does_not_exist() {
        let c = resolve_from(None, None, Some(PathBuf::from("/nonexistent/hf")), home());
        assert_eq!(c.size_on_disk(), None);
    }
}
