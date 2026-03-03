use anyhow::{Result, bail};
use console::style;
use std::sync::mpsc;

use crate::core::executor::{Executor, JobHandle};
use crate::core::job::{GenerateJobSpec, JobEvent, TrainJobSpec};

const CLOUD_WAITLIST_URL: &str = "https://modl.run/cloud";

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
// CloudExecutor
// ---------------------------------------------------------------------------

/// Cloud executor — dispatches jobs to a remote cloud provider.
///
/// Implements the same `Executor` trait as `LocalExecutor`. The CLI event loop
/// doesn't know or care which executor it's talking to.
///
/// ## Adding a new provider
///
/// 1. Add a variant to `CloudProvider`
/// 2. Implement the upload/submit/poll/download logic in the corresponding
///    match arm below
/// 3. Done — everything above (presets, spec, DB) and below (artifact
///    collection, symlinks) is shared code.
pub struct CloudExecutor {
    provider: CloudProvider,
    #[allow(dead_code)]
    api_key: Option<String>,
}

impl CloudExecutor {
    /// Create a cloud executor for the given provider.
    ///
    /// Reads API credentials from config or environment variables:
    ///   - Modal:     MODAL_TOKEN_ID + MODAL_TOKEN_SECRET
    ///   - Replicate: REPLICATE_API_TOKEN
    ///   - RunPod:    RUNPOD_API_KEY
    pub fn new(provider: CloudProvider) -> Result<Self> {
        let api_key = resolve_api_key(provider)?;
        Ok(Self {
            provider,
            api_key: Some(api_key),
        })
    }

    /// The provider this executor targets.
    #[allow(dead_code)]
    pub fn provider(&self) -> CloudProvider {
        self.provider
    }
}

impl Executor for CloudExecutor {
    fn submit(&mut self, _spec: &TrainJobSpec) -> Result<JobHandle> {
        print_cloud_waitlist("training", self.provider);
        std::process::exit(0);
    }

    fn submit_generate(&mut self, _spec: &GenerateJobSpec) -> Result<JobHandle> {
        print_cloud_waitlist("generation", self.provider);
        std::process::exit(0);
    }

    fn events(&mut self, _job_id: &str) -> Result<mpsc::Receiver<JobEvent>> {
        bail!("Cloud executor: no active job to receive events from");
    }

    fn cancel(&self, _job_id: &str) -> Result<()> {
        bail!(
            "Cloud executor: cancel not yet implemented for {}",
            self.provider
        );
    }
}

// ---------------------------------------------------------------------------
// Waitlist
// ---------------------------------------------------------------------------

fn print_cloud_waitlist(action: &str, provider: CloudProvider) {
    eprintln!();
    eprintln!(
        "  {}  Cloud {} is coming soon.",
        style("☁").cyan().bold(),
        action
    );
    eprintln!();
    eprintln!(
        "  We're building managed GPU {} via Modal, Replicate, and RunPod",
        action
    );
    eprintln!(
        "  so you can run {} without owning hardware.",
        style(format!(
            "modl {} --cloud --provider {}",
            if action == "training" {
                "train"
            } else {
                "generate"
            },
            provider
        ))
        .bold()
    );
    eprintln!();
    eprintln!(
        "  {} Sign up for early access: {}",
        style("→").green().bold(),
        style(CLOUD_WAITLIST_URL).underlined().bold()
    );
    eprintln!();
    eprintln!(
        "  For now, use local {} (remove the {} flag).",
        action,
        style("--cloud").yellow()
    );
    eprintln!();
}

// ---------------------------------------------------------------------------
// Credential resolution
// ---------------------------------------------------------------------------

/// Resolve API key for a provider from environment or config.
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

    // Try loading from modl config
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
    fn test_waitlist_url_is_set() {
        assert!(!CLOUD_WAITLIST_URL.is_empty());
        assert!(CLOUD_WAITLIST_URL.starts_with("https://"));
    }
}
