use anyhow::Result;
use clap::Subcommand;
use console::style;

use crate::core::runtime;

#[derive(Subcommand)]
pub enum RuntimeCommands {
    /// Install managed runtime profile
    Install {
        /// Runtime profile to install
        #[arg(long)]
        profile: Option<String>,
        /// Runtime channel: stable or beta
        #[arg(long)]
        channel: Option<String>,
    },

    /// Show runtime installation status
    Status,

    /// Run runtime health checks
    Doctor,

    /// Bootstrap managed profile environment and install dependencies
    Bootstrap {
        /// Runtime profile to bootstrap
        #[arg(long)]
        profile: Option<String>,
        /// Runtime channel: stable or beta
        #[arg(long)]
        channel: Option<String>,
    },

    /// Upgrade runtime profile to latest channel-compatible version
    Upgrade {
        /// Runtime channel: stable or beta
        #[arg(long)]
        channel: Option<String>,
    },

    /// Reset runtime state
    Reset {
        /// Also remove wheel/manifests cache
        #[arg(long)]
        purge_cache: bool,
    },
}

pub async fn run(command: RuntimeCommands) -> Result<()> {
    match command {
        RuntimeCommands::Install { profile, channel } => {
            // Only check for training capability if explicitly requesting a trainer profile
            if profile.as_deref() == Some("trainer-cu124") {
                crate::core::preflight::check_device_for_training()?;
            }
            let result = runtime::install(profile.as_deref(), channel.as_deref())?;

            println!("{} Runtime installed", style("✓").green().bold());
            println!("  Profile: {}", result.profile);
            println!("  Channel: {}", result.channel);
            println!("  Root: {}", result.runtime_root.display());
            println!("  Lock: {}", result.lock_path.display());
            println!();
            println!(
                "{} Python packages are bootstrapped lazily on first train/generate execution.",
                style("i").dim()
            );

            Ok(())
        }
        RuntimeCommands::Status => {
            let status = runtime::status()?;
            println!("{} Runtime status", style("modl runtime").cyan().bold());
            println!("  Root: {}", status.runtime_root.display());
            println!("  Lock: {}", status.lock_path.display());
            println!();

            if status.installed {
                println!("{} Installed", style("✓").green().bold());
                println!(
                    "  Profile: {}",
                    status.profile.unwrap_or_else(|| "unknown".to_string())
                );
                println!(
                    "  Channel: {}",
                    status.channel.unwrap_or_else(|| "unknown".to_string())
                );
            } else {
                println!("{} Not installed", style("!").yellow().bold());
                println!(
                    "  Run {} to install.",
                    style("modl runtime install --profile trainer-cu124").cyan()
                );
            }

            Ok(())
        }
        RuntimeCommands::Doctor => {
            let report = runtime::doctor()?;
            println!("{} Runtime doctor", style("modl runtime").cyan().bold());

            let mut issues = 0;

            if report.runtime_root_exists {
                println!("  {} runtime root exists", style("✓").green());
            } else {
                println!("  {} runtime root missing", style("✗").red());
                issues += 1;
            }

            if report.lock_exists {
                println!("  {} runtime lock exists", style("✓").green());
            } else {
                println!("  {} runtime lock missing", style("✗").red());
                issues += 1;
            }

            if report.profile_known {
                println!("  {} runtime profile recognized", style("✓").green());
            } else {
                println!("  {} runtime profile not recognized", style("✗").red());
                issues += 1;
            }

            if report.python_exists {
                println!("  {} python binary present", style("✓").green());
            } else {
                println!(
                    "  {} python binary not found yet (lazy bootstrap pending)",
                    style("i").yellow()
                );
            }

            println!();
            if issues == 0 {
                println!("{} Runtime health looks good", style("✓").green().bold());
            } else {
                println!(
                    "{} Found {} issue{}",
                    style("!").yellow().bold(),
                    issues,
                    if issues == 1 { "" } else { "s" }
                );
            }

            Ok(())
        }
        RuntimeCommands::Bootstrap { profile, channel } => {
            let result = runtime::bootstrap(profile.as_deref(), channel.as_deref()).await?;

            println!("{} Runtime bootstrap complete", style("✓").green().bold());
            println!("  Profile: {}", result.profile);
            println!("  Env: {}", result.env_dir.display());
            println!("  Python: {}", result.python_path.display());
            println!("  Requirements: {}", result.requirements_path.display());
            if result.created_env {
                println!("  Status: created new environment");
            } else {
                println!("  Status: reused existing environment");
            }

            Ok(())
        }
        RuntimeCommands::Upgrade { channel } => {
            let result = runtime::upgrade(channel.as_deref())?;

            println!("{} Runtime upgraded", style("✓").green().bold());
            println!("  Profile: {}", result.profile);
            println!("  Channel: {}", result.channel);
            println!("  Lock: {}", result.lock_path.display());

            Ok(())
        }
        RuntimeCommands::Reset { purge_cache } => {
            runtime::reset(purge_cache)?;
            if purge_cache {
                println!(
                    "{} Runtime reset complete (cache purged)",
                    style("✓").green().bold()
                );
            } else {
                println!(
                    "{} Runtime reset complete (manifests/wheel cache preserved)",
                    style("✓").green().bold()
                );
            }
            Ok(())
        }
    }
}
