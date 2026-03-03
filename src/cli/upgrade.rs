use anyhow::{Context, Result};
use console::style;
use dialoguer::Confirm;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

const REPO: &str = "modl-org/modl";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(serde::Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(serde::Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

pub async fn run() -> Result<()> {
    println!(
        "{} Checking for updates... (current: {})",
        style("\u{2192}").cyan(),
        style(format!("v{}", CURRENT_VERSION)).dim()
    );

    let client = reqwest::Client::builder()
        .user_agent("modl-cli")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let release: GitHubRelease = client
        .get(format!(
            "https://api.github.com/repos/{}/releases/latest",
            REPO
        ))
        .send()
        .await
        .context("Failed to check for updates")?
        .json()
        .await
        .context("Failed to parse release info")?;

    let latest = release.tag_name.trim_start_matches('v');

    if !is_newer(latest, CURRENT_VERSION) {
        println!(
            "{} Already up to date (v{})",
            style("\u{2713}").green(),
            CURRENT_VERSION
        );
        return Ok(());
    }

    let target = detect_target()?;
    let asset_name = format!("modl-{}-{}.tar.gz", release.tag_name, target);

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No release binary found for your platform ({}).\n  \
                 You can update from source: cargo install --git https://github.com/{}",
                target,
                REPO
            )
        })?;

    println!();
    println!(
        "  {} {} {} {}",
        style("New version available:").bold(),
        style(format!("v{}", CURRENT_VERSION)).red(),
        style("\u{2192}").dim(),
        style(format!("v{}", latest)).green().bold()
    );
    println!(
        "  {} {}",
        style("Download size:").dim(),
        indicatif::HumanBytes(asset.size)
    );
    println!();

    let confirm = Confirm::new()
        .with_prompt("  Install update?")
        .default(true)
        .interact()?;

    if !confirm {
        println!("{}", style("Update cancelled.").dim());
        return Ok(());
    }

    // Determine where the current binary lives (don't canonicalize — we want
    // the path the user's shell resolves, not a symlink target)
    let current_exe = std::env::current_exe().context("Could not determine current binary path")?;

    // Download to the system temp directory so we never need elevated
    // permissions just to fetch the archive
    let tmp_dir = std::env::temp_dir();
    let tmp_archive = tmp_dir.join("modl-upgrade.tar.gz");
    let tmp_binary = tmp_dir.join("modl-upgrade-bin");

    let pb = ProgressBar::new(asset.size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  Downloading [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("\u{2588}\u{2593}\u{2591}"),
    );

    // Download the tarball
    let response = client
        .get(&asset.browser_download_url)
        .send()
        .await
        .context("Failed to download update")?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", response.status());
    }

    {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        let mut file = tokio::fs::File::create(&tmp_archive).await?;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading download stream")?;
            file.write_all(&chunk).await?;
            pb.inc(chunk.len() as u64);
        }

        file.flush().await?;
    }

    pb.finish_and_clear();
    println!("  {} Downloaded", style("\u{2713}").green());

    // Extract the binary from the tarball
    extract_binary(&tmp_archive, &tmp_binary).context("Failed to extract binary from archive")?;

    // Clean up the archive
    std::fs::remove_file(&tmp_archive).ok();

    // Try to replace the current binary
    match replace_binary(&current_exe, &tmp_binary) {
        Ok(()) => {
            println!();
            println!(
                "{} Updated to {} successfully!",
                style("\u{2713}").green().bold(),
                style(format!("v{}", latest)).bold()
            );
        }
        Err(_) => {
            // Leave the downloaded binary in temp and print manual instructions
            println!();
            println!(
                "  {} Could not replace {} (permission denied)",
                style("!").yellow().bold(),
                current_exe.display()
            );
            println!();
            println!("  Run this to finish the update:");
            println!();
            println!(
                "    sudo install {} {}",
                tmp_binary.display(),
                current_exe.display()
            );
            println!();
        }
    }

    Ok(())
}

/// Extract the `modl` binary from a .tar.gz archive
fn extract_binary(archive_path: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    let file = std::fs::File::open(archive_path)?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        if name == "modl" || name == "modl.exe" {
            let mut out = std::fs::File::create(dest)?;
            std::io::copy(&mut entry, &mut out)?;

            // Preserve executable permission on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
            }

            return Ok(());
        }
    }

    anyhow::bail!("Could not find modl binary inside the archive")
}

/// Replace the running binary atomically
fn replace_binary(current: &PathBuf, new: &PathBuf) -> Result<()> {
    let backup = current.with_extension("bak");

    // Move current -> backup
    if current.exists() {
        std::fs::rename(current, &backup).context("Failed to back up current binary")?;
    }

    // Copy new -> current (copy instead of rename to handle cross-filesystem)
    match std::fs::copy(new, current) {
        Ok(_) => {
            // Set executable permission
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(current, std::fs::Permissions::from_mode(0o755)).ok();
            }
            // Clean up backup and temp
            std::fs::remove_file(&backup).ok();
            std::fs::remove_file(new).ok();
            Ok(())
        }
        Err(e) => {
            // Restore backup
            std::fs::rename(&backup, current).ok();
            Err(e).context("Failed to install new binary")
        }
    }
}

/// Compare semver strings: returns true if `latest` is newer than `current`
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

/// Detect the release target triple for the current platform
fn detect_target() -> Result<&'static str> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("linux", "x86_64") => Ok("x86_64-unknown-linux-gnu"),
        ("linux", "aarch64") => Ok("aarch64-unknown-linux-gnu"),
        ("macos", "x86_64") => Ok("x86_64-apple-darwin"),
        ("macos", "aarch64") => Ok("aarch64-apple-darwin"),
        ("windows", "x86_64") => Ok("x86_64-pc-windows-msvc"),
        _ => anyhow::bail!("Unsupported platform: {}-{}", os, arch),
    }
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
    fn test_detect_target() {
        // Just verify it doesn't panic on current platform
        let target = detect_target();
        assert!(target.is_ok());
    }
}
