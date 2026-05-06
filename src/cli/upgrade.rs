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

    // Extract the binary and python worker from the tarball
    let tmp_python = tmp_dir.join("modl-upgrade-python");
    extract_archive(&tmp_archive, &tmp_binary, &tmp_python).context("Failed to extract archive")?;

    // Clean up the archive
    std::fs::remove_file(&tmp_archive).ok();

    // Try to replace the current binary
    match replace_binary(&current_exe, &tmp_binary) {
        Ok(()) => {
            // Install python worker next to the binary
            install_python_worker(&current_exe, &tmp_python);

            println!();
            println!(
                "{} Updated to {} successfully!",
                style("\u{2713}").green().bold(),
                style(format!("v{}", latest)).bold()
            );

            // Check if the runtime deps are stale (e.g. pinned diffusers or
            // ai-toolkit SHA bumped in this release). The marker file was
            // written by the old version, so we compare against the new binary's
            // constants.
            if let Ok(Some(profile)) = crate::core::runtime::stale_runtime_profile() {
                println!();
                println!(
                    "  {} Runtime dependencies have changed (profile \"{profile}\").",
                    style("\u{2139}").dim()
                );
                println!(
                    "  {} Run {} to update Python packages and ai-toolkit.",
                    style("   ").dim(),
                    style("modl runtime install").cyan()
                );
            }
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
            let bin_dir = current_exe
                .parent()
                .unwrap_or(std::path::Path::new("/usr/local/bin"));
            println!(
                "    sudo install {} {} && sudo rm -rf {}/python && sudo mv {} {}/python",
                tmp_binary.display(),
                current_exe.display(),
                bin_dir.display(),
                tmp_python.display(),
                bin_dir.display(),
            );
            println!();
        }
    }

    Ok(())
}

/// Extract the `modl` binary and `python/` worker from a .tar.gz archive
fn extract_archive(
    archive_path: &std::path::Path,
    binary_dest: &std::path::Path,
    python_dest: &std::path::Path,
) -> Result<()> {
    let file = std::fs::File::open(archive_path)?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);

    // Clean any previous extraction
    if python_dest.exists() {
        std::fs::remove_dir_all(python_dest).ok();
    }

    let mut found_binary = false;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let path_str = path.to_string_lossy().to_string();

        // Extract the modl binary
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if (name == "modl" || name == "modl.exe") && !path_str.contains("python") {
            let mut out = std::fs::File::create(binary_dest)?;
            std::io::copy(&mut entry, &mut out)?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(binary_dest, std::fs::Permissions::from_mode(0o755))?;
            }

            found_binary = true;
            continue;
        }

        // Extract python/ directory contents
        if path_str.starts_with("python/") || path_str.starts_with("./python/") {
            let relative = path_str
                .trim_start_matches("./")
                .strip_prefix("python/")
                .unwrap_or(&path_str);
            let dest_path = python_dest.join(relative);

            if entry.header().entry_type().is_dir() {
                std::fs::create_dir_all(&dest_path).ok();
            } else {
                if let Some(parent) = dest_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut out = std::fs::File::create(&dest_path)?;
                std::io::copy(&mut entry, &mut out)?;
            }
        }
    }

    if !found_binary {
        anyhow::bail!("Could not find modl binary inside the archive");
    }

    Ok(())
}

/// Install the python worker directory next to the binary
fn install_python_worker(binary_path: &std::path::Path, extracted_python: &std::path::Path) {
    // Only proceed if we actually extracted python files
    if !extracted_python.exists()
        || extracted_python
            .read_dir()
            .map_or(true, |mut d| d.next().is_none())
    {
        return;
    }

    let bin_dir = match binary_path.parent() {
        Some(d) => d,
        None => return,
    };
    let target = bin_dir.join("python");

    // Remove old python worker
    if target.exists() {
        std::fs::remove_dir_all(&target).ok();
    }

    // Move extracted python into place
    if std::fs::rename(extracted_python, &target).is_err() {
        // Cross-filesystem: fall back to copy
        if let Err(e) = copy_dir_recursive(extracted_python, &target) {
            eprintln!(
                "  {} Could not install python worker: {}",
                console::style("!").yellow(),
                e
            );
            eprintln!(
                "  Set MODL_WORKER_PYTHON_ROOT to {} as a workaround",
                extracted_python.display()
            );
        } else {
            std::fs::remove_dir_all(extracted_python).ok();
        }
    }
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let dest_path = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
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
