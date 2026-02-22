use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// Download a file with progress bar and resume support
pub async fn download_file(
    url: &str,
    dest: &Path,
    expected_size: Option<u64>,
    auth_token: Option<&str>,
) -> Result<()> {
    let client = reqwest::Client::new();

    // Check for existing partial download
    let mut start_byte: u64 = 0;
    let partial_path = dest.with_extension("partial");

    if partial_path.exists() {
        start_byte = std::fs::metadata(&partial_path)
            .map(|m| m.len())
            .unwrap_or(0);
    }

    let mut request = client.get(url);

    if let Some(token) = auth_token {
        request = request.header("Authorization", format!("Bearer {}", token));
    }

    if start_byte > 0 {
        request = request.header("Range", format!("bytes={}-", start_byte));
    }

    let response = request.send().await.context("Failed to send request")?;

    if !response.status().is_success() && response.status().as_u16() != 206 {
        anyhow::bail!(
            "Download failed: HTTP {} for {}",
            response.status(),
            url
        );
    }

    let total_size = if response.status().as_u16() == 206 {
        // Partial content — total size from content-range or expected
        expected_size.unwrap_or(0)
    } else {
        // Full download — reset partial
        start_byte = 0;
        response
            .content_length()
            .or(expected_size)
            .unwrap_or(0)
    };

    // Progress bar
    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.set_position(start_byte);
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{msg} {bytes} {spinner}")
                .unwrap(),
        );
        pb
    };

    let file_name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    pb.set_message(file_name.to_string());

    // Ensure parent directory exists
    if let Some(parent) = partial_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Open file for append (resume) or create
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(start_byte > 0)
        .write(true)
        .truncate(start_byte == 0)
        .open(&partial_path)
        .await
        .context("Failed to open file for writing")?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading download stream")?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);

    pb.finish_with_message(format!("{} ✓", file_name));

    // Move partial to final location
    std::fs::rename(&partial_path, dest)
        .context("Failed to move downloaded file to final location")?;

    Ok(())
}
