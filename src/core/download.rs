use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tokio::io::AsyncWriteExt;

const MAX_RETRIES: u32 = 5;
const RETRY_DELAY_SECS: u64 = 3;

/// Download a file with progress bar, resume support, and automatic retry.
///
/// If the connection drops mid-download, the function will automatically
/// retry up to MAX_RETRIES times, resuming from the last byte written.
/// This is critical for large model files (e.g. 24 GB FLUX).
pub async fn download_file(
    url: &str,
    dest: &Path,
    expected_size: Option<u64>,
    auth_token: Option<&str>,
) -> Result<()> {
    let partial_path = dest.with_extension("partial");
    let file_name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file");

    // Ensure parent directory exists
    if let Some(parent) = partial_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut retries = 0;

    // Build client once — reused across retries to benefit from connection pooling.
    // connect_timeout guards against stalled TCP handshakes; the overall download
    // progress is monitored chunk-by-chunk below (a read timeout would kill large
    // file downloads that are simply slow).
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")?;

    loop {
        // Check for existing partial download
        let start_byte: u64 = if partial_path.exists() {
            std::fs::metadata(&partial_path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        // If we already have the expected size, we're done
        if let Some(expected) = expected_size
            && start_byte >= expected
        {
            // Already fully downloaded, just rename
            std::fs::rename(&partial_path, dest)
                .context("Failed to move downloaded file to final location")?;
            return Ok(());
        }

        let mut request = client.get(url);

        if let Some(token) = auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        if start_byte > 0 {
            request = request.header("Range", format!("bytes={}-", start_byte));
        }

        let response = request.send().await.context("Failed to send request")?;

        // Handle HuggingFace-style redirects: the initial response is a 302 to a
        // pre-signed CDN URL. The CDN URL already contains auth in query params,
        // so we follow it without the Authorization header.
        let response = if response.status().is_redirection() {
            let redirect_url = response
                .headers()
                .get("location")
                .context("Redirect response missing Location header")?
                .to_str()
                .context("Invalid Location header")?
                .to_string();

            let mut redirect_req = client.get(&redirect_url);

            if start_byte > 0 {
                redirect_req = redirect_req.header("Range", format!("bytes={}-", start_byte));
            }

            redirect_req
                .send()
                .await
                .context("Failed to follow redirect")?
        } else {
            response
        };

        if !response.status().is_success() && response.status().as_u16() != 206 {
            anyhow::bail!("Download failed: HTTP {} for {}", response.status(), url);
        }

        let total_size = if response.status().as_u16() == 206 {
            // Partial content — total size from expected
            expected_size.unwrap_or(0)
        } else {
            // Full download — we'll write from scratch
            response.content_length().or(expected_size).unwrap_or(0)
        };

        let effective_start = if response.status().as_u16() == 206 {
            start_byte
        } else {
            0
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
            pb.set_position(effective_start);
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

        if retries > 0 {
            pb.set_message(format!("{} (retry {}/{})", file_name, retries, MAX_RETRIES));
        } else {
            pb.set_message(file_name.to_string());
        }

        // Open file for append (resume) or create
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(effective_start > 0)
            .write(true)
            .truncate(effective_start == 0)
            .open(&partial_path)
            .await
            .context("Failed to open file for writing")?;

        let mut stream = response.bytes_stream();
        let mut stream_error: Option<anyhow::Error> = None;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if let Err(e) = file.write_all(&chunk).await {
                        stream_error = Some(e.into());
                        break;
                    }
                    pb.inc(chunk.len() as u64);
                }
                Err(e) => {
                    stream_error = Some(e.into());
                    break;
                }
            }
        }

        // Flush whatever we have
        let _ = file.flush().await;
        drop(file);

        if let Some(err) = stream_error {
            retries += 1;
            if retries > MAX_RETRIES {
                pb.abandon_with_message(format!(
                    "{} ✗ failed after {} retries",
                    file_name, MAX_RETRIES
                ));
                return Err(err).context(format!(
                    "Download failed after {} retries. Partial file saved at {}. \
                     Re-run the same command to resume.",
                    MAX_RETRIES,
                    partial_path.display()
                ));
            }

            pb.suspend(|| {
                eprintln!(
                    "  ⚠ Connection lost ({}). Retrying in {}s ({}/{})...",
                    err, RETRY_DELAY_SECS, retries, MAX_RETRIES
                );
            });
            pb.finish_and_clear();

            tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
            continue;
        }

        // Success
        pb.finish_with_message(format!("{} ✓", file_name));

        // Move partial to final location
        std::fs::rename(&partial_path, dest)
            .context("Failed to move downloaded file to final location")?;

        return Ok(());
    }
}
