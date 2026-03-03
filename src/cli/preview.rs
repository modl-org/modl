use anyhow::Result;
use console::style;

pub async fn run(port: u16, no_open: bool, foreground: bool) -> Result<()> {
    if foreground {
        // Foreground mode: blocks the terminal, shows logs
        eprintln!(
            "{}",
            style("  Starting training preview UI...").cyan().bold()
        );
        return crate::ui::server::start(port, !no_open).await;
    }

    // Default: background mode
    // If already running, just open the browser — no restart needed
    if is_listening(port) {
        let url = format!("http://127.0.0.1:{port}");
        eprintln!(
            "  {} Preview already running at {url}",
            style("✓").green().bold(),
        );
        if !no_open {
            let _ = open::that(&url);
        }
        return Ok(());
    }

    spawn_background(port, no_open)
}

/// Check if something is already listening on the given port.
fn is_listening(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        std::time::Duration::from_millis(200),
    )
    .is_ok()
}

/// Re-exec ourselves as a detached background process with --foreground.
fn spawn_background(port: u16, no_open: bool) -> Result<()> {
    use std::process::{Command, Stdio};

    let exe = std::env::current_exe()?;
    let child = Command::new(&exe)
        .args([
            "preview",
            "--foreground",
            "--no-open",
            "--port",
            &port.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    // Give the server a moment to bind
    std::thread::sleep(std::time::Duration::from_millis(300));

    let url = format!("http://127.0.0.1:{port}");
    eprintln!(
        "  {} Preview server started in background (PID {})",
        style("✓").green().bold(),
        child.id()
    );
    eprintln!("  {url}");

    if !no_open {
        let _ = open::that(&url);
    }

    Ok(())
}
