#[cfg(any(target_os = "linux", target_os = "macos"))]
use anyhow::Context;
use anyhow::Result;
use console::style;

pub async fn run(port: u16, no_open: bool, foreground: bool) -> Result<()> {
    if foreground {
        // Foreground mode: blocks the terminal, shows logs
        eprintln!("{}", style("  Starting modl UI...").cyan().bold());
        return crate::ui::server::start(port, !no_open).await;
    }

    // Default: background mode
    // If already running, just open the browser — no restart needed
    if is_listening(port) {
        let url = format!("http://127.0.0.1:{port}");
        eprintln!(
            "  {} modl UI already running at {url}",
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
            "serve",
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
        "  {} modl UI started (PID {})",
        style("✓").green().bold(),
        child.id()
    );
    eprintln!("  {url}");

    if !no_open {
        let _ = open::that(&url);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Service management
// ---------------------------------------------------------------------------

pub async fn install_service(port: u16) -> Result<()> {
    let exe = std::env::current_exe()?.to_string_lossy().to_string();

    #[cfg(target_os = "linux")]
    return install_systemd_service(&exe, port);

    #[cfg(target_os = "macos")]
    return install_launchd_service(&exe, port);

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (exe, port);
        anyhow::bail!(
            "Service installation is only supported on Linux (systemd) and macOS (launchd)"
        );
    }
}

pub async fn remove_service() -> Result<()> {
    #[cfg(target_os = "linux")]
    return remove_systemd_service();

    #[cfg(target_os = "macos")]
    return remove_launchd_service();

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    anyhow::bail!("Service removal is only supported on Linux (systemd) and macOS (launchd)");
}

#[cfg(target_os = "linux")]
fn systemd_unit_path() -> std::path::PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".config/systemd/user/modl.service")
}

#[cfg(target_os = "linux")]
fn install_systemd_service(exe_path: &str, port: u16) -> Result<()> {
    let unit_path = systemd_unit_path();
    if let Some(parent) = unit_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create systemd user directory")?;
    }

    let unit = format!(
        "[Unit]\n\
         Description=modl web UI\n\
         After=network.target\n\
         \n\
         [Service]\n\
         ExecStart={exe_path} serve --foreground --port {port}\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n"
    );

    std::fs::write(&unit_path, &unit)
        .with_context(|| format!("Failed to write {}", unit_path.display()))?;

    // Enable and start the service
    let status = std::process::Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .status()
        .context("Failed to run systemctl daemon-reload")?;
    if !status.success() {
        anyhow::bail!("systemctl daemon-reload failed");
    }

    let status = std::process::Command::new("systemctl")
        .args(["--user", "enable", "--now", "modl.service"])
        .status()
        .context("Failed to enable modl service")?;
    if !status.success() {
        anyhow::bail!("systemctl enable --now modl.service failed");
    }

    println!(
        "  {} Installed systemd service at {}",
        style("✓").green(),
        unit_path.display()
    );
    println!(
        "  {} modl serve will start on boot at http://localhost:{}",
        style("→").cyan(),
        port
    );
    println!();
    println!("  Manage with:");
    println!("    systemctl --user status modl");
    println!("    systemctl --user stop modl");
    println!("    systemctl --user restart modl");
    println!("    modl serve --remove-service");

    Ok(())
}

#[cfg(target_os = "linux")]
fn remove_systemd_service() -> Result<()> {
    let unit_path = systemd_unit_path();

    // Stop and disable
    let _ = std::process::Command::new("systemctl")
        .args(["--user", "disable", "--now", "modl.service"])
        .status();

    if unit_path.exists() {
        std::fs::remove_file(&unit_path)
            .with_context(|| format!("Failed to remove {}", unit_path.display()))?;
    }

    let _ = std::process::Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .status();

    println!("  {} Removed modl service", style("✓").green());
    Ok(())
}

#[cfg(target_os = "macos")]
fn launchd_plist_path() -> std::path::PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join("Library/LaunchAgents/run.modl.plist")
}

#[cfg(target_os = "macos")]
fn install_launchd_service(exe_path: &str, port: u16) -> Result<()> {
    let plist_path = launchd_plist_path();
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create LaunchAgents directory")?;
    }

    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>run.modl</string>
  <key>ProgramArguments</key>
  <array>
    <string>{exe_path}</string>
    <string>serve</string>
    <string>--foreground</string>
    <string>--port</string>
    <string>{port}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/modl-serve.log</string>
  <key>StandardErrorPath</key><string>/tmp/modl-serve.log</string>
</dict>
</plist>
"#
    );

    std::fs::write(&plist_path, &plist)
        .with_context(|| format!("Failed to write {}", plist_path.display()))?;

    let status = std::process::Command::new("launchctl")
        .args(["load", "-w", &plist_path.to_string_lossy()])
        .status()
        .context("Failed to run launchctl load")?;
    if !status.success() {
        anyhow::bail!("launchctl load failed");
    }

    println!(
        "  {} Installed launchd service at {}",
        style("✓").green(),
        plist_path.display()
    );
    println!(
        "  {} modl serve will start on boot at http://localhost:{}",
        style("→").cyan(),
        port
    );
    println!();
    println!("  Manage with:");
    println!("    launchctl list | grep modl");
    println!("    launchctl unload ~/Library/LaunchAgents/run.modl.plist");
    println!("    modl serve --remove-service");

    Ok(())
}

#[cfg(target_os = "macos")]
fn remove_launchd_service() -> Result<()> {
    let plist_path = launchd_plist_path();

    if plist_path.exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["unload", &plist_path.to_string_lossy()])
            .status();
        std::fs::remove_file(&plist_path)
            .with_context(|| format!("Failed to remove {}", plist_path.display()))?;
    }

    println!("  {} Removed modl service", style("✓").green());
    Ok(())
}
