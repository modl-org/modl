use anyhow::{Result, bail};
use console::style;

use crate::core::gpu_session::{self, GpuClient, SessionState};

pub async fn attach(spec: &str, idle: &str) -> Result<()> {
    println!(
        "{} Provisioning {} GPU (idle timeout: {})...",
        style("→").cyan(),
        style(spec).bold(),
        idle
    );

    let session = gpu_session::provision_session(spec, idle, &[]).await?;

    println!();
    println!(
        "  {} GPU session {} is {}",
        style("✓").green().bold(),
        style(&session.session_id).bold(),
        style(&session.state).green()
    );
    println!("  GPU type: {}", session.gpu_type);
    if let Some(price) = session.price_per_hour {
        println!("  Cost:     ${:.2}/hr", price);
    }
    if let Some(ref host) = session.instance_host {
        println!("  Host:     {}", host);
    }
    println!();
    println!(
        "  Run {} to use it, or {} to shut down.",
        style("modl generate \"...\" --attach-gpu").bold(),
        style("modl gpu detach").bold()
    );
    println!();

    Ok(())
}

pub async fn detach() -> Result<()> {
    let session = match gpu_session::load_session()? {
        Some(s) => s,
        None => bail!("No active GPU session. Nothing to detach."),
    };

    if session.state == SessionState::Destroyed {
        gpu_session::remove_session()?;
        println!("Session was already destroyed. Cleaned up local state.");
        return Ok(());
    }

    println!(
        "{} Destroying GPU session {}...",
        style("→").cyan(),
        style(&session.session_id).bold()
    );

    let client = GpuClient::from_session(&session)?;
    client.destroy_session(&session.session_id).await?;
    gpu_session::remove_session()?;

    println!("  {} GPU session destroyed.", style("✓").green().bold());

    Ok(())
}

pub async fn status() -> Result<()> {
    let session = match gpu_session::load_session()? {
        Some(s) => s,
        None => {
            println!("No active GPU sessions.");
            return Ok(());
        }
    };

    // Fetch live status from orchestrator
    let client = GpuClient::from_session(&session)?;
    match client.get_session(&session.session_id).await {
        Ok(status) => {
            println!("{} GPU Session", style("●").cyan().bold());
            println!("  ID:       {}", style(&status.session_id).bold());
            println!("  State:    {}", format_state(&status.state));
            println!("  GPU:      {}", status.gpu_type);
            println!("  Timeout:  {}", status.idle_timeout);
            println!("  Created:  {}", status.created_at);

            if let Some(price) = status.price_per_hour {
                println!("  Rate:     ${:.2}/hr", price);
            }
            if let Some(cost) = status.total_cost {
                println!("  Total:    ${:.2}", cost);
            }
            if let Some(runtime) = status.runtime_seconds {
                let hours = runtime / 3600;
                let mins = (runtime % 3600) / 60;
                println!("  Runtime:  {}h {}m", hours, mins);
            }
            if let Some(ref host) = status.instance_host {
                println!("  Host:     {}", host);
            }
            if let Some(ref msg) = status.error_message {
                println!("  Error:    {}", style(msg).red());
            }

            // Update local cache
            let updated = gpu_session::GpuSession {
                session_id: status.session_id,
                gpu_type: status.gpu_type,
                state: status.state,
                idle_timeout: status.idle_timeout,
                created_at: status.created_at,
                api_base: session.api_base,
                price_per_hour: status.price_per_hour,
                instance_host: status.instance_host,
                ssh_port: status.ssh_port,
            };

            if updated.state == SessionState::Destroyed {
                gpu_session::remove_session()?;
            } else {
                gpu_session::save_session(&updated)?;
            }
        }
        Err(e) => {
            println!(
                "{} GPU Session (cached — cannot reach orchestrator)",
                style("●").yellow()
            );
            println!("  ID:       {}", style(&session.session_id).bold());
            println!("  State:    {} (last known)", format_state(&session.state));
            println!("  GPU:      {}", session.gpu_type);
            println!("  Error:    {}", style(format!("{e:#}")).dim());
        }
    }

    Ok(())
}

pub async fn ssh() -> Result<()> {
    let session = match gpu_session::load_session()? {
        Some(s) => s,
        None => bail!("No active GPU session. Run `modl gpu attach` first."),
    };

    let host = session
        .instance_host
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Session has no SSH host. It may still be provisioning."))?;

    let port = session.ssh_port.unwrap_or(22);

    println!(
        "{} Connecting to {} (port {})...",
        style("→").cyan(),
        style(host).bold(),
        port
    );

    let status = std::process::Command::new("ssh")
        .arg("-p")
        .arg(port.to_string())
        .arg(format!("root@{host}"))
        .status()
        .context("Failed to launch ssh")?;

    if !status.success() {
        bail!("SSH exited with status: {}", status);
    }

    Ok(())
}

fn format_state(state: &SessionState) -> String {
    match state {
        SessionState::Ready | SessionState::Idle => style(state.to_string()).green().to_string(),
        SessionState::Busy => style(state.to_string()).yellow().to_string(),
        SessionState::Provisioning | SessionState::Installing => {
            style(state.to_string()).cyan().to_string()
        }
        SessionState::Error => style(state.to_string()).red().to_string(),
        SessionState::Destroying | SessionState::Destroyed => {
            style(state.to_string()).dim().to_string()
        }
    }
}

use anyhow::Context;
