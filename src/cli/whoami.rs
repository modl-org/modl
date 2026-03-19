use anyhow::Result;
use console::style;

use crate::core::hub::HubClient;

pub async fn run() -> Result<()> {
    let client = HubClient::from_config(true)?;
    let me = client.me().await?;

    println!("{}", style("Cloud Account").bold().cyan());
    println!(
        "  User:      {}",
        me.username.unwrap_or_else(|| "(no username)".to_string())
    );
    println!("  Email:     {}", me.email);
    println!("  Plan:      {}", me.plan);
    println!(
        "  Storage:   {} / {} ({} files)",
        bytes(me.storage.used_bytes),
        bytes(me.storage.limit_bytes),
        me.storage.file_count
    );
    if !me.usage.is_empty() {
        println!("  Usage:");
        let mut metrics: Vec<_> = me.usage.into_iter().collect();
        metrics.sort_by(|a, b| a.0.cmp(&b.0));
        for (k, v) in metrics {
            println!("    - {k}: {v}");
        }
    }
    Ok(())
}

fn bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let n_f = n as f64;
    if n_f < KB {
        format!("{n} B")
    } else if n_f < MB {
        format!("{:.1} KB", n_f / KB)
    } else if n_f < GB {
        format!("{:.1} MB", n_f / MB)
    } else {
        format!("{:.2} GB", n_f / GB)
    }
}
