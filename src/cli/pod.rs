//! `modl pod` — manage Vast.ai instances rented with your own API key.
//!
//! Safety net for `modl train --pod`: list what's running (and billing),
//! destroy stragglers, print the SSH command for debugging.

use anyhow::Result;
use console::style;

use crate::core::vast;

pub async fn ls() -> Result<()> {
    let instances = vast::list_instances().await?;
    if instances.is_empty() {
        println!("No Vast.ai instances on this account.");
        return Ok(());
    }

    println!(
        "{:<12} {:<14} {:<10} {:>8}  {}",
        style("ID").bold(),
        style("GPU").bold(),
        style("STATUS").bold(),
        style("$/HR").bold(),
        style("SSH").bold()
    );
    for i in &instances {
        let ssh = match (&i.ssh_host, i.ssh_port) {
            (Some(h), Some(p)) => format!("ssh -p {p} root@{h}"),
            _ => "-".to_string(),
        };
        println!(
            "{:<12} {:<14} {:<10} {:>8.3}  {}",
            i.id, i.gpu_name, i.actual_status, i.dph_total, ssh
        );
    }
    println!(
        "\n{} Instances bill until destroyed: modl pod rm <id>",
        style("!").yellow()
    );
    Ok(())
}

pub async fn rm(instance_id: u64, yes: bool) -> Result<()> {
    if !yes {
        let ok = dialoguer::Confirm::new()
            .with_prompt(format!(
                "Destroy Vast.ai instance {instance_id}? (billing stops)"
            ))
            .default(true)
            .interact()?;
        if !ok {
            return Ok(());
        }
    }
    vast::destroy_instance(instance_id).await?;
    println!(
        "{} Instance {instance_id} destroyed — billing stopped.",
        style("✓").green()
    );
    Ok(())
}

pub async fn ssh(instance_id: u64) -> Result<()> {
    let inst = vast::get_instance(instance_id).await?;
    match (inst.ssh_host, inst.ssh_port) {
        (Some(host), Some(port)) => {
            println!("ssh -p {port} root@{host}");
            Ok(())
        }
        _ => anyhow::bail!(
            "Instance {instance_id} has no SSH details yet (status: {})",
            inst.actual_status
        ),
    }
}
