use anyhow::{Context, Result, bail};
use console::style;

use crate::core::config::{CloudConfig, Config};
use crate::core::hub::HubClient;

pub async fn run() -> Result<()> {
    let client = HubClient::from_config(false)?;
    let start = client.device_start().await?;

    let verify_url = start.verify_url.unwrap_or_else(|| {
        format!(
            "{}/device?code={}",
            client.api_base,
            urlencoding::encode(&start.user_code)
        )
    });

    println!();
    println!("{} Device login", style("modl login").bold().cyan());
    println!();
    println!(
        "  1. Open this URL in your browser:\n     {}",
        style(&verify_url).underlined()
    );
    println!(
        "  2. Enter/confirm this code:\n     {}",
        style(&start.user_code).bold()
    );
    println!();

    let _ = open::that(&verify_url);

    let mut tries = 0u32;
    let max_tries = 900 / (start.interval.max(1) as u32) + 2; // ~15 min

    loop {
        if tries > max_tries {
            bail!("Timed out waiting for device confirmation");
        }
        tries += 1;

        let poll = client.device_poll(&start.device_code).await?;
        match poll.status.as_str() {
            "pending" => {
                tokio::time::sleep(std::time::Duration::from_secs(start.interval.max(1))).await;
            }
            "expired" => bail!("Device code expired. Run `modl login` again."),
            "confirmed" => {
                let api_key = poll
                    .api_key
                    .filter(|k| !k.trim().is_empty())
                    .context("Device was confirmed but no API key was returned")?;
                save_auth(&client.api_base, &api_key).await?;
                return Ok(());
            }
            other => bail!("Unexpected device poll status: {other}"),
        }
    }
}

async fn save_auth(api_base: &str, api_key: &str) -> Result<()> {
    let mut config = Config::load()?;
    let cloud = config.cloud.get_or_insert(CloudConfig {
        api_base: None,
        api_key: None,
        username: None,
        default_provider: None,
        modal_token: None,
        replicate_token: None,
        runpod_key: None,
    });

    cloud.api_base = Some(api_base.to_string());
    cloud.api_key = Some(api_key.to_string());

    // Resolve username immediately for better push UX.
    let mut authed = HubClient::from_config(false)?;
    authed.api_key = Some(api_key.to_string());
    if let Ok(me) = authed.me().await {
        cloud.username = me.username;
        println!(
            "{} Logged in as {}",
            style("✓").green().bold(),
            style(me.email).bold()
        );
    } else {
        println!(
            "{} Logged in (could not fetch account profile yet)",
            style("✓").green().bold()
        );
    }

    config.save()?;
    println!(
        "  Saved credentials to {}",
        style("~/.modl/config.yaml").dim()
    );
    Ok(())
}
