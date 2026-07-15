//! Vast.ai REST API client — direct, BYO-key provisioning.
//!
//! Talks to console.vast.ai with the user's own API key (`VASTAI_API_KEY`).
//! No hosted orchestrator involved: search offers, rent an instance, poll it,
//! destroy it. Used by `modl train --pod` and the `modl pod` commands.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;

const API_BASE: &str = "https://console.vast.ai/api/v0";

/// Docker image for rented pods. Vast.ai's own PyTorch template is pre-cached
/// across their fleet, so it boots in ~1 min (vs 5-20 min for custom images).
pub const POD_IMAGE: &str = "vastai/pytorch:latest";

#[derive(Debug, Clone)]
#[allow(dead_code)] // full marketplace row — some fields are display-only for now
pub struct Offer {
    pub id: u64,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub gpu_ram_mb: u64,
    pub disk_gb: f64,
    pub dph_total: f64,
    pub inet_down: f64,
    pub inet_up: f64,
    pub reliability: f64,
    pub dlperf: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Instance {
    pub id: u64,
    pub actual_status: String,
    pub gpu_name: String,
    pub ssh_host: Option<String>,
    pub ssh_port: Option<u16>,
    pub dph_total: f64,
    pub label: Option<String>,
}

/// Resolve the user's Vast.ai API key. BYO-key: env var only, no hosted auth.
pub fn api_key() -> Result<String> {
    match std::env::var("VASTAI_API_KEY") {
        Ok(k) if !k.trim().is_empty() => Ok(k.trim().to_string()),
        _ => bail!(
            "VASTAI_API_KEY is not set.\n\n\
             modl train --pod rents a GPU on Vast.ai using your own account:\n\
             1. Create an account at https://cloud.vast.ai\n\
             2. Copy your API key from https://cloud.vast.ai/account\n\
             3. export VASTAI_API_KEY=<your-key>\n\n\
             Also add an SSH key to your Vast.ai account — modl connects to the\n\
             pod over SSH to ship the dataset and sync results back."
        ),
    }
}

/// Map a friendly GPU type to Vast.ai's gpu_name filter value.
fn gpu_name_filter(gpu_type: &str) -> String {
    match gpu_type.to_lowercase().as_str() {
        "rtx3090" | "3090" => "RTX 3090".into(),
        "rtx4090" | "4090" => "RTX 4090".into(),
        "rtx5090" | "5090" => "RTX 5090".into(),
        "a10g" => "A10G".into(),
        "a100" => "A100".into(),
        "a100-80gb" => "A100_80GB".into(),
        "h100" => "H100".into(),
        "h200" => "H200".into(),
        // Unknown types pass through as-is (Vast.ai names are case-sensitive)
        _ => gpu_type.to_string(),
    }
}

fn client() -> Result<(reqwest::Client, String)> {
    let key = api_key()?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    Ok((client, key))
}

async fn check(resp: reqwest::Response, what: &str) -> Result<serde_json::Value> {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("Vast.ai {what} failed ({status}): {}", body.trim());
    }
    serde_json::from_str(&body).with_context(|| format!("Vast.ai {what}: invalid JSON response"))
}

/// Search the marketplace for rentable offers matching a GPU type.
/// Results are sorted fastest-host-first (dlperf desc), then by price.
pub async fn search_offers(gpu_type: &str, max_price_per_hour: f64) -> Result<Vec<Offer>> {
    let (client, key) = client()?;

    let query = json!({
        "verified": {"eq": true},
        "external": {"eq": false},
        "rentable": {"eq": true},
        "gpu_name": {"eq": gpu_name_filter(gpu_type)},
        "num_gpus": {"eq": 1},
        "dph_total": {"lte": max_price_per_hour},
        "inet_down": {"gte": 500},
        "inet_up": {"gte": 200},
        "disk_space": {"gte": 60},
        "reliability2": {"gte": 0.95},
        "direct_port_count": {"gte": 1},
        "cuda_max_good": {"gte": 12.0},
        "order": [["dlperf", "desc"]],
        "limit": 20,
    });

    let resp = client
        .post(format!("{API_BASE}/bundles/"))
        .bearer_auth(&key)
        .json(&query)
        .send()
        .await
        .context("Vast.ai offer search request failed")?;
    let data = check(resp, "offer search").await?;

    let mut offers = parse_offers(&data);
    offers.sort_by(|a, b| {
        b.dlperf
            .partial_cmp(&a.dlperf)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.dph_total
                    .partial_cmp(&b.dph_total)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    Ok(offers)
}

fn parse_offers(data: &serde_json::Value) -> Vec<Offer> {
    let empty = vec![];
    let raw = data
        .get("offers")
        .and_then(|o| o.as_array())
        .unwrap_or(&empty);
    raw.iter()
        .filter_map(|o| {
            Some(Offer {
                id: o.get("id")?.as_u64()?,
                gpu_name: o.get("gpu_name")?.as_str().unwrap_or("").to_string(),
                num_gpus: o.get("num_gpus").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
                gpu_ram_mb: o.get("gpu_ram").and_then(|v| v.as_f64()).unwrap_or(0.0) as u64,
                disk_gb: o.get("disk_space").and_then(|v| v.as_f64()).unwrap_or(0.0),
                dph_total: o.get("dph_total").and_then(|v| v.as_f64()).unwrap_or(0.0),
                inet_down: o.get("inet_down").and_then(|v| v.as_f64()).unwrap_or(0.0),
                inet_up: o.get("inet_up").and_then(|v| v.as_f64()).unwrap_or(0.0),
                reliability: o
                    .get("reliability2")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                dlperf: o.get("dlperf").and_then(|v| v.as_f64()).unwrap_or(0.0),
            })
        })
        .collect()
}

/// Rent an instance from an offer. Returns the new instance ID.
pub async fn create_instance(
    offer_id: u64,
    image: &str,
    onstart: &str,
    env: &[(String, String)],
    disk_gb: f64,
    label: &str,
) -> Result<u64> {
    let (client, key) = client()?;

    let env_map: serde_json::Map<String, serde_json::Value> = env
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    let extra_env: Vec<serde_json::Value> = env.iter().map(|(k, v)| json!([k, v])).collect();

    let payload = json!({
        "client_id": "me",
        "image": image,
        "disk": disk_gb,
        "onstart": onstart,
        "label": label,
        "env": env_map,
        "extra_env": extra_env,
    });

    let resp = client
        .put(format!("{API_BASE}/asks/{offer_id}/"))
        .bearer_auth(&key)
        .json(&payload)
        .send()
        .await
        .context("Vast.ai instance create request failed")?;
    let data = check(resp, "instance create").await?;

    data.get("new_contract")
        .and_then(|v| v.as_u64())
        .with_context(|| format!("Vast.ai did not return an instance ID: {data}"))
}

/// Current status of one instance.
pub async fn get_instance(instance_id: u64) -> Result<Instance> {
    let (client, key) = client()?;
    let resp = client
        .get(format!("{API_BASE}/instances/{instance_id}/"))
        .bearer_auth(&key)
        .send()
        .await
        .context("Vast.ai instance status request failed")?;
    let data = check(resp, "instance status").await?;

    // Single instance: {"instances": {...}}; some responses use a list.
    let inst = match data.get("instances") {
        Some(serde_json::Value::Array(list)) => list.first().cloned().unwrap_or(data.clone()),
        Some(obj @ serde_json::Value::Object(_)) => obj.clone(),
        _ => data.clone(),
    };
    parse_instance(&inst, instance_id)
}

/// All instances on the account.
pub async fn list_instances() -> Result<Vec<Instance>> {
    let (client, key) = client()?;
    let resp = client
        .get(format!("{API_BASE}/instances/"))
        .bearer_auth(&key)
        .send()
        .await
        .context("Vast.ai instance list request failed")?;
    let data = check(resp, "instance list").await?;

    let empty = vec![];
    let list = data
        .get("instances")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    Ok(list
        .iter()
        .filter_map(|i| parse_instance(i, 0).ok())
        .collect())
}

fn parse_instance(inst: &serde_json::Value, fallback_id: u64) -> Result<Instance> {
    #[derive(Deserialize)]
    struct Raw {
        id: Option<u64>,
        actual_status: Option<String>,
        gpu_name: Option<String>,
        ssh_host: Option<String>,
        ssh_port: Option<u16>,
        dph_total: Option<f64>,
        label: Option<String>,
    }
    let raw: Raw = serde_json::from_value(inst.clone())
        .context("Vast.ai instance: unexpected response shape")?;
    Ok(Instance {
        id: raw.id.unwrap_or(fallback_id),
        actual_status: raw.actual_status.unwrap_or_else(|| "unknown".into()),
        gpu_name: raw.gpu_name.unwrap_or_default(),
        ssh_host: raw.ssh_host,
        ssh_port: raw.ssh_port,
        dph_total: raw.dph_total.unwrap_or(0.0),
        label: raw.label,
    })
}

/// Destroy (terminate) an instance. Billing stops.
pub async fn destroy_instance(instance_id: u64) -> Result<()> {
    let (client, key) = client()?;
    let resp = client
        .delete(format!("{API_BASE}/instances/{instance_id}/"))
        .bearer_auth(&key)
        .send()
        .await
        .context("Vast.ai instance destroy request failed")?;
    check(resp, "instance destroy").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_offers_from_search_response() {
        let data = serde_json::json!({
            "offers": [
                {"id": 111, "gpu_name": "RTX 4090", "num_gpus": 1, "gpu_ram": 24576.0,
                 "disk_space": 120.5, "dph_total": 0.42, "inet_down": 5000.0,
                 "inet_up": 800.0, "reliability2": 0.997, "dlperf": 62.1},
                {"id": 222, "gpu_name": "RTX 4090", "dph_total": 0.31}
            ]
        });
        let offers = parse_offers(&data);
        assert_eq!(offers.len(), 2);
        assert_eq!(offers[0].id, 111);
        assert_eq!(offers[0].gpu_ram_mb, 24576);
        assert!((offers[0].dph_total - 0.42).abs() < 1e-9);
        assert_eq!(offers[1].num_gpus, 1);
    }

    #[test]
    fn parses_instance_shapes() {
        let single = serde_json::json!({
            "id": 12345, "actual_status": "running", "gpu_name": "RTX 4090",
            "ssh_host": "ssh4.vast.ai", "ssh_port": 22222, "dph_total": 0.44,
            "label": "modl-pod-train"
        });
        let inst = parse_instance(&single, 0).unwrap();
        assert_eq!(inst.id, 12345);
        assert_eq!(inst.ssh_port, Some(22222));
        assert_eq!(inst.actual_status, "running");

        // Missing optional fields
        let sparse = serde_json::json!({"actual_status": "loading"});
        let inst = parse_instance(&sparse, 777).unwrap();
        assert_eq!(inst.id, 777);
        assert!(inst.ssh_host.is_none());
    }

    #[test]
    fn gpu_name_filter_maps_aliases() {
        assert_eq!(gpu_name_filter("rtx4090"), "RTX 4090");
        assert_eq!(gpu_name_filter("4090"), "RTX 4090");
        assert_eq!(gpu_name_filter("a100-80gb"), "A100_80GB");
        assert_eq!(gpu_name_filter("H100"), "H100");
        assert_eq!(gpu_name_filter("Custom GPU"), "Custom GPU");
    }
}
