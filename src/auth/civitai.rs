use anyhow::Result;

/// Verify a Civitai API key is valid
pub async fn verify_key(api_key: &str) -> Result<bool> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://civitai.com/api/v1/models")
        .query(&[("limit", "1")])
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await?;
    Ok(resp.status().is_success())
}
