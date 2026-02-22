use anyhow::Result;

/// Verify a HuggingFace token is valid
pub async fn verify_token(token: &str) -> Result<bool> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://huggingface.co/api/whoami-v2")
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await?;
    Ok(resp.status().is_success())
}
