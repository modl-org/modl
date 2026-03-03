use anyhow::Result;
use console::style;
use dialoguer::Password;

use crate::auth::{AuthStore, CivitaiAuth, HuggingFaceAuth};
use crate::cli::AuthProvider;

pub async fn run(provider: AuthProvider) -> Result<()> {
    match provider {
        AuthProvider::Huggingface => configure_huggingface().await,
        AuthProvider::Civitai => configure_civitai().await,
    }
}

async fn configure_huggingface() -> Result<()> {
    println!("{} Configure HuggingFace authentication", style("→").cyan());

    // Check for HF_TOKEN env var first (non-interactive / CI-friendly)
    let token: String = if let Ok(env_token) = std::env::var("HF_TOKEN") {
        if !env_token.is_empty() {
            println!("  {} Found HF_TOKEN in environment", style("→").dim());
            env_token
        } else {
            prompt_hf_token()?
        }
    } else {
        prompt_hf_token()?
    };

    if !token.starts_with("hf_") {
        println!(
            "  {} Token doesn't start with 'hf_' — are you sure it's correct?",
            style("!").yellow()
        );
    }

    // Verify token
    println!("  Verifying token...");
    match crate::auth::huggingface::verify_token(&token).await {
        Ok(true) => {
            println!("  {} Token verified!", style("✓").green());
        }
        Ok(false) => {
            println!(
                "  {} Token rejected by HuggingFace. Check it's valid.",
                style("✗").red()
            );
            anyhow::bail!("Invalid token");
        }
        Err(e) => {
            println!(
                "  {} Could not verify (network issue?): {}",
                style("!").yellow(),
                e
            );
            println!("  Saving anyway — you can re-run `modl auth huggingface` later.");
        }
    }

    let mut store = AuthStore::load().unwrap_or_default();
    store.huggingface = Some(HuggingFaceAuth { token });
    store.save()?;

    println!(
        "{} HuggingFace auth saved to {}",
        style("✓").green(),
        style("~/.modl/auth.yaml").dim()
    );

    Ok(())
}

async fn configure_civitai() -> Result<()> {
    println!("{} Configure Civitai authentication", style("→").cyan());
    println!();
    println!(
        "  1. Go to {}",
        style("https://civitai.com/user/account").underlined()
    );
    println!("  2. Scroll to 'API Keys' and create one");
    println!();

    let api_key: String = Password::new().with_prompt("Civitai API key").interact()?;

    // Verify key
    println!("  Verifying API key...");
    match crate::auth::civitai::verify_key(&api_key).await {
        Ok(true) => {
            println!("  {} API key verified!", style("✓").green());
        }
        Ok(false) => {
            println!("  {} API key rejected by Civitai.", style("✗").red());
            anyhow::bail!("Invalid API key");
        }
        Err(e) => {
            println!(
                "  {} Could not verify (network issue?): {}",
                style("!").yellow(),
                e
            );
            println!("  Saving anyway.");
        }
    }

    let mut store = AuthStore::load().unwrap_or_default();
    store.civitai = Some(CivitaiAuth { api_key });
    store.save()?;

    println!(
        "{} Civitai auth saved to {}",
        style("✓").green(),
        style("~/.modl/auth.yaml").dim()
    );

    Ok(())
}

fn prompt_hf_token() -> Result<String> {
    println!();
    println!(
        "  1. Go to {}",
        style("https://huggingface.co/settings/tokens").underlined()
    );
    println!("  2. Create a token with 'Read' access");
    println!("  3. For gated models (e.g., Flux Dev), accept terms on the model page first");
    println!();

    let token: String = Password::new()
        .with_prompt("HuggingFace token (hf_...)")
        .interact()?;
    Ok(token)
}
