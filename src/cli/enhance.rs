use anyhow::Result;
use console::style;

use crate::core::enhance::{self, EnhanceIntensity};

pub async fn run(prompt: &str, model: Option<&str>, intensity: &str, json: bool) -> Result<()> {
    let intensity: EnhanceIntensity = intensity.parse()?;
    let result = enhance::enhance_prompt(prompt, model, intensity)?;

    if json {
        println!("{}", serde_json::to_string(&result)?);
    } else {
        println!(
            "{} Enhanced prompt ({})",
            style("✓").green().bold(),
            style(&result.backend).dim()
        );
        println!();
        println!("  {} {}", style("Original:").dim(), result.original);
        println!("  {} {}", style("Enhanced:").cyan(), result.enhanced);
    }

    Ok(())
}
