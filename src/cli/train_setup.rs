use anyhow::Result;
use console::style;

use crate::core::runtime;

pub async fn run(reinstall: bool) -> Result<()> {
    println!(
        "{} Preparing training runtime ({})",
        style("→").cyan(),
        style("profile: trainer-cu124").dim()
    );

    let result = runtime::setup_training(reinstall).await?;

    println!("{} Training setup complete", style("✓").green().bold());
    println!("  Profile: {}", result.profile);
    println!("  Python: {}", result.python_path.display());

    if let Some(cmd) = result.train_command_template {
        println!("  Train cmd: {}", cmd);
        println!("{} Runtime is ready for `modl train`", style("✓").green());
    } else {
        println!(
            "{} Could not auto-detect ai-toolkit train command.",
            style("!").yellow().bold()
        );
        println!(
            "  Set {} and re-run setup if needed.",
            style("MODL_AITOOLKIT_TRAIN_CMD").cyan()
        );
    }

    Ok(())
}
