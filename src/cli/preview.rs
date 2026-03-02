use anyhow::Result;
use console::style;

pub async fn run(port: u16, no_open: bool) -> Result<()> {
    eprintln!(
        "{}",
        style("  Starting training preview UI...").cyan().bold()
    );
    crate::ui::server::start(port, !no_open).await
}
