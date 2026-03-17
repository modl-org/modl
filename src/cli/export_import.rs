use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::core::backup::{
    self, ArchiveManifest, ExportOptions, ExportProgress, ImportOptions, ImportProgress,
};

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

struct CliExportProgress {
    pb: ProgressBar,
}

impl ExportProgress for CliExportProgress {
    fn set_total(&self, total: usize) {
        self.pb.set_length(total as u64);
    }

    fn tick(&self, _item: &str) {
        self.pb.inc(1);
    }
}

pub fn run_export(output: &str, no_outputs: bool, since: Option<&str>) -> Result<()> {
    let output_path = PathBuf::from(output);

    // Ensure parent directory exists
    if let Some(parent) = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty() && !p.exists())
    {
        std::fs::create_dir_all(parent)?;
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:30.cyan/dim}] {pos}/{len} files")
            .unwrap()
            .progress_chars("━╸─"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));

    let progress = CliExportProgress { pb: pb.clone() };

    let opts = ExportOptions {
        output_path: output_path.clone(),
        include_outputs: !no_outputs,
        since: since.map(String::from),
    };

    let result = backup::export(&opts, &progress)?;
    pb.finish_and_clear();

    let m = &result.manifest;
    println!(
        "\n{} Exported to {}\n",
        style("✓").green().bold(),
        style(output_path.display()).cyan()
    );
    println!("  Database:  {} jobs", style(m.stats.job_count).bold());
    println!("  LoRAs:     {} trained", style(m.stats.lora_count).bold());
    if m.contents.outputs {
        println!("  Outputs:   {} files", style(m.stats.output_count).bold());
    } else {
        println!("  Outputs:   {}", style("excluded").dim());
    }
    println!(
        "  Auth:      {}",
        if m.contents.auth {
            style("included").to_string()
        } else {
            style("not found").dim().to_string()
        }
    );
    println!(
        "  Size:      {}",
        style(format_bytes(m.stats.total_size)).bold()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

struct CliImportProgress {
    dry_run: bool,
    count: AtomicUsize,
}

impl ImportProgress for CliImportProgress {
    fn summary(
        &self,
        manifest: &ArchiveManifest,
        has_db: bool,
        has_auth: bool,
        lora_count: usize,
        output_count: usize,
    ) {
        if self.dry_run {
            println!(
                "\n{} Dry run — no changes will be made\n",
                style("→").cyan().bold()
            );
        } else {
            println!();
        }

        println!(
            "  Archive:   modl {} ({})",
            style(&manifest.modl_version).bold(),
            style(&manifest.created_at[..10]).dim()
        );
        if has_db {
            println!(
                "  Database:  {} jobs",
                style(manifest.stats.job_count).bold()
            );
        }
        if lora_count > 0 {
            println!("  LoRAs:     {}", style(lora_count).bold());
        }
        if output_count > 0 {
            println!("  Outputs:   {}", style(output_count).bold());
        }
        if has_auth {
            println!("  Auth:      {}", style("present").bold());
        }
        println!();
    }

    fn tick(&self, msg: &str) {
        self.count.fetch_add(1, Ordering::Relaxed);
        println!("  {} {}", style("✓").green(), msg);
    }
}

pub fn run_import(path: &str, dry_run: bool, overwrite: bool) -> Result<()> {
    let archive_path = PathBuf::from(path);

    let progress = CliImportProgress {
        dry_run,
        count: AtomicUsize::new(0),
    };

    let opts = ImportOptions {
        archive_path,
        dry_run,
        overwrite,
    };

    let _result = backup::import(&opts, &progress)?;

    if dry_run {
        println!(
            "  {}",
            style("Run without --dry-run to apply changes.").dim()
        );
    } else {
        println!("\n{} Import complete.", style("✓").green().bold());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
