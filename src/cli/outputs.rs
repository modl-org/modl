use anyhow::{Context, Result, bail};
use clap::Subcommand;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;

use crate::core::db::{ArtifactRecord, Database, JobRecord};

#[derive(Subcommand)]
pub enum OutputCommands {
    /// List recent generation outputs
    Ls {
        /// Show only the last N outputs (default: 20)
        #[arg(long, short = 'n', default_value = "20")]
        limit: usize,
        /// Filter by kind: image, lora, sample_image
        #[arg(long, short = 'k')]
        kind: Option<String>,
    },
    /// Show full metadata for an output (prompt, seed, model, params)
    Show {
        /// Output ID or job ID (prefix match supported)
        id: String,
    },
    /// Open an output image in the system viewer
    Open {
        /// Output ID (prefix match supported)
        id: String,
    },
    /// Search outputs by prompt, model, or LoRA name
    Search {
        /// Search query (matches prompt, model id, lora name)
        query: String,
        /// Maximum results to show
        #[arg(long, short = 'n', default_value = "20")]
        limit: usize,
    },
}

pub async fn run(command: OutputCommands) -> Result<()> {
    match command {
        OutputCommands::Ls { limit, kind } => run_list(limit, kind.as_deref()).await,
        OutputCommands::Show { id } => run_show(&id).await,
        OutputCommands::Open { id } => run_open(&id).await,
        OutputCommands::Search { query, limit } => run_search(&query, limit).await,
    }
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

async fn run_list(limit: usize, kind_filter: Option<&str>) -> Result<()> {
    let db = Database::open()?;
    let artifacts = db.list_artifacts(None)?;

    if artifacts.is_empty() {
        println!("No outputs yet.");
        println!(
            "  Run {} to generate images.",
            style("modl generate \"a photo of...\"").cyan()
        );
        return Ok(());
    }

    // Filter by kind if requested
    let filtered: Vec<&ArtifactRecord> = artifacts
        .iter()
        .filter(|a| {
            if let Some(k) = kind_filter {
                a.kind == k
            } else {
                true
            }
        })
        .take(limit)
        .collect();

    if filtered.is_empty() {
        if let Some(k) = kind_filter {
            println!("No outputs of kind '{k}'.");
        }
        return Ok(());
    }

    // For each artifact, try to extract summary from its job's spec_json
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("ID").fg(Color::Cyan),
        Cell::new("Kind").fg(Color::Cyan),
        Cell::new("Prompt / Name").fg(Color::Cyan),
        Cell::new("Model").fg(Color::Cyan),
        Cell::new("Time").fg(Color::Cyan),
    ]);

    for artifact in &filtered {
        let short_id = short_id(&artifact.artifact_id);
        let summary = artifact_summary(artifact, &db);

        table.add_row(vec![
            Cell::new(&short_id).fg(Color::Yellow),
            Cell::new(&artifact.kind),
            Cell::new(truncate(&summary.prompt_or_name, 50)),
            Cell::new(&summary.model),
            Cell::new(&artifact.created_at),
        ]);
    }

    println!("{table}");
    println!(
        "\n  {} outputs shown. Use {} for details.",
        style(filtered.len()).bold(),
        style("modl outputs show <id>").cyan()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Show
// ---------------------------------------------------------------------------

async fn run_show(id: &str) -> Result<()> {
    let db = Database::open()?;
    let artifact = find_artifact_by_prefix(id, &db)?;

    println!("{}", style("Output Details").bold().underlined());
    println!();
    println!(
        "  {}  {}",
        style("Artifact ID:").dim(),
        artifact.artifact_id
    );
    println!("  {}  {}", style("Kind:").dim(), artifact.kind);
    println!("  {}  {}", style("Path:").dim(), artifact.path);
    println!("  {}  {}", style("Created:").dim(), artifact.created_at);

    if !artifact.sha256.is_empty() {
        println!("  {}  {}", style("SHA256:").dim(), artifact.sha256);
    }
    if artifact.size_bytes > 0 {
        println!(
            "  {}  {}",
            style("Size:").dim(),
            indicatif::HumanBytes(artifact.size_bytes)
        );
    }

    // Show job spec details
    if let Some(ref job_id) = artifact.job_id
        && let Ok(Some(job)) = db.get_job(job_id)
    {
        println!();
        println!("  {}", style("── Job ──").dim());
        println!("  {}  {}", style("Job ID:").dim(), job.job_id);
        println!("  {}  {}", style("Kind:").dim(), job.kind);
        println!("  {}  {}", style("Status:").dim(), job.status);
        println!("  {}  {}", style("Target:").dim(), job.target);

        // Parse and display the spec
        print_spec_details(&job)?;
    }

    // Show metadata JSON if present
    if let Some(ref meta) = artifact.metadata
        && !meta.is_empty()
        && meta != "null"
    {
        println!();
        println!("  {}", style("── Metadata ──").dim());
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta) {
            let pretty = serde_json::to_string_pretty(&parsed).unwrap_or_default();
            for line in pretty.lines() {
                println!("  {}", line);
            }
        } else {
            println!("  {meta}");
        }
    }

    println!();
    Ok(())
}

/// Print the details from a job's spec_json.
fn print_spec_details(job: &JobRecord) -> Result<()> {
    let v: serde_json::Value =
        serde_json::from_str(&job.spec_json).context("Failed to parse spec_json")?;

    match job.kind.as_str() {
        "generate" => {
            if let Some(prompt) = v.get("prompt").and_then(|p| p.as_str()) {
                println!("  {}  {}", style("Prompt:").dim(), prompt);
            }
            if let Some(model) = v.get("model")
                && let Some(id) = model.get("base_model_id").and_then(|m| m.as_str())
            {
                println!("  {}  {}", style("Model:").dim(), id);
            }
            if let Some(lora) = v.get("lora")
                && let Some(name) = lora.get("name").and_then(|n| n.as_str())
            {
                let weight = lora.get("weight").and_then(|w| w.as_f64()).unwrap_or(1.0);
                println!("  {}  {} (weight: {})", style("LoRA:").dim(), name, weight);
            }
            if let Some(params) = v.get("params") {
                let width = params.get("width").and_then(|w| w.as_u64()).unwrap_or(0);
                let height = params.get("height").and_then(|h| h.as_u64()).unwrap_or(0);
                let steps = params.get("steps").and_then(|s| s.as_u64()).unwrap_or(0);
                let guidance = params
                    .get("guidance")
                    .and_then(|g| g.as_f64())
                    .unwrap_or(0.0);
                let seed = params.get("seed").and_then(|s| s.as_u64());
                let count = params.get("count").and_then(|c| c.as_u64()).unwrap_or(1);

                println!("  {}  {}×{}", style("Size:").dim(), width, height);
                println!("  {}  {}", style("Steps:").dim(), steps);
                println!("  {}  {}", style("Guidance:").dim(), guidance);
                if let Some(s) = seed {
                    println!("  {}  {}", style("Seed:").dim(), s);
                }
                if count > 1 {
                    println!("  {}  {}", style("Count:").dim(), count);
                }
            }
        }
        "train" => {
            if let Some(dataset) = v.get("dataset")
                && let Some(name) = dataset.get("name").and_then(|n| n.as_str())
            {
                println!("  {}  {}", style("Dataset:").dim(), name);
            }
            if let Some(model) = v.get("model")
                && let Some(id) = model.get("base_model_id").and_then(|m| m.as_str())
            {
                println!("  {}  {}", style("Base model:").dim(), id);
            }
            if let Some(output) = v.get("output")
                && let Some(name) = output.get("lora_name").and_then(|n| n.as_str())
            {
                println!("  {}  {}", style("LoRA name:").dim(), name);
            }
            if let Some(params) = v.get("params") {
                if let Some(preset) = params.get("preset").and_then(|p| p.as_str()) {
                    println!("  {}  {}", style("Preset:").dim(), preset);
                }
                if let Some(trigger) = params.get("trigger_word").and_then(|t| t.as_str()) {
                    println!("  {}  {}", style("Trigger:").dim(), trigger);
                }
                if let Some(steps) = params.get("steps").and_then(|s| s.as_u64()) {
                    println!("  {}  {}", style("Steps:").dim(), steps);
                }
                if let Some(rank) = params.get("rank").and_then(|r| r.as_u64()) {
                    println!("  {}  {}", style("Rank:").dim(), rank);
                }
                if let Some(lr) = params.get("learning_rate").and_then(|l| l.as_f64()) {
                    println!("  {}  {}", style("LR:").dim(), lr);
                }
            }
        }
        _ => {
            // Unknown kind — dump raw spec
            println!("  {}", serde_json::to_string_pretty(&v).unwrap_or_default());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Open
// ---------------------------------------------------------------------------

async fn run_open(id: &str) -> Result<()> {
    let db = Database::open()?;
    let artifact = find_artifact_by_prefix(id, &db)?;

    let path = std::path::Path::new(&artifact.path);
    if !path.exists() {
        bail!(
            "File not found: {}. It may have been moved or deleted.",
            artifact.path
        );
    }

    println!(
        "{} Opening {}...",
        style("→").cyan(),
        style(&artifact.path).underlined()
    );

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&artifact.path)
            .spawn()
            .context("Failed to open file with xdg-open")?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&artifact.path)
            .spawn()
            .context("Failed to open file")?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", &artifact.path])
            .spawn()
            .context("Failed to open file")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async fn run_search(query: &str, limit: usize) -> Result<()> {
    let db = Database::open()?;
    let jobs = db.list_jobs(None)?;
    let query_lower = query.to_lowercase();

    let mut matching_artifacts: Vec<(ArtifactRecord, String)> = Vec::new();

    for job in &jobs {
        // Search inside spec_json
        let spec_lower = job.spec_json.to_lowercase();
        if !spec_lower.contains(&query_lower) {
            continue;
        }

        // This job matches — pull its artifacts
        let artifacts = db.list_artifacts(Some(&job.job_id))?;
        let summary = job_summary(job);

        for artifact in artifacts {
            matching_artifacts.push((artifact, summary.prompt_or_name.clone()));
            if matching_artifacts.len() >= limit {
                break;
            }
        }
        if matching_artifacts.len() >= limit {
            break;
        }
    }

    if matching_artifacts.is_empty() {
        println!("No outputs matching '{}'.", style(query).italic());
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("ID").fg(Color::Cyan),
        Cell::new("Kind").fg(Color::Cyan),
        Cell::new("Prompt / Name").fg(Color::Cyan),
        Cell::new("Path").fg(Color::Cyan),
        Cell::new("Time").fg(Color::Cyan),
    ]);

    for (artifact, prompt) in &matching_artifacts {
        let short = short_id(&artifact.artifact_id);
        table.add_row(vec![
            Cell::new(&short).fg(Color::Yellow),
            Cell::new(&artifact.kind),
            Cell::new(truncate(prompt, 40)),
            Cell::new(truncate(&artifact.path, 40)),
            Cell::new(&artifact.created_at),
        ]);
    }

    println!("{table}");
    println!(
        "\n  {} results for '{}'",
        style(matching_artifacts.len()).bold(),
        style(query).italic()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the first 8 chars of an ID for display.
fn short_id(id: &str) -> String {
    if id.len() > 12 {
        id[..12].to_string()
    } else {
        id.to_string()
    }
}

/// Truncate a string for table display.
fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}…", &s[..max - 1])
    } else {
        s.to_string()
    }
}

/// Summary extracted from a job's spec_json.
struct JobSummary {
    prompt_or_name: String,
    model: String,
}

/// Extract summary fields from an ArtifactRecord by loading its parent job.
fn artifact_summary(artifact: &ArtifactRecord, db: &Database) -> JobSummary {
    if let Some(ref job_id) = artifact.job_id
        && let Ok(Some(job)) = db.get_job(job_id)
    {
        return job_summary(&job);
    }
    JobSummary {
        prompt_or_name: "—".into(),
        model: "—".into(),
    }
}

/// Extract summary fields from a JobRecord's spec_json.
fn job_summary(job: &JobRecord) -> JobSummary {
    let default = JobSummary {
        prompt_or_name: "—".into(),
        model: "—".into(),
    };

    let v: serde_json::Value = match serde_json::from_str(&job.spec_json) {
        Ok(v) => v,
        Err(_) => return default,
    };

    let model = v
        .get("model")
        .and_then(|m| m.get("base_model_id"))
        .and_then(|m| m.as_str())
        .unwrap_or("—")
        .to_string();

    let prompt_or_name = match job.kind.as_str() {
        "generate" => v
            .get("prompt")
            .and_then(|p| p.as_str())
            .unwrap_or("—")
            .to_string(),
        "train" => v
            .get("output")
            .and_then(|o| o.get("lora_name"))
            .and_then(|n| n.as_str())
            .unwrap_or("—")
            .to_string(),
        _ => "—".into(),
    };

    JobSummary {
        prompt_or_name,
        model,
    }
}

/// Find an artifact by prefix match on artifact_id.
fn find_artifact_by_prefix(prefix: &str, db: &Database) -> Result<ArtifactRecord> {
    let artifacts = db.list_artifacts(None)?;
    let matches: Vec<_> = artifacts
        .into_iter()
        .filter(|a| a.artifact_id.starts_with(prefix))
        .collect();

    match matches.len() {
        0 => bail!("No output found matching '{prefix}'."),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => {
            let ids: Vec<_> = matches.iter().map(|a| short_id(&a.artifact_id)).collect();
            bail!(
                "Ambiguous ID '{prefix}' matches {n} outputs: {}. Be more specific.",
                ids.join(", ")
            );
        }
    }
}
