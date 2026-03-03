use anyhow::Result;
use console::style;
use indicatif::HumanBytes;

use crate::core::db::Database;
use crate::core::registry::RegistryIndex;

pub async fn run(id: &str) -> Result<()> {
    let index = RegistryIndex::load_or_fetch().await?;
    let db = Database::open()?;

    // Try registry first, fall back to trained artifact lookup
    match index.find(id) {
        Some(manifest) => show_registry_model(&db, id, manifest),
        None => show_trained_artifact(&db, id),
    }
}

fn show_registry_model(
    db: &Database,
    id: &str,
    manifest: &crate::core::manifest::Manifest,
) -> Result<()> {
    let installed = db.is_installed(id)?;

    // Header
    println!(
        "{} {}",
        style(&manifest.name).bold().cyan(),
        if installed {
            style("[installed]").green().to_string()
        } else {
            String::new()
        }
    );
    println!(
        "  {} · {}",
        style(&manifest.asset_type).dim(),
        style(id).dim()
    );
    println!();

    // Description
    if let Some(ref desc) = manifest.description {
        for line in desc.trim().lines() {
            println!("{}", line);
        }
        println!();
    }

    // Metadata
    if let Some(ref author) = manifest.author {
        println!("  Author:       {}", author);
    }
    if let Some(ref arch) = manifest.architecture {
        println!("  Architecture: {}", arch);
    }
    if let Some(ref license) = manifest.license {
        println!("  License:      {}", license);
    }
    if let Some(ref homepage) = manifest.homepage {
        println!("  Homepage:     {}", style(homepage).underlined());
    }
    if let Some(rating) = manifest.rating {
        println!("  Rating:       {:.1} / 5.0", rating);
    }
    if !manifest.tags.is_empty() {
        println!("  Tags:         {}", manifest.tags.join(", "));
    }

    // Variants
    if !manifest.variants.is_empty() {
        println!();
        println!("  {}", style("Variants:").bold());
        for v in &manifest.variants {
            let vram = v
                .vram_required
                .map(|mb| format!(" ({}+ MB VRAM)", mb))
                .unwrap_or_default();
            println!(
                "    {} — {} {}{}",
                style(&v.id).cyan(),
                HumanBytes(v.size),
                v.precision.as_deref().unwrap_or(""),
                style(vram).dim()
            );
            if let Some(ref note) = v.note {
                println!("      {}", style(note).dim());
            }
        }
    }

    // Single file
    if let Some(ref file) = manifest.file {
        println!();
        println!("  Size: {}", HumanBytes(file.size));
    }

    // Dependencies
    if !manifest.requires.is_empty() {
        println!();
        println!("  {}", style("Dependencies:").bold());
        for dep in &manifest.requires {
            let reason = dep
                .reason
                .as_deref()
                .map(|r| format!(" — {}", r))
                .unwrap_or_default();
            println!(
                "    {} ({}){}",
                style(&dep.id).cyan(),
                dep.dep_type,
                style(reason).dim()
            );
        }
    }

    // Auth
    if let Some(ref auth) = manifest.auth
        && auth.gated
    {
        println!();
        println!(
            "  {} Requires {} authentication",
            style("!").yellow(),
            style(&auth.provider).bold()
        );
        if let Some(ref url) = auth.terms_url {
            println!("    Accept terms: {}", style(url).underlined());
        }
    }

    // LoRA-specific
    if !manifest.base_models.is_empty() {
        println!();
        println!("  Compatible with: {}", manifest.base_models.join(", "));
    }
    if !manifest.trigger_words.is_empty() {
        println!("  Trigger words:   {}", manifest.trigger_words.join(", "));
    }
    if let Some(w) = manifest.recommended_weight {
        println!("  Recommended weight: {}", w);
    }

    // Defaults
    if let Some(ref defaults) = manifest.defaults {
        println!();
        println!("  {}", style("Recommended settings:").bold());
        if let Some(steps) = defaults.steps {
            println!("    Steps:     {}", steps);
        }
        if let Some(cfg) = defaults.cfg {
            println!("    CFG:       {}", cfg);
        }
        if let Some(ref sampler) = defaults.sampler {
            println!("    Sampler:   {}", sampler);
        }
        if let Some(ref scheduler) = defaults.scheduler {
            println!("    Scheduler: {}", scheduler);
        }
    }

    // Installed status
    if installed {
        let models = db.list_installed(None)?;
        if let Some(m) = models.iter().find(|m| m.id == id) {
            println!();
            println!("  {}", style("Installed:").bold().green());
            if let Some(ref v) = m.variant {
                println!("    Variant:  {}", v);
            }
            println!("    Size:     {}", HumanBytes(m.size));
            println!("    Path:     {}", m.store_path);
        }
    }

    Ok(())
}

fn show_trained_artifact(db: &Database, query: &str) -> Result<()> {
    let artifact = db.find_artifact(query)?.ok_or_else(|| {
        anyhow::anyhow!(
            "'{}' not found in registry or trained outputs. Run `modl update` first?",
            query
        )
    })?;

    // Parse metadata JSON
    let meta: serde_json::Value = artifact
        .metadata
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Null);

    let lora_name = meta
        .get("lora_name")
        .and_then(|v| v.as_str())
        .unwrap_or(query);

    // Header
    println!(
        "{} {}",
        style(lora_name).bold().cyan(),
        style("[trained]").magenta()
    );
    println!(
        "  {} · {}",
        style(&artifact.kind).dim(),
        style(&artifact.artifact_id).dim()
    );
    println!();

    // Look up the job for training params
    let job = artifact
        .job_id
        .as_deref()
        .and_then(|jid| db.get_job(jid).ok().flatten());

    if let Some(ref job) = job {
        // Parse spec_json for training parameters
        let spec: serde_json::Value =
            serde_json::from_str(&job.spec_json).unwrap_or(serde_json::Value::Null);

        let params = &spec["params"];
        let dataset = &spec["dataset"];
        let model = &spec["model"];

        // Training details
        println!("  {}", style("Training:").bold());
        if let Some(base) = model.get("base_model_id").and_then(|v| v.as_str()) {
            println!("    Base model:    {}", style(base).cyan());
        }
        if let Some(trigger) = params.get("trigger_word").and_then(|v| v.as_str()) {
            println!("    Trigger word:  {}", style(trigger).yellow().bold());
        }
        if let Some(preset) = params.get("preset").and_then(|v| v.as_str()) {
            println!("    Preset:        {}", preset);
        }
        if let Some(steps) = params.get("steps").and_then(|v| v.as_u64()) {
            println!("    Steps:         {}", steps);
        }
        if let Some(rank) = params.get("rank").and_then(|v| v.as_u64()) {
            println!("    Rank:          {}", rank);
        }
        if let Some(lr) = params.get("learning_rate").and_then(|v| v.as_f64()) {
            println!("    Learning rate: {}", lr);
        }
        if let Some(optimizer) = params.get("optimizer").and_then(|v| v.as_str()) {
            println!("    Optimizer:     {}", optimizer);
        }
        if let Some(resolution) = params.get("resolution").and_then(|v| v.as_u64()) {
            println!("    Resolution:    {}", resolution);
        }

        // Dataset info
        if let Some(ds_name) = dataset.get("name").and_then(|v| v.as_str()) {
            println!();
            println!("  {}", style("Dataset:").bold());
            println!("    Name:          {}", ds_name);
            if let Some(count) = dataset.get("image_count").and_then(|v| v.as_u64()) {
                println!("    Images:        {}", count);
            }
        }

        // Job status
        println!();
        println!("  {}", style("Job:").bold());
        println!("    ID:            {}", &job.job_id);
        println!(
            "    Status:        {}",
            match job.status.as_str() {
                "completed" => style(&job.status).green(),
                "failed" => style(&job.status).red(),
                _ => style(&job.status).yellow(),
            }
        );
        println!("    Target:        {}", &job.target);
        if let Some(ref started) = job.started_at {
            println!("    Started:       {}", started);
        }
        if let Some(ref completed) = job.completed_at {
            println!("    Completed:     {}", completed);
        }
    }

    // Artifact details
    println!();
    println!("  {}", style("Output:").bold());
    println!("    Path:          {}", &artifact.path);
    println!("    Size:          {}", HumanBytes(artifact.size_bytes));
    println!("    SHA256:        {}", &artifact.sha256);
    println!("    Created:       {}", &artifact.created_at);

    // Symlink info
    let loras_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".modl")
        .join("loras");
    let symlink_path = loras_dir.join(format!("{}.safetensors", lora_name));
    if symlink_path.exists() {
        println!("    Symlink:       {}", style(symlink_path.display()).dim());
    }

    Ok(())
}
