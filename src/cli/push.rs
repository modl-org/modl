use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use console::style;
use serde_json::{Map, Value};

use crate::core::dataset;
use crate::core::db::Database;
use crate::core::hub::{CreateItemRequest, HubClient, upload_file_presigned};
use crate::core::run_manifest::{self, RunManifest};
use crate::core::store::Store;

struct LoraSourceContext {
    files: Vec<PathBuf>,
    manifest: Option<RunManifest>,
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    kind: &str,
    source: &str,
    name: &str,
    visibility: &str,
    description: Option<&str>,
    base_model: Option<&str>,
    trigger_words: &[String],
    owner: Option<&str>,
) -> Result<()> {
    if visibility != "public" && visibility != "private" {
        bail!("visibility must be 'public' or 'private'");
    }

    let client = HubClient::from_config(true)?;
    let me = client.me().await?;
    let username = owner
        .map(str::to_string)
        .or(me.username.clone())
        .context("No username on account. Set a username in the hub UI first.")?;

    match kind {
        "lora" => {
            let source_ctx = resolve_lora_source_context(source)?;
            let files = source_ctx.files;
            if files.is_empty() {
                bail!("No .safetensors files found to push");
            }

            let db = Database::open().ok();
            let first_meta = db
                .as_ref()
                .and_then(|d| file_local_metadata(d, &files[0]).ok())
                .flatten();
            let inferred_base = base_model
                .map(str::to_string)
                .or_else(|| {
                    source_ctx
                        .manifest
                        .as_ref()
                        .and_then(manifest_inferred_base_model)
                })
                .or_else(|| first_meta.as_ref().and_then(extract_base_model));
            let inferred_triggers = if trigger_words.is_empty() {
                source_ctx
                    .manifest
                    .as_ref()
                    .map(manifest_inferred_trigger_words)
                    .filter(|v| !v.is_empty())
                    .or_else(|| first_meta.as_ref().map(extract_trigger_words))
                    .unwrap_or_default()
            } else {
                trigger_words.to_vec()
            };

            ensure_item(
                &client,
                &username,
                name,
                "lora",
                visibility,
                description,
                inferred_base.as_deref(),
                &inferred_triggers,
            )
            .await?;

            println!(
                "{} Pushing {} file(s) to {}/{}",
                style("→").cyan(),
                files.len(),
                username,
                name
            );

            for (idx, path) in files.iter().enumerate() {
                let mut metadata = base_version_metadata(path, idx, files.len());
                if let Some(ref manifest) = source_ctx.manifest {
                    let manifest_meta = manifest_metadata_for_file(manifest, path);
                    merge_value_object(&mut metadata, &manifest_meta);
                }
                if let Some(ref db) = db
                    && let Ok(Some(local)) = file_local_metadata(db, path)
                {
                    merge_value_object(&mut metadata, &local);
                }
                push_one_file(&client, &username, name, path, Some(metadata)).await?;
            }
        }
        "dataset" => {
            let src_path = PathBuf::from(source);
            if !src_path.exists() {
                bail!("Dataset source not found: {}", src_path.display());
            }

            let (archive_path, metadata, _temp_zip) = if src_path.is_dir() {
                let temp_zip = std::env::temp_dir().join(format!(
                    "modl-dataset-{}-{}.zip",
                    name,
                    Utc::now().timestamp()
                ));
                create_zip_from_dir(&src_path, &temp_zip)?;
                let stats = dataset::scan(&src_path).ok();

                let mut meta = Map::new();
                meta.insert("source".to_string(), Value::String("modl-cli".to_string()));
                meta.insert("kind".to_string(), Value::String("dataset".to_string()));
                meta.insert(
                    "pushed_at".to_string(),
                    Value::String(Utc::now().to_rfc3339()),
                );
                meta.insert(
                    "source_path".to_string(),
                    Value::String(src_path.display().to_string()),
                );
                if let Some(info) = stats {
                    meta.insert("dataset_name".to_string(), Value::String(info.name));
                    meta.insert("image_count".to_string(), Value::from(info.image_count));
                    meta.insert(
                        "captioned_count".to_string(),
                        Value::from(info.captioned_count),
                    );
                    meta.insert(
                        "caption_coverage".to_string(),
                        Value::from(info.caption_coverage),
                    );
                }
                (temp_zip.clone(), Value::Object(meta), Some(temp_zip))
            } else if src_path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("zip"))
            {
                let mut meta = Map::new();
                meta.insert("source".to_string(), Value::String("modl-cli".to_string()));
                meta.insert("kind".to_string(), Value::String("dataset".to_string()));
                meta.insert(
                    "pushed_at".to_string(),
                    Value::String(Utc::now().to_rfc3339()),
                );
                meta.insert(
                    "source_path".to_string(),
                    Value::String(src_path.display().to_string()),
                );
                (src_path.clone(), Value::Object(meta), None)
            } else {
                bail!(
                    "Dataset source must be a directory or .zip file: {}",
                    src_path.display()
                );
            };

            ensure_item(
                &client,
                &username,
                name,
                "dataset",
                visibility,
                description,
                base_model,
                trigger_words,
            )
            .await?;

            push_one_file(
                &client,
                &username,
                name,
                &archive_path,
                Some(metadata.clone()),
            )
            .await?;

            if let Some(temp_zip) = _temp_zip {
                let _ = std::fs::remove_file(temp_zip);
            }
        }
        _ => bail!("Unsupported push kind: {kind}"),
    }

    println!(
        "{} Done. Pull with: {}",
        style("✓").green().bold(),
        style(format!("modl pull {}/{}", username, name)).bold()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn ensure_item(
    client: &HubClient,
    username: &str,
    slug: &str,
    item_type: &str,
    visibility: &str,
    description: Option<&str>,
    base_model: Option<&str>,
    trigger_words: &[String],
) -> Result<()> {
    match client.get_item(username, slug).await {
        Ok(item) => {
            if item.item.item_type != item_type {
                bail!(
                    "Item {}/{} exists with type '{}', not '{}'",
                    username,
                    slug,
                    item.item.item_type,
                    item_type
                );
            }
        }
        Err(e) => {
            if !e.to_string().contains("Item not found") {
                return Err(e).with_context(|| format!("Failed checking item {username}/{slug}"));
            }
            let req = CreateItemRequest {
                slug: slug.to_string(),
                item_type: item_type.to_string(),
                visibility: visibility.to_string(),
                description: description.map(str::to_string),
                tags: Vec::new(),
                base_model: base_model.map(str::to_string),
                trigger_words: trigger_words.to_vec(),
            };
            client.create_item(&req).await?;
            println!(
                "{} Created hub item {}/{} ({})",
                style("✓").green(),
                username,
                slug,
                item_type
            );
        }
    }
    Ok(())
}

async fn push_one_file(
    client: &HubClient,
    username: &str,
    slug: &str,
    path: &Path,
    metadata: Option<Value>,
) -> Result<()> {
    let size_bytes = std::fs::metadata(path)
        .with_context(|| format!("Failed to stat {}", path.display()))?
        .len();
    let sha256 = Store::hash_file(path)?;

    let start = client.push_start(username, slug).await?;
    println!(
        "  {} v{} {}",
        style("↑").cyan(),
        start.version,
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
    );

    upload_file_presigned(&start.upload_url, path, "application/octet-stream").await?;

    client
        .push_complete(
            username,
            slug,
            &start.version_id,
            size_bytes,
            &sha256,
            metadata,
        )
        .await?;
    Ok(())
}

fn resolve_lora_source_context(source: &str) -> Result<LoraSourceContext> {
    let files = resolve_lora_files(source)?;
    let run_name = detect_run_name_for_source(source, &files);
    let manifest = run_name.as_deref().and_then(load_run_manifest_for_name);
    Ok(LoraSourceContext { files, manifest })
}

fn detect_run_name_for_source(source: &str, files: &[PathBuf]) -> Option<String> {
    let source_path = PathBuf::from(source);
    if !source_path.exists() {
        let run_inner = run_manifest::run_inner_dir_for_name(source);
        if run_inner.exists() {
            return Some(source.to_string());
        }
    } else if let Some(run_name) = run_name_from_training_output_path(&source_path) {
        return Some(run_name);
    }

    files
        .iter()
        .find_map(|p| run_name_from_training_output_path(p))
}

fn run_name_from_training_output_path(path: &Path) -> Option<String> {
    let canonical_path = path.canonicalize().ok()?;
    for root in training_output_roots() {
        let Ok(canonical_root) = root.canonicalize() else {
            continue;
        };
        if let Ok(rel) = canonical_path.strip_prefix(&canonical_root) {
            let run_name = rel.components().next()?.as_os_str().to_str()?;
            return Some(run_name.to_string());
        }
    }
    None
}

fn load_run_manifest_for_name(run_name: &str) -> Option<RunManifest> {
    match run_manifest::refresh_manifest_for_run_name(run_name, "prepared") {
        Ok(m) => Some(m),
        Err(e) => {
            let run_inner = run_manifest::run_inner_dir_for_name(run_name);
            match run_manifest::load_manifest(&run_inner) {
                Ok(Some(m)) => Some(m),
                _ => {
                    eprintln!(
                        "{} Could not read run manifest for '{}': {e}",
                        style("⚠").yellow(),
                        run_name
                    );
                    None
                }
            }
        }
    }
}

fn manifest_inferred_base_model(manifest: &RunManifest) -> Option<String> {
    manifest
        .model
        .base_model_id
        .clone()
        .or_else(|| manifest.model.model_name_or_path.clone())
        .or_else(|| manifest.model.base_model_path.clone())
}

fn manifest_inferred_trigger_words(manifest: &RunManifest) -> Vec<String> {
    manifest.model.trigger_words.clone()
}

fn manifest_metadata_for_file(manifest: &RunManifest, path: &Path) -> Value {
    let mut obj = Map::new();
    obj.insert(
        "manifest_file".to_string(),
        Value::String(run_manifest::RUN_MANIFEST_FILE.to_string()),
    );
    obj.insert(
        "manifest_schema_version".to_string(),
        Value::from(manifest.schema_version),
    );
    obj.insert(
        "manifest_generated_at".to_string(),
        Value::String(manifest.generated_at.clone()),
    );
    obj.insert(
        "run_name".to_string(),
        Value::String(manifest.run_name.clone()),
    );
    obj.insert(
        "run_status".to_string(),
        Value::String(manifest.status.clone()),
    );

    if let Some(ref job_id) = manifest.job_id {
        obj.insert("job_id".to_string(), Value::String(job_id.clone()));
    }
    if let Some(base) = manifest_inferred_base_model(manifest) {
        obj.insert("base_model".to_string(), Value::String(base));
    }
    if !manifest.model.trigger_words.is_empty() {
        obj.insert(
            "trigger_words".to_string(),
            Value::Array(
                manifest
                    .model
                    .trigger_words
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(ref t) = manifest.model.lora_type {
        obj.insert("lora_type".to_string(), Value::String(t.clone()));
    }

    let mut dataset_obj = Map::new();
    if let Some(ref v) = manifest.dataset.name {
        dataset_obj.insert("name".to_string(), Value::String(v.clone()));
    }
    if let Some(ref v) = manifest.dataset.local_path {
        dataset_obj.insert("local_path".to_string(), Value::String(v.clone()));
    }
    if let Some(v) = manifest.dataset.image_count {
        dataset_obj.insert("image_count".to_string(), Value::from(v));
    }
    if let Some(v) = manifest.dataset.caption_coverage {
        dataset_obj.insert("caption_coverage".to_string(), Value::from(v));
    }
    if let Some(ref v) = manifest.dataset.hub_ref {
        dataset_obj.insert("hub_ref".to_string(), Value::String(v.clone()));
    }
    if let Some(ref v) = manifest.dataset.default_caption {
        dataset_obj.insert("default_caption".to_string(), Value::String(v.clone()));
    }
    if !dataset_obj.is_empty() {
        obj.insert("dataset".to_string(), Value::Object(dataset_obj));
    }

    let mut training_obj = Map::new();
    if let Some(ref v) = manifest.training.preset {
        training_obj.insert("preset".to_string(), Value::String(v.clone()));
    }
    if let Some(v) = manifest.training.steps {
        training_obj.insert("steps".to_string(), Value::from(v));
    }
    if let Some(v) = manifest.training.save_every {
        training_obj.insert("save_every".to_string(), Value::from(v));
    }
    if let Some(v) = manifest.training.sample_every {
        training_obj.insert("sample_every".to_string(), Value::from(v));
    }
    if let Some(ref v) = manifest.training.optimizer {
        training_obj.insert("optimizer".to_string(), Value::String(v.clone()));
    }
    if let Some(v) = manifest.training.learning_rate {
        training_obj.insert("learning_rate".to_string(), Value::from(v));
    }
    if let Some(v) = manifest.training.batch_size {
        training_obj.insert("batch_size".to_string(), Value::from(v));
    }
    if !manifest.training.resolution.is_empty() {
        training_obj.insert(
            "resolution".to_string(),
            Value::Array(
                manifest
                    .training
                    .resolution
                    .iter()
                    .copied()
                    .map(Value::from)
                    .collect(),
            ),
        );
    }
    if let Some(v) = manifest.training.seed {
        training_obj.insert("seed".to_string(), Value::from(v));
    }
    if let Some(ref v) = manifest.training.resume_from {
        training_obj.insert("resume_from".to_string(), Value::String(v.clone()));
    }
    if !training_obj.is_empty() {
        obj.insert("training".to_string(), Value::Object(training_obj));
    }

    if !manifest.sample_prompts.is_empty() {
        obj.insert(
            "sample_prompts".to_string(),
            Value::Array(
                manifest
                    .sample_prompts
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }

    if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
        if let Some(checkpoint) = manifest
            .checkpoints
            .iter()
            .find(|c| c.file_name == file_name)
        {
            if let Some(step) = checkpoint.step {
                obj.insert("checkpoint_step".to_string(), Value::from(step));
                if let Some(group) = manifest.sample_groups.iter().find(|g| g.step == step) {
                    let samples: Vec<Value> = group
                        .images
                        .iter()
                        .map(|img| {
                            let mut one = Map::new();
                            one.insert(
                                "relative_path".to_string(),
                                Value::String(img.relative_path.clone()),
                            );
                            one.insert(
                                "file_name".to_string(),
                                Value::String(img.file_name.clone()),
                            );
                            if let Some(idx) = img.prompt_index {
                                one.insert("prompt_index".to_string(), Value::from(idx));
                            }
                            if let Some(ref p) = img.prompt {
                                one.insert("prompt".to_string(), Value::String(p.clone()));
                            }
                            Value::Object(one)
                        })
                        .collect();
                    obj.insert("sample_images".to_string(), Value::Array(samples));
                }
            }
            obj.insert("is_final".to_string(), Value::Bool(checkpoint.is_final));
        } else if let Some(step) = step_from_checkpoint_file_name(file_name) {
            obj.insert("checkpoint_step".to_string(), Value::from(step));
        }
    }

    Value::Object(obj)
}

fn resolve_lora_files(source: &str) -> Result<Vec<PathBuf>> {
    let path = PathBuf::from(source);
    if path.exists() {
        if path.is_file() {
            if !is_safetensors(&path) {
                bail!("Expected a .safetensors file: {}", path.display());
            }
            return Ok(vec![path]);
        }
        if path.is_dir() {
            let mut files = Vec::new();
            collect_safetensors(&path, &mut files)?;
            sort_lora_files(&mut files);
            return Ok(files);
        }
    }

    // Fallback: try artifact id/name from SQLite.
    let db = Database::open().context("Failed to open local DB")?;
    if let Some(artifact) = db.find_artifact(source)? {
        let p = PathBuf::from(artifact.path);
        if p.exists() && is_safetensors(&p) {
            return Ok(vec![p]);
        }
    }

    // Fallback: treat source as training run name.
    for root in training_output_roots() {
        let run_dir = root.join(source);
        if run_dir.exists() && run_dir.is_dir() {
            let mut files = Vec::new();
            collect_safetensors(&run_dir, &mut files)?;
            sort_lora_files(&mut files);
            if !files.is_empty() {
                return Ok(files);
            }
        }
    }

    bail!("Could not resolve LoRA source: {source}")
}

fn collect_safetensors(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_safetensors(&path, out)?;
        } else if is_safetensors(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_safetensors(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
}

fn sort_lora_files(files: &mut [PathBuf]) {
    files.sort_by(|a, b| {
        let a_name = a.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let b_name = b.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        match (
            step_from_checkpoint_file_name(a_name),
            step_from_checkpoint_file_name(b_name),
        ) {
            (Some(sa), Some(sb)) => sa.cmp(&sb),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a_name.cmp(b_name),
        }
    });
}

fn step_from_checkpoint_file_name(file_name: &str) -> Option<u32> {
    let stem = Path::new(file_name).file_stem()?.to_str()?;
    let (_, suffix) = stem.rsplit_once('_')?;
    if !suffix.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    suffix.parse::<u32>().ok()
}

fn training_output_roots() -> Vec<PathBuf> {
    let root = crate::core::paths::modl_root();
    vec![
        root.join("training_output"),
        root.join(".modl").join("training_output"),
    ]
}

fn file_local_metadata(db: &Database, path: &Path) -> Result<Option<Value>> {
    let p = path.display().to_string();
    let Some(artifact) = db.find_artifact_by_path(&p)? else {
        return Ok(None);
    };

    let mut obj = Map::new();
    obj.insert(
        "local_artifact_id".to_string(),
        Value::String(artifact.artifact_id),
    );
    obj.insert("local_kind".to_string(), Value::String(artifact.kind));
    obj.insert("local_sha256".to_string(), Value::String(artifact.sha256));
    obj.insert(
        "local_size_bytes".to_string(),
        Value::from(artifact.size_bytes),
    );
    if let Some(meta) = artifact.metadata
        && let Ok(v) = serde_json::from_str::<Value>(&meta)
        && let Some(extra) = v.as_object()
    {
        for (k, val) in extra {
            obj.insert(k.clone(), val.clone());
        }
    }
    Ok(Some(Value::Object(obj)))
}

fn base_version_metadata(path: &Path, index: usize, total: usize) -> Value {
    let mut obj = Map::new();
    obj.insert("source".to_string(), Value::String("modl-cli".to_string()));
    obj.insert("kind".to_string(), Value::String("lora".to_string()));
    obj.insert(
        "pushed_at".to_string(),
        Value::String(Utc::now().to_rfc3339()),
    );
    obj.insert(
        "file_name".to_string(),
        Value::String(
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model.safetensors")
                .to_string(),
        ),
    );
    obj.insert(
        "local_path".to_string(),
        Value::String(path.display().to_string()),
    );
    obj.insert("checkpoint_index".to_string(), Value::from(index + 1));
    obj.insert("checkpoint_total".to_string(), Value::from(total));
    obj.insert(
        "is_checkpoint".to_string(),
        Value::Bool(total > 1 && index < total - 1),
    );
    Value::Object(obj)
}

fn merge_value_object(target: &mut Value, extra: &Value) {
    if let (Some(t), Some(e)) = (target.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            t.insert(k.clone(), v.clone());
        }
    }
}

fn extract_base_model(meta: &Value) -> Option<String> {
    meta.get("base_model")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn extract_trigger_words(meta: &Value) -> Vec<String> {
    if let Some(arr) = meta.get("trigger_words").and_then(|v| v.as_array()) {
        return arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
    }
    if let Some(one) = meta.get("trigger_word").and_then(|v| v.as_str()) {
        return vec![one.to_string()];
    }
    Vec::new()
}

fn create_zip_from_dir(source_dir: &Path, zip_path: &Path) -> Result<()> {
    let file = std::fs::File::create(zip_path)
        .with_context(|| format!("Failed to create zip {}", zip_path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    let base = source_dir.canonicalize()?;
    add_dir_to_zip(&base, &base, &mut zip, &options)?;
    zip.finish().context("Failed to finalize zip file")?;
    Ok(())
}

fn add_dir_to_zip(
    base: &Path,
    dir: &Path,
    zip: &mut zip::ZipWriter<std::fs::File>,
    options: &zip::write::SimpleFileOptions,
) -> Result<()> {
    use std::io::Write;

    for entry in
        std::fs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            add_dir_to_zip(base, &path, zip, options)?;
            continue;
        }
        let rel = path
            .strip_prefix(base)
            .with_context(|| format!("Failed to strip base prefix for {}", path.display()))?;
        let rel_name = rel.to_string_lossy().replace('\\', "/");

        zip.start_file(rel_name, *options)
            .context("Failed to start zip entry")?;
        let bytes =
            std::fs::read(&path).with_context(|| format!("Failed to read {}", path.display()))?;
        zip.write_all(&bytes)
            .context("Failed to write zip entry bytes")?;
    }
    Ok(())
}
