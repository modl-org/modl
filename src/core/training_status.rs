//! Parse live training progress from log files and process state.
//!
//! Works regardless of whether training was launched via `modl train` or
//! directly via `python run.py`. Scans `~/.modl/training_output/` for
//! active log files and parses tqdm-style progress lines.

use anyhow::Result;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Live status of a single training run.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingProgress {
    /// Run name (e.g. "art-style-v5")
    pub name: String,
    /// Whether a Python process is actively training this run
    pub is_running: bool,
    /// Current step (parsed from log)
    pub current_step: Option<u64>,
    /// Total steps
    pub total_steps: Option<u64>,
    /// Percentage complete (0-100)
    pub percent: Option<f32>,
    /// Elapsed time string (e.g. "30:31")
    pub elapsed: Option<String>,
    /// Remaining time string (e.g. "1:46:04")
    pub eta: Option<String>,
    /// Iterations per second
    pub speed: Option<f32>,
    /// Current learning rate
    pub lr: Option<String>,
    /// Latest loss value
    pub loss: Option<f64>,
    /// Base model architecture (from config)
    pub arch: Option<String>,
    /// Base model name (from config)
    pub base_model: Option<String>,
    /// Trigger word
    pub trigger_word: Option<String>,
    /// Path to log file
    pub log_file: Option<String>,
    /// Last modified time of log file (epoch seconds)
    pub log_updated_at: Option<u64>,
    /// Latest sample images (relative paths for /files/ endpoint)
    pub latest_samples: Vec<String>,
    /// Latest checkpoint file
    pub latest_checkpoint: Option<String>,
}

/// Get status of all training runs, or just active ones.
pub fn get_all_status(active_only: bool) -> Result<Vec<TrainingProgress>> {
    let output_dir = training_output_dir();
    if !output_dir.exists() {
        return Ok(vec![]);
    }

    let running_configs = find_running_training_configs();

    let mut results = Vec::new();

    for entry in std::fs::read_dir(&output_dir)?.flatten() {
        if !entry.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }

        let name = entry.file_name().to_string_lossy().to_string();
        let log_path = output_dir.join(format!("{name}.log"));

        let is_running = running_configs.iter().any(|c| c == &name);

        if active_only && !is_running {
            // Also include recently-active logs (modified in last 60s)
            if !is_recently_modified(&log_path, 60) {
                continue;
            }
        }

        let mut progress = parse_log_progress(&name, &log_path);
        progress.is_running = is_running;

        // Parse config for metadata
        if let Some(config) = load_run_config(&name) {
            progress.arch = config.arch;
            progress.base_model = config.base_model;
            progress.trigger_word = config.trigger_word;
        }

        // Find latest samples and checkpoint
        let run_inner = output_dir.join(&name).join(&name);
        progress.latest_samples = find_latest_samples(&name, &run_inner);
        progress.latest_checkpoint = find_latest_checkpoint(&run_inner);

        results.push(progress);
    }

    // Sort: running first, then by name
    results.sort_by(|a, b| {
        b.is_running
            .cmp(&a.is_running)
            .then_with(|| a.name.cmp(&b.name))
    });

    Ok(results)
}

/// Get status of a specific run by name.
pub fn get_status(name: &str) -> Result<TrainingProgress> {
    let output_dir = training_output_dir();
    let log_path = output_dir.join(format!("{name}.log"));
    let running_configs = find_running_training_configs();

    let mut progress = parse_log_progress(name, &log_path);
    progress.is_running = running_configs.iter().any(|c| c == name);

    if let Some(config) = load_run_config(name) {
        progress.arch = config.arch;
        progress.base_model = config.base_model;
        progress.trigger_word = config.trigger_word;
    }

    let run_inner = output_dir.join(name).join(name);
    progress.latest_samples = find_latest_samples(name, &run_inner);
    progress.latest_checkpoint = find_latest_checkpoint(&run_inner);

    Ok(progress)
}

// ---------------------------------------------------------------------------
// Log parsing
// ---------------------------------------------------------------------------

/// Parse a tqdm-style progress line from the end of a log file.
///
/// Expected format:
/// `art-style-v5:  24%|██▍       | 2440/10175 [30:31<1:46:04,  1.22it/s, lr: 1.0e-04 loss: 4.170e-02]`
fn parse_log_progress(name: &str, log_path: &Path) -> TrainingProgress {
    let mut progress = TrainingProgress {
        name: name.to_string(),
        is_running: false,
        current_step: None,
        total_steps: None,
        percent: None,
        elapsed: None,
        eta: None,
        speed: None,
        lr: None,
        loss: None,
        arch: None,
        base_model: None,
        trigger_word: None,
        log_file: None,
        log_updated_at: None,
        latest_samples: vec![],
        latest_checkpoint: None,
    };

    if !log_path.exists() {
        return progress;
    }

    progress.log_file = Some(log_path.to_string_lossy().to_string());

    if let Ok(meta) = log_path.metadata()
        && let Ok(modified) = meta.modified()
        && let Ok(duration) = modified.duration_since(SystemTime::UNIX_EPOCH)
    {
        progress.log_updated_at = Some(duration.as_secs());
    }

    // Read the last chunk of the log file (tqdm overwrites lines with \r)
    let tail = read_tail(log_path, 8192);
    if tail.is_empty() {
        return progress;
    }

    // Find the last tqdm progress line
    // Split on \r (carriage return) since tqdm uses \r to overwrite
    let segments: Vec<&str> = tail.split('\r').collect();

    for segment in segments.iter().rev() {
        let line = segment.trim();
        if line.is_empty() {
            continue;
        }

        // Try to match: <name>:  <pct>%|...|  <step>/<total> [<elapsed><eta>, <speed>, ...]
        if let Some(parsed) = parse_tqdm_line(line) {
            progress.current_step = Some(parsed.step);
            progress.total_steps = Some(parsed.total);
            progress.percent = Some(parsed.percent);
            progress.elapsed = parsed.elapsed;
            progress.eta = parsed.eta;
            progress.speed = parsed.speed;
            progress.lr = parsed.lr;
            progress.loss = parsed.loss;
            break;
        }
    }

    progress
}

struct TqdmParsed {
    step: u64,
    total: u64,
    percent: f32,
    elapsed: Option<String>,
    eta: Option<String>,
    speed: Option<f32>,
    lr: Option<String>,
    loss: Option<f64>,
}

fn parse_tqdm_line(line: &str) -> Option<TqdmParsed> {
    // Find step/total pattern: <digits>/<digits>
    // We look for the pattern after the progress bar: | <step>/<total> [
    let step_re = find_step_total(line)?;

    let percent = if step_re.total > 0 {
        (step_re.step as f32 / step_re.total as f32) * 100.0
    } else {
        0.0
    };

    // Parse time info: [elapsed<eta, speed]
    let elapsed = extract_between(line, "[", "<");
    let eta = extract_between(line, "<", ",");

    // Parse speed: e.g. "1.22it/s"
    let speed = extract_speed(line);

    // Parse lr: e.g. "lr: 1.0e-04"
    let lr = extract_kv(line, "lr:");

    // Parse loss: e.g. "loss: 4.170e-02"
    let loss = extract_kv(line, "loss:").and_then(|v| v.trim().parse::<f64>().ok());

    Some(TqdmParsed {
        step: step_re.step,
        total: step_re.total,
        percent,
        elapsed,
        eta,
        speed,
        lr,
        loss,
    })
}

struct StepTotal {
    step: u64,
    total: u64,
}

fn find_step_total(line: &str) -> Option<StepTotal> {
    // Look for pattern: | <digits>/<digits> [
    // Or just <digits>/<digits> anywhere meaningful
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    // Find the last occurrence of <digits>/<digits> that's preceded by '| '
    let mut best: Option<StepTotal> = None;

    while i < len {
        // Look for a digit
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < len && bytes[i] == b'/' {
                let step_str = &line[start..i];
                i += 1; // skip '/'
                let total_start = i;
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i > total_start {
                    let total_str = &line[total_start..i];
                    if let (Ok(step), Ok(total)) =
                        (step_str.parse::<u64>(), total_str.parse::<u64>())
                    {
                        // Prefer matches that look like tqdm (preceded by | or whitespace)
                        if total > 10 {
                            best = Some(StepTotal { step, total });
                        }
                    }
                }
            }
        }
        i += 1;
    }

    best
}

fn extract_between(s: &str, start_delim: &str, end_delim: &str) -> Option<String> {
    // Find the last occurrence of brackets context [elapsed<eta, ...]
    let bracket_start = s.rfind(start_delim)?;
    let rest = &s[bracket_start + start_delim.len()..];
    let end = rest.find(end_delim)?;
    let value = rest[..end].trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

fn extract_speed(s: &str) -> Option<f32> {
    // Look for patterns like "1.22it/s" or "1.38s/it"
    for word in s.split_whitespace() {
        let w = word.trim_end_matches(']').trim_end_matches(',');
        if w.ends_with("it/s") {
            let num = w.strip_suffix("it/s")?;
            return num.parse::<f32>().ok();
        }
        if w.ends_with("s/it") {
            let num = w.strip_suffix("s/it")?;
            let secs_per_it: f32 = num.parse().ok()?;
            if secs_per_it > 0.0 {
                return Some(1.0 / secs_per_it);
            }
        }
    }
    None
}

fn extract_kv(s: &str, key: &str) -> Option<String> {
    let idx = s.find(key)?;
    let rest = &s[idx + key.len()..];
    // Take the first whitespace-delimited token after the key
    let value = rest
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches(']')
        .trim_end_matches(',')
        .to_string();
    if value.is_empty() { None } else { Some(value) }
}

fn read_tail(path: &Path, max_bytes: u64) -> String {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };

    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

    if file_len > max_bytes {
        let _ = file.seek(SeekFrom::End(-(max_bytes as i64)));
    }

    let mut buf = String::new();
    let _ = file.read_to_string(&mut buf);
    buf
}

// ---------------------------------------------------------------------------
// Process detection
// ---------------------------------------------------------------------------

/// Find running `python run.py <config>` processes and return the run names
/// they are training. We read each config file to extract the run name since
/// the process may use a temp file path (e.g. /tmp/tmpXXXXXX.yaml).
fn find_running_training_configs() -> Vec<String> {
    let output = std::process::Command::new("ps")
        .args(["ax", "-o", "args="])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return vec![],
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut names = Vec::new();

    for line in stdout.lines() {
        let trimmed = line.trim();
        // Match: python run.py <config_path>
        if !trimmed.contains("run.py") && !trimmed.contains("modl_worker") {
            continue;
        }

        // Extract the config path (last arg that ends in .yaml/.yml)
        let config_path = trimmed
            .split_whitespace()
            .last()
            .filter(|a| a.ends_with(".yaml") || a.ends_with(".yml"));

        if let Some(path) = config_path {
            // Try to read the config and extract the run name
            if let Ok(contents) = std::fs::read_to_string(path) {
                // Look for "name: <run_name>" in the YAML
                for cfg_line in contents.lines() {
                    let cfg_trimmed = cfg_line.trim();
                    if let Some(name) = cfg_trimmed.strip_prefix("name:") {
                        let run_name = name.trim().trim_matches('"').trim_matches('\'');
                        if !run_name.is_empty() {
                            names.push(run_name.to_string());
                            break;
                        }
                    }
                }
                // Also check training_folder for the run name
                for cfg_line in contents.lines() {
                    let cfg_trimmed = cfg_line.trim();
                    if let Some(folder) = cfg_trimmed.strip_prefix("training_folder:") {
                        let folder = folder.trim().trim_matches('"').trim_matches('\'');
                        if let Some(name) = folder.rsplit('/').next()
                            && !name.is_empty()
                            && !names.contains(&name.to_string())
                        {
                            names.push(name.to_string());
                        }
                    }
                }
            }
            // Fallback: if the path itself contains training_output, extract name
            if names.is_empty() && trimmed.contains("training_output") {
                names.push(trimmed.to_string());
            }
        }
    }

    names
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

struct RunConfig {
    arch: Option<String>,
    base_model: Option<String>,
    trigger_word: Option<String>,
}

fn load_run_config(name: &str) -> Option<RunConfig> {
    let output_dir = training_output_dir();

    // Try the top-level config first (e.g. art-style-v5-config.yaml)
    let config_path = output_dir.join(format!("{name}-config.yaml"));
    let config_path = if config_path.exists() {
        config_path
    } else {
        // Fall back to inner config
        output_dir.join(name).join(name).join("config.yaml")
    };

    let yaml_str = std::fs::read_to_string(&config_path).ok()?;
    let yaml: serde_yaml::Value = serde_yaml::from_str(&yaml_str).ok()?;

    // Navigate: config.process[0].model.arch / config.process[0].model.name_or_path
    let process = yaml
        .get("config")
        .and_then(|c| c.get("process"))
        .and_then(|p| p.as_sequence())
        .and_then(|s| s.first())?;

    let arch = process
        .get("model")
        .and_then(|m| m.get("arch"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let base_model = process
        .get("model")
        .and_then(|m| m.get("name_or_path"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let trigger_word = process
        .get("trigger_word")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Some(RunConfig {
        arch,
        base_model,
        trigger_word,
    })
}

// ---------------------------------------------------------------------------
// Sample/checkpoint scanning
// ---------------------------------------------------------------------------

fn find_latest_samples(name: &str, run_inner: &Path) -> Vec<String> {
    let samples_dir = run_inner.join("samples");
    if !samples_dir.exists() {
        return vec![];
    }

    let mut entries: Vec<_> = std::fs::read_dir(&samples_dir)
        .ok()
        .into_iter()
        .flat_map(|rd| rd.flatten())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "jpg" || ext == "png")
        })
        .collect();

    // Sort by modification time descending
    entries.sort_by(|a, b| {
        let mt_a = a
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mt_b = b
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        mt_b.cmp(&mt_a)
    });

    // Return the most recent group (same step)
    let mut latest_step: Option<u64> = None;
    let mut result = Vec::new();

    for entry in &entries {
        let fname = entry.file_name().to_string_lossy().to_string();
        if let Some(step) = parse_step_from_filename(&fname) {
            match latest_step {
                None => {
                    latest_step = Some(step);
                    result.push(format!("training_output/{name}/{name}/samples/{fname}"));
                }
                Some(s) if s == step => {
                    result.push(format!("training_output/{name}/{name}/samples/{fname}"));
                }
                _ => break,
            }
        }
    }

    result
}

fn parse_step_from_filename(filename: &str) -> Option<u64> {
    let parts: Vec<&str> = filename.split("__").collect();
    if parts.len() == 2 {
        let rest = parts[1];
        let step_str = rest.split('_').next()?;
        step_str.parse::<u64>().ok()
    } else {
        None
    }
}

fn find_latest_checkpoint(run_inner: &Path) -> Option<String> {
    if !run_inner.exists() {
        return None;
    }

    let mut checkpoints: Vec<_> = std::fs::read_dir(run_inner)
        .ok()?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .collect();

    checkpoints.sort_by(|a, b| {
        let mt_a = a
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mt_b = b
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        mt_b.cmp(&mt_a)
    });

    checkpoints
        .first()
        .map(|e| e.path().to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn training_output_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("training_output")
}

fn is_recently_modified(path: &Path, max_age_secs: u64) -> bool {
    path.metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age.as_secs() < max_age_secs)
}
