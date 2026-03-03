use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tar::Archive;
use zstd::stream::read::Decoder as ZstdDecoder;

use console::style;

use crate::core::download;
use crate::core::store::Store;

const DEFAULT_PROFILE: &str = "trainer-cu124";
const DEFAULT_CHANNEL: &str = "stable";
const PYTHON_VERSION: &str = "3.11.11";
const TRAINER_CU126_INDEX_URL: &str = "https://download.pytorch.org/whl/cu126";
const TRAINER_TORCH_VERSION: &str = "2.7.0";
const TRAINER_TORCHVISION_VERSION: &str = "0.22.0";
const TRAINER_TORCHAUDIO_VERSION: &str = "2.7.0";
const AITOOLKIT_REPO_URL: &str = "https://github.com/ostris/ai-toolkit.git";
const AITOOLKIT_CLONE_DIR: &str = "ai-toolkit";
const DEFAULT_PYTHON_ARTIFACT_URL: &str = "https://github.com/modl/modl-runtime-manifests/releases/download/v2026.02.1/cpython-3.11.11-linux-x86_64.tar.gz";
const PYTHON_ARTIFACT_URL_ENV: &str = "MODL_PYTHON_ARTIFACT_URL";

#[derive(Debug, Clone)]
pub struct RuntimeInstallResult {
    pub profile: String,
    pub channel: String,
    pub runtime_root: PathBuf,
    pub lock_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RuntimeStatus {
    pub installed: bool,
    pub profile: Option<String>,
    pub channel: Option<String>,
    pub runtime_root: PathBuf,
    pub lock_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RuntimeDoctorReport {
    pub runtime_root_exists: bool,
    pub lock_exists: bool,
    pub profile_known: bool,
    pub python_exists: bool,
}

#[derive(Debug, Clone)]
pub struct RuntimeBootstrapResult {
    pub profile: String,
    pub env_dir: PathBuf,
    pub python_path: PathBuf,
    pub requirements_path: PathBuf,
    pub created_env: bool,
}

#[derive(Debug, Clone)]
pub struct TrainingSetupResult {
    pub profile: String,
    pub python_path: PathBuf,
    pub train_command_template: Option<String>,
    pub ready: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RuntimeLock {
    schema_version: String,
    installed_at: String,
    profile: String,
    channel: String,
    python_version: String,
    #[serde(default)]
    train_command_template: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RuntimeProfileManifest {
    schema_version: String,
    id: String,
    version: String,
    channel: String,
    python: PythonArtifact,
}

#[derive(Debug, Serialize, Deserialize)]
struct PythonArtifact {
    version: String,
    artifact_uri: String,
    sha256: String,
}

pub fn install(profile: Option<&str>, channel: Option<&str>) -> Result<RuntimeInstallResult> {
    let profile = profile.unwrap_or(DEFAULT_PROFILE);
    validate_profile(profile)?;

    let channel = channel.unwrap_or(DEFAULT_CHANNEL);
    validate_channel(channel)?;

    let runtime_root = runtime_root()?;
    ensure_layout(runtime_root.as_path())?;

    write_manifest_index_if_missing(runtime_root.as_path(), channel)?;
    ensure_profile_seed_files(runtime_root.as_path())?;
    let lock_path = lock_path(runtime_root.as_path());
    write_lock(lock_path.as_path(), profile, channel)?;

    Ok(RuntimeInstallResult {
        profile: profile.to_string(),
        channel: channel.to_string(),
        runtime_root,
        lock_path,
    })
}

pub fn status() -> Result<RuntimeStatus> {
    let runtime_root = runtime_root()?;
    let lock_path = lock_path(runtime_root.as_path());

    if !lock_path.exists() {
        return Ok(RuntimeStatus {
            installed: false,
            profile: None,
            channel: None,
            runtime_root,
            lock_path,
        });
    }

    let lock = read_lock(lock_path.as_path())?;

    Ok(RuntimeStatus {
        installed: true,
        profile: Some(lock.profile),
        channel: Some(lock.channel),
        runtime_root,
        lock_path,
    })
}

pub fn train_command_template() -> Result<Option<String>> {
    let runtime_root = runtime_root()?;
    let lock_path = lock_path(runtime_root.as_path());
    if !lock_path.exists() {
        return Ok(None);
    }

    let lock = read_lock(lock_path.as_path())?;
    Ok(lock.train_command_template)
}

pub async fn setup_training(reinstall: bool) -> Result<TrainingSetupResult> {
    let install = install(Some(DEFAULT_PROFILE), None)?;
    let env_dir = install.runtime_root.join("envs").join(DEFAULT_PROFILE);

    if reinstall {
        let marker = bootstrap_marker_path(env_dir.as_path());
        if marker.exists() {
            fs::remove_file(&marker)
                .with_context(|| format!("Failed to remove {}", marker.display()))?;
        }
    }

    let boot = bootstrap(Some(DEFAULT_PROFILE), None).await?;
    let template = train_command_template()?;

    Ok(TrainingSetupResult {
        profile: boot.profile,
        python_path: boot.python_path,
        ready: template.is_some(),
        train_command_template: template,
    })
}

pub fn doctor() -> Result<RuntimeDoctorReport> {
    let runtime_root = runtime_root()?;
    let lock_path = lock_path(runtime_root.as_path());

    let runtime_root_exists = runtime_root.exists();
    let lock_exists = lock_path.exists();

    let mut profile_known = false;
    let mut python_exists = false;

    if lock_exists {
        let lock = read_lock(lock_path.as_path())?;
        profile_known = validate_profile(&lock.profile).is_ok();

        let env_python = runtime_root
            .join("envs")
            .join(lock.profile)
            .join("bin")
            .join("python");
        let base_python = runtime_root
            .join("python")
            .join(PYTHON_VERSION)
            .join("bin")
            .join("python");
        python_exists = env_python.exists() || base_python.exists();
    }

    Ok(RuntimeDoctorReport {
        runtime_root_exists,
        lock_exists,
        profile_known,
        python_exists,
    })
}

pub fn upgrade(channel: Option<&str>) -> Result<RuntimeInstallResult> {
    let current = status()?;
    let profile = current.profile.as_deref().unwrap_or(DEFAULT_PROFILE);
    install(Some(profile), channel)
}

pub async fn bootstrap(
    profile: Option<&str>,
    channel: Option<&str>,
) -> Result<RuntimeBootstrapResult> {
    let install_result = install(profile, channel)?;
    let lock = read_lock(install_result.lock_path.as_path())?;

    let env_dir = install_result.runtime_root.join("envs").join(&lock.profile);
    let python_path = env_dir.join("bin").join("python");
    let requirements_path =
        profile_requirements_path(install_result.runtime_root.as_path(), &lock.profile);

    let mut created_env = false;
    if !python_path.exists() {
        install_managed_python(install_result.runtime_root.as_path(), &lock.profile).await?;
        let bootstrap_python = managed_python_path(install_result.runtime_root.as_path());
        run_command(
            Command::new(&bootstrap_python)
                .arg("-m")
                .arg("venv")
                .arg(&env_dir),
            "Failed to create runtime virtual environment",
        )?;
        created_env = true;
    }

    if created_env {
        run_command(
            Command::new(&python_path)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("--upgrade")
                .arg("pip")
                .arg("setuptools")
                .arg("wheel"),
            "Failed to upgrade pip tooling in runtime environment",
        )?;
    }

    ensure_profile_dependencies(
        python_path.as_path(),
        env_dir.as_path(),
        &lock.profile,
        requirements_path.as_path(),
        created_env,
        install_result.runtime_root.as_path(),
    )?;

    let aitoolkit_dir = aitoolkit_clone_dir(install_result.runtime_root.as_path());
    let aitoolkit_opt = if aitoolkit_dir.join("toolkit").exists() {
        Some(aitoolkit_dir.as_path())
    } else {
        None
    };
    let detected = detect_train_command_template(python_path.as_path(), aitoolkit_opt);
    if detected.is_some() {
        update_train_command_template(install_result.runtime_root.as_path(), detected)?;
    }

    Ok(RuntimeBootstrapResult {
        profile: lock.profile,
        env_dir,
        python_path,
        requirements_path,
        created_env,
    })
}

pub fn reset(purge_cache: bool) -> Result<()> {
    let runtime_root = runtime_root()?;

    if !runtime_root.exists() {
        return Ok(());
    }

    if purge_cache {
        fs::remove_dir_all(&runtime_root)
            .with_context(|| format!("Failed to remove {}", runtime_root.display()))?;
        return Ok(());
    }

    let envs = runtime_root.join("envs");
    let locks = runtime_root.join("locks");

    if envs.exists() {
        fs::remove_dir_all(&envs)
            .with_context(|| format!("Failed to remove {}", envs.display()))?;
    }

    if locks.exists() {
        fs::remove_dir_all(&locks)
            .with_context(|| format!("Failed to remove {}", locks.display()))?;
    }

    fs::create_dir_all(runtime_root.join("envs"))?;
    fs::create_dir_all(runtime_root.join("locks"))?;

    Ok(())
}

fn runtime_root() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".modl").join("runtime"))
}

fn ensure_layout(root: &Path) -> Result<()> {
    let python_bin = root.join("python").join(PYTHON_VERSION).join("bin");
    let downloads = root.join("python").join("downloads");
    let envs = root.join("envs");
    let wheelhouse = root.join("wheelhouse");
    let profiles = root.join("manifests").join("profiles");
    let locks = root.join("locks");

    fs::create_dir_all(&python_bin)
        .with_context(|| format!("Failed to create {}", python_bin.display()))?;
    fs::create_dir_all(&downloads)
        .with_context(|| format!("Failed to create {}", downloads.display()))?;
    fs::create_dir_all(&envs).with_context(|| format!("Failed to create {}", envs.display()))?;
    fs::create_dir_all(&wheelhouse)
        .with_context(|| format!("Failed to create {}", wheelhouse.display()))?;
    fs::create_dir_all(&profiles)
        .with_context(|| format!("Failed to create {}", profiles.display()))?;
    fs::create_dir_all(&locks).with_context(|| format!("Failed to create {}", locks.display()))?;
    Ok(())
}

fn write_manifest_index_if_missing(root: &Path, channel: &str) -> Result<()> {
    let index_path = root.join("manifests").join("index.json");
    if index_path.exists() {
        return Ok(());
    }

    let manifest = serde_json::json!({
        "schema_version": "v1",
        "channel": channel,
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "profiles": [
            {
                "id": DEFAULT_PROFILE,
                "version": "2026.02.1",
                "manifest_uri": "https://github.com/modl/modl-runtime-manifests/releases/download/v2026.02.1/trainer-cu124.json",
                "sha256": ""
            }
        ]
    });

    let data = serde_json::to_string_pretty(&manifest)?;
    fs::write(&index_path, data).with_context(|| {
        format!(
            "Failed to write runtime manifest index at {}",
            index_path.display()
        )
    })?;

    Ok(())
}

fn ensure_profile_seed_files(root: &Path) -> Result<()> {
    write_profile_manifest_if_missing(root, DEFAULT_PROFILE)?;
    write_profile_manifest_if_missing(root, "inference-cu124")?;

    write_profile_requirements_if_missing(root, DEFAULT_PROFILE)?;
    write_profile_requirements_if_missing(root, "inference-cu124")?;

    Ok(())
}

fn write_profile_manifest_if_missing(root: &Path, profile: &str) -> Result<()> {
    let manifest_path = profile_manifest_path(root, profile);
    if manifest_path.exists() {
        return Ok(());
    }

    let manifest = RuntimeProfileManifest {
        schema_version: "v1".to_string(),
        id: profile.to_string(),
        version: "2026.02.1".to_string(),
        channel: DEFAULT_CHANNEL.to_string(),
        python: PythonArtifact {
            version: PYTHON_VERSION.to_string(),
            artifact_uri: python_artifact_url(),
            sha256: String::new(),
        },
    };

    let json = serde_json::to_string_pretty(&manifest)?;
    fs::write(&manifest_path, json)
        .with_context(|| format!("Failed to write {}", manifest_path.display()))?;

    Ok(())
}

fn write_profile_requirements_if_missing(root: &Path, profile: &str) -> Result<()> {
    let requirements_path = profile_requirements_path(root, profile);
    if requirements_path.exists() {
        return Ok(());
    }

    let content = match profile {
        "trainer-cu124" => {
            "# Additional trainer-cu124 requirements (base torch + ai-toolkit are installed by bootstrap logic)\naccelerate>=0.33\nsafetensors>=0.5\ntransformers>=4.51\ndiffusers>=0.31\npillow>=10.0\n"
        }
        "inference-cu124" => {
            "# Runtime profile requirements for inference-cu124\n# Add/adjust heavy dependencies as manifests stabilize.\n\n"
        }
        _ => "# Runtime profile requirements\n\n",
    };

    fs::write(&requirements_path, content)
        .with_context(|| format!("Failed to write {}", requirements_path.display()))?;

    Ok(())
}

fn profile_requirements_path(root: &Path, profile: &str) -> PathBuf {
    root.join("manifests")
        .join("profiles")
        .join(format!("{}.requirements.txt", profile))
}

fn profile_manifest_path(root: &Path, profile: &str) -> PathBuf {
    root.join("manifests")
        .join("profiles")
        .join(format!("{}.json", profile))
}

fn lock_path(root: &Path) -> PathBuf {
    root.join("locks").join("runtime.lock.json")
}

fn write_lock(path: &Path, profile: &str, channel: &str) -> Result<()> {
    let lock = RuntimeLock {
        schema_version: "v1".to_string(),
        installed_at: chrono::Utc::now().to_rfc3339(),
        profile: profile.to_string(),
        channel: channel.to_string(),
        python_version: PYTHON_VERSION.to_string(),
        train_command_template: None,
    };

    let json = serde_json::to_string_pretty(&lock)?;
    fs::write(path, json).with_context(|| {
        format!(
            "Failed to write runtime lock file at {}",
            path.to_string_lossy()
        )
    })?;

    Ok(())
}

fn read_lock(path: &Path) -> Result<RuntimeLock> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read lock file at {}", path.display()))?;
    let lock: RuntimeLock =
        serde_json::from_str(&contents).context("Failed to parse runtime lock file")?;
    Ok(lock)
}

fn validate_profile(profile: &str) -> Result<()> {
    match profile {
        "trainer-cu124" => Ok(()),
        "inference-cu124" => Ok(()),
        _ => bail!(
            "Unsupported runtime profile '{}'. Supported: trainer-cu124, inference-cu124",
            profile
        ),
    }
}

fn validate_channel(channel: &str) -> Result<()> {
    match channel {
        "stable" | "beta" => Ok(()),
        _ => bail!(
            "Unsupported channel '{}'. Supported channels: stable, beta",
            channel
        ),
    }
}

fn managed_python_path(root: &Path) -> PathBuf {
    root.join("python")
        .join(PYTHON_VERSION)
        .join("bin")
        .join("python")
}

fn run_command(cmd: &mut Command, context_msg: &str) -> Result<()> {
    let output = cmd.output().with_context(|| context_msg.to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "{}\nstdout: {}\nstderr: {}",
            context_msg,
            stdout.trim(),
            stderr.trim()
        );
    }
    Ok(())
}

fn run_command_status(cmd: &mut Command) -> bool {
    match cmd.stdout(Stdio::null()).stderr(Stdio::null()).status() {
        Ok(status) => status.success(),
        Err(_) => false,
    }
}

fn detect_train_command_template(
    python_path: &Path,
    aitoolkit_dir: Option<&Path>,
) -> Option<String> {
    let mut cmd = Command::new(python_path);
    cmd.arg("-m").arg("toolkit.job").arg("--help");
    if let Some(dir) = aitoolkit_dir {
        cmd.env("PYTHONPATH", dir);
    }
    if run_command_status(&mut cmd) {
        return Some("{python} -m toolkit.job --config {config}".to_string());
    }

    let env_bin = python_path
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("bin"));
    if let Some(bin_dir) = env_bin {
        let aitk = bin_dir.join("aitk");
        if aitk.exists() && run_command_status(Command::new(&aitk).arg("--help")) {
            return Some(format!("{} train --config {{config}}", aitk.display()));
        }
    }

    None
}

fn update_train_command_template(root: &Path, template: Option<String>) -> Result<()> {
    let lock_path = lock_path(root);
    if !lock_path.exists() {
        return Ok(());
    }

    let mut lock = read_lock(lock_path.as_path())?;
    lock.train_command_template = template;

    let json = serde_json::to_string_pretty(&lock)?;
    fs::write(&lock_path, json)
        .with_context(|| format!("Failed to update {}", lock_path.display()))?;

    Ok(())
}

fn ensure_profile_dependencies(
    python_path: &Path,
    env_dir: &Path,
    profile: &str,
    requirements_path: &Path,
    created_env: bool,
    runtime_root: &Path,
) -> Result<()> {
    let marker_path = bootstrap_marker_path(env_dir);

    if marker_path.exists() && !created_env {
        return Ok(());
    }

    if profile == "trainer-cu124" {
        println!(
            "  {} Installing PyTorch {} …",
            style("→").dim(),
            style(TRAINER_TORCH_VERSION).dim()
        );
        run_command(
            Command::new(python_path)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("--no-cache-dir")
                .arg(format!("torch=={}", TRAINER_TORCH_VERSION))
                .arg(format!("torchvision=={}", TRAINER_TORCHVISION_VERSION))
                .arg(format!("torchaudio=={}", TRAINER_TORCHAUDIO_VERSION))
                .arg("--index-url")
                .arg(TRAINER_CU126_INDEX_URL),
            "Failed to install pinned torch dependencies for trainer-cu124",
        )?;

        println!("  {} Cloning ai-toolkit …", style("→").dim());
        clone_or_update_aitoolkit(runtime_root)?;

        let aitoolkit_dir = aitoolkit_clone_dir(runtime_root);
        let aitoolkit_reqs = aitoolkit_dir.join("requirements.txt");
        if aitoolkit_reqs.exists() {
            println!(
                "  {} Installing ai-toolkit dependencies …",
                style("→").dim()
            );
            run_command(
                Command::new(python_path)
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("--no-cache-dir")
                    .arg("-r")
                    .arg(&aitoolkit_reqs),
                "Failed to install ai-toolkit requirements",
            )?;
        }
    }

    if requirements_path.exists() {
        println!("  {} Installing profile requirements …", style("→").dim());
        run_command(
            Command::new(python_path)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("-r")
                .arg(requirements_path),
            "Failed to install runtime profile requirements",
        )?;
    }

    let marker_contents = format!(
        "profile={profile}\ntorch={TRAINER_TORCH_VERSION}\ntorchvision={TRAINER_TORCHVISION_VERSION}\ntorchaudio={TRAINER_TORCHAUDIO_VERSION}\n"
    );
    fs::write(&marker_path, marker_contents)
        .with_context(|| format!("Failed to write {}", marker_path.display()))?;

    Ok(())
}

fn aitoolkit_clone_dir(root: &Path) -> PathBuf {
    root.join(AITOOLKIT_CLONE_DIR)
}

fn clone_or_update_aitoolkit(root: &Path) -> Result<()> {
    let dir = aitoolkit_clone_dir(root);
    if dir.join("toolkit").exists() {
        // Already cloned – fast-forward update
        run_command(
            Command::new("git")
                .arg("pull")
                .arg("--ff-only")
                .current_dir(&dir),
            "Failed to update ai-toolkit (git pull)",
        )?;
    } else {
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .with_context(|| format!("Failed to remove stale {}", dir.display()))?;
        }
        run_command(
            Command::new("git")
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg(AITOOLKIT_REPO_URL)
                .arg(&dir),
            "Failed to clone ai-toolkit",
        )?;
    }
    Ok(())
}

pub fn aitoolkit_path() -> Result<Option<PathBuf>> {
    let root = runtime_root()?;
    let dir = aitoolkit_clone_dir(&root);
    if dir.join("toolkit").exists() {
        Ok(Some(dir))
    } else {
        Ok(None)
    }
}

fn bootstrap_marker_path(env_dir: &Path) -> PathBuf {
    env_dir.join(".modl-bootstrap-complete")
}

async fn install_managed_python(root: &Path, profile: &str) -> Result<()> {
    ensure_profile_seed_files(root)?;

    let managed_python = managed_python_path(root);
    if managed_python.exists() {
        return Ok(());
    }

    let profile_manifest = read_profile_manifest(root, profile)?;
    let url = profile_manifest.python.artifact_uri;
    let expected_hash = profile_manifest.python.sha256;

    let file_name = url
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("cpython.tar.gz");
    let archive_path = root.join("python").join("downloads").join(file_name);

    if !archive_path.exists() {
        download::download_file(&url, &archive_path, None, None).await.with_context(|| {
            format!(
                "Failed to download managed Python artifact from {}. Set {} to a valid CPython artifact URL if needed.",
                url,
                PYTHON_ARTIFACT_URL_ENV
            )
        })?;
    }

    if !expected_hash.trim().is_empty() {
        let hash_ok = Store::verify_hash(archive_path.as_path(), &expected_hash)
            .context("Failed to verify managed Python artifact hash")?;
        if !hash_ok {
            bail!(
                "Managed Python artifact hash mismatch for {}",
                archive_path.display()
            );
        }
    }

    extract_python_archive(archive_path.as_path(), &root.join("python"))?;

    if !managed_python.exists() {
        bail!(
            "Managed Python extraction completed, but {} is missing. Ensure artifact layout contains python/{}/bin/python",
            managed_python.display(),
            PYTHON_VERSION
        );
    }

    Ok(())
}

fn python_artifact_url() -> String {
    std::env::var(PYTHON_ARTIFACT_URL_ENV)
        .unwrap_or_else(|_| DEFAULT_PYTHON_ARTIFACT_URL.to_string())
}

fn read_profile_manifest(root: &Path, profile: &str) -> Result<RuntimeProfileManifest> {
    let manifest_path = profile_manifest_path(root, profile);
    let contents = fs::read_to_string(&manifest_path)
        .with_context(|| format!("Failed to read {}", manifest_path.display()))?;
    let manifest: RuntimeProfileManifest = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;
    Ok(manifest)
}

fn extract_python_archive(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = File::open(archive_path)
        .with_context(|| format!("Failed to open {}", archive_path.display()))?;

    let file_name = archive_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if file_name.ends_with(".tar.gz") || file_name.ends_with(".tgz") {
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        archive
            .unpack(destination)
            .with_context(|| format!("Failed to unpack {}", archive_path.display()))?;
        return Ok(());
    }

    if file_name.ends_with(".tar.zst") || file_name.ends_with(".tzst") {
        let decoder = ZstdDecoder::new(file).with_context(|| {
            format!(
                "Failed to initialize zstd decoder for {}",
                archive_path.display()
            )
        })?;
        let mut archive = Archive::new(decoder);
        archive
            .unpack(destination)
            .with_context(|| format!("Failed to unpack {}", archive_path.display()))?;
        return Ok(());
    }

    bail!(
        "Unsupported Python artifact format for {}. Expected .tar.gz/.tgz/.tar.zst/.tzst archive.",
        archive_path.display()
    )
}
