use assert_cmd::Command;
use predicates::str::contains;

#[allow(deprecated)]
fn modl_cmd() -> Command {
    Command::cargo_bin("modl").unwrap()
}

// ---------------------------------------------------------------------------
// Basic CLI smoke tests
// ---------------------------------------------------------------------------

#[test]
fn help_shows_description() {
    modl_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("AI image generation toolkit"));
}

#[test]
fn version_flag() {
    modl_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(contains("modl"));
}

#[test]
fn invalid_subcommand_fails() {
    modl_cmd().arg("yolo").assert().failure();
}

// ---------------------------------------------------------------------------
// ValueEnum validation — clap rejects invalid values before our code runs
// ---------------------------------------------------------------------------

#[test]
fn model_ls_rejects_invalid_type() {
    modl_cmd()
        .args(["ls", "--type", "banana"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

#[test]
fn model_ls_accepts_valid_type() {
    let result = modl_cmd().args(["ls", "--type", "lora"]).assert();

    // We just verify it doesn't fail with a clap error
    let output = result.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("possible values"),
        "should accept 'lora' as valid type"
    );
}

#[test]
fn model_search_rejects_invalid_type() {
    modl_cmd()
        .args(["search", "flux", "--type", "nope"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

#[test]
fn auth_rejects_invalid_provider() {
    modl_cmd()
        .args(["auth", "add", "dropbox"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

#[test]
fn train_rejects_invalid_provider() {
    modl_cmd()
        .args(["train", "--provider", "aws"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

#[test]
fn train_rejects_invalid_preset() {
    modl_cmd()
        .args(["train", "--preset", "extreme"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

#[test]
fn generate_rejects_invalid_provider() {
    modl_cmd()
        .args(["generate", "a cat", "--provider", "lambda"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

// ---------------------------------------------------------------------------
// HuggingFace cache location (see src/core/hf_cache.rs)
// ---------------------------------------------------------------------------

/// An isolated HOME so these never read or write the developer's real config.
fn isolated_home() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

#[test]
fn config_reports_hf_cache_default() {
    let home = isolated_home();
    modl_cmd()
        .env("HOME", home.path())
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .args(["config", "storage.hf_cache"])
        .assert()
        .success()
        .stdout(contains(".cache/huggingface"));
}

#[test]
fn config_reports_hf_cache_from_env() {
    let home = isolated_home();
    modl_cmd()
        .env("HOME", home.path())
        .env("HF_HOME", "/mnt/elsewhere/hf")
        .args(["config", "storage.hf_cache"])
        .assert()
        .success()
        .stdout(contains("/mnt/elsewhere/hf"));
}

#[test]
fn config_set_hf_cache_persists_and_is_read_back() {
    let home = isolated_home();
    modl_cmd()
        .env("HOME", home.path())
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .args(["config", "storage.hf_cache", "/srv/disk2/hf"])
        .assert()
        .success()
        .stdout(contains("/srv/disk2/hf"));

    // A fresh process must resolve the same path from config.yaml alone.
    modl_cmd()
        .env("HOME", home.path())
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .args(["config", "storage.hf_cache"])
        .assert()
        .success()
        .stdout(contains("/srv/disk2/hf"));
}

#[test]
fn env_hf_home_beats_configured_hf_cache() {
    let home = isolated_home();
    modl_cmd()
        .env("HOME", home.path())
        .env_remove("HF_HOME")
        .args(["config", "storage.hf_cache", "/srv/disk2/hf"])
        .assert()
        .success();

    modl_cmd()
        .env("HOME", home.path())
        .env("HF_HOME", "/mnt/preexisting/hf")
        .args(["config", "storage.hf_cache"])
        .assert()
        .success()
        .stdout(contains("/mnt/preexisting/hf"));
}

// ---------------------------------------------------------------------------
// Aliases
// ---------------------------------------------------------------------------

#[test]
fn model_ls_accepts_textencoder_alias() {
    let result = modl_cmd()
        .args(["model", "ls", "--type", "textencoder"])
        .assert();

    let output = result.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("possible values"),
        "should accept 'textencoder' as alias for text_encoder"
    );
}
