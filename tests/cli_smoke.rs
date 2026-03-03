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
        .stdout(contains("Model manager"));
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
        .args(["model", "ls", "--type", "banana"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

#[test]
fn model_ls_accepts_valid_type() {
    let result = modl_cmd().args(["model", "ls", "--type", "lora"]).assert();

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
        .args(["model", "search", "flux", "--type", "nope"])
        .assert()
        .failure()
        .stderr(contains("possible values"));
}

#[test]
fn auth_rejects_invalid_provider() {
    modl_cmd()
        .args(["auth", "dropbox"])
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
