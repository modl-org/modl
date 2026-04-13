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
// Video generation flags
// ---------------------------------------------------------------------------

#[test]
fn generate_accepts_frames_flag() {
    // --frames should be accepted without clap error (will fail on missing model, not flag parse)
    let result = modl_cmd()
        .args([
            "generate",
            "a cat",
            "--base",
            "ltx-video-2-3",
            "--frames",
            "97",
        ])
        .assert();

    let output = result.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("unexpected argument"),
        "--frames should be accepted by generate"
    );
}

#[test]
fn generate_accepts_fps_flag() {
    let result = modl_cmd()
        .args([
            "generate",
            "a cat",
            "--base",
            "ltx-video-2-3",
            "--fps",
            "24",
        ])
        .assert();

    let output = result.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("unexpected argument"),
        "--fps should be accepted by generate"
    );
}

#[test]
fn generate_help_shows_video_flags() {
    modl_cmd()
        .args(["generate", "--help"])
        .assert()
        .success()
        .stdout(contains("--frames"))
        .stdout(contains("--fps"));
}

#[test]
fn generate_frames_flag_requires_value() {
    // --frames without value should fail
    modl_cmd()
        .args(["generate", "a cat", "--frames"])
        .assert()
        .failure()
        .stderr(contains("a value is required"));
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
