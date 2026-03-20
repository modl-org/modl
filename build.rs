use std::process::Command;

fn main() {
    let web_dir = "src/ui/web";
    let dist_dir = "src/ui/dist";

    // Re-run this script if any frontend source file changes
    println!("cargo:rerun-if-changed={web_dir}/src");
    println!("cargo:rerun-if-changed={web_dir}/package.json");
    println!("cargo:rerun-if-changed={web_dir}/vite.config.ts");
    println!("cargo:rerun-if-changed={web_dir}/index.html");
    println!("cargo:rerun-if-changed=.env");

    // Skip the build if we're on docs.rs
    if std::env::var("DOCS_RS").is_ok() {
        ensure_stub_dist(dist_dir);
        return;
    }

    // When MODL_SKIP_UI_BUILD=1, just create placeholder files so that
    // `include_str!` compiles. Useful for `cargo check` / `cargo test` in CI
    // where bundling the UI isn't required.
    if std::env::var("MODL_SKIP_UI_BUILD").is_ok() {
        ensure_stub_dist(dist_dir);
        return;
    }

    // Ensure npm is available
    let npm = which_npm();

    // Install dependencies (no-op if node_modules is up to date)
    let install = Command::new(&npm)
        .args(["install", "--frozen-lockfile"])
        .current_dir(web_dir)
        .status()
        .expect("build.rs: failed to run `npm install`");
    assert!(install.success(), "build.rs: `npm install` failed");

    // Build
    let build = Command::new(&npm)
        .args(["run", "build"])
        .current_dir(web_dir)
        .status()
        .expect("build.rs: failed to run `npm run build`");
    assert!(build.success(), "build.rs: `npm run build` failed");
}

/// Create the minimum placeholder files so that `include_str!` macros compile
/// without a real frontend build.
fn ensure_stub_dist(dist_dir: &str) {
    let assets = format!("{dist_dir}/assets");
    std::fs::create_dir_all(&assets).expect("build.rs: could not create dist/assets");
    for (path, content) in [
        (format!("{assets}/app.js"), "/* stub */"),
        (format!("{assets}/index.css"), "/* stub */"),
        (
            format!("{dist_dir}/index.html"),
            "<!doctype html><html><body>stub</body></html>",
        ),
    ] {
        if !std::path::Path::new(&path).exists() {
            std::fs::write(&path, content).expect("build.rs: could not write stub file");
        }
    }
}

fn which_npm() -> String {
    // On Windows, Command::new("npm") won't find npm.cmd automatically,
    // so we check npm.cmd first on Windows.
    let candidates: &[&str] = if cfg!(windows) {
        &["npm.cmd", "npm", "npm.exe"]
    } else {
        &["npm", "/usr/local/bin/npm", "/usr/bin/npm"]
    };
    for candidate in candidates {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return candidate.to_string();
        }
    }
    panic!("build.rs: `npm` not found — install Node.js to build the UI");
}
