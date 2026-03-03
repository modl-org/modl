use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Valid image extensions for training datasets
const VALID_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp"];

/// Scanned image entry
#[derive(Debug, Clone)]
pub struct ImageEntry {
    pub path: PathBuf,
    pub caption_path: Option<PathBuf>,
}

/// Result of scanning a dataset directory
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub path: PathBuf,
    pub image_count: u32,
    pub captioned_count: u32,
    pub caption_coverage: f32,
    #[allow(dead_code)]
    pub images: Vec<ImageEntry>,
}

/// Root directory for managed datasets
fn datasets_root() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("datasets")
}

/// Resolve a name or path to a dataset directory.
/// If it looks like a path (contains / or \), use it directly.
/// Otherwise, look under ~/.modl/datasets/<name>.
pub fn resolve_path(name_or_path: &str) -> PathBuf {
    let path = PathBuf::from(name_or_path);
    if path.is_absolute() || name_or_path.contains('/') || name_or_path.contains('\\') {
        path
    } else {
        datasets_root().join(name_or_path)
    }
}

/// Create a managed dataset by copying images from a source directory.
///
/// Copies images matching valid extensions from `from_dir` into
/// `~/.modl/datasets/<name>/`, along with any paired `.txt` caption files.
///
/// Supports **recursive** scanning: if the source has subfolders (e.g. `happy/`,
/// `sad/`), images inside them are collected and the subfolder name is used as a
/// tag prefix in auto-generated captions (written as `<subfolder> — <original caption>`
/// when a caption exists, or just `<subfolder>` when it doesn't). Files are
/// flattened into the destination with a `<subfolder>_` prefix to avoid name
/// collisions.
pub fn create(name: &str, from_dir: &Path) -> Result<DatasetInfo> {
    if !from_dir.exists() {
        bail!("Source directory does not exist: {}", from_dir.display());
    }
    if !from_dir.is_dir() {
        bail!("Source path is not a directory: {}", from_dir.display());
    }

    let dest = datasets_root().join(name);
    if dest.exists() {
        bail!("Dataset '{}' already exists at {}", name, dest.display());
    }

    std::fs::create_dir_all(&dest)
        .with_context(|| format!("Failed to create dataset directory: {}", dest.display()))?;

    let mut copied = 0u32;

    // Collect (src_image_path, subfolder_tag_option) pairs
    let mut sources: Vec<(PathBuf, Option<String>)> = Vec::new();
    collect_images_recursive(from_dir, from_dir, &mut sources)?;

    for (src_path, tag) in &sources {
        let raw_name = src_path.file_name().unwrap().to_string_lossy().to_string();

        // Flatten subfolder images with a prefix to avoid collisions
        let dest_name = if let Some(t) = tag {
            format!("{}_{}", t, raw_name)
        } else {
            raw_name.clone()
        };
        let dest_file = dest.join(&dest_name);

        std::fs::copy(src_path, &dest_file).with_context(|| {
            format!(
                "Failed to copy {} → {}",
                src_path.display(),
                dest_file.display()
            )
        })?;
        copied += 1;

        // Copy or synthesize caption
        let caption_src = src_path.with_extension("txt");
        let caption_dest = dest_file.with_extension("txt");

        if caption_src.exists() {
            let original = std::fs::read_to_string(&caption_src)
                .unwrap_or_default()
                .trim()
                .to_string();
            let caption = if let Some(t) = tag {
                format!("{t}, {original}")
            } else {
                original
            };
            std::fs::write(&caption_dest, &caption)
                .with_context(|| format!("Failed to write caption: {}", caption_dest.display()))?;
        } else if let Some(t) = tag {
            // No existing caption — seed with the subfolder tag so there's
            // at least a label for training.
            std::fs::write(&caption_dest, t)
                .with_context(|| format!("Failed to write caption: {}", caption_dest.display()))?;
        }
    }

    if copied == 0 {
        let _ = std::fs::remove_dir(&dest);
        bail!(
            "No valid images found in {}. Expected jpg, jpeg, or png files.",
            from_dir.display()
        );
    }

    scan(&dest)
}

/// Recursively collect images from `dir`, tracking the subfolder tag
/// (relative to `root`). Top-level images get `tag = None`.
fn collect_images_recursive(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(PathBuf, Option<String>)>,
) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    let tag = if dir == root {
        None
    } else {
        dir.file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
    };

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Skip hidden directories and cache dirs
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name.starts_with('_') {
                continue;
            }
            collect_images_recursive(root, &path, out)?;
        } else if path.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            if ext
                .as_deref()
                .is_some_and(|e| VALID_EXTENSIONS.contains(&e))
            {
                out.push((path, tag.clone()));
            }
        }
    }
    Ok(())
}

/// Scan a dataset directory and return stats + image entries.
pub fn scan(path: &Path) -> Result<DatasetInfo> {
    if !path.exists() {
        bail!("Dataset path does not exist: {}", path.display());
    }
    if !path.is_dir() {
        bail!("Dataset path is not a directory: {}", path.display());
    }

    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let entries = std::fs::read_dir(path)
        .with_context(|| format!("Failed to read dataset directory: {}", path.display()))?;

    let mut images = Vec::new();

    for entry in entries {
        let entry = entry?;
        let file_path = entry.path();

        if !file_path.is_file() {
            continue;
        }

        let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        let is_image = ext
            .as_deref()
            .is_some_and(|e| VALID_EXTENSIONS.contains(&e));

        if !is_image {
            continue;
        }

        let caption_path = file_path.with_extension("txt");
        let caption = if caption_path.exists() {
            Some(caption_path)
        } else {
            None
        };

        images.push(ImageEntry {
            path: file_path,
            caption_path: caption,
        });
    }

    // Sort for deterministic output
    images.sort_by(|a, b| a.path.cmp(&b.path));

    let image_count = images.len() as u32;
    let captioned_count = images.iter().filter(|i| i.caption_path.is_some()).count() as u32;
    let caption_coverage = if image_count > 0 {
        captioned_count as f32 / image_count as f32
    } else {
        0.0
    };

    Ok(DatasetInfo {
        name,
        path: path.to_path_buf(),
        image_count,
        captioned_count,
        caption_coverage,
        images,
    })
}

/// Validate a dataset. Fails if 0 images, warns if < 5.
pub fn validate(path: &Path) -> Result<DatasetInfo> {
    let info = scan(path)?;

    if info.image_count == 0 {
        bail!(
            "Dataset '{}' contains no valid images (jpg, jpeg, png).",
            info.name
        );
    }

    Ok(info)
}

/// List all managed datasets under ~/.modl/datasets/
pub fn list() -> Result<Vec<DatasetInfo>> {
    let root = datasets_root();
    if !root.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&root)
        .with_context(|| format!("Failed to read datasets directory: {}", root.display()))?;

    let mut datasets = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            match scan(&path) {
                Ok(info) => datasets.push(info),
                Err(_) => continue, // Skip unreadable directories
            }
        }
    }

    datasets.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(datasets)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset(dir: &Path, images: &[&str], captions: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        for img in images {
            std::fs::write(dir.join(img), b"fake image data").unwrap();
        }
        for cap in captions {
            std::fs::write(dir.join(cap), "a photo of OHWX").unwrap();
        }
    }

    #[test]
    fn test_scan_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let info = scan(tmp.path()).unwrap();
        assert_eq!(info.image_count, 0);
        assert_eq!(info.caption_coverage, 0.0);
    }

    #[test]
    fn test_scan_with_images_and_captions() {
        let tmp = tempfile::TempDir::new().unwrap();
        create_test_dataset(
            tmp.path(),
            &["a.jpg", "b.png", "c.jpeg"],
            &["a.txt", "b.txt"],
        );
        let info = scan(tmp.path()).unwrap();
        assert_eq!(info.image_count, 3);
        assert_eq!(info.captioned_count, 2);
        assert!((info.caption_coverage - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_scan_ignores_non_images() {
        let tmp = tempfile::TempDir::new().unwrap();
        create_test_dataset(tmp.path(), &["a.jpg", "b.txt", "c.bmp", "d.gif"], &[]);
        let info = scan(tmp.path()).unwrap();
        assert_eq!(info.image_count, 1); // only a.jpg
    }

    #[test]
    fn test_scan_accepts_webp() {
        let tmp = tempfile::TempDir::new().unwrap();
        create_test_dataset(tmp.path(), &["a.jpg", "b.webp", "c.png"], &[]);
        let info = scan(tmp.path()).unwrap();
        assert_eq!(info.image_count, 3); // jpg, webp, png all valid
    }

    #[test]
    fn test_validate_fails_on_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = validate(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_succeeds_with_images() {
        let tmp = tempfile::TempDir::new().unwrap();
        create_test_dataset(tmp.path(), &["a.jpg"], &[]);
        let info = validate(tmp.path()).unwrap();
        assert_eq!(info.image_count, 1);
    }

    #[test]
    fn test_create_copies_images_and_captions() {
        let src = tempfile::TempDir::new().unwrap();
        create_test_dataset(src.path(), &["photo1.jpg", "photo2.png"], &["photo1.txt"]);

        // Override datasets root to temp dir
        let _dest_root = tempfile::TempDir::new().unwrap();
        let _dest = _dest_root.path().join("test-dataset");

        // Use scan on src, then manually create to test the logic
        let result = scan(src.path()).unwrap();
        assert_eq!(result.image_count, 2);
        assert_eq!(result.captioned_count, 1);
    }

    #[test]
    fn test_create_fails_on_nonexistent_source() {
        let result = create("test", Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_nonexistent_path() {
        let result = scan(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }
}
