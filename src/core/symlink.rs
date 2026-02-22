use anyhow::{Context, Result};
use std::path::Path;

/// Create a symlink from `link` pointing to `target`
pub fn create(link: &Path, target: &Path) -> Result<()> {
    // Ensure parent directory of symlink exists
    if let Some(parent) = link.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    // Remove existing symlink if present
    if link.is_symlink() {
        std::fs::remove_file(link)
            .with_context(|| format!("Failed to remove existing symlink: {}", link.display()))?;
    } else if link.exists() {
        // Real file exists — don't overwrite
        anyhow::bail!(
            "File already exists at {} (not a symlink). Use `mods link` to register it.",
            link.display()
        );
    }

    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link)
        .with_context(|| format!("Failed to create symlink: {} -> {}", link.display(), target.display()))?;

    #[cfg(windows)]
    std::os::windows::fs::symlink_file(target, link)
        .with_context(|| format!("Failed to create symlink: {} -> {}", link.display(), target.display()))?;

    Ok(())
}

/// Remove a symlink (only if it is a symlink, not a real file)
pub fn remove(link: &Path) -> Result<()> {
    if link.is_symlink() {
        std::fs::remove_file(link)
            .with_context(|| format!("Failed to remove symlink: {}", link.display()))?;
    }
    Ok(())
}

/// Check if a symlink is valid (target exists)
pub fn is_valid(link: &Path) -> bool {
    link.is_symlink() && link.exists()
}

/// Find all broken symlinks in a directory
pub fn find_broken(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut broken = Vec::new();

    if !dir.exists() {
        return Ok(broken);
    }

    for entry in std::fs::read_dir(dir).context("Failed to read directory")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_symlink() && !path.exists() {
            broken.push(path);
        } else if path.is_dir() && !path.is_symlink() {
            broken.extend(find_broken(&path)?);
        }
    }

    Ok(broken)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_remove_symlink() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("target.txt");
        std::fs::write(&target, "hello").unwrap();

        let link = tmp.path().join("link.txt");
        create(&link, &target).unwrap();

        assert!(link.is_symlink());
        assert!(is_valid(&link));

        remove(&link).unwrap();
        assert!(!link.exists());
    }

    #[test]
    fn test_find_broken_symlinks() {
        let tmp = TempDir::new().unwrap();

        // Create a valid symlink
        let target = tmp.path().join("exists.txt");
        std::fs::write(&target, "hello").unwrap();
        let good_link = tmp.path().join("good.txt");
        create(&good_link, &target).unwrap();

        // Create a broken symlink
        let bad_target = tmp.path().join("missing.txt");
        let bad_link = tmp.path().join("broken.txt");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&bad_target, &bad_link).unwrap();

        let broken = find_broken(tmp.path()).unwrap();
        assert_eq!(broken.len(), 1);
        assert_eq!(broken[0], bad_link);
    }
}
