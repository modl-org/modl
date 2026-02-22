use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::PathBuf;

/// Local database for tracking installed models
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open (or create) the database at ~/.mods/state.db
    pub fn open() -> Result<Self> {
        let path = Self::default_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }
        let conn = Connection::open(&path).context("Failed to open database")?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    pub fn open_at(path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(path).context("Failed to open database")?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    fn default_path() -> PathBuf {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".mods")
            .join("state.db")
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "
            CREATE TABLE IF NOT EXISTS installed (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                asset_type  TEXT NOT NULL,
                variant     TEXT,
                sha256      TEXT NOT NULL,
                size        INTEGER NOT NULL,
                file_name   TEXT NOT NULL,
                store_path  TEXT NOT NULL,
                installed_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS symlinks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id    TEXT NOT NULL REFERENCES installed(id),
                link_path   TEXT NOT NULL,
                target_path TEXT NOT NULL,
                UNIQUE(link_path)
            );

            CREATE TABLE IF NOT EXISTS dependencies (
                parent_id   TEXT NOT NULL REFERENCES installed(id),
                child_id    TEXT NOT NULL REFERENCES installed(id),
                PRIMARY KEY (parent_id, child_id)
            );
            ",
            )
            .context("Failed to run database migrations")?;
        Ok(())
    }

    /// Record a model as installed
    pub fn insert_installed(
        &self,
        id: &str,
        name: &str,
        asset_type: &str,
        variant: Option<&str>,
        sha256: &str,
        size: u64,
        file_name: &str,
        store_path: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO installed (id, name, asset_type, variant, sha256, size, file_name, store_path)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![id, name, asset_type, variant, sha256, size as i64, file_name, store_path],
            )
            .context("Failed to insert installed model")?;
        Ok(())
    }

    /// Check if a model is installed
    pub fn is_installed(&self, id: &str) -> Result<bool> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM installed WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .context("Failed to check installed status")?;
        Ok(count > 0)
    }

    /// Remove a model from the installed table
    pub fn remove_installed(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM installed WHERE id = ?1", params![id])
            .context("Failed to remove installed model")?;
        Ok(())
    }

    /// List all installed models
    pub fn list_installed(
        &self,
        type_filter: Option<&str>,
    ) -> Result<Vec<InstalledModel>> {
        let mut stmt = if let Some(t) = type_filter {
            let mut s = self
                .conn
                .prepare("SELECT id, name, asset_type, variant, sha256, size, file_name, store_path FROM installed WHERE asset_type = ?1 ORDER BY name")
                .context("Failed to prepare query")?;
            let rows = s
                .query_map(params![t], |row| {
                    Ok(InstalledModel {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        asset_type: row.get(2)?,
                        variant: row.get(3)?,
                        sha256: row.get(4)?,
                        size: row.get::<_, i64>(5)? as u64,
                        file_name: row.get(6)?,
                        store_path: row.get(7)?,
                    })
                })
                .context("Failed to query installed models")?;
            return rows.collect::<std::result::Result<Vec<_>, _>>().context("Failed to collect results");
        } else {
            self.conn
                .prepare("SELECT id, name, asset_type, variant, sha256, size, file_name, store_path FROM installed ORDER BY name")
                .context("Failed to prepare query")?
        };

        let rows = stmt
            .query_map([], |row| {
                Ok(InstalledModel {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    asset_type: row.get(2)?,
                    variant: row.get(3)?,
                    sha256: row.get(4)?,
                    size: row.get::<_, i64>(5)? as u64,
                    file_name: row.get(6)?,
                    store_path: row.get(7)?,
                })
            })
            .context("Failed to query installed models")?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect results")
    }
}

#[derive(Debug)]
pub struct InstalledModel {
    pub id: String,
    pub name: String,
    pub asset_type: String,
    pub variant: Option<String>,
    pub sha256: String,
    pub size: u64,
    pub file_name: String,
    pub store_path: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db = Database::open_at(tmp.path()).unwrap();

        db.insert_installed(
            "test-model",
            "Test Model",
            "checkpoint",
            Some("fp16"),
            "abcdef1234567890",
            1024,
            "test.safetensors",
            "/store/checkpoints/abcdef/test.safetensors",
        )
        .unwrap();

        assert!(db.is_installed("test-model").unwrap());
        assert!(!db.is_installed("nonexistent").unwrap());

        let models = db.list_installed(None).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "test-model");

        db.remove_installed("test-model").unwrap();
        assert!(!db.is_installed("test-model").unwrap());
    }
}
