use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::PathBuf;

/// Local database for tracking installed models
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open (or create) the database at ~/.modl/state.db
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

    #[allow(dead_code)]
    pub fn open_at(path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(path).context("Failed to open database")?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    fn default_path() -> PathBuf {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".modl")
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

            CREATE TABLE IF NOT EXISTS jobs (
                job_id       TEXT PRIMARY KEY,
                kind         TEXT NOT NULL DEFAULT 'train',
                status       TEXT NOT NULL DEFAULT 'queued',
                spec_json    TEXT NOT NULL,
                target       TEXT NOT NULL DEFAULT 'local',
                provider     TEXT,
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                started_at   TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS job_events (
                job_id    TEXT NOT NULL REFERENCES jobs(job_id),
                sequence  INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                timestamp  TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(job_id, sequence)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                job_id      TEXT REFERENCES jobs(job_id),
                kind        TEXT NOT NULL,
                path        TEXT NOT NULL,
                sha256      TEXT NOT NULL,
                size_bytes  INTEGER NOT NULL,
                metadata    TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            ",
            )
            .context("Failed to run database migrations")?;
        Ok(())
    }

    /// Record a model as installed
    #[allow(clippy::too_many_arguments)]
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

    /// Update the store_path for a model (used during storage migration)
    pub fn update_store_path(&self, id: &str, new_path: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE installed SET store_path = ?1 WHERE id = ?2",
                params![new_path, id],
            )
            .context("Failed to update store path")?;
        Ok(())
    }

    /// Find an installed model by ID
    pub fn find_installed(&self, id: &str) -> Result<Option<InstalledModel>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, asset_type, variant, sha256, size, file_name, store_path FROM installed WHERE id = ?1")
            .context("Failed to prepare query")?;
        let mut rows = stmt
            .query_map(params![id], |row| {
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
            .context("Failed to query installed model")?;
        match rows.next() {
            Some(Ok(model)) => Ok(Some(model)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// List all installed models
    pub fn list_installed(&self, type_filter: Option<&str>) -> Result<Vec<InstalledModel>> {
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
            return rows
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Failed to collect results");
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

    // -----------------------------------------------------------------------
    // Jobs CRUD
    // -----------------------------------------------------------------------

    /// Insert a new training job
    pub fn insert_job(
        &self,
        job_id: &str,
        kind: &str,
        status: &str,
        spec_json: &str,
        target: &str,
        provider: Option<&str>,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO jobs (job_id, kind, status, spec_json, target, provider)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![job_id, kind, status, spec_json, target, provider],
            )
            .context("Failed to insert job")?;
        Ok(())
    }

    /// Update job status (and optional timestamp fields)
    pub fn update_job_status(&self, job_id: &str, status: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        match status {
            "running" => {
                self.conn
                    .execute(
                        "UPDATE jobs SET status = ?1, started_at = ?2 WHERE job_id = ?3",
                        params![status, now, job_id],
                    )
                    .context("Failed to update job status")?;
            }
            "completed" | "error" | "cancelled" => {
                self.conn
                    .execute(
                        "UPDATE jobs SET status = ?1, completed_at = ?2 WHERE job_id = ?3",
                        params![status, now, job_id],
                    )
                    .context("Failed to update job status")?;
            }
            _ => {
                self.conn
                    .execute(
                        "UPDATE jobs SET status = ?1 WHERE job_id = ?2",
                        params![status, job_id],
                    )
                    .context("Failed to update job status")?;
            }
        }
        Ok(())
    }

    /// Get a single job by ID (or prefix match)
    pub fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT job_id, kind, status, spec_json, target, provider, created_at, started_at, completed_at FROM jobs WHERE job_id = ?1"
        ).context("Failed to prepare query")?;

        let mut rows = stmt
            .query_map(params![job_id], JobRecord::from_row)
            .context("Failed to query job")?;

        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// List all jobs, optionally filtered by status
    pub fn list_jobs(&self, status_filter: Option<&str>) -> Result<Vec<JobRecord>> {
        let sql = if status_filter.is_some() {
            "SELECT job_id, kind, status, spec_json, target, provider, created_at, started_at, completed_at FROM jobs WHERE status = ?1 ORDER BY created_at DESC"
        } else {
            "SELECT job_id, kind, status, spec_json, target, provider, created_at, started_at, completed_at FROM jobs ORDER BY created_at DESC"
        };

        let mut stmt = self.conn.prepare(sql).context("Failed to prepare query")?;

        let rows = if let Some(s) = status_filter {
            stmt.query_map(params![s], JobRecord::from_row)
                .context("Failed to query jobs")?
        } else {
            stmt.query_map([], JobRecord::from_row)
                .context("Failed to query jobs")?
        };

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect job results")
    }

    /// Find jobs whose lora_name matches (via job_id pattern `job-<name>-*`)
    pub fn find_jobs_by_lora_name(&self, lora_name: &str) -> Result<Vec<JobRecord>> {
        let pattern = format!("job-{lora_name}-%");
        let mut stmt = self.conn.prepare(
            "SELECT job_id, kind, status, spec_json, target, provider, created_at, started_at, completed_at \
             FROM jobs WHERE job_id LIKE ?1 ORDER BY created_at ASC"
        ).context("Failed to prepare query")?;

        let rows = stmt
            .query_map(params![pattern], JobRecord::from_row)
            .context("Failed to query jobs by lora name")?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect job results")
    }

    /// Delete all jobs and their events for a given LoRA name
    pub fn delete_jobs_by_lora_name(&self, lora_name: &str) -> Result<()> {
        let pattern = format!("job-{lora_name}-%");
        // Delete events first (child records)
        self.conn
            .execute(
                "DELETE FROM job_events WHERE job_id LIKE ?1",
                params![pattern],
            )
            .context("Failed to delete job events")?;
        // Delete the jobs themselves
        let deleted = self
            .conn
            .execute("DELETE FROM jobs WHERE job_id LIKE ?1", params![pattern])
            .context("Failed to delete jobs")?;
        if deleted > 0 {
            println!(
                "  {} Removed {} job record(s)",
                console::style("×").red(),
                deleted
            );
        }
        Ok(())
    }

    /// Insert a job event
    pub fn insert_job_event(&self, job_id: &str, sequence: u64, event_json: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO job_events (job_id, sequence, event_json)
                 VALUES (?1, ?2, ?3)",
                params![job_id, sequence as i64, event_json],
            )
            .context("Failed to insert job event")?;
        Ok(())
    }

    /// Insert an artifact record
    #[allow(clippy::too_many_arguments)]
    pub fn insert_artifact(
        &self,
        artifact_id: &str,
        job_id: Option<&str>,
        kind: &str,
        path: &str,
        sha256: &str,
        size_bytes: u64,
        metadata: Option<&str>,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO artifacts (artifact_id, job_id, kind, path, sha256, size_bytes, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![artifact_id, job_id, kind, path, sha256, size_bytes as i64, metadata],
            )
            .context("Failed to insert artifact")?;
        Ok(())
    }

    /// Find an artifact by name or artifact_id.
    ///
    /// Matches exact artifact_id first, then tries matching by LoRA name
    /// (the middle segment of `train:<name>:<hash>` artifact IDs).
    pub fn find_artifact(&self, query: &str) -> Result<Option<ArtifactRecord>> {
        // Exact match first
        let mut stmt = self
            .conn
            .prepare(
                "SELECT artifact_id, job_id, kind, path, sha256, size_bytes, metadata, created_at
             FROM artifacts WHERE artifact_id = ?1",
            )
            .context("Failed to prepare query")?;

        let mut rows = stmt
            .query_map(params![query], ArtifactRecord::from_row)
            .context("Failed to query artifact")?;

        if let Some(Ok(record)) = rows.next() {
            return Ok(Some(record));
        }

        // Fuzzy match: look for artifacts whose ID contains the query as the name segment
        let pattern = format!("train:{}:%", query);
        let mut stmt = self
            .conn
            .prepare(
                "SELECT artifact_id, job_id, kind, path, sha256, size_bytes, metadata, created_at
             FROM artifacts WHERE artifact_id LIKE ?1 ORDER BY created_at DESC LIMIT 1",
            )
            .context("Failed to prepare query")?;

        let mut rows = stmt
            .query_map(params![pattern], ArtifactRecord::from_row)
            .context("Failed to query artifact by name")?;

        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// List artifacts, optionally filtered by job_id
    #[allow(dead_code)]
    pub fn list_artifacts(&self, job_id: Option<&str>) -> Result<Vec<ArtifactRecord>> {
        let sql = if job_id.is_some() {
            "SELECT artifact_id, job_id, kind, path, sha256, size_bytes, metadata, created_at FROM artifacts WHERE job_id = ?1 ORDER BY created_at DESC"
        } else {
            "SELECT artifact_id, job_id, kind, path, sha256, size_bytes, metadata, created_at FROM artifacts ORDER BY created_at DESC"
        };

        let mut stmt = self.conn.prepare(sql).context("Failed to prepare query")?;

        let rows = if let Some(jid) = job_id {
            stmt.query_map(params![jid], ArtifactRecord::from_row)
                .context("Failed to query artifacts")?
        } else {
            stmt.query_map([], ArtifactRecord::from_row)
                .context("Failed to query artifacts")?
        };

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect artifact results")
    }

    /// Delete an artifact record by artifact_id
    pub fn delete_artifact(&self, artifact_id: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM artifacts WHERE artifact_id = ?1",
                params![artifact_id],
            )
            .context("Failed to delete artifact")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

#[derive(Debug)]
#[allow(dead_code)]
pub struct JobRecord {
    pub job_id: String,
    pub kind: String,
    pub status: String,
    pub spec_json: String,
    pub target: String,
    pub provider: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

impl JobRecord {
    fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        Ok(Self {
            job_id: row.get(0)?,
            kind: row.get(1)?,
            status: row.get(2)?,
            spec_json: row.get(3)?,
            target: row.get(4)?,
            provider: row.get(5)?,
            created_at: row.get(6)?,
            started_at: row.get(7)?,
            completed_at: row.get(8)?,
        })
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct ArtifactRecord {
    pub artifact_id: String,
    pub job_id: Option<String>,
    pub kind: String,
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub metadata: Option<String>,
    pub created_at: String,
}

impl ArtifactRecord {
    fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        Ok(Self {
            artifact_id: row.get(0)?,
            job_id: row.get(1)?,
            kind: row.get(2)?,
            path: row.get(3)?,
            sha256: row.get(4)?,
            size_bytes: row.get::<_, i64>(5)? as u64,
            metadata: row.get(6)?,
            created_at: row.get(7)?,
        })
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
