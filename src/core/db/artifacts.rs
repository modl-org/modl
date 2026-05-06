use anyhow::{Context, Result};
use rusqlite::params;

use super::Database;

impl Database {
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

    /// Get an artifact by exact artifact_id (no fuzzy matching).
    pub fn get_artifact_exact(&self, artifact_id: &str) -> Result<Option<ArtifactRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT artifact_id, job_id, kind, path, sha256, size_bytes, metadata, created_at
                 FROM artifacts WHERE artifact_id = ?1",
            )
            .context("Failed to prepare query")?;

        let mut rows = stmt
            .query_map(params![artifact_id], ArtifactRecord::from_row)
            .context("Failed to query artifact")?;

        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Find the most recent artifact whose file path matches (exact).
    pub fn find_artifact_by_path(&self, path: &str) -> Result<Option<ArtifactRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT artifact_id, job_id, kind, path, sha256, size_bytes, metadata, created_at
                 FROM artifacts WHERE path = ?1 ORDER BY created_at DESC LIMIT 1",
            )
            .context("Failed to prepare query")?;

        let mut rows = stmt
            .query_map(params![path], ArtifactRecord::from_row)
            .context("Failed to query artifact by path")?;

        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// List artifacts, optionally filtered by job_id
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

    /// Search artifacts whose job spec_json contains `query` (case-insensitive).
    ///
    /// Returns `(ArtifactRecord, spec_json)` pairs ordered by job creation time,
    /// newest first. Single JOIN query — no N+1.
    pub fn search_artifacts(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(ArtifactRecord, String)>> {
        let pattern = format!("%{}%", query.to_lowercase());
        let mut stmt = self.conn.prepare(
            "SELECT a.artifact_id, a.job_id, a.kind, a.path, a.sha256, a.size_bytes, a.metadata, a.created_at,
                    j.spec_json
             FROM artifacts a
             JOIN jobs j ON j.job_id = a.job_id
             WHERE lower(j.spec_json) LIKE ?1
             ORDER BY j.created_at DESC
             LIMIT ?2",
        ).context("Failed to prepare search query")?;

        let rows = stmt
            .query_map(params![pattern, limit as i64], |row| {
                let artifact = ArtifactRecord {
                    artifact_id: row.get(0)?,
                    job_id: row.get(1)?,
                    kind: row.get(2)?,
                    path: row.get(3)?,
                    sha256: row.get(4)?,
                    size_bytes: row.get::<_, i64>(5)? as u64,
                    metadata: row.get(6)?,
                    created_at: row.get(7)?,
                };
                let spec_json: String = row.get(8)?;
                Ok((artifact, spec_json))
            })
            .context("Failed to search artifacts")?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect search results")
    }

    /// Delete artifact records by file path. Returns number of deleted rows.
    pub fn delete_artifacts_by_path(&self, path: &str) -> Result<usize> {
        let deleted = self
            .conn
            .execute("DELETE FROM artifacts WHERE path = ?1", params![path])
            .context("Failed to delete artifacts by path")?;
        Ok(deleted)
    }
}

#[derive(Debug)]
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
    pub(crate) fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
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
