use anyhow::{Context, Result};
use rusqlite::params;

use super::Database;

impl Database {
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

    /// Update status for all jobs matching a lora name where current status matches.
    pub fn update_job_status_by_lora_name(
        &self,
        lora_name: &str,
        from_status: &str,
        to_status: &str,
    ) -> Result<usize> {
        let pattern = format!("job-{lora_name}-%");
        let now = chrono::Utc::now().to_rfc3339();
        let updated = self
            .conn
            .execute(
                "UPDATE jobs SET status = ?1, completed_at = ?2 WHERE job_id LIKE ?3 AND status = ?4",
                params![to_status, now, pattern, from_status],
            )
            .context("Failed to update job status by lora name")?;
        Ok(updated)
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

    /// Count total jobs in the database.
    pub fn count_jobs(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM jobs", [], |row| row.get(0))?;
        Ok(count as usize)
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
}

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
    pub(crate) fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
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
