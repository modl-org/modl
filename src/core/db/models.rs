use anyhow::{Context, Result};
use rusqlite::params;

use super::Database;

impl Database {
    /// Record a model as installed
    pub fn insert_installed(&self, record: &InstalledModelRecord) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO installed (id, name, asset_type, variant, sha256, size, file_name, store_path)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    record.id,
                    record.name,
                    record.asset_type,
                    record.variant,
                    record.sha256,
                    record.size as i64,
                    record.file_name,
                    record.store_path
                ],
            )
            .context("Failed to insert installed model")?;
        // Keep the shared store index in sync (best-effort — errors ignored).
        crate::core::store_index::upsert(record);
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
        crate::core::store_index::remove_by_id(id);
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
}

/// Input record for inserting an installed model
pub struct InstalledModelRecord<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub asset_type: &'a str,
    pub variant: Option<&'a str>,
    pub sha256: &'a str,
    pub size: u64,
    pub file_name: &'a str,
    pub store_path: &'a str,
}

#[derive(Debug, serde::Serialize)]
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
