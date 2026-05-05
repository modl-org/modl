mod artifacts;
mod favorites;
mod jobs;
mod lora_library;
mod models;
mod sessions;
mod training_queue;

pub use artifacts::ArtifactRecord;
pub use jobs::JobRecord;
pub use lora_library::LibraryLoraRecord;
pub use models::{InstalledModel, InstalledModelRecord};

use anyhow::{Context, Result};
use rusqlite::{Connection, ErrorCode};
use std::path::Path;
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
        Self::open_resilient(&path)
    }

    #[allow(dead_code)]
    pub fn open_at(path: &Path) -> Result<Self> {
        Self::open_resilient(path)
    }

    /// Try opening with WAL (concurrent reads, used by web UI). If WAL setup
    /// or migrations fail with an I/O error — typically a full disk preventing
    /// SQLite from allocating its shared-memory file — fall back to DELETE
    /// journaling so destructive commands like `modl rm` can still run.
    fn open_resilient(path: &Path) -> Result<Self> {
        match Self::try_open(path, "WAL") {
            Ok(db) => Ok(db),
            Err(e) if is_sqlite_io_error(&e) => {
                eprintln!(
                    "warning: SQLite WAL setup failed ({}); falling back to legacy journaling. \
                     Free disk space to restore normal mode.",
                    short_error(&e)
                );
                Self::try_open(path, "DELETE")
            }
            Err(e) => Err(e),
        }
    }

    fn try_open(path: &Path, journal_mode: &str) -> Result<Self> {
        let conn = Connection::open(path).context("Failed to open database")?;
        let _ = conn.pragma_update(None, "journal_mode", journal_mode);
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    fn default_path() -> PathBuf {
        super::paths::modl_root().join("state.db")
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

            CREATE TABLE IF NOT EXISTS favorites (
                path       TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS studio_sessions (
                id           TEXT PRIMARY KEY,
                intent       TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS session_events (
                session_id TEXT NOT NULL REFERENCES studio_sessions(id),
                sequence   INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                timestamp  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS session_images (
                session_id TEXT NOT NULL REFERENCES studio_sessions(id),
                image_path TEXT NOT NULL,
                role       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lora_library (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                trigger_word  TEXT,
                base_model    TEXT,
                lora_path     TEXT NOT NULL,
                thumbnail     TEXT,
                step          INTEGER,
                training_run  TEXT,
                config_json   TEXT,
                tags          TEXT,
                notes         TEXT,
                size_bytes    INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS training_queue (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                position   INTEGER NOT NULL,
                name       TEXT NOT NULL,
                spec_json  TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            ",
            )
            .context("Failed to run database migrations")?;
        Ok(())
    }
}

fn is_sqlite_io_error(err: &anyhow::Error) -> bool {
    err.chain().any(|e| {
        e.downcast_ref::<rusqlite::Error>()
            .and_then(|sqlite_err| match sqlite_err {
                rusqlite::Error::SqliteFailure(f, _) => Some(f.code),
                _ => None,
            })
            .is_some_and(|code| {
                matches!(
                    code,
                    ErrorCode::SystemIoFailure | ErrorCode::DiskFull | ErrorCode::CannotOpen
                )
            })
    })
}

fn short_error(err: &anyhow::Error) -> String {
    err.chain()
        .last()
        .map(|e| e.to_string())
        .unwrap_or_else(|| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db = Database::open_at(tmp.path()).unwrap();

        db.insert_installed(&InstalledModelRecord {
            id: "test-model",
            name: "Test Model",
            asset_type: "checkpoint",
            variant: Some("fp16"),
            sha256: "abcdef1234567890",
            size: 1024,
            file_name: "test.safetensors",
            store_path: "/store/checkpoints/abcdef/test.safetensors",
        })
        .unwrap();

        assert!(db.is_installed("test-model").unwrap());
        assert!(!db.is_installed("nonexistent").unwrap());

        let models = db.list_installed(None).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "test-model");

        db.remove_installed("test-model").unwrap();
        assert!(!db.is_installed("test-model").unwrap());
    }

    #[test]
    fn is_sqlite_io_error_recognizes_disk_full() {
        let err = anyhow::Error::from(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_FULL),
            Some("database or disk is full".to_string()),
        ))
        .context("Failed to run database migrations");
        assert!(is_sqlite_io_error(&err));
    }

    #[test]
    fn is_sqlite_io_error_recognizes_shmmap_failure() {
        // SQLITE_IOERR_SHMMAP — the exact error users hit when the disk is
        // full and SQLite can't grow the WAL shared-memory segment.
        let err = anyhow::Error::from(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR_SHMMAP),
            Some("I/O error within the xShmMap method".to_string()),
        ));
        assert!(is_sqlite_io_error(&err));
    }

    #[test]
    fn is_sqlite_io_error_ignores_unrelated_errors() {
        let err = anyhow::Error::from(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
            Some("constraint failed".to_string()),
        ));
        assert!(!is_sqlite_io_error(&err));
    }
}
