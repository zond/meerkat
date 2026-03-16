//! State directory management for mob-runner.
//!
//! Provides paths to persistent state files and detection of prior runs.

use std::path::{Path, PathBuf};

/// Paths and helpers for the `.mob-runner/` state directory.
pub struct StateDir {
    root: PathBuf,
}

impl StateDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Create the state directory and any required subdirectories.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.sessions_dir())?;
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Session store directory for `AgentFactory`.
    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    /// Session persistence (redb) — agent conversations survive restarts.
    pub fn sessions_redb(&self) -> PathBuf {
        self.root.join("sessions.redb")
    }

    /// Persistent mob storage (redb).
    pub fn mob_redb(&self) -> PathBuf {
        self.root.join("mob.redb")
    }

    /// Planner's mob definition output.
    pub fn mob_toml(&self) -> PathBuf {
        self.root.join("mob.toml")
    }

    /// Planner session ID file (for resuming planning conversations).
    pub fn planner_session_id(&self) -> PathBuf {
        self.root.join("planner_session_id")
    }

    /// Append-only event log.
    pub fn events_jsonl(&self) -> PathBuf {
        self.root.join("events.jsonl")
    }

    /// Whether a mob has been created (both redb and toml exist).
    pub fn has_mob(&self) -> bool {
        self.mob_redb().exists() && self.mob_toml().exists()
    }

    /// Whether a planner session can be resumed.
    pub fn has_planner_session(&self) -> bool {
        self.planner_session_id().exists()
    }
}
