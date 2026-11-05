//! Workspace and realm resolution.
//!
//! By default all state for a workspace (sessions, realm config, rules, input
//! history, logs) lives inside it, under `<workspace>/.rkat/tui/`, in a realm
//! with the constant id [`WORKSPACE_REALM_ID`]. Nothing is keyed by the
//! workspace's absolute path, so renaming or moving the directory keeps every
//! session reachable with `rkat-tui -c`. A `.gitignore` is written into the
//! state directory so it is never committed.
//!
//! `--realm`, `--isolated` and `--state-root` opt back into the shared realm
//! model `rkat` uses (`resolve_runtime_scope` / `load_config` in
//! `meerkat-cli/src/main.rs`).

use std::path::{Path, PathBuf};

use meerkat_core::{Config, ConfigStore, FileConfigStore, RealmConfig, RealmSelection};
use meerkat_store::{RealmBackend, RealmOrigin, realm_paths_in};

use super::dirs::GlobalDirs;
use crate::cli::{Args, RealmBackendArg};
use crate::error::TuiError;
use crate::paths::{PathError, ProjectRoot};

/// Realm id used for workspace-local state.
pub const WORKSPACE_REALM_ID: &str = "tui";
/// Directory under the workspace that holds realm directories.
pub const WORKSPACE_STATE_DIR: &str = ".rkat";

/// Where this TUI process lives: project, conventions roots and realm.
#[derive(Debug, Clone)]
pub struct RuntimeScope {
    /// Canonical project root (path policy and tool resolution).
    pub project_root: ProjectRoot,
    /// Conventions root for `AGENTS.md`, `.rkat/` hooks, skills and MCP config.
    /// Equal to the project root path.
    pub context_root: PathBuf,
    /// User-global conventions root (`--user-config-root`, else the home
    /// directory). `None` when neither is available.
    pub user_config_root: Option<PathBuf>,
    /// User-global `rkat-tui` directories (`--config-dir`, else
    /// `~/.config/rkat-tui`). `None` when nothing resolves.
    pub global: Option<GlobalDirs>,
    /// Realm the sessions and config are stored in.
    pub realm_id: String,
    /// Directory containing realm directories.
    pub state_root: PathBuf,
    /// Backend for a *new* realm; existing realms keep their pinned backend.
    pub backend_hint: Option<RealmBackend>,
    /// How the realm was selected (recorded in a new realm's manifest).
    pub origin: RealmOrigin,
}

impl RuntimeScope {
    /// Root directory of the active realm (`<state_root>/<realm_id>`).
    #[must_use]
    pub fn realm_root(&self) -> PathBuf {
        realm_paths_in(&self.state_root, &self.realm_id).root
    }

    /// Path of the realm's `config.toml`.
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        realm_paths_in(&self.state_root, &self.realm_id).config_path
    }
}

impl From<RealmBackendArg> for RealmBackend {
    fn from(value: RealmBackendArg) -> Self {
        match value {
            RealmBackendArg::Jsonl => Self::Jsonl,
            RealmBackendArg::Sqlite => Self::Sqlite,
            RealmBackendArg::Redb => Self::Redb,
        }
    }
}

/// Resolve the workspace, conventions roots and realm from the arguments.
///
/// The workspace is `--workspace`, else the nearest ancestor of the working
/// directory that already contains `.rkat/tui/`, else the working directory.
/// State lives in `<workspace>/.rkat/tui/` unless `--realm`, `--isolated` or
/// `--state-root` select the shared realm model.
///
/// # Errors
///
/// Returns [`TuiError::Realm`] for conflicting/invalid realm flags,
/// [`TuiError::ProjectRoot`] when the root cannot be canonicalized, and
/// [`TuiError::Io`] when the state directory cannot be created.
pub fn resolve_scope(args: &Args) -> Result<RuntimeScope, TuiError> {
    let context_root = match &args.context_root {
        Some(root) => root.clone(),
        None => {
            let cwd = std::env::current_dir().map_err(|source| TuiError::ProjectRoot {
                path: PathBuf::from("."),
                source,
            })?;
            find_workspace_above(&cwd).unwrap_or(cwd)
        }
    };
    let project_root = project_root_from(&context_root)?;
    let context_root = project_root.as_path().to_path_buf();
    let user_config_root = args.user_config_root.clone().or_else(dirs::home_dir);
    let global = GlobalDirs::resolve(args.config_dir.as_deref(), args.user_config_root.as_deref());

    let uses_shared_realm = args.realm.is_some() || args.isolated || args.state_root.is_some();
    if !uses_shared_realm {
        let state_root = context_root.join(WORKSPACE_STATE_DIR);
        ensure_workspace_state_dir(&state_root)?;
        return Ok(RuntimeScope {
            project_root,
            context_root,
            user_config_root,
            global,
            realm_id: WORKSPACE_REALM_ID.to_string(),
            state_root,
            backend_hint: args.realm_backend.map(Into::into),
            origin: RealmOrigin::Workspace,
        });
    }

    let selection = RealmConfig::selection_from_inputs(
        args.realm.clone(),
        args.isolated,
        RealmSelection::WorkspaceDerived {
            root: context_root.clone(),
        },
    )?;
    let origin = match &selection {
        RealmSelection::Explicit { .. } => RealmOrigin::Explicit,
        RealmSelection::Isolated => RealmOrigin::Generated,
        RealmSelection::WorkspaceDerived { .. } => RealmOrigin::Workspace,
    };
    let backend_hint: Option<RealmBackend> = args.realm_backend.map(Into::into);
    let realm_cfg = RealmConfig {
        selection,
        instance_id: None,
        backend_hint: backend_hint.map(|b| b.as_str().to_string()),
        state_root: args.state_root.clone(),
    };
    let locator = realm_cfg.resolve_locator()?;

    Ok(RuntimeScope {
        project_root,
        context_root,
        user_config_root,
        global,
        realm_id: locator.realm_id,
        state_root: locator.state_root,
        // Only an explicit request pins a backend; existing realms are opened
        // with the backend recorded in their manifest.
        backend_hint,
        origin,
    })
}

/// The nearest ancestor of `start` (inclusive) that already holds a
/// workspace-local state directory, if any.
fn find_workspace_above(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current
            .join(WORKSPACE_STATE_DIR)
            .join(WORKSPACE_REALM_ID)
            .is_dir()
        {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Create `<workspace>/.rkat/tui/` and drop a `.gitignore` into it so session
/// state, history, logs and realm rules are never committed.
fn ensure_workspace_state_dir(state_root: &Path) -> Result<(), TuiError> {
    let realm_dir = state_root.join(WORKSPACE_REALM_ID);
    std::fs::create_dir_all(&realm_dir)
        .map_err(|e| TuiError::io("create workspace state directory", e))?;
    let gitignore = realm_dir.join(".gitignore");
    if !gitignore.exists() {
        std::fs::write(&gitignore, "# rkat-tui session state; never commit.\n*\n")
            .map_err(|e| TuiError::io("write workspace state .gitignore", e))?;
    }
    Ok(())
}

fn project_root_from(path: &Path) -> Result<ProjectRoot, TuiError> {
    ProjectRoot::new(path).map_err(|err| match err {
        PathError::Io { path, source } => TuiError::ProjectRoot { path, source },
        other => TuiError::ProjectRoot {
            path: path.to_path_buf(),
            source: std::io::Error::other(other.to_string()),
        },
    })
}

/// Load the realm config (`<state_root>/<realm>/config.toml`) and apply
/// environment overrides, exactly like `rkat`.
///
/// # Errors
///
/// Returns [`TuiError::Io`] when the realm directory cannot be created and
/// [`TuiError::Config`] when the file cannot be parsed or env overrides fail.
pub async fn load_config(scope: &RuntimeScope) -> Result<Config, TuiError> {
    let config_path = scope.config_path();
    if let Some(parent) = config_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| TuiError::io("failed to create realm config directory", e))?;
    }
    let mut config = FileConfigStore::new(config_path)
        .get()
        .await
        .map_err(|source| TuiError::Config {
            context: "failed to load config",
            source,
        })?;
    config
        .apply_env_overrides()
        .map_err(|source| TuiError::Config {
            context: "failed to apply env overrides",
            source,
        })?;
    Ok(config)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec!["rkat-tui"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).unwrap()
    }

    #[test]
    fn default_state_lives_inside_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().to_str().unwrap();
        let scope = resolve_scope(&args(&["-w", ws])).unwrap();
        let canonical = std::fs::canonicalize(dir.path()).unwrap();
        assert_eq!(scope.realm_id, WORKSPACE_REALM_ID);
        assert_eq!(scope.state_root, canonical.join(".rkat"));
        assert_eq!(scope.realm_root(), canonical.join(".rkat").join("tui"));
        assert_eq!(scope.origin, RealmOrigin::Workspace);
        let gitignore = scope.realm_root().join(".gitignore");
        assert!(gitignore.is_file());
        assert!(std::fs::read_to_string(gitignore).unwrap().contains('*'));
    }

    #[test]
    fn workspace_is_found_from_a_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".rkat/tui")).unwrap();
        let sub = dir.path().join("src/deep");
        std::fs::create_dir_all(&sub).unwrap();
        assert_eq!(find_workspace_above(&sub).as_deref(), Some(dir.path()));
        let other = tempfile::tempdir().unwrap();
        assert!(find_workspace_above(other.path()).is_none());
    }

    #[test]
    fn realm_flags_opt_into_shared_state() {
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let scope = resolve_scope(&args(&[
            "-w",
            dir.path().to_str().unwrap(),
            "--isolated",
            "--state-root",
            state.path().to_str().unwrap(),
        ]))
        .unwrap();
        assert_ne!(scope.realm_id, WORKSPACE_REALM_ID);
        assert_eq!(scope.state_root, state.path());
        assert!(!dir.path().join(".rkat/tui").exists());
    }

    #[test]
    fn global_dirs_follow_flags() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().to_str().unwrap();
        let cfg = dir.path().join("cfg");
        let scope = resolve_scope(&args(&[
            "-w",
            ws,
            "--config-dir",
            cfg.to_str().unwrap(),
            "--user-config-root",
            ws,
        ]))
        .unwrap();
        assert_eq!(scope.global, Some(GlobalDirs { config: cfg }));

        let scope = resolve_scope(&args(&["-w", ws, "--user-config-root", ws])).unwrap();
        assert_eq!(
            scope.global.map(|g| g.config),
            Some(dir.path().join(".config").join("rkat-tui"))
        );
    }
}
