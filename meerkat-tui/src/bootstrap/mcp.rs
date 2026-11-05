//! MCP server loading, mirroring `rkat`'s `create_mcp_tools`
//! (`meerkat-cli/src/main.rs`).
//!
//! Servers from the project (`<context_root>/.rkat/mcp.toml`) and user
//! (`<user_config_root>/.rkat/mcp.toml`) configs are staged on one
//! [`McpRouter`] and connected in parallel in the background. The returned
//! dispatcher is an [`McpRouterAdapter`], whose `poll_external_updates` lets the
//! agent loop pick up tools as each server finishes its handshake.

use std::sync::Arc;
use std::time::Duration;

use meerkat::{McpConnection, McpRouter, McpRouterAdapter};
use meerkat_core::AgentToolDispatcher;
use meerkat_core::mcp_config::{McpConfig, McpScope, McpTransportConfig};

use super::RuntimeScope;
use crate::error::TuiError;

/// Grace added on top of the slowest server's connect timeout when waiting.
const WAIT_GRACE_SECS: u64 = 5;
/// Upper bound on how long `wait_for_mcp` blocks.
const WAIT_CAP_SECS: u64 = 60;

/// Loaded MCP servers, ready to be injected as external tools.
pub struct McpTools {
    /// Tool dispatcher backed by the MCP router (supports external updates).
    pub dispatcher: Arc<dyn AgentToolDispatcher>,
    /// Names of the configured servers, in load order (project first).
    pub server_names: Vec<String>,
}

impl std::fmt::Debug for McpTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpTools")
            .field("server_names", &self.server_names)
            .field("tools", &mcp_tool_names(self.dispatcher.as_ref()))
            .finish()
    }
}

/// Load configured MCP servers and start connecting them in the background.
///
/// Returns `Ok(None)` when no servers are configured. With `wait_for_mcp`,
/// blocks until every server has connected or failed (bounded by the slowest
/// server's `connect_timeout_secs` plus a grace period, capped at 60 s);
/// failures are logged, not returned.
///
/// # Errors
///
/// Returns [`TuiError::McpConfig`] when an `mcp.toml` cannot be read or parsed
/// and [`TuiError::Mcp`] when the router cannot apply the staged servers.
pub async fn load_mcp_tools(
    scope: &RuntimeScope,
    wait_for_mcp: bool,
) -> Result<Option<McpTools>, TuiError> {
    let servers = McpConfig::load_with_scopes_from_roots(
        Some(scope.context_root.as_path()),
        scope.user_config_root.as_deref(),
    )
    .await?;
    if servers.is_empty() {
        return Ok(None);
    }

    for s in servers.iter().filter(|s| s.scope == McpScope::Project) {
        let target = match &s.server.transport {
            McpTransportConfig::Stdio(stdio) if stdio.args.is_empty() => stdio.command.clone(),
            McpTransportConfig::Stdio(stdio) => {
                format!("{} {}", stdio.command, stdio.args.join(" "))
            }
            McpTransportConfig::Http(http) => http.url.clone(),
        };
        tracing::warn!(server = %s.server.name, %target, "loading MCP server from project config");
    }
    tracing::info!("loading {} MCP server(s)", servers.len());

    let mut router = McpRouter::new();
    for s in &servers {
        router.stage_add(s.server.clone());
    }
    let applied = router
        .apply_staged()
        .await
        .map_err(|e| TuiError::Mcp(format!("failed to start MCP servers: {e}")))?;
    let adapter = Arc::new(McpRouterAdapter::new(router));

    if wait_for_mcp && applied.pending_count > 0 {
        let slowest = servers
            .iter()
            .filter_map(|s| s.server.connect_timeout_secs)
            .max()
            .unwrap_or(McpConnection::DEFAULT_CONNECT_TIMEOUT_SECS);
        let timeout =
            Duration::from_secs((u64::from(slowest) + WAIT_GRACE_SECS).min(WAIT_CAP_SECS));
        tracing::info!(
            "waiting for {} MCP server(s) to connect (timeout: {}s)",
            applied.pending_count,
            timeout.as_secs()
        );
        for notice in adapter.wait_until_ready(timeout).await {
            if notice.status.starts_with("failed") {
                tracing::warn!(server = %notice.server, status = %notice.status, "MCP server failed");
            }
        }
    }

    adapter
        .refresh_tools()
        .await
        .map_err(|e| TuiError::Mcp(format!("failed to refresh MCP tools: {e}")))?;

    Ok(Some(McpTools {
        dispatcher: adapter,
        server_names: servers.into_iter().map(|s| s.server.name).collect(),
    }))
}

/// Names of the tools a dispatcher currently exposes.
#[must_use]
pub fn mcp_tool_names(dispatcher: &dyn AgentToolDispatcher) -> Vec<String> {
    dispatcher.tools().iter().map(|t| t.name.clone()).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::path::{Path, PathBuf};

    use meerkat_store::RealmOrigin;

    use super::*;
    use crate::paths::ProjectRoot;

    fn scope_in(dir: &Path) -> RuntimeScope {
        let project_root = ProjectRoot::new(dir).expect("project root");
        let context_root = project_root.as_path().to_path_buf();
        RuntimeScope {
            project_root,
            context_root,
            // A separate empty dir so the real `~/.rkat/mcp.toml` is not read.
            user_config_root: Some(dir.join("user")),
            realm_id: "test".to_string(),
            state_root: dir.join("state"),
            backend_hint: None,
            origin: RealmOrigin::Explicit,
        }
    }

    #[tokio::test]
    async fn no_config_yields_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tools = load_mcp_tools(&scope_in(dir.path()), false)
            .await
            .expect("load");
        assert!(tools.is_none());
    }

    #[tokio::test]
    async fn invalid_config_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join(".rkat")).expect("mkdir");
        std::fs::write(dir.path().join(".rkat/mcp.toml"), "not = [valid").expect("write");
        let err = load_mcp_tools(&scope_in(dir.path()), false)
            .await
            .expect_err("parse error");
        assert!(matches!(err, TuiError::McpConfig(_)), "{err:?}");
    }

    fn fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("target/debug/mcp-test-server")
    }

    /// Requires `cargo build -p mcp-test-server`.
    #[tokio::test]
    #[ignore = "integration-real: spawns the mcp-test-server fixture"]
    async fn integration_real_mcp_fixture_connects() {
        let server = fixture_path();
        assert!(
            server.exists(),
            "build the fixture first: cargo build -p mcp-test-server"
        );
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join(".rkat")).expect("mkdir");
        let toml = format!(
            "[[servers]]\nname = \"fixture\"\ncommand = {:?}\n",
            server.to_string_lossy()
        );
        std::fs::write(dir.path().join(".rkat/mcp.toml"), toml).expect("write");

        let tools = load_mcp_tools(&scope_in(dir.path()), true)
            .await
            .expect("load")
            .expect("servers configured");
        assert_eq!(tools.server_names, vec!["fixture".to_string()]);
        let names = mcp_tool_names(tools.dispatcher.as_ref());
        assert!(!names.is_empty(), "no tools from fixture: {tools:?}");
    }
}
