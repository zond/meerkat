//! Error type for the `rkat-tui` surface.

use std::path::PathBuf;

/// Top-level error for the TUI surface.
#[derive(Debug, thiserror::Error)]
pub enum TuiError {
    /// The working directory or an explicit root could not be resolved.
    #[error("failed to resolve project root from {path}: {source}")]
    ProjectRoot {
        /// Path that was being resolved.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A generic I/O failure with context.
    #[error("{context}: {source}")]
    Io {
        /// What was being attempted.
        context: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Realm selection flags were invalid (e.g. `--realm` with `--isolated`).
    #[error("invalid realm selection: {0}")]
    Realm(#[from] meerkat_core::RuntimeBootstrapError),
    /// The realm config could not be loaded or env overrides were invalid.
    #[error("{context}: {source}")]
    Config {
        /// What was being attempted.
        context: &'static str,
        /// Underlying config error.
        #[source]
        source: meerkat_core::ConfigError,
    },
    /// The realm persistence backend could not be opened.
    #[error("failed to open realm persistence: {0}")]
    Persistence(#[from] meerkat::PersistenceError),
    /// A session store operation failed.
    #[error("session store error: {0}")]
    Store(#[from] meerkat_store::StoreError),
    /// The provider could not be determined or is not supported.
    #[error("{0}")]
    Provider(String),
    /// No API key was found for the selected provider.
    #[error(
        "no API key for provider `{provider}`: set it in the realm config or the provider's environment variable"
    )]
    MissingApiKey {
        /// Canonical provider name.
        provider: &'static str,
    },
    /// The LLM client could not be constructed.
    #[error("failed to build LLM client: {0}")]
    LlmClient(#[from] meerkat_client::FactoryError),
    /// A session service operation failed.
    #[error("session error: {0}")]
    Session(#[from] meerkat_core::service::SessionError),
    /// A session reference (`last`, `~N`, id, or prefix) did not resolve.
    #[error("cannot resolve session `{reference}`: {reason}")]
    SessionRef {
        /// The reference supplied by the user.
        reference: String,
        /// Why it did not resolve.
        reason: String,
    },
    /// Approval rules could not be loaded or saved.
    #[error("approval rules: {0}")]
    Rules(#[from] crate::approval::rules::RulesError),
    /// The approval engine could not be configured.
    #[error("approval engine: {0}")]
    Approval(#[from] crate::approval::ApprovalError),
    /// An MCP config file (`.rkat/mcp.toml`) could not be read or parsed.
    #[error("MCP config error: {0}")]
    McpConfig(#[from] meerkat_core::mcp_config::McpConfigError),
    /// No endpoint/model could be selected, or an endpoint is unusable.
    #[error("{0}")]
    Endpoint(String),
    /// A command-line `tui.toml` setting is invalid.
    #[error("invalid command-line setting: {0}")]
    Overlay(#[from] crate::overlay::OverlayError),
    /// MCP servers could not be started.
    #[error("MCP: {0}")]
    Mcp(String),
    /// The terminal UI failed.
    #[error("terminal UI: {0}")]
    Ui(#[from] crate::ui::UiError),
}

impl TuiError {
    /// Wrap an I/O error with a short description of what was being attempted.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Process exit code for this error.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::ProjectRoot { .. }
            | Self::Realm(_)
            | Self::Config { .. }
            | Self::Provider(_)
            | Self::MissingApiKey { .. }
            | Self::SessionRef { .. }
            | Self::Endpoint(_)
            | Self::Overlay(_)
            | Self::McpConfig(_) => 2,
            Self::Io { .. }
            | Self::Persistence(_)
            | Self::Store(_)
            | Self::LlmClient(_)
            | Self::Session(_)
            | Self::Rules(_)
            | Self::Approval(_)
            | Self::Mcp(_)
            | Self::Ui(_) => 1,
        }
    }
}
