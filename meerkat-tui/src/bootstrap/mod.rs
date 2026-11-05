//! Realm, config, LLM client, factory and session-service construction.
//!
//! Bootstrap is split in two so the approval engine can be built in between:
//!
//! 1. [`prepare`] resolves the [`RuntimeScope`] and loads the realm [`Config`].
//!    (For `--isolated` this generates a fresh realm id, so call it once.)
//! 2. The caller builds its hook engine, typically from
//!    [`layered_hooks_config`] plus its own entries.
//! 3. [`bootstrap_in`] opens persistence, builds the shared LLM client and the
//!    [`EphemeralSessionService`].
//!
//! [`bootstrap`] runs steps 1 and 3 back to back for callers without step 2.

pub mod agent_builder;
pub mod credentials;
pub mod dirs;
pub mod instructions;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod persona;
pub mod scope;

use std::path::PathBuf;
use std::sync::Arc;

use meerkat::{AgentFactory, EphemeralSessionService, LlmClient, SessionStore};
use meerkat_core::{CommsRuntimeMode, Config, HookEngine, HooksConfig, Provider};
use meerkat_store::{RealmBackend, RealmManifest, realm_paths_in};

use crate::cli::Args;
use crate::error::TuiError;

pub use agent_builder::{TuiAgent, TuiAgentBuilder};
pub use credentials::{
    ProviderCredentials, build_llm_client, resolve_model_and_provider, resolve_provider_credentials,
};
pub use dirs::GlobalDirs;
pub use persona::CODING_PERSONA;
pub use scope::{RuntimeScope, load_config, resolve_scope};

/// Maximum concurrently live sessions in one TUI process (the primary session
/// plus a few for `/resume` switching).
const MAX_SESSIONS: usize = 8;

/// The session service type the TUI drives.
pub type TuiSessionService = EphemeralSessionService<TuiAgentBuilder>;

/// Everything a running TUI needs from the engine.
pub struct Runtime {
    /// Project, conventions roots and realm.
    pub scope: RuntimeScope,
    /// Realm config with env overrides applied.
    pub config: Config,
    /// Model for new sessions.
    pub model: String,
    /// Provider of [`Runtime::model`] and [`Runtime::llm`].
    pub provider: Provider,
    /// The one LLM client shared by sessions and the approval classifier.
    pub llm: Arc<dyn LlmClient>,
    /// The factory the session service builds agents with (cloned).
    pub factory: AgentFactory,
    /// Session lifecycle service.
    pub service: Arc<TuiSessionService>,
    /// Realm session store (for listing and loading persisted sessions).
    pub store: Arc<dyn SessionStore>,
    /// Backend pinned in the realm manifest.
    pub backend: RealmBackend,
    /// Hook engine injected into every agent, if any.
    pub hook_engine: Option<Arc<dyn HookEngine>>,
    /// Directory of the semantic memory index (`<factory.store_path>/memory`)
    /// the factory opens for every agent. It is a single-writer redb file:
    /// only one live session may hold it at a time.
    pub memory_dir: PathBuf,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("scope", &self.scope)
            .field("model", &self.model)
            .field("provider", &self.provider)
            .field("backend", &self.backend)
            .field("hook_engine", &self.hook_engine.is_some())
            .field("memory_dir", &self.memory_dir)
            .finish_non_exhaustive()
    }
}

/// Resolve the runtime scope and load the realm config.
///
/// # Errors
///
/// See [`resolve_scope`] and [`load_config`].
pub async fn prepare(args: &Args) -> Result<(RuntimeScope, Config), TuiError> {
    let scope = resolve_scope(args)?;
    let config = load_config(&scope).await?;
    Ok((scope, config))
}

/// Hooks configured for this scope: user-level, then project-level `.rkat`
/// hook files layered over the realm config's `[hooks]`.
///
/// The returned config is what the factory would use on its own; because a
/// hook-engine override replaces that resolution, callers that build their own
/// engine should start from this and append their entries.
pub async fn layered_hooks_config(scope: &RuntimeScope, config: &Config) -> HooksConfig {
    meerkat::resolve_layered_hooks_config(
        Some(scope.context_root.as_path()),
        scope.user_config_root.as_deref(),
        config,
    )
    .await
}

/// [`prepare`] followed by [`bootstrap_in`].
///
/// # Errors
///
/// See [`prepare`] and [`bootstrap_in`].
pub async fn bootstrap(
    args: &Args,
    hook_engine: Option<Arc<dyn HookEngine>>,
    llm_override: Option<Arc<dyn LlmClient>>,
) -> Result<Runtime, TuiError> {
    let (scope, config) = prepare(args).await?;
    bootstrap_in(scope, config, args, hook_engine, llm_override).await
}

/// Open realm persistence, build the LLM client and the session service.
///
/// `llm_override` skips credential resolution entirely (tests, or a caller
/// that already built a client).
///
/// # Errors
///
/// Returns [`TuiError::Persistence`] when the realm cannot be opened,
/// [`TuiError::Provider`]/[`TuiError::MissingApiKey`]/[`TuiError::LlmClient`]
/// when no client can be built.
pub async fn bootstrap_in(
    scope: RuntimeScope,
    config: Config,
    args: &Args,
    hook_engine: Option<Arc<dyn HookEngine>>,
    llm_override: Option<Arc<dyn LlmClient>>,
) -> Result<Runtime, TuiError> {
    let mut config = config;
    force_inproc_comms(&mut config);
    let (model, provider) = resolve_model_and_provider(args, &config)?;

    let (manifest, persistence) = meerkat::open_realm_persistence_in(
        &scope.state_root,
        &scope.realm_id,
        scope.backend_hint,
        Some(scope.origin),
    )
    .await?;
    let store = persistence.session_store();
    let store_path = persistence
        .store_path()
        .map_or_else(|| realm_store_path(&manifest, &scope), PathBuf::from);

    let mut factory = AgentFactory::new(store_path)
        .session_store(Arc::clone(&store))
        .runtime_root(scope.realm_root())
        .project_root(scope.project_root.as_path())
        .context_root(scope.context_root.clone())
        .builtins(true)
        .shell(true)
        .subagents(cfg!(feature = "sub-agents"));
    if let Some(user_root) = &scope.user_config_root {
        factory = factory.user_config_root(user_root.clone());
    }

    let llm = match llm_override {
        Some(client) => client,
        None => {
            let creds = resolve_provider_credentials(&config, provider)?;
            build_llm_client(&factory, creds).await?
        }
    };

    let builder = TuiAgentBuilder {
        factory: factory.clone(),
        config: config.clone(),
        hook_engine: hook_engine.clone(),
        llm: Some(Arc::clone(&llm)),
    };
    let service = Arc::new(EphemeralSessionService::new(builder, MAX_SESSIONS));
    let memory_dir = factory.store_path.join("memory");

    Ok(Runtime {
        scope,
        config,
        model,
        provider,
        llm,
        factory,
        service,
        store,
        backend: manifest.backend,
        hook_engine,
        memory_dir,
    })
}

/// Comms exists only to deliver messages typed mid-turn into the agent's own
/// inbox: keep it in-process (no sockets, no key files) and never let it
/// spread to sub-agents, whatever the realm config says.
fn force_inproc_comms(config: &mut Config) {
    config.comms.mode = CommsRuntimeMode::Inproc;
    config.comms.auto_enable_for_subagents = false;
}

/// Store path for the factory, matching `rkat`'s `realm_store_path`.
fn realm_store_path(manifest: &RealmManifest, scope: &RuntimeScope) -> PathBuf {
    let paths = realm_paths_in(&scope.state_root, &scope.realm_id);
    match manifest.backend {
        RealmBackend::Jsonl => paths.sessions_jsonl_dir,
        RealmBackend::Sqlite | RealmBackend::Redb => paths.root,
    }
}
