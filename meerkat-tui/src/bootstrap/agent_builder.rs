//! A `SessionAgentBuilder` that injects the TUI's hook engine and shared LLM
//! client into every agent the session service builds.
//!
//! `meerkat::FactoryAgentBuilder` has no way to pass a hook engine through
//! `SessionBuildOptions`, and `FactoryAgent` cannot be constructed outside the
//! facade, so this module carries its own builder and agent newtype.
//!
//! NOTE: [`TuiAgent`]'s `SessionAgent` impl and [`TuiAgentBuilder::build_agent`]
//! are copies of `FactoryAgent` / `FactoryAgentBuilder` in
//! `meerkat/src/service_factory.rs` (line ~70 onwards). Any upstream change to
//! that delegation must be mirrored here.

use std::sync::Arc;

use async_trait::async_trait;
use meerkat::{AgentBuildConfig, AgentFactory, DynAgent, LlmClient};
use meerkat::{SessionAgent, SessionAgentBuilder};
use meerkat_core::error::AgentError;
use meerkat_core::event::AgentEvent;
use meerkat_core::service::{CreateSessionRequest, SessionError, TurnToolOverlay};
use meerkat_core::types::{ContentInput, RunResult, SessionId};
use meerkat_core::{Config, HookEngine, Session};
use meerkat_session::SessionSnapshot;
use tokio::sync::mpsc;

/// Builds [`TuiAgent`]s through [`AgentFactory::build_agent`] with the TUI's
/// hook engine and (optionally) a shared LLM client injected.
pub struct TuiAgentBuilder {
    /// Factory carrying roots, store and tool flags.
    pub factory: AgentFactory,
    /// Config snapshot used for every build.
    pub config: Config,
    /// Hook engine replacing filesystem hook resolution for every agent.
    /// `None` keeps the factory's default (layered `.rkat` hooks).
    pub hook_engine: Option<Arc<dyn HookEngine>>,
    /// LLM client used when the request carries no override.
    pub llm: Option<Arc<dyn LlmClient>>,
}

#[async_trait]
impl SessionAgentBuilder for TuiAgentBuilder {
    type Agent = TuiAgent;

    async fn build_agent(
        &self,
        req: &CreateSessionRequest,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> Result<TuiAgent, SessionError> {
        let mut build_config = AgentBuildConfig::from_create_session_request(req, event_tx);
        build_config
            .hook_engine_override
            .clone_from(&self.hook_engine);
        if build_config.llm_client_override.is_none() {
            build_config.llm_client_override.clone_from(&self.llm);
        }
        let agent = self
            .factory
            .build_agent(build_config, &self.config)
            .await
            .map_err(|e| SessionError::Agent(AgentError::InternalError(e.to_string())))?;
        Ok(TuiAgent(agent))
    }
}

/// [`DynAgent`] newtype implementing [`SessionAgent`].
pub struct TuiAgent(pub DynAgent);

impl TuiAgent {
    /// The current session.
    #[must_use]
    pub fn session(&self) -> &Session {
        self.0.session()
    }
}

#[async_trait]
impl SessionAgent for TuiAgent {
    async fn run_with_events(
        &mut self,
        prompt: ContentInput,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> Result<RunResult, AgentError> {
        self.0.run_with_events(prompt, event_tx).await
    }

    async fn run_host_mode(&mut self, prompt: ContentInput) -> Result<RunResult, AgentError> {
        self.0.run_host_mode(prompt).await
    }

    fn set_skill_references(&mut self, refs: Option<Vec<meerkat_core::skills::SkillKey>>) {
        self.0.pending_skill_references = refs;
    }

    fn set_flow_tool_overlay(
        &mut self,
        overlay: Option<TurnToolOverlay>,
    ) -> Result<(), AgentError> {
        self.0
            .set_flow_tool_overlay(overlay)
            .map_err(|error| AgentError::ConfigError(error.to_string()))
    }

    fn replace_client(&mut self, client: Arc<dyn meerkat_core::AgentLlmClient>) {
        self.0.replace_client(client);
    }

    fn stage_external_tool_filter(
        &mut self,
        filter: meerkat_core::ToolFilter,
    ) -> Result<(), AgentError> {
        self.0
            .stage_external_tool_filter(filter)
            .map(|_| ())
            .map_err(|error| AgentError::ConfigError(error.to_string()))
    }

    fn cancel(&mut self) {
        self.0.cancel();
    }

    fn session_id(&self) -> SessionId {
        self.0.session().id().clone()
    }

    fn snapshot(&self) -> SessionSnapshot {
        let s = self.0.session();
        SessionSnapshot {
            created_at: s.created_at(),
            updated_at: s.updated_at(),
            message_count: s.messages().len(),
            total_tokens: s.total_tokens(),
            usage: s.total_usage(),
            last_assistant_text: s.last_assistant_text(),
        }
    }

    fn session_clone(&self) -> Session {
        self.0.session_with_system_context_state()
    }

    fn apply_runtime_system_context(
        &mut self,
        appends: &[meerkat_core::PendingSystemContextAppend],
    ) {
        self.0.session_mut().append_system_context_blocks(appends);
    }

    fn system_context_state(
        &self,
    ) -> Arc<std::sync::Mutex<meerkat_core::SessionSystemContextState>> {
        self.0.system_context_state()
    }

    fn event_injector(&self) -> Option<Arc<dyn meerkat_core::EventInjector>> {
        self.0.comms_arc()?.event_injector()
    }

    #[doc(hidden)]
    fn interaction_event_injector(
        &self,
    ) -> Option<Arc<dyn meerkat_core::event_injector::SubscribableInjector>> {
        self.0.comms_arc()?.interaction_event_injector()
    }

    fn comms_runtime(&self) -> Option<Arc<dyn meerkat_core::agent::CommsRuntime>> {
        self.0.comms_arc()
    }
}
