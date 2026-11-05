//! A handle on one live session in the [`TuiSessionService`].
//!
//! Turns are serialized by the service: calling [`SessionHandle::start_turn`]
//! while a turn is running returns [`SessionError::Busy`]. The UI owns a prompt
//! queue and submits the next prompt once the running turn's result arrives;
//! it must not retry `Busy` in a loop.

pub mod delivery;
pub mod lifecycle;
pub mod resolve;

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use meerkat_core::error::AgentError;
use meerkat_core::event::{AgentEvent, EventEnvelope, ScopedAgentEvent, StreamScopeFrame};
use meerkat_core::service::{
    CreateSessionRequest, InitialTurnPolicy, SessionBuildOptions, SessionError, SessionService,
    StartTurnRequest,
};
use meerkat_core::types::{RunResult, SessionId};
use meerkat_core::{
    AgentToolDispatcher, EventInjectorError, InteractionContent, PendingSystemContextAppend,
    PlainEventSource, Provider, Session,
};
use tokio::sync::{mpsc, oneshot};

use crate::bootstrap::{CODING_PERSONA, Runtime, TuiSessionService};
use crate::delivery::InjectFailure;
use crate::error::TuiError;

pub use resolve::{list_sessions, load_session, resolve_session_ref};

/// Options for [`SessionHandle::create`].
#[derive(Default)]
pub struct SessionOptions {
    /// A persisted session to continue; `None` starts a fresh one.
    pub resume: Option<Session>,
    /// Extra tools (file tools, MCP) composed into the agent's dispatcher.
    pub external_tools: Option<Arc<dyn AgentToolDispatcher>>,
    /// Sink for attributed events from the primary agent and its sub-agents.
    pub scoped_event_tx: Option<mpsc::Sender<ScopedAgentEvent>>,
    /// Instruction sections appended after [`CODING_PERSONA`].
    pub additional_instructions: Vec<String>,
    /// Model to run on, overriding the resumed session's stored model (or the
    /// runtime's default for a fresh session). Must belong to the runtime's
    /// provider.
    pub model: Option<String>,
}

/// A live session plus the model it runs on.
#[derive(Clone)]
pub struct SessionHandle {
    service: Arc<TuiSessionService>,
    id: SessionId,
    model: String,
    provider: Provider,
}

impl std::fmt::Debug for SessionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionHandle")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl SessionHandle {
    /// Register a new (or resumed) session with the service without running a
    /// turn.
    ///
    /// [`SessionOptions::model`] wins when set. Otherwise, on resume, the
    /// session's stored model is kept when it belongs to the runtime's
    /// provider (the shared LLM client is provider-specific), and the
    /// runtime's model is used in every other case.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Session`] when the agent cannot be built.
    pub async fn create(runtime: &Runtime, opts: SessionOptions) -> Result<Self, TuiError> {
        let SessionOptions {
            resume,
            external_tools,
            scoped_event_tx,
            additional_instructions,
            model: model_override,
        } = opts;

        let session = resume.unwrap_or_default();
        let model = match (model_override, session.session_metadata()) {
            (Some(model), _) => model,
            (None, meta) => stored_or_default_model(runtime, meta),
        };
        let id = session.id().clone();

        let mut instructions = Vec::with_capacity(1 + additional_instructions.len());
        instructions.push(CODING_PERSONA.to_string());
        instructions.extend(additional_instructions);

        let build = SessionBuildOptions {
            provider: Some(runtime.provider),
            resume_session: Some(session),
            external_tools,
            scoped_event_tx,
            scoped_event_path: Some(vec![StreamScopeFrame::Primary {
                session_id: id.to_string(),
            }]),
            // Semantic memory: the factory opens the realm's HNSW index and
            // registers `memory_search`. The index is single-writer; callers
            // keep one live session (see `session::lifecycle`).
            override_memory: Some(true),
            override_mob: Some(false),
            additional_instructions: Some(instructions),
            realm_id: Some(runtime.scope.realm_id.clone()),
            backend: Some(runtime.backend.as_str().to_string()),
            // An in-process comms runtime gives the agent an inbox for
            // messages typed mid-turn. The name is unique per session; a
            // rebuild of the same session re-registers it, evicting the stale
            // entry.
            comms_name: Some(comms_name(&id)),
            ..SessionBuildOptions::default()
        };
        let req = CreateSessionRequest {
            model: model.clone(),
            prompt: String::new().into(),
            system_prompt: None,
            max_tokens: None,
            event_tx: None,
            host_mode: false,
            skill_references: None,
            initial_turn: InitialTurnPolicy::Defer,
            build: Some(build),
            labels: Some(BTreeMap::from([("surface".to_string(), "tui".to_string())])),
        };
        let created = runtime.service.create_session(req).await?;
        debug_assert_eq!(created.session_id, id);

        Ok(Self {
            service: Arc::clone(&runtime.service),
            id: created.session_id,
            model,
            provider: runtime.provider,
        })
    }

    /// The session id.
    #[must_use]
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// The model this session runs on.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The provider of [`SessionHandle::model`].
    #[must_use]
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Run one turn to completion, streaming events to `event_tx`.
    ///
    /// # Errors
    ///
    /// [`SessionError::Busy`] when a turn is already running (queue the prompt
    /// instead), [`SessionError::Agent`] with [`AgentError::Cancelled`] after
    /// [`SessionHandle::interrupt`], or any other agent failure.
    pub async fn start_turn(
        &self,
        prompt: String,
        event_tx: mpsc::Sender<EventEnvelope<AgentEvent>>,
    ) -> Result<RunResult, SessionError> {
        let req = StartTurnRequest {
            prompt: prompt.into(),
            event_tx: Some(event_tx),
            host_mode: false,
            skill_references: None,
            flow_tool_overlay: None,
            additional_instructions: None,
        };
        self.service.start_turn(&self.id, req).await
    }

    /// Spawn [`SessionHandle::start_turn`] on the tokio runtime so a UI loop
    /// can `select!` on the returned [`TurnTask`].
    #[must_use]
    pub fn spawn_turn(
        &self,
        prompt: String,
        event_tx: mpsc::Sender<EventEnvelope<AgentEvent>>,
    ) -> TurnTask {
        let (tx, rx) = oneshot::channel();
        let handle = self.clone();
        tokio::spawn(async move {
            let result = handle.start_turn(prompt, event_tx).await;
            // The receiver may have been dropped (UI quit); nothing to report then.
            let _ = tx.send(result);
        });
        TurnTask { rx }
    }

    /// Append `text` to the live session's system prompt as a
    /// `[Runtime System Context]` block with source `rkat-tui`.
    ///
    /// The block becomes part of the session's canonical system prompt, so
    /// every later LLM call of this session sees it. (The service's staged
    /// `SessionServiceControlExt::append_system_context` path is not used: on
    /// the ephemeral service it reaches only the next single LLM request.)
    ///
    /// The session task applies it between turns: while a turn is running
    /// this future waits for the turn to finish, so a UI loop should spawn it.
    /// `idempotency_key` is recorded on the block for provenance; repeated
    /// calls are not deduplicated.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotFound`] when the session is not live, or an internal
    /// error when the session task has exited.
    pub async fn append_system_context(
        &self,
        text: String,
        idempotency_key: String,
    ) -> Result<(), SessionError> {
        let append = PendingSystemContextAppend {
            text,
            source: Some("rkat-tui".to_string()),
            idempotency_key: Some(idempotency_key),
            accepted_at: meerkat_core::time_compat::SystemTime::now(),
        };
        self.service
            .apply_runtime_system_context(&self.id, vec![append])
            .await
    }

    /// Put `framed_body` (see [`crate::delivery::frame`]) into the agent's
    /// comms inbox. The agent picks it up at its next turn boundary and records
    /// it as a user message `[EVENT via stdin] <framed_body>`. Nothing wakes an
    /// idle agent, so only inject while a turn is running.
    ///
    /// # Errors
    ///
    /// [`InjectFailure::Unavailable`] when the session has no comms runtime,
    /// [`InjectFailure::Full`]/[`InjectFailure::Closed`] from the inbox.
    pub async fn inject(&self, framed_body: String) -> Result<(), InjectFailure> {
        // Fetched per call: a rebuilt session has a new runtime and inbox.
        let injector = self
            .service
            .event_injector(&self.id)
            .await
            .ok_or(InjectFailure::Unavailable)?;
        injector
            .inject(framed_body, PlainEventSource::Stdin)
            .map_err(|err| match err {
                EventInjectorError::Full => InjectFailure::Full,
                EventInjectorError::Closed => InjectFailure::Closed,
            })
    }

    /// Remove and return the bodies of plain events still waiting in the
    /// agent's inbox (they never reached the model). Empty when the session
    /// has no comms runtime.
    ///
    /// Other inbox items (peer messages) cannot occur in the TUI, which has no
    /// peers; any that do are dropped with a warning.
    pub async fn drain_inbox(&self) -> Vec<String> {
        let Some(runtime) = self.service.comms_runtime(&self.id).await else {
            return Vec::new();
        };
        runtime
            .drain_inbox_interactions()
            .await
            .into_iter()
            .filter_map(|interaction| match interaction.content {
                InteractionContent::Message { body, .. }
                    if interaction.from.starts_with("event:") =>
                {
                    Some(body)
                }
                other => {
                    tracing::warn!(from = %interaction.from, content = ?other, "dropping unexpected inbox item");
                    None
                }
            })
            .collect()
    }

    /// A snapshot of the live session. Waits for a running turn to finish.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotFound`] when the session is not live.
    pub async fn export(&self) -> Result<Session, SessionError> {
        self.service.export_session(&self.id).await
    }

    /// Cancel the running turn. The session stays usable.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotRunning`] when no turn is in flight.
    pub async fn interrupt(&self) -> Result<(), SessionError> {
        self.service.interrupt(&self.id).await
    }
}

/// The model a session runs on without an explicit override: its stored
/// model when that belongs to the runtime's provider (the shared LLM client is
/// provider-specific), otherwise the runtime's model.
fn stored_or_default_model(
    runtime: &Runtime,
    meta: Option<meerkat_core::SessionMetadata>,
) -> String {
    match meta {
        Some(meta) if meta.provider == runtime.provider => meta.model,
        Some(meta) => {
            tracing::warn!(
                stored_provider = meta.provider.as_str(),
                stored_model = %meta.model,
                active_provider = runtime.provider.as_str(),
                "resumed session used a different provider; continuing with {}",
                runtime.model
            );
            runtime.model.clone()
        }
        None => runtime.model.clone(),
    }
}

/// Inproc comms name of a session.
fn comms_name(id: &SessionId) -> String {
    format!("tui-{id}")
}

/// A turn running in the background. Resolves to the turn's result.
///
/// If the spawned task disappears without reporting (it panicked or the
/// runtime shut down), resolves to an internal [`SessionError::Agent`] error.
#[derive(Debug)]
pub struct TurnTask {
    rx: oneshot::Receiver<Result<RunResult, SessionError>>,
}

impl TurnTask {
    /// The raw receiver, for callers that prefer to handle `RecvError` themselves.
    #[must_use]
    pub fn into_receiver(self) -> oneshot::Receiver<Result<RunResult, SessionError>> {
        self.rx
    }
}

impl Future for TurnTask {
    type Output = Result<RunResult, SessionError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.rx).poll(cx).map(|received| {
            received.unwrap_or_else(|_| {
                Err(SessionError::Agent(AgentError::InternalError(
                    "turn task ended without a result".to_string(),
                )))
            })
        })
    }
}
