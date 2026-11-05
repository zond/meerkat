//! A handle on one live session in the [`TuiSessionService`].
//!
//! Turns are serialized by the service: calling [`SessionHandle::start_turn`]
//! while a turn is running returns [`SessionError::Busy`]. The UI owns a prompt
//! queue and submits the next prompt once the running turn's result arrives;
//! it must not retry `Busy` in a loop.

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
use meerkat_core::{AgentToolDispatcher, Provider, Session};
use tokio::sync::{mpsc, oneshot};

use crate::bootstrap::{CODING_PERSONA, Runtime, TuiSessionService};
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
    /// On resume, the session's stored model is kept when it belongs to the
    /// runtime's provider (the shared LLM client is provider-specific);
    /// otherwise the runtime's model is used.
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
        } = opts;

        let session = resume.unwrap_or_default();
        let model = match session.session_metadata() {
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
            override_memory: Some(false),
            override_mob: Some(false),
            additional_instructions: Some(instructions),
            realm_id: Some(runtime.scope.realm_id.clone()),
            backend: Some(runtime.backend.as_str().to_string()),
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

    /// Cancel the running turn. The session stays usable.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotRunning`] when no turn is in flight.
    pub async fn interrupt(&self) -> Result<(), SessionError> {
        self.service.interrupt(&self.id).await
    }
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
