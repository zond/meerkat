//! Test seam: drive the integrator headless, without a terminal.
//!
//! [`spawn_integrator_for_tests`] wires a bootstrapped [`Runtime`] the way
//! [`super::run`] does (file and memory-note tools, instruction files, a
//! permissive approval engine without a classifier or human broker) and
//! hands back the channels the UI would own. Not a stable API.

use std::sync::Arc;

use meerkat::ToolGatewayBuilder;
use meerkat_core::{AgentEvent, AgentToolDispatcher, EventEnvelope, ScopedAgentEvent, SessionId};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub use super::quit::SavedUnsent;
use super::quit::UNSENT_FILE;
use super::{
    AGENT_EVENT_CAPACITY, ESCALATION_CAPACITY, Integrator, IntegratorParts, SessionTemplate,
    workspace_notes_dir,
};
use crate::approval::classifier::{Environment, UnavailableClassifier};
use crate::approval::escalation::EscalationBroker;
use crate::approval::rules::{AutoSettings, RuleSet};
use crate::approval::{ApprovalConfig, ApprovalEngine};
use crate::bootstrap::Runtime;
use crate::bootstrap::credentials::ProviderCredentials;
use crate::error::TuiError;
use crate::memory::{MemoryToolDispatcher, NoteStore, Scope};
use crate::session::SessionHandle;
use crate::tools::FileToolDispatcher;
use crate::ui::{Effect, Msg};

/// A running integrator and the UI's ends of its channels.
pub struct IntegratorHarness {
    /// Effects, as the UI would send them.
    pub effects: mpsc::UnboundedSender<Effect>,
    /// Integrator messages to the UI.
    pub ui_rx: mpsc::UnboundedReceiver<Msg>,
    /// Primary agent events.
    pub agent_rx: mpsc::Receiver<EventEnvelope<AgentEvent>>,
    /// Scoped (sub-agent) events.
    pub scoped_rx: mpsc::Receiver<ScopedAgentEvent>,
    /// Id of the first session.
    pub first_session: SessionId,
    /// Where quitting saves unsent messages.
    pub unsent_path: std::path::PathBuf,
    /// The integrator; resolves after [`Effect::Quit`] (or when `effects`
    /// is dropped) with where unsent messages were saved.
    pub task: JoinHandle<Option<SavedUnsent>>,
}

/// Spawn an integrator on `runtime` with a fresh first session and no
/// list-models endpoint (`/model` offers only the current model).
///
/// # Errors
///
/// When the approval engine or the first session cannot be built.
pub async fn spawn_integrator_for_tests(runtime: Runtime) -> Result<IntegratorHarness, TuiError> {
    spawn_integrator_with_model_list(runtime, None).await
}

/// Like [`spawn_integrator_for_tests`], fetching the model list with `creds`
/// (point `base_url` at a stub server).
///
/// # Errors
///
/// When the approval engine or the first session cannot be built.
pub async fn spawn_integrator_with_model_list(
    runtime: Runtime,
    creds: Option<ProviderCredentials>,
) -> Result<IntegratorHarness, TuiError> {
    let (approval_tx, approval_rx) = mpsc::unbounded_channel();
    let (broker, escalation_rx) = EscalationBroker::new(ESCALATION_CAPACITY);
    let root = runtime.scope.project_root.clone();
    let engine = ApprovalEngine::new(ApprovalConfig {
        environment: Environment::basic(root.as_path(), root.as_path()),
        root,
        rules: RuleSet::default(),
        settings: AutoSettings::default(),
        classifier: Arc::new(UnavailableClassifier),
        broker: Some(broker),
        posix_shell: cfg!(unix),
        events: approval_tx,
        rules_path: None,
    });

    let (diff_tx, diff_rx) = mpsc::unbounded_channel();
    let file_tools = Arc::new(FileToolDispatcher::new(
        runtime.scope.project_root.clone(),
        Some(diff_tx),
    ));
    let notes = Arc::new(MemoryToolDispatcher::new(
        None,
        NoteStore::open(
            workspace_notes_dir(&runtime.scope.context_root),
            Scope::Workspace,
        ),
    ));
    let gateway = ToolGatewayBuilder::new()
        .add_dispatcher(file_tools as Arc<dyn AgentToolDispatcher>)
        .add_dispatcher(Arc::clone(&notes) as Arc<dyn AgentToolDispatcher>)
        .build()
        .map_err(|e| TuiError::Mcp(format!("compose tool dispatchers: {e}")))?;
    let (scoped_tx, scoped_rx) = mpsc::channel(AGENT_EVENT_CAPACITY);
    let template = SessionTemplate {
        workspace: runtime.scope.context_root.clone(),
        global: runtime.scope.global.clone(),
        notes,
        external_tools: Arc::new(gateway),
        scoped_tx,
    };
    let session = SessionHandle::create(&runtime, template.options(None, None)).await?;
    let first_session = session.id().clone();
    let unsent_path = runtime.scope.realm_root().join(UNSENT_FILE);

    let (agent_tx, agent_rx) = mpsc::channel(AGENT_EVENT_CAPACITY);
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    let (effects, effect_rx) = mpsc::unbounded_channel();
    let integrator = Integrator::new(IntegratorParts {
        runtime,
        engine,
        session,
        template,
        agent_tx,
        ui_tx,
        effect_rx,
        escalation_rx,
        approval_rx,
        diff_rx,
        last_context: None,
        creds,
    });
    Ok(IntegratorHarness {
        effects,
        ui_rx,
        agent_rx,
        scoped_rx,
        first_session,
        unsent_path,
        task: tokio::spawn(integrator.run()),
    })
}
