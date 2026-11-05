//! Integrator: wires bootstrap, approval, file tools, session and UI together.
//!
//! The UI loop ([`crate::ui::run_ui`]) owns the terminal and emits
//! [`Effect`]s; this module turns them into session-service calls and feeds
//! agent events, approval decisions, escalations and diffs back as
//! [`Msg`]s. Nothing here touches the terminal directly.

mod lifecycle;

use crate::approval::classifier::{
    ActionClassifier, Environment, LlmClassifier, UnavailableClassifier, classifier_default_model,
};
use crate::approval::counters::{AutoModeState, PauseCause};
use crate::approval::escalation::{EscalationBroker, EscalationRequest, HumanDecision};
use crate::approval::policy::VerdictSummary;
use crate::approval::rules::{self, AutoSettings, Rule, RuleId, RuleKind, RuleOrigin, RuleSet};
use crate::approval::{ApprovalConfig, ApprovalEngine, ApprovalEvent};
use crate::bootstrap::credentials::{
    build_llm_client, resolve_model_and_provider, resolve_provider_credentials,
};
use crate::bootstrap::{self, Runtime};
use crate::cli::Args;
use crate::delivery::{InjectFailure, PendingId, frame};
use crate::error::TuiError;
use crate::memory::{MemoryToolDispatcher, NoteStore, Scope};
use crate::session::delivery::{recall, reconcile, user_texts};
use crate::session::lifecycle::{
    context_window, estimate_tokens, probe_memory, session_instructions,
};
use crate::session::{
    SessionHandle, SessionOptions, TurnTask, list_sessions, load_session, resolve_session_ref,
};
use crate::tools::{FileDiff, FileToolDispatcher};
use crate::ui::{
    App, ApprovalNotice, Block, Effect, EscalationAnswer, EscalationId, EscalationPrompt, History,
    Msg, NoticeLevel, StatusInfo, StatusUpdate, ToolState, TurnOutcome, TurnSummary, UiChannels,
    run_ui,
};
use meerkat::{AgentFactory, SessionService, ToolGatewayBuilder};
use meerkat_client::LlmClient;
use meerkat_core::AgentToolDispatcher;
use meerkat_core::service::SessionError;
use meerkat_core::{
    AgentError, AgentEvent, AssistantBlock, EventEnvelope, Message, Provider, ScopedAgentEvent,
    Session, SessionMeta,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Capacity of the primary agent-event channel.
const AGENT_EVENT_CAPACITY: usize = 256;
/// Capacity of the escalation channel.
const ESCALATION_CAPACITY: usize = 16;
/// Name of the rules file inside the realm directory.
const RULES_FILE: &str = "tui.toml";

/// Run the TUI end to end.
///
/// # Errors
///
/// Returns a [`TuiError`] when bootstrap fails, the session cannot be created,
/// or the terminal cannot be driven.
pub async fn run(args: Args) -> Result<(), TuiError> {
    let (scope, mut config) = bootstrap::prepare(&args).await?;
    init_logging(&scope.realm_root())?;
    // Before `bootstrap_in`, which hands copies of the config to the factory.
    bootstrap::instructions::apply_system_md(&mut config, scope.global.as_ref());

    // Shared LLM client: the session and the classifier use the same credentials.
    let (model, provider) = resolve_model_and_provider(&args, &config)?;
    let creds = resolve_provider_credentials(&config, provider)?;
    let llm = build_llm_client(&AgentFactory::new(scope.realm_root()), creds).await?;

    // Approval engine, built before the agent so its hooks are in the engine.
    let realm_dir = scope.realm_root();
    let global_rules = scope.global.as_ref().map(bootstrap::GlobalDirs::rules_file);
    let (mut rule_set, mut settings, mut notices) = rules::load_layered(
        &realm_dir,
        global_rules.as_deref(),
        &scope.context_root.join(".rkat"),
    )?;
    if let Some(notice) = rules::legacy_global_rules_notice(
        scope.user_config_root.as_deref(),
        global_rules.as_deref(),
    ) {
        notices.push(notice);
    }
    // The realm rules file lives inside the workspace. If it is tracked by git it
    // arrived with the repository, so its allow rules are not the user's.
    let realm_rules = realm_dir.join(RULES_FILE);
    if rules_file_is_tracked(&scope.context_root, &realm_rules).await {
        let dropped = rule_set.allow.len();
        rule_set
            .allow
            .retain(|rule| rule.origin != RuleOrigin::Realm);
        let dropped = dropped - rule_set.allow.len();
        if dropped > 0 {
            notices.push(rules::Notice::TrackedRealmAllowIgnored {
                path: realm_rules,
                count: dropped,
            });
        }
    }
    if args.no_auto {
        settings.enabled = false;
    }
    if let Some(m) = &args.classifier_model {
        settings.classifier_model = Some(m.clone());
    }
    let (approval_tx, approval_rx) = mpsc::unbounded_channel();
    let (broker, escalation_rx) = EscalationBroker::new(ESCALATION_CAPACITY);
    let cwd = std::env::current_dir().map_err(|e| TuiError::io("read current directory", e))?;
    let environment = Environment::detect(scope.project_root.as_path(), &cwd).await;
    let classifier = make_classifier(&settings, Arc::clone(&llm), provider);
    let engine = ApprovalEngine::new(ApprovalConfig {
        root: scope.project_root.clone(),
        rules: rule_set,
        settings,
        classifier,
        broker: Some(broker),
        environment,
        posix_shell: cfg!(unix),
        events: approval_tx,
        rules_path: Some(realm_dir.join(RULES_FILE)),
    });
    let base_hooks = bootstrap::layered_hooks_config(&scope, &config).await;
    let hook_engine = engine.build_hook_engine(base_hooks)?;

    let runtime =
        bootstrap::bootstrap_in(scope, config, &args, Some(hook_engine), Some(llm)).await?;

    // File tools and the first session.
    let (diff_tx, diff_rx) = mpsc::unbounded_channel();
    let file_tools: Arc<FileToolDispatcher> = Arc::new(FileToolDispatcher::new(
        runtime.scope.project_root.clone(),
        Some(diff_tx),
    ));
    let (scoped_tx, scoped_rx) = mpsc::channel(AGENT_EVENT_CAPACITY);
    let notes = Arc::new(MemoryToolDispatcher::new(
        runtime
            .scope
            .global
            .as_ref()
            .map(|global| NoteStore::open(global.notes_dir(), Scope::User)),
        NoteStore::open(
            workspace_notes_dir(&runtime.scope.context_root),
            Scope::Workspace,
        ),
    ));
    let (external_tools, mcp_notice) =
        compose_external_tools(&runtime, &args, &file_tools, &notes).await?;
    // The semantic index is single-writer: another rkat-tui on the same realm
    // keeps this one's agents from opening it (they then lack `memory_search`).
    let memory_notice = probe_memory(&runtime.memory_dir).await.err().map(|err| {
        tracing::warn!(error = %err, dir = %runtime.memory_dir.display(), "semantic memory probe failed");
        "semantic memory unavailable: another rkat-tui holds the index".to_string()
    });
    let resume_ref = if args.continue_last {
        Some("last")
    } else {
        args.resume.as_deref()
    };
    let resumed = match resume_ref {
        Some(reference) => {
            let id = resolve_session_ref(&*runtime.store, reference).await?;
            Some(load_session(&*runtime.store, &id).await?)
        }
        None => None,
    };
    let resumed_blocks = resumed.as_ref().map(blocks_from_session);
    let resumed_context = resumed.as_ref().map(|s| estimate_tokens(s.messages()));
    if let Some(session) = &resumed {
        engine.seed_transcript(session.id(), session);
    }
    let template = SessionTemplate {
        workspace: runtime.scope.context_root.clone(),
        global: runtime.scope.global.clone(),
        notes,
        external_tools,
        scoped_tx: scoped_tx.clone(),
    };
    let session = SessionHandle::create(&runtime, template.options(resumed, None)).await?;
    let current_model = session.model().to_string();

    // UI state and channels.
    let mut status = StatusInfo::default();
    status.set_model(provider_name(provider), &model);
    status.set_session(&session.id().to_string());
    let (auto_state, denials) = auto_status(&engine);
    status.set_auto(auto_state, denials);
    let history = History::load(runtime.scope.realm_root().join("tui_history"));
    let global_history = runtime
        .scope
        .global
        .as_ref()
        .map(|global| History::load(global.history_file()));
    let app = App::new(status)
        .with_history(history.entries().to_vec())
        .with_global_history(
            global_history
                .as_ref()
                .map(|h| h.entries().to_vec())
                .unwrap_or_default(),
        );
    let histories: Vec<History> = std::iter::once(history).chain(global_history).collect();
    let (agent_tx, agent_rx) = mpsc::channel(AGENT_EVENT_CAPACITY);
    let (ui_tx, ui_rx) = mpsc::unbounded_channel::<Msg>();
    let (effect_tx, effect_rx) = mpsc::unbounded_channel::<Effect>();
    for notice in notices {
        let _ = ui_tx.send(Msg::Notice {
            level: NoticeLevel::Warn,
            text: notice.to_string(),
        });
    }
    if let Some(text) = mcp_notice {
        let _ = ui_tx.send(Msg::Notice {
            level: NoticeLevel::Info,
            text,
        });
    }
    if let Some(text) = memory_notice {
        let _ = ui_tx.send(Msg::Notice {
            level: NoticeLevel::Warn,
            text,
        });
    }
    let _ = ui_tx.send(Msg::Status(StatusUpdate::ContextWindow(context_window(
        provider,
        &current_model,
    ))));
    if let Some(used) = resumed_context {
        let _ = ui_tx.send(Msg::Status(StatusUpdate::Context {
            used: Some(used),
            approx: true,
        }));
    }
    if let Some(blocks) = resumed_blocks {
        let _ = ui_tx.send(Msg::ResetTranscript(blocks));
    }
    if let Some(prompt) = args.prompt.clone() {
        let _ = effect_tx.send(Effect::StartTurn(prompt));
    }

    let integrator = Integrator {
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
        pending: HashMap::new(),
        next_escalation: 0,
        turn: None,
        inflight: Vec::new(),
        current_model,
        last_context: resumed_context,
    };
    let integrator_task = tokio::spawn(integrator.run());

    let ui_result = run_ui(
        app,
        UiChannels {
            agent_rx,
            scoped_rx,
            ui_rx,
        },
        effect_tx,
        args.mouse,
        histories,
    )
    .await;
    integrator_task.abort();
    ui_result.map_err(TuiError::from)
}

/// Directory of the workspace's memory notes.
fn workspace_notes_dir(workspace: &Path) -> PathBuf {
    workspace.join(".rkat").join("tui").join("notes")
}

/// The session's external tools: the in-crate file and memory-note tools plus
/// any configured MCP servers, composed behind one gateway. Returns a notice
/// describing the MCP servers when any are configured.
async fn compose_external_tools(
    runtime: &Runtime,
    args: &Args,
    file_tools: &Arc<FileToolDispatcher>,
    notes: &Arc<MemoryToolDispatcher>,
) -> Result<(Arc<dyn AgentToolDispatcher>, Option<String>), TuiError> {
    #[cfg_attr(not(feature = "mcp"), allow(unused_mut))]
    let mut builder = ToolGatewayBuilder::new()
        .add_dispatcher(Arc::clone(file_tools) as Arc<dyn AgentToolDispatcher>)
        .add_dispatcher(Arc::clone(notes) as Arc<dyn AgentToolDispatcher>);
    #[cfg_attr(not(feature = "mcp"), allow(unused_mut))]
    let mut notice = None;
    #[cfg(feature = "mcp")]
    {
        use crate::bootstrap::mcp::load_mcp_tools;
        if let Some(mcp) = load_mcp_tools(&runtime.scope, args.wait_for_mcp).await? {
            builder = builder.add_dispatcher(Arc::clone(&mcp.dispatcher));
            notice = Some(format!(
                "MCP servers: {} ({})",
                mcp.server_names.join(", "),
                if args.wait_for_mcp {
                    "connected"
                } else {
                    "connecting in the background"
                }
            ));
        }
    }
    #[cfg(not(feature = "mcp"))]
    {
        let _ = (runtime, args);
    }
    let gateway = builder
        .build()
        .map_err(|e| TuiError::Mcp(format!("compose tool dispatchers: {e}")))?;
    Ok((Arc::new(gateway), notice))
}

/// Whether `path` is tracked by the git repository at `workspace`.
/// Best effort: no git, no repository, or an untracked file all yield `false`.
async fn rules_file_is_tracked(workspace: &std::path::Path, path: &std::path::Path) -> bool {
    if !path.is_file() {
        return false;
    }
    tokio::process::Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["ls-files", "--error-unmatch", "--"])
        .arg(path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success())
}

/// Route tracing to a log file inside the realm so nothing writes to the
/// terminal while the UI owns it.
fn init_logging(realm_root: &std::path::Path) -> Result<(), TuiError> {
    let log_dir = realm_root.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| TuiError::io("create log directory", e))?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("rkat-tui.log"))
        .map_err(|e| TuiError::io("open log file", e))?;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    // A second init (tests) is not an error worth failing the run for.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_writer(std::sync::Mutex::new(file))
        .try_init();
    Ok(())
}

/// The classifier for Tier-3 actions: the two-stage LLM classifier on the
/// session's provider, or an always-escalating stub when no model is known.
fn make_classifier(
    settings: &AutoSettings,
    llm: Arc<dyn LlmClient>,
    provider: Provider,
) -> Arc<dyn ActionClassifier> {
    let model = settings
        .classifier_model
        .clone()
        .or_else(|| classifier_default_model(provider).map(str::to_string));
    match model {
        Some(model) => Arc::new(LlmClassifier::new(llm, provider, model)),
        None => Arc::new(UnavailableClassifier),
    }
}

fn provider_name(provider: Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "anthropic",
        Provider::OpenAI => "openai",
        Provider::Gemini => "gemini",
        Provider::Other => "other",
    }
}

/// Status-bar form of the auto-mode state.
fn auto_status(engine: &ApprovalEngine) -> (String, (u32, u32, u32, u32)) {
    use crate::approval::counters::{MAX_CONSECUTIVE_DENIALS, MAX_TOTAL_DENIALS};
    let (counters, state) = engine.auto_state();
    let label = match state {
        AutoModeState::Active => "active".to_string(),
        AutoModeState::Paused {
            cause: PauseCause::UserDisabled,
        } => "off".to_string(),
        AutoModeState::Paused { .. } => "PAUSED".to_string(),
    };
    (
        label,
        (
            counters.consecutive,
            MAX_CONSECUTIVE_DENIALS,
            counters.total,
            MAX_TOTAL_DENIALS,
        ),
    )
}

/// Transcript blocks for a resumed session: user text, assistant text and
/// tool calls. Tool results are summarized as completed calls.
fn blocks_from_session(session: &Session) -> Vec<Block> {
    let mut blocks = Vec::new();
    for message in session.messages() {
        match message {
            Message::User(user) => {
                blocks.extend(
                    user_texts(user)
                        .into_iter()
                        .map(|text| Block::User { text }),
                );
            }
            Message::Assistant(assistant) => {
                if !assistant.content.is_empty() {
                    blocks.push(Block::Assistant {
                        md: assistant.content.clone(),
                    });
                }
                for call in &assistant.tool_calls {
                    blocks.push(restored_tool_block(&call.name, call.args.clone()));
                }
            }
            Message::BlockAssistant(assistant) => {
                for block in &assistant.blocks {
                    match block {
                        AssistantBlock::Text { text, .. } if !text.is_empty() => {
                            blocks.push(Block::Assistant { md: text.clone() });
                        }
                        _ => {}
                    }
                }
                for call in assistant.tool_calls() {
                    let args = serde_json::from_str(call.args.get())
                        .unwrap_or_else(|_| serde_json::Value::String(call.args.get().to_string()));
                    blocks.push(restored_tool_block(call.name, args));
                }
            }
            _ => {}
        }
    }
    blocks
}

fn restored_tool_block(name: &str, args: serde_json::Value) -> Block {
    Block::Tool {
        name: name.to_string(),
        args,
        state: ToolState::Done {
            is_error: false,
            duration_ms: 0,
        },
        result_preview: None,
        diff: None,
        depth: 0,
    }
}

/// Escape glob metacharacters so a literal string matches only itself.
fn glob_escape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '*' | '?' | '[' | ']' | '{' | '}' => {
                out.push('[');
                out.push(ch);
                out.push(']');
            }
            other => out.push(other),
        }
    }
    out
}

/// An escalation the UI has not answered yet.
struct PendingEscalation {
    respond: oneshot::Sender<HumanDecision>,
    tool: String,
    args_preview: String,
}

/// What every session build shares: where instructions and notes come from,
/// the external tools and the scoped-event sink.
struct SessionTemplate {
    workspace: PathBuf,
    global: Option<bootstrap::GlobalDirs>,
    notes: Arc<MemoryToolDispatcher>,
    external_tools: Arc<dyn AgentToolDispatcher>,
    scoped_tx: mpsc::Sender<ScopedAgentEvent>,
}

impl SessionTemplate {
    /// Options for a session build. Instruction files and the notes index are
    /// re-read every time, so `/clear`, `/resume` and `/compact` reload them.
    fn options(&self, resume: Option<Session>, model: Option<String>) -> SessionOptions {
        SessionOptions {
            resume,
            external_tools: Some(Arc::clone(&self.external_tools)),
            scoped_event_tx: Some(self.scoped_tx.clone()),
            additional_instructions: session_instructions(
                &self.workspace,
                self.global.as_ref(),
                Some(&self.notes),
            ),
            model,
        }
    }
}

struct Integrator {
    runtime: Runtime,
    engine: Arc<ApprovalEngine>,
    session: SessionHandle,
    template: SessionTemplate,
    agent_tx: mpsc::Sender<EventEnvelope<AgentEvent>>,
    ui_tx: mpsc::UnboundedSender<Msg>,
    effect_rx: mpsc::UnboundedReceiver<Effect>,
    escalation_rx: mpsc::Receiver<EscalationRequest>,
    approval_rx: mpsc::UnboundedReceiver<ApprovalEvent>,
    diff_rx: mpsc::UnboundedReceiver<FileDiff>,
    pending: HashMap<u64, PendingEscalation>,
    next_escalation: u64,
    turn: Option<RunningTurn>,
    /// Messages injected into the running turn, `(id, framed body)` in
    /// injection order; reconciled when the turn ends.
    inflight: Vec<(PendingId, String)>,
    /// The model sessions run on (start-up model, then `/model`); kept across
    /// `/clear` and `/compact` rebuilds.
    current_model: String,
    /// Estimated context of the current session after its last turn, for
    /// `CompactionStarted`.
    last_context: Option<u64>,
}

/// A turn in flight and the session it runs on (which stays the one to
/// reconcile against even if the UI switches sessions meanwhile).
struct RunningTurn {
    task: TurnTask,
    session: SessionHandle,
}

impl Integrator {
    async fn run(mut self) {
        loop {
            tokio::select! {
                effect = self.effect_rx.recv() => {
                    let Some(effect) = effect else { break };
                    if matches!(effect, Effect::Quit) {
                        self.stop_pending();
                        break;
                    }
                    self.handle_effect(effect).await;
                }
                Some(request) = self.escalation_rx.recv() => self.on_escalation(request),
                Some(event) = self.approval_rx.recv() => self.on_approval_event(event),
                Some(diff) = self.diff_rx.recv() => self.send(Msg::Diff {
                    tool_use_id: diff.tool_use_id,
                    path: diff.path.display().to_string(),
                    unified: diff.unified,
                    added: diff.added,
                    removed: diff.removed,
                }),
                result = async { match self.turn.as_mut() { Some(turn) => (&mut turn.task).await, None => std::future::pending().await } } => {
                    self.turn_finished(result).await;
                }
            }
        }
    }

    /// Branch order: persist on error, reconcile injected messages, report.
    async fn turn_finished(&mut self, result: Result<meerkat_core::RunResult, SessionError>) {
        let Some(RunningTurn { session, .. }) = self.turn.take() else {
            return;
        };
        if result.is_err() {
            self.persist_session(&session).await;
        }
        let delivery = reconcile(&session, &mut self.inflight).await;
        if session.id() == self.session.id() {
            match session.export().await {
                Ok(snapshot) => self.last_context = Some(estimate_tokens(snapshot.messages())),
                Err(err) => {
                    tracing::debug!(error = %err, "could not export session for the context estimate");
                }
            }
        }
        self.send(Msg::TurnFinished(TurnOutcome {
            result: turn_summary(result),
            delivery,
        }));
    }

    /// Answer [`Effect::Inject`]: frame the text and put it in the running
    /// turn's inbox.
    async fn inject(&mut self, id: PendingId, text: &str) {
        let result = match &self.turn {
            None => Err(InjectFailure::NotRunning),
            Some(turn) => {
                let framed = frame(text);
                let result = turn.session.inject(framed.clone()).await;
                if result.is_ok() {
                    // The classifier must see mid-turn instructions as user intent.
                    self.engine.record_prompt(turn.session.id(), text);
                    self.inflight.push((id, framed));
                }
                result
            }
        };
        self.send(Msg::InjectAck { id, result });
    }

    /// Answer [`Effect::RecallPending`].
    async fn recall_pending(&mut self) {
        let recalled = match &self.turn {
            Some(turn) => recall(&turn.session, &mut self.inflight).await,
            None => Vec::new(),
        };
        self.send(Msg::Recalled(recalled));
    }

    fn send(&self, msg: Msg) {
        let _ = self.ui_tx.send(msg);
    }

    /// Save the live session to the store. The engine only persists a session
    /// when a run completes, so after a failed or interrupted turn the user's
    /// prompt would otherwise be lost to `--continue`.
    async fn persist_session(&self, session: &SessionHandle) {
        match session.export().await {
            Ok(session) => {
                if let Err(err) = self.runtime.store.save(&session).await {
                    tracing::warn!(error = %err, "could not persist session after failed turn");
                }
            }
            Err(err) => tracing::warn!(error = %err, "could not export session for persistence"),
        }
    }

    fn notice(&self, level: NoticeLevel, text: impl Into<String>) {
        self.send(Msg::Notice {
            level,
            text: text.into(),
        });
    }

    fn push_auto_status(&self) {
        let (state, denials) = auto_status(&self.engine);
        self.send(Msg::Status(StatusUpdate::Auto { state, denials }));
    }

    /// Deny every pending escalation as `Stop`; used on quit.
    fn stop_pending(&mut self) {
        for (_, pending) in self.pending.drain() {
            let _ = pending.respond.send(HumanDecision::Stop);
        }
    }

    async fn handle_effect(&mut self, effect: Effect) {
        match effect {
            Effect::StartTurn(prompt) => self.start_turn(prompt),
            Effect::Inject { id, text } => self.inject(id, &text).await,
            Effect::RecallPending => self.recall_pending().await,
            // The UI keeps the text of `InterruptAndSend` and starts the next
            // turn itself once `TurnFinished` arrives.
            Effect::Interrupt | Effect::InterruptAndSend(_) => {
                self.stop_pending();
                if let Err(err) = self.session.interrupt().await
                    && !matches!(err, SessionError::NotRunning { .. })
                {
                    self.notice(NoticeLevel::Warn, format!("interrupt failed: {err}"));
                }
            }
            Effect::Quit => {}
            Effect::NewSession => self.new_session().await,
            Effect::ResumeSession(reference) => self.resume_session(&reference).await,
            Effect::RequestResumePicker => self.show_resume_picker().await,
            Effect::RespondEscalation(id, answer) => self.respond_escalation(id, answer),
            Effect::AddRule {
                effect,
                tool,
                pattern,
            } => {
                let kind = match effect {
                    crate::ui::RuleEffect::Allow => RuleKind::Allow,
                    crate::ui::RuleEffect::Deny => RuleKind::Deny,
                };
                self.add_rule(kind, &tool, pattern.as_deref());
            }
            Effect::RemoveRule(n) => self.remove_rule(n),
            Effect::ListRules => self.list_rules(),
            Effect::SetAutoMode(enabled) => {
                self.engine.set_enabled(enabled);
                self.push_auto_status();
                self.notice(
                    NoticeLevel::Info,
                    if enabled {
                        "auto mode on"
                    } else {
                        "auto mode off: tier-3 actions are denied"
                    },
                );
            }
            Effect::ResumeAutoMode => {
                self.engine.resume_auto();
                self.push_auto_status();
                self.notice(NoticeLevel::Info, "auto mode resumed");
            }
            Effect::SetModel(model) => self.set_model(model).await,
            Effect::Remember(text) => self.remember(&text),
            Effect::Compact => self.compact().await,
        }
    }

    fn start_turn(&mut self, prompt: String) {
        if self.turn.is_some() {
            self.notice(NoticeLevel::Warn, "a turn is already running");
            return;
        }
        self.turn = Some(RunningTurn {
            task: self.session.spawn_turn(prompt, self.agent_tx.clone()),
            session: self.session.clone(),
        });
    }

    async fn show_resume_picker(&self) {
        match list_sessions(&*self.runtime.store).await {
            Ok(sessions) => {
                let items = sessions.iter().map(picker_item).collect();
                self.send(Msg::ShowResumePicker(items));
            }
            Err(err) => self.notice(NoticeLevel::Error, err.to_string()),
        }
    }

    fn on_escalation(&mut self, request: EscalationRequest) {
        let id = self.next_escalation;
        self.next_escalation += 1;
        let prompt = EscalationPrompt {
            tool: request.tool.clone(),
            args_preview: request.args_preview.clone(),
            hint: request.classifier_hint,
        };
        self.pending.insert(
            id,
            PendingEscalation {
                respond: request.respond,
                tool: request.tool,
                args_preview: request.args_preview,
            },
        );
        self.send(Msg::Escalation(prompt, EscalationId(id)));
    }

    fn respond_escalation(&mut self, id: EscalationId, answer: EscalationAnswer) {
        let Some(pending) = self.pending.remove(&id.0) else {
            return;
        };
        let decision = match answer {
            EscalationAnswer::AllowOnce => HumanDecision::AllowOnce,
            EscalationAnswer::Deny => HumanDecision::Deny,
            EscalationAnswer::Stop => HumanDecision::Stop,
            EscalationAnswer::AllowAndAddRule => {
                let pattern = glob_escape(&pending.args_preview);
                match Rule::new(
                    RuleKind::Allow,
                    &pending.tool,
                    Some(&pattern),
                    RuleOrigin::Realm,
                ) {
                    Ok(rule) => HumanDecision::AllowAndAddRule(Box::new(rule)),
                    Err(err) => {
                        self.notice(NoticeLevel::Warn, format!("rule not added: {err}"));
                        HumanDecision::AllowOnce
                    }
                }
            }
        };
        let _ = pending.respond.send(decision);
        self.send(Msg::EscalationResolved(id));
    }

    fn on_approval_event(&self, event: ApprovalEvent) {
        match event {
            ApprovalEvent::Decision {
                tool,
                tier,
                verdict,
                ..
            } => {
                let (allowed, reason) = match verdict {
                    VerdictSummary::Allowed { .. } => (true, None),
                    VerdictSummary::Denied { by, .. } => (false, Some(format!("denied by {by:?}"))),
                };
                // Tier-1/2 allows are routine; only reviewed actions are worth a line.
                if !allowed || tier == crate::approval::policy::Tier::Classifier {
                    self.send(Msg::Approval(ApprovalNotice {
                        tool,
                        tier: tier.to_string(),
                        allowed,
                        reason,
                    }));
                }
                self.push_auto_status();
            }
            ApprovalEvent::Paused(cause) => {
                self.push_auto_status();
                self.notice(NoticeLevel::Warn, format!("auto mode paused: {cause}"));
            }
            ApprovalEvent::Resumed => {
                self.push_auto_status();
            }
            ApprovalEvent::RulesDropped(rules) => {
                let listed: Vec<String> = rules.iter().map(describe_rule).collect();
                self.notice(
                    NoticeLevel::Warn,
                    format!(
                        "auto mode dropped {} broad allow rule(s): {}",
                        listed.len(),
                        listed.join(", ")
                    ),
                );
            }
        }
    }

    fn add_rule(&self, kind: RuleKind, tool: &str, pattern: Option<&str>) {
        match Rule::new(kind, tool, pattern, RuleOrigin::Realm)
            .map_err(TuiError::from)
            .and_then(|rule| self.engine.add_rule(rule).map_err(TuiError::from))
        {
            Ok(()) => self.notice(NoticeLevel::Info, "rule added"),
            Err(err) => self.notice(NoticeLevel::Error, format!("rule not added: {err}")),
        }
    }

    fn remove_rule(&self, n: usize) {
        let Some(id) = nth_rule_id(&self.engine.rules(), n) else {
            self.notice(NoticeLevel::Error, format!("no rule #{n}"));
            return;
        };
        match self.engine.remove_rule(id) {
            Ok(Some(rule)) => self.notice(
                NoticeLevel::Info,
                format!("removed {}", describe_rule(&rule)),
            ),
            Ok(None) => self.notice(NoticeLevel::Error, format!("no rule #{n}")),
            Err(err) => self.notice(NoticeLevel::Error, format!("rule not removed: {err}")),
        }
    }

    fn list_rules(&self) {
        let rules = self.engine.rules();
        let lines: Vec<String> = numbered_rules(&rules)
            .map(|(n, rule)| format!("{n}. {}", describe_rule(rule)))
            .collect();
        if lines.is_empty() {
            self.notice(NoticeLevel::Info, "no rules");
        } else {
            self.notice(NoticeLevel::Info, lines.join("\n"));
        }
    }

    async fn set_model(&mut self, model: String) {
        if Provider::infer_from_model(&model) != Some(self.session.provider()) {
            self.notice(
                NoticeLevel::Warn,
                format!(
                    "{model} is not a {} model; restart with --model to switch providers",
                    provider_name(self.session.provider())
                ),
            );
            return;
        }
        let adapter = self
            .runtime
            .factory
            .build_llm_adapter(Arc::clone(&self.runtime.llm), model.clone())
            .await;
        match self
            .runtime
            .service
            .set_session_client(self.session.id(), Arc::new(adapter))
            .await
        {
            Ok(()) => {
                self.send(Msg::Status(StatusUpdate::Model {
                    provider: provider_name(self.session.provider()).to_string(),
                    model: model.clone(),
                }));
                self.current_model = model;
                self.push_context_window();
            }
            Err(err) => self.notice(NoticeLevel::Error, format!("model not changed: {err}")),
        }
    }
}

fn turn_summary(
    result: Result<meerkat_core::RunResult, SessionError>,
) -> Result<TurnSummary, String> {
    match result {
        Ok(run) => Ok(TurnSummary {
            text: run.text,
            input_tokens: run.usage.input_tokens,
            output_tokens: run.usage.output_tokens,
        }),
        Err(SessionError::Agent(AgentError::Cancelled)) => Err("interrupted".to_string()),
        Err(err) => Err(err.to_string()),
    }
}

fn picker_item(meta: &SessionMeta) -> (String, String) {
    let age = meta
        .updated_at
        .elapsed()
        .map_or_else(|_| "just now".to_string(), humanize);
    let model = meta
        .metadata
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    (
        meta.id.to_string(),
        format!(
            "{} · {} ago · {} msgs · {model}",
            short_id(&meta.id.to_string()),
            age,
            meta.message_count
        ),
    )
}

fn short_id(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}

fn humanize(elapsed: std::time::Duration) -> String {
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86_400)
    }
}

/// Rules in display order: allow rules first, then deny rules, 1-based.
fn numbered_rules(rules: &RuleSet) -> impl Iterator<Item = (usize, &Rule)> {
    rules
        .allow
        .iter()
        .chain(rules.deny.iter())
        .enumerate()
        .map(|(i, r)| (i + 1, r))
}

fn nth_rule_id(rules: &RuleSet, n: usize) -> Option<RuleId> {
    let index = n.checked_sub(1)?;
    if index < rules.allow.len() {
        Some(RuleId {
            kind: RuleKind::Allow,
            index,
        })
    } else {
        let index = index - rules.allow.len();
        (index < rules.deny.len()).then_some(RuleId {
            kind: RuleKind::Deny,
            index,
        })
    }
}

fn describe_rule(rule: &Rule) -> String {
    let kind = match rule.kind {
        RuleKind::Allow => "allow",
        RuleKind::Deny => "deny",
    };
    match &rule.arg {
        Some(arg) => format!("{kind} {} {}", rule.tool.as_str(), arg.as_str()),
        None => format!("{kind} {}", rule.tool.as_str()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn glob_escape_neutralizes_metacharacters() {
        assert_eq!(glob_escape("cargo test *"), "cargo test [*]");
        assert_eq!(glob_escape("a?b[c]{d}"), "a[?]b[[]c[]][{]d[}]");
        assert_eq!(glob_escape("plain"), "plain");
    }

    #[test]
    fn nth_rule_id_spans_allow_then_deny() {
        let mut rules = RuleSet::default();
        rules
            .allow
            .push(Rule::new(RuleKind::Allow, "shell", Some("ls*"), RuleOrigin::Realm).unwrap());
        rules
            .deny
            .push(Rule::new(RuleKind::Deny, "shell", Some("rm*"), RuleOrigin::Realm).unwrap());
        assert_eq!(nth_rule_id(&rules, 0), None);
        assert_eq!(
            nth_rule_id(&rules, 1).map(|r| r.kind),
            Some(RuleKind::Allow)
        );
        assert_eq!(
            nth_rule_id(&rules, 2).map(|r| (r.kind, r.index)),
            Some((RuleKind::Deny, 0))
        );
        assert_eq!(nth_rule_id(&rules, 3), None);
    }

    #[test]
    fn humanize_buckets() {
        assert_eq!(humanize(std::time::Duration::from_secs(5)), "5s");
        assert_eq!(humanize(std::time::Duration::from_secs(120)), "2m");
        assert_eq!(humanize(std::time::Duration::from_secs(7200)), "2h");
        assert_eq!(humanize(std::time::Duration::from_secs(200_000)), "2d");
    }
}
