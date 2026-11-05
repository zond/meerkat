//! Integrator: wires bootstrap, approval, file tools, session and UI together.
//!
//! The UI loop ([`crate::ui::run_ui`]) owns the terminal and emits
//! [`Effect`]s; this module turns them into session-service calls and feeds
//! agent events, approval decisions, escalations and diffs back as
//! [`Msg`]s. Nothing here touches the terminal directly.

mod cost;
mod lifecycle;
mod models;
mod quit;
#[doc(hidden)]
pub mod testing;

use crate::approval::classifier::{
    ActionClassifier, Environment, LlmClassifier, UnavailableClassifier, classifier_default_model,
};
use crate::approval::counters::{AutoModeState, PauseCause};
use crate::approval::escalation::{EscalationBroker, EscalationRequest, HumanDecision};
use crate::approval::policy::VerdictSummary;
use crate::approval::rules::{self, AutoSettings, Rule, RuleId, RuleKind, RuleOrigin, RuleSet};
use crate::approval::{ApprovalConfig, ApprovalEngine, ApprovalEvent};
use crate::bootstrap::{self, Runtime, StartModel};
use crate::cli::Args;
use crate::delivery::{InjectFailure, PendingId, frame};
use crate::endpoints::{
    EndpointRef, EndpointRegistry, StartupPlan, StartupRequest, load_choice, needs_lookup,
    plan_startup, restore_choice,
};
use crate::error::TuiError;
use crate::memory::{MemoryToolDispatcher, NoteStore, Scope};
use crate::models::list_endpoint_models;
use crate::overlay::ConfigOverlay;
use crate::pricing::{PriceTable, PricingSetup};
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
use meerkat::{AgentFactory, ToolGatewayBuilder};
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
    // Command-line settings are validated before anything else happens.
    let overlay = ConfigOverlay::from_args(&args)?;
    let cli_layer = overlay.rules_file()?;
    let (scope, mut config) = bootstrap::prepare(&args).await?;
    init_logging(&scope.realm_root())?;
    // Before `bootstrap_with`, which hands copies of the config to the factory.
    bootstrap::instructions::apply_system_md(&mut config, scope.global.as_ref());

    // tui.toml layers (global → realm → command line → project).
    let realm_dir = scope.realm_root();
    let global_rules = scope.global.as_ref().map(bootstrap::GlobalDirs::rules_file);
    // The realm rules file lives inside the workspace. If it is (or may be)
    // tracked by git it arrived with the repository, so only its deny rules apply.
    let realm_rules = realm_dir.join(RULES_FILE);
    let mut tracking_notice = None;
    let realm_trust = match rules_file_is_tracked(&scope.context_root, &realm_rules).await {
        Ok(false) => rules::RealmTrust::Owned,
        Ok(true) => rules::RealmTrust::Tracked,
        Err(reason) => {
            tracking_notice = Some(rules::Notice::RealmTrackingUnknown {
                path: realm_rules.clone(),
                reason,
            });
            rules::RealmTrust::Tracked
        }
    };
    let rules::Layered {
        rules: rule_set,
        settings,
        endpoints: endpoint_sections,
        pricing: price_overrides,
        mut notices,
    } = rules::load_layered_with(
        &realm_dir,
        global_rules.as_deref(),
        &scope.context_root.join(".rkat"),
        realm_trust,
        cli_layer.as_ref(),
    )?;
    notices.extend(tracking_notice);
    if let Some(notice) = rules::legacy_global_rules_notice(
        scope.user_config_root.as_deref(),
        global_rules.as_deref(),
    ) {
        notices.push(notice);
    }

    // Endpoints and the start-up model. The start-up endpoint's client is
    // shared by its sessions and the approval classifier.
    let (registry, endpoint_notices) = EndpointRegistry::discover(&config, &endpoint_sections);
    notices.extend(endpoint_notices);
    let startup = resolve_startup(&args, &config, &registry).await?;
    let mut pricing = PricingSetup::load(
        price_overrides,
        scope
            .global
            .as_ref()
            .map(bootstrap::GlobalDirs::pricing_cache),
        !args.no_pricing_fetch,
    );
    pricing.table.set_endpoint_prices(startup.prices.clone());
    let start_endpoint = registry.get(&startup.choice.endpoint).ok_or_else(|| {
        TuiError::Endpoint(format!("unknown endpoint `{}`", startup.choice.endpoint))
    })?;
    let provider: Provider = start_endpoint.dialect.into();
    let llm = start_endpoint
        .build_client(&AgentFactory::new(scope.realm_root()))
        .await?;

    let (approval_tx, approval_rx) = mpsc::unbounded_channel();
    let (broker, escalation_rx) = EscalationBroker::new(ESCALATION_CAPACITY);
    let cwd = std::env::current_dir().map_err(|e| TuiError::io("read current directory", e))?;
    let environment = Environment::detect(scope.project_root.as_path(), &cwd).await;
    let compaction_policy = settings.compaction_policy(None);
    let classifier = make_classifier(
        &settings,
        Arc::clone(&llm),
        provider,
        scope.project_root.clone(),
    );
    let engine = ApprovalEngine::new(ApprovalConfig {
        root: scope.project_root.clone(),
        rules: rule_set,
        settings,
        classifier,
        broker: Some(broker),
        environment,
        posix_shell: bootstrap::shell_is_posix(),
        events: approval_tx,
        rules_path: Some(realm_dir.join(RULES_FILE)),
    });
    let base_hooks = bootstrap::layered_hooks_config(&scope, &config).await;
    let hook_engine = engine.build_hook_engine(base_hooks)?;

    let runtime = bootstrap::bootstrap_with(
        scope,
        config,
        Some(hook_engine),
        StartModel {
            endpoint: startup.choice.endpoint.clone(),
            provider,
            model: startup.choice.model.clone(),
            llm: Arc::clone(&llm),
        },
    )
    .await?;

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

    // A resumed session continues on its own endpoint and model unless the
    // command line picked one.
    let mut clients: HashMap<String, Arc<dyn LlmClient>> =
        HashMap::from([(startup.choice.endpoint.clone(), Arc::clone(&llm))]);
    let explicit = args.model.is_some() || args.endpoint.is_some() || args.provider.is_some();
    let (choice, mut choice_notice) = match &resumed {
        Some(session) if !explicit => restore_choice(
            load_choice(&runtime.scope.realm_root(), &session.id().to_string()),
            session
                .session_metadata()
                .map(|meta| (meta.provider, meta.model)),
            registry.as_slice(),
            |name| registry.get(name).is_some(),
            &startup.choice,
        ),
        _ => (startup.choice.clone(), None),
    };
    let (choice, choice_llm) = match clients.get(&choice.endpoint) {
        Some(client) => (choice, Arc::clone(client)),
        None => match registry.get(&choice.endpoint) {
            Some(endpoint) => match endpoint.build_client(&runtime.factory).await {
                Ok(client) => {
                    clients.insert(choice.endpoint.clone(), Arc::clone(&client));
                    (choice, client)
                }
                Err(err) => {
                    choice_notice = Some(format!(
                        "could not use {choice} ({err}); continuing on {}",
                        startup.choice
                    ));
                    (startup.choice.clone(), Arc::clone(&llm))
                }
            },
            None => (startup.choice.clone(), Arc::clone(&llm)),
        },
    };
    let target = SessionTarget {
        provider: registry
            .get(&choice.endpoint)
            .map_or(provider, |e| e.dialect.into()),
        llm: choice_llm,
        choice,
    };
    let resumed_cost = resumed
        .as_ref()
        .map(|s| cost::session_cost(&runtime.scope.realm_root(), s, &target.choice.model));
    let session_endpoint = target.choice.endpoint.clone();
    let session = SessionHandle::create(&runtime, template.options(resumed, Some(target))).await?;
    engine.set_primary_session(session.id());
    let current_model = session.model().to_string();

    // UI state and channels.
    let mut status = StatusInfo::default();
    status.set_model(&session_endpoint, &current_model);
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
        .with_compaction_policy(compaction_policy)
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
    for text in startup.notice.into_iter().chain(choice_notice) {
        let _ = ui_tx.send(Msg::Notice {
            level: NoticeLevel::Info,
            text,
        });
    }
    let _ = ui_tx.send(Msg::Status(window_status(
        session.provider(),
        &current_model,
    )));
    if let Some(text) = quit::unsent_notice(&runtime.scope.realm_root()) {
        let _ = ui_tx.send(Msg::Notice {
            level: NoticeLevel::Warn,
            text,
        });
    }
    if let Some(used) = resumed_context {
        let _ = ui_tx.send(Msg::Status(StatusUpdate::Context {
            used: Some(used),
            approx: true,
        }));
    }
    if let Some(blocks) = resumed_blocks {
        let _ = ui_tx.send(Msg::ResetTranscript(blocks));
    }
    if let Some(cost) = resumed_cost {
        let _ = ui_tx.send(Msg::Status(StatusUpdate::CostTotals(cost)));
    }
    if let Some(prompt) = args.prompt.clone() {
        let _ = effect_tx.send(Effect::StartTurn(prompt));
    }

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
        last_context: resumed_context,
        endpoints: registry,
        current_endpoint: session_endpoint,
        clients,
        models: startup.models,
        pricing,
    });
    let integrator_task = tokio::spawn(integrator.run());

    let ui_result = run_ui(
        app,
        UiChannels {
            agent_rx,
            scoped_rx,
            ui_rx,
        },
        effect_tx,
        args.mouse_capture(),
        histories,
    )
    .await;
    // The UI has gone (and closed the effect channel): the integrator stops
    // the turn, persists the session and saves unsent messages, then exits.
    match tokio::time::timeout(quit::SHUTDOWN_TIMEOUT, integrator_task).await {
        Ok(Ok(Some(saved))) => eprintln!(
            "rkat-tui: {} unsent message(s) saved in {}",
            saved.count,
            saved.path.display()
        ),
        Ok(Ok(None)) => {}
        Ok(Err(err)) => tracing::warn!(error = %err, "integrator task failed"),
        Err(_) => tracing::warn!("integrator did not shut down in time"),
    }
    ui_result.map_err(TuiError::from)
}

/// The start-up endpoint and model, the model lists when they had to be
/// fetched to decide, and a notice when the model was picked automatically.
struct Startup {
    choice: EndpointRef,
    models: Option<models::ModelCache>,
    /// Prices the fetched lists reported (LiteLLM `/model/info`).
    prices: models::EndpointPrices,
    notice: Option<String>,
}

/// Decide the start-up endpoint and model (see [`plan_startup`]). A bare
/// `--model` id with several endpoints is looked up in their lists, fetched
/// now (10 s cap, in parallel) and kept as the picker cache.
async fn resolve_startup(
    args: &Args,
    config: &meerkat_core::Config,
    registry: &EndpointRegistry,
) -> Result<Startup, TuiError> {
    let endpoint = args.endpoint.as_deref().or(args.provider.as_deref());
    let request = StartupRequest {
        model: args.model.as_deref(),
        endpoint,
        config_model: &config.agent.model,
    };
    let (models, prices) = match request.model {
        Some(model) if endpoint.is_none() && needs_lookup(model, registry.as_slice()) => {
            let (cache, prices) = models::fetch_cache(registry).await;
            (Some(cache), prices)
        }
        _ => (None, Vec::new()),
    };
    let index = models::index_of(models.as_ref());
    match plan_startup(request, registry.as_slice(), &index)? {
        StartupPlan::Resolved(choice) => Ok(Startup {
            choice,
            models,
            prices,
            notice: None,
        }),
        StartupPlan::FirstListed(name) => {
            let endpoint = registry
                .get(&name)
                .ok_or_else(|| TuiError::Endpoint(format!("unknown endpoint `{name}`")))?;
            let listed = list_endpoint_models(endpoint, models::LIST_MODELS_TIMEOUT)
                .await
                .map_err(|err| {
                    TuiError::Endpoint(format!(
                        "no model selected and listing the models of `{name}` failed ({err}); pass --model"
                    ))
                })?;
            let first = listed.first().ok_or_else(|| {
                TuiError::Endpoint(format!(
                    "no model selected and `{name}` lists no models; pass --model"
                ))
            })?;
            let choice = EndpointRef::new(name, first.id.clone());
            Ok(Startup {
                notice: Some(format!(
                    "no model configured; using {choice}, the first model it lists"
                )),
                choice,
                models,
                prices,
            })
        }
    }
}

/// Status update for the context window of `model`.
fn window_status(provider: Provider, model: &str) -> StatusUpdate {
    match context_window(provider, model) {
        Some(window) if window.approx => StatusUpdate::ApproxContextWindow(window.tokens),
        Some(window) => StatusUpdate::ContextWindow(Some(window.tokens)),
        None => StatusUpdate::ContextWindow(None),
    }
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
///
/// `Ok(false)` only when that is known: the file does not exist, lies outside
/// the workspace, the workspace is not a git repository, or git reports it as
/// untracked (exit code 1). Anything else (git missing, other failures) is an
/// `Err` with the reason, which callers treat as tracked (fail closed).
async fn rules_file_is_tracked(
    workspace: &std::path::Path,
    path: &std::path::Path,
) -> Result<bool, String> {
    if !path.is_file() {
        return Ok(false);
    }
    let (Ok(workspace), Ok(path)) = (workspace.canonicalize(), path.canonicalize()) else {
        return Err("cannot resolve the rules file or workspace path".to_string());
    };
    if !path.starts_with(&workspace) {
        return Ok(false);
    }
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(&workspace)
        .args(["ls-files", "--error-unmatch", "--"])
        .arg(&path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("cannot run git: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ if stderr.contains("not a git repository") => Ok(false),
        code => Err(format!(
            "git ls-files exited with {}: {}",
            code.map_or_else(|| "a signal".to_string(), |c| c.to_string()),
            stderr.trim()
        )),
    }
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
    project_root: crate::paths::ProjectRoot,
) -> Arc<dyn ActionClassifier> {
    let model = settings
        .classifier_model
        .clone()
        .or_else(|| classifier_default_model(provider).map(str::to_string));
    match model {
        // Stage 2 may read project files (read-only, bounded) to judge what a
        // command will actually execute.
        Some(model) => Arc::new(LlmClassifier::new(llm, provider, model).with_project_tools(
            Arc::new(crate::tools::ReadOnlyProjectTools::new(project_root)),
        )),
        None => Arc::new(UnavailableClassifier),
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
        collapsed: true,
    }
}

/// An escalation the UI has not answered yet.
struct PendingEscalation {
    respond: oneshot::Sender<HumanDecision>,
    tool: String,
    /// Canonical rule subject "allow + add rule" turns into a literal rule.
    subject: Option<String>,
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

/// Where a session runs: endpoint and model, the endpoint's dialect and client.
struct SessionTarget {
    choice: EndpointRef,
    provider: Provider,
    llm: Arc<dyn LlmClient>,
}

impl SessionTemplate {
    /// Options for a session build. Instruction files and the notes index are
    /// re-read every time, so `/clear`, `/resume` and `/compact` reload them.
    /// Without a `target` the runtime's endpoint and model apply.
    fn options(&self, resume: Option<Session>, target: Option<SessionTarget>) -> SessionOptions {
        let (model, provider, llm, endpoint) = match target {
            Some(SessionTarget {
                choice,
                provider,
                llm,
            }) => (
                Some(choice.model),
                Some(provider),
                Some(llm),
                Some(choice.endpoint),
            ),
            None => (None, None, None, None),
        };
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
            provider,
            llm,
            endpoint,
        }
    }
}

/// The already-built pieces an [`Integrator`] drives.
struct IntegratorParts {
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
    last_context: Option<u64>,
    /// Configured endpoints (empty disables listing: tests).
    endpoints: EndpointRegistry,
    /// Endpoint of `session`.
    current_endpoint: String,
    /// Clients already built, by endpoint name (at least `current_endpoint`).
    clients: HashMap<String, Arc<dyn LlmClient>>,
    /// Model lists fetched before start-up, if any (else fetched in the background).
    models: Option<models::ModelCache>,
    /// Start-up prices and how to refresh the community list.
    pricing: PricingSetup,
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
    /// The endpoint sessions run on (start-up endpoint, then `/model`); kept
    /// across `/clear` and `/compact` rebuilds. The approval classifier
    /// keeps the start-up endpoint's client.
    current_endpoint: String,
    /// Dialect of `current_endpoint`.
    current_provider: Provider,
    /// The model sessions run on (start-up model, then `/model`); kept across
    /// `/clear` and `/compact` rebuilds.
    current_model: String,
    /// Estimated context of the current session after its last turn, for
    /// `CompactionStarted`.
    last_context: Option<u64>,
    /// No live session: a `/compact` rebuild and the restore both failed.
    /// Turns are refused until `/resume` or `/clear` opens one.
    dead: bool,
    /// Messages the UI still held when the user quit ([`Effect::SaveUnsent`]).
    held_unsent: Vec<(Option<PendingId>, String)>,
    /// Configured endpoints.
    endpoints: EndpointRegistry,
    /// Clients by endpoint name, built on first use.
    clients: HashMap<String, Arc<dyn LlmClient>>,
    /// Every endpoint's model list, cached after the first fetch.
    models: Option<models::ModelCache>,
    /// A fetch of the lists in flight.
    model_fetch: Option<models::ModelFetch>,
    /// `/model` is waiting for [`Integrator::model_fetch`] to open the picker.
    picker_waiting: bool,
    /// `/model <bare id>` waiting for the lists to resolve it.
    pending_set: Option<String>,
    /// Prices for the cost readout (shared with the UI).
    prices: Arc<PriceTable>,
    /// A community price-list refresh in flight.
    price_fetch: Option<cost::PriceFetch>,
}

/// A turn in flight and the session it runs on (which stays the one to
/// reconcile against even if the UI switches sessions meanwhile).
struct RunningTurn {
    task: TurnTask,
    session: SessionHandle,
}

impl Integrator {
    fn new(parts: IntegratorParts) -> Self {
        let IntegratorParts {
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
            last_context,
            endpoints,
            current_endpoint,
            clients,
            models,
            pricing,
        } = parts;
        let current_model = session.model().to_string();
        let current_provider = session.provider();
        // Fetch the model lists right away so the context gauge uses the
        // endpoint's window from the first turn.
        let model_fetch = (models.is_none() && !endpoints.is_empty())
            .then(|| models::spawn_model_fetch(&endpoints));
        let price_fetch = pricing
            .refresh_url()
            .map(|url| cost::spawn_price_fetch(url.to_string(), pricing.cache_path.clone()));
        let integrator = Self {
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
            current_endpoint,
            current_provider,
            current_model,
            last_context,
            dead: false,
            held_unsent: Vec::new(),
            endpoints,
            clients,
            models,
            model_fetch,
            picker_waiting: false,
            pending_set: None,
            prices: Arc::new(pricing.table),
            price_fetch,
        };
        integrator.save_current_choice();
        integrator.push_prices();
        if integrator.models.is_some() {
            integrator.push_context_window();
        }
        integrator
    }

    /// Serve effects until the UI quits (or drops the effect channel), then
    /// shut down: see [`Integrator::shutdown`]. Returns where unsent messages
    /// were saved, if any.
    async fn run(mut self) -> Option<quit::SavedUnsent> {
        loop {
            tokio::select! {
                effect = self.effect_rx.recv() => {
                    match effect {
                        None | Some(Effect::Quit) => break,
                        Some(effect) => self.handle_effect(effect).await,
                    }
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
                joined = async { match self.model_fetch.as_mut() { Some(fetch) => fetch.await, None => std::future::pending().await } } => {
                    self.model_fetch_finished(joined).await;
                }
                joined = async { match self.price_fetch.as_mut() { Some(fetch) => fetch.await, None => std::future::pending().await } } => {
                    self.price_fetch_finished(joined);
                }
            }
        }
        self.shutdown().await
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
            Some(turn) => {
                let before = self.inflight.clone();
                let recalled = recall(&turn.session, &mut self.inflight).await;
                // A recalled message never reached the model, so it must not
                // count as the user's intent for the approval classifier either.
                for (id, framed) in &before {
                    if recalled.contains(id) {
                        let text = framed
                            .strip_prefix(crate::delivery::FRAME)
                            .unwrap_or(framed);
                        self.engine.forget_prompt(turn.session.id(), text);
                    }
                }
                recalled
            }
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
            Effect::SaveUnsent(held) => self.held_unsent.extend(held),
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
            // `set_model` waits for the session task, which serves commands
            // only between turns: never await it while a turn runs.
            Effect::SetModel(_) if self.turn.is_some() || !self.inflight.is_empty() => {
                self.notice(
                    NoticeLevel::Warn,
                    "/model is not available while a turn is running",
                );
            }
            Effect::SetModel(model) => self.set_model(model).await,
            Effect::RequestModelPicker { refresh } => self.request_model_picker(refresh),
            Effect::Remember(text) => self.remember(&text),
            Effect::Compact => self.compact().await,
            Effect::CostSnapshot(snapshot) => self.save_cost_snapshot(&snapshot),
        }
    }

    fn start_turn(&mut self, prompt: String) {
        if self.turn.is_some() {
            self.notice(NoticeLevel::Warn, "a turn is already running");
            return;
        }
        if self.dead {
            // The UI is waiting for this turn: finish it right away.
            self.send(Msg::TurnFinished(TurnOutcome {
                result: Err("no live session; use /resume last".to_string()),
                delivery: crate::delivery::DeliveryReport::default(),
            }));
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
                subject: request.subjects.into_iter().next(),
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
                // A literal rule over the canonical subject: matches exactly
                // this command / path, no glob semantics to escape.
                let rule = match pending.subject {
                    Some(subject) => {
                        Rule::literal(RuleKind::Allow, &pending.tool, subject, RuleOrigin::Realm)
                            .map_err(|e| e.to_string())
                    }
                    None => Err("the action has no rule subject".to_string()),
                };
                match rule {
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
            .map(|(n, rule)| rule_list_line(n, rule))
            .collect();
        if lines.is_empty() {
            self.notice(NoticeLevel::Info, "no rules");
        } else {
            self.notice(NoticeLevel::Info, lines.join("\n"));
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

/// One `/rules list` line: number, rule and the layer it came from.
fn rule_list_line(n: usize, rule: &Rule) -> String {
    format!("{n}. {} ({})", describe_rule(rule), rule.origin.as_str())
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
    fn rules_list_shows_the_origin() {
        let rule = Rule::new(RuleKind::Allow, "shell", Some("ls*"), RuleOrigin::Cli).unwrap();
        assert_eq!(rule_list_line(3, &rule), "3. allow shell ls* (cli)");
        let rule = Rule::new(RuleKind::Deny, "shell", None, RuleOrigin::Project).unwrap();
        assert_eq!(rule_list_line(1, &rule), "1. deny shell (project)");
    }

    #[test]
    fn humanize_buckets() {
        assert_eq!(humanize(std::time::Duration::from_secs(5)), "5s");
        assert_eq!(humanize(std::time::Duration::from_secs(120)), "2m");
        assert_eq!(humanize(std::time::Duration::from_secs(7200)), "2h");
        assert_eq!(humanize(std::time::Duration::from_secs(200_000)), "2d");
    }
}
