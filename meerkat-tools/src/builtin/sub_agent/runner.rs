//! Sub-agent runner - Actually executes sub-agents with comms support
//!
//! This module handles the full lifecycle of sub-agent execution:
//! - Creating LLM client adapters
//! - Setting up comms infrastructure for parent-child communication
//! - Running the agent loop in a background task
//! - Reporting results back to the parent

use meerkat_client::{LlmClient, LlmClientAdapter};
#[cfg(feature = "comms")]
use meerkat_comms::agent::wrap_with_comms;
#[cfg(feature = "comms")]
use meerkat_comms::runtime::{CommsBootstrap, CommsRuntime, CoreCommsConfig, ParentCommsContext};
#[cfg(feature = "comms")]
use meerkat_comms::{PubKey, TrustedPeer, TrustedPeers};
use meerkat_core::config::HookRunOverrides;
use meerkat_core::ops::{OperationId, OperationResult};
use meerkat_core::session::Session;
use meerkat_core::sub_agent::SubAgentManager;
use meerkat_core::types::{Message, SystemMessage, UserMessage};
use meerkat_core::{
    AgentBuilder, AgentEvent, AgentLlmClient, AgentSessionStore, AgentToolDispatcher, BudgetLimits,
    HookEngine, MeteredLlmClient, ScopedAgentEvent, StreamScopeFrame, UsageObserver,
};
#[cfg(feature = "comms")]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// Configuration for spawning a sub-agent with comms
#[cfg(feature = "comms")]
#[derive(Debug, Clone)]
pub struct SubAgentCommsConfig {
    /// Name for this sub-agent
    pub name: String,
    /// Base directory for comms identity files
    pub base_dir: PathBuf,
    /// Parent's context for establishing trust
    pub parent_context: ParentCommsContext,
}

/// Handle to a running sub-agent
#[derive(Debug)]
pub struct SubAgentHandle {
    /// Operation ID
    pub id: OperationId,
    /// Name of the sub-agent
    pub name: String,
    /// Child's public key (for parent to send messages)
    pub child_pubkey: [u8; 32],
    /// Address to reach the child
    pub child_addr: String,
}

/// Set up mutual trust between parent and child
///
/// Creates a TrustedPeers list that trusts the parent, for use by the child.
#[cfg(feature = "comms")]
pub fn create_child_trusted_peers(parent_context: &ParentCommsContext) -> TrustedPeers {
    TrustedPeers {
        peers: vec![TrustedPeer {
            name: parent_context.parent_name.clone(),
            pubkey: PubKey::new(parent_context.parent_pubkey),
            addr: parent_context.parent_addr.clone(),
            meta: meerkat_comms::PeerMeta::default(),
        }],
    }
}

/// Create comms configuration for a child agent
#[cfg(feature = "comms")]
pub fn create_child_comms_config(child_name: &str, base_dir: &std::path::Path) -> CoreCommsConfig {
    CoreCommsConfig {
        enabled: true,
        name: child_name.to_string(),
        // Use UDS for local communication (more efficient than TCP)
        listen_uds: Some(base_dir.join(format!("{child_name}.sock"))),
        listen_tcp: None,
        identity_dir: base_dir.join(format!("{child_name}/identity")),
        trusted_peers_path: base_dir.join(format!("{child_name}/trusted_peers.json")),
        ack_timeout_secs: 30,
        max_message_bytes: 1_048_576,
        ..Default::default()
    }
}

/// Initialize a session for a spawned sub-agent (clean context)
pub fn create_spawn_session(prompt: &str, system_prompt: Option<&str>) -> Session {
    let mut session = Session::new();

    // Add system prompt if provided
    if let Some(sys) = system_prompt {
        session.push(Message::System(SystemMessage {
            content: sys.to_string(),
        }));
    }

    // Add the user prompt
    session.push(Message::User(UserMessage::text(prompt.to_string())));

    session
}

/// Initialize a session for a forked sub-agent (with history)
pub fn create_fork_session(parent_session: &Session, fork_prompt: &str) -> Session {
    let mut session = parent_session.clone();

    // Append the fork prompt as a new user message
    session.push(Message::User(UserMessage::text(fork_prompt.to_string())));

    session
}

/// Scope path of a child: the parent's path (or an unknown primary when the
/// parent has none) followed by the child's own frame.
fn child_scope_path(
    parent_scope_path: &[StreamScopeFrame],
    child_scope_frame: StreamScopeFrame,
) -> Vec<StreamScopeFrame> {
    let mut path = if parent_scope_path.is_empty() {
        vec![StreamScopeFrame::Primary {
            session_id: "unknown".to_string(),
        }]
    } else {
        parent_scope_path.to_vec()
    };
    path.push(child_scope_frame);
    path
}

/// The client a child runs against: metered under `scope_path` when an
/// observer is set, otherwise the given client unchanged.
fn child_llm_client(
    client: Arc<dyn AgentLlmClient>,
    usage_observer: Option<Arc<dyn UsageObserver>>,
    scope_path: &[StreamScopeFrame],
    model: &str,
) -> Arc<dyn AgentLlmClient> {
    match usage_observer {
        Some(observer) => Arc::new(MeteredLlmClient::new(
            client,
            observer,
            scope_path.to_vec(),
            model,
        )),
        None => client,
    }
}

fn spawn_scoped_forwarder(
    mut child_event_rx: mpsc::Receiver<AgentEvent>,
    scoped_tx: mpsc::Sender<ScopedAgentEvent>,
    scope_path: Vec<StreamScopeFrame>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(event) = child_event_rx.recv().await {
            let scoped = ScopedAgentEvent::new(scope_path.clone(), event);
            if scoped_tx.send(scoped).await.is_err() {
                break;
            }
        }
    })
}

/// Set up the comms infrastructure for a child agent
///
/// This creates the identity, trust files, and comms runtime for a child agent.
/// Returns the child's public key and address for the parent to use.
#[cfg(feature = "comms")]
pub async fn setup_child_comms(
    config: &SubAgentCommsConfig,
) -> Result<(CommsRuntime, [u8; 32], String), SubAgentRunnerError> {
    // Create trusted peers (trusting the parent)
    let trusted_peers = create_child_trusted_peers(&config.parent_context);
    let comms_config = create_child_comms_config(&config.name, &config.base_dir);
    let resolved_config = comms_config.resolve_paths(&config.base_dir);

    // Ensure directories exist without blocking the async runtime.
    tokio::fs::create_dir_all(&resolved_config.identity_dir)
        .await
        .map_err(|e| {
            SubAgentRunnerError::CommsSetup(format!("Failed to create identity dir: {e}"))
        })?;

    if let Some(parent) = resolved_config.trusted_peers_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| SubAgentRunnerError::CommsSetup(format!("Failed to create dir: {e}")))?;
    }

    // Save trusted peers
    let trusted_peers_path = resolved_config.trusted_peers_path.clone();
    trusted_peers.save(&trusted_peers_path).await.map_err(|e| {
        SubAgentRunnerError::CommsSetup(format!("Failed to save trusted peers: {e}"))
    })?;

    // Create and start comms runtime
    let mut runtime = CommsRuntime::new(resolved_config.clone())
        .await
        .map_err(|e| {
            SubAgentRunnerError::CommsSetup(format!("Failed to create comms runtime: {e}"))
        })?;

    // Get child's public key before starting listeners
    let child_pubkey = *runtime.public_key().as_bytes();
    let child_addr = format!(
        "uds://{}",
        config
            .base_dir
            .join(format!("{}.sock", config.name))
            .display()
    );

    // Start listeners
    runtime
        .start_listeners()
        .await
        .map_err(|e| SubAgentRunnerError::CommsSetup(format!("Failed to start listeners: {e}")))?;

    Ok((runtime, child_pubkey, child_addr))
}

/// Create a TrustedPeer entry for the parent to trust the child
///
/// After setting up child comms, the parent needs to add this peer to its trust list.
#[cfg(feature = "comms")]
pub fn create_child_peer_entry(
    child_name: &str,
    child_pubkey: [u8; 32],
    child_addr: &str,
) -> TrustedPeer {
    TrustedPeer {
        name: child_name.to_string(),
        pubkey: PubKey::new(child_pubkey),
        addr: child_addr.to_string(),
        meta: meerkat_comms::PeerMeta::default(),
    }
}

/// Specification for spawning a sub-agent (generic version)
pub struct SubAgentSpec<C, T, S>
where
    C: LlmClient + 'static,
    T: AgentToolDispatcher + 'static,
    S: AgentSessionStore + 'static,
{
    /// LLM client to use
    pub client: Arc<C>,
    /// Model name
    pub model: String,
    /// Tool dispatcher
    pub tools: Arc<T>,
    /// Session store
    pub store: Arc<S>,
    /// Initial session
    pub session: Session,
    /// Budget limits
    pub budget: Option<BudgetLimits>,
    /// Nesting depth (parent depth + 1)
    pub depth: u32,
    /// System prompt override
    pub system_prompt: Option<String>,
}

/// Specification for spawning a sub-agent (trait object version)
///
/// This is used when the concrete types aren't known at compile time,
/// such as when the client comes from a `SubAgentModelResolver`.
pub struct DynSubAgentSpec {
    /// Agent-ready LLM client (already adapted for `model`).
    pub client: Arc<dyn AgentLlmClient>,
    /// Model name (also the model usage is attributed to)
    pub model: String,
    /// Tool dispatcher (trait object)
    pub tools: Arc<dyn AgentToolDispatcher>,
    /// Session store (trait object)
    pub store: Arc<dyn AgentSessionStore>,
    /// Initial session
    pub session: Session,
    /// Budget limits
    pub budget: Option<BudgetLimits>,
    /// Nesting depth (parent depth + 1)
    pub depth: u32,
    /// System prompt override
    pub system_prompt: Option<String>,
    /// Comms configuration (if comms should be enabled for this sub-agent)
    #[cfg(feature = "comms")]
    pub comms_config: Option<SubAgentCommsConfig>,
    /// Parent's trusted peers (for adding this sub-agent as trusted)
    /// When set, the sub-agent will be added to this list after comms setup
    /// so the parent can accept connections from the sub-agent.
    #[cfg(feature = "comms")]
    pub parent_trusted_peers: Option<Arc<parking_lot::RwLock<TrustedPeers>>>,
    /// Parent's sync classification sidecar. Updated alongside parent_trusted_peers.
    #[cfg(feature = "comms")]
    pub parent_classification_peers: Option<Arc<parking_lot::RwLock<TrustedPeers>>>,
    /// Host mode - agent stays alive processing comms messages after initial prompt
    pub host_mode: bool,
    /// Optional scoped stream sink for attributed child events.
    pub scoped_event_tx: Option<tokio::sync::mpsc::Sender<ScopedAgentEvent>>,
    /// Parent scope path used as the base for child attribution.
    pub parent_scope_path: Vec<StreamScopeFrame>,
    /// Hook engine inherited from the parent so guardrail hooks (e.g.
    /// `PreToolExecution`) also apply inside the sub-agent.
    pub hook_engine: Option<Arc<dyn HookEngine>>,
    /// Run-scoped hook overrides inherited from the parent.
    pub hook_run_overrides: HookRunOverrides,
    /// When set, the child's client is wrapped in a [`MeteredLlmClient`]
    /// reporting under the child's scope path.
    pub usage_observer: Option<Arc<dyn UsageObserver>>,
}

/// Spawn and run a sub-agent in a background task (trait object version)
///
/// The client in `spec` is used as-is (it is already an `AgentLlmClient`);
/// when `spec.usage_observer` is set it is wrapped in a [`MeteredLlmClient`].
///
/// ## Comms Setup
///
/// If `spec.comms_config` is provided, this function uses `CommsBootstrap::for_child`
/// to set up comms for the sub-agent in a uniform way. The tools are then wrapped
/// with comms tools using `wrap_with_comms`, ensuring sub-agents have full comms
/// capabilities just like the main agent.
pub async fn spawn_sub_agent_dyn(
    id: OperationId,
    name: String,
    spec: DynSubAgentSpec,
    manager: Arc<SubAgentManager>,
) -> Result<(), SubAgentRunnerError> {
    #[cfg(feature = "comms")]
    use meerkat_core::sub_agent::SubAgentCommsInfo;

    let started_at = Instant::now();
    let scoped_event_tx = spec.scoped_event_tx.clone();
    let scope_path = child_scope_path(
        &spec.parent_scope_path,
        StreamScopeFrame::SubAgent {
            agent_id: id.to_string(),
            tool_call_id: None,
            label: Some(name.clone()),
        },
    );

    let llm_client = child_llm_client(spec.client, spec.usage_observer, &scope_path, &spec.model);

    // Build the agent - now supports trait objects directly via ?Sized bounds
    let mut builder = AgentBuilder::new()
        .model(&spec.model)
        .depth(spec.depth)
        .resume_session(spec.session)
        .with_hook_run_overrides(spec.hook_run_overrides);

    if let Some(sys_prompt) = &spec.system_prompt {
        builder = builder.system_prompt(sys_prompt);
    }

    if let Some(budget) = spec.budget {
        builder = builder.budget(budget);
    }

    if let Some(engine) = spec.hook_engine {
        builder = builder.with_hook_engine(engine);
    }

    // Set up comms for the sub-agent if configured
    // Uses CommsBootstrap::for_child_inproc for lightweight in-process communication
    // No network listeners or filesystem resources needed
    #[cfg(feature = "comms")]
    let (tools, comms_info) = if let Some(comms_config) = spec.comms_config {
        let bootstrap = CommsBootstrap::for_child_inproc(
            comms_config.name.clone(),
            comms_config.parent_context.clone(),
        );

        match bootstrap.prepare().await {
            Ok(Some(prepared)) => {
                // Wrap tools with comms - this is the uniform way to add comms tools
                let tools_with_comms = wrap_with_comms(spec.tools.clone(), &prepared.runtime);

                // Extract advertise info for parent trust
                let comms_info = prepared.advertise.map(|adv| SubAgentCommsInfo {
                    pubkey: adv.pubkey,
                    addr: adv.addr,
                });

                // Add the child to the parent's trusted peers so the parent
                // will accept connections from this sub-agent
                if let (Some(info), Some(parent_peers)) = (&comms_info, &spec.parent_trusted_peers)
                {
                    let child_peer = TrustedPeer {
                        name: name.clone(),
                        pubkey: PubKey::new(info.pubkey),
                        addr: info.addr.clone(),
                        meta: meerkat_comms::PeerMeta::default(),
                    };
                    // Update both the async store and the sync classification
                    // sidecar so the parent accepts this child at ingress
                    // immediately, before the next drain.
                    parent_peers.write().upsert(child_peer.clone());
                    if let Some(ref classification_peers) = spec.parent_classification_peers {
                        classification_peers.write().upsert(child_peer);
                    }
                    tracing::debug!("Added sub-agent '{}' to parent's trusted peers", name);
                }

                // Pass the comms runtime to the builder
                builder = builder.with_comms_runtime(Arc::new(prepared.runtime));

                (tools_with_comms, comms_info)
            }
            Ok(None) => {
                // Comms disabled (shouldn't happen for Child mode, but handle gracefully)
                (spec.tools, None)
            }
            Err(e) => {
                tracing::warn!("Failed to set up comms for sub-agent '{}': {}", name, e);
                (spec.tools, None)
            }
        }
    } else {
        (spec.tools, None)
    };

    #[cfg(not(feature = "comms"))]
    let (tools, comms_info) = (spec.tools, None);

    // Pass trait objects directly - no wrappers needed thanks to ?Sized bounds
    let mut agent = builder.build(llm_client, tools, spec.store).await;

    // Register the sub-agent with the manager BEFORE spawning the task
    // This is critical - without registration, status/steer/cancel won't find the agent
    manager
        .register_with_comms(id.clone(), name.clone(), comms_info)
        .await
        .map_err(|e| {
            SubAgentRunnerError::ExecutionError(format!("Failed to register sub-agent: {e}"))
        })?;

    // Clone what we need for the spawned task
    let id_for_task = id.clone();
    let _name_for_task = name.clone();
    let manager_for_task = manager.clone();
    let host_mode = spec.host_mode;

    // Spawn the agent execution task
    tokio::spawn(async move {
        let (result, forwarder_task) = if let Some(scoped_tx) = scoped_event_tx {
            let (child_event_tx, child_event_rx) = mpsc::channel::<AgentEvent>(64);
            let forwarder = spawn_scoped_forwarder(child_event_rx, scoped_tx, scope_path);

            let result = if host_mode {
                #[cfg(feature = "comms")]
                {
                    agent
                        .run_host_mode_with_events(String::new().into(), child_event_tx)
                        .await
                }
                #[cfg(not(feature = "comms"))]
                {
                    agent.run_host_mode(String::new().into()).await
                }
            } else {
                // The session already has the user prompt, so use run_pending()
                // which runs from the existing session without adding a new message.
                agent.run_pending_with_events(child_event_tx).await
            };
            (result, Some(forwarder))
        } else {
            let result = if host_mode {
                agent.run_host_mode(String::new().into()).await
            } else {
                // The session already has the user prompt, so use run_pending()
                // which runs from the existing session without adding a new message.
                agent.run_pending().await
            };
            (result, None)
        };

        if let Some(forwarder) = forwarder_task {
            let _ = forwarder.await;
        }

        let duration_ms = started_at.elapsed().as_millis() as u64;

        match result {
            Ok(run_result) => {
                manager_for_task
                    .complete(
                        &id_for_task,
                        OperationResult {
                            id: id_for_task.clone(),
                            content: run_result.text,
                            is_error: false,
                            duration_ms,
                            tokens_used: run_result.usage.total_tokens(),
                        },
                    )
                    .await;
            }
            Err(e) => {
                manager_for_task.fail(&id_for_task, e.to_string()).await;
            }
        }
    });

    Ok(())
}

/// Spawn and run a sub-agent in a background task (generic version)
///
/// This function:
/// 1. Creates an Agent with the given configuration
/// 2. Spawns a tokio task that runs the agent loop
/// 3. Reports results back to the manager when complete
pub async fn spawn_sub_agent<C, T, S>(
    id: OperationId,
    name: String,
    spec: SubAgentSpec<C, T, S>,
    manager: Arc<SubAgentManager>,
) -> Result<(), SubAgentRunnerError>
where
    C: LlmClient + 'static,
    T: AgentToolDispatcher + 'static,
    S: AgentSessionStore + 'static,
{
    let started_at = Instant::now();

    // Create the LLM client adapter
    let llm_adapter = Arc::new(LlmClientAdapter::new(spec.client, spec.model.clone()));

    // Build the agent
    let mut builder = AgentBuilder::new()
        .model(&spec.model)
        .depth(spec.depth)
        .resume_session(spec.session);

    if let Some(sys_prompt) = &spec.system_prompt {
        builder = builder.system_prompt(sys_prompt);
    }

    if let Some(budget) = spec.budget {
        builder = builder.budget(budget);
    }

    let mut agent = builder.build(llm_adapter, spec.tools, spec.store).await;

    // Register the sub-agent with the manager BEFORE spawning the task
    // This is critical - without registration, status/steer/cancel won't find the agent
    manager
        .register(id.clone(), name.clone())
        .await
        .map_err(|e| {
            SubAgentRunnerError::ExecutionError(format!("Failed to register sub-agent: {e}"))
        })?;

    // Clone what we need for the spawned task
    let id_for_task = id.clone();
    let _name_for_task = name.clone(); // Kept for potential future use (logging, etc.)
    let manager_for_task = manager.clone();

    // Spawn the agent execution task
    tokio::spawn(async move {
        // The session already has the user prompt via create_spawn_session/create_fork_session.
        // Use run_pending() to run from the existing session without adding a new message.
        let result = agent.run_pending().await;

        let duration_ms = started_at.elapsed().as_millis() as u64;

        match result {
            Ok(run_result) => {
                manager_for_task
                    .complete(
                        &id_for_task,
                        OperationResult {
                            id: id_for_task.clone(),
                            content: run_result.text,
                            is_error: false,
                            duration_ms,
                            tokens_used: run_result.usage.total_tokens(),
                        },
                    )
                    .await;
            }
            Err(e) => {
                manager_for_task.fail(&id_for_task, e.to_string()).await;
            }
        }
    });

    Ok(())
}

/// Errors that can occur during sub-agent execution
#[derive(Debug, thiserror::Error)]
pub enum SubAgentRunnerError {
    #[error("Comms setup failed: {0}")]
    CommsSetup(String),

    #[error("Agent execution error: {0}")]
    ExecutionError(String),

    #[error("Session error: {0}")]
    SessionError(String),

    #[error("LLM client error: {0}")]
    LlmError(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_create_spawn_session() {
        let session = create_spawn_session("Do this task", None);
        assert_eq!(session.messages().len(), 1);
        match &session.messages()[0] {
            Message::User(u) => assert_eq!(u.text_content(), "Do this task"),
            _ => unreachable!("Expected User message"),
        }
    }

    #[test]
    fn test_create_spawn_session_with_system() {
        let session = create_spawn_session("Do this task", Some("You are helpful"));
        assert_eq!(session.messages().len(), 2);
        match &session.messages()[0] {
            Message::System(s) => assert_eq!(s.content, "You are helpful"),
            _ => unreachable!("Expected System message"),
        }
        match &session.messages()[1] {
            Message::User(u) => assert_eq!(u.text_content(), "Do this task"),
            _ => unreachable!("Expected User message"),
        }
    }

    #[test]
    fn test_create_fork_session() {
        let mut parent = Session::new();
        parent.push(Message::User(UserMessage::text(
            "Original prompt".to_string(),
        )));

        let forked = create_fork_session(&parent, "Continue with this");
        assert_eq!(forked.messages().len(), 2);
        match &forked.messages()[1] {
            Message::User(u) => assert_eq!(u.text_content(), "Continue with this"),
            _ => unreachable!("Expected User message"),
        }
    }

    #[cfg(feature = "comms")]
    #[test]
    fn test_create_child_trusted_peers() {
        let parent_context = ParentCommsContext {
            parent_name: "parent-agent".to_string(),
            parent_pubkey: [42u8; 32],
            parent_addr: "uds:///tmp/parent.sock".to_string(),
            comms_base_dir: PathBuf::from("/tmp/comms"),
            inproc_namespace: None,
        };

        let trusted = create_child_trusted_peers(&parent_context);
        assert_eq!(trusted.peers.len(), 1);
        assert_eq!(trusted.peers[0].name, "parent-agent");
        assert_eq!(*trusted.peers[0].pubkey.as_bytes(), [42u8; 32]);
    }

    #[cfg(feature = "comms")]
    #[test]
    fn test_create_child_comms_config() {
        let base_dir = PathBuf::from("/tmp/agents");
        let config = create_child_comms_config("child-1", &base_dir);

        assert!(config.enabled);
        assert_eq!(config.name, "child-1");
        assert!(config.listen_uds.is_some());
        assert!(config.listen_tcp.is_none());
    }

    #[cfg(feature = "comms")]
    #[test]
    fn test_create_child_peer_entry() {
        let peer = create_child_peer_entry("child-1", [1u8; 32], "uds:///tmp/child.sock");

        assert_eq!(peer.name, "child-1");
        assert_eq!(*peer.pubkey.as_bytes(), [1u8; 32]);
        assert_eq!(peer.addr, "uds:///tmp/child.sock");
    }

    #[test]
    fn test_sub_agent_runner_error_display() {
        let err = SubAgentRunnerError::CommsSetup("test error".to_string());
        assert!(err.to_string().contains("Comms setup failed"));
        assert!(err.to_string().contains("test error"));

        let err = SubAgentRunnerError::ExecutionError("runtime error".to_string());
        assert!(err.to_string().contains("execution error"));
    }

    mod usage_metering {
        use super::hook_propagation::{base_spec, run_to_end};
        use super::*;
        use crate::builtin::sub_agent::test_support::FixedUsageClient;
        use async_trait::async_trait;
        use meerkat_core::error::ToolError;
        use meerkat_core::ops::SubAgentState;
        use meerkat_core::{ToolCallView, ToolDef, ToolResult, Usage, UsageLedger};

        struct NoTools;

        #[async_trait]
        impl AgentToolDispatcher for NoTools {
            fn tools(&self) -> Arc<[Arc<ToolDef>]> {
                Arc::from([Arc::new(ToolDef {
                    name: "noop".to_string(),
                    description: "noop".to_string(),
                    input_schema: serde_json::json!({"type": "object"}),
                })])
            }

            async fn dispatch(&self, call: ToolCallView<'_>) -> Result<ToolResult, ToolError> {
                Ok(ToolResult::new(
                    call.id.to_string(),
                    "ok".to_string(),
                    false,
                ))
            }
        }

        fn usage() -> Usage {
            Usage {
                input_tokens: 1000,
                output_tokens: 200,
                cache_creation_tokens: Some(30),
                cache_read_tokens: Some(400),
            }
        }

        fn parent_path() -> Vec<StreamScopeFrame> {
            vec![StreamScopeFrame::Primary {
                session_id: "root".to_string(),
            }]
        }

        #[tokio::test]
        async fn completed_child_usage_is_metered_with_scope_and_model() {
            let ledger = Arc::new(UsageLedger::new());
            let mut spec = base_spec(Arc::new(FixedUsageClient::new(usage())), Arc::new(NoTools));
            spec.model = "child-model".to_string();
            spec.parent_scope_path = parent_path();
            spec.usage_observer = Some(Arc::clone(&ledger) as Arc<dyn UsageObserver>);

            let (id, state) = run_to_end(spec).await;
            assert_eq!(state, SubAgentState::Completed);

            let rows = ledger.snapshot();
            assert_eq!(rows.len(), 1, "{rows:?}");
            assert_eq!(rows[0].scope_id, format!("primary/sub:{id}"));
            assert_eq!(rows[0].model, "child-model");
            assert_eq!(rows[0].usage, usage());
        }

        #[tokio::test]
        async fn failed_child_usage_is_retained() {
            let ledger = Arc::new(UsageLedger::new());
            let mut spec = base_spec(
                Arc::new(FixedUsageClient::tool_then_fail(usage(), "noop")),
                Arc::new(NoTools),
            );
            spec.usage_observer = Some(Arc::clone(&ledger) as Arc<dyn UsageObserver>);

            let (id, state) = run_to_end(spec).await;
            assert_eq!(state, SubAgentState::Failed);

            let rows = ledger.snapshot();
            assert_eq!(rows.len(), 1, "{rows:?}");
            // No parent path: the child is attributed under an unknown primary.
            assert_eq!(rows[0].scope_id, format!("primary/sub:{id}"));
            assert_eq!(rows[0].usage, usage());
        }

        #[test]
        fn without_observer_the_spec_client_is_used_unwrapped() {
            let client: Arc<dyn AgentLlmClient> = Arc::new(FixedUsageClient::new(usage()));
            let used = child_llm_client(Arc::clone(&client), None, &parent_path(), "m");
            assert!(Arc::ptr_eq(&used, &client));

            let ledger: Arc<dyn UsageObserver> = Arc::new(UsageLedger::new());
            let used = child_llm_client(Arc::clone(&client), Some(ledger), &parent_path(), "m");
            assert!(!Arc::ptr_eq(&used, &client));
        }

        #[test]
        fn child_scope_path_appends_frame_to_parent_or_unknown_primary() {
            let frame = StreamScopeFrame::SubAgent {
                agent_id: "a".to_string(),
                tool_call_id: None,
                label: None,
            };
            let path = child_scope_path(&parent_path(), frame.clone());
            assert_eq!(path, vec![parent_path()[0].clone(), frame.clone()]);
            let path = child_scope_path(&[], frame.clone());
            assert_eq!(
                path,
                vec![
                    StreamScopeFrame::Primary {
                        session_id: "unknown".to_string()
                    },
                    frame
                ]
            );
        }
    }

    mod hook_propagation {
        use super::*;
        use async_trait::async_trait;
        use meerkat_client::{
            LlmClient, LlmClientAdapter, LlmDoneOutcome, LlmError, LlmEvent, LlmRequest,
        };
        use meerkat_core::error::{AgentError, ToolError};
        use meerkat_core::hooks::{
            HookDecision, HookEngineError, HookExecutionReport, HookId, HookInvocation, HookPoint,
            HookReasonCode,
        };
        use meerkat_core::ops::{ConcurrencyLimits, SubAgentState};
        use meerkat_core::{StopReason, ToolCallView, ToolDef, ToolResult};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        const TOOL_NAME: &str = "guarded_tool";

        /// First call emits one tool call; subsequent calls end the turn.
        struct ToolCallingClient {
            calls: AtomicUsize,
        }

        #[async_trait]
        impl LlmClient for ToolCallingClient {
            fn stream<'a>(
                &'a self,
                _request: &'a LlmRequest,
            ) -> meerkat_client::types::LlmStream<'a> {
                let events = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    vec![
                        Ok(LlmEvent::ToolCallComplete {
                            id: "tc_1".to_string(),
                            name: TOOL_NAME.to_string(),
                            args: serde_json::json!({}),
                            meta: None,
                        }),
                        Ok(LlmEvent::Done {
                            outcome: LlmDoneOutcome::Success {
                                stop_reason: StopReason::ToolUse,
                            },
                        }),
                    ]
                } else {
                    vec![
                        Ok(LlmEvent::TextDelta {
                            delta: "done".to_string(),
                            meta: None,
                        }),
                        Ok(LlmEvent::Done {
                            outcome: LlmDoneOutcome::Success {
                                stop_reason: StopReason::EndTurn,
                            },
                        }),
                    ]
                };
                Box::pin(futures::stream::iter(events))
            }

            fn provider(&self) -> &'static str {
                "mock"
            }

            async fn health_check(&self) -> Result<(), LlmError> {
                Ok(())
            }
        }

        struct CountingDispatcher {
            dispatched: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl AgentToolDispatcher for CountingDispatcher {
            fn tools(&self) -> Arc<[Arc<ToolDef>]> {
                Arc::from([Arc::new(ToolDef {
                    name: TOOL_NAME.to_string(),
                    description: "guarded".to_string(),
                    input_schema: serde_json::json!({"type": "object"}),
                })])
            }

            async fn dispatch(&self, call: ToolCallView<'_>) -> Result<ToolResult, ToolError> {
                self.dispatched.fetch_add(1, Ordering::SeqCst);
                Ok(ToolResult::new(
                    call.id.to_string(),
                    "ok".to_string(),
                    false,
                ))
            }
        }

        struct NoopStore;

        #[async_trait]
        impl AgentSessionStore for NoopStore {
            async fn save(&self, _session: &Session) -> Result<(), AgentError> {
                Ok(())
            }

            async fn load(&self, _id: &str) -> Result<Option<Session>, AgentError> {
                Ok(None)
            }
        }

        /// Denies every `PreToolExecution` and counts how often it was asked.
        struct DenyAllHooks {
            pre_tool_calls: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl HookEngine for DenyAllHooks {
            async fn execute(
                &self,
                invocation: HookInvocation,
                _overrides: Option<&meerkat_core::config::HookRunOverrides>,
            ) -> Result<HookExecutionReport, HookEngineError> {
                if invocation.point != HookPoint::PreToolExecution {
                    return Ok(HookExecutionReport::empty());
                }
                self.pre_tool_calls.fetch_add(1, Ordering::SeqCst);
                Ok(HookExecutionReport {
                    decision: Some(HookDecision::deny(
                        HookId::new("deny-all"),
                        HookReasonCode::PolicyViolation,
                        "denied in test",
                        None,
                    )),
                    ..HookExecutionReport::default()
                })
            }
        }

        pub(super) fn base_spec(
            client: Arc<dyn AgentLlmClient>,
            tools: Arc<dyn AgentToolDispatcher>,
        ) -> DynSubAgentSpec {
            DynSubAgentSpec {
                client,
                model: "mock-model".to_string(),
                tools,
                store: Arc::new(NoopStore),
                session: create_spawn_session("use the tool", None),
                budget: None,
                depth: 1,
                system_prompt: None,
                #[cfg(feature = "comms")]
                comms_config: None,
                #[cfg(feature = "comms")]
                parent_trusted_peers: None,
                #[cfg(feature = "comms")]
                parent_classification_peers: None,
                host_mode: false,
                scoped_event_tx: None,
                parent_scope_path: Vec::new(),
                hook_engine: None,
                hook_run_overrides: HookRunOverrides::default(),
                usage_observer: None,
            }
        }

        /// Spawn `spec`, wait until the child leaves `Running`, return the
        /// operation id and final state.
        pub(super) async fn run_to_end(spec: DynSubAgentSpec) -> (OperationId, SubAgentState) {
            let manager = Arc::new(SubAgentManager::new(ConcurrencyLimits::default(), 0));
            let id = OperationId::new();
            spawn_sub_agent_dyn(id.clone(), "child".to_string(), spec, Arc::clone(&manager))
                .await
                .expect("spawn sub-agent");
            tokio::time::timeout(Duration::from_secs(10), async {
                while matches!(
                    manager.get_state(&id).await,
                    Some(SubAgentState::Running) | None
                ) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("sub-agent finished in time");
            let state = manager.get_state(&id).await.expect("sub-agent registered");
            (id, state)
        }

        pub(super) fn tool_calling_client() -> Arc<dyn AgentLlmClient> {
            Arc::new(LlmClientAdapter::new(
                Arc::new(ToolCallingClient {
                    calls: AtomicUsize::new(0),
                }),
                "mock-model".to_string(),
            ))
        }

        /// Spawn a sub-agent that makes one tool call; returns how often the
        /// tool was actually dispatched.
        async fn run_sub_agent(
            hook_engine: Option<Arc<dyn HookEngine>>,
            hook_run_overrides: HookRunOverrides,
        ) -> usize {
            let dispatched = Arc::new(AtomicUsize::new(0));
            let mut spec = base_spec(
                tool_calling_client(),
                Arc::new(CountingDispatcher {
                    dispatched: Arc::clone(&dispatched),
                }),
            );
            spec.hook_engine = hook_engine;
            spec.hook_run_overrides = hook_run_overrides;
            let (_, state) = run_to_end(spec).await;
            assert_eq!(state, SubAgentState::Completed);
            dispatched.load(Ordering::SeqCst)
        }

        #[tokio::test]
        async fn sub_agent_pre_tool_hook_denies_tool_call() {
            let pre_tool_calls = Arc::new(AtomicUsize::new(0));
            let engine: Arc<dyn HookEngine> = Arc::new(DenyAllHooks {
                pre_tool_calls: Arc::clone(&pre_tool_calls),
            });

            let dispatched = run_sub_agent(Some(engine), HookRunOverrides::default()).await;

            assert_eq!(pre_tool_calls.load(Ordering::SeqCst), 1);
            assert_eq!(dispatched, 0, "denied tool must not be dispatched");
        }

        #[tokio::test]
        async fn sub_agent_without_hook_engine_dispatches_tool_call() {
            assert_eq!(run_sub_agent(None, HookRunOverrides::default()).await, 1);
        }

        /// Records the run overrides handed to every hook invocation.
        struct OverrideSpy {
            seen: Arc<parking_lot::Mutex<Vec<Option<HookRunOverrides>>>>,
        }

        #[async_trait]
        impl HookEngine for OverrideSpy {
            async fn execute(
                &self,
                _invocation: HookInvocation,
                overrides: Option<&HookRunOverrides>,
            ) -> Result<HookExecutionReport, HookEngineError> {
                self.seen.lock().push(overrides.cloned());
                Ok(HookExecutionReport::empty())
            }
        }

        #[tokio::test]
        async fn sub_agent_receives_parent_hook_run_overrides() {
            let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
            let engine: Arc<dyn HookEngine> = Arc::new(OverrideSpy {
                seen: Arc::clone(&seen),
            });
            let overrides = HookRunOverrides {
                entries: Vec::new(),
                disable: vec![HookId::new("parent-disabled")],
            };

            run_sub_agent(Some(engine), overrides.clone()).await;

            let seen = seen.lock();
            assert!(!seen.is_empty(), "hooks were invoked");
            assert!(
                seen.iter().all(|o| o.as_ref() == Some(&overrides)),
                "every hook call sees the parent's overrides: {seen:?}"
            );
        }
    }
}
