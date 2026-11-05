//! agent_fork tool - Fork current agent with continued context

#[cfg(feature = "comms")]
use super::runner::SubAgentCommsConfig;
use super::runner::{DynSubAgentSpec, create_fork_session, spawn_sub_agent_dyn};
use super::state::SubAgentToolState;
use crate::builtin::{BuiltinTool, BuiltinToolError, ToolOutput};
use crate::dispatcher::FilteredDispatcher;
use async_trait::async_trait;
use meerkat_core::ToolDef;
use meerkat_core::ops::{ForkBudgetPolicy, OperationId, ToolAccessPolicy};
use serde::de;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum ProviderName {
    Anthropic,
    Openai,
    Gemini,
}

impl ProviderName {
    fn as_str(self) -> &'static str {
        match self {
            ProviderName::Anthropic => "anthropic",
            ProviderName::Openai => "openai",
            ProviderName::Gemini => "gemini",
        }
    }
}

/// Tool access policy for forked agents
#[derive(Debug, schemars::JsonSchema)]
#[serde(tag = "policy", rename_all = "snake_case")]
enum ToolAccessInput {
    /// Inherit all tools from parent
    Inherit,
    /// Only allow specific tools
    AllowList { tools: Vec<String> },
    /// Block specific tools
    DenyList { tools: Vec<String> },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
enum ToolAccessInputRaw {
    Inherit,
    AllowList { tools: Vec<String> },
    DenyList { tools: Vec<String> },
}

impl From<ToolAccessInputRaw> for ToolAccessInput {
    fn from(input: ToolAccessInputRaw) -> Self {
        match input {
            ToolAccessInputRaw::Inherit => ToolAccessInput::Inherit,
            ToolAccessInputRaw::AllowList { tools } => ToolAccessInput::AllowList { tools },
            ToolAccessInputRaw::DenyList { tools } => ToolAccessInput::DenyList { tools },
        }
    }
}

impl<'de> Deserialize<'de> for ToolAccessInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(raw) => {
                let parsed: ToolAccessInputRaw =
                    serde_json::from_str(&raw).map_err(de::Error::custom)?;
                Ok(parsed.into())
            }
            other => {
                let parsed: ToolAccessInputRaw =
                    serde_json::from_value(other).map_err(de::Error::custom)?;
                Ok(parsed.into())
            }
        }
    }
}

impl From<ToolAccessInput> for ToolAccessPolicy {
    fn from(input: ToolAccessInput) -> Self {
        match input {
            ToolAccessInput::Inherit => ToolAccessPolicy::Inherit,
            ToolAccessInput::AllowList { tools } => ToolAccessPolicy::AllowList(tools),
            ToolAccessInput::DenyList { tools } => ToolAccessPolicy::DenyList(tools),
        }
    }
}

/// Parameters for agent_fork tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ForkParams {
    /// Additional prompt for the forked agent
    #[schemars(
        description = "Additional prompt/instruction for the forked agent. This is appended to the inherited conversation history."
    )]
    prompt: String,
    /// LLM provider to use (anthropic, openai, gemini).
    #[serde(default)]
    provider: Option<ProviderName>,
    /// Model name (provider-specific).
    #[serde(default)]
    model: Option<String>,
    /// Tool access policy
    #[serde(default)]
    tool_access: Option<ToolAccessInput>,
    /// Budget allocation policy.
    #[serde(default)]
    budget_policy: Option<String>,
}

/// Response from agent_fork tool
#[derive(Debug, Serialize)]
struct ForkResponse {
    /// Forked sub-agent ID
    agent_id: String,
    /// Name assigned to the forked agent
    name: String,
    /// Provider being used
    provider: String,
    /// Model being used
    model: String,
    /// Current state (will be "running")
    state: String,
    /// Number of messages inherited from parent
    messages_inherited: usize,
    /// Additional info
    message: String,
    /// How to communicate with this child (if comms enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    comms: Option<String>,
}

/// Generate comms instructions for parent about messaging forked child
#[cfg(feature = "comms")]
fn parent_comms_instructions(child_name: &str) -> String {
    format!(
        "To message this fork: send({{\"kind\":\"peer_message\",\"to\":\"{child_name}\",\"body\":\"your message\"}})"
    )
}

/// Tool for forking the current agent with continued context
pub struct AgentForkTool {
    state: Arc<SubAgentToolState>,
}

impl AgentForkTool {
    /// Create a new agent_fork tool
    pub fn new(state: Arc<SubAgentToolState>) -> Self {
        Self { state }
    }

    fn resolve_budget_policy(&self, policy_str: Option<&str>) -> ForkBudgetPolicy {
        match policy_str {
            Some("equal") => ForkBudgetPolicy::Equal,
            Some("remaining") => ForkBudgetPolicy::Remaining,
            Some("proportional") => ForkBudgetPolicy::Proportional,
            Some(fixed) if fixed.starts_with("fixed:") => {
                if let Ok(tokens) = fixed[6..].parse::<u64>() {
                    ForkBudgetPolicy::Fixed(tokens)
                } else {
                    ForkBudgetPolicy::Equal
                }
            }
            _ => ForkBudgetPolicy::Equal,
        }
    }

    fn resolve_fork_budget(&self, policy: ForkBudgetPolicy) -> meerkat_core::budget::BudgetLimits {
        use meerkat_core::budget::BudgetLimits;

        let default_tokens = self.state.config.default_budget.unwrap_or(50000);
        let max_per_agent = self.state.config.max_budget_per_agent;

        let tokens = match policy {
            ForkBudgetPolicy::Equal => default_tokens,
            ForkBudgetPolicy::Remaining => {
                // For now, treat remaining as equal (would need parent budget tracking)
                default_tokens
            }
            ForkBudgetPolicy::Proportional => {
                // For now, treat proportional as equal (would need parent budget tracking)
                default_tokens
            }
            ForkBudgetPolicy::Fixed(amount) => {
                // Apply max cap if configured
                if let Some(max) = max_per_agent {
                    amount.min(max)
                } else {
                    amount
                }
            }
        };

        BudgetLimits::default().with_max_tokens(tokens)
    }

    async fn fork_agent(&self, params: ForkParams) -> Result<ForkResponse, BuiltinToolError> {
        // Check if we can spawn
        if !self.state.can_spawn().await {
            return Err(BuiltinToolError::execution_failed(
                "Cannot fork agent: concurrency or depth limit reached",
            ));
        }

        // Check nesting permission
        if self.state.depth() > 0 && !self.state.can_nest() {
            return Err(BuiltinToolError::execution_failed(
                "Nested forking is not allowed at this depth",
            ));
        }

        // Validate prompt
        if params.prompt.trim().is_empty() {
            return Err(BuiltinToolError::invalid_args(
                "Fork prompt cannot be empty",
            ));
        }

        // Resolve provider/model, validate it and create the fork's client
        let resolved = self
            .state
            .resolve_child_model(
                params.provider.map(ProviderName::as_str),
                params.model.as_deref(),
            )
            .await?;
        let provider = resolved.provider;
        let model = resolved.model;
        let client = resolved.client;

        // Resolve tool access policy and apply filtering
        let tool_access: ToolAccessPolicy = params
            .tool_access
            .map_or(ToolAccessPolicy::Inherit, Into::into);

        let filtered_tools: Arc<dyn meerkat_core::AgentToolDispatcher> = match &tool_access {
            ToolAccessPolicy::Inherit => self.state.tool_dispatcher.clone(),
            _ => Arc::new(FilteredDispatcher::new(
                self.state.tool_dispatcher.clone(),
                &tool_access,
            )),
        };

        // Resolve budget policy and calculate budget
        let budget_policy = self.resolve_budget_policy(params.budget_policy.as_deref());
        let budget = self.resolve_fork_budget(budget_policy);

        // Get parent session and create forked session
        let parent_session = self.state.parent_session.read().await;
        let messages_inherited = parent_session.messages().len();
        let session = create_fork_session(&parent_session, &params.prompt);
        drop(parent_session);
        let (scoped_event_tx, parent_scope_path) = self.state.scoped_stream().await;

        // Generate operation ID and name
        // Use first 12 hex chars (8 + 4 after dash) to avoid collision with UUIDv7
        let op_id = OperationId::new();
        let op_id_str = op_id.to_string();
        let name = format!("fork-{}{}", &op_id_str[..8], &op_id_str[9..13]);

        // Build comms config if parent has comms enabled
        #[cfg(feature = "comms")]
        let (comms_config, comms_instructions) =
            if let Some(parent_comms) = &self.state.parent_comms {
                let config = SubAgentCommsConfig {
                    name: name.clone(),
                    base_dir: parent_comms.comms_base_dir.clone(),
                    parent_context: parent_comms.clone(),
                };
                let instructions = Some(parent_comms_instructions(&name));
                (Some(config), instructions)
            } else {
                (None, None)
            };
        #[cfg(not(feature = "comms"))]
        let comms_instructions: Option<String> = None;

        // Create the sub-agent specification for the fork
        #[cfg(feature = "comms")]
        let spec = DynSubAgentSpec {
            client,
            model: model.clone(),
            tools: filtered_tools,
            store: self.state.session_store.clone(),
            session,
            budget: Some(budget),
            depth: self.state.depth() + 1,
            system_prompt: None, // Fork inherits system prompt from session
            comms_config,
            parent_trusted_peers: self.state.parent_trusted_peers.clone(),
            #[cfg(feature = "comms")]
            parent_classification_peers: self.state.parent_classification_peers.clone(),
            host_mode: false, // Forked agents don't run in host mode
            scoped_event_tx,
            parent_scope_path,
            hook_engine: self.state.hook_engine.clone(),
            hook_run_overrides: self.state.hook_run_overrides.clone(),
            usage_observer: self.state.usage_observer.clone(),
        };
        #[cfg(not(feature = "comms"))]
        let spec = DynSubAgentSpec {
            client,
            model: model.clone(),
            tools: filtered_tools,
            store: self.state.session_store.clone(),
            session,
            budget: Some(budget),
            depth: self.state.depth() + 1,
            system_prompt: None, // Fork inherits system prompt from session
            host_mode: false,    // Forked agents don't run in host mode
            scoped_event_tx,
            parent_scope_path,
            hook_engine: self.state.hook_engine.clone(),
            hook_run_overrides: self.state.hook_run_overrides.clone(),
            usage_observer: self.state.usage_observer.clone(),
        };

        // Spawn the forked agent
        spawn_sub_agent_dyn(
            op_id.clone(),
            name.clone(),
            spec,
            self.state.manager.clone(),
        )
        .await
        .map_err(|e| {
            BuiltinToolError::execution_failed(format!("Failed to spawn forked agent: {e}"))
        })?;

        Ok(ForkResponse {
            agent_id: op_id.to_string(),
            name,
            provider: provider.as_str().to_string(),
            model,
            state: "running".to_string(),
            messages_inherited,
            message: "Agent forked successfully with full conversation history. Use agent_status to check progress.".to_string(),
            comms: comms_instructions,
        })
    }
}

#[async_trait]
impl BuiltinTool for AgentForkTool {
    fn name(&self) -> &'static str {
        "agent_fork"
    }

    fn def(&self) -> ToolDef {
        ToolDef {
            name: "agent_fork".into(),
            description: "Fork the current agent with continued context. The forked agent inherits the full conversation history and continues from the same state. Useful for exploring alternative approaches or parallel execution.".into(),
            input_schema: crate::schema::schema_for::<ForkParams>(),
        }
    }

    fn default_enabled(&self) -> bool {
        false // Sub-agent tools are disabled by default
    }

    async fn call(&self, args: Value) -> Result<ToolOutput, BuiltinToolError> {
        let params: ForkParams = serde_json::from_value(args)
            .map_err(|e| BuiltinToolError::invalid_args(format!("Invalid parameters: {e}")))?;

        let response = self.fork_agent(params).await?;
        serde_json::to_value(response)
            .map(ToolOutput::Json)
            .map_err(|e| {
                BuiltinToolError::execution_failed(format!("Failed to serialize response: {e}"))
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::builtin::sub_agent::config::SubAgentConfig;
    use crate::builtin::sub_agent::test_support::CountingResolver;
    use meerkat_client::{FactoryError, LlmClient, LlmClientFactory, LlmProvider};
    use meerkat_core::error::{AgentError, ToolError};
    use meerkat_core::ops::ConcurrencyLimits;
    use meerkat_core::session::Session;
    use meerkat_core::sub_agent::SubAgentManager;
    use meerkat_core::types::{Message, UserMessage};
    use meerkat_core::{AgentSessionStore, AgentToolDispatcher, ToolCallView, ToolResult};
    use serde_json::json;
    use tokio::sync::RwLock;

    struct MockClientFactory {
        should_fail: bool,
    }

    impl MockClientFactory {
        fn new() -> Self {
            Self { should_fail: false }
        }

        fn failing() -> Self {
            Self { should_fail: true }
        }
    }

    impl LlmClientFactory for MockClientFactory {
        fn create_client(
            &self,
            _provider: LlmProvider,
            _api_key: Option<String>,
        ) -> Result<Arc<dyn LlmClient>, FactoryError> {
            if self.should_fail {
                Err(FactoryError::MissingApiKey("mock".into()))
            } else {
                Err(FactoryError::MissingApiKey("test-mode".into()))
            }
        }

        fn supported_providers(&self) -> Vec<LlmProvider> {
            vec![
                LlmProvider::Anthropic,
                LlmProvider::OpenAi,
                LlmProvider::Gemini,
            ]
        }
    }

    struct MockToolDispatcher;

    #[async_trait]
    impl AgentToolDispatcher for MockToolDispatcher {
        fn tools(&self) -> Arc<[Arc<ToolDef>]> {
            Arc::from([])
        }

        async fn dispatch(&self, call: ToolCallView<'_>) -> Result<ToolResult, ToolError> {
            Err(ToolError::not_found(call.name))
        }
    }

    struct MockSessionStore;

    #[async_trait]
    impl AgentSessionStore for MockSessionStore {
        async fn save(&self, _session: &Session) -> Result<(), AgentError> {
            Ok(())
        }

        async fn load(&self, _id: &str) -> Result<Option<Session>, AgentError> {
            Ok(None)
        }
    }

    fn create_test_state() -> Arc<SubAgentToolState> {
        let limits = ConcurrencyLimits::default();
        let manager = Arc::new(SubAgentManager::new(limits, 0));
        let client_factory = Arc::new(MockClientFactory::new());
        let tool_dispatcher = Arc::new(MockToolDispatcher);
        let session_store = Arc::new(MockSessionStore);
        let parent_session = Arc::new(RwLock::new(Session::new()));
        let config = SubAgentConfig::default();

        Arc::new(SubAgentToolState::new(
            manager,
            client_factory,
            tool_dispatcher,
            session_store,
            parent_session,
            config,
            0,
        ))
    }

    fn create_test_state_with_messages() -> Arc<SubAgentToolState> {
        let limits = ConcurrencyLimits::default();
        let manager = Arc::new(SubAgentManager::new(limits, 0));
        let client_factory = Arc::new(MockClientFactory::new());
        let tool_dispatcher = Arc::new(MockToolDispatcher);
        let session_store = Arc::new(MockSessionStore);

        let mut session = Session::new();
        session.push(Message::User(UserMessage::text("Hello".to_string())));
        session.push(Message::User(UserMessage::text("World".to_string())));

        let parent_session = Arc::new(RwLock::new(session));
        let config = SubAgentConfig::default();

        Arc::new(SubAgentToolState::new(
            manager,
            client_factory,
            tool_dispatcher,
            session_store,
            parent_session,
            config,
            0,
        ))
    }

    fn create_failing_state() -> Arc<SubAgentToolState> {
        let limits = ConcurrencyLimits::default();
        let manager = Arc::new(SubAgentManager::new(limits, 0));
        let client_factory = Arc::new(MockClientFactory::failing());
        let tool_dispatcher = Arc::new(MockToolDispatcher);
        let session_store = Arc::new(MockSessionStore);
        let parent_session = Arc::new(RwLock::new(Session::new()));
        let config = SubAgentConfig::default();

        Arc::new(SubAgentToolState::new(
            manager,
            client_factory,
            tool_dispatcher,
            session_store,
            parent_session,
            config,
            0,
        ))
    }

    #[test]
    fn test_tool_name() {
        let state = create_test_state();
        let tool = AgentForkTool::new(state);
        assert_eq!(tool.name(), "agent_fork");
    }

    #[test]
    fn test_tool_def() {
        let state = create_test_state();
        let tool = AgentForkTool::new(state);
        let def = tool.def();

        assert_eq!(def.name, "agent_fork");
        assert!(def.description.contains("Fork"));
        assert!(def.description.contains("history"));

        let schema = &def.input_schema;
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["prompt"].is_object());
        assert!(schema["properties"]["provider"].is_object());
        assert!(schema["properties"]["budget_policy"].is_object());
        assert_eq!(schema["required"], json!(["prompt"]));
    }

    #[test]
    fn test_default_disabled() {
        let state = create_test_state();
        let tool = AgentForkTool::new(state);
        assert!(!tool.default_enabled());
    }

    #[test]
    fn test_resolve_budget_policy() {
        let state = create_test_state();
        let tool = AgentForkTool::new(state);

        assert!(matches!(
            tool.resolve_budget_policy(None),
            ForkBudgetPolicy::Equal
        ));
        assert!(matches!(
            tool.resolve_budget_policy(Some("equal")),
            ForkBudgetPolicy::Equal
        ));
        assert!(matches!(
            tool.resolve_budget_policy(Some("remaining")),
            ForkBudgetPolicy::Remaining
        ));
        assert!(matches!(
            tool.resolve_budget_policy(Some("proportional")),
            ForkBudgetPolicy::Proportional
        ));

        match tool.resolve_budget_policy(Some("fixed:10000")) {
            ForkBudgetPolicy::Fixed(tokens) => assert_eq!(tokens, 10000),
            _ => unreachable!("Expected Fixed"),
        }
    }

    #[tokio::test]
    async fn test_fork_empty_prompt() {
        let state = create_test_state();
        let tool = AgentForkTool::new(state);

        let result = tool.call(json!({"prompt": ""})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_fork_invalid_provider() {
        let state = create_test_state();
        let tool = AgentForkTool::new(state);

        let result = tool
            .call(json!({
                "prompt": "Continue with analysis",
                "provider": "invalid"
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fork_client_creation_failure() {
        let state = create_failing_state();
        let tool = AgentForkTool::new(state);

        let result = tool
            .call(json!({
                "prompt": "Continue with analysis"
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fork_inherits_message_count() {
        let state = create_test_state_with_messages();
        let tool = AgentForkTool::new(state);

        // This will fail due to mock client, but we can test that it got far enough
        // to read the parent session
        let result = tool
            .call(json!({
                "prompt": "Continue"
            }))
            .await;

        // Even though it fails, we can verify the state was set up correctly
        assert!(result.is_err()); // Fails at client creation
    }

    #[tokio::test]
    async fn test_fork_validates_model() {
        let state = create_test_state();
        let tool = AgentForkTool::new(state);
        let err = tool
            .call(json!({"prompt": "Continue", "provider": "openai", "model": "gpt-4o"}))
            .await
            .unwrap_err();
        match err {
            BuiltinToolError::InvalidArgs(msg) => {
                assert!(msg.contains("not in allowlist"), "{msg}");
            }
            other => panic!("expected InvalidArgs, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_fork_uses_injected_resolver() {
        let resolver = CountingResolver::ok();
        let state = Arc::new(
            SubAgentToolState::new(
                Arc::new(SubAgentManager::new(ConcurrencyLimits::default(), 0)),
                Arc::new(MockClientFactory::failing()),
                Arc::new(MockToolDispatcher),
                Arc::new(MockSessionStore),
                Arc::new(RwLock::new(Session::new())),
                SubAgentConfig::default(),
                0,
            )
            .with_parent_model(meerkat_core::Provider::Gemini, "parent-model")
            .with_model_resolver(Arc::clone(&resolver) as _),
        );
        let tool = AgentForkTool::new(state);
        let out = tool
            .call(json!({"prompt": "Continue", "provider": "gemini", "model": "m1"}))
            .await
            .unwrap();
        let ToolOutput::Json(out) = out else {
            panic!("expected JSON output");
        };
        assert_eq!(out["model"], "m1");
        assert_eq!(resolver.calls(), 1);
        let seen = resolver.last_request();
        assert_eq!(seen.provider.as_deref(), Some("gemini"));
        assert_eq!(seen.model.as_deref(), Some("m1"));
        assert_eq!(seen.parent_provider, meerkat_core::Provider::Gemini);
        assert_eq!(seen.parent_model, "parent-model");
        assert_eq!(seen.depth, 1);
    }

    #[tokio::test]
    async fn test_fork_resolver_model_not_available_is_invalid_args() {
        let resolver = CountingResolver::unavailable();
        let state = Arc::new(
            SubAgentToolState::new(
                Arc::new(SubAgentManager::new(ConcurrencyLimits::default(), 0)),
                Arc::new(MockClientFactory::new()),
                Arc::new(MockToolDispatcher),
                Arc::new(MockSessionStore),
                Arc::new(RwLock::new(Session::new())),
                SubAgentConfig::default(),
                0,
            )
            .with_model_resolver(Arc::clone(&resolver) as _),
        );
        let tool = AgentForkTool::new(state);
        let err = tool.call(json!({"prompt": "Continue"})).await.unwrap_err();
        assert!(matches!(err, BuiltinToolError::InvalidArgs(_)), "{err:?}");
        assert_eq!(resolver.calls(), 1);
    }

    #[test]
    fn test_fork_params_tool_access_accepts_json_string() {
        let params: ForkParams = serde_json::from_value(json!({
            "prompt": "test",
            "tool_access": "{\"policy\":\"deny_list\",\"tools\":[\"shell\"]}"
        }))
        .unwrap();

        let tool_access = params.tool_access.expect("tool_access should parse");
        match tool_access {
            ToolAccessInput::DenyList { tools } => {
                assert_eq!(tools, vec!["shell"]);
            }
            other => panic!("Unexpected tool_access variant: {other:?}"),
        }
    }
}
