//! Shared test doubles for sub-agent tool tests.

use async_trait::async_trait;
use meerkat_core::error::AgentError;
use meerkat_core::{
    AgentLlmClient, AssistantBlock, LlmStreamResult, Message, Provider, ResolvedSubAgentModel,
    StopReason, SubAgentModelError, SubAgentModelRequest, SubAgentModelResolver, ToolDef, Usage,
};
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// What [`CountingResolver::describe`] returns.
pub(super) const COUNTING_RESOLVER_DESCRIPTION: &str = "Endpoints - test: any-model";

/// Owned copy of a [`SubAgentModelRequest`].
#[derive(Debug, Clone)]
pub(super) struct SeenRequest {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub parent_provider: Provider,
    pub parent_model: String,
    pub depth: u32,
}

/// Resolver that counts calls and either hands out a [`FixedUsageClient`] or
/// rejects every model.
pub(super) struct CountingResolver {
    reject: bool,
    usage: Usage,
    calls: AtomicUsize,
    last: parking_lot::Mutex<Option<SeenRequest>>,
}

impl CountingResolver {
    pub fn ok() -> Arc<Self> {
        Self::with_usage(Usage::default())
    }

    pub fn with_usage(usage: Usage) -> Arc<Self> {
        Arc::new(Self {
            reject: false,
            usage,
            calls: AtomicUsize::new(0),
            last: parking_lot::Mutex::new(None),
        })
    }

    pub fn unavailable() -> Arc<Self> {
        Arc::new(Self {
            reject: true,
            usage: Usage::default(),
            calls: AtomicUsize::new(0),
            last: parking_lot::Mutex::new(None),
        })
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn last_request(&self) -> SeenRequest {
        self.last
            .lock()
            .clone()
            .unwrap_or_else(|| unreachable!("resolver was never called"))
    }
}

#[async_trait]
impl SubAgentModelResolver for CountingResolver {
    async fn resolve(
        &self,
        req: SubAgentModelRequest<'_>,
    ) -> Result<ResolvedSubAgentModel, SubAgentModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last.lock() = Some(SeenRequest {
            provider: req.provider.map(str::to_string),
            model: req.model.map(str::to_string),
            parent_provider: req.parent_provider,
            parent_model: req.parent_model.to_string(),
            depth: req.depth,
        });
        let model = req.model.unwrap_or("default-model").to_string();
        if self.reject {
            return Err(SubAgentModelError::ModelNotAvailable {
                model,
                endpoint: "test".to_string(),
                hint: "Available: any-model".to_string(),
            });
        }
        Ok(ResolvedSubAgentModel {
            client: Arc::new(FixedUsageClient::new(self.usage.clone())),
            provider: Provider::Other,
            display: format!("test:{model}"),
            model,
        })
    }

    fn describe(&self) -> String {
        COUNTING_RESOLVER_DESCRIPTION.to_string()
    }
}

/// Agent client reporting a fixed usage per call.
///
/// By default every call ends the turn. In `tool_then_fail` mode the first
/// call requests `tool` and every later call fails (non-recoverably).
pub(super) struct FixedUsageClient {
    usage: Usage,
    fail_after_tool: Option<String>,
    calls: AtomicUsize,
}

impl FixedUsageClient {
    pub fn new(usage: Usage) -> Self {
        Self {
            usage,
            fail_after_tool: None,
            calls: AtomicUsize::new(0),
        }
    }

    pub fn tool_then_fail(usage: Usage, tool: &str) -> Self {
        Self {
            usage,
            fail_after_tool: Some(tool.to_string()),
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl AgentLlmClient for FixedUsageClient {
    async fn stream_response(
        &self,
        _messages: &[Message],
        _tools: &[Arc<ToolDef>],
        _max_tokens: u32,
        _temperature: Option<f32>,
        _provider_params: Option<&Value>,
    ) -> Result<LlmStreamResult, AgentError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(tool) = &self.fail_after_tool {
            if call > 0 {
                return Err(AgentError::InternalError("scripted failure".to_string()));
            }
            let args = serde_json::value::RawValue::from_string("{}".to_string())
                .map_err(|e| AgentError::InternalError(e.to_string()))?;
            return Ok(LlmStreamResult::new(
                vec![AssistantBlock::ToolUse {
                    id: "tc_1".to_string(),
                    name: tool.clone(),
                    args,
                    meta: None,
                }],
                StopReason::ToolUse,
                self.usage.clone(),
            ));
        }
        Ok(LlmStreamResult::new(
            Vec::new(),
            StopReason::EndTurn,
            self.usage.clone(),
        ))
    }

    fn provider(&self) -> &'static str {
        "test"
    }
}
