//! `build_agent` threads the sub-agent model resolver and usage observer from
//! `AgentBuildConfig` into the `agent_spawn` tool.

#![cfg(feature = "sub-agents")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use meerkat::{
    AgentBuildConfig, AgentFactory, LlmClient, LlmDoneOutcome, LlmEvent, LlmRequest,
    ResolvedSubAgentModel, SubAgentModelError, SubAgentModelRequest, SubAgentModelResolver,
    UsageLedger, UsageObserver, adapt_llm_client,
};
use meerkat_core::{Config, Provider, StopReason, Usage};

/// Primary: first call spawns a child, then ends the turn. Child: reports
/// usage and ends the turn.
struct ScriptedClient {
    calls: AtomicUsize,
    spawn_first: bool,
}

#[async_trait]
impl LlmClient for ScriptedClient {
    fn stream<'a>(&'a self, _request: &'a LlmRequest) -> meerkat_client::types::LlmStream<'a> {
        let first = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
        let mut events = Vec::new();
        if first && self.spawn_first {
            events.push(Ok(LlmEvent::ToolCallComplete {
                id: "tc_spawn".to_string(),
                name: "agent_spawn".to_string(),
                args: serde_json::json!({
                    "prompt": "child task",
                    "provider": "litellm",
                    "model": "child-model",
                }),
                meta: None,
            }));
            events.push(Ok(LlmEvent::Done {
                outcome: LlmDoneOutcome::Success {
                    stop_reason: StopReason::ToolUse,
                },
            }));
        } else {
            events.push(Ok(LlmEvent::TextDelta {
                delta: "done".to_string(),
                meta: None,
            }));
            if !self.spawn_first {
                events.push(Ok(LlmEvent::UsageUpdate {
                    usage: Usage {
                        input_tokens: 11,
                        output_tokens: 7,
                        cache_creation_tokens: Some(3),
                        cache_read_tokens: Some(5),
                    },
                }));
            }
            events.push(Ok(LlmEvent::Done {
                outcome: LlmDoneOutcome::Success {
                    stop_reason: StopReason::EndTurn,
                },
            }));
        }
        Box::pin(stream::iter(events))
    }

    fn provider(&self) -> &'static str {
        "mock"
    }

    async fn health_check(&self) -> Result<(), meerkat_client::LlmError> {
        Ok(())
    }
}

/// `(provider, model, parent_provider, parent_model, depth)` of one request.
type SeenRequest = (Option<String>, Option<String>, Provider, String, u32);

#[derive(Default)]
struct RecordingResolver {
    seen: std::sync::Mutex<Vec<SeenRequest>>,
}

#[async_trait]
impl SubAgentModelResolver for RecordingResolver {
    async fn resolve(
        &self,
        req: SubAgentModelRequest<'_>,
    ) -> Result<ResolvedSubAgentModel, SubAgentModelError> {
        self.seen.lock().unwrap().push((
            req.provider.map(str::to_string),
            req.model.map(str::to_string),
            req.parent_provider,
            req.parent_model.to_string(),
            req.depth,
        ));
        let model = req.model.unwrap_or("default").to_string();
        let client: Arc<dyn LlmClient> = Arc::new(ScriptedClient {
            calls: AtomicUsize::new(0),
            spawn_first: false,
        });
        Ok(ResolvedSubAgentModel {
            client: adapt_llm_client(client, model.clone()),
            provider: Provider::Other,
            display: format!("litellm:{model}"),
            model,
        })
    }

    fn describe(&self) -> String {
        "Endpoints - litellm".to_string()
    }
}

#[tokio::test]
async fn build_agent_wires_model_resolver_and_usage_observer_into_sub_agents() {
    let temp = tempfile::tempdir().unwrap();
    let factory = AgentFactory::new(temp.path().join("sessions"))
        .builtins(true)
        .subagents(true);
    let resolver = Arc::new(RecordingResolver::default());
    let ledger = Arc::new(UsageLedger::new());

    let mut build = AgentBuildConfig::new("claude-sonnet-4-5");
    build.llm_client_override = Some(Arc::new(ScriptedClient {
        calls: AtomicUsize::new(0),
        spawn_first: true,
    }));
    build.sub_agent_model_resolver = Some(Arc::clone(&resolver) as Arc<dyn SubAgentModelResolver>);
    build.usage_observer = Some(Arc::clone(&ledger) as Arc<dyn UsageObserver>);

    let mut agent = factory
        .build_agent(build, &Config::default())
        .await
        .unwrap();
    let result = agent.run("delegate".to_string().into()).await.unwrap();
    assert_eq!(result.text, "done");

    tokio::time::timeout(Duration::from_secs(10), async {
        while ledger.snapshot().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("child usage metered");

    let seen = resolver.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![(
            Some("litellm".to_string()),
            Some("child-model".to_string()),
            Provider::Anthropic,
            "claude-sonnet-4-5".to_string(),
            1,
        )]
    );

    // Only the child is metered; the primary is accounted elsewhere.
    let rows = ledger.snapshot();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0].scope_id.starts_with("primary/sub:"), "{rows:?}");
    assert_eq!(rows[0].model, "child-model");
    assert_eq!(
        rows[0].usage,
        Usage {
            input_tokens: 11,
            output_tokens: 7,
            cache_creation_tokens: Some(3),
            cache_read_tokens: Some(5),
        }
    );
}
