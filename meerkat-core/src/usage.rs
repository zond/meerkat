//! Per-call LLM usage metering.
//!
//! [`MeteredLlmClient`] decorates an [`AgentLlmClient`] and reports the exact
//! [`Usage`] of every completed LLM call to a [`UsageObserver`], tagged with
//! the caller's stream scope and model. [`UsageLedger`] is an in-memory
//! observer that aggregates usage per `(scope_id, model)`.
//!
//! Used to account for sub-agents: their usage never reaches the parent's
//! session or budget, so surfaces attach an observer to see it.

use crate::agent::{AgentLlmClient, LlmStreamResult};
use crate::error::AgentError;
use crate::event::{ScopedAgentEvent, StreamScopeFrame};
use crate::schema::{CompiledSchema, SchemaError};
use crate::types::{Message, OutputSchema, ToolDef, Usage};
use async_trait::async_trait;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Receives the usage of each completed LLM call.
///
/// Called synchronously on the agent's task; implementations must be cheap
/// and must not block (forward to a channel if more work is needed).
pub trait UsageObserver: Send + Sync {
    /// One LLM call made under `scope` with `model` consumed `usage`.
    fn on_usage(&self, scope: &[StreamScopeFrame], model: &str, usage: &Usage);
}

/// [`AgentLlmClient`] decorator that reports each response's usage.
///
/// Failed calls report nothing (no usage is available for them).
pub struct MeteredLlmClient<C: AgentLlmClient + ?Sized> {
    inner: Arc<C>,
    observer: Arc<dyn UsageObserver>,
    scope: Arc<[StreamScopeFrame]>,
    model: String,
}

impl<C: AgentLlmClient + ?Sized> MeteredLlmClient<C> {
    /// Wrap `inner`, attributing its usage to `scope` and `model`.
    pub fn new(
        inner: Arc<C>,
        observer: Arc<dyn UsageObserver>,
        scope: impl Into<Arc<[StreamScopeFrame]>>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            observer,
            scope: scope.into(),
            model: model.into(),
        }
    }

    /// Scope path the usage is attributed to.
    pub fn scope(&self) -> &[StreamScopeFrame] {
        &self.scope
    }

    /// Model name the usage is attributed to.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl<C: AgentLlmClient + ?Sized> std::fmt::Debug for MeteredLlmClient<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeteredLlmClient")
            .field("scope", &self.scope)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<C: AgentLlmClient + ?Sized> AgentLlmClient for MeteredLlmClient<C> {
    async fn stream_response(
        &self,
        messages: &[Message],
        tools: &[Arc<ToolDef>],
        max_tokens: u32,
        temperature: Option<f32>,
        provider_params: Option<&Value>,
    ) -> Result<LlmStreamResult, AgentError> {
        let result = self
            .inner
            .stream_response(messages, tools, max_tokens, temperature, provider_params)
            .await?;
        self.observer
            .on_usage(&self.scope, &self.model, result.usage());
        Ok(result)
    }

    fn provider(&self) -> &'static str {
        self.inner.provider()
    }

    fn compile_schema(&self, output_schema: &OutputSchema) -> Result<CompiledSchema, SchemaError> {
        self.inner.compile_schema(output_schema)
    }
}

/// One aggregated row of a [`UsageLedger`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageEntry {
    /// Canonical scope id (see [`ScopedAgentEvent::scope_id_from_path`]).
    pub scope_id: String,
    /// Model the usage was consumed by.
    pub model: String,
    /// Accumulated usage.
    pub usage: Usage,
}

/// In-memory [`UsageObserver`] aggregating usage per `(scope_id, model)`.
///
/// Rows keep first-seen order. Usage of children that later fail or are
/// cancelled stays in the ledger.
#[derive(Debug, Default)]
pub struct UsageLedger {
    entries: parking_lot::Mutex<IndexMap<(String, String), Usage>>,
}

impl UsageLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy of all rows in first-seen order.
    pub fn snapshot(&self) -> Vec<UsageEntry> {
        self.entries
            .lock()
            .iter()
            .map(|((scope_id, model), usage)| UsageEntry {
                scope_id: scope_id.clone(),
                model: model.clone(),
                usage: usage.clone(),
            })
            .collect()
    }

    /// Sum of all rows.
    pub fn total(&self) -> Usage {
        let mut total = Usage::default();
        for usage in self.entries.lock().values() {
            total.add(usage);
        }
        total
    }
}

impl UsageObserver for UsageLedger {
    fn on_usage(&self, scope: &[StreamScopeFrame], model: &str, usage: &Usage) {
        let key = (
            ScopedAgentEvent::scope_id_from_path(scope),
            model.to_string(),
        );
        self.entries.lock().entry(key).or_default().add(usage);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::types::StopReason;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FixedUsageClient {
        usage: Usage,
        fail: bool,
        calls: AtomicUsize,
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
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(AgentError::InternalError("boom".to_string()));
            }
            Ok(LlmStreamResult::new(
                Vec::new(),
                StopReason::EndTurn,
                self.usage.clone(),
            ))
        }

        fn provider(&self) -> &'static str {
            "fixed"
        }
    }

    #[derive(Default)]
    struct Recorder {
        calls: parking_lot::Mutex<Vec<(Vec<StreamScopeFrame>, String, Usage)>>,
    }

    impl UsageObserver for Recorder {
        fn on_usage(&self, scope: &[StreamScopeFrame], model: &str, usage: &Usage) {
            self.calls
                .lock()
                .push((scope.to_vec(), model.to_string(), usage.clone()));
        }
    }

    fn usage() -> Usage {
        Usage {
            input_tokens: 120,
            output_tokens: 34,
            cache_creation_tokens: Some(56),
            cache_read_tokens: Some(78),
        }
    }

    fn child_scope() -> Vec<StreamScopeFrame> {
        vec![
            StreamScopeFrame::Primary {
                session_id: "s1".to_string(),
            },
            StreamScopeFrame::SubAgent {
                agent_id: "a1".to_string(),
                tool_call_id: None,
                label: Some("child".to_string()),
            },
        ]
    }

    #[tokio::test]
    async fn metered_client_reports_exact_usage_scope_and_model() {
        let inner = Arc::new(FixedUsageClient {
            usage: usage(),
            fail: false,
            calls: AtomicUsize::new(0),
        });
        let recorder = Arc::new(Recorder::default());
        let client = MeteredLlmClient::new(
            inner,
            Arc::clone(&recorder) as Arc<dyn UsageObserver>,
            child_scope(),
            "model-x",
        );

        let result = client
            .stream_response(&[], &[], 100, None, None)
            .await
            .unwrap();
        assert_eq!(result.usage(), &usage());
        assert_eq!(client.provider(), "fixed");

        let calls = recorder.calls.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, child_scope());
        assert_eq!(calls[0].1, "model-x");
        assert_eq!(calls[0].2, usage());
    }

    #[tokio::test]
    async fn metered_client_reports_nothing_on_failure() {
        let inner = Arc::new(FixedUsageClient {
            usage: usage(),
            fail: true,
            calls: AtomicUsize::new(0),
        });
        let recorder = Arc::new(Recorder::default());
        let client = MeteredLlmClient::new(
            Arc::clone(&inner),
            Arc::clone(&recorder) as Arc<dyn UsageObserver>,
            child_scope(),
            "model-x",
        );
        assert!(
            client
                .stream_response(&[], &[], 100, None, None)
                .await
                .is_err()
        );
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
        assert!(recorder.calls.lock().is_empty());
    }

    #[tokio::test]
    async fn metered_client_works_over_trait_objects() {
        let inner: Arc<dyn AgentLlmClient> = Arc::new(FixedUsageClient {
            usage: usage(),
            fail: false,
            calls: AtomicUsize::new(0),
        });
        let ledger = Arc::new(UsageLedger::new());
        let client: Arc<dyn AgentLlmClient> = Arc::new(MeteredLlmClient::new(
            inner,
            Arc::clone(&ledger) as Arc<dyn UsageObserver>,
            child_scope(),
            "model-x",
        ));
        client
            .stream_response(&[], &[], 100, None, None)
            .await
            .unwrap();
        client
            .stream_response(&[], &[], 100, None, None)
            .await
            .unwrap();
        let rows = ledger.snapshot();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].scope_id, "primary/sub:a1");
        assert_eq!(
            rows[0].usage,
            Usage {
                input_tokens: 240,
                output_tokens: 68,
                cache_creation_tokens: Some(112),
                cache_read_tokens: Some(156),
            }
        );
    }

    #[test]
    fn ledger_aggregates_per_scope_and_model_in_first_seen_order() {
        let ledger = UsageLedger::new();
        let other_scope = vec![StreamScopeFrame::Primary {
            session_id: "s1".to_string(),
        }];
        ledger.on_usage(&child_scope(), "b", &usage());
        ledger.on_usage(&other_scope, "a", &usage());
        ledger.on_usage(&child_scope(), "a", &usage());
        ledger.on_usage(&child_scope(), "b", &usage());

        let rows = ledger.snapshot();
        let keys: Vec<_> = rows
            .iter()
            .map(|r| (r.scope_id.as_str(), r.model.as_str()))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("primary/sub:a1", "b"),
                ("primary", "a"),
                ("primary/sub:a1", "a")
            ]
        );
        assert_eq!(rows[0].usage.input_tokens, 240);
        assert_eq!(ledger.total().input_tokens, 480);
        assert_eq!(ledger.total().cache_read_tokens, Some(312));
    }

    #[test]
    fn ledger_snapshot_serde_round_trip() {
        let ledger = UsageLedger::new();
        ledger.on_usage(&child_scope(), "model-x", &usage());
        let snapshot = ledger.snapshot();
        let json = serde_json::to_string(&snapshot).unwrap();
        let back: Vec<UsageEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, snapshot);
        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value[0]["scope_id"], "primary/sub:a1");
        assert_eq!(value[0]["usage"]["cache_creation_tokens"], 56);
    }
}
