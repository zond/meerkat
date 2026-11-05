//! Shared harness for the `/compact` and semantic-recall tests: an isolated
//! realm whose shared LLM client replays a script and records every request.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use futures::stream;
use meerkat_client::{LlmClient, LlmDoneOutcome, LlmError, LlmEvent, LlmRequest};
use meerkat_core::event::{AgentEvent, EventEnvelope};
use meerkat_core::{Message, StopReason};
use meerkat_tui::Args;
use meerkat_tui::bootstrap::{self, Runtime};
use meerkat_tui::session::{SessionHandle, SessionOptions};
use parking_lot::Mutex;
use serde_json::Value;
use tempfile::TempDir;
use tokio::sync::mpsc;

/// Upper bound on any await that could hang on a wiring bug.
pub const TIMEOUT: Duration = Duration::from_secs(30);

pub async fn within<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(TIMEOUT, future)
        .await
        .expect("timed out")
}

/// One scripted LLM response.
#[derive(Clone)]
pub enum Step {
    /// Text, then `Done(EndTurn)`.
    Text(&'static str),
    /// A single tool call, then `Done(ToolUse)`.
    Tool {
        id: &'static str,
        name: &'static str,
        args: Value,
    },
    /// A non-retryable error.
    Fail,
}

impl Step {
    fn events(self) -> Vec<Result<LlmEvent, LlmError>> {
        let done = |stop_reason| {
            Ok(LlmEvent::Done {
                outcome: LlmDoneOutcome::Success { stop_reason },
            })
        };
        match self {
            Self::Text(text) => vec![
                Ok(LlmEvent::TextDelta {
                    delta: text.to_string(),
                    meta: None,
                }),
                done(StopReason::EndTurn),
            ],
            Self::Tool { id, name, args } => vec![
                Ok(LlmEvent::ToolCallComplete {
                    id: id.to_string(),
                    name: name.to_string(),
                    args,
                    meta: None,
                }),
                done(StopReason::ToolUse),
            ],
            Self::Fail => vec![Err(LlmError::InvalidRequest {
                message: "scripted failure".to_string(),
            })],
        }
    }
}

type LlmStream<'a> = Pin<Box<dyn futures::Stream<Item = Result<LlmEvent, LlmError>> + Send + 'a>>;

/// Returns one scripted response per call and records every request.
pub struct ScriptedLlmClient {
    script: Vec<Step>,
    next: AtomicUsize,
    requests: Mutex<Vec<LlmRequest>>,
}

impl ScriptedLlmClient {
    pub fn requests(&self) -> Vec<LlmRequest> {
        self.requests.lock().clone()
    }
}

#[async_trait]
impl LlmClient for ScriptedLlmClient {
    fn stream<'a>(&'a self, request: &'a LlmRequest) -> LlmStream<'a> {
        self.requests.lock().push(request.clone());
        let index = self.next.fetch_add(1, Ordering::SeqCst);
        let step = self
            .script
            .get(index)
            .cloned()
            .unwrap_or(Step::Text("script exhausted"));
        Box::pin(stream::iter(step.events()))
    }

    fn provider(&self) -> &'static str {
        "scripted"
    }

    async fn health_check(&self) -> Result<(), LlmError> {
        Ok(())
    }
}

/// A tempdir project, isolated realm and one live session.
pub struct Harness {
    _project_dir: TempDir,
    _state_dir: TempDir,
    pub runtime: Runtime,
    pub session: SessionHandle,
    pub llm: Arc<ScriptedLlmClient>,
}

impl Harness {
    pub async fn new(script: Vec<Step>) -> Self {
        let project_dir = tempfile::tempdir().unwrap();
        let state_dir = tempfile::tempdir().unwrap();
        let project = std::fs::canonicalize(project_dir.path()).unwrap();
        let args = Args::try_parse_from([
            "rkat-tui",
            "--isolated",
            "--state-root",
            state_dir.path().to_str().unwrap(),
            "--context-root",
            project.to_str().unwrap(),
            "--user-config-root",
            project.to_str().unwrap(),
            "--model",
            "claude-sonnet-4-6",
        ])
        .unwrap();
        let llm = Arc::new(ScriptedLlmClient {
            script,
            next: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
        });
        let runtime =
            bootstrap::bootstrap(&args, None, Some(Arc::clone(&llm) as Arc<dyn LlmClient>))
                .await
                .expect("bootstrap");
        let session = SessionHandle::create(&runtime, SessionOptions::default())
            .await
            .expect("create session");
        Self {
            _project_dir: project_dir,
            _state_dir: state_dir,
            runtime,
            session,
            llm,
        }
    }

    /// Run one turn on the live session; returns the final text.
    pub async fn turn(&self, prompt: &str) -> String {
        let (tx, mut rx) = mpsc::channel::<EventEnvelope<AgentEvent>>(4096);
        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let result = within(self.session.spawn_turn(prompt.to_string(), tx))
            .await
            .expect("turn");
        drain.abort();
        result.text
    }
}

/// Indexable text of every message, one per line.
pub fn all_text(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| match m {
            Message::System(s) => s.content.clone(),
            Message::ToolResults { results } => results
                .iter()
                .map(meerkat_core::types::ToolResult::text_content)
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.as_indexable_text(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
