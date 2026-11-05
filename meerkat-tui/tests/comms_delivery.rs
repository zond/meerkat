//! Mid-turn message delivery through the in-process comms inbox, end to end
//! through the real session service with a scripted, gateable LLM and a
//! gateable tool: delivery at a tool boundary, the final-boundary edge case,
//! idle injection, recall, and the absence of comms side effects.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use futures::{StreamExt, stream};
use meerkat::ToolGatewayBuilder;
use meerkat_client::{LlmClient, LlmDoneOutcome, LlmError, LlmEvent, LlmRequest};
use meerkat_core::error::ToolError;
use meerkat_core::types::{
    AssistantMessage, ToolCallView, ToolDef, ToolResult, Usage, UserMessage,
};
use meerkat_core::{AgentToolDispatcher, Message, RunResult, Session, StopReason};
use meerkat_tui::Args;
use meerkat_tui::bootstrap::{self, Runtime};
use meerkat_tui::delivery::{DeliveryReport, FRAME, PendingId, frame};
use meerkat_tui::session::delivery::{recall, reconcile};
use meerkat_tui::session::{SessionHandle, SessionOptions};
use meerkat_tui::tools::FileToolDispatcher;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;

/// Upper bound on any await that could hang on a wiring bug.
const TIMEOUT: Duration = Duration::from_secs(30);

const GATE_TOOL: &str = "gate_tool";

#[tokio::test]
async fn message_injected_during_a_tool_reaches_the_next_llm_call() {
    let h = Harness::new(vec![
        Step::tool("call-gate", GATE_TOOL, json!({})),
        Step::Text("got it"),
    ])
    .await;

    let turn = h.spawn_turn("start");
    within(h.gate.entered.notified()).await;
    let framed = frame("hello");
    within(h.session.inject(framed.clone())).await.unwrap();
    h.gate.release.notify_one();
    within(turn).await.unwrap().expect("turn failed");

    let requests = h.llm.requests();
    assert_eq!(requests.len(), 2, "expected two LLM calls");
    assert!(
        has_user_containing(&requests[1].messages, &format!("{FRAME}hello")),
        "second request lacks the injected message: {:?}",
        requests[1].messages
    );

    let mut inflight = vec![(PendingId(1), framed)];
    let report = within(reconcile(&h.session, &mut inflight)).await;
    assert_eq!(
        report,
        DeliveryReport {
            answered: vec![PendingId(1)],
            ..DeliveryReport::default()
        }
    );
    assert!(inflight.is_empty());
}

#[tokio::test]
async fn message_injected_during_the_final_llm_call_is_recorded_unanswered() {
    let llm_gate = Arc::new(Notify::new());
    let h = Harness::new(vec![Step::Gated(
        Arc::clone(&llm_gate),
        Box::new(Step::Text("all done")),
    )])
    .await;

    let turn = h.spawn_turn("start");
    within(h.llm.called.notified()).await;
    let framed = frame("one more thing");
    within(h.session.inject(framed.clone())).await.unwrap();
    llm_gate.notify_one();
    within(turn).await.unwrap().expect("turn failed");

    let session = within(h.session.export()).await.unwrap();
    let last = session.messages().last().expect("session has messages");
    assert!(
        matches!(last, Message::User(user) if user.text_content().contains(&framed)),
        "expected the injected message last, got {last:?}"
    );

    let mut inflight = vec![(PendingId(7), framed)];
    let report = within(reconcile(&h.session, &mut inflight)).await;
    assert_eq!(
        report,
        DeliveryReport {
            unanswered: vec![PendingId(7)],
            ..DeliveryReport::default()
        }
    );
}

#[tokio::test]
async fn message_injected_while_idle_is_undelivered_and_drained() {
    let h = Harness::new(vec![Step::Text("unused")]).await;

    let framed = frame("anyone there?");
    within(h.session.inject(framed.clone())).await.unwrap();
    let mut inflight = vec![(PendingId(3), framed)];
    let report = within(reconcile(&h.session, &mut inflight)).await;
    assert_eq!(
        report,
        DeliveryReport {
            undelivered: vec![PendingId(3)],
            ..DeliveryReport::default()
        }
    );
    assert!(within(h.session.drain_inbox()).await.is_empty());
    assert!(
        h.llm.requests().is_empty(),
        "nothing may wake an idle agent"
    );
}

#[tokio::test]
async fn recalled_message_never_reaches_the_model() {
    let h = Harness::new(vec![
        Step::tool("call-gate", GATE_TOOL, json!({})),
        Step::Text("done"),
    ])
    .await;

    let turn = h.spawn_turn("start");
    within(h.gate.entered.notified()).await;
    let keep = frame("keep me");
    let take = frame("take me back");
    within(h.session.inject(keep.clone())).await.unwrap();
    within(h.session.inject(take.clone())).await.unwrap();
    let mut inflight = vec![(PendingId(1), keep.clone()), (PendingId(2), take.clone())];
    // Recall drains the inbox: both come back, in injection order.
    let recalled = within(recall(&h.session, &mut inflight)).await;
    assert_eq!(recalled, vec![PendingId(1), PendingId(2)]);
    assert!(inflight.is_empty());
    h.gate.release.notify_one();
    within(turn).await.unwrap().expect("turn failed");

    let requests = h.llm.requests();
    assert_eq!(requests.len(), 2);
    for body in [&keep, &take] {
        assert!(
            !has_user_containing(&requests[1].messages, body),
            "recalled message reached the model: {:?}",
            requests[1].messages
        );
    }
    let report = within(reconcile(&h.session, &mut inflight)).await;
    assert!(report.is_empty());
}

#[tokio::test]
async fn comms_adds_no_peer_tools_and_pre_comms_sessions_resume() {
    let h = Harness::new(vec![Step::Text("first"), Step::Text("second")]).await;
    within(h.spawn_turn("hi"))
        .await
        .unwrap()
        .expect("turn failed");

    let requests = h.llm.requests();
    let names: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&GATE_TOOL), "{names:?}");
    assert!(names.contains(&"read_file"), "{names:?}");
    for hidden in ["send", "peers"] {
        assert!(!names.contains(&hidden), "{hidden} exposed: {names:?}");
    }

    // A session persisted before comms existed: no comms metadata at all.
    let mut old = Session::new();
    old.push(Message::User(UserMessage::text("earlier question")));
    old.push(Message::Assistant(AssistantMessage {
        content: "earlier answer".into(),
        tool_calls: Vec::new(),
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
    }));
    let resumed = SessionHandle::create(
        &h.runtime,
        SessionOptions {
            resume: Some(old.clone()),
            external_tools: Some(Arc::clone(&h.tools)),
            ..SessionOptions::default()
        },
    )
    .await
    .expect("resume pre-comms session");
    assert_eq!(resumed.id(), old.id());
    let (tx, _rx) = mpsc::channel(4096);
    within(resumed.start_turn("and now?".into(), tx))
        .await
        .expect("resumed turn failed");
    // The rebuilt session has an inbox of its own.
    within(resumed.inject(frame("ping"))).await.unwrap();
    assert_eq!(within(resumed.drain_inbox()).await, vec![frame("ping")]);
    let requests = h.llm.requests();
    assert!(has_user_containing(
        &requests[1].messages,
        "earlier question"
    ));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn within<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(TIMEOUT, future)
        .await
        .expect("timed out")
}

fn has_user_containing(messages: &[Message], needle: &str) -> bool {
    messages
        .iter()
        .any(|m| matches!(m, Message::User(user) if user.text_content().contains(needle)))
}

/// One scripted LLM response.
#[derive(Clone)]
enum Step {
    /// A single tool call, then `Done(ToolUse)`.
    Tool {
        id: &'static str,
        name: &'static str,
        args: Value,
    },
    /// Text, then `Done(EndTurn)`.
    Text(&'static str),
    /// The inner step, streamed only once the notify fires.
    Gated(Arc<Notify>, Box<Step>),
}

impl Step {
    fn tool(id: &'static str, name: &'static str, args: Value) -> Self {
        Self::Tool { id, name, args }
    }

    fn events(self) -> (Option<Arc<Notify>>, Vec<Result<LlmEvent, LlmError>>) {
        match self {
            Self::Tool { id, name, args } => (
                None,
                vec![
                    Ok(LlmEvent::ToolCallComplete {
                        id: id.to_string(),
                        name: name.to_string(),
                        args,
                        meta: None,
                    }),
                    done(StopReason::ToolUse),
                ],
            ),
            Self::Text(text) => (
                None,
                vec![
                    Ok(LlmEvent::TextDelta {
                        delta: text.to_string(),
                        meta: None,
                    }),
                    done(StopReason::EndTurn),
                ],
            ),
            Self::Gated(gate, inner) => (Some(gate), inner.events().1),
        }
    }
}

fn done(stop_reason: StopReason) -> Result<LlmEvent, LlmError> {
    Ok(LlmEvent::Done {
        outcome: LlmDoneOutcome::Success { stop_reason },
    })
}

type LlmStream<'a> = Pin<Box<dyn futures::Stream<Item = Result<LlmEvent, LlmError>> + Send + 'a>>;

/// Returns one scripted response per call and records every request.
struct ScriptedLlmClient {
    script: Vec<Step>,
    next: AtomicUsize,
    requests: Mutex<Vec<LlmRequest>>,
    /// Fires on every call, after the request is recorded.
    called: Notify,
}

impl ScriptedLlmClient {
    fn requests(&self) -> Vec<LlmRequest> {
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
        self.called.notify_one();
        let (gate, events) = step.events();
        Box::pin(
            stream::once(async move {
                if let Some(gate) = gate {
                    gate.notified().await;
                }
                stream::iter(events)
            })
            .flatten(),
        )
    }

    fn provider(&self) -> &'static str {
        "scripted"
    }

    async fn health_check(&self) -> Result<(), LlmError> {
        Ok(())
    }
}

/// A tool that signals `entered` and then blocks until `release` fires.
struct GateTool {
    defs: Arc<[Arc<ToolDef>]>,
    entered: Notify,
    release: Notify,
}

impl GateTool {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            defs: Arc::from(vec![Arc::new(ToolDef {
                name: GATE_TOOL.to_string(),
                description: "Blocks until the test releases it.".to_string(),
                input_schema: json!({"type": "object", "properties": {}}),
            })]),
            entered: Notify::new(),
            release: Notify::new(),
        })
    }
}

#[async_trait]
impl AgentToolDispatcher for GateTool {
    fn tools(&self) -> Arc<[Arc<ToolDef>]> {
        Arc::clone(&self.defs)
    }

    async fn dispatch(&self, call: ToolCallView<'_>) -> Result<ToolResult, ToolError> {
        if call.name != GATE_TOOL {
            return Err(ToolError::NotFound {
                name: call.name.to_string(),
            });
        }
        self.entered.notify_one();
        self.release.notified().await;
        Ok(ToolResult::new(
            call.id.to_string(),
            "released".to_string(),
            false,
        ))
    }
}

/// A tempdir project, isolated realm and live session (no approval engine).
struct Harness {
    _project_dir: TempDir,
    _state_dir: TempDir,
    runtime: Runtime,
    session: SessionHandle,
    llm: Arc<ScriptedLlmClient>,
    gate: Arc<GateTool>,
    tools: Arc<dyn AgentToolDispatcher>,
}

impl Harness {
    async fn new(script: Vec<Step>) -> Self {
        let project_dir = tempfile::tempdir().unwrap();
        let state_dir = tempfile::tempdir().unwrap();
        let project = std::fs::canonicalize(project_dir.path()).unwrap();
        std::fs::create_dir_all(project.join(".rkat")).unwrap();

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
            called: Notify::new(),
        });
        let runtime =
            bootstrap::bootstrap(&args, None, Some(Arc::clone(&llm) as Arc<dyn LlmClient>))
                .await
                .expect("bootstrap");

        let gate = GateTool::new();
        let file_tools = Arc::new(FileToolDispatcher::new(
            runtime.scope.project_root.clone(),
            None,
        ));
        let tools: Arc<dyn AgentToolDispatcher> = Arc::new(
            ToolGatewayBuilder::new()
                .add_dispatcher(file_tools)
                .add_dispatcher(Arc::clone(&gate) as Arc<dyn AgentToolDispatcher>)
                .build()
                .expect("compose tools"),
        );
        let session = SessionHandle::create(
            &runtime,
            SessionOptions {
                external_tools: Some(Arc::clone(&tools)),
                ..SessionOptions::default()
            },
        )
        .await
        .expect("create session");

        Self {
            _project_dir: project_dir,
            _state_dir: state_dir,
            runtime,
            session,
            llm,
            gate,
            tools,
        }
    }

    /// Start a turn in the background; events are discarded.
    fn spawn_turn(
        &self,
        prompt: &str,
    ) -> JoinHandle<Result<RunResult, meerkat_core::service::SessionError>> {
        let session = self.session.clone();
        let prompt = prompt.to_string();
        tokio::spawn(async move {
            let (tx, mut rx) = mpsc::channel(4096);
            tokio::spawn(async move { while rx.recv().await.is_some() {} });
            session.start_turn(prompt, tx).await
        })
    }
}
