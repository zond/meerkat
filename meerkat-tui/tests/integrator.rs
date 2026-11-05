//! The integrator end to end, headless: effects in, UI messages out, through
//! the real session service with a scripted LLM. Covers session replacement
//! (`/clear`, `/compact`), the model and instruction files across rebuilds,
//! refusing `/model` mid-turn without stalling, and quitting mid-turn.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::{ScriptedLlmClient, Step, all_text, isolated_args, within};
use meerkat::SessionStore;
use meerkat_client::LlmClient;
use meerkat_core::Provider;
use meerkat_core::service::{SessionQuery, SessionService};
use meerkat_core::{Message, SessionId};
use meerkat_tui::bootstrap::credentials::ProviderCredentials;
use meerkat_tui::bootstrap::{self, TuiSessionService};
use meerkat_tui::delivery::PendingId;
use meerkat_tui::models::ModelSource;
use meerkat_tui::runner::testing::{IntegratorHarness, spawn_integrator_with_model_list};
use meerkat_tui::session::load_session;
use meerkat_tui::ui::{Effect, Msg, NoticeLevel, StatusUpdate};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Notify, mpsc};

const OTHER_MODEL: &str = "claude-sonnet-4-5";

struct Rig {
    project: std::path::PathBuf,
    _dirs: (TempDir, TempDir),
    llm: Arc<ScriptedLlmClient>,
    service: Arc<TuiSessionService>,
    store: Arc<dyn SessionStore>,
    effects: mpsc::UnboundedSender<Effect>,
    ui_rx: mpsc::UnboundedReceiver<Msg>,
    first_session: SessionId,
    unsent_path: std::path::PathBuf,
    task: tokio::task::JoinHandle<Option<meerkat_tui::runner::testing::SavedUnsent>>,
}

impl Rig {
    async fn new(script: Vec<Step>) -> Self {
        Self::with_model_list(script, None).await
    }

    async fn with_model_list(script: Vec<Step>, creds: Option<ProviderCredentials>) -> Self {
        let project_dir = tempfile::tempdir().unwrap();
        let state_dir = tempfile::tempdir().unwrap();
        let project = std::fs::canonicalize(project_dir.path()).unwrap();
        let llm = ScriptedLlmClient::new(script);
        let runtime = bootstrap::bootstrap(
            &isolated_args(&project, state_dir.path()),
            None,
            Some(Arc::clone(&llm) as Arc<dyn LlmClient>),
        )
        .await
        .expect("bootstrap");
        let service = Arc::clone(&runtime.service);
        let store = Arc::clone(&runtime.store);
        let IntegratorHarness {
            effects,
            ui_rx,
            mut agent_rx,
            mut scoped_rx,
            first_session,
            unsent_path,
            task,
        } = spawn_integrator_with_model_list(runtime, creds)
            .await
            .expect("integrator");
        // The UI would render these; nobody may let the session task block.
        tokio::spawn(async move { while agent_rx.recv().await.is_some() {} });
        tokio::spawn(async move { while scoped_rx.recv().await.is_some() {} });
        Self {
            project,
            _dirs: (project_dir, state_dir),
            llm,
            service,
            store,
            effects,
            ui_rx,
            first_session,
            unsent_path,
            task,
        }
    }

    fn send(&self, effect: Effect) {
        self.effects.send(effect).unwrap();
    }

    /// The next UI message matching `pred` (earlier ones are skipped).
    async fn expect(&mut self, what: &str, pred: impl Fn(&Msg) -> bool) -> Msg {
        let deadline = tokio::time::Instant::now() + common::TIMEOUT;
        loop {
            match tokio::time::timeout_at(deadline, self.ui_rx.recv()).await {
                Ok(Some(msg)) if pred(&msg) => return msg,
                Ok(Some(_)) => {}
                Ok(None) => panic!("ui channel closed while waiting for {what}"),
                Err(elapsed) => panic!("waiting for {what}: {elapsed}"),
            }
        }
    }

    async fn turn(&mut self, prompt: &str) {
        self.send(Effect::StartTurn(prompt.to_string()));
        match self
            .expect("TurnFinished", |m| matches!(m, Msg::TurnFinished(_)))
            .await
        {
            Msg::TurnFinished(outcome) => {
                assert!(outcome.result.is_ok(), "turn failed: {:?}", outcome.result);
            }
            _ => unreachable!(),
        }
    }

    async fn clear(&mut self) {
        self.send(Effect::NewSession);
        self.expect(
            "ResetTranscript",
            |m| matches!(m, Msg::ResetTranscript(b) if b.is_empty()),
        )
        .await;
    }

    async fn compact(&mut self) {
        self.send(Effect::Compact);
        match self
            .expect("CompactFinished", |m| matches!(m, Msg::CompactFinished(_)))
            .await
        {
            Msg::CompactFinished(result) => result.expect("compaction"),
            _ => unreachable!(),
        }
    }

    async fn live_sessions(&self) -> Vec<SessionId> {
        self.service
            .list(SessionQuery::default())
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.session_id)
            .collect()
    }

    fn last_request(&self) -> meerkat_client::LlmRequest {
        self.llm.requests().pop().expect("an LLM request")
    }
}

fn is_notice(msg: &Msg, level: NoticeLevel, needle: &str) -> bool {
    matches!(msg, Msg::Notice { level: l, text } if *l == level && text.contains(needle))
}

#[tokio::test]
async fn repeated_clear_keeps_exactly_one_live_session() {
    let mut rig = Rig::new(Vec::new()).await;
    rig.turn("hello").await;
    for _ in 0..3 {
        rig.clear().await;
    }
    let live = rig.live_sessions().await;
    assert_eq!(live.len(), 1, "{live:?}");
    assert_ne!(live[0], rig.first_session);
}

#[tokio::test]
async fn model_survives_clear_and_compact() {
    let mut rig = Rig::new(Vec::new()).await;
    rig.send(Effect::SetModel(OTHER_MODEL.to_string()));
    rig.expect(
        "model status",
        |m| matches!(m, Msg::Status(StatusUpdate::Model { model, .. }) if model == OTHER_MODEL),
    )
    .await;
    rig.turn("one").await;
    assert_eq!(rig.last_request().model, OTHER_MODEL);

    rig.clear().await;
    rig.turn("two").await;
    assert_eq!(rig.last_request().model, OTHER_MODEL);

    rig.turn("three").await;
    rig.turn("four").await;
    rig.compact().await;
    assert_eq!(rig.last_request().model, OTHER_MODEL, "summarization model");
    rig.turn("five").await;
    assert_eq!(rig.last_request().model, OTHER_MODEL);
}

#[tokio::test]
async fn instruction_file_added_between_turns_applies_after_clear() {
    let mut rig = Rig::new(Vec::new()).await;
    rig.turn("before").await;
    std::fs::write(
        rig.project.join("AGENTS.local.md"),
        "LOCAL-INSTRUCTION-MARKER: answer in haiku.\n",
    )
    .unwrap();
    rig.clear().await;
    rig.turn("after").await;
    let system = all_text(&rig.last_request().messages);
    assert!(
        system.contains("LOCAL-INSTRUCTION-MARKER"),
        "system prompt lacks AGENTS.local.md: {system}"
    );
}

#[tokio::test]
async fn compact_keeps_the_session_id_and_memory_search() {
    let mut rig = Rig::new(Vec::new()).await;
    let has_memory_search = |rig: &Rig| {
        rig.last_request()
            .tools
            .iter()
            .any(|tool| tool.name == "memory_search")
    };
    rig.turn("one").await;
    assert!(has_memory_search(&rig), "memory_search missing at start");
    rig.turn("two").await;
    rig.turn("three").await;
    rig.compact().await;
    rig.turn("four").await;
    rig.turn("five").await;
    rig.compact().await;
    rig.turn("six").await;
    assert!(
        has_memory_search(&rig),
        "memory_search missing after two compactions"
    );
    assert_eq!(rig.live_sessions().await, vec![rig.first_session.clone()]);
}

#[tokio::test]
async fn set_model_mid_turn_is_refused_without_blocking_injection() {
    let gate = Arc::new(Notify::new());
    let mut rig = Rig::new(vec![Step::Gated(
        Arc::clone(&gate),
        Box::new(Step::Text("done")),
    )])
    .await;
    rig.send(Effect::StartTurn("work".to_string()));
    within(rig.llm.called.notified()).await;

    rig.send(Effect::SetModel(OTHER_MODEL.to_string()));
    rig.expect("model refusal", |m| {
        is_notice(m, NoticeLevel::Warn, "/model is not available")
    })
    .await;
    rig.send(Effect::Inject {
        id: PendingId(1),
        text: "also this".to_string(),
    });
    // The integrator must still be responsive while the turn runs.
    let ack = tokio::time::timeout(
        Duration::from_secs(5),
        rig.expect("inject ack", |m| matches!(m, Msg::InjectAck { .. })),
    )
    .await
    .expect("the integrator stalled");
    assert!(
        matches!(ack, Msg::InjectAck { result: Ok(()), .. }),
        "{ack:?}"
    );

    gate.notify_one();
    rig.expect("TurnFinished", |m| matches!(m, Msg::TurnFinished(_)))
        .await;
    rig.turn("next").await;
    assert_ne!(rig.last_request().model, OTHER_MODEL);
}

#[tokio::test]
async fn quit_mid_turn_interrupts_persists_and_saves_unsent() {
    // Never released: only the interrupt ends the turn.
    let gate = Arc::new(Notify::new());
    let mut rig = Rig::new(vec![Step::Gated(gate, Box::new(Step::Text("never")))]).await;
    rig.send(Effect::StartTurn("FIRST-PROMPT".to_string()));
    within(rig.llm.called.notified()).await;
    rig.send(Effect::Inject {
        id: PendingId(1),
        text: "QUEUED-TEXT".to_string(),
    });
    rig.expect("inject ack", |m| {
        matches!(m, Msg::InjectAck { result: Ok(()), .. })
    })
    .await;
    rig.send(Effect::SaveUnsent(vec![
        (Some(PendingId(1)), "QUEUED-TEXT".to_string()),
        (Some(PendingId(2)), "DEFERRED-TEXT".to_string()),
        (None, "NEXT-PROMPT".to_string()),
    ]));
    rig.send(Effect::Quit);

    let saved = within(&mut rig.task)
        .await
        .unwrap()
        .expect("unsent messages saved");
    assert_eq!(saved.count, 3);
    assert_eq!(saved.path, rig.unsent_path);
    let content = std::fs::read_to_string(&rig.unsent_path).unwrap();
    for text in ["QUEUED-TEXT", "DEFERRED-TEXT", "NEXT-PROMPT"] {
        assert_eq!(content.matches(text).count(), 1, "{text} in {content}");
    }

    // The interrupted turn's prompt was persisted for `--continue`.
    let stored = load_session(&*rig.store, &rig.first_session)
        .await
        .expect("session persisted on quit");
    let users: Vec<String> = stored
        .messages()
        .iter()
        .filter_map(|m| match m {
            Message::User(u) => Some(u.text_content()),
            _ => None,
        })
        .collect();
    assert!(
        users.iter().any(|t| t.contains("FIRST-PROMPT")),
        "{users:?}"
    );
}

/// A stub Anthropic `/v1/models` endpoint answering every request with
/// `body`; returns its base URL and a request counter.
async fn models_stub(body: &'static str) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
            counter.fetch_add(1, Ordering::SeqCst);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });
    (format!("http://{addr}"), hits)
}

#[tokio::test]
async fn model_list_feeds_the_gauge_and_the_picker() {
    let (base, hits) = models_stub(
        r#"{"data":[{"id":"claude-opus-4-6","display_name":"Opus"},{"id":"claude-sonnet-4-6","display_name":"Sonnet","max_input_tokens":123456}],"has_more":false}"#,
    )
    .await;
    let creds = ProviderCredentials {
        provider: Provider::Anthropic,
        api_key: "test-key".to_string(),
        base_url: Some(base),
    };
    let mut rig = Rig::with_model_list(Vec::new(), Some(creds)).await;
    // Start-up fetch: the gauge window comes from the endpoint.
    rig.expect("endpoint window", |m| {
        matches!(m, Msg::Status(StatusUpdate::ContextWindow(Some(123_456))))
    })
    .await;

    rig.send(Effect::RequestModelPicker { refresh: false });
    let Msg::ShowModelPicker(items, warning) = rig
        .expect("picker", |m| matches!(m, Msg::ShowModelPicker(..)))
        .await
    else {
        unreachable!()
    };
    assert!(warning.is_none());
    let ids: Vec<(&str, ModelSource)> = items.iter().map(|e| (e.id.as_str(), e.source)).collect();
    assert_eq!(
        ids,
        vec![
            ("claude-sonnet-4-6", ModelSource::Current),
            ("claude-opus-4-6", ModelSource::Provider),
        ]
    );
    assert_eq!(items[0].context_window, Some(123_456));
    assert_eq!(hits.load(Ordering::SeqCst), 1, "picker reuses the cache");

    rig.send(Effect::RequestModelPicker { refresh: true });
    rig.expect("refreshed picker", |m| {
        matches!(m, Msg::ShowModelPicker(..))
    })
    .await;
    assert_eq!(hits.load(Ordering::SeqCst), 2, "refresh refetches");

    // The endpoint has no window for opus: the catalog estimate is used.
    rig.send(Effect::SetModel("claude-opus-4-6".to_string()));
    rig.expect("fallback window", |m| {
        matches!(
            m,
            Msg::Status(StatusUpdate::ContextWindow(Some(w)) | StatusUpdate::ApproxContextWindow(w))
                if *w != 123_456
        )
    })
    .await;
}

#[tokio::test]
async fn model_picker_without_endpoint_offers_current_with_warning() {
    let mut rig = Rig::new(Vec::new()).await;
    rig.send(Effect::RequestModelPicker { refresh: false });
    let Msg::ShowModelPicker(items, warning) = rig
        .expect("picker", |m| matches!(m, Msg::ShowModelPicker(..)))
        .await
    else {
        unreachable!()
    };
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].id, "claude-sonnet-4-6");
    assert_eq!(items[0].source, ModelSource::Current);
    assert!(warning.is_some());
}
