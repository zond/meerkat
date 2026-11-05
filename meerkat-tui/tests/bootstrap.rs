//! Headless bootstrap smoke test: isolated realm, scripted LLM, no network.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use clap::Parser;
use meerkat_client::TestClient;
use meerkat_core::event::{AgentEvent, EventEnvelope};
use meerkat_core::{Message, SessionService};
use meerkat_tui::Args;
use meerkat_tui::bootstrap::{self, Runtime};
use meerkat_tui::session::{SessionHandle, SessionOptions, load_session, resolve_session_ref};
use tokio::sync::mpsc;

async fn runtime(project: &std::path::Path, state: &std::path::Path) -> Runtime {
    let args = Args::try_parse_from([
        "rkat-tui",
        "--isolated",
        "--state-root",
        state.to_str().unwrap(),
        "--context-root",
        project.to_str().unwrap(),
        "--user-config-root",
        project.to_str().unwrap(),
        "--model",
        "claude-sonnet-4-6",
    ])
    .unwrap();
    bootstrap::bootstrap(&args, None, Some(Arc::new(TestClient::default())))
        .await
        .expect("bootstrap")
}

async fn run_turn(handle: &SessionHandle, prompt: &str) -> (String, Vec<AgentEvent>) {
    let (tx, mut rx) = mpsc::channel::<EventEnvelope<AgentEvent>>(256);
    let result = handle
        .spawn_turn(prompt.to_string(), tx)
        .await
        .expect("turn");
    // Events are forwarded asynchronously; drain until the sender is dropped
    // or the stream goes quiet.
    let mut events = Vec::new();
    while let Ok(Some(env)) =
        tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
    {
        events.push(env.payload);
    }
    (result.text, events)
}

#[tokio::test]
async fn bootstrap_session_turn_and_resolve() {
    let project = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("AGENTS.md"), "Be terse.\n").unwrap();
    let state = tempfile::tempdir().unwrap();

    let rt = runtime(project.path(), state.path()).await;
    assert!(rt.scope.realm_id.starts_with("realm-"));
    assert_eq!(
        rt.scope.project_root.as_path(),
        std::fs::canonicalize(project.path()).unwrap()
    );
    assert!(rt.scope.realm_root().starts_with(state.path()));
    assert_eq!(rt.model, "claude-sonnet-4-6");

    // First session: one turn.
    let first = SessionHandle::create(&rt, SessionOptions::default())
        .await
        .unwrap();
    let (text, events) = run_turn(&first, "hi").await;
    assert_eq!(text, "ok");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::TextDelta { .. })),
        "expected a TextDelta event, got {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::TurnCompleted { .. })),
        "expected a TurnCompleted event, got {events:?}"
    );

    // Second session: also one turn so it is persisted and most recent.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let second = SessionHandle::create(&rt, SessionOptions::default())
        .await
        .unwrap();
    assert_ne!(first.id(), second.id());
    let (text, _) = run_turn(&second, "hello again").await;
    assert_eq!(text, "ok");

    let store = rt.store.as_ref();
    assert_eq!(
        &resolve_session_ref(store, "last").await.unwrap(),
        second.id()
    );
    assert_eq!(
        &resolve_session_ref(store, "~0").await.unwrap(),
        second.id()
    );
    assert_eq!(&resolve_session_ref(store, "~1").await.unwrap(), first.id());
    assert!(resolve_session_ref(store, "~2").await.is_err());
    assert_eq!(
        &resolve_session_ref(store, &first.id().to_string())
            .await
            .unwrap(),
        first.id()
    );

    // Shortest prefix that distinguishes the two ids (UUIDv7 share a time prefix).
    let a = first.id().to_string();
    let b = second.id().to_string();
    let common = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    let prefix = &a[..=common];
    assert_eq!(
        &resolve_session_ref(store, prefix).await.unwrap(),
        first.id()
    );
    if common > 0 {
        assert!(resolve_session_ref(store, &a[..common]).await.is_err());
    }
    assert!(resolve_session_ref(store, "zzzz").await.is_err());

    // The persisted session contains the user message.
    let loaded = load_session(store, first.id()).await.unwrap();
    let has_user_hi = loaded.messages().iter().any(|m| match m {
        Message::User(u) => format!("{u:?}").contains("hi"),
        _ => false,
    });
    assert!(has_user_hi, "loaded session lacks the user message");

    // Resuming registers a live session with the same id.
    rt.service
        .archive(first.id())
        .await
        .expect("archive first live session");
    let resumed = SessionHandle::create(
        &rt,
        SessionOptions {
            resume: Some(loaded),
            ..SessionOptions::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(resumed.id(), first.id());
    assert_eq!(resumed.model(), "claude-sonnet-4-6");
}

#[tokio::test]
async fn empty_realm_has_no_last_session() {
    let project = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let rt = runtime(project.path(), state.path()).await;
    let err = resolve_session_ref(rt.store.as_ref(), "last")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no sessions"), "{err}");
}

/// A workspace-local realm: bootstrap with no realm flags, so state lands in
/// `<workspace>/.rkat/tui/`.
async fn workspace_runtime(workspace: &std::path::Path) -> Runtime {
    let args = Args::try_parse_from([
        "rkat-tui",
        "--workspace",
        workspace.to_str().unwrap(),
        "--user-config-root",
        workspace.to_str().unwrap(),
        "--model",
        "claude-sonnet-4-6",
    ])
    .unwrap();
    bootstrap::bootstrap(&args, None, Some(Arc::new(TestClient::default())))
        .await
        .expect("bootstrap")
}

#[tokio::test]
async fn workspace_state_survives_a_rename() {
    let parent = tempfile::tempdir().unwrap();
    let before = parent.path().join("project-a");
    std::fs::create_dir_all(&before).unwrap();
    std::fs::write(before.join("AGENTS.md"), "Be terse.\n").unwrap();

    // Work in the workspace at its first location.
    let rt = workspace_runtime(&before).await;
    assert_eq!(rt.scope.realm_id, "tui");
    assert!(before.join(".rkat/tui/.gitignore").is_file());
    let handle = SessionHandle::create(&rt, SessionOptions::default())
        .await
        .unwrap();
    let (text, _) = run_turn(&handle, "remember me").await;
    assert_eq!(text, "ok");
    let original_id = handle.id().clone();
    rt.service.archive(&original_id).await.unwrap();
    drop(rt);

    // Rename the whole workspace, then continue the latest session there.
    let after = parent.path().join("project-b");
    std::fs::rename(&before, &after).unwrap();
    let rt = workspace_runtime(&after).await;
    let last = resolve_session_ref(&*rt.store, "last").await.unwrap();
    assert_eq!(last, original_id);
    let session = load_session(&*rt.store, &last).await.unwrap();
    assert!(session.messages().iter().any(|m| matches!(
        m,
        Message::User(u) if u.text_content() == "remember me"
    )));
    let resumed = SessionHandle::create(
        &rt,
        SessionOptions {
            resume: Some(session),
            ..SessionOptions::default()
        },
    )
    .await
    .unwrap();
    let (text, _) = run_turn(&resumed, "still here?").await;
    assert_eq!(text, "ok");
}
