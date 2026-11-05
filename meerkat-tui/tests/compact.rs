//! `/compact` building blocks end to end through the real session service:
//! summarization with the shared LLM client, the rebuilt history, the saved
//! session, and the refusal and failure paths that leave everything untouched.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{Harness, Step, all_text, within};
use meerkat_core::{AgentEvent, Message};
use meerkat_tui::session::lifecycle::{CompactError, MANUAL_RECENT_TURNS, compact_session};
use meerkat_tui::session::load_session;
use tokio::sync::mpsc;

const SUMMARY: &str = "SUMMARY-OF-EARLY-TURNS";

fn four_turns_then_summary() -> Vec<Step> {
    vec![
        Step::Text("reply one"),
        Step::Text("reply two"),
        Step::Text("reply three"),
        Step::Text("reply four"),
        Step::Text(SUMMARY),
    ]
}

async fn four_turns(h: &Harness) {
    for prompt in [
        "turn-one ALPHA",
        "turn-two BRAVO",
        "turn-three CHARLIE",
        "turn-four DELTA",
    ] {
        h.turn(prompt).await;
    }
}

fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::User(u) => Some(u.text_content()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn compact_summarizes_saves_and_keeps_recent_turns() {
    let h = Harness::new(four_turns_then_summary()).await;
    four_turns(&h).await;
    let before = h.session.export().await.unwrap();

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(16);
    let compacted = within(compact_session(
        &h.runtime,
        &h.session,
        h.session.model(),
        1234,
        Some(tx),
    ))
    .await
    .expect("compact");
    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

    // The fifth call is the summarization request and sees the early turns.
    let requests = h.llm.requests();
    assert_eq!(requests.len(), 5);
    let summarize = all_text(&requests[4].messages);
    assert!(summarize.contains("turn-one ALPHA"), "{summarize}");

    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::CompactionStarted { .. })),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::CompactionCompleted { .. })),
        "{events:?}"
    );

    // Same session id; summary first, then only the recent turns verbatim.
    assert_eq!(compacted.session.id(), h.session.id());
    assert_eq!(compacted.messages_before, before.messages().len());
    let messages = compacted.session.messages();
    assert!(messages.len() < compacted.messages_before);
    assert!(matches!(messages.first(), Some(Message::System(_))));
    let users = user_texts(messages);
    assert_eq!(users.len(), 1 + MANUAL_RECENT_TURNS, "{users:?}");
    assert!(users[0].contains(SUMMARY), "{users:?}");
    assert_eq!(users[1], "turn-three CHARLIE");
    assert_eq!(users[2], "turn-four DELTA");

    // The discarded part is exactly the early turns (for semantic indexing).
    let discarded = all_text(&compacted.discarded);
    assert!(discarded.contains("turn-one ALPHA"), "{discarded}");
    assert!(discarded.contains("reply two"), "{discarded}");
    assert!(!discarded.contains("CHARLIE"), "{discarded}");

    // Saved to the realm store under the same id.
    let stored = load_session(&*h.runtime.store, h.session.id())
        .await
        .unwrap();
    assert_eq!(user_texts(stored.messages()), users);

    // The live session itself is untouched; the integrator rebuilds it.
    let live = h.session.export().await.unwrap();
    assert_eq!(live.messages().len(), compacted.messages_before);
}

#[tokio::test]
async fn compact_refuses_a_short_history_without_calling_the_llm() {
    let h = Harness::new(vec![Step::Text("only reply")]).await;
    h.turn("just one turn").await;
    let err = compact_session(&h.runtime, &h.session, h.session.model(), 0, None)
        .await
        .expect_err("too short");
    assert!(matches!(err, CompactError::TooShort(_)), "{err}");
    assert_eq!(h.llm.requests().len(), 1);
}

#[tokio::test]
async fn compact_with_only_recent_turns_has_no_gain() {
    let h = Harness::new(vec![
        Step::Text("reply one"),
        Step::Text("reply two"),
        Step::Text(SUMMARY),
    ])
    .await;
    h.turn("turn-one").await;
    h.turn("turn-two").await;
    let before = load_session(&*h.runtime.store, h.session.id()).await.ok();
    let err = compact_session(&h.runtime, &h.session, h.session.model(), 0, None)
        .await
        .expect_err("no gain");
    assert!(matches!(err, CompactError::NoGain(_)), "{err}");
    let after = load_session(&*h.runtime.store, h.session.id()).await.ok();
    assert_eq!(
        before.map(|s| s.messages().len()),
        after.map(|s| s.messages().len()),
        "nothing may be saved when compaction does not shrink the history"
    );
}

#[tokio::test]
async fn failed_summarization_leaves_the_session_and_store_untouched() {
    let mut script = four_turns_then_summary();
    *script.last_mut().unwrap() = Step::Fail;
    let h = Harness::new(script).await;
    four_turns(&h).await;
    let stored_before = load_session(&*h.runtime.store, h.session.id())
        .await
        .ok()
        .map(|s| s.messages().len());

    let err = within(compact_session(
        &h.runtime,
        &h.session,
        h.session.model(),
        0,
        None,
    ))
    .await
    .expect_err("summarization fails");
    assert!(matches!(err, CompactError::Compaction(_)), "{err}");

    let stored_after = load_session(&*h.runtime.store, h.session.id())
        .await
        .ok()
        .map(|s| s.messages().len());
    assert_eq!(stored_before, stored_after);
    let live = h.session.export().await.unwrap();
    assert_eq!(user_texts(live.messages()).len(), 4);
}
