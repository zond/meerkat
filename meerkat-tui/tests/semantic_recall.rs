//! Semantic recall after `/compact`, mirroring the integrator's sequence:
//! compact, retire the live session (releasing the single-writer index),
//! index the discarded history, rebuild the session from the compacted copy,
//! and let the agent find the compacted-away text through `memory_search`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{Harness, Step, all_text, within};
use meerkat_core::Message;
use meerkat_core::memory::MemoryStore;
use meerkat_memory::HnswMemoryStore;
use meerkat_tui::session::lifecycle::{
    compact_session, index_discarded, probe_memory, retire_session,
};
use meerkat_tui::session::{SessionHandle, SessionOptions};
use serde_json::json;

const SECRET: &str = "the zebra pancake recipe uses saffron";

fn tool_results(messages: &[Message]) -> String {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::ToolResults { results } => Some(
                results
                    .iter()
                    .map(meerkat_core::types::ToolResult::text_content)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn compacted_history_is_recalled_through_memory_search() {
    let mut h = Harness::new(vec![
        Step::Text("noted"),
        Step::Text("reply two"),
        Step::Text("reply three"),
        Step::Text("reply four"),
        Step::Text("SUMMARY without details"),
        // First call of the rebuilt session: search memory, then answer.
        Step::Tool {
            id: "call-1",
            name: "memory_search",
            args: json!({ "query": "zebra pancake recipe" }),
        },
        Step::Text("saffron"),
    ])
    .await;
    h.turn(&format!("remember this: {SECRET}")).await;
    h.turn("turn-two").await;
    h.turn("turn-three").await;
    h.turn("turn-four").await;

    // The live agent holds the index.
    assert!(probe_memory(&h.runtime.memory_dir).await.is_err());

    let compacted = within(compact_session(
        &h.runtime,
        &h.session,
        h.session.model(),
        0,
        None,
    ))
    .await
    .expect("compact");
    assert!(all_text(&compacted.discarded).contains(SECRET));

    let released = within(retire_session(&h.runtime, h.session.id()))
        .await
        .expect("retire");
    assert!(released, "retiring the session must release the index");
    let indexed = index_discarded(
        &h.runtime.memory_dir,
        compacted.session.id(),
        &compacted.discarded,
    )
    .await
    .expect("index");
    assert_eq!(indexed, 4, "two user and two assistant messages");

    let rebuilt = SessionHandle::create(
        &h.runtime,
        SessionOptions {
            resume: Some(compacted.session.clone()),
            ..SessionOptions::default()
        },
    )
    .await
    .expect("rebuild");
    assert_eq!(rebuilt.id(), h.session.id());
    h.session = rebuilt;
    assert_eq!(h.turn("what was the recipe?").await, "saffron");

    let requests = h.llm.requests();
    assert_eq!(requests.len(), 7);
    let first = &requests[5];
    assert!(
        first.tools.iter().any(|t| t.name == "memory_search"),
        "rebuilt session must offer memory_search"
    );
    assert!(
        !all_text(&first.messages).contains("zebra"),
        "the secret must only come back through memory_search"
    );
    let recalled = tool_results(&requests[6].messages);
    assert!(recalled.contains(SECRET), "{recalled}");
    assert!(
        recalled.contains(&h.session.id().to_string()),
        "results carry the session id: {recalled}"
    );
}

#[tokio::test]
async fn index_discarded_is_searchable_and_skips_empty_text() {
    let dir = tempfile::tempdir().unwrap();
    let id = meerkat_core::SessionId::new();
    let discarded = vec![
        Message::User(meerkat_core::types::UserMessage::text(SECRET)),
        Message::User(meerkat_core::types::UserMessage::text("   ")),
        Message::ToolResults {
            results: Vec::new(),
        },
    ];
    assert_eq!(
        index_discarded(dir.path(), &id, &discarded).await.unwrap(),
        1
    );
    assert_eq!(index_discarded(dir.path(), &id, &[]).await.unwrap(), 0);

    let store = HnswMemoryStore::open(dir.path()).unwrap();
    let hits = store.search("zebra pancake", 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].content, SECRET);
    assert_eq!(hits[0].metadata.session_id, id);
}
