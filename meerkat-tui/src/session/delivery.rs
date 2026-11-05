//! Reconciling messages injected mid-turn against what the agent actually saw.
//!
//! The engine reports nothing when it drains its inbox, so after a turn ends
//! the integrator compares the bodies it injected with the inbox leftovers and
//! the recorded session (see [`reconcile`]). [`recall`] pulls queued messages
//! back out of the inbox before the agent reaches them.

use meerkat_core::types::{ContentBlock, UserMessage};
use meerkat_core::{Message, Session};

use super::SessionHandle;
use crate::delivery::{CONTINUATION_PROMPT, DeliveryReport, PendingId, strip_frame};

/// What the user actually typed in a recorded user message.
///
/// One drain records every injected message as a separate text block of one
/// user message; each comes back as its own entry with the frame stripped.
/// A plain prompt comes back whole, and a [`CONTINUATION_PROMPT`] not at all.
#[must_use]
pub fn user_texts(user: &UserMessage) -> Vec<String> {
    let injected = user
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::Text { text } if strip_frame(text).is_some()));
    if injected {
        return user
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(strip_frame(text).unwrap_or(text).to_string()),
                _ => None,
            })
            .collect();
    }
    let text = user.text_content();
    if text == CONTINUATION_PROMPT {
        Vec::new()
    } else {
        vec![text]
    }
}

/// Classify every injected message after a turn ended (ok, failed or
/// cancelled) and clear `inflight`.
///
/// `inflight` holds `(id, framed body)` pairs in injection order. A body still
/// in the inbox is drained and reported *undelivered*; a body recorded as a
/// user message is *answered* when an assistant message follows the last such
/// message and *unanswered* otherwise; a body found nowhere is *undelivered*.
///
/// Must be called only once the turn task has completed: exporting the session
/// waits for a running turn.
pub async fn reconcile(
    handle: &SessionHandle,
    inflight: &mut Vec<(PendingId, String)>,
) -> DeliveryReport {
    if inflight.is_empty() {
        return DeliveryReport::default();
    }
    let leftovers = handle.drain_inbox().await;
    let session = match handle.export().await {
        Ok(session) => Some(session),
        Err(err) => {
            tracing::warn!(error = %err, "could not export session to reconcile mid-turn messages");
            None
        }
    };
    classify(std::mem::take(inflight), leftovers, session.as_ref())
}

/// Drain the inbox and take back the injected messages that were still in it.
/// Returns their ids in injection order; they never reached the model.
pub async fn recall(
    handle: &SessionHandle,
    inflight: &mut Vec<(PendingId, String)>,
) -> Vec<PendingId> {
    let mut drained = handle.drain_inbox().await;
    let mut recalled = Vec::new();
    inflight.retain(|(id, body)| {
        if take_matching(&mut drained, body) {
            recalled.push(*id);
            false
        } else {
            true
        }
    });
    if !drained.is_empty() {
        tracing::warn!(
            count = drained.len(),
            "drained inbox items that were not injected by this TUI"
        );
    }
    recalled
}

/// Pure part of [`reconcile`]. Without a session snapshot, everything not in
/// the inbox counts as undelivered (re-sending risks a duplicate, not a loss).
fn classify(
    inflight: Vec<(PendingId, String)>,
    mut leftovers: Vec<String>,
    session: Option<&Session>,
) -> DeliveryReport {
    let mut report = DeliveryReport::default();
    for (id, body) in inflight {
        if take_matching(&mut leftovers, &body) {
            report.undelivered.push(id);
            continue;
        }
        match session.map(|s| recorded_state(s.messages(), &body)) {
            Some(Recorded::Answered) => report.answered.push(id),
            Some(Recorded::Unanswered) => report.unanswered.push(id),
            Some(Recorded::Missing) | None => report.undelivered.push(id),
        }
    }
    report
}

/// Remove the first entry equal to `body`; whether there was one.
fn take_matching(items: &mut Vec<String>, body: &str) -> bool {
    match items.iter().position(|item| item == body) {
        Some(index) => {
            items.remove(index);
            true
        }
        None => false,
    }
}

enum Recorded {
    Missing,
    Unanswered,
    Answered,
}

/// Where the last user message containing `body` sits in `messages`.
fn recorded_state(messages: &[Message], body: &str) -> Recorded {
    let Some(index) = messages
        .iter()
        .rposition(|m| matches!(m, Message::User(user) if user.text_content().contains(body)))
    else {
        return Recorded::Missing;
    };
    let answered = messages[index + 1..]
        .iter()
        .any(|m| matches!(m, Message::Assistant(_) | Message::BlockAssistant(_)));
    if answered {
        Recorded::Answered
    } else {
        Recorded::Unanswered
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::delivery::frame;
    use meerkat_core::StopReason;
    use meerkat_core::types::{AssistantMessage, Usage, UserMessage};

    fn user(text: &str) -> Message {
        Message::User(UserMessage::text(text))
    }

    fn assistant() -> Message {
        Message::Assistant(AssistantMessage {
            content: "ok".into(),
            tool_calls: Vec::new(),
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }

    #[test]
    fn classifies_leftover_recorded_and_missing_bodies() {
        let a = frame("a");
        let b = frame("b");
        let c = frame("c");
        let d = frame("d");
        let mut session = Session::new();
        session.push(user("start"));
        session.push(user(&format!("[EVENT via stdin] {a}")));
        session.push(assistant());
        session.push(user(&format!("[EVENT via stdin] {b}")));
        let report = classify(
            vec![
                (PendingId(1), a),
                (PendingId(2), b),
                (PendingId(3), c.clone()),
                (PendingId(4), d),
            ],
            vec![c],
            Some(&session),
        );
        assert_eq!(report.answered, vec![PendingId(1)]);
        assert_eq!(report.unanswered, vec![PendingId(2)]);
        assert_eq!(report.undelivered, vec![PendingId(3), PendingId(4)]);
    }

    #[test]
    fn user_texts_strips_frames_per_block_and_hides_continuations() {
        let drained = UserMessage::with_blocks(vec![
            ContentBlock::Text {
                text: format!("[EVENT via stdin] {}", frame("one")),
            },
            ContentBlock::Text {
                text: format!("[EVENT via stdin] {}", frame("two")),
            },
        ]);
        assert_eq!(user_texts(&drained), vec!["one", "two"]);
        assert_eq!(user_texts(&UserMessage::text("hi")), vec!["hi"]);
        assert!(user_texts(&UserMessage::text(CONTINUATION_PROMPT)).is_empty());
    }

    #[test]
    fn no_snapshot_means_undelivered() {
        let report = classify(vec![(PendingId(1), frame("x"))], Vec::new(), None);
        assert_eq!(report.undelivered, vec![PendingId(1)]);
    }
}
