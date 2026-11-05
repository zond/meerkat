//! Shutdown on the integrator: stop the running turn, persist the session and
//! save every message the model never received to `<realm>/tui_unsent.txt`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use meerkat_core::service::SessionError;

use super::lifecycle::has_conversation;
use super::{Integrator, RunningTurn};
use crate::approval::escalation::HumanDecision;
use crate::delivery::{FRAME, PendingId};
use crate::session::TurnTask;
use crate::session::delivery::reconcile;

/// How long quitting waits for an interrupted turn to wind down.
pub const QUIT_TURN_TIMEOUT: Duration = Duration::from_secs(10);

/// Upper bound on the whole integrator shutdown (turn, reconcile, persist).
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);

/// File in the realm directory that collects unsent messages.
pub const UNSENT_FILE: &str = "tui_unsent.txt";

/// Line written in front of every saved message.
const UNSENT_HEADER: &str = "--- unsent message ---";

/// Where unsent messages were saved on quit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedUnsent {
    /// Messages written by this shutdown.
    pub count: usize,
    /// The file they were appended to.
    pub path: PathBuf,
}

impl Integrator {
    /// Quit: deny pending escalations, interrupt a running turn and wait (up
    /// to [`QUIT_TURN_TIMEOUT`]) for it, reconcile what was injected,
    /// persist the session, and append every message the model never got to
    /// [`UNSENT_FILE`].
    pub(super) async fn shutdown(&mut self) -> Option<SavedUnsent> {
        self.stop_pending();
        let mut unsent: Vec<(Option<PendingId>, String)> = Vec::new();
        let mut delivered: Vec<PendingId> = Vec::new();
        let mut session_idle = true;
        if let Some(RunningTurn { task, session }) = self.turn.take() {
            if let Err(err) = session.interrupt().await
                && !matches!(err, SessionError::NotRunning { .. })
            {
                tracing::warn!(error = %err, "could not interrupt the turn on quit");
            }
            if self.await_turn_on_quit(task).await {
                let bodies = self.inflight.clone();
                let report = reconcile(&session, &mut self.inflight).await;
                delivered.extend(report.answered.iter().chain(&report.unanswered));
                for id in &report.undelivered {
                    if let Some((_, body)) = bodies.iter().find(|(known, _)| known == id) {
                        unsent.push((Some(*id), unframe(body).to_string()));
                    }
                }
            } else {
                // Still running: exporting would wait for it. Keep every
                // injected message (a duplicate beats a loss).
                tracing::warn!("turn did not stop in time on quit");
                session_idle = false;
                unsent.extend(
                    self.inflight
                        .drain(..)
                        .map(|(id, body)| (Some(id), unframe(&body).to_string())),
                );
            }
        }
        for (id, text) in std::mem::take(&mut self.held_unsent) {
            let known = id.is_some_and(|id| {
                delivered.contains(&id) || unsent.iter().any(|(other, _)| *other == Some(id))
            });
            if !known {
                unsent.push((id, text));
            }
        }
        unsent.sort_by_key(|(id, _)| id.map_or(u64::MAX, |id| id.0));
        if session_idle && !self.dead {
            self.persist_on_quit().await;
        }
        if unsent.is_empty() {
            return None;
        }
        let path = self.runtime.scope.realm_root().join(UNSENT_FILE);
        let texts: Vec<String> = unsent.into_iter().map(|(_, text)| text).collect();
        match append_unsent(&path, &texts) {
            Ok(()) => Some(SavedUnsent {
                count: texts.len(),
                path,
            }),
            Err(err) => {
                tracing::warn!(error = %err, path = %path.display(), "could not save unsent messages");
                None
            }
        }
    }

    /// Wait for the interrupted turn, answering any new escalation with
    /// `Stop`. Returns whether the turn finished in time.
    async fn await_turn_on_quit(&mut self, mut task: TurnTask) -> bool {
        let deadline = tokio::time::sleep(QUIT_TURN_TIMEOUT);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                result = &mut task => {
                    if let Err(err) = result {
                        tracing::debug!(error = %err, "turn ended on quit");
                    }
                    return true;
                }
                Some(request) = self.escalation_rx.recv() => {
                    let _ = request.respond.send(HumanDecision::Stop);
                }
                () = &mut deadline => return false,
            }
        }
    }

    /// Save the live session when it holds a conversation. The engine saves
    /// only after completed runs; an interrupted turn would be lost otherwise.
    async fn persist_on_quit(&self) {
        match self.session.export().await {
            Ok(session) if has_conversation(&session) => {
                if let Err(err) = self.runtime.store.save(&session).await {
                    tracing::warn!(error = %err, "could not persist the session on quit");
                }
            }
            Ok(_) => {}
            Err(err) => tracing::warn!(error = %err, "could not export the session on quit"),
        }
    }
}

/// The text the user typed, from an injected (framed) body.
fn unframe(body: &str) -> &str {
    body.strip_prefix(FRAME).unwrap_or(body)
}

/// Append `texts` to the unsent-messages file, each after a header line.
fn append_unsent(path: &Path, texts: &[String]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for text in texts {
        writeln!(file, "{UNSENT_HEADER}")?;
        writeln!(file, "{}", text.trim_end())?;
    }
    file.flush()
}

/// Number of messages in an unsent-messages file.
pub fn count_unsent(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .map(|content| content.lines().filter(|l| *l == UNSENT_HEADER).count())
        .unwrap_or(0)
}

/// Start-up notice when earlier unsent messages are waiting in the realm.
pub fn unsent_notice(realm_root: &Path) -> Option<String> {
    let path = realm_root.join(UNSENT_FILE);
    match count_unsent(&path) {
        0 => None,
        n => Some(format!(
            "{n} unsent message{} from an earlier run saved in {} (delete the file once handled)",
            if n == 1 { "" } else { "s" },
            path.display()
        )),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::delivery::frame;

    #[test]
    fn unsent_file_appends_and_counts() {
        let dir = tempfile::tempdir().unwrap();
        assert!(unsent_notice(dir.path()).is_none());
        let path = dir.path().join(UNSENT_FILE);
        append_unsent(&path, &["one".into(), "two\nlines\n".into()]).unwrap();
        append_unsent(&path, &["three".into()]).unwrap();
        assert_eq!(count_unsent(&path), 3);
        let notice = unsent_notice(dir.path()).unwrap();
        assert!(notice.starts_with("3 unsent messages"), "{notice}");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("two\nlines\n---"), "{content}");
    }

    #[test]
    fn unframe_strips_the_mid_turn_frame() {
        assert_eq!(unframe(&frame("hi")), "hi");
        assert_eq!(unframe("plain"), "plain");
    }
}
