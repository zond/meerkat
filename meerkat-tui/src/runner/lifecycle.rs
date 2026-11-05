//! Session lifecycle on the integrator: `/clear`, `/resume`, `/compact`,
//! `/remember`, and the context-gauge status feed.
//!
//! Exactly one session is live at a time: replacing it retires the old one
//! first (see [`crate::session::lifecycle`]), which also frees the
//! single-writer semantic memory index for the next agent.

use std::hash::{Hash, Hasher};

use meerkat_core::service::SessionError;
use meerkat_core::{AgentEvent, EventEnvelope, Message, Session};
use tokio::sync::mpsc;

use super::{Integrator, blocks_from_session, provider_name};
use crate::bootstrap::instructions::{append_remember_line, remember_target, remembered_context};
use crate::error::TuiError;
use crate::session::lifecycle::{
    compact_session, context_window, estimate_tokens, index_discarded, retire_session,
};
use crate::session::{SessionHandle, load_session, resolve_session_ref};
use crate::ui::{Msg, NoticeLevel, StatusUpdate};

/// Capacity of the compaction event channel.
const COMPACTION_EVENT_CAPACITY: usize = 16;

impl Integrator {
    /// Whether a turn (or a delivery still being reconciled) blocks a session
    /// switch; tells the user so.
    fn busy(&self, what: &str) -> bool {
        if self.turn.is_some() || !self.inflight.is_empty() {
            self.notice(
                NoticeLevel::Warn,
                format!("{what} is not available while a turn is running"),
            );
            return true;
        }
        false
    }

    /// `/clear`: a fresh session on the current model.
    pub(super) async fn new_session(&mut self) {
        if self.busy("/clear") {
            return;
        }
        let model = Some(self.current_model.clone());
        if let Err(err) = self.replace_session(None, model).await {
            self.notice(NoticeLevel::Error, format!("session: {err}"));
        }
    }

    /// `/resume <ref>`: continue a persisted session on its stored model.
    pub(super) async fn resume_session(&mut self, reference: &str) {
        if self.busy("/resume") {
            return;
        }
        let store = std::sync::Arc::clone(&self.runtime.store);
        let loaded = match resolve_session_ref(&*store, reference).await {
            Ok(id) => load_session(&*store, &id).await,
            Err(err) => Err(err),
        };
        let session = match loaded {
            Ok(session) => session,
            Err(err) => {
                self.notice(NoticeLevel::Error, err.to_string());
                return;
            }
        };
        let usage = session.total_usage();
        let blocks = blocks_from_session(&session);
        self.engine.seed_transcript(session.id(), &session);
        match self.replace_session(Some(session), None).await {
            Ok(()) => {
                self.send(Msg::ResetTranscript(blocks));
                self.send(Msg::Status(StatusUpdate::Tokens {
                    input: usage.input_tokens,
                    output: usage.output_tokens,
                }));
            }
            Err(err) => self.notice(NoticeLevel::Error, format!("session: {err}")),
        }
    }

    /// Persist and retire the live session, then build `resume` (or a fresh
    /// session) in its place. When the build fails, the previous session is
    /// rebuilt from its snapshot so the UI keeps a live session.
    async fn replace_session(
        &mut self,
        resume: Option<Session>,
        model: Option<String>,
    ) -> Result<(), TuiError> {
        let previous = match self.session.export().await {
            Ok(session) => Some(session),
            Err(err) => {
                tracing::warn!(error = %err, "could not export the session before replacing it");
                None
            }
        };
        if let Some(previous) = previous.as_ref().filter(|s| has_conversation(s))
            && let Err(err) = self.runtime.store.save(previous).await
        {
            tracing::warn!(error = %err, "could not persist the session before replacing it");
        }
        self.retire_current().await;
        match self.open_session(resume, model).await {
            Ok(()) => Ok(()),
            Err(err) => {
                if let Some(previous) = previous {
                    let model = Some(self.current_model.clone());
                    if let Err(restore) = self.open_session(Some(previous), model).await {
                        tracing::warn!(error = %restore, "could not restore the previous session");
                    }
                }
                Err(err)
            }
        }
    }

    /// Shut the live session down and wait for its agent to release the
    /// semantic index.
    async fn retire_current(&self) {
        match retire_session(&self.runtime, self.session.id()).await {
            Ok(true) | Err(SessionError::NotFound { .. }) => {}
            Ok(false) => self.notice(
                NoticeLevel::Warn,
                "semantic memory index is still busy; memory_search may be missing in this session",
            ),
            Err(err) => tracing::warn!(error = %err, "could not retire the live session"),
        }
    }

    /// Build a session (fresh or `resume`) and make it current; pushes the
    /// session, model and context status.
    async fn open_session(
        &mut self,
        resume: Option<Session>,
        model: Option<String>,
    ) -> Result<(), TuiError> {
        let context = resume.as_ref().map(|s| estimate_tokens(s.messages()));
        let handle =
            SessionHandle::create(&self.runtime, self.template.options(resume, model)).await?;
        self.current_model = handle.model().to_string();
        self.last_context = context;
        self.send(Msg::Status(StatusUpdate::Session(handle.id().to_string())));
        self.send(Msg::Status(StatusUpdate::Model {
            provider: provider_name(handle.provider()).to_string(),
            model: handle.model().to_string(),
        }));
        self.session = handle;
        self.push_context_window();
        self.send(Msg::Status(StatusUpdate::Context {
            used: context,
            approx: context.is_some(),
        }));
        Ok(())
    }

    /// Context window of the current model, for the gauge.
    pub(super) fn push_context_window(&self) {
        self.send(Msg::Status(StatusUpdate::ContextWindow(context_window(
            self.session.provider(),
            &self.current_model,
        ))));
    }

    /// `/compact`: summarize the history, index what the summary replaces
    /// into the semantic memory, and rebuild the session (same id) from the
    /// compacted history, which also reloads instructions and notes.
    pub(super) async fn compact(&mut self) {
        if self.turn.is_some() || !self.inflight.is_empty() {
            self.send(Msg::CompactFinished(Err(
                "cannot compact while a turn is running".to_string(),
            )));
            return;
        }
        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(COMPACTION_EVENT_CAPACITY);
        let ui_tx = self.ui_tx.clone();
        let source = self.session.id().to_string();
        let forward = tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                let _ = ui_tx.send(Msg::Agent(EventEnvelope::new(
                    source.clone(),
                    0,
                    None,
                    event,
                )));
            }
        });
        let result = compact_session(
            &self.runtime,
            &self.session,
            &self.current_model,
            self.last_context.unwrap_or(0),
            Some(event_tx),
        )
        .await;
        // The sender is gone once `compact_session` returns; flush the events
        // before reporting the outcome.
        let _ = forward.await;
        let compacted = match result {
            Ok(compacted) => compacted,
            Err(err) => {
                self.send(Msg::CompactFinished(Err(err.to_string())));
                return;
            }
        };

        // The compacted session is saved; the live one is retired unsaved.
        self.retire_current().await;
        let id = compacted.session.id().clone();
        match index_discarded(&self.runtime.memory_dir, &id, &compacted.discarded).await {
            Ok(indexed) => tracing::debug!(indexed, "indexed compacted history"),
            Err(err) => self.notice(
                NoticeLevel::Warn,
                format!("compacted history was not indexed for memory_search: {err}"),
            ),
        }
        let model = Some(self.current_model.clone());
        let mut opened = self
            .open_session(Some(compacted.session.clone()), model.clone())
            .await;
        if opened.is_err() {
            opened = self
                .open_session(Some(compacted.session.clone()), model)
                .await;
        }
        match opened {
            Ok(()) => {
                self.engine.seed_transcript(&id, &compacted.session);
                self.notice(
                    NoticeLevel::Info,
                    format!(
                        "compacted {} → {} messages",
                        compacted.messages_before,
                        compacted.session.messages().len()
                    ),
                );
                self.send(Msg::CompactFinished(Ok(())));
            }
            Err(err) => {
                let text = format!("compacted session saved; use /resume last ({err})");
                self.notice(NoticeLevel::Error, text.clone());
                self.send(Msg::CompactFinished(Err(text)));
            }
        }
    }

    /// `/remember` and `#text`: append the line to the workspace AGENTS.md and
    /// to the live session's system prompt (applied between turns).
    pub(super) fn remember(&self, text: &str) {
        let target = remember_target(&self.template.workspace);
        if let Err(err) = append_remember_line(&target, text) {
            self.notice(
                NoticeLevel::Error,
                format!("could not remember in {}: {err}", target.display()),
            );
            return;
        }
        let session = self.session.clone();
        let ui_tx = self.ui_tx.clone();
        let context = remembered_context(text);
        let key = format!("remember:{:016x}", text_hash(text));
        tokio::spawn(async move {
            if let Err(err) = session.append_system_context(context, key).await {
                let _ = ui_tx.send(Msg::Notice {
                    level: NoticeLevel::Warn,
                    text: format!(
                        "remembered line not applied to the live session ({err}); it applies to new sessions"
                    ),
                });
            }
        });
        self.notice(
            NoticeLevel::Info,
            format!(
                "remembered in {}; active for new calls, fully applied after /compact or a new session",
                target.display()
            ),
        );
    }
}

/// Whether `session` holds anything worth persisting.
fn has_conversation(session: &Session) -> bool {
    session
        .messages()
        .iter()
        .any(|m| !matches!(m, Message::System(_)))
}

fn text_hash(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}
