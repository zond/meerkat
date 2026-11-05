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

use super::{Integrator, SessionTarget, blocks_from_session};
use crate::bootstrap::instructions::{append_remember_line, remember_target, remembered_context};
use crate::endpoints::{EndpointRef, load_choice, restore_choice};
use crate::error::TuiError;
use crate::session::lifecycle::{
    CompactError, compact_session_with, estimate_tokens, index_discarded, retire_session,
};
use crate::session::{SessionHandle, load_session, resolve_session_ref};
use crate::ui::{CompactFailure, Msg, NoticeLevel, StatusUpdate};

/// Capacity of the compaction event channel.
const COMPACTION_EVENT_CAPACITY: usize = 16;

/// The UI's view of a [`CompactError`]: whether retrying at the same usage
/// is pointless ([`CompactFailure::NoGain`]/[`CompactFailure::TooShort`]).
fn compact_failure(err: CompactError) -> CompactFailure {
    match err {
        CompactError::NoGain(messages) => CompactFailure::NoGain(messages),
        CompactError::TooShort(messages) => CompactFailure::TooShort(messages),
        other => CompactFailure::Other(other.to_string()),
    }
}

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

    /// `/clear`: a fresh session on the current model. The UI keeps its
    /// transcript until the new session is live.
    pub(super) async fn new_session(&mut self) {
        if self.busy("/clear") {
            return;
        }
        let choice = self.current_choice();
        match self.replace_session(None, choice, None).await {
            Ok(()) => {
                self.send(Msg::ResetTranscript(Vec::new()));
                self.send(Msg::Status(StatusUpdate::Tokens {
                    input: 0,
                    output: 0,
                }));
            }
            Err(err) => self.notice(
                NoticeLevel::Error,
                format!("new session not started: {err}"),
            ),
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
        let id = session.id().clone();
        let seed = session.clone();
        let known = self.known_endpoint_list();
        let (choice, notice) = restore_choice(
            load_choice(&self.runtime.scope.realm_root(), &id.to_string()),
            session
                .session_metadata()
                .map(|meta| (meta.provider, meta.model)),
            &known,
            |name| self.endpoint_exists(name),
            &self.current_choice(),
        );
        match self.replace_session(Some(session), choice, notice).await {
            Ok(()) => {
                self.engine.seed_transcript(&id, &seed);
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
        choice: EndpointRef,
        notice: Option<String>,
    ) -> Result<(), TuiError> {
        let previous_choice = self.current_choice();
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
        match self.open_session(resume, choice).await {
            Ok(()) => {
                if let Some(text) = notice {
                    self.notice(NoticeLevel::Warn, text);
                }
                Ok(())
            }
            Err(err) => {
                let restored = match previous {
                    Some(previous) => {
                        match self.open_session(Some(previous), previous_choice).await {
                            Ok(()) => true,
                            Err(restore) => {
                                tracing::warn!(error = %restore, "could not restore the previous session");
                                false
                            }
                        }
                    }
                    None => false,
                };
                if !restored {
                    self.dead = true;
                    self.notice(NoticeLevel::Error, "no live session; use /resume last");
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

    /// Build a session (fresh or `resume`) on `choice` and make it current;
    /// pushes the session, model and context status.
    async fn open_session(
        &mut self,
        resume: Option<Session>,
        choice: EndpointRef,
    ) -> Result<(), TuiError> {
        let context = resume.as_ref().map(|s| estimate_tokens(s.messages()));
        let llm = self
            .client_for(&choice.endpoint)
            .await
            .map_err(TuiError::Endpoint)?;
        let target = SessionTarget {
            provider: self.provider_of(&choice.endpoint),
            llm,
            choice,
        };
        let endpoint = target.choice.endpoint.clone();
        let handle =
            SessionHandle::create(&self.runtime, self.template.options(resume, Some(target)))
                .await?;
        self.engine.set_primary_session(handle.id());
        self.current_endpoint = endpoint;
        self.current_provider = handle.provider();
        self.current_model = handle.model().to_string();
        self.last_context = context;
        self.dead = false;
        self.send(Msg::Status(StatusUpdate::Session(handle.id().to_string())));
        self.send(Msg::Status(StatusUpdate::Model {
            provider: self.current_endpoint.clone(),
            model: handle.model().to_string(),
        }));
        self.session = handle;
        self.save_current_choice();
        self.push_context_window();
        self.send(Msg::Status(StatusUpdate::Context {
            used: context,
            approx: context.is_some(),
        }));
        Ok(())
    }

    /// `/compact`: summarize the history, index what the summary replaces
    /// into the semantic memory, and rebuild the session (same id) from the
    /// compacted history, which also reloads instructions and notes.
    pub(super) async fn compact(&mut self) {
        if self.turn.is_some() || !self.inflight.is_empty() {
            self.send(Msg::CompactFinished(Err(CompactFailure::Other(
                "cannot compact while a turn is running".to_string(),
            ))));
            return;
        }
        if self.dead {
            self.send(Msg::CompactFinished(Err(CompactFailure::Other(
                "no live session; use /resume last".to_string(),
            ))));
            return;
        }
        let choice = self.current_choice();
        let llm = match self.client_for(&choice.endpoint).await {
            Ok(llm) => llm,
            Err(err) => {
                self.send(Msg::CompactFinished(Err(CompactFailure::Other(err))));
                return;
            }
        };
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
        let result = compact_session_with(
            &self.runtime,
            &self.session,
            llm,
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
                self.send(Msg::CompactFinished(Err(compact_failure(err))));
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
        let mut opened = self
            .open_session(Some(compacted.session.clone()), choice.clone())
            .await;
        if opened.is_err() {
            opened = self
                .open_session(Some(compacted.session.clone()), choice)
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
                self.restore_after_failed_compaction(compacted.original, &err)
                    .await;
            }
        }
    }

    /// The compacted session could not be opened: bring the pre-compaction
    /// session back so `self.session` never names a retired session. When
    /// even that fails, mark the integrator dead until `/resume` or `/clear`.
    async fn restore_after_failed_compaction(&mut self, original: Session, err: &TuiError) {
        tracing::warn!(error = %err, "could not open the compacted session; restoring");
        let id = original.id().clone();
        let seed = original.clone();
        let choice = self.current_choice();
        match self.open_session(Some(original), choice).await {
            Ok(()) => {
                self.engine.seed_transcript(&id, &seed);
                // The store holds the compacted version; the restored
                // session overwrites it after its next completed turn.
                let text = format!(
                    "compaction not applied to the live session ({err}); the compacted copy is saved, the session continues uncompacted"
                );
                self.notice(NoticeLevel::Warn, text.clone());
                self.send(Msg::CompactFinished(Err(CompactFailure::Other(text))));
            }
            Err(restore) => {
                tracing::warn!(error = %restore, "could not restore the session after compaction");
                self.dead = true;
                let text = format!(
                    "no live session ({err}); the compacted session is saved: use /resume last"
                );
                self.notice(NoticeLevel::Error, text.clone());
                self.send(Msg::CompactFinished(Err(CompactFailure::Other(text))));
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
pub(super) fn has_conversation(session: &Session) -> bool {
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
