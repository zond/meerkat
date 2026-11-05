//! Session lifecycle building blocks: the instruction sections of a new
//! session, context estimates, `/compact`, and the semantic memory index.
//!
//! The factory opens the realm's semantic index (`Runtime::memory_dir`, a
//! single-writer redb database) for every agent it builds. A live session
//! therefore holds the index until its agent is dropped, which happens
//! asynchronously after [`retire_session`] asks the session task to shut
//! down. Callers keep exactly one live session and wait for the index to be
//! released (see [`wait_memory_released`]) before building the next one;
//! otherwise the new agent silently comes up without `memory_search`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use meerkat_core::agent::compact::{CompactionError, run_compaction};
use meerkat_core::memory::{MemoryMetadata, MemoryStore, MemoryStoreError};
use meerkat_core::service::SessionError;
use meerkat_core::types::{AssistantBlock, SessionId};
use meerkat_core::{AgentEvent, CompactionConfig, Compactor, Message, Provider, Session};
use meerkat_memory::HnswMemoryStore;
use tokio::sync::mpsc;

use super::SessionHandle;
use crate::bootstrap::GlobalDirs;
use crate::bootstrap::Runtime;
use crate::bootstrap::instructions::InstructionSources;
use crate::memory::{MemoryToolDispatcher, prompt_index};

/// Most recent complete turns `/compact` keeps verbatim (the configured
/// `compaction.recent_turn_budget` is honoured when it is smaller). A manual
/// compaction is asked for to shrink the context now, so it keeps less than
/// the automatic one.
pub const MANUAL_RECENT_TURNS: usize = 2;

/// Fewest messages worth compacting.
pub const MIN_COMPACT_MESSAGES: usize = 4;

/// How long [`wait_memory_released`] waits for a retired agent to let go of
/// the semantic index.
pub const MEMORY_RELEASE_TIMEOUT: Duration = Duration::from_secs(2);

/// Poll interval of [`wait_memory_released`].
const MEMORY_RELEASE_STEP: Duration = Duration::from_millis(25);

/// The instruction sections of a new session, after the persona: discovered
/// instruction files (user-global, ancestors, `AGENTS.local.md`) followed by
/// the memory-notes index when there are notes.
///
/// Re-run on every session build so `/clear`, `/resume` and `/compact` pick
/// up edited instruction files and notes.
#[must_use]
pub fn session_instructions(
    workspace: &Path,
    global: Option<&GlobalDirs>,
    notes: Option<&MemoryToolDispatcher>,
) -> Vec<String> {
    let mut sections =
        InstructionSources::render_all(&InstructionSources::discover(workspace, global));
    if let Some(index) = notes.and_then(|notes| prompt_index(notes.user(), notes.workspace())) {
        sections.push(index);
    }
    sections
}

/// Rough token count of `messages`: characters of all text content (system,
/// user, assistant, tool results) divided by four.
#[must_use]
pub fn estimate_tokens(messages: &[Message]) -> u64 {
    let chars: usize = messages
        .iter()
        .map(|message| match message {
            Message::System(system) => system.content.chars().count(),
            Message::User(user) => user.text_content().chars().count(),
            Message::Assistant(assistant) => assistant.content.chars().count(),
            Message::BlockAssistant(assistant) => assistant
                .blocks
                .iter()
                .map(|block| match block {
                    AssistantBlock::Text { text, .. } => text.chars().count(),
                    _ => 0,
                })
                .sum(),
            Message::ToolResults { results } => results
                .iter()
                .map(|result| result.text_content().chars().count())
                .sum(),
        })
        .sum();
    u64::try_from(chars / 4).unwrap_or(u64::MAX)
}

/// The model's context window from the curated catalog, `None` for models
/// the catalog does not know.
#[must_use]
pub fn context_window(provider: Provider, model: &str) -> Option<u64> {
    meerkat_models::catalog::entry_for(provider.as_str(), model)
        .and_then(|entry| entry.context_window)
        .map(u64::from)
}

/// Shut down the live session `id` (without archiving or persisting it) and
/// wait until its agent has released the semantic index.
///
/// Returns whether the index was released in time; `false` means the next
/// session may come up without `memory_search`.
///
/// # Errors
///
/// [`SessionError::NotFound`] when `id` is not live.
pub async fn retire_session(runtime: &Runtime, id: &SessionId) -> Result<bool, SessionError> {
    runtime.service.discard_live_session(id).await?;
    Ok(wait_memory_released(&runtime.memory_dir).await)
}

/// Open the semantic index once and drop it again.
///
/// # Errors
///
/// The open error, typically because another live agent (in this process
/// or another `rkat-tui`) holds the index.
pub async fn probe_memory(dir: &Path) -> Result<(), MemoryStoreError> {
    open_memory(dir).await.map(drop)
}

/// Poll [`probe_memory`] until it succeeds or [`MEMORY_RELEASE_TIMEOUT`]
/// passes. Returns whether the index is free.
pub async fn wait_memory_released(dir: &Path) -> bool {
    let deadline = tokio::time::Instant::now() + MEMORY_RELEASE_TIMEOUT;
    loop {
        let err = match probe_memory(dir).await {
            Ok(()) => return true,
            Err(err) => err,
        };
        if tokio::time::Instant::now() >= deadline {
            tracing::warn!(
                dir = %dir.display(),
                error = %err,
                "semantic memory index still held after retiring the session"
            );
            return false;
        }
        tokio::time::sleep(MEMORY_RELEASE_STEP).await;
    }
}

/// Open the semantic index off the async runtime (it reads the whole index).
async fn open_memory(dir: &Path) -> Result<HnswMemoryStore, MemoryStoreError> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || HnswMemoryStore::open(dir))
        .await
        .map_err(|err| MemoryStoreError::Index(format!("index open task failed: {err}")))?
}

/// Index the indexable text of `discarded` into the semantic index at `dir`.
/// Returns how many messages were indexed. The index must not be held by a
/// live agent: call after [`retire_session`].
///
/// # Errors
///
/// When the index cannot be opened or a write fails; messages indexed before
/// the failure stay indexed.
pub async fn index_discarded(
    dir: &Path,
    session_id: &SessionId,
    discarded: &[Message],
) -> Result<usize, MemoryStoreError> {
    let texts: Vec<String> = discarded
        .iter()
        .map(Message::as_indexable_text)
        .filter(|text| !text.trim().is_empty())
        .collect();
    if texts.is_empty() {
        return Ok(0);
    }
    let store = open_memory(dir).await?;
    for text in &texts {
        store
            .index(
                text,
                MemoryMetadata {
                    session_id: session_id.clone(),
                    turn: None,
                    indexed_at: meerkat_core::time_compat::SystemTime::now(),
                },
            )
            .await?;
    }
    drop(store);
    Ok(texts.len())
}

/// Why [`compact_session`] did not compact.
#[derive(Debug, thiserror::Error)]
pub enum CompactError {
    /// The live session could not be exported.
    #[error("could not read the session: {0}")]
    Export(#[from] SessionError),
    /// Too little history to be worth summarizing.
    #[error("nothing to compact yet ({0} messages)")]
    TooShort(usize),
    /// Summarization failed; the session is untouched.
    #[error("{0}")]
    Compaction(#[from] CompactionError),
    /// Compaction did not reduce the history (everything is recent).
    #[error("nothing to compact: all {0} messages are recent turns")]
    NoGain(usize),
    /// The compacted session could not be saved; the live session is untouched.
    #[error("could not save the compacted session: {0}")]
    Save(String),
}

/// The result of [`compact_session`].
#[derive(Debug)]
pub struct Compacted {
    /// The compacted session (same id), already saved to the realm store.
    pub session: Session,
    /// The messages the summary replaced, for semantic indexing.
    pub discarded: Vec<Message>,
    /// Message count before compaction.
    pub messages_before: usize,
}

/// Summarize the history of the idle session `handle` with Meerkat's
/// [`meerkat::DefaultCompactor`] on `model`, and save the compacted session
/// to the realm store.
///
/// The live session is not changed: the caller retires it and rebuilds the
/// session from [`Compacted::session`] (which also reloads instructions).
/// `last_input_tokens` is only reported in `CompactionStarted`. Compaction
/// events go to `event_tx`.
///
/// # Errors
///
/// See [`CompactError`]. On every error, nothing was saved.
pub async fn compact_session(
    runtime: &Runtime,
    handle: &SessionHandle,
    model: &str,
    last_input_tokens: u64,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
) -> Result<Compacted, CompactError> {
    let mut session = handle.export().await?;
    let messages_before = session.messages().len();
    if messages_before < MIN_COMPACT_MESSAGES {
        return Err(CompactError::TooShort(messages_before));
    }

    let mut config: CompactionConfig = runtime.config.compaction.clone().into();
    config.recent_turn_budget = config.recent_turn_budget.min(MANUAL_RECENT_TURNS);
    let compactor: Arc<dyn Compactor> = Arc::new(meerkat::DefaultCompactor::new(config));
    let client = runtime
        .factory
        .build_llm_adapter(Arc::clone(&runtime.llm), model)
        .await;
    let outcome = run_compaction(
        &client,
        &compactor,
        session.messages(),
        last_input_tokens,
        0,
        &event_tx,
        &meerkat_core::new_event_tap(),
    )
    .await?;
    if outcome.discarded.is_empty() {
        return Err(CompactError::NoGain(messages_before));
    }

    *session.messages_mut() = outcome.new_messages;
    session.record_usage(outcome.summary_usage);
    runtime
        .store
        .save(&session)
        .await
        .map_err(|err| CompactError::Save(err.to_string()))?;
    Ok(Compacted {
        session,
        discarded: outcome.discarded,
        messages_before,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use meerkat_core::StopReason;
    use meerkat_core::types::{AssistantMessage, SystemMessage, ToolResult, Usage, UserMessage};

    #[test]
    fn estimate_counts_text_of_every_kind() {
        let messages = vec![
            Message::System(SystemMessage {
                content: "s".repeat(40),
            }),
            Message::User(UserMessage::text("u".repeat(40))),
            Message::Assistant(AssistantMessage {
                content: "a".repeat(40),
                tool_calls: Vec::new(),
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
            Message::ToolResults {
                results: vec![ToolResult::new("t".into(), "r".repeat(40), false)],
            },
        ];
        assert_eq!(estimate_tokens(&messages), 40);
        assert_eq!(estimate_tokens(&[]), 0);
    }

    #[test]
    fn catalog_window() {
        assert_eq!(
            context_window(Provider::Anthropic, "claude-sonnet-4-6"),
            Some(200_000)
        );
        assert_eq!(context_window(Provider::Anthropic, "no-such-model"), None);
    }

    #[tokio::test]
    async fn index_lock_is_detected_and_released() {
        let dir = tempfile::tempdir().unwrap();
        let held = HnswMemoryStore::open(dir.path()).unwrap();
        assert!(probe_memory(dir.path()).await.is_err());
        let waiter = tokio::spawn({
            let dir = dir.path().to_path_buf();
            async move { wait_memory_released(&dir).await }
        });
        tokio::time::sleep(Duration::from_millis(60)).await;
        drop(held);
        assert!(waiter.await.unwrap());
    }
}
