//! ratatui application: Elm-style state + pure `update`, a renderer and a terminal runtime.
//!
//! [`App::update`] is pure — it never touches the terminal or performs I/O — and
//! returns [`Effect`]s that the integrator (session/approval wiring) executes.
//! Everything the integrator wants to show is pushed back in as a [`Msg`].

pub mod commands;
pub mod delivery;
pub mod history;
pub mod input;
pub mod keymap;
pub mod markdown;
pub mod modal;
pub mod render;
pub mod runtime;
pub mod search;
pub mod transcript;

use crate::delivery::{CONTINUATION_PROMPT, DeliveryReport, InjectFailure, PendingId};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use meerkat_core::{AgentEvent, EventEnvelope, ScopedAgentEvent};
use ratatui::crossterm::event::{KeyEvent, MouseEvent, MouseEventKind};

pub use commands::{AutoCommand, Command, CommandError, RuleEffect, RulesCommand};
pub use delivery::{Pending, PendingQueue, PendingState};
pub use history::History;
pub use input::InputEditor;
pub use keymap::{Action, EditAction};
pub use modal::{EscalationAnswer, EscalationId, EscalationPrompt, Modal, ResumePicker};
pub use runtime::{TerminalGuard, UiChannels, UiError, install_panic_hook, run_ui};
pub use search::{HistorySearch, SearchOutcome};
pub use transcript::{Block, BlockId, DiffView, NoticeLevel, QueuedLabel, ToolState, Transcript};

/// Window in which a second Ctrl-C on an idle, empty prompt quits.
pub const DOUBLE_CTRL_C_WINDOW: Duration = Duration::from_millis(1500);

/// Window in which a second Ctrl-D / `/quit` confirms quitting while a turn
/// is running or messages are queued.
pub const QUIT_CONFIRM_WINDOW: Duration = Duration::from_secs(3);

/// Window after Ctrl-X in which the second key of a chord is accepted.
pub const CHORD_WINDOW: Duration = Duration::from_millis(1500);

/// How keys reach the input area.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum InputMode {
    /// Keys edit the prompt.
    #[default]
    Normal,
    /// Keys drive a reverse history search (Ctrl-R).
    Search(HistorySearch),
}

/// Result of a finished turn, reported by the integrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnSummary {
    /// Final assistant text.
    pub text: String,
    /// Input tokens consumed by the run.
    pub input_tokens: u64,
    /// Output tokens produced by the run.
    pub output_tokens: u64,
}

/// Result of a turn plus what happened to messages injected during it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    /// The turn's result.
    pub result: Result<TurnSummary, String>,
    /// Delivery status of every message injected during the turn.
    pub delivery: DeliveryReport,
}

impl TurnOutcome {
    /// An outcome with nothing pending.
    #[must_use]
    pub fn plain(result: Result<TurnSummary, String>) -> Self {
        Self {
            result,
            delivery: DeliveryReport::default(),
        }
    }
}

/// An approval decision to show in the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalNotice {
    /// Tool name.
    pub tool: String,
    /// Tier / mechanism that decided (e.g. `rule`, `classifier`, `user`).
    pub tier: String,
    /// Whether the call was allowed.
    pub allowed: bool,
    /// Optional explanation.
    pub reason: Option<String>,
}

/// Status-bar update pushed by the integrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusUpdate {
    /// Provider and model changed.
    Model {
        /// Provider name.
        provider: String,
        /// Model id.
        model: String,
    },
    /// Active session id (shortened for display).
    Session(String),
    /// Absolute token counters (e.g. after resuming).
    Tokens {
        /// Input tokens.
        input: u64,
        /// Output tokens.
        output: u64,
    },
    /// Auto-mode state and denial counters.
    Auto {
        /// Display state, e.g. `active`, `PAUSED`, `off`.
        state: String,
        /// `(consecutive, consecutive_limit, total, total_limit)`.
        denials: (u32, u32, u32, u32),
    },
    /// Whether the classifier is currently running.
    Classifying(bool),
    /// Current context usage (tokens in the last model call); `approx` after
    /// compaction or resume when the number is an estimate.
    Context {
        /// Tokens used, `None` when unknown.
        used: Option<u64>,
        /// Whether `used` is an estimate.
        approx: bool,
    },
    /// The model's context window in tokens, `None` when unknown.
    ContextWindow(Option<u64>),
    /// An estimated context window (the model is not in the catalog; the
    /// window is a family or provider default). The gauge shows `~`.
    ApproxContextWindow(u64),
}

/// Input to [`App::update`].
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Msg {
    /// A key press.
    Key(KeyEvent),
    /// Bracketed paste.
    Paste(String),
    /// Terminal resized to `(columns, rows)`.
    Resize(u16, u16),
    /// Mouse event (only when mouse capture is enabled).
    Mouse(MouseEvent),
    /// Primary agent stream event.
    Agent(EventEnvelope<AgentEvent>),
    /// Scoped (sub-agent) stream event.
    Scoped(ScopedAgentEvent),
    /// The turn started by [`Effect::StartTurn`] finished.
    TurnFinished(TurnOutcome),
    /// The integrator's answer to [`Effect::Inject`].
    InjectAck {
        /// The queued message.
        id: PendingId,
        /// `Ok` when the message is in the agent's inbox.
        result: Result<(), InjectFailure>,
    },
    /// Queued messages pulled back out of the agent's inbox by
    /// [`Effect::RecallPending`]; these never reached the model.
    Recalled(Vec<PendingId>),
    /// A file tool produced a diff.
    Diff {
        /// Tool-use id of the producing call.
        tool_use_id: String,
        /// File path.
        path: String,
        /// Unified diff.
        unified: String,
        /// Added lines.
        added: usize,
        /// Removed lines.
        removed: usize,
    },
    /// The approval engine needs a human decision.
    Escalation(EscalationPrompt, EscalationId),
    /// An escalation was resolved elsewhere (timeout, dropped); close it.
    EscalationResolved(EscalationId),
    /// An approval decision to log in the transcript.
    Approval(ApprovalNotice),
    /// A notice to show in the transcript.
    Notice {
        /// Severity.
        level: NoticeLevel,
        /// Text.
        text: String,
    },
    /// Status-bar update.
    Status(StatusUpdate),
    /// Open the resume picker with `(ref, label)` items.
    ShowResumePicker(Vec<(String, String)>),
    /// Replace the transcript (e.g. after resuming a session).
    ResetTranscript(Vec<Block>),
    /// `/compact` finished (the integrator rebuilt the session).
    CompactFinished(Result<(), String>),
    /// Animation tick.
    Tick,
}

/// Side effect requested by [`App::update`], executed by the integrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Run a turn with this prompt; answer with [`Msg::TurnFinished`].
    StartTurn(String),
    /// Interrupt the running turn.
    Interrupt,
    /// Interrupt the running turn and send this prompt next (Ctrl-X Ctrl-S).
    InterruptAndSend(String),
    /// Deliver a message to the running turn at its next boundary; answered
    /// with [`Msg::InjectAck`].
    Inject {
        /// Queue id allocated by the UI.
        id: PendingId,
        /// Raw user text (the integrator adds the frame).
        text: String,
    },
    /// Pull every not-yet-consumed injected message back; answered with
    /// [`Msg::Recalled`].
    RecallPending,
    /// Leave the application. The integrator interrupts a running turn,
    /// persists the session and saves undelivered messages before exiting.
    Quit,
    /// Sent right before [`Effect::Quit`]: every message the UI still holds
    /// (queued, deferred, or the Ctrl-X Ctrl-S prompt as `None`). The
    /// integrator saves those the model never received.
    SaveUnsent(Vec<(Option<PendingId>, String)>),
    /// Start a fresh session.
    NewSession,
    /// Resume a session by reference (`last`, `~N`, id prefix).
    ResumeSession(String),
    /// Load session list; answer with [`Msg::ShowResumePicker`].
    RequestResumePicker,
    /// Answer an escalation.
    RespondEscalation(EscalationId, EscalationAnswer),
    /// Persist an approval rule.
    AddRule {
        /// Allow or deny.
        effect: RuleEffect,
        /// Tool name.
        tool: String,
        /// Optional argument pattern.
        pattern: Option<String>,
    },
    /// Remove rule number `n`.
    RemoveRule(usize),
    /// List rules; answer with [`Msg::Notice`].
    ListRules,
    /// Enable/disable auto mode.
    SetAutoMode(bool),
    /// Resume auto mode after a denial-limit pause.
    ResumeAutoMode,
    /// Switch model.
    SetModel(String),
    /// Append a line to the workspace AGENTS.md and push it live.
    Remember(String),
    /// Compact the session's context (idle only); answered with
    /// [`Msg::CompactFinished`].
    Compact,
}

/// Everything the status bar shows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatusInfo {
    /// Provider name.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// Short session id.
    pub session_short: String,
    /// Cumulative input tokens.
    pub input_tokens: u64,
    /// Cumulative output tokens.
    pub output_tokens: u64,
    /// Auto-mode display state (`active`, `PAUSED`, `off`; empty hides the segment).
    pub auto_state: String,
    /// `(consecutive, consecutive_limit, total, total_limit)` denials.
    pub denials: (u32, u32, u32, u32),
    /// When the running turn started.
    pub running_since: Option<Instant>,
    /// Number of messages typed mid-turn that have not been delivered yet.
    pub queued: usize,
    /// Whether the classifier is running.
    pub classifying: bool,
    /// Tokens in the last model call, `None` when unknown.
    pub context_used: Option<u64>,
    /// Model context window in tokens, `None` when unknown.
    pub context_window: Option<u64>,
    /// Whether `context_used` is an estimate.
    pub context_approx: bool,
    /// Whether `context_window` is an estimate (model not in the catalog).
    pub window_approx: bool,
}

/// Number of session-id characters shown in the status bar.
const SESSION_SHORT_LEN: usize = 8;

impl StatusInfo {
    /// Set provider and model.
    pub fn set_model(&mut self, provider: impl Into<String>, model: impl Into<String>) {
        self.provider = provider.into();
        self.model = model.into();
    }

    /// Set the session id (shortened to its first 8 characters).
    pub fn set_session(&mut self, session_id: &str) {
        self.session_short = session_id.chars().take(SESSION_SHORT_LEN).collect();
    }

    /// Set absolute token counters.
    pub fn set_tokens(&mut self, input: u64, output: u64) {
        self.input_tokens = input;
        self.output_tokens = output;
    }

    /// Add to the token counters.
    pub fn add_tokens(&mut self, input: u64, output: u64) {
        self.input_tokens = self.input_tokens.saturating_add(input);
        self.output_tokens = self.output_tokens.saturating_add(output);
    }

    /// Set auto-mode state and denial counters.
    pub fn set_auto(&mut self, state: impl Into<String>, denials: (u32, u32, u32, u32)) {
        self.auto_state = state.into();
        self.denials = denials;
    }

    /// Set the classifier-running flag.
    pub fn set_classifying(&mut self, classifying: bool) {
        self.classifying = classifying;
    }

    /// Apply an integrator update.
    pub fn apply(&mut self, update: StatusUpdate) {
        match update {
            StatusUpdate::Model { provider, model } => self.set_model(provider, model),
            StatusUpdate::Session(id) => self.set_session(&id),
            StatusUpdate::Tokens { input, output } => self.set_tokens(input, output),
            StatusUpdate::Auto { state, denials } => self.set_auto(state, denials),
            StatusUpdate::Classifying(c) => self.set_classifying(c),
            StatusUpdate::Context { used, approx } => {
                self.context_used = used;
                self.context_approx = approx;
            }
            StatusUpdate::ContextWindow(window) => {
                self.context_window = window;
                self.window_approx = false;
            }
            StatusUpdate::ApproxContextWindow(window) => {
                self.context_window = Some(window);
                self.window_approx = true;
            }
        }
    }
}

/// Longest quoted excerpt of a message in a notice, in characters.
const QUOTE_MAX_CHARS: usize = 40;

/// `“text”`, shortened to its first line and [`QUOTE_MAX_CHARS`] characters.
fn quote_short(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default().trim();
    let mut out: String = line.chars().take(QUOTE_MAX_CHARS).collect();
    if out.len() < line.len() || text.trim().lines().count() > 1 {
        out.push('…');
    }
    format!("“{out}”")
}

/// Tokens in the context after a model call: everything sent plus the reply.
fn context_tokens(usage: &meerkat_core::Usage) -> u64 {
    usage
        .input_tokens
        .saturating_add(usage.output_tokens)
        .saturating_add(usage.cache_creation_tokens.unwrap_or(0))
        .saturating_add(usage.cache_read_tokens.unwrap_or(0))
}

/// Transcript scroll position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollState {
    /// Lines between the bottom of the viewport and the end of the transcript.
    pub offset_from_bottom: usize,
    /// Whether the view sticks to the bottom as output arrives.
    pub follow: bool,
    last_total: usize,
}

impl Default for ScrollState {
    fn default() -> Self {
        Self {
            offset_from_bottom: 0,
            follow: true,
            last_total: 0,
        }
    }
}

impl ScrollState {
    /// Scroll towards older output.
    pub fn up(&mut self, lines: usize) {
        self.offset_from_bottom = self.offset_from_bottom.saturating_add(lines);
        self.follow = false;
    }

    /// Scroll towards newer output; reaching the bottom re-enables follow.
    pub fn down(&mut self, lines: usize) {
        self.offset_from_bottom = self.offset_from_bottom.saturating_sub(lines);
        if self.offset_from_bottom == 0 {
            self.follow = true;
        }
    }

    /// Jump to the oldest output.
    pub fn top(&mut self) {
        self.offset_from_bottom = usize::MAX;
        self.follow = false;
    }

    /// Jump to the newest output and follow.
    pub fn bottom(&mut self) {
        self.offset_from_bottom = 0;
        self.follow = true;
    }

    /// Reconcile with the laid-out transcript and return the first visible line.
    ///
    /// When not following, growth at the bottom shifts the offset so the visible
    /// window stays put.
    pub fn window(&mut self, total: usize, height: usize) -> usize {
        if self.follow {
            self.offset_from_bottom = 0;
        } else if total > self.last_total && self.last_total > 0 {
            self.offset_from_bottom = self
                .offset_from_bottom
                .saturating_add(total - self.last_total);
        }
        self.last_total = total;
        let max_offset = total.saturating_sub(height);
        self.offset_from_bottom = self.offset_from_bottom.min(max_offset);
        if self.offset_from_bottom == 0 {
            self.follow = true;
        }
        total.saturating_sub(height) - self.offset_from_bottom
    }
}

/// Application state.
///
/// `App` is pure: it performs no I/O. Prompt-history persistence is the
/// integrator's job — seed recall with [`App::with_history`] (e.g. from
/// [`History::load`]) and, after each [`App::update`], drain newly submitted
/// lines with [`App::take_submitted`] and append them to the [`History`] file.
#[derive(Debug)]
pub struct App {
    /// Rendered conversation.
    pub transcript: Transcript,
    /// Prompt editor.
    pub input: InputEditor,
    /// Transcript scroll position.
    pub scroll: ScrollState,
    /// Status-bar data.
    pub status: StatusInfo,
    /// Open modal, if any.
    pub modal: Option<Modal>,
    /// Messages typed while a turn was running, not yet delivered.
    pub pending: PendingQueue,
    /// Prompt to send once the interrupted turn ends (Ctrl-X Ctrl-S).
    pub send_next: Option<String>,
    /// Sequence number of the last primary-stream event processed.
    pub last_seq: u64,
    /// Continuation turns still allowed for unanswered mid-turn messages;
    /// reset to 1 whenever a turn starts with user content.
    pub continuations_left: u8,
    /// Normal editing or history search.
    pub mode: InputMode,
    /// When Ctrl-X was pressed, while a chord is pending.
    pub chord: Option<Instant>,
    /// Whether `/compact` is running ([`Effect::Compact`] sent, no
    /// [`Msg::CompactFinished`] yet). Prompts are deferred meanwhile.
    pub compacting: bool,
    /// When a Ctrl-D / `/quit` was refused because work is in flight; a
    /// second one within [`QUIT_CONFIRM_WINDOW`] quits anyway.
    pub quit_confirm: Option<Instant>,
    /// Whether assistant text streamed during the current turn (the turn
    /// summary is shown only when nothing did).
    streamed_text: bool,
    /// Highest pending id allocated when the last continuation turn started;
    /// an unanswered message above it was typed during the continuation.
    continuation_watermark: Option<PendingId>,
    /// A mid-turn message was recorded but not answered and no continuation
    /// is left: Enter on an empty editor asks the model to answer it.
    answer_hint: bool,
    /// Texts of the messages the last report listed as unanswered.
    unanswered_texts: Vec<(PendingId, String)>,
    /// Global (cross-workspace) prompt history, oldest first; searched after the
    /// workspace history.
    global_history: Vec<String>,
    /// Search corpus (newest first, deduplicated), built when a search starts.
    search_corpus: Vec<String>,
    pending_escalations: VecDeque<(EscalationId, EscalationPrompt)>,
    /// Injection failures already announced during the current turn.
    deferred_notified: Vec<InjectFailure>,
    submitted: Vec<String>,
    running: bool,
    run_first_block: usize,
    last_ctrl_c: Option<Instant>,
    viewport_height: usize,
    dirty: bool,
}

impl App {
    /// Create an app with the given status-bar data.
    pub fn new(status: StatusInfo) -> Self {
        Self {
            transcript: Transcript::new(),
            input: InputEditor::new(),
            scroll: ScrollState::default(),
            status,
            modal: None,
            pending: PendingQueue::new(),
            send_next: None,
            last_seq: 0,
            continuations_left: 1,
            mode: InputMode::Normal,
            chord: None,
            compacting: false,
            quit_confirm: None,
            streamed_text: false,
            continuation_watermark: None,
            answer_hint: false,
            unanswered_texts: Vec::new(),
            global_history: Vec::new(),
            search_corpus: Vec::new(),
            pending_escalations: VecDeque::new(),
            deferred_notified: Vec::new(),
            submitted: Vec::new(),
            running: false,
            run_first_block: 0,
            last_ctrl_c: None,
            viewport_height: 20,
            dirty: true,
        }
    }

    /// Seed the prompt history (oldest first).
    pub fn with_history(mut self, history: Vec<String>) -> Self {
        self.input = InputEditor::new().with_history(history);
        self
    }

    /// Seed the global prompt history (oldest first), searched by Ctrl-R after
    /// the workspace history. Up/Down recall stays workspace-only.
    pub fn with_global_history(mut self, history: Vec<String>) -> Self {
        self.global_history = history;
        self
    }

    /// The active history search and its corpus, if searching.
    pub fn search(&self) -> Option<(&HistorySearch, &[String])> {
        match &self.mode {
            InputMode::Search(s) => Some((s, &self.search_corpus)),
            InputMode::Normal => None,
        }
    }

    /// Whether a Ctrl-X chord is pending at `now` (used for the input hint).
    pub fn chord_pending(&self, now: Instant) -> bool {
        self.chord
            .is_some_and(|t| now.saturating_duration_since(t) < CHORD_WINDOW)
    }

    /// Lines submitted with Enter (prompts and slash commands, as typed) since the
    /// last call, oldest first. The integrator appends them to the persistent
    /// [`History`]; blank submissions are never recorded.
    pub fn take_submitted(&mut self) -> Vec<String> {
        std::mem::take(&mut self.submitted)
    }

    /// Whether a turn is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Whether a queued message can be recalled (Esc / ↑ on an empty editor).
    pub fn has_recallable(&self) -> bool {
        self.pending.newest_recallable().is_some()
    }

    /// Escalations waiting behind the open modal.
    pub fn pending_escalations(&self) -> usize {
        self.pending_escalations.len()
    }

    /// Whether a second Ctrl-C at `now` would quit (used for the input hint).
    pub fn quit_armed(&self, now: Instant) -> bool {
        self.last_ctrl_c
            .is_some_and(|t| now.saturating_duration_since(t) < DOUBLE_CTRL_C_WINDOW)
    }

    /// Whether a second Ctrl-D / `/quit` at `now` would quit despite a running
    /// turn or queued messages (used for the input hint).
    pub fn quit_confirm_armed(&self, now: Instant) -> bool {
        self.quit_confirm
            .is_some_and(|t| now.saturating_duration_since(t) < QUIT_CONFIRM_WINDOW)
    }

    /// Record the transcript viewport height (set by the renderer).
    pub fn set_viewport_height(&mut self, height: usize) {
        self.viewport_height = height.max(1);
    }

    /// Return and reset the redraw flag.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// Process one message. Pure: no I/O, only state changes and returned effects.
    pub fn update(&mut self, msg: Msg) -> Vec<Effect> {
        self.update_at(msg, Instant::now())
    }

    /// [`Self::update`] with an explicit clock (for tests).
    pub fn update_at(&mut self, msg: Msg, now: Instant) -> Vec<Effect> {
        let mut fx = Vec::new();
        let mut dirty = true;
        match msg {
            Msg::Key(key) => {
                if self.modal.is_some() {
                    self.modal_key(&key, &mut fx);
                } else {
                    self.main_key(&key, now, &mut fx);
                }
            }
            Msg::Paste(text) => {
                if self.modal.is_none() {
                    self.chord = None;
                    if let InputMode::Search(search) = &mut self.mode {
                        for c in text.chars().filter(|c| !c.is_control()) {
                            search.push(c, &self.search_corpus);
                        }
                    } else {
                        self.input.insert_str(&text);
                    }
                }
            }
            Msg::Resize(..) => self.transcript.invalidate_all(),
            Msg::Mouse(m) => match m.kind {
                MouseEventKind::ScrollUp => self.scroll.up(keymap::LINE_SCROLL),
                MouseEventKind::ScrollDown => self.scroll.down(keymap::LINE_SCROLL),
                _ => dirty = false,
            },
            Msg::Agent(env) => {
                self.last_seq = env.seq;
                match &env.payload {
                    AgentEvent::TurnStarted { .. } => self.mark_seen(env.seq),
                    AgentEvent::TextDelta { delta } if !delta.is_empty() => {
                        self.streamed_text = true;
                    }
                    AgentEvent::TextComplete { content } if !content.is_empty() => {
                        self.streamed_text = true;
                    }
                    AgentEvent::TurnCompleted { usage, .. } => {
                        self.status.context_used = Some(context_tokens(usage));
                        self.status.context_approx = false;
                    }
                    AgentEvent::CompactionCompleted { summary_tokens, .. } => {
                        self.status.context_used = Some(*summary_tokens);
                        self.status.context_approx = true;
                    }
                    _ => {}
                }
                if let Some(usage) = self.transcript.apply_agent_event_at(&env.payload, now) {
                    self.status
                        .add_tokens(usage.input_tokens, usage.output_tokens);
                }
            }
            Msg::Scoped(ev) => self.transcript.apply_scoped_event_at(&ev, now),
            Msg::TurnFinished(outcome) => self.turn_finished(outcome, now, &mut fx),
            Msg::InjectAck { id, result } => self.inject_ack(id, result, now, &mut fx),
            Msg::Recalled(ids) => self.recalled(&ids, &mut fx),
            Msg::Diff {
                tool_use_id,
                path,
                unified,
                added,
                removed,
            } => {
                let view = DiffView {
                    path: path.clone(),
                    unified,
                    added,
                    removed,
                };
                if !self.transcript.attach_diff(&tool_use_id, view) {
                    self.transcript.push_notice(
                        NoticeLevel::Info,
                        format!("edited {path} (+{added} -{removed})"),
                    );
                }
            }
            Msg::Escalation(prompt, id) => self.escalation(prompt, id),
            Msg::EscalationResolved(id) => self.escalation_resolved(id),
            Msg::Approval(n) => {
                self.transcript.push(Block::Approval {
                    tool: n.tool,
                    tier: n.tier,
                    allowed: n.allowed,
                    reason: n.reason,
                });
            }
            Msg::Notice { level, text } => {
                self.transcript.push_notice(level, text);
            }
            Msg::Status(update) => {
                // Envelope sequence numbers restart with every session task.
                if matches!(update, StatusUpdate::Session(_)) {
                    self.last_seq = 0;
                }
                self.status.apply(update);
            }
            Msg::ShowResumePicker(items) => {
                if self.modal.is_none() {
                    self.modal = Some(Modal::Resume(ResumePicker { items, selected: 0 }));
                }
            }
            Msg::ResetTranscript(blocks) => {
                self.transcript.reset(blocks);
                self.last_seq = 0;
                self.pending.clear_settled();
                // Keep placeholders for messages still in flight.
                for item in self.pending.iter_mut() {
                    item.block = self.transcript.push(Block::Queued {
                        id: item.id,
                        text: item.text.clone(),
                        label: item.state.label(),
                    });
                }
                self.run_first_block = self.transcript.len();
                self.scroll.bottom();
            }
            Msg::CompactFinished(result) => self.compact_finished(result, now, &mut fx),
            Msg::Tick => {
                dirty = self.status.running_since.is_some()
                    || self.status.classifying
                    || self.last_ctrl_c.is_some()
                    || self.quit_confirm.is_some()
                    || self.chord.is_some();
                if self.last_ctrl_c.is_some() && !self.quit_armed(now) {
                    self.last_ctrl_c = None;
                }
                if self.quit_confirm.is_some() && !self.quit_confirm_armed(now) {
                    self.quit_confirm = None;
                }
                if self.chord.is_some() && !self.chord_pending(now) {
                    self.chord = None;
                }
            }
        }
        self.status.queued = self.pending.len();
        self.dirty |= dirty;
        fx
    }

    fn main_key(&mut self, key: &KeyEvent, now: Instant, fx: &mut Vec<Effect>) {
        if let InputMode::Search(search) = &mut self.mode {
            self.last_ctrl_c = None;
            match search.key(key, &self.search_corpus) {
                SearchOutcome::Continue => {}
                SearchOutcome::Accept(text) | SearchOutcome::Cancel(text) => {
                    self.end_search(&text);
                }
            }
            return;
        }
        if let Some(started) = self.chord.take()
            && now.saturating_duration_since(started) < CHORD_WINDOW
        {
            use ratatui::crossterm::event::{KeyCode, KeyModifiers};
            match key.code {
                KeyCode::Esc => {
                    self.last_ctrl_c = None;
                    return;
                }
                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.last_ctrl_c = None;
                    self.interrupt_and_send(now, fx);
                    return;
                }
                // Anything else cancels the chord and is processed normally.
                _ => {}
            }
        }
        let action = keymap::map_key(key, self.input.is_empty());
        if action != Action::CtrlC {
            self.last_ctrl_c = None;
        }
        let page = self.viewport_height.saturating_sub(2).max(1);
        match action {
            Action::Submit => self.submit(now, fx),
            Action::Newline => self.input.newline(),
            Action::CtrlC => self.ctrl_c(now, fx),
            Action::CtrlD => {
                if self.input.is_empty() {
                    self.request_quit(now, fx);
                } else {
                    self.input.delete();
                }
            }
            Action::Escape => {
                if self.has_recallable() {
                    self.recall_all(fx);
                } else if self.running {
                    self.interrupt(fx);
                }
            }
            Action::Edit(EditAction::Up) if self.input.is_empty() && self.has_recallable() => {
                self.recall_newest(fx);
            }
            Action::ToggleReasoning => {
                self.transcript.toggle_last_reasoning();
            }
            Action::HistorySearch => self.start_search(),
            Action::ChordPrefix => self.chord = Some(now),
            Action::ScrollUp(n) => self.scroll.up(n),
            Action::ScrollDown(n) => self.scroll.down(n),
            Action::PageUp => self.scroll.up(page),
            Action::PageDown => self.scroll.down(page),
            Action::ScrollTop => self.scroll.top(),
            Action::ScrollBottom => self.scroll.bottom(),
            Action::Edit(edit) => self.edit(edit),
            Action::None => {}
        }
    }

    fn start_search(&mut self) {
        self.search_corpus = search::build_corpus(self.input.history(), &self.global_history);
        self.mode = InputMode::Search(HistorySearch::start(self.input.text(), &self.search_corpus));
    }

    /// Leave search mode with `text` in the editor (cursor at the end).
    fn end_search(&mut self, text: &str) {
        self.mode = InputMode::Normal;
        self.search_corpus = Vec::new();
        self.input.clear();
        self.input.set_text(text);
    }

    /// Ctrl-X Ctrl-S: when idle, the same as Enter; when running, interrupt and
    /// send the editor contents next (plain interrupt when the editor is empty).
    fn interrupt_and_send(&mut self, now: Instant, fx: &mut Vec<Effect>) {
        if !self.running {
            self.submit(now, fx);
        } else if self.input.text().trim().is_empty() {
            self.interrupt(fx);
        } else {
            let text = self.input.take();
            self.submitted.push(text.clone());
            self.send_next = Some(match self.send_next.take() {
                Some(prev) => format!("{prev}\n\n{text}"),
                None => text.clone(),
            });
            fx.push(Effect::InterruptAndSend(text));
        }
    }

    fn edit(&mut self, edit: EditAction) {
        let e = &mut self.input;
        match edit {
            EditAction::Insert(c) => e.insert_char(c),
            EditAction::Backspace => e.backspace(),
            EditAction::Delete => e.delete(),
            EditAction::Left => e.left(),
            EditAction::Right => e.right(),
            EditAction::Home => e.home(),
            EditAction::End => e.end(),
            EditAction::WordLeft => e.word_left(),
            EditAction::WordRight => e.word_right(),
            EditAction::Up => e.up(),
            EditAction::Down => e.down(),
            EditAction::DeleteWordBack => e.delete_word_back(),
            EditAction::KillToLineStart => e.kill_to_line_start(),
        }
    }

    fn modal_key(&mut self, key: &KeyEvent, fx: &mut Vec<Effect>) {
        use ratatui::crossterm::event::{KeyCode, KeyModifiers};
        let ctrl_c =
            key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c');
        match &mut self.modal {
            Some(Modal::Escalation { id, .. }) => {
                let id = *id;
                if ctrl_c {
                    self.stop_all_escalations(fx);
                    return;
                }
                let answer = match key.code {
                    KeyCode::Char('y' | 'Y') => EscalationAnswer::AllowOnce,
                    KeyCode::Char('a' | 'A') => EscalationAnswer::AllowAndAddRule,
                    KeyCode::Char('n' | 'N') | KeyCode::Esc => EscalationAnswer::Deny,
                    KeyCode::Char('s' | 'S') => EscalationAnswer::Stop,
                    _ => return,
                };
                fx.push(Effect::RespondEscalation(id, answer));
                self.advance_escalation();
            }
            Some(Modal::Resume(picker)) => match key.code {
                _ if ctrl_c => self.modal = None,
                KeyCode::Up | KeyCode::Char('k') => picker.up(),
                KeyCode::Down | KeyCode::Char('j') => picker.down(),
                KeyCode::Enter => {
                    let selected = picker.selected_ref().map(str::to_string);
                    self.modal = None;
                    if let Some(r) = selected {
                        fx.push(Effect::ResumeSession(r));
                    }
                }
                KeyCode::Esc | KeyCode::Char('q') => self.modal = None,
                _ => {}
            },
            None => {}
        }
    }

    fn submit(&mut self, now: Instant, fx: &mut Vec<Effect>) {
        if self.input.text().trim().is_empty() {
            if self.answer_hint && !self.running && !self.compacting {
                // Ask the model to answer the recorded-but-unanswered message.
                self.input.clear();
                self.continuations_left = 0;
                self.continuation_watermark = Some(self.pending.last_id());
                self.begin_turn(CONTINUATION_PROMPT.to_string(), false, now, fx);
            }
            return;
        }
        let raw = self.input.take();
        self.submitted.push(raw.clone());
        match commands::parse(&raw) {
            Some(Ok(cmd)) => self.run_command(cmd, now, fx),
            Some(Err(err)) => {
                self.transcript
                    .push_notice(NoticeLevel::Warn, err.to_string());
            }
            None => {
                if let Some(text) = commands::parse_remember(&raw) {
                    self.run_command(Command::Remember(text), now, fx);
                    return;
                }
                let prompt = match raw.trim_start().strip_prefix("//") {
                    Some(rest) => format!("/{rest}"),
                    None => commands::unescape_hash(&raw).unwrap_or(raw),
                };
                if self.compacting {
                    self.defer(prompt);
                } else if self.running {
                    self.enqueue(prompt, fx);
                } else {
                    self.start_turn(prompt, now, fx);
                }
            }
        }
    }

    /// Start a turn with user content: resets the continuation budget and
    /// shows `prompt` as a user block.
    fn start_turn(&mut self, prompt: String, now: Instant, fx: &mut Vec<Effect>) {
        self.continuations_left = 1;
        self.continuation_watermark = None;
        self.begin_turn(prompt, true, now, fx);
    }

    fn begin_turn(&mut self, prompt: String, user_block: bool, now: Instant, fx: &mut Vec<Effect>) {
        self.run_first_block = self.transcript.len();
        self.streamed_text = false;
        self.answer_hint = false;
        if user_block {
            self.transcript.push(Block::User {
                text: prompt.clone(),
            });
        }
        self.running = true;
        self.status.running_since = Some(now);
        self.scroll.bottom();
        fx.push(Effect::StartTurn(prompt));
    }

    /// Enter while a turn runs: queue the message and inject it.
    fn enqueue(&mut self, text: String, fx: &mut Vec<Effect>) {
        let id = self.pending.alloc_id();
        let block = self.transcript.push(Block::Queued {
            id,
            text: text.clone(),
            label: QueuedLabel::Queued,
        });
        self.pending.push(Pending {
            id,
            text: text.clone(),
            state: PendingState::Queued,
            block,
        });
        self.scroll.bottom();
        fx.push(Effect::Inject { id, text });
    }

    /// Enter while compacting: hold the message until compaction finishes.
    fn defer(&mut self, text: String) {
        let id = self.pending.alloc_id();
        let block = self.transcript.push(Block::Queued {
            id,
            text: text.clone(),
            label: QueuedLabel::Deferred,
        });
        self.pending.push(Pending {
            id,
            text,
            state: PendingState::Deferred,
            block,
        });
        self.scroll.bottom();
    }

    /// `/compact` finished: send what was typed meanwhile, or hand it back to
    /// the editor when compaction failed.
    fn compact_finished(&mut self, result: Result<(), String>, now: Instant, fx: &mut Vec<Effect>) {
        self.compacting = false;
        let deferred = self
            .pending
            .take_where(|p| p.state == PendingState::Deferred);
        match result {
            Ok(()) => {
                self.transcript.push_notice(NoticeLevel::Info, "compacted");
                if self.running {
                    // Cannot happen (compaction is idle-only); keep them for
                    // the post-turn logic rather than starting a second turn.
                    for item in deferred {
                        self.pending.push(item);
                    }
                } else if !deferred.is_empty() {
                    self.send_unsent(deferred, now, fx);
                }
            }
            Err(err) => {
                self.transcript
                    .push_notice(NoticeLevel::Error, format!("compaction failed: {err}"));
                self.drop_blocks(deferred.iter().map(|p| p.block));
                self.prepend_to_editor(deferred.into_iter().map(|p| p.text).collect());
            }
        }
    }

    fn set_pending_state(&mut self, id: PendingId, state: PendingState) {
        let Some(item) = self.pending.get_mut(id) else {
            return;
        };
        item.state = state;
        let block = item.block;
        if let Some(Block::Queued { label, .. }) = self.transcript.block_mut(block) {
            *label = state.label();
        }
    }

    /// Show a pending message as delivered (its placeholder becomes a user
    /// block in place) and keep its text until the post-turn report.
    fn settle(&mut self, id: PendingId) {
        if let Some(item) = self.pending.settle(id) {
            let (block, text) = (item.block, item.text.clone());
            self.transcript.replace_block(block, Block::User { text });
        }
    }

    /// Remove transcript blocks (highest index first, so the remaining ids
    /// stay valid) and fix up every id the app holds.
    fn drop_blocks(&mut self, blocks: impl IntoIterator<Item = BlockId>) {
        let mut blocks: Vec<BlockId> = blocks.into_iter().collect();
        blocks.sort_by_key(|b| std::cmp::Reverse(b.0));
        for block in blocks {
            if self.transcript.remove_block(block).is_some() {
                self.pending.block_removed(block);
                if self.run_first_block > block.0 {
                    self.run_first_block -= 1;
                }
            }
        }
    }

    /// Take pending items out of the queue and the transcript, returning their
    /// texts oldest first.
    fn withdraw(&mut self, ids: &[PendingId]) -> Vec<String> {
        let taken = self.pending.take_where(|p| ids.contains(&p.id));
        self.drop_blocks(taken.iter().map(|p| p.block));
        taken.into_iter().map(|p| p.text).collect()
    }

    /// Put recalled texts (oldest first) before the editor contents.
    fn prepend_to_editor(&mut self, texts: Vec<String>) {
        if texts.is_empty() {
            return;
        }
        let mut text = texts.join("\n");
        let current = self.input.text();
        if !text.is_empty() && !current.is_empty() {
            text.push('\n');
        }
        text.push_str(&current);
        self.input.clear();
        self.input.set_text(&text);
    }

    /// A `TurnStarted` with `seq`: messages acknowledged before it reached the model.
    fn mark_seen(&mut self, seq: u64) {
        let seen: Vec<PendingId> = self
            .pending
            .items()
            .iter()
            .filter(|p| matches!(p.state, PendingState::Delivering { ack_seq } if seq > ack_seq))
            .map(|p| p.id)
            .collect();
        for id in seen {
            self.settle(id);
        }
    }

    fn inject_ack(
        &mut self,
        id: PendingId,
        result: Result<(), InjectFailure>,
        now: Instant,
        fx: &mut Vec<Effect>,
    ) {
        let Some((state, text)) = self.pending.get(id).map(|p| (p.state, p.text.clone())) else {
            return;
        };
        if state == PendingState::Recalling && result.is_err() {
            // Never reached the inbox: hand it straight back.
            let texts = self.withdraw(&[id]);
            self.prepend_to_editor(texts);
            return;
        }
        match result {
            Ok(()) => {
                if state == PendingState::Queued {
                    let ack_seq = self.last_seq;
                    self.set_pending_state(id, PendingState::Delivering { ack_seq });
                }
            }
            Err(InjectFailure::NotRunning) if self.running => {
                fx.push(Effect::Inject { id, text });
            }
            Err(InjectFailure::NotRunning) => {
                if let Some(item) = self.pending.remove(id) {
                    self.transcript
                        .replace_block(item.block, Block::User { text: item.text });
                }
                self.continuations_left = 1;
                self.continuation_watermark = None;
                self.begin_turn(text, false, now, fx);
            }
            Err(cause) => {
                self.set_pending_state(id, PendingState::Deferred);
                if !self.deferred_notified.contains(&cause) {
                    self.transcript.push_notice(
                        NoticeLevel::Warn,
                        format!("couldn't deliver mid-turn: {cause}; will send after this turn"),
                    );
                    self.deferred_notified.push(cause);
                }
            }
        }
    }

    /// Esc with messages pending: pull every one of them back into the editor.
    fn recall_all(&mut self, fx: &mut Vec<Effect>) {
        let deferred: Vec<PendingId> = self
            .pending
            .items()
            .iter()
            .filter(|p| p.state == PendingState::Deferred)
            .map(|p| p.id)
            .collect();
        let in_inbox: Vec<PendingId> = self
            .pending
            .items()
            .iter()
            .filter(|p| p.state.in_inbox())
            .map(|p| p.id)
            .collect();
        let texts = self.withdraw(&deferred);
        self.prepend_to_editor(texts);
        for id in &in_inbox {
            self.set_pending_state(*id, PendingState::Recalling);
        }
        if !in_inbox.is_empty() {
            fx.push(Effect::RecallPending);
        }
    }

    /// ↑ on an empty editor with messages pending: pull back the newest one.
    fn recall_newest(&mut self, fx: &mut Vec<Effect>) {
        let Some((id, state)) = self.pending.newest_recallable().map(|p| (p.id, p.state)) else {
            return;
        };
        if state.in_inbox() {
            self.set_pending_state(id, PendingState::Recalling);
            fx.push(Effect::RecallPending);
        } else {
            let texts = self.withdraw(&[id]);
            self.prepend_to_editor(texts);
        }
    }

    /// The integrator drained `ids` from the inbox before the model saw them.
    fn recalled(&mut self, ids: &[PendingId], fx: &mut Vec<Effect>) {
        let wanted: Vec<PendingId> = self
            .pending
            .items()
            .iter()
            .filter(|p| p.state == PendingState::Recalling && ids.contains(&p.id))
            .map(|p| p.id)
            .collect();
        // Recalling items that were not drained had already reached the model.
        let delivered: Vec<PendingId> = self
            .pending
            .items()
            .iter()
            .filter(|p| p.state == PendingState::Recalling && !ids.contains(&p.id))
            .map(|p| p.id)
            .collect();
        for id in delivered {
            self.settle(id);
        }
        // Drained but not asked for (a single-message recall drains the whole
        // inbox, or a message was wrongly shown as delivered): send it again.
        for &id in ids {
            if wanted.contains(&id) {
                continue;
            }
            if let Some(item) = self.pending.take_settled(id) {
                let (block, text) = (item.block, item.text.clone());
                self.transcript.replace_block(
                    block,
                    Block::Queued {
                        id,
                        text: text.clone(),
                        label: QueuedLabel::Queued,
                    },
                );
                self.pending.push(Pending {
                    state: PendingState::Queued,
                    ..item
                });
                fx.push(Effect::Inject { id, text });
            } else if let Some(text) = self.pending.get(id).map(|p| p.text.clone()) {
                self.set_pending_state(id, PendingState::Queued);
                fx.push(Effect::Inject { id, text });
            }
        }
        let texts = self.withdraw(&wanted);
        self.prepend_to_editor(texts);
    }

    fn turn_finished(&mut self, outcome: TurnOutcome, now: Instant, fx: &mut Vec<Effect>) {
        let TurnOutcome { result, delivery } = outcome;
        self.running = false;
        self.status.running_since = None;
        self.status.classifying = false;
        let interrupted = matches!(&result, Err(e) if e == "interrupted");
        let succeeded = result.is_ok();
        match result {
            Ok(summary) => {
                if !self.streamed_text && !summary.text.is_empty() {
                    self.transcript.push(Block::Assistant { md: summary.text });
                }
            }
            Err(err) => {
                self.transcript
                    .push_notice(NoticeLevel::Error, format!("turn failed: {err}"));
            }
        }
        self.streamed_text = false;
        self.deferred_notified.clear();
        // Texts of unanswered messages, kept across continuation turns (the
        // queue forgets settled items once the report is applied).
        let unanswered: Vec<(PendingId, String)> = delivery
            .unanswered
            .iter()
            .filter_map(|id| {
                let text = self.pending.text_of(*id).map(str::to_string).or_else(|| {
                    self.unanswered_texts
                        .iter()
                        .find(|(known, _)| known == id)
                        .map(|(_, text)| text.clone())
                })?;
                Some((*id, text))
            })
            .collect();
        let unsent = self.apply_report(&delivery);
        self.next_after_turn(unsent, &delivery, interrupted, succeeded, now, fx);
        if !self.running && succeeded && !delivery.unanswered.is_empty() {
            self.answer_hint = true;
            let quoted: Vec<String> = unanswered.iter().map(|(_, t)| quote_short(t)).collect();
            let what = if quoted.is_empty() {
                "your message".to_string()
            } else {
                format!("your message {}", quoted.join(", "))
            };
            self.transcript.push_notice(
                NoticeLevel::Warn,
                format!("{what} was recorded but not answered; send it again or press Enter"),
            );
        }
        self.unanswered_texts = unanswered;
    }

    /// Apply a delivery report: delivered messages become user blocks; returns
    /// the messages that never reached the model (undelivered or deferred),
    /// oldest first. Messages the user asked back go to the editor.
    fn apply_report(&mut self, report: &DeliveryReport) -> Vec<Pending> {
        for id in report.answered.iter().chain(&report.unanswered) {
            self.settle(*id);
        }
        let mut unsent: Vec<Pending> = report
            .undelivered
            .iter()
            .filter_map(|id| self.pending.take_settled(*id))
            .collect();
        // Deferred messages were never injected; a message still shown as
        // delivering that the report does not mention is treated as unsent.
        unsent.extend(self.pending.take_where(|p| {
            report.undelivered.contains(&p.id)
                || p.state == PendingState::Deferred
                || matches!(p.state, PendingState::Delivering { .. })
        }));
        self.pending.clear_settled();
        unsent.sort_by_key(|p| p.id);
        let (recalled, unsent): (Vec<Pending>, Vec<Pending>) = unsent
            .into_iter()
            .partition(|p| p.state == PendingState::Recalling);
        self.drop_blocks(recalled.iter().map(|p| p.block));
        self.prepend_to_editor(recalled.into_iter().map(|p| p.text).collect());
        unsent
    }

    /// Decide what to send after a turn, in priority order: the Ctrl-X Ctrl-S
    /// prompt (with anything unsent), unsent messages back to the editor after
    /// an interrupt, unsent messages as a new turn, one continuation turn for
    /// messages the model saw but did not answer.
    fn next_after_turn(
        &mut self,
        unsent: Vec<Pending>,
        report: &DeliveryReport,
        interrupted: bool,
        succeeded: bool,
        now: Instant,
        fx: &mut Vec<Effect>,
    ) {
        if let Some(next) = self.send_next.take() {
            let mut texts = self.unsent_as_user(unsent);
            texts.push(next.clone());
            self.continuations_left = 1;
            self.transcript.push(Block::User { text: next });
            self.begin_turn(texts.join("\n\n"), false, now, fx);
        } else if interrupted {
            self.drop_blocks(unsent.iter().map(|p| p.block));
            self.prepend_to_editor(unsent.into_iter().map(|p| p.text).collect());
        } else if !unsent.is_empty() {
            self.send_unsent(unsent, now, fx);
        } else if !report.unanswered.is_empty() && succeeded {
            // A message typed during the continuation turn earns its own.
            if self
                .continuation_watermark
                .is_some_and(|mark| report.unanswered.iter().any(|id| *id > mark))
            {
                self.continuations_left = 1;
            }
            if self.continuations_left > 0 {
                self.continuations_left -= 1;
                self.continuation_watermark = Some(self.pending.last_id());
                self.transcript
                    .push_notice(NoticeLevel::Info, "continuing: answering your message");
                self.begin_turn(CONTINUATION_PROMPT.to_string(), false, now, fx);
            }
        }
    }

    /// Start one turn with unsent messages (joined by blank lines), showing
    /// them as user blocks in place.
    fn send_unsent(&mut self, unsent: Vec<Pending>, now: Instant, fx: &mut Vec<Effect>) {
        let texts = self.unsent_as_user(unsent);
        self.continuations_left = 1;
        self.continuation_watermark = None;
        self.begin_turn(texts.join("\n\n"), false, now, fx);
    }

    /// Show unsent messages as user blocks (they are about to be sent) and
    /// return their texts.
    fn unsent_as_user(&mut self, unsent: Vec<Pending>) -> Vec<String> {
        unsent
            .into_iter()
            .map(|p| {
                self.transcript.replace_block(
                    p.block,
                    Block::User {
                        text: p.text.clone(),
                    },
                );
                p.text
            })
            .collect()
    }

    fn ctrl_c(&mut self, now: Instant, fx: &mut Vec<Effect>) {
        if self.running {
            self.last_ctrl_c = None;
            self.interrupt(fx);
        } else if !self.input.is_empty() {
            self.last_ctrl_c = None;
            self.input.clear();
        } else if self.quit_armed(now) {
            fx.push(Effect::Quit);
        } else {
            self.last_ctrl_c = Some(now);
        }
    }

    /// Interrupt the running turn. Pending messages are settled by the
    /// post-turn delivery report.
    fn interrupt(&mut self, fx: &mut Vec<Effect>) {
        fx.push(Effect::Interrupt);
    }

    /// Ctrl-D on an empty editor or `/quit`: quit right away when idle;
    /// otherwise warn and quit only on a second request within
    /// [`QUIT_CONFIRM_WINDOW`]. Quitting hands every held message to the
    /// integrator ([`Effect::SaveUnsent`]) so undelivered ones are saved.
    fn request_quit(&mut self, now: Instant, fx: &mut Vec<Effect>) {
        let queued = self.pending.len() + usize::from(self.send_next.is_some());
        let busy = self.running || queued > 0;
        if busy && !self.quit_confirm_armed(now) {
            self.quit_confirm = Some(now);
            let mut what = Vec::new();
            if self.running {
                what.push("a turn is running".to_string());
            }
            if queued > 0 {
                what.push(format!(
                    "{queued} message{} queued",
                    if queued == 1 { "" } else { "s" }
                ));
            }
            self.transcript.push_notice(
                NoticeLevel::Warn,
                format!(
                    "{}; press again to quit anyway (the turn is interrupted, undelivered messages are saved)",
                    what.join(" / ")
                ),
            );
            self.scroll.bottom();
            return;
        }
        self.quit_confirm = None;
        let mut held: Vec<(Option<PendingId>, String)> = self
            .pending
            .all()
            .map(|p| (Some(p.id), p.text.clone()))
            .collect();
        if let Some(next) = &self.send_next {
            held.push((None, next.clone()));
        }
        if !held.is_empty() {
            fx.push(Effect::SaveUnsent(held));
        }
        fx.push(Effect::Quit);
    }

    fn run_command(&mut self, cmd: Command, now: Instant, fx: &mut Vec<Effect>) {
        let busy = |app: &mut Self| {
            app.transcript.push_notice(
                NoticeLevel::Warn,
                "a turn is running; interrupt it first (Ctrl-C)",
            );
        };
        let compacting = |app: &mut Self| {
            app.transcript.push_notice(
                NoticeLevel::Warn,
                "compaction is running; wait for it to finish",
            );
        };
        match cmd {
            Command::Quit => self.request_quit(now, fx),
            Command::Help => {
                self.transcript
                    .push_notice(NoticeLevel::Info, commands::HELP_TEXT);
            }
            Command::Clear => {
                if self.running {
                    busy(self);
                } else if self.compacting {
                    compacting(self);
                } else if !self.pending.is_empty() {
                    self.transcript.push_notice(
                        NoticeLevel::Warn,
                        "messages are still queued; wait for them or recall them (Esc)",
                    );
                } else {
                    // The integrator resets the transcript and counters once
                    // the new session is live (and keeps them on failure).
                    fx.push(Effect::NewSession);
                }
            }
            Command::Resume(reference) => {
                if self.running {
                    busy(self);
                } else if self.compacting {
                    compacting(self);
                } else if !self.pending.is_empty() {
                    self.transcript.push_notice(
                        NoticeLevel::Warn,
                        "messages are still queued; wait for them or recall them (Esc)",
                    );
                } else {
                    fx.push(reference.map_or(Effect::RequestResumePicker, Effect::ResumeSession));
                }
            }
            Command::Remember(text) => {
                self.transcript
                    .push_notice(NoticeLevel::Info, format!("remembering: {text}"));
                fx.push(Effect::Remember(text));
            }
            Command::Compact => {
                if self.compacting {
                    compacting(self);
                } else if self.running || !self.pending.is_empty() {
                    self.transcript.push_notice(
                        NoticeLevel::Warn,
                        "finish or interrupt the current turn first",
                    );
                } else {
                    self.compacting = true;
                    self.transcript
                        .push_notice(NoticeLevel::Info, "compacting…");
                    self.scroll.bottom();
                    fx.push(Effect::Compact);
                }
            }
            Command::Model(Some(model)) => {
                if self.running || !self.pending.is_empty() {
                    self.transcript.push_notice(
                        NoticeLevel::Warn,
                        "/model is only available while idle; wait for the turn or interrupt it (Ctrl-C)",
                    );
                } else if self.compacting {
                    compacting(self);
                } else {
                    fx.push(Effect::SetModel(model));
                }
            }
            Command::Model(None) => {
                let text = format!("model: {}/{}", self.status.provider, self.status.model);
                self.transcript.push_notice(NoticeLevel::Info, text);
            }
            Command::Auto(AutoCommand::Status) => {
                let (c, cm, t, tm) = self.status.denials;
                let text = format!(
                    "auto mode: {} (consecutive denials {c}/{cm}, total {t}/{tm})",
                    self.status.auto_state
                );
                self.transcript.push_notice(NoticeLevel::Info, text);
            }
            Command::Auto(AutoCommand::On) => fx.push(Effect::SetAutoMode(true)),
            Command::Auto(AutoCommand::Off) => fx.push(Effect::SetAutoMode(false)),
            Command::Auto(AutoCommand::Resume) => fx.push(Effect::ResumeAutoMode),
            Command::Rules(RulesCommand::List) => fx.push(Effect::ListRules),
            Command::Rules(RulesCommand::Add {
                effect,
                tool,
                pattern,
            }) => fx.push(Effect::AddRule {
                effect,
                tool,
                pattern,
            }),
            Command::Rules(RulesCommand::Remove(n)) => fx.push(Effect::RemoveRule(n)),
        }
    }

    fn escalation(&mut self, prompt: EscalationPrompt, id: EscalationId) {
        match &mut self.modal {
            Some(Modal::Escalation { queued_behind, .. }) => {
                self.pending_escalations.push_back((id, prompt));
                *queued_behind = self.pending_escalations.len();
            }
            _ => {
                self.modal = Some(Modal::Escalation {
                    id,
                    prompt,
                    queued_behind: self.pending_escalations.len(),
                });
            }
        }
    }

    fn advance_escalation(&mut self) {
        self.modal = self
            .pending_escalations
            .pop_front()
            .map(|(id, prompt)| Modal::Escalation {
                id,
                prompt,
                queued_behind: self.pending_escalations.len(),
            });
    }

    fn escalation_resolved(&mut self, resolved: EscalationId) {
        match &mut self.modal {
            Some(Modal::Escalation { id, .. }) if *id == resolved => self.advance_escalation(),
            Some(Modal::Escalation { queued_behind, .. }) => {
                self.pending_escalations.retain(|(id, _)| *id != resolved);
                *queued_behind = self.pending_escalations.len();
            }
            _ => self.pending_escalations.retain(|(id, _)| *id != resolved),
        }
    }

    fn stop_all_escalations(&mut self, fx: &mut Vec<Effect>) {
        if let Some(Modal::Escalation { id, .. }) = self.modal.take() {
            fx.push(Effect::RespondEscalation(id, EscalationAnswer::Stop));
        }
        for (id, _) in self.pending_escalations.drain(..) {
            fx.push(Effect::RespondEscalation(id, EscalationAnswer::Stop));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use meerkat_core::{StopReason, Usage};
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    use serde_json::json;

    fn key(code: KeyCode) -> Msg {
        Msg::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn ctrl(c: char) -> Msg {
        Msg::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }

    fn agent(ev: AgentEvent) -> Msg {
        Msg::Agent(EventEnvelope::new("s", 0, None, ev))
    }

    fn type_str(app: &mut App, s: &str) {
        for c in s.chars() {
            app.update(key(KeyCode::Char(c)));
        }
    }

    fn summary() -> TurnSummary {
        TurnSummary {
            text: String::new(),
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    #[test]
    fn submit_starts_turn_and_streams_into_one_block() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "hi");
        let fx = app.update(key(KeyCode::Enter));
        assert_eq!(fx, vec![Effect::StartTurn("hi".into())]);
        assert!(app.is_running());
        assert!(app.status.running_since.is_some());
        app.update(agent(AgentEvent::TurnStarted { turn_number: 0 }));
        for d in ["Hel", "lo ", "there"] {
            app.update(agent(AgentEvent::TextDelta { delta: d.into() }));
        }
        app.update(agent(AgentEvent::TurnCompleted {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 4,
                ..Usage::default()
            },
        }));
        assert_eq!(
            app.transcript.blocks(),
            &[
                Block::User { text: "hi".into() },
                Block::Assistant {
                    md: "Hello there".into()
                }
            ]
        );
        assert_eq!((app.status.input_tokens, app.status.output_tokens), (10, 4));
        app.update(Msg::TurnFinished(TurnOutcome::plain(Ok(summary()))));
        assert!(!app.is_running());
        assert!(app.status.running_since.is_none());
    }

    #[test]
    fn turn_summary_text_used_when_nothing_streamed() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "q");
        app.update(key(KeyCode::Enter));
        app.update(Msg::TurnFinished(TurnOutcome::plain(Ok(TurnSummary {
            text: "answer".into(),
            ..summary()
        }))));
        assert_eq!(
            app.transcript.blocks().last(),
            Some(&Block::Assistant {
                md: "answer".into()
            })
        );
    }

    #[test]
    fn tool_lifecycle_updates_block_states() {
        let mut app = App::new(StatusInfo::default());
        app.update(agent(AgentEvent::ToolCallRequested {
            id: "t1".into(),
            name: "edit_file".into(),
            args: json!({"path": "src/x.rs"}),
        }));
        let id = app.transcript.tool_block("t1").unwrap();
        app.update(agent(AgentEvent::ToolExecutionStarted {
            id: "t1".into(),
            name: "edit_file".into(),
        }));
        assert!(matches!(
            app.transcript.block(id),
            Some(Block::Tool {
                state: ToolState::Running { .. },
                ..
            })
        ));
        app.update(Msg::Diff {
            tool_use_id: "t1".into(),
            path: "src/x.rs".into(),
            unified: "-a\n+b".into(),
            added: 1,
            removed: 1,
        });
        app.update(agent(AgentEvent::ToolExecutionCompleted {
            id: "t1".into(),
            name: "edit_file".into(),
            result: "ok".into(),
            is_error: true,
            duration_ms: 12,
            has_images: false,
        }));
        match app.transcript.block(id) {
            Some(Block::Tool {
                state,
                result_preview,
                diff: Some(d),
                ..
            }) => {
                assert_eq!(
                    *state,
                    ToolState::Done {
                        is_error: true,
                        duration_ms: 12
                    }
                );
                assert_eq!(result_preview.as_deref(), Some("ok"));
                assert_eq!(d.added, 1);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    fn agent_at(ev: AgentEvent, seq: u64) -> Msg {
        Msg::Agent(EventEnvelope::new("s", seq, None, ev))
    }

    fn submit(app: &mut App, s: &str) -> Vec<Effect> {
        type_str(app, s);
        app.update(key(KeyCode::Enter))
    }

    fn finished(result: Result<TurnSummary, String>, delivery: DeliveryReport) -> Msg {
        Msg::TurnFinished(TurnOutcome { result, delivery })
    }

    fn ack(id: u64, result: Result<(), InjectFailure>) -> Msg {
        Msg::InjectAck {
            id: PendingId(id),
            result,
        }
    }

    fn queued_block(id: u64, text: &str, label: QueuedLabel) -> Block {
        Block::Queued {
            id: PendingId(id),
            text: text.into(),
            label,
        }
    }

    fn user(text: &str) -> Block {
        Block::User { text: text.into() }
    }

    /// A running app with "one" as the turn prompt.
    fn running_app() -> App {
        let mut app = App::new(StatusInfo::default());
        assert_eq!(
            submit(&mut app, "one"),
            vec![Effect::StartTurn("one".into())]
        );
        app
    }

    #[test]
    fn enter_while_running_queues_and_injects() {
        let mut app = running_app();
        assert_eq!(
            submit(&mut app, "two"),
            vec![Effect::Inject {
                id: PendingId(1),
                text: "two".into()
            }]
        );
        assert_eq!(
            submit(&mut app, "three"),
            vec![Effect::Inject {
                id: PendingId(2),
                text: "three".into()
            }]
        );
        assert_eq!(app.status.queued, 2);
        assert_eq!(
            app.transcript.blocks(),
            &[
                user("one"),
                queued_block(1, "two", QueuedLabel::Queued),
                queued_block(2, "three", QueuedLabel::Queued),
            ]
        );
        assert!(app.has_recallable());
    }

    #[test]
    fn ack_ok_then_turn_started_delivers_in_place() {
        let mut app = running_app();
        app.update(agent_at(AgentEvent::TurnStarted { turn_number: 0 }, 3));
        app.update(agent_at(AgentEvent::TextDelta { delta: "a".into() }, 4));
        submit(&mut app, "two");
        app.update(ack(1, Ok(())));
        assert_eq!(
            app.pending.get(PendingId(1)).map(|p| p.state),
            Some(PendingState::Delivering { ack_seq: 4 })
        );
        assert_eq!(
            app.transcript.block(BlockId(2)),
            Some(&queued_block(1, "two", QueuedLabel::Delivering))
        );
        // An event at or below ack_seq does not count.
        app.update(agent_at(AgentEvent::TurnStarted { turn_number: 1 }, 4));
        assert_eq!(app.status.queued, 1);
        app.update(agent_at(AgentEvent::TurnStarted { turn_number: 1 }, 5));
        app.update(agent_at(AgentEvent::TextDelta { delta: "b".into() }, 6));
        assert_eq!(app.status.queued, 0);
        // Replaced in place: transcript order is preserved.
        assert_eq!(
            app.transcript.blocks(),
            &[
                user("one"),
                Block::Assistant { md: "a".into() },
                user("two"),
                Block::Assistant { md: "b".into() },
            ]
        );
        // The report confirms it; nothing more is sent.
        let fx = app.update(finished(
            Ok(summary()),
            DeliveryReport {
                answered: vec![PendingId(1)],
                ..DeliveryReport::default()
            },
        ));
        assert!(fx.is_empty());
        assert!(!app.is_running());
    }

    #[test]
    fn ack_not_running_reinjects_or_starts_turn() {
        let mut app = running_app();
        submit(&mut app, "two");
        // Still running (e.g. a continuation started): inject again.
        assert_eq!(
            app.update(ack(1, Err(InjectFailure::NotRunning))),
            vec![Effect::Inject {
                id: PendingId(1),
                text: "two".into()
            }]
        );
        app.update(finished(Ok(summary()), DeliveryReport::default()));
        assert!(!app.is_running());
        assert_eq!(app.status.queued, 1);
        // Idle: send it as a prompt, converting the placeholder in place.
        app.continuations_left = 0;
        assert_eq!(
            app.update(ack(1, Err(InjectFailure::NotRunning))),
            vec![Effect::StartTurn("two".into())]
        );
        assert!(app.is_running());
        assert_eq!(app.continuations_left, 1);
        assert_eq!(app.status.queued, 0);
        assert_eq!(app.transcript.blocks(), &[user("one"), user("two")]);
    }

    #[test]
    fn ack_failure_defers_with_one_notice_per_cause_and_sends_after_turn() {
        let mut app = running_app();
        submit(&mut app, "two");
        submit(&mut app, "three");
        submit(&mut app, "four");
        assert!(app.update(ack(1, Err(InjectFailure::Full))).is_empty());
        app.update(ack(2, Err(InjectFailure::Full)));
        app.update(ack(3, Err(InjectFailure::Unavailable)));
        let notices = app
            .transcript
            .blocks()
            .iter()
            .filter(|b| {
                matches!(
                    b,
                    Block::Notice {
                        level: NoticeLevel::Warn,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(notices, 2);
        assert_eq!(
            app.transcript.block(BlockId(1)),
            Some(&queued_block(1, "two", QueuedLabel::Deferred))
        );
        assert_eq!(app.status.queued, 3);
        let fx = app.update(finished(Ok(summary()), DeliveryReport::default()));
        assert_eq!(fx, vec![Effect::StartTurn("two\n\nthree\n\nfour".into())]);
        assert_eq!(app.status.queued, 0);
        assert_eq!(app.transcript.block(BlockId(1)), Some(&user("two")));
        assert_eq!(app.transcript.block(BlockId(3)), Some(&user("four")));
    }

    #[test]
    fn esc_recalls_everything_pending_into_a_non_empty_editor() {
        let mut app = running_app();
        submit(&mut app, "two");
        submit(&mut app, "three");
        submit(&mut app, "four");
        app.update(ack(1, Ok(())));
        app.update(ack(2, Err(InjectFailure::Full)));
        type_str(&mut app, "draft");
        // Deferred "three" comes back at once; the rest is drained by the integrator.
        assert_eq!(app.update(key(KeyCode::Esc)), vec![Effect::RecallPending]);
        assert_eq!(app.input.text(), "three\ndraft");
        let non_notice = |app: &App| -> Vec<Block> {
            app.transcript
                .blocks()
                .iter()
                .filter(|b| !matches!(b, Block::Notice { .. }))
                .cloned()
                .collect()
        };
        assert_eq!(
            non_notice(&app),
            vec![
                user("one"),
                queued_block(1, "two", QueuedLabel::Recalling),
                queued_block(3, "four", QueuedLabel::Recalling),
            ]
        );
        // Esc again with only recalling items: plain interrupt.
        assert_eq!(app.update(key(KeyCode::Esc)), vec![Effect::Interrupt]);
        // "two" had already reached the model; "four" was drained.
        assert!(app.update(Msg::Recalled(vec![PendingId(3)])).is_empty());
        assert_eq!(app.input.text(), "four\nthree\ndraft");
        assert_eq!(app.input.cursor(), (2, 5));
        assert_eq!(non_notice(&app), vec![user("one"), user("two")]);
        assert_eq!(app.status.queued, 0);
    }

    #[test]
    fn esc_with_nothing_pending_interrupts() {
        let mut app = running_app();
        assert_eq!(app.update(key(KeyCode::Esc)), vec![Effect::Interrupt]);
        let mut idle = App::new(StatusInfo::default());
        assert!(idle.update(key(KeyCode::Esc)).is_empty());
    }

    #[test]
    fn up_recalls_only_the_newest_and_reinjects_the_rest() {
        let mut app = running_app().with_history(vec!["old".into()]);
        submit(&mut app, "two");
        submit(&mut app, "three");
        app.update(ack(1, Ok(())));
        app.update(ack(2, Ok(())));
        assert_eq!(app.update(key(KeyCode::Up)), vec![Effect::RecallPending]);
        assert_eq!(
            app.pending.get(PendingId(1)).map(|p| p.state),
            Some(PendingState::Delivering { ack_seq: 0 })
        );
        // The integrator drained both: only "three" returns, "two" is re-injected.
        assert_eq!(
            app.update(Msg::Recalled(vec![PendingId(1), PendingId(2)])),
            vec![Effect::Inject {
                id: PendingId(1),
                text: "two".into()
            }]
        );
        assert_eq!(app.input.text(), "three");
        assert_eq!(
            app.transcript.blocks(),
            &[user("one"), queued_block(1, "two", QueuedLabel::Queued)]
        );
        // With a non-empty editor ↑ edits as usual.
        app.update(key(KeyCode::Up));
        assert_eq!(app.input.text(), "three");
        assert_eq!(app.status.queued, 1);
    }

    #[test]
    fn up_on_deferred_item_recalls_locally() {
        let mut app = running_app();
        submit(&mut app, "two");
        app.update(ack(1, Err(InjectFailure::Closed)));
        assert!(app.update(key(KeyCode::Up)).is_empty());
        assert_eq!(app.input.text(), "two");
        assert!(app.pending.is_empty());
        // With nothing pending, ↑ is history recall again.
        app.update(ctrl('u'));
        app.update(key(KeyCode::Up));
        assert_eq!(app.input.text(), "two");
    }

    #[test]
    fn recall_reply_for_failed_injection_returns_text() {
        let mut app = running_app();
        submit(&mut app, "two");
        assert_eq!(app.update(key(KeyCode::Esc)), vec![Effect::RecallPending]);
        app.update(ack(1, Err(InjectFailure::NotRunning)));
        assert_eq!(app.input.text(), "two");
        assert!(app.pending.is_empty());
        assert!(app.update(Msg::Recalled(vec![])).is_empty());
        assert_eq!(app.transcript.blocks(), &[user("one")]);
    }

    #[test]
    fn report_undelivered_starts_turn_and_unanswered_continues_once() {
        let mut app = running_app();
        submit(&mut app, "two");
        submit(&mut app, "three");
        app.update(ack(1, Ok(())));
        app.update(ack(2, Ok(())));
        let fx = app.update(finished(
            Ok(summary()),
            DeliveryReport {
                unanswered: vec![PendingId(1)],
                undelivered: vec![PendingId(2)],
                ..DeliveryReport::default()
            },
        ));
        // Undelivered wins over the continuation.
        assert_eq!(fx, vec![Effect::StartTurn("three".into())]);
        assert_eq!(
            app.transcript.blocks(),
            &[user("one"), user("two"), user("three")]
        );
        // Next turn: an unanswered message triggers one continuation…
        submit(&mut app, "four");
        app.update(ack(3, Ok(())));
        let report = DeliveryReport {
            unanswered: vec![PendingId(3)],
            ..DeliveryReport::default()
        };
        let fx = app.update(finished(Ok(summary()), report));
        assert_eq!(fx, vec![Effect::StartTurn(CONTINUATION_PROMPT.into())]);
        assert_eq!(app.continuations_left, 0);
        assert!(matches!(
            app.transcript.blocks().last(),
            Some(Block::Notice { level: NoticeLevel::Info, text }) if text.contains("continuing")
        ));
        assert!(
            !app.transcript
                .blocks()
                .iter()
                .any(|b| matches!(b, Block::User { text } if text == CONTINUATION_PROMPT))
        );
        // A message typed during the continuation earns another one…
        submit(&mut app, "five");
        app.update(ack(4, Ok(())));
        let report = DeliveryReport {
            unanswered: vec![PendingId(4)],
            ..DeliveryReport::default()
        };
        let fx = app.update(finished(Ok(summary()), report));
        assert_eq!(fx, vec![Effect::StartTurn(CONTINUATION_PROMPT.into())]);
        // …but the same message left unanswered again does not: the user is
        // told, and Enter on an empty editor asks for the answer.
        let report = DeliveryReport {
            unanswered: vec![PendingId(4)],
            ..DeliveryReport::default()
        };
        assert!(app.update(finished(Ok(summary()), report)).is_empty());
        assert!(!app.is_running());
        assert!(matches!(
            last_notice(&app),
            Some((NoticeLevel::Warn, text))
                if text.contains("recorded but not answered") && text.contains("“five”")
        ));
        assert_eq!(
            app.update(key(KeyCode::Enter)),
            vec![Effect::StartTurn(CONTINUATION_PROMPT.into())]
        );
        app.update(finished(Ok(summary()), DeliveryReport::default()));
        // A failed run never continues.
        submit(&mut app, "six");
        submit(&mut app, "seven");
        app.update(ack(5, Ok(())));
        let report = DeliveryReport {
            unanswered: vec![PendingId(5)],
            ..DeliveryReport::default()
        };
        assert!(app.update(finished(Err("boom".into()), report)).is_empty());
    }

    #[test]
    fn interrupted_turn_returns_undelivered_to_the_editor() {
        let mut app = running_app();
        submit(&mut app, "two");
        submit(&mut app, "three");
        app.update(ack(1, Ok(())));
        app.update(ack(2, Ok(())));
        assert_eq!(app.update(ctrl('c')), vec![Effect::Interrupt]);
        assert_eq!(app.status.queued, 2);
        type_str(&mut app, "draft");
        let fx = app.update(finished(
            Err("interrupted".into()),
            DeliveryReport {
                answered: vec![PendingId(1)],
                undelivered: vec![PendingId(2)],
                ..DeliveryReport::default()
            },
        ));
        assert!(fx.is_empty());
        assert_eq!(app.input.text(), "three\ndraft");
        assert!(app.transcript.blocks().contains(&user("two")));
        assert!(
            !app.transcript
                .blocks()
                .iter()
                .any(|b| matches!(b, Block::Queued { .. })
                    || matches!(b, Block::User { text } if text == "three"))
        );
        assert_eq!(app.status.queued, 0);
    }

    #[test]
    fn send_next_takes_priority_over_undelivered() {
        let mut app = running_app();
        let t0 = Instant::now();
        submit(&mut app, "two");
        app.update(ack(1, Ok(())));
        type_str(&mut app, "urgent");
        app.update_at(ctrl('x'), t0);
        assert_eq!(
            app.update_at(ctrl('s'), t0),
            vec![Effect::InterruptAndSend("urgent".into())]
        );
        assert_eq!(app.send_next.as_deref(), Some("urgent"));
        // The queued message stays in place until the report.
        assert_eq!(app.status.queued, 1);
        app.continuations_left = 0;
        let fx = app.update(finished(
            Err("interrupted".into()),
            DeliveryReport {
                undelivered: vec![PendingId(1)],
                ..DeliveryReport::default()
            },
        ));
        assert_eq!(fx, vec![Effect::StartTurn("two\n\nurgent".into())]);
        assert!(app.send_next.is_none());
        assert!(app.is_running());
        assert_eq!(app.continuations_left, 1);
        assert!(app.input.is_empty());
        let users: Vec<&Block> = app
            .transcript
            .blocks()
            .iter()
            .filter(|b| matches!(b, Block::User { .. }))
            .collect();
        assert_eq!(users, vec![&user("one"), &user("two"), &user("urgent")]);
    }

    #[test]
    fn clear_is_refused_while_messages_are_pending() {
        let mut app = running_app();
        submit(&mut app, "two");
        app.update(finished(Ok(summary()), DeliveryReport::default()));
        assert_eq!(app.status.queued, 1);
        assert!(submit(&mut app, "/clear").is_empty());
        assert!(matches!(
            app.transcript.blocks().last(),
            Some(Block::Notice { level: NoticeLevel::Warn, text }) if text.contains("queued")
        ));
        assert!(!app.transcript.is_empty());
    }

    #[test]
    fn reset_transcript_keeps_queued_placeholders() {
        let mut app = running_app();
        submit(&mut app, "two");
        app.update(Msg::ResetTranscript(vec![user("restored")]));
        assert_eq!(
            app.transcript.blocks(),
            &[
                user("restored"),
                queued_block(1, "two", QueuedLabel::Queued)
            ]
        );
        assert_eq!(
            app.pending.get(PendingId(1)).map(|p| p.block),
            Some(BlockId(1))
        );
    }

    #[test]
    fn ctrl_c_interrupts_running_turn_and_keeps_pending() {
        let mut app = running_app();
        submit(&mut app, "two");
        assert_eq!(app.update(ctrl('c')), vec![Effect::Interrupt]);
        assert_eq!(app.status.queued, 1);
        assert!(app.input.is_empty());
    }

    #[test]
    fn ctrl_c_clears_input_then_double_press_quits() {
        let mut app = App::new(StatusInfo::default());
        let t0 = Instant::now();
        type_str(&mut app, "draft");
        assert!(app.update_at(ctrl('c'), t0).is_empty());
        assert!(app.input.is_empty());
        assert!(app.update_at(ctrl('c'), t0).is_empty());
        assert!(app.quit_armed(t0));
        // Too late: re-arms instead of quitting.
        let late = t0 + Duration::from_secs(2);
        assert!(app.update_at(ctrl('c'), late).is_empty());
        let soon = late + Duration::from_millis(500);
        assert_eq!(app.update_at(ctrl('c'), soon), vec![Effect::Quit]);
    }

    #[test]
    fn ctrl_c_with_modal_stops_every_pending_escalation() {
        let mut app = App::new(StatusInfo::default());
        let prompt = |t: &str| EscalationPrompt {
            tool: t.into(),
            args_preview: "{}".into(),
            hint: None,
        };
        app.update(Msg::Escalation(prompt("a"), EscalationId(1)));
        app.update(Msg::Escalation(prompt("b"), EscalationId(2)));
        assert_eq!(app.pending_escalations(), 1);
        assert!(matches!(
            app.modal,
            Some(Modal::Escalation {
                queued_behind: 1,
                ..
            })
        ));
        let fx = app.update(ctrl('c'));
        assert_eq!(
            fx,
            vec![
                Effect::RespondEscalation(EscalationId(1), EscalationAnswer::Stop),
                Effect::RespondEscalation(EscalationId(2), EscalationAnswer::Stop),
            ]
        );
        assert!(app.modal.is_none());
        assert_eq!(app.pending_escalations(), 0);
    }

    #[test]
    fn escalation_keys_answer_and_advance() {
        let mut app = App::new(StatusInfo::default());
        let p = EscalationPrompt {
            tool: "shell".into(),
            args_preview: "rm x".into(),
            hint: Some("deletes".into()),
        };
        app.update(Msg::Escalation(p.clone(), EscalationId(7)));
        app.update(Msg::Escalation(p.clone(), EscalationId(8)));
        app.update(Msg::Escalation(p, EscalationId(9)));
        // Typing goes to the modal, not the input.
        assert!(app.update(key(KeyCode::Char('x'))).is_empty());
        assert!(app.input.is_empty());
        assert_eq!(
            app.update(key(KeyCode::Char('y'))),
            vec![Effect::RespondEscalation(
                EscalationId(7),
                EscalationAnswer::AllowOnce
            )]
        );
        app.update(Msg::EscalationResolved(EscalationId(8)));
        assert!(matches!(
            app.modal,
            Some(Modal::Escalation {
                id: EscalationId(9),
                ..
            })
        ));
        assert_eq!(
            app.update(key(KeyCode::Char('a'))),
            vec![Effect::RespondEscalation(
                EscalationId(9),
                EscalationAnswer::AllowAndAddRule
            )]
        );
        assert!(app.modal.is_none());
    }

    #[test]
    fn ctrl_d_quits_only_on_empty_input() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "ab");
        app.update(key(KeyCode::Left));
        assert!(app.update(ctrl('d')).is_empty());
        assert_eq!(app.input.text(), "a");
        app.update(key(KeyCode::Backspace));
        assert_eq!(app.update(ctrl('d')), vec![Effect::Quit]);
    }

    #[test]
    fn ctrl_o_toggles_reasoning() {
        let mut app = App::new(StatusInfo::default());
        app.update(agent(AgentEvent::ReasoningDelta {
            delta: "think".into(),
        }));
        app.update(ctrl('o'));
        assert!(matches!(
            app.transcript.blocks()[0],
            Block::Reasoning {
                collapsed: false,
                ..
            }
        ));
    }

    #[test]
    fn newline_variants_and_paste() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "a");
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)));
        type_str(&mut app, "b");
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)));
        app.update(ctrl('j'));
        app.update(Msg::Paste("x\ny".into()));
        assert_eq!(app.input.text(), "a\nb\n\nx\ny");
        assert_eq!(
            app.update(key(KeyCode::Enter)),
            vec![Effect::StartTurn("a\nb\n\nx\ny".into())]
        );
    }

    #[test]
    fn scroll_follow_semantics() {
        let mut s = ScrollState::default();
        assert_eq!(s.window(100, 10), 90);
        s.up(5);
        assert!(!s.follow);
        assert_eq!(s.window(100, 10), 85);
        // New output does not move the view while scrolled back.
        assert_eq!(s.window(110, 10), 85);
        s.down(100);
        assert!(s.follow);
        assert_eq!(s.window(120, 10), 110);
        s.top();
        assert_eq!(s.window(120, 10), 0);
        s.bottom();
        assert!(s.follow);
        assert_eq!(s.window(120, 10), 110);
        // Short transcripts never scroll.
        let mut short = ScrollState::default();
        short.up(3);
        assert_eq!(short.window(4, 10), 0);
        assert!(short.follow);
    }

    #[test]
    fn page_and_end_keys_drive_scroll_state() {
        let mut app = App::new(StatusInfo::default());
        app.set_viewport_height(12);
        app.update(key(KeyCode::PageUp));
        assert_eq!(app.scroll.offset_from_bottom, 10);
        assert!(!app.scroll.follow);
        app.update(Msg::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::SHIFT)));
        assert_eq!(app.scroll.offset_from_bottom, 7);
        app.update(key(KeyCode::End));
        assert!(app.scroll.follow);
        assert_eq!(app.scroll.offset_from_bottom, 0);
    }

    #[test]
    fn slash_commands_map_to_effects() {
        let mut app = App::new(StatusInfo::default());
        let run = |app: &mut App, s: &str| {
            type_str(app, s);
            app.update(key(KeyCode::Enter))
        };
        assert_eq!(
            run(&mut app, "/model gpt-5.4"),
            vec![Effect::SetModel("gpt-5.4".into())]
        );
        assert_eq!(run(&mut app, "/auto off"), vec![Effect::SetAutoMode(false)]);
        assert_eq!(run(&mut app, "/resume"), vec![Effect::RequestResumePicker]);
        assert_eq!(
            run(&mut app, "/resume last"),
            vec![Effect::ResumeSession("last".into())]
        );
        assert_eq!(run(&mut app, "/rules rm 2"), vec![Effect::RemoveRule(2)]);
        // The transcript is reset only once the integrator confirms.
        app.transcript.push_notice(NoticeLevel::Info, "old");
        assert_eq!(run(&mut app, "/clear"), vec![Effect::NewSession]);
        assert!(!app.transcript.is_empty());
        app.update(Msg::ResetTranscript(Vec::new()));
        assert!(app.transcript.is_empty());
        assert!(run(&mut app, "/bogus").is_empty());
        assert!(matches!(
            app.transcript.blocks().last(),
            Some(Block::Notice { level: NoticeLevel::Warn, text }) if text.contains("/bogus")
        ));
        assert_eq!(
            run(&mut app, "//etc"),
            vec![Effect::StartTurn("/etc".into())]
        );
        // While running, /clear and /model are refused, /quit asks again.
        assert!(run(&mut app, "/clear").is_empty());
        assert!(run(&mut app, "/model gpt-5.2").is_empty());
        assert!(matches!(
            app.transcript.blocks().last(),
            Some(Block::Notice { level: NoticeLevel::Warn, text }) if text.contains("idle")
        ));
        assert!(run(&mut app, "/quit").is_empty());
        assert_eq!(run(&mut app, "/quit"), vec![Effect::Quit]);
    }

    #[test]
    fn model_is_refused_while_messages_are_pending() {
        let mut app = running_app();
        submit(&mut app, "two");
        app.update(finished(Ok(summary()), DeliveryReport::default()));
        assert!(!app.is_running());
        assert_eq!(app.status.queued, 1);
        assert!(submit(&mut app, "/model gpt-5.2").is_empty());
        app.pending = PendingQueue::new();
        assert_eq!(
            submit(&mut app, "/model gpt-5.2"),
            vec![Effect::SetModel("gpt-5.2".into())]
        );
    }

    #[test]
    fn quit_while_busy_needs_confirmation_and_hands_over_held_messages() {
        let mut app = running_app();
        let t0 = Instant::now();
        submit(&mut app, "two");
        app.update(ack(1, Err(InjectFailure::Full)));
        // First Ctrl-D warns and arms.
        assert!(app.update_at(ctrl('d'), t0).is_empty());
        assert!(app.quit_confirm_armed(t0));
        assert!(matches!(
            last_notice(&app),
            Some((NoticeLevel::Warn, text))
                if text.contains("a turn is running") && text.contains("1 message queued")
        ));
        // The window expires on a tick; the next press only re-arms.
        app.update_at(Msg::Tick, t0 + QUIT_CONFIRM_WINDOW);
        assert!(app.quit_confirm.is_none());
        assert!(
            app.update_at(ctrl('d'), t0 + QUIT_CONFIRM_WINDOW)
                .is_empty()
        );
        assert!(app.quit_confirm_armed(t0 + QUIT_CONFIRM_WINDOW));
        // Ctrl-D then /quit (typing does not disarm) within the window quits,
        // handing over the text.
        let t1 = t0 + Duration::from_secs(10);
        assert!(app.update_at(ctrl('d'), t1).is_empty());
        for c in "/quit".chars() {
            app.update_at(key(KeyCode::Char(c)), t1);
        }
        assert_eq!(
            app.update_at(key(KeyCode::Enter), t1 + Duration::from_secs(1)),
            vec![
                Effect::SaveUnsent(vec![(Some(PendingId(1)), "two".into())]),
                Effect::Quit
            ]
        );
    }

    #[test]
    fn idle_quit_is_immediate() {
        let mut app = App::new(StatusInfo::default());
        assert_eq!(app.update(ctrl('d')), vec![Effect::Quit]);
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "/quit");
        assert_eq!(app.update(key(KeyCode::Enter)), vec![Effect::Quit]);
    }

    #[test]
    fn last_seq_resets_with_the_session() {
        let mut app = running_app();
        app.update(agent_at(AgentEvent::TurnStarted { turn_number: 0 }, 41));
        assert_eq!(app.last_seq, 41);
        app.update(Msg::Status(StatusUpdate::Session("next".into())));
        assert_eq!(app.last_seq, 0);
        app.update(agent_at(AgentEvent::TurnStarted { turn_number: 0 }, 7));
        app.update(Msg::ResetTranscript(Vec::new()));
        assert_eq!(app.last_seq, 0);
    }

    #[test]
    fn summary_text_is_not_repeated_after_streamed_text() {
        let mut app = running_app();
        app.update(agent(AgentEvent::TextDelta {
            delta: "streamed".into(),
        }));
        // A notice after the text must not make the summary look new.
        app.update(Msg::Notice {
            level: NoticeLevel::Info,
            text: "x".into(),
        });
        app.update(Msg::TurnFinished(TurnOutcome::plain(Ok(TurnSummary {
            text: "streamed".into(),
            ..summary()
        }))));
        let assistant = app
            .transcript
            .blocks()
            .iter()
            .filter(|b| matches!(b, Block::Assistant { .. }))
            .count();
        assert_eq!(assistant, 1);
    }

    #[test]
    fn resume_picker_selects() {
        let mut app = App::new(StatusInfo::default());
        app.update(Msg::ShowResumePicker(vec![
            ("aaa".into(), "first".into()),
            ("bbb".into(), "second".into()),
        ]));
        app.update(key(KeyCode::Down));
        assert_eq!(
            app.update(key(KeyCode::Enter)),
            vec![Effect::ResumeSession("bbb".into())]
        );
        assert!(app.modal.is_none());
    }

    #[test]
    fn status_updates_and_dirty_tracking() {
        let mut app = App::new(StatusInfo::default());
        assert!(app.take_dirty());
        app.update(Msg::Tick);
        assert!(!app.take_dirty());
        app.update(Msg::Status(StatusUpdate::Auto {
            state: "PAUSED".into(),
            denials: (3, 3, 5, 20),
        }));
        assert!(app.take_dirty());
        assert_eq!(app.status.auto_state, "PAUSED");
        app.update(Msg::Status(StatusUpdate::Session(
            "0123456789abcdef".into(),
        )));
        assert_eq!(app.status.session_short, "01234567");
    }

    #[test]
    fn history_recall_with_up() {
        let mut app = App::new(StatusInfo::default()).with_history(vec!["old".into()]);
        app.update(key(KeyCode::Up));
        assert_eq!(app.input.text(), "old");
    }

    #[test]
    fn take_submitted_drains_prompts_and_commands() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "hello");
        app.update(key(KeyCode::Enter));
        app.update(key(KeyCode::Enter)); // blank: ignored
        type_str(&mut app, "/help");
        app.update(key(KeyCode::Enter));
        assert_eq!(
            app.take_submitted(),
            vec!["hello".to_string(), "/help".to_string()]
        );
        assert!(app.take_submitted().is_empty());
    }

    #[test]
    fn ctrl_r_enters_search_and_enter_accepts() {
        let mut app = App::new(StatusInfo::default())
            .with_history(vec!["cargo build".into(), "git status".into()])
            .with_global_history(vec!["cargo test --global".into(), "git status".into()]);
        type_str(&mut app, "draft");
        app.update(ctrl('r'));
        assert!(matches!(app.mode, InputMode::Search(_)));
        let (_, corpus) = app.search().unwrap();
        assert_eq!(corpus, ["git status", "cargo build", "cargo test --global"]);
        type_str(&mut app, "cargo");
        let (s, corpus) = app.search().unwrap();
        assert_eq!(s.current(corpus), Some("cargo build"));
        app.update(ctrl('r'));
        let (s, corpus) = app.search().unwrap();
        assert_eq!(s.current(corpus), Some("cargo test --global"));
        // Enter accepts into the editor without submitting.
        assert!(app.update(key(KeyCode::Enter)).is_empty());
        assert_eq!(app.mode, InputMode::Normal);
        assert_eq!(app.input.text(), "cargo test --global");
        assert_eq!(app.input.cursor(), (0, 19));
        assert!(app.take_submitted().is_empty());
        // Searching while a turn runs is fine.
        app.update(key(KeyCode::Enter));
        assert!(app.is_running());
        app.update(ctrl('r'));
        type_str(&mut app, "git");
        app.update(key(KeyCode::End));
        assert_eq!(app.input.text(), "git status");
    }

    #[test]
    fn esc_cancels_search_restores_input() {
        let mut app = App::new(StatusInfo::default()).with_history(vec!["hello".into()]);
        type_str(&mut app, "draft");
        app.update(ctrl('r'));
        type_str(&mut app, "hel");
        assert_eq!(app.input.text(), "draft");
        assert!(app.update(key(KeyCode::Esc)).is_empty());
        assert_eq!(app.mode, InputMode::Normal);
        assert_eq!(app.input.text(), "draft");
        // Ctrl-C cancels too and does not arm quit / clear the input.
        app.update(ctrl('r'));
        assert!(app.update(ctrl('c')).is_empty());
        assert_eq!(app.input.text(), "draft");
        assert!(!app.quit_armed(Instant::now()));
    }

    #[test]
    fn chord_ctrl_x_ctrl_s_interrupt_and_send_when_running() {
        let mut app = App::new(StatusInfo::default());
        let t0 = Instant::now();
        // Idle: behaves like Enter.
        type_str(&mut app, "first");
        assert!(app.update_at(ctrl('x'), t0).is_empty());
        assert!(app.chord_pending(t0));
        assert_eq!(
            app.update_at(ctrl('s'), t0),
            vec![Effect::StartTurn("first".into())]
        );
        assert!(app.chord.is_none());
        // Running with input: interrupt and send.
        type_str(&mut app, "next");
        app.update_at(ctrl('x'), t0);
        assert_eq!(
            app.update_at(ctrl('s'), t0),
            vec![Effect::InterruptAndSend("next".into())]
        );
        assert!(app.input.is_empty());
        assert_eq!(
            app.take_submitted(),
            vec!["first".to_string(), "next".into()]
        );
        assert_eq!(app.input.history().last().map(String::as_str), Some("next"));
        // Running with an empty editor: plain interrupt.
        app.update_at(ctrl('x'), t0);
        assert_eq!(app.update_at(ctrl('s'), t0), vec![Effect::Interrupt]);
        // Esc cancels the chord without interrupting.
        app.update_at(ctrl('x'), t0);
        assert!(app.update_at(key(KeyCode::Esc), t0).is_empty());
        assert!(app.chord.is_none());
        // Any other key cancels the chord and is processed normally.
        app.update_at(ctrl('x'), t0);
        app.update_at(key(KeyCode::Char('a')), t0);
        assert!(app.chord.is_none());
        assert_eq!(app.input.text(), "a");
        assert!(app.update_at(ctrl('s'), t0).is_empty());
    }

    #[test]
    fn chord_expires_after_window() {
        let mut app = App::new(StatusInfo::default());
        let t0 = Instant::now();
        type_str(&mut app, "x");
        app.update_at(ctrl('x'), t0);
        let late = t0 + CHORD_WINDOW + Duration::from_millis(1);
        assert!(!app.chord_pending(late));
        assert!(app.update_at(ctrl('s'), late).is_empty());
        assert_eq!(app.input.text(), "x");
        assert!(!app.is_running());
        // A tick after the window clears the pending chord.
        app.update_at(ctrl('x'), t0);
        app.update_at(Msg::Tick, late);
        assert!(app.chord.is_none());
    }

    #[test]
    fn ctrl_s_alone_is_noop() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "text");
        assert!(app.update(ctrl('s')).is_empty());
        assert_eq!(app.input.text(), "text");
        assert!(!app.is_running());
    }

    fn last_notice(app: &App) -> Option<(NoticeLevel, String)> {
        app.transcript.blocks().iter().rev().find_map(|b| match b {
            Block::Notice { level, text } => Some((*level, text.clone())),
            _ => None,
        })
    }

    #[test]
    fn hash_input_remembers_and_escapes_do_not() {
        let mut app = App::new(StatusInfo::default());
        assert_eq!(
            submit(&mut app, "#  prefer tabs "),
            vec![Effect::Remember("prefer tabs".into())]
        );
        assert!(!app.is_running());
        assert_eq!(
            last_notice(&app),
            Some((NoticeLevel::Info, "remembering: prefer tabs".into()))
        );
        assert_eq!(
            submit(&mut app, "/remember use nextest"),
            vec![Effect::Remember("use nextest".into())]
        );
        assert_eq!(
            submit(&mut app, "## heading"),
            vec![Effect::StartTurn("## heading".into())]
        );
        app.update(finished(Ok(summary()), DeliveryReport::default()));
        assert_eq!(
            submit(&mut app, "\\#tag"),
            vec![Effect::StartTurn("#tag".into())]
        );
        app.update(finished(Ok(summary()), DeliveryReport::default()));
        type_str(&mut app, "#a");
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)));
        assert_eq!(
            submit(&mut app, "b"),
            vec![Effect::StartTurn("#a\nb".into())]
        );
    }

    #[test]
    fn remember_is_allowed_while_running() {
        let mut app = running_app();
        assert_eq!(
            submit(&mut app, "#note"),
            vec![Effect::Remember("note".into())]
        );
        assert!(app.pending.is_empty());
        assert!(app.is_running());
    }

    #[test]
    fn compact_refused_while_running_or_pending() {
        let mut app = running_app();
        assert!(submit(&mut app, "/compact").is_empty());
        assert_eq!(
            last_notice(&app),
            Some((
                NoticeLevel::Warn,
                "finish or interrupt the current turn first".into()
            ))
        );
        submit(&mut app, "two");
        app.update(finished(Ok(summary()), DeliveryReport::default()));
        assert!(!app.is_running());
        assert_eq!(app.status.queued, 1);
        assert!(submit(&mut app, "/compact").is_empty());
        assert!(!app.compacting);
        assert_eq!(last_notice(&app).map(|(l, _)| l), Some(NoticeLevel::Warn));
    }

    #[test]
    fn prompts_are_deferred_while_compacting_and_sent_after() {
        let mut app = App::new(StatusInfo::default());
        assert_eq!(submit(&mut app, "/compact"), vec![Effect::Compact]);
        assert!(app.compacting);
        assert_eq!(
            last_notice(&app),
            Some((NoticeLevel::Info, "compacting…".into()))
        );
        assert!(submit(&mut app, "one").is_empty());
        assert!(submit(&mut app, "two").is_empty());
        assert_eq!(app.status.queued, 2);
        assert_eq!(
            app.transcript.blocks().last(),
            Some(&queued_block(2, "two", QueuedLabel::Deferred))
        );
        // Lifecycle commands are refused; /remember is not.
        for cmd in ["/clear", "/resume last", "/compact"] {
            assert!(submit(&mut app, cmd).is_empty(), "{cmd}");
            assert_eq!(
                last_notice(&app).map(|(l, _)| l),
                Some(NoticeLevel::Warn),
                "{cmd}"
            );
        }
        assert_eq!(submit(&mut app, "#x"), vec![Effect::Remember("x".into())]);
        let fx = app.update(Msg::CompactFinished(Ok(())));
        assert_eq!(fx, vec![Effect::StartTurn("one\n\ntwo".into())]);
        assert!(!app.compacting);
        assert!(app.is_running());
        assert_eq!(app.status.queued, 0);
        assert!(app.transcript.blocks().contains(&user("one")));
        assert!(app.transcript.blocks().contains(&user("two")));
        assert!(app.transcript.blocks().iter().any(
            |b| matches!(b, Block::Notice { level: NoticeLevel::Info, text } if text == "compacted")
        ));
    }

    #[test]
    fn compact_success_without_deferred_sends_nothing() {
        let mut app = App::new(StatusInfo::default());
        submit(&mut app, "/compact");
        assert!(app.update(Msg::CompactFinished(Ok(()))).is_empty());
        assert!(!app.is_running());
        assert_eq!(
            last_notice(&app),
            Some((NoticeLevel::Info, "compacted".into()))
        );
    }

    #[test]
    fn failed_compaction_returns_deferred_to_the_editor() {
        let mut app = App::new(StatusInfo::default());
        submit(&mut app, "/compact");
        submit(&mut app, "one");
        submit(&mut app, "two");
        type_str(&mut app, "draft");
        let fx = app.update(Msg::CompactFinished(Err("boom".into())));
        assert!(fx.is_empty());
        assert!(!app.compacting);
        assert!(!app.is_running());
        assert!(app.pending.is_empty());
        assert_eq!(app.status.queued, 0);
        assert_eq!(app.input.text(), "one\ntwo\ndraft");
        assert_eq!(
            last_notice(&app),
            Some((NoticeLevel::Error, "compaction failed: boom".into()))
        );
        assert!(
            !app.transcript
                .blocks()
                .iter()
                .any(|b| matches!(b, Block::Queued { .. }))
        );
    }

    #[test]
    fn context_gauge_tracks_usage_compaction_and_clear() {
        let mut app = App::new(StatusInfo::default());
        app.update(Msg::Status(StatusUpdate::ContextWindow(Some(200_000))));
        app.update(agent(AgentEvent::TurnCompleted {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 1_000,
                output_tokens: 200,
                cache_creation_tokens: Some(30),
                cache_read_tokens: Some(80_000),
            },
        }));
        assert_eq!(app.status.context_used, Some(81_230));
        assert!(!app.status.context_approx);
        app.update(agent(AgentEvent::CompactionCompleted {
            summary_tokens: 2_500,
            messages_before: 40,
            messages_after: 3,
        }));
        assert_eq!(app.status.context_used, Some(2_500));
        assert!(app.status.context_approx);
        app.update(agent(AgentEvent::TurnCompleted {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 3_000,
                output_tokens: 100,
                ..Usage::default()
            },
        }));
        assert_eq!(app.status.context_used, Some(3_100));
        assert!(!app.status.context_approx);
        assert_eq!(submit(&mut app, "/clear"), vec![Effect::NewSession]);
        // What the integrator sends once the new session is live.
        app.update(Msg::Status(StatusUpdate::Context {
            used: None,
            approx: false,
        }));
        app.update(Msg::Status(StatusUpdate::ContextWindow(Some(200_000))));
        assert_eq!(app.status.context_used, None);
        assert_eq!(app.status.context_window, Some(200_000));
        app.update(Msg::Status(StatusUpdate::ApproxContextWindow(128_000)));
        assert!(app.status.window_approx);
        assert_eq!(app.status.context_window, Some(128_000));
        app.update(Msg::Status(StatusUpdate::ContextWindow(Some(200_000))));
        assert!(!app.status.window_approx);
    }
}
