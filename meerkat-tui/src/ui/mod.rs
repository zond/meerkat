//! ratatui application: Elm-style state + pure `update`, a renderer and a terminal runtime.
//!
//! [`App::update`] is pure — it never touches the terminal or performs I/O — and
//! returns [`Effect`]s that the integrator (session/approval wiring) executes.
//! Everything the integrator wants to show is pushed back in as a [`Msg`].

pub mod commands;
pub mod history;
pub mod input;
pub mod keymap;
pub mod markdown;
pub mod modal;
pub mod render;
pub mod runtime;
pub mod transcript;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use meerkat_core::{AgentEvent, EventEnvelope, ScopedAgentEvent};
use ratatui::crossterm::event::{KeyEvent, MouseEvent, MouseEventKind};

pub use commands::{AutoCommand, Command, CommandError, RuleEffect, RulesCommand};
pub use history::History;
pub use input::InputEditor;
pub use keymap::{Action, EditAction};
pub use modal::{EscalationAnswer, EscalationId, EscalationPrompt, Modal, ResumePicker};
pub use runtime::{TerminalGuard, UiChannels, UiError, install_panic_hook, run_ui};
pub use transcript::{Block, BlockId, DiffView, NoticeLevel, ToolState, Transcript};

/// Window in which a second Ctrl-C on an idle, empty prompt quits.
pub const DOUBLE_CTRL_C_WINDOW: Duration = Duration::from_millis(1500);

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
    TurnFinished(Result<TurnSummary, String>),
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
    /// Leave the application.
    Quit,
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
    /// Number of queued prompts.
    pub queued: usize,
    /// Whether the classifier is running.
    pub classifying: bool,
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
        }
    }
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
    /// Prompts submitted while a turn was running.
    pub queued: VecDeque<String>,
    pending_escalations: VecDeque<(EscalationId, EscalationPrompt)>,
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
            queued: VecDeque::new(),
            pending_escalations: VecDeque::new(),
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

    /// Escalations waiting behind the open modal.
    pub fn pending_escalations(&self) -> usize {
        self.pending_escalations.len()
    }

    /// Whether a second Ctrl-C at `now` would quit (used for the input hint).
    pub fn quit_armed(&self, now: Instant) -> bool {
        self.last_ctrl_c
            .is_some_and(|t| now.saturating_duration_since(t) < DOUBLE_CTRL_C_WINDOW)
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
                    self.input.insert_str(&text);
                }
            }
            Msg::Resize(..) => self.transcript.invalidate_all(),
            Msg::Mouse(m) => match m.kind {
                MouseEventKind::ScrollUp => self.scroll.up(keymap::LINE_SCROLL),
                MouseEventKind::ScrollDown => self.scroll.down(keymap::LINE_SCROLL),
                _ => dirty = false,
            },
            Msg::Agent(env) => {
                if let Some(usage) = self.transcript.apply_agent_event_at(&env.payload, now) {
                    self.status
                        .add_tokens(usage.input_tokens, usage.output_tokens);
                }
            }
            Msg::Scoped(ev) => self.transcript.apply_scoped_event_at(&ev, now),
            Msg::TurnFinished(result) => self.turn_finished(result, now, &mut fx),
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
            Msg::Status(update) => self.status.apply(update),
            Msg::ShowResumePicker(items) => {
                if self.modal.is_none() {
                    self.modal = Some(Modal::Resume(ResumePicker { items, selected: 0 }));
                }
            }
            Msg::ResetTranscript(blocks) => {
                self.transcript.reset(blocks);
                self.scroll.bottom();
            }
            Msg::Tick => {
                dirty = self.status.running_since.is_some()
                    || self.status.classifying
                    || self.last_ctrl_c.is_some();
                if self.last_ctrl_c.is_some() && !self.quit_armed(now) {
                    self.last_ctrl_c = None;
                }
            }
        }
        self.status.queued = self.queued.len();
        self.dirty |= dirty;
        fx
    }

    fn main_key(&mut self, key: &KeyEvent, now: Instant, fx: &mut Vec<Effect>) {
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
                    fx.push(Effect::Quit);
                } else {
                    self.input.delete();
                }
            }
            Action::Escape => {
                if self.running {
                    self.interrupt(fx);
                }
            }
            Action::ToggleReasoning => {
                self.transcript.toggle_last_reasoning();
            }
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
            return;
        }
        let raw = self.input.take();
        self.submitted.push(raw.clone());
        match commands::parse(&raw) {
            Some(Ok(cmd)) => self.run_command(cmd, fx),
            Some(Err(err)) => {
                self.transcript
                    .push_notice(NoticeLevel::Warn, err.to_string());
            }
            None => {
                let prompt = match raw.trim_start().strip_prefix("//") {
                    Some(rest) => format!("/{rest}"),
                    None => raw,
                };
                if self.running {
                    self.queued.push_back(prompt);
                } else {
                    self.start_turn(prompt, now, fx);
                }
            }
        }
    }

    fn start_turn(&mut self, prompt: String, now: Instant, fx: &mut Vec<Effect>) {
        self.run_first_block = self.transcript.len();
        self.transcript.push(Block::User {
            text: prompt.clone(),
        });
        self.running = true;
        self.status.running_since = Some(now);
        self.scroll.bottom();
        fx.push(Effect::StartTurn(prompt));
    }

    fn turn_finished(
        &mut self,
        result: Result<TurnSummary, String>,
        now: Instant,
        fx: &mut Vec<Effect>,
    ) {
        self.running = false;
        self.status.running_since = None;
        self.status.classifying = false;
        match result {
            Ok(summary) => {
                let has_text = self
                    .transcript
                    .blocks()
                    .iter()
                    .skip(self.run_first_block)
                    .any(|b| matches!(b, Block::Assistant { .. }));
                if !has_text && !summary.text.is_empty() {
                    self.transcript.push(Block::Assistant { md: summary.text });
                }
            }
            Err(err) => {
                self.transcript
                    .push_notice(NoticeLevel::Error, format!("turn failed: {err}"));
            }
        }
        if let Some(next) = self.queued.pop_front() {
            self.start_turn(next, now, fx);
        }
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

    fn interrupt(&mut self, fx: &mut Vec<Effect>) {
        fx.push(Effect::Interrupt);
        if self.queued.is_empty() {
            return;
        }
        let restored: Vec<String> = self.queued.drain(..).collect();
        let mut text = self.input.text();
        for prompt in restored {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&prompt);
        }
        self.input.set_text(&text);
        self.transcript.push_notice(
            NoticeLevel::Info,
            "interrupted; queued prompts moved back to the input",
        );
    }

    fn run_command(&mut self, cmd: Command, fx: &mut Vec<Effect>) {
        let busy = |app: &mut Self| {
            app.transcript.push_notice(
                NoticeLevel::Warn,
                "a turn is running; interrupt it first (Ctrl-C)",
            );
        };
        match cmd {
            Command::Quit => fx.push(Effect::Quit),
            Command::Help => {
                self.transcript
                    .push_notice(NoticeLevel::Info, commands::HELP_TEXT);
            }
            Command::Clear => {
                if self.running {
                    busy(self);
                } else {
                    self.transcript.clear();
                    self.scroll.bottom();
                    self.status.set_tokens(0, 0);
                    fx.push(Effect::NewSession);
                }
            }
            Command::Resume(reference) => {
                if self.running {
                    busy(self);
                } else {
                    fx.push(reference.map_or(Effect::RequestResumePicker, Effect::ResumeSession));
                }
            }
            Command::Model(Some(model)) => fx.push(Effect::SetModel(model)),
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
        app.update(Msg::TurnFinished(Ok(summary())));
        assert!(!app.is_running());
        assert!(app.status.running_since.is_none());
    }

    #[test]
    fn turn_summary_text_used_when_nothing_streamed() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "q");
        app.update(key(KeyCode::Enter));
        app.update(Msg::TurnFinished(Ok(TurnSummary {
            text: "answer".into(),
            ..summary()
        })));
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

    #[test]
    fn prompts_queue_while_running_and_drain_on_finish() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "one");
        assert_eq!(app.update(key(KeyCode::Enter)).len(), 1);
        type_str(&mut app, "two");
        assert!(app.update(key(KeyCode::Enter)).is_empty());
        type_str(&mut app, "three");
        assert!(app.update(key(KeyCode::Enter)).is_empty());
        assert_eq!(app.status.queued, 2);
        let fx = app.update(Msg::TurnFinished(Ok(summary())));
        assert_eq!(fx, vec![Effect::StartTurn("two".into())]);
        assert!(app.is_running());
        assert_eq!(app.status.queued, 1);
        let fx = app.update(Msg::TurnFinished(Err("boom".into())));
        assert_eq!(fx, vec![Effect::StartTurn("three".into())]);
        assert!(app.transcript.blocks().iter().any(
            |b| matches!(b, Block::Notice { level: NoticeLevel::Error, text } if text.contains("boom"))
        ));
        assert!(app.update(Msg::TurnFinished(Ok(summary()))).is_empty());
        assert!(!app.is_running());
    }

    #[test]
    fn ctrl_c_interrupts_running_turn_and_restores_queue() {
        let mut app = App::new(StatusInfo::default());
        type_str(&mut app, "one");
        app.update(key(KeyCode::Enter));
        type_str(&mut app, "two");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.update(ctrl('c')), vec![Effect::Interrupt]);
        assert!(app.queued.is_empty());
        assert_eq!(app.input.text(), "two");
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
    fn ctrl_r_toggles_reasoning() {
        let mut app = App::new(StatusInfo::default());
        app.update(agent(AgentEvent::ReasoningDelta {
            delta: "think".into(),
        }));
        app.update(ctrl('r'));
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
        assert_eq!(run(&mut app, "/clear"), vec![Effect::NewSession]);
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
        // While running, /clear is refused.
        assert!(run(&mut app, "/clear").is_empty());
        assert_eq!(run(&mut app, "/quit"), vec![Effect::Quit]);
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
}
