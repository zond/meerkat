//! Rich terminal input editor with box-drawn UI.
//!
//! Provides a Claude Code-style input box with full keyboard support:
//! - Arrow keys for cursor movement
//! - Ctrl+Left/Right for word navigation
//! - Up/Down for command history
//! - Home/End for line start/end
//! - Ctrl+A/E for line start/end (Emacs bindings)
//! - Ctrl+W to delete previous word
//! - Ctrl+U to clear line before cursor
//! - Ctrl+K to clear line after cursor
//! - Tab completion for slash commands
//! - Enter to submit, Ctrl+C to signal interrupt, Ctrl+D on empty to signal EOF

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    style,
    terminal::{self, ClearType},
    ExecutableCommand, QueueableCommand,
};
use std::io::{BufRead, Write};
use std::sync::mpsc as std_mpsc;
use unicode_width::UnicodeWidthStr;

/// Result of processing a key event.
pub enum InputAction {
    /// User submitted a line (pressed Enter).
    Submit(String),
    /// User pressed Ctrl+C.
    Interrupt,
    /// User pressed Ctrl+D on empty input (EOF).
    Eof,
    /// Key was handled, redraw needed.
    Redraw,
    /// Key was handled, no redraw needed.
    Noop,
}

/// Result sent from the input thread.
pub enum InputResult {
    Line(String),
    Interrupt,
    Eof,
}

/// Signal sent to the input thread to coordinate with output.
pub enum OutputSignal {
    /// Hide the input box. The `SyncSender` is used to acknowledge completion,
    /// so the caller can safely write to stderr without overlap.
    HideBox(std_mpsc::SyncSender<()>),
    /// Re-show the input box (fire-and-forget).
    ShowBox,
    /// Update the label in the input box.
    UpdateLabel(String),
}

/// Print a line to stderr with `\r\n` line endings, which is required in raw mode.
/// Regular `eprintln!` only outputs `\n` (LF), which doesn't return the cursor
/// to column 0 in raw terminal mode.
#[macro_export]
macro_rules! raw_eprintln {
    () => {
        eprint!("\r\n")
    };
    ($($arg:tt)*) => {
        eprint!("{}\r\n", format_args!($($arg)*))
    };
}

/// Hide the input box and block until the input thread confirms it is cleared.
pub fn hide_box(output_tx: &std_mpsc::Sender<OutputSignal>) {
    let (ack_tx, ack_rx) = std_mpsc::sync_channel(0);
    let _ = output_tx.send(OutputSignal::HideBox(ack_tx));
    // Block until ack. If the input thread is gone, recv() returns Err — fine.
    let _ = ack_rx.recv();
}

/// Hide the input box, run a closure that writes to stderr, then re-show it.
pub fn with_output(output_tx: &std_mpsc::Sender<OutputSignal>, f: impl FnOnce()) {
    hide_box(output_tx);
    f();
    let _ = std::io::stderr().flush();
    let _ = output_tx.send(OutputSignal::ShowBox);
}

/// Slash commands available for tab completion.
const SLASH_COMMANDS: &[&str] = &["/quit", "/status", "/members", "/tasks", "/send"];

/// A line editor with history and box-drawn UI.
pub struct InputEditor {
    buf: String,
    /// Byte offset of cursor within `buf`.
    cursor: usize,
    /// Command history (most recent last).
    history: Vec<String>,
    /// Index into history when browsing (None = editing current input).
    history_idx: Option<usize>,
    /// Saved current input when browsing history.
    saved_input: String,
    /// Label shown in the box header.
    label: String,
    /// Terminal width (updated on each render).
    term_width: u16,
    /// Number of lines the box occupied on last render (for clearing).
    last_render_height: u16,
}

impl InputEditor {
    pub fn new(label: impl Into<String>) -> Self {
        let (w, _) = terminal::size().unwrap_or((80, 24));
        Self {
            buf: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            saved_input: String::new(),
            label: label.into(),
            term_width: w,
            last_render_height: 0,
        }
    }

    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
    }

    /// Process a single key event and return the action to take.
    pub fn handle_key(&mut self, key: KeyEvent) -> InputAction {
        match (key.code, key.modifiers) {
            (KeyCode::Enter, _) => {
                let line = std::mem::take(&mut self.buf);
                if !line.trim().is_empty() {
                    self.history.push(line.clone());
                }
                self.cursor = 0;
                self.history_idx = None;
                self.saved_input.clear();
                InputAction::Submit(line)
            }

            (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => {
                InputAction::Interrupt
            }

            (KeyCode::Char('d'), m)
                if m.contains(KeyModifiers::CONTROL) && self.buf.is_empty() =>
            {
                InputAction::Eof
            }

            // Word-level cursor movement
            (KeyCode::Left, m) if m.contains(KeyModifiers::CONTROL) => {
                self.move_word_left();
                InputAction::Redraw
            }
            (KeyCode::Right, m) if m.contains(KeyModifiers::CONTROL) => {
                self.move_word_right();
                InputAction::Redraw
            }

            // Character-level cursor movement
            (KeyCode::Left, _) => {
                self.move_left();
                InputAction::Redraw
            }
            (KeyCode::Right, _) => {
                self.move_right();
                InputAction::Redraw
            }

            // Line start
            (KeyCode::Home, _) => {
                self.cursor = 0;
                InputAction::Redraw
            }
            (KeyCode::Char('a'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.cursor = 0;
                InputAction::Redraw
            }

            // Line end
            (KeyCode::End, _) => {
                self.cursor = self.buf.len();
                InputAction::Redraw
            }
            (KeyCode::Char('e'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.cursor = self.buf.len();
                InputAction::Redraw
            }

            // History
            (KeyCode::Up, _) => {
                self.history_up();
                InputAction::Redraw
            }
            (KeyCode::Down, _) => {
                self.history_down();
                InputAction::Redraw
            }

            // Deletion
            (KeyCode::Backspace, m) if m.contains(KeyModifiers::CONTROL) => {
                self.delete_word_back();
                InputAction::Redraw
            }
            (KeyCode::Backspace, _) => {
                if self.cursor > 0 {
                    let prev = self.prev_char_boundary();
                    self.buf.drain(prev..self.cursor);
                    self.cursor = prev;
                }
                InputAction::Redraw
            }
            (KeyCode::Delete, _) => {
                if self.cursor < self.buf.len() {
                    let next = self.next_char_boundary();
                    self.buf.drain(self.cursor..next);
                }
                InputAction::Redraw
            }
            (KeyCode::Char('w'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.delete_word_back();
                InputAction::Redraw
            }
            (KeyCode::Char('u'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.buf.drain(..self.cursor);
                self.cursor = 0;
                InputAction::Redraw
            }
            (KeyCode::Char('k'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.buf.truncate(self.cursor);
                InputAction::Redraw
            }

            // Tab completion
            (KeyCode::Tab, _) => {
                self.try_complete();
                InputAction::Redraw
            }

            // Regular character input
            (KeyCode::Char(c), m) => {
                if m.contains(KeyModifiers::CONTROL) {
                    return InputAction::Noop;
                }
                self.buf.insert(self.cursor, c);
                self.cursor += c.len_utf8();
                InputAction::Redraw
            }

            _ => InputAction::Noop,
        }
    }

    /// Compute the cursor's logical row inside the box content area.
    fn cursor_content_row(&self, inner_width: usize) -> u16 {
        let text_before = &self.buf[..self.cursor];
        let before_lines = wrap_text(
            if text_before.is_empty() { " " } else { text_before },
            inner_width.max(1),
        );
        before_lines.len().saturating_sub(1) as u16
    }

    /// Render the input box to stderr.
    pub fn render(&mut self, out: &mut impl Write) -> std::io::Result<()> {
        self.term_width = terminal::size().unwrap_or((80, 24)).0;

        let box_width = self.term_width as usize;
        if box_width < 10 {
            write!(out, "\r{} {}", self.label, self.buf)?;
            out.flush()?;
            return Ok(());
        }

        // Clear previous render.
        if self.last_render_height > 0 {
            for _ in 0..self.last_render_height {
                out.queue(cursor::MoveUp(1))?;
                out.queue(terminal::Clear(ClearType::CurrentLine))?;
            }
        }

        let inner_width = box_width - 4; // 2 border + 2 padding

        // Top border: ╭─ label ──...──╮
        let label_display = if self.label.is_empty() {
            String::new()
        } else {
            format!(" {} ", self.label)
        };
        let label_w = UnicodeWidthStr::width(label_display.as_str());
        let top_rule_len = box_width.saturating_sub(2 + label_w);
        out.queue(style::SetForegroundColor(style::Color::DarkGrey))?;
        write!(out, "\r\u{256d}{}{}\u{256e}\r\n", label_display, "\u{2500}".repeat(top_rule_len))?;

        // Content lines (wrap long input).
        let buf_ref = if self.buf.is_empty() { " " } else { &self.buf };
        let lines = wrap_text(buf_ref, inner_width);
        let line_count = lines.len();

        // Cursor position within the box.
        let cursor_display_row = self.cursor_content_row(inner_width) as usize;
        let cursor_display_col = {
            let text_before = &self.buf[..self.cursor];
            let before_lines = wrap_text(
                if text_before.is_empty() { " " } else { text_before },
                inner_width,
            );
            let last = before_lines.last().map_or("", |s| s.as_str());
            if self.buf.is_empty() { 0 } else { UnicodeWidthStr::width(last) }
        };

        for (i, line) in lines.iter().enumerate() {
            let line_w = UnicodeWidthStr::width(line.as_str());
            let pad = inner_width.saturating_sub(line_w);
            write!(out, "\u{2502} ")?;
            out.queue(style::ResetColor)?;

            if i == 0 && self.buf.is_empty() {
                out.queue(style::SetForegroundColor(style::Color::DarkGrey))?;
                let placeholder = "Type a message...";
                let ph_w = UnicodeWidthStr::width(placeholder);
                write!(out, "{placeholder}{}", " ".repeat(inner_width.saturating_sub(ph_w)))?;
                out.queue(style::ResetColor)?;
            } else {
                write!(out, "{}{}", line, " ".repeat(pad))?;
            }

            out.queue(style::SetForegroundColor(style::Color::DarkGrey))?;
            write!(out, " \u{2502}\r\n")?;
        }

        // Bottom border: ╰──...── hints ──╯
        let hint = "enter send | ctrl+c cancel";
        let hint_w = UnicodeWidthStr::width(hint);
        let bottom_rule_len = box_width.saturating_sub(2);
        let hint_start = bottom_rule_len.saturating_sub(hint_w + 1);
        write!(out, "\u{2570}")?;
        write!(out, "{}", "\u{2500}".repeat(hint_start))?;
        out.queue(style::SetForegroundColor(style::Color::DarkGrey))?;
        write!(out, "{hint}")?;
        write!(out, "{}", "\u{2500}".repeat(bottom_rule_len.saturating_sub(hint_start + hint_w)))?;
        write!(out, "\u{256f}")?;
        out.queue(style::ResetColor)?;

        self.last_render_height = (2 + line_count) as u16;

        // Position cursor inside the box.
        let lines_up = (line_count - cursor_display_row) as u16;
        out.queue(cursor::MoveUp(lines_up))?;
        out.queue(cursor::MoveToColumn((2 + cursor_display_col) as u16))?;
        out.flush()
    }

    /// Erase the input box from the terminal.
    pub fn clear_box(&mut self, out: &mut impl Write) -> std::io::Result<()> {
        if self.last_render_height == 0 {
            return Ok(());
        }

        let inner_width = (self.term_width as usize).saturating_sub(4);
        let cursor_row = self.cursor_content_row(inner_width);

        // Move from cursor position to top border, then clear each line downward.
        out.queue(cursor::MoveToColumn(0))?;
        out.queue(cursor::MoveUp(cursor_row + 1))?;

        for i in 0..self.last_render_height {
            out.queue(terminal::Clear(ClearType::CurrentLine))?;
            if i < self.last_render_height - 1 {
                out.queue(cursor::MoveDown(1))?;
            }
        }

        // Return to where the top border was.
        if self.last_render_height > 1 {
            out.queue(cursor::MoveUp(self.last_render_height - 1))?;
        }
        out.queue(cursor::MoveToColumn(0))?;
        self.last_render_height = 0;
        out.flush()
    }

    // -- Cursor movement --

    fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.prev_char_boundary();
        }
    }

    fn move_right(&mut self) {
        if self.cursor < self.buf.len() {
            self.cursor = self.next_char_boundary();
        }
    }

    fn move_word_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut pos = self.cursor;
        // Skip whitespace backwards.
        while pos > 0 {
            pos = prev_boundary(&self.buf, pos);
            let ch = self.buf[pos..].chars().next().unwrap();
            if !ch.is_whitespace() {
                break;
            }
        }
        // Skip word characters backwards.
        while pos > 0 {
            let prev = prev_boundary(&self.buf, pos);
            let ch = self.buf[prev..].chars().next().unwrap();
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                break;
            }
            pos = prev;
        }
        self.cursor = pos;
    }

    fn move_word_right(&mut self) {
        let len = self.buf.len();
        if self.cursor >= len {
            return;
        }
        let mut pos = self.cursor;
        // Skip word characters forward.
        while pos < len {
            let ch = self.buf[pos..].chars().next().unwrap();
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                break;
            }
            pos = next_boundary(&self.buf, pos);
        }
        // Skip whitespace/punctuation forward.
        while pos < len {
            let ch = self.buf[pos..].chars().next().unwrap();
            if !ch.is_whitespace() && !ch.is_ascii_punctuation() {
                break;
            }
            pos = next_boundary(&self.buf, pos);
        }
        self.cursor = pos;
    }

    fn delete_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let old = self.cursor;
        self.move_word_left();
        self.buf.drain(self.cursor..old);
    }

    fn prev_char_boundary(&self) -> usize {
        prev_boundary(&self.buf, self.cursor)
    }

    fn next_char_boundary(&self) -> usize {
        next_boundary(&self.buf, self.cursor)
    }

    // -- History --

    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.history_idx {
            None => {
                self.saved_input = self.buf.clone();
                let idx = self.history.len() - 1;
                self.history_idx = Some(idx);
                self.buf = self.history[idx].clone();
                self.cursor = self.buf.len();
            }
            Some(idx) if idx > 0 => {
                let new_idx = idx - 1;
                self.history_idx = Some(new_idx);
                self.buf = self.history[new_idx].clone();
                self.cursor = self.buf.len();
            }
            _ => {}
        }
    }

    fn history_down(&mut self) {
        if let Some(idx) = self.history_idx {
            if idx + 1 < self.history.len() {
                let new_idx = idx + 1;
                self.history_idx = Some(new_idx);
                self.buf = self.history[new_idx].clone();
            } else {
                self.history_idx = None;
                self.buf = self.saved_input.clone();
            }
            self.cursor = self.buf.len();
        }
    }

    // -- Tab completion --

    fn try_complete(&mut self) {
        if !self.buf.starts_with('/') {
            return;
        }
        let prefix = &self.buf;
        let mut matches = SLASH_COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(prefix))
            .take(2);
        let first = matches.next();
        let second = matches.next();
        if let (Some(only), None) = (first, second) {
            self.buf = format!("{only} ");
            self.cursor = self.buf.len();
        }
    }
}

// -- String helpers --

fn prev_boundary(s: &str, pos: usize) -> usize {
    let mut p = pos.saturating_sub(1);
    while p > 0 && !s.is_char_boundary(p) {
        p -= 1;
    }
    p
}

fn next_boundary(s: &str, pos: usize) -> usize {
    let mut p = pos + 1;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Wrap text to fit within `max_width` display columns.
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0;

    for ch in text.chars() {
        let ch_w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if current_width + ch_w > max_width && !current.is_empty() {
            lines.push(current);
            current = String::new();
            current_width = 0;
        }
        current.push(ch);
        current_width += ch_w;
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

/// Run the input editor in a dedicated thread. Enables raw mode on entry,
/// restores on exit. Renders to stderr. The `output_rx` channel coordinates
/// with the async side to hide/show the box around output.
pub fn run_input_thread(
    label: String,
    line_tx: tokio::sync::mpsc::Sender<InputResult>,
    output_rx: std_mpsc::Receiver<OutputSignal>,
) {
    let mut editor = InputEditor::new(label);
    let mut stderr = std::io::stderr();
    let mut box_visible = true;

    if terminal::enable_raw_mode().is_err() {
        drop(output_rx);
        run_fallback_input(line_tx);
        return;
    }

    let _ = stderr.execute(cursor::SetCursorStyle::SteadyBar);
    let _ = editor.render(&mut stderr);

    loop {
        // Drain all pending output signals.
        while let Ok(signal) = output_rx.try_recv() {
            match signal {
                OutputSignal::HideBox(ack) => {
                    let _ = editor.clear_box(&mut stderr);
                    box_visible = false;
                    let _ = ack.send(());
                }
                OutputSignal::ShowBox => {
                    let _ = editor.render(&mut stderr);
                    box_visible = true;
                }
                OutputSignal::UpdateLabel(new_label) => {
                    editor.set_label(new_label);
                    if box_visible {
                        let _ = editor.render(&mut stderr);
                    }
                }
            }
        }

        // Poll with short timeout so output signals are handled promptly.
        if event::poll(std::time::Duration::from_millis(16)).unwrap_or(false) {
            let ev = match event::read() {
                Ok(ev) => ev,
                Err(_) => break,
            };

            match ev {
                Event::Key(key) => match editor.handle_key(key) {
                    InputAction::Submit(line) => {
                        if box_visible {
                            let _ = editor.clear_box(&mut stderr);
                        }
                        let _ = write!(stderr, "\x1b[1myou>\x1b[0m {line}\r\n");
                        let _ = editor.render(&mut stderr);
                        box_visible = true;
                        if line_tx.blocking_send(InputResult::Line(line)).is_err() {
                            break;
                        }
                    }
                    InputAction::Interrupt => {
                        if box_visible {
                            let _ = editor.clear_box(&mut stderr);
                        }
                        if line_tx.blocking_send(InputResult::Interrupt).is_err() {
                            break;
                        }
                        let _ = editor.render(&mut stderr);
                        box_visible = true;
                    }
                    InputAction::Eof => {
                        if box_visible {
                            let _ = editor.clear_box(&mut stderr);
                        }
                        let _ = line_tx.blocking_send(InputResult::Eof);
                        break;
                    }
                    InputAction::Redraw if box_visible => {
                        let _ = editor.render(&mut stderr);
                    }
                    _ => {}
                },
                Event::Resize(w, _h) => {
                    editor.term_width = w;
                    if box_visible {
                        let _ = editor.render(&mut stderr);
                    }
                }
                _ => {}
            }
        }
    }

    let _ = terminal::disable_raw_mode();
    let _ = stderr.execute(cursor::SetCursorStyle::DefaultUserShape);
}

/// Fallback for environments that don't support raw mode.
fn run_fallback_input(line_tx: tokio::sync::mpsc::Sender<InputResult>) {
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        eprint!("you> ");
        let _ = std::io::stderr().flush();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => {
                let _ = line_tx.blocking_send(InputResult::Eof);
                break;
            }
            Ok(_) => {
                if line_tx
                    .blocking_send(InputResult::Line(line.trim().to_string()))
                    .is_err()
                {
                    break;
                }
            }
        }
    }
}
