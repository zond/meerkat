//! Modal dialogs: the escalation (approval) prompt and the resume picker.
//!
//! Modals are driven by plain data so the UI stays decoupled from the approval
//! engine and session modules.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block as BorderBlock, Borders, Clear, Paragraph};

use super::markdown::{truncate_to_width, wrap_text};

/// Maximum lines of the args preview shown in the escalation dialog.
const MAX_ARGS_LINES: usize = 10;

/// Identifier of a pending escalation, assigned by the approval engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EscalationId(pub u64);

/// The user's answer to an escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationAnswer {
    /// Allow this call only.
    AllowOnce,
    /// Allow and persist a narrow allow rule.
    AllowAndAddRule,
    /// Deny this call.
    Deny,
    /// Deny and stop the turn.
    Stop,
}

/// What the escalation dialog shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EscalationPrompt {
    /// Tool name.
    pub tool: String,
    /// Human-readable preview of the arguments.
    pub args_preview: String,
    /// Classifier explanation, if any.
    pub hint: Option<String>,
}

/// Session picker for `/resume` without a reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumePicker {
    /// `(session ref, label)` pairs, newest first.
    pub items: Vec<(String, String)>,
    /// Selected index.
    pub selected: usize,
}

impl ResumePicker {
    /// Move the selection up.
    pub fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    /// Move the selection down.
    pub fn down(&mut self) {
        if self.selected + 1 < self.items.len() {
            self.selected += 1;
        }
    }

    /// The selected session ref.
    pub fn selected_ref(&self) -> Option<&str> {
        self.items.get(self.selected).map(|(r, _)| r.as_str())
    }
}

/// An open modal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Modal {
    /// A tool call awaiting a human decision.
    Escalation {
        /// Escalation id to answer.
        id: EscalationId,
        /// What to show.
        prompt: EscalationPrompt,
        /// How many more escalations are queued behind this one.
        queued_behind: usize,
    },
    /// Session picker.
    Resume(ResumePicker),
}

/// Center a `width`×`height` rectangle inside `area` (clamped to fit).
pub fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

/// Draw `modal` centered over `area`.
pub fn draw_modal(frame: &mut Frame<'_>, area: Rect, modal: &Modal) {
    let width = area.width.saturating_sub(4).clamp(20, 80);
    let inner_w = usize::from(width.saturating_sub(2));
    let (title, lines) = match modal {
        Modal::Escalation {
            prompt,
            queued_behind,
            ..
        } => {
            let title = if *queued_behind > 0 {
                format!(" Approval needed (+{queued_behind} queued) ")
            } else {
                " Approval needed ".to_string()
            };
            (title, escalation_lines(prompt, inner_w))
        }
        Modal::Resume(picker) => (
            " Resume session ".to_string(),
            picker_lines(picker, inner_w, usize::from(area.height.saturating_sub(4))),
        ),
    };
    let height = (lines.len() as u16).saturating_add(2);
    let rect = centered(area, width, height);
    frame.render_widget(Clear, rect);
    let block = BorderBlock::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(title);
    frame.render_widget(Paragraph::new(lines).block(block), rect);
}

fn escalation_lines(prompt: &EscalationPrompt, width: usize) -> Vec<Line<'static>> {
    let dim = Style::default().add_modifier(Modifier::DIM);
    let mut lines = vec![Line::from(vec![
        Span::styled("Tool: ", dim),
        Span::styled(
            prompt.tool.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ])];
    let args = wrap_text(&prompt.args_preview, width);
    let extra = args.len().saturating_sub(MAX_ARGS_LINES);
    lines.extend(args.into_iter().take(MAX_ARGS_LINES).map(Line::from));
    if extra > 0 {
        lines.push(Line::styled(format!("… ({extra} more lines)"), dim));
    }
    if let Some(hint) = &prompt.hint {
        lines.push(Line::default());
        lines.extend(
            wrap_text(&format!("Classifier: {hint}"), width)
                .into_iter()
                .map(|l| Line::styled(l, Style::default().fg(Color::Yellow))),
        );
    }
    lines.push(Line::default());
    let key = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    lines.push(Line::from(vec![
        Span::styled("[y]", key),
        Span::raw(" allow once  "),
        Span::styled("[a]", key),
        Span::raw(" allow + rule  "),
        Span::styled("[n]", key),
        Span::raw(" deny  "),
        Span::styled("[s]", key),
        Span::raw(" stop"),
    ]));
    lines
}

fn picker_lines(picker: &ResumePicker, width: usize, max_rows: usize) -> Vec<Line<'static>> {
    if picker.items.is_empty() {
        return vec![Line::styled(
            "no sessions found (Esc to close)",
            Style::default().add_modifier(Modifier::DIM),
        )];
    }
    let rows = max_rows.max(1);
    let start = picker.selected.saturating_sub(rows.saturating_sub(1));
    picker
        .items
        .iter()
        .enumerate()
        .skip(start)
        .take(rows)
        .map(|(i, (id, label))| {
            let text = truncate_to_width(&format!("{id}  {label}"), width);
            if i == picker.selected {
                Line::styled(text, Style::default().add_modifier(Modifier::REVERSED))
            } else {
                Line::from(text)
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn picker_navigation_is_clamped() {
        let mut p = ResumePicker {
            items: vec![("a".into(), "x".into()), ("b".into(), "y".into())],
            selected: 0,
        };
        p.up();
        assert_eq!(p.selected_ref(), Some("a"));
        p.down();
        p.down();
        assert_eq!(p.selected_ref(), Some("b"));
    }

    #[test]
    fn centered_rect_fits() {
        let r = centered(Rect::new(0, 0, 10, 10), 20, 4);
        assert_eq!(r, Rect::new(0, 3, 10, 4));
    }
}
