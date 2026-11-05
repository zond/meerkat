//! Terminal ownership and the UI event loop.
//!
//! All crossterm types come from `ratatui::crossterm` so the backend and our
//! commands can never disagree on the crossterm version; the `crossterm` crate is
//! used directly only for its `event-stream` feature (`EventStream`).

use std::io::{self, Stdout, Write};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use meerkat_core::{AgentEvent, EventEnvelope, ScopedAgentEvent};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    KeyEventKind, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    supports_keyboard_enhancement,
};
use tokio::sync::mpsc;

use super::render::draw;
use super::{App, Effect, History, Msg};

/// Render tick interval (~30 fps).
pub const TICK: Duration = Duration::from_millis(33);

/// Whether keyboard-enhancement flags were pushed (so restore knows to pop them).
static KEYBOARD_ENHANCED: AtomicBool = AtomicBool::new(false);
/// Whether mouse capture is enabled.
static MOUSE_CAPTURED: AtomicBool = AtomicBool::new(false);
/// Whether the terminal is currently in TUI mode.
static TERMINAL_ACTIVE: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK: Once = Once::new();

/// Errors from the UI runtime.
#[derive(Debug, thiserror::Error)]
pub enum UiError {
    /// Terminal I/O failed.
    #[error("terminal I/O failed: {0}")]
    Io(#[from] io::Error),
    /// The integrator dropped the effect receiver.
    #[error("effect channel closed")]
    EffectChannelClosed,
}

/// Restore the terminal to cooked mode. Idempotent and safe to call from a panic hook.
fn restore_terminal() {
    if !TERMINAL_ACTIVE.swap(false, Ordering::SeqCst) {
        return;
    }
    let mut out = io::stdout();
    if KEYBOARD_ENHANCED.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    if MOUSE_CAPTURED.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, DisableMouseCapture);
    }
    let _ = execute!(
        out,
        DisableBracketedPaste,
        LeaveAlternateScreen,
        ratatui::crossterm::cursor::Show
    );
    let _ = disable_raw_mode();
    let _ = out.flush();
}

/// RAII owner of the terminal's TUI mode (raw mode, alternate screen, paste, keyboard flags).
#[derive(Debug)]
pub struct TerminalGuard {
    _private: (),
}

impl TerminalGuard {
    /// Enter TUI mode. `mouse` enables mouse capture (wheel scrolling, but no native selection).
    pub fn enter(mouse: bool) -> io::Result<Self> {
        enable_raw_mode()?;
        TERMINAL_ACTIVE.store(true, Ordering::SeqCst);
        // From here on, `Drop` restores whatever was enabled if a later step fails.
        let guard = Self { _private: () };
        let mut out = io::stdout();
        execute!(out, EnterAlternateScreen, EnableBracketedPaste)?;
        if matches!(supports_keyboard_enhancement(), Ok(true)) {
            execute!(
                out,
                PushKeyboardEnhancementFlags(
                    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                        | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                )
            )?;
            KEYBOARD_ENHANCED.store(true, Ordering::SeqCst);
        }
        if mouse {
            execute!(out, EnableMouseCapture)?;
            MOUSE_CAPTURED.store(true, Ordering::SeqCst);
        }
        Ok(guard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// Install (once) a panic hook that restores the terminal before delegating to the
/// previous hook, so panic messages are readable.
pub fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore_terminal();
            previous(info);
        }));
    });
}

/// Receivers feeding the UI loop.
#[derive(Debug)]
pub struct UiChannels {
    /// Primary agent stream events.
    pub agent_rx: mpsc::Receiver<EventEnvelope<AgentEvent>>,
    /// Scoped (sub-agent) stream events.
    pub scoped_rx: mpsc::Receiver<ScopedAgentEvent>,
    /// Integrator messages: escalations, diffs, approvals, turn completion, status, notices.
    pub ui_rx: mpsc::UnboundedReceiver<Msg>,
}

fn translate(event: Event, mouse: bool) -> Option<Msg> {
    match event {
        // Windows reports Press and Release; enhanced terminals also report Repeat.
        Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
            Some(Msg::Key(key))
        }
        Event::Paste(text) => Some(Msg::Paste(text)),
        Event::Resize(w, h) => Some(Msg::Resize(w, h)),
        Event::Mouse(m) if mouse => Some(Msg::Mouse(m)),
        _ => None,
    }
}

/// Run the UI until the user quits.
///
/// Effects produced by [`App::update`] are forwarded on `effects`; [`Effect::Quit`]
/// is forwarded too and then ends the loop. Every submitted line is appended to
/// each of `histories` (e.g. the workspace and the global history file). The
/// terminal is restored on return, on error and on panic.
pub async fn run_ui(
    mut app: App,
    mut channels: UiChannels,
    effects: mpsc::UnboundedSender<Effect>,
    mouse: bool,
    mut histories: Vec<History>,
) -> Result<(), UiError> {
    install_panic_hook();
    let _guard = TerminalGuard::enter(mouse)?;
    let mut terminal: Terminal<CrosstermBackend<Stdout>> =
        Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

    let mut events = crossterm::event::EventStream::new();
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        if app.take_dirty() {
            terminal.draw(|frame| draw(frame, &mut app))?;
        }
        let msg = tokio::select! {
            maybe = events.next() => match maybe {
                Some(Ok(event)) => match translate(event, mouse) {
                    Some(msg) => msg,
                    None => continue,
                },
                Some(Err(err)) => return Err(UiError::Io(err)),
                None => {
                    // Input stream ended (terminal gone): tell the integrator to shut down.
                    let _ = effects.send(Effect::Quit);
                    return Ok(());
                }
            },
            Some(env) = channels.agent_rx.recv() => Msg::Agent(env),
            Some(ev) = channels.scoped_rx.recv() => Msg::Scoped(ev),
            Some(msg) = channels.ui_rx.recv() => msg,
            _ = tick.tick() => Msg::Tick,
        };
        let effects_now = app.update(msg);
        for entry in app.take_submitted() {
            for history in &mut histories {
                if let Err(err) = history.append(&entry) {
                    tracing::warn!(
                        error = %err,
                        path = %history.path().display(),
                        "could not append to input history"
                    );
                }
            }
        }
        for effect in effects_now {
            let quit = effect == Effect::Quit;
            if effects.send(effect).is_err() && !quit {
                return Err(UiError::EffectChannelClosed);
            }
            if quit {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventState, KeyModifiers};

    #[test]
    fn translate_filters_release_events() {
        let mut key = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        assert!(matches!(
            translate(Event::Key(key), false),
            Some(Msg::Key(_))
        ));
        key.kind = KeyEventKind::Release;
        key.state = KeyEventState::NONE;
        assert!(translate(Event::Key(key), false).is_none());
        assert!(matches!(
            translate(Event::Paste("x".into()), false),
            Some(Msg::Paste(_))
        ));
        assert!(matches!(
            translate(Event::Resize(1, 2), false),
            Some(Msg::Resize(1, 2))
        ));
        assert!(translate(Event::FocusGained, false).is_none());
    }
}
