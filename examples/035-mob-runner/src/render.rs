//! Event rendering to the terminal.
//!
//! Prints attributed agent events from the mob event router to stderr,
//! tracking the "current speaker" and buffering text per-agent to avoid
//! character-level interleaving when multiple agents stream simultaneously.
//!
//! Coordinates with the input box via `with_output`: hides the box,
//! writes output, then re-shows it so the box stays at the bottom.

use crate::format;
use crate::input::OutputSignal;
use meerkat_core::event::AgentEvent;
use meerkat_mob::{AttributedEvent, MeerkatId, ProfileName};
use std::collections::HashMap;
use std::sync::mpsc as std_mpsc;

/// Tracks current speaker and buffers text per-agent for coherent output.
pub struct EventRenderer {
    text_buffers: HashMap<MeerkatId, String>,
    current_source: Option<MeerkatId>,
    output_tx: std_mpsc::Sender<OutputSignal>,
}

impl EventRenderer {
    pub fn new(output_tx: std_mpsc::Sender<OutputSignal>) -> Self {
        Self {
            text_buffers: HashMap::new(),
            current_source: None,
            output_tx,
        }
    }

    fn emit(&self, msg: String) {
        crate::input::with_output(&self.output_tx, || eprint!("{msg}"));
    }

    fn flush_buffer(&mut self, meerkat_id: &MeerkatId, profile: &ProfileName) {
        if let Some(text) = self.text_buffers.remove(meerkat_id) {
            if text.is_empty() {
                return;
            }
            let source_changed = self
                .current_source
                .as_ref()
                .is_none_or(|id| id != meerkat_id);
            self.current_source = Some(meerkat_id.clone());

            let prefix = if source_changed { "\r\n" } else { "" };
            let text = text.replace('\n', "\r\n");
            self.emit(format!(
                "{prefix}\x1b[36m[{profile}/{meerkat_id}]\x1b[0m {text}\r\n"
            ));
        }
    }

    pub fn render(&mut self, event: &AttributedEvent) -> String {
        let source = &event.source;
        let profile = &event.profile;
        let payload = &event.envelope.payload;

        match payload {
            AgentEvent::TextDelta { delta } => {
                self.text_buffers
                    .entry(source.clone())
                    .or_default()
                    .push_str(delta);
            }
            AgentEvent::TextComplete { .. } => {
                self.flush_buffer(source, profile);
            }
            AgentEvent::RunFailed { error, .. } => {
                self.flush_buffer(source, profile);
                self.current_source = Some(source.clone());
                self.emit(format!(
                    "\x1b[36m[{profile}/{source}]\x1b[0m \x1b[31mFAILED: {error}\x1b[0m\r\n"
                ));
            }
            AgentEvent::ToolCallRequested { name, args, .. } => {
                if !format::should_show_tool(name) {
                    return self.serialize(event);
                }
                self.flush_buffer(source, profile);
                self.current_source = Some(source.clone());
                let preview = format::tool_args_preview(name, args);
                if preview.is_empty() {
                    self.emit(format!(
                        "\x1b[36m[{profile}/{source}]\x1b[0m \x1b[33m{name}\x1b[0m\r\n"
                    ));
                } else {
                    self.emit(format!(
                        "\x1b[36m[{profile}/{source}]\x1b[0m \x1b[33m{name}\x1b[0m {preview}\r\n"
                    ));
                }
            }
            AgentEvent::ToolExecutionCompleted {
                name,
                result,
                is_error,
                duration_ms,
                ..
            } => {
                if !format::should_show_tool(name) {
                    return self.serialize(event);
                }
                let status = if *is_error {
                    "\x1b[31mERR\x1b[0m"
                } else {
                    "\x1b[32mok\x1b[0m"
                };
                let preview = format::tool_result_preview(name, result, 3);
                self.current_source = Some(source.clone());
                if preview.is_empty() {
                    self.emit(format!(
                        "\x1b[36m[{profile}/{source}]\x1b[0m \x1b[2m{name} {status}\x1b[2m {duration_ms}ms\x1b[0m\r\n"
                    ));
                } else {
                    self.emit(format!(
                        "\x1b[36m[{profile}/{source}]\x1b[0m \x1b[2m{name} {status}\x1b[2m {duration_ms}ms\x1b[0m {preview}\r\n"
                    ));
                }
            }
            // Suppress noisy lifecycle events:
            // RunStarted, RunCompleted, TurnStarted, TurnCompleted
            _ => {}
        }

        self.serialize(event)
    }

    fn serialize(&self, event: &AttributedEvent) -> String {
        match serde_json::to_string(event) {
            Ok(line) => line,
            Err(e) => {
                tracing::warn!("failed to serialize event for log: {e}");
                String::new()
            }
        }
    }
}
