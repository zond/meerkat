//! Event rendering to the terminal.
//!
//! Prints attributed agent events from the mob event router to stderr,
//! tracking the "current speaker" and buffering text per-agent to avoid
//! character-level interleaving when multiple agents stream simultaneously.

use meerkat_core::event::AgentEvent;
use meerkat_mob::{AttributedEvent, MeerkatId, ProfileName};
use std::collections::HashMap;

/// Tracks current speaker and buffers text per-agent for coherent output.
pub struct EventRenderer {
    /// Text buffer per agent — flushed on TextComplete or non-text events.
    text_buffers: HashMap<MeerkatId, String>,
    /// Which agent was last printed to stderr (for header changes).
    current_source: Option<MeerkatId>,
}

impl EventRenderer {
    pub fn new() -> Self {
        Self {
            text_buffers: HashMap::new(),
            current_source: None,
        }
    }

    /// Flush the text buffer for a specific agent to stderr.
    fn flush_buffer(&mut self, meerkat_id: &MeerkatId, profile: &ProfileName) {
        if let Some(text) = self.text_buffers.remove(meerkat_id) {
            if text.is_empty() {
                return;
            }
            let source_changed = self
                .current_source
                .as_ref()
                .map_or(true, |id| id != meerkat_id);
            if source_changed {
                eprintln!();
                self.current_source = Some(meerkat_id.clone());
            }
            eprint!("[{profile}/{meerkat_id}] {text}");
            eprintln!();
        }
    }

    /// Render one attributed event. Returns a JSONL line for logging.
    pub fn render(&mut self, event: &AttributedEvent) -> String {
        let source = &event.source;
        let profile = &event.profile;
        let payload = &event.envelope.payload;

        match payload {
            AgentEvent::TextDelta { delta } => {
                // Buffer text deltas — flushed on TextComplete.
                self.text_buffers
                    .entry(source.clone())
                    .or_default()
                    .push_str(delta);
            }
            AgentEvent::TextComplete { .. } => {
                self.flush_buffer(source, profile);
            }
            AgentEvent::RunStarted { .. } => {
                self.flush_buffer(source, profile);
                self.track_source(source);
                eprintln!("[{profile}/{source}] turn started");
            }
            AgentEvent::RunCompleted { .. } => {
                self.flush_buffer(source, profile);
                self.track_source(source);
                eprintln!("[{profile}/{source}] turn completed");
                eprintln!("---");
            }
            AgentEvent::RunFailed { error, .. } => {
                self.flush_buffer(source, profile);
                self.track_source(source);
                eprintln!("[{profile}/{source}] FAILED: {error}");
            }
            AgentEvent::ToolCallRequested { name, .. } => {
                self.flush_buffer(source, profile);
                self.track_source(source);
                eprintln!("[{profile}/{source}] tool: {name}");
            }
            AgentEvent::ToolExecutionCompleted {
                name,
                is_error,
                duration_ms,
                ..
            } => {
                let status = if *is_error { "ERR" } else { "ok" };
                self.track_source(source);
                eprintln!(
                    "[{profile}/{source}] tool done: {name} ({status}, {duration_ms}ms)"
                );
            }
            AgentEvent::TurnStarted { turn_number } => {
                self.flush_buffer(source, profile);
                self.track_source(source);
                eprintln!("[{profile}/{source}] LLM turn {turn_number}");
            }
            AgentEvent::TurnCompleted { usage, .. } => {
                let total = usage.input_tokens + usage.output_tokens;
                self.track_source(source);
                eprintln!("[{profile}/{source}] turn done ({total} tokens)");
            }
            // Skip verbose events
            _ => {}
        }

        // Return a JSONL line for the event log.
        match serde_json::to_string(event) {
            Ok(line) => line,
            Err(e) => {
                tracing::warn!("failed to serialize event for log: {e}");
                String::new()
            }
        }
    }

    /// Track source changes for non-text events.
    fn track_source(&mut self, source: &MeerkatId) {
        let source_changed = self
            .current_source
            .as_ref()
            .map_or(true, |id| id != source);
        if source_changed {
            self.current_source = Some(source.clone());
        }
    }
}
