//! Planning phase: interactive multi-turn chat with a planner agent.
//!
//! The planner explores the codebase, chats with the user, and outputs a
//! ```toml mob definition when the plan is ready.

use color_eyre::eyre::{self, WrapErr};
use meerkat::SessionService;
use meerkat_core::event::{AgentEvent, EventEnvelope};
use meerkat_core::service::{CreateSessionRequest, InitialTurnPolicy, StartTurnRequest};
use meerkat_core::types::SessionId;
use std::sync::{Arc, LazyLock};
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc;

use crate::raw_eprintln;
use crate::input::{self, InputResult, OutputSignal};
use crate::state::StateDir;

use crate::format;

type SessionSvc = dyn SessionService;

/// Compiled regex for extracting ```toml fenced code blocks.
/// Captures the last ```toml block to avoid partial fragments the LLM
/// might emit during explanation.
static TOML_BLOCK_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?si)```toml\s*\n(.*?)```").expect("valid regex"));

/// Detect available providers based on environment variables.
fn available_providers() -> Vec<(&'static str, &'static str, &'static [&'static str])> {
    let mut providers = Vec::new();
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        providers.push((
            "Anthropic",
            "ANTHROPIC_API_KEY",
            &["claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-4-5"][..],
        ));
    }
    if std::env::var("GEMINI_API_KEY").is_ok() {
        providers.push((
            "Gemini",
            "GEMINI_API_KEY",
            &[
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-lite",
                "gemini-3-flash-preview",
            ][..],
        ));
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        providers.push((
            "OpenAI",
            "OPENAI_API_KEY",
            &["gpt-5.4", "gpt-5.3-codex", "gpt-5.2"][..],
        ));
    }
    providers
}

/// Pick a default model from available providers.
/// Priority: Anthropic > Gemini > OpenAI.
pub fn pick_default_model() -> Option<String> {
    let providers = available_providers();
    providers.first().map(|(_, _, models)| models[0].to_string())
}

/// Build the provider context block for the planner's system prompt.
fn provider_context() -> String {
    let providers = available_providers();
    if providers.is_empty() {
        return "WARNING: No API keys detected. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.".to_string();
    }
    let mut lines = vec!["Available providers and models:".to_string()];
    for (name, key, models) in &providers {
        let model_list = models.join(", ");
        lines.push(format!("- {name} ({key} is set): {model_list}"));
    }
    lines.push("Use only models from providers with available API keys.".to_string());
    lines.join("\n")
}

/// Build the planner's system prompt.
fn planner_system_prompt() -> String {
    let provider_ctx = provider_context();
    format!(
        r#"You are a mob planning agent for the Meerkat multi-agent framework.

Your job is to chat with the user, understand what they want to accomplish, and design a mob of collaborating AI agents to carry out the work.

## How to interact

1. Ask clarifying questions about the user's goal.
2. Explore the codebase using your shell and builtin tools as needed.
3. When the plan is clear and the user agrees, output the mob definition as a fenced TOML block.

## Mob definition format

When you're ready to deploy, output a ```toml block containing the mob definition. Example:

```toml
[mob]
id = "my-mob"
orchestrator = "lead"

[profiles.lead]
model = "claude-sonnet-4-6"
skills = ["orchestration"]
peer_description = "Orchestrator that coordinates all workers"
external_addressable = true

[profiles.lead.tools]
builtins = true
comms = true
mob = true
mob_tasks = true

[profiles.worker]
model = "claude-sonnet-4-6"
skills = ["implementation"]
peer_description = "Implementation worker"

[profiles.worker.tools]
builtins = true
shell = true
comms = true
mob_tasks = true

[wiring]
auto_wire_orchestrator = true

[skills.orchestration]
source = "inline"
content = "You are the orchestrator. Break the task into subtasks, assign to workers via comms, and track progress on the task board."

[skills.implementation]
source = "inline"
content = "You are a worker. Implement assigned tasks, report progress via comms."
```

## Required fields

- `[mob]` MUST have `id` (unique string) and `orchestrator` (profile name of the orchestrator)
- Each `[profiles.<name>]` MUST have `model` (LLM model name)

## Rules for mob definitions

**Hard constraints (must follow):**
- The orchestrator profile MUST have `external_addressable = true` (receives user messages)
- The orchestrator profile MUST have `comms = true` and `mob_tasks = true`
- The orchestrator SHOULD have `mob = true` so it can spawn/retire agents at runtime

**Design guidance:**
- Workers that need file/shell access get `builtins = true`, `shell = true`
- Workers that need to talk to peers get `comms = true`
- Wire workers together if they need direct collaboration (or use `auto_wire_orchestrator = true`)
- Write detailed, specific skill content for each role — this is the agent's primary context
- Use appropriate models for each role (more capable for orchestrator, efficient for workers)
- Define enough profiles upfront: the orchestrator can spawn/retire agents at runtime but cannot define new profiles
- `memory = true` in `[profiles.<name>.tools]` enables semantic memory tools for that agent

## Available tools (use these exact names in skill content)

Agents have these tools depending on their `[profiles.X.tools]` config:

**comms = true:**
- `send` — Send message to a peer. Args: `to` (peer name), `body` (text), `kind` ("peer_message")
- `peers` — List connected peers

**mob_tasks = true:**
- `mob_task_create` — Create a task. Args: `subject` (short title), `description` (details), `blocked_by` (optional list of task IDs)
- `mob_task_list` — List all tasks with status
- `mob_task_update` — Update task status. Args: `task_id`, `status` ("in_progress"/"completed"/"blocked"), `owner` (optional)

**mob = true (orchestrator only):**
- `spawn_meerkat` — Spawn a new agent
- `retire_meerkat` — Remove an agent
- `wire_peers` — Connect two agents for direct communication. Args: `a`, `b`
- `list_meerkats` — List all agents

**shell = true:**
- `shell` — Run a shell command. Args: `command` (string)

**builtins = true:**
- `read_file`, `write_file`, `list_directory` — File operations

## Skill content guidelines

- ALWAYS reference exact tool names (e.g. `mob_task_create`, NOT `task_create`)
- Workers should NOT passively wait. They should poll `mob_task_list` periodically to find assigned tasks
- Workers should use `send` with `kind: "peer_message"` to report progress
- The orchestrator should create tasks with `mob_task_create`, assign them by setting `owner` via `mob_task_update`, and send instructions via `send`
- Reviewers should poll `mob_task_list` for tasks assigned to them
- The orchestrator's TEXT OUTPUT (not `send`) is what the user sees. To reply to the user, just output text — do NOT use `send` to target "user" (there is no peer named "user")
- Only use `send` to communicate with other agents in the mob

## {provider_ctx}

When the user says "go", "deploy", "let's do it", or similar — finalize and output the TOML definition."#
    )
}

/// Extract the last ```toml fenced code block from text.
/// Uses the last match to avoid capturing partial fragments the LLM
/// might output during explanation before the final definition.
fn extract_toml_block(text: &str) -> Option<String> {
    TOML_BLOCK_RE
        .captures_iter(text)
        .last()
        .map(|caps| caps[1].to_string())
}

/// Run a turn on the planner session and stream text events to stderr.
/// Returns the accumulated text from the turn.
///
/// Hides the input box during output, then re-shows it when done.
async fn run_turn_streaming(
    session_service: &Arc<SessionSvc>,
    session_id: &SessionId,
    prompt: String,
    output_tx: &std_mpsc::Sender<OutputSignal>,
) -> color_eyre::Result<String> {
    let (turn_tx, mut turn_rx) = mpsc::channel::<EventEnvelope<AgentEvent>>(256);

    let turn_handle = {
        let svc = session_service.clone();
        let sid = session_id.clone();
        tokio::spawn(async move {
            svc.start_turn(
                &sid,
                StartTurnRequest {
                    prompt: prompt.into(),
                    event_tx: Some(turn_tx),
                    host_mode: false,
                    skill_references: None,
                    flow_tool_overlay: None,
                    additional_instructions: None,
                },
            )
            .await
        })
    };

    // Hide the input box while streaming output.
    input::hide_box(output_tx);

    let mut full_text = String::new();
    let mut in_text = false;
    while let Some(envelope) = turn_rx.recv().await {
        match &envelope.payload {
            AgentEvent::TextDelta { delta } => {
                if !in_text {
                    eprint!("\x1b[36mplanner>\x1b[0m ");
                    in_text = true;
                }
                // In raw mode, \n alone doesn't return to column 0.
                eprint!("{}", delta.replace('\n', "\r\n"));
                full_text.push_str(delta);
            }
            AgentEvent::TextComplete { content } => {
                if in_text {
                    raw_eprintln!();
                    in_text = false;
                }
                if full_text.is_empty() {
                    full_text = content.clone();
                }
            }
            AgentEvent::ToolCallRequested { name, args, .. } => {
                if in_text {
                    raw_eprintln!();
                    in_text = false;
                }
                let preview = format::tool_args_preview(name, args);
                if preview.is_empty() {
                    raw_eprintln!("  \x1b[33m[tool: {name}]\x1b[0m");
                } else {
                    raw_eprintln!("  \x1b[33m[tool: {name}]\x1b[0m {preview}");
                }
            }
            AgentEvent::ToolExecutionCompleted {
                name,
                result,
                is_error,
                duration_ms,
                ..
            } => {
                let status = if *is_error { "\x1b[31mERR\x1b[0m" } else { "\x1b[32mok\x1b[0m" };
                let preview = format::tool_result_preview(name, result, 3);
                if preview.is_empty() {
                    raw_eprintln!("  \x1b[2m[{name} {status}\x1b[2m {duration_ms}ms]\x1b[0m");
                } else {
                    raw_eprintln!("  \x1b[2m[{name} {status}\x1b[2m {duration_ms}ms]\x1b[0m {preview}");
                }
            }
            AgentEvent::RunFailed { error, .. } => {
                raw_eprintln!("\r\n\x1b[31m[ERROR: {error}]\x1b[0m");
            }
            _ => {}
        }
    }

    if in_text {
        raw_eprintln!();
    }

    // Re-show the input box.
    let _ = output_tx.send(OutputSignal::ShowBox);

    let turn_result = turn_handle
        .await
        .wrap_err("planner turn task panicked")?
        .wrap_err("planner turn failed")?;
    if full_text.is_empty() {
        full_text = turn_result.text.clone();
    }

    Ok(full_text)
}

/// Validate a TOML block as a mob definition.
/// Returns `Ok(toml_string)` if valid, `Err(feedback_message)` if invalid.
fn validate_mob_toml(
    toml_block: &str,
    output_tx: &std_mpsc::Sender<OutputSignal>,
) -> Result<String, String> {
    match meerkat_mob::MobDefinition::from_toml(toml_block) {
        Ok(def) => {
            let diagnostics = meerkat_mob::validate_definition(&def);
            let errors: Vec<_> = diagnostics
                .iter()
                .filter(|d| d.severity == meerkat_mob::DiagnosticSeverity::Error)
                .collect();
            let warnings: Vec<_> = diagnostics
                .iter()
                .filter(|d| d.severity == meerkat_mob::DiagnosticSeverity::Warning)
                .collect();
            if !warnings.is_empty() {
                input::with_output(output_tx, || {
                    for w in &warnings {
                        raw_eprintln!("  \x1b[33mWARNING: {}\x1b[0m", w.message);
                    }
                });
            }
            if errors.is_empty() {
                let id = def.id.clone();
                input::with_output(output_tx, || {
                    raw_eprintln!("\x1b[32m[Valid mob definition -- deploying '{id}']\x1b[0m");
                });
                Ok(toml_block.to_string())
            } else {
                let mut feedback = String::from(
                    "[SYSTEM] Your mob definition has validation errors. \
                     Please fix and output a corrected ```toml block:\n",
                );
                input::with_output(output_tx, || {
                    for d in &errors {
                        raw_eprintln!("  \x1b[31mERROR: {}\x1b[0m", d.message);
                    }
                });
                for d in &errors {
                    feedback.push_str(&format!("- {}\n", d.message));
                }
                Err(feedback)
            }
        }
        Err(e) => {
            input::with_output(output_tx, || {
                raw_eprintln!("\x1b[31m[TOML parse error: {e}]\x1b[0m");
            });
            Err(format!(
                "[SYSTEM] Your TOML mob definition failed to parse: {e}\n\
                 Please fix and output a corrected ```toml block."
            ))
        }
    }
}

/// Create a fresh planner session with the system prompt.
async fn create_planner_session(
    session_service: &Arc<SessionSvc>,
    model: &str,
) -> color_eyre::Result<SessionId> {
    let result = session_service
        .create_session(CreateSessionRequest {
            model: model.to_string(),
            prompt: String::new().into(),
            system_prompt: Some(planner_system_prompt()),
            max_tokens: None,
            event_tx: None,
            host_mode: false,
            skill_references: None,
            initial_turn: InitialTurnPolicy::Defer,
            build: None,
            labels: None,
        })
        .await
        .wrap_err("failed to create planner session")?;
    Ok(result.session_id)
}

/// Run the interactive planning phase.
///
/// Uses the rich input editor. Returns the TOML mob definition and the
/// input channels for reuse by the execution phase.
///
/// If a planner session ID file exists from a previous run, the conversation
/// is resumed rather than starting fresh (the session data is in the persistent
/// redb store).
pub async fn run_planner(
    session_service: Arc<SessionSvc>,
    model: &str,
    state: &StateDir,
    mut input_rx: mpsc::Receiver<InputResult>,
    output_tx: std_mpsc::Sender<OutputSignal>,
) -> color_eyre::Result<(String, mpsc::Receiver<InputResult>, std_mpsc::Sender<OutputSignal>)> {
    input::with_output(&output_tx, || {
        raw_eprintln!("\x1b[1m=== Mob Runner -- Planning Phase ===\x1b[0m");
        raw_eprintln!();
        raw_eprintln!("Chat with the planner to design your mob. The planner can explore");
        raw_eprintln!("the codebase, ask questions, and help you design a team of agents.");
        raw_eprintln!("When the plan is ready, it will output a mob definition.");
        raw_eprintln!();
    });

    // Resume an existing planner session or create a new one.
    let session_id = if state.has_planner_session() {
        let id_str = std::fs::read_to_string(state.planner_session_id())
            .wrap_err("failed to read planner session ID")?;
        let id: SessionId = serde_json::from_str(&id_str)
            .wrap_err("failed to parse planner session ID")?;
        // Verify the session still exists in the store.
        match session_service.read(&id).await {
            Ok(_) => {
                raw_eprintln!("[Resuming planner session {id}]");
                id
            }
            Err(_) => {
                raw_eprintln!("[Previous planner session not found, starting fresh]");
                create_planner_session(&session_service, model).await?
            }
        }
    } else {
        create_planner_session(&session_service, model).await?
    };

    // Persist the session ID so planning can resume across restarts.
    std::fs::write(
        state.planner_session_id(),
        serde_json::to_string(&session_id)?,
    )
    .wrap_err("failed to save planner session ID")?;

    let mut session_id = session_id;
    let mut first_turn = true;

    // Interactive loop.
    loop {
        let input = match input_rx.recv().await {
            Some(InputResult::Line(line)) => line,
            Some(InputResult::Interrupt) => {
                input::hide_box(&output_tx);
                raw_eprintln!("\r\n\x1b[2m[Interrupted -- exiting]\x1b[0m");
                break;
            }
            Some(InputResult::Eof) | None => {
                input::hide_box(&output_tx);
                raw_eprintln!("\r\n\x1b[2m[EOF -- exiting]\x1b[0m");
                break;
            }
        };

        if input.is_empty() {
            continue;
        }

        // On the first turn, if the session is stale (e.g. redb was deleted),
        // start_turn will fail. Recover by creating a fresh session.
        let text = match run_turn_streaming(&session_service, &session_id, input.clone(), &output_tx).await {
            Ok(text) => text,
            Err(e) if first_turn => {
                input::with_output(&output_tx, || {
                    raw_eprintln!("\x1b[33m[Session error: {e:#}]\x1b[0m");
                    raw_eprintln!("\x1b[2m[Starting fresh planner session...]\x1b[0m");
                });
                session_id = create_planner_session(&session_service, model).await?;
                std::fs::write(
                    state.planner_session_id(),
                    serde_json::to_string(&session_id)?,
                ).wrap_err("failed to save planner session ID")?;
                run_turn_streaming(&session_service, &session_id, input, &output_tx).await?
            }
            Err(e) => return Err(e),
        };
        first_turn = false;

        // Check for a TOML mob definition. On validation failure, feed the error
        // back to the planner as a correction turn so it can self-correct.
        if let Some(toml_block) = extract_toml_block(&text) {
            match validate_mob_toml(&toml_block, &output_tx) {
                Ok(valid_toml) => return Ok((valid_toml, input_rx, output_tx)),
                Err(feedback) => {
                    input::with_output(&output_tx, || {
                        raw_eprintln!("\x1b[2m[Feeding errors back to planner for self-correction]\x1b[0m");
                    });
                    let _text = run_turn_streaming(&session_service, &session_id, feedback, &output_tx).await?;

                    if let Some(fixed_toml) = extract_toml_block(&_text) {
                        if let Ok(valid_toml) = validate_mob_toml(&fixed_toml, &output_tx) {
                            return Ok((valid_toml, input_rx, output_tx));
                        }
                    }
                    input::with_output(&output_tx, || {
                        raw_eprintln!("\x1b[2m[Still invalid -- continue chatting to fix]\x1b[0m");
                    });
                }
            }
        }
    }

    Err(eyre::eyre!("Planning cancelled"))
}
