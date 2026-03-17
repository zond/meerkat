//! Interactive mob loop: input box + event streaming + Ctrl+C handling.
//!
//! Routes user input to the orchestrator and streams events from all agents.

use crate::raw_eprintln;
use crate::input::{self, InputResult, OutputSignal};
use crate::render::EventRenderer;
use crate::state::StateDir;
use meerkat_mob::{MeerkatId, MobHandle};
use std::io::Write;
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc;

/// Find the orchestrator's meerkat ID from the mob definition.
fn orchestrator_id(handle: &MobHandle) -> Option<MeerkatId> {
    let def = handle.definition();
    def.orchestrator
        .as_ref()
        .map(|o| MeerkatId::from(o.profile.as_str()))
}

/// Run the interactive mob loop.
///
/// Uses the rich input editor for user input.
pub async fn run_mob_loop(
    handle: MobHandle,
    state: &StateDir,
    mut input_rx: mpsc::Receiver<InputResult>,
    output_tx: std_mpsc::Sender<OutputSignal>,
) -> color_eyre::Result<()> {
    let orch_id = orchestrator_id(&handle);

    // Print header.
    input::with_output(&output_tx, || {
        raw_eprintln!("\x1b[1m=== Mob Runner -- Execution Phase ===\x1b[0m");
        raw_eprintln!();
        if let Some(ref id) = orch_id {
            raw_eprintln!("Orchestrator: \x1b[36m{id}\x1b[0m");
        }
        raw_eprintln!("\x1b[2mCommands: /status, /members, /tasks, /send <agent> <msg>, /quit\x1b[0m");
        raw_eprintln!("\x1b[2mType a message to send to the orchestrator.\x1b[0m");
        raw_eprintln!();
    });

    let mut router_handle = handle.subscribe_mob_events();
    let mut renderer = EventRenderer::new(output_tx.clone());

    let log_path = state.events_jsonl();
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let mut input_open = true;

    loop {
        tokio::select! {
            result = input_rx.recv(), if input_open => {
                let Some(input_result) = result else {
                    input_open = false;
                    input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[input closed]\x1b[0m"));
                    continue;
                };
                match input_result {
                    InputResult::Line(line) => {
                        if line.is_empty() {
                            continue;
                        }

                        if line == "/quit" {
                            input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[Shutting down mob]\x1b[0m"));
                            break;
                        }

                        if line == "/status" {
                            let status = handle.status();
                            let members = handle.list_members().await;
                            let msg = format!("[Status: {status:?}, Members: {}]", members.len());
                            input::with_output(&output_tx, || raw_eprintln!("\x1b[2m{msg}\x1b[0m"));
                            continue;
                        }

                        if line == "/members" {
                            let members = handle.list_members().await;
                            input::with_output(&output_tx, || {
                                raw_eprintln!("\x1b[2m[Members ({}):]", members.len());
                                for m in &members {
                                    raw_eprintln!(
                                        "  {} (profile: {}, state: {:?}, wired_to: {:?})",
                                        m.meerkat_id, m.profile, m.state, m.wired_to
                                    );
                                }
                                eprint!("\x1b[0m");
                            });
                            continue;
                        }

                        if line == "/tasks" {
                            let events = handle.poll_events(0, 1000).await?;
                            input::with_output(&output_tx, || {
                                let mut task_count = 0;
                                for event in &events {
                                    match &event.kind {
                                        meerkat_mob::MobEventKind::TaskCreated { task_id, subject, .. } => {
                                            raw_eprintln!("  \x1b[33m[{task_id}]\x1b[0m {subject}");
                                            task_count += 1;
                                        }
                                        meerkat_mob::MobEventKind::TaskUpdated { task_id, status, owner } => {
                                            let owner_str = owner
                                                .as_ref()
                                                .map_or("unassigned", |o| o.as_ref());
                                            raw_eprintln!("  \x1b[33m[{task_id}]\x1b[0m status={status:?} owner={owner_str}");
                                        }
                                        _ => {}
                                    }
                                }
                                if task_count == 0 {
                                    raw_eprintln!("\x1b[2m[No tasks yet]\x1b[0m");
                                }
                            });
                            continue;
                        }

                        if let Some(rest) = line.strip_prefix("/send ") {
                            let mut parts = rest.splitn(2, ' ');
                            let (Some(target_str), Some(msg_str)) = (parts.next(), parts.next()) else {
                                input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[Usage: /send <agent> <message>]\x1b[0m"));
                                continue;
                            };
                            let target = MeerkatId::from(target_str);
                            let msg = msg_str.to_string();
                            match handle.send_message(target.clone(), msg).await {
                                Ok(_) => input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[Message sent to {target}]\x1b[0m")),
                                Err(e) => input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[Failed to send to {target}: {e}]\x1b[0m")),
                            }
                            continue;
                        }

                        // Default: send to orchestrator.
                        if let Some(ref orch) = orch_id {
                            match handle.send_message(orch.clone(), line).await {
                                Ok(_) => input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[Sent to {orch}]\x1b[0m")),
                                Err(e) => input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[Failed to send to orchestrator: {e}]\x1b[0m")),
                            }
                        } else {
                            input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[No orchestrator defined -- use /send <agent> <msg>]\x1b[0m"));
                        }
                    }
                    InputResult::Interrupt => {
                        input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[Ctrl+C -- shutting down]\x1b[0m"));
                        break;
                    }
                    InputResult::Eof => {
                        input_open = false;
                        input::with_output(&output_tx, || raw_eprintln!("\x1b[2m[input closed]\x1b[0m"));
                    }
                }
            }

            Some(event) = router_handle.event_rx.recv() => {
                let log_line = renderer.render(&event);
                if !log_line.is_empty() {
                    if let Err(e) = writeln!(log_file, "{log_line}") {
                        tracing::warn!("failed to write event log: {e}");
                    }
                }
            }

            _ = tokio::signal::ctrl_c() => {
                input::hide_box(&output_tx);
                raw_eprintln!("\r\n\x1b[2m[Ctrl+C -- shutting down]\x1b[0m");
                break;
            }
        }
    }

    // Don't retire agents — just stop the event router and exit.
    // Retiring writes MeerkatRetired events which would cause the next
    // resume to lose all agent conversation history (since reconcile_resume
    // archives "orphan" sessions and creates fresh ones). By leaving the
    // agents un-retired, resume reconnects to their existing sessions.
    input::hide_box(&output_tx);
    router_handle.cancel();
    raw_eprintln!(
        "\x1b[2m[Mob runner stopped. Sessions preserved in {}]\x1b[0m",
        state.root().display()
    );

    Ok(())
}
