//! Interactive mob loop: stdin + event streaming + Ctrl+C handling.
//!
//! Routes user input to the orchestrator and streams events from all agents.

use meerkat_mob::{MeerkatId, MobHandle};
use std::io::Write;
use tokio::sync::mpsc;

use crate::render::EventRenderer;
use crate::state::StateDir;

/// Find the orchestrator's meerkat ID from the mob definition.
fn orchestrator_id(handle: &MobHandle) -> Option<MeerkatId> {
    let def = handle.definition();
    def.orchestrator
        .as_ref()
        .map(|o| MeerkatId::from(o.profile.as_str()))
}

/// Run the interactive mob loop.
///
/// Accepts a shared stdin receiver (spawned once in main).
pub async fn run_mob_loop(
    handle: MobHandle,
    state: &StateDir,
    mut stdin_rx: mpsc::Receiver<String>,
) -> color_eyre::Result<()> {
    let orch_id = orchestrator_id(&handle);

    eprintln!("=== Mob Runner — Execution Phase ===");
    eprintln!();
    if let Some(ref id) = orch_id {
        eprintln!("Orchestrator: {id}");
    }
    eprintln!("Commands: /status, /members, /tasks, /send <agent> <msg>, /quit");
    eprintln!("Type a message to send to the orchestrator.");
    eprintln!();

    // Subscribe to all agent events via the event router.
    let mut router_handle = handle.subscribe_mob_events();
    let mut renderer = EventRenderer::new();

    // Open event log file for appending.
    let log_path = state.events_jsonl();
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let mut stdin_open = true;

    loop {
        tokio::select! {
            // User input from stdin (disabled once channel closes to avoid busy-spin).
            result = stdin_rx.recv(), if stdin_open => {
                let Some(input) = result else {
                    stdin_open = false;
                    eprintln!("[stdin closed]");
                    continue;
                };
                if input.is_empty() {
                    continue;
                }

                if input == "/quit" {
                    eprintln!("[Shutting down mob]");
                    break;
                }

                if input == "/status" {
                    let status = handle.status();
                    let members = handle.list_members().await;
                    eprintln!("[Status: {status:?}, Members: {}]", members.len());
                    continue;
                }

                if input == "/members" {
                    let members = handle.list_members().await;
                    eprintln!("[Members ({}):]", members.len());
                    for m in &members {
                        eprintln!(
                            "  {} (profile: {}, state: {:?}, wired_to: {:?})",
                            m.meerkat_id, m.profile, m.state, m.wired_to
                        );
                    }
                    continue;
                }

                if input == "/tasks" {
                    // Scan mob events for task state. This re-reads from cursor 0
                    // for simplicity — acceptable for an interactive command.
                    let events = handle.poll_events(0, 1000).await?;
                    let mut task_count = 0;
                    for event in &events {
                        match &event.kind {
                            meerkat_mob::MobEventKind::TaskCreated { task_id, subject, .. } => {
                                eprintln!("  [{task_id}] {subject}");
                                task_count += 1;
                            }
                            meerkat_mob::MobEventKind::TaskUpdated { task_id, status, owner } => {
                                let owner_str = owner
                                    .as_ref()
                                    .map_or("unassigned", |o| o.as_ref());
                                eprintln!("  [{task_id}] status={status:?} owner={owner_str}");
                            }
                            _ => {}
                        }
                    }
                    if task_count == 0 {
                        eprintln!("[No tasks yet]");
                    }
                    continue;
                }

                if let Some(rest) = input.strip_prefix("/send ") {
                    let mut parts = rest.splitn(2, ' ');
                    let (Some(target_str), Some(msg_str)) = (parts.next(), parts.next()) else {
                        eprintln!("[Usage: /send <agent> <message>]");
                        continue;
                    };
                    let target = MeerkatId::from(target_str);
                    let msg = msg_str.to_string();
                    match handle.send_message(target.clone(), msg).await {
                        Ok(_) => eprintln!("[Message sent to {target}]"),
                        Err(e) => eprintln!("[Failed to send to {target}: {e}]"),
                    }
                    continue;
                }

                // Default: send to orchestrator.
                if let Some(ref orch) = orch_id {
                    match handle.send_message(orch.clone(), input.clone()).await {
                        Ok(_) => {}
                        Err(e) => eprintln!("[Failed to send to orchestrator: {e}]"),
                    }
                } else {
                    eprintln!("[No orchestrator defined — use /send <agent> <msg>]");
                }
            }

            // Events from the mob event router.
            Some(event) = router_handle.event_rx.recv() => {
                let log_line = renderer.render(&event);
                if !log_line.is_empty() {
                    if let Err(e) = writeln!(log_file, "{log_line}") {
                        tracing::warn!("failed to write event log: {e}");
                    }
                }
            }

            // Ctrl+C handler.
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n[Ctrl+C — shutting down]");
                break;
            }
        }
    }

    // Clean up: stop the event router and retire all agents.
    router_handle.cancel();
    eprintln!("[Retiring all agents...]");
    if let Err(e) = handle.retire_all().await {
        eprintln!("[Warning: failed to retire agents: {e}]");
    }
    eprintln!("[Mob runner stopped. State saved in {}]", state.root().display());

    Ok(())
}
