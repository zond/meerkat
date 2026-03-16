//! Mob deployment and resume.
//!
//! Creates or resumes a mob from a parsed definition using `MobBuilder`.

use color_eyre::eyre::WrapErr;
use meerkat_mob::{
    MeerkatId, MobBuilder, MobDefinition, MobHandle, MobSessionService, MobStorage,
    SpawnMemberSpec,
};
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;

use crate::input::{self, OutputSignal};
use crate::raw_eprintln;
use crate::state::StateDir;

/// Spawn all agents defined in a mob definition on the given handle.
async fn spawn_all_members(
    handle: &MobHandle,
    definition: &MobDefinition,
    output_tx: &std_mpsc::Sender<OutputSignal>,
) -> color_eyre::Result<()> {
    let orchestrator_profile = definition
        .orchestrator
        .as_ref()
        .map(|o| o.profile.clone());

    for (name, _profile) in &definition.profiles {
        let meerkat_id = MeerkatId::from(name.as_str());
        let is_orchestrator = orchestrator_profile
            .as_ref()
            .is_some_and(|o| o == name);

        let initial_msg = if is_orchestrator {
            "You are the orchestrator of this mob. Wait for instructions from the user.".to_string()
        } else {
            format!(
                "You are a worker in this mob with profile '{name}'. Wait for instructions from the orchestrator."
            )
        };

        let spec = SpawnMemberSpec::new(name.clone(), meerkat_id.clone())
            .with_initial_message(initial_msg);

        match handle.spawn_spec(spec).await {
            Ok(member_ref) => {
                input::with_output(output_tx, || {
                    raw_eprintln!("[Spawned {name}/{meerkat_id}: {member_ref:?}]");
                });
            }
            Err(e) => {
                if is_orchestrator {
                    return Err(e).wrap_err(format!(
                        "failed to spawn orchestrator {name}/{meerkat_id}"
                    ));
                }
                input::with_output(output_tx, || {
                    raw_eprintln!("\x1b[33m[Failed to spawn {name}/{meerkat_id}: {e}]\x1b[0m");
                });
            }
        }
    }
    Ok(())
}

/// Wait for the roster to reach the expected member count.
async fn wait_for_roster(handle: &MobHandle, expected: usize) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let members = handle.list_members().await;
        if members.len() >= expected {
            break;
        }
        if tokio::time::Instant::now() > deadline {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

/// Print the current roster.
fn print_roster(members: &[meerkat_mob::RosterEntry], output_tx: &std_mpsc::Sender<OutputSignal>) {
    input::with_output(output_tx, || {
        raw_eprintln!("[Roster ({} members):]", members.len());
        for m in members {
            raw_eprintln!(
                "  {} (profile: {}, wired_to: {:?})",
                m.meerkat_id, m.profile, m.wired_to
            );
        }
        raw_eprintln!();
    });
}

/// Deploy a fresh mob from a definition.
pub async fn deploy_mob(
    definition: MobDefinition,
    session_service: Arc<dyn MobSessionService>,
    state: &StateDir,
    output_tx: &std_mpsc::Sender<OutputSignal>,
) -> color_eyre::Result<MobHandle> {
    let storage = MobStorage::redb(state.mob_redb())
        .wrap_err("failed to open mob storage")?;

    let expected_count = definition.profiles.len();

    let handle = MobBuilder::new(definition.clone(), storage)
        .with_session_service(session_service)
        .create()
        .await
        .wrap_err("failed to create mob")?;

    let mob_id = handle.mob_id().to_string();
    let status = format!("{:?}", handle.status());
    input::with_output(output_tx, || {
        raw_eprintln!("[Mob '{mob_id}' created (status: {status})]");
    });

    spawn_all_members(&handle, &definition, output_tx).await?;
    wait_for_roster(&handle, expected_count).await;

    let members = handle.list_members().await;
    if members.len() < expected_count {
        input::with_output(output_tx, || {
            raw_eprintln!(
                "\x1b[33m[Warning: only {}/{} members ready after timeout]\x1b[0m",
                members.len(),
                expected_count
            );
        });
    }
    print_roster(&members, output_tx);

    Ok(handle)
}

/// Resume an existing mob from persistent storage.
/// If the roster is empty after resume (e.g. agents were retired on shutdown),
/// re-spawns all agents from the mob definition.
pub async fn resume_mob(
    session_service: Arc<dyn MobSessionService>,
    state: &StateDir,
    output_tx: &std_mpsc::Sender<OutputSignal>,
) -> color_eyre::Result<MobHandle> {
    input::with_output(output_tx, || {
        raw_eprintln!("[Resuming mob from {}]", state.mob_redb().display());
    });

    let storage = MobStorage::redb(state.mob_redb())
        .wrap_err("failed to open mob storage for resume")?;

    let handle = MobBuilder::for_resume(storage)
        .with_session_service(session_service)
        .resume()
        .await
        .wrap_err("failed to resume mob")?;

    let mob_id = handle.mob_id().to_string();
    let status = format!("{:?}", handle.status());
    input::with_output(output_tx, || {
        raw_eprintln!("[Mob '{mob_id}' resumed (status: {status})]");
    });

    let members = handle.list_members().await;
    if members.is_empty() {
        // Roster is empty — agents were retired during shutdown.
        // Re-read the definition and re-spawn all agents.
        input::with_output(output_tx, || {
            raw_eprintln!("\x1b[33m[No members found — re-spawning from definition...]\x1b[0m");
        });

        let mob_toml = std::fs::read_to_string(state.mob_toml())
            .wrap_err("failed to read mob.toml for re-spawn")?;
        let definition = MobDefinition::from_toml(&mob_toml)
            .wrap_err("failed to parse mob.toml for re-spawn")?;

        spawn_all_members(&handle, &definition, output_tx).await?;
        wait_for_roster(&handle, definition.profiles.len()).await;

        let members = handle.list_members().await;
        print_roster(&members, output_tx);
    } else {
        print_roster(&members, output_tx);
    }

    Ok(handle)
}
