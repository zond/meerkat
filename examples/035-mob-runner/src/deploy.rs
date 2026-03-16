//! Mob deployment and resume.
//!
//! Creates or resumes a mob from a parsed definition using `MobBuilder`.

use color_eyre::eyre::WrapErr;
use meerkat_mob::{
    MeerkatId, MobBuilder, MobDefinition, MobHandle, MobSessionService, MobStorage,
    SpawnMemberSpec,
};
use std::sync::Arc;

use crate::state::StateDir;

/// Deploy a fresh mob from a definition.
pub async fn deploy_mob(
    definition: MobDefinition,
    session_service: Arc<dyn MobSessionService>,
    state: &StateDir,
) -> color_eyre::Result<MobHandle> {
    let storage = MobStorage::redb(state.mob_redb())
        .wrap_err("failed to open mob storage")?;

    // Extract data we need for spawning before moving definition into builder.
    let orchestrator_profile = definition
        .orchestrator
        .as_ref()
        .map(|o| o.profile.clone());
    let profile_names: Vec<_> = definition.profiles.keys().cloned().collect();
    let expected_count = definition.profiles.len();

    let handle = MobBuilder::new(definition, storage)
        .with_session_service(session_service)
        .create()
        .await
        .wrap_err("failed to create mob")?;

    eprintln!(
        "[Mob '{}' created (status: {:?})]",
        handle.mob_id(),
        handle.status()
    );

    // Spawn one agent per profile.
    // Convention: meerkat_id = profile name (1:1 mapping in this example).
    for name in &profile_names {
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
                eprintln!("[Spawned {name}/{meerkat_id}: {member_ref:?}]");
            }
            Err(e) => {
                if is_orchestrator {
                    // Orchestrator is required — fail the deployment.
                    return Err(e).wrap_err(format!(
                        "failed to spawn orchestrator {name}/{meerkat_id}"
                    ));
                }
                // Best-effort for workers: log and continue with partial roster.
                eprintln!("[Failed to spawn {name}/{meerkat_id}: {e}]");
            }
        }
    }

    // Wait for roster to be ready.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let members = handle.list_members().await;
        if members.len() == expected_count {
            break;
        }
        if tokio::time::Instant::now() > deadline {
            eprintln!(
                "[Warning: only {}/{} members ready after timeout]",
                members.len(),
                expected_count
            );
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    let members = handle.list_members().await;
    eprintln!("\n[Roster ({} members):]", members.len());
    for m in &members {
        eprintln!(
            "  {} (profile: {}, wired_to: {:?})",
            m.meerkat_id, m.profile, m.wired_to
        );
    }
    eprintln!();

    Ok(handle)
}

/// Resume an existing mob from persistent storage.
pub async fn resume_mob(
    session_service: Arc<dyn MobSessionService>,
    state: &StateDir,
) -> color_eyre::Result<MobHandle> {
    eprintln!("[Resuming mob from {}]", state.mob_redb().display());

    let storage = MobStorage::redb(state.mob_redb())
        .wrap_err("failed to open mob storage for resume")?;

    let handle = MobBuilder::for_resume(storage)
        .with_session_service(session_service)
        .resume()
        .await
        .wrap_err("failed to resume mob")?;

    eprintln!(
        "[Mob '{}' resumed (status: {:?})]",
        handle.mob_id(),
        handle.status()
    );

    let members = handle.list_members().await;
    eprintln!("[Roster ({} members):]", members.len());
    for m in &members {
        eprintln!(
            "  {} (profile: {}, wired_to: {:?})",
            m.meerkat_id, m.profile, m.wired_to
        );
    }
    eprintln!();

    Ok(handle)
}
