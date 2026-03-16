//! # 035 — Interactive Mob Runner
//!
//! A conversational agent launcher: chat with a planning agent to design a task,
//! then deploy a mob of collaborating agents. Continue chatting with the
//! orchestrator while watching all agents work.
//!
//! ## Two phases
//!
//! 1. **Planning** — multi-turn chat with a planner agent that has shell + builtins.
//!    When ready, the planner outputs a ```toml mob definition.
//! 2. **Execution** — mob deployed from the definition, user chats with the
//!    orchestrator, events stream from all agents.
//!
//! ## Resume
//!
//! State persists in `.mob-runner/`. Re-run to resume a mob in progress.
//! Agent sessions are stored in a redb database so conversations survive restarts.
//!
//! ## Run
//!
//! ```bash
//! ANTHROPIC_API_KEY=... cargo run
//! # or
//! GEMINI_API_KEY=... cargo run
//! # or
//! OPENAI_API_KEY=... cargo run -- --model gpt-5.4
//! ```

mod deploy;
mod format;
mod input;
mod planner;
mod render;
mod state;

// Avoid Rust keyword collision with `loop`.
#[path = "loop.rs"]
mod event_loop;

use color_eyre::eyre::{self, WrapErr};
use input::{InputResult, OutputSignal};
use meerkat::{
    AgentFactory, Config, PersistenceBundle, RedbSessionStore, SessionStore,
    build_persistent_service,
};
use meerkat_mob::MobDefinition;
use state::StateDir;
use std::sync::Arc;
use tokio::sync::mpsc;

fn parse_args() -> (Option<String>, String) {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<String> = None;
    let mut state_dir = ".mob-runner".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() {
                    model = Some(args[i].clone());
                }
            }
            "--state-dir" => {
                i += 1;
                if i < args.len() {
                    state_dir = args[i].clone();
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: mob-runner [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --model <MODEL>       LLM model for the planner (auto-detected from API keys)");
                eprintln!("  --state-dir <DIR>     State directory (default: .mob-runner)");
                eprintln!("  --help                Show this help");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (model, state_dir)
}

/// Spawn the rich input editor thread. Returns:
/// - `mpsc::Receiver<InputResult>` for receiving user input events
/// - `std::sync::mpsc::Sender<OutputSignal>` for coordinating output with the box
fn spawn_input_editor(label: &str) -> (mpsc::Receiver<InputResult>, std::sync::mpsc::Sender<OutputSignal>) {
    let (line_tx, line_rx) = mpsc::channel::<InputResult>(16);
    let (output_tx, output_rx) = std::sync::mpsc::channel::<OutputSignal>();
    let label = label.to_string();
    std::thread::spawn(move || {
        input::run_input_thread(label, line_tx, output_rx);
    });
    (line_rx, output_tx)
}

/// Open a persistent session service backed by redb.
fn open_persistent_service(
    state: &StateDir,
) -> color_eyre::Result<Arc<dyn meerkat_mob::MobSessionService>> {
    let factory = AgentFactory::new(state.sessions_dir())
        .builtins(true)
        .shell(true)
        .comms(true)
        .mob(true);
    let config = Config::default();

    let redb_path = state.sessions_redb();
    let session_store = Arc::new(
        RedbSessionStore::open(&redb_path)
            .wrap_err_with(|| format!("failed to open session store at {}", redb_path.display()))?,
    );
    let runtime_store = Arc::new(
        meerkat_runtime::store::RedbRuntimeStore::new(session_store.database())
            .wrap_err("failed to create runtime store")?,
    ) as Arc<dyn meerkat_runtime::RuntimeStore>;

    let persistence = PersistenceBundle::new(
        session_store as Arc<dyn SessionStore>,
        Some(runtime_store),
    );

    let service = build_persistent_service(factory, config, 64, persistence);
    Ok(Arc::new(service))
}

async fn run() -> color_eyre::Result<()> {
    let (model_override, state_dir_path) = parse_args();

    // Resolve model.
    let model = model_override
        .or_else(|| planner::pick_default_model())
        .ok_or_else(|| eyre::eyre!(
            "No API keys found. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY."
        ))?;

    eprintln!("Using model: {model}");

    // Set up state directory.
    let state = StateDir::new(&state_dir_path);
    state.ensure_dirs()?;

    // Open persistent session service (sessions survive restarts).
    let session_service = open_persistent_service(&state)?;

    // Rich input editor shared across both phases.
    let initial_label = if state.has_mob() { "mob" } else { "planner" };
    let (input_rx, output_tx) = spawn_input_editor(initial_label);

    if state.has_mob() {
        // Resume existing mob.
        let handle = deploy::resume_mob(session_service, &state).await?;
        event_loop::run_mob_loop(handle, &state, input_rx, output_tx).await?;
    } else {
        // Planning phase.
        let (mob_toml, input_rx, output_tx) = planner::run_planner(
            session_service.clone(), &model, &state, input_rx, output_tx,
        ).await?;

        // Save the definition for resume.
        std::fs::write(state.mob_toml(), &mob_toml)
            .wrap_err("failed to save mob definition")?;

        // Update input label for execution phase.
        let _ = output_tx.send(OutputSignal::UpdateLabel("mob".to_string()));

        // Parse and deploy.
        let definition = MobDefinition::from_toml(&mob_toml)
            .wrap_err("failed to parse mob definition")?;
        let handle = deploy::deploy_mob(definition, session_service, &state).await?;

        // Execution phase.
        event_loop::run_mob_loop(handle, &state, input_rx, output_tx).await?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    // Initialize tracing (controlled via RUST_LOG).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let result = run().await;

    // Always restore terminal before printing errors. disable_raw_mode is
    // idempotent — harmless if raw mode was never enabled or already disabled.
    let _ = crossterm::terminal::disable_raw_mode();
    let _ = crossterm::execute!(
        std::io::stderr(),
        crossterm::cursor::SetCursorStyle::DefaultUserShape
    );

    result
}
