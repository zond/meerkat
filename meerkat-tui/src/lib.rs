//! `rkat-tui`: an interactive terminal coding harness built on the Meerkat agent engine.
//!
//! The binary embeds the agent in-process and gates every tool call through an
//! auto-mode approval model: read-only and in-project edits pass without review,
//! everything else is judged by a reasoning-blind classifier, and a human is only
//! consulted after repeated blocks.
//!
//! Module map:
//! - [`cli`] — command-line arguments
//! - [`paths`] — project-root path resolution and protected-path policy (shared)
//! - [`bootstrap`] — realm/config/factory/service construction
//! - [`endpoints`] — LLM endpoints (built-in providers, `[endpoints.<name>]`) and `endpoint:model` references
//! - [`overlay`] — the `--set`/`--allow`/… command-line layer over `tui.toml`
//! - [`session`] — session handle: create, resume, turns, interrupt, prompt queue
//! - [`tools`] — in-crate file tools (`read_file`, `write_file`, `edit_file`, `glob`, `grep`)
//! - [`approval`] — auto-mode approval engine (tiers, rules, shell analysis, classifier, escalation)
//! - [`ui`] — ratatui application

pub mod approval;
pub mod bootstrap;
pub mod cli;
pub mod delivery;
pub mod endpoints;
pub mod error;
pub mod memory;
pub mod models;
pub mod overlay;
pub mod paths;
pub mod runner;
pub mod session;
pub mod tools;
pub mod ui;

pub use cli::Args;
pub use error::TuiError;

/// Run the TUI with the given arguments.
///
/// # Errors
///
/// Returns a [`TuiError`] when bootstrap fails or the terminal cannot be driven.
pub async fn run(args: Args) -> Result<(), TuiError> {
    runner::run(args).await
}
