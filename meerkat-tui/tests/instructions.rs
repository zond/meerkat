//! Instruction files (user-global SYSTEM.md / AGENTS.md, ancestor AGENTS.md,
//! AGENTS.local.md), the memory-notes index, and runtime `/remember` context
//! reach the LLM request.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use clap::Parser;
use futures::stream;
use meerkat_client::{LlmClient, LlmDoneOutcome, LlmError, LlmEvent, LlmRequest};
use meerkat_core::event::{AgentEvent, EventEnvelope};
use meerkat_core::{Message, StopReason};
use meerkat_tui::Args;
use meerkat_tui::bootstrap::instructions::{
    InstructionSources, SectionKind, apply_system_md, remembered_context,
};
use meerkat_tui::bootstrap::{self, Runtime};
use meerkat_tui::memory::{MemoryToolDispatcher, NoteStore, Scope};
use meerkat_tui::session::lifecycle::session_instructions;
use meerkat_tui::session::{SessionHandle, SessionOptions};
use parking_lot::Mutex;
use tokio::sync::mpsc;

type LlmStream<'a> = Pin<Box<dyn futures::Stream<Item = Result<LlmEvent, LlmError>> + Send + 'a>>;

/// One recorded request: the system prompt and every non-system message.
#[derive(Debug, Clone)]
struct Recorded {
    system: String,
    messages: Vec<String>,
}

/// Answers every call with "ok" and records the request.
#[derive(Default)]
struct RecordingLlmClient {
    requests: Mutex<Vec<Recorded>>,
}

impl RecordingLlmClient {
    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().clone()
    }
}

#[async_trait]
impl LlmClient for RecordingLlmClient {
    fn stream<'a>(&'a self, request: &'a LlmRequest) -> LlmStream<'a> {
        let mut system = String::new();
        let mut messages = Vec::new();
        for message in &request.messages {
            match message {
                Message::System(s) => system.push_str(&s.content),
                Message::User(u) => messages.push(u.text_content()),
                other => messages.push(format!("{other:?}")),
            }
        }
        self.requests.lock().push(Recorded { system, messages });
        Box::pin(stream::iter(vec![
            Ok(LlmEvent::TextDelta {
                delta: "ok".to_string(),
                meta: None,
            }),
            Ok(LlmEvent::Done {
                outcome: LlmDoneOutcome::Success {
                    stop_reason: StopReason::EndTurn,
                },
            }),
        ]))
    }

    fn provider(&self) -> &'static str {
        "scripted"
    }

    async fn health_check(&self) -> Result<(), LlmError> {
        Ok(())
    }
}

struct Fixture {
    _tmp: tempfile::TempDir,
    git_root: PathBuf,
    workspace: PathBuf,
    home: PathBuf,
}

fn write(path: &Path, content: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

/// `<tmp>/repo/.git/`, `<tmp>/repo/AGENTS.md`, workspace `<tmp>/repo/ws` with
/// `AGENTS.md` and `AGENTS.local.md`, and `<tmp>/home/.config/rkat-tui/`
/// with `SYSTEM.md` and `AGENTS.md`.
fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(tmp.path()).unwrap();
    let git_root = base.join("repo");
    let workspace = git_root.join("ws");
    let home = base.join("home");
    std::fs::create_dir_all(git_root.join(".git")).unwrap();
    write(&git_root.join("AGENTS.md"), "ANCESTOR-RULES apply here.\n");
    write(
        &workspace.join("AGENTS.md"),
        "WORKSPACE-RULES apply here.\n",
    );
    write(
        &workspace.join("AGENTS.local.md"),
        "LOCAL-RULES apply here.\n",
    );
    let global = home.join(".config/rkat-tui");
    write(&global.join("SYSTEM.md"), "You are TESTPERSONA.\n");
    write(&global.join("AGENTS.md"), "GLOBAL-RULES apply here.\n");
    Fixture {
        _tmp: tmp,
        git_root,
        workspace,
        home,
    }
}

async fn runtime(f: &Fixture, llm: Arc<RecordingLlmClient>) -> Runtime {
    let args = Args::try_parse_from([
        "rkat-tui",
        "--workspace",
        f.workspace.to_str().unwrap(),
        "--user-config-root",
        f.home.to_str().unwrap(),
        "--model",
        "claude-sonnet-4-6",
    ])
    .unwrap();
    let (scope, mut config) = bootstrap::prepare(&args).await.expect("prepare");
    let applied = apply_system_md(&mut config, scope.global.as_ref());
    assert_eq!(
        applied,
        Some(f.home.join(".config/rkat-tui/SYSTEM.md")),
        "SYSTEM.md should apply to a config without a prompt"
    );
    bootstrap::bootstrap_in(scope, config, &args, None, Some(llm))
        .await
        .expect("bootstrap")
}

async fn session(rt: &Runtime) -> SessionHandle {
    let sections = InstructionSources::discover(&rt.scope.context_root, rt.scope.global.as_ref());
    let kinds: Vec<_> = sections.iter().map(|s| s.kind).collect();
    assert_eq!(
        kinds,
        [
            SectionKind::UserGlobal,
            SectionKind::Ancestor,
            SectionKind::Local
        ]
    );
    SessionHandle::create(
        rt,
        SessionOptions {
            additional_instructions: InstructionSources::render_all(&sections),
            ..SessionOptions::default()
        },
    )
    .await
    .expect("create session")
}

async fn run_turn(handle: &SessionHandle, prompt: &str) -> String {
    let (tx, mut rx) = mpsc::channel::<EventEnvelope<AgentEvent>>(256);
    let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let result = handle
        .spawn_turn(prompt.to_string(), tx)
        .await
        .expect("turn");
    drain.abort();
    result.text
}

fn position(haystack: &str, needle: &str) -> usize {
    haystack
        .find(needle)
        .unwrap_or_else(|| panic!("system prompt lacks {needle:?}:\n{haystack}"))
}

#[tokio::test]
async fn instruction_files_reach_the_system_prompt() {
    let f = fixture();
    let llm = Arc::new(RecordingLlmClient::default());
    let rt = runtime(&f, Arc::clone(&llm)).await;
    let handle = session(&rt).await;
    assert_eq!(run_turn(&handle, "hi").await, "ok");

    let requests = llm.requests();
    let system = &requests.first().expect("one request").system;
    assert!(system.contains("TESTPERSONA"), "{system}");
    assert!(
        !system.contains("You are an autonomous agent"),
        "default prompt should be replaced by SYSTEM.md:\n{system}"
    );
    assert!(system.contains("WORKSPACE-RULES"), "{system}");

    let global = position(
        system,
        &format!(
            "# Instructions (from {})",
            f.home.join(".config/rkat-tui/AGENTS.md").display()
        ),
    );
    let ancestor = position(
        system,
        &format!(
            "# Instructions (from {})",
            f.git_root.join("AGENTS.md").display()
        ),
    );
    let local = position(
        system,
        &format!(
            "# Instructions (from {})",
            f.workspace.join("AGENTS.local.md").display()
        ),
    );
    assert!(global < ancestor && ancestor < local, "{system}");
    assert!(system.contains("GLOBAL-RULES"));
    assert!(system.contains("ANCESTOR-RULES"));
    assert!(system.contains("LOCAL-RULES"));
    // The workspace AGENTS.md is loaded once, by Meerkat, not as a section.
    assert_eq!(system.matches("WORKSPACE-RULES").count(), 1, "{system}");
    assert!(
        !system.contains(&format!(
            "# Instructions (from {})",
            f.workspace.join("AGENTS.md").display()
        )),
        "{system}"
    );
}

#[tokio::test]
async fn appended_system_context_reaches_the_next_request() {
    let f = fixture();
    let llm = Arc::new(RecordingLlmClient::default());
    let rt = runtime(&f, Arc::clone(&llm)).await;
    let handle = session(&rt).await;

    handle
        .append_system_context(
            remembered_context("REMEMBER-ME always run clippy"),
            "remember-1".to_string(),
        )
        .await
        .expect("append");
    assert_eq!(run_turn(&handle, "hi").await, "ok");
    let requests = llm.requests();
    let first = requests.first().expect("one request");
    let all = format!("{}\n{}", first.system, first.messages.join("\n"));
    assert!(all.contains("# Remembered instruction"), "{all}");
    assert_eq!(
        all.matches("REMEMBER-ME always run clippy").count(),
        1,
        "{all}"
    );
    assert!(
        first.system.contains("source: rkat-tui"),
        "{}",
        first.system
    );

    // Still present on the following turn (it is part of the session's
    // system prompt, not a one-shot staged append).
    assert_eq!(run_turn(&handle, "again").await, "ok");
    let requests = llm.requests();
    let second = requests.last().unwrap();
    assert!(second.system.contains("REMEMBER-ME"), "{}", second.system);
}

#[tokio::test]
async fn memory_notes_index_follows_the_instruction_files() {
    let f = fixture();
    let llm = Arc::new(RecordingLlmClient::default());
    let rt = runtime(&f, Arc::clone(&llm)).await;
    let global = rt.scope.global.clone().expect("global config dir");
    let user = NoteStore::open(global.notes_dir(), Scope::User);
    let workspace = NoteStore::open(
        rt.scope.context_root.join(".rkat/tui/notes"),
        Scope::Workspace,
    );
    user.save("style", "USER-NOTE prefer early returns\nmore detail")
        .unwrap();
    workspace.save("build", "WS-NOTE run make ci").unwrap();
    let notes = MemoryToolDispatcher::new(Some(user), workspace);

    let sections = session_instructions(&rt.scope.context_root, Some(&global), Some(&notes));
    let handle = SessionHandle::create(
        &rt,
        SessionOptions {
            additional_instructions: sections,
            ..SessionOptions::default()
        },
    )
    .await
    .expect("create session");
    assert_eq!(run_turn(&handle, "hi").await, "ok");

    let requests = llm.requests();
    let system = &requests.first().expect("one request").system;
    let local = position(
        system,
        &format!(
            "# Instructions (from {})",
            f.workspace.join("AGENTS.local.md").display()
        ),
    );
    let memory = position(system, "# Memory\n\nNotes you saved earlier.");
    let user_at = position(system, "## User notes (");
    let ws_at = position(system, "## Workspace notes (");
    assert!(
        local < memory && memory < user_at && user_at < ws_at,
        "{system}"
    );
    // Only the first line of each note is in the index; bodies stay on disk.
    assert!(
        system.contains("- style: USER-NOTE prefer early returns\n"),
        "{system}"
    );
    assert!(
        system.contains("- build: WS-NOTE run make ci\n"),
        "{system}"
    );
    assert!(!system.contains("more detail"), "{system}");
}

#[tokio::test]
async fn no_memory_section_without_notes() {
    let f = fixture();
    let llm = Arc::new(RecordingLlmClient::default());
    let rt = runtime(&f, Arc::clone(&llm)).await;
    let global = rt.scope.global.clone().expect("global config dir");
    let notes = MemoryToolDispatcher::new(
        Some(NoteStore::open(global.notes_dir(), Scope::User)),
        NoteStore::open(
            rt.scope.context_root.join(".rkat/tui/notes"),
            Scope::Workspace,
        ),
    );
    let sections = session_instructions(&rt.scope.context_root, Some(&global), Some(&notes));
    assert!(
        sections.iter().all(|s| !s.starts_with("# Memory")),
        "{sections:?}"
    );
}
