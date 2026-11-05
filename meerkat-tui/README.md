# meerkat-tui (`rkat-tui`)

An interactive terminal coding harness built on the Meerkat agent engine. It
embeds the agent in-process (no RPC server) and gates every tool call through an
**auto-mode** approval model modelled on Claude Code's: routine actions run
without interruption, risky ones are judged by a reasoning-blind classifier, and
you are only asked after repeated blocks.

```bash
cargo run -p meerkat-tui -- "fix the failing test"
rkat-tui -c                   # continue the latest session of this workspace
rkat-tui --resume ~2
rkat-tui --model gpt-5.2 --classifier-model gpt-5.2
rkat-tui --no-auto            # tier-3 actions are denied instead of classified
```

## How approval works

Every tool call passes through a `PreToolExecution` hook. First match wins:

| Tier | What | Review |
|---|---|---|
| 0 | A user **deny** rule matches | denied |
| 1 | Read-only tools (`read_file`, `glob`, `grep`, task tools, …), read-only shell commands (`cat`, `ls`, `rg`, `git status/diff/log`, `cargo check/test`, …), narrow user **allow** rules | allowed |
| 2 | `write_file` / `edit_file` / `apply_patch` where every target is inside the project and not protected (`.rkat/`, `.git/`, `.env*`, …) | allowed |
| 3 | Everything else: other shell commands, MCP tools, sub-agent spawns, out-of-project paths, protected paths | classifier |

The classifier is a separate LLM call on the session's provider. It sees only the
user's messages and the agent's prior tool calls, never the agent's reasoning or
tool outputs, so it cannot be argued or prompt-injected past. Stage 1 is a
one-word ALLOW/BLOCK told to err on the side of blocking; stage 2 re-evaluates
flagged actions with reasoning. A block is returned to the agent as a tool error
with the reason and an instruction to find a safer path.

After **3 consecutive** or **20 total** classifier blocks, auto mode pauses and
each tier-3 action opens a dialog: allow once, allow and add a narrow rule, deny,
or stop the turn. `/auto resume` resumes.

Sub-agents spawned by the agent inherit the same hooks, so they are gated too.

## Rules

Rules live in TOML, layered (later settings win): the user-global
`~/.config/rkat-tui/tui.toml` (see below), then `<realm>/tui.toml` (written by
the UI), then `<project>/.rkat/tui.toml`. The old global location
`~/.rkat/tui.toml` is no longer read; rkat-tui shows a notice if it still exists.
The project file may only
add deny rules and `[auto]` settings; broad allow rules (blanket shell,
interpreters, package-manager run commands) are dropped when auto mode starts.

```toml
[auto]
enabled = true
classifier_model = "claude-sonnet-4-6"   # optional, per-provider default otherwise
trust_project_builds = true              # cargo build/test/check are tier 1

[[allow]]
tool = "shell"
arg = "cargo nextest run*"

[[deny]]
tool = "shell"
arg = "git push*"
```

## Keys and commands

Enter sends; Alt+Enter / Ctrl+J insert a newline; Ctrl-C interrupts a running
turn (twice to quit when idle); Ctrl-O toggles the last reasoning block; Ctrl-R
searches prompt history (workspace, then global; Ctrl-R again for older
matches, Enter/arrows accept into the editor, Esc cancels); Ctrl-X Ctrl-S
interrupts the running turn and sends the input; PgUp/PgDn and Shift+↑/↓ scroll.

While the agent is working, Enter queues the message and delivers it at the
agent's next step (between tool calls). Esc recalls every queued message into
the editor (and interrupts the turn when nothing is queued); ↑ on an empty
editor takes back only the newest queued message for editing. When a message
reached the model too late to be answered, rkat-tui asks for an answer once on
its own; if it still goes unanswered you are told, and Enter on an empty editor
asks again.

Ctrl-D on an empty editor (or `/quit`) quits at once when idle. While a turn is
running or messages are queued it warns first; pressing it again within 3
seconds interrupts the turn, saves the session, and appends every message the
model never received to `<workspace>/.rkat/tui/tui_unsent.txt` (the next start
says so).

`/help`, `/quit`, `/clear`, `/resume [ref]`, `/rules [list|add allow|deny <tool> [pattern]|rm <n>]`,
`/auto [on|off|resume]`, `/model <id>`, `/remember <text>`, `/compact`.
`/clear`, `/resume`, `/model <id>` and `/compact` work only while idle (no turn
running, nothing queued); the transcript is replaced only once the new session
is live.

A single-line prompt starting with `#` (`# use cargo nextest`) is the same as
`/remember`; `##…`, multi-line input and `\#…` are sent as prompts.

The status bar shows a context gauge, `ctx 42% (84k/200k)`, yellow from 70%
and red from 90% (`~` marks an estimate, e.g. right after `/resume` or
`/compact`, or a window borrowed from a related model or the provider default
for a model the catalog does not know).

## Instructions and memory

The system prompt is built from, in order:

1. the base prompt: the user-global `SYSTEM.md` replaces Meerkat's default
   when it is non-empty and the realm config sets no `system_prompt` /
   `system_prompt_file` of its own;
2. the workspace `AGENTS.md` (or `.rkat/AGENTS.md`), loaded by Meerkat;
3. the coding persona, then the user-global `AGENTS.md`, every `AGENTS.md`
   from the git root down to the workspace's parent directory, and the
   workspace `AGENTS.local.md` (or `.rkat/AGENTS.local.md`, meant to stay
   uncommitted), each under `# Instructions (from <path>)`. Each file is
   capped at the Meerkat `AGENTS.md` limit and all of them together at 96 KiB;
4. the memory-notes index (below), when there are notes.

Instruction files and notes are re-read whenever a session is built, so
`/clear`, `/resume` and `/compact` pick up edits.

**`/remember <text>`** (or `#text`) appends `- <text>` to the workspace
`AGENTS.md` (created if missing) and adds it to the live session's system
prompt between turns, so it applies at once and to every later session.

**Memory notes** are small Markdown files the agent keeps across sessions with
`memory_save`, `memory_read`, `memory_list` and `memory_forget`. Workspace
notes (facts about this project) live in `<workspace>/.rkat/tui/notes/`, user
notes (personal preferences that follow you into every project) in
`<config-dir>/notes/`; each directory has a generated `MEMORY.md` index. The
system prompt carries only a capped index (name and first line of each note);
the agent reads full notes on demand. Reads are always allowed and workspace
writes are treated like in-project edits; user-note writes go to the
classifier, since they reach every project's prompt.

**`/compact`** summarizes the conversation with the session's model, keeps the
last two turns verbatim, and continues the same session (same id) from the
summary. The messages the summary replaced are indexed into the realm's
semantic memory, so the agent can still find them with `memory_search` in this
and later sessions. The semantic index is single-writer: a second `rkat-tui`
on the same workspace runs without `memory_search` and says so at start-up.

## Where things live

All state for a workspace lives inside it, under `<workspace>/.rkat/tui/`:
sessions, the realm config, the rules file written by the UI, input history and
logs. Nothing is keyed by the absolute path, so you can rename or move the
directory and `rkat-tui -c` still continues the latest session. A `.gitignore`
is written into that directory so it is never committed.

The workspace is `--workspace <dir>` (`-w`), else the nearest ancestor of the
current directory that already has `.rkat/tui/` (the search stops at the git
root and never picks your home directory), else the current directory. Running
from your home directory requires an explicit `--workspace`.

`--realm`, `--isolated` and `--state-root` opt into the shared realm model that
`rkat` uses (`$XDG_DATA_HOME/meerkat/realms/<realm>/`) instead.

User-global files live in one config directory: `--config-dir <dir>`, else
`$XDG_CONFIG_HOME/rkat-tui`, else `~/.config/rkat-tui` (on Windows the platform
config dir, `%APPDATA%\rkat-tui`). With `--user-config-root <root>` and no
`--config-dir` it is `<root>/.config/rkat-tui`. It holds `tui.toml` (global
rules), `AGENTS.md`, `SYSTEM.md`, `notes/` and `history`.

Project conventions are read from `<project>/AGENTS.md` and `AGENTS.local.md`
(plus ancestor `AGENTS.md` files up to the git root), `.rkat/mcp.toml` (MCP
servers), `.rkat/config.toml` (hooks), and `.rkat/tui.toml` (deny rules).
Workspace memory notes live in `.rkat/tui/notes/` and the semantic index
behind `memory_search` in the realm's `memory/` directory.

## Testing

```bash
cargo test -p meerkat-tui                                   # unit + integration-fast
cargo test -p meerkat-tui --test e2e_classifier -- --ignored  # live classifier, needs provider keys
```
