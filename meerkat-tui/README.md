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

Rules live in TOML, layered: `<realm>/tui.toml` (written by the UI), then
`~/.rkat/tui.toml`, then `<project>/.rkat/tui.toml`. The project file may only
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
turn (twice to quit when idle); Ctrl-D quits; Ctrl-R toggles the last reasoning
block; PgUp/PgDn and Shift+↑/↓ scroll.

`/help`, `/quit`, `/clear`, `/resume [ref]`, `/rules [list|add allow|deny <tool> [pattern]|rm <n>]`,
`/auto [on|off|resume]`, `/model <id>`.

## Where things live

All state for a workspace lives inside it, under `<workspace>/.rkat/tui/`:
sessions, the realm config, the rules file written by the UI, input history and
logs. Nothing is keyed by the absolute path, so you can rename or move the
directory and `rkat-tui -c` still continues the latest session. A `.gitignore`
is written into that directory so it is never committed.

The workspace is `--workspace <dir>` (`-w`), else the nearest ancestor of the
current directory that already has `.rkat/tui/`, else the current directory.

`--realm`, `--isolated` and `--state-root` opt into the shared realm model that
`rkat` uses (`$XDG_DATA_HOME/meerkat/realms/<realm>/`) instead.

Project conventions are read from `<project>/AGENTS.md`, `.rkat/mcp.toml`
(MCP servers), `.rkat/config.toml` (hooks), and `.rkat/tui.toml` (deny rules).

## Testing

```bash
cargo test -p meerkat-tui                                   # unit + integration-fast
cargo test -p meerkat-tui --test e2e_classifier -- --ignored  # live classifier, needs provider keys
```
