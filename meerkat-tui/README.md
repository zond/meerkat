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
rkat-tui --model gpt-x        # bare ids are looked up in every endpoint's model list
rkat-tui --endpoint-def litellm=openai=http://localhost:4000,key_env=LITELLM_KEY --model litellm:gpt-5.2
rkat-tui --allow 'shell=cargo nextest run*' --set auto.compact=auto
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
the UI), then the command line (see the configuration reference), then
`<project>/.rkat/tui.toml`. The old global location
`~/.rkat/tui.toml` is no longer read; rkat-tui shows a notice if it still exists.
The project file may only
add deny rules and `[auto]` settings; broad allow rules (blanket shell,
interpreters, package-manager run commands) are dropped when auto mode starts.

```toml
[auto]
enabled = true
classifier_model = "claude-sonnet-4-6"   # optional, per-provider default otherwise
trust_project_builds = true              # cargo build/test/check are tier 1
compact = "warn"                         # or "auto"; see "Context and compaction"
compact_warn_percent = 75
compact_auto_percent = 85

[[allow]]
tool = "shell"
arg = "cargo nextest run*"

[[deny]]
tool = "shell"
arg = "git push*"
```

## Configuration reference (`tui.toml`)

Every key can be set in any user layer (`<config-dir>/tui.toml`,
`<realm>/tui.toml`) or for one run on the command line with
`--set <dotted.key>=<value>`. The value is TOML (`true`, `85`, `"x"`,
`[{tool = "shell"}]`); anything that does not parse as TOML is a string, so
`--set auto.compact=auto` needs no quotes. Command-line settings form a layer
after the global and realm files and before the project file: tables merge,
scalars replace, arrays **append** (`--set 'allow=[{tool="shell",arg="cargo nextest run*"}]'`
adds a rule) and endpoints merge by name (`--set endpoints.litellm.base_url=http://host:4000`
changes only the URL). The project file's deny rules still apply, and it may
still lower `trust_project_builds`; its other `[auto]` keys apply only where the
command line set nothing. A bad `--set` (unknown key, wrong type, invalid
pattern or dialect) stops start-up and names the argument. Command-line rules
show as `(cli)` in `/rules list` and are never written to a file.

| Key | Flag | Default |
|---|---|---|
| `auto.enabled` | `--no-auto` (= `false`) | `true` |
| `auto.classifier_model` | `--classifier-model <id>` | per-dialect default of the start-up endpoint |
| `auto.trust_project_builds` | `--trust-project-builds` / `--no-trust-project-builds` | `false` |
| `auto.compact` (`"warn"`/`"auto"`) | `--auto-compact` / `--no-auto-compact` | `"warn"` |
| `auto.compact_warn_percent` | `--set auto.compact_warn_percent=N` | `75` |
| `auto.compact_auto_percent` | `--set auto.compact_auto_percent=N` | `85` |
| `[[allow]]` `tool`, `arg` (glob) or `literal` | `--allow <tool>[=<glob>]` (repeatable) | none |
| `[[deny]]` `tool`, `arg` (glob) or `literal` | `--deny <tool>[=<glob>]` (repeatable) | none |
| `endpoints.<name>.dialect` (`anthropic`/`openai`/`gemini`) | `--endpoint-def <name>=<dialect>=<url>[,key_env=VAR]` | required |
| `endpoints.<name>.base_url` | (same) | the dialect's public API |
| `endpoints.<name>.api_key_env` | (same, `key_env=`) | — (preferred) |
| `endpoints.<name>.api_key` | `--set endpoints.<name>.api_key=…` | — (plain text; warned about) |
| `endpoints.<name>.priority` | `--set endpoints.<name>.priority=N` | `0` |
| anything | `--set <dotted.key>=<value>` (repeatable) | |

Other flags: `--model`/`-m <[endpoint:]id>`, `--endpoint <name>` (endpoint
for a bare `--model` and the default; `--provider <name>` is the same for the
built-in ones), `--resume <ref>`, `-c`/`--continue`, `-w`/`--workspace`,
`--config-dir`, `--user-config-root`, `--realm`, `--isolated`,
`--realm-backend`, `--state-root`, `--mouse`, `--wait-for-mcp`. `--auto-compact`
conflicts with `--no-auto-compact`, `--trust-project-builds` with
`--no-trust-project-builds`, `--endpoint` with `--provider`.

## Models and endpoints

An **endpoint** is a named place to send requests in one of Meerkat's client
dialects: `anthropic` (Messages API), `openai` (the Responses API,
`/v1/responses`; also LiteLLM and other OpenAI-compatible proxies that speak
it) or `gemini`. The built-in endpoints `anthropic`, `openai` and `gemini`
exist when the realm config (`providers.api_keys`, `providers.base_urls`) or
the provider's environment variable has a key. More come from `tui.toml`:

```toml
[endpoints.litellm]
dialect = "openai"
base_url = "http://localhost:4000"
api_key_env = "LITELLM_KEY"   # or api_key = "…" (plain text, warned about)
priority = 1                  # optional tie-break, default 0
```

Names use letters, digits, `_` and `-` and may not be a built-in name.
Endpoints with an unknown or missing dialect or no key are skipped with a
notice. The project file may add endpoints but not change yours, and may not
read keys from the environment (`api_key_env`), so a cloned repository cannot
point your key at its own server; a git-tracked realm file contributes none.

**Model references** are `endpoint:id` (explicit; `:` separates, ids may
contain dots and further colons) or a bare id, resolved by lookup in the
fetched model lists:

1. exactly one endpoint lists the id → that endpoint;
2. several list it → the current endpoint, else the built-in endpoint of the
   id's dialect, else the highest `priority`, else the first configured
   (built-ins first, then by name);
3. none lists it (new id, or the fetch failed) → the id's prefix decides:
   `claude-*` → anthropic, `gpt-*`/`chatgpt-*`/`o<digit>*` → openai,
   `gemini-*` → gemini — the built-in endpoint of that dialect, else the only
   endpoint of that dialect; else the only configured endpoint; else an error
   listing the endpoints.

At start-up `--model <bare id>` with several endpoints fetches every list first
(in parallel, 10 s each) and reuses the result for the picker; `--endpoint`
skips the lookup. Without `--model`: the realm config's `agent.model` when its
prefix picks an endpoint, else with a single endpoint its catalog default (or
the first model it lists, with a notice), else start-up fails listing the
endpoints.

`/model` opens a picker with one group per endpoint
(`── litellm (openai dialect, http://localhost:4000) ──`), the current
endpoint and model first (bold), each row `id  display name  200k ctx`; type to
filter on `endpoint:id` and display name, Enter switches. Every endpoint's
list is fetched in parallel in the background at start-up (10 s each) and
cached; `/model refresh` refetches. A failed endpoint shows its error under
its header. For OpenAI-dialect endpoints with a base URL, LiteLLM's
`/model/info` fills in context windows. `/model <ref>` switches directly
(fetching the lists first if a bare id needs them).

Switching swaps the live session's client between turns, also across
endpoints and dialects; `/clear`, `/compact` and `/resume` keep the endpoint.
Each session's endpoint and model are recorded in
`<realm>/tui_endpoints/<session-id>.toml` and restored by `-c`/`/resume`
(unless `--model`/`--endpoint` is given); when the endpoint no longer exists
it is inferred from the model id's prefix, with a notice. The approval
classifier always uses the start-up endpoint. Sub-agents spawned with
`agent_spawn` do not follow the endpoint: Meerkat builds their clients from
its own provider defaults and checks models against its curated allowlist.

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
`/auto [on|off|resume]`, `/model [[endpoint:]id|refresh]`, `/remember <text>`, `/compact`.
`/clear`, `/resume`, `/model` and `/compact` work only while idle (no turn
running, nothing queued); the transcript is replaced only once the new session
is live.

`/model` is described under [Models and endpoints](#models-and-endpoints).

A single-line prompt starting with `#` (`# use cargo nextest`) is the same as
`/remember`; `##…`, multi-line input and `\#…` are sent as prompts.

The status bar shows a context gauge, `ctx 42% (84k/200k)`, yellow from 70%
and red from 90% (`~` marks an estimate, e.g. right after `/resume` or
`/compact`, or a window borrowed from a related model or the provider default
for a model the catalog does not know). The window comes from the provider's
list-models endpoint when it reports one (Anthropic, Gemini, LiteLLM's
`/model/info`); otherwise (OpenAI, or the fetch failed) from the curated catalog.

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

### Context and compaction

Meerkat's built-in auto-compaction is switched off (its compactor keeps whole
user turns, which in agentic sessions often frees nothing); rkat-tui decides
instead, after every turn, from the context gauge. With `compact = "warn"`
(the default, `[auto]` in `tui.toml`, any layer including the project file)
it warns once when the context crosses `compact_warn_percent` (75%) and after
every turn from `compact_auto_percent` (85%) on, suggesting `/compact`. With
`compact = "auto"` (or `--auto-compact`; `--no-auto-compact` forces warnings)
it runs `/compact` itself at `compact_auto_percent`, once the turn is over and
nothing is queued; messages typed meanwhile wait for it. When a compaction frees
nothing it is not retried until usage grows another 10 points. For a model
whose window is unknown the thresholds are 120k (warn) and 150k (auto) tokens.
`compact_auto_percent` below `compact_warn_percent` is raised to it, with a
notice. The realm config's other `[compaction]` settings still apply to
`/compact` (which keeps at most two turns).

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
