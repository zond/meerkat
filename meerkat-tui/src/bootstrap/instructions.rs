//! Instruction files beyond the workspace `AGENTS.md`.
//!
//! Meerkat itself loads the workspace `AGENTS.md` (or `.rkat/AGENTS.md`) into
//! the system prompt. The TUI adds, as `additional_instructions` sections after
//! the persona and in this order:
//!
//! 1. the user-global `AGENTS.md` in the [`GlobalDirs`] config directory,
//! 2. ancestor `AGENTS.md` files from the git root down to the workspace's
//!    parent directory,
//! 3. the workspace `AGENTS.local.md` (or `.rkat/AGENTS.local.md`).
//!
//! It also lets a user-global `SYSTEM.md` replace the base system prompt
//! ([`apply_system_md`]) and appends `/remember` lines to the workspace
//! `AGENTS.md` ([`append_remember_line`]).

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use meerkat_core::Config;
use meerkat_core::prompt::{AGENTS_MD_MAX_BYTES, find_project_agents_md_in};

use super::dirs::GlobalDirs;

/// Cap on the combined rendered size of all discovered sections.
pub const TOTAL_MAX_BYTES: usize = 96 * 1024;

/// File name of the workspace-local, uncommitted instruction file.
pub const LOCAL_FILE_NAME: &str = "AGENTS.local.md";

/// Where an instruction section came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// The user-global `AGENTS.md`.
    UserGlobal,
    /// An `AGENTS.md` in a directory between the git root and the workspace.
    Ancestor,
    /// The workspace `AGENTS.local.md`.
    Local,
}

/// One instruction file, ready to render into the system prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Where the file came from.
    pub kind: SectionKind,
    /// The file's path.
    pub path: PathBuf,
    /// The file's content (possibly truncated, trailing whitespace trimmed).
    pub body: String,
    /// Whether `body` was cut at [`AGENTS_MD_MAX_BYTES`].
    pub truncated: bool,
}

impl Section {
    /// `# Instructions (from <path>)`.
    #[must_use]
    pub fn heading(&self) -> String {
        format!("# Instructions (from {})", self.path.display())
    }

    /// Heading, body and a truncation marker when the body was cut.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = self.heading();
        out.push_str("\n\n");
        out.push_str(&self.body);
        if self.truncated {
            let _ = write!(out, "\n\n[truncated at {} KiB]", AGENTS_MD_MAX_BYTES / 1024);
        }
        out
    }
}

/// Heading of runtime system context added by `/remember` (the persona refers
/// to it).
pub const REMEMBERED_HEADING: &str = "# Remembered instruction";

/// The runtime system context text for a `/remember` line:
/// [`REMEMBERED_HEADING`], a blank line and `text` trimmed.
#[must_use]
pub fn remembered_context(text: &str) -> String {
    format!("{REMEMBERED_HEADING}\n\n{}", text.trim())
}

/// Discovery of the TUI's extra instruction files.
#[derive(Debug, Clone, Copy)]
pub struct InstructionSources;

impl InstructionSources {
    /// Discover instruction sections for `workspace`.
    ///
    /// Order: user-global `AGENTS.md`; ancestor `AGENTS.md` files from
    /// [`git_root`] down to the workspace's parent (the workspace's own
    /// `AGENTS.md` is loaded by Meerkat and excluded here); the workspace
    /// `AGENTS.local.md`. Empty and unreadable files are skipped; each file is
    /// capped at [`AGENTS_MD_MAX_BYTES`] and sections stop being added once the
    /// rendered total would exceed [`TOTAL_MAX_BYTES`].
    #[must_use]
    pub fn discover(workspace: &Path, global: Option<&GlobalDirs>) -> Vec<Section> {
        let mut candidates: Vec<(SectionKind, PathBuf)> = Vec::new();

        if let Some(global) = global {
            let path = global.agents_md();
            if path.is_file() {
                candidates.push((SectionKind::UserGlobal, path));
            }
        }

        if let Some(root) = git_root(workspace) {
            let ancestors: Vec<&Path> = workspace
                .ancestors()
                .skip(1)
                .take_while(|dir| dir.starts_with(&root))
                .collect();
            for dir in ancestors.into_iter().rev() {
                if let Some(path) = find_project_agents_md_in(dir) {
                    candidates.push((SectionKind::Ancestor, path));
                }
            }
        }

        let local = [
            workspace.join(LOCAL_FILE_NAME),
            workspace.join(".rkat").join(LOCAL_FILE_NAME),
        ]
        .into_iter()
        .find(|p| p.is_file());
        if let Some(path) = local {
            candidates.push((SectionKind::Local, path));
        }

        let mut sections = Vec::with_capacity(candidates.len());
        let mut total = 0usize;
        for (kind, path) in candidates {
            let Some(section) = load_section(kind, path) else {
                continue;
            };
            let size = section.render().len();
            if total + size > TOTAL_MAX_BYTES {
                tracing::warn!(
                    path = %section.path.display(),
                    limit = TOTAL_MAX_BYTES,
                    "instruction files exceed the total size cap; skipping this and later files"
                );
                break;
            }
            total += size;
            sections.push(section);
        }
        sections
    }

    /// One rendered string per section, for
    /// `SessionOptions.additional_instructions`.
    #[must_use]
    pub fn render_all(sections: &[Section]) -> Vec<String> {
        sections.iter().map(Section::render).collect()
    }
}

/// Read one instruction file; `None` when empty or unreadable.
fn load_section(kind: SectionKind, path: PathBuf) -> Option<Section> {
    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(err) => {
            tracing::warn!(path = %path.display(), %err, "cannot read instruction file; skipping");
            return None;
        }
    };
    let trimmed = content.trim_end();
    if trimmed.trim_start().is_empty() {
        return None;
    }
    let (body, truncated) = if trimmed.len() > AGENTS_MD_MAX_BYTES {
        let mut cut = AGENTS_MD_MAX_BYTES;
        while !trimmed.is_char_boundary(cut) {
            cut -= 1;
        }
        (trimmed[..cut].to_string(), true)
    } else {
        (trimmed.to_string(), false)
    };
    Some(Section {
        kind,
        path,
        body,
        truncated,
    })
}

/// The nearest ancestor of `start` (inclusive) containing `.git`, as a
/// directory or as a file (worktrees, submodules).
#[must_use]
pub fn git_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(".git").exists())
        .map(Path::to_path_buf)
}

/// Point `config.agent.system_prompt_file` at the user-global `SYSTEM.md`.
///
/// Applies only when `SYSTEM.md` exists and is non-empty and the config sets
/// neither `system_prompt_file` nor a non-empty `system_prompt`. Returns the
/// applied path.
pub fn apply_system_md(config: &mut Config, global: Option<&GlobalDirs>) -> Option<PathBuf> {
    let path = global?.system_md();
    let file_set = config
        .agent
        .system_prompt_file
        .as_ref()
        .is_some_and(|p| !p.as_os_str().is_empty());
    let inline_set = config
        .agent
        .system_prompt
        .as_ref()
        .is_some_and(|p| !p.trim().is_empty());
    if file_set || inline_set {
        return None;
    }
    match std::fs::read_to_string(&path) {
        Ok(content) if !content.trim().is_empty() => {}
        Ok(_) => return None,
        Err(err) => {
            if err.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %path.display(), %err, "cannot read SYSTEM.md; ignoring");
            }
            return None;
        }
    }
    config.agent.system_prompt_file = Some(path.clone());
    Some(path)
}

/// The file `/remember` appends to: the workspace `AGENTS.md` Meerkat loads
/// (`AGENTS.md`, else `.rkat/AGENTS.md`), or `<workspace>/AGENTS.md` when
/// neither exists.
#[must_use]
pub fn remember_target(workspace: &Path) -> PathBuf {
    find_project_agents_md_in(workspace).unwrap_or_else(|| workspace.join("AGENTS.md"))
}

/// Append `- <text>` to `path`.
///
/// Line breaks in `text` are collapsed to single spaces and the result is
/// trimmed. A missing file is created with an `# AGENTS.md` header; an
/// existing file without a trailing newline gets one first.
///
/// # Errors
///
/// [`std::io::ErrorKind::InvalidInput`] when `text` is blank, or any I/O
/// error reading or writing `path`.
pub fn append_remember_line(path: &Path, text: &str) -> std::io::Result<()> {
    let line = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if line.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "nothing to remember",
        ));
    }

    let existing = match std::fs::read_to_string(path) {
        Ok(content) => Some(content),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => return Err(err),
    };
    let mut out = String::new();
    match existing.as_deref() {
        None => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            out.push_str("# AGENTS.md\n\n");
        }
        Some(content) if !content.is_empty() && !content.ends_with('\n') => out.push('\n'),
        Some(_) => {}
    }
    out.push_str("- ");
    out.push_str(&line);
    out.push('\n');

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(out.as_bytes())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn write(path: &Path, content: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    struct Tree {
        _tmp: tempfile::TempDir,
        root: PathBuf,
        a: PathBuf,
        ws: PathBuf,
        global: GlobalDirs,
    }

    fn tree() -> Tree {
        let tmp = tempfile::tempdir().unwrap();
        let base = std::fs::canonicalize(tmp.path()).unwrap();
        let root = base.join("root");
        let a = root.join("a");
        let ws = a.join("b");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        write(&root.join("AGENTS.md"), "root rules\n");
        write(&a.join(".rkat/AGENTS.md"), "a rules\n");
        write(&ws.join("AGENTS.md"), "workspace rules\n");
        write(&ws.join(LOCAL_FILE_NAME), "local rules\n");
        let global = GlobalDirs {
            config: base.join("config"),
        };
        write(&global.agents_md(), "global rules\n");
        Tree {
            _tmp: tmp,
            root,
            a,
            ws,
            global,
        }
    }

    #[test]
    fn discovers_in_order_and_excludes_workspace_agents_md() {
        let t = tree();
        let sections = InstructionSources::discover(&t.ws, Some(&t.global));
        let kinds: Vec<_> = sections.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            [
                SectionKind::UserGlobal,
                SectionKind::Ancestor,
                SectionKind::Ancestor,
                SectionKind::Local
            ]
        );
        let paths: Vec<_> = sections.iter().map(|s| s.path.clone()).collect();
        assert_eq!(
            paths,
            [
                t.global.agents_md(),
                t.root.join("AGENTS.md"),
                t.a.join(".rkat/AGENTS.md"),
                t.ws.join(LOCAL_FILE_NAME),
            ]
        );
        for s in &sections {
            assert!(s.heading().contains(&s.path.display().to_string()));
            assert!(!s.truncated);
        }
        assert_eq!(sections[1].body, "root rules");
        let rendered = InstructionSources::render_all(&sections);
        assert_eq!(rendered.len(), 4);
        assert!(rendered.iter().all(|r| !r.contains("workspace rules")));
        assert_eq!(
            rendered[3],
            format!(
                "# Instructions (from {})\n\nlocal rules",
                t.ws.join(LOCAL_FILE_NAME).display()
            )
        );
    }

    #[test]
    fn rkat_local_file_is_accepted() {
        let t = tree();
        std::fs::remove_file(t.ws.join(LOCAL_FILE_NAME)).unwrap();
        write(&t.ws.join(".rkat").join(LOCAL_FILE_NAME), "hidden local\n");
        let sections = InstructionSources::discover(&t.ws, None);
        let last = sections.last().unwrap();
        assert_eq!(last.kind, SectionKind::Local);
        assert_eq!(last.body, "hidden local");
    }

    #[test]
    fn oversized_file_is_truncated_with_marker() {
        let t = tree();
        // Multi-byte characters straddling the cap.
        write(&t.global.agents_md(), &"é".repeat(AGENTS_MD_MAX_BYTES));
        let sections = InstructionSources::discover(&t.ws, Some(&t.global));
        let global = &sections[0];
        assert!(global.truncated);
        assert!(global.body.len() <= AGENTS_MD_MAX_BYTES);
        assert!(global.render().ends_with("\n\n[truncated at 32 KiB]"));
    }

    #[test]
    fn total_cap_stops_adding_sections() {
        let t = tree();
        let big = "x".repeat(AGENTS_MD_MAX_BYTES - 10);
        write(&t.global.agents_md(), &big);
        write(&t.root.join("AGENTS.md"), &big);
        write(&t.a.join(".rkat/AGENTS.md"), &big);
        write(&t.ws.join(LOCAL_FILE_NAME), &big);
        let sections = InstructionSources::discover(&t.ws, Some(&t.global));
        assert_eq!(sections.len(), 2, "third ~32 KiB section exceeds the cap");
        let total: usize = InstructionSources::render_all(&sections)
            .iter()
            .map(String::len)
            .sum();
        assert!(total <= TOTAL_MAX_BYTES);
    }

    #[test]
    fn git_file_worktree_is_detected() {
        let t = tree();
        std::fs::remove_dir_all(t.root.join(".git")).unwrap();
        write(&t.a.join(".git"), "gitdir: /elsewhere\n");
        assert_eq!(git_root(&t.ws), Some(t.a.clone()));
        let sections = InstructionSources::discover(&t.ws, None);
        let kinds: Vec<_> = sections.iter().map(|s| (s.kind, s.path.clone())).collect();
        assert_eq!(
            kinds,
            [
                (SectionKind::Ancestor, t.a.join(".rkat/AGENTS.md")),
                (SectionKind::Local, t.ws.join(LOCAL_FILE_NAME)),
            ]
        );
    }

    #[test]
    fn workspace_at_git_root_has_no_ancestors() {
        let t = tree();
        std::fs::create_dir_all(t.ws.join(".git")).unwrap();
        let sections = InstructionSources::discover(&t.ws, None);
        assert!(sections.iter().all(|s| s.kind != SectionKind::Ancestor));
    }

    #[test]
    fn no_git_means_no_ancestors() {
        let t = tree();
        std::fs::remove_dir_all(t.root.join(".git")).unwrap();
        // The tempdir itself may live under a git checkout; only assert when not.
        if git_root(&t.ws).is_none() {
            let sections = InstructionSources::discover(&t.ws, Some(&t.global));
            let kinds: Vec<_> = sections.iter().map(|s| s.kind).collect();
            assert_eq!(kinds, [SectionKind::UserGlobal, SectionKind::Local]);
        }
    }

    #[test]
    fn empty_files_are_skipped() {
        let t = tree();
        write(&t.root.join("AGENTS.md"), "  \n\t\n");
        write(&t.global.agents_md(), "");
        let sections = InstructionSources::discover(&t.ws, Some(&t.global));
        let kinds: Vec<_> = sections.iter().map(|s| s.kind).collect();
        assert_eq!(kinds, [SectionKind::Ancestor, SectionKind::Local]);
        assert_eq!(sections[0].path, t.a.join(".rkat/AGENTS.md"));
    }

    #[test]
    fn system_md_applies_only_without_configured_prompt() {
        let t = tree();
        assert_eq!(
            apply_system_md(&mut Config::default(), Some(&t.global)),
            None
        );
        write(&t.global.system_md(), "You are X.\n");

        let mut config = Config::default();
        assert_eq!(apply_system_md(&mut config, None), None);
        assert_eq!(
            apply_system_md(&mut config, Some(&t.global)),
            Some(t.global.system_md())
        );
        assert_eq!(config.agent.system_prompt_file, Some(t.global.system_md()));

        let mut config = Config::default();
        config.agent.system_prompt = Some("inline".into());
        assert_eq!(apply_system_md(&mut config, Some(&t.global)), None);
        assert_eq!(config.agent.system_prompt_file, None);

        let mut config = Config::default();
        config.agent.system_prompt = Some("   ".into());
        assert!(apply_system_md(&mut config, Some(&t.global)).is_some());

        let mut config = Config::default();
        config.agent.system_prompt_file = Some(PathBuf::from("/custom.md"));
        assert_eq!(apply_system_md(&mut config, Some(&t.global)), None);
        assert_eq!(
            config.agent.system_prompt_file,
            Some(PathBuf::from("/custom.md"))
        );

        write(&t.global.system_md(), " \n");
        assert_eq!(
            apply_system_md(&mut Config::default(), Some(&t.global)),
            None
        );
    }

    #[test]
    fn remember_target_resolution() {
        let t = tree();
        assert_eq!(remember_target(&t.ws), t.ws.join("AGENTS.md"));
        assert_eq!(remember_target(&t.a), t.a.join(".rkat/AGENTS.md"));
        let empty = t.root.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(remember_target(&empty), empty.join("AGENTS.md"));
    }

    #[test]
    fn append_remember_line_creates_and_appends() {
        let t = tree();
        let path = t.root.join("new/AGENTS.md");
        append_remember_line(&path, "  use tabs\n  not spaces  ").unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# AGENTS.md\n\n- use tabs not spaces\n"
        );
        append_remember_line(&path, "second").unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# AGENTS.md\n\n- use tabs not spaces\n- second\n"
        );

        let existing = t.root.join("AGENTS.md");
        std::fs::write(&existing, "no newline").unwrap();
        append_remember_line(&existing, "x\r\ny").unwrap();
        assert_eq!(
            std::fs::read_to_string(&existing).unwrap(),
            "no newline\n- x y\n"
        );

        let err = append_remember_line(&existing, " \n ").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
