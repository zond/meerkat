//! Tier-1 shell analysis: a hand-written POSIX-subset lexer plus a read-only allowlist.
//!
//! The lexer deliberately understands only a small, predictable subset of POSIX
//! shell syntax. Anything that could hide a second command or expand at runtime
//! (command substitution, process substitution, parameter expansion, heredocs,
//! subshells, brace groups) is reported as [`Unsupported`], which routes the
//! command to the classifier instead of auto-allowing it.
//!
//! [`allowlisted`] then requires *every* segment of a `&&`/`||`/`;`/`|` chain to
//! pass the allowlist; one unknown segment sends the whole command to Tier 3.

use super::policy::PolicyContext;
use std::fmt;
use std::path::PathBuf;

/// Control operators separating simple commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `&&`
    AndAnd,
    /// `||`
    OrOr,
    /// `;`
    Semi,
    /// `|` (also `|&`)
    Pipe,
    /// `&` (background)
    Amp,
    /// An unescaped newline.
    Newline,
}

/// Redirection operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedirectKind {
    /// `<`
    In,
    /// `>`
    Out,
    /// `>>`
    Append,
    /// `>|`
    Clobber,
    /// `>&`
    DupOut,
    /// `<&`
    DupIn,
    /// `<>`
    ReadWrite,
    /// `&>` / `&>>`
    OutErr,
}

/// A redirection with its (unexpanded) target word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redirect {
    /// Explicit file descriptor (`2>`), if any.
    pub fd: Option<u32>,
    /// Operator kind.
    pub kind: RedirectKind,
    /// Target word after quote removal.
    pub target: String,
}

/// A word after quote removal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// Text after quote removal and escape processing.
    pub text: String,
    /// Whether the word contains an unquoted glob character (`*`, `?`, `[`).
    pub unquoted_glob: bool,
}

/// A lexer token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tok {
    /// A word.
    Word(Word),
    /// A control operator.
    Op(Op),
    /// A redirection including its target.
    Redirect(Redirect),
}

/// The command uses syntax outside the supported subset.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported shell syntax: {0}")]
pub struct Unsupported(pub &'static str);

/// The command is not on the Tier-1 allowlist; the reason is human readable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("not allowlisted: {0}")]
pub struct NotAllowlisted(pub String);

impl NotAllowlisted {
    fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }
}

impl From<Unsupported> for NotAllowlisted {
    fn from(err: Unsupported) -> Self {
        Self(err.to_string())
    }
}

/// Raw tokens before redirect targets are attached.
enum Raw {
    Word(Word),
    Op(Op),
    Redir { fd: Option<u32>, kind: RedirectKind },
}

#[derive(Default)]
struct WordBuf {
    text: String,
    active: bool,
    quoted: bool,
    unquoted_glob: bool,
}

impl WordBuf {
    fn push_unquoted(&mut self, c: char) {
        self.active = true;
        if matches!(c, '*' | '?' | '[') {
            self.unquoted_glob = true;
        }
        self.text.push(c);
    }

    fn push_quoted(&mut self, c: char) {
        self.active = true;
        self.quoted = true;
        self.text.push(c);
    }

    fn mark_quoted(&mut self) {
        self.active = true;
        self.quoted = true;
    }

    /// The pending word is an unquoted run of digits (a possible fd prefix).
    fn as_fd(&self) -> Option<u32> {
        if self.active
            && !self.quoted
            && !self.text.is_empty()
            && self.text.bytes().all(|b| b.is_ascii_digit())
        {
            return self.text.parse().ok();
        }
        None
    }

    fn flush(&mut self, out: &mut Vec<Raw>) {
        if self.active {
            let buf = std::mem::take(self);
            out.push(Raw::Word(Word {
                text: buf.text,
                unquoted_glob: buf.unquoted_glob,
            }));
        }
    }
}

/// Lex a command string into tokens.
///
/// # Errors
///
/// Returns [`Unsupported`] for any syntax outside the supported subset.
pub fn lex(input: &str) -> Result<Vec<Tok>, Unsupported> {
    let raw = lex_raw(input)?;
    attach_redirect_targets(raw)
}

fn lex_raw(input: &str) -> Result<Vec<Raw>, Unsupported> {
    let mut out = Vec::new();
    let mut word = WordBuf::default();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\r' => word.flush(&mut out),
            '\n' => {
                word.flush(&mut out);
                out.push(Raw::Op(Op::Newline));
            }
            '#' if !word.active => {
                // Comment until end of line.
                while let Some(&n) = chars.peek() {
                    if n == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '\'' => {
                word.mark_quoted();
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(q) => word.push_quoted(q),
                        None => return Err(Unsupported("unterminated single quote")),
                    }
                }
            }
            '"' => {
                word.mark_quoted();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(e @ ('$' | '`' | '"' | '\\')) => word.push_quoted(e),
                            Some('\n') => {}
                            Some(other) => {
                                word.push_quoted('\\');
                                word.push_quoted(other);
                            }
                            None => return Err(Unsupported("unterminated double quote")),
                        },
                        Some('$') => return Err(Unsupported("parameter expansion")),
                        Some('`') => return Err(Unsupported("command substitution")),
                        Some(q) => word.push_quoted(q),
                        None => return Err(Unsupported("unterminated double quote")),
                    }
                }
            }
            '\\' => match chars.next() {
                Some('\n') => {}
                Some(e) => word.push_quoted(e),
                None => return Err(Unsupported("trailing backslash")),
            },
            '$' => return Err(Unsupported("parameter expansion")),
            '`' => return Err(Unsupported("command substitution")),
            '(' | ')' => return Err(Unsupported("subshell or grouping")),
            '{' | '}' => return Err(Unsupported("brace group or expansion")),
            ';' => {
                word.flush(&mut out);
                if chars.peek() == Some(&';') {
                    return Err(Unsupported("case terminator"));
                }
                out.push(Raw::Op(Op::Semi));
            }
            '|' => {
                word.flush(&mut out);
                match chars.peek() {
                    Some('|') => {
                        chars.next();
                        out.push(Raw::Op(Op::OrOr));
                    }
                    Some('&') => {
                        chars.next();
                        out.push(Raw::Op(Op::Pipe));
                    }
                    _ => out.push(Raw::Op(Op::Pipe)),
                }
            }
            '&' => {
                word.flush(&mut out);
                match chars.peek() {
                    Some('&') => {
                        chars.next();
                        out.push(Raw::Op(Op::AndAnd));
                    }
                    Some('>') => {
                        chars.next();
                        if chars.peek() == Some(&'>') {
                            chars.next();
                        }
                        out.push(Raw::Redir {
                            fd: None,
                            kind: RedirectKind::OutErr,
                        });
                    }
                    _ => out.push(Raw::Op(Op::Amp)),
                }
            }
            '<' | '>' => {
                let fd = word.as_fd();
                if fd.is_some() {
                    word = WordBuf::default();
                } else {
                    word.flush(&mut out);
                }
                let kind = if c == '<' {
                    match chars.peek() {
                        Some('<') => return Err(Unsupported("heredoc or here-string")),
                        Some('(') => return Err(Unsupported("process substitution")),
                        Some('&') => {
                            chars.next();
                            RedirectKind::DupIn
                        }
                        Some('>') => {
                            chars.next();
                            RedirectKind::ReadWrite
                        }
                        _ => RedirectKind::In,
                    }
                } else {
                    match chars.peek() {
                        Some('(') => return Err(Unsupported("process substitution")),
                        Some('>') => {
                            chars.next();
                            RedirectKind::Append
                        }
                        Some('&') => {
                            chars.next();
                            RedirectKind::DupOut
                        }
                        Some('|') => {
                            chars.next();
                            RedirectKind::Clobber
                        }
                        _ => RedirectKind::Out,
                    }
                };
                out.push(Raw::Redir { fd, kind });
            }
            other => word.push_unquoted(other),
        }
    }
    word.flush(&mut out);
    Ok(out)
}

fn attach_redirect_targets(raw: Vec<Raw>) -> Result<Vec<Tok>, Unsupported> {
    let mut out = Vec::with_capacity(raw.len());
    let mut iter = raw.into_iter();
    while let Some(tok) = iter.next() {
        match tok {
            Raw::Word(w) => out.push(Tok::Word(w)),
            Raw::Op(op) => out.push(Tok::Op(op)),
            Raw::Redir { fd, kind } => match iter.next() {
                Some(Raw::Word(target)) => out.push(Tok::Redirect(Redirect {
                    fd,
                    kind,
                    target: target.text,
                })),
                _ => return Err(Unsupported("redirect without a target")),
            },
        }
    }
    Ok(out)
}

/// One simple command: its words, redirects, and the operator that ended it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Segment {
    /// Words in order (command name first).
    pub words: Vec<Word>,
    /// Redirections attached to this command.
    pub redirects: Vec<Redirect>,
    /// Operator that terminated the segment (`None` at end of input).
    pub terminator: Option<Op>,
}

impl Segment {
    fn is_empty(&self) -> bool {
        self.words.is_empty() && self.redirects.is_empty()
    }
}

/// Split a token stream on control operators.
pub fn segments(tokens: Vec<Tok>) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut current = Segment::default();
    for tok in tokens {
        match tok {
            Tok::Word(w) => current.words.push(w),
            Tok::Redirect(r) => current.redirects.push(r),
            Tok::Op(op) => {
                current.terminator = Some(op);
                out.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Commands that run another command; never Tier 1.
const WRAPPERS: &[&str] = &[
    "sudo", "doas", "env", "xargs", "exec", "eval", "nohup", "timeout", "nice", "time", "su",
    "setsid", "stdbuf", "chroot", "watch", "parallel",
];

/// How positional words of a command are checked against the project root.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PathArgs {
    /// Every non-flag word (and every `--flag=value` value) must resolve inside the project.
    InsideProject,
    /// Words are plain strings (e.g. `echo`), never touched as paths.
    Ignore,
}

/// Command-specific checks beyond forbidden flags.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Extra {
    None,
    /// `uniq` with at most one positional (the two-positional form writes a file).
    UniqSingleInput,
    /// `git branch`: listing only.
    GitBranchList,
    /// `git remote`: only `-v`/`--verbose`.
    GitRemoteList,
    /// `cargo fmt` must carry `--check`.
    CargoFmtCheck,
    /// Executes project code (build scripts, tests); requires `trust_project_builds`.
    ProjectBuild,
}

struct Allowed {
    name: &'static str,
    /// Subcommand that must directly follow the command name (`git status`).
    sub: Option<&'static str>,
    /// Forbidden argument prefixes (`--pre` matches `--pre-glob`, `-exec` matches `-execdir`).
    forbidden_prefixes: &'static [&'static str],
    /// Characters forbidden inside single-dash short-flag clusters (`-o`, `-uo`).
    forbidden_short: &'static [char],
    paths: PathArgs,
    extra: Extra,
}

const fn simple(name: &'static str, paths: PathArgs) -> Allowed {
    Allowed {
        name,
        sub: None,
        forbidden_prefixes: &[],
        forbidden_short: &[],
        paths,
        extra: Extra::None,
    }
}

const GIT_FORBIDDEN: &[&str] = &["--ext-diff", "--output", "-o", "--textconv"];
const CARGO_FORBIDDEN: &[&str] = &["--fix", "--config", "-Z", "--allow-dirty"];

const fn git(sub: &'static str, extra: Extra) -> Allowed {
    Allowed {
        name: "git",
        sub: Some(sub),
        forbidden_prefixes: GIT_FORBIDDEN,
        forbidden_short: &[],
        paths: PathArgs::InsideProject,
        extra,
    }
}

const fn cargo(sub: &'static str, extra: Extra) -> Allowed {
    Allowed {
        name: "cargo",
        sub: Some(sub),
        forbidden_prefixes: CARGO_FORBIDDEN,
        forbidden_short: &[],
        paths: PathArgs::InsideProject,
        extra,
    }
}

/// The Tier-1 allowlist.
const ALLOWLIST: &[Allowed] = &[
    simple("cat", PathArgs::InsideProject),
    simple("head", PathArgs::InsideProject),
    simple("tail", PathArgs::InsideProject),
    simple("wc", PathArgs::InsideProject),
    simple("ls", PathArgs::InsideProject),
    simple("pwd", PathArgs::Ignore),
    simple("echo", PathArgs::Ignore),
    simple("which", PathArgs::Ignore),
    Allowed {
        name: "file",
        sub: None,
        forbidden_prefixes: &["--compile"],
        forbidden_short: &['C'],
        paths: PathArgs::InsideProject,
        extra: Extra::None,
    },
    simple("stat", PathArgs::InsideProject),
    simple("du", PathArgs::InsideProject),
    Allowed {
        name: "tree",
        sub: None,
        forbidden_prefixes: &["--output"],
        forbidden_short: &['o'],
        paths: PathArgs::InsideProject,
        extra: Extra::None,
    },
    simple("diff", PathArgs::InsideProject),
    simple("cmp", PathArgs::InsideProject),
    simple("basename", PathArgs::Ignore),
    simple("dirname", PathArgs::Ignore),
    simple("realpath", PathArgs::InsideProject),
    Allowed {
        name: "sort",
        sub: None,
        forbidden_prefixes: &["--output", "--compress-program"],
        forbidden_short: &['o'],
        paths: PathArgs::InsideProject,
        extra: Extra::None,
    },
    Allowed {
        name: "uniq",
        sub: None,
        forbidden_prefixes: &[],
        forbidden_short: &[],
        paths: PathArgs::InsideProject,
        extra: Extra::UniqSingleInput,
    },
    Allowed {
        name: "rg",
        sub: None,
        forbidden_prefixes: &["--pre"],
        forbidden_short: &[],
        paths: PathArgs::InsideProject,
        extra: Extra::None,
    },
    simple("grep", PathArgs::InsideProject),
    Allowed {
        name: "find",
        sub: None,
        forbidden_prefixes: &["-exec", "-ok", "-delete", "-fprint", "-fls"],
        forbidden_short: &[],
        paths: PathArgs::InsideProject,
        extra: Extra::None,
    },
    git("status", Extra::None),
    git("diff", Extra::None),
    git("log", Extra::None),
    git("show", Extra::None),
    git("branch", Extra::GitBranchList),
    git("rev-parse", Extra::None),
    git("ls-files", Extra::None),
    git("blame", Extra::None),
    git("remote", Extra::GitRemoteList),
    cargo("check", Extra::ProjectBuild),
    cargo("clippy", Extra::ProjectBuild),
    cargo("fmt", Extra::CargoFmtCheck),
    cargo("tree", Extra::None),
    cargo("metadata", Extra::None),
    cargo("build", Extra::ProjectBuild),
    cargo("test", Extra::ProjectBuild),
    cargo("nextest", Extra::ProjectBuild),
];

/// Global options accepted between `git` and its subcommand.
const GIT_GLOBAL_OK: &[&str] = &["--no-pager"];

/// Whether `command` passes the Tier-1 read-only allowlist, with relative paths
/// resolved against the project root.
///
/// # Errors
///
/// Returns [`NotAllowlisted`] with a reason when any segment fails.
pub fn allowlisted(command: &str, ctx: &PolicyContext<'_>) -> Result<(), NotAllowlisted> {
    allowlisted_in(command, None, ctx)
}

/// Like [`allowlisted`], with an explicit working directory (the shell tool's
/// `working_dir`, relative to the project root). The working directory itself
/// must lie inside the project.
///
/// # Errors
///
/// Returns [`NotAllowlisted`] with a reason when any segment fails.
pub fn allowlisted_in(
    command: &str,
    working_dir: Option<&str>,
    ctx: &PolicyContext<'_>,
) -> Result<(), NotAllowlisted> {
    if !ctx.posix_shell {
        return Err(NotAllowlisted::new(
            "shell analysis is disabled on non-POSIX shells",
        ));
    }
    let base = match working_dir {
        Some(dir) => {
            let resolved = ctx
                .root
                .resolve(dir)
                .map_err(|e| NotAllowlisted::new(format!("working directory: {e}")))?;
            if !resolved.inside() {
                return Err(NotAllowlisted::new(
                    "working directory is outside the project",
                ));
            }
            resolved.abs
        }
        None => ctx.root.as_path().to_path_buf(),
    };
    let segs = segments(lex(command)?);
    let mut any = false;
    for seg in &segs {
        if seg.terminator == Some(Op::Amp) {
            return Err(NotAllowlisted::new("background execution (`&`)"));
        }
        if seg.is_empty() {
            continue;
        }
        any = true;
        check_segment(seg, &base, ctx)?;
    }
    if any {
        Ok(())
    } else {
        Err(NotAllowlisted::new("empty command"))
    }
}

fn check_segment(
    seg: &Segment,
    base: &std::path::Path,
    ctx: &PolicyContext<'_>,
) -> Result<(), NotAllowlisted> {
    for redirect in &seg.redirects {
        check_redirect(redirect, base, ctx)?;
    }
    let mut words: &[Word] = &seg.words;
    while let Some((first, rest)) = words.split_first() {
        if first.text == "command" || first.text == "builtin" {
            words = rest;
        } else {
            break;
        }
    }
    let Some((name, args)) = words.split_first() else {
        return Err(NotAllowlisted::new("redirect without a command"));
    };
    if name.unquoted_glob {
        return Err(NotAllowlisted::new("glob in command name"));
    }
    if is_assignment(&name.text) {
        return Err(NotAllowlisted::new("leading variable assignment"));
    }
    if WRAPPERS.contains(&name.text.as_str()) {
        return Err(NotAllowlisted::new(format!(
            "`{}` runs another command",
            name.text
        )));
    }
    let candidates: Vec<&Allowed> = ALLOWLIST.iter().filter(|a| a.name == name.text).collect();
    if candidates.is_empty() {
        return Err(NotAllowlisted::new(format!(
            "`{}` is not on the read-only allowlist",
            name.text
        )));
    }

    let args: Vec<&str> = args.iter().map(|w| w.text.as_str()).collect();
    let (entry, rest) = match_subcommand(&candidates, &name.text, &args)?;
    check_entry(entry, &args, rest, base, ctx)
}

/// Pick the allowlist entry for this invocation and return the args after the subcommand.
fn match_subcommand<'e, 'a>(
    candidates: &[&'e Allowed],
    name: &str,
    args: &'a [&'a str],
) -> Result<(&'e Allowed, &'a [&'a str]), NotAllowlisted> {
    let Some(first) = candidates.first() else {
        return Err(NotAllowlisted::new(format!("`{name}` is not allowlisted")));
    };
    if first.sub.is_none() {
        return Ok((first, args));
    }
    let mut idx = 0;
    if name == "git" {
        while args.get(idx).is_some_and(|a| GIT_GLOBAL_OK.contains(a)) {
            idx += 1;
        }
    }
    let Some(sub) = args.get(idx) else {
        return Err(NotAllowlisted::new(format!(
            "`{name}` without a subcommand"
        )));
    };
    if sub.starts_with('-') || sub.starts_with('+') {
        return Err(NotAllowlisted::new(format!(
            "`{name}` option `{sub}` before the subcommand"
        )));
    }
    let entry = candidates
        .iter()
        .find(|c| c.sub == Some(*sub))
        .ok_or_else(|| {
            NotAllowlisted::new(format!("`{name} {sub}` is not on the read-only allowlist"))
        })?;
    Ok((entry, args.get(idx + 1..).unwrap_or(&[])))
}

fn check_entry(
    entry: &Allowed,
    all_args: &[&str],
    args: &[&str],
    base: &std::path::Path,
    ctx: &PolicyContext<'_>,
) -> Result<(), NotAllowlisted> {
    let label = match entry.sub {
        Some(sub) => format!("{} {sub}", entry.name),
        None => entry.name.to_string(),
    };
    for arg in all_args {
        if let Some(prefix) = entry
            .forbidden_prefixes
            .iter()
            .find(|p| arg.starts_with(**p))
        {
            return Err(NotAllowlisted::new(format!("`{label}` with `{prefix}`")));
        }
        if is_short_cluster(arg)
            && let Some(c) = entry
                .forbidden_short
                .iter()
                .find(|c| arg.chars().skip(1).any(|a| a == **c))
        {
            return Err(NotAllowlisted::new(format!("`{label}` with `-{c}`")));
        }
    }

    match entry.extra {
        Extra::None => {}
        Extra::UniqSingleInput => {
            if positionals(args).count() > 1 {
                return Err(NotAllowlisted::new("`uniq` with an output file"));
            }
        }
        Extra::GitBranchList => check_git_branch(args)?,
        Extra::GitRemoteList => {
            if args.iter().any(|a| !matches!(*a, "-v" | "--verbose")) {
                return Err(NotAllowlisted::new("`git remote` beyond listing"));
            }
        }
        Extra::CargoFmtCheck => {
            if !args.contains(&"--check") {
                return Err(NotAllowlisted::new("`cargo fmt` without `--check`"));
            }
        }
        Extra::ProjectBuild => {
            if !ctx.trust_project_builds {
                return Err(NotAllowlisted::new(format!(
                    "`{label}` executes project code and trust_project_builds is off"
                )));
            }
        }
    }

    if entry.paths == PathArgs::InsideProject {
        for value in path_like_values(args) {
            check_inside(value, base, ctx)?;
        }
    }
    Ok(())
}

fn check_git_branch(args: &[&str]) -> Result<(), NotAllowlisted> {
    const MUTATING_LONG: &[&str] = &[
        "--delete",
        "--move",
        "--copy",
        "--force",
        "--set-upstream",
        "--unset-upstream",
        "--edit-description",
        "--track",
        "--no-track",
        "--create-reflog",
    ];
    const MUTATING_SHORT: &[char] = &['d', 'D', 'm', 'M', 'c', 'C', 'f', 'u', 't'];
    const LISTING: &[&str] = &[
        "-l",
        "--list",
        "--contains",
        "--no-contains",
        "--merged",
        "--no-merged",
        "--points-at",
    ];
    for arg in args {
        if MUTATING_LONG.iter().any(|m| arg.starts_with(m)) {
            return Err(NotAllowlisted::new(format!("`git branch {arg}` mutates")));
        }
        if is_short_cluster(arg) && arg.chars().skip(1).any(|c| MUTATING_SHORT.contains(&c)) {
            return Err(NotAllowlisted::new(format!("`git branch {arg}` mutates")));
        }
    }
    let listing = args.iter().any(|a| {
        LISTING
            .iter()
            .any(|l| a == l || a.starts_with(&format!("{l}=")))
    });
    if positionals(args).next().is_some() && !listing {
        return Err(NotAllowlisted::new("`git branch <name>` creates a branch"));
    }
    Ok(())
}

/// Single-dash short-flag cluster such as `-o` or `-nru` (not `--long`, not `-`).
fn is_short_cluster(arg: &str) -> bool {
    arg.len() > 1 && arg.starts_with('-') && !arg.starts_with("--")
}

fn positionals<'a>(args: &'a [&'a str]) -> impl Iterator<Item = &'a str> + 'a {
    args.iter()
        .copied()
        .filter(|a| !(a.starts_with('-') && a.len() > 1))
}

/// Non-flag words, plus the value part of `--flag=value`.
fn path_like_values<'a>(args: &'a [&'a str]) -> impl Iterator<Item = &'a str> + 'a {
    args.iter().filter_map(|a| {
        if a.starts_with('-') && a.len() > 1 {
            a.split_once('=').map(|(_, v)| v).filter(|v| !v.is_empty())
        } else {
            Some(*a)
        }
    })
}

fn check_inside(
    value: &str,
    base: &std::path::Path,
    ctx: &PolicyContext<'_>,
) -> Result<(), NotAllowlisted> {
    if value == "-" {
        return Ok(());
    }
    let candidate: PathBuf = if value.starts_with('/') || value.starts_with('~') {
        PathBuf::from(value)
    } else {
        base.join(value)
    };
    let Some(as_str) = candidate.to_str() else {
        return Err(NotAllowlisted::new("non UTF-8 path argument"));
    };
    let resolved = ctx
        .root
        .resolve(as_str)
        .map_err(|e| NotAllowlisted::new(format!("path `{value}`: {e}")))?;
    if resolved.inside() {
        Ok(())
    } else {
        Err(NotAllowlisted::new(format!(
            "path `{value}` is outside the project"
        )))
    }
}

fn check_redirect(
    redirect: &Redirect,
    base: &std::path::Path,
    ctx: &PolicyContext<'_>,
) -> Result<(), NotAllowlisted> {
    let ok = match (redirect.kind, redirect.fd) {
        (RedirectKind::DupOut, Some(2)) => redirect.target == "1",
        (RedirectKind::Out, None | Some(1 | 2)) => redirect.target == "/dev/null",
        (RedirectKind::In, None | Some(0)) => {
            check_inside(&redirect.target, base, ctx)?;
            true
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(NotAllowlisted::new(format!("redirect {redirect}")))
    }
}

impl fmt::Display for Redirect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(fd) = self.fd {
            write!(f, "{fd}")?;
        }
        let op = match self.kind {
            RedirectKind::In => "<",
            RedirectKind::Out => ">",
            RedirectKind::Append => ">>",
            RedirectKind::Clobber => ">|",
            RedirectKind::DupOut => ">&",
            RedirectKind::DupIn => "<&",
            RedirectKind::ReadWrite => "<>",
            RedirectKind::OutErr => "&>",
        };
        write!(f, "{op}{}", self.target)
    }
}

fn is_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Interpreters whose first positional argument is a script file.
const INTERPRETERS: &[&str] = &[
    "sh", "bash", "zsh", "dash", "python", "python3", "node", "ruby", "perl", "deno", "bun",
];

/// If the command runs a script file (`./x.sh`, `bash x.sh`, `python3 tool.py`),
/// return the script path as written. Used to attach in-project script content to
/// the classifier payload. Returns `None` when the command cannot be lexed.
pub fn script_path(command: &str) -> Option<String> {
    let segs = segments(lex(command).ok()?);
    for seg in segs {
        let mut words = seg.words.iter().map(|w| w.text.as_str());
        let Some(name) = words.next() else {
            continue;
        };
        if name.contains('/') {
            return Some(name.to_string());
        }
        let base = name.rsplit('/').next().unwrap_or(name);
        if INTERPRETERS.contains(&base) {
            // `deno run x.ts`, `bun run x.ts`
            let found = words.find(|w| !w.starts_with('-') && *w != "run");
            if let Some(path) = found {
                return Some(path.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::paths::ProjectRoot;

    fn project() -> (tempfile::TempDir, ProjectRoot) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.path().join("input.txt"), "x\n").unwrap();
        let root = ProjectRoot::new(dir.path()).unwrap();
        (dir, root)
    }

    fn ctx(root: &ProjectRoot, trust: bool) -> PolicyContext<'_> {
        PolicyContext {
            root,
            trust_project_builds: trust,
            posix_shell: true,
        }
    }

    fn words(cmd: &str) -> Vec<String> {
        lex(cmd)
            .unwrap()
            .into_iter()
            .filter_map(|t| match t {
                Tok::Word(w) => Some(w.text),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn lexer_quotes_and_escapes() {
        assert_eq!(
            words("echo 'a b' \"c d\" e\\ f"),
            ["echo", "a b", "c d", "e f"]
        );
        assert_eq!(words("echo 'a$b'"), ["echo", "a$b"]);
        assert_eq!(words(r#"echo "a\"b""#), ["echo", "a\"b"]);
        assert_eq!(words(r#"echo "a\nb""#), ["echo", "a\\nb"]);
        assert_eq!(words("echo ''"), ["echo", ""]);
        assert_eq!(words("ls # comment $(x)"), ["ls"]);
        assert_eq!(words("echo a#b"), ["echo", "a#b"]);
    }

    #[test]
    fn lexer_operators() {
        let toks = lex("a && b || c; d | e & f\ng |& h").unwrap();
        let ops: Vec<Op> = toks
            .iter()
            .filter_map(|t| match t {
                Tok::Op(o) => Some(*o),
                _ => None,
            })
            .collect();
        assert_eq!(
            ops,
            [
                Op::AndAnd,
                Op::OrOr,
                Op::Semi,
                Op::Pipe,
                Op::Amp,
                Op::Newline,
                Op::Pipe
            ]
        );
    }

    #[test]
    fn lexer_redirects() {
        let toks = lex("cmd 2>&1 >/dev/null 2>/dev/null <in.txt >>log &>all 3<>rw").unwrap();
        let redirects: Vec<Redirect> = toks
            .into_iter()
            .filter_map(|t| match t {
                Tok::Redirect(r) => Some(r),
                _ => None,
            })
            .collect();
        assert_eq!(redirects.len(), 7);
        assert_eq!(
            redirects[0],
            Redirect {
                fd: Some(2),
                kind: RedirectKind::DupOut,
                target: "1".into()
            }
        );
        assert_eq!(redirects[1].kind, RedirectKind::Out);
        assert_eq!(redirects[1].fd, None);
        assert_eq!(redirects[2].fd, Some(2));
        assert_eq!(redirects[3].kind, RedirectKind::In);
        assert_eq!(redirects[4].kind, RedirectKind::Append);
        assert_eq!(redirects[5].kind, RedirectKind::OutErr);
        assert_eq!(redirects[6].kind, RedirectKind::ReadWrite);
        assert_eq!(redirects[6].fd, Some(3));
    }

    #[test]
    fn quoted_digits_are_not_fds() {
        let toks = lex("echo '2'>x").unwrap();
        assert!(toks.contains(&Tok::Word(Word {
            text: "2".into(),
            unquoted_glob: false
        })));
    }

    #[test]
    fn lexer_unsupported() {
        for cmd in [
            "echo $(id)",
            "echo `id`",
            "echo $HOME",
            "echo \"$HOME\"",
            "echo \"`id`\"",
            "diff <(ls) <(ls a)",
            "tee >(cat)",
            "(ls)",
            "{ ls; }",
            "echo {a,b}",
            "cat <<EOF",
            "cat <<<word",
            "echo 'unterminated",
            "echo \"unterminated",
            "echo trailing\\",
            "ls >",
            "case x in a) ;; esac",
        ] {
            assert!(lex(cmd).is_err(), "{cmd} should be unsupported");
        }
    }

    #[test]
    fn segments_split_on_ops() {
        let segs = segments(lex("ls -la && cat a | rg x; pwd").unwrap());
        assert_eq!(segs.len(), 4);
        assert_eq!(segs[0].terminator, Some(Op::AndAnd));
        assert_eq!(segs[1].terminator, Some(Op::Pipe));
        assert_eq!(segs[2].terminator, Some(Op::Semi));
        assert_eq!(segs[3].terminator, None);
        assert_eq!(segs[3].words[0].text, "pwd");
    }

    #[test]
    fn allowlist_table() {
        let (dir, root) = project();
        let outside = dir.path().parent().unwrap().join("elsewhere.txt");
        let outside_cat = format!("cat {}", outside.display());
        let inside_abs = format!("cat {}", dir.path().join("src/main.rs").display());
        let cases: Vec<(&str, bool)> = vec![
            // Plan examples
            ("ls && rm -rf /", false),
            ("git -c core.pager=sh status", false),
            ("cat a | rg x", true),
            ("echo $(id)", false),
            ("find . -delete", false),
            ("cargo test 2>&1", true),
            // Simple read-only commands
            ("ls", true),
            ("ls -la src", true),
            ("pwd", true),
            ("echo hello world", true),
            ("echo /etc/passwd", true),
            ("which cargo", true),
            ("cat src/main.rs", true),
            ("head -n 20 src/main.rs", true),
            ("tail -n 5 src/main.rs", true),
            ("wc -l src/main.rs", true),
            ("file src/main.rs", true),
            ("file -C -m magic", false),
            ("stat src/main.rs", true),
            ("du -sh src", true),
            ("tree src", true),
            ("tree -o out.txt", false),
            ("diff src/main.rs input.txt", true),
            ("cmp src/main.rs input.txt", true),
            ("basename /a/b/c", true),
            ("dirname /a/b/c", true),
            ("realpath src", true),
            ("sort input.txt", true),
            ("sort -o out.txt input.txt", false),
            ("sort -uo out.txt input.txt", false),
            ("sort --output=out.txt input.txt", false),
            ("uniq input.txt", true),
            ("uniq input.txt out.txt", false),
            ("rg TODO src", true),
            ("rg --pre cat TODO", false),
            ("rg --pre-glob '*.gz' TODO", false),
            ("grep -rn foo src", true),
            ("find . -name '*.rs'", true),
            ("find . -exec rm {} \\;", false),
            ("find . -execdir ls ';'", false),
            ("find . -ok rm ';'", false),
            ("find . -fprint out", false),
            ("find . -fls out", false),
            ("find / -name passwd", false),
            // Paths
            ("cat ../secret", false),
            ("cat ~/.ssh/id_rsa", false),
            ("cat /etc/passwd", false),
            (outside_cat.as_str(), false),
            (inside_abs.as_str(), true),
            ("rg -f /etc/patterns x", false),
            ("ls --directory=/etc", false),
            // Chains
            ("ls && pwd", true),
            ("ls || pwd", true),
            ("ls; pwd", true),
            ("ls\npwd", true),
            ("ls;", true),
            ("cat a | sort | uniq", true),
            ("ls | xargs rm", false),
            ("ls & pwd", false),
            ("ls &", false),
            ("", false),
            ("   ", false),
            // Redirects
            ("ls >/dev/null", true),
            ("ls 2>/dev/null", true),
            ("ls >/dev/null 2>&1", true),
            ("ls > out.txt", false),
            ("ls >> out.txt", false),
            ("ls 1>&2", false),
            ("ls &>/dev/null", false),
            ("cat < input.txt", true),
            ("cat < /etc/passwd", false),
            // Prefixes and wrappers
            ("FOO=bar ls", false),
            ("command ls", true),
            ("builtin echo hi", true),
            ("sudo ls", false),
            ("env ls", false),
            ("exec ls", false),
            ("eval ls", false),
            ("nohup ls", false),
            ("timeout 5 ls", false),
            ("nice ls", false),
            ("time ls", false),
            ("l* src", false),
            ("/bin/ls", false),
            ("./script.sh", false),
            ("rm -rf target", false),
            ("python3 -c 'print(1)'", false),
            // git
            ("git status", true),
            ("git --no-pager log --oneline -5", true),
            ("git diff HEAD~1", true),
            ("git diff --ext-diff", false),
            ("git diff --output=patch.txt", false),
            ("git log -o x", false),
            ("git show HEAD:src/main.rs", true),
            ("git branch", true),
            ("git branch -a", true),
            ("git branch -D main", false),
            ("git branch newbranch", false),
            ("git branch --contains abc123", true),
            ("git rev-parse HEAD", true),
            ("git ls-files", true),
            ("git blame src/main.rs", true),
            ("git remote -v", true),
            ("git remote add evil https://x", false),
            ("git push --force", false),
            ("git commit -m x", false),
            ("git -C /tmp status", false),
            ("git", false),
            // cargo
            ("cargo fmt --check", true),
            ("cargo fmt", false),
            ("cargo tree", true),
            ("cargo metadata --format-version 1", true),
            ("cargo check --workspace", true),
            ("cargo clippy --fix", false),
            ("cargo check --config build.rustc-wrapper=x", false),
            ("cargo +nightly check", false),
            ("cargo build --release", true),
            ("cargo nextest run", true),
            ("cargo run", false),
            ("cargo install ripgrep", false),
            ("cargo build --target-dir /tmp/x", false),
        ];
        let c = ctx(&root, true);
        let mut failures = Vec::new();
        for (cmd, expected) in &cases {
            let got = allowlisted(cmd, &c);
            if got.is_ok() != *expected {
                failures.push(format!("{cmd:?}: expected {expected}, got {got:?}"));
            }
        }
        assert!(cases.len() >= 60, "table has {} cases", cases.len());
        assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
    }

    #[test]
    fn project_builds_require_trust() {
        let (_dir, root) = project();
        let untrusted = ctx(&root, false);
        for cmd in [
            "cargo test",
            "cargo build",
            "cargo nextest run",
            "cargo check",
            "cargo clippy",
        ] {
            assert!(allowlisted(cmd, &untrusted).is_err(), "{cmd}");
        }
        assert!(allowlisted("cargo fmt --check", &untrusted).is_ok());
        assert!(allowlisted("cargo tree", &untrusted).is_ok());
    }

    #[test]
    fn non_posix_shell_never_allowlists() {
        let (_dir, root) = project();
        let c = PolicyContext {
            root: &root,
            trust_project_builds: true,
            posix_shell: false,
        };
        assert!(allowlisted("ls", &c).is_err());
    }

    #[test]
    fn working_dir_must_be_inside() {
        let (_dir, root) = project();
        let c = ctx(&root, true);
        assert!(allowlisted_in("ls", Some("src"), &c).is_ok());
        assert!(allowlisted_in("ls", Some("/etc"), &c).is_err());
        assert!(allowlisted_in("cat ../input.txt", Some("src"), &c).is_ok());
        assert!(allowlisted_in("cat ../../x", Some("src"), &c).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_argument_escaping_is_rejected() {
        let (dir, root) = project();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("s"), "x").unwrap();
        std::os::unix::fs::symlink(outside.path().join("s"), dir.path().join("link")).unwrap();
        assert!(allowlisted("cat link", &ctx(&root, true)).is_err());
    }

    #[test]
    fn script_paths() {
        assert_eq!(
            script_path("./build.sh --fast").as_deref(),
            Some("./build.sh")
        );
        assert_eq!(
            script_path("bash scripts/x.sh").as_deref(),
            Some("scripts/x.sh")
        );
        assert_eq!(
            script_path("python3 -u tool.py").as_deref(),
            Some("tool.py")
        );
        assert_eq!(script_path("deno run app.ts").as_deref(), Some("app.ts"));
        assert_eq!(script_path("ls -la"), None);
        assert_eq!(script_path("echo $(x)"), None);
    }
}
