//! User allow/deny rules and their TOML persistence.
//!
//! Rules match a tool name ([`ToolPattern`]) and optionally a glob over the
//! action's canonical subject ([`ArgPattern`]): the command string for `shell`,
//! the display path for file tools, pretty JSON for everything else.
//!
//! Three files are layered: `~/.rkat/tui.toml` (global), `<realm>/tui.toml`
//! (per-workspace, the default write target) and `<project>/.rkat/tui.toml`
//! (committed with the repository, therefore untrusted: it may only add deny
//! rules and `[auto]` settings, and may not loosen `trust_project_builds`).

use globset::{Glob, GlobBuilder, GlobMatcher};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// File name of every rules layer.
pub const RULES_FILE_NAME: &str = "tui.toml";

/// Errors from rule parsing and persistence.
#[derive(Debug, thiserror::Error)]
pub enum RulesError {
    /// A glob pattern failed to compile.
    #[error("invalid pattern `{pattern}`: {source}")]
    Pattern {
        /// The offending pattern.
        pattern: String,
        /// Underlying globset error.
        #[source]
        source: globset::Error,
    },
    /// A tool pattern was empty.
    #[error("tool pattern is empty")]
    EmptyTool,
    /// Reading or writing a rules file failed.
    #[error("{path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A rules file is not valid TOML for [`RulesFile`].
    #[error("{path}: {source}")]
    Parse {
        /// File path.
        path: PathBuf,
        /// Underlying TOML error.
        #[source]
        source: toml::de::Error,
    },
    /// Serializing a rules file failed.
    #[error("failed to serialize rules: {0}")]
    Serialize(#[from] toml::ser::Error),
}

/// Whether a rule allows or denies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleKind {
    /// Allow without review.
    Allow,
    /// Deny outright.
    Deny,
}

/// Which layer a rule came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleOrigin {
    /// `~/.rkat/tui.toml`.
    Global,
    /// `<realm>/tui.toml`.
    Realm,
    /// `<project>/.rkat/tui.toml`.
    Project,
    /// Added during this process (not yet persisted).
    Session,
}

/// Tool-name matcher: an exact name or a glob such as `mcp__github__*`.
#[derive(Clone)]
pub struct ToolPattern {
    raw: String,
    glob: Option<GlobMatcher>,
}

impl ToolPattern {
    /// Parse a tool pattern.
    ///
    /// # Errors
    ///
    /// Returns [`RulesError`] for empty or invalid glob patterns.
    pub fn new(raw: impl Into<String>) -> Result<Self, RulesError> {
        let raw = raw.into();
        if raw.trim().is_empty() {
            return Err(RulesError::EmptyTool);
        }
        let glob = if has_glob_meta(&raw) {
            Some(compile(&raw)?)
        } else {
            None
        };
        Ok(Self { raw, glob })
    }

    /// The pattern as written.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Whether the pattern is a plain tool name.
    pub fn is_exact(&self) -> bool {
        self.glob.is_none()
    }

    /// Whether `tool` matches.
    pub fn matches(&self, tool: &str) -> bool {
        match &self.glob {
            Some(glob) => glob.is_match(tool),
            None => self.raw == tool,
        }
    }
}

impl PartialEq for ToolPattern {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for ToolPattern {}

impl fmt::Debug for ToolPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ToolPattern").field(&self.raw).finish()
    }
}

/// Glob over the canonical action subject (`*` also matches `/` and spaces).
#[derive(Clone)]
pub struct ArgPattern {
    raw: String,
    matcher: GlobMatcher,
}

impl ArgPattern {
    /// Parse an argument pattern.
    ///
    /// # Errors
    ///
    /// Returns [`RulesError::Pattern`] when the glob does not compile.
    pub fn new(raw: impl Into<String>) -> Result<Self, RulesError> {
        let raw = raw.into();
        let matcher = compile(&raw)?;
        Ok(Self { raw, matcher })
    }

    /// The pattern as written.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Whether `subject` matches.
    pub fn matches(&self, subject: &str) -> bool {
        self.matcher.is_match(subject)
    }
}

impl PartialEq for ArgPattern {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for ArgPattern {}

impl fmt::Debug for ArgPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ArgPattern").field(&self.raw).finish()
    }
}

fn has_glob_meta(s: &str) -> bool {
    s.contains(['*', '?', '[', '{'])
}

fn compile(pattern: &str) -> Result<GlobMatcher, RulesError> {
    let glob: Glob = GlobBuilder::new(pattern)
        .literal_separator(false)
        .backslash_escape(true)
        .build()
        .map_err(|source| RulesError::Pattern {
            pattern: pattern.to_string(),
            source,
        })?;
    Ok(glob.compile_matcher())
}

/// One allow or deny rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// Tool-name matcher.
    pub tool: ToolPattern,
    /// Optional subject matcher; `None` matches every invocation of the tool.
    pub arg: Option<ArgPattern>,
    /// Allow or deny.
    pub kind: RuleKind,
    /// Layer the rule came from.
    pub origin: RuleOrigin,
}

impl Rule {
    /// Build a rule from raw patterns.
    ///
    /// # Errors
    ///
    /// Returns [`RulesError`] when a pattern is invalid.
    pub fn new(
        kind: RuleKind,
        tool: &str,
        arg: Option<&str>,
        origin: RuleOrigin,
    ) -> Result<Self, RulesError> {
        Ok(Self {
            tool: ToolPattern::new(tool)?,
            arg: arg.map(ArgPattern::new).transpose()?,
            kind,
            origin,
        })
    }

    /// Whether this rule matches the tool and subjects.
    ///
    /// Deny rules match when **any** subject matches; allow rules require
    /// **every** subject to match (an `apply_patch` touching two files is
    /// allowed only when both are covered).
    pub fn matches(&self, tool: &str, subjects: &[String]) -> bool {
        if !self.tool.matches(tool) {
            return false;
        }
        let Some(arg) = &self.arg else {
            return true;
        };
        match self.kind {
            RuleKind::Deny => subjects.iter().any(|s| arg.matches(s)),
            RuleKind::Allow => !subjects.is_empty() && subjects.iter().all(|s| arg.matches(s)),
        }
    }

    /// The serialized form.
    pub fn to_entry(&self) -> RuleEntry {
        RuleEntry {
            tool: self.tool.as_str().to_string(),
            arg: self.arg.as_ref().map(|a| a.as_str().to_string()),
        }
    }

    /// Why this rule is too broad to keep in auto mode, if it is.
    pub fn broad_reason(&self) -> Option<&'static str> {
        if self.kind != RuleKind::Allow {
            return None;
        }
        let tool = self.tool.as_str();
        if tool.trim_matches('*').is_empty() {
            return Some("matches every tool");
        }
        if let Some(rest) = tool.strip_prefix("mcp__")
            && !self.tool.is_exact()
        {
            let tool_part = rest.split_once("__").map_or("", |(_, t)| t);
            if tool_part.trim_matches('*').is_empty() {
                return Some("matches every tool of an MCP server");
            }
        }
        if !self.tool.matches("shell") {
            return None;
        }
        let Some(arg) = &self.arg else {
            return Some("allows every shell command");
        };
        broad_shell_arg(arg.as_str())
    }
}

/// Interpreters whose wildcard allow rules amount to arbitrary code execution.
const INTERPRETER_PREFIXES: &[&str] = &[
    "python", "node", "bash", "sh", "zsh", "ruby", "perl", "deno", "bun",
];

/// Task runners that execute project-defined scripts.
const RUNNER_PREFIXES: &[&[&str]] = &[
    &["npm", "run"],
    &["npm", "exec"],
    &["npx"],
    &["pnpm"],
    &["yarn"],
    &["cargo", "run"],
    &["make"],
];

fn broad_shell_arg(arg: &str) -> Option<&'static str> {
    let trimmed = arg.trim();
    if trimmed.is_empty() || trimmed.starts_with(['*', '?', '[', '{']) {
        return Some("matches any shell command");
    }
    let mut words = trimmed.split_whitespace();
    let first = words.next().unwrap_or_default();
    let first_bare = first.trim_end_matches(['*', '?']);
    let wildcard = has_glob_meta(trimmed);
    if wildcard
        && INTERPRETER_PREFIXES.iter().any(|p| {
            if *p == "python" {
                first_bare.starts_with(p)
            } else {
                first_bare == *p
            }
        })
    {
        return Some("runs an interpreter on arbitrary input");
    }
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    for prefix in RUNNER_PREFIXES {
        let matches = prefix.iter().enumerate().all(|(i, p)| {
            tokens
                .get(i)
                .is_some_and(|t| t.trim_end_matches(['*', '?']) == *p)
        });
        if matches {
            return Some("runs project-defined scripts");
        }
    }
    None
}

/// Identifies a rule inside a [`RuleSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuleId {
    /// Which list.
    pub kind: RuleKind,
    /// Index into that list.
    pub index: usize,
}

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            RuleKind::Allow => "allow",
            RuleKind::Deny => "deny",
        };
        write!(f, "{kind}#{}", self.index + 1)
    }
}

/// Merged rules from every layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuleSet {
    /// Allow rules.
    pub allow: Vec<Rule>,
    /// Deny rules.
    pub deny: Vec<Rule>,
}

impl RuleSet {
    /// First deny rule matching the action.
    pub fn matches_deny(&self, tool: &str, subjects: &[String]) -> Option<RuleId> {
        self.deny
            .iter()
            .position(|r| r.matches(tool, subjects))
            .map(|index| RuleId {
                kind: RuleKind::Deny,
                index,
            })
    }

    /// First allow rule matching the action.
    pub fn matches_allow(&self, tool: &str, subjects: &[String]) -> Option<RuleId> {
        self.allow
            .iter()
            .position(|r| r.matches(tool, subjects))
            .map(|index| RuleId {
                kind: RuleKind::Allow,
                index,
            })
    }

    /// Look up a rule by id.
    pub fn get(&self, id: RuleId) -> Option<&Rule> {
        match id.kind {
            RuleKind::Allow => self.allow.get(id.index),
            RuleKind::Deny => self.deny.get(id.index),
        }
    }

    /// Add a rule to the matching list (ignores exact duplicates).
    pub fn push(&mut self, rule: Rule) {
        let list = match rule.kind {
            RuleKind::Allow => &mut self.allow,
            RuleKind::Deny => &mut self.deny,
        };
        let duplicate = list
            .iter()
            .any(|r| r.tool == rule.tool && r.arg == rule.arg);
        if !duplicate {
            list.push(rule);
        }
    }

    /// Remove a rule by id.
    pub fn remove(&mut self, id: RuleId) -> Option<Rule> {
        let list = match id.kind {
            RuleKind::Allow => &mut self.allow,
            RuleKind::Deny => &mut self.deny,
        };
        (id.index < list.len()).then(|| list.remove(id.index))
    }

    /// Drop allow rules too broad for auto mode; returns the dropped rules.
    pub fn sanitize_for_auto(&mut self) -> Vec<Rule> {
        let (keep, dropped): (Vec<Rule>, Vec<Rule>) = std::mem::take(&mut self.allow)
            .into_iter()
            .partition(|r| r.broad_reason().is_none());
        self.allow = keep;
        dropped
    }

    /// Rules from one layer, as a serializable file body (without `[auto]`).
    pub fn layer_entries(&self, origin: RuleOrigin) -> (Vec<RuleEntry>, Vec<RuleEntry>) {
        let pick = |rules: &[Rule]| {
            rules
                .iter()
                .filter(|r| r.origin == origin)
                .map(Rule::to_entry)
                .collect()
        };
        (pick(&self.allow), pick(&self.deny))
    }
}

/// Auto-mode settings after layering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoSettings {
    /// Whether auto mode is on; when off, Tier-3 actions are denied outright.
    pub enabled: bool,
    /// Classifier model override.
    pub classifier_model: Option<String>,
    /// Whether `cargo build|test|nextest|check|clippy` are Tier 1.
    pub trust_project_builds: bool,
}

impl Default for AutoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            classifier_model: None,
            trust_project_builds: true,
        }
    }
}

/// `[auto]` section of a rules file. Unset fields inherit from lower layers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoSection {
    /// See [`AutoSettings::enabled`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// See [`AutoSettings::classifier_model`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classifier_model: Option<String>,
    /// See [`AutoSettings::trust_project_builds`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_project_builds: Option<bool>,
}

/// One `[[allow]]` / `[[deny]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleEntry {
    /// Tool name or glob.
    pub tool: String,
    /// Optional subject glob.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arg: Option<String>,
}

/// On-disk rules file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulesFile {
    /// Auto-mode settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto: Option<AutoSection>,
    /// Allow rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<RuleEntry>,
    /// Deny rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<RuleEntry>,
}

impl RulesFile {
    /// Load a rules file; a missing file yields an empty one.
    ///
    /// # Errors
    ///
    /// Returns [`RulesError`] on I/O or parse failures.
    pub fn load(path: &Path) -> Result<Self, RulesError> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|source| RulesError::Parse {
                path: path.to_path_buf(),
                source,
            }),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(RulesError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    fn rules(&self, origin: RuleOrigin) -> Result<Vec<Rule>, RulesError> {
        let allow = self
            .allow
            .iter()
            .map(|e| Rule::new(RuleKind::Allow, &e.tool, e.arg.as_deref(), origin));
        let deny = self
            .deny
            .iter()
            .map(|e| Rule::new(RuleKind::Deny, &e.tool, e.arg.as_deref(), origin));
        allow.chain(deny).collect()
    }
}

/// Non-fatal observations from loading rules, shown to the user once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Notice {
    /// The project-level file contained allow rules, which are ignored.
    ProjectAllowIgnored {
        /// Project rules file.
        path: PathBuf,
        /// Number of ignored entries.
        count: usize,
    },
    /// The project-level file tried to enable `trust_project_builds` against a user setting.
    ProjectTrustIgnored {
        /// Project rules file.
        path: PathBuf,
    },
    /// The realm rules file is tracked by git (it came with the repository), so
    /// its allow rules are ignored.
    TrackedRealmAllowIgnored {
        /// Realm rules file.
        path: PathBuf,
        /// Number of ignored allow rules.
        count: usize,
    },
}

impl fmt::Display for Notice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectAllowIgnored { path, count } => write!(
                f,
                "ignored {count} allow rule(s) in {}: project rules files may only add deny rules",
                path.display()
            ),
            Self::ProjectTrustIgnored { path } => write!(
                f,
                "ignored trust_project_builds=true in {}: a project file cannot loosen your setting",
                path.display()
            ),
            Self::TrackedRealmAllowIgnored { path, count } => write!(
                f,
                "{} is tracked by git; ignored {count} allow rule(s) from it",
                path.display()
            ),
        }
    }
}

/// Load and merge the three rule layers.
///
/// Settings precedence (later wins): global, realm, project. The project layer
/// contributes only deny rules and `[auto]` settings, and cannot turn
/// `trust_project_builds` on when a user layer turned it off.
///
/// # Errors
///
/// Returns [`RulesError`] when a file cannot be read or parsed.
pub fn load_layered(
    realm_dir: &Path,
    home_rkat_dir: Option<&Path>,
    project_rkat_dir: &Path,
) -> Result<(RuleSet, AutoSettings, Vec<Notice>), RulesError> {
    let mut rules = RuleSet::default();
    let mut settings = AutoSettings::default();
    let mut notices = Vec::new();

    let mut user_layers = Vec::new();
    if let Some(home) = home_rkat_dir {
        user_layers.push((home.join(RULES_FILE_NAME), RuleOrigin::Global));
    }
    user_layers.push((realm_dir.join(RULES_FILE_NAME), RuleOrigin::Realm));

    let mut user_trust: Option<bool> = None;
    for (path, origin) in user_layers {
        let file = RulesFile::load(&path)?;
        for rule in file.rules(origin)? {
            rules.push(rule);
        }
        if let Some(auto) = &file.auto {
            apply_auto(&mut settings, auto);
            if auto.trust_project_builds.is_some() {
                user_trust = auto.trust_project_builds;
            }
        }
    }

    let project_path = project_rkat_dir.join(RULES_FILE_NAME);
    let project = RulesFile::load(&project_path)?;
    if !project.allow.is_empty() {
        notices.push(Notice::ProjectAllowIgnored {
            path: project_path.clone(),
            count: project.allow.len(),
        });
    }
    for entry in &project.deny {
        rules.push(Rule::new(
            RuleKind::Deny,
            &entry.tool,
            entry.arg.as_deref(),
            RuleOrigin::Project,
        )?);
    }
    if let Some(mut auto) = project.auto {
        if auto.trust_project_builds == Some(true) && user_trust == Some(false) {
            auto.trust_project_builds = None;
            notices.push(Notice::ProjectTrustIgnored { path: project_path });
        }
        apply_auto(&mut settings, &auto);
    }
    Ok((rules, settings, notices))
}

fn apply_auto(settings: &mut AutoSettings, auto: &AutoSection) {
    if let Some(enabled) = auto.enabled {
        settings.enabled = enabled;
    }
    if let Some(model) = &auto.classifier_model {
        settings.classifier_model = Some(model.clone());
    }
    if let Some(trust) = auto.trust_project_builds {
        settings.trust_project_builds = trust;
    }
}

/// Atomically write a rules file (temp file in the same directory + rename).
///
/// # Errors
///
/// Returns [`RulesError`] on serialization or I/O failures.
pub fn save_realm(path: &Path, file: &RulesFile) -> Result<(), RulesError> {
    let text = toml::to_string_pretty(file)?;
    let io = |source| RulesError::Io {
        path: path.to_path_buf(),
        source,
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(format!(".{}.tmp", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    std::fs::write(&tmp, text).map_err(io)?;
    std::fs::rename(&tmp, path).map_err(|source| {
        let _ = std::fs::remove_file(&tmp);
        RulesError::Io {
            path: path.to_path_buf(),
            source,
        }
    })
}

/// Rewrite the rule lists of the realm file from the realm/session rules in
/// `rules`, preserving its `[auto]` section.
///
/// # Errors
///
/// Returns [`RulesError`] on I/O, parse or serialization failures.
pub fn persist_realm_rules(path: &Path, rules: &RuleSet) -> Result<(), RulesError> {
    let mut file = RulesFile::load(path)?;
    let (mut allow, mut deny) = rules.layer_entries(RuleOrigin::Realm);
    let (session_allow, session_deny) = rules.layer_entries(RuleOrigin::Session);
    allow.extend(session_allow);
    deny.extend(session_deny);
    file.allow = allow;
    file.deny = deny;
    save_realm(path, &file)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn allow(tool: &str, arg: Option<&str>) -> Rule {
        Rule::new(RuleKind::Allow, tool, arg, RuleOrigin::Realm).unwrap()
    }

    #[test]
    fn tool_patterns() {
        let exact = ToolPattern::new("shell").unwrap();
        assert!(exact.is_exact());
        assert!(exact.matches("shell"));
        assert!(!exact.matches("shell_jobs"));
        let glob = ToolPattern::new("mcp__github__*").unwrap();
        assert!(glob.matches("mcp__github__create_issue"));
        assert!(!glob.matches("mcp__gitlab__x"));
        assert!(ToolPattern::new("  ").is_err());
    }

    #[test]
    fn arg_patterns_span_separators() {
        let p = ArgPattern::new("cargo nextest run*").unwrap();
        assert!(p.matches("cargo nextest run -p meerkat-tui --lib"));
        assert!(!p.matches("cargo run"));
        let path = ArgPattern::new("docs/**").unwrap();
        assert!(path.matches("docs/a/b.md"));
    }

    #[test]
    fn allow_requires_all_subjects_deny_any() {
        let a = allow("apply_patch", Some("src/*"));
        let subjects = vec!["src/a.rs".to_string(), "README.md".to_string()];
        assert!(!a.matches("apply_patch", &subjects));
        let d = Rule::new(
            RuleKind::Deny,
            "apply_patch",
            Some("src/*"),
            RuleOrigin::Realm,
        )
        .unwrap();
        assert!(d.matches("apply_patch", &subjects));
    }

    #[test]
    fn ruleset_ids_and_matching() {
        let mut set = RuleSet::default();
        set.push(
            Rule::new(
                RuleKind::Deny,
                "shell",
                Some("git push*"),
                RuleOrigin::Global,
            )
            .unwrap(),
        );
        set.push(allow("shell", Some("make lint")));
        set.push(allow("shell", Some("make lint")));
        assert_eq!(set.allow.len(), 1, "duplicates are ignored");
        let id = set
            .matches_deny("shell", &["git push --force".into()])
            .unwrap();
        assert_eq!(id.to_string(), "deny#1");
        assert!(set.get(id).is_some());
        assert!(set.matches_deny("shell", &["git status".into()]).is_none());
        assert!(set.remove(id).is_some());
        assert!(set.deny.is_empty());
    }

    #[test]
    fn sanitizer_drops_broad_rules() {
        let mut set = RuleSet::default();
        let broad = [
            allow("shell", None),
            allow("shell", Some("*")),
            allow("shell", Some("python*")),
            allow("shell", Some("python3 *")),
            allow("shell", Some("node *")),
            allow("shell", Some("bash *")),
            allow("shell", Some("sh -c *")),
            allow("shell", Some("ruby *")),
            allow("shell", Some("perl *")),
            allow("shell", Some("deno *")),
            allow("shell", Some("bun *")),
            allow("shell", Some("npm run *")),
            allow("shell", Some("npx prettier")),
            allow("shell", Some("pnpm test")),
            allow("shell", Some("yarn *")),
            allow("shell", Some("cargo run")),
            allow("shell", Some("make")),
            allow("shell", Some("make *")),
            allow("mcp__*", None),
            allow("mcp__github__*", None),
            allow("*", None),
            allow("sh*", None),
        ];
        let narrow = [
            allow("shell", Some("cargo nextest run*")),
            allow("shell", Some("python3 scripts/gen.py")),
            allow("shell", Some("npm test")),
            allow("mcp__github__list_issues", None),
            allow("read_file", Some("/opt/docs/**")),
        ];
        for r in broad.iter().chain(narrow.iter()) {
            set.push(r.clone());
        }
        set.push(Rule::new(RuleKind::Deny, "shell", None, RuleOrigin::Realm).unwrap());
        let dropped = set.sanitize_for_auto();
        assert_eq!(dropped.len(), broad.len(), "dropped: {dropped:#?}");
        assert_eq!(set.allow.len(), narrow.len());
        assert_eq!(set.deny.len(), 1, "deny rules are never sanitized");
    }

    #[test]
    fn rules_file_round_trip() {
        let file = RulesFile {
            auto: Some(AutoSection {
                enabled: Some(true),
                classifier_model: Some("claude-sonnet-4-6".into()),
                trust_project_builds: Some(false),
            }),
            allow: vec![RuleEntry {
                tool: "shell".into(),
                arg: Some("cargo nextest run*".into()),
            }],
            deny: vec![RuleEntry {
                tool: "mcp__prod__*".into(),
                arg: None,
            }],
        };
        let text = toml::to_string_pretty(&file).unwrap();
        let back: RulesFile = toml::from_str(&text).unwrap();
        assert_eq!(back, file);
        let parsed: RulesFile = toml::from_str(
            "[auto]\nenabled = true\n\n[[allow]]\ntool = \"shell\"\narg = \"cargo nextest run*\"\n",
        )
        .unwrap();
        assert_eq!(parsed.allow.len(), 1);
        assert!(toml::from_str::<RulesFile>("[[allow]]\ntool=\"x\"\nbogus=1\n").is_err());
    }

    #[test]
    fn layered_loading() {
        let tmp = tempfile::tempdir().unwrap();
        let realm = tmp.path().join("realm");
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        std::fs::create_dir_all(&realm).unwrap();
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(
            home.join(RULES_FILE_NAME),
            "[auto]\ntrust_project_builds = false\n[[deny]]\ntool = \"shell\"\narg = \"git push*\"\n",
        )
        .unwrap();
        std::fs::write(
            realm.join(RULES_FILE_NAME),
            "[auto]\nclassifier_model = \"gpt-5.2\"\n[[allow]]\ntool = \"shell\"\narg = \"make lint\"\n",
        )
        .unwrap();
        std::fs::write(
            project.join(RULES_FILE_NAME),
            "[auto]\ntrust_project_builds = true\nenabled = false\n[[allow]]\ntool = \"shell\"\n[[deny]]\ntool = \"agent_spawn\"\n",
        )
        .unwrap();
        let (rules, settings, notices) = load_layered(&realm, Some(&home), &project).unwrap();
        assert_eq!(rules.allow.len(), 1);
        assert_eq!(rules.allow[0].origin, RuleOrigin::Realm);
        assert_eq!(rules.deny.len(), 2);
        assert!(!settings.trust_project_builds);
        assert!(!settings.enabled);
        assert_eq!(settings.classifier_model.as_deref(), Some("gpt-5.2"));
        assert_eq!(notices.len(), 2);
        assert!(
            notices
                .iter()
                .any(|n| matches!(n, Notice::ProjectAllowIgnored { count: 1, .. }))
        );
    }

    #[test]
    fn missing_files_give_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let (rules, settings, notices) =
            load_layered(tmp.path(), None, &tmp.path().join("nope")).unwrap();
        assert_eq!(rules, RuleSet::default());
        assert_eq!(settings, AutoSettings::default());
        assert!(notices.is_empty());
    }

    #[test]
    fn invalid_file_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join(RULES_FILE_NAME),
            "[[allow]]\ntool = \"[\"\n",
        )
        .unwrap();
        assert!(matches!(
            load_layered(tmp.path(), None, tmp.path()),
            Err(RulesError::Pattern { .. })
        ));
        std::fs::write(tmp.path().join(RULES_FILE_NAME), "not toml [").unwrap();
        assert!(matches!(
            load_layered(tmp.path(), None, tmp.path()),
            Err(RulesError::Parse { .. })
        ));
    }

    #[test]
    fn save_and_persist_preserve_auto_section() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("deep/realm").join(RULES_FILE_NAME);
        save_realm(
            &path,
            &RulesFile {
                auto: Some(AutoSection {
                    enabled: Some(false),
                    ..AutoSection::default()
                }),
                ..RulesFile::default()
            },
        )
        .unwrap();
        let mut set = RuleSet::default();
        set.push(allow("shell", Some("make lint")));
        set.push(Rule::new(RuleKind::Deny, "shell", Some("rm *"), RuleOrigin::Session).unwrap());
        set.push(Rule::new(RuleKind::Deny, "shell", Some("x"), RuleOrigin::Project).unwrap());
        persist_realm_rules(&path, &set).unwrap();
        let back = RulesFile::load(&path).unwrap();
        assert_eq!(back.auto.unwrap().enabled, Some(false));
        assert_eq!(back.allow.len(), 1);
        assert_eq!(
            back.deny.len(),
            1,
            "project rules are not copied into the realm file"
        );
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }
}
