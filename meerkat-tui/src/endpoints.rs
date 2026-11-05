//! LLM endpoints: the built-in provider endpoints plus any `[endpoints.<name>]`
//! from `tui.toml` (e.g. a LiteLLM proxy), and `endpoint:model` references.
//!
//! An [`Endpoint`] is a named place to send requests in one of Meerkat's
//! client [`Dialect`]s. The built-in `anthropic`, `openai` and `gemini`
//! endpoints exist when the realm config or the environment has a key for
//! them; configured endpoints come from the layered `tui.toml`.
//!
//! The endpoint/model choice of every session is kept in a small sidecar file
//! (see [`save_choice`]): Meerkat persists the session's model, but neither
//! the endpoint nor a model switched mid-session.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use meerkat::{AgentFactory, LlmClient};
use meerkat_core::{Config, Provider};
use serde::{Deserialize, Serialize};

use crate::approval::rules::{EndpointSection, LayeredEndpoint, Notice, RuleOrigin};
use crate::bootstrap::credentials::{ProviderCredentials, resolve_provider_credentials};
use crate::error::TuiError;
use crate::models::ModelEntry;

/// Names of the built-in endpoints, in display order.
pub const BUILTIN_NAMES: [&str; 3] = ["anthropic", "openai", "gemini"];

/// Directory (inside the realm root) of the per-session endpoint sidecars.
const CHOICE_DIR: &str = "tui_endpoints";

/// The wire protocol an endpoint speaks (one of Meerkat's LLM clients).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI Responses API (`/v1/responses`); also LiteLLM and other proxies.
    OpenAi,
    /// Gemini `generateContent`.
    Gemini,
}

impl Dialect {
    /// Every dialect, in display order.
    pub const ALL: [Self; 3] = [Self::Anthropic, Self::OpenAi, Self::Gemini];

    /// Canonical name (`anthropic`, `openai`, `gemini`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Gemini => "gemini",
        }
    }

    /// Parse a canonical name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.as_str() == name)
    }

    /// The dialect of a concrete provider.
    #[must_use]
    pub fn from_provider(provider: Provider) -> Option<Self> {
        match provider {
            Provider::Anthropic => Some(Self::Anthropic),
            Provider::OpenAI => Some(Self::OpenAi),
            Provider::Gemini => Some(Self::Gemini),
            Provider::Other => None,
        }
    }

    /// Infer the dialect from a model id prefix: `claude-*` → Anthropic,
    /// `gpt-*`, `chatgpt-*`, `o<digit>*` → OpenAI, `gemini-*` → Gemini.
    #[must_use]
    pub fn infer_from_model(model: &str) -> Option<Self> {
        let lower = model.to_ascii_lowercase();
        if lower.starts_with("claude-") {
            Some(Self::Anthropic)
        } else if lower.starts_with("gpt-")
            || lower.starts_with("chatgpt-")
            || (lower.starts_with('o') && lower[1..].starts_with(|c: char| c.is_ascii_digit()))
        {
            Some(Self::OpenAi)
        } else if lower.starts_with("gemini-") {
            Some(Self::Gemini)
        } else {
            None
        }
    }
}

impl From<Dialect> for Provider {
    fn from(dialect: Dialect) -> Self {
        match dialect {
            Dialect::Anthropic => Self::Anthropic,
            Dialect::OpenAi => Self::OpenAI,
            Dialect::Gemini => Self::Gemini,
        }
    }
}

impl fmt::Display for Dialect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A secret that never shows up in `Debug` output and has no `Display`.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    /// Wrap a secret.
    #[must_use]
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    /// The secret itself; only for request headers and client construction.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// One place to send LLM requests.
#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Unique name (`anthropic`, `litellm`, …); the `endpoint` of `endpoint:model`.
    pub name: String,
    /// Protocol the endpoint speaks.
    pub dialect: Dialect,
    /// Base URL override; `None` uses the dialect's public API.
    pub base_url: Option<String>,
    /// API key.
    pub api_key: SecretString,
    /// Whether this is one of the built-in provider endpoints.
    pub builtin: bool,
    /// Tie-break when several endpoints list the same bare model id
    /// (higher wins; built-ins are 0).
    pub priority: i64,
}

impl Endpoint {
    /// Credentials for the list-models request.
    #[must_use]
    pub fn credentials(&self) -> ProviderCredentials {
        ProviderCredentials {
            provider: self.dialect.into(),
            api_key: self.api_key.expose().to_string(),
            base_url: self.base_url.clone(),
        }
    }

    /// Build a Meerkat LLM client for this endpoint.
    ///
    /// # Errors
    ///
    /// [`TuiError::LlmClient`] when the client cannot be constructed.
    pub async fn build_client(
        &self,
        factory: &AgentFactory,
    ) -> Result<Arc<dyn LlmClient>, TuiError> {
        Ok(factory
            .build_llm_client(
                self.dialect.into(),
                Some(self.api_key.expose().to_string()),
                self.base_url.clone(),
            )
            .await?)
    }

    /// `anthropic dialect` or `openai dialect, http://localhost:4000`.
    #[must_use]
    pub fn describe(&self) -> String {
        describe_endpoint(self.dialect, self.base_url.as_deref())
    }
}

/// `<dialect> dialect[, <base url>]`.
#[must_use]
pub fn describe_endpoint(dialect: Dialect, base_url: Option<&str>) -> String {
    match base_url {
        Some(url) => format!("{dialect} dialect, {url}"),
        None => format!("{dialect} dialect"),
    }
}

/// A model on a named endpoint; displayed (and parsed) as `endpoint:model`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointRef {
    /// Endpoint name.
    pub endpoint: String,
    /// Model id as the endpoint knows it.
    pub model: String,
}

impl EndpointRef {
    /// A reference to `model` on `endpoint`.
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
        }
    }
}

impl fmt::Display for EndpointRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.endpoint, self.model)
    }
}

/// Endpoint names, comma-separated (`none` when empty).
struct NameList<'a>(&'a [String]);

impl fmt::Display for NameList<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("none");
        }
        for (i, name) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(name)?;
        }
        Ok(())
    }
}

/// A model reference that does not resolve to one endpoint.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelRefError {
    /// The reference was empty (or `endpoint:` without a model).
    #[error("empty model reference")]
    Empty,
    /// No endpoint is configured at all.
    #[error(
        "no LLM endpoint is configured: set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY, or add [endpoints.<name>] to tui.toml"
    )]
    NoEndpoints,
    /// `name:` (or `--endpoint name`) names no configured endpoint.
    #[error("unknown endpoint `{name}` (configured: {})", NameList(.known))]
    UnknownEndpoint {
        /// The name given.
        name: String,
        /// Configured endpoint names.
        known: Vec<String>,
    },
    /// A bare model id that could belong to several endpoints (or none).
    #[error(
        "cannot tell which endpoint serves `{model}`; use `<endpoint>:{model}` (configured: {})",
        NameList(.known)
    )]
    Ambiguous {
        /// The model id.
        model: String,
        /// Configured endpoint names.
        known: Vec<String>,
    },
    /// No model was given and none could be chosen.
    #[error(
        "no model selected and none can be chosen automatically; pass --model <endpoint>:<id> (configured: {})",
        NameList(.known)
    )]
    NoModel {
        /// Configured endpoint names.
        known: Vec<String>,
    },
}

impl From<ModelRefError> for TuiError {
    fn from(err: ModelRefError) -> Self {
        Self::Endpoint(err.to_string())
    }
}

fn names(endpoints: &[Endpoint]) -> Vec<String> {
    endpoints.iter().map(|e| e.name.clone()).collect()
}

/// Split `name:model` when `name` is a configured endpoint.
fn split_explicit<'a>(s: &'a str, endpoints: &[Endpoint]) -> Option<(&'a str, &'a str)> {
    let (name, model) = s.split_once(':')?;
    endpoints
        .iter()
        .any(|e| e.name == name)
        .then_some((name, model))
}

/// The endpoint serving a bare id by dialect: the built-in endpoint of the
/// inferred dialect when configured, else the only endpoint of that dialect.
fn by_dialect<'a>(model: &str, endpoints: &'a [Endpoint]) -> Option<&'a Endpoint> {
    let dialect = Dialect::infer_from_model(model)?;
    if let Some(builtin) = endpoints.iter().find(|e| e.builtin && e.dialect == dialect) {
        return Some(builtin);
    }
    let mut same = endpoints.iter().filter(|e| e.dialect == dialect);
    match (same.next(), same.next()) {
        (Some(only), None) => Some(only),
        _ => None,
    }
}

/// Which endpoints list which model ids (from the fetched model lists).
#[derive(Debug, Clone, Default)]
pub struct ModelIndex {
    /// Model id → endpoints listing it, in insertion order.
    listed: HashMap<String, Vec<String>>,
}

impl ModelIndex {
    /// An index over `(endpoint, models)` pairs.
    pub fn from_lists<'a>(lists: impl IntoIterator<Item = (&'a str, &'a [ModelEntry])>) -> Self {
        let mut index = Self::default();
        for (endpoint, models) in lists {
            index.insert(endpoint, models.iter().map(|m| m.id.as_str()));
        }
        index
    }

    /// Record that `endpoint` lists `ids`.
    pub fn insert<'a>(&mut self, endpoint: &str, ids: impl IntoIterator<Item = &'a str>) {
        for id in ids {
            let entry = self.listed.entry(id.to_string()).or_default();
            if !entry.iter().any(|e| e == endpoint) {
                entry.push(endpoint.to_string());
            }
        }
    }

    /// Whether `endpoint` lists `id`.
    #[must_use]
    pub fn lists(&self, endpoint: &str, id: &str) -> bool {
        self.listed
            .get(id)
            .is_some_and(|endpoints| endpoints.iter().any(|e| e == endpoint))
    }

    /// Whether nothing is indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.listed.is_empty()
    }
}

/// Whether resolving `s` would consult the model lists: it is a bare id and
/// more than one endpoint could serve it.
#[must_use]
pub fn needs_lookup(s: &str, endpoints: &[Endpoint]) -> bool {
    endpoints.len() > 1 && split_explicit(s.trim(), endpoints).is_none()
}

/// Pick among several endpoints listing the same id: the current endpoint,
/// then the built-in endpoint of the id's dialect, then the highest
/// `priority`, then the first configured.
fn tie_break<'a>(
    model: &str,
    candidates: &[&'a Endpoint],
    current: Option<&str>,
) -> Option<&'a Endpoint> {
    if let Some(current) = current
        && let Some(found) = candidates.iter().find(|e| e.name == current)
    {
        return Some(found);
    }
    if let Some(dialect) = Dialect::infer_from_model(model)
        && let Some(found) = candidates
            .iter()
            .find(|e| e.builtin && e.dialect == dialect)
    {
        return Some(found);
    }
    let best = candidates.iter().map(|e| e.priority).max()?;
    candidates.iter().find(|e| e.priority == best).copied()
}

/// Parse a model reference.
///
/// `name:id` is explicit when `name` is a configured endpoint (model ids may
/// themselves contain `:`). A bare id is looked up in `index` (the fetched
/// model lists): one endpoint listing it → that endpoint; several → the
/// `current` endpoint, then the built-in endpoint of the id's dialect, then
/// the highest `priority`, then the first configured. When no endpoint lists
/// it, the prefix decides (`claude-*` → anthropic, `gpt-*`/`o<digit>*` →
/// openai, `gemini-*` → gemini: the built-in endpoint of that dialect, else
/// the only endpoint of that dialect), else the only configured endpoint.
///
/// # Errors
///
/// [`ModelRefError`] when the reference is empty or does not pick exactly one
/// endpoint; the error lists the configured endpoints.
pub fn parse_model_ref(
    s: &str,
    endpoints: &[Endpoint],
    index: &ModelIndex,
    current: Option<&str>,
) -> Result<EndpointRef, ModelRefError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ModelRefError::Empty);
    }
    if endpoints.is_empty() {
        return Err(ModelRefError::NoEndpoints);
    }
    if let Some((name, model)) = split_explicit(s, endpoints) {
        if model.is_empty() {
            return Err(ModelRefError::Empty);
        }
        return Ok(EndpointRef::new(name, model));
    }
    let listing: Vec<&Endpoint> = endpoints
        .iter()
        .filter(|e| index.lists(&e.name, s))
        .collect();
    if let Some(endpoint) = tie_break(s, &listing, current) {
        return Ok(EndpointRef::new(endpoint.name.clone(), s));
    }
    if let Some(endpoint) = by_dialect(s, endpoints) {
        return Ok(EndpointRef::new(endpoint.name.clone(), s));
    }
    if let [only] = endpoints {
        return Ok(EndpointRef::new(only.name.clone(), s));
    }
    // `foo:bar` where `foo` looks like an endpoint name the user forgot to configure.
    if let Some((name, _)) = s.split_once(':')
        && is_valid_endpoint_name(name)
    {
        return Err(ModelRefError::UnknownEndpoint {
            name: name.to_string(),
            known: names(endpoints),
        });
    }
    Err(ModelRefError::Ambiguous {
        model: s.to_string(),
        known: names(endpoints),
    })
}

/// Whether `name` is usable as an endpoint name: ASCII letters, digits, `_`, `-`.
#[must_use]
pub fn is_valid_endpoint_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// The endpoints available to this process.
#[derive(Debug, Clone, Default)]
pub struct EndpointRegistry {
    endpoints: Vec<Endpoint>,
}

impl EndpointRegistry {
    /// A registry over exactly these endpoints (tests, or a caller that
    /// resolved them itself).
    #[must_use]
    pub fn from_endpoints(endpoints: Vec<Endpoint>) -> Self {
        Self { endpoints }
    }

    /// The built-in endpoints whose key resolves (realm config
    /// `providers.api_keys`/legacy `[provider]`, else the provider's
    /// environment variable; base URL from `providers.base_urls`), then the
    /// layered `[endpoints.<name>]` of `tui.toml`.
    ///
    /// Invalid configured endpoints (unknown or missing dialect, no key,
    /// invalid name, a name clashing with a built-in) are skipped with a
    /// [`Notice`]; a plain-text `api_key` is used with a notice.
    #[must_use]
    pub fn discover(
        config: &Config,
        configured: &BTreeMap<String, LayeredEndpoint>,
    ) -> (Self, Vec<Notice>) {
        Self::discover_with(config, configured, |var| std::env::var(var).ok())
    }

    /// [`Self::discover`] with an explicit environment lookup for `api_key_env`.
    pub fn discover_with(
        config: &Config,
        configured: &BTreeMap<String, LayeredEndpoint>,
        env: impl Fn(&str) -> Option<String>,
    ) -> (Self, Vec<Notice>) {
        let mut endpoints = Vec::new();
        for dialect in Dialect::ALL {
            if let Ok(creds) = resolve_provider_credentials(config, dialect.into()) {
                endpoints.push(Endpoint {
                    name: dialect.as_str().to_string(),
                    dialect,
                    base_url: creds.base_url,
                    api_key: SecretString::new(creds.api_key),
                    builtin: true,
                    priority: 0,
                });
            }
        }
        let mut notices = Vec::new();
        for (name, layered) in configured {
            match configured_endpoint(name, layered, &env) {
                Ok((endpoint, notice)) => {
                    notices.extend(notice);
                    endpoints.push(endpoint);
                }
                Err(reason) => notices.push(Notice::EndpointSkipped {
                    name: name.clone(),
                    reason,
                }),
            }
        }
        (Self { endpoints }, notices)
    }

    /// The endpoint called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Endpoint> {
        self.endpoints.iter().find(|e| e.name == name)
    }

    /// Every endpoint: built-ins first, then configured ones by name.
    pub fn iter(&self) -> impl Iterator<Item = &Endpoint> {
        self.endpoints.iter()
    }

    /// The endpoints as a slice (for [`parse_model_ref`]).
    #[must_use]
    pub fn as_slice(&self) -> &[Endpoint] {
        &self.endpoints
    }

    /// Whether no endpoint is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }

    /// The endpoint used when nothing else selects one: the first built-in
    /// (anthropic, openai, gemini), else the first configured endpoint.
    #[must_use]
    pub fn default_endpoint(&self) -> Option<&Endpoint> {
        self.endpoints.first()
    }

    /// Endpoint names, in [`Self::iter`] order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        names(&self.endpoints)
    }
}

/// Validate one `[endpoints.<name>]` entry.
fn configured_endpoint(
    name: &str,
    layered: &LayeredEndpoint,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<(Endpoint, Option<Notice>), String> {
    let EndpointSection {
        dialect,
        base_url,
        api_key_env,
        api_key,
        priority,
    } = &layered.section;
    if !is_valid_endpoint_name(name) {
        return Err("names may only use letters, digits, `_` and `-`".to_string());
    }
    if BUILTIN_NAMES.contains(&name) {
        return Err(format!(
            "`{name}` is a built-in endpoint; configure its key and base URL in the realm config, or pick another name"
        ));
    }
    let dialect = match dialect.as_deref() {
        None => return Err("`dialect` is missing (anthropic, openai or gemini)".to_string()),
        Some(raw) => Dialect::parse(raw).ok_or_else(|| {
            format!("unknown dialect `{raw}` (expected anthropic, openai or gemini)")
        })?,
    };
    let mut notice = None;
    let key = match (api_key_env, api_key) {
        (Some(var), _) if layered.origin == RuleOrigin::Project => {
            return Err(format!(
                "a project tui.toml may not read keys from the environment (`api_key_env = \"{var}\"`); define the endpoint in your own tui.toml"
            ));
        }
        (Some(var), _) => env(var)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("environment variable `{var}` (api_key_env) is not set"))?,
        (None, Some(key)) => {
            notice = Some(Notice::EndpointPlainKey {
                name: name.to_string(),
            });
            key.clone()
        }
        (None, None) => {
            return Err("no key: set `api_key_env` (or `api_key`)".to_string());
        }
    };
    Ok((
        Endpoint {
            name: name.to_string(),
            dialect,
            base_url: base_url.clone(),
            api_key: SecretString::new(key),
            builtin: false,
            priority: priority.unwrap_or(0),
        },
        notice,
    ))
}

/// What the start-up flags and config ask for.
#[derive(Debug, Clone, Copy, Default)]
pub struct StartupRequest<'a> {
    /// `--model` (`endpoint:id` or a bare id).
    pub model: Option<&'a str>,
    /// `--endpoint` (or `--provider`): endpoint for a bare id and the default.
    pub endpoint: Option<&'a str>,
    /// The realm config's `agent.model` (may be empty).
    pub config_model: &'a str,
}

/// The start-up model, or the endpoint whose list must be fetched to pick one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupPlan {
    /// The model is known.
    Resolved(EndpointRef),
    /// Use the first model `endpoint` lists.
    FirstListed(String),
}

/// Decide the start-up model without network access.
///
/// - `--model` wins: explicit `endpoint:id`, else a bare id on `--endpoint`
///   when given, else [`parse_model_ref`] over `index` (fetch the lists first
///   when [`needs_lookup`] says so).
/// - `--endpoint` alone: a built-in endpoint uses the config model when its
///   dialect matches, else the catalog default; a configured endpoint uses
///   the first model it lists.
/// - Neither: the config model when its prefix picks an endpoint, else (one
///   endpoint) the catalog default of a built-in or the first listed model.
///
/// # Errors
///
/// [`ModelRefError`] (listing the endpoints) when nothing resolves.
pub fn plan_startup(
    request: StartupRequest<'_>,
    endpoints: &[Endpoint],
    index: &ModelIndex,
) -> Result<StartupPlan, ModelRefError> {
    if endpoints.is_empty() {
        return Err(ModelRefError::NoEndpoints);
    }
    let selected = request
        .endpoint
        .map(|name| {
            endpoints.iter().find(|e| e.name == name).ok_or_else(|| {
                ModelRefError::UnknownEndpoint {
                    name: name.to_string(),
                    known: names(endpoints),
                }
            })
        })
        .transpose()?;
    if let Some(model) = request.model {
        if let Some((name, id)) = split_explicit(model, endpoints) {
            if id.is_empty() {
                return Err(ModelRefError::Empty);
            }
            return Ok(StartupPlan::Resolved(EndpointRef::new(name, id)));
        }
        return match selected {
            Some(endpoint) if !model.trim().is_empty() => Ok(StartupPlan::Resolved(
                EndpointRef::new(endpoint.name.clone(), model.trim()),
            )),
            _ => parse_model_ref(model, endpoints, index, None).map(StartupPlan::Resolved),
        };
    }
    let config_model = request.config_model.trim();
    if let Some(endpoint) = selected {
        if !endpoint.builtin {
            return Ok(StartupPlan::FirstListed(endpoint.name.clone()));
        }
        return Ok(StartupPlan::Resolved(builtin_default(
            endpoint,
            config_model,
        )));
    }
    if !config_model.is_empty() {
        if let Some((name, id)) = split_explicit(config_model, endpoints)
            && !id.is_empty()
        {
            return Ok(StartupPlan::Resolved(EndpointRef::new(name, id)));
        }
        if let Some(endpoint) = by_dialect(config_model, endpoints) {
            return Ok(StartupPlan::Resolved(EndpointRef::new(
                endpoint.name.clone(),
                config_model,
            )));
        }
    }
    match endpoints {
        [only] if only.builtin => Ok(StartupPlan::Resolved(builtin_default(only, config_model))),
        [only] => Ok(StartupPlan::FirstListed(only.name.clone())),
        _ => Err(ModelRefError::NoModel {
            known: names(endpoints),
        }),
    }
}

/// The config model when it belongs to `endpoint`'s dialect, else the
/// catalog default for that dialect.
fn builtin_default(endpoint: &Endpoint, config_model: &str) -> EndpointRef {
    let model = if !config_model.is_empty()
        && Dialect::infer_from_model(config_model) == Some(endpoint.dialect)
    {
        config_model.to_string()
    } else {
        meerkat_models::default_model(endpoint.dialect.as_str())
            .unwrap_or_default()
            .to_string()
    };
    EndpointRef::new(endpoint.name.clone(), model)
}

/// Where a session's endpoint choice should come from on resume.
///
/// Inference here only follows the model id's prefix (no "only endpoint"
/// fallback), so a session never silently moves to an unrelated endpoint.
///
/// `stored` is the session's sidecar, `meta` its persisted provider and
/// model; `exists` says whether an endpoint name is usable. Returns the
/// choice and a notice when it had to be inferred or fell back.
pub fn restore_choice(
    stored: Option<EndpointRef>,
    meta: Option<(Provider, String)>,
    endpoints: &[Endpoint],
    exists: impl Fn(&str) -> bool,
    fallback: &EndpointRef,
) -> (EndpointRef, Option<String>) {
    if let Some(stored) = stored {
        if exists(&stored.endpoint) {
            return (stored, None);
        }
        return match by_dialect(&stored.model, endpoints) {
            Some(endpoint) => {
                let inferred = EndpointRef::new(endpoint.name.clone(), stored.model);
                let text = format!(
                    "endpoint `{}` of this session no longer exists; continuing on {inferred}",
                    stored.endpoint
                );
                (inferred, Some(text))
            }
            None => (
                fallback.clone(),
                Some(format!(
                    "endpoint `{}` of this session no longer exists; continuing on {fallback}",
                    stored.endpoint
                )),
            ),
        };
    }
    let Some((provider, model)) = meta else {
        return (fallback.clone(), None);
    };
    if let Some(dialect) = Dialect::from_provider(provider)
        && exists(dialect.as_str())
    {
        return (EndpointRef::new(dialect.as_str(), model), None);
    }
    match by_dialect(&model, endpoints) {
        Some(endpoint) if exists(&endpoint.name) => {
            (EndpointRef::new(endpoint.name.clone(), model), None)
        }
        _ => (
            fallback.clone(),
            Some(format!(
                "this session ran on {} model `{model}`, which no configured endpoint serves; continuing on {fallback}",
                provider.as_str()
            )),
        ),
    }
}

/// On-disk form of a session's endpoint choice.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChoiceFile {
    endpoint: String,
    model: String,
}

fn choice_path(realm_root: &Path, session_id: &str) -> PathBuf {
    realm_root
        .join(CHOICE_DIR)
        .join(format!("{session_id}.toml"))
}

/// Record that session `session_id` runs on `choice`
/// (`<realm>/tui_endpoints/<id>.toml`).
///
/// # Errors
///
/// I/O or serialization failures.
pub fn save_choice(
    realm_root: &Path,
    session_id: &str,
    choice: &EndpointRef,
) -> Result<(), TuiError> {
    let path = choice_path(realm_root, session_id);
    let text = toml::to_string(&ChoiceFile {
        endpoint: choice.endpoint.clone(),
        model: choice.model.clone(),
    })
    .map_err(|e| TuiError::Endpoint(format!("serialize endpoint choice: {e}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| TuiError::io("create endpoint choice directory", e))?;
    }
    std::fs::write(&path, text).map_err(|e| TuiError::io("write endpoint choice", e))
}

/// The recorded choice of session `session_id`, if any (unreadable files
/// count as missing).
#[must_use]
pub fn load_choice(realm_root: &Path, session_id: &str) -> Option<EndpointRef> {
    let text = std::fs::read_to_string(choice_path(realm_root, session_id)).ok()?;
    match toml::from_str::<ChoiceFile>(&text) {
        Ok(file) => Some(EndpointRef::new(file.endpoint, file.model)),
        Err(err) => {
            tracing::warn!(error = %err, session_id, "ignoring unreadable endpoint choice");
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn ep(name: &str, dialect: Dialect, builtin: bool) -> Endpoint {
        Endpoint {
            name: name.to_string(),
            dialect,
            base_url: None,
            api_key: SecretString::new("k"),
            builtin,
            priority: 0,
        }
    }

    fn r(endpoint: &str, model: &str) -> EndpointRef {
        EndpointRef::new(endpoint, model)
    }

    #[test]
    fn dialect_inference() {
        let cases = [
            ("claude-opus-4-6", Some(Dialect::Anthropic)),
            ("gpt-5.4", Some(Dialect::OpenAi)),
            ("chatgpt-4o-latest", Some(Dialect::OpenAi)),
            ("o3", Some(Dialect::OpenAi)),
            ("o4-mini", Some(Dialect::OpenAi)),
            ("gemini-3.1-pro-preview", Some(Dialect::Gemini)),
            ("ollama", None),
            ("o", None),
            ("llama3:8b", None),
        ];
        for (model, expected) in cases {
            assert_eq!(Dialect::infer_from_model(model), expected, "{model}");
        }
    }

    #[test]
    fn parse_model_ref_table() {
        let builtins = vec![
            ep("anthropic", Dialect::Anthropic, true),
            ep("openai", Dialect::OpenAi, true),
            ep("litellm", Dialect::OpenAi, false),
            ep("vertex", Dialect::Gemini, false),
        ];
        let two_openai = vec![
            ep("litellm", Dialect::OpenAi, false),
            ep("proxy", Dialect::OpenAi, false),
        ];
        let single = vec![ep("litellm", Dialect::OpenAi, false)];
        type Case<'a> = (&'a str, &'a [Endpoint], Result<EndpointRef, &'a str>);
        let cases: Vec<Case<'_>> = vec![
            // explicit
            ("litellm:gpt-5.2", &builtins, Ok(r("litellm", "gpt-5.2"))),
            ("anthropic:gpt-x", &builtins, Ok(r("anthropic", "gpt-x"))),
            // model ids containing ':' after an explicit endpoint
            (
                "litellm:llama3:8b",
                &builtins,
                Ok(r("litellm", "llama3:8b")),
            ),
            // bare: builtin of the inferred dialect wins over configured ones
            ("gpt-5.2", &builtins, Ok(r("openai", "gpt-5.2"))),
            (
                "claude-sonnet-4-6",
                &builtins,
                Ok(r("anthropic", "claude-sonnet-4-6")),
            ),
            ("o3", &builtins, Ok(r("openai", "o3"))),
            // bare: the only endpoint of the dialect
            (
                "gemini-3-flash-preview",
                &builtins,
                Ok(r("vertex", "gemini-3-flash-preview")),
            ),
            // bare: the only configured endpoint
            ("llama3:8b", &single, Ok(r("litellm", "llama3:8b"))),
            (
                "claude-opus-4-6",
                &single,
                Ok(r("litellm", "claude-opus-4-6")),
            ),
            // ambiguous
            ("gpt-5.2", &two_openai, Err("cannot tell which endpoint")),
            ("mystery", &builtins, Err("cannot tell which endpoint")),
            ("nope:gpt-5.2", &two_openai, Err("unknown endpoint `nope`")),
            ("", &builtins, Err("empty")),
            ("litellm:", &builtins, Err("empty")),
            ("gpt-5.2", &[], Err("no LLM endpoint")),
        ];
        for (input, endpoints, expected) in cases {
            let got = parse_model_ref(input, endpoints, &ModelIndex::default(), None);
            match expected {
                Ok(want) => assert_eq!(got.as_ref(), Ok(&want), "{input}"),
                Err(needle) => {
                    let err = got.expect_err(input).to_string();
                    assert!(err.contains(needle), "{input}: {err}");
                }
            }
        }
        let err = parse_model_ref("gpt-5.2", &two_openai, &ModelIndex::default(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("litellm, proxy"), "{err}");
    }

    #[test]
    fn bare_ids_resolve_by_lookup_with_tie_breaks() {
        let mut high = ep("proxy", Dialect::OpenAi, false);
        high.priority = 5;
        let endpoints = vec![
            ep("anthropic", Dialect::Anthropic, true),
            ep("openai", Dialect::OpenAi, true),
            ep("litellm", Dialect::OpenAi, false),
            high,
        ];
        let mut index = ModelIndex::default();
        index.insert(
            "litellm",
            ["only-on-litellm", "gpt-5.2", "shared", "claude-opus-4-6"],
        );
        index.insert("openai", ["gpt-5.2"]);
        index.insert("proxy", ["shared", "gpt-5.2", "claude-opus-4-6"]);
        index.insert("anthropic", ["claude-opus-4-6"]);
        type Case<'a> = (&'a str, Option<&'a str>, Result<EndpointRef, &'a str>);
        let cases: Vec<Case<'_>> = vec![
            // unique: listed by one endpoint, whatever the prefix says
            ("only-on-litellm", None, Ok(r("litellm", "only-on-litellm"))),
            // several: the current endpoint first
            ("gpt-5.2", Some("litellm"), Ok(r("litellm", "gpt-5.2"))),
            (
                "claude-opus-4-6",
                Some("proxy"),
                Ok(r("proxy", "claude-opus-4-6")),
            ),
            // then the builtin of the id's dialect
            ("gpt-5.2", Some("anthropic"), Ok(r("openai", "gpt-5.2"))),
            (
                "claude-opus-4-6",
                None,
                Ok(r("anthropic", "claude-opus-4-6")),
            ),
            // then the highest priority
            ("shared", None, Ok(r("proxy", "shared"))),
            // unlisted → prefix inference
            ("gpt-6", None, Ok(r("openai", "gpt-6"))),
            (
                "claude-new",
                Some("litellm"),
                Ok(r("anthropic", "claude-new")),
            ),
            // unlisted, unknown prefix, several endpoints → error
            ("mystery", None, Err("cannot tell which endpoint")),
            // explicit override
            ("litellm:gpt-6", None, Ok(r("litellm", "gpt-6"))),
        ];
        for (input, current, expected) in cases {
            let got = parse_model_ref(input, &endpoints, &index, current);
            match expected {
                Ok(want) => assert_eq!(got.as_ref(), Ok(&want), "{input}"),
                Err(needle) => {
                    let err = got.expect_err(input).to_string();
                    assert!(err.contains(needle), "{input}: {err}");
                }
            }
        }
        // Equal priorities: the first configured wins.
        let tied = vec![
            ep("a", Dialect::OpenAi, false),
            ep("b", Dialect::OpenAi, false),
        ];
        let mut index = ModelIndex::default();
        index.insert("b", ["m"]);
        index.insert("a", ["m"]);
        assert_eq!(parse_model_ref("m", &tied, &index, None), Ok(r("a", "m")));
        // Unlisted, unknown prefix, single endpoint → that endpoint.
        let single = vec![ep("litellm", Dialect::OpenAi, false)];
        assert_eq!(
            parse_model_ref("mystery", &single, &ModelIndex::default(), None),
            Ok(r("litellm", "mystery"))
        );
        assert!(needs_lookup("gpt-5.2", &endpoints));
        assert!(!needs_lookup("litellm:gpt-5.2", &endpoints));
        assert!(!needs_lookup("gpt-5.2", &single));
    }

    #[test]
    fn endpoint_ref_display() {
        assert_eq!(r("litellm", "gpt-5.2").to_string(), "litellm:gpt-5.2");
    }

    #[test]
    fn secret_never_debugs() {
        let e = ep("x", Dialect::OpenAi, false);
        let mut e = e;
        e.api_key = SecretString::new("sk-very-secret");
        assert!(!format!("{e:?}").contains("sk-very-secret"));
    }

    fn layered(section: EndpointSection, origin: RuleOrigin) -> LayeredEndpoint {
        LayeredEndpoint { section, origin }
    }

    fn section(dialect: Option<&str>, key_env: Option<&str>, key: Option<&str>) -> EndpointSection {
        EndpointSection {
            dialect: dialect.map(str::to_string),
            base_url: Some("http://localhost:4000".to_string()),
            api_key_env: key_env.map(str::to_string),
            api_key: key.map(str::to_string),
            priority: None,
        }
    }

    #[test]
    fn discover_validates_configured_endpoints() {
        let mut config = Config::default();
        // No built-in keys from the config; the environment is not consulted
        // for built-ins in this test unless set, so only assert on configured.
        config.providers.api_keys = None;
        let configured: BTreeMap<String, LayeredEndpoint> = [
            (
                "litellm",
                layered(
                    section(Some("openai"), Some("LITELLM_KEY"), None),
                    RuleOrigin::Realm,
                ),
            ),
            (
                "plain",
                layered(
                    section(Some("anthropic"), None, Some("k")),
                    RuleOrigin::Global,
                ),
            ),
            (
                "baddialect",
                layered(section(Some("cohere"), None, Some("k")), RuleOrigin::Realm),
            ),
            (
                "nodialect",
                layered(section(None, None, Some("k")), RuleOrigin::Realm),
            ),
            (
                "nokey",
                layered(section(Some("openai"), None, None), RuleOrigin::Realm),
            ),
            (
                "unsetenv",
                layered(
                    section(Some("openai"), Some("UNSET_VAR"), None),
                    RuleOrigin::Realm,
                ),
            ),
            (
                "openai",
                layered(section(Some("openai"), None, Some("k")), RuleOrigin::Realm),
            ),
            (
                "projenv",
                layered(
                    section(Some("openai"), Some("LITELLM_KEY"), None),
                    RuleOrigin::Project,
                ),
            ),
            (
                "bad.name",
                layered(section(Some("openai"), None, Some("k")), RuleOrigin::Realm),
            ),
        ]
        .into_iter()
        .map(|(n, l)| (n.to_string(), l))
        .collect();
        let (registry, notices) = EndpointRegistry::discover_with(&config, &configured, |var| {
            (var == "LITELLM_KEY").then(|| "sk-litellm".to_string())
        });
        let litellm = registry.get("litellm").expect("litellm");
        assert_eq!(litellm.dialect, Dialect::OpenAi);
        assert_eq!(litellm.api_key.expose(), "sk-litellm");
        assert!(!litellm.builtin);
        assert!(registry.get("plain").is_some());
        for skipped in [
            "baddialect",
            "nodialect",
            "nokey",
            "unsetenv",
            "projenv",
            "bad.name",
        ] {
            assert!(registry.get(skipped).is_none(), "{skipped}");
            assert!(
                notices
                    .iter()
                    .any(|n| matches!(n, Notice::EndpointSkipped { name, .. } if name == skipped)),
                "{skipped}: {notices:?}"
            );
        }
        // The configured `openai` clashes with the built-in name.
        assert!(registry.iter().all(|e| e.name != "openai" || e.builtin));
        assert!(
            notices
                .iter()
                .any(|n| matches!(n, Notice::EndpointSkipped { name, .. } if name == "openai"))
        );
        assert!(
            notices
                .iter()
                .any(|n| matches!(n, Notice::EndpointPlainKey { name } if name == "plain"))
        );
        let text: Vec<String> = notices.iter().map(ToString::to_string).collect();
        assert!(text.iter().all(|t| !t.contains("sk-litellm")));
    }

    fn plan_startup_t(
        request: StartupRequest<'_>,
        endpoints: &[Endpoint],
    ) -> Result<StartupPlan, ModelRefError> {
        plan_startup(request, endpoints, &ModelIndex::default())
    }

    #[test]
    fn startup_plans() {
        let both = vec![
            ep("anthropic", Dialect::Anthropic, true),
            ep("litellm", Dialect::OpenAi, false),
        ];
        let only_litellm = vec![ep("litellm", Dialect::OpenAi, false)];
        let only_openai = vec![ep("openai", Dialect::OpenAi, true)];
        let req = |model, endpoint, config_model| StartupRequest {
            model,
            endpoint,
            config_model,
        };
        let resolved = |e: &str, m: &str| Ok(StartupPlan::Resolved(r(e, m)));
        assert_eq!(
            plan_startup_t(req(Some("litellm:gpt-x"), None, ""), &both),
            resolved("litellm", "gpt-x")
        );
        assert_eq!(
            plan_startup_t(req(Some("gpt-x"), Some("litellm"), ""), &both),
            resolved("litellm", "gpt-x")
        );
        assert_eq!(
            plan_startup_t(req(Some("claude-opus-4-6"), None, ""), &both),
            resolved("anthropic", "claude-opus-4-6")
        );
        assert_eq!(
            plan_startup_t(req(None, Some("litellm"), "claude-opus-4-6"), &both),
            Ok(StartupPlan::FirstListed("litellm".into()))
        );
        assert_eq!(
            plan_startup_t(req(None, None, "claude-opus-4-6"), &both),
            resolved("anthropic", "claude-opus-4-6")
        );
        // Config model of a dialect nobody serves, one endpoint: list it.
        assert_eq!(
            plan_startup_t(req(None, None, "claude-opus-4-6"), &only_litellm),
            Ok(StartupPlan::FirstListed("litellm".into()))
        );
        // ... or use the catalog default of a built-in.
        let Ok(StartupPlan::Resolved(choice)) =
            plan_startup_t(req(None, None, "claude-opus-4-6"), &only_openai)
        else {
            panic!("expected a resolved model");
        };
        assert_eq!(choice.endpoint, "openai");
        assert_eq!(
            Some(choice.model.as_str()),
            meerkat_models::default_model("openai")
        );
        let err = plan_startup_t(req(None, Some("nope"), ""), &both).unwrap_err();
        assert!(err.to_string().contains("anthropic, litellm"), "{err}");
        let err = plan_startup_t(
            req(None, None, "mystery"),
            &[
                ep("litellm", Dialect::OpenAi, false),
                ep("proxy", Dialect::OpenAi, false),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, ModelRefError::NoModel { .. }), "{err}");
        assert_eq!(
            plan_startup_t(req(None, None, ""), &[]),
            Err(ModelRefError::NoEndpoints)
        );
    }

    #[test]
    fn restore_prefers_sidecar_then_infers() {
        let endpoints = vec![
            ep("anthropic", Dialect::Anthropic, true),
            ep("litellm", Dialect::OpenAi, false),
        ];
        let exists = |name: &str| endpoints.iter().any(|e| e.name == name);
        let fallback = r("anthropic", "claude-sonnet-4-6");
        let (choice, notice) = restore_choice(
            Some(r("litellm", "gpt-x")),
            None,
            &endpoints,
            exists,
            &fallback,
        );
        assert_eq!(choice, r("litellm", "gpt-x"));
        assert!(notice.is_none());
        let (choice, notice) = restore_choice(
            Some(r("gone", "gpt-x")),
            None,
            &endpoints,
            exists,
            &fallback,
        );
        assert_eq!(choice, r("litellm", "gpt-x"));
        assert!(notice.unwrap().contains("`gone`"));
        let (choice, notice) = restore_choice(
            None,
            Some((Provider::Anthropic, "claude-opus-4-6".into())),
            &endpoints,
            exists,
            &fallback,
        );
        assert_eq!(choice, r("anthropic", "claude-opus-4-6"));
        assert!(notice.is_none());
        let (choice, notice) = restore_choice(
            None,
            Some((Provider::Gemini, "gemini-3-flash-preview".into())),
            &[ep("anthropic", Dialect::Anthropic, true)],
            |name| name == "anthropic",
            &fallback,
        );
        assert_eq!(choice, fallback);
        assert!(notice.is_some());
    }

    #[test]
    fn choice_sidecar_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load_choice(tmp.path(), "abc").is_none());
        save_choice(tmp.path(), "abc", &r("litellm", "llama3:8b")).unwrap();
        assert_eq!(
            load_choice(tmp.path(), "abc"),
            Some(r("litellm", "llama3:8b"))
        );
    }
}
