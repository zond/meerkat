//! Auto-mode approval engine.
//!
//! Every tool call passes through a `PreToolExecution` guardrail hook
//! (`tui-auto-mode`) that calls [`ApprovalEngine::decide`]:
//!
//! 0. a user deny rule matches ⇒ deny;
//! 1. read-only builtins, in-project reads/searches, allowlisted shell commands
//!    and narrow user allow rules ⇒ allow;
//! 2. writes whose every target is inside the project and unprotected ⇒ allow;
//! 3. everything else ⇒ the reasoning-blind [`ActionClassifier`].
//!
//! Classifier blocks are returned to the model as hook denials with a reason.
//! After three consecutive (or twenty total) blocks, auto mode pauses and Tier-3
//! actions are escalated to the human through the [`EscalationBroker`].
//!
//! A second, observe-only `RunStarted` hook (`tui-transcript`) records user
//! prompts, so the classifier transcript never depends on the UI.

pub mod classifier;
pub mod counters;
pub mod escalation;
pub mod policy;
pub mod prompts;
pub mod rules;
pub mod shell;
pub mod transcript;

use classifier::{ActionClassifier, ClassifierInput, ClassifierVerdict, Environment};
use counters::{AutoMode, AutoModeState, DenialCounters, PauseCause};
use escalation::{EscalationBroker, EscalationPrompt, HumanDecision};
use meerkat_core::{
    HookCapability, HookDecision, HookEngine, HookEntryConfig, HookExecutionMode,
    HookFailurePolicy, HookId, HookInvocation, HookPoint, HookReasonCode, HookRuntimeConfig,
    HooksConfig, Session, SessionId,
};
use meerkat_hooks::{DefaultHookEngine, InProcessHookHandler, RuntimeHookResponse};
use parking_lot::{Mutex, RwLock};
use policy::{Action, ActionKind, Decider, PolicyContext, Tier, VerdictSummary, classify_tier};
use rules::{AutoSettings, Rule, RuleId, RuleOrigin, RuleSet, RulesError};
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use transcript::ApprovalTranscript;

use crate::paths::ProjectRoot;

/// Hook id and in-process handler name of the transcript hook.
pub const TRANSCRIPT_HOOK_ID: &str = "tui-transcript";
/// Hook id and in-process handler name of the guardrail hook.
pub const AUTO_MODE_HOOK_ID: &str = "tui-auto-mode";
/// Guardrail timeout: long enough for a human to answer an escalation.
pub const AUTO_MODE_HOOK_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
/// Longest argument preview shown in escalation dialogs.
const ARGS_PREVIEW_BYTES: usize = 2048;

/// Final decision for one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Run the tool.
    Allow {
        /// Tier the action fell into.
        tier: Tier,
        /// Who allowed it.
        by: Decider,
        /// Short explanation.
        why: String,
    },
    /// Do not run the tool; the model receives the reason.
    Deny {
        /// Tier the action fell into.
        tier: Tier,
        /// Why.
        reason: DenyReason,
    },
}

impl Verdict {
    /// Compact form for the UI and the transcript.
    pub fn summary(&self) -> VerdictSummary {
        match self {
            Self::Allow { tier, by, .. } => VerdictSummary::Allowed {
                tier: *tier,
                by: *by,
            },
            Self::Deny { tier, reason } => VerdictSummary::Denied {
                tier: *tier,
                by: reason.decider(),
            },
        }
    }

    /// Whether the tool may run.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow { .. })
    }
}

/// Why a call was denied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenyReason {
    /// A user deny rule matched.
    UserRule {
        /// Rule id.
        id: RuleId,
        /// The rule as written (`shell git push*`).
        rule: String,
    },
    /// The classifier blocked the action.
    Classifier {
        /// Block-rule id, if named.
        rule: Option<String>,
        /// Explanation.
        explanation: String,
    },
    /// The human denied it.
    Human,
    /// The human stopped the turn.
    Stopped,
    /// Auto mode is disabled; Tier-3 actions are not run.
    AutoDisabled,
    /// The classifier failed and no human could be asked.
    ClassifierUnavailable(String),
}

impl DenyReason {
    fn decider(&self) -> Decider {
        match self {
            Self::UserRule { .. } | Self::AutoDisabled | Self::ClassifierUnavailable(_) => {
                Decider::Policy
            }
            Self::Classifier { .. } => Decider::Classifier,
            Self::Human | Self::Stopped => Decider::Human,
        }
    }

    fn reason_code(&self) -> HookReasonCode {
        match self {
            Self::Classifier { .. } => HookReasonCode::SafetyViolation,
            _ => HookReasonCode::PolicyViolation,
        }
    }

    fn rule_label(&self) -> Option<String> {
        match self {
            Self::UserRule { id, .. } => Some(id.to_string()),
            Self::Classifier { rule, .. } => rule.clone(),
            _ => None,
        }
    }
}

impl fmt::Display for DenyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserRule { id, rule } => write!(f, "the user's deny rule {id} ({rule}) matches"),
            Self::Classifier {
                rule: Some(rule),
                explanation,
            } => write!(f, "{explanation} [{rule}]"),
            Self::Classifier {
                rule: None,
                explanation,
            } => f.write_str(explanation),
            Self::Human => f.write_str("the user denied this action"),
            Self::Stopped => f.write_str("the user stopped this turn"),
            Self::AutoDisabled => {
                f.write_str("auto mode is disabled, so actions that need safety review are not run")
            }
            Self::ClassifierUnavailable(err) => {
                write!(f, "the safety classifier is unavailable ({err})")
            }
        }
    }
}

/// Typed payload attached to hook denials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DenyPayload {
    /// Always `"auto-mode"`.
    pub policy: &'static str,
    /// Tool-use id of the denied call (lets a UI match the denial to its tool block).
    pub tool_use_id: String,
    /// Tool name.
    pub tool: String,
    /// Tier the action fell into.
    pub tier: Tier,
    /// Rule id (user rule or classifier block rule), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
    /// Consecutive classifier blocks.
    pub consecutive: u32,
    /// Total classifier blocks.
    pub total: u32,
}

/// Model-facing denial message.
pub fn deny_message(reason: &DenyReason) -> String {
    format!(
        "Blocked by auto-mode safety policy: {reason}. Do not try to achieve the same effect \
         another way (different tool, rewritten or encoded command, or a script). Choose a safer \
         approach within the user's request, or stop and ask the user."
    )
}

/// UI-facing notifications.
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalEvent {
    /// A tool call was decided.
    Decision {
        /// Session.
        session_id: SessionId,
        /// Tool-use id.
        tool_use_id: String,
        /// Tool name.
        tool: String,
        /// Tier.
        tier: Tier,
        /// Outcome.
        verdict: VerdictSummary,
    },
    /// Auto mode paused.
    Paused(PauseCause),
    /// Auto mode resumed.
    Resumed,
    /// Broad allow rules were dropped on entering auto mode.
    RulesDropped(Vec<Rule>),
}

/// Errors from engine configuration.
#[derive(Debug, thiserror::Error)]
pub enum ApprovalError {
    /// Rules could not be persisted.
    #[error(transparent)]
    Rules(#[from] RulesError),
    /// A hook runtime config could not be built.
    #[error("invalid hook runtime config: {0}")]
    HookConfig(#[from] serde_json::Error),
}

/// Construction parameters for [`ApprovalEngine`].
pub struct ApprovalConfig {
    /// Canonical project root.
    pub root: ProjectRoot,
    /// Layered rules (sanitized for auto mode on construction).
    pub rules: RuleSet,
    /// Layered settings.
    pub settings: AutoSettings,
    /// Tier-3 reviewer.
    pub classifier: Arc<dyn ActionClassifier>,
    /// Human escalation; `None` ⇒ escalations are denied.
    pub broker: Option<EscalationBroker>,
    /// Environment facts for the classifier.
    pub environment: Environment,
    /// Whether the shell tool runs a POSIX shell.
    pub posix_shell: bool,
    /// UI notifications.
    pub events: mpsc::UnboundedSender<ApprovalEvent>,
    /// Realm rules file; rules added at runtime are persisted here when set.
    pub rules_path: Option<PathBuf>,
}

type CacheKey = (String, u64);

/// The approval engine. Shared behind an [`Arc`] by both hook handlers.
pub struct ApprovalEngine {
    root: ProjectRoot,
    rules: RwLock<RuleSet>,
    settings: RwLock<AutoSettings>,
    transcripts: RwLock<HashMap<SessionId, ApprovalTranscript>>,
    auto: Mutex<AutoMode>,
    allow_cache: Mutex<HashMap<SessionId, HashSet<CacheKey>>>,
    stopped: Mutex<HashSet<SessionId>>,
    classifier: Arc<dyn ActionClassifier>,
    broker: Option<EscalationBroker>,
    environment: Environment,
    posix_shell: bool,
    events: mpsc::UnboundedSender<ApprovalEvent>,
    rules_path: Option<PathBuf>,
}

impl fmt::Debug for ApprovalEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalEngine")
            .field("root", &self.root)
            .field("settings", &*self.settings.read())
            .field("auto", &*self.auto.lock())
            .finish_non_exhaustive()
    }
}

impl ApprovalEngine {
    /// Build the engine. Broad allow rules are dropped and reported once via
    /// [`ApprovalEvent::RulesDropped`].
    pub fn new(config: ApprovalConfig) -> Arc<Self> {
        let ApprovalConfig {
            root,
            mut rules,
            settings,
            classifier,
            broker,
            environment,
            posix_shell,
            events,
            rules_path,
        } = config;
        let dropped = rules.sanitize_for_auto();
        if !dropped.is_empty() {
            let _ = events.send(ApprovalEvent::RulesDropped(dropped));
        }
        let auto = AutoMode::new(settings.enabled);
        Arc::new(Self {
            root,
            rules: RwLock::new(rules),
            settings: RwLock::new(settings),
            transcripts: RwLock::new(HashMap::new()),
            auto: Mutex::new(auto),
            allow_cache: Mutex::new(HashMap::new()),
            stopped: Mutex::new(HashSet::new()),
            classifier,
            broker,
            environment,
            posix_shell,
            events,
            rules_path,
        })
    }

    /// The project root.
    pub fn root(&self) -> &ProjectRoot {
        &self.root
    }

    /// Current settings.
    pub fn settings(&self) -> AutoSettings {
        self.settings.read().clone()
    }

    /// Current rules.
    pub fn rules(&self) -> RuleSet {
        self.rules.read().clone()
    }

    /// Current counters and state.
    pub fn auto_state(&self) -> (DenialCounters, AutoModeState) {
        let auto = self.auto.lock();
        (auto.counters, auto.state)
    }

    /// Rendered approval transcript for a session (for diagnostics and tests).
    pub fn transcript(&self, session_id: &SessionId) -> Option<String> {
        self.transcripts
            .read()
            .get(session_id)
            .map(ApprovalTranscript::render)
    }

    /// Seed a session's transcript from persisted history (on resume).
    pub fn seed_transcript(&self, session_id: &SessionId, session: &Session) {
        let mut transcript = ApprovalTranscript::default();
        transcript.seed_from_session(session);
        self.transcripts
            .write()
            .insert(session_id.clone(), transcript);
    }

    /// Record a user prompt (what the `tui-transcript` hook does) and clear a
    /// previous stop for the session.
    pub fn record_prompt(&self, session_id: &SessionId, prompt: &str) {
        self.stopped.lock().remove(session_id);
        self.transcripts
            .write()
            .entry(session_id.clone())
            .or_default()
            .push_user(prompt);
    }

    /// Add a rule, persisting it to the realm rules file when configured.
    ///
    /// # Errors
    ///
    /// Returns [`RulesError`] when persistence fails (the rule stays active).
    pub fn add_rule(&self, mut rule: Rule) -> Result<(), RulesError> {
        if self.rules_path.is_some() {
            rule.origin = RuleOrigin::Realm;
        }
        let snapshot = {
            let mut rules = self.rules.write();
            rules.push(rule);
            rules.clone()
        };
        self.persist(&snapshot)
    }

    /// Remove a rule (global and project rules are removed for this process only).
    ///
    /// # Errors
    ///
    /// Returns [`RulesError`] when persistence fails.
    pub fn remove_rule(&self, id: RuleId) -> Result<Option<Rule>, RulesError> {
        let (removed, snapshot) = {
            let mut rules = self.rules.write();
            let removed = rules.remove(id);
            (removed, rules.clone())
        };
        if removed.is_some() {
            self.persist(&snapshot)?;
        }
        Ok(removed)
    }

    fn persist(&self, rules: &RuleSet) -> Result<(), RulesError> {
        match &self.rules_path {
            Some(path) => rules::persist_realm_rules(path, rules),
            None => Ok(()),
        }
    }

    /// Turn auto mode on (`/auto on`, resets counters) or off (`/auto off`:
    /// Tier-3 actions are denied outright).
    pub fn set_enabled(&self, enabled: bool) {
        self.settings.write().enabled = enabled;
        let event = {
            let mut auto = self.auto.lock();
            if enabled {
                auto.resume(true).then_some(ApprovalEvent::Resumed)
            } else {
                auto.pause(PauseCause::UserDisabled)
                    .then_some(ApprovalEvent::Paused(PauseCause::UserDisabled))
            }
        };
        if let Some(event) = event {
            self.emit(event);
        }
    }

    /// `/auto resume`: clear a pause (including the total-denials pause) and reset counters.
    pub fn resume_auto(&self) {
        let resumed = self.auto.lock().resume(true);
        if resumed {
            self.emit(ApprovalEvent::Resumed);
        }
    }

    fn emit(&self, event: ApprovalEvent) {
        let _ = self.events.send(event);
    }

    /// Decide one tool call. Records it in the session's approval transcript and
    /// emits an [`ApprovalEvent::Decision`].
    pub async fn decide(
        &self,
        session_id: &SessionId,
        tool_use_id: &str,
        name: &str,
        args: &Value,
    ) -> Verdict {
        let verdict = self.evaluate(session_id, name, args).await;
        let tier = match &verdict {
            Verdict::Allow { tier, .. } | Verdict::Deny { tier, .. } => *tier,
        };
        let summary = verdict.summary();
        self.transcripts
            .write()
            .entry(session_id.clone())
            .or_default()
            .push_tool_call(tool_use_id, name, args.clone(), Some(summary));
        tracing::debug!(tool = name, %tier, ?summary, "auto-mode decision");
        self.emit(ApprovalEvent::Decision {
            session_id: session_id.clone(),
            tool_use_id: tool_use_id.to_string(),
            tool: name.to_string(),
            tier,
            verdict: summary,
        });
        verdict
    }

    fn tier_of(&self, action: &Action<'_>) -> policy::TierDecision {
        let trust_project_builds = self.settings.read().trust_project_builds;
        let ctx = PolicyContext {
            root: &self.root,
            trust_project_builds,
            posix_shell: self.posix_shell,
        };
        classify_tier(action, &self.rules.read(), &ctx)
    }

    /// Verdict for tiers 0–2; `None` for Tier 3.
    fn static_verdict(&self, decision: policy::TierDecision) -> Option<Verdict> {
        match decision.tier {
            Tier::UserDeny => {
                let id = decision.rule?;
                let rule = self
                    .rules
                    .read()
                    .get(id)
                    .map(describe_rule)
                    .unwrap_or_default();
                Some(Verdict::Deny {
                    tier: Tier::UserDeny,
                    reason: DenyReason::UserRule { id, rule },
                })
            }
            Tier::AlwaysAllow | Tier::InProjectEdit => Some(Verdict::Allow {
                tier: decision.tier,
                by: Decider::Policy,
                why: decision.detail,
            }),
            Tier::Classifier => None,
        }
    }

    async fn evaluate(&self, session_id: &SessionId, name: &str, args: &Value) -> Verdict {
        let action = Action::parse(name, args);
        if let Some(verdict) = self.static_verdict(self.tier_of(&action)) {
            return verdict;
        }

        let (disabled, escalating) = {
            let auto = self.auto.lock();
            (auto.disabled(), auto.escalating())
        };
        if disabled {
            return deny3(DenyReason::AutoDisabled);
        }

        let key = cache_key(name, args);
        if !escalating
            && self
                .allow_cache
                .lock()
                .get(session_id)
                .is_some_and(|set| set.contains(&key))
        {
            self.auto.lock().record_allow();
            return allow3(Decider::Classifier, "previously allowed by the classifier");
        }

        let script = self.script_payload(&action).await;
        let result = {
            let transcript = self
                .transcripts
                .read()
                .get(session_id)
                .cloned()
                .unwrap_or_default();
            let input = ClassifierInput {
                environment: &self.environment,
                transcript: &transcript,
                tool: name,
                args,
                script_payload: script.as_deref(),
            };
            self.classifier.classify(input).await
        };

        if escalating {
            let hint = match &result {
                Ok(v) => format!("classifier: {v}"),
                Err(e) => format!("classifier error: {e}"),
            };
            return self.escalate(session_id, &action, Some(hint)).await;
        }

        match result {
            Ok(ClassifierVerdict::Allow) => {
                self.allow_cache
                    .lock()
                    .entry(session_id.clone())
                    .or_default()
                    .insert(key);
                self.auto.lock().record_allow();
                allow3(Decider::Classifier, "allowed by the classifier")
            }
            Ok(ClassifierVerdict::Block { rule, reason }) => {
                let paused = self.auto.lock().record_classifier_deny();
                if let Some(cause) = paused {
                    self.emit(ApprovalEvent::Paused(cause));
                }
                deny3(DenyReason::Classifier {
                    rule,
                    explanation: reason,
                })
            }
            Err(err) => {
                tracing::warn!(error = %err, tool = name, "classifier failed; escalating");
                if self.broker.is_some() {
                    self.escalate(
                        session_id,
                        &action,
                        Some(format!("classifier error: {err}")),
                    )
                    .await
                } else {
                    deny3(DenyReason::ClassifierUnavailable(err.to_string()))
                }
            }
        }
    }

    async fn escalate(
        &self,
        session_id: &SessionId,
        action: &Action<'_>,
        hint: Option<String>,
    ) -> Verdict {
        let Some(broker) = &self.broker else {
            return deny3(DenyReason::ClassifierUnavailable(
                "no human is available to review this action".to_string(),
            ));
        };
        let guard = broker.serial().await;
        // Re-check after waiting: a Stop or a new rule may have arrived meanwhile.
        if self.stopped.lock().contains(session_id) {
            return deny3(DenyReason::Stopped);
        }
        if let Some(verdict) = self.static_verdict(self.tier_of(action)) {
            return verdict;
        }
        if self.auto.lock().disabled() {
            return deny3(DenyReason::AutoDisabled);
        }
        let prompt = EscalationPrompt {
            session_id: session_id.clone(),
            tool: action.name.to_string(),
            args_preview: args_preview(action),
            classifier_hint: hint,
        };
        let decision = broker.ask_locked(&guard, prompt).await;
        drop(guard);
        match decision {
            HumanDecision::AllowOnce => {
                self.human_allowed();
                allow3(Decider::Human, "allowed once by the user")
            }
            HumanDecision::AllowAndAddRule(rule) => {
                if let Err(err) = self.add_rule(*rule) {
                    tracing::warn!(error = %err, "failed to persist rule");
                }
                self.human_allowed();
                allow3(Decider::Human, "allowed by the user (rule added)")
            }
            HumanDecision::Deny => deny3(DenyReason::Human),
            HumanDecision::Stop => {
                self.stopped.lock().insert(session_id.clone());
                deny3(DenyReason::Stopped)
            }
        }
    }

    fn human_allowed(&self) {
        let resumed = {
            let mut auto = self.auto.lock();
            auto.record_allow();
            auto.resume(false)
        };
        if resumed {
            self.emit(ApprovalEvent::Resumed);
        }
    }

    async fn script_payload(&self, action: &Action<'_>) -> Option<String> {
        let ActionKind::Shell {
            command,
            working_dir,
            ..
        } = &action.kind
        else {
            return None;
        };
        let script = shell::script_path(command)?;
        let joined = match working_dir {
            Some(dir) if !script.starts_with('/') => format!("{dir}/{script}"),
            _ => script,
        };
        let resolved = self.root.resolve(&joined).ok()?;
        if !resolved.inside() {
            return None;
        }
        let bytes = tokio::fs::read(&resolved.abs).await.ok()?;
        let cut = bytes.len().min(prompts::MAX_SCRIPT_BYTES);
        let slice = bytes.get(..cut)?;
        Some(String::from_utf8_lossy(slice).into_owned())
    }

    /// Hook decision for a verdict on the call `tool_use_id` of tool `tool`.
    pub fn hook_decision(&self, verdict: &Verdict, tool_use_id: &str, tool: &str) -> HookDecision {
        match verdict {
            Verdict::Allow { .. } => HookDecision::Allow,
            Verdict::Deny { tier, reason } => {
                let counters = self.auto.lock().counters;
                let payload = DenyPayload {
                    policy: "auto-mode",
                    tool_use_id: tool_use_id.to_string(),
                    tool: tool.to_string(),
                    tier: *tier,
                    rule: reason.rule_label(),
                    consecutive: counters.consecutive,
                    total: counters.total,
                };
                HookDecision::deny(
                    HookId::new(AUTO_MODE_HOOK_ID),
                    reason.reason_code(),
                    deny_message(reason),
                    serde_json::to_value(payload).ok(),
                )
            }
        }
    }

    /// The `(RunStarted, PreToolExecution)` in-process hook handlers.
    pub fn into_hook_handlers(self: &Arc<Self>) -> (InProcessHookHandler, InProcessHookHandler) {
        let engine = Arc::clone(self);
        let run_started: InProcessHookHandler = Arc::new(move |invocation: HookInvocation| {
            let engine = Arc::clone(&engine);
            Box::pin(async move {
                if let Some(prompt) = invocation.prompt.as_deref() {
                    engine.record_prompt(&invocation.session_id, prompt);
                }
                Ok(RuntimeHookResponse::default())
            })
        });
        let engine = Arc::clone(self);
        let pre_tool: InProcessHookHandler = Arc::new(move |invocation: HookInvocation| {
            let engine = Arc::clone(&engine);
            Box::pin(async move {
                let Some(call) = invocation.tool_call else {
                    return Err("pre-tool invocation without a tool call".to_string());
                };
                let verdict = engine
                    .decide(
                        &invocation.session_id,
                        &call.tool_use_id,
                        &call.name,
                        &call.args,
                    )
                    .await;
                Ok(RuntimeHookResponse {
                    decision: Some(engine.hook_decision(&verdict, &call.tool_use_id, &call.name)),
                    patches: Vec::new(),
                })
            })
        });
        (run_started, pre_tool)
    }

    /// The two hook registrations.
    ///
    /// # Errors
    ///
    /// Returns [`ApprovalError::HookConfig`] if a runtime config cannot be serialized.
    pub fn hook_entries() -> Result<Vec<HookEntryConfig>, ApprovalError> {
        let runtime = |name: &str| {
            HookRuntimeConfig::new("in_process", Some(serde_json::json!({ "name": name })))
        };
        Ok(vec![
            HookEntryConfig {
                id: HookId::new(TRANSCRIPT_HOOK_ID),
                enabled: true,
                point: HookPoint::RunStarted,
                mode: HookExecutionMode::Foreground,
                capability: HookCapability::Observe,
                priority: i32::MIN,
                failure_policy: Some(HookFailurePolicy::FailOpen),
                timeout_ms: None,
                runtime: runtime(TRANSCRIPT_HOOK_ID)?,
            },
            HookEntryConfig {
                id: HookId::new(AUTO_MODE_HOOK_ID),
                enabled: true,
                point: HookPoint::PreToolExecution,
                mode: HookExecutionMode::Foreground,
                capability: HookCapability::Guardrail,
                priority: i32::MIN,
                failure_policy: Some(HookFailurePolicy::FailClosed),
                timeout_ms: Some(AUTO_MODE_HOOK_TIMEOUT_MS),
                runtime: runtime(AUTO_MODE_HOOK_ID)?,
            },
        ])
    }

    /// Build a hook engine from `base` (layered user hooks) plus our two entries.
    ///
    /// # Errors
    ///
    /// Returns [`ApprovalError::HookConfig`] if a runtime config cannot be serialized.
    pub fn build_hook_engine(
        self: &Arc<Self>,
        mut base: HooksConfig,
    ) -> Result<Arc<dyn HookEngine>, ApprovalError> {
        base.entries
            .retain(|e| e.id.0 != TRANSCRIPT_HOOK_ID && e.id.0 != AUTO_MODE_HOOK_ID);
        let rewriters: Vec<String> = base
            .entries
            .iter()
            .filter(|e| {
                e.enabled
                    && e.point == HookPoint::PreToolExecution
                    && e.capability == HookCapability::Rewrite
            })
            .map(|e| e.id.to_string())
            .collect();
        if !rewriters.is_empty() {
            tracing::warn!(
                hooks = ?rewriters,
                "rewrite-capable pre-tool hooks may change tool arguments after auto-mode review"
            );
        }
        base.entries.extend(Self::hook_entries()?);
        let (run_started, pre_tool) = self.into_hook_handlers();
        let engine = DefaultHookEngine::new(base)
            .with_in_process_handler(TRANSCRIPT_HOOK_ID, run_started)
            .with_in_process_handler(AUTO_MODE_HOOK_ID, pre_tool);
        Ok(Arc::new(engine))
    }
}

fn allow3(by: Decider, why: impl Into<String>) -> Verdict {
    Verdict::Allow {
        tier: Tier::Classifier,
        by,
        why: why.into(),
    }
}

fn deny3(reason: DenyReason) -> Verdict {
    Verdict::Deny {
        tier: Tier::Classifier,
        reason,
    }
}

fn describe_rule(rule: &Rule) -> String {
    match &rule.arg {
        Some(arg) => format!("{} {}", rule.tool.as_str(), arg.as_str()),
        None => rule.tool.as_str().to_string(),
    }
}

fn args_preview(action: &Action<'_>) -> String {
    let text = match &action.kind {
        ActionKind::Shell { command, .. } => (*command).to_string(),
        _ => serde_json::to_string_pretty(action.args).unwrap_or_default(),
    };
    if text.len() <= ARGS_PREVIEW_BYTES {
        return text;
    }
    let mut cut = ARGS_PREVIEW_BYTES;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &text[..cut])
}

/// Allow-cache key: tool name plus a hash of the key-sorted argument JSON.
fn cache_key(name: &str, args: &Value) -> CacheKey {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_value(args, &mut hasher);
    (name.to_string(), hasher.finish())
}

fn hash_value(value: &Value, hasher: &mut impl Hasher) {
    match value {
        Value::Null => 0u8.hash(hasher),
        Value::Bool(b) => {
            1u8.hash(hasher);
            b.hash(hasher);
        }
        Value::Number(n) => {
            2u8.hash(hasher);
            n.to_string().hash(hasher);
        }
        Value::String(s) => {
            3u8.hash(hasher);
            s.hash(hasher);
        }
        Value::Array(items) => {
            4u8.hash(hasher);
            items.len().hash(hasher);
            for item in items {
                hash_value(item, hasher);
            }
        }
        Value::Object(map) => {
            5u8.hash(hasher);
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            keys.len().hash(hasher);
            for key in keys {
                key.hash(hasher);
                if let Some(v) = map.get(key) {
                    hash_value(v, hasher);
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cache_key_ignores_key_order() {
        let a = json!({"a": 1, "b": [true, null, "x"]});
        let b: Value = serde_json::from_str(r#"{"b": [true, null, "x"], "a": 1}"#).unwrap();
        assert_eq!(cache_key("t", &a), cache_key("t", &b));
        assert_ne!(cache_key("t", &a), cache_key("u", &a));
        assert_ne!(cache_key("t", &a), cache_key("t", &json!({"a": 2})));
    }

    #[test]
    fn deny_message_wording() {
        let msg = deny_message(&DenyReason::Classifier {
            rule: Some("D2".into()),
            explanation: "force push rewrites shared history".into(),
        });
        assert!(msg.starts_with(
            "Blocked by auto-mode safety policy: force push rewrites shared history [D2]. Do not try"
        ));
        assert!(msg.ends_with("or stop and ask the user."));
    }

    #[test]
    fn payload_serializes_snake_case_tier() {
        let payload = DenyPayload {
            policy: "auto-mode",
            tool_use_id: "call_1".to_string(),
            tool: "shell".to_string(),
            tier: Tier::Classifier,
            rule: None,
            consecutive: 1,
            total: 2,
        };
        assert_eq!(
            serde_json::to_value(payload).unwrap(),
            json!({
                "policy": "auto-mode",
                "tool_use_id": "call_1",
                "tool": "shell",
                "tier": "classifier",
                "consecutive": 1,
                "total": 2
            })
        );
    }

    #[test]
    fn hook_entries_shape() {
        let entries = ApprovalEngine::hook_entries().unwrap();
        assert_eq!(entries.len(), 2);
        let guard = &entries[1];
        assert_eq!(guard.point, HookPoint::PreToolExecution);
        assert_eq!(guard.capability, HookCapability::Guardrail);
        assert_eq!(guard.priority, i32::MIN);
        assert_eq!(guard.failure_policy, Some(HookFailurePolicy::FailClosed));
        assert_eq!(guard.timeout_ms, Some(AUTO_MODE_HOOK_TIMEOUT_MS));
        assert_eq!(
            guard.runtime.config_value().unwrap(),
            json!({"name": AUTO_MODE_HOOK_ID})
        );
        assert_eq!(entries[0].point, HookPoint::RunStarted);
        assert_eq!(entries[0].capability, HookCapability::Observe);
    }
}
