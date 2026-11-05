//! The Tier-3 classifier seam.
//!
//! [`ActionClassifier`] is the only interface the approval engine uses; the
//! two-stage LLM implementation lives behind it (prompts and response parsing in
//! [`super::prompts`]). Errors never count as denials: the engine escalates to
//! the human instead.

use super::prompts::{Stage1, parse_stage1, render_stage1, render_stage2, stage2_or_block};
use super::transcript::ApprovalTranscript;
use async_trait::async_trait;
use futures::StreamExt;
use meerkat_client::{LlmClient, LlmDoneOutcome, LlmEvent, LlmRequest};
use meerkat_core::{Message, Provider, SystemMessage, UserMessage};
use serde_json::Value;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Facts about the environment shown to the classifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    /// Canonical project root (the trust boundary).
    pub project_root: PathBuf,
    /// Process working directory.
    pub cwd: PathBuf,
    /// `origin` remote URL, if any.
    pub git_remote: Option<String>,
    /// Default branch name, if known.
    pub default_branch: Option<String>,
    /// Operating system (`std::env::consts::OS`).
    pub os: String,
}

impl Environment {
    /// Environment without git facts.
    pub fn basic(project_root: impl Into<PathBuf>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            cwd: cwd.into(),
            git_remote: None,
            default_branch: None,
            os: std::env::consts::OS.to_string(),
        }
    }

    /// Environment with git remote and default branch discovered via `git`
    /// (best effort; failures leave the fields empty).
    pub async fn detect(project_root: &Path, cwd: &Path) -> Self {
        let mut env = Self::basic(project_root, cwd);
        env.git_remote = git_output(project_root, &["remote", "get-url", "origin"]).await;
        env.default_branch = git_output(
            project_root,
            &["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        )
        .await
        .map(|b| b.strip_prefix("origin/").map_or(b.clone(), str::to_string));
        env
    }
}

async fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Everything the classifier may see. Deliberately excludes assistant prose,
/// reasoning and tool outputs.
#[derive(Debug, Clone, Copy)]
pub struct ClassifierInput<'a> {
    /// Environment and trust boundary.
    pub environment: &'a Environment,
    /// Reasoning-blind transcript (user prompts and prior tool calls).
    pub transcript: &'a ApprovalTranscript,
    /// Tool under review.
    pub tool: &'a str,
    /// Its arguments.
    pub args: &'a Value,
    /// Content of an in-project script the action runs (≤ 8 KiB), if any.
    pub script_payload: Option<&'a str>,
}

/// The classifier's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifierVerdict {
    /// Allow the action.
    Allow,
    /// Block the action.
    Block {
        /// Block-rule id (e.g. `D1`), if the classifier named one.
        rule: Option<String>,
        /// Explanation passed back to the agent.
        reason: String,
    },
}

impl fmt::Display for ClassifierVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => f.write_str("ALLOW"),
            Self::Block {
                rule: Some(rule),
                reason,
            } => write!(f, "BLOCK [{rule}]: {reason}"),
            Self::Block { rule: None, reason } => write!(f, "BLOCK: {reason}"),
        }
    }
}

/// Classifier failures; never counted as denials.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClassifierError {
    /// A stage exceeded its time budget.
    #[error("classifier timed out")]
    Timeout,
    /// The provider call failed.
    #[error("classifier provider error: {0}")]
    Provider(String),
    /// The response could not be parsed.
    #[error("unparseable classifier response: {0}")]
    Unparseable(String),
}

/// Reviews Tier-3 actions.
#[async_trait]
pub trait ActionClassifier: Send + Sync {
    /// Decide whether the action may run.
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError>;
}

/// Placeholder used until an LLM classifier is configured: every Tier-3 action
/// is escalated to the human (or denied when no human is available).
#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableClassifier;

#[async_trait]
impl ActionClassifier for UnavailableClassifier {
    async fn classify(
        &self,
        _input: ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError> {
        Err(ClassifierError::Provider(
            "no classifier is configured".to_string(),
        ))
    }
}

/// Default per-stage time budget.
pub const DEFAULT_STAGE_TIMEOUT: Duration = Duration::from_secs(30);
/// Default stage-1 output budget (one word).
pub const DEFAULT_STAGE1_MAX_TOKENS: u32 = 32;
/// Default stage-2 output budget (reasoning plus verdict lines).
pub const DEFAULT_STAGE2_MAX_TOKENS: u32 = 1024;

/// Default classifier model for `provider` (`None` for [`Provider::Other`]).
pub fn classifier_default_model(provider: Provider) -> Option<&'static str> {
    match provider {
        Provider::Anthropic => Some("claude-sonnet-4-6"),
        Provider::OpenAI => Some("gpt-5.2"),
        Provider::Gemini => Some("gemini-3.1-flash-lite"),
        Provider::Other => None,
    }
}

/// Provider parameters that minimise hidden reasoning, so the tiny stage-1
/// budget is spent on the answer.
///
/// OpenAI reads `reasoning_effort` (`gpt-5` / `gpt-5-mini` accept `minimal`,
/// codex models start at `low`, later `gpt-5.x` accept `none`); Gemini reads
/// `thinking_budget`; Anthropic thinking is off unless requested.
pub fn low_reasoning_params(provider: Provider, model: &str) -> Option<Value> {
    match provider {
        Provider::OpenAI => {
            let effort = if model.contains("codex") {
                "low"
            } else if model == "gpt-5" || model.starts_with("gpt-5-") {
                "minimal"
            } else if model.starts_with("gpt-5.") {
                "none"
            } else {
                return None;
            };
            Some(serde_json::json!({ "reasoning_effort": effort }))
        }
        Provider::Gemini => Some(serde_json::json!({ "thinking_budget": 0 })),
        Provider::Anthropic | Provider::Other => None,
    }
}

/// Classifier stage, for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    One,
    Two,
}

impl Stage {
    fn number(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }
}

/// Two-stage LLM classifier: a one-word fast pass, and a reasoned second
/// pass for anything not clearly allowed.
pub struct LlmClassifier {
    client: Arc<dyn LlmClient>,
    provider: Provider,
    model: String,
    stage_timeout: Duration,
    stage1_max_tokens: u32,
    stage2_max_tokens: u32,
}

impl fmt::Debug for LlmClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmClassifier")
            .field("client", &self.client.provider())
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("stage_timeout", &self.stage_timeout)
            .field("stage1_max_tokens", &self.stage1_max_tokens)
            .field("stage2_max_tokens", &self.stage2_max_tokens)
            .finish()
    }
}

impl LlmClassifier {
    /// Classifier on `client` using `model` (served by `provider`).
    pub fn new(client: Arc<dyn LlmClient>, provider: Provider, model: impl Into<String>) -> Self {
        Self {
            client,
            provider,
            model: model.into(),
            stage_timeout: DEFAULT_STAGE_TIMEOUT,
            stage1_max_tokens: DEFAULT_STAGE1_MAX_TOKENS,
            stage2_max_tokens: DEFAULT_STAGE2_MAX_TOKENS,
        }
    }

    /// Override the per-stage time budget.
    #[must_use]
    pub fn with_stage_timeout(mut self, timeout: Duration) -> Self {
        self.stage_timeout = timeout;
        self
    }

    /// Override the output token budgets of both stages.
    #[must_use]
    pub fn with_max_tokens(mut self, stage1: u32, stage2: u32) -> Self {
        self.stage1_max_tokens = stage1;
        self.stage2_max_tokens = stage2;
        self
    }

    /// The classifier model.
    pub fn model(&self) -> &str {
        &self.model
    }

    fn request(&self, system: String, user: String, max_tokens: u32) -> LlmRequest {
        LlmRequest {
            model: self.model.clone(),
            messages: vec![
                Message::System(SystemMessage { content: system }),
                Message::User(UserMessage::text(user)),
            ],
            tools: Vec::new(),
            max_tokens,
            temperature: Some(0.0),
            stop_sequences: None,
            provider_params: low_reasoning_params(self.provider, &self.model),
        }
    }

    /// Run one completion and return its concatenated text.
    async fn complete(
        &self,
        stage: Stage,
        request: &LlmRequest,
    ) -> Result<String, ClassifierError> {
        let started = Instant::now();
        let result = tokio::time::timeout(self.stage_timeout, collect_text(&*self.client, request))
            .await
            .unwrap_or(Err(ClassifierError::Timeout));
        tracing::debug!(
            stage = stage.number(),
            model = %self.model,
            elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            ok = result.is_ok(),
            "classifier stage finished"
        );
        result
    }
}

async fn collect_text(
    client: &dyn LlmClient,
    request: &LlmRequest,
) -> Result<String, ClassifierError> {
    let mut stream = client.stream(request);
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event.map_err(|e| ClassifierError::Provider(e.to_string()))? {
            LlmEvent::TextDelta { delta, .. } => text.push_str(&delta),
            LlmEvent::Done {
                outcome: LlmDoneOutcome::Success { .. },
            } => break,
            LlmEvent::Done {
                outcome: LlmDoneOutcome::Error { error },
            } => return Err(ClassifierError::Provider(error.to_string())),
            _ => {}
        }
    }
    Ok(text)
}

#[async_trait]
impl ActionClassifier for LlmClassifier {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError> {
        let (system, user) = render_stage1(&input);
        let request = self.request(system, user, self.stage1_max_tokens);
        let answer = self.complete(Stage::One, &request).await?;
        let stage1 = parse_stage1(answer.trim());
        tracing::debug!(model = %self.model, tool = input.tool, ?stage1, "classifier stage 1 verdict");
        if stage1 == Stage1::Allow {
            return Ok(ClassifierVerdict::Allow);
        }

        let (system, user) = render_stage2(&input);
        let request = self.request(system, user, self.stage2_max_tokens);
        let answer = self.complete(Stage::Two, &request).await?;
        let verdict = stage2_or_block(&answer);
        tracing::debug!(model = %self.model, tool = input.tool, %verdict, "classifier stage 2 verdict");
        Ok(verdict)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unavailable_classifier_errors() {
        let env = Environment::basic("/p", "/p");
        let transcript = ApprovalTranscript::default();
        let args = serde_json::json!({});
        let input = ClassifierInput {
            environment: &env,
            transcript: &transcript,
            tool: "x",
            args: &args,
            script_payload: None,
        };
        assert!(matches!(
            UnavailableClassifier.classify(input).await,
            Err(ClassifierError::Provider(_))
        ));
    }

    #[test]
    fn verdict_display() {
        let v = ClassifierVerdict::Block {
            rule: Some("D1".into()),
            reason: "force push".into(),
        };
        assert_eq!(v.to_string(), "BLOCK [D1]: force push");
        assert_eq!(ClassifierVerdict::Allow.to_string(), "ALLOW");
    }
}
