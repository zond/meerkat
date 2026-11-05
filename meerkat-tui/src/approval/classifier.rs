//! The Tier-3 classifier seam.
//!
//! [`ActionClassifier`] is the only interface the approval engine uses; the
//! two-stage LLM implementation lives behind it (prompts and response parsing in
//! [`super::prompts`]). Errors never count as denials: the engine escalates to
//! the human instead.

use super::prompts::{
    Evidence, Stage1, Stage2Mode, UNTRUSTED_TOOL_RESULT_PREFIX, parse_stage1, render_stage1,
    render_stage2, render_stage2_final, stage2_or_block,
};
use super::transcript::ApprovalTranscript;
use crate::tools::{READ_FILE, ReadOnlyProjectTools};
use async_trait::async_trait;
use futures::StreamExt;
use meerkat_client::{BlockAssembler, LlmClient, LlmDoneOutcome, LlmError, LlmEvent, LlmRequest};
use meerkat_core::types::{AssistantBlock, BlockAssistantMessage, ToolCallView, ToolResult};
use meerkat_core::{
    AgentToolDispatcher, Message, Provider, StopReason, SystemMessage, ToolDef, UserMessage,
};
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

/// Default stage-1 time budget.
pub const DEFAULT_STAGE_TIMEOUT: Duration = Duration::from_secs(30);
/// Default stage-2 time budget: the whole tool loop, tool runs included.
pub const DEFAULT_STAGE2_TIMEOUT: Duration = Duration::from_secs(60);
/// Maximum tool calls stage 2 may execute.
pub const MAX_STAGE2_TOOL_CALLS: usize = 6;
/// Maximum bytes of one tool result shown to stage 2 (excluding the prefix).
pub const MAX_TOOL_RESULT_BYTES: usize = 16 * 1024;
/// Maximum tool-enabled LLM round-trips in stage 2 (the final tool-less call
/// after budget exhaustion comes on top).
pub const MAX_STAGE2_ROUND_TRIPS: usize = 8;
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

/// Two-stage LLM classifier: a one-word fast pass, and a reasoned second
/// pass for anything not clearly allowed.
pub struct LlmClassifier {
    client: Arc<dyn LlmClient>,
    provider: Provider,
    model: String,
    stage_timeout: Duration,
    stage2_timeout: Duration,
    stage1_max_tokens: u32,
    stage2_max_tokens: u32,
    project_tools: Option<Arc<ReadOnlyProjectTools>>,
}

impl fmt::Debug for LlmClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmClassifier")
            .field("client", &self.client.provider())
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("stage_timeout", &self.stage_timeout)
            .field("stage2_timeout", &self.stage2_timeout)
            .field("stage1_max_tokens", &self.stage1_max_tokens)
            .field("stage2_max_tokens", &self.stage2_max_tokens)
            .field("project_tools", &self.project_tools.is_some())
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
            stage2_timeout: DEFAULT_STAGE2_TIMEOUT,
            stage1_max_tokens: DEFAULT_STAGE1_MAX_TOKENS,
            stage2_max_tokens: DEFAULT_STAGE2_MAX_TOKENS,
            project_tools: None,
        }
    }

    /// Override the time budget of both stages.
    #[must_use]
    pub fn with_stage_timeout(mut self, timeout: Duration) -> Self {
        self.stage_timeout = timeout;
        self.stage2_timeout = timeout;
        self
    }

    /// Override the stage-2 time budget (the whole tool loop).
    #[must_use]
    pub fn with_stage2_timeout(mut self, timeout: Duration) -> Self {
        self.stage2_timeout = timeout;
        self
    }

    /// Let stage 2 read project files through `tools`. Without them stage 2
    /// decides from the attached context only.
    #[must_use]
    pub fn with_project_tools(mut self, tools: Arc<ReadOnlyProjectTools>) -> Self {
        self.project_tools = Some(tools);
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

    /// Run the stage-1 completion and return its concatenated text.
    async fn complete_stage1(&self, request: &LlmRequest) -> Result<String, ClassifierError> {
        let started = Instant::now();
        let result = tokio::time::timeout(self.stage_timeout, collect_text(&*self.client, request))
            .await
            .unwrap_or(Err(ClassifierError::Timeout));
        tracing::debug!(
            stage = 1,
            model = %self.model,
            elapsed_ms = elapsed_ms(started),
            ok = result.is_ok(),
            "classifier stage finished"
        );
        result
    }

    /// Stage 2 within its time budget.
    async fn stage2(
        &self,
        input: &ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError> {
        let started = Instant::now();
        let result = tokio::time::timeout(self.stage2_timeout, self.stage2_inner(input))
            .await
            .unwrap_or(Err(ClassifierError::Timeout));
        tracing::debug!(
            stage = 2,
            model = %self.model,
            elapsed_ms = elapsed_ms(started),
            ok = result.is_ok(),
            "classifier stage finished"
        );
        result
    }

    async fn stage2_inner(
        &self,
        input: &ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError> {
        let (text, run) = match &self.project_tools {
            None => (self.stage2_plain(input).await?, Stage2Run::Plain),
            Some(tools) => match self.stage2_tool_loop(input, tools).await {
                Ok((text, stats)) => (text, Stage2Run::Tools(stats)),
                Err(LoopError::ToolsRejected(error)) => {
                    tracing::debug!(
                        model = %self.model,
                        %error,
                        "classifier endpoint rejected tools; retrying stage 2 without tools"
                    );
                    (self.stage2_plain(input).await?, Stage2Run::ToolsRejected)
                }
                Err(LoopError::Classifier(err)) => return Err(err),
            },
        };
        tracing::debug!(model = %self.model, tool = input.tool, ?run, "classifier stage 2 mode");
        let verdict = match stage2_or_block(&text) {
            ClassifierVerdict::Block { rule, reason } => ClassifierVerdict::Block {
                rule,
                reason: match run.suffix() {
                    Some(suffix) => format!("{reason} {suffix}"),
                    None => reason,
                },
            },
            ClassifierVerdict::Allow => ClassifierVerdict::Allow,
        };
        Ok(verdict)
    }

    /// Tool-less stage 2 (no project tools, or the endpoint rejected them).
    async fn stage2_plain(&self, input: &ClassifierInput<'_>) -> Result<String, ClassifierError> {
        let (system, user) = render_stage2(input, Stage2Mode::NoTools);
        let request = self.request(system, user, self.stage2_max_tokens);
        collect_text(&*self.client, &request).await
    }

    /// Stage 2 as a bounded tool loop over the read-only project tools.
    async fn stage2_tool_loop(
        &self,
        input: &ClassifierInput<'_>,
        tools: &ReadOnlyProjectTools,
    ) -> Result<(String, ToolStats), LoopError> {
        let (system, user) = render_stage2(input, Stage2Mode::WithTools);
        let mut request = self.request(system.clone(), user.clone(), self.stage2_max_tokens);
        request.tools = tools.tools().iter().cloned().collect::<Vec<Arc<ToolDef>>>();
        let mut stats = ToolStats::default();
        let mut evidence: Vec<OwnedEvidence> = Vec::new();

        for round in 0..MAX_STAGE2_ROUND_TRIPS {
            if stats.calls >= MAX_STAGE2_TOOL_CALLS {
                break;
            }
            let reply = match collect_turn(&*self.client, &request).await {
                Ok(reply) => reply,
                Err(err) if round == 0 && tools_rejected(&err) => {
                    return Err(LoopError::ToolsRejected(err.to_string()));
                }
                Err(err) => {
                    return Err(LoopError::Classifier(ClassifierError::Provider(
                        err.to_string(),
                    )));
                }
            };
            if !reply.has_tool_calls() {
                return Ok((assistant_text(&reply), stats));
            }
            let mut results = Vec::new();
            for call in reply.tool_calls() {
                let result = if stats.calls >= MAX_STAGE2_TOOL_CALLS {
                    ToolResult::new(
                        call.id.to_string(),
                        format!(
                            "{UNTRUSTED_TOOL_RESULT_PREFIX}\nnot executed: {}",
                            super::prompts::TOOL_BUDGET_EXHAUSTED
                        ),
                        true,
                    )
                } else {
                    stats.calls += 1;
                    run_tool(tools, call, &mut stats).await
                };
                evidence.push(OwnedEvidence {
                    tool: call.name.to_string(),
                    args: call.args.get().to_string(),
                    result: result.text_content(),
                    is_error: result.is_error,
                });
                results.push(result);
            }
            request.messages.push(Message::BlockAssistant(reply));
            request.messages.push(Message::ToolResults { results });
        }

        // Budget exhausted: one last call, tools disabled, with the evidence
        // replayed as text (providers reject tool blocks without tool definitions).
        stats.exhausted = true;
        let views: Vec<Evidence<'_>> = evidence
            .iter()
            .map(|e| Evidence {
                tool: &e.tool,
                args: &e.args,
                result: &e.result,
                is_error: e.is_error,
            })
            .collect();
        let request = self.request(
            system,
            render_stage2_final(&user, &views),
            self.stage2_max_tokens,
        );
        let text = collect_text(&*self.client, &request)
            .await
            .map_err(LoopError::Classifier)?;
        Ok((text, stats))
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Why the stage-2 tool loop stopped without a text answer.
enum LoopError {
    /// The endpoint rejected the first tool-enabled request.
    ToolsRejected(String),
    /// Any other failure.
    Classifier(ClassifierError),
}

/// What the stage-2 tool loop did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ToolStats {
    /// Tool calls executed.
    calls: usize,
    /// Successful `read_file` calls.
    files: usize,
    /// Successful `glob` / `grep` calls.
    searches: usize,
    /// Whether the budget ran out and the final tool-less call was made.
    exhausted: bool,
}

/// How stage 2 ran, for logging and the verdict's reason suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage2Run {
    /// No project tools configured.
    Plain,
    /// The tool loop ran.
    Tools(ToolStats),
    /// Tools were rejected; stage 2 ran without them.
    ToolsRejected,
}

impl Stage2Run {
    fn suffix(self) -> Option<String> {
        match self {
            Self::Plain => None,
            Self::ToolsRejected => Some("[tools unavailable]".to_string()),
            Self::Tools(stats) => {
                let plural = |n: usize, one: &str, many: &str| {
                    format!("{n} {}", if n == 1 { one } else { many })
                };
                let mut out = format!("[read {}", plural(stats.files, "file", "files"));
                if stats.searches > 0 {
                    out.push_str(", ");
                    out.push_str(&plural(stats.searches, "search", "searches"));
                }
                if stats.exhausted {
                    out.push_str("; tool budget exhausted");
                }
                out.push(']');
                Some(out)
            }
        }
    }
}

/// A tool round-trip kept for the final tool-less call.
struct OwnedEvidence {
    tool: String,
    args: String,
    result: String,
    is_error: bool,
}

/// Whether a failed first tool-enabled request means the endpoint does not
/// accept tools (a 400, or an error that mentions tools).
fn tools_rejected(err: &LlmError) -> bool {
    matches!(err, LlmError::InvalidRequest { .. })
        || err.to_string().to_ascii_lowercase().contains("tool")
}

/// Dispatch one call and wrap its output for the classifier: untrusted-data
/// prefix, truncation at [`MAX_TOOL_RESULT_BYTES`]. Dispatch errors become
/// error results so the loop continues.
async fn run_tool(
    tools: &ReadOnlyProjectTools,
    call: ToolCallView<'_>,
    stats: &mut ToolStats,
) -> ToolResult {
    let (text, is_error) = match tools.dispatch(call).await {
        Ok(result) => (result.text_content(), result.is_error),
        Err(err) => (err.to_string(), true),
    };
    if !is_error {
        if call.name == READ_FILE {
            stats.files += 1;
        } else {
            stats.searches += 1;
        }
    }
    ToolResult::new(
        call.id.to_string(),
        format!(
            "{UNTRUSTED_TOOL_RESULT_PREFIX}\n{}",
            truncate_result(&text, MAX_TOOL_RESULT_BYTES)
        ),
        is_error,
    )
}

/// Cut `text` to at most `max` bytes on a char boundary, with a marker.
fn truncate_result(text: &str, max: usize) -> std::borrow::Cow<'_, str> {
    if text.len() <= max {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut cut = max;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    std::borrow::Cow::Owned(format!(
        "{}\n[truncated: showing {cut} of {} bytes]",
        &text[..cut],
        text.len()
    ))
}

/// Concatenated text blocks of an assistant reply.
fn assistant_text(reply: &BlockAssistantMessage) -> String {
    let mut out = String::new();
    for block in &reply.blocks {
        if let AssistantBlock::Text { text, .. } = block {
            out.push_str(text);
        }
    }
    out
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

/// Fold one streamed reply (text, reasoning and tool calls) into an assistant
/// message, keeping provider metadata so the reply can be sent back.
async fn collect_turn(
    client: &dyn LlmClient,
    request: &LlmRequest,
) -> Result<BlockAssistantMessage, LlmError> {
    let mut stream = client.stream(request);
    let mut assembler = BlockAssembler::new();
    let mut reasoning_started = false;
    let mut stop_reason = StopReason::EndTurn;
    while let Some(event) = stream.next().await {
        match event? {
            LlmEvent::TextDelta { delta, meta } => assembler.on_text_delta(&delta, meta),
            LlmEvent::ReasoningDelta { delta } => {
                if !reasoning_started {
                    reasoning_started = true;
                    assembler.on_reasoning_start();
                }
                assembler.on_reasoning_delta(&delta).map_err(stream_error)?;
            }
            LlmEvent::ReasoningComplete { text, meta } => {
                if !reasoning_started {
                    assembler.on_reasoning_start();
                    assembler.on_reasoning_delta(&text).map_err(stream_error)?;
                }
                assembler.on_reasoning_complete(meta);
                reasoning_started = false;
            }
            LlmEvent::ToolCallDelta {
                id,
                name,
                args_delta,
            } => {
                if assembler
                    .on_tool_call_delta(&id, name.as_deref(), &args_delta)
                    .is_err()
                {
                    assembler
                        .on_tool_call_start(id.clone())
                        .map_err(stream_error)?;
                    assembler
                        .on_tool_call_delta(&id, name.as_deref(), &args_delta)
                        .map_err(stream_error)?;
                }
            }
            LlmEvent::ToolCallComplete {
                id,
                name,
                args,
                meta,
            } => {
                let args = serde_json::value::to_raw_value(&args).map_err(stream_error)?;
                assembler
                    .on_tool_call_complete(id, name, args, meta)
                    .map_err(stream_error)?;
            }
            LlmEvent::Done {
                outcome: LlmDoneOutcome::Success { stop_reason: sr },
            } => {
                stop_reason = sr;
                break;
            }
            LlmEvent::Done {
                outcome: LlmDoneOutcome::Error { error },
            } => return Err(error),
            LlmEvent::UsageUpdate { .. } => {}
        }
    }
    Ok(BlockAssistantMessage {
        blocks: assembler.finalize(),
        stop_reason,
    })
}

fn stream_error(err: impl fmt::Display) -> LlmError {
    LlmError::StreamParseError {
        message: err.to_string(),
    }
}

#[async_trait]
impl ActionClassifier for LlmClassifier {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError> {
        let (system, user) = render_stage1(&input);
        let request = self.request(system, user, self.stage1_max_tokens);
        let answer = self.complete_stage1(&request).await?;
        let stage1 = parse_stage1(answer.trim());
        tracing::debug!(model = %self.model, tool = input.tool, ?stage1, "classifier stage 1 verdict");
        if stage1 == Stage1::Allow {
            return Ok(ClassifierVerdict::Allow);
        }

        let verdict = self.stage2(&input).await?;
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
    fn truncation_marks_and_respects_boundaries() {
        assert_eq!(truncate_result("abc", 5), "abc");
        let long = "é".repeat(10);
        let out = truncate_result(&long, 5);
        assert!(out.starts_with("éé\n[truncated: showing 4 of 20 bytes]"));
    }

    #[test]
    fn run_suffixes() {
        assert_eq!(Stage2Run::Plain.suffix(), None);
        assert_eq!(
            Stage2Run::ToolsRejected.suffix().as_deref(),
            Some("[tools unavailable]")
        );
        let stats = ToolStats {
            calls: 4,
            files: 3,
            searches: 1,
            exhausted: false,
        };
        assert_eq!(
            Stage2Run::Tools(stats).suffix().as_deref(),
            Some("[read 3 files, 1 search]")
        );
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
