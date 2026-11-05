//! Pure-logic tests for the auto-mode approval engine: tiers, rules, counters,
//! escalation and the hook handlers, driven with a scripted classifier.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use async_trait::async_trait;
use meerkat_core::{
    HookDecision, HookEngine, HookInvocation, HookPoint, HookReasonCode, HookToolCall, HooksConfig,
    SessionId,
};
use meerkat_tui::approval::classifier::{
    ActionClassifier, ClassifierError, ClassifierInput, ClassifierVerdict, Environment,
};
use meerkat_tui::approval::counters::{AutoModeState, PauseCause};
use meerkat_tui::approval::escalation::{EscalationBroker, EscalationRequest, HumanDecision};
use meerkat_tui::approval::policy::{Decider, Tier};
use meerkat_tui::approval::rules::{AutoSettings, Rule, RuleKind, RuleOrigin, RuleSet};
use meerkat_tui::approval::{ApprovalConfig, ApprovalEngine, ApprovalEvent, DenyReason, Verdict};
use meerkat_tui::paths::ProjectRoot;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;

/// Classifier returning scripted verdicts (ALLOW once the script runs out) and
/// recording what it was shown.
#[derive(Default)]
struct ScriptedClassifier {
    script: parking_lot::Mutex<VecDeque<Result<ClassifierVerdict, ClassifierError>>>,
    calls: AtomicUsize,
    seen_transcripts: parking_lot::Mutex<Vec<String>>,
}

impl ScriptedClassifier {
    fn with(verdicts: Vec<Result<ClassifierVerdict, ClassifierError>>) -> Arc<Self> {
        Arc::new(Self {
            script: parking_lot::Mutex::new(verdicts.into()),
            ..Self::default()
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ActionClassifier for ScriptedClassifier {
    async fn classify(
        &self,
        input: ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.seen_transcripts.lock().push(input.transcript.render());
        self.script
            .lock()
            .pop_front()
            .unwrap_or(Ok(ClassifierVerdict::Allow))
    }
}

fn block(reason: &str) -> Result<ClassifierVerdict, ClassifierError> {
    Ok(ClassifierVerdict::Block {
        rule: Some("D2".into()),
        reason: reason.into(),
    })
}

struct Harness {
    _dir: tempfile::TempDir,
    engine: Arc<ApprovalEngine>,
    classifier: Arc<ScriptedClassifier>,
    events: mpsc::UnboundedReceiver<ApprovalEvent>,
    escalations: Option<mpsc::Receiver<EscalationRequest>>,
    session: SessionId,
    outside: String,
}

struct Options {
    rules: RuleSet,
    trust_project_builds: bool,
    enabled: bool,
    broker: bool,
    verdicts: Vec<Result<ClassifierVerdict, ClassifierError>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            rules: RuleSet::default(),
            trust_project_builds: true,
            enabled: true,
            broker: true,
            verdicts: Vec::new(),
        }
    }
}

fn harness(opts: Options) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::create_dir_all(project.join(".rkat")).unwrap();
    std::fs::write(project.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(project.join(".rkat/config.toml"), "").unwrap();
    std::fs::write(dir.path().join("outside.txt"), "secret").unwrap();
    let root = ProjectRoot::new(&project).unwrap();
    let classifier = ScriptedClassifier::with(opts.verdicts);
    let (events_tx, events) = mpsc::unbounded_channel();
    let (broker, escalations) = if opts.broker {
        let (b, rx) = EscalationBroker::new(4);
        (Some(b), Some(rx))
    } else {
        (None, None)
    };
    let engine = ApprovalEngine::new(ApprovalConfig {
        environment: Environment::basic(root.as_path(), root.as_path()),
        root,
        rules: opts.rules,
        settings: AutoSettings {
            enabled: opts.enabled,
            classifier_model: None,
            trust_project_builds: opts.trust_project_builds,
        },
        classifier: classifier.clone(),
        broker,
        posix_shell: true,
        events: events_tx,
        rules_path: None,
    });
    let outside = dir.path().join("outside.txt").to_str().unwrap().to_string();
    Harness {
        _dir: dir,
        engine,
        classifier,
        events,
        escalations,
        session: SessionId::new(),
        outside,
    }
}

impl Harness {
    async fn decide(&self, name: &str, args: Value) -> Verdict {
        self.engine.decide(&self.session, "tu", name, &args).await
    }

    fn drain_events(&mut self) -> Vec<ApprovalEvent> {
        let mut out = Vec::new();
        while let Ok(e) = self.events.try_recv() {
            out.push(e);
        }
        out
    }
}

fn tier_of(v: &Verdict) -> Tier {
    match v {
        Verdict::Allow { tier, .. } | Verdict::Deny { tier, .. } => *tier,
    }
}

fn deny_rule(tool: &str, arg: &str) -> RuleSet {
    let mut rules = RuleSet::default();
    rules.push(Rule::new(RuleKind::Deny, tool, Some(arg), RuleOrigin::Realm).unwrap());
    rules
}

#[tokio::test]
async fn read_inside_project_is_always_allowed() {
    let h = harness(Options::default());
    let v = h.decide("read_file", json!({"path": "src/main.rs"})).await;
    assert!(v.is_allowed());
    assert_eq!(tier_of(&v), Tier::AlwaysAllow);
    assert_eq!(h.classifier.calls(), 0);
}

#[tokio::test]
async fn read_outside_project_invokes_classifier() {
    let h = harness(Options::default());
    let v = h.decide("read_file", json!({"path": h.outside})).await;
    assert_eq!(h.classifier.calls(), 1);
    assert!(matches!(
        v,
        Verdict::Allow {
            tier: Tier::Classifier,
            by: Decider::Classifier,
            ..
        }
    ));
}

#[tokio::test]
async fn classifier_allow_is_cached_per_session() {
    let h = harness(Options::default());
    let args = json!({"path": h.outside});
    h.decide("read_file", args.clone()).await;
    h.decide("read_file", args.clone()).await;
    assert_eq!(h.classifier.calls(), 1);
    // A different session does not share the cache.
    h.engine
        .decide(&SessionId::new(), "tu", "read_file", &args)
        .await;
    assert_eq!(h.classifier.calls(), 2);
}

#[tokio::test]
async fn in_project_edit_is_tier_two() {
    let h = harness(Options::default());
    let v = h
        .decide(
            "edit_file",
            json!({"path": "src/main.rs", "old_string": "a", "new_string": "b"}),
        )
        .await;
    assert!(v.is_allowed());
    assert_eq!(tier_of(&v), Tier::InProjectEdit);
    assert_eq!(h.classifier.calls(), 0);
}

#[tokio::test]
async fn protected_path_edit_goes_to_classifier() {
    let h = harness(Options {
        verdicts: vec![block("editing the agent's own configuration")],
        ..Options::default()
    });
    let v = h
        .decide(
            "edit_file",
            json!({"path": ".rkat/config.toml", "old_string": "", "new_string": "x"}),
        )
        .await;
    assert_eq!(h.classifier.calls(), 1);
    assert!(matches!(
        v,
        Verdict::Deny {
            tier: Tier::Classifier,
            reason: DenyReason::Classifier { .. }
        }
    ));
}

#[tokio::test]
async fn deny_rule_wins_without_classifier() {
    let h = harness(Options {
        rules: deny_rule("shell", "git push*"),
        ..Options::default()
    });
    let v = h
        .decide("shell", json!({"command": "git push --force"}))
        .await;
    assert_eq!(h.classifier.calls(), 0);
    assert!(matches!(
        v,
        Verdict::Deny {
            tier: Tier::UserDeny,
            reason: DenyReason::UserRule { .. }
        }
    ));
}

#[tokio::test]
async fn cargo_test_depends_on_trust_project_builds() {
    let trusted = harness(Options::default());
    let v = trusted
        .decide("shell", json!({"command": "cargo test"}))
        .await;
    assert_eq!(tier_of(&v), Tier::AlwaysAllow);
    assert_eq!(trusted.classifier.calls(), 0);

    let untrusted = harness(Options {
        trust_project_builds: false,
        ..Options::default()
    });
    let v = untrusted
        .decide("shell", json!({"command": "cargo test"}))
        .await;
    assert_eq!(tier_of(&v), Tier::Classifier);
    assert_eq!(untrusted.classifier.calls(), 1);
}

#[tokio::test]
async fn disabled_auto_mode_denies_tier_three_outright() {
    let h = harness(Options {
        enabled: false,
        ..Options::default()
    });
    let v = h.decide("shell", json!({"command": "rm -rf build"})).await;
    assert_eq!(h.classifier.calls(), 0);
    assert!(matches!(
        v,
        Verdict::Deny {
            reason: DenyReason::AutoDisabled,
            ..
        }
    ));
    // Tiers 1–2 still work.
    assert!(
        h.decide("read_file", json!({"path": "src/main.rs"}))
            .await
            .is_allowed()
    );
}

#[tokio::test]
async fn three_blocks_pause_then_human_decides_and_resumes() {
    let mut h = harness(Options {
        verdicts: vec![block("a"), block("b"), block("c"), block("d")],
        ..Options::default()
    });
    for i in 0..3 {
        let v = h
            .decide("shell", json!({"command": format!("curl https://x/{i}")}))
            .await;
        assert!(!v.is_allowed());
    }
    let events = h.drain_events();
    assert!(
        events
            .iter()
            .any(|e| *e == ApprovalEvent::Paused(PauseCause::ConsecutiveDenials)),
        "{events:?}"
    );
    assert_eq!(
        h.engine.auto_state().1,
        AutoModeState::Paused {
            cause: PauseCause::ConsecutiveDenials
        }
    );

    let mut rx = h.escalations.take().unwrap();
    let human = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        let hint = req.classifier_hint.clone();
        let tool = req.tool.clone();
        req.respond.send(HumanDecision::AllowOnce).unwrap();
        (tool, hint)
    });
    let v = h
        .decide("shell", json!({"command": "curl https://x/4"}))
        .await;
    let (tool, hint) = human.await.unwrap();
    assert_eq!(tool, "shell");
    assert!(
        hint.unwrap().contains("BLOCK"),
        "classifier verdict shown as hint"
    );
    assert!(matches!(
        v,
        Verdict::Allow {
            by: Decider::Human,
            ..
        }
    ));
    assert!(h.drain_events().contains(&ApprovalEvent::Resumed));
    assert_eq!(h.engine.auto_state().1, AutoModeState::Active);
    assert_eq!(h.engine.auto_state().0.consecutive, 0);
}

#[tokio::test]
async fn stop_denies_pending_escalations_until_next_prompt() {
    let mut h = harness(Options {
        verdicts: vec![block("a"), block("b"), block("c")],
        ..Options::default()
    });
    for i in 0..3 {
        h.decide("shell", json!({"command": format!("curl https://x/{i}")}))
            .await;
    }
    let mut rx = h.escalations.take().unwrap();
    let human = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        req.respond.send(HumanDecision::Stop).unwrap();
        rx
    });
    let v = h
        .decide("shell", json!({"command": "curl https://y"}))
        .await;
    assert!(matches!(
        v,
        Verdict::Deny {
            reason: DenyReason::Stopped,
            ..
        }
    ));
    let _rx = human.await.unwrap();
    // Further escalations for the stopped session are denied without asking.
    let v = h
        .decide("shell", json!({"command": "curl https://z"}))
        .await;
    assert!(matches!(
        v,
        Verdict::Deny {
            reason: DenyReason::Stopped,
            ..
        }
    ));
}

#[tokio::test]
async fn classifier_error_escalates_and_is_not_counted() {
    let mut h = harness(Options {
        verdicts: vec![Err(ClassifierError::Timeout)],
        ..Options::default()
    });
    let mut rx = h.escalations.take().unwrap();
    let human = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        req.respond.send(HumanDecision::Deny).unwrap();
    });
    let v = h
        .decide("shell", json!({"command": "curl https://x"}))
        .await;
    human.await.unwrap();
    assert!(matches!(
        v,
        Verdict::Deny {
            reason: DenyReason::Human,
            ..
        }
    ));
    assert_eq!(h.engine.auto_state().0.total, 0);
}

#[tokio::test]
async fn classifier_error_without_broker_denies() {
    let h = harness(Options {
        broker: false,
        verdicts: vec![Err(ClassifierError::Provider("down".into()))],
        ..Options::default()
    });
    let v = h
        .decide("shell", json!({"command": "curl https://x"}))
        .await;
    assert!(matches!(
        v,
        Verdict::Deny {
            reason: DenyReason::ClassifierUnavailable(_),
            ..
        }
    ));
}

#[tokio::test]
async fn human_allow_and_add_rule_allows_future_calls() {
    let mut h = harness(Options {
        verdicts: vec![block("a"), block("b"), block("c")],
        ..Options::default()
    });
    for i in 0..3 {
        h.decide("shell", json!({"command": format!("curl https://x/{i}")}))
            .await;
    }
    let mut rx = h.escalations.take().unwrap();
    let human = tokio::spawn(async move {
        let req = rx.recv().await.unwrap();
        let rule = Rule::new(
            RuleKind::Allow,
            "shell",
            Some("make lint*"),
            RuleOrigin::Session,
        )
        .unwrap();
        req.respond
            .send(HumanDecision::AllowAndAddRule(Box::new(rule)))
            .unwrap();
    });
    assert!(
        h.decide("shell", json!({"command": "make lint"}))
            .await
            .is_allowed()
    );
    human.await.unwrap();
    let calls = h.classifier.calls();
    let v = h
        .decide("shell", json!({"command": "make lint --fix"}))
        .await;
    assert_eq!(tier_of(&v), Tier::AlwaysAllow);
    assert_eq!(h.classifier.calls(), calls);
}

#[tokio::test]
async fn broad_rules_are_dropped_with_an_event() {
    let mut rules = RuleSet::default();
    rules.push(Rule::new(RuleKind::Allow, "shell", None, RuleOrigin::Global).unwrap());
    rules.push(
        Rule::new(
            RuleKind::Allow,
            "shell",
            Some("cargo nextest run*"),
            RuleOrigin::Global,
        )
        .unwrap(),
    );
    let mut h = harness(Options {
        rules,
        ..Options::default()
    });
    let events = h.drain_events();
    assert!(
        matches!(&events[..], [ApprovalEvent::RulesDropped(d)] if d.len() == 1),
        "{events:?}"
    );
    assert_eq!(h.engine.rules().allow.len(), 1);
    // The dropped blanket rule no longer bypasses review.
    h.decide("shell", json!({"command": "rm -rf target"})).await;
    assert_eq!(h.classifier.calls(), 1);
}

fn invocation(session: &SessionId, point: HookPoint) -> HookInvocation {
    HookInvocation {
        point,
        session_id: session.clone(),
        turn_number: Some(1),
        prompt: None,
        error: None,
        llm_request: None,
        llm_response: None,
        tool_call: None,
        tool_result: None,
    }
}

fn pre_tool(session: &SessionId, name: &str, args: Value) -> HookInvocation {
    HookInvocation {
        tool_call: Some(HookToolCall {
            tool_use_id: "call-1".into(),
            name: name.into(),
            args,
        }),
        ..invocation(session, HookPoint::PreToolExecution)
    }
}

#[tokio::test]
async fn hook_engine_end_to_end() {
    let h = harness(Options {
        rules: deny_rule("shell", "git push*"),
        ..Options::default()
    });
    let hooks: Arc<dyn HookEngine> = h.engine.build_hook_engine(HooksConfig::default()).unwrap();

    // RunStarted feeds the reasoning-blind transcript.
    let mut started = invocation(&h.session, HookPoint::RunStarted);
    started.prompt = Some("please tidy up the repo".into());
    let report = hooks.execute(started, None).await.unwrap();
    assert!(report.decision.is_none());
    assert!(
        h.engine
            .transcript(&h.session)
            .unwrap()
            .contains("please tidy up the repo")
    );

    let report = hooks
        .execute(
            pre_tool(&h.session, "shell", json!({"command": "git push --force"})),
            None,
        )
        .await
        .unwrap();
    match report.decision {
        Some(HookDecision::Deny {
            hook_id,
            reason_code,
            message,
            payload,
        }) => {
            assert_eq!(hook_id.0, "tui-auto-mode");
            assert_eq!(reason_code, HookReasonCode::PolicyViolation);
            assert!(
                message.starts_with("Blocked by auto-mode safety policy: "),
                "{message}"
            );
            assert!(message.contains("Do not try to achieve the same effect another way"));
            let payload = payload.unwrap();
            assert_eq!(payload["policy"], "auto-mode");
            assert_eq!(payload["tier"], "user_deny");
            assert_eq!(payload["rule"], "deny#1");
        }
        other => panic!("expected deny, got {other:?}"),
    }

    let report = hooks
        .execute(
            pre_tool(
                &h.session,
                "edit_file",
                json!({"path": "src/main.rs", "old_string": "a", "new_string": "b"}),
            ),
            None,
        )
        .await
        .unwrap();
    assert_eq!(report.decision, Some(HookDecision::Allow));

    // The transcript records the tool calls with verdicts and the user prompt,
    // and is what the classifier sees.
    let transcript = h.engine.transcript(&h.session).unwrap();
    assert!(transcript.contains("git push --force"));
    assert!(transcript.contains("-> blocked"));
    assert!(transcript.contains("-> allowed"));
}

#[tokio::test]
async fn hook_classifier_block_reports_safety_violation() {
    let h = harness(Options {
        verdicts: vec![block("force pushing rewrites shared history")],
        ..Options::default()
    });
    let hooks = h.engine.build_hook_engine(HooksConfig::default()).unwrap();
    let mut started = invocation(&h.session, HookPoint::RunStarted);
    started.prompt = Some("ship it".into());
    hooks.execute(started, None).await.unwrap();
    let report = hooks
        .execute(
            pre_tool(&h.session, "shell", json!({"command": "git push --force"})),
            None,
        )
        .await
        .unwrap();
    let Some(HookDecision::Deny {
        reason_code,
        message,
        payload,
        ..
    }) = report.decision
    else {
        panic!("expected deny");
    };
    assert_eq!(reason_code, HookReasonCode::SafetyViolation);
    assert!(message.contains("force pushing rewrites shared history [D2]"));
    let payload = payload.unwrap();
    assert_eq!(payload["tier"], "classifier");
    assert_eq!(payload["consecutive"], 1);
    assert_eq!(payload["total"], 1);
    // The classifier saw the user's prompt.
    assert!(h.classifier.seen_transcripts.lock()[0].contains("ship it"));
}

#[tokio::test]
async fn workspace_memory_save_is_in_project_edit() {
    let h = harness(Options::default());
    let v = h
        .decide(
            "memory_save",
            json!({"scope": "workspace", "name": "build", "content": "make ci"}),
        )
        .await;
    assert!(v.is_allowed());
    assert_eq!(tier_of(&v), Tier::InProjectEdit);
    assert_eq!(h.classifier.calls(), 0);
}

#[tokio::test]
async fn user_memory_save_reaches_classifier() {
    let h = harness(Options {
        verdicts: vec![block("planting a user-level note")],
        ..Options::default()
    });
    let v = h
        .decide(
            "memory_save",
            json!({"scope": "user", "name": "style", "content": "x"}),
        )
        .await;
    assert_eq!(h.classifier.calls(), 1);
    assert!(matches!(
        v,
        Verdict::Deny {
            tier: Tier::Classifier,
            ..
        }
    ));
}

#[tokio::test]
async fn memory_list_is_tier_one() {
    let h = harness(Options::default());
    let v = h.decide("memory_list", json!({})).await;
    assert!(v.is_allowed());
    assert_eq!(tier_of(&v), Tier::AlwaysAllow);
    assert_eq!(h.classifier.calls(), 0);
}

#[tokio::test]
async fn allow_rule_makes_user_memory_save_tier_one() {
    let mut rules = RuleSet::default();
    rules.push(
        Rule::new(
            RuleKind::Allow,
            "memory_save",
            Some("user/*"),
            RuleOrigin::Realm,
        )
        .unwrap(),
    );
    let h = harness(Options {
        rules,
        ..Options::default()
    });
    let v = h
        .decide(
            "memory_save",
            json!({"scope": "user", "name": "style", "content": "x"}),
        )
        .await;
    assert!(v.is_allowed());
    assert_eq!(tier_of(&v), Tier::AlwaysAllow);
    assert_eq!(h.classifier.calls(), 0);
}
