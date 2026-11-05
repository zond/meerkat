//! End-to-end memory notes: approval tiers through the `ApprovalEngine` with a
//! stub classifier, then the actual write through `MemoryToolDispatcher`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use async_trait::async_trait;
use meerkat_core::{AgentToolDispatcher, SessionId, ToolCallView};
use meerkat_tui::approval::classifier::{
    ActionClassifier, ClassifierError, ClassifierInput, ClassifierVerdict, Environment,
};
use meerkat_tui::approval::policy::Tier;
use meerkat_tui::approval::rules::{AutoSettings, RuleSet};
use meerkat_tui::approval::{ApprovalConfig, ApprovalEngine, Verdict};
use meerkat_tui::memory::{
    INDEX_FILE, MEMORY_LIST, MEMORY_SAVE, MemoryToolDispatcher, NoteStore, Scope, prompt_index,
};
use meerkat_tui::paths::ProjectRoot;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;

/// Classifier that always allows and counts its calls.
#[derive(Default)]
struct CountingClassifier {
    calls: AtomicUsize,
}

#[async_trait]
impl ActionClassifier for CountingClassifier {
    async fn classify(
        &self,
        _input: ClassifierInput<'_>,
    ) -> Result<ClassifierVerdict, ClassifierError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ClassifierVerdict::Allow)
    }
}

struct Harness {
    tmp: tempfile::TempDir,
    engine: Arc<ApprovalEngine>,
    classifier: Arc<CountingClassifier>,
    session: SessionId,
    _events: mpsc::UnboundedReceiver<meerkat_tui::approval::ApprovalEvent>,
}

fn harness() -> Harness {
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    let root = ProjectRoot::new(&project).unwrap();
    let classifier = Arc::new(CountingClassifier::default());
    let (events_tx, events) = mpsc::unbounded_channel();
    let engine = ApprovalEngine::new(ApprovalConfig {
        environment: Environment::basic(root.as_path(), root.as_path()),
        root,
        rules: RuleSet::default(),
        settings: AutoSettings {
            enabled: true,
            classifier_model: None,
            trust_project_builds: true,
            ..AutoSettings::default()
        },
        classifier: classifier.clone(),
        broker: None,
        posix_shell: true,
        events: events_tx,
        rules_path: None,
    });
    Harness {
        tmp,
        engine,
        classifier,
        session: SessionId::new(),
        _events: events,
    }
}

impl Harness {
    async fn decide(&self, name: &str, args: &Value) -> Verdict {
        self.engine.decide(&self.session, "tu", name, args).await
    }

    fn calls(&self) -> usize {
        self.classifier.calls.load(Ordering::SeqCst)
    }

    fn stores(&self) -> (NoteStore, NoteStore) {
        (
            NoteStore::open(self.tmp.path().join("config/notes"), Scope::User),
            NoteStore::open(
                self.tmp.path().join("project/.rkat/tui/notes"),
                Scope::Workspace,
            ),
        )
    }
}

fn tier_of(v: &Verdict) -> Tier {
    match v {
        Verdict::Allow { tier, .. } | Verdict::Deny { tier, .. } => *tier,
    }
}

async fn dispatch(d: &MemoryToolDispatcher, name: &str, args: &Value) -> (String, bool) {
    let raw = serde_json::value::to_raw_value(args).unwrap();
    let result = d
        .dispatch(ToolCallView {
            id: "call-1",
            name,
            args: &raw,
        })
        .await
        .unwrap();
    (result.text_content(), result.is_error)
}

#[tokio::test]
async fn workspace_save_is_allowed_without_classifier_and_lands_on_disk() {
    let h = harness();
    let args = json!({"scope": "workspace", "name": "Build", "content": "Run make ci.\nDetails."});
    let verdict = h.decide(MEMORY_SAVE, &args).await;
    assert!(verdict.is_allowed());
    assert_eq!(tier_of(&verdict), Tier::InProjectEdit);
    assert_eq!(h.calls(), 0);

    let list = json!({});
    let verdict = h.decide(MEMORY_LIST, &list).await;
    assert_eq!(tier_of(&verdict), Tier::AlwaysAllow);
    assert_eq!(h.calls(), 0);

    let (user, ws) = h.stores();
    let dispatcher = MemoryToolDispatcher::new(Some(user.clone()), ws.clone());
    let (text, is_error) = dispatch(&dispatcher, MEMORY_SAVE, &args).await;
    assert!(!is_error, "{text}");
    assert_eq!(text, "saved workspace/build (new)");
    assert_eq!(
        std::fs::read_to_string(ws.dir().join("build.md")).unwrap(),
        "Run make ci.\nDetails.\n"
    );
    let index = std::fs::read_to_string(ws.dir().join(INDEX_FILE)).unwrap();
    assert!(index.contains("- build: Run make ci."), "{index}");
    let prompt = prompt_index(Some(&user), &ws).unwrap();
    assert!(prompt.contains("## Workspace notes ("), "{prompt}");
    assert!(prompt.contains("- build: Run make ci."), "{prompt}");
    assert!(!prompt.contains("## User notes"), "{prompt}");
}

#[tokio::test]
async fn user_save_reaches_the_classifier_then_lands_on_disk() {
    let h = harness();
    let args = json!({"scope": "user", "name": "style", "content": "Prefer early returns."});
    let verdict = h.decide(MEMORY_SAVE, &args).await;
    assert_eq!(h.calls(), 1);
    assert!(verdict.is_allowed());
    assert_eq!(tier_of(&verdict), Tier::Classifier);

    let (user, ws) = h.stores();
    let dispatcher = MemoryToolDispatcher::new(Some(user.clone()), ws.clone());
    let (text, is_error) = dispatch(&dispatcher, MEMORY_SAVE, &args).await;
    assert!(!is_error, "{text}");
    assert_eq!(text, "saved user/style (new)");
    assert!(user.dir().join("style.md").is_file());
    assert!(user.dir().join(INDEX_FILE).is_file());
    let prompt = prompt_index(Some(&user), &ws).unwrap();
    assert!(prompt.contains("## User notes ("), "{prompt}");
    assert!(
        prompt.contains("- style: Prefer early returns."),
        "{prompt}"
    );
}
