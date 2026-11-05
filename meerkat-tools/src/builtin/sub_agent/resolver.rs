//! Default sub-agent model resolution.
//!
//! [`DefaultSubAgentModelResolver`] implements the built-in policy used by
//! `agent_spawn` and `agent_fork` when no resolver is injected: providers are
//! the three built-in families, models are validated against the configured
//! allowlist (the resolved `sub_agents` policy, or the static catalog
//! defaults), and clients come from an [`LlmClientFactory`].

use super::config::SubAgentConfig;
use async_trait::async_trait;
use meerkat_client::{LlmClientAdapter, LlmClientFactory, LlmProvider};
use meerkat_core::config::SubAgentsConfig;
use meerkat_core::{
    Provider, ResolvedSubAgentModel, SubAgentModelError, SubAgentModelRequest,
    SubAgentModelResolver,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Allowlist-validating resolver backed by an [`LlmClientFactory`].
pub struct DefaultSubAgentModelResolver {
    factory: Arc<dyn LlmClientFactory>,
    /// Allowed models per provider (`anthropic` / `openai` / `gemini`).
    allowed: BTreeMap<String, Vec<String>>,
    /// Provider used when the tool call names none.
    default_provider: String,
    /// Model used when the tool call names none (`None` = catalog default
    /// for the resolved provider).
    default_model: Option<String>,
}

impl DefaultSubAgentModelResolver {
    /// Build the resolver from the sub-agent tool config.
    ///
    /// The resolved policy (`config.resolved_policy`) wins when present;
    /// otherwise the config's `default_provider`/`default_model` and the
    /// static [`SubAgentsConfig::default`] allowlist apply.
    pub fn new(factory: Arc<dyn LlmClientFactory>, config: &SubAgentConfig) -> Self {
        match &config.resolved_policy {
            Some(policy) => Self {
                factory,
                allowed: policy.allowed_models.clone(),
                default_provider: policy.default_provider.as_str().to_string(),
                default_model: Some(policy.default_model.clone()),
            },
            None => Self {
                factory,
                allowed: SubAgentsConfig::default().allowed_models,
                default_provider: config.default_provider.clone(),
                default_model: config.default_model.clone(),
            },
        }
    }

    /// Resolve the provider: explicit name, else the configured default.
    pub fn resolve_provider(
        &self,
        provider: Option<&str>,
    ) -> Result<LlmProvider, SubAgentModelError> {
        let name = provider.unwrap_or(&self.default_provider);
        LlmProvider::parse(name)
            .ok_or_else(|| SubAgentModelError::InvalidProvider(name.to_string()))
    }

    /// Resolve the model: explicit name, else the configured default, else
    /// the catalog default for `provider`.
    pub fn resolve_model(&self, provider: LlmProvider, model: Option<&str>) -> String {
        if let Some(model) = model {
            return model.to_string();
        }
        if let Some(default) = &self.default_model {
            return default.clone();
        }
        meerkat_models::catalog::default_model(core_provider(provider).as_str())
            .unwrap_or_default()
            .to_string()
    }

    /// Check that `model` is in the allowlist for `provider`.
    pub fn validate_model(
        &self,
        provider: LlmProvider,
        model: &str,
    ) -> Result<(), SubAgentModelError> {
        let allowed = self
            .allowed
            .get(core_provider(provider).as_str())
            .map(Vec::as_slice)
            .unwrap_or_default();
        if allowed.iter().any(|m| m == model) {
            Ok(())
        } else {
            Err(SubAgentModelError::ModelNotAvailable {
                model: model.to_string(),
                endpoint: provider.to_string(),
                hint: format!("Allowed models: {}", allowed.join(", ")),
            })
        }
    }
}

impl std::fmt::Debug for DefaultSubAgentModelResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultSubAgentModelResolver")
            .field("allowed", &self.allowed)
            .field("default_provider", &self.default_provider)
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SubAgentModelResolver for DefaultSubAgentModelResolver {
    async fn resolve(
        &self,
        req: SubAgentModelRequest<'_>,
    ) -> Result<ResolvedSubAgentModel, SubAgentModelError> {
        let provider = self.resolve_provider(req.provider)?;
        let model = self.resolve_model(provider, req.model);
        self.validate_model(provider, &model)?;
        let client = self
            .factory
            .create_client(provider, None)
            .map_err(|e| SubAgentModelError::Client(e.to_string()))?;
        Ok(ResolvedSubAgentModel {
            client: Arc::new(LlmClientAdapter::new(client, model.clone())),
            provider: core_provider(provider),
            display: format!("{provider}:{model}"),
            model,
        })
    }

    fn describe(&self) -> String {
        let parts: Vec<String> = self
            .allowed
            .iter()
            .map(|(provider, models)| {
                let title = match provider.as_str() {
                    "anthropic" => "Anthropic",
                    "openai" => "OpenAI",
                    "gemini" => "Gemini",
                    other => other,
                };
                format!("{}: {}", title, models.join(", "))
            })
            .collect();
        format!("Allowed models - {}", parts.join("; "))
    }
}

fn core_provider(provider: LlmProvider) -> Provider {
    match provider {
        LlmProvider::Anthropic => Provider::Anthropic,
        LlmProvider::OpenAi => Provider::OpenAI,
        LlmProvider::Gemini => Provider::Gemini,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use meerkat_client::{FactoryError, LlmClient};
    use meerkat_core::config::ResolvedSubAgentConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockClientFactory {
        created: AtomicUsize,
        fail: bool,
    }

    impl MockClientFactory {
        fn new(fail: bool) -> Arc<Self> {
            Arc::new(Self {
                created: AtomicUsize::new(0),
                fail,
            })
        }
    }

    impl LlmClientFactory for MockClientFactory {
        fn create_client(
            &self,
            provider: LlmProvider,
            _api_key: Option<String>,
        ) -> Result<Arc<dyn LlmClient>, FactoryError> {
            self.created.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(FactoryError::MissingApiKey(provider.to_string()));
            }
            Ok(Arc::new(meerkat_client::TestClient::default()))
        }

        fn supported_providers(&self) -> Vec<LlmProvider> {
            vec![
                LlmProvider::Anthropic,
                LlmProvider::OpenAi,
                LlmProvider::Gemini,
            ]
        }
    }

    fn resolver() -> DefaultSubAgentModelResolver {
        DefaultSubAgentModelResolver::new(MockClientFactory::new(false), &SubAgentConfig::default())
    }

    fn request<'a>(provider: Option<&'a str>, model: Option<&'a str>) -> SubAgentModelRequest<'a> {
        SubAgentModelRequest {
            provider,
            model,
            parent_provider: Provider::Anthropic,
            parent_model: "claude-sonnet-4-5",
            depth: 1,
        }
    }

    #[test]
    fn allowed_models_pass_validation() {
        let r = resolver();
        for (provider, model) in [
            (LlmProvider::Anthropic, "claude-opus-4-6"),
            (LlmProvider::Anthropic, "claude-sonnet-4-5"),
            (LlmProvider::Anthropic, "claude-opus-4-5"),
            (LlmProvider::OpenAi, "gpt-5.2"),
            (LlmProvider::OpenAi, "gpt-5.2-pro"),
            (LlmProvider::Gemini, "gemini-3-flash-preview"),
            (LlmProvider::Gemini, "gemini-3-pro-preview"),
        ] {
            assert!(
                r.validate_model(provider, model).is_ok(),
                "{provider}:{model} should be allowed"
            );
        }
    }

    #[test]
    fn disallowed_models_fail_validation_with_legacy_message() {
        let r = resolver();
        let err = r.validate_model(LlmProvider::OpenAi, "gpt-4o").unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with(
            "Model 'gpt-4o' is not in allowlist for provider 'openai'. Allowed models: "
        ));
        assert!(msg.contains("gpt-5.2"));

        let err = r
            .validate_model(LlmProvider::Anthropic, "claude-3-5-sonnet-20241022")
            .unwrap_err();
        assert!(err.to_string().contains("not in allowlist"));

        let err = r
            .validate_model(LlmProvider::Gemini, "gemini-1.5-flash")
            .unwrap_err();
        assert!(matches!(err, SubAgentModelError::ModelNotAvailable { .. }));
    }

    #[test]
    fn provider_defaults_and_aliases() {
        let r = resolver();
        assert_eq!(r.resolve_provider(None).unwrap(), LlmProvider::Anthropic);
        assert_eq!(
            r.resolve_provider(Some("openai")).unwrap(),
            LlmProvider::OpenAi
        );
        assert_eq!(
            r.resolve_provider(Some("claude")).unwrap(),
            LlmProvider::Anthropic
        );
        let err = r.resolve_provider(Some("unknown")).unwrap_err();
        assert_eq!(err.to_string(), "Invalid provider: unknown");
    }

    #[test]
    fn model_defaults_come_from_catalog() {
        let r = resolver();
        assert!(
            r.resolve_model(LlmProvider::Anthropic, None)
                .contains("claude")
        );
        assert!(r.resolve_model(LlmProvider::OpenAi, None).contains("gpt"));
        assert!(
            r.resolve_model(LlmProvider::Gemini, None)
                .contains("gemini")
        );
        assert_eq!(r.resolve_model(LlmProvider::Anthropic, Some("x")), "x");
    }

    #[test]
    fn resolved_policy_overrides_defaults() {
        let mut allowed = BTreeMap::new();
        allowed.insert("openai".to_string(), vec!["gpt-5.4".to_string()]);
        let config = SubAgentConfig::default().with_resolved_policy(ResolvedSubAgentConfig {
            default_provider: Provider::OpenAI,
            default_model: "gpt-5.4".to_string(),
            allowed_models: allowed,
        });
        let r = DefaultSubAgentModelResolver::new(MockClientFactory::new(false), &config);
        assert_eq!(r.resolve_provider(None).unwrap(), LlmProvider::OpenAi);
        assert_eq!(r.resolve_model(LlmProvider::OpenAi, None), "gpt-5.4");
        assert!(r.validate_model(LlmProvider::OpenAi, "gpt-5.4").is_ok());
        assert!(r.validate_model(LlmProvider::OpenAi, "gpt-5.2").is_err());
        assert_eq!(r.describe(), "Allowed models - OpenAI: gpt-5.4");
    }

    #[test]
    fn other_policy_provider_is_invalid() {
        let config = SubAgentConfig::default().with_resolved_policy(ResolvedSubAgentConfig {
            default_provider: Provider::Other,
            default_model: "m".to_string(),
            allowed_models: BTreeMap::new(),
        });
        let r = DefaultSubAgentModelResolver::new(MockClientFactory::new(false), &config);
        assert_eq!(
            r.resolve_provider(None).unwrap_err().to_string(),
            "Invalid provider: other"
        );
    }

    #[test]
    fn describe_lists_default_allowlist() {
        let desc = resolver().describe();
        assert!(desc.starts_with("Allowed models - "));
        assert!(desc.contains("Anthropic: "));
        assert!(desc.contains("claude-opus-4-6"));
        assert!(desc.contains("OpenAI: "));
        assert!(desc.contains("gpt-5.2"));
        assert!(desc.contains("Gemini: "));
        assert!(desc.contains("gemini-3-flash-preview"));
    }

    #[tokio::test]
    async fn resolve_creates_adapted_client() {
        let factory = MockClientFactory::new(false);
        let r = DefaultSubAgentModelResolver::new(
            Arc::clone(&factory) as Arc<dyn LlmClientFactory>,
            &SubAgentConfig::default(),
        );
        let resolved = r
            .resolve(request(Some("openai"), Some("gpt-5.2")))
            .await
            .unwrap();
        assert_eq!(resolved.provider, Provider::OpenAI);
        assert_eq!(resolved.model, "gpt-5.2");
        assert_eq!(resolved.display, "openai:gpt-5.2");
        assert_eq!(factory.created.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn resolve_validates_before_creating_client() {
        let factory = MockClientFactory::new(false);
        let r = DefaultSubAgentModelResolver::new(
            Arc::clone(&factory) as Arc<dyn LlmClientFactory>,
            &SubAgentConfig::default(),
        );
        let err = r
            .resolve(request(Some("openai"), Some("gpt-4o")))
            .await
            .unwrap_err();
        assert!(matches!(err, SubAgentModelError::ModelNotAvailable { .. }));
        assert_eq!(factory.created.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn resolve_maps_factory_errors_to_client() {
        let r = DefaultSubAgentModelResolver::new(
            MockClientFactory::new(true),
            &SubAgentConfig::default(),
        );
        let err = r.resolve(request(None, None)).await.unwrap_err();
        match err {
            SubAgentModelError::Client(msg) => {
                assert_eq!(msg, "Missing API key for provider: anthropic");
            }
            other => unreachable!("expected Client error, got {other:?}"),
        }
    }
}
