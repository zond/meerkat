//! Provider, model and credential resolution.
//!
//! The LLM client is built exactly once per process from these credentials and
//! shared between the session and the approval classifier, so the two can never
//! disagree about which account or endpoint they talk to.

use std::sync::Arc;

use meerkat::{AgentFactory, LlmClient};
use meerkat_client::ProviderResolver;
use meerkat_core::{Config, Provider, ProviderConfig};

use crate::cli::Args;
use crate::error::TuiError;

/// Resolved credentials for one provider.
#[derive(Clone)]
pub struct ProviderCredentials {
    /// The provider these credentials belong to.
    pub provider: Provider,
    /// API key (never logged).
    pub api_key: String,
    /// Optional base URL override.
    pub base_url: Option<String>,
}

impl std::fmt::Debug for ProviderCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderCredentials")
            .field("provider", &self.provider)
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Resolve the session model and provider.
///
/// Model: `--model`, else `config.agent.model` when it belongs to the selected
/// provider (or no provider was requested), else the provider's catalog
/// default. Provider: `--provider`, else inferred from the model name.
///
/// # Errors
///
/// Returns [`TuiError::Provider`] for an unknown `--provider` name or when the
/// provider cannot be inferred from the model name.
pub fn resolve_model_and_provider(
    args: &Args,
    config: &Config,
) -> Result<(String, Provider), TuiError> {
    let explicit_provider = args
        .provider
        .as_deref()
        .map(|name| {
            Provider::parse_strict(name).ok_or_else(|| {
                TuiError::Provider(format!(
                    "unknown provider `{name}` (expected anthropic, openai or gemini)"
                ))
            })
        })
        .transpose()?;

    let model = match (&args.model, explicit_provider) {
        (Some(model), _) => model.clone(),
        (None, None) => config.agent.model.clone(),
        (None, Some(provider)) => {
            let configured = &config.agent.model;
            if !configured.is_empty() && Provider::infer_from_model(configured) == Some(provider) {
                configured.clone()
            } else {
                meerkat_models::default_model(provider.as_str())
                    .map(str::to_string)
                    .ok_or_else(|| {
                        TuiError::Provider(format!(
                            "no default model for provider `{}`",
                            provider.as_str()
                        ))
                    })?
            }
        }
    };

    let provider = match explicit_provider {
        Some(provider) => provider,
        None => match ProviderResolver::infer_from_model(&model) {
            Provider::Other => {
                return Err(TuiError::Provider(format!(
                    "cannot infer provider from model `{model}`; pass --provider"
                )));
            }
            provider => provider,
        },
    };
    Ok((model, provider))
}

/// Resolve API key and base URL for `provider`.
///
/// Mirrors the factory's private `resolve_provider_credentials`
/// (`meerkat/src/factory.rs`): `providers.api_keys`/`providers.base_urls`
/// keyed by provider name, overridden by a matching legacy `[provider]` block,
/// with environment variables as the final fallback for the key.
///
/// # Errors
///
/// Returns [`TuiError::MissingApiKey`] when no key is configured anywhere.
pub fn resolve_provider_credentials(
    config: &Config,
    provider: Provider,
) -> Result<ProviderCredentials, TuiError> {
    let key = meerkat::provider_key(provider);
    let mut base_url = config
        .providers
        .base_urls
        .as_ref()
        .and_then(|map| map.get(key).cloned());
    let mut api_key = config
        .providers
        .api_keys
        .as_ref()
        .and_then(|map| map.get(key).cloned());

    match (&config.provider, provider) {
        (
            ProviderConfig::Anthropic {
                api_key: cfg_key,
                base_url: cfg_url,
            },
            Provider::Anthropic,
        )
        | (
            ProviderConfig::OpenAI {
                api_key: cfg_key,
                base_url: cfg_url,
            },
            Provider::OpenAI,
        ) => {
            if cfg_key.is_some() {
                api_key.clone_from(cfg_key);
            }
            if cfg_url.is_some() {
                base_url.clone_from(cfg_url);
            }
        }
        (ProviderConfig::Gemini { api_key: cfg_key }, Provider::Gemini) => {
            if cfg_key.is_some() {
                api_key.clone_from(cfg_key);
            }
        }
        _ => {}
    }

    if api_key.is_none() {
        api_key = ProviderResolver::api_key_for(provider);
    }

    let api_key = api_key.ok_or(TuiError::MissingApiKey {
        provider: provider.as_str(),
    })?;
    Ok(ProviderCredentials {
        provider,
        api_key,
        base_url,
    })
}

/// Build the process-wide LLM client from resolved credentials.
///
/// # Errors
///
/// Returns [`TuiError::LlmClient`] when the provider is unsupported or the
/// client cannot be constructed.
pub async fn build_llm_client(
    factory: &AgentFactory,
    creds: ProviderCredentials,
) -> Result<Arc<dyn LlmClient>, TuiError> {
    Ok(factory
        .build_llm_client(creds.provider, Some(creds.api_key), creds.base_url)
        .await?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec!["rkat-tui"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).unwrap()
    }

    #[test]
    fn explicit_model_infers_provider() {
        let (model, provider) =
            resolve_model_and_provider(&args(&["--model", "gpt-5.2"]), &Config::default()).unwrap();
        assert_eq!(model, "gpt-5.2");
        assert_eq!(provider, Provider::OpenAI);
    }

    #[test]
    fn provider_without_model_uses_catalog_default_when_config_mismatches() {
        let mut config = Config::default();
        config.agent.model = "claude-sonnet-4-6".to_string();
        let (model, provider) =
            resolve_model_and_provider(&args(&["--provider", "gemini"]), &config).unwrap();
        assert_eq!(provider, Provider::Gemini);
        assert_eq!(
            Some(model.as_str()),
            meerkat_models::default_model("gemini")
        );
    }

    #[test]
    fn config_model_used_by_default() {
        let mut config = Config::default();
        config.agent.model = "claude-opus-4-6".to_string();
        let (model, provider) = resolve_model_and_provider(&args(&[]), &config).unwrap();
        assert_eq!(model, "claude-opus-4-6");
        assert_eq!(provider, Provider::Anthropic);
    }

    #[test]
    fn unknown_provider_is_rejected() {
        let err = resolve_model_and_provider(&args(&["--provider", "nope"]), &Config::default())
            .unwrap_err();
        assert!(matches!(err, TuiError::Provider(_)));
    }

    #[test]
    fn config_api_key_map_wins_over_env() {
        let mut config = Config::default();
        config.providers.api_keys = Some(
            [("openai".to_string(), "sk-from-config".to_string())]
                .into_iter()
                .collect(),
        );
        config.providers.base_urls = Some(
            [("openai".to_string(), "http://localhost:1".to_string())]
                .into_iter()
                .collect(),
        );
        let creds = resolve_provider_credentials(&config, Provider::OpenAI).unwrap();
        assert_eq!(creds.api_key, "sk-from-config");
        assert_eq!(creds.base_url.as_deref(), Some("http://localhost:1"));
        assert!(!format!("{creds:?}").contains("sk-from-config"));
    }
}
