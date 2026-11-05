//! `/model` across endpoints: the picker, switching, and endpoint-sourced
//! context windows.
//!
//! Every endpoint's model list is fetched in parallel in the background at
//! start-up (or synchronously before the first session when a bare
//! `--model` id needs the lists to pick its endpoint) and cached for the
//! process; `/model` opens the picker from the cache (fetching when it is
//! empty), `/model refresh` refetches. The cache also resolves bare ids (see
//! [`parse_model_ref`]) and is the source of the context-gauge window: the
//! catalog-based estimate in [`window_status`] is used only when the
//! endpoint reported no window for the current model or the fetch failed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use meerkat::{LlmClient, SessionService};
use meerkat_core::Provider;
use tokio::task::JoinHandle;

use super::{Integrator, window_status};
use crate::endpoints::{
    BUILTIN_NAMES, Dialect, Endpoint, EndpointRef, EndpointRegistry, ModelIndex, ModelRefError,
    SecretString, needs_lookup, parse_model_ref, save_choice,
};
use crate::models::{
    ListModelsError, ModelEntry, ModelGroup, fetch_warning, list_all_endpoints, with_current,
};
use crate::ui::{Msg, NoticeLevel, StatusUpdate};

/// Timeout for each list-models request.
pub(super) const LIST_MODELS_TIMEOUT: Duration = Duration::from_secs(10);

/// Model lists by endpoint name: the list, or a one-line warning.
pub(crate) type ModelCache = HashMap<String, Result<Vec<ModelEntry>, String>>;

/// A fetch of every endpoint's list running in the background.
pub(super) type ModelFetch = JoinHandle<FetchResults>;

/// Per-endpoint results of one fetch, in endpoint order.
pub(super) type FetchResults = Vec<(String, Result<Vec<ModelEntry>, ListModelsError>)>;

/// Start fetching every endpoint's model list.
pub(super) fn spawn_model_fetch(endpoints: &EndpointRegistry) -> ModelFetch {
    let endpoints: Vec<Endpoint> = endpoints.iter().cloned().collect();
    tokio::spawn(async move { list_all_endpoints(&endpoints, LIST_MODELS_TIMEOUT).await })
}

/// Fetch every endpoint's list now (start-up lookups).
pub(super) async fn fetch_cache(endpoints: &EndpointRegistry) -> ModelCache {
    let endpoints: Vec<Endpoint> = endpoints.iter().cloned().collect();
    cache_from(list_all_endpoints(&endpoints, LIST_MODELS_TIMEOUT).await)
}

/// Turn fetch results into a cache, logging failures.
pub(super) fn cache_from(
    results: Vec<(String, Result<Vec<ModelEntry>, ListModelsError>)>,
) -> ModelCache {
    results
        .into_iter()
        .map(|(name, result)| {
            let result = result.map_err(|err| {
                tracing::warn!(endpoint = %name, error = %err, "could not list models");
                fetch_warning(&err)
            });
            (name, result)
        })
        .collect()
}

/// Which endpoints list which ids, from the successful entries of `cache`.
pub(super) fn index_of(cache: Option<&ModelCache>) -> ModelIndex {
    let Some(cache) = cache else {
        return ModelIndex::default();
    };
    ModelIndex::from_lists(
        cache
            .iter()
            .filter_map(|(name, list)| list.as_ref().ok().map(|l| (name.as_str(), l.as_slice()))),
    )
}

impl Integrator {
    /// The configured endpoints plus, when it is not configured (tests, or a
    /// start-up client supplied by the caller), the current endpoint.
    fn known_endpoints(&self) -> Vec<Endpoint> {
        let mut known: Vec<Endpoint> = self.endpoints.iter().cloned().collect();
        if self.endpoints.get(&self.current_endpoint).is_none()
            && let Some(dialect) = Dialect::from_provider(self.current_provider)
        {
            known.push(Endpoint {
                name: self.current_endpoint.clone(),
                dialect,
                base_url: None,
                api_key: SecretString::default(),
                builtin: BUILTIN_NAMES.contains(&self.current_endpoint.as_str()),
                priority: 0,
            });
        }
        known
    }

    /// [`Self::known_endpoints`], for callers outside this module.
    pub(super) fn known_endpoint_list(&self) -> Vec<Endpoint> {
        self.known_endpoints()
    }

    /// The current endpoint and model.
    pub(super) fn current_choice(&self) -> EndpointRef {
        EndpointRef::new(self.current_endpoint.clone(), self.current_model.clone())
    }

    /// Whether `name` is an endpoint sessions can run on.
    pub(super) fn endpoint_exists(&self, name: &str) -> bool {
        self.endpoints.get(name).is_some() || self.clients.contains_key(name)
    }

    /// The dialect (as a provider) of endpoint `name`.
    pub(super) fn provider_of(&self, name: &str) -> Provider {
        match self.endpoints.get(name) {
            Some(endpoint) => endpoint.dialect.into(),
            None if name == self.current_endpoint => self.current_provider,
            None => self.runtime.provider,
        }
    }

    /// Resolve `/model <reference>` against the cached lists.
    fn resolve_ref(&self, reference: &str) -> Result<EndpointRef, ModelRefError> {
        parse_model_ref(
            reference,
            &self.known_endpoints(),
            &index_of(self.models.as_ref()),
            Some(&self.current_endpoint),
        )
    }

    /// The client of endpoint `name`, built and cached on first use.
    pub(super) async fn client_for(&mut self, name: &str) -> Result<Arc<dyn LlmClient>, String> {
        if let Some(client) = self.clients.get(name) {
            return Ok(Arc::clone(client));
        }
        let endpoint = self
            .endpoints
            .get(name)
            .ok_or_else(|| format!("unknown endpoint `{name}`"))?;
        let client = endpoint
            .build_client(&self.runtime.factory)
            .await
            .map_err(|err| err.to_string())?;
        self.clients.insert(name.to_string(), Arc::clone(&client));
        Ok(client)
    }

    /// Record the current choice next to the session, for `-c`/`/resume`.
    pub(super) fn save_current_choice(&self) {
        let realm = self.runtime.scope.realm_root();
        if let Err(err) = save_choice(
            &realm,
            &self.session.id().to_string(),
            &self.current_choice(),
        ) {
            tracing::warn!(error = %err, "could not record the session's endpoint");
        }
    }

    /// Start the background fetch unless one is running.
    fn ensure_fetch(&mut self) {
        if self.model_fetch.is_none() {
            self.model_fetch = Some(spawn_model_fetch(&self.endpoints));
            self.notice(NoticeLevel::Info, "fetching the model lists…");
        }
    }

    /// Answer [`crate::ui::Effect::RequestModelPicker`].
    pub(super) fn request_model_picker(&mut self, refresh: bool) {
        if refresh {
            self.models = None;
        }
        if self.models.is_some() || self.endpoints.is_empty() {
            self.send(Msg::ShowModelPicker(self.picker_groups()));
            return;
        }
        self.picker_waiting = true;
        self.ensure_fetch();
    }

    /// The picker: one group per endpoint, the current endpoint first with
    /// the current model first.
    fn picker_groups(&self) -> Vec<ModelGroup> {
        let known = self.known_endpoints();
        let current = known.iter().filter(|e| e.name == self.current_endpoint);
        let others = known.iter().filter(|e| e.name != self.current_endpoint);
        current
            .chain(others)
            .map(|endpoint| {
                let (models, warning) = match self.models.as_ref().map(|m| m.get(&endpoint.name)) {
                    Some(Some(Ok(list))) => (list.clone(), None),
                    Some(Some(Err(warning))) => (Vec::new(), Some(warning.clone())),
                    _ if self.endpoints.get(&endpoint.name).is_none() => (
                        Vec::new(),
                        Some("model listing is unavailable for this endpoint".to_string()),
                    ),
                    _ => (Vec::new(), None),
                };
                let models = if endpoint.name == self.current_endpoint {
                    with_current(&self.current_model, models)
                } else {
                    with_current("", models)
                };
                ModelGroup {
                    endpoint: endpoint.name.clone(),
                    dialect: endpoint.dialect,
                    base_url: endpoint.base_url.clone(),
                    models,
                    warning,
                }
            })
            .collect()
    }

    /// The background fetch finished: cache the lists, update the context
    /// window, open the picker and finish a `/model <id>` waiting for them.
    pub(super) async fn model_fetch_finished(
        &mut self,
        joined: Result<FetchResults, tokio::task::JoinError>,
    ) {
        self.model_fetch = None;
        let cache = match joined {
            Ok(results) => cache_from(results),
            Err(err) => {
                tracing::warn!(error = %err, "list-models task failed");
                self.endpoints
                    .iter()
                    .map(|e| {
                        (
                            e.name.clone(),
                            Err(format!("list-models task failed: {err}")),
                        )
                    })
                    .collect()
            }
        };
        self.models = Some(cache);
        self.push_context_window();
        if std::mem::take(&mut self.picker_waiting) {
            self.send(Msg::ShowModelPicker(self.picker_groups()));
        }
        if let Some(reference) = self.pending_set.take() {
            if self.turn.is_some() || !self.inflight.is_empty() {
                self.notice(
                    NoticeLevel::Warn,
                    format!("/model {reference} dropped: a turn started meanwhile"),
                );
            } else {
                self.set_model(reference).await;
            }
        }
    }

    /// `/model <reference>` or a picker selection: switch the live session's
    /// client between turns.
    ///
    /// A bare id is looked up in the cached lists (fetched first when the
    /// cache is empty). Sub-agents are not affected: Meerkat builds their
    /// clients from its own defaults (see the crate README).
    pub(super) async fn set_model(&mut self, reference: String) {
        if self.models.is_none()
            && !self.endpoints.is_empty()
            && needs_lookup(&reference, &self.known_endpoints())
        {
            self.pending_set = Some(reference);
            self.ensure_fetch();
            return;
        }
        let target = match self.resolve_ref(&reference) {
            Ok(target) => target,
            Err(err) => {
                self.notice(NoticeLevel::Error, format!("model not changed: {err}"));
                return;
            }
        };
        let client = match self.client_for(&target.endpoint).await {
            Ok(client) => client,
            Err(err) => {
                self.notice(NoticeLevel::Error, format!("model not changed: {err}"));
                return;
            }
        };
        let adapter = self
            .runtime
            .factory
            .build_llm_adapter(client, target.model.clone())
            .await;
        match self
            .runtime
            .service
            .set_session_client(self.session.id(), Arc::new(adapter))
            .await
        {
            Ok(()) => {
                self.current_provider = self.provider_of(&target.endpoint);
                self.current_endpoint = target.endpoint;
                self.current_model = target.model;
                self.send(Msg::Status(StatusUpdate::Model {
                    provider: self.current_endpoint.clone(),
                    model: self.current_model.clone(),
                }));
                self.push_context_window();
                self.save_current_choice();
            }
            Err(err) => self.notice(NoticeLevel::Error, format!("model not changed: {err}")),
        }
    }

    /// Push the current model's context window: the endpoint's number when
    /// its cached list has one, else the catalog/family estimate.
    pub(super) fn push_context_window(&self) {
        let listed = self
            .models
            .as_ref()
            .and_then(|cache| cache.get(&self.current_endpoint))
            .and_then(|list| list.as_ref().ok())
            .and_then(|models| models.iter().find(|entry| entry.id == self.current_model))
            .and_then(|entry| entry.context_window);
        let update = match listed {
            Some(window) => StatusUpdate::ContextWindow(Some(window)),
            None => window_status(self.current_provider, &self.current_model),
        };
        self.send(Msg::Status(update));
    }
}
