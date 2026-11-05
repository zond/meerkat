//! `/model` picker and endpoint-sourced context windows.
//!
//! The provider's model list is fetched once in the background at start-up
//! and cached for the process; `/model` opens the picker from the cache
//! (fetching when it is empty), `/model refresh` refetches. The cached list
//! is also the source of the context-gauge window: the catalog-based
//! estimate in [`window_status`] is used only when the endpoint reported no
//! window for the current model or the fetch failed.

use std::time::Duration;

use tokio::task::JoinHandle;

use super::{Integrator, window_status};
use crate::bootstrap::credentials::ProviderCredentials;
use crate::models::{ListModelsError, ModelEntry, fetch_warning, list_models, with_current};
use crate::ui::{Msg, NoticeLevel, StatusUpdate};

/// Timeout for each list-models request.
pub(super) const LIST_MODELS_TIMEOUT: Duration = Duration::from_secs(10);

/// A list-models request running in the background.
pub(super) type ModelFetch = JoinHandle<Result<Vec<ModelEntry>, ListModelsError>>;

/// Start fetching the model list for `creds`.
pub(super) fn spawn_model_fetch(creds: &ProviderCredentials) -> ModelFetch {
    let creds = creds.clone();
    tokio::spawn(async move { list_models(&creds, LIST_MODELS_TIMEOUT).await })
}

impl Integrator {
    /// Answer [`crate::ui::Effect::RequestModelPicker`].
    pub(super) fn request_model_picker(&mut self, refresh: bool) {
        if refresh {
            self.models = None;
        }
        if let Some(models) = &self.models {
            let items = with_current(&self.current_model, models.clone());
            self.send(Msg::ShowModelPicker(items, None));
            return;
        }
        let Some(creds) = &self.creds else {
            let items = with_current(&self.current_model, Vec::new());
            let warning = "model listing is unavailable (no provider credentials)".to_string();
            self.send(Msg::ShowModelPicker(items, Some(warning)));
            return;
        };
        self.picker_waiting = true;
        if self.model_fetch.is_none() {
            let fetch = spawn_model_fetch(creds);
            self.model_fetch = Some(fetch);
            self.notice(NoticeLevel::Info, "fetching the model list…");
        }
    }

    /// A background list-models request finished: cache the list, update the
    /// context window and open the picker if `/model` is waiting for it.
    pub(super) fn model_fetch_finished(
        &mut self,
        joined: Result<Result<Vec<ModelEntry>, ListModelsError>, tokio::task::JoinError>,
    ) {
        self.model_fetch = None;
        let result = joined.unwrap_or_else(|err| {
            Err(ListModelsError::Decode(format!(
                "list-models task failed: {err}"
            )))
        });
        let warning = match result {
            Ok(models) => {
                self.models = Some(models);
                self.push_context_window();
                None
            }
            Err(err) => {
                tracing::warn!(error = %err, "could not list models");
                Some(fetch_warning(&err))
            }
        };
        if std::mem::take(&mut self.picker_waiting) {
            let live = self.models.clone().unwrap_or_default();
            self.send(Msg::ShowModelPicker(
                with_current(&self.current_model, live),
                warning,
            ));
        }
    }

    /// Whether `model` is listed by the provider endpoint.
    pub(super) fn model_is_listed(&self, model: &str) -> bool {
        self.models
            .as_ref()
            .is_some_and(|models| models.iter().any(|entry| entry.id == model))
    }

    /// Push the current model's context window: the endpoint's number when
    /// the cached list has one, else the catalog/family estimate.
    pub(super) fn push_context_window(&self) {
        let provider = self.session.provider();
        let listed = self
            .creds
            .as_ref()
            .filter(|creds| creds.provider == provider)
            .and(self.models.as_ref())
            .and_then(|models| models.iter().find(|entry| entry.id == self.current_model))
            .and_then(|entry| entry.context_window);
        let update = match listed {
            Some(window) => StatusUpdate::ContextWindow(Some(window)),
            None => window_status(provider, &self.current_model),
        };
        self.send(Msg::Status(update));
    }
}
