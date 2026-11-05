//! Cost readout on the integrator: the price table (overrides, endpoint
//! prices, the community list refreshed in the background) and the per-session
//! cost totals persisted in the endpoint sidecar.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use meerkat_core::Session;
use tokio::task::JoinHandle;

use super::Integrator;
use super::models::EndpointPrices;
use crate::endpoints::{load_cost, save_cost};
use crate::pricing::{CommunityPrices, FETCH_TIMEOUT, PricingError, refresh_community};
use crate::ui::{Msg, SessionCost, SessionCostSnapshot, StatusUpdate};

/// A community price-list refresh running in the background.
pub(super) type PriceFetch = JoinHandle<Result<CommunityPrices, PricingError>>;

/// Start refreshing the community list from `url` (cached at `cache`).
pub(super) fn spawn_price_fetch(url: String, cache: Option<PathBuf>) -> PriceFetch {
    tokio::spawn(async move { refresh_community(&url, cache.as_deref(), FETCH_TIMEOUT).await })
}

/// The cost totals of `session`: the sidecar's, else derived from the
/// session's cumulative usage and attributed to `model` (approximate).
pub(super) fn session_cost(realm_root: &Path, session: &Session, model: &str) -> SessionCost {
    load_cost(realm_root, &session.id().to_string())
        .unwrap_or_else(|| SessionCost::derived(model, &session.total_usage()))
}

impl Integrator {
    /// Send the current price table to the UI.
    pub(super) fn push_prices(&self) {
        self.send(Msg::Status(StatusUpdate::Prices(Arc::clone(&self.prices))));
    }

    /// Replace the endpoint (LiteLLM `/model/info`) prices and tell the UI.
    pub(super) fn set_endpoint_prices(&mut self, prices: EndpointPrices) {
        Arc::make_mut(&mut self.prices).set_endpoint_prices(prices);
        self.push_prices();
    }

    /// The community refresh finished: use the new list (the cache was
    /// written by the task) or keep the old one.
    pub(super) fn price_fetch_finished(
        &mut self,
        joined: Result<Result<CommunityPrices, PricingError>, tokio::task::JoinError>,
    ) {
        self.price_fetch = None;
        match joined {
            Ok(Ok(community)) => {
                tracing::debug!(models = community.len(), "community price list refreshed");
                Arc::make_mut(&mut self.prices).set_community(community);
                self.push_prices();
            }
            Ok(Err(err)) => {
                tracing::warn!(error = %err, "could not fetch the community price list");
            }
            Err(err) => tracing::warn!(error = %err, "price-list task failed"),
        }
    }

    /// Answer [`crate::ui::Effect::CostSnapshot`]: store the totals next to
    /// the live session.
    pub(super) fn save_cost_snapshot(&self, snapshot: &SessionCostSnapshot) {
        if let Err(err) = save_cost(
            &self.runtime.scope.realm_root(),
            &self.session.id().to_string(),
            &self.current_choice(),
            snapshot,
        ) {
            tracing::warn!(error = %err, "could not record the session's cost");
        }
    }

    /// Send the cost totals of the (just resumed) `session`.
    pub(super) fn push_session_cost(&self, session: &Session) {
        let cost = session_cost(
            &self.runtime.scope.realm_root(),
            session,
            &self.current_model,
        );
        self.send(Msg::Status(StatusUpdate::CostTotals(cost)));
    }
}
