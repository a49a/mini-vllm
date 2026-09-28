//! Router construction.

use axum::routing::{get, post};
use axum::Router;

use crate::api;
use crate::SharedState;

/// Body size guard (design doc §61): reject oversized request payloads
/// before they cause unbounded memory growth.
const MAX_BODY_BYTES: usize = 1024 * 1024;

pub fn router(state: SharedState) -> Router {
    router_with_preprocessing(state, Default::default())
        .expect("valid default preprocessing config")
}

pub fn router_with_preprocessing(
    state: SharedState,
    config: crate::preprocessing::PreprocessConfig,
) -> Result<Router, &'static str> {
    let processor = crate::preprocessing::Preprocessor::new(config)?;
    Ok(Router::new()
        .route("/health", get(api::health))
        .route("/v1/models", get(api::models))
        .route("/metrics", get(api::metrics))
        .route("/v1/completions", post(api::completions))
        .route("/v1/chat/completions", post(api::chat_completions))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(axum::Extension(processor))
        .with_state(state))
}
