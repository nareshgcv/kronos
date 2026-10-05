//! Axum HTTP/REST endpoints.

use crate::engine::{Decision, DecisionRequest, KronosError};
use crate::EmbeddedKronos;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct AppState {
    pub kronos: EmbeddedKronos,
    /// Shared with the IPC server: caps total in-flight + queued work.
    pub queue: Arc<Semaphore>,
}

pub fn create_router(state: AppState, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/decision", post(decide))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "model": state.kronos.model_kind(),
        "device": state.kronos.device_label(),
        "prompt_format": state.kronos.prompt_format(),
    }))
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.kronos.stats())
}

async fn decide(
    State(state): State<AppState>,
    Json(req): Json<DecisionRequest>,
) -> Result<Json<Decision>, KronosError> {
    let permit = Arc::clone(&state.queue)
        .try_acquire_owned()
        .map_err(|_| KronosError::Busy)?;
    let kronos = state.kronos.clone();

    // The permit moves into the blocking task so it stays held even if the
    // client disconnects and this future is dropped.
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        kronos.evaluate(&req)
    })
    .await
    .map_err(|e| KronosError::Inference(format!("worker task failed: {e}")))?
    .map(Json)
}

impl IntoResponse for KronosError {
    fn into_response(self) -> Response {
        let status = match &self {
            KronosError::InvalidRequest(_) | KronosError::Schema(_) => StatusCode::BAD_REQUEST,
            KronosError::Busy => StatusCode::SERVICE_UNAVAILABLE,
            KronosError::Inference(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(json!({ "error": self.to_string(), "kind": self.kind() }));
        (status, body).into_response()
    }
}
