//! HTTP ingress for incoming payloads.

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

#[derive(Clone)]
pub struct AppState {
    pub kronos: EmbeddedKronos,
}

pub fn create_router(state: AppState, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/decision", post(decide))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "model": state.kronos.model_kind(),
        "device": state.kronos.device_label(),
    }))
}

async fn decide(
    State(state): State<AppState>,
    Json(req): Json<DecisionRequest>,
) -> Result<Json<Decision>, KronosError> {
    let kronos = state.kronos.clone();
    tokio::task::spawn_blocking(move || kronos.evaluate(&req))
        .await
        .map_err(|e| KronosError::Inference(format!("worker task failed: {e}")))?
        .map(Json)
}

impl IntoResponse for KronosError {
    fn into_response(self) -> Response {
        let status = match &self {
            KronosError::InvalidRequest(_) | KronosError::Schema(_) => StatusCode::BAD_REQUEST,
            KronosError::Inference(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(json!({ "error": self.to_string(), "kind": self.kind() }));
        (status, body).into_response()
    }
}
