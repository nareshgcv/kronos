use crate::engine::candle_backend::CandleEngine;
use crate::engine::logit_extractor::LogitExtractor;
use crate::tokenizer::schema_mapper::SchemaMapper;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize)]
pub struct ApiRequest {
    pub prompt: String,
    pub choices: Vec<String>,
    #[serde(default = "default_temp")]
    pub temperature: f32,
}

fn default_temp() -> f32 {
    1.0
}

#[derive(Serialize)]
pub struct ApiResponse {
    pub selected_choice: String,
    pub confidence: f32,
    pub probabilities: HashMap<String, f32>,
    pub latency_ms: f64,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub struct AppState {
    pub engine: CandleEngine,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handle_health))
        .route("/v1/decision", post(handle_decision))
        .with_state(state)
}

async fn handle_health() -> impl IntoResponse {
    StatusCode::OK
}

async fn handle_decision(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ApiRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();

    // Offload CPU/GPU intensive execution to a blocking thread pool
    tokio::task::spawn_blocking(move || {
        // 1. Map choices to single token IDs
        let resolved_choices = SchemaMapper::resolve_choices(
            state.engine.tokenizer(),
            &payload.choices,
        )
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Schema Mapper Error: {e}"),
                }),
            )
        })?;

        // 2. Run single forward prefill pass
        let (logits, _) = state
            .engine
            .forward_prefill_logits(&payload.prompt)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Inference Engine Error: {e}"),
                    }),
                )
            })?;

        // 3. Extract Softmax probabilities
        let output = LogitExtractor::compute_probabilities(
            &logits,
            &resolved_choices,
            payload.temperature,
        );

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(Json(ApiResponse {
            selected_choice: output.selected_choice,
            confidence: output.confidence,
            probabilities: output.probabilities,
            latency_ms,
        }))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Task execution failed: {e}"),
            }),
        )
    })?
}
