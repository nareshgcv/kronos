use crate::engine::candle_backend::CandleEngine;
use crate::engine::logit_extractor::LogitExtractor;
use crate::tokenizer::schema_mapper::SchemaMapper;
use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
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
    pub probabilities: std::collections::HashMap<String, f32>,
    pub latency_ms: f64,
}

pub struct AppState {
    pub engine: CandleEngine,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/decision", post(handle_decision))
        .with_state(state)
}

async fn handle_decision(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ApiRequest>,
) -> Json<ApiResponse> {
    let start = Instant::now();

    // 1. Map choices to single token IDs
    let resolved_choices = SchemaMapper::resolve_choices(state.engine.tokenizer(), &payload.choices)
        .expect("Choice schema resolution failed");

    // 2. Run single forward prefill pass
    let (logits, _) = state
        .engine
        .forward_prefill_logits(&payload.prompt)
        .expect("Inference execution failed");

    // 3. Extract Softmax probabilities
    let output = LogitExtractor::compute_probabilities(&logits, &resolved_choices, payload.temperature);

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Json(ApiResponse {
        selected_choice: output.selected_choice,
        confidence: output.confidence,
        probabilities: output.probabilities,
        latency_ms,
    })
}
