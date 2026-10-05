//! Engine interface definitions: request/response types, errors, backend trait.

pub mod candle_backend;
pub mod logit_extractor;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub prompt: String,
    pub choices: Vec<String>,
    /// Defaults to the server's configured temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// If the winning probability is below this, `abstained` is true.
    #[serde(default)]
    pub min_confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChoiceProbability {
    pub choice: String,
    pub token_id: u32,
    pub logit: f32,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub selected_choice: String,
    /// Probability of the selected choice, renormalized over the allowed choices.
    pub confidence: f32,
    /// Share of the model's full next-token distribution (T=1) that landed on the
    /// allowed choices. Low values mean the model "wanted" to say something else.
    pub choice_mass: f32,
    /// True when `min_confidence` was set and not reached. Caller should escalate.
    pub abstained: bool,
    /// In the same order as the request's `choices`.
    pub probabilities: Vec<ChoiceProbability>,
    pub prompt_tokens: usize,
    pub latency_us: u64,
}

#[derive(Debug, Error)]
pub enum KronosError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("schema violation: {0}")]
    Schema(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("server busy, retry later")]
    Busy,
}

impl KronosError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::Schema(_) => "schema_violation",
            Self::Inference(_) => "inference_error",
            Self::Busy => "busy",
        }
    }
}

pub type KronosResult<T> = Result<T, KronosError>;

/// Anything that can turn a token sequence into next-token logits.
pub trait LogitBackend: Send {
    /// Logits for the position after the last input token (length = model vocab).
    fn next_token_logits(&mut self, input_ids: &[u32]) -> anyhow::Result<Vec<f32>>;
    fn vocab_size(&self) -> usize;
}
