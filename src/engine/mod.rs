//! Engine types: request, structured decision, downstream action, errors, backend trait.

pub mod candle_backend;
pub mod logit_extractor;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Incoming payload.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub prompt: String,
    pub choices: Vec<String>,
    /// Defaults to the configured temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChoiceProbability {
    pub choice: String,
    pub token_id: u32,
    pub logit: f32,
    pub probability: f32,
}

/// Downstream execution for the selected choice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Action {
    /// Pass to the primary LLM pipeline.
    Allow,
    /// Reject the request.
    Block,
    /// Hand off to the agent / model / service named by the choice.
    Route { target: String },
}

impl Action {
    /// `allow` and `block` (any case) are reserved; every other choice is a route target.
    pub fn from_choice(choice: &str) -> Self {
        match choice.trim().to_ascii_lowercase().as_str() {
            "allow" => Self::Allow,
            "block" => Self::Block,
            _ => Self::Route {
                target: choice.to_owned(),
            },
        }
    }
}

/// Structured decision + confidence.
#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub selected_choice: String,
    /// Probability of the selected choice, renormalized over the allowed choices.
    pub confidence: f32,
    pub action: Action,
    /// Same order as the request's `choices`.
    pub probabilities: Vec<ChoiceProbability>,
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
}

impl KronosError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::Schema(_) => "schema_violation",
            Self::Inference(_) => "inference_error",
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

#[cfg(test)]
mod tests {
    use super::Action;

    #[test]
    fn maps_choices_to_actions() {
        assert_eq!(Action::from_choice("allow"), Action::Allow);
        assert_eq!(Action::from_choice("BLOCK"), Action::Block);
        assert_eq!(
            Action::from_choice("billing"),
            Action::Route {
                target: "billing".into()
            }
        );
    }
}
