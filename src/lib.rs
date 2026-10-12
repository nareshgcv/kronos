//! Kronos: a zero-generation decision runtime.
//!
//! Incoming payload -> Schema Mapper -> Candle single prefill pass ->
//! candidate logit extraction -> temperature-scaled softmax ->
//! structured decision + confidence -> downstream action (route / allow / block).

pub mod config;
pub mod engine;
pub mod server;
pub mod tokenizer;
pub mod utils;

pub use config::{select_best_device, KronosConfig, PromptFormat};
pub use engine::candle_backend::{CandleEngine, ModelKind};
pub use engine::logit_extractor::{Distribution, LogitExtractor};
pub use engine::{
    Action, ChoiceProbability, Decision, DecisionRequest, KronosError, KronosResult, LogitBackend,
};
pub use tokenizer::{ResolvedChoice, SchemaMapper, SpacePolicy, TokenizerManager};
pub use utils::metrics::LatencyTimer;

use anyhow::Result;
use candle_core::Device;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

/// Embedded, in-process Kronos. Cheap to clone and safe to share across threads.
/// `evaluate` blocks, so from async code call it inside `spawn_blocking`.
#[derive(Clone)]
pub struct EmbeddedKronos {
    inner: Arc<Inner>,
}

struct Inner {
    engine: Mutex<CandleEngine>,
    tokenizer: TokenizerManager,
    model_kind: ModelKind,
    device_label: String,
    default_temperature: f32,
    max_choices: usize,
    max_prompt_tokens: usize,
}

impl EmbeddedKronos {
    /// Loads a model directory with default settings and the best available device.
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config = KronosConfig {
            model_dir: model_dir.as_ref().to_path_buf(),
            ..KronosConfig::default()
        };
        Self::from_config(&config)
    }

    pub fn from_config(config: &KronosConfig) -> Result<Self> {
        Self::with_device(config, select_best_device())
    }

    pub fn with_device(config: &KronosConfig, device: Device) -> Result<Self> {
        config.validate()?;
        let device_label = format!("{device:?}");

        let engine = CandleEngine::load(&config.model_dir, device)?;
        let model_kind = engine.kind();
        let format = config
            .prompt_format
            .unwrap_or_else(|| model_kind.default_prompt_format());

        let tokenizer = TokenizerManager::load(
            &config.model_dir,
            format,
            config.system_prompt.clone(),
            config.append_choices_to_prompt,
        )?;

        Ok(Self {
            inner: Arc::new(Inner {
                engine: Mutex::new(engine),
                tokenizer,
                model_kind,
                device_label,
                default_temperature: config.default_temperature,
                max_choices: config.max_choices,
                max_prompt_tokens: config.max_prompt_tokens,
            }),
        })
    }

    /// Convenience wrapper with the default temperature.
    pub fn decide(&self, prompt: &str, choices: &[&str]) -> KronosResult<Decision> {
        self.evaluate(&DecisionRequest {
            prompt: prompt.to_owned(),
            choices: choices.iter().map(|c| (*c).to_owned()).collect(),
            temperature: None,
        })
    }

    /// Runs the full pipeline for one request.
    pub fn evaluate(&self, req: &DecisionRequest) -> KronosResult<Decision> {
        let timer = LatencyTimer::start();
        let inner = &self.inner;

        // Incoming payload / prompt
        if req.prompt.trim().is_empty() {
            return Err(KronosError::InvalidRequest("prompt must not be empty".into()));
        }
        let n = req.choices.len();
        if n < 2 || n > inner.max_choices {
            return Err(KronosError::InvalidRequest(format!(
                "expected between 2 and {} choices, got {n}",
                inner.max_choices
            )));
        }
        let temperature = req.temperature.unwrap_or(inner.default_temperature);
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(KronosError::InvalidRequest(
                "temperature must be a finite value > 0".into(),
            ));
        }

        // 1. Schema Mapper: choices -> single token IDs, prompt -> template + tokens
        let candidates = inner
            .tokenizer
            .resolve_choices(&req.choices)
            .map_err(|e| KronosError::Schema(format!("{e:#}")))?;
        let input_ids = inner
            .tokenizer
            .encode_prompt(&req.prompt, &req.choices)
            .map_err(|e| KronosError::InvalidRequest(format!("{e:#}")))?;
        if input_ids.len() > inner.max_prompt_tokens {
            return Err(KronosError::InvalidRequest(format!(
                "prompt is {} tokens, limit is {}",
                input_ids.len(),
                inner.max_prompt_tokens
            )));
        }

        // 2. Candle engine: single prefill pass (lock held only for this call)
        let logits = self
            .lock_engine()
            .next_token_logits(&input_ids)
            .map_err(|e| KronosError::Inference(format!("{e:#}")))?;

        // 3 + 4. Candidate logit extraction + temperature-scaled softmax
        let dist = LogitExtractor::compute(&logits, &candidates, temperature)
            .map_err(|e| KronosError::Inference(format!("{e:#}")))?;

        // 5. Structured decision + confidence
        let selected_choice = dist.choices[dist.best].choice.clone();
        let confidence = dist.choices[dist.best].probability;

        // 6. Downstream execution: route / allow / block
        let action = Action::from_choice(&selected_choice);

        Ok(Decision {
            selected_choice,
            confidence,
            action,
            probabilities: dist.choices,
            latency_us: timer.elapsed_us(),
        })
    }

    pub fn model_kind(&self) -> ModelKind {
        self.inner.model_kind
    }

    pub fn device_label(&self) -> &str {
        &self.inner.device_label
    }

    pub fn prompt_format(&self) -> PromptFormat {
        self.inner.tokenizer.format()
    }

    fn lock_engine(&self) -> MutexGuard<'_, CandleEngine> {
        // The KV cache is cleared at the start of every pass, so a poisoned
        // guard from an earlier panic is safe to reuse.
        self.inner
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
