//! Kronos: a zero-generation decision runtime.
//!
//! One prefill pass over the prompt, then a temperature-scaled softmax over the
//! logits of a fixed set of single-token choices. No autoregressive decoding.

pub mod config;
pub mod engine;
pub mod server;
pub mod tokenizer;
pub mod utils;

pub use config::{select_best_device, KronosConfig, PromptFormat};
pub use engine::candle_backend::{CandleEngine, ModelKind};
pub use engine::logit_extractor::{Distribution, LogitExtractor};
pub use engine::{ChoiceProbability, Decision, DecisionRequest, KronosError, KronosResult, LogitBackend};
pub use tokenizer::{ResolvedChoice, SchemaMapper, SpacePolicy, TokenizerManager};
pub use utils::metrics::{LatencySnapshot, LatencyStats, LatencyTimer};

use anyhow::Result;
use candle_core::Device;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

/// Embedded, in-process Kronos API.
///
/// Cheap to clone and safe to share across threads. Forward passes are
/// serialized per instance because the model keeps a KV cache; tokenization
/// and choice resolution run outside the lock. `evaluate` blocks the calling
/// thread, so from async code call it inside `tokio::task::spawn_blocking`.
#[derive(Clone)]
pub struct EmbeddedKronos {
    inner: Arc<Inner>,
}

struct Inner {
    engine: Mutex<CandleEngine>,
    tokenizer: TokenizerManager,
    stats: LatencyStats,
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
                stats: LatencyStats::new(),
                model_kind,
                device_label,
                default_temperature: config.default_temperature,
                max_choices: config.max_choices,
                max_prompt_tokens: config.max_prompt_tokens,
            }),
        })
    }

        let result = self.evaluate_inner(req, &timer);
        match &result {
            Ok(_) => self.inner.stats.record(timer.elapsed_us()),
            Err(_) => self.inner.stats.record_error(),
        }
        result
    }

    fn evaluate_inner(&self, req: &DecisionRequest, timer: &LatencyTimer) -> KronosResult<Decision> {
        let inner = &self.inner;

        // 1. Validate request
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
        if let Some(m) = req.min_confidence {
            if !(0.0..=1.0).contains(&m) {
                return Err(KronosError::InvalidRequest(
                    "min_confidence must be between 0 and 1".into(),
                ));
            }
        }

        // 2. Resolve choices to single token IDs (cached, outside the engine lock)
        let resolved = inner
            .tokenizer
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

        // 4. Single prefill pass; the lock is held only for this statement
        let logits = self
            .lock_engine()
            .next_token_logits(&input_ids)
            .map_err(|e| KronosError::Inference(format!("{e:#}")))?;

        // 5. Constrained softmax over the candidate tokens
        let dist = LogitExtractor::compute(&logits, &resolved, temperature)
            .map_err(|e| KronosError::Inference(format!("{e:#}")))?;

        let selected_choice = dist.choices[dist.best].choice.clone();
        let confidence = dist.choices[dist.best].probability;

        Ok(Decision {
            selected_choice,
            confidence,
            choice_mass: dist.choice_mass,
            abstained: req.min_confidence.is_some_and(|m| confidence < m),
            probabilities: dist.choices,
            prompt_tokens: input_ids.len(),
            latency_us: timer.elapsed_us(),
        })
    }

    /// Runs one throwaway forward pass so kernels/shaders are compiled before traffic.
    pub fn warmup(&self) -> Result<()> {
        let ids = self
            .inner
            .tokenizer
            .encode_prompt("warmup", &["yes".to_owned(), "no".to_owned()])?;
        self.lock_engine().next_token_logits(&ids)?;
        Ok(())
    }

    pub fn stats(&self) -> LatencySnapshot {
        self.inner.stats.snapshot()
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
        // A panic mid-forward poisons the mutex. The KV cache is cleared at the
        // start of every pass, so recovering the guard is safe.
        self.inner
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
