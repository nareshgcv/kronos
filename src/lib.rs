pub mod config;
pub mod engine;
pub mod server;
pub mod tokenizer;
pub mod utils;

// Re-export key structures for clean public API access
pub use config::{select_best_device, KronosConfig};
pub use engine::candle_backend::CandleEngine;
pub use engine::logit_extractor::{LogitExtractor, ProbabilityOutput};
pub use tokenizer::schema_mapper::{ResolvedChoice, SchemaMapper};
pub use utils::metrics::LatencyTimer;

use anyhow::Result;
use std::sync::Arc;

/// High-level embedded library API for running direct in-process Kronos decisions
pub struct EmbeddedKronos {
    engine: Arc<CandleEngine>,
}

impl EmbeddedKronos {
    /// Initializes Kronos directly inside another Rust process or C++ host application
    pub fn new(model_dir: &str) -> Result<Self> {
        let device = select_best_device();
        let engine = CandleEngine::load(model_dir, device)?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    /// Evaluates a System 1 decision synchronously in-memory (< 10ms)
    pub fn evaluate(
        &self,
        prompt: &str,
        choices: &[String],
        temperature: f32,
    ) -> Result<ProbabilityOutput> {
        // 1. Resolve choices to single token IDs
        let resolved = SchemaMapper::resolve_choices(self.engine.tokenizer(), choices)?;

        // 2. Run single prefill forward pass
        let (logits, _) = self.engine.forward_prefill_logits(prompt)?;

        // 3. Extract Softmax probability vector
        let output = LogitExtractor::compute_probabilities(&logits, &resolved, temperature);

        Ok(output)
    }
}
