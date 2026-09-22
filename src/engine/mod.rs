pub mod candle_backend;
pub mod logit_extractor;

use candle_backend::CandleEngine;

impl CandleEngine {
    /// Safe shallow handle wrapper to share engine references across Tokio tasks
    pub fn clone_engine(&self) -> Self {
        Self {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
        }
    }
}
