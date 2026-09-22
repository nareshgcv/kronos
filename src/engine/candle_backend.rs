use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as QwenConfig, ModelForCausalLM as QwenModel};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct CandleEngine {
    model: QwenModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleEngine {
    pub fn load<P: AsRef<Path>>(model_dir: P, device: Device) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");

        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Missing config at {:?}", config_path))?;
        let config: QwenConfig = serde_json::from_str(&config_str)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed loading tokenizer: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::BF16, &device)?
        };

        let model = QwenModel::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Evaluates sequence context and extracts output logits at the final position
    pub fn forward_prefill_logits(&self, prompt: &str) -> Result<(Vec<f32>, usize)> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Prompt encoding failed: {}", e))?;

        let input_ids = encoding.get_ids();
        let seq_len = input_ids.len();

        if seq_len == 0 {
            anyhow::bail!("Prompt token sequence is empty.");
        }

        // Tensor shape: [1, seq_len]
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

        // Single forward pass execution (No decoding loop)
        let logits = self.model.forward(&input_tensor, 0)?;

        // Extract last position logits: [1, seq_len, vocab_size] -> [vocab_size]
        let last_logits = logits.i((0, seq_len - 1))?;
        let logits_vec: Vec<f32> = last_logits.to_dtype(DType::F32)?.to_vec1()?;

        Ok((logits_vec, seq_len))
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
  }
