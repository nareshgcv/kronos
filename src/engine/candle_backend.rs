use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    llama::{Config as LlamaConfig, Model as LlamaModel},
    mistral::{Config as MistralConfig, Model as MistralModel},
    qwen2::{Config as QwenConfig, ModelForCausalLM as QwenModel},
};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Supported open-weight model architectures for Kronos
pub enum ModelArchitecture {
    Qwen2(QwenModel),
    Llama(LlamaModel),
    Mistral(MistralModel),
}

impl ModelArchitecture {
    pub fn forward(&self, input_tensor: &Tensor, seq_len_offset: usize) -> Result<Tensor> {
        match self {
            Self::Qwen2(m) => Ok(m.forward(input_tensor, seq_len_offset)?),
            Self::Llama(m) => Ok(m.forward(input_tensor, seq_len_offset)?),
            Self::Mistral(m) => Ok(m.forward(input_tensor, seq_len_offset)?),
        }
    }
}

pub struct CandleEngine {
    model: ModelArchitecture,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleEngine {
    pub fn load<P: AsRef<Path>>(model_dir: P, device: Device) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");

        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Missing config at {:?}", config_path))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed loading tokenizer: {}", e))?;

        // 1. Automatically collect all .safetensors files (supports single-file and sharded models)
        let mut weights_paths: Vec<PathBuf> = Vec::new();
        if dir.join("model.safetensors").exists() {
            weights_paths.push(dir.join("model.safetensors"));
        } else {
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if let Some(ext) = path.extension() {
                    if ext == "safetensors" {
                        weights_paths.push(path);
                    }
                }
            }
        }

        if weights_paths.is_empty() {
            anyhow::bail!("No .safetensors weight files found in {:?}", dir);
        }

        // Select precision according to hardware device
        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, &device)?
        };

        // 2. Auto-detect model architecture from config.json (`architectures` or `model_type`)
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("qwen2");

        let model = match model_type {
            "qwen2" => {
                let config: QwenConfig = serde_json::from_str(&config_str)?;
                ModelArchitecture::Qwen2(QwenModel::new(&config, vb)?)
            }
            "llama" => {
                let config: LlamaConfig = serde_json::from_str(&config_str)?;
                ModelArchitecture::Llama(LlamaModel::new(&config, vb)?)
            }
            "mistral" => {
                let config: MistralConfig = serde_json::from_str(&config_str)?;
                ModelArchitecture::Mistral(MistralModel::new(&config, vb)?)
            }
            other => anyhow::bail!("Unsupported model architecture: '{}'", other),
        };

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

        // Execute single forward pass using dynamic model architecture
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
