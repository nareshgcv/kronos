//! Candle transformer prefill runner.

use super::LogitBackend;
use crate::config::{dtype_for_device, PromptFormat};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{llama, mistral, qwen2};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Qwen2,
    Llama,
    Mistral,
}

impl ModelKind {
    /// Note: `llama` defaults to the Llama 3 template. For Llama 2 checkpoints,
    /// set the prompt format explicitly.
    pub fn default_prompt_format(self) -> PromptFormat {
        match self {
            Self::Qwen2 => PromptFormat::ChatMl,
            Self::Llama => PromptFormat::Llama3,
            Self::Mistral => PromptFormat::Mistral,
        }
    }
}

enum Model {
    Qwen2(qwen2::ModelForCausalLM),
    Mistral(mistral::Model),
    /// Llama keeps its cache outside the model; it's created with KV caching off.
    Llama { model: llama::Llama, cache: llama::Cache },
}

pub struct CandleEngine {
    model: Model,
    device: Device,
    kind: ModelKind,
    vocab_size: usize,
}

impl CandleEngine {
    pub fn load(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .with_context(|| format!("parsing {}", config_path.display()))?;

        let kind = detect_kind(&config_json)?;
        let vocab_size = config_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let weights = collect_safetensors(dir)?;
        let dtype = dtype_for_device(&device);
        // SAFETY: weight files must not be modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };

        let model = match kind {
            ModelKind::Qwen2 => {
                let cfg: qwen2::Config = serde_json::from_str(&config_str)?;
                Model::Qwen2(qwen2::ModelForCausalLM::new(&cfg, vb)?)
            }
            ModelKind::Mistral => {
                let cfg: mistral::Config = serde_json::from_str(&config_str)?;
            }
            ModelKind::Llama => {
                let raw: llama::LlamaConfig = serde_json::from_str(&config_str)?;
                let cfg = raw.into_config(false);
                let cache = llama::Cache::new(false, dtype, &cfg, &device)?;
                Model::Llama {
                    model: llama::Llama::load(vb, &cfg)?,
                    cache,
                }
            }
        };

        Ok(Self {
            model,
            device,
            kind,
            vocab_size,
        })
    }

    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl LogitBackend for CandleEngine {
    fn next_token_logits(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            bail!("prompt produced no tokens");
        }
        // Shape [1, seq_len]
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

        let logits = match &mut self.model {
            // Stateful models: drop anything left from the previous request.
            Model::Qwen2(m) => {
                m.clear_kv_cache();
                m.forward(&input, 0)?
            }
            Model::Mistral(m) => {
                m.clear_kv_cache();
                m.forward(&input, 0)?
            }
            Model::Llama { model, cache } => model.forward(&input, 0, cache)?,
        };

        // All three heads already return only the last position
        // ([1, 1, vocab] or [1, vocab]), so flatten instead of indexing seq_len - 1.
        let logits = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn detect_kind(config: &serde_json::Value) -> Result<ModelKind> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .context("config.json has no `model_type` field")?;
    match model_type {
        "qwen2" => Ok(ModelKind::Qwen2),
        "llama" => Ok(ModelKind::Llama),
        "mistral" => Ok(ModelKind::Mistral),
        other => bail!("unsupported model_type '{other}' (supported: qwen2, llama, mistral)"),
    }
}

/// `model.safetensors`, else the shards listed in the index, else every
/// `*.safetensors` except Mistral's duplicate `consolidated` file.
fn collect_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    }

    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let map = json
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("safetensors index has no `weight_map`")?;
        let mut files: Vec<PathBuf> = map
            .values()
            .filter_map(|v| v.as_str())
            .map(|f| dir.join(f))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files);
    }

    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let is_safetensors = path.extension().is_some_and(|e| e == "safetensors");
        let is_consolidated = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("consolidated"));
        if is_safetensors && !is_consolidated {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        bail!("no .safetensors files found in {}", dir.display());
    }
    Ok(files)
}
