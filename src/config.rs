use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

/// How the user prompt is wrapped before tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptFormat {
    /// Fed to the model as-is (base models or your own template).
    Raw,
    /// Qwen / ChatML: `<|im_start|>role ... <|im_end|>`
    ChatMl,
    /// Llama 3 instruct headers.
    Llama3,
    /// Mistral instruct: `[INST] ... [/INST]`
    Mistral,
}

impl FromStr for PromptFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "raw" => Ok(Self::Raw),
            "chatml" | "qwen" | "qwen2" => Ok(Self::ChatMl),
            "llama3" => Ok(Self::Llama3),
            "mistral" => Ok(Self::Mistral),
            other => bail!("unknown prompt format '{other}' (raw | chatml | llama3 | mistral)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KronosConfig {
    pub max_choices: usize,
    /// Max HTTP body / IPC line size in bytes.
    pub max_request_bytes: usize,
    /// Requests allowed in flight or queued before returning "busy".
    pub max_queued_requests: usize,
    pub warmup: bool,
}

impl Default for KronosConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("./models/qwen2.5-0.5b-instruct"),
            http_addr: SocketAddr::from(([127, 0, 0, 1], 3000)),
            use_ipc: cfg!(unix),
            ipc_socket_path: PathBuf::from("/tmp/kronos.sock"),
            prompt_format: None,
            system_prompt: "You are a decision classifier. Reply with exactly one of the allowed answers and nothing else.".to_owned(),
            append_choices_to_prompt: true,
            default_temperature: 1.0,
            max_prompt_tokens: 4096,
            max_choices: 64,
            max_request_bytes: 64 * 1024,
            max_queued_requests: 64,
            warmup: true,
        }
    }
}

impl KronosConfig {
    /// Defaults overridden by `KRONOS_*` environment variables.
    pub fn from_env() -> Result<Self> {
        let mut c = Self::default();
        if let Some(v) = env_var::<PathBuf>("KRONOS_MODEL_DIR")? { c.model_dir = v; }
        if let Some(v) = env_var::<SocketAddr>("KRONOS_HTTP_ADDR")? { c.http_addr = v; }
        if let Some(v) = env_bool("KRONOS_IPC")? { c.use_ipc = v; }
        if let Some(v) = env_var::<PathBuf>("KRONOS_IPC_SOCKET")? { c.ipc_socket_path = v; }
        if let Some(v) = env_var::<PromptFormat>("KRONOS_PROMPT_FORMAT")? { c.prompt_format = Some(v); }
        if let Some(v) = env_var::<String>("KRONOS_SYSTEM_PROMPT")? { c.system_prompt = v; }
        if let Some(v) = env_bool("KRONOS_APPEND_CHOICES")? { c.append_choices_to_prompt = v; }
        }
        if !self.default_temperature.is_finite() || self.default_temperature <= 0.0 {
            bail!("default_temperature must be a finite value > 0");
        }
        if self.max_choices < 2 {
            bail!("max_choices must be at least 2");
        }
        if self.max_prompt_tokens == 0 {
            bail!("max_prompt_tokens must be > 0");
        }
        if self.max_request_bytes < 1024 {
            bail!("max_request_bytes must be at least 1024");
        }
        if self.max_queued_requests == 0 {
            bail!("max_queued_requests must be > 0");
        }
        Ok(())
    }
}

/// CUDA, then Metal, then CPU. Only succeeds for backends compiled in via features.
pub fn select_best_device() -> Device {
    if candle_core::utils::cuda_is_available() {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    if candle_core::utils::metal_is_available() {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

/// F32 on CPU, BF16 on accelerators.
pub fn dtype_for_device(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    }
}

fn env_var<T>(key: &str) -> Result<Option<T>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(key) {
        Ok(v) => v
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow!("{key}={v:?}: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(anyhow!("{key}: {e}")),
    }
}

fn env_bool(key: &str) -> Result<Option<bool>> {
    match env_var::<String>(key)? {
        None => Ok(None),
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(Some(true)),
            "0" | "false" | "no" | "off" => Ok(Some(false)),
            _ => bail!("{key}={v:?}: expected true/false"),
        },
    }
}
