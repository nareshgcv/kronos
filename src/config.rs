use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server_addr: String,
    pub model_id: String,
    pub revision: String,
    pub device: DeviceConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),
    Metal,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_addr: "127.0.0.1:3000".to_string(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            revision: "main".to_string(),
            device: DeviceConfig::Cpu,
        }
    }
}
