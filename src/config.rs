use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KronosConfig {
    pub model_dir: PathBuf,
    pub server_port: u16,
    pub ipc_socket_path: String,
    pub default_temperature: f32,
    pub use_ipc: bool,
}

impl Default for KronosConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("./model_weights"),
            server_port: 8080,
            ipc_socket_path: "/tmp/kronos.sock".to_string(),
            default_temperature: 1.0,
            use_ipc: true,
        }
    }
}

pub fn select_best_device() -> Device {
    Device::cuda_if_available(0)
        .or_else(|_| Device::new_metal(0))
        .unwrap_or(Device::Cpu)
}
