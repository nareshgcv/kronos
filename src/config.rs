use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Qwen2,
    Llama,
    Mistral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KronosConfig {
    pub server_port: u16,
    pub model_id: String,
    pub model_family: ModelFamily,
    pub revision: String,
    pub use_ipc: bool,
    pub ipc_socket_path: String,
}

impl Default for KronosConfig {
    fn default() -> Self {
        Self {
            server_port: 3000,
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            model_family: ModelFamily::Qwen2,
            revision: "main".to_string(),
            use_ipc: true,
            ipc_socket_path: "/tmp/kronos.sock".to_string(),
        }
    }
}
