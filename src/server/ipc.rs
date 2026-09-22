use crate::engine::candle_backend::CandleEngine;
use crate::engine::logit_extractor::LogitExtractor;
use crate::tokenizer::schema_mapper::SchemaMapper;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;

#[derive(Deserialize)]
pub struct IpcRequest {
    pub prompt: String,
    pub choices: Vec<String>,
    pub temperature: Option<f32>,
}

#[derive(Serialize)]
pub struct IpcResponse {
    pub selected_choice: String,
    pub confidence: f32,
    pub latency_ms: f64,
}

pub struct IpcServer {
    socket_path: String,
    engine: Arc<CandleEngine>,
}

impl IpcServer {
    pub fn new(socket_path: String, engine: Arc<CandleEngine>) -> Self {
        Self { socket_path, engine }
    }

    pub async fn run(&self) -> Result<()> {
        // Clean up stale sockets
        let _ = tokio::fs::remove_file(&self.socket_path).await;

        let listener = UnixListener::bind(&self.socket_path)?;
        println!("[IPC] Listening on Unix Socket: {}", self.socket_path);

        loop {
            let (mut stream, _) = listener.accept().await?;
            let engine = Arc::clone(&self.engine);

            tokio::spawn(async move {
                let mut buffer = vec![0u8; 4096];

                while let Ok(bytes_read) = stream.read(&mut buffer).await {
                    if bytes_read == 0 {
                        break;
                    }

                    let start = std::time::Instant::now();

                    // 1. Deserialize request packet
                    let req: IpcRequest = match serde_json::from_slice(&buffer[..bytes_read]) {
                        Ok(val) => val,
                        Err(e) => {
                            eprintln!("[IPC Error] Invalid Payload: {}", e);
                            break;
                        }
                    };

                    // 2. Resolve target tokens & execute prefill pass
                    let resolved = SchemaMapper::resolve_choices(engine.tokenizer(), &req.choices)
                        .expect("Choice schema resolution failed");

                    let (logits, _) = engine
                        .forward_prefill_logits(&req.prompt)
                        .expect("Inference execution failed");

                    // 3. Extract probabilities
                    let output = LogitExtractor::compute_probabilities(
                        &logits,
                        &resolved,
                        req.temperature.unwrap_or(1.0),
                    );

                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                    let res = IpcResponse {
                        selected_choice: output.selected_choice,
                        confidence: output.confidence,
                        latency_ms,
                    };

                    let response_bytes = serde_json::to_vec(&res).unwrap();
                    if stream.write_all(&response_bytes).await.is_err() {
                        break;
                    }
                }
            });
        }
    }
              }
