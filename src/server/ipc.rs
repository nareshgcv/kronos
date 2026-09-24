use crate::engine::candle_backend::CandleEngine;
use crate::engine::logit_extractor::LogitExtractor;
use crate::tokenizer::schema_mapper::SchemaMapper;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
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
    pub probabilities: HashMap<String, f32>,
    pub latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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
        // Clean up stale socket file before binding
        let _ = tokio::fs::remove_file(&self.socket_path).await;

        let listener = UnixListener::bind(&self.socket_path)?;
        println!("[IPC] Listening on Unix Socket: {}", self.socket_path);

        loop {
            let (stream, _) = listener.accept().await?;
            let engine = Arc::clone(&self.engine);

            tokio::spawn(async move {
                let (reader, mut writer) = stream.into_split();
                let mut buffered_reader = BufReader::new(reader);
                let mut line = String::new();

                // Reads line-delimited JSON messages (\n framing) to handle variable payload sizes cleanly
                while buffered_reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    let start = std::time::Instant::now();
                    let payload_str = line.trim().to_string();
                    line.clear();

                    if payload_str.is_empty() {
                        continue;
                    }

                    let engine = Arc::clone(&engine);

                    // Execute model inference on blocking thread pool
                    let response = tokio::task::spawn_blocking(move || {
                        // 1. Deserialize request payload
                        let req: IpcRequest = match serde_json::from_str(&payload_str) {
                            Ok(val) => val,
                            Err(e) => {
                                return IpcResponse {
                                    selected_choice: String::new(),
                                    confidence: 0.0,
                                    probabilities: HashMap::new(),
                                    latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                                    error: Some(format!("Invalid Payload JSON: {e}")),
                                };
                            }
                        };

                        // 2. Resolve choices strictly to single token IDs
                        let resolved = match SchemaMapper::resolve_choices(
                            engine.tokenizer(),
                            &req.choices,
                        ) {
                            Ok(res) => res,
                            Err(e) => {
                                return IpcResponse {
                                    selected_choice: String::new(),
                                    confidence: 0.0,
                                    probabilities: HashMap::new(),
                                    latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                                    error: Some(format!("Schema Violation: {e}")),
                                };
                            }
                        };

                        // 3. Run single forward prefill pass
                        let (logits, _) = match engine.forward_prefill_logits(&req.prompt) {
                            Ok(res) => res,
                            Err(e) => {
                                return IpcResponse {
                                    selected_choice: String::new(),
                                    confidence: 0.0,
                                    probabilities: HashMap::new(),
                                    latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                                    error: Some(format!("Inference Error: {e}")),
                                };
                            }
                        };

                        // 4. Extract Softmax probability vector
                        let output = LogitExtractor::compute_probabilities(
                            &logits,
                            &resolved,
                            req.temperature.unwrap_or(1.0),
                        );

                        IpcResponse {
                            selected_choice: output.selected_choice,
                            confidence: output.confidence,
                            probabilities: output.probabilities,
                            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                            error: None,
                        }
                    })
                    .await
                    .unwrap_or_else(|e| IpcResponse {
                        selected_choice: String::new(),
                        confidence: 0.0,
                        probabilities: HashMap::new(),
                        latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                        error: Some(format!("Worker Task Error: {e}")),
                    });

                    // Write JSON response followed by newline character delimiter
                    let mut response_bytes = serde_json::to_vec(&response).unwrap();
                    response_bytes.push(b'\n');

                    if writer.write_all(&response_bytes).await.is_err() {
                        break;
                    }
                }
            });
        }
    }
}
