mod config;
mod engine;
mod server;
mod tokenizer;

use anyhow::Result;
use config::{select_best_device, KronosConfig};
use engine::candle_backend::CandleEngine;
use server::http::{create_router, AppState};
use server::ipc::IpcServer;
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("==================================================");
    println!("     KRONOS CORE :: System 1 Engine Initializing   ");
    println!("==================================================");

    let config = KronosConfig::default();
    let device = select_best_device();

    println!("[Device] Selected Hardware Acceleration: {:?}", device);

    // 1. Load model weights into memory
    let engine = CandleEngine::load(&config.model_dir, device)?;
    let shared_engine = Arc::new(engine);

    // 2. Spawn Sub-2ms Unix IPC Socket Listener
    if config.use_ipc {
        let ipc_server = IpcServer::new(config.ipc_socket_path.clone(), Arc::clone(&shared_engine));
        tokio::spawn(async move {
            if let Err(e) = ipc_server.run().await {
                eprintln!("[IPC Fatal] Server crashed: {}", e);
            }
        });
    }

    // 3. Start HTTP/REST API Server (Axum)
    let state = Arc::new(AppState {
        engine: (*shared_engine).clone_engine(),
    });

    let app = create_router(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], config.server_port));

    println!("[HTTP] REST API listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}    /// Loads the model weights and tokenizer onto GPU/CPU
    pub fn new(model_dir: &str, device: Device) -> Result<Self> {
        let config_path = format!("{}/config.json", model_dir);
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        let weights_path = format!("{}/model.safetensors", model_dir);

        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path))?;
        let config: QwenConfig = serde_json::from_str(&config_str)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                DType::BF16,
                &device,
            )?
        };

        let model = QwenModel::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Evaluates decision in a single forward pass (< 15ms)
    pub fn evaluate_decision(&self, request: DecisionRequest) -> Result<DecisionResponse> {
        let start_time = Instant::now();

        // 1. Resolve candidate string choices into single token IDs
        let mut candidate_tokens: Vec<(String, u32)> = Vec::new();
        for choice in &request.candidate_choices {
            let encoding = self
                .tokenizer
                .encode(choice.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

            let tokens = encoding.get_ids();
            if tokens.is_empty() {
                anyhow::bail!("Choice '{}' produced 0 tokens.", choice);
            }
            // In Kronos System 1 schema, choices map strictly to single token IDs
            candidate_tokens.push((choice.clone(), tokens[0]));
        }

        // 2. Encode the full context prompt
        let prompt_encoding = self
            .tokenizer
            .encode(request.context_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Prompt tokenization error: {}", e))?;
        let input_ids = prompt_encoding.get_ids();
        let seq_len = input_ids.len();

        if seq_len == 0 {
            anyhow::bail!("Input context prompt cannot be empty.");
        }

        // 3. Prepare Tensors
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

        // 4. Executing SINGLE-PASS Prefill Forward Pass (No autoregressive generation loop)
        let logits = self.model.forward(&input_tensor, 0)?;

        // 5. Extract raw logits specifically at the last sequence position: shape [vocab_size]
        let last_logits = logits.i((0, seq_len - 1))?;
        let last_logits_vec: Vec<f32> = last_logits.to_dtype(DType::F32)?.to_vec1()?;

        // 6. Temperature scaling & Max logit extraction for numerical stability
        let temp = if request.temperature <= 0.0 { 1.0 } else { request.temperature };
        let mut extracted_logits: Vec<f32> = Vec::new();
        let mut max_logit = f32::NEG_INFINITY;

        for (_, token_id) in &candidate_tokens {
            let raw_logit = last_logits_vec[*token_id as usize] / temp;
            if raw_logit > max_logit {
                max_logit = raw_logit;
            }
            extracted_logits.push(raw_logit);
        }

        // 7. Compute Softmax over candidate logits only
        let mut sum_exp = 0.0f32;
        let mut exps: Vec<f32> = Vec::with_capacity(extracted_logits.len());

        for logit in &extracted_logits {
            let exp_val = (logit - max_logit).exp();
            exps.push(exp_val);
            sum_exp += exp_val;
        }

        let mut probabilities: HashMap<String, f32> = HashMap::new();
        let mut best_choice = String::new();
        let mut max_prob = -1.0f32;

        for (i, (choice_name, _)) in candidate_tokens.iter().enumerate() {
            let prob = exps[i] / sum_exp;
            probabilities.insert(choice_name.clone(), prob);

            if prob > max_prob {
                max_prob = prob;
                best_choice = choice_name.clone();
            }
        }

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(DecisionResponse {
            selected_choice: best_choice,
            confidence: max_prob,
            probabilities,
            execution_time_ms: elapsed_ms,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Select CUDA if available, or fall back to Apple Metal / CPU
    let device = Device::cuda_if_available(0)
        .or_else(|_| Device::new_metal(0))
        .unwrap_or(Device::Cpu);

    println!("Initializing Kronos Engine on device: {:?}", device);

    // Initialize Engine (expects directory with config.json, model.safetensors, tokenizer.json)
    let engine = Arc::new(KronosEngine::new("./model_weights", device)?);

    // Simulated Real-Time Game Telemetry Decision Request
    let request = DecisionRequest {
        context_prompt: "SYS_STATE: PLAYER_HP=12% AMMO=5% ENEMIES_NEARBY=8 DISTANCE=2m | ACTION_DECISION:".to_string(),
        candidate_choices: vec![
            "SPAWN_HEALTH".to_string(),
            "SPAWN_AMMO".to_string(),
            "TRIGGER_FLANK".to_string(),
            "HOLD".to_string(),
        ],
        temperature: 0.8,
    };

    println!("Running Kronos Sub-15ms Single-Pass Decision...");
    let response = engine.evaluate_decision(request)?;

    println!("\n--- Kronos Execution Result ---");
    println!("Selected Decision : {}", response.selected_choice);
    println!("Confidence        : {:.2}%", response.confidence * 100.0);
    println!("Latency           : {:.2} ms", response.execution_time_ms);
    println!("Probability Map   : {:#?}", response.probabilities);

    Ok(())
}
