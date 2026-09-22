# ⚡ Kronos (`kronos`)

> **Sub-10ms System 1 Decision Engine for Bare-Metal & Real-Time AI Systems**

[![Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Inference SLA](https://img.shields.io/badge/Inference_SLA-%3C10ms-brightgreen.svg)]()
[![Hardware Acceleration](https://img.shields.io/badge/Acceleration-CUDA_%7C_Metal-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Kronos Core** is a ultra-low-latency, zero-token, non-generative AI inference engine built in pure Rust. Unlike generative Large Language Models that decode output token-by-token (taking 200ms–2000ms), Kronos executes a **single prefill forward pass**, extracts target choice logits, and applies a temperature-scaled Softmax over a pre-defined schema in **4ms to 12ms**.

It is engineered for real-time applications such as **60 FPS game engine loops (Unreal/Unity/Godot)**, **sub-10ms High-Frequency Trading (HFT) risk filters**, and **local agent security firewalls**.

---

## 🏗️ Architecture Overview

Kronos bypasses autoregressive token generation entirely, mapping static enum choices directly to fixed token IDs in model vocabulary.# kronos
┌────────────────────────────────────────────────────────────────────────┐
│                        Telemetry State / Prompt                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Candle Engine :: Single Prefill Pass (4–8ms)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│        Schema Mapper :: Extract Choice Logits (Fixed Token IDs)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         Logit Extractor :: Temperature-Scaled Softmax Vector          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Structured Choice + Confidence Score Output                │
└───────────────────────────────────┬───────────────────────────────────

─┘
---

## 🚀 Key Features

* **Sub-10ms Latency SLA:** Eliminates token-by-token generation overhead for ultra-fast decision loops.
* **0% Structural Hallucination:** Mathematically constrained to return only choices defined in your input schema.
* **Dual Runtime Modes:**
  * **Daemon Mode:** High-performance REST API (Axum) + Zero-Copy Unix Domain Socket (`/tmp/kronos.sock`).
  * **Embedded Library Mode:** Direct in-process crate integration with zero inter-process communication (IPC) overhead.
* **Multi-Platform Hardware Acceleration:** Native Apple Metal (macOS) and NVIDIA CUDA (Linux/Windows) via Hugging Face `candle-core`.

  ---

## 🛠️ Installation & Setup

### Prerequisites

* [Rust Toolchain](https://rustup.rs/) (edition 2021+)
* **For macOS Acceleration:** Apple Silicon Mac with Xcode command-line tools.
* **For NVIDIA Acceleration:** CUDA Toolkit 11.8+ or 12.x installed.

### 1. Download Model Weights

Place fine-tuned model weights (e.g., Qwen2.5-1.5B or Llama-3.2-1B in `.safetensors` format) into the `./model_weights` directory:

```bash
mkdir -p model_weights
# Place config.json, tokenizer.json, and model.safetensors 

2. Build & Run Kronos Server
For macOS (Apple Metal):

cargo run --release --features metal
For NVIDIA GPUs (CUDA):

cargo run --release --features cuda

💻 Usage Examples
1. Embedded Crate Usage (In-Memory Rust Crate)
Add kronos- to your Cargo.toml:

use kronos_core::EmbeddedKronos;

fn main() -> anyhow::Result<()> {
    // Initialize Kronos in-process
    let kronos = EmbeddedKronos::new("./model_weights")?;

    let prompt = "SYS_STATE: PLAYER_HP=12% AMMO=5% ENEMIES=8 | ACTION_DECISION:";
    let choices = vec![
        "SPAWN_HEALTH".to_string(),
        "SPAWN_AMMO".to_string(),
        "HOLD".to_string()
    ];




    // Synchronous decision in < 8ms
    let decision = kronos.evaluate(prompt, &choices, 0.8)?;

    println!("Selected Action: {}", decision.selected_choice);
    println!("Confidence     : {:.2}%", decision.confidence * 100.0);

    Ok(())
}

______________________________________

2. Sub-2ms Unix Domain Socket IPC (Local Client)
For game engines (C++/Unreal/Unity) communicating with the Kronos daemon locally over /tmp/kronos.sock:

Request Payload:

{
  "prompt": "SYS_STATE: PLAYER_HP=12% AMMO=5% ENEMIES=8 | ACTION_DECISION:",
  "choices": ["SPAWN_HEALTH", "SPAWN_AMMO", "HOLD"],
  "temperature": 0.8
}

Response Payload:




