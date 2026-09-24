 # ⚡ Kronos Core

**Kronos Core** is an ultra-low-latency, zero-generation **System 1 decision engine** built in pure Rust and powered by [`candle`](https://github.com/huggingface/candle). 

Instead of waiting for autoregressive text generation (token-by-token parsing taking 200ms–2000ms), Kronos executes a single prefill forward pass over prompt context, extracts target choice logits directly, and applies a temperature-scaled Softmax over a pre-defined schema in **1ms to 5ms**.

---

## ⚡ Key Features

* **Zero-Generation System 1 Engine Core:**
  * **Single Prefill Forward Pass:** Eliminates autoregressive token decoding loops, delivering decisions in 1ms to 5ms.
  * **Deterministic Single-Token Resolution:** Maps candidate choices directly to target vocabulary token IDs, guaranteeing $100\%$ schema compliance without text parsing risk.
  * **Numerically Stable Softmax:** Computes temperature-scaled probabilities using max-logit subtraction ($e^{x - \text{max\_x}}$) to prevent floating-point underflow or overflow.

* **Dynamic Model Architecture & Hardware Dispatch:**
  * **Model Agnosticism:** Dynamic architecture dispatch inspecting `config.json` to support **Qwen2**, **Llama 3**, and **Mistral** open-weight model families.
  * **Multi-Shard Weight Support:** Automatically discovers and memory-maps single or multi-shard `.safetensors` weight files (`VarBuilder::from_mmaped_safetensors`).
  * **Hardware-Aware Precision:** Deploys `BF16` precision on CUDA and Metal acceleration targets, falling back gracefully to `F32` on CPU.

* **Dual Network Interfaces & Concurrency:**
  * **High-Throughput REST API (Axum):** Asynchronous HTTP server providing standard endpoints (`POST /v1/decision`, `GET /health`).
  * **Sub-2ms Local IPC:** Unix Domain Socket server utilizing newline-delimited JSON framing (`\n`) for low-latency local sidecar integration.
  * **Non-Blocking Execution:** Offloads heavy tensor matrix operations to dedicated worker threads (`tokio::task::spawn_blocking`), preserving Tokio runtime responsiveness.

* **Embedded Library API (`lib.rs`):**
  * **`EmbeddedKronos` Interface:** Thread-safe (`Send + Sync`) struct wrapping `Arc<CandleEngine>` for direct in-process execution inside Rust binaries or C++ hosts with zero network overhead.

* **Production Reliability & Operations:**
  * **Graceful Lifecycle:** Automatic stale Unix socket cleanup and signal handling (`SIGINT`/`Ctrl+C`) during server startup and shutdown.
  * **Zero-Panic Error Propagation:** Structured JSON error responses replace thread panics across network workers.
  * **Sub-Millisecond Micro-Metrics:** Built-in `LatencyTimer` utility providing microsecond (`elapsed_us`) and millisecond (`elapsed_ms`) timing accuracy.

---

## 🏗️ Architecture & Execution Flow

```mermaid
flowchart TD
    A["Incoming Payload / Prompt"] --> B["Schema Mapper :: Resolve Single-Token Schema"]
    B --> C["Candle Engine :: Single Prefill Pass (1–5ms)"]
    C --> D["Logit Extractor :: Softmax Calculation"]
    D --> E["Structured Intent & Confidence Output"]
    
    E --> F{"Downstream Execution"}
    F -->|Route Intent| G["Target Agent / Microservice"]
    F -->|Guardrail Pass| H["Primary LLM Pipeline"]
    F -->|Guardrail Fail| I["Block Request"]
