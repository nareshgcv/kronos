# ⚡ Kronos

**Kronos** is an ultra-low-latency, zero-generation **System 1 decision runtime** built in pure Rust and powered by [Hugging Face Candle](https://github.com/huggingface/candle).

Instead of generating an answer token-by-token, Kronos performs a **single prefill forward pass**, extracts logits for a predefined decision schema, and converts those logits into confidence scores.

The result is a local decision primitive for **AI routing, guardrails, agent gating, tool selection, and real-time AI systems**.

> **Kronos — The Decision Runtime for Real-Time AI**

---

## 🚀 Why Kronos?

Modern AI systems frequently invoke expensive models for decisions that do not require a generated response.

Before executing an LLM, agent, or tool, an application may only need to determine:

* Which model should handle this request?
* Should this request be allowed?
* Which agent should receive it?
* Which tool should be selected?
* Is the request safe?
* Should expensive inference be skipped?
* Which execution path should be selected?

Kronos is designed to make these decisions locally with minimal inference overhead.

```text
Incoming Request
       │
       ▼
┌──────────────────────┐
│       KRONOS         │
│                      │
│  Local inference     │
│  Schema-constrained  │
│  Zero generation     │
│  Low latency         │
└──────────┬───────────┘
           │
        Decision
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  Route   Allow  Block
    │      │      │
    ▼      ▼      ▼
 Agent    LLM   Reject
```

---

# 🧠 Core Idea

Traditional LLM inference generally relies on autoregressive generation:

```text
Prompt
  │
  ▼
Prefill
  │
  ▼
Token → Token → Token → Token
  │
  ▼
Generated response
```

Kronos uses a different execution path:

```text
Prompt
  │
  ▼
Single Prefill Forward Pass
  │
  ▼
Relevant Choice Logits
  │
  ▼
Temperature-Scaled Softmax
  │
  ▼
Structured Decision
```

There is no autoregressive decoding loop and no generated text that needs to be parsed back into a decision.

---

# 🎯 Schema-Constrained Decisions

Kronos evaluates a predefined set of candidate choices.

For example:

```rust
let decision = kronos.decide(
    "Is this transaction safe to process?",
    &["APPROVE", "REJECT", "HALT"],
)?;
```

Conceptually, the model produces logits such as:

```text
APPROVE  →  4.82
REJECT   →  1.27
HALT     →  0.63
```

Kronos extracts the relevant logits and applies temperature-scaled Softmax:

```text
APPROVE  →  0.94
REJECT   →  0.04
HALT     →  0.02
```

The resulting decision is selected from the predefined schema.

### What this guarantees

At the output layer, Kronos can guarantee that the selected decision is one of the allowed schema choices.

In other words:

> **Zero invalid-choice outputs.**

### What this does not guarantee

Schema constraints do **not** guarantee semantic correctness.

If the underlying model incorrectly interprets the input, Kronos can still produce an incorrect decision with a high confidence score.

For example:

```text
Correct semantic judgment:
SAFE

Model judgment:
UNSAFE

Kronos:
UNSAFE → 0.91
```

The output is structurally valid, but the underlying model judgment can still be wrong.

Therefore, **schema compliance and semantic accuracy are separate properties**.

---

# ⚠️ Schema & Tokenizer Considerations

Kronos currently supports **single-token choices only**.

Each choice must map to exactly one distinct token in the selected model's vocabulary. Kronos first tries every choice as written, then every choice with a leading space. The same spelling is used for all choices so none gets an unfair advantage.

A request is rejected with a schema error if:

* any choice tokenizes to more than one token
* two choices map to the same token
* choices are duplicated or empty

Natural-language choices such as:

```text
"approve transaction"
"send to security agent"
"requires human review"
```

usually span multiple tokens and are **not supported** in the current version.

Whether a label is a single token depends on the tokenizer. Uppercase labels like `APPROVE` may be split into several tokens by some models, so test custom labels against your model before deploying.

### Roadmap

Multi-token choice support (for example, sequence-level scoring across a candidate's tokens) is under consideration. It would carry different latency characteristics from the single-token path.

All Kronos performance claims apply to the single-token path.

---

# ✨ Key Features

## ⚡ Zero-Generation Decision Engine

* Single prefill forward pass
* No autoregressive decoding loop
* No generated-text parsing
* Direct candidate-logit extraction
* Temperature-scaled probability calculation
* Structured decision output

## 🎯 Deterministic Output Space

The application defines the valid decision space before inference by passing a list of `choices` with each request.

```text
choices
  │
  ├── allow    → action: allow   (reserved)
  ├── block    → action: block   (reserved)
  └── billing  → action: route { target: "billing" }
```

Kronos selects from that predefined list rather than allowing arbitrary generated text.

**Choice rules**

* `allow` and `block` (any case) are reserved and map directly to allow / block actions.
* Every other choice becomes a `route` action, with the choice name as the target.
* Each request needs between 2 and `max_choices` choices (default 64).
* Choices must be unique and non-empty.
* Each choice must map to exactly one distinct token in the model's vocabulary. Multi-token choices are rejected with a schema error, so test custom labels against your model's tokenizer.

This provides **structural output constraints**, not guaranteed semantic correctness.

## 🦀 Pure Rust Runtime

Kronos is implemented in Rust and built around [Candle](https://github.com/huggingface/candle).

The runtime is designed for:

* Embedded inference
* Low-latency services
* Edge systems
* On-premise deployments
* Local AI infrastructure

## 🧩 Multi-Architecture Model Support

Kronos is designed to support multiple open-weight model architectures through dynamic model configuration.

Current architecture targets include:

* Qwen3
* Llama 3
* Mistral

Model support depends on the corresponding Candle implementation and Kronos integration.

## 💾 Safetensors Support

Kronos can load single or multi-shard `.safetensors` model weights using Candle's memory-mapped weight loading.

## 🖥️ Hardware Acceleration

Kronos supports the following Candle backends, enabled through Cargo features:

* CUDA (`--features cuda`)
* Metal (`--features metal`)
* CPU (default)

At startup, Kronos picks the best compiled-in device in the order CUDA, Metal, CPU. It uses F32 on CPU and BF16 on accelerators.

## 🌐 Multiple Interfaces

### Embedded Rust API

Kronos can run directly inside a Rust application through `EmbeddedKronos`:

```rust
let kronos = EmbeddedKronos::new("./models/Qwen3-0.6B")?;

let decision = kronos.decide(
    "Customer asks for a refund on order #4821",
    &["allow", "block", "billing"],
)?;

println!("{} ({:.2})", decision.selected_choice, decision.confidence);
```

For per-request temperature, use `evaluate` with a `DecisionRequest`:

```rust
let decision = kronos.evaluate(&DecisionRequest {
    prompt: "Customer asks for a refund on order #4821".into(),
    choices: vec!["allow".into(), "block".into(), "billing".into()],
    temperature: Some(0.7),
})?;
```

`evaluate` is blocking. From async code, call it inside `tokio::task::spawn_blocking`.

This avoids network overhead and is intended for latency-sensitive applications.
### REST API

An Axum-based HTTP interface provides endpoints such as:

```text
POST /v1/decision
GET  /health
```

### Unix Domain Socket

For local sidecar deployments, Kronos provides a Unix Domain Socket interface using newline-delimited JSON framing.

This allows applications written in other languages to communicate with a local Kronos process without requiring a remote service.

---

# 🏗️ Architecture

```mermaid
flowchart TD
    A["Incoming Payload / Prompt"] --> B["Schema Mapper"]
    B --> C["Candle Engine<br/>Single Prefill Pass"]
    C --> D["Candidate Logit Extraction"]
    D --> E["Temperature-Scaled Softmax"]
    E --> F["Structured Decision + Confidence"]

    F --> G{"Downstream Execution"}

    G -->|Route| H["Target Agent / Model / Service"]
    G -->|Allow| I["Primary LLM Pipeline"]
    G -->|Block| J["Reject Request"]
```

---

# 🔥 Where Kronos Fits

Kronos is intentionally **not a full AI gateway**.

A gateway typically handles concerns such as:

* Provider integrations
* API keys
* Provider failover
* Rate limiting
* Load balancing
* Authentication
* Budgets
* Billing
* Observability
* Request management

Kronos focuses on a narrower problem:

> **Making a local semantic decision before expensive AI execution.**

It can therefore operate underneath or alongside an AI gateway.

```text
                AI APPLICATION
                      │
                      ▼
              ┌───────────────┐
              │    KRONOS     │
              │               │
              │ Local decision│
              │ Runtime       │
              └───────┬───────┘
                      │
              ┌───────┼────────┐
              ▼       ▼        ▼
            Route    Allow    Block
              │       │
              ▼       ▼
        AI Gateway / Application
              │
       ┌──────┼─────────┐
       ▼      ▼         ▼
      LLM   Agent      Tools
```

Kronos can therefore complement systems such as LLM gateways rather than attempting to replace their infrastructure responsibilities.

---

# 🚫 What Kronos Is Not

## Not an AI Gateway

Kronos does not attempt to provide:

* Provider API-key management
* Provider failover
* Billing
* Rate-limit management
* Multi-provider load balancing
* Model marketplaces
* Gateway-level observability

Those are gateway/control-plane responsibilities.

## Not a General-Purpose Text Generator

Kronos is designed to make constrained decisions.

It is not intended to replace a general-purpose LLM for:

* Long-form generation
* Open-ended conversations
* Complex reasoning
* Code generation
* Document generation
* Creative writing

The expected output is a structured decision such as:

```text
ALLOW
BLOCK
ROUTE_A
ROUTE_B
SAFE
UNSAFE
SIMPLE
COMPLEX
```

---

# 🎯 Use Cases

## Model Routing

Select an appropriate model before invoking expensive inference.

```text
Request
   │
   ▼
 Kronos
   │
   ├── SIMPLE ──────► Small Model
   │
   ├── COMPLEX ─────► Large Model
   │
   └── CODE ────────► Coding Model
```

---

## Agent Gating

Determine whether an agent should execute.

```text
User Request
     │
     ▼
   Kronos
     │
 ┌───┴────┐
 ▼        ▼
ALLOW    BLOCK
 │
 ▼
Agent
```

---

## AI Guardrails

Run a local decision before expensive generation.

```text
Request
   │
   ▼
Kronos Guardrail
   │
   ├── SAFE ──────► LLM
   │
   └── UNSAFE ───► Block
```

---

## Tool Selection

Determine which tool or execution path should handle a request.

```text
Request
   │
   ▼
 Kronos
   │
   ├── DATABASE
   ├── WEB
   ├── CALCULATOR
   └── NO_TOOL
```

---

## Model Cascading

Determine whether a larger model is necessary.

```text
Request
   │
   ▼
 Kronos
   │
   ├── Easy ───► Small Model
   │
   └── Hard ───► Large Model
```

The potential value is not simply lower decision latency.

The larger objective is to **avoid unnecessary expensive inference**.

---

# 📊 Latency

Kronos is designed for low-latency local decision inference, with a current development target of approximately **1–5 ms warm inference** under suitable workloads and hardware.

This should be treated as a **benchmark target**, not a universal guarantee.

Actual latency depends on:

* Model architecture
* Model size
* Input length
* Tokenizer
* Hardware
* Backend
* Memory bandwidth
* Schema size
* Runtime configuration
* GPU synchronization
* Concurrency

Meaningful latency reporting should include:

```text
p50
p95
p99
p99.9
max
```

along with the hardware and workload used to produce the measurements.

---

# 📈 Benchmarking

A basic latency benchmark can be run using:


```text
 need to update
```
For production benchmarking, measure the pipeline separately:

```text
Tokenization
     │
     ▼
Schema Resolution
     │
     ▼
Candle Forward Pass
     │
     ▼
GPU Synchronization
     │
     ▼
Logit Extraction
     │
     ▼
Softmax
     │
     ▼
Decision Construction
```

---

# 🧪 Accuracy Matters

Latency alone does not establish whether a decision engine is useful.

A meaningful Kronos evaluation should measure:

```text
Decision Quality
+
Decision Latency
+
Resource Cost
```

Recommended metrics include:

* Accuracy
* Macro F1
* Precision / Recall
* Confusion matrix
* Calibration
* p50 latency
* p95 latency
* p99 latency
* p99.9 latency
* Memory usage
* GPU utilization

The key product question is:

> **Can Kronos make sufficiently accurate semantic decisions at substantially lower latency and cost than invoking a larger downstream model?**

---

# 🔬 Design Philosophy

Kronos follows a simple principle:

> **Don't generate when you only need to decide.**

Many AI workflows do not require a paragraph of generated text.

They require:

```text
ALLOW
BLOCK
ROUTE_A
ROUTE_B
ESCALATE
SAFE
UNSAFE
SIMPLE
COMPLEX
```

When the output space is known in advance, generation can be replaced with constrained decision inference.

---

# 🛠️ Project Structure

A typical Kronos deployment can expose three layers:

```text
┌───────────────────────────────────────┐
│             Application               │
└───────────────────┬───────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    Embedded API          REST / UDS
          │                   │
          └─────────┬─────────┘
                    ▼
             Kronos Engine
                    │
                    ▼
              Candle Runtime
                    │
             ┌──────┴──────┐
             ▼             ▼
           CPU          GPU Backend
```

---

# 💻 Quick Start

## Requirements

* Recent stable Rust
* A local Hugging Face model directory containing:
  * `config.json`
  * `tokenizer.json`
  * `.safetensors` weights (single file or sharded)
* CPU (default), or CUDA / Metal when built with the matching feature

## Run the Server

```bash
cargo run --release -- ./models/Qwen3-0.6B

# With GPU acceleration
cargo run --release --features cuda -- ./models/Qwen3-0.6B
```

By default the HTTP server listens on `127.0.0.1:3000`. On Unix, the socket server also starts at `/tmp/kronos.sock`. Settings can be changed with `KRONOS_*` environment variables (for example `KRONOS_HTTP_ADDR`, `KRONOS_TEMPERATURE`, `KRONOS_IPC`).

## Use as a Library

```rust
use kronos::EmbeddedKronos;

let kronos = EmbeddedKronos::new("./models/Qwen3-0.6B")?;

let decision = kronos.decide(
    "Market volatility: high. Order size: 50,000 USD.",
    &["APPROVE", "REJECT", "HALT"],
)?;

println!("{:?}", decision);
```

---

# 🌐 API Example

```http
POST /v1/decision
Content-Type: application/json
```

```json
{
  "prompt": "Market volatility: high. Order size: 50,000 USD.",
  "choices": ["APPROVE", "REJECT", "HALT"],
  "temperature": 1.0
}
```

`temperature` is optional; the configured default is used if omitted.

Example response:

```json
{
  "selected_choice": "HALT",
  "confidence": 0.91,
  "action": { "type": "route", "target": "HALT" },
  "probabilities": [
    { "choice": "APPROVE", "token_id": 0, "logit": 1.12, "probability": 0.04 },
    { "choice": "REJECT",  "token_id": 0, "logit": 1.30, "probability": 0.05 },
    { "choice": "HALT",    "token_id": 0, "logit": 4.18, "probability": 0.91 }
  ],
  "latency_us": 18250
}
```

Token IDs and latency are illustrative. Errors return HTTP 400 or 500 with:

```json
{ "error": "schema violation: ...", "kind": "schema_violation" }
```

---

# 🔒 Reliability

Kronos is designed around explicit failure handling rather than generated-text parsing.

The runtime provides:

* Structured error propagation
* Health endpoint
* Graceful shutdown
* Unix socket cleanup
* Async HTTP handling
* Dedicated blocking workers for tensor operations
* Microsecond-level latency measurement

Production deployments should still validate:

* OOM behavior
* GPU failures
* Model loading failures
* Invalid schemas
* Invalid model files
* Concurrent requests
* Process restarts
* Resource exhaustion

---

# 🧭 Roadmap

* [ ] More model architectures
* [ ] CUDA optimization
* [ ] Metal optimization
* [ ] CPU SIMD optimization
* [ ] Dynamic batching
* [ ] Multi-token candidate scoring
* [ ] Streaming decision APIs
* [ ] Comprehensive benchmark suite
* [ ] Accuracy evaluation datasets
* [ ] Model routing benchmarks
* [ ] Guardrail benchmarks
* [ ] OpenTelemetry integration
* [ ] C-compatible FFI
* [ ] Python bindings
* [ ] WASM / edge targets
* [ ] Quantized model support
* [ ] Production deployment examples

---

# 🤝 Contributing

Contributions are welcome.

Areas that are particularly useful:

* New Candle model architectures
* Hardware backend optimization
* Benchmarking
* Accuracy datasets
* Routing strategies
* Guardrail evaluation
* API integrations
* Documentation
* Performance profiling

---

# 📜 License

See the repository license for current licensing terms.

---

# ⚡ The Kronos Thesis

Large language models are powerful, but many AI workflows do not require generation for every step.

Sometimes the application only needs to decide:

```text
Should I call the model?
Which model should I call?
Which agent should execute?
Which tool should run?
Should this request be blocked?
```

Kronos provides a local execution layer for those decisions.

```text
             EXPENSIVE AI
                  ▲
                  │
           Only when needed
                  │
           ┌─────────────┐
           │   KRONOS    │
           │             │
           │   Decide    │
           │   Route     │
           │   Gate      │
           │   Filter    │
           └──────┬──────┘
                  │
                  ▼
              AI SYSTEM
```

**Make the decision first.
Generate only when necessary.**
