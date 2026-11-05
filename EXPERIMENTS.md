# Kronos experiments: baselines, calibration and out-of-distribution inputs

Follow-up experiments after [Gate 1](Gate1.md). Run on 2026-11-05.

## Setup

| Item | Value |
|---|---|
| Hardware | Google Colab, NVIDIA Tesla T4 (compute capability 7.5, 15 GB) |
| CUDA | 13.0 |
| Candle | 0.11 |
| Kronos model | Qwen3-4B, F16 (`KRONOS_DTYPE=f16`; BF16 kernels are not available on the T4) |
| Datasets | 8 project datasets in `eval/datasets`: 182 `test` cases, 42 `calibration` cases |

Accuracy is always measured on the 182 `test` cases only. Latency runs cycle through all 224 prompts (test + calibration) for both Kronos and Laya, so the two use the same inputs.

## How Kronos decides

Kronos does not generate text. It runs one forward pass over the prompt, reads the next-token logit of each label's token, and applies a softmax over the labels only. The answer is therefore always one of the given labels, and each label must be a single token for the model.

## 1. Embeddings baseline

Each input is matched to the closest label using `BAAI/bge-small-en-v1.5` embeddings.

- **Zero-shot:** label = `"label: definition"` (the same information Kronos gets).
- **Few-shot:** each label vector also includes the 6 calibration examples per dataset.

| Dataset | Embeddings zero-shot | Embeddings few-shot | Laya |
|---|---|---|---|
| agent_gating | 66.7% | 70.8% | 62.5% |
| customer_support_routing | 90.6% | 100.0% | 96.9% |
| guardrail | 54.2% | 50.0% | 66.7% |
| guardrail_yesno | 37.5% | 62.5% | 75.0% |
| sanity | 41.7% | 41.7% | 83.3% |
| sentiment | 58.3% | 66.7% | 95.8% |
| tool_selection | 79.2% | 83.3% | 87.5% |
| trade_risk | 61.1% | 33.3% | 66.7% |
| **Overall** | **63.7%** | **67.6%** | **80.2%** |

**Kronos overall on the same 182 test cases: 94.5%** (per-dataset results in [Gate1.md](Gate1.md)).

**Finding:** embeddings handle topic routing well (100% on support routing with 6 examples) but fail on decisions that need judgment, such as guardrails, gating and risk (33–71%). Kronos's advantage is on judgment decisions, not topic routing.

Each dataset has only 12–32 test cases, so one case moves a score by 3–8 points. Treat large gaps as meaningful and small gaps as noise.

## 2. Latency and memory

| System | p50 | p95 | p99 | max | Throughput | GPU memory |
|---|---|---|---|---|---|---|
| Embeddings (bge-small) | 11.5 ms | 20.1 ms | 22.7 ms | – | – | 0.16 GB |
| Laya (10,000 passes) | 38.6 ms | 49.3 ms | 62.7 ms | 182.6 ms | 24.9/s | 2.43 GB* |
| Kronos, Qwen3-4B (2,000 passes) | 98.2 ms | 108.0 ms | 110.7 ms | 112.6 ms | 10.9/s | 8.2 GB |

\* PyTorch-allocated memory only (`torch.cuda.max_memory_allocated`), which excludes the CUDA context. Kronos memory was measured with `nvidia-smi` during the run.

The embeddings latency excludes the label embeddings, which are computed once; Kronos reads the labels on every request.

### T4 power throttling

Long runs on the Colab T4 slow down because the GPU hits its 70 W power cap:

- Repeated 10,000-pass Kronos runs measured p50 99.2 → 110.1 → 120.8 ms.
- During a 2,000-pass run, clocks fell from 1590 MHz to about 750–810 MHz at ~67 W and 78 °C, with throttle reason `0x4` (power cap).

Short runs in a fresh runtime (2,000 passes) match the Gate 1 figure (96.9 ms). Treat Colab T4 latency as varying by roughly 10–25% between runs.

## 3. Confidence calibration

Kronos on the 182 test cases, grouped by reported confidence:

| Confidence | Cases | Accuracy | Avg confidence |
|---|---|---|---|
| 0.90–0.99 | 2 | 50.0% | 92.9% |
| 0.99–1.00 | 180 | 95.0% | 100.0% |

- Accuracy: **94.5%**
- Expected calibration error (ECE): **5.4 points**

**Finding:** Kronos is overconfident. Almost every answer, including the wrong ones, is reported at about 100% confidence, because the label logits are far apart (e.g. 37.5 vs 14.4). **The confidence score cannot yet be used to send uncertain cases to a human.** Next step: temperature scaling fitted on the calibration split.

## 4. Out-of-distribution inputs

Eight clearly off-topic inputs (weather, a poem request, random characters, arithmetic, etc.) were sent to the support-routing decision.

| Setup | Result |
|---|---|
| Original labels only | Every input forced into a real category, at 61–100% confidence (e.g. a poem request → `sales`, 100%) |
| With an added `none` label | **8/8** caught as `none` |
| Routing accuracy on real tickets with `none` added | **100%** (no real tickets lost to `none`) |

In-distribution routing confidence: median 100%, lowest 99.4%.

**Finding:** always include a `none` label, defined as "the request does not fit any of the other categories". These inputs were obviously off-topic; near-topic inputs still need testing.

## 5. Closely related labels

Three label pairs with close definitions (`refund` / `dispute`, `escalate` / `review`, `cancel` / `pause`), 4 cases each: **12/12 correct**, all at ~100% confidence.

**Limitation:** the cases were written to be clear-cut, so this shows Kronos separates close definitions, but it is not a test of genuinely ambiguous inputs.

## Build notes for Colab (CUDA 13)

- **BF16 on the T4:** fails with `CUDA_ERROR_NOT_FOUND: named symbol not found`. Use `KRONOS_DTYPE=f16`.
- **Candle 0.11 + CUDA 13:** `candle-kernels/src/compatibility.cuh` defines `__hmax_nan` / `__hmin_nan`, which CUDA 13's `cuda_fp16.hpp` now also defines. Renaming Candle's two definitions fixes the build.
- **`CUDA_COMPUTE_CAP=75`:** set it explicitly for the T4.

## Summary

| Claim | Status |
|---|---|
| Beats embeddings and Laya on judgment decisions without training | ✅ Supported (94.5% vs 63.7% and 80.2%) |
| Rejects off-topic input when a `none` label is included | ✅ Supported (8/8, no accuracy loss) |
| Confidence score usable for human review | ❌ Not yet (overconfident; calibration needed) |
| Fastest option | ❌ No (embeddings ~8× and Laya ~2.5× faster) |
| Safe for untrusted input | ❌ No (prompt-injectable; see Gate 1) |

## Next steps

1. Temperature scaling on the calibration split; re-measure ECE.
2. Near-topic out-of-distribution and genuinely ambiguous test cases.
3. Larger test sets (100+ cases) for the most important datasets.
4. Comparison with llama.cpp running Qwen3-4B, and 4-bit quantization.
