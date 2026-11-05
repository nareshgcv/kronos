# Kronos experiments: model sizes, baselines, order sensitivity, calibration

Follow-up experiments after [Gate 1](Gate1.md). Run on 2026-11-05.

## Setup

| Item | Value |
|---|---|
| Hardware | Google Colab, NVIDIA Tesla T4 (compute capability 7.5, 15 GB) |
| CUDA | 13.0 |
| Candle | 0.11 |
| Models | Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, all F16 (`KRONOS_DTYPE=f16`; BF16 kernels are not available on the T4) |
| Datasets | 8 project datasets in `eval/datasets`: 182 `test` cases, 42 `calibration` cases |

Accuracy is always measured on the 182 `test` cases only. Latency benchmarks cycle through all 224 prompts (test + calibration) for both Kronos and Laya, so the two use the same inputs.

Each dataset has only 12–32 test cases, so one case moves a score by 3–8 points. Treat large gaps as meaningful and small gaps as noise.

## How Kronos decides

Kronos does not generate text. It runs one forward pass over the prompt, reads the next-token logit of each label's token, and applies a softmax over the labels only. The answer is therefore always one of the given labels, and each label must be a single token for the model.

## 1. Model size: accuracy

`kronos-eval --style defined|fewshot --split test --order-check`, all 8 datasets.

- **defined:** each label comes with its definition.
- **fewshot:** definitions plus worked examples.

| Dataset | 0.6B defined | 0.6B few-shot | 1.7B defined | 1.7B few-shot | 4B defined | 4B few-shot |
|---|---|---|---|---|---|---|
| agent_gating | 50.0% | 50.0% | 100% | 95.8% | 95.8% | 95.8% |
| customer_support_routing | 53.1% | 34.4% | 78.1% | 100% | 100% | 100% |
| guardrail | 50.0% | 79.2% | 91.7% | 87.5% | 95.8% | 100% |
| guardrail_yesno | 50.0% | 50.0% | 91.7% | 66.7% | 95.8% | 95.8% |
| sanity | 50.0% | 50.0% | 100% | 83.3% | 100% | 100% |
| sentiment | 54.2% | 33.3% | 100% | 95.8% | 95.8% | 95.8% |
| tool_selection | 50.0% | 50.0% | 66.7% | 95.8% | 100% | 100% |
| trade_risk | 33.3% | 33.3% | 100% | 88.9% | 66.7% | 100% |
| **Overall (182)** | **49.5%** | **47.3%** | **89.6%** | **90.1%** | **94.5%** | **98.4%** |

Findings:

- **4B few-shot is the most accurate setup: 179 of 182 correct.** Examples help 4B most on trade_risk (66.7% → 100%).
- **1.7B reaches about 90%** in both styles.
- **0.6B is unusable.** On several datasets it gives the same answer for every input (50% accuracy with macro F1 0.333 on two-label tasks), so it is not reading the input. Few-shot prompts make it worse.

## 2. Model size: latency and memory

`cargo bench --bench latency`, 2,000 passes over 224 prompts, `defined`-style prompts. GPU memory is the peak `nvidia-smi` reading during the run.

| System | p50 | p95 | p99 | max | Throughput | GPU memory |
|---|---|---|---|---|---|---|
| Embeddings (bge-small) | 11.5 ms | 20.1 ms | 22.7 ms | – | – | 0.16 GB |
| Kronos + Qwen3-0.6B | 22.3 ms | 23.8 ms | 24.4 ms | 31.4 ms | 46.6/s | 1.6 GB |
| Laya (421M, 10,000 passes) | 38.6 ms | 49.3 ms | 62.7 ms | 182.6 ms | 24.9/s | 2.43 GB* |
| Kronos + Qwen3-1.7B | 45.8 ms | 47.9 ms | 49.1 ms | 50.5 ms | 23.2/s | 3.8 GB |
| Kronos + Qwen3-4B (fresh runtime) | 98.2 ms | 108.0 ms | 110.7 ms | 112.6 ms | 10.9/s | 8.2 GB |
| Kronos + Qwen3-4B (after earlier runs) | 109.4 ms | 115.8 ms | 122.0 ms | 127.0 ms | 9.9/s | 8.3 GB |

\* PyTorch-allocated memory only (`torch.cuda.max_memory_allocated`), which excludes the CUDA context, so Laya's real footprint is likely 0.3–0.5 GB higher.

The embeddings latency excludes the label embeddings, which are computed once; Kronos reads the labels on every request. Few-shot prompts are longer and add roughly 10–30% to Kronos latency.

Findings:

- **The runtime is fast; the model size sets the speed.** With a model of similar size to Laya (0.6B), Kronos is faster than Laya (22 ms vs 39 ms p50) with a much tighter tail (max 31 ms vs 183 ms). The Gate 1 speed gap came from running a 4B model.
- **1.7B runs at close to Laya's speed** (46 ms vs 39 ms p50), with a steadier tail, and fits in under 4 GB of GPU memory.

### T4 power throttling

Long runs on the Colab T4 slow down because the GPU hits its 70 W power cap:

- Repeated 10,000-pass Kronos 4B runs measured p50 99.2 → 110.1 → 120.8 ms.
- During a 2,000-pass run, clocks fell from 1590 MHz to about 750–810 MHz at ~67 W and 78 °C, with throttle reason `0x4` (power cap).

Short runs in a fresh runtime match the Gate 1 figure (96.9 ms). Treat Colab T4 latency as varying by roughly 10–25% between runs.

## 3. Choice-order sensitivity

Each test case was also run with the labels in reversed order (`--order-check`). A "flip" is a case whose answer changed.

| Model | Style | Flips |
|---|---|---|
| 0.6B | defined | 20 / 182 |
| 0.6B | few-shot | 55 / 182 (100% on few-shot sentiment) |
| 1.7B | defined | 20 / 182 (11.0%) |
| 1.7B | few-shot | 25 / 182 (13.7%) |
| 4B | defined | 2 / 182 (1.1%) |
| 4B | few-shot | 4 / 182 (2.2%) |

**Order sensitivity shrinks with model size.** 4B is close to stable; at 1.7B about 1 in 10 answers depends on the order the labels are listed. The worst single cases were 1.7B defined agent_gating (8 of 24 flipped; accuracy 100% → 66.7% reversed) and 1.7B few-shot trade_risk (8 of 18 flipped; 88.9% → 44.4%).

### Fix tested: averaging both orders (1.7B)

Each case was scored in both orders through the HTTP API, and the two probability sets were averaged. The few-shot prompts here used the 2 `examples` in each dataset file, so the one-order numbers differ slightly from `kronos-eval`.

| Style | Original order | Reversed order | Averaged | Flips | p50, one order | p50, averaged |
|---|---|---|---|---|---|---|
| defined | 89.6% | 83.0% | 90.7% | 20 / 182 | 50.0 ms | 99.8 ms |
| few-shot | 91.2% | 92.3% | **94.0%** | 18 / 182 | 55.3 ms | 111.2 ms |

Latencies include HTTP overhead.

Findings:

- **Averaging removes order dependence and improves accuracy** (few-shot 91.2% → 94.0%).
- **But it doubles latency** to about the same as 4B, which is more accurate (98.4% few-shot). Averaged 1.7B is only worth it where 4B does not fit in memory.
- Reversed order scored higher than original in few-shot, so this is a position bias, not random noise.

Idea to test: score both orders in a single batched forward pass. At batch size 1 the GPU is underused, so two orders together may cost well under 2×.

## 4. Embeddings baseline

Each input is matched to the closest label using `BAAI/bge-small-en-v1.5` embeddings.

- **Zero-shot:** label = `"label: definition"` (the same information Kronos gets).
- **Few-shot:** each label vector also includes the calibration examples for that label.

| Dataset | Embeddings zero-shot | Embeddings few-shot | Laya | Kronos 4B defined |
|---|---|---|---|---|
| agent_gating | 66.7% | 70.8% | 62.5% | 95.8% |
| customer_support_routing | 90.6% | 100.0% | 96.9% | 100% |
| guardrail | 54.2% | 50.0%† | 66.7% | 95.8% |
| guardrail_yesno | 37.5% | 62.5% | 75.0% | 95.8% |
| sanity | 41.7% | 41.7% | 83.3% | 100% |
| sentiment | 58.3% | 66.7% | 95.8% | 95.8% |
| tool_selection | 79.2% | 83.3% | 87.5% | 100% |
| trade_risk | 61.1% | 33.3% | 66.7% | 66.7% |
| **Overall** | **63.7%** | **67.6%** | **80.2%** | **94.5%** |

† Run before the dataset fix below, so it used only 3 of guardrail's 6 calibration examples.

**Finding:** embeddings handle topic routing well (100% on support routing with a few examples) but fail on decisions that need judgment, such as guardrails, gating and risk (33–71%). Kronos's advantage is on judgment decisions, not topic routing.

## 5. Confidence calibration (4B, defined)

Kronos on the 182 test cases, grouped by reported confidence:

| Confidence | Cases | Accuracy | Avg confidence |
|---|---|---|---|
| 0.90–0.99 | 2 | 50.0% | 92.9% |
| 0.99–1.00 | 180 | 95.0% | 100.0% |

- Accuracy: **94.5%**
- Expected calibration error (ECE): **5.4 points**

**Finding:** Kronos is overconfident. Almost every answer, including the wrong ones, is reported at about 100% confidence, because the label logits are far apart (e.g. 37.5 vs 14.4). An earlier 0.6B probe showed the same pattern: wrong answers averaged higher confidence (0.88) than right ones (0.85). **The confidence score cannot yet be used to send uncertain cases to a human.** Next step: temperature scaling fitted on the calibration split.

## 6. Out-of-distribution inputs (4B)

Eight clearly off-topic inputs (weather, a poem request, random characters, arithmetic, etc.) were sent to the support-routing decision.

| Setup | Result |
|---|---|
| Original labels only | Every input forced into a real category, at 61–100% confidence (e.g. a poem request → `sales`, 100%) |
| With an added `none` label | **8/8** caught as `none` |
| Routing accuracy on real tickets with `none` added | **100%** (no real tickets lost to `none`) |

In-distribution routing confidence: median 100%, lowest 99.4%.

**Finding:** always include a `none` label, defined as "the request does not fit any of the other categories". These inputs were obviously off-topic; near-topic inputs still need testing.

## 7. Closely related labels (4B)

Three label pairs with close definitions (`refund` / `dispute`, `escalate` / `review`, `cancel` / `pause`), 4 cases each: **12/12 correct**, all at ~100% confidence.

**Limitation:** the cases were written to be clear-cut, so this shows Kronos separates close definitions, but it is not a test of genuinely ambiguous inputs.

## Fixes made during these runs

- **Dataset:** three calibration cases in `guardrail.json` used the label `Safe` instead of `safe`, which stopped `kronos-eval` at that dataset. All labels are now lowercase. Test-split results for Kronos and Laya were not affected.
- **BF16 on the T4:** fails with `CUDA_ERROR_NOT_FOUND: named symbol not found`. Use `KRONOS_DTYPE=f16`.
- **Candle 0.11 + CUDA 13:** `candle-kernels/src/compatibility.cuh` defines `__hmax_nan` / `__hmin_nan`, which CUDA 13's `cuda_fp16.hpp` now also defines. Renaming Candle's two definitions fixes the build. Set `CUDA_COMPUTE_CAP=75` for the T4.
- **`main.rs`:** the IPC error log needs `tracing::error!` (full path) to compile.

## Which model to use

| Mode | Setup | Accuracy | p50 | GPU memory | Caveat |
|---|---|---|---|---|---|
| **Default** | 4B few-shot | 98.4% | ~110 ms | 8.3 GB | needs an 8 GB GPU |
| **Low memory** | 1.7B few-shot, both orders averaged | 94.0% | ~111 ms | 3.8 GB | twice the compute of one order |
| **Fast** | 1.7B, one order | 89.6–91.2% | ~46–55 ms | 3.8 GB | ~1 in 10 answers depends on label order |
| Not recommended | 0.6B | ~48% | 22 ms | 1.6 GB | does not read the input reliably |

## Summary

| Claim | Status |
|---|---|
| Beats embeddings and Laya on judgment decisions without training | ✅ Supported (4B: 94.5–98.4%; 1.7B: ~90%; Laya 80.2%; embeddings 63.7–67.6%) |
| Runtime speed competitive with Laya | ✅ At similar model size (0.6B: 22 ms vs 39 ms), ⚠️ near but not faster at usable accuracy (1.7B: 46 ms) |
| Answers independent of label order | ✅ 4B (1–2% flips), ❌ 1.7B (11–14% flips) unless both orders are averaged |
| Rejects off-topic input when a `none` label is included | ✅ Supported (8/8, no accuracy loss) |
| Confidence score usable for human review | ❌ Not yet (overconfident; calibration needed) |
| Safe for untrusted input | ❌ No (prompt-injectable; see Gate 1) |

## Next steps

1. Batch both label orders into one forward pass and measure the real cost of order averaging.
2. Temperature scaling on the calibration split; re-measure ECE.
3. Near-topic out-of-distribution and genuinely ambiguous test cases.
4. Larger test sets (100+ cases) for the most important datasets.
5. Comparison with llama.cpp running the same models, and 4-bit quantization of 4B (to fit a 4 GB GPU).
6. Re-run the embeddings few-shot baseline with the fixed `guardrail.json`.
