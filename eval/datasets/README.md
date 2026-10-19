# Kronos evaluation harness

Measures decision quality on labeled datasets: accuracy, macro F1, per-class
precision/recall, confusion matrix, calibration (ECE) and abstention thresholds,
plus an optional choice-order check.

## Install into the repo

Copy these paths into the Kronos repository:

```text
src/utils/eval_metrics.rs     metric functions + unit tests
src/bin/kronos-eval.rs        the harness program
eval/datasets/*.json          routing, guardrail, tool_selection
eval/results/                 reports are written here
```

Then make two small edits:

1. In `src/utils/mod.rs`, add `pub mod eval_metrics;`
2. Make sure `Cargo.toml` has `serde` (with the `derive` feature), `serde_json`
   and `anyhow` as dependencies. Kronos already uses them for the HTTP/IPC
   servers and benchmarks, so usually nothing changes.

Cargo picks up `src/bin/kronos-eval.rs` automatically.

## Check the metric math

```bash
cargo test --lib eval_metrics
```

## Run

```bash
cargo run --release --bin kronos-eval -- \
    --model ./models/Qwen3-0.6B \
    --dataset eval/datasets/routing.json \
    --dataset eval/datasets/guardrail.json \
    --dataset eval/datasets/tool_selection.json \
    --style defined --order-check
```

Add `--features cuda` or `--features metal` for GPU runs.

| Option | Meaning |
|---|---|
| `--style bare` | labels only |
| `--style defined` | labels with one-line definitions (default) |
| `--style fewshot` | definitions plus the dataset's examples |
| `--split test` | the 24 test cases per dataset (default) |
| `--split calibration` | the 6 held-out cases, for choosing a threshold |
| `--order-check` | rerun with choices reversed and list answers that changed |
| `--out DIR` | report directory (default `eval/results`) |

Each run writes a Markdown report and a JSONL file of per-case predictions,
named `<unix-time>-<task>-<style>-<split>`.

## Step 4 of the plan: compare prompt styles

Run the same datasets three times with `--style bare`, `--style defined` and
`--style fewshot`, all with `--order-check`.

## Reading the report

- **Errors (no decision)**: usually a label that is not a single token for
  this model. The first error message is shown; rename the label in the
  dataset (and its examples) and rerun.
- **Mean choice mass**: low values mean the model wanted to answer with
  something other than the labels, which points to a prompt problem rather
  than a model-judgment problem.
- **Mean confidence, correct vs wrong**: if wrong answers are as confident as
  right ones, confidence cannot be used to catch mistakes.
- **Abstention thresholds**: pick a threshold on the calibration split, then
  confirm it on the test split. Never tune on the test split.

## Dataset format

```json
{
  "task": "routing",
  "instruction": "One sentence describing the decision.",
  "labels": [{ "label": "billing", "definition": "charges, invoices, refunds" }],
  "examples": [{ "input": "I was charged twice.", "label": "billing" }],
  "cases": [
    { "id": "routing-001", "input": "...", "label": "billing",
      "split": "test", "tags": ["refund"] }
  ]
}
```

`split` is `test` or `calibration`; `tags` are free-form and appear next to
wrong answers in the report. Few-shot examples are kept separate from the
cases so they never leak into the scored set.
