# Kronos evaluation: tool_selection

| Setting | Value |
|---|---|
| Dataset | `eval/datasets/tool_selection.json` |
| Split | test |
| Prompt style | defined |
| Model dir | `./models/Qwen3-0.6B` |
| Model kind | Qwen3 |
| Prompt format | Qwen3 |
| Device | Cpu |
| Git commit | bff89ad |
| Unix time | 1793001316 |

## Summary

| Metric | Value |
|---|---|
| Cases | 24 |
| Errors (no decision) | 0 |
| Accuracy | 50.0% |
| Macro F1 | 0.343 |
| Mean confidence, correct | 0.932 |
| Mean confidence, wrong | 0.747 |
| ECE (10 bins) | 0.339 |
| Latency p50 / p95 / p99 / max | 1006.7 / 1049.1 / 1053.3 / 1053.3 ms |

## Per-class results

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| math | 0.667 | 1.000 | 0.800 | 6 |
| search | 0.000 | 0.000 | 0.000 | 6 |
| database | 0.000 | 0.000 | 0.000 | 6 |
| none | 0.400 | 1.000 | 0.571 | 6 |

## Confusion matrix (rows = correct label, columns = Kronos)

| | math | search | database | none | error |
|---|---|---|---|---|---|
| **math** | 6 | 0 | 0 | 0 | 0 |
| **search** | 1 | 0 | 0 | 5 | 0 |
| **database** | 2 | 0 | 0 | 4 | 0 |
| **none** | 0 | 0 | 0 | 6 | 0 |

## Calibration

| Confidence bin | Cases | Avg confidence | Accuracy |
|---|---|---|---|
| 0.5–0.6 | 2 | 0.571 | 0.0% |
| 0.6–0.7 | 3 | 0.654 | 33.3% |
| 0.7–0.8 | 5 | 0.756 | 0.0% |
| 0.8–0.9 | 4 | 0.882 | 50.0% |
| 0.9–1.0 | 10 | 0.973 | 90.0% |

## Abstention thresholds

Abstain when confidence is below the threshold.

| Threshold | Coverage | Accuracy when answering | Wrong answers caught | Correct answers lost |
|---|---|---|---|---|
| 0.00 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.50 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.60 | 91.7% | 54.5% | 16.7% | 0.0% |
| 0.70 | 79.2% | 57.9% | 33.3% | 8.3% |
| 0.80 | 58.3% | 78.6% | 75.0% | 8.3% |
| 0.90 | 41.7% | 90.0% | 91.7% | 25.0% |
| 0.95 | 33.3% | 87.5% | 91.7% | 41.7% |

## Choice-order check

| Metric | Value |
|---|---|
| Accuracy, original order | 50.0% |
| Accuracy, reversed order | 45.8% |
| Answers that changed | 3 of 24 (12.5%) |

- `tool_selection-005` (correct: math): math → none

- `tool_selection-010` (correct: search): math → none

- `tool_selection-017` (correct: database): math → none

## Wrong answers (highest confidence first)

| Case | Correct | Kronos | Confidence | Tags |
|---|---|---|---|---|
| tool_selection-012 | search | none | 0.973 |  |
| tool_selection-018 | database | none | 0.894 |  |
| tool_selection-007 | search | none | 0.864 |  |
| tool_selection-011 | search | none | 0.798 |  |
| tool_selection-013 | database | none | 0.785 |  |
| tool_selection-015 | database | none | 0.759 |  |
| tool_selection-009 | search | none | 0.718 |  |
| tool_selection-014 | database | math | 0.718 |  |
| tool_selection-016 | database | none | 0.670 |  |
| tool_selection-008 | search | none | 0.641 |  |
| tool_selection-017 | database | math | 0.594 |  |
| tool_selection-010 | search | math | 0.549 |  |

---
Confidence is relative to the allowed choices only, so it is not a calibrated probability of being right. Check the calibration table before using it as a threshold.
