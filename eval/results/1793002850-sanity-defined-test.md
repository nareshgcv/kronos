# Kronos evaluation: sanity

| Setting | Value |
|---|---|
| Dataset | `eval/datasets/sanity.json` |
| Split | test |
| Prompt style | defined |
| Model dir | `./models/Qwen3-0.6B` |
| Model kind | Qwen3 |
| Prompt format | Qwen3 |
| Device | Cpu |
| Git commit | bff89ad |
| Unix time | 1793002850 |

## Summary

| Metric | Value |
|---|---|
| Cases | 12 |
| Errors (no decision) | 0 |
| Accuracy | 50.0% |
| Macro F1 | 0.333 |
| Mean confidence, correct | 0.957 |
| Mean confidence, wrong | 0.905 |
| ECE (10 bins) | 0.431 |
| Latency p50 / p95 / p99 / max | 615.5 / 730.3 / 730.3 / 730.3 ms |

## Per-class results

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Yes | 0.500 | 1.000 | 0.667 | 6 |
| No | 0.000 | 0.000 | 0.000 | 6 |

## Confusion matrix (rows = correct label, columns = Kronos)

| | Yes | No | error |
|---|---|---|---|
| **Yes** | 6 | 0 | 0 |
| **No** | 6 | 0 | 0 |

## Calibration

| Confidence bin | Cases | Avg confidence | Accuracy |
|---|---|---|---|
| 0.8–0.9 | 3 | 0.892 | 0.0% |
| 0.9–1.0 | 9 | 0.945 | 66.7% |

## Abstention thresholds

Abstain when confidence is below the threshold.

| Threshold | Coverage | Accuracy when answering | Wrong answers caught | Correct answers lost |
|---|---|---|---|---|
| 0.00 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.50 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.60 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.70 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.80 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.90 | 75.0% | 66.7% | 50.0% | 0.0% |
| 0.95 | 33.3% | 100.0% | 100.0% | 33.3% |

## Wrong answers (highest confidence first)

| Case | Correct | Kronos | Confidence | Tags |
|---|---|---|---|---|
| sanity-006 | No | Yes | 0.948 |  |
| sanity-010 | No | Yes | 0.906 |  |
| sanity-004 | No | Yes | 0.904 |  |
| sanity-002 | No | Yes | 0.899 |  |
| sanity-008 | No | Yes | 0.893 |  |
| sanity-012 | No | Yes | 0.883 |  |

---
Confidence is relative to the allowed choices only, so it is not a calibrated probability of being right. Check the calibration table before using it as a threshold.
