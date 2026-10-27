# Kronos evaluation: customer_support_routing

| Setting | Value |
|---|---|
| Dataset | `eval/datasets/customer_support_routing.json` |
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
| Cases | 32 |
| Errors (no decision) | 0 |
| Accuracy | 50.0% |
| Macro F1 | 0.458 |
| Mean confidence, correct | 0.653 |
| Mean confidence, wrong | 0.582 |
| ECE (10 bins) | 0.173 |
| Latency p50 / p95 / p99 / max | 1043.0 / 1182.6 / 1192.4 / 1192.4 ms |

## Per-class results

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| billing | 0.000 | 0.000 | 0.000 | 8 |
| technical | 1.000 | 0.500 | 0.667 | 8 |
| account | 0.333 | 1.000 | 0.500 | 8 |
| sales | 1.000 | 0.500 | 0.667 | 8 |

## Confusion matrix (rows = correct label, columns = Kronos)

| | billing | technical | account | sales | error |
|---|---|---|---|---|---|
| **billing** | 0 | 0 | 8 | 0 | 0 |
| **technical** | 0 | 4 | 4 | 0 | 0 |
| **account** | 0 | 0 | 8 | 0 | 0 |
| **sales** | 0 | 0 | 4 | 4 | 0 |

## Calibration

| Confidence bin | Cases | Avg confidence | Accuracy |
|---|---|---|---|
| 0.4–0.5 | 3 | 0.469 | 0.0% |
| 0.5–0.6 | 11 | 0.556 | 36.4% |
| 0.6–0.7 | 10 | 0.631 | 60.0% |
| 0.7–0.8 | 7 | 0.732 | 85.7% |
| 0.8–0.9 | 1 | 0.808 | 0.0% |

## Abstention thresholds

Abstain when confidence is below the threshold.

| Threshold | Coverage | Accuracy when answering | Wrong answers caught | Correct answers lost |
|---|---|---|---|---|
| 0.00 | 100.0% | 50.0% | 0.0% | 0.0% |
| 0.50 | 90.6% | 55.2% | 18.8% | 0.0% |
| 0.60 | 56.2% | 66.7% | 62.5% | 25.0% |
| 0.70 | 25.0% | 75.0% | 87.5% | 62.5% |
| 0.80 | 3.1% | 0.0% | 93.8% | 100.0% |
| 0.90 | 0.0% | 0.0% | 100.0% | 100.0% |
| 0.95 | 0.0% | 0.0% | 100.0% | 100.0% |

## Choice-order check

| Metric | Value |
|---|---|
| Accuracy, original order | 50.0% |
| Accuracy, reversed order | 50.0% |
| Answers that changed | 6 of 32 (18.8%) |

- `customer_support_routing-008` (correct: technical): account → technical

- `customer_support_routing-011` (correct: technical): account → technical

- `customer_support_routing-017` (correct: account): account → sales

- `customer_support_routing-019` (correct: sales): sales → account

- `customer_support_routing-021` (correct: sales): sales → account

- `customer_support_routing-023` (correct: sales): account → sales

## Wrong answers (highest confidence first)

| Case | Correct | Kronos | Confidence | Tags |
|---|---|---|---|---|
| customer_support_routing-028 | technical | account | 0.808 |  |
| customer_support_routing-005 | billing | account | 0.727 | negative-tone |
| customer_support_routing-026 | billing | account | 0.656 |  |
| customer_support_routing-027 | technical | account | 0.636 |  |
| customer_support_routing-025 | billing | account | 0.612 |  |
| customer_support_routing-006 | billing | account | 0.611 | refund |
| customer_support_routing-008 | technical | account | 0.597 |  |
| customer_support_routing-003 | billing | account | 0.582 |  |
| customer_support_routing-032 | sales | account | 0.550 |  |
| customer_support_routing-001 | billing | account | 0.545 | refund |
| customer_support_routing-023 | sales | account | 0.540 |  |
| customer_support_routing-011 | technical | account | 0.530 |  |
| customer_support_routing-004 | billing | account | 0.514 |  |
| customer_support_routing-024 | sales | account | 0.496 |  |
| customer_support_routing-020 | sales | account | 0.457 |  |
| customer_support_routing-002 | billing | account | 0.453 |  |

---
Confidence is relative to the allowed choices only, so it is not a calibrated probability of being right. Check the calibration table before using it as a threshold.
