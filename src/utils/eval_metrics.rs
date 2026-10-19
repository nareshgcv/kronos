//! Classification metrics for the Kronos evaluation harness.
//!
//! Pure functions with no model involved: accuracy, per-class precision /
//! recall / F1, macro F1, confusion matrix, calibration (ECE) and a
//! confidence-threshold (abstention) sweep.

/// One evaluated case. `predicted` is `None` when Kronos returned an error.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub gold: String,
    pub predicted: Option<String>,
    pub confidence: f64,
}

impl Outcome {
    pub fn is_correct(&self) -> bool {
        self.predicted.as_deref() == Some(self.gold.as_str())
    }
}

/// Rows are gold labels, columns are predicted labels (same order as `labels`).
/// `errors[i]` counts gold-label-`i` cases that produced no valid prediction.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub labels: Vec<String>,
    pub counts: Vec<Vec<usize>>,
    pub errors: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ClassStats {
    pub label: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Clone)]
pub struct CalibrationBin {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub avg_confidence: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct ThresholdRow {
    pub threshold: f64,
    /// Share of all cases answered (confidence >= threshold).
    pub coverage: f64,
    /// Accuracy on the answered cases only.
    pub selective_accuracy: f64,
    /// Share of wrong answers that the threshold would have abstained on.
    pub wrong_caught: f64,
    /// Share of correct answers that the threshold would have thrown away.
    pub correct_lost: f64,
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

pub fn accuracy(outcomes: &[Outcome]) -> f64 {
    ratio(outcomes.iter().filter(|o| o.is_correct()).count(), outcomes.len())
}

pub fn confusion_matrix(labels: &[String], outcomes: &[Outcome]) -> ConfusionMatrix {
    let n = labels.len();
    let mut counts = vec![vec![0usize; n]; n];
    let mut errors = vec![0usize; n];
    let index = |l: &str| labels.iter().position(|x| x == l);

    for o in outcomes {
        let Some(g) = index(o.gold.as_str()) else { continue };
        match o.predicted.as_deref().and_then(index) {
            Some(p) => counts[g][p] += 1,
            None => errors[g] += 1,
        }
    }
    ConfusionMatrix {
        labels: labels.to_vec(),
        counts,
        errors,
    }
}

pub fn per_class(cm: &ConfusionMatrix) -> Vec<ClassStats> {
    let n = cm.labels.len();
    (0..n)
        .map(|i| {
            let tp = cm.counts[i][i];
            let predicted_total: usize = (0..n).map(|r| cm.counts[r][i]).sum();
            let support = cm.counts[i].iter().sum::<usize>() + cm.errors[i];
            let precision = ratio(tp, predicted_total);
            let recall = ratio(tp, support);
            let f1 = if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            };
            ClassStats {
                label: cm.labels[i].clone(),
                precision,
                recall,
                f1,
                support,
            }
        })
        .collect()
}

/// Unweighted mean F1 over classes that appear in the gold data.
pub fn macro_f1(stats: &[ClassStats]) -> f64 {
    let present: Vec<&ClassStats> = stats.iter().filter(|s| s.support > 0).collect();
    if present.is_empty() {
        return 0.0;
    }
    present.iter().map(|s| s.f1).sum::<f64>() / present.len() as f64
}

/// Reliability bins and expected calibration error over answered cases.
pub fn calibration(outcomes: &[Outcome], bins: usize) -> (Vec<CalibrationBin>, f64) {
    let bins = bins.max(1);
    let answered: Vec<&Outcome> = outcomes.iter().filter(|o| o.predicted.is_some()).collect();
    let mut sum_conf = vec![0.0f64; bins];
    let mut correct = vec![0usize; bins];
    let mut count = vec![0usize; bins];

    for o in &answered {
        let c = o.confidence.clamp(0.0, 1.0);
        let b = ((c * bins as f64) as usize).min(bins - 1);
        sum_conf[b] += c;
        count[b] += 1;
        if o.is_correct() {
            correct[b] += 1;
        }
    }

    let total = answered.len();
    let mut ece = 0.0;
    let table: Vec<CalibrationBin> = (0..bins)
        .map(|b| {
            let avg_confidence = if count[b] == 0 { 0.0 } else { sum_conf[b] / count[b] as f64 };
            let acc = ratio(correct[b], count[b]);
            if count[b] > 0 {
                ece += ratio(count[b], total) * (acc - avg_confidence).abs();
            }
            CalibrationBin {
                lower: b as f64 / bins as f64,
                upper: (b + 1) as f64 / bins as f64,
                count: count[b],
                avg_confidence,
                accuracy: acc,
            }
        })
        .collect();
    (table, ece)
}

pub fn threshold_sweep(outcomes: &[Outcome], thresholds: &[f64]) -> Vec<ThresholdRow> {
    let total = outcomes.len();
    let answered: Vec<&Outcome> = outcomes.iter().filter(|o| o.predicted.is_some()).collect();
    let wrong_total = answered.iter().filter(|o| !o.is_correct()).count();
    let correct_total = answered.len() - wrong_total;

    thresholds
        .iter()
        .map(|&t| {
            let kept: Vec<&&Outcome> = answered.iter().filter(|o| o.confidence >= t).collect();
            let kept_correct = kept.iter().filter(|o| o.is_correct()).count();
            let wrong_caught = answered
                .iter()
                .filter(|o| !o.is_correct() && o.confidence < t)
                .count();
            let correct_lost = answered
                .iter()
                .filter(|o| o.is_correct() && o.confidence < t)
                .count();
            ThresholdRow {
                threshold: t,
                coverage: ratio(kept.len(), total),
                selective_accuracy: ratio(kept_correct, kept.len()),
                wrong_caught: ratio(wrong_caught, wrong_total),
                correct_lost: ratio(correct_lost, correct_total),
            }
        })
        .collect()
}

/// Mean confidence on correct and on wrong answers (`None` if there are none).
pub fn mean_confidence_split(outcomes: &[Outcome]) -> (Option<f64>, Option<f64>) {
    let mean = |it: Vec<f64>| {
        if it.is_empty() {
            None
        } else {
            Some(it.iter().sum::<f64>() / it.len() as f64)
        }
    };
    let answered = outcomes.iter().filter(|o| o.predicted.is_some());
    let (right, wrong): (Vec<&Outcome>, Vec<&Outcome>) = answered.partition(|o| o.is_correct());
    (
        mean(right.iter().map(|o| o.confidence).collect()),
        mean(wrong.iter().map(|o| o.confidence).collect()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(gold: &str, pred: Option<&str>, conf: f64) -> Outcome {
        Outcome {
            gold: gold.into(),
            predicted: pred.map(Into::into),
            confidence: conf,
        }
    }

    fn labels() -> Vec<String> {
        vec!["a".into(), "b".into()]
    }

    fn sample() -> Vec<Outcome> {
        vec![
            o("a", Some("a"), 0.9),
            o("a", Some("b"), 0.8),
            o("b", Some("b"), 0.7),
            o("b", None, 0.0),
        ]
    }

    #[test]
    fn accuracy_counts_errors_as_wrong() {
        assert!((accuracy(&sample()) - 0.5).abs() < 1e-12);
        assert_eq!(accuracy(&[]), 0.0);
    }

    #[test]
    fn confusion_matrix_rows_are_gold() {
        let cm = confusion_matrix(&labels(), &sample());
        assert_eq!(cm.counts, vec![vec![1, 1], vec![0, 1]]);
        assert_eq!(cm.errors, vec![0, 1]);
    }

    #[test]
    fn per_class_and_macro_f1() {
        let stats = per_class(&confusion_matrix(&labels(), &sample()));
        // a: tp=1, predicted=1, support=2 -> p=1.0 r=0.5 f1=2/3
        assert!((stats[0].precision - 1.0).abs() < 1e-12);
        assert!((stats[0].recall - 0.5).abs() < 1e-12);
        // b: tp=1, predicted=2, support=2 (one error) -> p=0.5 r=0.5 f1=0.5
        assert!((stats[1].f1 - 0.5).abs() < 1e-12);
        assert_eq!(stats[1].support, 2);
        let expected = (2.0 / 3.0 + 0.5) / 2.0;
        assert!((macro_f1(&stats) - expected).abs() < 1e-12);
    }

    #[test]
    fn perfectly_calibrated_has_zero_ece() {
        let data = vec![o("a", Some("a"), 1.0), o("b", Some("b"), 1.0)];
        let (_, ece) = calibration(&data, 10);
        assert!(ece.abs() < 1e-12);
    }

    #[test]
    fn overconfident_wrong_answers_raise_ece() {
        let data = vec![o("a", Some("b"), 0.95), o("b", Some("a"), 0.95)];
        let (bins, ece) = calibration(&data, 10);
        assert!((ece - 0.95).abs() < 1e-12);
        assert_eq!(bins[9].count, 2);
    }

    #[test]
    fn threshold_sweep_tracks_abstention() {
        let rows = threshold_sweep(&sample(), &[0.0, 0.85]);
        // t=0: all three answered cases kept (coverage 3/4), 2 of 3 correct.
        assert!((rows[0].coverage - 0.75).abs() < 1e-12);
        assert!((rows[0].selective_accuracy - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(rows[0].wrong_caught, 0.0);
        // t=0.85: only the 0.9 case is kept; the single wrong answer (0.8) is caught.
        assert!((rows[1].coverage - 0.25).abs() < 1e-12);
        assert!((rows[1].selective_accuracy - 1.0).abs() < 1e-12);
        assert!((rows[1].wrong_caught - 1.0).abs() < 1e-12);
        assert!((rows[1].correct_lost - 0.5).abs() < 1e-12);
    }

    #[test]
    fn confidence_split() {
        let (right, wrong) = mean_confidence_split(&sample());
        assert!((right.unwrap() - 0.8).abs() < 1e-12);
        assert!((wrong.unwrap() - 0.8).abs() < 1e-12);
    }
}
