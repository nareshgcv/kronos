//! Candidate logit extraction and temperature-scaled softmax.

use super::ChoiceProbability;
use crate::tokenizer::schema_mapper::ResolvedChoice;
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct Distribution {
    /// Same order as the candidates passed in.
    pub choices: Vec<ChoiceProbability>,
    /// Index of the highest-probability choice (first one wins ties).
    pub best: usize,
    /// Full-vocab probability mass on the candidates at T = 1.
    pub choice_mass: f32,
}

pub struct LogitExtractor;

impl LogitExtractor {
    pub fn compute(logits: &[f32], candidates: &[ResolvedChoice], temperature: f32) -> Result<Distribution> {
        if candidates.is_empty() {
            bail!("no candidate choices");
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            bail!("temperature must be a finite value > 0");
        }

        let mut raw = Vec::with_capacity(candidates.len());
        for c in candidates {
            let Some(&logit) = logits.get(c.token_id as usize) else {
                bail!(
                    "token id {} for '{}' is outside the logits vector (len {})",
                    c.token_id,
                    c.choice,
                    logits.len()
                );
            };
            if !logit.is_finite() {
                bail!("non-finite logit for '{}'", c.choice);
            }
            raw.push(logit);
        }

        let scaled: Vec<f32> = raw.iter().map(|l| l / temperature).collect();
        let probs = softmax(&scaled);

        let choice_mass = (log_sum_exp(&raw) - log_sum_exp(logits)).exp().clamp(0.0, 1.0);

        let mut best = 0;
        for (i, p) in probs.iter().enumerate() {
            if *p > probs[best] {
                best = i;
            }
        }

        let choices = candidates
            .iter()
            .zip(raw.iter().zip(probs.iter()))
            .map(|(c, (&logit, &probability))| ChoiceProbability {
                choice: c.choice.clone(),
                token_id: c.token_id,
                logit,
                probability,
            })
            .collect();

        Ok(Distribution {
            choices,
            best,
            choice_mass,
        })
    }
}

/// Numerically stable softmax (accumulates in f64).
fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = xs.iter().map(|&x| ((x - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| (e / sum) as f32).collect()
}

/// log(sum(exp(x))) over finite values; -inf if there are none.
fn log_sum_exp(xs: &[f32]) -> f32 {
    let max = xs
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::NEG_INFINITY;
    }
    let sum: f64 = xs
        .iter()
        .filter(|x| x.is_finite())
        .map(|&x| ((x - max) as f64).exp())
        .sum();
    max + sum.ln() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rc(choice: &str, token_id: u32) -> ResolvedChoice {
        ResolvedChoice {
            choice: choice.into(),
            token_id,
        }
    }

    #[test]
    fn sums_to_one_and_picks_argmax() {
        let logits = [0.0, 2.0, 1.0, -1.0];
        let d = LogitExtractor::compute(&logits, &[rc("a", 1), rc("b", 2), rc("c", 3)], 1.0).unwrap();
        let sum: f32 = d.choices.iter().map(|c| c.probability).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(d.best, 0);
        assert_eq!(d.choices[0].choice, "a");
    }

    #[test]
    fn lower_temperature_sharpens() {
        let logits = [2.0, 1.0];
        let cands = [rc("a", 0), rc("b", 1)];
        let cold = LogitExtractor::compute(&logits, &cands, 0.5).unwrap();
        let hot = LogitExtractor::compute(&logits, &cands, 2.0).unwrap();
        assert!(cold.choices[0].probability > hot.choices[0].probability);
    }

    #[test]
    fn out_of_range_token_is_an_error() {
        assert!(LogitExtractor::compute(&[0.0, 1.0], &[rc("a", 0), rc("b", 5)], 1.0).is_err());
    }

    #[test]
    fn rejects_bad_temperature() {
        let cands = [rc("a", 0), rc("b", 1)];
        assert!(LogitExtractor::compute(&[0.0, 1.0], &cands, 0.0).is_err());
        assert!(LogitExtractor::compute(&[0.0, 1.0], &cands, f32::NAN).is_err());
    }

    #[test]
    fn choice_mass_tracks_full_vocab() {
        // Model strongly prefers token 0, which isn't a candidate.
        let logits = [10.0, 0.0, 0.0];
        let off = LogitExtractor::compute(&logits, &[rc("a", 1), rc("b", 2)], 1.0).unwrap();
        assert!(off.choice_mass < 0.01);
        let on = LogitExtractor::compute(&logits, &[rc("x", 0), rc("a", 1)], 1.0).unwrap();
        assert!(on.choice_mass > 0.99);
    }
}
