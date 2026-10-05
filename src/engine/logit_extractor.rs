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
