use crate::tokenizer::schema_mapper::ResolvedChoice;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityOutput {
    pub selected_choice: String,
    pub confidence: f32,
    pub probabilities: HashMap<String, f32>,
}

pub struct LogitExtractor;

impl LogitExtractor {
    /// Computes temperature-scaled Softmax probability distribution strictly across schema choice tokens.
    pub fn compute_probabilities(
        raw_logits: &[f32],
        candidates: &[ResolvedChoice],
        temperature: f32,
    ) -> ProbabilityOutput {
        if candidates.is_empty() {
            return ProbabilityOutput {
                selected_choice: String::new(),
                confidence: 0.0,
                probabilities: HashMap::new(),
            };
        }

        let temp = if temperature <= 0.0 { 1.0 } else { temperature };

        let mut max_logit = f32::NEG_INFINITY;
        let mut scaled_logits = Vec::with_capacity(candidates.len());

        // 1. Extract and scale target logits safely with temperature
        for candidate in candidates {
            let token_id = candidate.token_id as usize;
            let raw_logit = raw_logits.get(token_id).copied().unwrap_or(0.0);
            let logit = raw_logit / temp;

            if logit > max_logit {
                max_logit = logit;
            }
            scaled_logits.push(logit);
        }

        // 2. Compute numerically stable Softmax exponents
        let mut sum_exp = 0.0f32;
        let mut exps = Vec::with_capacity(scaled_logits.len());
        for &logit in &scaled_logits {
            let exp_val = (logit - max_logit).exp();
            exps.push(exp_val);
            sum_exp += exp_val;
        }

        // Defensive check against potential 0.0 or NaN division
        if sum_exp <= 0.0 || sum_exp.is_nan() {
            sum_exp = 1.0;
        }

        // 3. Normalize to probabilities & identify ArgMax choice
        let mut probabilities = HashMap::with_capacity(candidates.len());
        let mut best_choice = candidates[0].choice_text.clone();
        let mut max_prob = -1.0f32;

        for (i, candidate) in candidates.iter().enumerate() {
            let prob = exps[i] / sum_exp;
            probabilities.insert(candidate.choice_text.clone(), prob);

            if prob > max_prob {
                max_prob = prob;
                best_choice = candidate.choice_text.clone();
            }
        }

        ProbabilityOutput {
            selected_choice: best_choice,
            confidence: max_prob,
            probabilities,
        }
    }
}        for &logit in &scaled_logits {
            let exp_val = (logit - max_logit).exp();
            exps.push(exp_val);
            sum_exp += exp_val;
        }

        let mut probabilities = HashMap::new();
        let mut best_choice = String::new();
        let mut max_prob = -1.0f32;

        for (i, candidate) in candidates.iter().enumerate() {
            let prob = exps[i] / sum_exp;
            probabilities.insert(candidate.choice_text.clone(), prob);

            if prob > max_prob {
                max_prob = prob;
                best_choice = candidate.choice_text.clone();
            }
        }

        ProbabilityOutput {
            selected_choice: best_choice,
            confidence: max_prob,
            probabilities,
        }
    }
}
