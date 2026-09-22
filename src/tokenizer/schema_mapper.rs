
use anyhow::{bail, Result};
use std::collections::HashMap;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ResolvedChoice {
    pub choice_text: String,
    pub token_id: u32,
}

pub struct SchemaMapper;

impl SchemaMapper {
    /// Maps a list of string choices strictly to single token IDs in the vocabulary
    pub fn resolve_choices(
        tokenizer: &Tokenizer,
        choices: &[String],
    ) -> Result<Vec<ResolvedChoice>> {
        let mut resolved = Vec::with_capacity(choices.len());

        for choice in choices {
            let encoding = tokenizer
                .encode(choice.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization error for choice '{}': {}", choice, e))?;

            let token_ids = encoding.get_ids();
            if token_ids.len() != 1 {
                bail!(
                    "System 1 Schema Violation: Choice '{}' maps to {} tokens. Must map to exactly 1 token ID.",
                    choice,
                    token_ids.len()
                );
            }

            resolved.push(ResolvedChoice {
                choice_text: choice.clone(),
                token_id: token_ids[0],
            });
        }

        Ok(resolved)
    }
}
