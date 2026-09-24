use anyhow::{bail, Result};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ResolvedChoice {
    pub choice_text: String,
    pub token_id: u32,
}

pub struct SchemaMapper;

impl SchemaMapper {
    /// Maps a list of string choices strictly to single token IDs in the vocabulary.
    pub fn resolve_choices(
        tokenizer: &Tokenizer,
        choices: &[String],
    ) -> Result<Vec<ResolvedChoice>> {
        let mut resolved = Vec::with_capacity(choices.len());

        for choice in choices {
            let choice_str = choice.as_str();

            // 1. Try exact tokenization
            let mut encoding = tokenizer
                .encode(choice_str, false)
                .map_err(|e| anyhow::anyhow!("Tokenization error for choice '{choice}': {e}"))?;

            // 2. Fallback: Try with leading space if exact match produced != 1 token (common in BPE)
            if encoding.get_ids().len() != 1 {
                let leading_space_choice = format!(" {choice_str}");
                if let Ok(space_encoding) = tokenizer.encode(leading_space_choice.as_str(), false) {
                    if space_encoding.get_ids().len() == 1 {
                        encoding = space_encoding;
                    }
                }
            }

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
