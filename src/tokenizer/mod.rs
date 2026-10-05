//! Tokenizer manager: chat templating, prompt encoding, cached choice resolution.

pub mod schema_mapper;

pub use schema_mapper::{ResolvedChoice, SchemaMapper, SpacePolicy};

use crate::config::PromptFormat;
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use tokenizers::Tokenizer;

const CHOICE_CACHE_CAPACITY: usize = 1024;

type ChoiceCache = HashMap<Vec<String>, Arc<Vec<ResolvedChoice>>>;

pub struct TokenizerManager {
    tokenizer: Tokenizer,
    format: PromptFormat,
    system_prompt: String,
    append_choices: bool,
    choice_cache: Mutex<ChoiceCache>,
}

impl TokenizerManager {
    pub fn load(model_dir: &Path, format: PromptFormat, system_prompt: String, append_choices: bool) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        Ok(Self {
            tokenizer,
            format,
            system_prompt,
            append_choices,
            choice_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Resolves the choice set to single, distinct token IDs. Results are cached
    /// per choice set because decision schemas repeat across requests.
    pub fn resolve_choices(&self, choices: &[String]) -> Result<Arc<Vec<ResolvedChoice>>> {
        if let Some(hit) = self.cache().get(choices) {
            return Ok(Arc::clone(hit));
        }

        for c in choices {
            self.reject_control_markers(c, "choice")?;
        }

        // In chat templates the answer starts right after a newline, so the
        // bare token is what the model emits. In raw prompts it usually follows
        // a word or colon, so the space-prefixed token is the likelier one.
        let policy = match self.format {
            PromptFormat::Raw => SpacePolicy::PreferLeadingSpace,
            _ => SpacePolicy::PreferNoSpace,
        };
        let resolved = Arc::new(SchemaMapper::resolve_choices(&self.tokenizer, choices, policy)?);

        let mut cache = self.cache();
        if cache.len() >= CHOICE_CACHE_CAPACITY {
            cache.clear();
        }
        cache.insert(choices.to_vec(), Arc::clone(&resolved));
        Ok(resolved)
    }

    /// Wraps the prompt in the model's chat template and tokenizes it.
    pub fn encode_prompt(&self, prompt: &str, choices: &[String]) -> Result<Vec<u32>> {
        self.reject_control_markers(prompt, "prompt")?;

        let user = if self.append_choices && self.format != PromptFormat::Raw {
            format!(
                "{prompt}\n\nAllowed answers: {}\nReply with exactly one allowed answer.",
                choices.join(", ")
            )
        } else {
            prompt.to_owned()
        };

        let (text, add_special_tokens) = self.render(&user);
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("prompt encoding failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn render(&self, user: &str) -> (String, bool) {
        let sys = self.system_prompt.trim();
        match self.format {
            PromptFormat::Raw => (user.to_owned(), true),
            PromptFormat::ChatMl => {
                let mut s = String::new();
                if !sys.is_empty() {
                    s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
                }
                s.push_str(&format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"));
                (s, false)
            }
            PromptFormat::Llama3 => {
                let mut s = String::from("<|begin_of_text|>");
                if !sys.is_empty() {
                    s.push_str(&format!(
                        "<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
                    ));
                }
                s.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                ));
                (s, false)
            }
            PromptFormat::Mistral => {
                let body = if sys.is_empty() {
                    user.to_owned()
                } else {
                    format!("{sys}\n\n{user}")
                };
                (format!("<s>[INST] {body} [/INST]"), false)
            }
        }
    }

    /// Blocks template injection: user text must not contain the template's
    /// control tokens, or it could close the user turn and forge an answer.
    fn reject_control_markers(&self, text: &str, what: &str) -> Result<()> {
        let markers: &[&str] = match self.format {
            PromptFormat::Raw => &[],
            PromptFormat::ChatMl => &["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
            PromptFormat::Llama3 => &[
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ],
            PromptFormat::Mistral => &["[INST]", "[/INST]", "<s>", "</s>"],
        };
        if let Some(m) = markers.iter().find(|m| text.contains(**m)) {
            bail!("{what} contains reserved template token {m}");
        }
        Ok(())
    }

    fn cache(&self) -> MutexGuard<'_, ChoiceCache> {
        self.choice_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
