//! Maps a choice schema to single, distinct vocabulary token IDs.

use anyhow::{anyhow, bail, Result};
use std::collections::{HashMap, HashSet};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedChoice {
    pub choice: String,
    pub token_id: u32,
}

/// Which spelling to try first. The same spelling is used for every choice so
/// that no choice gets an unfair head start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpacePolicy {
    PreferNoSpace,
    PreferLeadingSpace,
}

pub struct SchemaMapper;

impl SchemaMapper {
    pub fn resolve_choices(tokenizer: &Tokenizer, choices: &[String], policy: SpacePolicy) -> Result<Vec<ResolvedChoice>> {
        Self::resolve_with(
            |text: &str| {
                tokenizer
                    .encode(text, false)
                    .map(|e| e.get_ids().to_vec())
                    .map_err(|e| anyhow!("tokenization failed for {text:?}: {e}"))
            },
            choices,
            policy,
        )
    }

    /// Tokenizer-agnostic core, so it can be unit-tested with a fake encoder.
    pub fn resolve_with<F>(encode: F, choices: &[String], policy: SpacePolicy) -> Result<Vec<ResolvedChoice>>
    where
        F: Fn(&str) -> Result<Vec<u32>>,
    {
        if choices.is_empty() {
            bail!("no choices provided");
        }
        let mut seen = HashSet::new();
        for c in choices {
            if c.trim().is_empty() {
                bail!("choices must not be empty or whitespace");
            }
            if !seen.insert(c.as_str()) {
                bail!("duplicate choice '{c}'");
            }
        }

        let order = match policy {
            SpacePolicy::PreferNoSpace => [false, true],
            SpacePolicy::PreferLeadingSpace => [true, false],
        };

        let mut failures = Vec::with_capacity(2);
        for leading_space in order {
            match Self::try_variant(&encode, choices, leading_space) {
                Ok(resolved) => return Ok(resolved),
                Err(e) => failures.push(format!(
                    "{}: {e}",
                    if leading_space { "with leading space" } else { "as written" }
                )),
            }
        }
        bail!(
            "each choice must map to exactly one distinct token, spelled consistently. {}",
            failures.join("; ")
        )
    }

    fn try_variant<F>(encode: &F, choices: &[String], leading_space: bool) -> Result<Vec<ResolvedChoice>>
    where
        F: Fn(&str) -> Result<Vec<u32>>,
    {
        let mut resolved = Vec::with_capacity(choices.len());
        let mut by_id: HashMap<u32, &str> = HashMap::new();
        let mut multi_token = Vec::new();

        for choice in choices {
            let text = if leading_space {
                format!(" {choice}")
            } else {
                choice.clone()
            };
            let ids = encode(&text)?;
            if ids.len() != 1 {
                multi_token.push(format!("'{choice}' is {} tokens", ids.len()));
                continue;
            }
            let id = ids[0];
            if let Some(other) = by_id.insert(id, choice.as_str()) {
                bail!("'{other}' and '{choice}' share token id {id}");
            }
            resolved.push(ResolvedChoice {
                choice: choice.clone(),
                token_id: id,
            });
        }

        if !multi_token.is_empty() {
            bail!("{}", multi_token.join(", "));
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(vocab: &'static [(&'static str, u32)]) -> impl Fn(&str) -> Result<Vec<u32>> {
        move |text: &str| {
            Ok(match vocab.iter().find(|(t, _)| *t == text) {
                Some((_, id)) => vec![*id],
                None => vec![900, 901], // anything unknown is "two tokens"
            })
        }
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn prefers_bare_spelling() {
        let enc = fake(&[("yes", 1), ("no", 2), (" yes", 3), (" no", 4)]);
        let r = SchemaMapper::resolve_with(enc, &s(&["yes", "no"]), SpacePolicy::PreferNoSpace).unwrap();
        assert_eq!(r.iter().map(|c| c.token_id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn falls_back_consistently() {
        // "no" has no bare single token, so *both* choices switch to spaced.
        let enc = fake(&[("yes", 1), (" yes", 3), (" no", 4)]);
        let r = SchemaMapper::resolve_with(enc, &s(&["yes", "no"]), SpacePolicy::PreferNoSpace).unwrap();
        assert_eq!(r.iter().map(|c| c.token_id).collect::<Vec<_>>(), vec![3, 4]);
    }

    #[test]
    fn rejects_multi_token_choice() {
        let enc = fake(&[("yes", 1), (" yes", 3)]);
        let err = SchemaMapper::resolve_with(enc, &s(&["yes", "SPAWN_HEALTH"]), SpacePolicy::PreferNoSpace)
            .unwrap_err()
            .to_string();
        assert!(err.contains("SPAWN_HEALTH"));
    }

    #[test]
    fn rejects_token_collision() {
        let enc = fake(&[("a", 1), ("b", 1), (" a", 2), (" b", 2)]);
        assert!(SchemaMapper::resolve_with(enc, &s(&["a", "b"]), SpacePolicy::PreferNoSpace).is_err());
    }

    #[test]
    fn rejects_duplicates_and_blanks() {
        let enc = fake(&[("a", 1)]);
        assert!(SchemaMapper::resolve_with(&enc, &s(&["a", "a"]), SpacePolicy::PreferNoSpace).is_err());
        assert!(SchemaMapper::resolve_with(&enc, &s(&["a", " "]), SpacePolicy::PreferNoSpace).is_err());
    }
}
