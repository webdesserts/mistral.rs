mod chat_template;
mod content;
mod gguf_tokenizer;
use strum::EnumString;

use anyhow::{Context, Result};
pub(crate) use chat_template::get_gguf_chat_template;
pub(crate) use content::Content;
pub(crate) use gguf_tokenizer::{convert_gguf_to_hf_tokenizer, GgufTokenizerConversion};
use std::str::FromStr;

pub const GGUF_MULTI_FILE_DELIMITER: &str = " ";

#[derive(Debug, EnumString, Clone, Copy, strum::Display)]
#[strum(serialize_all = "lowercase")]
pub enum GGUFArchitecture {
    Llama,
    Mpt,
    Gptneox,
    Gptj,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    Phi2,
    Phi3,
    Starcoder2,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    #[strum(serialize = "minimax-m2")]
    MinimaxM2,
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
impl GGUFArchitecture {
    pub fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        Self::from_str(&value.as_ref().to_ascii_lowercase())
            .with_context(|| format!("Unknown GGUF architecture `{value}`"))
            .map_err(anyhow::Error::msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_m2_architecture_parsing() {
        // Test the exact GGUF architecture string (with hyphen)
        let arch = GGUFArchitecture::from_value("minimax-m2").unwrap();
        assert!(matches!(arch, GGUFArchitecture::MinimaxM2));

        // Test case insensitivity
        let arch = GGUFArchitecture::from_value("MINIMAX-M2").unwrap();
        assert!(matches!(arch, GGUFArchitecture::MinimaxM2));

        // Test display round-trip
        let arch = GGUFArchitecture::MinimaxM2;
        assert_eq!(arch.to_string(), "minimax-m2");
    }
}
