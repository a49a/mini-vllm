//! Hugging Face–compatible `config.json` parsing and validation.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

/// `bos`/`eos` fields may be a single id or a list of ids in HF configs.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIdField {
    One(i64),
    Many(Vec<i64>),
}

impl TokenIdField {
    fn to_ids(&self) -> Result<Vec<u32>> {
        match self {
            TokenIdField::One(v) => vec![*v],
            TokenIdField::Many(vs) => vs.clone(),
        }
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| Error::Config("token id must fit u32".into())))
        .collect()
    }
}

#[derive(Debug, Deserialize)]
struct RawModelConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default)]
    bos_token_id: Option<TokenIdField>,
    #[serde(default)]
    eos_token_id: Option<TokenIdField>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

/// Parsed and validated model configuration.
///
/// Tensor dimensions are always read from the config — never hard-coded.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    /// All ids that terminate generation (HF may list several).
    pub eos_token_ids: Vec<u32>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub architectures: Vec<String>,
}

impl ModelConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawModelConfig = serde_json::from_str(json)?;
        if raw.model_type.as_deref().is_some_and(|v| v != "qwen2") {
            return Err(Error::Config("only model_type=qwen2 is supported".into()));
        }
        if raw.quantization_config.is_some() {
            return Err(Error::Config(
                "quantized model configurations are not supported".into(),
            ));
        }
        if raw.head_dim.is_some_and(|dim| {
            raw.num_attention_heads == 0 || dim != raw.hidden_size / raw.num_attention_heads
        }) {
            return Err(Error::Config(
                "custom attention head_dim is not supported".into(),
            ));
        }
        if raw.rope_scaling.is_some() || raw.use_sliding_window {
            return Err(Error::Config(
                "RoPE scaling and sliding-window attention are not implemented".into(),
            ));
        }
        if raw.hidden_act.as_deref().is_some_and(|v| v != "silu") {
            return Err(Error::Config("only hidden_act=silu is supported".into()));
        }
        let cfg = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            bos_token_id: raw
                .bos_token_id
                .as_ref()
                .map(|b| b.to_ids())
                .transpose()?
                .and_then(|ids| ids.first().copied()),
            eos_token_ids: raw
                .eos_token_id
                .as_ref()
                .map(|e| e.to_ids())
                .transpose()?
                .unwrap_or_default(),
            tie_word_embeddings: raw.tie_word_embeddings,
            torch_dtype: raw.torch_dtype,
            architectures: raw.architectures.unwrap_or_default(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load and validate `<model_dir>/config.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let json = std::fs::read_to_string(&path)?;
        Self::from_json(&json).map_err(|e| match e {
            Error::Json(_) => Error::Config(format!("failed to parse {}", path.display())),
            other => other,
        })
    }

    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA group size (heads per KV head).
    pub fn heads_per_kv_head(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Fail early on configurations this runtime does not support.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
        {
            return Err(Error::Config("dimension fields must be > 0".into()));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::Config(format!(
                "hidden_size({}) not divisible by num_attention_heads({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::Config(format!(
                "num_attention_heads({}) not divisible by num_key_value_heads({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.num_key_value_heads > self.num_attention_heads {
            return Err(Error::Config(
                "num_key_value_heads must be <= num_attention_heads".into(),
            ));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(Error::Config("head_dim must be even for RoPE".into()));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(Error::Config(
                "rms_norm_eps and rope_theta must be finite and > 0".into(),
            ));
        }
        if self
            .eos_token_ids
            .iter()
            .chain(self.bos_token_id.iter())
            .any(|&id| id as usize >= self.vocab_size)
        {
            return Err(Error::Config("special token id outside vocabulary".into()));
        }
        if self.intermediate_size == 0 {
            return Err(Error::Config("intermediate_size must be > 0".into()));
        }
        if self
            .architectures
            .iter()
            .any(|arch| arch != "Qwen2ForCausalLM")
        {
            return Err(Error::Config(
                "only Qwen2ForCausalLM architecture is supported".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN2_05B: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "max_window_layers": 70,
        "model_type": "qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "rope_scaling": null,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "vocab_size": 151936,
        "eos_token_id": 151645,
        "bos_token_id": null
    }"#;

    #[test]
    fn parses_qwen25_config() {
        let cfg = ModelConfig::from_json(QWEN2_05B).unwrap();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.heads_per_kv_head(), 7);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.eos_token_ids, vec![151645]);
        assert_eq!(cfg.torch_dtype.as_deref(), Some("bfloat16"));
    }

    #[test]
    fn validates_head_divisibility() {
        let mut cfg = ModelConfig::from_json(QWEN2_05B).unwrap();
        cfg.num_attention_heads = 15;
        assert!(cfg.validate().is_err());
        cfg.num_attention_heads = 14;
        cfg.num_key_value_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_kv_heads_and_invalid_rotary_dimensions() {
        let mut cfg = ModelConfig::from_json(QWEN2_05B).unwrap();
        cfg.num_key_value_heads = 0;
        assert!(cfg.validate().is_err());
        cfg.num_key_value_heads = 2;
        cfg.hidden_size = 14 * 63;
        assert!(cfg.validate().is_err());
        cfg.hidden_size = 896;
        cfg.rope_theta = f64::NAN;
        assert!(cfg.validate().is_err());
        assert!(ModelConfig::from_json(&QWEN2_05B.replace("151645", "-1")).is_err());
    }

    #[test]
    fn rejects_unsupported_model_semantics() {
        for (key, value) in [
            ("model_type", serde_json::json!("qwen3")),
            ("architectures", serde_json::json!(["Qwen2MoeForCausalLM"])),
            (
                "rope_scaling",
                serde_json::json!({"type":"linear","factor":2}),
            ),
            ("use_sliding_window", serde_json::json!(true)),
            ("hidden_act", serde_json::json!("gelu")),
        ] {
            let mut config: serde_json::Value = serde_json::from_str(QWEN2_05B).unwrap();
            config[key] = value;
            assert!(
                ModelConfig::from_json(&config.to_string()).is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn eos_can_be_a_list() {
        let json = QWEN2_05B.replace(
            "\"eos_token_id\": 151645",
            "\"eos_token_id\": [151645, 151643]",
        );
        let cfg = ModelConfig::from_json(&json).unwrap();
        assert_eq!(cfg.eos_token_ids, vec![151645, 151643]);
    }
}
