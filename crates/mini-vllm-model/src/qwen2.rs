//! Qwen2-style decoder-only transformer.
//!
//! ```text
//! Token Embedding
//!       ↓
//! N × [ RMSNorm → GQA Self-Attention (+res) → RMSNorm → SwiGLU MLP (+res) ]
//!       ↓
//! RMSNorm
//!       ↓
//! LM Head (tied or untied per config)
//!       ↓
//! Logits
//! ```
//!
//! All dimensions come from `config.json`; nothing is hard-coded.

use candle_core::{DType, Device, Result, Tensor};
use mini_vllm_kv::KvCache;

use crate::attention::Attention;
use crate::config::ModelConfig;
use crate::loader::{get_weight, Tensors};
use crate::mlp::Mlp;
use crate::rms_norm::RmsNorm;
use crate::rope::{RopeCache, SharedRope};

/// One transformer decoder layer.
#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(
        prefix: &str,
        tensors: &Tensors,
        cfg: &ModelConfig,
        rope: SharedRope,
    ) -> crate::error::Result<Self> {
        let rms = |name: &str| -> crate::error::Result<RmsNorm> {
            Ok(RmsNorm::new(
                get_weight(tensors, &format!("{prefix}{name}"))?.clone(),
                cfg.rms_norm_eps,
            ))
        };
        Ok(Self {
            input_layernorm: rms("input_layernorm.weight")?,
            attn: Attention::load(&format!("{prefix}self_attn."), tensors, cfg, rope)?,
            post_attention_layernorm: rms("post_attention_layernorm.weight")?,
            mlp: Mlp::new(
                crate::linear::Linear::new(
                    get_weight(tensors, &format!("{prefix}mlp.gate_proj.weight"))?.clone(),
                    None,
                ),
                crate::linear::Linear::new(
                    get_weight(tensors, &format!("{prefix}mlp.up_proj.weight"))?.clone(),
                    None,
                ),
                crate::linear::Linear::new(
                    get_weight(tensors, &format!("{prefix}mlp.down_proj.weight"))?.clone(),
                    None,
                ),
            ),
        })
    }
}

/// The Qwen2 causal LM.
#[derive(Debug)]
pub struct Qwen2 {
    cfg: ModelConfig,
    dtype: DType,
    device: Device,
    embed_tokens: Tensor,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    /// `None` when word embeddings are tied (lm_head = embed^T).
    lm_head: Option<crate::linear::Linear>,
}

/// A mixed prefill/decode batch: concatenated tokens plus the per-sequence
/// token counts that partition them.
#[derive(Debug, Clone, Default)]
pub struct BatchTokens {
    pub token_ids: Vec<u32>,
    /// Absolute position of every token (same length as `token_ids`).
    pub positions: Vec<u32>,
    /// `token_ids[i]` belongs to sequence `si` where
    /// `sum(seq_lens[..si]) <= i < sum(seq_lens[..si+1])`.
    pub seq_lens: Vec<usize>,
}

impl BatchTokens {
    /// Full-prompt prefill for one sequence.
    pub fn prefill(token_ids: Vec<u32>) -> Self {
        let n = token_ids.len();
        Self {
            positions: (0..n as u32).collect(),
            token_ids,
            seq_lens: vec![n],
        }
    }

    /// Single-token decode step for one sequence.
    pub fn decode_one(token_id: u32, position: u32) -> Self {
        Self {
            token_ids: vec![token_id],
            positions: vec![position],
            seq_lens: vec![1],
        }
    }
}

impl Qwen2 {
    /// Build the model from an already-loaded tensor map (device + dtype set
    /// by the loader).
    pub fn load(
        cfg: ModelConfig,
        tensors: &Tensors,
        dtype: DType,
        device: &Device,
    ) -> crate::error::Result<Self> {
        cfg.validate()?;
        if device.is_cpu() && dtype == DType::BF16 {
            return Err(crate::error::Error::UnsupportedDtype(
                "BF16 on CPU: this Candle backend has no BF16 matmul; use F32 or F16".into(),
            ));
        }
        let rope: SharedRope = std::sync::Arc::new(RopeCache::new(
            cfg.max_position_embeddings,
            cfg.head_dim(),
            cfg.rope_theta,
            dtype,
            device,
        )?);
        let embed_tokens = get_weight(tensors, "model.embed_tokens.weight")?.clone();
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(crate::linear::Linear::new(
                get_weight(tensors, "lm_head.weight")?.clone(),
                None,
            ))
        };
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::load(&format!("model.layers.{i}."), tensors, &cfg, rope.clone()))
            .collect::<crate::error::Result<Vec<_>>>()?;
        let norm = RmsNorm::new(
            get_weight(tensors, "model.norm.weight")?.clone(),
            cfg.rms_norm_eps,
        );
        Ok(Self {
            cfg,
            dtype,
            device: device.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Shared forward path.
    ///
    /// Returns logits for the last token of every sequence
    /// (`[num_seqs, vocab]`), or for *all* tokens (`[total, vocab]`) when
    /// `all_logits` is set (used by the no-cache reference path).
    fn forward_impl(
        &self,
        token_ids: &[u32],
        positions: &[u32],
        seq_lens: &[usize],
        mut caches: Option<&mut [KvCache]>,
        all_logits: bool,
    ) -> Result<Tensor> {
        if token_ids.len() != positions.len()
            || token_ids.is_empty()
            || seq_lens.contains(&0)
            || seq_lens.iter().try_fold(0usize, |n, &v| n.checked_add(v)) != Some(token_ids.len())
        {
            candle_core::bail!("invalid batch token/position/sequence lengths");
        }
        if caches.as_ref().is_some_and(|c| c.len() < seq_lens.len()) {
            candle_core::bail!("missing sequence caches");
        }
        if token_ids
            .iter()
            .any(|&id| id as usize >= self.cfg.vocab_size)
            || positions
                .iter()
                .any(|&pos| pos as usize >= self.cfg.max_position_embeddings)
        {
            candle_core::bail!("token or position out of range");
        }
        let total = token_ids.len();
        let ids = Tensor::from_vec(token_ids.to_vec(), (total,), &self.device)?;
        let mut x = self.embed_tokens.index_select(&ids, 0)?; // [total, hidden]

        for (li, layer) in self.layers.iter().enumerate() {
            let h = layer.input_layernorm.forward(&x)?;
            let attn = layer
                .attn
                .forward(&h, seq_lens, positions, caches.as_deref_mut(), li)?;
            x = x.add(&attn)?;
            let h = layer.post_attention_layernorm.forward(&x)?;
            let mlp = layer.mlp.forward(&h)?;
            x = x.add(&mlp)?;
        }
        let x = self.norm.forward(&x)?;

        let final_hidden = if all_logits {
            x
        } else {
            // Only the last token of each sequence feeds the LM head.
            let mut rows = Vec::with_capacity(seq_lens.len());
            let mut off = 0usize;
            for &len in seq_lens {
                rows.push((off + len - 1) as u32);
                off += len;
            }
            let idx = Tensor::from_vec(rows, (seq_lens.len(),), &self.device)?;
            x.index_select(&idx, 0)?
        };

        match &self.lm_head {
            Some(head) => head.forward(&final_hidden),
            None => final_hidden.matmul(&self.embed_tokens.t()?),
        }
    }
}

/// Narrow model abstraction: the engine never touches Qwen2 internals.
pub trait CausalLm: Send + Sync {
    fn device(&self) -> &Device;

    fn config(&self) -> &ModelConfig;

    fn vocab_size(&self) -> usize;

    /// Dtype the model computes in (used to size KV caches).
    fn dtype(&self) -> DType;

    /// Cached forward: mixed prefill/decode batch over per-sequence KV
    /// caches. Returns `[num_seqs, vocab]` last-token logits.
    fn forward_cached(&self, input: &BatchTokens, caches: &mut [KvCache]) -> Result<Tensor>;

    /// Correctness-first reference path with no KV cache: run the entire
    /// sequence, return `[seq_len, vocab]` logits for every position.
    fn forward_nocache(&self, token_ids: &[u32], positions: &[u32]) -> Result<Tensor>;
}

impl CausalLm for Qwen2 {
    fn device(&self) -> &Device {
        &self.device
    }

    fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn forward_cached(&self, input: &BatchTokens, caches: &mut [KvCache]) -> Result<Tensor> {
        self.forward_impl(
            &input.token_ids,
            &input.positions,
            &input.seq_lens,
            Some(caches),
            false,
        )
    }

    fn forward_nocache(&self, token_ids: &[u32], positions: &[u32]) -> Result<Tensor> {
        let seq_lens = vec![token_ids.len()];
        self.forward_impl(token_ids, positions, &seq_lens, None, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{kv_cache_for, random_model};
    use candle_core::Device;

    fn model() -> Qwen2 {
        random_model(&Device::Cpu)
    }

    /// §52.4 — the critical equivalence test: cached prefill + incremental
    /// decode must match full-sequence recomputation at every position.
    #[test]
    fn cached_logits_match_nocache_logits() {
        let m = model();
        let tokens: Vec<u32> = (1..=12).collect();

        let reference = m
            .forward_nocache(&tokens, &(0..12).collect::<Vec<_>>())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        let mut caches = [kv_cache_for(&m, 32)];
        let split = 8;

        let prefill = m
            .forward_cached(&BatchTokens::prefill(tokens[..split].to_vec()), &mut caches)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (j, &r) in prefill[0].iter().enumerate() {
            assert!(
                (r - reference[split - 1][j]).abs() < 1e-4,
                "prefill logits diverge at vocab {j}"
            );
        }

        for i in split..12 {
            let out = m
                .forward_cached(&BatchTokens::decode_one(tokens[i], i as u32), &mut caches)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for (j, &r) in out[0].iter().enumerate() {
                assert!(
                    (r - reference[i][j]).abs() < 1e-4,
                    "cached decode logits diverge at pos {i}, vocab {j}"
                );
            }
        }
    }

    /// §52.5 — greedy decoding of a sequence must be identical whether the
    /// decode steps run alone or batched with another sequence.
    #[test]
    fn batched_decode_matches_independent_decode() {
        let m = model();
        let tokens_a: Vec<u32> = (1..=5).collect();
        let tokens_b: Vec<u32> = (40..=45).collect();

        // Independent runs.
        let greedy = |tokens: &[u32], steps: usize| -> Vec<u32> {
            let mut caches = [kv_cache_for(&m, 64)];
            let mut out = Vec::new();
            let logits = m
                .forward_cached(&BatchTokens::prefill(tokens.to_vec()), &mut caches)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let mut next = argmax(&logits[0]);
            out.push(next);
            for _ in 1..steps {
                let pos = (tokens.len() + out.len() - 1) as u32;
                let logits = m
                    .forward_cached(&BatchTokens::decode_one(next, pos), &mut caches)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                next = argmax(&logits[0]);
                out.push(next);
            }
            out
        };
        let solo_a = greedy(&tokens_a, 4);
        let solo_b = greedy(&tokens_b, 4);

        // Batched runs: separate prefills (each into its own cache), then
        // joint decode steps.
        let mut caches = [kv_cache_for(&m, 64), kv_cache_for(&m, 64)];
        let first_a = m
            .forward_cached(&BatchTokens::prefill(tokens_a.clone()), &mut caches)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let first_b = m
            .forward_cached(&BatchTokens::prefill(tokens_b.clone()), &mut caches[1..])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let mut next = [argmax(&first_a[0]), argmax(&first_b[0])];
        let mut batched_a = vec![next[0]];
        let mut batched_b = vec![next[1]];
        for _ in 1..4 {
            let pos_a = (tokens_a.len() + batched_a.len() - 1) as u32;
            let pos_b = (tokens_b.len() + batched_b.len() - 1) as u32;
            let input = BatchTokens {
                token_ids: vec![next[0], next[1]],
                positions: vec![pos_a, pos_b],
                seq_lens: vec![1, 1],
            };
            let logits = m
                .forward_cached(&input, &mut caches)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            next = [argmax(&logits[0]), argmax(&logits[1])];
            batched_a.push(next[0]);
            batched_b.push(next[1]);
        }

        assert_eq!(solo_a, batched_a, "batching changed sequence A");
        assert_eq!(solo_b, batched_b, "batching changed sequence B");
    }

    /// §16 — no-cache autoregressive generation produces plausible shapes.
    #[test]
    fn nocache_forward_returns_all_positions() {
        let m = model();
        let tokens: Vec<u32> = (0..7).collect();
        let out = m
            .forward_nocache(&tokens, &(0..7).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(out.dims(), &[7, m.vocab_size()]);
    }

    fn argmax(row: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }
}
