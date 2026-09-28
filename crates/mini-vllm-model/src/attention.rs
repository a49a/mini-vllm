//! Grouped Query Attention (GQA) with explicit prefill / decode paths.
//!
//! Prefill: the whole prompt goes through causal self-attention (mask built
//! once, offset by however many tokens are already cached).
//! Decode: one token per sequence — the new token attends to the entire
//! cache and itself, so no triangular mask is needed at all.
//!
//! Linear projections (Q/K/V) run over the whole token batch; only the
//! attention core is per-sequence, because each sequence owns an
//! independent KV cache.

use candle_core::{DType, Device, Result, Tensor};

use crate::config::ModelConfig;
use crate::linear::Linear;
use crate::rope::SharedRope;
use mini_vllm_kv::KvCache;

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    rope: SharedRope,
}

impl Attention {
    pub fn load(
        prefix: &str,
        tensors: &crate::loader::Tensors,
        cfg: &ModelConfig,
        rope: SharedRope,
    ) -> crate::error::Result<Self> {
        let get = |name: &str| -> crate::error::Result<Tensor> {
            Ok(crate::loader::get_weight(tensors, &format!("{prefix}{name}"))?.clone())
        };
        let opt = |name: &str| -> crate::error::Result<Option<Tensor>> {
            Ok(crate::loader::opt_weight(tensors, &format!("{prefix}{name}"))?.cloned())
        };
        Ok(Self {
            q_proj: Linear::new(get("q_proj.weight")?, opt("q_proj.bias")?),
            k_proj: Linear::new(get("k_proj.weight")?, opt("k_proj.bias")?),
            v_proj: Linear::new(get("v_proj.weight")?, opt("v_proj.bias")?),
            o_proj: Linear::new(get("o_proj.weight")?, None),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            scale: 1.0 / (cfg.head_dim() as f64).sqrt(),
            rope,
        })
    }

    /// Batched attention over multiple sequences sharing one token buffer.
    ///
    /// * `x`: `[total_tokens, hidden]` (Q/K/V projections run batched)
    /// * `seq_lens`: tokens belonging to each sequence (decode steps: all 1)
    /// * `positions`: absolute position of each token
    /// * `caches`: per-sequence KV caches, or `None` for the no-cache path
    pub fn forward(
        &self,
        x: &Tensor,
        seq_lens: &[usize],
        positions: &[u32],
        mut caches: Option<&mut [KvCache]>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        debug_assert_eq!(x.dim(0)?, positions.len());
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let hd = self.head_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;

        let mut outs = Vec::with_capacity(seq_lens.len());
        let mut offset = 0usize;
        for (si, &len) in seq_lens.iter().enumerate() {
            // [len, heads*hd] → [heads, len, hd]
            let qs = q
                .narrow(0, offset, len)?
                .reshape((len, nh, hd))?
                .transpose(0, 1)?
                .contiguous()?;
            let ks = k
                .narrow(0, offset, len)?
                .reshape((len, nkv, hd))?
                .transpose(0, 1)?
                .contiguous()?;
            let vs = v
                .narrow(0, offset, len)?
                .reshape((len, nkv, hd))?
                .transpose(0, 1)?
                .contiguous()?;
            let pos = &positions[offset..offset + len];
            let (qr, kr) = self.rope.apply(&qs, &ks, pos)?;

            let out = match caches.as_deref_mut() {
                Some(caches) => {
                    let pages = caches[si].write_layer_pages(layer_idx, &kr, &vs)?;
                    self.attend_pages(&qr, &pages)?
                }
                None => self.attend(&qr, &kr, &vs)?,
            };
            outs.push(out);
            offset += len;
        }
        let attn = if outs.len() == 1 {
            outs.remove(0)
        } else {
            Tensor::cat(&outs, 0)?
        };
        self.o_proj.forward(&attn)
    }

    /// Block-addressed attention: K/V stay in their physical pages. Scores
    /// share one softmax normalization across all pages, then each page's
    /// value contribution is accumulated. This is a portable reference,
    /// not a fused GPU PagedAttention kernel.
    fn attend_pages(&self, q: &Tensor, pages: &[(Tensor, Tensor)]) -> Result<Tensor> {
        let group = self.num_heads / self.num_kv_heads;
        let q_len = q.dim(1)?;
        let mut scores = Vec::with_capacity(pages.len());
        let mut values = Vec::with_capacity(pages.len());
        let mut kv_len = 0;
        for (k, v) in pages {
            let k = repeat_kv(k, group)?;
            scores.push(q.matmul(&k.transpose(1, 2)?)?.affine(self.scale, 0.0)?);
            values.push(repeat_kv(v, group)?);
            kv_len += k.dim(1)?;
        }
        let mut scores = Tensor::cat(&scores, 2)?;
        if q_len > 1 {
            scores = scores.broadcast_add(&causal_mask(
                q_len,
                kv_len,
                kv_len - q_len,
                q.dtype(),
                q.device(),
            )?)?;
        }
        let probs = softmax_last_dim(&scores)?;
        let mut offset = 0;
        let mut result: Option<Tensor> = None;
        for v in values {
            let len = v.dim(1)?;
            let part = probs.narrow(2, offset, len)?.contiguous()?.matmul(&v)?;
            result = Some(match result {
                None => part,
                Some(sum) => sum.add(&part)?,
            });
            offset += len;
        }
        result
            .ok_or_else(|| candle_core::Error::Msg("empty attention pages".into()))?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((q_len, self.num_heads * self.head_dim))
    }

    /// Scaled dot-product attention for one sequence.
    ///
    /// `q: [nh, q_len, hd]`, `k/v: [nkv, kv_len, hd]` → `[q_len, nh*hd]`.
    /// Causal masking is applied when `q_len > 1` (prefill); a single-token
    /// decode step attends to everything, so no mask is built.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q_len = q.dim(1)?;
        let kv_len = k.dim(1)?;
        let group = self.num_heads / self.num_kv_heads;
        let k_rep = repeat_kv(k, group)?;
        let v_rep = repeat_kv(v, group)?;

        let mut scores = q.matmul(&k_rep.transpose(1, 2)?)?; // [nh, q_len, kv_len]
        scores = scores.affine(self.scale, 0.0)?;
        if q_len > 1 {
            let offset = kv_len - q_len;
            let mask = causal_mask(q_len, kv_len, offset, q.dtype(), q.device())?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = softmax_last_dim(&scores)?;
        let out = probs.matmul(&v_rep)?; // [nh, q_len, hd]
        out.transpose(0, 1)?
            .contiguous()?
            .reshape((q_len, self.num_heads * self.head_dim))
    }
}

/// Numerically stable softmax over the last dimension.
/// (`Tensor::softmax` lives in candle-nn, which this crate avoids.)
fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    // Reductions drop the dim; restore it so broadcasts align.
    let max = t
        .max(candle_core::D::Minus1)?
        .unsqueeze(candle_core::D::Minus1)?;
    let shifted = t.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp
        .sum(candle_core::D::Minus1)?
        .unsqueeze(candle_core::D::Minus1)?;
    exp.broadcast_div(&sum)
}

/// Expand KV heads to query heads: each KV head is repeated `group` times
/// consecutively, matching the query head order `[kv0 ×g, kv1 ×g, …]`.
fn repeat_kv(t: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(t.clone());
    }
    let nkv = t.dim(0)?;
    let len = t.dim(1)?;
    let hd = t.dim(2)?;
    let copies: Vec<&Tensor> = (0..group).map(|_| t).collect();
    // Cat along the token dim then reshape: head order becomes
    // [kv0 ×group, kv1 ×group, ...] as required by GQA.
    Tensor::cat(&copies, 1)?.reshape((nkv * group, len, hd))
}

/// Additive causal mask: position `offset + i` may attend to keys `j <= offset + i`.
pub fn causal_mask(
    q_len: usize,
    kv_len: usize,
    offset: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; q_len * kv_len];
    for (i, row) in data.chunks_mut(kv_len).enumerate() {
        let g = offset + i;
        for (j, cell) in row.iter_mut().enumerate() {
            if j > g {
                *cell = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (q_len, kv_len), dev)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::rope::RopeCache;
    use candle_core::DType;
    use std::sync::Arc;

    fn tiny_cfg() -> ModelConfig {
        ModelConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 32,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_ids: vec![],
            tie_word_embeddings: true,
            torch_dtype: None,
            architectures: vec!["Qwen2ForCausalLM".into()],
        }
    }

    fn attention(cfg: &ModelConfig) -> Attention {
        let hd = cfg.head_dim();
        let linear = |out: usize| {
            Linear::new(
                Tensor::randn(0.0f32, 0.1f32, (out, cfg.hidden_size), &Device::Cpu).unwrap(),
                None,
            )
        };
        let rope =
            Arc::new(RopeCache::new(32, hd, cfg.rope_theta, DType::F32, &Device::Cpu).unwrap());
        Attention {
            q_proj: linear(cfg.num_attention_heads * hd),
            k_proj: linear(cfg.num_key_value_heads * hd),
            v_proj: linear(cfg.num_key_value_heads * hd),
            o_proj: linear(cfg.hidden_size),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            scale: 1.0 / (hd as f64).sqrt(),
            rope,
        }
    }

    fn kv_cache(cfg: &ModelConfig, capacity: usize) -> KvCache {
        KvCache::new(
            1,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            capacity,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    }

    fn tokens(n: usize, hidden: usize) -> Tensor {
        let v: Vec<f32> = (0..n * hidden)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.2)
            .collect();
        Tensor::from_vec(v, (n, hidden), &Device::Cpu).unwrap()
    }

    #[test]
    fn gqa_output_shape() {
        let cfg = tiny_cfg();
        let attn = attention(&cfg);
        let x = tokens(5, cfg.hidden_size);
        let out = attn.forward(&x, &[5], &[0, 1, 2, 3, 4], None, 0).unwrap();
        assert_eq!(out.dims(), &[5, cfg.hidden_size]);
    }

    #[test]
    fn decode_shape_with_cache() {
        let cfg = tiny_cfg();
        let attn = attention(&cfg);
        let mut caches = [kv_cache(&cfg, 16)];
        // prefill 3 tokens
        let x = tokens(3, cfg.hidden_size);
        let _ = attn
            .forward(&x, &[3], &[0, 1, 2], Some(&mut caches), 0)
            .unwrap();
        assert_eq!(caches[0].seq_len(), 3);
        // decode 1 token
        let x1 = tokens(1, cfg.hidden_size);
        let out = attn.forward(&x1, &[1], &[3], Some(&mut caches), 0).unwrap();
        assert_eq!(out.dims(), &[1, cfg.hidden_size]);
        assert_eq!(caches[0].seq_len(), 4);
    }

    #[test]
    fn batched_decode_matches_single_sequence_decode() {
        let cfg = tiny_cfg();
        let attn = attention(&cfg);
        let pos = |n: usize| (0..n as u32).collect::<Vec<_>>();

        // Two sequences decoded one at a time.
        let mut solo = [kv_cache(&cfg, 32)];
        let xa = tokens(4, cfg.hidden_size);
        let solo_prefill = attn
            .forward(&xa, &[4], &pos(4), Some(&mut solo), 0)
            .unwrap();
        let x1 = tokens(1, cfg.hidden_size);
        let solo_step = attn.forward(&x1, &[1], &[4], Some(&mut solo), 0).unwrap();

        // The same two-step flow, with a second unrelated sequence batched in.
        let mut batched = [kv_cache(&cfg, 32), kv_cache(&cfg, 32)];
        let _ = attn
            .forward(&xa, &[4], &pos(4), Some(&mut batched), 0)
            .unwrap();
        let xb = tokens(1, cfg.hidden_size);
        let both = Tensor::cat(&[&x1, &xb], 0).unwrap();
        let mixed = attn
            .forward(&both, &[1, 1], &[4, 0], Some(&mut batched), 0)
            .unwrap();

        let a1 = solo_step.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let a2 = mixed
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (x, y) in a1.iter().zip(a2.iter()) {
            assert!((x - y).abs() < 1e-5, "batching changed sequence A output");
        }
        let _ = solo_prefill;
    }

    #[test]
    fn causal_mask_blocks_future() {
        let m = causal_mask(3, 3, 0, DType::F32, &Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(m[0][1], f32::NEG_INFINITY);
        assert_eq!(m[2][0], 0.0);
        assert_eq!(m[2][2], 0.0);
    }

    #[test]
    fn repeat_kv_interleaves_heads() {
        // 2 kv heads × group 2 → heads [k0, k0, k1, k1]
        let t = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1, 1), &Device::Cpu).unwrap();
        let out = repeat_kv(&t, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0]);
    }
}
