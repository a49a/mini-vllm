//! Random-weight fixtures for tests (feature `test-util`).
//!
//! These exercise every code path with a tiny deterministic-shaped model —
//! KV-cache equivalence, batching equivalence, engine concurrency — without
//! needing real weights on disk.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use mini_vllm_kv::KvCache;

use crate::config::ModelConfig;
use crate::qwen2::{CausalLm, Qwen2};

/// A tiny Qwen2 config: GQA (4 query / 2 KV heads), 2 layers, tied embeddings.
pub fn tiny_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        vocab_size: 64,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        bos_token_id: None,
        eos_token_ids: vec![2],
        tie_word_embeddings: true,
        torch_dtype: Some("float32".into()),
        architectures: vec!["Qwen2ForCausalLM".into()],
    }
}

fn randn(shape: (usize, usize), dev: &Device) -> Tensor {
    Tensor::randn(0.0f32, 0.3f32, shape, dev).unwrap()
}

fn rand_vec(n: usize, dev: &Device) -> Tensor {
    Tensor::randn(0.0f32, 0.3f32, n, dev).unwrap()
}

/// Random weight map with every Qwen2 weight name for `cfg`.
pub fn random_tensors(
    cfg: &ModelConfig,
    untied_lm_head: bool,
    dev: &Device,
) -> HashMap<String, Tensor> {
    let hd = cfg.head_dim();
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        randn((cfg.vocab_size, cfg.hidden_size), dev),
    );
    if untied_lm_head {
        t.insert(
            "lm_head.weight".into(),
            randn((cfg.vocab_size, cfg.hidden_size), dev),
        );
    }
    t.insert("model.norm.weight".into(), rand_vec(cfg.hidden_size, dev));
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}.");
        let q = cfg.num_attention_heads * hd;
        let kv = cfg.num_key_value_heads * hd;
        t.insert(
            format!("{p}self_attn.q_proj.weight"),
            randn((q, cfg.hidden_size), dev),
        );
        t.insert(format!("{p}self_attn.q_proj.bias"), rand_vec(q, dev));
        t.insert(
            format!("{p}self_attn.k_proj.weight"),
            randn((kv, cfg.hidden_size), dev),
        );
        t.insert(format!("{p}self_attn.k_proj.bias"), rand_vec(kv, dev));
        t.insert(
            format!("{p}self_attn.v_proj.weight"),
            randn((kv, cfg.hidden_size), dev),
        );
        t.insert(format!("{p}self_attn.v_proj.bias"), rand_vec(kv, dev));
        t.insert(
            format!("{p}self_attn.o_proj.weight"),
            randn((cfg.hidden_size, q), dev),
        );
        t.insert(
            format!("{p}mlp.gate_proj.weight"),
            randn((cfg.intermediate_size, cfg.hidden_size), dev),
        );
        t.insert(
            format!("{p}mlp.up_proj.weight"),
            randn((cfg.intermediate_size, cfg.hidden_size), dev),
        );
        t.insert(
            format!("{p}mlp.down_proj.weight"),
            randn((cfg.hidden_size, cfg.intermediate_size), dev),
        );
        t.insert(
            format!("{p}input_layernorm.weight"),
            rand_vec(cfg.hidden_size, dev),
        );
        t.insert(
            format!("{p}post_attention_layernorm.weight"),
            rand_vec(cfg.hidden_size, dev),
        );
    }
    t
}

/// Random-weight tiny Qwen2 on `dev` (tied embeddings).
pub fn random_model(dev: &Device) -> Qwen2 {
    let cfg = tiny_config();
    let tensors = random_tensors(&cfg, false, dev);
    Qwen2::load(cfg, &tensors, DType::F32, dev).unwrap()
}

/// Random-weight tiny Qwen2 with an untied LM head.
pub fn random_model_untied(dev: &Device) -> Qwen2 {
    let mut cfg = tiny_config();
    cfg.tie_word_embeddings = false;
    let tensors = random_tensors(&cfg, true, dev);
    Qwen2::load(cfg, &tensors, DType::F32, dev).unwrap()
}

/// Fresh KV cache matching `model`.
pub fn kv_cache_for(model: &Qwen2, capacity: usize) -> KvCache {
    let cfg = model.config();
    KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        capacity,
        DType::F32,
        model.device(),
    )
    .unwrap()
}
