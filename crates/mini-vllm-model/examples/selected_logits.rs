//! CPU comparison of full vs selective chunk logits; pass a local model directory.
use candle_core::{DType, Device};
use mini_vllm_kv::KvCache;
use mini_vllm_model::{loader, BatchTokens, CausalLm};
use std::{path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected a local model directory")?;
    let model = loader::load_model(Path::new(&path), DType::F32, Device::Cpu)?;
    let cfg = model.config();
    let total = 256.min(cfg.max_position_embeddings);
    let chunk = 16;
    let run = |selected: bool| -> candle_core::Result<(f64, Vec<f32>)> {
        let mut cache = KvCache::new_paged(cfg.num_hidden_layers, total, chunk)?;
        let mut last = Vec::new();
        let start = Instant::now();
        for position in (0..total).step_by(chunk) {
            let count = chunk.min(total - position);
            let input = BatchTokens {
                token_ids: (position..position + count)
                    .map(|i| (i % cfg.vocab_size) as u32)
                    .collect(),
                positions: (position..position + count).map(|i| i as u32).collect(),
                seq_lens: vec![count],
            };
            let final_chunk = position + count == total;
            let logits = if selected {
                model.forward_cached_selected(
                    &input,
                    std::slice::from_mut(&mut cache),
                    if final_chunk { &[0] } else { &[] },
                )?
            } else {
                model.forward_cached(&input, std::slice::from_mut(&mut cache))?
            };
            if !selected || final_chunk {
                last = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?.remove(0);
                std::hint::black_box(&last);
            }
        }
        Ok((start.elapsed().as_secs_f64() * 1000., last))
    };
    run(false)?;
    run(true)?;
    let mut trials = Vec::new();
    for trial in 0..3 {
        let (baseline, selected) = if trial % 2 == 0 {
            (run(false)?, run(true)?)
        } else {
            let selected = run(true)?;
            (run(false)?, selected)
        };
        if baseline.1.iter().chain(&selected.1).any(|v| !v.is_finite()) {
            return Err("non-finite logits in comparison".into());
        }
        let error = baseline
            .1
            .iter()
            .zip(&selected.1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if !error.is_finite() || error > 1e-4 {
            return Err(format!("logit mismatch: {error}").into());
        }
        trials.push(
            serde_json::json!({"trial":trial,"full_logits_ms":baseline.0,
            "selected_logits_ms":selected.0,"max_absolute_error":error}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model":path,"device":"cpu","dtype":"f32","tokens":total,"chunk_tokens":chunk,
            "full_projection_rows":total.div_ceil(chunk),"selected_projection_rows":1,
            "warmup_per_mode":1,"trials":trials
        }))?
    );
    Ok(())
}
