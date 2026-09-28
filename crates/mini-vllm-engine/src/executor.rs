//! Model executor: owns the model and turns sequences into batched forward
//! passes plus sampled tokens.
//!
//! The linear layers (QKV, o_proj, MLP, LM head) always run over the whole
//! batch buffer; the attention core loops per sequence because each
//! sequence owns an independent KV cache. Logits are converted off the
//! device exactly once per forward pass and sampled row by row.

use std::sync::Arc;

use candle_core::{DType, Tensor};
use mini_vllm_model::{BatchTokens, CausalLm};

use crate::sequence::SequenceGroup;

pub struct Executor {
    model: Arc<dyn CausalLm>,
}

/// One sampled token per sequence, in batch order.
pub type SampledTokens = Vec<u32>;

impl Executor {
    pub fn new(model: Arc<dyn CausalLm>) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &Arc<dyn CausalLm> {
        &self.model
    }

    pub fn dtype(&self) -> DType {
        self.model.dtype()
    }

    /// Execute prefill chunks and decode tokens in one forward call. An
    /// unfinished prefill produces no sample and does not advance its RNG.
    pub fn run_mixed_batch(
        &self,
        seqs: &mut [&mut SequenceGroup],
        counts: &[usize],
    ) -> candle_core::Result<Vec<Option<u32>>> {
        if seqs.len() != counts.len() || seqs.is_empty() {
            candle_core::bail!("invalid mixed batch");
        }
        let mut input = BatchTokens::default();
        let mut sample = Vec::new();
        for (seq, &n) in seqs.iter().zip(counts) {
            if n == 0 {
                candle_core::bail!("empty chunk");
            }
            if seq.status == mini_vllm_core::SequenceStatus::Prefill {
                let start = seq.prefill_position;
                if n > seq.prompt_len().saturating_sub(start) {
                    candle_core::bail!("chunk beyond prompt");
                }
                input
                    .token_ids
                    .extend_from_slice(&seq.request.prompt_token_ids[start..start + n]);
                input.positions.extend((start..start + n).map(|p| p as u32));
                sample.push(start + n == seq.prompt_len());
            } else {
                if n != 1 {
                    candle_core::bail!("decode requires one token");
                }
                input.token_ids.push(seq.next_token);
                input.positions.push(seq.next_input_position());
                sample.push(true);
            }
            input.seq_lens.push(n);
        }
        let mut caches: Vec<_> = seqs
            .iter_mut()
            .map(|s| s.take_cache().expect("admitted cache"))
            .collect();
        let lengths: Vec<_> = caches.iter().map(|c| c.seq_len()).collect();
        let result = self
            .model
            .forward_cached(&input, &mut caches)
            .and_then(|t| f32_rows(&t))
            .and_then(|rows| {
                if rows.len() != seqs.len()
                    || rows.iter().any(|r| {
                        r.len() != self.model.vocab_size() || r.iter().any(|v| !v.is_finite())
                    })
                {
                    candle_core::bail!("invalid model logits shape or non-finite values");
                }
                Ok(rows)
            });
        if result.is_err() {
            // A failed forward can have appended to some caches. Roll back
            // to the pre-call lengths. A cache that cannot roll back breaks
            // the append-only invariant: doom that sequence (the engine's
            // isolated-retry loop skips terminal sequences, so a corrupt
            // cache is never retried) instead of panicking the engine thread.
            for ((seq, cache), len) in seqs.iter_mut().zip(caches.iter_mut()).zip(lengths) {
                if let Err(e) = cache.truncate(len) {
                    tracing::error!(request_id = %seq.request.id, error = %e, "kv rollback failed");
                    seq.failure = Some(format!("kv rollback failed: {e}"));
                    seq.finish_reason = Some(mini_vllm_core::FinishReason::Error);
                    seq.status = mini_vllm_core::SequenceStatus::Failed;
                }
            }
        }
        for (seq, cache) in seqs.iter_mut().zip(caches) {
            seq.restore_cache(cache);
        }
        Ok(seqs
            .iter_mut()
            .zip(result?)
            .zip(sample)
            .map(|((seq, row), sample)| sample.then(|| seq.sample_next(&row)))
            .collect())
    }

    /// Prefill one sequence: run the whole prompt through the model, fill
    /// its KV cache, and sample the first token.
    pub fn run_prefill(&self, seq: &mut SequenceGroup) -> candle_core::Result<u32> {
        let mut cache = seq
            .take_cache()
            .expect("prefill requires an admitted cache");
        let input = BatchTokens::prefill(seq.request.prompt_token_ids.clone());
        let result = self
            .model
            .forward_cached(&input, std::slice::from_mut(&mut cache));
        seq.restore_cache(cache);
        let logits = result?;
        let rows = f32_rows(&logits)?;
        Ok(seq.sample_next(&rows[0]))
    }

    /// Decode step for a batch of running sequences: one token each, batched
    /// into a single forward pass. Returns sampled tokens in batch order.
    pub fn run_decode_batch(
        &self,
        seqs: &mut [&mut SequenceGroup],
    ) -> candle_core::Result<SampledTokens> {
        let n = seqs.len();
        let input = BatchTokens {
            token_ids: seqs.iter().map(|s| s.next_token).collect(),
            positions: seqs.iter().map(|s| s.next_input_position()).collect(),
            seq_lens: vec![1; n],
        };
        let mut caches: Vec<mini_vllm_kv::KvCache> = seqs
            .iter_mut()
            .map(|s| s.take_cache().expect("decode requires an admitted cache"))
            .collect();

        let lengths: Vec<_> = caches.iter().map(|c| c.seq_len()).collect();
        let outcome = self
            .model
            .forward_cached(&input, &mut caches)
            .and_then(|logits| f32_rows(&logits))
            .and_then(|rows| {
                if rows.len() != n {
                    candle_core::bail!("expected {n} logits rows, got {}", rows.len());
                }
                Ok(rows)
            });

        // A failed forward can have appended only some sequences/layers.
        // Rewind before retrying; sampling has not advanced any RNG yet. A
        // cache that cannot rewind breaks the append-only invariant and
        // dooms its sequence rather than panicking the engine thread.
        if outcome.is_err() {
            for ((seq, cache), len) in seqs.iter_mut().zip(caches.iter_mut()).zip(lengths) {
                if let Err(e) = cache.truncate(len) {
                    tracing::error!(request_id = %seq.request.id, error = %e, "kv rollback failed");
                    seq.failure = Some(format!("kv rollback failed: {e}"));
                    seq.finish_reason = Some(mini_vllm_core::FinishReason::Error);
                    seq.status = mini_vllm_core::SequenceStatus::Failed;
                }
            }
        }

        // Every sequence gets its cache back before sampling or propagating.
        for (s, c) in seqs.iter_mut().zip(caches) {
            s.restore_cache(c);
        }
        let rows = outcome?;

        let mut sampled = Vec::with_capacity(n);
        for (i, s) in seqs.iter_mut().enumerate() {
            let row = rows.get(i).ok_or_else(|| {
                candle_core::Error::Msg(format!("logits row {i} missing ({} rows)", rows.len()))
            })?;
            sampled.push(s.sample_next(row));
        }
        Ok(sampled)
    }

    /// Single-sequence decode step, used to isolate a batch-wide failure to
    /// the sequence(s) that actually fail (§54: one bad sequence must not
    /// kill its batch mates).
    pub fn run_decode_single(&self, seq: &mut SequenceGroup) -> candle_core::Result<u32> {
        let mut cache = seq.take_cache().expect("decode requires an admitted cache");
        let input = BatchTokens::decode_one(seq.next_token, seq.next_input_position());
        let result = self
            .model
            .forward_cached(&input, std::slice::from_mut(&mut cache));
        seq.restore_cache(cache);
        let logits = result?;
        let rows = f32_rows(&logits)?;
        Ok(seq.sample_next(&rows[0]))
    }
}

/// Convert `[n, vocab]` logits to host F32 rows exactly once. Works from
/// any device (`to_vec*` performs the device→host copy internally).
fn f32_rows(logits: &Tensor) -> candle_core::Result<Vec<Vec<f32>>> {
    logits.to_dtype(DType::F32)?.to_vec2::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use mini_vllm_core::{GenerationRequest, SamplingParams};
    use mini_vllm_kv::KvCache;
    use mini_vllm_sampling::Sampler;

    #[test]
    fn failed_batch_rolls_back_before_individual_retry() {
        let model: Arc<dyn CausalLm> =
            Arc::new(mini_vllm_model::testutil::random_model(&Device::Cpu));
        let executor = Executor::new(model);
        let make = |capacity| {
            let (tx, rx) = tokio::sync::mpsc::channel(8);
            let mut seq = SequenceGroup::new(
                GenerationRequest {
                    id: format!("{capacity}"),
                    prompt_token_ids: vec![4],
                    max_new_tokens: 7,
                    sampling: SamplingParams {
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    stop_token_ids: vec![],
                    stop_strings: vec![],
                },
                Sampler::from_seed(1),
                None,
                tx,
                vec![],
                8,
            );
            seq.restore_cache(KvCache::new(2, 2, 8, capacity, DType::F32, &Device::Cpu).unwrap());
            let token = executor.run_prefill(&mut seq).unwrap();
            seq.record_sampled(token);
            (seq, rx)
        };
        let (mut healthy, _rx1) = make(8);
        let (mut failing, _rx2) = make(1);
        let (mut reference, _rx3) = make(8);
        assert!(executor
            .run_decode_batch(&mut [&mut healthy, &mut failing])
            .is_err());
        let mut cache = healthy.take_cache().unwrap();
        assert!(cache.layers().iter().all(|l| l.len() == 1));
        healthy.restore_cache(cache);
        assert_eq!(
            executor.run_decode_single(&mut healthy).unwrap(),
            executor.run_decode_single(&mut reference).unwrap()
        );
        let mut actual = healthy.take_cache().unwrap();
        let mut expected = reference.take_cache().unwrap();
        for (a, b) in actual.layers().iter().zip(expected.layers().iter()) {
            assert_eq!(a.len(), 2);
            for (x, y) in [(&a.key, &b.key), (&a.value, &b.value)] {
                let x = x
                    .narrow(1, 0, 2)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let y = y
                    .narrow(1, 0, 2)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert!(x.iter().zip(y).all(|(x, y)| (x - y).abs() < 1e-5));
            }
        }
    }
    struct LayerFault {
        cfg: mini_vllm_model::ModelConfig,
        device: Device,
        fail_after: usize,
    }
    impl CausalLm for LayerFault {
        fn config(&self) -> &mini_vllm_model::ModelConfig {
            &self.cfg
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn vocab_size(&self) -> usize {
            self.cfg.vocab_size
        }
        fn forward_nocache(&self, _: &[u32], _: &[u32]) -> candle_core::Result<Tensor> {
            unreachable!()
        }
        fn forward_cached(
            &self,
            _: &BatchTokens,
            caches: &mut [KvCache],
        ) -> candle_core::Result<Tensor> {
            for cache in caches {
                for layer in 0..=self.fail_after {
                    let t = Tensor::ones((2, 1, 8), DType::F32, &self.device)?;
                    cache.write_layer_pages(layer, &t, &t)?;
                }
            }
            candle_core::bail!("injected layer failure")
        }
    }
    #[test]
    fn partial_layer_failures_restore_all_cursors_on_both_storage_paths() {
        for paged in [false, true] {
            for fail_after in 0..2 {
                let model = Arc::new(LayerFault {
                    cfg: mini_vllm_model::testutil::tiny_config(),
                    device: Device::Cpu,
                    fail_after,
                });
                let executor = Executor::new(model);
                let (tx, _rx) = tokio::sync::mpsc::channel(4);
                let mut seq = SequenceGroup::new(
                    GenerationRequest {
                        id: "fault".into(),
                        prompt_token_ids: vec![4],
                        max_new_tokens: 2,
                        sampling: SamplingParams::default(),
                        stop_token_ids: vec![],
                        stop_strings: vec![],
                    },
                    Sampler::from_seed(1),
                    None,
                    tx,
                    vec![],
                    4,
                );
                seq.status = mini_vllm_core::SequenceStatus::Prefill;
                let mut cache = if paged {
                    KvCache::new_paged(2, 4, 2).unwrap()
                } else {
                    KvCache::new(2, 2, 8, 4, DType::F32, &Device::Cpu).unwrap()
                };
                let old = Tensor::zeros((2, 1, 8), DType::F32, &Device::Cpu).unwrap();
                for layer in 0..2 {
                    cache.write_layer_pages(layer, &old, &old).unwrap();
                }
                seq.restore_cache(cache);
                assert!(executor.run_mixed_batch(&mut [&mut seq], &[1]).is_err());
                let mut cache = seq.take_cache().unwrap();
                assert_eq!(cache.seq_len(), 1);
                for layer in 0..2 {
                    let views = cache.write_layer_pages(layer, &old, &old).unwrap();
                    assert_eq!(
                        views.iter().map(|(k, _)| k.dim(1).unwrap()).sum::<usize>(),
                        2
                    );
                    assert!(views.iter().all(|(k, _)| k
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .all(|v| *v == 0.)));
                }
            }
        }
    }
}
