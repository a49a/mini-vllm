//! Generated action sequences against the real scheduler, with failure replay
//! and deletion shrinking. No generated case requires external model weights.
use candle_core::{DType, Device};
use mini_vllm_core::{EngineConfig, GenerationEvent, GenerationRequest, SamplingParams};
use mini_vllm_engine::{spawn_engine, EngineApiError, ShutdownMode};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc, time::Duration};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Op {
    Submit { slot: usize, prefix: u32 },
    Poll(usize),
    Cancel(usize),
    Disconnect(usize),
    Pause,
    Shutdown(bool),
}
struct Observed {
    id: String,
    rx: mpsc::Receiver<GenerationEvent>,
    terminal: bool,
}
impl Observed {
    fn observe(&mut self, event: GenerationEvent) -> Result<(), String> {
        if self.terminal {
            return Err(format!("{}: event after terminal", self.id));
        }
        if matches!(
            event,
            GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
        ) {
            self.terminal = true;
        }
        Ok(())
    }
}
fn generate(mut seed: u64, count: usize) -> Vec<Op> {
    let mut ops = Vec::new();
    // Warm, reuse, then fill more unique prefixes than the cache holds.
    for prefix in [4, 4, 5, 6, 7, 8] {
        ops.extend([
            Op::Submit { slot: 0, prefix },
            Op::Pause,
            Op::Poll(0),
            Op::Disconnect(0),
        ]);
    }
    for _ in 0..count {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let slot = (seed as usize >> 8) % 4;
        ops.push(match seed % 8 {
            0..=2 => Op::Submit {
                slot,
                prefix: 4 + ((seed >> 16) % 5) as u32,
            },
            3 => Op::Poll(slot),
            4 => Op::Cancel(slot),
            5 => Op::Disconnect(slot),
            _ => Op::Pause,
        });
    }
    // Shutdown with pending receivers, then exercise repeated shutdown/submission.
    ops.extend([
        Op::Shutdown(seed & 1 == 0),
        Op::Submit { slot: 3, prefix: 4 },
        Op::Shutdown(true),
    ]);
    ops
}
async fn execute(ops: &[Op]) -> Result<(), String> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../mini-vllm-model/tests/fixtures/tiny-qwen2");
    let model = Arc::new(
        mini_vllm_model::loader::load_model(fixture, DType::F32, Device::Cpu)
            .map_err(|e| e.to_string())?,
    );
    let handle = spawn_engine(
        model,
        None,
        EngineConfig {
            max_model_len: 64,
            max_num_seqs: 2,
            max_batch_tokens: 3,
            max_prefill_chunk_tokens: 2,
            kv_block_size: 4,
            max_kv_tokens: 128,
            prefix_cache_tokens: 8,
            event_channel_capacity: 2,
            output_drain_timeout_ms: 5,
            ..Default::default()
        },
        17,
    )
    .map_err(|e| e.to_string())?;
    let mut slots: Vec<Option<Observed>> = (0..4).map(|_| None).collect();
    // Always clean up the engine even when an invariant fails during replay.
    let result = async {
        for (step, op) in ops.iter().enumerate() {
            match *op {
                Op::Submit { slot, prefix } => {
                    let slot = slot % slots.len();
                    if slots[slot].is_some() {
                        continue;
                    }
                    let id = format!("case-{step}");
                    let request = GenerationRequest {
                        id: id.clone(),
                        prompt_token_ids: vec![prefix % 60; 9],
                        max_new_tokens: 6,
                        sampling: SamplingParams {
                            temperature: 0.,
                            ..Default::default()
                        },
                        stop_token_ids: vec![],
                        stop_strings: vec![],
                    };
                    match handle.generate(request) {
                        Ok(rx) => {
                            slots[slot] = Some(Observed {
                                id,
                                rx,
                                terminal: false,
                            })
                        }
                        Err(EngineApiError::ShuttingDown | EngineApiError::QueueFull) => {}
                        Err(e) => return Err(format!("step {step}: unexpected submit error {e}")),
                    }
                }
                Op::Poll(slot) => {
                    let slot = slot % slots.len();
                    if let Some(observed) = &mut slots[slot] {
                        match observed.rx.try_recv() {
                            Ok(event) => observed.observe(event)?,
                            Err(mpsc::error::TryRecvError::Disconnected) => {
                                slots[slot] = None;
                            }
                            Err(mpsc::error::TryRecvError::Empty) => {}
                        }
                    }
                }
                Op::Cancel(slot) => {
                    if let Some(observed) = &slots[slot % slots.len()] {
                        let _ = handle.cancel(&observed.id);
                    }
                }
                Op::Disconnect(slot) => {
                    let index = slot % slots.len();
                    slots[index] = None;
                }
                Op::Pause => tokio::time::sleep(Duration::from_millis(2)).await,
                Op::Shutdown(cancel) => handle.request_shutdown(
                    if cancel {
                        ShutdownMode::Cancel
                    } else {
                        ShutdownMode::Drain
                    },
                    Duration::from_millis(10),
                ),
            }
            let m = handle.metrics().snapshot();
            if m.requests_running < 0
                || m.requests_waiting < 0
                || m.kv_blocks_used > m.kv_blocks_total
            {
                return Err(format!("invalid gauges: step {step}: {m:?}"));
            }
        }
        Ok(())
    }
    .await;
    handle.request_shutdown(ShutdownMode::Cancel, Duration::ZERO);
    handle.join(Duration::from_secs(5))?;
    result?;
    for observed in slots.iter_mut().flatten() {
        while let Some(event) = observed.rx.recv().await {
            observed.observe(event)?;
        }
        // Slow/disconnected consumers may close without a terminal event by
        // contract. If delivered, it must be unique and last.
    }
    let m = handle.metrics().snapshot();
    if m.kv_blocks_used != 0
        || m.kv_active_sequences != 0
        || m.cached_prefix_tokens != 0
        || m.requests_running != 0
        || m.requests_waiting != 0
        || m.requests_finishing != 0
    {
        return Err(format!("resources retained after shutdown: {m:?}"));
    }
    if m.requests_total != m.requests_finished + m.requests_failed + m.requests_cancelled {
        return Err(format!("requests not retired exactly once: {m:?}"));
    }
    Ok(())
}

// Bounded delta debugging: remove chunks only when the same invariant still
// fails. Timing-sensitive failures are replayed three times before shrinking.
async fn shrink<F, Fut>(mut ops: Vec<Op>, mut fails: F) -> Vec<Op>
where
    F: FnMut(Vec<Op>) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let mut chunk = (ops.len() / 2).max(1);
    let mut budget = 128;
    while chunk > 0 && budget > 0 {
        let mut start = 0;
        while start < ops.len() && budget > 0 {
            let mut candidate = ops.clone();
            candidate.drain(start..(start + chunk).min(candidate.len()));
            budget -= 1;
            if fails(candidate.clone()).await {
                ops = candidate;
            } else {
                start += chunk;
            }
        }
        chunk /= 2;
    }
    ops
}
async fn minimize(ops: Vec<Op>, failure: &str) -> Vec<Op> {
    let signature = failure.split(':').next().unwrap_or(failure).to_owned();
    shrink(ops, |candidate| {
        let signature = signature.clone();
        async move {
            for _ in 0..3 {
                if !execute(&candidate)
                    .await
                    .is_err_and(|e| e.starts_with(&signature))
                {
                    return false;
                }
            }
            true
        }
    })
    .await
}

#[tokio::test]
async fn shrinker_preserves_required_action_order() {
    let reduced = shrink(
        vec![
            Op::Pause,
            Op::Cancel(0),
            Op::Poll(1),
            Op::Shutdown(true),
            Op::Pause,
        ],
        |ops| async move {
            let cancel = ops.iter().position(|op| matches!(op, Op::Cancel(0)));
            let stop = ops.iter().position(|op| matches!(op, Op::Shutdown(true)));
            matches!((cancel, stop), (Some(c), Some(s)) if c < s)
        },
    )
    .await;
    assert!(matches!(
        reduced.as_slice(),
        [Op::Cancel(0), Op::Shutdown(true)]
    ));
}
#[tokio::test]
async fn generated_lifecycle_sequences_preserve_invariants() {
    let replay = std::env::var("MINI_VLLM_LIFECYCLE_CASE").ok();
    let cases: Vec<Vec<Op>> = if let Some(path) = replay {
        vec![serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()]
    } else {
        let seed = std::env::var("MINI_VLLM_LIFECYCLE_SEED")
            .ok()
            .map(|s| s.parse().unwrap())
            .unwrap_or(713);
        (0..24).map(|i| generate(seed + i * 104729, 64)).collect()
    };
    for ops in cases {
        if let Err(error) = execute(&ops).await {
            let original = ops.clone();
            let reduced = minimize(ops, &error).await;
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/lifecycle-failures");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("original.json"),
                serde_json::to_vec_pretty(&original).unwrap(),
            )
            .unwrap();
            std::fs::write(
                dir.join("minimized.json"),
                serde_json::to_vec_pretty(&reduced).unwrap(),
            )
            .unwrap();
            panic!("{error}; replay {}", dir.join("minimized.json").display());
        }
    }
}
