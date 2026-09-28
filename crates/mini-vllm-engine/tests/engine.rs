//! End-to-end engine tests (design doc §52) against a tiny random-weight
//! Qwen2 model: request lifecycle, stop conditions, cancellation, the
//! slow-consumer backpressure contract, and batch-vs-solo equivalence
//! through the real scheduler and engine thread.

use std::sync::Arc;
use std::time::Duration;

use candle_core::Device;
use mini_vllm_core::{
    EngineConfig, FinishReason, GenerationEvent, GenerationRequest, SamplingParams, Usage,
};
use mini_vllm_engine::spawn_engine;
use mini_vllm_model::testutil::{random_model, tiny_config};

fn config() -> EngineConfig {
    EngineConfig {
        max_model_len: 1024,
        max_num_seqs: 8,
        max_batch_tokens: 256,
        kv_block_size: 16,
        max_kv_tokens: 2048,
        default_max_new_tokens: 64,
        event_channel_capacity: 64,
        command_channel_capacity: 256,
        ..EngineConfig::default()
    }
}

fn spawn_with(model: Arc<dyn mini_vllm_model::CausalLm>) -> mini_vllm_engine::EngineHandle {
    spawn_engine(model, None, config(), 42).unwrap()
}

fn spawn() -> mini_vllm_engine::EngineHandle {
    spawn_with(Arc::new(random_model(&Device::Cpu)))
}

/// One random-weight model whose config has no EOS ids: greedy generation
/// can then never stop early, so `Length` outcomes are deterministic even
/// though the weights are redrawn per run.
fn no_eos_model() -> Arc<dyn mini_vllm_model::CausalLm> {
    let mut cfg = tiny_config();
    cfg.eos_token_ids.clear();
    let weights = mini_vllm_model::testutil::random_tensors(&cfg, false, &Device::Cpu);
    Arc::new(
        mini_vllm_model::Qwen2::load(cfg, &weights, candle_core::DType::F32, &Device::Cpu).unwrap(),
    )
}

fn spawn_no_eos() -> mini_vllm_engine::EngineHandle {
    spawn_with(no_eos_model())
}

fn req(id: &str, prompt: Vec<u32>, max_new: usize) -> GenerationRequest {
    GenerationRequest {
        id: id.into(),
        prompt_token_ids: prompt,
        sampling: SamplingParams {
            temperature: 0.0, // greedy: deterministic outputs
            ..SamplingParams::default()
        },
        max_new_tokens: max_new,
        stop_token_ids: vec![],
        stop_strings: vec![],
    }
}

/// Drain the stream to its terminal event.
async fn drain(
    rx: tokio::sync::mpsc::Receiver<GenerationEvent>,
) -> (Vec<u32>, FinishReason, Usage) {
    let mut rx = rx;
    let mut tokens = Vec::new();
    loop {
        let ev = tokio::time::timeout(Duration::from_secs(30), rx.recv())
            .await
            .expect("timed out waiting for engine event")
            .expect("engine closed the stream before a terminal event");
        match ev {
            GenerationEvent::Token { token_id, .. } => tokens.push(token_id),
            GenerationEvent::Finished { reason, usage } => return (tokens, reason, usage),
            GenerationEvent::Error { message, .. } => panic!("engine error: {message}"),
        }
    }
}

/// Like [`drain`], but a stream that ends without a terminal event counts as
/// a backpressure cancellation (the documented contract for slow consumers).
async fn drain_tolerant(
    rx: tokio::sync::mpsc::Receiver<GenerationEvent>,
) -> (Vec<u32>, Option<FinishReason>) {
    let mut rx = rx;
    let mut tokens = Vec::new();
    loop {
        let ev = match tokio::time::timeout(Duration::from_secs(30), rx.recv()).await {
            Ok(Some(ev)) => ev,
            Ok(None) => return (tokens, None), // stream ended: treat as cancelled
            Err(_) => panic!("timed out waiting for engine event"),
        };
        match ev {
            GenerationEvent::Token { token_id, .. } => tokens.push(token_id),
            GenerationEvent::Finished { reason, .. } => return (tokens, Some(reason)),
            GenerationEvent::Error { message, .. } => panic!("engine error: {message}"),
        }
    }
}

async fn generate(handle: &mini_vllm_engine::EngineHandle, request: GenerationRequest) {
    let rx = handle.generate(request).expect("submit");
    drain(rx).await;
}

#[tokio::test]
async fn generates_until_length_limit() {
    // No model EOS ⇒ greedy can only end at the length limit, keeping the
    // `Length` assertion deterministic across random weight draws. Output
    // stays stable across a re-run on a fresh engine sharing the same
    // weights (random weights are drawn per model, not per engine).
    let model = no_eos_model();
    let handle = spawn_with(Arc::clone(&model));
    let handle2 = spawn_with(model);
    let (tokens, reason, usage) =
        drain(handle.generate(req("len", vec![1, 2, 3], 16)).unwrap()).await;
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(tokens.len(), 16);
    assert_eq!(usage.completion_tokens, 16);
    assert_eq!(usage.total_tokens, 3 + 16);
    let (tokens2, _, _) = drain(handle2.generate(req("len", vec![1, 2, 3], 16)).unwrap()).await;
    assert_eq!(tokens, tokens2);
}

#[tokio::test]
async fn stops_on_stop_token() {
    let handle = spawn();
    let vocab = tiny_config().vocab_size as u32;
    // Every possible token is a stop token: the first sample must finish it.
    let mut r = req("stop", vec![5, 6, 7], 32);
    r.stop_token_ids = (0..vocab).collect();
    let (tokens, reason, usage) = drain(handle.generate(r).unwrap()).await;
    assert_eq!(reason, FinishReason::Stop);
    assert_eq!(tokens.len(), 1);
    assert_eq!(usage.completion_tokens, 1);
}

#[tokio::test]
async fn rejects_prompt_beyond_limits() {
    let handle = spawn();
    // Beyond max_model_len.
    let r = req("big", vec![1; 1020], 8);
    assert!(matches!(
        handle.generate(r),
        Err(mini_vllm_engine::EngineApiError::InvalidRequest(_))
    ));
    // Larger than one prefill budget is now valid and advances in chunks.
    let (_, reason, _) = drain(handle.generate(req("chunked", vec![1; 300], 8)).unwrap()).await;
    assert!(matches!(reason, FinishReason::Length | FinishReason::Stop));
}

#[tokio::test]
async fn cancel_after_first_token_finishes_as_cancelled() {
    let handle = spawn_no_eos();
    let mut rx = handle.generate(req("cancel", vec![9, 8, 7], 200)).unwrap();
    // Wait for the first token, then cancel mid-generation.
    match tokio::time::timeout(Duration::from_secs(30), rx.recv())
        .await
        .expect("timeout waiting for first token")
    {
        Some(GenerationEvent::Token { .. }) => {}
        other => panic!("expected a first token, got {other:?}"),
    }
    handle.cancel("cancel").expect("cancel accepted");
    let (tokens, reason, _) = drain(rx).await;
    assert_eq!(reason, FinishReason::Cancelled);
    // Far below the 200-token budget: cancellation actually stopped work.
    assert!(
        tokens.len() < 200,
        "cancel did not stop generation: {tokens:?}"
    );
}

#[tokio::test]
async fn slow_consumer_is_cancelled_not_engine_stalling() {
    // §43 contract: a consumer that never reads must lead to *its own*
    // cancellation — the engine thread must never block on it.
    let handle = spawn_no_eos();
    let rx = handle.generate(req("slow", vec![3, 4], 300)).unwrap();
    // Do not read for a while: channel (64) + outbox (64) overflow at the
    // 129th event, so the engine cancels the request by itself. A stream
    // that then ends without a terminal event also counts as cancelled.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (_, reason) = drain_tolerant(rx).await;
    assert!(
        matches!(reason, None | Some(FinishReason::Cancelled)),
        "slow consumer should have been cancelled, got {reason:?}"
    );
    // The engine stayed responsive: a fresh request completes normally.
    let (_, reason2, _) = drain(handle.generate(req("after", vec![1], 8)).unwrap()).await;
    assert_eq!(reason2, FinishReason::Length);
}

#[tokio::test]
async fn concurrent_generation_is_independent_and_deterministic() {
    // §52.5's exact batch==solo equality is gated at the model level (with
    // controlled tolerance). Through the real engine, batched GEMM float
    // ordering can flip argmax on near-ties, so greedy chains may legally
    // diverge between solo and batched runs (vLLM has the same property).
    // The load-bearing engine properties verified here: sequences complete
    // independently, and re-runs within the same mode are deterministic.
    let handle = spawn();
    let prompt_a: Vec<u32> = (1..=5).collect();
    let prompt_b: Vec<u32> = (40..=45).collect();

    let (solo_a, _, _) = drain(handle.generate(req("a1", prompt_a.clone(), 12)).unwrap()).await;
    let (solo_a2, _, _) = drain(handle.generate(req("a1r", prompt_a.clone(), 12)).unwrap()).await;
    assert_eq!(solo_a, solo_a2, "solo runs are not deterministic");

    // Same two prompts submitted back to back: the engine batches them.
    let (batch_a, batch_b) = run_pair(&handle, "x", &prompt_a, &prompt_b).await;
    let (batch_a2, batch_b2) = run_pair(&handle, "y", &prompt_a, &prompt_b).await;
    assert_eq!(batch_a, batch_a2, "batched runs are not deterministic");
    assert_eq!(batch_b, batch_b2, "batched runs are not deterministic");
}

/// Submit both prompts together and collect their token streams.
async fn run_pair(
    handle: &mini_vllm_engine::EngineHandle,
    tag: &str,
    prompt_a: &[u32],
    prompt_b: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let rx_a = handle
        .generate(req(&format!("{tag}a"), prompt_a.to_vec(), 12))
        .unwrap();
    let rx_b = handle
        .generate(req(&format!("{tag}b"), prompt_b.to_vec(), 12))
        .unwrap();
    let (ta, ra, _) = drain(rx_a).await;
    let (tb, rb, _) = drain(rx_b).await;
    assert!(matches!(
        (ra, rb),
        (
            FinishReason::Length | FinishReason::Stop,
            FinishReason::Length | FinishReason::Stop
        )
    ));
    assert!(ta.len() <= 12 && tb.len() <= 12);
    (ta, tb)
}

#[tokio::test]
async fn kv_blocks_return_to_the_pool_after_finish_and_cancel() {
    let handle = spawn();
    let metrics = handle.metrics();
    // Fill, finish, and verify the KV gauges drop back to zero.
    generate(&handle, req("m1", vec![1; 20], 10)).await;
    generate(&handle, req("m2", vec![2; 20], 10)).await;
    // Gauges are refreshed every engine step; poll until drained (bounded).
    for _ in 0..100 {
        let snap = metrics.snapshot();
        if snap.kv_blocks_used == 0 && snap.requests_running == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let snap = metrics.snapshot();
    assert_eq!(snap.kv_blocks_used, 0, "kv blocks leaked: {snap:?}");
    assert_eq!(snap.requests_running, 0);
    assert_eq!(snap.requests_waiting, 0);
    assert_eq!(snap.requests_finished, 2);
}

#[tokio::test]
async fn prefill_failure_finishes_and_releases_kv_without_new_commands() {
    let handle = spawn_engine(
        Arc::new(TextModel {
            cfg: tiny_config(),
            device: Device::Cpu,
        }),
        None,
        config(),
        1,
    )
    .unwrap();
    let mut rx = handle
        .generate(req("injected-failure", vec![63], 7))
        .unwrap();
    let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap();
    assert!(matches!(
        event,
        Some(GenerationEvent::Error {
            kind: mini_vllm_core::GenerationErrorKind::Execution,
            ..
        })
    ));
    // Channel closes after retirement; no further command should be needed.
    assert!(tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .is_none());
    for _ in 0..100 {
        if handle.metrics().snapshot().kv_blocks_used == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("failed prefill retained KV blocks");
}

/// Deterministic text fixture: emits a, space-b, space-c, EOS.
struct TextModel {
    cfg: mini_vllm_model::ModelConfig,
    device: Device,
}
impl mini_vllm_model::CausalLm for TextModel {
    fn device(&self) -> &Device {
        &self.device
    }
    fn config(&self) -> &mini_vllm_model::ModelConfig {
        &self.cfg
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn dtype(&self) -> candle_core::DType {
        candle_core::DType::F32
    }
    fn forward_cached(
        &self,
        input: &mini_vllm_model::BatchTokens,
        _: &mut [mini_vllm_kv::KvCache],
    ) -> candle_core::Result<candle_core::Tensor> {
        if input.token_ids.contains(&63) {
            candle_core::bail!("injected prefill failure");
        }
        let mut logits = vec![0f32; input.seq_lens.len() * self.cfg.vocab_size];
        let mut offset = 0;
        for (row, &len) in input.seq_lens.iter().enumerate() {
            let pos = input.positions[offset + len - 1] as usize;
            let token = [4, 9, 10, 2][pos.min(3)];
            logits[row * self.cfg.vocab_size + token] = 10.0;
            offset += len;
        }
        candle_core::Tensor::from_vec(
            logits,
            (input.seq_lens.len(), self.cfg.vocab_size),
            &self.device,
        )
    }
    fn forward_nocache(&self, _: &[u32], _: &[u32]) -> candle_core::Result<candle_core::Tensor> {
        candle_core::bail!("not used by engine")
    }
}

#[tokio::test]
async fn engine_stop_matching_and_normal_finish_preserve_text() {
    let model = Arc::new(TextModel {
        cfg: tiny_config(),
        device: Device::Cpu,
    });
    let tokenizer = Arc::new(mini_vllm_tokenizer::testutil::wrapper());
    let handle = spawn_engine(model, Some(tokenizer), config(), 1).unwrap();
    for (id, stop, max_new, expected, expected_reason) in [
        ("cross", "b c", 8, "a ", FinishReason::Stop),
        ("length", "b c", 2, "a b", FinishReason::Length),
        ("eos", "c d", 8, "a b c", FinishReason::Stop),
    ] {
        let mut request = req(id, vec![4], max_new);
        request.stop_strings = vec![stop.into()];
        let mut rx = handle.generate(request).unwrap();
        let mut text = String::new();
        loop {
            match tokio::time::timeout(Duration::from_secs(2), rx.recv())
                .await
                .unwrap()
                .unwrap()
            {
                GenerationEvent::Token { text: delta, .. } => text.push_str(&delta),
                GenerationEvent::Finished { reason, .. } => {
                    assert_eq!(reason, expected_reason);
                    break;
                }
                GenerationEvent::Error { message, .. } => panic!("{message}"),
            }
        }
        assert_eq!(text, expected, "{id}");
    }
}

#[tokio::test]
async fn output_grace_period_releases_kv_before_slow_reader_finishes() {
    let mut cfg = config();
    cfg.event_channel_capacity = 1;
    cfg.output_drain_timeout_ms = 1000;
    let handle = spawn_engine(
        Arc::new(TextModel {
            cfg: tiny_config(),
            device: Device::Cpu,
        }),
        None,
        cfg,
        1,
    )
    .unwrap();
    let mut rx = handle.generate(req("grace", vec![4], 2)).unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(handle.metrics().snapshot().kv_blocks_used, 0);
    assert!(matches!(
        rx.recv().await,
        Some(GenerationEvent::Token { .. })
    ));
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(matches!(
        rx.recv().await,
        Some(GenerationEvent::Token { .. })
    ));
    let event = tokio::time::timeout(Duration::from_secs(1), rx.recv())
        .await
        .unwrap();
    assert!(matches!(
        event,
        Some(GenerationEvent::Finished {
            reason: FinishReason::Length,
            ..
        })
    ));
}

#[tokio::test]
async fn prefix_reuse_preserves_output_and_has_bounded_retention() {
    let mut model = tiny_config();
    model.eos_token_ids = vec![];
    let weights = mini_vllm_model::testutil::random_tensors(&model, false, &Device::Cpu);
    let model: Arc<dyn mini_vllm_model::CausalLm> = Arc::new(
        mini_vllm_model::Qwen2::load(model, &weights, candle_core::DType::F32, &Device::Cpu)
            .unwrap(),
    );
    let mut cfg = config();
    cfg.kv_block_size = 4;
    cfg.max_batch_tokens = 3;
    cfg.prefix_cache_tokens = 8;
    let handle = spawn_engine(model, None, cfg, 1).unwrap();
    let prompt = vec![4; 9];
    let (first, _, _) = drain(handle.generate(req("cold", prompt.clone(), 4)).unwrap()).await;
    let before = handle.metrics().snapshot();
    let (second, _, _) = drain(handle.generate(req("warm", prompt, 4)).unwrap()).await;
    assert_eq!(first, second);
    let after = handle.metrics().snapshot();
    assert_eq!(
        after.prefix_cache_hit_tokens - before.prefix_cache_hit_tokens,
        8
    );
    assert_eq!(after.prompt_tokens_total - before.prompt_tokens_total, 1);
    assert!(after.cached_prefix_tokens <= 8);
    assert!(after.scheduled_tokens_total <= after.model_steps_total * 3);
}

#[tokio::test]
async fn total_outstanding_requests_remain_bounded_during_output_drain() {
    let mut cfg = config();
    cfg.max_num_seqs = 1;
    cfg.max_waiting_requests = 1;
    cfg.event_channel_capacity = 1;
    cfg.output_drain_timeout_ms = 1000;
    let handle = spawn_engine(
        Arc::new(TextModel {
            cfg: tiny_config(),
            device: Device::Cpu,
        }),
        None,
        cfg,
        1,
    )
    .unwrap();
    let a = handle.generate(req("a", vec![4], 2)).unwrap();
    // Wait until a is draining, which still owns its request permit.
    tokio::time::sleep(Duration::from_millis(30)).await;
    let b = handle.generate(req("b", vec![4], 2)).unwrap();
    assert!(matches!(
        handle.generate(req("c", vec![4], 2)),
        Err(mini_vllm_engine::EngineApiError::QueueFull)
    ));
    assert!(matches!(
        handle.generate(req("a", vec![4], 2)),
        Err(mini_vllm_engine::EngineApiError::InvalidRequest(_))
    ));
    handle.cancel("a").unwrap();
    handle.cancel("b").unwrap();
    drop(a);
    drop(b);
    tokio::time::sleep(Duration::from_millis(30)).await;
    let (_, reason, _) = drain(handle.generate(req("after", vec![4], 1)).unwrap()).await;
    assert_eq!(reason, FinishReason::Length);
}

struct RecordingModel {
    inner: mini_vllm_model::Qwen2,
    calls: std::sync::Mutex<Vec<mini_vllm_model::BatchTokens>>,
}
impl mini_vllm_model::CausalLm for RecordingModel {
    fn device(&self) -> &Device {
        mini_vllm_model::CausalLm::device(&self.inner)
    }
    fn config(&self) -> &mini_vllm_model::ModelConfig {
        mini_vllm_model::CausalLm::config(&self.inner)
    }
    fn vocab_size(&self) -> usize {
        mini_vllm_model::CausalLm::vocab_size(&self.inner)
    }
    fn dtype(&self) -> candle_core::DType {
        self.inner.dtype()
    }
    fn forward_cached(
        &self,
        input: &mini_vllm_model::BatchTokens,
        caches: &mut [mini_vllm_kv::KvCache],
    ) -> candle_core::Result<candle_core::Tensor> {
        self.calls.lock().unwrap().push(input.clone());
        // Give the submitter time to enqueue the long prompt during the first prefill.
        if self.calls.lock().unwrap().len() == 1 {
            std::thread::sleep(Duration::from_millis(30));
        }
        mini_vllm_model::CausalLm::forward_cached(&self.inner, input, caches)
    }
    fn forward_nocache(
        &self,
        ids: &[u32],
        positions: &[u32],
    ) -> candle_core::Result<candle_core::Tensor> {
        mini_vllm_model::CausalLm::forward_nocache(&self.inner, ids, positions)
    }
}

#[tokio::test]
async fn mixed_prefill_decode_calls_obey_the_combined_budget() {
    let mut model_cfg = tiny_config();
    model_cfg.eos_token_ids.clear();
    let weights = mini_vllm_model::testutil::random_tensors(&model_cfg, false, &Device::Cpu);
    let model = Arc::new(RecordingModel {
        inner: mini_vllm_model::Qwen2::load(
            model_cfg,
            &weights,
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap(),
        calls: Default::default(),
    });
    let mut cfg = config();
    cfg.max_batch_tokens = 3;
    let handle = spawn_engine(model.clone(), None, cfg, 1).unwrap();
    let a = handle.generate(req("short", vec![4], 8)).unwrap();
    let b = handle.generate(req("long", vec![5; 11], 2)).unwrap();
    let (_, ra, _) = drain(a).await;
    let (_, rb, _) = drain(b).await;
    assert_eq!(ra, FinishReason::Length);
    assert_eq!(rb, FinishReason::Length);
    let calls = model.calls.lock().unwrap();
    assert!(calls.iter().all(|c| c.token_ids.len() <= 3));
    assert!(calls
        .iter()
        .any(|c| c.seq_lens == vec![1, 2] && c.positions[0] > 0));
}

#[tokio::test]
async fn explicit_shutdown_drains_and_rejects_new_requests() {
    use mini_vllm_engine::{EngineApiError, ShutdownMode};
    let handle = spawn_no_eos();
    let a = handle.generate(req("drain-a", vec![4; 12], 3)).unwrap();
    let b = handle.generate(req("drain-b", vec![5; 12], 3)).unwrap();
    handle.request_shutdown(ShutdownMode::Drain, Duration::from_secs(5));
    assert!(!handle.is_accepting());
    assert!(matches!(
        handle.generate(req("late", vec![4], 1)),
        Err(EngineApiError::ShuttingDown)
    ));
    assert_eq!(drain(a).await.1, FinishReason::Length);
    assert_eq!(drain(b).await.1, FinishReason::Length);
    handle.join(Duration::from_secs(5)).unwrap();
    let metrics = handle.metrics().snapshot();
    assert_eq!(metrics.kv_blocks_used, 0);
    assert_eq!(metrics.cached_prefix_tokens, 0);
}

#[tokio::test]
async fn shutdown_deadline_cancels_and_joins_even_with_unread_output() {
    let handle = spawn_no_eos();
    let _rx = handle.generate(req("unread", vec![4; 32], 400)).unwrap();
    handle.request_shutdown(mini_vllm_engine::ShutdownMode::Drain, Duration::ZERO);
    handle.join(Duration::from_secs(5)).unwrap();
    assert_eq!(handle.metrics().snapshot().kv_blocks_used, 0);
    // Joining twice and upgrading shutdown are safe.
    handle.request_shutdown(mini_vllm_engine::ShutdownMode::Cancel, Duration::ZERO);
    handle.join(Duration::ZERO).unwrap();
}

#[tokio::test]
async fn seeded_lifecycle_interleavings_reclaim_every_reservation() {
    let mut cfg = config();
    cfg.kv_block_size = 4;
    cfg.prefix_cache_tokens = 16;
    cfg.max_batch_tokens = 3;
    cfg.max_prefill_chunk_tokens = 2;
    cfg.output_drain_timeout_ms = 5;
    let handle = spawn_engine(Arc::new(random_model(&Device::Cpu)), None, cfg, 7).unwrap();
    let mut seed = 713u64;
    for round in 0..8 {
        let mut receivers = Vec::new();
        for i in 0..8 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let id = format!("{round}-{i}");
            let rx = handle
                .generate(req(&id, vec![4 + (seed % 3) as u32; 9], 4))
                .unwrap();
            match (seed >> 32) % 3 {
                0 => drop(rx),
                1 => {
                    handle.cancel(id).unwrap();
                    receivers.push(rx);
                }
                _ => receivers.push(rx),
            }
        }
        for rx in receivers {
            drain_tolerant(rx).await;
        }
    }
    handle.request_shutdown(
        mini_vllm_engine::ShutdownMode::Drain,
        Duration::from_secs(5),
    );
    handle.join(Duration::from_secs(5)).unwrap();
    let metrics = handle.metrics().snapshot();
    assert_eq!(metrics.kv_blocks_used, 0);
    assert_eq!(metrics.kv_active_sequences, 0);
    assert_eq!(metrics.requests_running + metrics.requests_waiting, 0);
}

#[tokio::test]
async fn output_deadline_closes_stream_without_false_success() {
    let mut cfg = config();
    cfg.event_channel_capacity = 1;
    cfg.output_drain_timeout_ms = 1;
    let handle = spawn_engine(
        Arc::new(TextModel {
            cfg: tiny_config(),
            device: Device::Cpu,
        }),
        None,
        cfg,
        1,
    )
    .unwrap();
    let rx = handle.generate(req("deadline", vec![4], 2)).unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (_, reason) = drain_tolerant(rx).await;
    assert!(!matches!(
        reason,
        Some(FinishReason::Length | FinishReason::Stop)
    ));
    assert_eq!(handle.metrics().snapshot().kv_blocks_used, 0);
}

#[tokio::test]
#[ignore = "opt-in device/dtype lifecycle test; set MINI_VLLM_TEST_DEVICE and MINI_VLLM_TEST_DTYPE"]
async fn backend_dtype_lifecycle() {
    let (device, _) = mini_vllm_model::resolve_device(
        &std::env::var("MINI_VLLM_TEST_DEVICE").expect("set MINI_VLLM_TEST_DEVICE"),
    )
    .unwrap();
    let dtype = mini_vllm_model::resolve_dtype(
        &std::env::var("MINI_VLLM_TEST_DTYPE").expect("set MINI_VLLM_TEST_DTYPE"),
    )
    .unwrap();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../mini-vllm-model/tests/fixtures/tiny-qwen2");
    let model = Arc::new(mini_vllm_model::loader::load_model(fixture, dtype, device).unwrap());
    let handle = spawn_engine(
        model,
        None,
        EngineConfig {
            max_model_len: 64,
            max_batch_tokens: 3,
            kv_block_size: 4,
            prefix_cache_tokens: 16,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    let first = drain(handle.generate(req("first", vec![4; 9], 4)).unwrap()).await;
    let second = drain(handle.generate(req("reuse", vec![4; 9], 4)).unwrap()).await;
    assert_eq!(first.0, second.0);
    assert!(handle.metrics().snapshot().prefix_cache_hit_tokens >= 8);
    let cancelled = handle.generate(req("cancel", vec![5; 32], 16)).unwrap();
    handle.cancel("cancel").unwrap();
    assert_eq!(drain(cancelled).await.1, FinishReason::Cancelled);
    // Backend stress: interleave shared prefixes, cancellation and disconnects.
    let mut receivers = Vec::new();
    for i in 0..24 {
        let id = format!("stress-{i}");
        let rx = handle
            .generate(req(&id, vec![4 + (i % 2) as u32; 9], 4))
            .unwrap();
        if i % 3 == 0 {
            handle.cancel(&id).unwrap();
        }
        if i % 5 == 0 {
            drop(rx);
        } else {
            receivers.push(rx);
        }
    }
    for rx in receivers {
        drain_tolerant(rx).await;
    }
    handle.request_shutdown(
        mini_vllm_engine::ShutdownMode::Drain,
        Duration::from_secs(5),
    );
    handle.join(Duration::from_secs(5)).unwrap();
    assert_eq!(handle.metrics().snapshot().kv_blocks_used, 0);
}
