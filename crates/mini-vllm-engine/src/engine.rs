//! The engine: a single OS thread owning all runtime state.
//!
//! Each iteration checks cancellation, retires finished work, admits waiting
//! requests, executes a mixed prefill/decode batch, retires again, and updates
//! gauges. Output drains after KV reclamation within a bounded grace period.
//! Cancellation flags bypass command-channel capacity. Prefix snapshots share
//! immutable physical pages under a separate bounded LRU budget.

use crate::lifecycle::{Registry, RequestLease};
use crate::prefix::PrefixCache;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{atomic::AtomicBool, Arc, Mutex};
use std::time::{Duration, Instant};

use mini_vllm_core::{FinishReason, GenerationEvent, GenerationRequest, RequestId, SequenceStatus};
use mini_vllm_kv::{blocks_for_tokens, KvBlockManager, KvCache};
use mini_vllm_model::CausalLm;
use mini_vllm_sampling::Sampler;
use mini_vllm_tokenizer::{IncrementalDetokenizer, TokenizerWrapper};
use tokio::sync::mpsc;
use tracing::debug;

use crate::executor::Executor;
use crate::metrics::Metrics;
use crate::request::{EngineApiError, EngineCommand};
use crate::scheduler::Scheduler;
use crate::sequence::SequenceGroup;

/// Drain accepted work up to a deadline, or cancel it at the next model-step boundary.
#[derive(Debug, Clone, Copy)]
pub enum ShutdownMode {
    Drain,
    Cancel,
}
#[derive(Default)]
struct Control {
    shutdown: Mutex<Option<(ShutdownMode, Instant)>>,
    join: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// Handle used by HTTP/CLI callers to talk to the engine loop.
#[derive(Clone)]
pub struct EngineHandle {
    control: Arc<Control>,
    cmd_tx: mpsc::Sender<EngineCommand>,
    metrics: Arc<Metrics>,
    event_capacity: usize,
    registry: Registry,
    max_outstanding: usize,
    max_model_len: usize,
    vocab_size: usize,
    default_max_new_tokens: usize,
}

impl EngineHandle {
    /// Submit a generation request; returns the bounded event stream.
    pub fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<mpsc::Receiver<GenerationEvent>, EngineApiError> {
        let control = self
            .control
            .shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if control.is_some() {
            return Err(EngineApiError::ShuttingDown);
        }
        request
            .validate(self.max_model_len)
            .map_err(|e| EngineApiError::InvalidRequest(e.to_string()))?;
        if request
            .prompt_token_ids
            .iter()
            .chain(request.stop_token_ids.iter())
            .any(|&id| id as usize >= self.vocab_size)
        {
            return Err(EngineApiError::InvalidRequest(
                "token id outside vocabulary".into(),
            ));
        }
        let lease = {
            let mut registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
            if registry.contains_key(&request.id) {
                return Err(EngineApiError::InvalidRequest(
                    "duplicate request id".into(),
                ));
            }
            if registry.len() >= self.max_outstanding {
                return Err(EngineApiError::QueueFull);
            }
            let cancelled = Arc::new(AtomicBool::new(false));
            registry.insert(request.id.clone(), Arc::clone(&cancelled));
            RequestLease {
                id: request.id.clone(),
                submitted_at: Instant::now(),
                cancelled,
                registry: Arc::clone(&self.registry),
            }
        };
        let (tx, rx) = mpsc::channel(self.event_capacity);
        self.cmd_tx
            .try_send(EngineCommand::Generate {
                request,
                events: tx,
                lease,
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => EngineApiError::QueueFull,
                mpsc::error::TrySendError::Closed(_) => EngineApiError::ShuttingDown,
            })?;
        drop(control);
        self.wake();
        Ok(rx)
    }

    /// Request cancellation; the engine stops scheduling the sequence and
    /// releases its resources.
    pub fn cancel(&self, request_id: impl Into<RequestId>) -> Result<(), EngineApiError> {
        if self.cmd_tx.is_closed() {
            return Err(EngineApiError::ShuttingDown);
        }
        if let Some(flag) = self
            .registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&request_id.into())
        {
            flag.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    fn wake(&self) {
        if let Some(join) = self
            .control
            .join
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            join.thread().unpark();
        }
    }

    pub fn is_accepting(&self) -> bool {
        !self.cmd_tx.is_closed()
            && self
                .control
                .shutdown
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none()
    }

    /// Nonblocking; cancellation takes effect between model calls. A deadline
    /// cannot interrupt a device kernel that is already executing.
    pub fn request_shutdown(&self, mode: ShutdownMode, grace: Duration) {
        let mut state = self
            .control
            .shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + grace;
        match *state {
            None => *state = Some((mode, deadline)),
            Some((old, previous)) => {
                *state = Some((
                    if matches!(mode, ShutdownMode::Cancel) {
                        mode
                    } else {
                        old
                    },
                    previous.min(deadline),
                ))
            }
        }
        drop(state);
        self.wake();
    }

    /// Blocking bounded join. Invoke with spawn_blocking from async runtimes.
    /// On timeout the thread remains joinable; it is never forcibly terminated.
    pub fn join(&self, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            let mut slot = self.control.join.lock().unwrap_or_else(|e| e.into_inner());
            match slot.as_ref() {
                None => return Ok(()),
                Some(join) if join.is_finished() => {
                    return slot
                        .take()
                        .unwrap()
                        .join()
                        .map_err(|_| "engine thread panicked".into());
                }
                _ => {}
            }
            drop(slot);
            if Instant::now() >= deadline {
                return Err("engine join timed out".into());
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    pub fn default_max_new_tokens(&self) -> usize {
        self.default_max_new_tokens
    }

    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }
}

/// Spawn the engine on a dedicated OS thread (model execution is blocking
/// work; it must never run on a Tokio worker).
pub fn spawn_engine(
    model: Arc<dyn CausalLm>,
    tokenizer: Option<Arc<TokenizerWrapper>>,
    config: mini_vllm_core::EngineConfig,
    engine_seed: u64,
) -> mini_vllm_core::Result<EngineHandle> {
    config.validate()?;
    let metrics = Arc::new(Metrics::new());
    let trace = crate::trace::TraceWriter::new_with_counters(
        config.trace_jsonl.as_deref(),
        metrics.trace_events_dropped.clone(),
        metrics.trace_writer_errors.clone(),
        metrics.trace_shutdown_timeouts.clone(),
    )
    .map_err(|e| mini_vllm_core::Error::InvalidRequest(format!("opening trace: {e}")))?;
    model
        .config()
        .validate()
        .map_err(|e| mini_vllm_core::Error::InvalidRequest(e.to_string()))?;
    let control = Arc::new(Control::default());
    let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
    let max_outstanding = config.max_num_seqs + config.max_waiting_requests;
    let max_model_len = config
        .max_model_len
        .min(model.config().max_position_embeddings);
    let vocab_size = model.vocab_size();
    let default_max_new_tokens = config.default_max_new_tokens;
    let (cmd_tx, cmd_rx) = mpsc::channel(config.command_channel_capacity);
    let event_capacity = config.event_channel_capacity;
    let eos_ids = model.config().eos_token_ids.clone();
    let executor = Executor::new(Arc::clone(&model));
    let blocks = KvBlockManager::new(config.kv_block_size, config.max_kv_tokens);
    let scheduler = Scheduler::new(config.max_num_seqs, config.max_batch_tokens);
    let engine = Engine {
        trace,
        control: Arc::clone(&control),
        cmd_rx,
        executor,
        tokenizer,
        scheduler,
        blocks,
        prefix_cache: PrefixCache::new(config.kv_block_size, config.prefix_cache_tokens),
        config,
        eos_ids,
        metrics: Arc::clone(&metrics),
        engine_seed,
        rng_counter: 0,
        finishing: Vec::new(),

        decode_cursor: 0,
        prefill_cursor: 0,
        step_id: 0,
    };
    let join = std::thread::Builder::new()
        .name("mini-vllm-engine".to_string())
        .spawn(move || engine.run())
        .map_err(|e| mini_vllm_core::Error::Capacity(format!("spawning engine: {e}")))?;
    *control.join.lock().unwrap() = Some(join);
    Ok(EngineHandle {
        control,
        cmd_tx,
        metrics,
        event_capacity,
        registry,
        max_outstanding,
        max_model_len,
        vocab_size,
        default_max_new_tokens,
    })
}

struct Engine {
    trace: crate::trace::TraceWriter,
    control: Arc<Control>,
    cmd_rx: mpsc::Receiver<EngineCommand>,
    executor: Executor,
    tokenizer: Option<Arc<TokenizerWrapper>>,
    scheduler: Scheduler,
    blocks: KvBlockManager,
    config: mini_vllm_core::EngineConfig,
    eos_ids: Vec<u32>,
    metrics: Arc<Metrics>,
    engine_seed: u64,
    rng_counter: u64,
    finishing: Vec<(SequenceGroup, Instant)>,
    prefix_cache: PrefixCache,
    decode_cursor: usize,
    prefill_cursor: usize,
    step_id: u64,
}

impl Engine {
    fn run(mut self) {
        tracing::info!(
            max_num_seqs = self.scheduler.max_num_seqs(),
            max_batch_tokens = self.scheduler.max_batch_tokens(),
            kv_block_size = self.blocks.block_size(),
            kv_blocks_total = self.blocks.total_blocks(),
            "engine loop started"
        );
        loop {
            let shutdown = *self
                .control
                .shutdown
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if shutdown.is_some() {
                self.cmd_rx.close();
            }
            // 1. Drain every pending command without blocking.
            let mut disconnected = false;
            for _ in 0..self.config.command_channel_capacity {
                match self.cmd_rx.try_recv() {
                    Ok(cmd) => self.handle_command(cmd),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            if shutdown.is_some_and(|(mode, deadline)| {
                matches!(mode, ShutdownMode::Cancel) || Instant::now() >= deadline
            }) || (disconnected && shutdown.is_none())
            {
                self.shutdown_all();
                break;
            }

            // 2. One scheduling iteration.
            let worked = self.step();

            if shutdown.is_some()
                && self.scheduler.waiting_len() == 0
                && self.scheduler.running_len() == 0
                && self.finishing.is_empty()
                && self.cmd_rx.is_empty()
            {
                self.prefix_cache.clear();

                self.update_gauges();
                break;
            }
            // 3. Idle handling: block on commands, or pause if capacity-blocked.
            if !worked {
                std::thread::park_timeout(Duration::from_millis(10));
            }
        }
        tracing::info!("engine loop stopped");
    }

    fn handle_command(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::Generate {
                request,
                events,
                lease,
            } => self.enqueue_request_with_lease(request, events, Some(lease)),
            EngineCommand::Cancel { request_id } => {
                let in_running = self.scheduler.cancel(&request_id);
                debug!(request_id = %request_id, was_running = in_running, "cancellation requested");
            }
        }
    }

    #[cfg(test)]
    fn enqueue_request(
        &mut self,
        request: GenerationRequest,
        events: mpsc::Sender<GenerationEvent>,
    ) {
        self.enqueue_request_with_lease(request, events, None);
    }

    fn enqueue_request_with_lease(
        &mut self,
        request: GenerationRequest,
        events: mpsc::Sender<GenerationEvent>,
        lease: Option<RequestLease>,
    ) {
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        let reject = |kind, msg: String| {
            tracing::warn!(request_id = %request.id, "request rejected: {msg}");
            self.metrics.requests_failed.fetch_add(1, Ordering::Relaxed);
            // Fresh channel: try_send cannot be full here; a closed channel
            // means the caller already left, which is fine.
            let _ = events.try_send(GenerationEvent::Error { kind, message: msg });
        };
        let submitted = lease
            .as_ref()
            .map(|lease| lease.submitted_at)
            .unwrap_or_else(Instant::now);
        let elapsed = submitted.elapsed();
        if (self.config.queue_timeout_ms > 0
            && elapsed >= Duration::from_millis(self.config.queue_timeout_ms))
            || (self.config.request_timeout_ms > 0
                && elapsed >= Duration::from_millis(self.config.request_timeout_ms))
        {
            reject(
                mini_vllm_core::GenerationErrorKind::Timeout,
                "request expired in command queue".into(),
            );
            return;
        }
        if let Err(e) = request.validate(self.effective_max_model_len()) {
            reject(
                mini_vllm_core::GenerationErrorKind::InvalidRequest,
                e.to_string(),
            );
            return;
        }
        if self.scheduler.waiting_len() >= self.config.max_waiting_requests {
            reject(
                mini_vllm_core::GenerationErrorKind::Overloaded,
                "waiting queue is full".into(),
            );
            return;
        }
        if !self
            .blocks
            .can_fit_horizon(request.prompt_token_ids.len(), request.max_new_tokens)
        {
            reject(
                mini_vllm_core::GenerationErrorKind::InvalidRequest,
                format!(
                "prompt_len({}) + max_new_tokens({}) exceeds KV capacity ({} tokens in {} blocks)",
                request.prompt_token_ids.len(),
                request.max_new_tokens,
                self.blocks.total_blocks() * self.blocks.block_size(),
                self.blocks.total_blocks()
            ),
            );
            return;
        }

        let seed = request.sampling.seed.unwrap_or_else(|| self.next_seed());
        let sampler = Sampler::from_seed(seed);
        let detokenizer = self
            .tokenizer
            .as_ref()
            .map(|t| IncrementalDetokenizer::new(Arc::clone(t)));
        let mut stop_ids = request.stop_token_ids.clone();
        for eos in &self.eos_ids {
            if !stop_ids.contains(eos) {
                stop_ids.push(*eos);
            }
        }
        debug!(
            request_id = %request.id,
            prompt_len = request.prompt_token_ids.len(),
            max_new_tokens = request.max_new_tokens,
            "request queued"
        );
        let mut seq = SequenceGroup::new(
            request,
            sampler,
            detokenizer,
            events,
            stop_ids,
            self.config.event_channel_capacity,
        );
        self.trace
            .emit(|| serde_json::json!({"event":"queued", "request_id":seq.request.id}));
        seq.created_at = submitted;
        seq.lease = lease;
        if self.config.trace_requests {
            tracing::info!(target: "mini_vllm_trace", request_id = %seq.request.id, "request queued");
        }
        self.scheduler.enqueue(seq);
    }

    fn effective_max_model_len(&self) -> usize {
        self.config
            .max_model_len
            .min(self.executor.model().config().max_position_embeddings)
    }

    fn next_seed(&mut self) -> u64 {
        self.rng_counter = self.rng_counter.wrapping_add(1);
        self.engine_seed ^ self.rng_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    // ------------------------------------------------------------------
    // Scheduling iteration
    // ------------------------------------------------------------------

    fn step(&mut self) -> bool {
        self.scheduler.cancel_disconnected();
        self.scheduler
            .expire(self.config.queue_timeout_ms, self.config.request_timeout_ms);
        self.retire();
        self.admit_waiting();
        let did = self.run_mixed_step();
        // Failures also create terminal sequences; retire before any idle wait.
        self.retire();
        self.update_gauges();
        did
    }

    /// Release model resources before waiting for output delivery.
    fn retire(&mut self) {
        let mut retired = self.scheduler.take_retired_waiting();
        retired.extend(self.scheduler.retire_finished());
        for mut seq in retired {
            let _ = self.blocks.release(&seq.request.id);
            drop(seq.take_cache());
            seq.prefix_pin = None;
            self.trace.emit(|| {
                serde_json::json!({"event":"retired", "request_id":seq.request.id,
                "reason":seq.finish_reason.map(|r| r.as_str()), "error":seq.failure,
                "prefix_tokens":self.prefix_cache.tokens()})
            });
            if self.config.trace_requests {
                tracing::info!(target: "mini_vllm_trace", request_id = %seq.request.id, reason = ?seq.finish_reason, "KV released; output draining");
            }
            self.finishing.push((seq, Instant::now()));
        }
        let mut keep = Vec::new();
        for (mut seq, started) in self.finishing.drain(..) {
            if !seq.drain_terminal() {
                if started.elapsed() < Duration::from_millis(self.config.output_drain_timeout_ms) {
                    keep.push((seq, started));
                    continue;
                }
                seq.expire_delivery();
            }
            self.metrics.request_latency_us_sum.fetch_add(
                seq.created_at.elapsed().as_micros() as u64,
                Ordering::Relaxed,
            );
            self.metrics
                .request_latency_count
                .fetch_add(1, Ordering::Relaxed);
            match seq.finish_reason {
                Some(FinishReason::Stop | FinishReason::Length) => {
                    self.metrics
                        .requests_finished
                        .fetch_add(1, Ordering::Relaxed);
                }
                Some(FinishReason::Cancelled) => {
                    self.metrics
                        .requests_cancelled
                        .fetch_add(1, Ordering::Relaxed);
                }
                _ => {
                    self.metrics.requests_failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.finishing = keep;
    }

    /// Move waiting → running while capacity allows, then allocate KV blocks
    /// and caches for the newly admitted sequences.
    ///
    /// Reserve the horizon minus a pinned retained prefix. Physical pages are allocated
    /// lazily; contiguous reference storage is allocated at admission.
    fn admit_waiting(&mut self) {
        let block_size = self.blocks.block_size();
        let admitted = {
            let mut available = self.blocks.free_block_count();
            self.scheduler.admit_with_lookahead(
                self.config.admission_lookahead,
                std::time::Duration::from_millis(self.config.admission_max_wait_ms),
                |req| {
                    let shared = self.prefix_cache.matched_tokens(&req.prompt_token_ids);
                    let needed = blocks_for_tokens(
                        req.prompt_token_ids.len() + req.max_new_tokens - shared,
                        block_size,
                    );
                    if needed > available {
                        return false;
                    }
                    available -= needed;
                    true
                },
            )
        };
        if admitted == 0 {
            return;
        }

        let cfg = self.executor.model().config();
        let (num_layers, kv_heads, head_dim) = (
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        );
        let dtype = self.executor.dtype();
        let device = self.executor.model().device().clone();

        for idx in 0..self.scheduler.running_len() {
            let (id, horizon, needs_cache, is_prefill) = {
                let s = &self.scheduler.running()[idx];
                (
                    s.request.id.clone(),
                    s.kv_capacity(),
                    !s.has_cache(),
                    s.status == SequenceStatus::Prefill,
                )
            };
            if !is_prefill || !needs_cache {
                continue;
            }
            let prompt = &self.scheduler.running()[idx].request.prompt_token_ids;
            let shared_len = self.prefix_cache.matched_tokens(prompt);
            if let Err(e) = self.blocks.allocate(&id, horizon - shared_len) {
                tracing::error!(request_id = %id, error = %e, "block allocation failed after admission");
                let seq = self.scheduler.running_get_mut(idx).expect("idx valid");
                seq.finish_reason = Some(FinishReason::Error);
                seq.status = SequenceStatus::Failed;
                continue;
            }
            let fresh = if self.config.paged_kv {
                KvCache::new_paged(num_layers, horizon, block_size)
            } else {
                KvCache::new(num_layers, kv_heads, head_dim, horizon, dtype, &device)
            };
            match fresh {
                Ok(mut cache) => {
                    cache.track_allocations(Arc::clone(&self.metrics.kv_storage_allocations_total));
                    if shared_len > 0 {
                        let prompt = &self.scheduler.running()[idx].request.prompt_token_ids;
                        match self.prefix_cache.borrow(prompt, horizon) {
                            Ok(Some((shared, pin))) => {
                                let seq = self.scheduler.running_get_mut(idx).unwrap();
                                seq.prefill_position = shared_len;
                                seq.prefix_pin = Some(pin);
                                self.metrics
                                    .prefix_cache_hit_tokens
                                    .fetch_add(shared_len as u64, Ordering::Relaxed);
                                cache = shared;
                                if self.config.trace_requests {
                                    tracing::info!(target: "mini_vllm_trace", request_id = %id, prefix_tokens = shared_len, "prefix hit");
                                }
                            }
                            _ => {
                                let seq = self.scheduler.running_get_mut(idx).unwrap();
                                seq.failure =
                                    Some("failed to attach reserved shared prefix".into());
                                finish_sequence(seq, FinishReason::Error);
                                continue;
                            }
                        }
                    }
                    self.scheduler
                        .running_get_mut(idx)
                        .expect("idx valid")
                        .restore_cache(cache);
                    self.trace.emit(|| {
                        serde_json::json!({"event":"admitted", "request_id":id,
                        "reserved_tokens":horizon-shared_len, "shared_prefix_tokens":shared_len})
                    });
                    if self.config.trace_requests {
                        tracing::info!(target: "mini_vllm_trace", request_id = %id, reserved_tokens = horizon - shared_len, "request admitted");
                    }
                }
                Err(e) => {
                    tracing::error!(request_id = %id, error = %e, "kv cache allocation failed");
                    let seq = self.scheduler.running_get_mut(idx).expect("idx valid");
                    seq.finish_reason = Some(FinishReason::Error);
                    seq.status = SequenceStatus::Failed;
                }
            }
        }
    }

    /// Decode priority with a strict combined token budget. Rotate decode
    /// selection when the budget is smaller than the running set, and leave
    /// one token for waiting prefills when possible.
    fn run_mixed_step(&mut self) -> bool {
        self.step_id = self.step_id.wrapping_add(1);
        let mut budget = self.config.max_batch_tokens;
        let mut candidates = self.scheduler.decode_candidates();
        if !candidates.is_empty() {
            let offset = self.decode_cursor % candidates.len();
            candidates.rotate_left(offset);
        }
        let prefill = self.scheduler.prefill_candidate().is_some();
        let decode_budget = if prefill && budget == 1 && self.step_id.is_multiple_of(2) {
            0
        } else if prefill && budget > 1 {
            budget - 1
        } else {
            budget
        };
        candidates.truncate(decode_budget);
        self.decode_cursor = self.decode_cursor.wrapping_add(candidates.len());
        let mut plan: Vec<(usize, usize)> = candidates.into_iter().map(|idx| (idx, 1)).collect();
        budget -= plan.len();
        let mut prefills: Vec<_> = self
            .scheduler
            .running()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == SequenceStatus::Prefill && !s.is_terminal())
            .map(|(idx, _)| idx)
            .collect();
        if !prefills.is_empty() {
            let offset = self.prefill_cursor % prefills.len();
            prefills.rotate_left(offset);
        }
        let mut served = 0;
        for idx in prefills {
            if budget == 0 {
                break;
            }
            let s = &self.scheduler.running()[idx];
            let n = (s.prompt_len() - s.prefill_position)
                .min(budget)
                .min(self.config.max_prefill_chunk_tokens);
            if n > 0 {
                plan.push((idx, n));
                budget -= n;
                served += 1;
            }
        }
        self.prefill_cursor = self.prefill_cursor.wrapping_add(served);
        if plan.is_empty() {
            return false;
        }
        for (idx, count) in &plan {
            let seq = &self.scheduler.running()[*idx];
            let position = if seq.status == SequenceStatus::Prefill {
                seq.prefill_position
            } else {
                seq.next_input_position() as usize
            };
            self.trace.emit(|| serde_json::json!({"event":"scheduled", "step":self.step_id, "request_id":seq.request.id,
                "phase":format!("{:?}",seq.status), "position":position, "tokens":count,
                "pages_before":position.div_ceil(self.config.kv_block_size),
                "pages_after":(position+count).div_ceil(self.config.kv_block_size)}));
        }
        // Borrow in sequence order; every output maps to this sorted plan.
        plan.sort_unstable_by_key(|p| p.0);
        if self.config.trace_requests {
            for (idx, count) in &plan {
                let s = &self.scheduler.running()[*idx];
                let position = if s.status == SequenceStatus::Prefill {
                    s.prefill_position
                } else {
                    s.next_input_position() as usize
                };
                tracing::info!(target: "mini_vllm_trace", step = self.step_id, request_id = %s.request.id,
                    phase = ?s.status, position, input_tokens = count,
                    pages_before = position.div_ceil(self.config.kv_block_size),
                    pages_after = (position + count).div_ceil(self.config.kv_block_size), "request step");
            }
        }
        let counts: Vec<_> = plan.iter().map(|p| p.1).collect();
        let kinds: Vec<_> = plan
            .iter()
            .map(|(idx, _)| self.scheduler.running()[*idx].status)
            .collect();
        let outcome = {
            let mut seqs: Vec<_> = self
                .scheduler
                .running_iter_mut()
                .enumerate()
                .filter(|(i, _)| plan.iter().any(|p| p.0 == *i))
                .map(|(_, s)| s)
                .collect();
            self.executor.run_mixed_batch(&mut seqs, &counts)
        };
        let prefill_tokens: usize = counts
            .iter()
            .zip(&kinds)
            .filter(|(_, s)| **s == SequenceStatus::Prefill)
            .map(|(n, _)| *n)
            .sum();
        let decodes = kinds
            .iter()
            .filter(|s| **s == SequenceStatus::Running)
            .count();
        self.metrics
            .scheduled_tokens_total
            .fetch_add(counts.iter().sum::<usize>() as u64, Ordering::Relaxed);
        self.metrics
            .model_steps_total
            .fetch_add(1, Ordering::Relaxed);
        if prefill_tokens > 0 {
            self.metrics
                .prefill_steps_total
                .fetch_add(1, Ordering::Relaxed);
        }
        if decodes > 0 {
            self.metrics
                .decode_steps_total
                .fetch_add(1, Ordering::Relaxed);
            self.metrics
                .decode_batch_tokens_total
                .fetch_add(decodes as u64, Ordering::Relaxed);
        }
        match outcome {
            Ok(tokens) => {
                for ((idx, n), token) in plan.into_iter().zip(tokens) {
                    self.after_chunk(idx, n, token);
                }
            }
            Err(error) => {
                tracing::warn!(%error,"mixed batch failed; caches rolled back before isolated retries");
                for (idx, n) in plan {
                    let result = self.executor.run_mixed_batch(
                        &mut [self.scheduler.running_get_mut(idx).expect("idx valid")],
                        &[n],
                    );
                    match result {
                        Ok(tokens) => self.after_chunk(idx, n, tokens[0]),
                        Err(e) => {
                            tracing::warn!(%e,"isolated forward failed");
                            let seq = self.scheduler.running_get_mut(idx).expect("idx valid");
                            seq.failure = Some(format!("model execution failed: {e}"));
                            finish_sequence(seq, FinishReason::Error);
                        }
                    }
                }
            }
        }
        true
    }

    fn after_chunk(&mut self, idx: usize, n: usize, token: Option<u32>) {
        self.scheduler
            .expire(self.config.queue_timeout_ms, self.config.request_timeout_ms);
        let seq = self.scheduler.running_get_mut(idx).expect("idx valid");
        if seq.is_terminal() {
            return;
        }
        if seq.status == SequenceStatus::Prefill {
            seq.prefill_position += n;
            self.metrics
                .prompt_tokens_total
                .fetch_add(n as u64, Ordering::Relaxed);
            let prefix_len = seq.prefill_position.min(seq.prompt_len().saturating_sub(1))
                / self.config.kv_block_size
                * self.config.kv_block_size;
            if prefix_len > 0 && self.config.prefix_cache_tokens > 0 {
                if let Some(mut cache) = seq.take_cache() {
                    if let Err(error) = self
                        .prefix_cache
                        .insert(&seq.request.prompt_token_ids[..prefix_len], &mut cache)
                    {
                        tracing::warn!(%error, "prefix insertion failed");
                    }
                    seq.restore_cache(cache);
                }
            }
        }
        self.trace.emit(|| {
            serde_json::json!({"event":"computed", "step":self.step_id,
            "request_id":seq.request.id, "phase":format!("{:?}",seq.status),
            "prefix_tokens":self.prefix_cache.tokens()})
        });
        if let Some(token) = token {
            self.after_sample(idx, token);
        }
    }

    /// Shared post-sampling path: record the token, latency metrics,
    /// detokenization, stop checks, token event emission.
    fn after_sample(&mut self, idx: usize, token: u32) {
        let now = Instant::now();
        let seq = self.scheduler.running_get_mut(idx).expect("idx valid");
        seq.record_sampled(token);
        self.metrics.record_token();

        if seq.num_generated() == 1 {
            let us = now.duration_since(seq.created_at).as_micros() as u64;
            self.metrics.ttft_us_sum.fetch_add(us, Ordering::Relaxed);
            self.metrics.ttft_count.fetch_add(1, Ordering::Relaxed);
        } else if let Some(prev) = seq.last_token_at {
            let us = now.duration_since(prev).as_micros() as u64;
            self.metrics.itl_us_sum.fetch_add(us, Ordering::Relaxed);
            self.metrics.itl_count.fetch_add(1, Ordering::Relaxed);
        }
        seq.last_token_at = Some(now);

        let mut reason = None;
        if seq.stop_token_ids.contains(&token) {
            reason = Some(FinishReason::Stop);
        } else if seq.num_generated() >= seq.request.max_new_tokens {
            reason = Some(FinishReason::Length);
        }

        let mut delta = String::new();
        if let Some(detok) = seq.detokenizer.as_mut() {
            match detok.push(token, &seq.request.stop_strings) {
                Ok((d, stopped)) => {
                    delta = d;
                    if stopped {
                        reason = Some(FinishReason::Stop);
                    } else if reason.is_some() {
                        delta.push_str(&detok.finish());
                    }
                }
                Err(e) => {
                    tracing::warn!(request_id = %seq.request.id, error = %e, "detokenization failed");
                    seq.failure = Some(format!("detokenization failed: {e}"));
                    finish_sequence(seq, FinishReason::Error);
                    return;
                }
            }
        }

        if seq.consumer_gone
            || seq.emit(GenerationEvent::Token {
                token_id: token,
                text: delta,
            }) == crate::sequence::Emission::Gone
        {
            debug!(request_id = %seq.request.id, "consumer gone/too slow; cancelling");
            reason = Some(FinishReason::Cancelled);
        }

        self.metrics
            .generated_tokens_total
            .fetch_add(1, Ordering::Relaxed);

        if let Some(reason) = reason {
            finish_sequence(seq, reason);
        } else {
            seq.status = SequenceStatus::Running;
        }
    }

    fn update_gauges(&mut self) {
        self.metrics
            .requests_finishing
            .store(self.finishing.len() as u64, Ordering::Relaxed);
        self.metrics
            .cached_prefix_tokens
            .store(self.prefix_cache.tokens() as u64, Ordering::Relaxed);
        let usage = self.blocks.usage();
        self.metrics
            .kv_blocks_total
            .store(usage.total_blocks as u64, Ordering::Relaxed);
        self.metrics
            .kv_blocks_used
            .store(usage.used_blocks as u64, Ordering::Relaxed);
        self.metrics
            .kv_active_sequences
            .store(usage.active_sequences as u64, Ordering::Relaxed);
        // Single source of truth for queue gauges.
        self.metrics
            .requests_waiting
            .store(self.scheduler.waiting_len() as i64, Ordering::Relaxed);
        self.metrics
            .requests_running
            .store(self.scheduler.running_len() as i64, Ordering::Relaxed);
    }

    fn shutdown_all(&mut self) {
        let (waiting, running) = self.scheduler.drain_all();
        let n = waiting.len() + running.len() + self.finishing.len();
        self.metrics
            .requests_cancelled
            .fetch_add(n as u64, Ordering::Relaxed);
        let finishing = std::mem::take(&mut self.finishing)
            .into_iter()
            .map(|(s, _)| s);
        for mut seq in waiting.into_iter().chain(running).chain(finishing) {
            self.trace.emit(|| serde_json::json!({"event":"retired", "request_id":seq.request.id, "reason":"shutdown"}));
            // Running sequences own block tables; release them explicitly.
            let _ = self.blocks.release(&seq.request.id);
            seq.emit_terminal(GenerationEvent::Finished {
                reason: FinishReason::Cancelled,
                usage: seq.usage(),
            });
        }
        if n > 0 {
            tracing::info!(cancelled = n, "engine shutdown: cancelled requests");
        }
        self.prefix_cache.clear();

        self.update_gauges();
    }
}

/// Map a finish reason onto terminal status.
fn finish_sequence(seq: &mut SequenceGroup, reason: FinishReason) {
    seq.finish_reason = Some(reason);
    seq.status = match reason {
        FinishReason::Stop | FinishReason::Length => SequenceStatus::Finished,
        FinishReason::Cancelled => SequenceStatus::Cancelled,
        FinishReason::Error => SequenceStatus::Failed,
    };
    debug!(
        request_id = %seq.request.id,
        reason = reason.as_str(),
        generated = seq.num_generated(),
        "sequence reached terminal state"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use mini_vllm_core::{EngineConfig, SamplingParams};

    fn test_engine(config: EngineConfig) -> Engine {
        let model: Arc<dyn CausalLm> = Arc::new(mini_vllm_model::testutil::random_model(
            &candle_core::Device::Cpu,
        ));
        let (_tx, cmd_rx) = mpsc::channel(4);
        Engine {
            trace: crate::trace::TraceWriter::new(None).unwrap(),
            control: Arc::new(Control::default()),
            cmd_rx,
            executor: Executor::new(model),
            tokenizer: None,
            scheduler: Scheduler::new(config.max_num_seqs, config.max_batch_tokens),
            blocks: KvBlockManager::new(config.kv_block_size, config.max_kv_tokens),
            prefix_cache: PrefixCache::new(config.kv_block_size, config.prefix_cache_tokens),
            config,
            eos_ids: vec![],
            metrics: Arc::new(Metrics::new()),
            engine_seed: 1,
            rng_counter: 0,
            finishing: Vec::new(),

            decode_cursor: 0,
            prefill_cursor: 0,
            step_id: 0,
        }
    }

    #[test]
    fn admission_reserves_each_requests_horizon() {
        let config = EngineConfig {
            kv_block_size: 8,
            max_kv_tokens: 8,
            ..EngineConfig::default()
        };
        let mut engine = test_engine(config);
        let mut receivers = Vec::new();
        for id in ["a", "b"] {
            let (tx, rx) = mpsc::channel(64);
            receivers.push(rx);
            engine.enqueue_request(
                GenerationRequest {
                    id: id.into(),
                    prompt_token_ids: vec![4],
                    max_new_tokens: 7,
                    sampling: SamplingParams::default(),
                    stop_token_ids: vec![],
                    stop_strings: vec![],
                },
                tx,
            );
        }
        engine.admit_waiting();
        assert_eq!(engine.scheduler.running_len(), 1);
        assert_eq!(engine.scheduler.waiting_len(), 1);
        for _ in 0..20 {
            engine.step();
        }
        for mut rx in receivers {
            let mut finished = false;
            while let Ok(event) = rx.try_recv() {
                if let GenerationEvent::Finished { reason, .. } = event {
                    assert_eq!(reason, FinishReason::Length);
                    finished = true;
                }
            }
            assert!(finished);
        }
        assert_eq!(engine.blocks.free_block_count(), 1);
    }

    #[test]
    fn lookahead_admission_keeps_kv_reservations_bounded() {
        let mut engine = test_engine(EngineConfig {
            max_kv_tokens: 16,
            kv_block_size: 4,
            admission_lookahead: 3,
            admission_max_wait_ms: 10_000,
            ..EngineConfig::default()
        });
        let _active = enqueue(&mut engine, "active", vec![4], 7);
        engine.admit_waiting();
        let _big = enqueue(&mut engine, "big", vec![4], 11);
        let _small = enqueue(&mut engine, "small", vec![4], 3);
        let _small2 = enqueue(&mut engine, "small2", vec![4], 3);
        engine.admit_waiting();
        assert_eq!(engine.scheduler.running_len(), 3);
        assert_eq!(engine.scheduler.waiting_len(), 1);
        assert_eq!(engine.blocks.free_block_count(), 0);
        assert!(engine
            .scheduler
            .running()
            .iter()
            .all(|s| s.request.id != "big"));
        for id in ["active", "small", "small2"] {
            engine.scheduler.cancel(id);
        }
        for _ in 0..20 {
            engine.step();
        }
        assert_eq!(engine.scheduler.waiting_len(), 0);
        assert_eq!(engine.scheduler.running_len(), 0);
        assert_eq!(engine.blocks.free_block_count(), 4);
    }

    #[test]
    fn cancellation_does_not_need_command_queue_capacity() {
        let (tx, _rx) = mpsc::channel(1);
        tx.try_send(EngineCommand::Cancel {
            request_id: "occupied".into(),
        })
        .unwrap();
        let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
        let flag = Arc::new(AtomicBool::new(false));
        registry
            .lock()
            .unwrap()
            .insert("target".into(), flag.clone());
        let handle = EngineHandle {
            control: Arc::new(Control::default()),
            cmd_tx: tx,
            metrics: Arc::new(Metrics::new()),
            event_capacity: 1,
            registry,
            max_outstanding: 2,
            max_model_len: 64,
            vocab_size: 64,
            default_max_new_tokens: 512,
        };
        handle.cancel("target").unwrap();
        assert!(flag.load(Ordering::Relaxed));
    }
    fn enqueue(
        engine: &mut Engine,
        id: &str,
        prompt: Vec<u32>,
        max_new: usize,
    ) -> mpsc::Receiver<GenerationEvent> {
        let (tx, rx) = mpsc::channel(64);
        engine.enqueue_request(
            GenerationRequest {
                id: id.into(),
                prompt_token_ids: prompt,
                max_new_tokens: max_new,
                sampling: SamplingParams::default(),
                stop_token_ids: vec![],
                stop_strings: vec![],
            },
            tx,
        );
        rx
    }

    #[test]
    fn prefix_admission_saves_capacity_and_pins_eviction() {
        let mut engine = test_engine(EngineConfig {
            kv_block_size: 4,
            max_kv_tokens: 16,
            prefix_cache_tokens: 8,
            max_batch_tokens: 4,
            ..EngineConfig::default()
        });
        let _warm = enqueue(&mut engine, "warm", vec![4; 9], 3);
        for _ in 0..10 {
            engine.step();
        }
        assert_eq!(engine.prefix_cache.tokens(), 8);
        let _a = enqueue(&mut engine, "a", vec![4; 9], 3);
        let _b = enqueue(&mut engine, "b", vec![4; 9], 3);
        engine.admit_waiting();
        assert_eq!(
            engine.scheduler.running_len(),
            2,
            "both fit only when prefix is deducted"
        );
        assert_eq!(engine.blocks.usage().used_blocks, 2);
        assert_eq!(
            Arc::strong_count(
                engine
                    .scheduler
                    .running()
                    .iter()
                    .find(|s| s.request.id == "b")
                    .unwrap()
                    .prefix_pin
                    .as_ref()
                    .unwrap()
            ),
            3
        );
        // A third, unrelated request can use the remaining block, but cannot
        // evict the shared prefix while its active borrowers still hold it.
        let _c = enqueue(&mut engine, "c", vec![5; 5], 1);
        engine.admit_waiting();
        assert_eq!(engine.blocks.usage().used_blocks, 4);
        engine.run_mixed_step();
        engine.run_mixed_step();
        assert_eq!(engine.prefix_cache.tokens(), 8);
        assert_eq!(
            engine.prefix_cache.matched_tokens(&[4; 9]),
            8,
            "borrowed prefix must not be evicted"
        );
        engine.scheduler.cancel("a");
        engine.retire();
        assert_eq!(
            Arc::strong_count(
                engine
                    .scheduler
                    .running()
                    .iter()
                    .find(|s| s.request.id == "b")
                    .unwrap()
                    .prefix_pin
                    .as_ref()
                    .unwrap()
            ),
            2
        );
        engine.shutdown_all();
        assert_eq!(engine.blocks.usage().used_blocks, 0);
        assert_eq!(engine.prefix_cache.tokens(), 0);
    }

    #[test]
    fn prefill_rotation_prevents_long_prompt_monopoly() {
        let mut engine = test_engine(EngineConfig {
            max_batch_tokens: 2,
            max_prefill_chunk_tokens: 2,
            ..EngineConfig::default()
        });
        let _long = enqueue(&mut engine, "long", vec![4; 32], 2);
        let _short = enqueue(&mut engine, "short", vec![5; 3], 2);
        engine.step();
        engine.step();
        let short = engine
            .scheduler
            .running()
            .iter()
            .find(|s| s.request.id == "short")
            .unwrap();
        assert_eq!(short.prefill_position, 2);
        assert!(engine.scheduler.running()[0].prefill_position < 32);
    }

    #[test]
    fn detokenization_failure_is_terminal_and_reclaims_cache() {
        let mut engine = test_engine(EngineConfig::default());
        engine.tokenizer = Some(Arc::new(TokenizerWrapper::from_inner(
            mini_vllm_tokenizer::testutil::char_tokenizer(),
        )));
        let mut rx = enqueue(&mut engine, "bad-tokenizer", vec![4], 2);
        engine.admit_waiting();
        // This is a model-vocabulary id missing from the smaller tokenizer.
        engine.after_sample(0, 63);
        engine.retire();
        assert!(matches!(
            rx.try_recv().unwrap(),
            GenerationEvent::Error {
                kind: mini_vllm_core::GenerationErrorKind::Execution,
                ..
            }
        ));
        assert!(rx.try_recv().is_err());
        assert_eq!(engine.blocks.usage().used_blocks, 0);
    }
    #[test]
    fn one_token_budget_rotates_prefills_despite_alternating_decode_steps() {
        let mut engine = test_engine(EngineConfig {
            max_batch_tokens: 1,
            ..EngineConfig::default()
        });
        let _decode = enqueue(&mut engine, "decode", vec![4], 8);
        engine.step();
        let _a = enqueue(&mut engine, "a", vec![5; 12], 2);
        let _b = enqueue(&mut engine, "b", vec![6; 12], 2);
        for _ in 0..4 {
            engine.step();
        }
        for id in ["a", "b"] {
            let s = engine
                .scheduler
                .running()
                .iter()
                .find(|s| s.request.id == id)
                .unwrap();
            assert_eq!(s.prefill_position, 1, "both prefills should receive a turn");
        }
    }
    #[test]
    fn timeouts_expire_waiting_and_running_without_leaking_capacity() {
        let mut engine = test_engine(EngineConfig {
            max_num_seqs: 1,
            queue_timeout_ms: 1,
            request_timeout_ms: 10,
            ..EngineConfig::default()
        });
        let mut running = enqueue(&mut engine, "running", vec![4; 16], 2);
        engine.admit_waiting();
        let mut waiting = enqueue(&mut engine, "waiting", vec![5], 2);
        // Aging the running sequence avoids timing-dependent model speed.
        engine.scheduler.running_get_mut(0).unwrap().created_at =
            Instant::now() - Duration::from_millis(20);
        std::thread::sleep(Duration::from_millis(3));
        engine.step();
        for rx in [&mut running, &mut waiting] {
            assert!(matches!(
                rx.try_recv().unwrap(),
                GenerationEvent::Error {
                    kind: mini_vllm_core::GenerationErrorKind::Timeout,
                    ..
                }
            ));
        }
        assert_eq!(engine.blocks.usage().used_blocks, 0);
        assert_eq!(engine.scheduler.waiting_len(), 0);
    }

    #[test]
    fn timeout_in_command_queue_uses_submission_timestamp() {
        let mut engine = test_engine(EngineConfig {
            queue_timeout_ms: 1,
            ..EngineConfig::default()
        });
        let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
        let cancelled = Arc::new(AtomicBool::new(false));
        registry
            .lock()
            .unwrap()
            .insert("old".into(), cancelled.clone());
        let lease = RequestLease {
            id: "old".into(),
            submitted_at: Instant::now() - Duration::from_millis(20),
            cancelled,
            registry: registry.clone(),
        };
        let (tx, mut rx) = mpsc::channel(2);
        engine.enqueue_request_with_lease(
            GenerationRequest {
                id: "old".into(),
                prompt_token_ids: vec![4],
                max_new_tokens: 1,
                sampling: SamplingParams::default(),
                stop_token_ids: vec![],
                stop_strings: vec![],
            },
            tx,
            Some(lease),
        );
        assert!(matches!(
            rx.try_recv().unwrap(),
            GenerationEvent::Error {
                kind: mini_vllm_core::GenerationErrorKind::Timeout,
                ..
            }
        ));
        assert!(registry.lock().unwrap().is_empty());
        assert_eq!(engine.scheduler.waiting_len(), 0);
    }
    #[test]
    fn deadline_crossed_during_forward_does_not_emit_successful_token() {
        let mut engine = test_engine(EngineConfig {
            request_timeout_ms: 1,
            ..EngineConfig::default()
        });
        let mut rx = enqueue(&mut engine, "deadline", vec![4], 1);
        engine.admit_waiting();
        engine.scheduler.running_get_mut(0).unwrap().created_at =
            Instant::now() - Duration::from_millis(20);
        engine.after_chunk(0, 1, Some(4));
        engine.retire();
        assert!(matches!(
            rx.try_recv().unwrap(),
            GenerationEvent::Error {
                kind: mini_vllm_core::GenerationErrorKind::Timeout,
                ..
            }
        ));
        assert!(rx.try_recv().is_err());
        assert_eq!(engine.blocks.usage().used_blocks, 0);
    }
}
