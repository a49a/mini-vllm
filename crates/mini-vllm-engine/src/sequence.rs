//! Per-request runtime state.

use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use mini_vllm_core::{FinishReason, GenerationEvent, GenerationRequest, SequenceStatus, Usage};
use mini_vllm_kv::KvCache;
use mini_vllm_sampling::Sampler;
use mini_vllm_tokenizer::IncrementalDetokenizer;
use tokio::sync::mpsc;

/// Outcome of trying to hand an event to the consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emission {
    /// Delivered into the channel.
    Sent,
    /// The channel is full; parked in the bounded outbox for later flush.
    Queued,
    /// The consumer is gone (disconnected) or too slow (outbox overflow).
    /// The engine cancels the request on this outcome.
    Gone,
}

/// Everything the engine needs to drive one generation request.
pub struct SequenceGroup {
    pub request: GenerationRequest,
    pub lease: Option<crate::lifecycle::RequestLease>,
    /// Number of prompt positions already computed or reused.
    pub prefill_position: usize,
    /// Pins the separately charged shared prefix until active KV is released.
    pub prefix_pin: Option<std::sync::Arc<()>>,
    terminal_queued: bool,

    pub status: SequenceStatus,
    /// `None` while the sequence is still waiting for admission.
    kv_cache: Option<KvCache>,
    pub sampler: Sampler,
    /// Present when the engine was built with a tokenizer; used for
    /// incremental detokenization and stop-string matching.
    pub detokenizer: Option<IncrementalDetokenizer>,
    event_tx: mpsc::Sender<GenerationEvent>,
    /// Bounded backlog for events that did not fit the channel. The engine
    /// thread must never block on a consumer (§43): a slow consumer first
    /// fills this outbox, then gets cancelled.
    outbox: VecDeque<GenerationEvent>,
    outbox_limit: usize,
    /// Next token to feed into the model (the most recently sampled one).
    pub next_token: u32,
    pub generated_token_ids: Vec<u32>,
    /// Tokens seen so far (prompt ∪ generated) for the repetition penalty.
    seen_tokens: HashSet<u32>,
    /// Effective stop ids: request stops ∪ model EOS ids.
    pub stop_token_ids: Vec<u32>,
    pub created_at: Instant,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
    pub finish_reason: Option<FinishReason>,
    pub failure: Option<String>,
    /// Set when the consumer is gone (channel closed or outbox overflow).
    pub consumer_gone: bool,
}

impl SequenceGroup {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request: GenerationRequest,
        sampler: Sampler,
        detokenizer: Option<IncrementalDetokenizer>,
        event_tx: mpsc::Sender<GenerationEvent>,
        stop_token_ids: Vec<u32>,
        outbox_limit: usize,
    ) -> Self {
        let seen_tokens = request.prompt_token_ids.iter().copied().collect();
        Self {
            request,
            lease: None,
            prefill_position: 0,
            prefix_pin: None,
            terminal_queued: false,
            status: SequenceStatus::Waiting,
            kv_cache: None,
            sampler,
            detokenizer,
            event_tx,
            outbox: VecDeque::new(),
            outbox_limit,
            next_token: 0,
            generated_token_ids: Vec::new(),
            seen_tokens,
            stop_token_ids,
            created_at: Instant::now(),
            first_token_at: None,
            last_token_at: None,
            finish_reason: None,
            failure: None,
            consumer_gone: false,
        }
    }

    pub fn prompt_len(&self) -> usize {
        self.request.prompt_token_ids.len()
    }

    pub fn num_generated(&self) -> usize {
        self.generated_token_ids.len()
    }

    /// The repetition-penalty context, maintained incrementally.
    pub fn seen_context(&self) -> &HashSet<u32> {
        &self.seen_tokens
    }

    /// Sample one token from a logits row with this sequence's params and
    /// penalty context. (Destructured so the sampler's mutable borrow stays
    /// disjoint from the params/context borrows.)
    pub fn sample_next(&mut self, logits: &[f32]) -> u32 {
        let Self {
            sampler,
            request,
            seen_tokens,
            ..
        } = self;
        sampler.sample(logits, &request.sampling, seen_tokens)
    }

    /// Position at which `next_token` will be fed. Only meaningful after
    /// the first sampled token; decode candidates always have one.
    pub fn next_input_position(&self) -> u32 {
        (self.prompt_len() + self.num_generated().saturating_sub(1)) as u32
    }

    /// Tokens cached once the next step completes.
    pub fn kv_tokens_after_next_step(&self) -> usize {
        self.prompt_len() + self.num_generated() + 1
    }

    /// Full horizon used to size the cache at admission.
    pub fn kv_capacity(&self) -> usize {
        self.prompt_len() + self.request.max_new_tokens
    }

    pub fn is_terminal(&self) -> bool {
        self.finish_reason.is_some()
    }

    pub fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_len(),
            completion_tokens: self.num_generated(),
            total_tokens: self.prompt_len() + self.num_generated(),
        }
    }

    /// Record a freshly sampled token. (`last_token_at` is maintained by
    /// the engine so inter-token latency is measured at emission time.)
    pub fn record_sampled(&mut self, token: u32) {
        self.next_token = token;
        self.generated_token_ids.push(token);
        self.seen_tokens.insert(token);
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
    }

    /// Hand one event to the consumer **without ever blocking the engine
    /// thread**. Flushes the outbox first, then tries the channel. A full
    /// channel parks the event in the bounded outbox; an overflowing outbox
    /// (or a closed channel) marks the consumer gone.
    pub fn emit(&mut self, event: GenerationEvent) -> Emission {
        if self.consumer_gone {
            return Emission::Gone;
        }
        self.outbox.push_back(event);
        self.flush_outbox();
        if self.consumer_gone || self.outbox.len() > self.outbox_limit {
            self.consumer_gone = true;
            self.outbox.clear();
            Emission::Gone
        } else if self.outbox.is_empty() {
            Emission::Sent
        } else {
            Emission::Queued
        }
    }

    /// Always send from the front: a consumer freeing a channel slot must
    /// never allow a newer event to overtake an older queued token.
    fn flush_outbox(&mut self) {
        while let Some(event) = self.outbox.pop_front() {
            match self.event_tx.try_send(event) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(event)) => {
                    self.outbox.push_front(event);
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    self.consumer_gone = true;
                    self.outbox.clear();
                    break;
                }
            }
        }
    }

    /// Flush queued tokens before finishing. Retirement cannot wait on a
    /// slow consumer: if delivery is incomplete, close as cancellation,
    /// never as a successful truncated response. A missing terminal event
    /// is also cancellation and must be surfaced by stream adapters.
    pub fn emit_terminal(&mut self, event: GenerationEvent) {
        self.flush_outbox();
        if self.consumer_gone || !self.outbox.is_empty() {
            self.cancel_delivery();
            return;
        }
        if self.event_tx.try_send(event).is_err() {
            self.cancel_delivery();
        }
    }

    fn cancel_delivery(&mut self) {
        self.consumer_gone = true;
        self.outbox.clear();
        self.finish_reason = Some(FinishReason::Cancelled);
        self.status = SequenceStatus::Cancelled;
        let _ = self.event_tx.try_send(GenerationEvent::Finished {
            reason: FinishReason::Cancelled,
            usage: self.usage(),
        });
    }

    pub fn cancelled(&self) -> bool {
        self.consumer_gone
            || self.event_tx.is_closed()
            || self.lease.as_ref().is_some_and(|l| l.is_cancelled())
    }

    /// Queue terminal delivery once, then allow a bounded grace period after
    /// KV reclamation. The engine owns the timeout and keeps polling.
    pub fn drain_terminal(&mut self) -> bool {
        if self.cancelled() {
            self.cancel_delivery();
            return true;
        }
        if !self.terminal_queued {
            let reason = self.finish_reason.unwrap_or(FinishReason::Error);
            let event = if reason == FinishReason::Error {
                GenerationEvent::Error {
                    kind: mini_vllm_core::GenerationErrorKind::Execution,
                    message: self
                        .failure
                        .take()
                        .unwrap_or_else(|| "generation failed".into()),
                }
            } else {
                GenerationEvent::Finished {
                    reason,
                    usage: self.usage(),
                }
            };
            self.outbox.push_back(event);
            self.terminal_queued = true;
        }
        self.flush_outbox();
        self.consumer_gone || self.outbox.is_empty()
    }

    pub fn expire_delivery(&mut self) {
        self.cancel_delivery();
    }

    // -- KV cache slot management --------------------------------------

    /// Temporarily take the cache out of the sequence (for batched forward).
    pub fn take_cache(&mut self) -> Option<KvCache> {
        self.kv_cache.take()
    }

    /// Put the cache back after a forward pass.
    pub fn restore_cache(&mut self, cache: KvCache) {
        debug_assert!(self.kv_cache.is_none(), "cache slot must be empty");
        self.kv_cache = Some(cache);
    }

    pub fn has_cache(&self) -> bool {
        self.kv_cache.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mini_vllm_core::{GenerationRequest, SamplingParams};

    fn seq(prompt: usize, max_new: usize) -> SequenceGroup {
        let (tx, _rx) = mpsc::channel(4);
        SequenceGroup::new(
            GenerationRequest {
                id: "t".into(),
                prompt_token_ids: vec![1; prompt],
                sampling: SamplingParams::default(),
                max_new_tokens: max_new,
                stop_token_ids: vec![],
                stop_strings: vec![],
            },
            Sampler::from_seed(1),
            None,
            tx,
            vec![],
            4,
        )
    }

    fn token(id: u32) -> GenerationEvent {
        GenerationEvent::Token {
            token_id: id,
            text: String::new(),
        }
    }

    #[test]
    fn positions_track_generation() {
        let mut s = seq(5, 10);
        assert_eq!(s.kv_capacity(), 15);
        s.record_sampled(42);
        assert_eq!(s.next_input_position(), 5); // first generated token sits at pos = prompt_len
        s.record_sampled(43);
        assert_eq!(s.next_input_position(), 6);
        assert_eq!(s.seen_context().len(), 3); // prompt ids dedupe to 1, plus 2 sampled
        let u = s.usage();
        assert_eq!(
            (u.prompt_tokens, u.completion_tokens, u.total_tokens),
            (5, 2, 7)
        );
    }

    #[test]
    fn emit_queues_when_channel_full_and_cancels_on_overflow() {
        let (tx, _rx_keep) = mpsc::channel(1); // hold the receiver open
        let mut s = SequenceGroup::new(
            GenerationRequest {
                id: "t".into(),
                prompt_token_ids: vec![1],
                sampling: SamplingParams::default(),
                max_new_tokens: 4,
                stop_token_ids: vec![],
                stop_strings: vec![],
            },
            Sampler::from_seed(1),
            None,
            tx,
            vec![],
            2, // outbox limit
        );
        // First event fills the channel of capacity 1.
        assert_eq!(s.emit(token(1)), Emission::Sent);
        // Next events park in the outbox…
        assert_eq!(s.emit(token(2)), Emission::Queued);
        assert_eq!(s.emit(token(3)), Emission::Queued);
        // …and the fourth (over the limit) cancels the consumer.
        assert_eq!(s.emit(token(4)), Emission::Gone);
        assert!(s.consumer_gone);
        assert_eq!(s.emit(token(5)), Emission::Gone); // stays gone
    }

    #[test]
    fn emit_detects_closed_channel() {
        let (tx, rx) = mpsc::channel(4);
        let mut s = SequenceGroup::new(
            GenerationRequest {
                id: "t".into(),
                prompt_token_ids: vec![1],
                sampling: SamplingParams::default(),
                max_new_tokens: 4,
                stop_token_ids: vec![],
                stop_strings: vec![],
            },
            Sampler::from_seed(1),
            None,
            tx,
            vec![],
            4,
        );
        drop(rx);
        assert_eq!(s.emit(token(1)), Emission::Gone);
        assert!(s.consumer_gone);
    }

    #[tokio::test]
    async fn retirement_never_reports_success_after_dropping_queued_tokens() {
        let (tx, mut rx) = mpsc::channel(1);
        let mut s = seq(1, 2);
        s.event_tx = tx;
        assert_eq!(s.emit(token(1)), Emission::Sent);
        assert_eq!(s.emit(token(2)), Emission::Queued);
        assert!(matches!(
            rx.recv().await,
            Some(GenerationEvent::Token { token_id: 1, .. })
        ));
        s.emit_terminal(GenerationEvent::Finished {
            reason: FinishReason::Length,
            usage: s.usage(),
        });
        assert_eq!(s.finish_reason, Some(FinishReason::Cancelled));
        drop(s);
        assert!(matches!(
            rx.recv().await,
            Some(GenerationEvent::Token { token_id: 2, .. })
        ));
        while let Some(event) = rx.recv().await {
            assert!(matches!(
                event,
                GenerationEvent::Finished {
                    reason: FinishReason::Cancelled,
                    ..
                }
            ));
        }
    }

    #[tokio::test]
    async fn retirement_flushes_tokens_before_success_when_space_available() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut s = seq(1, 2);
        s.event_tx = tx;
        s.outbox.push_back(token(1));
        s.outbox.push_back(token(2));
        s.emit_terminal(GenerationEvent::Finished {
            reason: FinishReason::Length,
            usage: s.usage(),
        });
        drop(s);
        for id in [1, 2] {
            assert!(
                matches!(rx.recv().await, Some(GenerationEvent::Token { token_id, .. }) if token_id == id)
            );
        }
        assert!(matches!(
            rx.recv().await,
            Some(GenerationEvent::Finished {
                reason: FinishReason::Length,
                ..
            })
        ));
    }

    #[test]
    fn cache_slot_roundtrip() {
        let mut s = seq(2, 2);
        assert!(!s.has_cache());
        let cache = KvCache::new(
            1,
            1,
            4,
            8,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        s.restore_cache(cache);
        assert!(s.has_cache());
        assert!(s.take_cache().is_some());
        assert!(!s.has_cache());
    }
}
