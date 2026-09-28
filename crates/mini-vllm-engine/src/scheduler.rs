//! FIFO scheduler with admission control (design doc §27/§28).
//!
//! The engine calls [`Scheduler::admit`] every iteration with a capacity
//! gate (free KV blocks); admitted sequences move waiting → running and get
//! prefilled, finished sequences are retired — this is what makes the
//! batching *continuous*.

use std::collections::VecDeque;

use mini_vllm_core::SequenceStatus;

use crate::sequence::SequenceGroup;

pub struct Scheduler {
    waiting: VecDeque<SequenceGroup>,
    running: Vec<SequenceGroup>,
    /// Sequences cancelled while still waiting (never reached `running`).
    retired_waiting: Vec<SequenceGroup>,
    max_num_seqs: usize,
    max_batch_tokens: usize,
}

impl Scheduler {
    pub fn new(max_num_seqs: usize, max_batch_tokens: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            retired_waiting: Vec::new(),
            max_num_seqs: max_num_seqs.max(1),
            max_batch_tokens: max_batch_tokens.max(1),
        }
    }

    /// Read access to the running list (engine uses it for split borrows).
    pub fn running(&self) -> &[SequenceGroup] {
        &self.running
    }

    pub fn max_num_seqs(&self) -> usize {
        self.max_num_seqs
    }

    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn enqueue(&mut self, seq: SequenceGroup) {
        self.waiting.push_back(seq);
    }

    /// Move waiting sequences into `running` in FIFO order while
    /// `max_num_seqs` is respected and `gate` (capacity check) accepts the
    /// head of the queue. Returns the number of newly admitted sequences.
    pub fn admit(
        &mut self,
        mut gate: impl FnMut(&mini_vllm_core::GenerationRequest) -> bool,
    ) -> usize {
        self.admit_with_lookahead(1, std::time::Duration::ZERO, &mut gate)
    }

    /// Scan at most `lookahead` candidates per admission. A blocked, aged
    /// request forms a barrier: later requests cannot keep taking its capacity.
    /// `gate` must consume capacity only when returning true.
    pub fn admit_with_lookahead(
        &mut self,
        lookahead: usize,
        max_wait: std::time::Duration,
        mut gate: impl FnMut(&mini_vllm_core::GenerationRequest) -> bool,
    ) -> usize {
        let mut admitted = 0;
        while self.running.len() < self.max_num_seqs {
            let mut candidate = None;
            for (i, seq) in self.waiting.iter().take(lookahead.max(1)).enumerate() {
                if gate(&seq.request) {
                    candidate = Some(i);
                    break;
                }
                if seq.created_at.elapsed() >= max_wait {
                    break;
                }
            }
            let Some(index) = candidate else { break };
            let mut seq = self.waiting.remove(index).expect("candidate exists");
            tracing::debug!(request_id = %seq.request.id, bypassed = index, "request admitted to running set");
            seq.status = SequenceStatus::Prefill;
            seq.admitted_at = Some(std::time::Instant::now());
            self.running.push(seq);
            admitted += 1;
        }
        admitted
    }

    /// Remove terminal sequences from `running`; returns them so the engine
    /// can release resources and emit terminal events.
    pub fn retire_finished(&mut self) -> Vec<SequenceGroup> {
        let (keep, retired): (Vec<_>, Vec<_>) =
            self.running.drain(..).partition(|s| !s.is_terminal());
        self.running = keep;
        retired
    }

    /// Indices (into `running`) of sequences ready for a decode step.
    pub fn decode_candidates(&self) -> Vec<usize> {
        self.running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == SequenceStatus::Running && !s.is_terminal())
            .map(|(i, _)| i)
            .collect()
    }

    /// Index of the first sequence waiting for prefill, if any.
    pub fn prefill_candidate(&self) -> Option<usize> {
        self.running
            .iter()
            .position(|s| s.status == SequenceStatus::Prefill && !s.is_terminal())
    }

    pub fn running_iter_mut(&mut self) -> std::slice::IterMut<'_, SequenceGroup> {
        self.running.iter_mut()
    }

    pub fn running_get_mut(&mut self, idx: usize) -> Option<&mut SequenceGroup> {
        self.running.get_mut(idx)
    }

    pub fn has_decode_work(&self) -> bool {
        !self.decode_candidates().is_empty()
    }

    /// Mark a request cancelled, wherever it currently lives. Returns true
    /// if it was found in `running` (deferred handling) — false means it
    /// was removed from `waiting` (or not found at all).
    pub fn cancel(&mut self, request_id: &str) -> bool {
        if let Some(pos) = self.waiting.iter().position(|s| s.request.id == request_id) {
            let mut seq = self.waiting.remove(pos).expect("position just found");
            seq.finish_reason = Some(mini_vllm_core::FinishReason::Cancelled);
            self.retired_waiting.push(seq);
            return false;
        }
        if let Some(seq) = self
            .running
            .iter_mut()
            .find(|s| s.request.id == request_id && !s.is_terminal())
        {
            seq.finish_reason = Some(mini_vllm_core::FinishReason::Cancelled);
            seq.status = mini_vllm_core::SequenceStatus::Cancelled;
            return true;
        }
        false
    }

    /// Poll receiver closure and cancellation flags even for waiting requests.
    pub fn cancel_disconnected(&mut self) {
        let ids: Vec<_> = self
            .waiting
            .iter()
            .chain(self.running.iter())
            .filter(|s| !s.is_terminal() && s.cancelled())
            .map(|s| s.request.id.clone())
            .collect();
        for id in ids {
            self.cancel(&id);
        }
    }

    /// Cancellations of sequences that never reached `running`.
    pub fn take_retired_waiting(&mut self) -> Vec<SequenceGroup> {
        std::mem::take(&mut self.retired_waiting)
    }

    /// Expire at scheduling boundaries; device calls are never interrupted.
    pub fn expire(&mut self, queue_ms: u64, request_ms: u64) {
        for seq in self.waiting.iter_mut().chain(self.running.iter_mut()) {
            if seq.is_terminal() {
                continue;
            }
            let elapsed = seq.created_at.elapsed();
            let queue_expired = seq.admitted_at.is_none()
                && queue_ms > 0
                && elapsed >= std::time::Duration::from_millis(queue_ms);
            let request_expired =
                request_ms > 0 && elapsed >= std::time::Duration::from_millis(request_ms);
            if queue_expired || request_expired {
                seq.failure = Some(
                    if queue_expired {
                        "queue timeout"
                    } else {
                        "request deadline exceeded"
                    }
                    .into(),
                );
                seq.failure_kind = mini_vllm_core::GenerationErrorKind::Timeout;
                seq.finish_reason = Some(mini_vllm_core::FinishReason::Error);
                seq.status = SequenceStatus::Failed;
            }
        }
        let (retired, waiting): (Vec<_>, Vec<_>) =
            self.waiting.drain(..).partition(|s| s.is_terminal());
        self.waiting = waiting.into();
        self.retired_waiting.extend(retired);
    }

    /// Drain every sequence (engine shutdown path).
    pub fn drain_all(&mut self) -> (Vec<SequenceGroup>, Vec<SequenceGroup>) {
        let waiting: Vec<_> = self
            .waiting
            .drain(..)
            .chain(self.retired_waiting.drain(..))
            .collect();
        let running = std::mem::take(&mut self.running);
        (waiting, running)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mini_vllm_core::{FinishReason, GenerationRequest, SamplingParams};
    use tokio::sync::mpsc;

    fn seq(id: &str) -> SequenceGroup {
        let (tx, _rx) = mpsc::channel(4);
        SequenceGroup::new(
            GenerationRequest {
                id: id.into(),
                prompt_token_ids: vec![1, 2, 3],
                sampling: SamplingParams::default(),
                max_new_tokens: 4,
                stop_token_ids: vec![],
                stop_strings: vec![],
            },
            mini_vllm_sampling::Sampler::from_seed(0),
            None,
            tx,
            vec![],
            4,
        )
    }

    #[test]
    fn respects_max_num_seqs_fifo() {
        let mut s = Scheduler::new(2, 512);
        for i in 0..5 {
            s.enqueue(seq(&format!("r{i}")));
        }
        let n = s.admit(|_| true);
        assert_eq!(n, 2);
        assert_eq!(s.running_len(), 2);
        assert_eq!(s.waiting_len(), 3);
        // FIFO: first two admitted.
        let ids: Vec<String> = s.running_iter_mut().map(|x| x.request.id.clone()).collect();
        assert_eq!(ids, vec!["r0", "r1"]);
    }

    #[test]
    fn gate_blocks_head_of_line() {
        let mut s = Scheduler::new(8, 512);
        s.enqueue(seq("big"));
        s.enqueue(seq("small"));
        let n = s.admit(|_| false);
        assert_eq!(n, 0);
        let n = s.admit(|req| req.id == "big"); // big fits now
        assert_eq!(n, 1);
        let n = s.admit(|_| false);
        assert_eq!(n, 0);
    }

    #[test]
    fn lookahead_is_bounded_and_aged_requests_stop_bypasses() {
        use std::time::{Duration, Instant};
        let mut s = Scheduler::new(8, 512);
        for id in ["big", "medium", "small"] {
            s.enqueue(seq(id));
        }
        let wait = Duration::from_secs(10);
        assert_eq!(s.admit_with_lookahead(2, wait, |r| r.id == "small"), 0);
        assert_eq!(s.admit_with_lookahead(3, wait, |r| r.id == "small"), 1);
        assert_eq!(s.running()[0].request.id, "small");
        s.waiting[0].created_at = Instant::now() - wait;
        let mut probed = Vec::new();
        assert_eq!(
            s.admit_with_lookahead(3, wait, |r| {
                probed.push(r.id.clone());
                r.id != "big"
            }),
            0
        );
        assert_eq!(probed, ["big"]);
        // Once capacity is free the oldest request can proceed.
        assert_eq!(s.admit_with_lookahead(3, wait, |_| true), 2);
        assert_eq!(s.running()[1].request.id, "big");
    }

    #[test]
    fn lookahead_respects_aged_barrier_behind_the_head_and_slot_limit() {
        use std::time::{Duration, Instant};
        let mut s = Scheduler::new(1, 512);
        s.enqueue(seq("new-head"));
        let mut old = seq("old");
        old.created_at = Instant::now() - Duration::from_secs(20);
        s.enqueue(old);
        s.enqueue(seq("small"));
        assert_eq!(
            s.admit_with_lookahead(3, Duration::from_secs(10), |r| r.id == "small"),
            0
        );
        assert_eq!(
            s.admit_with_lookahead(3, Duration::from_secs(10), |_| true),
            1
        );
        assert_eq!(s.waiting_len(), 2);
    }

    #[test]
    fn retire_removes_only_terminal() {
        let mut s = Scheduler::new(8, 512);
        s.enqueue(seq("a"));
        s.enqueue(seq("b"));
        s.admit(|_| true);
        s.running_get_mut(0).unwrap().finish_reason = Some(FinishReason::Length);
        let retired = s.retire_finished();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].request.id, "a");
        assert_eq!(s.running_len(), 1);
    }

    #[test]
    fn cancel_in_waiting_returns_immediately() {
        let mut s = Scheduler::new(8, 512);
        s.enqueue(seq("w"));
        assert!(!s.cancel("w"));
        let retired = s.take_retired_waiting();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].finish_reason, Some(FinishReason::Cancelled));
    }

    #[test]
    fn decode_candidates_skip_prefill_and_terminal() {
        let mut s = Scheduler::new(8, 512);
        for id in ["a", "b", "c"] {
            s.enqueue(seq(id));
        }
        s.admit(|_| true);
        // a → running, b → finished, c → still prefill
        s.running_get_mut(0).unwrap().status = SequenceStatus::Running;
        s.running_get_mut(1).unwrap().finish_reason = Some(FinishReason::Stop);
        assert_eq!(s.decode_candidates(), vec![0]);
        assert_eq!(s.prefill_candidate(), Some(2));
        assert!(s.has_decode_work());
    }
}
