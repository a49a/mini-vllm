//! Fundamental domain objects: requests, sampling parameters, sequences,
//! generation events.
//!
//! HTTP-specific types never appear here; the server layer maps its own
//! request/response structs onto these.

/// Unique identifier of an in-flight generation request.
pub type RequestId = String;

/// Sampling configuration for a single request.
///
/// The pipeline applied by the sampler is:
/// `repetition penalty -> temperature -> top-k -> top-p -> softmax -> sample`.
/// `temperature == 0` short-circuits to greedy argmax.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// 0 means greedy decoding.
    pub temperature: f32,
    /// Keep only the `k` highest-probability logits. `None` / `Some(0)` disables.
    pub top_k: Option<usize>,
    /// Nucleus sampling: keep the smallest prefix of tokens whose cumulative
    /// probability reaches `top_p`. `None` disables.
    pub top_p: Option<f32>,
    /// Penalize tokens that already appeared in the prompt or generated text.
    pub repetition_penalty: Option<f32>,
    /// Per-request seed for deterministic sampling.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: None,
        }
    }
}

/// A fully materialized generation request (already tokenized).
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub sampling: SamplingParams,
    /// Upper bound on generated tokens; the hard cap is also bounded by the
    /// engine's `max_model_len`.
    pub max_new_tokens: usize,
    /// Additional token ids that finish generation (EOS ids are added by the engine).
    pub stop_token_ids: Vec<u32>,
    /// Stop strings: generation finishes once the decoded text contains one;
    /// the returned text is truncated at the first occurrence.
    pub stop_strings: Vec<String>,
}

impl GenerationRequest {
    /// Validate basic invariants before the request enters the engine.
    pub fn validate(&self, max_model_len: usize) -> crate::Result<()> {
        if self.prompt_token_ids.is_empty() {
            return Err(crate::Error::InvalidRequest("prompt is empty".into()));
        }
        if self.max_new_tokens == 0 {
            return Err(crate::Error::InvalidRequest(
                "max_new_tokens must be > 0".into(),
            ));
        }
        if self
            .prompt_token_ids
            .len()
            .checked_add(self.max_new_tokens)
            .map_or(true, |n| n > max_model_len)
        {
            return Err(crate::Error::InvalidRequest(format!(
                "prompt_len({}) + max_new_tokens({}) exceeds max_model_len({})",
                self.prompt_token_ids.len(),
                self.max_new_tokens,
                max_model_len
            )));
        }
        let t = self.sampling.temperature;
        if !t.is_finite() || t < 0.0 {
            return Err(crate::Error::InvalidRequest(
                "temperature must be finite and >= 0".into(),
            ));
        }
        if let Some(penalty) = self.sampling.repetition_penalty {
            if !penalty.is_finite() || penalty <= 0.0 {
                return Err(crate::Error::InvalidRequest(
                    "repetition_penalty must be finite and > 0".into(),
                ));
            }
        }
        if let Some(p) = self.sampling.top_p {
            if !(0.0 < p && p <= 1.0) {
                return Err(crate::Error::InvalidRequest(
                    "top_p must be in (0, 1]".into(),
                ));
            }
        }
        if let Some(k) = self.sampling.top_k {
            if k == 0 {
                return Err(crate::Error::InvalidRequest(
                    "top_k must be > 0 (omit to disable)".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Lifecycle status of a sequence inside the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Admitted to the running set but prefill has not executed yet.
    Waiting,
    /// Prefill scheduled / in progress.
    Prefill,
    /// Incremental decoding.
    Running,
    Finished,
    Cancelled,
    Failed,
}

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS / stop token / stop string.
    Stop,
    /// `max_new_tokens` reached.
    Length,
    Cancelled,
    Error,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
            FinishReason::Error => "error",
        }
    }
}

/// Token accounting for a finished request (OpenAI-style `usage`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Transport-independent failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationErrorKind {
    InvalidRequest,
    Overloaded,
    Timeout,
    Execution,
}
impl GenerationErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request_error",
            Self::Overloaded => "engine_overloaded",
            Self::Timeout => "request_timeout",
            Self::Execution => "internal_error",
        }
    }
}

/// Events streamed from the engine to a single request's consumer.
///
/// The engine is transport-agnostic: SSE conversion happens in the server
/// layer.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// A newly generated token. `text` is the incremental detokenized delta
    /// (possibly empty when a token does not produce new characters yet).
    Token { token_id: u32, text: String },
    /// Terminal event; the engine closes the channel after sending it.
    Finished { reason: FinishReason, usage: Usage },
    /// Fatal error for this request; the engine closes the channel after it.
    Error {
        kind: GenerationErrorKind,
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    fn req(prompt: usize, max_new: usize) -> GenerationRequest {
        GenerationRequest {
            id: "r1".into(),
            prompt_token_ids: vec![1; prompt],
            sampling: SamplingParams::default(),
            max_new_tokens: max_new,
            stop_token_ids: vec![],
            stop_strings: vec![],
        }
    }

    #[test]
    fn rejects_context_overflow() {
        assert!(matches!(
            req(10, 8).validate(16),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn rejects_empty_prompt_and_zero_max_new() {
        assert!(matches!(
            req(0, 4).validate(64),
            Err(Error::InvalidRequest(_))
        ));
        assert!(matches!(
            req(4, 0).validate(64),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn rejects_arithmetic_overflow_and_nonfinite_sampling() {
        assert!(req(1, usize::MAX).validate(usize::MAX).is_err());
        for value in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let mut r = req(1, 1);
            r.sampling.repetition_penalty = Some(value);
            assert!(r.validate(10).is_err());
        }
        let mut r = req(1, 1);
        r.sampling.temperature = f32::INFINITY;
        assert!(r.validate(10).is_err());
    }

    #[test]
    fn accepts_reasonable_request() {
        assert!(req(4, 12).validate(64).is_ok());
    }

    #[test]
    fn rejects_bad_sampling_params() {
        let mut r = req(4, 4);
        r.sampling.temperature = -1.0;
        assert!(r.validate(64).is_err());
        let mut r = req(4, 4);
        r.sampling.top_p = Some(1.5);
        assert!(r.validate(64).is_err());
        let mut r = req(4, 4);
        r.sampling.top_k = Some(0);
        assert!(r.validate(64).is_err());
    }
}
