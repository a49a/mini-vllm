//! Runtime (engine) configuration — deliberately separate from model
//! configuration, which lives in `mini-vllm-model`.

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Hard cap on `prompt_len + generated_len` per sequence.
    pub max_model_len: usize,
    /// Maximum concurrently running sequences (scheduler admission limit).
    pub max_num_seqs: usize,
    /// Maximum combined prefill and decode tokens in one engine step.
    pub max_batch_tokens: usize,
    /// Per-sequence prefill slice; the scheduler rotates between slices.
    pub max_prefill_chunk_tokens: usize,
    /// Emit teaching traces (request ids/positions only, never prompt text).
    pub trace_requests: bool,
    /// Optional teaching JSONL file. Must not already exist.
    pub trace_jsonl: Option<std::path::PathBuf>,
    /// Submission-to-admission timeout; zero disables.
    pub queue_timeout_ms: u64,
    /// Submission-to-generation-completion deadline; zero disables.
    pub request_timeout_ms: u64,
    /// Token count per KV "block" used by the block manager for accounting.
    pub kv_block_size: usize,
    /// Total KV cache budget expressed in tokens (converted to blocks).
    pub max_kv_tokens: usize,
    /// Maximum accepted requests waiting for admission.
    pub max_waiting_requests: usize,
    /// Milliseconds allowed to flush completed output after releasing KV.
    pub output_drain_timeout_ms: u64,
    /// Enable physical KV pages (contiguous storage remains the reference).
    pub paged_kv: bool,
    /// Separate bounded budget for retained prefix pages; zero disables reuse.
    pub prefix_cache_tokens: usize,
    /// Default `max_new_tokens` when a client does not specify one.
    pub default_max_new_tokens: usize,
    /// Bounded capacity of per-request streaming event channels (backpressure).
    pub event_channel_capacity: usize,
    /// Bounded capacity of the engine command queue.
    pub command_channel_capacity: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_model_len: 4096,
            max_num_seqs: 32,
            max_batch_tokens: 2048,
            max_prefill_chunk_tokens: 256,
            trace_requests: false,
            trace_jsonl: None,
            queue_timeout_ms: 0,
            request_timeout_ms: 0,
            kv_block_size: 16,
            max_kv_tokens: 32 * 1024,
            max_waiting_requests: 256,
            output_drain_timeout_ms: 1000,
            paged_kv: true,
            prefix_cache_tokens: 0,
            default_max_new_tokens: 512,
            event_channel_capacity: 64,
            command_channel_capacity: 256,
        }
    }
}

impl EngineConfig {
    pub fn validate(&self) -> crate::Result<()> {
        for (name, value) in [
            ("max_model_len", self.max_model_len),
            ("max_num_seqs", self.max_num_seqs),
            ("max_batch_tokens", self.max_batch_tokens),
            ("max_prefill_chunk_tokens", self.max_prefill_chunk_tokens),
            ("kv_block_size", self.kv_block_size),
            ("max_kv_tokens", self.max_kv_tokens),
            ("max_waiting_requests", self.max_waiting_requests),
            ("event_channel_capacity", self.event_channel_capacity),
            ("command_channel_capacity", self.command_channel_capacity),
            ("default_max_new_tokens", self.default_max_new_tokens),
        ] {
            if value == 0 {
                return Err(crate::Error::InvalidRequest(format!("{name} must be > 0")));
            }
        }
        if self.max_model_len > u32::MAX as usize || self.max_kv_tokens < self.kv_block_size {
            return Err(crate::Error::InvalidRequest(
                "context must fit u32 and KV pool must hold one block".into(),
            ));
        }
        if self
            .max_num_seqs
            .checked_add(self.max_waiting_requests)
            .is_none()
        {
            return Err(crate::Error::InvalidRequest(
                "request capacity overflow".into(),
            ));
        }
        if !self.paged_kv && self.prefix_cache_tokens > 0 {
            return Err(crate::Error::InvalidRequest(
                "prefix caching requires paged KV".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_limits_are_validated_before_channels_or_allocations() {
        for cfg in [
            EngineConfig {
                event_channel_capacity: 0,
                ..EngineConfig::default()
            },
            EngineConfig {
                max_batch_tokens: 0,
                ..EngineConfig::default()
            },
            EngineConfig {
                kv_block_size: 0,
                ..EngineConfig::default()
            },
            EngineConfig {
                max_kv_tokens: 1,
                ..EngineConfig::default()
            },
            EngineConfig {
                paged_kv: false,
                prefix_cache_tokens: 32,
                ..EngineConfig::default()
            },
        ] {
            assert!(cfg.validate().is_err());
        }
    }
}
