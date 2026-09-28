//! OpenAI-compatible HTTP server.
//!
//! Architectural boundary (design doc §65): the HTTP layer never touches
//! model internals. It validates requests, tokenizes, submits to the engine
//! command queue, and translates engine events into JSON or SSE.

pub mod api;
pub mod openai;
pub mod preprocessing;
pub mod preprocessing_metrics;
pub mod routes;
pub mod streaming;

use std::sync::Arc;

use mini_vllm_engine::EngineHandle;
use mini_vllm_tokenizer::{ChatTemplate, TokenizerWrapper};

/// Subset of engine-like behavior the HTTP layer needs; mockable in tests.
pub trait EngineApi: Send + Sync + 'static {
    fn generate(
        &self,
        request: mini_vllm_core::GenerationRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<mini_vllm_core::GenerationEvent>,
        mini_vllm_engine::EngineApiError,
    >;

    fn cancel(&self, request_id: &str);
    fn default_max_new_tokens(&self) -> usize {
        mini_vllm_core::EngineConfig::default().default_max_new_tokens
    }

    fn is_accepting(&self) -> bool {
        true
    }

    fn metrics(&self) -> mini_vllm_engine::MetricsSnapshot;
}

impl EngineApi for EngineHandle {
    fn generate(
        &self,
        request: mini_vllm_core::GenerationRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<mini_vllm_core::GenerationEvent>,
        mini_vllm_engine::EngineApiError,
    > {
        EngineHandle::generate(self, request)
    }

    fn default_max_new_tokens(&self) -> usize {
        EngineHandle::default_max_new_tokens(self)
    }

    fn cancel(&self, request_id: &str) {
        EngineHandle::cancel(self, request_id).ok();
    }

    fn is_accepting(&self) -> bool {
        EngineHandle::is_accepting(self)
    }

    fn metrics(&self) -> mini_vllm_engine::MetricsSnapshot {
        self.metrics().snapshot()
    }
}

/// Everything a handler needs.
pub struct AppState {
    pub engine: Arc<dyn EngineApi>,
    pub tokenizer: Arc<TokenizerWrapper>,
    pub template: Arc<dyn ChatTemplate>,
    /// Model id advertised through the API (directory name by default).
    pub model_id: String,
    pub max_model_len: usize,
    pub vocab_size: usize,
}

pub type SharedState = Arc<AppState>;
