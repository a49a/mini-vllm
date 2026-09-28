//! Engine command protocol and handle-side error type.

use mini_vllm_core::{GenerationEvent, GenerationRequest, RequestId};
use tokio::sync::mpsc;

/// Commands accepted by the engine loop (design doc §25).
#[derive(Debug)]
pub enum EngineCommand {
    Generate {
        request: GenerationRequest,
        /// Bounded event channel for this request; the engine closes it
        /// after emitting the terminal event.
        events: mpsc::Sender<GenerationEvent>,
        lease: crate::lifecycle::RequestLease,
    },
    Cancel {
        request_id: RequestId,
    },
}

/// Why a submission (or cancellation) could not reach the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineApiError {
    /// The bounded command queue is full — the server is overloaded, not
    /// shutting down. Clients may retry.
    QueueFull,
    InvalidRequest(String),
    /// The engine loop has exited.
    ShuttingDown,
}

impl std::fmt::Display for EngineApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineApiError::InvalidRequest(message) => write!(f, "{message}"),
            EngineApiError::QueueFull => {
                write!(f, "engine command queue is full (server overloaded)")
            }
            EngineApiError::ShuttingDown => {
                write!(f, "engine is not accepting requests (shutting down)")
            }
        }
    }
}

impl std::error::Error for EngineApiError {}
