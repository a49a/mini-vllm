//! Typed errors for model loading and execution.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("model config: {0}")]
    Config(String),

    #[error("missing weight `{0}` in safetensors files")]
    MissingWeight(String),

    #[error("unsupported device `{0}` (not compiled in or not available)")]
    UnsupportedDevice(String),

    #[error("unsupported dtype `{0}`")]
    UnsupportedDtype(String),

    #[error("no safetensors weights found for model")]
    NoWeights,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),
}
