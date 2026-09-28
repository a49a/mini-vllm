//! Typed errors for the KV subsystem.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    /// Admission-time or grow-time capacity exhaustion (never an OOM crash —
    /// the engine uses this as normal flow control).
    #[error("out of KV capacity: requested {requested} blocks, free {free}")]
    OutOfCapacity { requested: usize, free: usize },

    #[error("unknown sequence `{0}` in block table")]
    UnknownSequence(String),

    #[error("kv error: {0}")]
    Other(String),
}
