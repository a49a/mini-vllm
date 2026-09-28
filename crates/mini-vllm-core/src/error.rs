//! Typed errors for the core domain.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    /// The caller submitted something the engine refuses to run.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The request cannot run with current runtime limits / KV capacity.
    #[error("request rejected: {0}")]
    Capacity(String),
}
