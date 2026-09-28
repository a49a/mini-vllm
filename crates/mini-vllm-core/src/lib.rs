//! Core domain types shared by every mini-vllm-rs crate.
//!
//! This crate deliberately has no dependency on candle, tokio, or HTTP
//! frameworks: it defines the language in which the subsystems talk to
//! each other (requests, sampling parameters, events, engine limits).

pub mod config;
pub mod error;
pub mod types;

pub use config::EngineConfig;
pub use error::{Error, Result};
pub use types::*;
