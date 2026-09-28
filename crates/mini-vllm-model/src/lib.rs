//! Model subsystem: configuration parsing, device selection, Qwen2
//! components, and weight loading.
//!
//! Architectural rule: the engine depends on the [`CausalLm`] trait only —
//! never on Qwen-specific internals.

pub mod attention;
pub mod config;
pub mod device;
pub mod error;
pub mod linear;
pub mod loader;
pub mod mlp;
pub mod qwen2;
pub mod rms_norm;
pub mod rope;

/// Random-weight fixtures for tests. Purely helper functions, safe to ship;
/// engine/server tests build tiny models from them.
pub mod testutil;

pub use config::ModelConfig;
pub use device::{resolve_device, resolve_dtype};
pub use error::{Error, Result};
pub use qwen2::{BatchTokens, CausalLm, Qwen2};
