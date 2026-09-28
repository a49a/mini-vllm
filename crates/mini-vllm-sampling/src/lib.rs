//! Sampling subsystem.
//!
//! Owns the explicit logits → token pipeline:
//!
//! ```text
//! logits
//!   → repetition penalty
//!   → temperature (0 ⇒ greedy argmax)
//!   → top-k filtering
//!   → top-p filtering
//!   → softmax
//!   → sample
//! ```
//!
//! The sampler works on plain `&[f32]` so it stays independent of the tensor
//! runtime; the engine copies one row of logits out of the model output.

pub mod sampler;

pub use sampler::Sampler;
