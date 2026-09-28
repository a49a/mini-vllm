//! Inference engine.
//!
//! Concurrency model (see design doc §40):
//!
//! ```text
//! Tokio HTTP tasks
//!        ↓  mpsc commands (bounded)
//! single engine OS thread
//!        ↓
//! scheduler → batched model execution → per-request event channels (bounded)
//! ```
//!
//! The engine owns all mutable runtime state (scheduler queues, sequences,
//! KV caches, block manager), which makes the invariants easy to reason
//! about: nothing else mutates them.

pub mod batch;
pub mod engine;
pub mod executor;
pub mod metrics;
pub mod request;
pub mod scheduler;
pub mod sequence;

pub use engine::{spawn_engine, EngineHandle, ShutdownMode};
pub use metrics::{Metrics, MetricsSnapshot};
pub use request::{EngineApiError, EngineCommand};
pub use sequence::{Emission, SequenceGroup};

pub mod lifecycle;
