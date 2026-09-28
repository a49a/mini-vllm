//! KV storage: physical immutable pages with copy-on-write prefix sharing,
//! a contiguous numerical reference, and conservative logical admission blocks.
//! The logical block pool is not a global physical tensor allocator.

pub mod block;
pub mod cache;
pub mod error;
pub mod manager;

pub use block::{blocks_for_tokens, BlockId, BlockTable, DEFAULT_BLOCK_SIZE};
pub use cache::{KvCache, LayerKvCache};
pub use error::{Error, Result};
pub use manager::{KvBlockManager, KvMemoryUsage};

pub mod paged;
