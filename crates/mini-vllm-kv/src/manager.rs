//! Block-based KV memory manager.
//!
//! Fixed-size blocks (`kv_block_size` tokens each) tracked per sequence:
//! free pool, allocation, growth, release, and usage accounting. The engine
//! uses this for admission control and capacity reporting; running out of
//! blocks queues or rejects requests instead of crashing.
//!
//! Tensor storage can be paged or contiguous. This manager counts active
//! reservations; separately retained, pinned prefixes are charged by the engine.

use std::collections::HashMap;

use crate::block::{blocks_for_tokens, BlockId, BlockTable, DEFAULT_BLOCK_SIZE};
use crate::error::{Error, Result};

/// Snapshot of KV memory accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMemoryUsage {
    pub block_size: usize,
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub free_blocks: usize,
    pub active_sequences: usize,
    /// Approximate tokens held by allocated blocks.
    pub cached_tokens: usize,
}

/// Block pool manager: free list + per-sequence block tables.
#[derive(Debug)]
pub struct KvBlockManager {
    block_size: usize,
    total_blocks: usize,
    free_blocks: Vec<BlockId>,
    sequence_blocks: HashMap<String, BlockTable>,
}

impl KvBlockManager {
    pub fn new(block_size: usize, max_kv_tokens: usize) -> Self {
        let block_size = block_size.max(1);
        let total_blocks = blocks_for_tokens(max_kv_tokens, block_size);
        Self {
            block_size,
            total_blocks,
            free_blocks: (0..total_blocks).collect(),
            sequence_blocks: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Whether a sequence with this full horizon (`prompt + max_new_tokens`)
    /// can ever fit, regardless of current occupancy.
    pub fn can_fit_horizon(&self, prompt_tokens: usize, max_new_tokens: usize) -> bool {
        blocks_for_tokens(prompt_tokens + max_new_tokens, self.block_size) <= self.total_blocks
    }

    /// Whether `n_blocks` could be taken from the free pool right now.
    pub fn can_allocate(&self, n_blocks: usize) -> bool {
        n_blocks <= self.free_blocks.len()
    }

    /// Allocate enough blocks to hold `tokens` for `sequence_id`.
    ///
    /// Atomic on failure: a rejected allocation leaves no partial table.
    pub fn allocate(&mut self, sequence_id: &str, tokens: usize) -> Result<()> {
        let wanted = blocks_for_tokens(tokens, self.block_size);
        let existing = self
            .sequence_blocks
            .get(sequence_id)
            .map(BlockTable::len)
            .unwrap_or(0);
        let needed = wanted.saturating_sub(existing);
        if needed > self.free_blocks.len() {
            return Err(Error::OutOfCapacity {
                requested: needed,
                free: self.free_blocks.len(),
            });
        }
        let table = self
            .sequence_blocks
            .entry(sequence_id.to_string())
            .or_default();
        table.blocks.extend(self.free_blocks.drain(..needed)); // FIFO reuse
        Ok(())
    }

    /// Grow (or shrink-to-fit is not needed: only grow) the table so it
    /// covers `tokens` tokens.
    pub fn ensure_capacity(&mut self, sequence_id: &str, tokens: usize) -> Result<()> {
        self.allocate(sequence_id, tokens)
    }

    /// Release all blocks of a sequence (finish or cancel). Returns how many
    /// blocks were freed. Releasing twice is an error (leak detection).
    pub fn release(&mut self, sequence_id: &str) -> Result<usize> {
        let table = self
            .sequence_blocks
            .remove(sequence_id)
            .ok_or_else(|| Error::UnknownSequence(sequence_id.to_string()))?;
        let n = table.len();
        self.free_blocks.extend(table.blocks);
        Ok(n)
    }

    pub fn table_of(&self, sequence_id: &str) -> Option<&BlockTable> {
        self.sequence_blocks.get(sequence_id)
    }

    pub fn has_sequence(&self, sequence_id: &str) -> bool {
        self.sequence_blocks.contains_key(sequence_id)
    }

    pub fn usage(&self) -> KvMemoryUsage {
        let used = self.total_blocks - self.free_blocks.len();
        KvMemoryUsage {
            block_size: self.block_size,
            total_blocks: self.total_blocks,
            used_blocks: used,
            free_blocks: self.total_blocks - used,
            active_sequences: self.sequence_blocks.len(),
            cached_tokens: used * self.block_size,
        }
    }
}

impl Default for KvBlockManager {
    fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_release_reuse_no_leaks() {
        let mut m = KvBlockManager::new(16, 1024);
        assert_eq!(m.total_blocks(), 64);

        m.allocate("a", 40).unwrap(); // 3 blocks
        m.allocate("b", 16).unwrap(); // 1 block
        assert_eq!(m.usage().used_blocks, 4);
        assert_eq!(m.free_block_count(), 60);

        let freed = m.release("a").unwrap();
        assert_eq!(freed, 3);
        assert_eq!(m.free_block_count(), 63);
        assert!(!m.has_sequence("a"));

        // Reuse after release: a fresh sequence takes the recycled blocks.
        m.allocate("c", 63 * 16).unwrap();
        assert_eq!(m.free_block_count(), 0);
        let c = m.table_of("c").unwrap();
        assert_eq!(c.len(), 63);
    }

    #[test]
    fn double_release_is_error() {
        let mut m = KvBlockManager::new(16, 256);
        m.allocate("a", 1).unwrap();
        m.release("a").unwrap();
        assert!(m.release("a").is_err());
    }

    #[test]
    fn out_of_capacity_is_flow_control_not_panic() {
        let mut m = KvBlockManager::new(16, 32); // 2 blocks total
        assert!(matches!(
            m.allocate("big", 100),
            Err(Error::OutOfCapacity { .. })
        ));
        // Nothing was mutated by the failed allocation.
        assert!(!m.has_sequence("big"));
        m.allocate("small", 16).unwrap();
        assert_eq!(m.usage().used_blocks, 1);
    }

    #[test]
    fn grow_appends_only_missing_blocks() {
        let mut m = KvBlockManager::new(16, 512);
        m.allocate("a", 10).unwrap();
        let before = m.table_of("a").unwrap().blocks.clone();
        m.ensure_capacity("a", 40).unwrap();
        let after = m.table_of("a").unwrap();
        assert_eq!(after.len(), 3);
        assert_eq!(after.blocks[0], before[0]);
    }

    #[test]
    fn horizon_check_gates_admission() {
        let m = KvBlockManager::new(16, 1024); // 64 blocks
        assert!(m.can_fit_horizon(1000, 24)); // exactly 1024 tokens → 64 blocks
        assert!(!m.can_fit_horizon(1000, 25)); // 1025 → 65 blocks
        assert!(!m.can_fit_horizon(1, 2000));
    }

    #[test]
    fn usage_accounts_cached_tokens() {
        let mut m = KvBlockManager::new(16, 1024);
        m.allocate("a", 33).unwrap();
        let u = m.usage();
        assert_eq!(u.used_blocks, 3);
        assert_eq!(u.cached_tokens, 48);
        assert_eq!(u.active_sequences, 1);
    }
}
