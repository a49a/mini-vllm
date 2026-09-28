//! Fixed-size KV block types (logical view, inspired by paged KV caches).

/// Physical block identifier into the logical block pool.
pub type BlockId = usize;

/// Default number of tokens cached per physical block.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Mapping from a sequence's logical blocks to physical blocks.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    pub blocks: Vec<BlockId>,
}

impl BlockTable {
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn capacity_tokens(&self, block_size: usize) -> usize {
        self.blocks.len() * block_size
    }
}

/// Convert a token count to the number of blocks that hold it.
pub fn blocks_for_tokens(tokens: usize, block_size: usize) -> usize {
    if block_size == 0 {
        return 0;
    }
    tokens.div_ceil(block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_up_to_blocks() {
        assert_eq!(blocks_for_tokens(0, 16), 0);
        assert_eq!(blocks_for_tokens(1, 16), 1);
        assert_eq!(blocks_for_tokens(16, 16), 1);
        assert_eq!(blocks_for_tokens(17, 16), 2);
    }
}
