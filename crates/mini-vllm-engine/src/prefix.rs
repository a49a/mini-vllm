//! Block-keyed prefix trie. Each node owns one unique block across all layers;
//! nodes retain only their own pages. Only unpinned leaves can be evicted.
use mini_vllm_kv::KvCache;
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

struct Node {
    parent: Option<usize>,
    block: Arc<[u32]>,
    depth: usize,
    children: usize,
    used: u64,
    cache: KvCache,
    pin: Arc<()>,
}
pub struct PrefixCache {
    index: HashMap<Option<usize>, HashMap<Arc<[u32]>, usize>>,
    leaves: BTreeSet<(u64, usize)>,
    nodes: HashMap<usize, Node>,
    block_size: usize,
    max_blocks: usize,
    next: usize,
    clock: u64,
}
impl PrefixCache {
    pub fn new(block_size: usize, tokens: usize) -> Self {
        Self {
            index: HashMap::new(),
            leaves: BTreeSet::new(),
            nodes: HashMap::new(),
            block_size,
            max_blocks: tokens / block_size,
            next: 0,
            clock: 0,
        }
    }
    pub fn tokens(&self) -> usize {
        self.nodes.len() * self.block_size
    }
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.index.clear();
        self.leaves.clear();
    }
    fn lookup(&self, parent: Option<usize>, block: &[u32]) -> Option<&usize> {
        self.index.get(&parent)?.get(block)
    }
    fn touch(&mut self, id: usize) {
        let node = self.nodes.get_mut(&id).unwrap();
        if node.children == 0 {
            self.leaves.remove(&(node.used, id));
            self.leaves.insert((self.clock, id));
        }
        node.used = self.clock;
    }
    /// Logical retained metadata, independent of model weights and tensor sizes.
    /// These counts are not allocator bytes or process RSS.
    pub fn metadata_counts(&self) -> (usize, usize, usize) {
        (
            self.nodes.len(),
            self.nodes.values().map(|n| n.block.len()).sum(),
            self.nodes
                .values()
                .map(|n| n.cache.page_table_entries())
                .sum(),
        )
    }
    fn find(&self, prompt: &[u32]) -> Option<usize> {
        let mut parent = None;
        for block in prompt[..prompt.len().saturating_sub(1)].chunks_exact(self.block_size) {
            match self.lookup(parent, block) {
                Some(&id) => parent = Some(id),
                None => break,
            }
        }
        parent
    }
    pub fn matched_tokens(&self, prompt: &[u32]) -> usize {
        self.find(prompt)
            .map(|id| self.nodes[&id].depth * self.block_size)
            .unwrap_or(0)
    }
    pub fn borrow(
        &mut self,
        prompt: &[u32],
        capacity: usize,
    ) -> candle_core::Result<Option<(KvCache, Arc<()>)>> {
        let Some(id) = self.find(prompt) else {
            return Ok(None);
        };
        self.clock += 1;
        self.touch(id);
        let node = &self.nodes[&id];
        let pin = node.pin.clone();
        let mut path = vec![id];
        let mut parent = node.parent;
        while let Some(ancestor) = parent {
            path.push(ancestor);
            parent = self.nodes[&ancestor].parent;
        }
        path.reverse();
        let mut cache = self.nodes[&path[0]]
            .cache
            .fork_prefix(self.block_size, capacity)?;
        for ancestor in path.into_iter().skip(1) {
            cache.append_shared_block(&self.nodes[&ancestor].cache)?;
        }
        Ok(Some((cache, pin)))
    }
    fn evict_leaf(&mut self, protected: Option<usize>) -> bool {
        // Pinned leaves stay indexed: dropping a borrow needs no mutation here.
        // Search only leaves in LRU order, never the trie interior.
        let victim = self.leaves.iter().find_map(|&(_, id)| {
            (Some(id) != protected && Arc::strong_count(&self.nodes[&id].pin) == 1).then_some(id)
        });
        let Some(id) = victim else {
            return false;
        };
        let node = self.nodes.remove(&id).unwrap();
        self.leaves.remove(&(node.used, id));
        let siblings = self.index.get_mut(&node.parent).unwrap();
        siblings.remove(node.block.as_ref());
        if siblings.is_empty() {
            self.index.remove(&node.parent);
        }
        if let Some(parent) = node.parent {
            let ancestor = self.nodes.get_mut(&parent).unwrap();
            ancestor.children -= 1;
            if ancestor.children == 0 {
                self.leaves.insert((ancestor.used, parent));
            }
        }
        true
    }
    pub fn insert(&mut self, tokens: &[u32], cache: &mut KvCache) -> candle_core::Result<()> {
        if self.max_blocks == 0 {
            return Ok(());
        }
        let mut parent = None;
        for (i, block) in tokens.chunks_exact(self.block_size).enumerate() {
            self.clock += 1;
            if let Some(&id) = self.lookup(parent, block) {
                self.touch(id);
                let node = &self.nodes[&id];
                // Concurrent cold requests may have computed identical history.
                // Canonicalize before adding any descendants, so accounting
                // reflects unique tensors rather than overlapping snapshots.
                cache.share_block_from(&node.cache, i * self.block_size)?;
                parent = Some(id);
                continue;
            }
            while self.nodes.len() >= self.max_blocks {
                if !self.evict_leaf(parent) {
                    return Ok(());
                }
            }
            let snapshot = cache.fork_block(i * self.block_size)?;
            let id = self.next;
            self.next += 1;
            if let Some(parent) = parent {
                let ancestor = self.nodes.get_mut(&parent).unwrap();
                self.leaves.remove(&(ancestor.used, parent));
                ancestor.children += 1;
            }
            let block: Arc<[u32]> = block.into();
            self.nodes.insert(
                id,
                Node {
                    parent,
                    block: block.clone(),
                    depth: i + 1,
                    children: 0,
                    used: self.clock,
                    cache: snapshot,
                    pin: Arc::new(()),
                },
            );
            self.index.entry(parent).or_default().insert(block, id);
            self.leaves.insert((self.clock, id));
            parent = Some(id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cache(n: usize) -> KvCache {
        let mut cache = KvCache::new_paged(1, 16, 2).unwrap();
        let t = candle_core::Tensor::ones(
            (1, n, 2),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        cache.write_layer_pages(0, &t, &t).unwrap();
        cache
    }
    #[test]
    fn overlapping_prefixes_charge_each_block_once_and_preserve_borrowers() {
        let mut trie = PrefixCache::new(2, 6);
        let mut a = cache(6);
        trie.insert(&[1, 2, 3, 4], &mut a).unwrap();
        trie.insert(&[1, 2, 3, 4, 5, 6], &mut a).unwrap();
        assert_eq!(trie.tokens(), 6);
        assert_eq!(
            trie.nodes
                .values()
                .map(|n| n.cache.page_table_entries())
                .sum::<usize>(),
            3
        );
        assert_eq!(trie.matched_tokens(&[1, 2, 3, 4, 5, 6, 7]), 6);
        assert_eq!(
            trie.matched_tokens(&[1, 2, 3, 4]),
            2,
            "leave a token for logits"
        );
        let (borrowed, pin) = trie.borrow(&[1, 2, 3, 4, 5, 6, 7], 16).unwrap().unwrap();
        let mut b = cache(6);
        trie.insert(&[1, 2, 9, 9], &mut b).unwrap();
        assert_eq!(
            trie.matched_tokens(&[1, 2, 9, 9, 7]),
            2,
            "pinned path consumes whole pool"
        );
        assert_eq!(borrowed.seq_len(), 6);
        drop(pin);
        drop(borrowed);
        trie.insert(&[1, 2, 9, 9], &mut b).unwrap();
        assert_eq!(trie.matched_tokens(&[1, 2, 9, 9, 7]), 4);
        assert_eq!(trie.matched_tokens(&[1, 2, 3, 4, 7]), 4);
        assert_eq!(trie.tokens(), 6);
        trie.insert(&[1, 2, 9, 9], &mut b).unwrap();
        assert_eq!(trie.tokens(), 6, "duplicate insert must not charge again");
    }

    #[test]
    fn borrowed_path_reassembles_pages_in_order() {
        let mut source = KvCache::new_paged(1, 8, 2).unwrap();
        let values = candle_core::Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6.],
            (1, 6, 1),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        source.write_layer_pages(0, &values, &values).unwrap();
        let mut trie = PrefixCache::new(2, 6);
        trie.insert(&[1, 2, 3, 4, 5, 6], &mut source).unwrap();
        assert_eq!(
            trie.nodes
                .values()
                .map(|n| n.cache.page_table_entries())
                .sum::<usize>(),
            3
        );
        let (mut borrowed, _pin) = trie.borrow(&[1, 2, 3, 4, 5, 6, 7], 8).unwrap().unwrap();
        let next = candle_core::Tensor::from_vec(vec![7f32], (1, 1, 1), &candle_core::Device::Cpu)
            .unwrap();
        let pages = borrowed.write_layer_pages(0, &next, &next).unwrap();
        let flattened: Vec<f32> = pages
            .into_iter()
            .flat_map(|(k, _)| k.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .collect();
        assert_eq!(flattened, vec![1., 2., 3., 4., 5., 6., 7.]);
    }
    #[test]
    fn lru_refresh_and_parent_promotion_preserve_eviction_order() {
        let mut trie = PrefixCache::new(2, 6);
        trie.insert(&[1, 2, 3, 4], &mut cache(4)).unwrap();
        trie.insert(&[5, 6], &mut cache(2)).unwrap();
        let (_, pin) = trie.borrow(&[1, 2, 3, 4, 0], 16).unwrap().unwrap();
        drop(pin);
        trie.insert(&[7, 8], &mut cache(2)).unwrap();
        assert_eq!(trie.matched_tokens(&[5, 6, 0]), 0);
        assert!(trie.evict_leaf(None)); // [3,4] is now the oldest leaf.
        assert_eq!(trie.matched_tokens(&[1, 2, 3, 4, 0]), 2);
        assert!(trie.evict_leaf(None)); // promoted [1,2] retains its old age.
        assert_eq!(trie.matched_tokens(&[1, 2, 0]), 0);
        assert_eq!(trie.matched_tokens(&[7, 8, 0]), 2);
        trie.clear();
        assert!(trie.leaves.is_empty());
        assert!(trie.index.is_empty());
    }
}
