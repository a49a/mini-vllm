//! Block-keyed prefix trie. Each node owns one unique block across all layers;
//! nodes retain only their own pages. Only unpinned leaves can be evicted.
use mini_vllm_kv::KvCache;
use std::{collections::HashMap, sync::Arc};

struct Node {
    parent: Option<usize>,
    block: Vec<u32>,
    depth: usize,
    children: usize,
    used: u64,
    cache: KvCache,
    pin: Arc<()>,
}
pub struct PrefixCache {
    index: HashMap<(Option<usize>, Vec<u32>), usize>,
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
    }
    fn find(&self, prompt: &[u32]) -> Option<usize> {
        let mut parent = None;
        for block in prompt[..prompt.len().saturating_sub(1)].chunks_exact(self.block_size) {
            match self.index.get(&(parent, block.to_vec())) {
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
        let node = self.nodes.get_mut(&id).unwrap();
        node.used = self.clock;
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
        let victim = self
            .nodes
            .iter()
            .filter(|(id, n)| {
                Some(**id) != protected && n.children == 0 && Arc::strong_count(&n.pin) == 1
            })
            .min_by_key(|(_, n)| n.used)
            .map(|(&id, _)| id);
        let Some(id) = victim else {
            return false;
        };
        let node = self.nodes.remove(&id).unwrap();
        self.index.remove(&(node.parent, node.block));
        if let Some(parent) = node.parent {
            self.nodes.get_mut(&parent).unwrap().children -= 1;
        }
        true
    }
    pub fn insert(&mut self, tokens: &[u32], cache: &mut KvCache) -> candle_core::Result<()> {
        if self.max_blocks == 0 {
            return Ok(());
        }
        let mut parent = None;
        for (i, block) in tokens.chunks_exact(self.block_size).enumerate() {
            let key = (parent, block.to_vec());
            self.clock += 1;
            if let Some(&id) = self.index.get(&key) {
                let node = self.nodes.get_mut(&id).unwrap();
                node.used = self.clock;
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
                self.nodes.get_mut(&parent).unwrap().children += 1;
            }
            self.nodes.insert(
                id,
                Node {
                    parent,
                    block: block.to_vec(),
                    depth: i + 1,
                    children: 0,
                    used: self.clock,
                    cache: snapshot,
                    pin: Arc::new(()),
                },
            );
            self.index.insert(key, id);
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
}
