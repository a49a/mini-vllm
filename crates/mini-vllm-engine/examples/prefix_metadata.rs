//! No model: measure trie metadata scaling and allocations in matched_tokens.
use mini_vllm_engine::prefix::PrefixCache;
use mini_vllm_kv::KvCache;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};
struct Counting;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
fn main() -> candle_core::Result<()> {
    let mut rows = Vec::new();
    for blocks in [16, 64, 256, 1024] {
        let n = blocks * 8;
        let mut source = KvCache::new_paged(1, n + 8, 8)?;
        let t = candle_core::Tensor::ones(
            (1, n, 1),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )?;
        source.write_layer_pages(0, &t, &t)?;
        let mut trie = PrefixCache::new(8, n);
        let tokens: Vec<u32> = (0..=n as u32).collect();
        let start = Instant::now();
        trie.insert(&tokens[..n], &mut source)?;
        let insert_us = start.elapsed().as_micros();
        let start = Instant::now();
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        for _ in 0..1000 {
            assert_eq!(
                std::hint::black_box(trie.matched_tokens(std::hint::black_box(&tokens))),
                n
            );
        }
        let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
        let lookup_us = start.elapsed().as_micros();
        assert_eq!(allocations, 0, "lookup must not allocate temporary keys");
        let (nodes, key_tokens, page_refs) = trie.metadata_counts();
        assert_eq!((nodes, key_tokens, page_refs), (blocks, n, blocks));
        // Evict a whole deep chain by inserting a disjoint prefix.
        let other: Vec<u32> = tokens.iter().map(|v| v + n as u32 + 1).collect();
        let start = Instant::now();
        trie.insert(&other[..n], &mut source)?;
        let replace_us = start.elapsed().as_micros();
        assert_eq!(trie.matched_tokens(&tokens), 0);
        rows.push(serde_json::json!({"blocks":blocks,"key_tokens":key_tokens,
            "page_references":page_refs,"insert_us":insert_us,"replace_us":replace_us,
            "lookup_1000_us":lookup_us,"lookup_allocations":allocations}));
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
    Ok(())
}
