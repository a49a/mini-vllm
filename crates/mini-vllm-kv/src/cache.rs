//! Per-sequence KV tensor storage.
//!
//! Layout (per layer): `key/value: [num_kv_heads, capacity, head_dim]`,
//! pre-allocated once at admission for the full `prompt + max_new_tokens`
//! horizon and filled **in place** with `scatter_set` along the token
//! dimension — no reallocation and no data copies per step. Reads use
//! `narrow` views over the used prefix.

use candle_core::{DType, Device, Result, Tensor};

/// Key/value storage for one transformer layer of one sequence.
#[derive(Debug)]
pub struct LayerKvCache {
    /// `[num_kv_heads, capacity, head_dim]`
    pub key: Tensor,
    /// `[num_kv_heads, capacity, head_dim]`
    pub value: Tensor,
    /// Tokens currently stored.
    len: usize,
}

impl LayerKvCache {
    fn new(
        num_kv_heads: usize,
        capacity: usize,
        head_dim: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        Ok(Self {
            key: Tensor::zeros((num_kv_heads, capacity, head_dim), dtype, dev)?,
            value: Tensor::zeros((num_kv_heads, capacity, head_dim), dtype, dev)?,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.key.dim(1).unwrap_or(0)
    }

    /// Append new `k`/`v` blocks of shape `[num_kv_heads, n, head_dim]` at
    /// the write cursor and return full views
    /// `[num_kv_heads, len, head_dim]` covering everything cached so far.
    pub fn write(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (nkv, n, hd) = (k.dim(0)?, k.dim(1)?, k.dim(2)?);
        debug_assert_eq!(k.dims(), v.dims());
        if self.len + n > self.capacity() {
            candle_core::bail!(
                "kv cache overflow: len {} + n {} > capacity {}",
                self.len,
                n,
                self.capacity()
            );
        }
        // Scatter positions `[len .. len+n)` along the token dim; the same
        // position applies to every head and every feature.
        let mut idx: Vec<u32> = Vec::with_capacity(nkv * n * hd);
        for _ in 0..nkv {
            for t in 0..n {
                let e = (self.len + t) as u32;
                for _ in 0..hd {
                    idx.push(e);
                }
            }
        }
        let idx = Tensor::from_vec(idx, (nkv, n, hd), self.key.device())?;
        self.key.scatter_set(&idx, k, 1)?;
        self.value.scatter_set(&idx, v, 1)?;
        self.len += n;
        Ok((
            self.key.narrow(1, 0, self.len)?,
            self.value.narrow(1, 0, self.len)?,
        ))
    }
}

/// All-layer KV cache of a single sequence.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    paged: Option<Vec<crate::paged::PagedLayer>>,
    /// Number of tokens currently cached (identical across layers; enforced
    /// by the model writing every layer once per step).
    seq_len: usize,
    capacity: usize,
}

impl KvCache {
    /// Allocate an empty cache for `num_layers` layers.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new(num_kv_heads, capacity, head_dim, dtype, dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            paged: None,
            seq_len: 0,
            capacity,
        })
    }

    /// Allocate physical pages lazily. The contiguous constructor is retained
    /// as a numerical reference and for callers requiring contiguous views.
    pub fn new_paged(num_layers: usize, capacity: usize, block_size: usize) -> Result<Self> {
        if num_layers == 0 || capacity == 0 || block_size == 0 {
            candle_core::bail!("paged KV dimensions must be positive");
        }
        Ok(Self {
            layers: vec![],
            paged: Some(
                (0..num_layers)
                    .map(|_| crate::paged::PagedLayer::new(block_size, capacity))
                    .collect(),
            ),
            seq_len: 0,
            capacity,
        })
    }

    /// Share immutable prefix pages; future partial-page writes use COW.
    pub fn fork_prefix(&self, len: usize, capacity: usize) -> Result<Self> {
        if len > capacity || len > self.seq_len {
            candle_core::bail!("invalid prefix length");
        }
        let Some(layers) = &self.paged else {
            candle_core::bail!("prefix sharing requires paged storage");
        };
        let mut layers = layers.clone();
        for layer in &mut layers {
            layer.truncate(len)?;
        }
        Ok(Self {
            layers: vec![],
            paged: Some(
                layers
                    .into_iter()
                    .map(|l| l.with_capacity(capacity))
                    .collect(),
            ),
            seq_len: len,
            capacity,
        })
    }

    /// Count persistent K/V storage tensor allocations, excluding temporary
    /// attention tensors and index buffers. Prefix forks share this counter.
    pub fn track_allocations(&mut self, counter: std::sync::Arc<std::sync::atomic::AtomicU64>) {
        if let Some(layers) = &mut self.paged {
            for layer in layers {
                layer.track_allocations(counter.clone());
            }
        } else {
            counter.fetch_add(
                (2 * self.layers.len()) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }

    /// Replace identical, block-aligned history with canonical shared pages.
    pub fn share_prefix_from(&mut self, source: &Self, len: usize) -> Result<()> {
        let (Some(dst), Some(src)) = (&mut self.paged, &source.paged) else {
            candle_core::bail!("canonical sharing requires paged storage");
        };
        if dst.len() != src.len() {
            candle_core::bail!("prefix layer mismatch");
        }
        for (dst, src) in dst.iter_mut().zip(src) {
            dst.share_prefix_from(src, len)?;
        }
        Ok(())
    }

    pub fn is_paged(&self) -> bool {
        self.paged.is_some()
    }

    /// Append and return individual page views for block-addressed attention.
    pub fn write_layer_pages(
        &mut self,
        i: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        if let Some(layers) = &mut self.paged {
            let pages = layers[i].write(k, v)?;
            if i == 0 {
                self.seq_len = layers[0].len();
            }
            Ok(pages)
        } else {
            Ok(vec![self.write_layer(i, k, v)?])
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn layers(&mut self) -> &mut [LayerKvCache] {
        &mut self.layers
    }

    /// Rewind an append-only forward pass to its last committed length.
    /// Stale tail values are hidden and overwritten by the next append.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if let Some(layers) = &mut self.paged {
            if layers.iter().any(|l| l.len() < len) {
                candle_core::bail!("invalid paged rollback length");
            }
            for layer in layers {
                layer.truncate(len)?;
            }
            self.seq_len = len;
            return Ok(());
        }
        if self.layers.iter().any(|layer| layer.len < len) {
            candle_core::bail!("cannot extend KV cache when rolling back to {len}");
        }
        for layer in &mut self.layers {
            layer.len = len;
        }
        self.seq_len = len;
        Ok(())
    }

    /// Write the step's `k`/`v` into layer `i` and return full-length views.
    pub fn write_layer(&mut self, i: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.paged.is_some() {
            let pages = self.write_layer_pages(i, k, v)?;
            let keys: Vec<_> = pages.iter().map(|p| &p.0).collect();
            let values: Vec<_> = pages.iter().map(|p| &p.1).collect();
            return Ok((Tensor::cat(&keys, 1)?, Tensor::cat(&values, 1)?));
        }
        let views = self.layers[i].write(k, v)?;
        if i == 0 {
            self.seq_len = self.layers[0].len();
        }
        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(capacity: usize) -> KvCache {
        KvCache::new(2, 2, 4, capacity, DType::F32, &Device::Cpu).unwrap()
    }

    fn kv(heads: usize, n: usize, head_dim: usize, fill: f32) -> Tensor {
        let v: Vec<f32> = vec![fill; heads * n * head_dim];
        Tensor::from_vec(v, (heads, n, head_dim), &Device::Cpu).unwrap()
    }

    #[test]
    fn write_then_read_back_views() {
        let mut c = cache(8);
        let (k, v) = c
            .write_layer(0, &kv(2, 3, 4, 1.0), &kv(2, 3, 4, 2.0))
            .unwrap();
        assert_eq!(k.dims(), &[2, 3, 4]);
        assert_eq!(v.dims(), &[2, 3, 4]);
        assert_eq!(c.seq_len(), 3);
        // Per-head check: the first 3 token slots hold the data, untouched
        // capacity stays zero.
        let head0 = c.layers[0]
            .key
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(head0[..3 * 4].iter().all(|&x| x == 1.0));
        assert!(head0[3 * 4..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn layers_are_independent() {
        let mut c = cache(8);
        c.write_layer(0, &kv(2, 2, 4, 1.0), &kv(2, 2, 4, 1.0))
            .unwrap();
        c.write_layer(1, &kv(2, 2, 4, 9.0), &kv(2, 2, 4, 9.0))
            .unwrap();
        let l1 = c.layers[1]
            .key
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(l1[..8].iter().all(|&x| x == 9.0));
    }

    #[test]
    fn overflow_is_an_error() {
        let mut c = cache(4);
        c.write_layer(0, &kv(2, 3, 4, 1.0), &kv(2, 3, 4, 1.0))
            .unwrap();
        let r = c.write_layer(0, &kv(2, 3, 4, 1.0), &kv(2, 3, 4, 1.0));
        assert!(r.is_err());
    }

    #[test]
    fn sequential_writes_keep_prefix() {
        let mut c = cache(8);
        c.write_layer(0, &kv(2, 2, 4, 1.0), &kv(2, 2, 4, 1.0))
            .unwrap();
        c.write_layer(0, &kv(2, 1, 4, 5.0), &kv(2, 1, 4, 5.0))
            .unwrap();
        let head0 = c.layers[0]
            .key
            .narrow(0, 0, 1)
            .unwrap()
            .narrow(1, 0, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(head0[..8].iter().all(|&x| x == 1.0));
        assert!(head0[8..].iter().all(|&x| x == 5.0));
        assert_eq!(c.seq_len(), 3);
    }
}
