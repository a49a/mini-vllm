//! Immutable physical KV pages. Full pages can be shared by prefix snapshots;
//! appending to a partial page replaces only that page (copy on write).
use candle_core::{Result, Tensor};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Debug)]
pub struct Page {
    pub key: Tensor,
    pub value: Tensor,
}

#[derive(Debug, Clone)]
pub struct PagedLayer {
    pages: Vec<Arc<Page>>,
    block_size: usize,
    len: usize,
    capacity: usize,
    allocations: Option<Arc<AtomicU64>>,
}

impl PagedLayer {
    pub fn new(block_size: usize, capacity: usize) -> Self {
        Self {
            pages: Vec::new(),
            block_size,
            len: 0,
            capacity,
            allocations: None,
        }
    }
    pub fn track_allocations(&mut self, counter: Arc<AtomicU64>) {
        self.allocations = Some(counter);
    }
    fn count_allocation(&self) {
        if let Some(counter) = &self.allocations {
            counter.fetch_add(2, Ordering::Relaxed);
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn write(&mut self, k: &Tensor, v: &Tensor) -> Result<Vec<(Tensor, Tensor)>> {
        if k.dims() != v.dims() {
            candle_core::bail!("KV shape mismatch");
        }
        let n = k.dim(1)?;
        if n > self.capacity.saturating_sub(self.len) {
            candle_core::bail!("paged KV capacity exceeded");
        }
        let mut offset = 0;
        while offset < n {
            let used = self.len % self.block_size;
            let count = (self.block_size - used).min(n - offset);
            if used == 0 {
                self.pages.push(Arc::new(Page {
                    key: Tensor::zeros(
                        (k.dim(0)?, self.block_size, k.dim(2)?),
                        k.dtype(),
                        k.device(),
                    )?,
                    value: Tensor::zeros(
                        (v.dim(0)?, self.block_size, v.dim(2)?),
                        v.dtype(),
                        v.device(),
                    )?,
                }));
                self.count_allocation();
            } else if Arc::strong_count(self.pages.last().unwrap()) > 1 {
                // Copy on write: the partial tail page is shared with a
                // prefix snapshot or another sequence's fork.
                //
                // Threading invariant: the `strong_count` check and the
                // `Arc::get_mut` below are only sound because every
                // mutation of PagedLayer happens on the single engine
                // thread — no other thread can acquire a new reference
                // between the check and the exclusive access. Sharing
                // across threads would need an atomic scheme here.
                let old = self.pages.pop().unwrap();
                self.pages.push(Arc::new(Page {
                    key: old.key.copy()?,
                    value: old.value.copy()?,
                }));
                self.count_allocation();
            }
            let page = Arc::get_mut(self.pages.last_mut().unwrap())
                .expect("exclusive page after copy-on-write");
            let (heads, dim) = (k.dim(0)?, k.dim(2)?);
            let mut indices = Vec::with_capacity(heads * count * dim);
            for _ in 0..heads {
                for pos in used..used + count {
                    indices.extend(std::iter::repeat(pos as u32).take(dim));
                }
            }
            let indices = Tensor::from_vec(indices, (heads, count, dim), k.device())?;
            page.key
                .scatter_set(&indices, &k.narrow(1, offset, count)?.contiguous()?, 1)?;
            page.value
                .scatter_set(&indices, &v.narrow(1, offset, count)?.contiguous()?, 1)?;
            self.len += count;
            offset += count;
        }
        Ok(self.views())
    }
    pub fn views(&self) -> Vec<(Tensor, Tensor)> {
        self.pages
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let used = (self.len - i * self.block_size).min(self.block_size);
                (
                    p.key.narrow(1, 0, used).expect("page bounds"),
                    p.value.narrow(1, 0, used).expect("page bounds"),
                )
            })
            .collect()
    }
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            candle_core::bail!("cannot extend paged KV on rollback");
        }
        self.pages.truncate(len.div_ceil(self.block_size));
        self.len = len;
        Ok(())
    }
    pub fn share_prefix_from(&mut self, source: &Self, len: usize) -> Result<()> {
        if self.block_size != source.block_size
            || len % self.block_size != 0
            || len > self.len
            || len > source.len
        {
            candle_core::bail!("invalid canonical prefix");
        }
        for i in 0..len / self.block_size {
            self.pages[i] = Arc::clone(&source.pages[i]);
        }
        Ok(())
    }
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }
    #[cfg(test)]
    fn shares_first_page(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.pages[0], &other.pages[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn prefix_pages_are_shared_and_tail_writes_are_isolated() {
        let t = |n, v| Tensor::from_vec(vec![v; n], (1, n, 1), &Device::Cpu).unwrap();
        let mut a = PagedLayer::new(2, 8);
        a.write(&t(3, 1f32), &t(3, 2f32)).unwrap();
        let mut b = a.clone();
        assert!(a.shares_first_page(&b));
        b.write(&t(1, 9f32), &t(1, 8f32)).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 4);
        assert_eq!(a.views()[1].0.elem_count(), 1);
        b.truncate(2).unwrap();
        b.write(&t(2, 7f32), &t(2, 7f32)).unwrap();
        assert_eq!(
            a.views()[1]
                .0
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.]
        );
        assert!(a.shares_first_page(&b));
    }
    #[test]
    fn exclusive_tail_reuses_storage_but_shared_tail_copies() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut a = PagedLayer::new(4, 8);
        a.track_allocations(counter.clone());
        let t = Tensor::from_vec(vec![1f32], (1, 1, 1), &Device::Cpu).unwrap();
        a.write(&t, &t).unwrap();
        a.write(&t, &t).unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        let mut b = a.clone();
        b.truncate(1).unwrap();
        let nine = Tensor::from_vec(vec![9f32], (1, 1, 1), &Device::Cpu).unwrap();
        b.write(&nine, &nine).unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 4);
        assert_eq!(
            a.views()[0]
                .0
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1., 1.]
        );
        assert_eq!(
            b.views()[0]
                .0
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1., 9.]
        );
        a.truncate(1).unwrap();
        a.write(&nine, &nine).unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }
}
