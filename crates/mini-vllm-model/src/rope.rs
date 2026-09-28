//! Rotary positional embeddings (RoPE), HF `rotate_half` convention.
//!
//! `q_rot = q * cos + rotate_half(q) * sin` where `rotate_half(x) =
//! cat(-x2, x1)` and cos/sin tables are precomputed per position.
//! Kept as a standalone module (never buried inside a forward method) with
//! shape, determinism and rotation-invariance tests.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug)]
pub struct RopeCache {
    /// `[max_pos, head_dim]`
    cos: Tensor,
    /// `[max_pos, head_dim]`
    sin: Tensor,
    head_dim: usize,
}

impl RopeCache {
    pub fn new(
        max_pos: usize,
        head_dim: usize,
        theta: f64,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        assert!(head_dim.is_multiple_of(2), "head_dim must be even for RoPE");
        let half = head_dim / 2;
        // inv_freq[j] = theta^(-2j / head_dim)
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| theta.powf(-2.0 * j as f64 / head_dim as f64) as f32)
            .collect();
        let inv = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let pos: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (max_pos, 1), dev)?;
        let angles = pos.matmul(&inv)?; // [max_pos, half]
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        // Duplicate halves: [c0..c_{half-1}, c0..c_{half-1}] per position.
        let cos = Tensor::cat(&[&cos, &cos], 1)?.to_dtype(dtype)?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.to_dtype(dtype)?;
        Ok(Self { cos, sin, head_dim })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Apply rotary embeddings.
    ///
    /// * `q`: `[num_heads, q_len, head_dim]`
    /// * `k`: `[num_kv_heads, q_len, head_dim]`
    /// * `positions`: absolute position ids, length `q_len`
    pub fn apply(&self, q: &Tensor, k: &Tensor, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        let len = positions.len();
        let idx = Tensor::from_vec(positions.to_vec(), (len,), self.cos.device())?;
        let cos = self.cos.index_select(&idx, 0)?.unsqueeze(0)?; // [1, len, hd]
        let sin = self.sin.index_select(&idx, 0)?.unsqueeze(0)?;
        Ok((rope_rotate(q, &cos, &sin)?, rope_rotate(k, &cos, &sin)?))
    }
}

fn rope_rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(2)?;
    let half = d / 2;
    let x1 = x.narrow(2, 0, half)?;
    let x2 = x.narrow(2, half, d - half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 2)?;
    x.broadcast_mul(cos)?
        .broadcast_add(&rotated.broadcast_mul(sin)?)
}

/// Shared, cheap-to-clone handle used by every attention layer.
pub type SharedRope = Arc<RopeCache>;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn rope() -> RopeCache {
        RopeCache::new(16, 4, 10_000.0, DType::F32, &Device::Cpu).unwrap()
    }

    fn tensor(nh: usize, len: usize, hd: usize) -> Tensor {
        let v: Vec<f32> = (0..nh * len * hd)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.25)
            .collect();
        Tensor::from_vec(v, (nh, len, hd), &Device::Cpu).unwrap()
    }

    #[test]
    fn shapes_are_preserved() {
        let r = rope();
        let q = tensor(4, 3, 4);
        let k = tensor(2, 3, 4);
        let (q2, k2) = r.apply(&q, &k, &[0, 1, 2]).unwrap();
        assert_eq!(q2.dims(), q.dims());
        assert_eq!(k2.dims(), k.dims());
    }

    #[test]
    fn rotation_preserves_norms() {
        // RoPE is a rotation: per-position vector norm must not change.
        let r = rope();
        let q = tensor(2, 4, 4);
        let (q2, _) = r.apply(&q, &q, &[3, 7, 1, 0]).unwrap();
        let before = q
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let after = q2
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((before - after).abs() / before < 1e-5);
    }

    #[test]
    fn position_zero_is_identity() {
        // cos(0)=1, sin(0)=0 → output equals input.
        let r = rope();
        let q = tensor(2, 1, 4);
        let (q2, _) = r.apply(&q, &q, &[0]).unwrap();
        let a = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = q2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn deterministic_and_position_dependent() {
        let r = rope();
        let q = tensor(2, 2, 4);
        let (a, _) = r.apply(&q, &q, &[5, 5]).unwrap();
        let (b, _) = r.apply(&q, &q, &[5, 5]).unwrap();
        let (c, _) = r.apply(&q, &q, &[6, 6]).unwrap();
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cv = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av, bv);
        assert_ne!(av, cv);
    }
}
