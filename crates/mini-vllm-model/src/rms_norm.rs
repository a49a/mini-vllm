//! RMSNorm: `x / sqrt(mean(x^2) + eps) * weight`, computed in F32 for
//! numerical stability.

use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        // [.., H] → [.., 1] so the broadcast aligns on the feature dim.
        let var = x
            .sqr()?
            .mean(candle_core::D::Minus1)?
            .unsqueeze(candle_core::D::Minus1)?;
        let normed = x.broadcast_mul(&var.affine(1.0, self.eps)?.sqrt()?.recip()?)?;
        // Reshape the weight to the input rank so Candle can broadcast.
        let rank = x.rank();
        let mut w_shape = vec![1usize; rank];
        w_shape[rank - 1] = self.weight.elem_count();
        let w = self.weight.to_dtype(DType::F32)?.reshape(w_shape)?;
        normed.broadcast_mul(&w)?.to_dtype(in_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn normalizes_known_values() {
        // x = [3, 4]: mean(x^2) = 12.5 → factor = 1/sqrt(12.5)
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &Device::Cpu).unwrap();
        let w = Tensor::from_vec(vec![1.0f32, 1.0], (2,), &Device::Cpu).unwrap();
        let norm = RmsNorm::new(w, 0.0);
        let out = norm
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let f = (12.5f32).sqrt().recip();
        assert!((out[0] - 3.0 * f).abs() < 1e-6);
        assert!((out[1] - 4.0 * f).abs() < 1e-6);
    }

    #[test]
    fn weight_scales_output() {
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &Device::Cpu).unwrap();
        let w = Tensor::from_vec(vec![2.0f32, 3.0], (2,), &Device::Cpu).unwrap();
        let norm = RmsNorm::new(w, 1e-6);
        let out = norm
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((out[0] - 2.0).abs() < 1e-3);
        assert!((out[1] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn preserves_input_dtype() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &Device::Cpu).unwrap();
        let w = Tensor::from_vec(vec![1.0f32, 1.0], (2,), &Device::Cpu).unwrap();
        let norm = RmsNorm::new(w, 1e-6);
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dtype(), DType::F32);
    }
}
