//! SwiGLU feed-forward block: `down( silu(gate(x)) ⊙ up(x) )`.
//!
//! SiLU is computed as `x * sigmoid(x)` with a numerically stable sigmoid
//! (`0.5 * tanh(x/2) + 0.5`) so the crate does not depend on candle-nn.

use candle_core::{Result, Tensor};

use crate::linear::Linear;

#[derive(Debug, Clone)]
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = silu(&gate)?;
        self.down_proj.forward(&activated.broadcast_mul(&up)?)
    }
}

/// Stable SiLU: `x * sigmoid(x)`, `sigmoid(x) = 0.5 * tanh(x/2) + 0.5`.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = x.affine(0.5, 0.0)?.tanh()?.affine(0.5, 0.5)?;
    x.broadcast_mul(&sigmoid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn linear(rows: usize, cols: usize, fill: f32) -> Linear {
        Linear::new(
            Tensor::from_vec(vec![fill; rows * cols], (rows, cols), &Device::Cpu).unwrap(),
            None,
        )
    }

    #[test]
    fn silu_matches_definition() {
        for &v in &[-4.0f32, -0.5, 0.0, 0.5, 4.0] {
            let x = Tensor::from_vec(vec![v], (1,), &Device::Cpu).unwrap();
            let out = silu(&x).unwrap().to_vec1::<f32>().unwrap()[0];
            let expected = v / (1.0 + (-v).exp());
            assert!((out - expected).abs() < 1e-5, "silu({v}) = {out}");
        }
    }

    #[test]
    fn silu_is_stable_for_large_inputs() {
        let x = Tensor::from_vec(vec![-100.0f32, 100.0], (2,), &Device::Cpu).unwrap();
        let out = silu(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(out[0].abs() < 1e-5); // silu(-100) ≈ 0, never NaN
        assert!((out[1] - 100.0).abs() < 1e-2);
        assert!(out[0].is_finite() && out[1].is_finite());
    }

    #[test]
    fn mlp_output_shape() {
        let mlp = Mlp::new(linear(8, 4, 0.5), linear(8, 4, 0.25), linear(4, 8, 0.1));
        let x = Tensor::from_vec(vec![0.1f32; 2 * 4], (2, 4), &Device::Cpu).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 4]);
    }
}
