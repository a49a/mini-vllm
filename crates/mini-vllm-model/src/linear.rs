//! Minimal linear layer: `y = x @ W^T + b` with `W: [out, in]` (HF layout).

use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}
