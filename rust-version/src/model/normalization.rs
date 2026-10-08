//! RMS Normalization layer.

use tch::{ nn, Kind, Tensor };

/// Root Mean Square Layer Normalization.
///
/// More efficient than LayerNorm as it doesn't require mean computation.
/// Formula: x * rsqrt(mean(x^2) + eps) * weight
#[derive(Debug)]
pub struct RMSNorm {
    w: Tensor,
    eps: f64,
}

impl RMSNorm {
    /// Create a new RMSNorm layer.
    ///
    /// # Arguments
    /// * `vs` - Variable store path
    /// * `dim` - Feature dimension
    pub fn new(vs: &nn::Path, dim: i64) -> Self {
        let w = vs.var("w", &[dim], nn::Init::Const(1.0));
        Self { w, eps: 1e-6 }
    }

    /// Apply RMS normalization.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let norm =
            x *
            (x.pow_tensor_scalar(2.0).mean_dim(&[-1i64][..], true, Kind::Float) + self.eps).rsqrt();
        norm * &self.w
    }
}
