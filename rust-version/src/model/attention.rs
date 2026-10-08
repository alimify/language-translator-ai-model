//! Multi-Head Attention with RoPE support.

use tch::{ nn, Kind, Tensor };
use super::rope::apply_rope;

/// Multi-Head Attention with Rotary Position Embedding (RoPE) support.
///
/// Features:
/// - Q, K, V projections without bias
/// - RoPE applied to Q always, K only for self-attention
/// - Scaled dot-product attention with masking
#[derive(Debug)]
pub struct MultiHeadAttention {
    h: i64, // Number of heads
    dh: i64, // Head dimension
    wq: nn::Linear,
    wk: nn::Linear,
    wv: nn::Linear,
    o: nn::Linear,
    dropout: f64,
}

impl MultiHeadAttention {
    /// Create a new Multi-Head Attention layer.
    pub fn new(vs: &nn::Path, dim: i64, heads: i64, dropout: f64) -> Self {
        let config = nn::LinearConfig { bias: false, ..Default::default() };
        Self {
            h: heads,
            dh: dim / heads,
            wq: nn::linear(vs / "wq", dim, dim, config),
            wk: nn::linear(vs / "wk", dim, dim, config),
            wv: nn::linear(vs / "wv", dim, dim, config),
            o: nn::linear(vs / "o", dim, dim, config),
            dropout,
        }
    }

    /// Compute multi-head attention.
    ///
    /// # Arguments
    /// * `x` - Query input (B, T, D)
    /// * `kv` - Key-Value input for cross-attention (None = self-attention)
    /// * `mask` - Attention mask
    /// * `freqs` - RoPE frequency tensor
    /// * `train` - Training mode flag
    pub fn forward(
        &self,
        x: &Tensor,
        kv: Option<&Tensor>,
        mask: Option<&Tensor>,
        freqs: Option<&Tensor>,
        train: bool
    ) -> Tensor {
        let (b, t, _d) = x.size3().unwrap();
        let kv_tensor = kv.unwrap_or(x);
        let tk = kv_tensor.size()[1];

        // Compute Q, K, V projections
        let q = x.apply(&self.wq).reshape(&[b, t, self.h, self.dh]).transpose(1, 2);
        let k = kv_tensor.apply(&self.wk).reshape(&[b, tk, self.h, self.dh]).transpose(1, 2);
        let v = kv_tensor.apply(&self.wv).reshape(&[b, tk, self.h, self.dh]).transpose(1, 2);

        // Apply RoPE: always to Q, only to K for self-attention
        let (q_rot, k_rot) = if let Some(f) = freqs {
            let q_r = apply_rope(&q, f);
            // Python: k = apply_rope(k, freqs) if kv is x else k
            let k_r = if kv.is_none() { apply_rope(&k, f) } else { k };
            (q_r, k_r)
        } else {
            (q, k)
        };

        // Scaled dot-product attention
        let mut att = q_rot.matmul(&k_rot.transpose(-2, -1)) * (self.dh as f64).powf(-0.5);

        // Apply mask
        if let Some(m) = mask {
            att = att.masked_fill(&m.eq(0), -1e9);
        }

        // Softmax and dropout
        att = att.softmax(-1, Kind::Float).dropout(self.dropout, train);

        // Combine heads
        let d = self.h * self.dh;
        let out = att.matmul(&v).transpose(1, 2).reshape(&[b, t, d]);
        out.apply(&self.o)
    }
}
