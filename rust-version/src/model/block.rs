//! Transformer Block with optional cross-attention for encoder/decoder.

use tch::{ nn, Tensor };
use crate::config::Config;
use super::normalization::RMSNorm;
use super::attention::MultiHeadAttention;

/// Transformer Block supporting both encoder and decoder configurations.
///
/// Architecture:
/// - Pre-norm with RMSNorm
/// - Self-attention (always)
/// - Cross-attention (decoder only)
/// - Feed-forward network with SiLU activation
#[derive(Debug)]
pub struct Block {
    is_dec: bool,
    sa: MultiHeadAttention,
    ca: Option<MultiHeadAttention>,
    ff1: nn::Linear,
    ff2: nn::Linear,
    n1: RMSNorm,
    n2: RMSNorm,
    n3: Option<RMSNorm>,
    dropout: f64,
}

impl Block {
    /// Create a new Transformer Block.
    pub fn new(vs: &nn::Path, cfg: &Config, is_dec: bool) -> Self {
        let sa = MultiHeadAttention::new(&(vs / "sa"), cfg.dim, cfg.heads, cfg.dropout);
        let ca = if is_dec {
            Some(MultiHeadAttention::new(&(vs / "ca"), cfg.dim, cfg.heads, cfg.dropout))
        } else {
            None
        };

        // Feed-forward network: Linear -> SiLU -> Dropout -> Linear -> Dropout
        let ff1 = nn::linear(vs / "ff" / "0", cfg.dim, cfg.dim * cfg.ff_mult, Default::default());
        let ff2 = nn::linear(vs / "ff" / "3", cfg.dim * cfg.ff_mult, cfg.dim, Default::default());

        Self {
            is_dec,
            sa,
            ca,
            ff1,
            ff2,
            n1: RMSNorm::new(&(vs / "n1"), cfg.dim),
            n2: RMSNorm::new(&(vs / "n2"), cfg.dim),
            n3: if is_dec {
                Some(RMSNorm::new(&(vs / "n3"), cfg.dim))
            } else {
                None
            },
            dropout: cfg.dropout,
        }
    }

    /// Forward pass through the block.
    pub fn forward(
        &self,
        x: &Tensor,
        e: Option<&Tensor>,
        sm: Option<&Tensor>,
        cm: Option<&Tensor>,
        freqs: Option<&Tensor>,
        train: bool
    ) -> Tensor {
        // Self-attention with residual (no block-level dropout — dropout is inside MHA)
        let sa_out = self.sa.forward(&self.n1.forward(x), None, sm, freqs, train);
        let mut x_out = x + sa_out;

        // Cross-attention with residual (decoder only)
        if self.is_dec {
            let ca_out = self.ca
                .as_ref()
                .unwrap()
                .forward(&self.n2.forward(&x_out), e, cm, None, train);
            x_out = x_out + ca_out;
        }

        // Feed-forward with residual: Linear -> SiLU -> Dropout -> Linear -> Dropout
        let norm_input = if self.is_dec {
            self.n3.as_ref().unwrap().forward(&x_out)
        } else {
            self.n2.forward(&x_out)
        };
        let ff_out = norm_input
            .apply(&self.ff1)
            .silu()
            .dropout(self.dropout, train)
            .apply(&self.ff2)
            .dropout(self.dropout, train);
        x_out + ff_out
    }
}
