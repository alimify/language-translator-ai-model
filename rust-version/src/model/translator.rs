//! Main Translator model - Encoder-Decoder Transformer.

use tch::{ nn, nn::Module, Kind, Tensor };
use crate::config::Config;
use super::rope::precompute_rope_freqs;
use super::normalization::RMSNorm;
use super::block::Block;

/// Encoder-Decoder Transformer for sequence-to-sequence translation.
///
/// Architecture:
/// - Separate source and target embeddings
/// - Embedding scaling by sqrt(dim)
/// - Weight tying between target embedding and output projection
/// - RoPE positional encoding
/// - Xavier initialization for multi-dimensional parameters
#[derive(Debug)]
pub struct Translator {
    se: nn::Embedding,
    te: nn::Embedding,
    enc: Vec<Block>,
    dec: Vec<Block>,
    norm: RMSNorm,
    freqs: Tensor,
    dim: i64,
}

impl Translator {
    /// Create a new Translator model.
    pub fn new(vs: &nn::Path, cfg: &Config, sv: i64, tv: i64) -> Self {
        let se = nn::embedding(vs / "se", sv, cfg.dim, Default::default());
        let te = nn::embedding(vs / "te", tv, cfg.dim, Default::default());

        // Encoder stack
        let mut enc = Vec::new();
        for i in 0..cfg.layers {
            enc.push(Block::new(&(vs / "enc" / i), cfg, false));
        }

        // Decoder stack
        let mut dec = Vec::new();
        for i in 0..cfg.layers {
            dec.push(Block::new(&(vs / "dec" / i), cfg, true));
        }

        let norm = RMSNorm::new(&(vs / "norm"), cfg.dim);

        // Precompute RoPE frequencies
        let freqs = precompute_rope_freqs(cfg.dim / cfg.heads, cfg.max_len);

        // Note: Weight tying in Rust tch is complex; we use te.ws directly in
        // forward. Xavier initialization of multi-dimensional parameters is
        // applied by the caller over the VarStore (see xavier_init_varstore),
        // matching Python's post-construction init loop.

        Self { se, te, enc, dec, norm, freqs, dim: cfg.dim }
    }

    /// Forward pass through encoder-decoder.
    pub fn forward(&self, src: &Tensor, trg: &Tensor, train: bool) -> Tensor {
        let device = trg.device();
        let sz = trg.size()[1];

        // Source mask: (B, 1, 1, T_src)
        let sm = src.ne(0).unsqueeze(1).unsqueeze(2);

        // Target mask: causal mask (1, 1, T_trg, T_trg)
        let tm = Tensor::tril(&Tensor::ones(&[sz, sz], (Kind::Bool, device)), 0).view([
            1,
            1,
            sz,
            sz,
        ]);

        // Move frequencies to device
        let freqs = self.freqs.to_device(device);

        // Encode source with embedding scaling
        let mut e = self.se.forward(src) * (self.dim as f64).sqrt();
        for layer in &self.enc {
            e = layer.forward(&e, None, Some(&sm), None, Some(&freqs), train);
        }

        // Decode target with embedding scaling
        let mut d = self.te.forward(trg) * (self.dim as f64).sqrt();
        for layer in &self.dec {
            d = layer.forward(&d, Some(&e), Some(&tm), Some(&sm), Some(&freqs), train);
        }

        // Output projection using weight tying (te.ws as weight)
        let norm_out = self.norm.forward(&d);
        norm_out.matmul(&self.te.ws.transpose(0, 1))
    }

    /// Xavier-uniform initialization for all multi-dimensional parameters in
    /// the VarStore (matches Python's post-construction init loop:
    /// `for p in self.parameters(): if p.dim() > 1: xavier_uniform_(p)`).
    pub fn xavier_init_varstore(vs: &nn::VarStore) {
        for (_name, var) in vs.variables() {
            let dims = var.size();
            if dims.len() > 1 {
                tch::no_grad(|| {
                    let fan_in = dims[dims.len() - 1] as f64;
                    let fan_out = dims[dims.len() - 2] as f64;
                    let limit = (6.0f64 / (fan_in + fan_out)).sqrt();
                    let mut uniform = Tensor::zeros(&dims, (Kind::Float, var.device()));
                    let _ = uniform.uniform_(-limit, limit);
                    let mut v = var;
                    let _ = v.copy_(&uniform);
                });
            }
        }
    }

    /// Print E2 shape checkpoints (equivalence-protocol validation mode).
    pub fn validate_shapes(&self, src: &Tensor, trg: &Tensor) {
        let fmt = |t: &Tensor| -> String {
            let strs: Vec<String> = t
                .size()
                .iter()
                .map(|s| s.to_string())
                .collect();
            format!("[{}]", strs.join(", "))
        };
        println!("E2 enc_input_ids shape={}", fmt(src));
        let sm = src.ne(0).unsqueeze(1).unsqueeze(2);
        let freqs = self.freqs.to_device(src.device());
        let emb = self.se.forward(src) * (self.dim as f64).sqrt();
        println!("E2 enc_embed shape={}", fmt(&emb));
        let mut e = emb;
        for layer in &self.enc {
            e = layer.forward(&e, None, Some(&sm), None, Some(&freqs), false);
        }
        println!("E2 enc_out shape={}", fmt(&e));
        println!("E2 dec_input_ids shape={}", fmt(trg));
        let logits = self.forward(src, trg, false);
        println!("E2 dec_logits shape={}", fmt(&logits));
    }
}
