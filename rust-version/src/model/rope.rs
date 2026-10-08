//! Rotary Position Embedding (RoPE) implementation.

use tch::{ Device, Kind, Tensor };

/// Precompute RoPE frequency tensor.
///
/// Formula: freq = 1.0 / (10000 ^ (2i / dim)) for i in [0, dim/2)
///
/// # Arguments
/// * `dim` - Head dimension (model_dim / num_heads)
/// * `max_len` - Maximum sequence length
///
/// # Returns
/// Frequency tensor of shape (max_len, dim/2)
pub fn precompute_rope_freqs(dim: i64, max_len: i64) -> Tensor {
    // inv_freq = 1.0 / (10000 ^ (2i / dim))  — scalar base, tensor exponent
    let exponent = Tensor::arange_start_step(0, dim, 2, (Kind::Float, Device::Cpu)) / (dim as f64);
    let inv_freq = 1.0 / Tensor::pow_scalar(10000.0, &exponent);
    let t = Tensor::arange(max_len, (Kind::Float, Device::Cpu));
    Tensor::outer(&t, &inv_freq)
}

/// Apply rotary position embedding to input tensor.
///
/// # Arguments
/// * `x` - Input tensor of shape (B, H, T, D)
/// * `freqs` - Precomputed frequency tensor
///
/// # Returns
/// Tensor with RoPE applied
pub fn apply_rope(x: &Tensor, freqs: &Tensor) -> Tensor {
    let (b, h, t, d) = x.size4().unwrap();

    // Reshape to (B, H, T, D/2, 2) for rotation
    let x_reshaped = x.reshape(&[b, h, t, -1, 2]);
    let x0 = x_reshaped.select(-1, 0);
    let x1 = x_reshaped.select(-1, 1);

    // Get frequencies for current sequence length
    let freqs_t = freqs.slice(0, 0, t, 1).unsqueeze(0).unsqueeze(0); // [1, 1, T, D/2]

    let cos = freqs_t.cos();
    let sin = freqs_t.sin();

    // Apply rotation: [x*cos - y*sin, x*sin + y*cos]
    let out0 = &x0 * &cos - &x1 * &sin;
    let out1 = &x0 * &sin + &x1 * &cos;

    Tensor::stack(&[out0, out1], -1).reshape(&[b, h, t, d])
}
