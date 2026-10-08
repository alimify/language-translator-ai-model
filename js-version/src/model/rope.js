/**Rotary Position Embedding (RoPE) implementation.*/
import { tf } from "../tf.js";

/**Precompute RoPE frequency tensors using sin/cos form.
 * 
 * Formula: freq = 1.0 / (10000 ^ (2i / dim)) for i in [0, dim/2)
 * 
 * Args:
 *     dim: Head dimension (model_dim // num_heads)
 *     max_len: Maximum sequence length
 *     
 * Returns:
 *     Object {cos, sin} each of shape (max_len, dim/2)
 */
function precompute_rope_freqs(dim, max_len) {
    // inv_freq = 1.0 / (10000 ^ (arange(0, dim, 2) / dim))
    const indices = tf.range(0, dim, 2).toFloat();
    const inv_freq = tf.div(1.0, tf.pow(10000, tf.div(indices, dim)));

    // freqs = outer(arange(max_len), inv_freq)
    const positions = tf.range(0, max_len).toFloat().reshape([-1, 1]);
    const freqs = tf.matMul(positions, inv_freq.reshape([1, -1]));

    // Return cos and sin (equivalent to torch.polar(ones, freqs))
    return {
        cos: tf.cos(freqs),
        sin: tf.sin(freqs),
    };
}

/**Apply rotary position embedding to input tensor.
 * 
 * Args:
 *     x: Input tensor of shape (B, H, T, D)
 *     freqs: Object {cos, sin} each of shape (max_len, dim/2)
 *     
 * Returns:
 *     Tensor with RoPE applied, same shape as input
 */
function apply_rope(x, freqs) {
    const [B, H, T, D] = x.shape;
    const half = D / 2;

    // Reshape to consecutive pairs: (B, H, T, D/2, 2) — matches Python's view_as_complex
    const x_r = x.reshape([B, H, T, half, 2]);
    const x0 = x_r.slice([0, 0, 0, 0, 0], [B, H, T, half, 1]).reshape([B, H, T, half]);
    const x1 = x_r.slice([0, 0, 0, 0, 1], [B, H, T, half, 1]).reshape([B, H, T, half]);

    // Get cos and sin for sequence length T: (T, D/2) -> (1, 1, T, D/2)
    const cos_t = freqs.cos.slice([0, 0], [T, half]).reshape([1, 1, T, half]);
    const sin_t = freqs.sin.slice([0, 0], [T, half]).reshape([1, 1, T, half]);

    // Apply rotation: complex multiplication via sin/cos
    // real_part = x0 * cos - x1 * sin
    // imag_part = x0 * sin + x1 * cos
    const out0 = tf.sub(tf.mul(x0, cos_t), tf.mul(x1, sin_t));
    const out1 = tf.add(tf.mul(x0, sin_t), tf.mul(x1, cos_t));

    // Stack back to (B, H, T, D/2, 2) then reshape to (B, H, T, D)
    return tf.stack([out0, out1], -1).reshape([B, H, T, D]);
}

export { precompute_rope_freqs, apply_rope };
