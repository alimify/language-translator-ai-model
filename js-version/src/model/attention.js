/**Multi-Head Attention with RoPE support.*/
import { tf } from "../tf.js";
import { apply_rope } from "./rope.js";

class MultiHeadAttention {
  /**Multi-Head Attention with Rotary Position Embedding (RoPE) support.
   *
   * Matches Python: class MultiHeadAttention(nn.Module)
   *
   * Members: h, dh, wq, wk, wv, o, dropout
   */

  /**Initialize Multi-Head Attention.
   *
   * Matches Python: def __init__(self, cfg)
   *
   * Args:
   *     cfg: Config object with dim, heads, dropout
   */
  constructor(cfg) {
    this.h = cfg.heads;
    this.dh = Math.floor(cfg.dim / cfg.heads);
    this.dropout = cfg.dropout;

    // Q, K, V projections without bias
    this.wq = tf.variable(tf.randomNormal([cfg.dim, cfg.dim]));
    this.wk = tf.variable(tf.randomNormal([cfg.dim, cfg.dim]));
    this.wv = tf.variable(tf.randomNormal([cfg.dim, cfg.dim]));

    // Output projection
    this.o = tf.variable(tf.randomNormal([cfg.dim, cfg.dim]));
  }

  linear3d(x, w) {
    const [B, T, D] = x.shape;
    return tf.matMul(x.reshape([B * T, D]), w).reshape([B, T, w.shape[1]]);
  }

  /**Compute multi-head attention.
   *
   * Matches Python: def forward(self, x, kv=None, mask=None, freqs=None)
   *
   * Args:
   *     x: Query input of shape (B, T, D)
   *     kv: Key-Value input for cross-attention (default: self-attention)
   *     mask: Attention mask (1 = attend, 0 = mask)
   *     freqs: RoPE frequency object {cos, sin}
   *     training: Whether in training mode (for dropout)
   */
  forward(x, kv = null, mask = null, freqs = null, training = false) {
    if (kv === null) kv = x;

    const [B, T, D] = x.shape;
    const Tk = kv.shape[1];

    // Compute Q, K, V: (B, T, D) @ (D, D) -> (B, T, D)
    let q = this.linear3d(x, this.wq);
    let k = this.linear3d(kv, this.wk);
    let v = this.linear3d(kv, this.wv);

    // Reshape to (B, H, T, D_h)
    q = q.reshape([B, T, this.h, this.dh]).transpose([0, 2, 1, 3]);
    k = k.reshape([B, Tk, this.h, this.dh]).transpose([0, 2, 1, 3]);
    v = v.reshape([B, Tk, this.h, this.dh]).transpose([0, 2, 1, 3]);

    // Apply RoPE: always to Q, only to K for self-attention
    // Self-attention check: x.data_ptr() == kv.data_ptr() (Python)
    // In JS, after kv = x above, kv === x for self-attention
    if (freqs !== null) {
      q = apply_rope(q, freqs);
      if (kv === x) {
        // Self-attention
        k = apply_rope(k, freqs);
      }
    }

    // Scaled dot-product attention
    const scale = Math.pow(this.dh, -0.5);
    let att = tf.mul(tf.matMul(q, k, false, true), scale);

    // Apply mask
    if (mask !== null) {
      // mask == 0 -> fill with -1e9
      const neg_inf = tf.mul(tf.sub(1, mask.toFloat()), -1e9);
      att = tf.add(att, neg_inf);
    }

    // Softmax and dropout
    att = tf.softmax(att, -1);
    if (training && this.dropout > 0) {
      att = tf.dropout(att, this.dropout);
    }

    // Combine heads: (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, D)
    let out = tf.matMul(att, v);
    out = out.transpose([0, 2, 1, 3]).reshape([B, T, D]);

    // Output projection
    return this.linear3d(out, this.o);
  }

  getWeights() {
    return [this.wq, this.wk, this.wv, this.o];
  }
}

export { MultiHeadAttention };
