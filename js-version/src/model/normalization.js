/**RMS Normalization layer.*/
import { tf } from "../tf.js";

class RMSNorm {
    /**Root Mean Square Layer Normalization.
     * 
     * More efficient than LayerNorm as it doesn't require mean computation.
     * Formula: x * rsqrt(mean(x^2) + eps) * weight
     */

    /**Initialize RMSNorm.
     * 
     * Args:
     *     d: Feature dimension
     *     eps: Small constant for numerical stability
     */
    constructor(d, eps = 1e-6) {
        this.eps = eps;
        this.w = tf.variable(tf.ones([d]));
    }

    /**Apply RMS normalization.
     * 
     * Args:
     *     x: Input tensor of shape (..., d)
     *     
     * Returns:
     *     Normalized tensor of same shape
     */
    forward(x) {
        // x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        const x_sq = tf.square(x);
        const mean_sq = tf.mean(x_sq, -1, true);
        const rsqrt = tf.rsqrt(tf.add(mean_sq, this.eps));
        return tf.mul(tf.mul(x, rsqrt), this.w);
    }

    getWeights() {
        return [this.w];
    }
}

export { RMSNorm };
