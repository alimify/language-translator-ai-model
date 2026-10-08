/**Transformer Block with optional cross-attention for encoder/decoder.*/
import { tf } from "../tf.js";
import { RMSNorm } from "./normalization.js";
import { MultiHeadAttention } from "./attention.js";

class Block {
    /**Transformer Block supporting both encoder and decoder configurations.
     * 
     * Matches Python: class Block(nn.Module)
     * 
     * Architecture:
     *     - Pre-norm with RMSNorm
     *     - Self-attention (always)
     *     - Cross-attention (decoder only)
     *     - Feed-forward network with SiLU activation
     * 
     * Members: sa, ca, n1, n2, n3, ff_w1, ff_w2, is_dec, dropout
     */

    /**Initialize Transformer Block.
     * 
     * Matches Python: def __init__(self, cfg, is_dec=False)
     * 
     * Args:
     *     cfg: Config object with dim, heads, ff_mult, dropout
     *     is_dec: If true, include cross-attention for decoder
     */
    constructor(cfg, is_dec = false) {
        this.is_dec = is_dec;
        this.dropout = cfg.dropout;

        // Self-attention
        this.sa = new MultiHeadAttention(cfg);

        // Cross-attention (decoder only)
        this.ca = is_dec
            ? new MultiHeadAttention(cfg)
            : null;

        // Feed-forward network weights (Linear -> SiLU -> Dropout -> Linear -> Dropout)
        // Python nn.Linear defaults to bias=True for the FFN layers, so the
        // biases exist here too. Weights are re-initialized with Xavier by the
        // Translator; biases keep PyTorch's Linear default U(-1/sqrt(fan_in), +).
        this.ff_w1 = tf.variable(
            tf.randomNormal([cfg.dim, cfg.dim * cfg.ff_mult]),
            true
        );
        const b1_bound = 1.0 / Math.sqrt(cfg.dim);
        this.ff_b1 = tf.variable(
            tf.randomUniform([cfg.dim * cfg.ff_mult], -b1_bound, b1_bound),
            true
        );
        this.ff_w2 = tf.variable(
            tf.randomNormal([cfg.dim * cfg.ff_mult, cfg.dim]),
            true
        );
        const b2_bound = 1.0 / Math.sqrt(cfg.dim * cfg.ff_mult);
        this.ff_b2 = tf.variable(
            tf.randomUniform([cfg.dim], -b2_bound, b2_bound),
            true
        );

        // Normalization layers
        this.n1 = new RMSNorm(cfg.dim);
        this.n2 = new RMSNorm(cfg.dim);
        if (is_dec) {
            this.n3 = new RMSNorm(cfg.dim);
        }
    }

    /**SiLU activation: x * sigmoid(x)*/
    silu(x) {
        return tf.mul(x, tf.sigmoid(x));
    }

    linear3d(x, w, b = null) {
        const [B, T, D] = x.shape;
        let out = tf.matMul(x.reshape([B * T, D]), w);
        if (b !== null) {
            out = tf.add(out, b);
        }
        return out.reshape([B, T, w.shape[1]]);
    }

    /**Feed-forward network: Linear -> SiLU -> Dropout -> Linear -> Dropout
     *
     * Matches Python: self.ff = nn.Sequential(Linear, SiLU, Dropout, Linear, Dropout)
     * (nn.Linear bias=True by default, so both layers carry a bias term)
     */
    ff(x, training = false) {
        let out = this.linear3d(x, this.ff_w1, this.ff_b1);
        out = this.silu(out);
        if (training && this.dropout > 0) {
            out = tf.dropout(out, this.dropout);
        }
        out = this.linear3d(out, this.ff_w2, this.ff_b2);
        if (training && this.dropout > 0) {
            out = tf.dropout(out, this.dropout);
        }
        return out;
    }

    /**Forward pass through the block.
     * 
     * Matches Python: def forward(self, x, e=None, sm=None, cm=None, freqs=None)
     * 
     * Args:
     *     x: Input tensor (B, T, D)
     *     e: Encoder output for cross-attention (decoder only)
     *     sm: Self-attention mask
     *     cm: Cross-attention mask
     *     freqs: RoPE frequencies
     *     training: Whether in training mode
     */
    forward(x, e = null, sm = null, cm = null, freqs = null, training = false) {
        // Self-attention with residual
        x = tf.add(x, this.sa.forward(this.n1.forward(x), null, sm, freqs, training));

        // Cross-attention with residual (decoder only)
        if (this.is_dec) {
            x = tf.add(
                x,
                this.ca.forward(this.n2.forward(x), e, cm, null, training)
            );
        }

        // Feed-forward with residual (n3 for decoder, n2 for encoder)
        const norm = this.is_dec ? this.n3 : this.n2;
        return tf.add(x, this.ff(norm.forward(x), training));
    }

    getWeights() {
        const weights = [
            ...this.sa.getWeights(),
            this.ff_w1,
            this.ff_b1,
            this.ff_w2,
            this.ff_b2,
            ...this.n1.getWeights(),
            ...this.n2.getWeights(),
        ];
        if (this.is_dec) {
            weights.push(...this.ca.getWeights());
            weights.push(...this.n3.getWeights());
        }
        return weights;
    }
}

export { Block };
