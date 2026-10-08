/**Main Translator model - Encoder-Decoder Transformer.*/
import { tf } from "../tf.js";
import { precompute_rope_freqs } from "./rope.js";
import { RMSNorm } from "./normalization.js";
import { Block } from "./block.js";

class Translator {
    /**Encoder-Decoder Transformer for sequence-to-sequence translation.
     * 
     * Architecture:
     *     - Separate source and target embeddings
     *     - Embedding scaling by sqrt(dim)
     *     - Weight tying between target embedding and output projection
     *     - RoPE positional encoding
     *     - Xavier initialization for multi-dimensional parameters
     */

    /**Initialize Translator model.
     * 
     * Args:
     *     cfg: Config object with model hyperparameters
     *     sv: Source vocabulary size
     *     tv: Target vocabulary size
     */
    constructor(cfg, sv, tv) {
        this.dim = cfg.dim;

        // Embeddings
        this.se = tf.variable(tf.randomNormal([sv, cfg.dim])); // Source embedding
        this.te = tf.variable(tf.randomNormal([tv, cfg.dim])); // Target embedding

        // Encoder and Decoder stacks
        this.enc = [];
        for (let i = 0; i < cfg.layers; i++) {
            this.enc.push(new Block(cfg));
        }
        this.dec = [];
        for (let i = 0; i < cfg.layers; i++) {
            this.dec.push(new Block(cfg, true));
        }

        // Output layers
        this.norm = new RMSNorm(cfg.dim);
        // head shares weights with te (weight tying) — no separate head variable needed

        // Precompute RoPE frequencies
        this.freqs = precompute_rope_freqs(
            Math.floor(cfg.dim / cfg.heads),
            cfg.max_len
        );

        // Xavier initialization for multi-dimensional parameters
        this._xavierInit();
    }

    /**Xavier uniform initialization for all multi-dimensional parameters.*/
    _xavierInit() {
        const allWeights = this.getWeights();
        for (const w of allWeights) {
            if (w.shape.length > 1) {
                const [fan_in, fan_out] = w.shape;
                const limit = Math.sqrt(6.0 / (fan_in + fan_out));
                w.assign(tf.randomUniform(w.shape, -limit, limit));
            }
        }
    }

    /**Forward pass through encoder-decoder.
     * 
     * Args:
     *     src: Source token IDs (B, T_src) - integer tensor
     *     trg: Target token IDs (B, T_trg) - integer tensor
     *     training: Whether in training mode
     *     
     * Returns:
     *     Logits tensor (B, T_trg, vocab_size)
     */
    forward(src, trg, training = false) {
        // Create masks
        // Source mask: (B, 1, 1, T_src) - mask padding tokens
        const sm = tf.cast(tf.notEqual(src, 0), "float32")
            .expandDims(1)
            .expandDims(2);

        // Target mask: causal mask (1, 1, T_trg, T_trg)
        const sz = trg.shape[1];
        const ones = tf.ones([sz, sz]);
        const tm = tf.linalg.bandPart(ones, -1, 0) // tril
            .reshape([1, 1, sz, sz]);

        // Encode source with embedding scaling
        const scale = Math.sqrt(this.dim);
        let e = tf.mul(tf.gather(this.se, src), scale);
        for (const layer of this.enc) {
            e = layer.forward(e, null, sm, null, this.freqs, training);
        }

        // Decode target with embedding scaling
        let d = tf.mul(tf.gather(this.te, trg), scale);
        for (const layer of this.dec) {
            d = layer.forward(d, e, tm, sm, this.freqs, training);
        }

        // Output projection (weight tying: head.weight = te.weight)
        // logits = d @ te.T
        const normed = this.norm.forward(d);
        const [B, T, D] = normed.shape;
        const logits2d = tf.matMul(normed.reshape([B * T, D]), this.te, false, true);
        return logits2d.reshape([B, T, this.te.shape[0]]);
    }

    /**Get all trainable weights.*/
    getWeights() {
        const weights = [this.se, this.te, ...this.norm.getWeights()];
        for (const layer of this.enc) {
            weights.push(...layer.getWeights());
        }
        for (const layer of this.dec) {
            weights.push(...layer.getWeights());
        }
        return weights;
    }

    /**Save model weights to a JSON-compatible object.*/
    saveWeights() {
        const data = {};
        const weights = this.getWeights();
        for (let i = 0; i < weights.length; i++) {
            data[`w_${i}`] = Array.from(weights[i].dataSync());
            data[`s_${i}`] = weights[i].shape;
        }
        return data;
    }

    /**Load model weights from a JSON-compatible object.*/
    loadWeights(data) {
        const weights = this.getWeights();
        for (let i = 0; i < weights.length; i++) {
            const values = data[`w_${i}`];
            const shape = data[`s_${i}`];
            weights[i].assign(tf.tensor(values, shape));
        }
    }
}

export { Translator };
