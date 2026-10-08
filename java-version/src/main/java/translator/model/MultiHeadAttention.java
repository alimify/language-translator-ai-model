package translator.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;
import translator.Config;

/**
 * Multi-Head Attention with RoPE support.
 * Matches Python: model/attention.py
 * 
 * Members: h, dh, wq, wk, wv, o, dropout
 */
public class MultiHeadAttention extends Module {

    private final int h;
    private final int dh;
    private final LinearImpl wq, wk, wv, o;
    private final float dropout;

    public MultiHeadAttention(Config cfg) {
        this.h = cfg.heads;
        this.dh = cfg.dim / cfg.heads;
        this.dropout = cfg.dropout;

        LinearOptions opts = new LinearOptions(cfg.dim, cfg.dim);
        opts.bias().put(false);
        this.wq = register_module("wq", new LinearImpl(opts));
        this.wk = register_module("wk", new LinearImpl(opts));
        this.wv = register_module("wv", new LinearImpl(opts));
        this.o = register_module("o", new LinearImpl(opts));
    }

    /**
     * Forward pass.
     * Matches Python: def forward(self, x, kv=None, mask=None, freqs=None)
     */
    /** Projection weights in reference order, for the E2b loader. */
    public Tensor[] projectionWeights() {
        return new Tensor[] { wq.weight(), wk.weight(), wv.weight(), o.weight() };
    }

    public Tensor forward(Tensor x, Tensor kv, Tensor mask, Tensor freqs) {
        long B = x.size(0), T = x.size(1), D = x.size(2);

        if (kv == null)
            kv = x;
        long Tk = kv.size(1);

        // Q, K, V projections
        Tensor q = wq.forward(x).reshape(B, T, h, dh).permute(0, 2, 1, 3);
        Tensor k = wk.forward(kv).reshape(B, Tk, h, dh).permute(0, 2, 1, 3);
        Tensor v = wv.forward(kv).reshape(B, Tk, h, dh).permute(0, 2, 1, 3);

        // Apply RoPE
        if (freqs != null) {
            q = RoPE.apply_rope(q, freqs);
            // Self-attention check: apply RoPE to K only for self-attention
            if (kv.data_ptr().address() == x.data_ptr().address()) {
                k = RoPE.apply_rope(k, freqs);
            }
        }

        // Scaled dot-product attention
        Tensor att = q.matmul(k.transpose(2, 3)).mul(new Scalar(Math.pow(dh, -0.5)));

        // Apply mask
        if (mask != null) {
            att = att.masked_fill(mask.eq(new Scalar(0)), new Scalar(-1e9));
        }

        // Softmax and dropout
        att = softmax(att, -1);
        if (is_training()) {
            att = dropout(att, dropout, true);
        }

        // Combine heads
        Tensor out = att.matmul(v).permute(0, 2, 1, 3).reshape(B, T, D);
        return o.forward(out);
    }
}
