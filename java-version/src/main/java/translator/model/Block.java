package translator.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;
import translator.Config;

/**
 * Transformer Block (encoder or decoder).
 * Matches Python: model/block.py
 * 
 * Members: sa, ca (decoder only), n1, n2, n3, ff (Sequential), is_dec
 */
public class Block extends Module {

    private final boolean is_dec;
    private final float dropout;

    private final MultiHeadAttention sa;
    private MultiHeadAttention ca;
    private final RMSNorm n1, n2;
    private RMSNorm n3;
    private final LinearImpl ff1, ff2;

    public Block(Config cfg, boolean is_dec) {
        this.is_dec = is_dec;
        this.dropout = cfg.dropout;

        // Self-attention
        this.sa = new MultiHeadAttention(cfg);
        register_module("sa", sa);

        // Norms
        this.n1 = new RMSNorm(cfg.dim);
        register_module("n1", n1);
        this.n2 = new RMSNorm(cfg.dim);
        register_module("n2", n2);

        // Cross-attention (decoder only)
        if (is_dec) {
            this.ca = new MultiHeadAttention(cfg);
            register_module("ca", ca);
            this.n3 = new RMSNorm(cfg.dim);
            register_module("n3", n3);
        }

        // Feed-forward: nn.Sequential(Linear(dim, dim*ff_mult), SiLU(), Dropout,
        // Linear(dim*ff_mult, dim), Dropout)
        this.ff1 = register_module("ff1", new LinearImpl(cfg.dim, cfg.dim * cfg.ff_mult));
        this.ff2 = register_module("ff2", new LinearImpl(cfg.dim * cfg.ff_mult, cfg.dim));
    }

    /**
     * Forward pass.
     * Matches Python: def forward(self, x, e=None, sm=None, cm=None, freqs=None)
     */
    /**
     * Tensors keyed by the *reference implementation's* parameter names, for
     * the E2b neutral-weight loader. This class registers its feed-forward
     * layers as ff1/ff2 because JavaCPP has no Sequential equivalent that
     * reproduces the reference's ff.0 / ff.3 indices; the mapping is made
     * explicit here rather than left to positional order.
     */
    public java.util.Map<String, Tensor> namedTensors(String prefix) {
        java.util.Map<String, Tensor> m = new java.util.LinkedHashMap<>();
        Tensor[] sap = sa.projectionWeights();
        String[] pn = { "wq", "wk", "wv", "o" };
        for (int i = 0; i < 4; i++) m.put(prefix + ".sa." + pn[i] + ".weight", sap[i]);
        if (is_dec) {
            Tensor[] cap = ca.projectionWeights();
            for (int i = 0; i < 4; i++) m.put(prefix + ".ca." + pn[i] + ".weight", cap[i]);
        }
        m.put(prefix + ".ff.0.weight", ff1.weight());
        m.put(prefix + ".ff.0.bias", ff1.bias());
        m.put(prefix + ".ff.3.weight", ff2.weight());
        m.put(prefix + ".ff.3.bias", ff2.bias());
        m.put(prefix + ".n1.w", n1.weight());
        m.put(prefix + ".n2.w", n2.weight());
        if (is_dec) m.put(prefix + ".n3.w", n3.weight());
        return m;
    }

    public Tensor forward(Tensor x, Tensor e, Tensor sm, Tensor cm, Tensor freqs) {
        // Self-attention with residual
        x = x.add(sa.forward(n1.forward(x), null, sm, freqs));

        // Cross-attention with residual (decoder only)
        if (is_dec && e != null) {
            x = x.add(ca.forward(n2.forward(x), e, cm, null));
        }

        // Feed-forward with residual
        Tensor norm_out = is_dec ? n3.forward(x) : n2.forward(x);
        Tensor ff_out = silu(ff1.forward(norm_out));
        if (is_training()) {
            ff_out = dropout(ff_out, dropout, true);
        }
        ff_out = ff2.forward(ff_out);
        if (is_training()) {
            ff_out = dropout(ff_out, dropout, true);
        }

        return x.add(ff_out);
    }
}
