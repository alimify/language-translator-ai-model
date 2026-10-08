package translator.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;
import translator.Config;

/**
 * Encoder-Decoder Transformer.
 * Matches Python: model/translator.py
 * 
 * Members: se, te, enc[], dec[], norm, head, dim, sv, tv
 */
public class Translator extends Module {

    private final int dim;
    private final EmbeddingImpl se, te;
    private final Block[] enc, dec;
    private final RMSNorm norm;
    // Output projection is computed functionally from te.weight() (true weight
    // tying: one shared parameter, exactly like Python's
    // `self.head.weight = self.te.weight` where parameters() dedups the tensor).
    private Tensor freqs;
    private final Config cfg;

    public Translator(Config cfg, int sv, int tv) {
        this.dim = cfg.dim;
        this.cfg = cfg;

        // Source and target embeddings
        this.se = register_module("se", new EmbeddingImpl(sv, cfg.dim));
        this.te = register_module("te", new EmbeddingImpl(tv, cfg.dim));

        // Encoder stack
        this.enc = new Block[cfg.layers];
        for (int i = 0; i < cfg.layers; i++) {
            enc[i] = new Block(cfg, false);
            register_module("enc" + i, enc[i]);
        }

        // Decoder stack
        this.dec = new Block[cfg.layers];
        for (int i = 0; i < cfg.layers; i++) {
            dec[i] = new Block(cfg, true);
            register_module("dec" + i, dec[i]);
        }

        // Output norm
        this.norm = new RMSNorm(cfg.dim);
        register_module("norm", norm);

        // Output projection reuses te.weight() directly in forward() (weight
        // tying with a single shared parameter; no separate head parameter).

        // Precompute RoPE frequencies
        this.freqs = RoPE.precompute_rope_freqs(cfg.dim / cfg.heads, cfg.max_len);

        // Xavier init for all linear layers
        _init_weights();
    }

    /**
     * Every trained tensor keyed by the reference implementation's parameter
     * name, for the E2b neutral-weight loader (thesis 8.8). The RoPE frequency
     * buffer is deliberately absent: it is recomputed in the constructor, so an
     * incorrect rotation cannot borrow the reference's table.
     */
    public java.util.Map<String, Tensor> namedTensors() {
        java.util.Map<String, Tensor> m = new java.util.LinkedHashMap<>();
        m.put("se.weight", se.weight());
        m.put("te.weight", te.weight());
        for (int i = 0; i < enc.length; i++) m.putAll(enc[i].namedTensors("enc." + i));
        for (int i = 0; i < dec.length; i++) m.putAll(dec[i].namedTensors("dec." + i));
        m.put("norm.w", norm.weight());
        return m;
    }

    private void _init_weights() {
        TensorVector params = parameters();
        for (long i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.dim() >= 2) {
                xavier_uniform_(p);
            }
        }
    }

    /**
     * Forward pass.
     * Matches Python: def forward(self, src, trg)
     */
    public Tensor forward(Tensor src, Tensor trg) {
        long sz = trg.size(1);

        // Source mask: (B, 1, 1, T_src)
        Tensor sm = src.ne(new Scalar(0)).unsqueeze(1).unsqueeze(2);

        // Target causal mask: (1, 1, sz, sz)
        Tensor tm = ones(sz, sz).tril(0).unsqueeze(0).unsqueeze(0);

        // Encode: embedding * sqrt(dim)
        Tensor e = se.forward(src).mul(new Scalar(Math.sqrt(dim)));
        for (Block layer : enc) {
            e = layer.forward(e, null, sm, null, freqs);
        }

        // Decode: embedding * sqrt(dim)
        Tensor d = te.forward(trg).mul(new Scalar(Math.sqrt(dim)));
        for (Block layer : dec) {
            d = layer.forward(d, e, tm, sm, freqs);
        }

        // Output: norm -> projection tied to te.weight (logits = norm(d) @ te.weight^T)
        return matmul(norm.forward(d), te.weight().t());
    }

    /**
     * Equivalence-protocol E2 checkpoints: mirrors forward() computing the
     * source embeddings and the encoder stack, printing the intermediate
     * shapes. Matches the Python `validate` branch in main.py.
     */
    public void printShapeCheckpoints(Tensor src, Tensor trg) {
        Tensor emb = se.forward(src).mul(new Scalar(Math.sqrt(dim)));
        System.out.println("E2 enc_embed shape=" + shapeList(emb));
        Tensor sm = src.ne(new Scalar(0)).unsqueeze(1).unsqueeze(2);
        Tensor e = emb;
        for (Block layer : enc) {
            e = layer.forward(e, null, sm, null, freqs);
        }
        System.out.println("E2 enc_out shape=" + shapeList(e));
    }

    /** Format a tensor shape like Python's list(t.shape), e.g. "[2, 9, 256]". */
    public static String shapeList(Tensor t) {
        StringBuilder sb = new StringBuilder("[");
        for (long i = 0; i < t.dim(); i++) {
            if (i > 0)
                sb.append(", ");
            sb.append(t.size(i));
        }
        return sb.append("]").toString();
    }
}
