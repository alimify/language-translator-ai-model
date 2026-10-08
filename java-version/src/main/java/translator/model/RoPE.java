package translator.model;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Rotary Position Embedding (RoPE).
 * Matches Python: model/rope.py
 *
 * Functions:
 * precompute_rope_freqs(dim, max_len) -> Tensor
 * apply_rope(x, freqs) -> Tensor
 */
public class RoPE {

    /**
     * Precompute RoPE frequency tensor.
     * Matches Python: def precompute_rope_freqs(dim, max_len)
     *
     * inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
     * t = torch.arange(max_len)
     * freqs = torch.outer(t, inv_freq)
     * return torch.polar(torch.ones_like(freqs), freqs)
     */
    public static Tensor precompute_rope_freqs(int dim, int max_len) {
        // inv_freq = 1.0 / (10000 ^ (arange(0, dim, 2) / dim))
        Tensor inv_freq = arange(new Scalar(0), new Scalar(dim), new Scalar(2))
                .to(ScalarType.Float).div(new Scalar((double) dim));
        inv_freq = pow(new Scalar(10000.0), inv_freq).reciprocal();

        // t = arange(max_len)
        Tensor t = arange(new Scalar(max_len)).to(ScalarType.Float);

        // freqs = outer(t, inv_freq)
        Tensor freqs = outer(t, inv_freq);

        // return polar(ones_like(freqs), freqs)
        return polar(ones_like(freqs), freqs);
    }

    /**
     * Apply rotary position embedding.
     * Matches Python: def apply_rope(x, freqs)
     */
    public static Tensor apply_rope(Tensor x, Tensor freqs) {
        long B = x.size(0), H = x.size(1), T = x.size(2);

        // x_ = x.float().reshape(B, H, T, -1, 2)
        Tensor x_ = view_as_complex(x.to(ScalarType.Float).reshape(B, H, T, -1, 2));

        // freqs = freqs[:T].view(1, 1, T, -1)
        Tensor f = freqs.slice(0, new LongOptional(0), new LongOptional(T), 1)
                .reshape(1, 1, T, -1);

        // return view_as_real(x_ * freqs).flatten(3).type_as(x)
        return view_as_real(x_.mul(f)).flatten(3, -1).to(x.scalar_type());
    }
}
