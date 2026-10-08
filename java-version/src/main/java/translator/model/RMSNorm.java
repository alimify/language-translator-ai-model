package translator.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Root Mean Square Layer Normalization.
 * Matches Python: model/normalization.py
 * 
 * Formula: x * rsqrt(mean(x^2) + eps) * w
 */
public class RMSNorm extends Module {

    private final Tensor w;
    private final float eps;

    public RMSNorm(int dim) {
        this.eps = 1e-6f;
        this.w = register_parameter("w", ones(dim));
    }

    /** The learned scale, exposed for the E2b neutral-weight loader. */
    public Tensor weight() { return w; }

    public Tensor forward(Tensor x) {
        // x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w
        Tensor variance = x.pow(new Scalar(2)).mean(new long[] { -1 }, true, new ScalarTypeOptional());
        return x.mul(rsqrt(variance.add(new Scalar(eps)))).mul(w);
    }
}
