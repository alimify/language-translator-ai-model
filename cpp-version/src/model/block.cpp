#include "block.hpp"

BlockImpl::BlockImpl(const Config &cfg, bool is_dec) : is_dec(is_dec)
{
    // Self-attention
    sa = register_module("sa", MultiHeadAttention(cfg.dim, cfg.heads, cfg.dropout));

    // Cross-attention (decoder only)
    if (is_dec)
    {
        ca = register_module("ca", MultiHeadAttention(cfg.dim, cfg.heads, cfg.dropout));
        n3 = register_module("n3", RMSNorm(cfg.dim));
    }

    // Feed-forward network with SiLU activation
    ff = register_module("ff", torch::nn::Sequential(
                                   torch::nn::Linear(cfg.dim, cfg.dim * cfg.ff_mult),
                                   torch::nn::SiLU(),
                                   torch::nn::Dropout(cfg.dropout),
                                   torch::nn::Linear(cfg.dim * cfg.ff_mult, cfg.dim),
                                   torch::nn::Dropout(cfg.dropout)));

    // Normalization layers
    n1 = register_module("n1", RMSNorm(cfg.dim));
    n2 = register_module("n2", RMSNorm(cfg.dim));
}

torch::Tensor BlockImpl::forward(torch::Tensor x, torch::Tensor e,
                                 torch::Tensor sm, torch::Tensor cm,
                                 torch::Tensor freqs)
{
    // Self-attention with residual
    x = x + sa->forward(n1->forward(x), {}, sm, freqs);

    // Cross-attention with residual (decoder only)
    if (is_dec)
    {
        x = x + ca->forward(n2->forward(x), e, cm);
    }

    // Feed-forward with residual
    // Python: (self.n3 if self.is_dec else self.n2)(x)
    auto norm_x = is_dec ? n3->forward(x) : n2->forward(x);
    return x + ff->forward(norm_x);
}
