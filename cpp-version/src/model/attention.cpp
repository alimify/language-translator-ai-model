#include "attention.hpp"
#include "rope.hpp"
#include <cmath>

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int d, int h, double dropout)
    : h(h), dh(d / h)
{
    wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
    wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
    wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
    o = register_module("o", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
    this->dropout = register_module("dropout", torch::nn::Dropout(dropout));
}

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x, torch::Tensor kv,
                                              torch::Tensor mask, torch::Tensor freqs)
{
    int B = x.size(0);
    int T = x.size(1);
    int D = x.size(2);

    // Default kv to x for self-attention
    if (!kv.defined())
        kv = x;
    int Tk = kv.size(1);

    // Compute Q, K, V
    auto q = wq(x).view({B, T, h, dh}).transpose(1, 2); // [B, H, T, dh]
    auto k = wk(kv).view({B, Tk, h, dh}).transpose(1, 2);
    auto v = wv(kv).view({B, Tk, h, dh}).transpose(1, 2);

    // Apply RoPE: always to Q, only to K for self-attention
    if (freqs.defined())
    {
        q = apply_rope(q, freqs);
        if (kv.data_ptr() == x.data_ptr())
        { // Self-attention
            k = apply_rope(k, freqs);
        }
    }

    // Scaled dot-product attention
    auto att = torch::matmul(q, k.transpose(-2, -1)) * std::pow(dh, -0.5);

    // Apply mask
    if (mask.defined())
    {
        att = att.masked_fill(mask == 0, -1e9);
    }

    // Softmax and dropout
    att = dropout(torch::softmax(att, -1));

    // Combine heads
    auto out = torch::matmul(att, v).transpose(1, 2).reshape({B, T, D});
    return o(out);
}
