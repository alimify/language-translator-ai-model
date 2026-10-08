#include "rope.hpp"

torch::Tensor precompute_rope_freqs(int dim, int max_len)
{
    // inv_freq = 1.0 / (10000 ^ (2i / dim)) for i in [0, dim/2)
    auto inv_freq = 1.0 / torch::pow(10000, torch::arange(0, dim, 2).to(torch::kFloat) / dim);
    auto t = torch::arange(max_len).to(torch::kFloat);
    auto freqs = torch::outer(t, inv_freq);
    // Use polar form: e^(i*theta) = cos(theta) + i*sin(theta)
    return torch::polar(torch::ones_like(freqs), freqs);
}

torch::Tensor apply_rope(torch::Tensor x, torch::Tensor freqs)
{
    // x: [B, H, T, D]
    int B = x.size(0);
    int H = x.size(1);
    int T = x.size(2);
    int D = x.size(3);

    // Reshape to complex: (B, H, T, D/2, 2) -> complex (B, H, T, D/2)
    auto x_float = x.to(torch::kFloat).reshape({B, H, T, -1, 2});
    auto x_c = torch::view_as_complex(x_float);

    // Apply rotation via complex multiplication
    auto f = freqs.slice(0, 0, T).view({1, 1, T, -1});
    auto rot = x_c * f;

    return torch::view_as_real(rot).reshape({B, H, T, D}).type_as(x);
}
