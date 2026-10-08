#pragma once
#include <torch/torch.h>

/**
 * @brief Root Mean Square Layer Normalization.
 * 
 * More efficient than LayerNorm as it doesn't require mean computation.
 * Formula: x * rsqrt(mean(x^2) + eps) * weight
 */
struct RMSNormImpl : torch::nn::Module {
    RMSNormImpl(int d, double eps = 1e-6);
    torch::Tensor forward(torch::Tensor x);
    
    double eps;
    torch::Tensor w;
};
TORCH_MODULE(RMSNorm);
