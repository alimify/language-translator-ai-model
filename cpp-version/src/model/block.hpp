#pragma once
#include <torch/torch.h>
#include "../config.hpp"
#include "normalization.hpp"
#include "attention.hpp"

/**
 * @brief Transformer Block supporting encoder and decoder configurations.
 *
 * Architecture:
 *   - Pre-norm with RMSNorm
 *   - Self-attention (always)
 *   - Cross-attention (decoder only)
 *   - Feed-forward network with SiLU activation
 */
struct BlockImpl : torch::nn::Module
{
    BlockImpl(const Config &cfg, bool is_dec = false);

    /**
     * @brief Forward pass through the block.
     * @param x Input tensor (B, T, D)
     * @param e Encoder output for cross-attention (decoder only)
     * @param sm Self-attention mask
     * @param cm Cross-attention mask
     * @param freqs RoPE frequencies
     * @return Output tensor (B, T, D)
     */
    torch::Tensor forward(torch::Tensor x,
                          torch::Tensor e = {},
                          torch::Tensor sm = {},
                          torch::Tensor cm = {},
                          torch::Tensor freqs = {});

    bool is_dec;
    MultiHeadAttention sa = nullptr; // Self-attention
    MultiHeadAttention ca = nullptr; // Cross-attention (decoder only)
    torch::nn::Sequential ff = nullptr;
    RMSNorm n1 = nullptr, n2 = nullptr, n3 = nullptr;
};
TORCH_MODULE(Block);
