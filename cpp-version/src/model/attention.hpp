#pragma once
#include <torch/torch.h>

/**
 * @brief Multi-Head Attention with RoPE support.
 *
 * Features:
 *   - Q, K, V projections without bias
 *   - RoPE applied to Q always, K only for self-attention
 *   - Scaled dot-product attention with masking
 */
struct MultiHeadAttentionImpl : torch::nn::Module
{
    MultiHeadAttentionImpl(int d, int h, double dropout = 0.0);

    /**
     * @brief Compute multi-head attention.
     * @param x Query input (B, T, D)
     * @param kv Key-Value input for cross-attention (default: self-attention)
     * @param mask Attention mask (1 = attend, 0 = mask)
     * @param freqs RoPE frequency tensor
     * @return Output tensor (B, T, D)
     */
    torch::Tensor forward(torch::Tensor x,
                          torch::Tensor kv = {},
                          torch::Tensor mask = {},
                          torch::Tensor freqs = {});

    int h;  // Number of heads
    int dh; // Head dimension
    torch::nn::Linear wq = nullptr, wk = nullptr, wv = nullptr, o = nullptr;
    torch::nn::Dropout dropout = nullptr;
};
TORCH_MODULE(MultiHeadAttention);
