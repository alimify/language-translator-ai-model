#pragma once
#include <torch/torch.h>

/**
 * @brief Rotary Position Embedding (RoPE) functions.
 * 
 * Uses complex polar form for rotation:
 * freq = 1.0 / (10000 ^ (2i / dim))
 */

/**
 * @brief Precompute RoPE frequency tensor.
 * @param dim Head dimension (model_dim / num_heads)
 * @param max_len Maximum sequence length
 * @return Complex tensor of shape (max_len, dim/2)
 */
torch::Tensor precompute_rope_freqs(int dim, int max_len);

/**
 * @brief Apply rotary position embedding to input tensor.
 * @param x Input tensor of shape (B, H, T, D)
 * @param freqs Precomputed frequency tensor
 * @return Tensor with RoPE applied
 */
torch::Tensor apply_rope(torch::Tensor x, torch::Tensor freqs);
