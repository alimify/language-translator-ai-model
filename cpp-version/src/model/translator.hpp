#pragma once
#include <torch/torch.h>
#include "../config.hpp"
#include "rope.hpp"
#include "normalization.hpp"
#include "block.hpp"

/**
 * @brief Encoder-Decoder Transformer for sequence-to-sequence translation.
 * 
 * Architecture:
 *   - Separate source and target embeddings
 *   - Embedding scaling by sqrt(dim)
 *   - Weight tying between target embedding and output projection
 *   - RoPE positional encoding
 *   - Xavier initialization for multi-dimensional parameters
 */
struct TranslatorImpl : torch::nn::Module {
    TranslatorImpl(const Config& cfg, int sv, int tv);
    
    /**
     * @brief Forward pass through encoder-decoder.
     * @param src Source token IDs (B, T_src)
     * @param trg Target token IDs (B, T_trg)
     * @return Logits tensor (B, T_trg, vocab_size)
     */
    torch::Tensor forward(torch::Tensor src, torch::Tensor trg);
    
    int dim;
    torch::nn::Embedding se = nullptr, te = nullptr;  // Source/Target embeddings
    torch::nn::ModuleList enc = nullptr, dec = nullptr;
    RMSNorm norm = nullptr;
    // Output projection is computed functionally from te->weight (true weight
    // tying: a single shared parameter, exactly like Python's
    // `self.head.weight = self.te.weight`).
    torch::Tensor freqs;  // Precomputed RoPE frequencies
};
TORCH_MODULE(Translator);
