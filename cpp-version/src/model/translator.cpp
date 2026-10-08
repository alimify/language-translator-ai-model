#include "translator.hpp"
#include <cmath>

TranslatorImpl::TranslatorImpl(const Config& cfg, int sv, int tv) : dim(cfg.dim) {
    // Embeddings
    se = register_module("se", torch::nn::Embedding(sv, cfg.dim));
    te = register_module("te", torch::nn::Embedding(tv, cfg.dim));
    
    // Encoder stack
    enc = register_module("enc", torch::nn::ModuleList());
    for (int i = 0; i < cfg.layers; ++i) {
        enc->push_back(Block(cfg, false));
    }
    
    // Decoder stack
    dec = register_module("dec", torch::nn::ModuleList());
    for (int i = 0; i < cfg.layers; ++i) {
        dec->push_back(Block(cfg, true));
    }
    
    // Output layers. The projection reuses te->weight directly in forward()
    // (true weight tying, matching Python's `self.head.weight = self.te.weight`
    // where parameters() deduplicates the shared tensor).
    norm = register_module("norm", RMSNorm(cfg.dim));

    // Precompute RoPE frequencies
    freqs = register_buffer("freqs", precompute_rope_freqs(cfg.dim / cfg.heads, cfg.max_len));
    
    // Xavier initialization for multi-dimensional parameters
    for (auto& p : parameters()) {
        if (p.dim() > 1) {
            torch::nn::init::xavier_uniform_(p);
        }
    }
}

torch::Tensor TranslatorImpl::forward(torch::Tensor src, torch::Tensor trg) {
    auto device = trg.device();
    int sz = trg.size(1);
    
    // Source mask: (B, 1, 1, T_src)
    auto sm = (src != 0).unsqueeze(1).unsqueeze(2);
    
    // Target mask: causal mask (1, 1, T_trg, T_trg)
    auto tm = torch::tril(torch::ones({sz, sz}, torch::TensorOptions().device(device)))
                  .to(torch::kBool).view({1, 1, sz, sz});
    
    // Encode source with embedding scaling
    auto e = se(src) * std::sqrt(static_cast<double>(dim));
    for (size_t i = 0; i < enc->size(); ++i) {
        e = enc->at<BlockImpl>(i).forward(e, {}, sm, {}, freqs);
    }

    // Decode target with embedding scaling
    auto d = te(trg) * std::sqrt(static_cast<double>(dim));
    for (size_t i = 0; i < dec->size(); ++i) {
        d = dec->at<BlockImpl>(i).forward(d, e, tm, sm, freqs);
    }

    // Output projection (tied to target embedding weight, no bias)
    return torch::nn::functional::linear(norm->forward(d), te->weight);
}
