#pragma once
#include <vector>
#include <string>
#include <utility>
#include "config.hpp"
#include "model/translator.hpp"
#include "tokenizer.hpp"

/**
 * @brief Run full training loop with all original features.
 * 
 * Features:
 *   - 95/5 train/val split
 *   - AdamW optimizer with weight_decay=0.01
 *   - OneCycleLR scheduler with pct_start=0.3, cosine annealing
 *   - Gradient clipping with max_norm=1.0
 *   - Label smoothing
 *   - Model checkpoint saving
 * 
 * @param cfg Configuration object
 * @param data Training data pairs
 * @param device Training device (CPU/CUDA)
 */
void run_training(const Config& cfg, 
                  std::vector<std::pair<std::string, std::string>>& data,
                  torch::Device device);
