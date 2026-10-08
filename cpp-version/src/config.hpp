#pragma once

/**
 * @brief Configuration for the Translator model.
 * All values match the original Python implementation exactly.
 */
struct Config
{
    int max_len = 64;             // Maximum sequence length
    int dim = 256;                // Model dimension
    int heads = 8;                // Number of attention heads
    int layers = 6;               // Number of encoder/decoder layers
    int ff_mult = 4;              // Feed-forward multiplier
    int vocab_size = 12000;       // Maximum vocabulary size
    double lr = 8e-4;             // Learning rate
    int epochs = 40;              // Training epochs
    int batch_size = 32;          // Batch size
    double label_smoothing = 0.1; // Label smoothing factor
    double dropout = 0.2;         // Dropout rate
};
