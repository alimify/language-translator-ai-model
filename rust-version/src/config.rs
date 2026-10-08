//! Configuration for the Translator model.

/// Hyperparameters for the Translator model.
/// All values match the original Python implementation exactly.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_len: i64, // Maximum sequence length
    pub dim: i64, // Model dimension
    pub heads: i64, // Number of attention heads
    pub layers: i64, // Number of encoder/decoder layers
    pub ff_mult: i64, // Feed-forward multiplier
    pub vocab_size: i64, // Maximum vocabulary size
    pub lr: f64, // Learning rate
    pub epochs: i64, // Training epochs
    pub batch_size: i64, // Batch size
    pub label_smoothing: f64, // Label smoothing factor
    pub dropout: f64, // Dropout rate
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_len: 64,
            dim: 256,
            heads: 8,
            layers: 6,
            ff_mult: 4,
            vocab_size: 12000,
            lr: 8e-4,
            epochs: 40,
            batch_size: 32,
            label_smoothing: 0.1,
            dropout: 0.2,
        }
    }
}
