"""Configuration for the Translator model."""
from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters for the Translator model.
    
    All values match the original implementation exactly.
    """
    max_len: int = 64           # Maximum sequence length
    dim: int = 256              # Model dimension
    heads: int = 8              # Number of attention heads
    layers: int = 6             # Number of encoder/decoder layers
    ff_mult: int = 4            # Feed-forward multiplier
    vocab_size: int = 12000     # Maximum vocabulary size
    lr: float = 8e-4            # Learning rate
    epochs: int = 40            # Training epochs
    batch_size: int = 32        # Batch size
    label_smoothing: float = 0.1  # Label smoothing factor
    dropout: float = 0.2        # Dropout rate
