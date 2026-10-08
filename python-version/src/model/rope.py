"""Rotary Position Embedding (RoPE) implementation."""
import torch
from torch import Tensor


def precompute_rope_freqs(dim: int, max_len: int) -> Tensor:
    """Precompute RoPE frequency tensor using complex polar form.
    
    Formula: freq = 1.0 / (10000 ^ (2i / dim)) for i in [0, dim/2)
    
    Args:
        dim: Head dimension (model_dim // num_heads)
        max_len: Maximum sequence length
        
    Returns:
        Complex tensor of shape (max_len, dim/2) containing rotation frequencies
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary position embedding to input tensor.
    
    Args:
        x: Input tensor of shape (B, H, T, D)
        freqs: Precomputed frequency tensor
        
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    B, H, T, D = x.shape
    # Reshape to complex: (B, H, T, D/2, 2) -> complex (B, H, T, D/2)
    x_c = torch.view_as_complex(x.float().reshape(B, H, T, -1, 2))
    # Apply rotation via complex multiplication
    result = torch.view_as_real(x_c * freqs[:T].view(1, 1, T, -1))
    return result.reshape(B, H, T, D).type_as(x)
