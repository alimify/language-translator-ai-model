"""RMS Normalization layer."""
import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't require mean computation.
    Formula: x * rsqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, d: int, eps: float = 1e-6):
        """Initialize RMSNorm.
        
        Args:
            d: Feature dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., d)
            
        Returns:
            Normalized tensor of same shape
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w
