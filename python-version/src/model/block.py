"""Transformer Block with optional cross-attention for encoder/decoder."""
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .normalization import RMSNorm
from .attention import MultiHeadAttention


class Block(nn.Module):
    """Transformer Block supporting both encoder and decoder configurations.
    
    Architecture:
        - Pre-norm with RMSNorm
        - Self-attention (always)
        - Cross-attention (decoder only)
        - Feed-forward network with SiLU activation
    """
    
    def __init__(self, cfg, is_dec: bool = False):
        """Initialize Transformer Block.
        
        Args:
            cfg: Config object with dim, heads, ff_mult, dropout
            is_dec: If True, include cross-attention for decoder
        """
        super().__init__()
        self.is_dec = is_dec
        
        # Self-attention
        self.sa = MultiHeadAttention(cfg.dim, cfg.heads, cfg.dropout)
        
        # Cross-attention (decoder only)
        self.ca = MultiHeadAttention(cfg.dim, cfg.heads, cfg.dropout) if is_dec else None
        
        # Feed-forward network with SiLU activation
        self.ff = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * cfg.ff_mult),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim * cfg.ff_mult, cfg.dim),
            nn.Dropout(cfg.dropout),
        )
        
        # Normalization layers
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        if is_dec:
            self.n3 = RMSNorm(cfg.dim)
    
    def forward(
        self,
        x: Tensor,
        e: Optional[Tensor] = None,
        sm: Optional[Tensor] = None,
        cm: Optional[Tensor] = None,
        freqs: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the block.
        
        Args:
            x: Input tensor (B, T, D)
            e: Encoder output for cross-attention (decoder only)
            sm: Self-attention mask
            cm: Cross-attention mask
            freqs: RoPE frequencies
            
        Returns:
            Output tensor (B, T, D)
        """
        # Self-attention with residual
        x = x + self.sa(self.n1(x), mask=sm, freqs=freqs)
        
        # Cross-attention with residual (decoder only)
        if self.is_dec:
            x = x + self.ca(self.n2(x), kv=e, mask=cm)
        
        # Feed-forward with residual
        norm = self.n3 if self.is_dec else self.n2
        return x + self.ff(norm(x))
