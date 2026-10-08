"""Multi-Head Attention with RoPE support."""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .rope import apply_rope


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Rotary Position Embedding (RoPE) support.
    
    Features:
        - Q, K, V projections without bias (as per original)
        - RoPE applied to Q always, K only for self-attention
        - Scaled dot-product attention with masking
    """
    
    def __init__(self, d: int, h: int, dropout: float = 0.0):
        """Initialize Multi-Head Attention.
        
        Args:
            d: Model dimension
            h: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.h = h
        self.dh = d // h  # Head dimension
        
        # Q, K, V projections without bias
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        
        # Output projection
        self.o = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        kv: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        freqs: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute multi-head attention.
        
        Args:
            x: Query input of shape (B, T, D)
            kv: Key-Value input for cross-attention (default: self-attention)
            mask: Attention mask (1 = attend, 0 = mask)
            freqs: RoPE frequency tensor
            
        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape
        kv = kv if kv is not None else x
        Tk = kv.shape[1]
        
        # Compute Q, K, V
        q, k, v = self.wq(x), self.wk(kv), self.wv(kv)
        
        # Reshape to (B, H, T, D_h)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = v.view(B, Tk, self.h, self.dh).transpose(1, 2)
        
        # Apply RoPE: always to Q, only to K for self-attention
        if freqs is not None:
            q = apply_rope(q, freqs)
            if kv is x:  # Self-attention
                k = apply_rope(k, freqs)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        
        # Apply mask
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        att = self.dropout(att.softmax(dim=-1))
        
        # Combine heads
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)
