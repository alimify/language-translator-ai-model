"""Main Translator model - Encoder-Decoder Transformer."""
import torch
import torch.nn as nn
from torch import Tensor

from .rope import precompute_rope_freqs
from .normalization import RMSNorm
from .block import Block


class Translator(nn.Module):
    """Encoder-Decoder Transformer for sequence-to-sequence translation.
    
    Architecture:
        - Separate source and target embeddings
        - Embedding scaling by sqrt(dim)
        - Weight tying between target embedding and output projection
        - RoPE positional encoding
        - Xavier initialization for multi-dimensional parameters
    """
    
    def __init__(self, cfg, sv: int, tv: int):
        """Initialize Translator model.
        
        Args:
            cfg: Config object with model hyperparameters
            sv: Source vocabulary size
            tv: Target vocabulary size
        """
        super().__init__()
        self.dim = cfg.dim
        
        # Embeddings
        self.se = nn.Embedding(sv, cfg.dim)  # Source embedding
        self.te = nn.Embedding(tv, cfg.dim)  # Target embedding
        
        # Encoder and Decoder stacks
        self.enc = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.dec = nn.ModuleList([Block(cfg, is_dec=True) for _ in range(cfg.layers)])
        
        # Output layers
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, tv, bias=False)
        
        # Weight tying: share weights between target embedding and output projection
        self.head.weight = self.te.weight
        
        # Precompute RoPE frequencies
        self.register_buffer("freqs", precompute_rope_freqs(cfg.dim // cfg.heads, cfg.max_len))
        
        # Xavier initialization for multi-dimensional parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        """Forward pass through encoder-decoder.
        
        Args:
            src: Source token IDs (B, T_src)
            trg: Target token IDs (B, T_trg)
            
        Returns:
            Logits tensor (B, T_trg, vocab_size)
        """
        # Create masks
        # Source mask: (B, 1, 1, T_src) - mask padding tokens
        sm = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Target mask: causal mask (1, 1, T_trg, T_trg)
        sz = trg.size(1)
        tm = torch.tril(torch.ones(sz, sz, device=trg.device)).bool().view(1, 1, sz, sz)
        
        # Encode source with embedding scaling
        e = self.se(src) * (self.dim ** 0.5)
        for layer in self.enc:
            e = layer(e, sm=sm, freqs=self.freqs)
        
        # Decode target with embedding scaling
        d = self.te(trg) * (self.dim ** 0.5)
        for layer in self.dec:
            d = layer(d, e, sm=tm, cm=sm, freqs=self.freqs)
        
        # Output projection
        return self.head(self.norm(d))
