"""
Rotary Positional Embeddings (RoPE)
Better than fixed positional encodings, used in LLaMA, Gemma, etc.
"""
import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Applies rotation to query and key vectors based on position.
    """
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
        # Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = None
    
    def forward(self, x, seq_len=None):
        """
        x: (B, Seq, H, D) where D is head_dim
        Returns: cos, sin for rotary embedding
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Use cache if available
        if self._cached_seq_len == seq_len and self._cached_cos is not None:
            return self._cached_cos, self._cached_sin
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        # Cache for future use
        self._cached_cos = cos
        self._cached_sin = sin
        self._cached_seq_len = seq_len
        
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embedding to query and key.
    q, k: (B, Seq, H, D)
    cos, sin: (Seq, D)
    """
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, Seq, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, Seq, 1, D)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

