"""
RMSNorm implementation (Root Mean Square Layer Normalization)
As used in Gemma, LLaMA, and other modern LLMs.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm and works well with residual connections.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        # RMSNorm: normalize by RMS (root mean square) instead of mean
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

