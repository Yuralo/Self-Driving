"""
SwiGLU Activation Function
Better than GELU for MLPs, used in LLaMA, PaLM, etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit
    SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    where Swish(x) = x * sigmoid(x)
    """
    def forward(self, x):
        # x should be split into two parts: gate and value
        # This is typically done in the MLP layer
        return x


def swiglu(x):
    """SwiGLU activation function."""
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x  # silu is swish


class SwiGLUMLP(nn.Module):
    """
    MLP with SwiGLU activation.
    Uses 2/3 expansion factor (like LLaMA) instead of 4x.
    """
    def __init__(self, d_model, expansion_factor=8/3, dropout=0.1):
        """
        expansion_factor: typically 8/3 for SwiGLU (vs 4 for GELU)
        This means hidden_dim = d_model * 8/3 ≈ d_model * 2.67
        """
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        # SwiGLU needs 2x hidden_dim for gate and value
        gate_proj_dim = hidden_dim * 2
        
        self.gate_proj = nn.Linear(d_model, gate_proj_dim)
        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: Swish(gate) * up
        gate = self.gate_proj(x)  # (..., 2*hidden_dim)
        up = self.up_proj(x)  # (..., hidden_dim)
        
        # Split gate into two parts for swiglu
        gate1, gate2 = gate.chunk(2, dim=-1)  # Each: (..., hidden_dim)
        # Apply swish to gate2 and multiply with gate1
        activated = F.silu(gate2) * gate1  # (..., hidden_dim)
        
        # Multiply with up projection
        out = activated * up  # (..., hidden_dim)
        out = self.down_proj(out)  # (..., d_model)
        return self.dropout(out)

