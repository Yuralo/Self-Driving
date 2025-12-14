import math

import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb
from .swiglu import SwiGLUMLP


class SpatialEncoder(nn.Module):
    def __init__(self, dropout=0.3):
        super(SpatialEncoder, self).__init__()
        # Improved Nvidia-style CNN with batch normalization and dropout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Dropout2d(dropout * 0.3),
            
            nn.Conv2d(24, 36, 5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Dropout2d(dropout * 0.3),
            
            nn.Conv2d(36, 48, 5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Dropout2d(dropout * 0.3),
            
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout * 0.2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(dropout * 0.2),
            
            nn.Flatten()
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv_layers(x)

class DrivingPolicy(nn.Module):
    def __init__(self, sequence_length=5, d_model=256, nhead=4, num_layers=2, dropout=0.3):
        super(DrivingPolicy, self).__init__()
        self.encoder = SpatialEncoder(dropout=dropout)
        
        # Calculate embedding size
        # Input: 66x200
        # Conv1: 33x100 (stride 2, kernel 5, padding 2)
        # Conv2: 17x50
        # Conv3: 9x25
        # Conv4: 9x25 (stride 1, kernel 3, padding 1)
        # Conv5: 9x25
        # Flatten: 64 * 9 * 25 = 14400
        self.feature_dim = 14400
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Improved head with residual connection idea
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1) # Steering angle
        )
        
    def forward(self, x):
        # x: (B, Seq, C, H, W)
        B, Seq, C, H, W = x.shape
        
        # Merge Batch and Seq for CNN
        x = x.view(B * Seq, C, H, W)
        features = self.encoder(x) # (B*Seq, feature_dim)
        
        # Reshape back
        features = features.view(B, Seq, -1)
        
        # Project to d_model
        features = self.projection(features) # (B, Seq, d_model)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        output = self.transformer_encoder(features) # (B, Seq, d_model)
        
        # Take the last token's output for prediction
        last_output = output[:, -1, :]
        
        steering = self.head(last_output)
        return steering

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MLPBlock(nn.Module):
    """MLP block with RMSNorm, SwiGLU, and residual connection."""
    def __init__(self, d_model, use_swiglu=True, expansion_factor=8/3, dropout=0.1, layer_scale=1.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        if use_swiglu:
            self.mlp = SwiGLUMLP(d_model, expansion_factor=expansion_factor, dropout=dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, int(d_model * expansion_factor * 1.5)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(d_model * expansion_factor * 1.5), d_model),
                nn.Dropout(dropout)
            )
        # Layer scaling (like LLaMA)
        self.layer_scale = nn.Parameter(torch.ones(d_model) * layer_scale) if layer_scale > 0 else None
    
    def forward(self, x):
        # x: (B, Seq, d_model) or (B, d_model)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        if self.layer_scale is not None:
            x = x * self.layer_scale
        return x + residual


class AttentionBlock(nn.Module):
    """Multi-head self-attention with RMSNorm, RoPE, and residual connection."""
    def __init__(self, d_model, nhead=4, dropout=0.1, use_rope=True, layer_scale=1.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rope = use_rope
        
        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=512)
        
        # Layer scaling
        self.layer_scale = nn.Parameter(torch.ones(d_model) * layer_scale) if layer_scale > 0 else None
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x):
        # x: (B, Seq, d_model)
        residual = x
        x = self.norm(x)
        
        B, Seq, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, Seq, self.nhead, self.head_dim)
        k = self.k_proj(x).view(B, Seq, self.nhead, self.head_dim)
        v = self.v_proj(x).view(B, Seq, self.nhead, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(q, seq_len=Seq)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for attention: (B, H, Seq, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v)  # (B, H, Seq, D)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, Seq, -1)
        out = self.o_proj(out)
        
        if self.layer_scale is not None:
            out = out * self.layer_scale
        
        return out + residual


class TransformerBlock(nn.Module):
    """Full transformer block with RMSNorm, attention (RoPE), and MLP (SwiGLU)."""
    def __init__(self, d_model, nhead=4, dropout=0.1, use_rope=True, use_swiglu=True, layer_scale=1.0):
        super().__init__()
        self.attention = AttentionBlock(d_model, nhead, dropout, use_rope=use_rope, layer_scale=layer_scale)
        self.mlp = MLPBlock(d_model, use_swiglu=use_swiglu, dropout=dropout, layer_scale=layer_scale)
    
    def forward(self, x):
        # x: (B, Seq, d_model)
        x = self.attention(x)
        x = self.mlp(x)
        return x


class ResidualStreamPolicy(nn.Module):
    """
    Advanced policy network with residual stream architecture.
    Inspired by modern LLM architectures (Gemma, LLaMA).
    
    Architecture:
    1. Input -> RMSNorm -> CNN Encoder -> Projection -> Residual Stream
    2. Residual Stream -> MLP Block -> Add to stream
    3. Residual Stream -> Attention Block -> Add to stream
    4. Residual Stream -> Transformer Block -> Add to stream
    5. Residual Stream -> Attention Block -> Add to stream
    6. Residual Stream -> RMSNorm -> Policy Head
    """
    def __init__(self, sequence_length=5, d_model=256, nhead=4, dropout=0.1, 
                 use_rope=True, use_swiglu=True, layer_scale=1e-5):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Initial CNN encoder
        self.encoder = SpatialEncoder(dropout=dropout)
        self.feature_dim = 14400  # Calculated from CNN output
        
        # Initial projection with RMSNorm
        self.input_norm = RMSNorm(self.feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Note: Positional encoding handled by RoPE in attention blocks
        
        # Residual stream processing blocks
        # Block 1: MLP with SwiGLU
        self.mlp_block1 = MLPBlock(d_model, use_swiglu=use_swiglu, dropout=dropout, layer_scale=layer_scale)
        
        # Block 2: Attention with RoPE
        self.attention_block1 = AttentionBlock(d_model, nhead=nhead, dropout=dropout, 
                                               use_rope=use_rope, layer_scale=layer_scale)
        
        # Block 3: Full Transformer (Attention + MLP)
        self.transformer_block = TransformerBlock(d_model, nhead=nhead, dropout=dropout,
                                                  use_rope=use_rope, use_swiglu=use_swiglu, 
                                                  layer_scale=layer_scale)
        
        # Block 4: Another Attention with RoPE
        self.attention_block2 = AttentionBlock(d_model, nhead=nhead, dropout=dropout,
                                               use_rope=use_rope, layer_scale=layer_scale)
        
        # Block 5: Another MLP with SwiGLU
        self.mlp_block2 = MLPBlock(d_model, use_swiglu=use_swiglu, dropout=dropout, layer_scale=layer_scale)
        
        # Final normalization and policy head
        self.output_norm = RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)  # Steering angle
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following LLaMA/Gemma practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform for most layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Special initialization for output layer
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.02)
        if self.head[-1].bias is not None:
            nn.init.constant_(self.head[-1].bias, 0)
    
    def forward(self, x):
        # x: (B, Seq, C, H, W)
        B, Seq, C, H, W = x.shape
        
        # Step 1: CNN encoding
        x = x.view(B * Seq, C, H, W)
        features = self.encoder(x)  # (B*Seq, feature_dim)
        features = features.view(B, Seq, -1)  # (B, Seq, feature_dim)
        
        # Step 2: Input normalization and projection to residual stream
        # Apply RMSNorm per sequence element
        features = self.input_norm(features)  # (B, Seq, feature_dim)
        residual_stream = self.projection(features)  # (B, Seq, d_model)
        
        # Note: Positional encoding is handled by RoPE in attention blocks
        
        # Step 3: Process through residual stream blocks
        # Each block adds to the residual stream
        
        # Block 1: MLP
        residual_stream = self.mlp_block1(residual_stream)
        
        # Block 2: Attention
        residual_stream = self.attention_block1(residual_stream)
        
        # Block 3: Transformer (Attention + MLP)
        residual_stream = self.transformer_block(residual_stream)
        
        # Block 4: Another Attention
        residual_stream = self.attention_block2(residual_stream)
        
        # Block 5: Another MLP
        residual_stream = self.mlp_block2(residual_stream)
        
        # Step 4: Final normalization and prediction
        # Take the last token's output
        last_output = residual_stream[:, -1, :]  # (B, d_model)
        last_output = self.output_norm(last_output)
        
        # Policy head
        steering = self.head(last_output)
        return steering
