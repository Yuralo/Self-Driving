import torch
import torch.nn as nn
import math

class SpatialEncoder(nn.Module):
    def __init__(self):
        super(SpatialEncoder, self).__init__()
        # Nvidia-style CNN for self-driving
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            nn.Flatten()
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv_layers(x)

class DrivingPolicy(nn.Module):
    def __init__(self, sequence_length=5, d_model=256, nhead=4, num_layers=2):
        super(DrivingPolicy, self).__init__()
        self.encoder = SpatialEncoder()
        
        # Calculate embedding size
        # Input: 66x200
        # Conv1: 31x98 (stride 2, kernel 5)
        # Conv2: 14x47
        # Conv3: 5x22
        # Conv4: 3x20 (stride 1, kernel 3)
        # Conv5: 1x18
        # Flatten: 64 * 1 * 18 = 1152
        self.feature_dim = 1152
        
        self.projection = nn.Linear(self.feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ELU(),
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
