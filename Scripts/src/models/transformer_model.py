"""
Transformer Model for Sequence Data.

This module implements a Transformer-based architecture for processing
time-series sequence data.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (Max_Len, 1, D_Model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Seq_Len, Batch, D_Model)
        return x + self.pe[:x.size(0), :]

class TransformerNetwork(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_dim=1):
        """
        Args:
            input_dim (int): Number of input features.
            d_model (int): Hidden dimension size.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of the FFN.
            dropout (float): Dropout rate.
            output_dim (int): Output dimension (default 1 for scalar loss).
        """
        super(TransformerNetwork, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, output_dim)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x shape from loader: (Batch, Seq_Len, Features)
        # Transformer expects: (Seq_Len, Batch, Features) for default batch_first=False
        
        x = x.permute(1, 0, 2) 
        
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x)
        
        # output shape: (Seq_Len, Batch, D_Model)
        
        # For sequence-to-scalar, take the output of the last time step? Or average?
        # Let's take the mean over the sequence length.
        output = output.mean(dim=0) # (Batch, D_Model)
        
        output = self.decoder(output) # (Batch, Output_Dim)
        
        return output
