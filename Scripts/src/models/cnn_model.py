"""
CNN Model for Sequence Data.

This module implements a 1D Convolutional Neural Network (CNN) for processing
time-series sequence data (e.g., flux density waveforms).
"""

import torch
import torch.nn as nn

class CNNNetwork(nn.Module):
    def __init__(self, input_dim=1, kernel_size=3, num_channels=64, num_layers=3, output_dim=1):
        """
        Args:
            input_dim (int): Number of input channels/features (default 1 for B waveform).
            kernel_size (int): Size of the convolutional kernel.
            num_channels (int): Number of channels in the convolutional layers.
            num_layers (int): Number of convolutional layers.
            output_dim (int): Number of output features (default 1 for scalar loss or sequence H).
        """
        super(CNNNetwork, self).__init__()
        
        layers = []
        
        # Initial Conv Layer
        # Input shape: (Batch, Channels, Seq_Len) -> Conv1d needs (N, C_in, L)
        # However, our data loader typically provides (Batch, Seq_Len, Features)
        # We'll handle the permute in forward().
        
        layers.append(nn.Conv1d(in_channels=input_dim, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(num_channels))
        
        # Hidden Conv Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(num_channels))
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output Head
        # For sequence-to-scalar, we might pool first. For sequence-to-sequence, we keep it.
        # Assuming we want to support both or default to something useful.
        # Let's assume Sequence-to-Sequence for now, or use a Global Average Pooling for Scalar.
        # User request didn't specify, but "CNN" usually implies feature extraction over time.
        # Let's add a Global Average Pooling + Linear for Scalar output by default since 'sequence_model' was seq-to-scalar.
        # But wait, seq2seq_model exists too.
        # Let's make it flexible or design for Sequence-to-Scaler (Loss prediction).
        # Given "Transformer" and "CNN" were added together, likely for the same task as "all" models.
        # 'sequence' model predicts Log Loss (Scalar). 'seq2seq' predicts H (Sequence).
        # Let's make CNN predict Scalar Loss (Sequence-to-Scalar) similar to 'sequence_model'.
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels, output_dim)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        # Conv1d expects (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)
        
        features = self.feature_extractor(x)
        
        # features shape: (Batch, Num_Channels, Seq_Len)
        
        # Global Average Pooling -> (Batch, Num_Channels, 1)
        pooled = self.global_pool(features).squeeze(-1)
        
        # FC -> (Batch, Output_Dim)
        out = self.fc(pooled)
        
        return out
