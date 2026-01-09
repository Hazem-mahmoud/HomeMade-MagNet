"""
Sequence-to-Scaler Model.

This module implements models for mapping time-series waveforms to single scalar values.
It uses an LSTM encoder followed by a linear head on the final hidden state.
"""

import torch
import torch.nn as nn

class SequenceToScalerNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, num_layers=2):
        """
        Args:
            input_dim (int): Number of features in input sequence (default 1: B or H).
            hidden_dim (int): LSTM hidden dimension.
            output_dim (int): Output size (default 1: Power Loss).
            num_layers (int): Number of LSTM layers.
        """
        super(SequenceToScalerNetwork, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Head to map final hidden state to output
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        
        # LSTM output: (Batch, Seq, Hidden), (h_n, c_n)
        # h_n shape: (Num_Layers, Batch, Hidden)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use final layer's final hidden state
        final_hidden = h_n[-1] # (Batch, Hidden)
        
        out = self.head(final_hidden)
        return out
