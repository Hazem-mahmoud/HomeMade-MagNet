"""
Sequence-to-Sequence Model.

This module implements Encoder-Decoder architectures for mapping input waveforms 
(Excitation, e.g., H) to output waveforms (Response, e.g., B).
"""

import torch
import torch.nn as nn

class Seq2SeqNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, num_layers=2):
        """
        Args:
            input_dim (int): Input feature size.
            hidden_dim (int): Hidden size.
            output_dim (int): Output feature size.
            num_layers (int): Depth of LSTM.
        """
        super(Seq2SeqNetwork, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Decoder
        # Input to decoder is previous output (or ground truth in training).
        # We model this as mapping Hidden State -> Sequence.
        
        # Simple Approach: Use LSTM to map (Batch, Seq, Hidden) -> (Batch, Seq, Out).
        # But standard Seq2Seq uses an Decoder LSTM.
        
        # Here we implement a simple LSTM-based mapping (Many-to-Many).
        # Since input and output length are same for hysteresis loops.
        # This acts like a Bi-Directional LSTM or just a mapped LSTM.
        
        self.decoder = nn.LSTM(
            input_size=hidden_dim, # We might feed encoder outputs or similar
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (Batch, Seq, Input)
        
        # Encoder
        # We want to map Sequence -> Sequence.
        # If lengths are same and it's 1:1 mapping (like filtering), 
        # a single LSTM (Encoder) + Linear Head per step is sufficient.
        
        # output: (Batch, Seq, Hidden)
        enc_out, _ = self.encoder(x)
        
        # Decode/Map
        # dec_out, _ = self.decoder(enc_out) # Optional: Deepen model
        
        # Project to output
        # out: (Batch, Seq, Output)
        out = self.head(enc_out)
        
        return out
