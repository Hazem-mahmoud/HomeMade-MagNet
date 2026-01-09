"""
Scaler-to-Scaler Model.

This module implements a standard Multi-Layer Perceptron (MLP) for predicting
scalar outputs (e.g., Power Loss) from scalar inputs (Frequency, Temperature, Hdc).
"""

import torch
import torch.nn as nn

class ScalerNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1, num_layers=3):
        """
        Args:
            input_dim (int): Number of input features (default 3: Freq, Temp, Hdc).
            hidden_dim (int): Number of neurons in hidden layers.
            output_dim (int): Number of output features (default 1: Power Loss).
            num_layers (int): Number of hidden layers.
        """
        super(ScalerNetwork, self).__init__()
        
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
